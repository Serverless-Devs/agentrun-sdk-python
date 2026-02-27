"""Google ADK Agent Server —— 使用 OTSSessionService 持久化会话。

集成步骤：
  Step 1: 初始化 SessionStore（OTS 后端）
  Step 2: 创建 OTSSessionService
  Step 3: 创建 ADK Agent + Runner，传入 session_service
  Step 4: 实现 invoke_agent，将 AgentRequest 转为 ADK 调用并流式输出
  Step 5: 通过 AgentRunServer 启动 HTTP 服务

使用方式：
  uv run --env-file .env python examples/conversation_service_adk_server.py
"""

from __future__ import annotations

import os
import sys
from typing import Any
import uuid

from dotenv import load_dotenv
from google.adk.agents import Agent  # type: ignore[import-untyped]
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner  # type: ignore[import-untyped]
from google.adk.tools import ToolContext  # type: ignore[import-untyped]
from google.genai import types  # type: ignore[import-untyped]

from agentrun import AgentRequest
from agentrun.conversation_service import SessionStore
from agentrun.conversation_service.adapters import OTSSessionService
from agentrun.server import AgentRunServer

load_dotenv()

# ── 配置参数 ──────────────────────────────────────────────────
APP_NAME = "adk_chat_server"
MEMORY_COLLECTION_NAME = os.getenv("MEMORY_COLLECTION_NAME", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

if not MEMORY_COLLECTION_NAME:
    print("ERROR: 请设置环境变量 MEMORY_COLLECTION_NAME")
    sys.exit(1)
if not DASHSCOPE_API_KEY:
    print("ERROR: 请设置环境变量 DASHSCOPE_API_KEY")
    sys.exit(1)


# ── 工具定义 ──────────────────────────────────────────────────


def get_weather(city: str) -> dict[str, Any]:
    """查询指定城市的天气信息。"""
    data = {
        "北京": {"weather": "晴", "temperature": "5~15°C"},
        "上海": {"weather": "多云", "temperature": "12~20°C"},
    }
    return data.get(city, {"error": "暂无该城市数据"})


def get_session_state(tool_context: ToolContext) -> dict[str, Any]:
    """获取当前会话的状态信息。

    当用户询问会话状态、对话轮数、历史记录等信息时调用此工具。
    返回 OTS 中持久化的完整 session state，包括：
    - turn_count: 对话轮数
    - last_user_input: 上一轮用户输入
    - last_reply: 上一轮 agent 回复（由 output_key 自动写入）
    - app:model_name: 使用的模型名称
    - user:language: 用户语言偏好
    """
    return tool_context.state.to_dict()


# ── Step 1: 初始化 SessionStore ──────────────────────────────

store = SessionStore.from_memory_collection(MEMORY_COLLECTION_NAME)
store.init_tables()

# ── Step 2: 创建 OTSSessionService ──────────────────────────

session_service = OTSSessionService(session_store=store)

# ── Step 3: 创建 ADK Agent + Runner ─────────────────────────

custom_model = LiteLlm(
    model="openai/qwen3-max",
    api_key=DASHSCOPE_API_KEY,
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
agent = Agent(
    name="smart_assistant",
    model=custom_model,
    instruction=(
        "你是一个友好的中文智能助手。\n"
        "- 用户问天气时调用 get_weather\n"
        "- 用户询问会话状态、对话轮数、历史记录等信息时调用 get_session_state"
    ),
    tools=[get_weather, get_session_state],
    output_key="last_reply",
)

runner = Runner(
    agent=agent,
    app_name=APP_NAME,
    session_service=session_service,
)


# ── 辅助函数 ──────────────────────────────────────────────────


def _get_session_id(req: AgentRequest) -> str:
    """从请求 header 中提取 session_id，没有则生成一个。"""
    raw_headers: dict[str, str] = {}
    if hasattr(req, "raw_request") and req.raw_request:
        raw_headers = dict(req.raw_request.headers)

    return (
        raw_headers.get("X-AgentRun-Session-ID")
        or raw_headers.get("x-agentrun-session-id")
        or raw_headers.get("X-Agentrun-Session-Id")
        or f"chat_{uuid.uuid4().hex[:8]}"
    )


def _get_user_id(req: AgentRequest) -> str:
    """从请求 header 中提取 user_id，没有则使用默认值。"""
    raw_headers: dict[str, str] = {}
    if hasattr(req, "raw_request") and req.raw_request:
        raw_headers = dict(req.raw_request.headers)

    return (
        raw_headers.get("X-AgentRun-User-ID")
        or raw_headers.get("x-agentrun-user-id")
        or "default_user"
    )


async def _get_or_create_session(user_id: str, session_id: str) -> Any:
    """获取已有 session，不存在则自动创建。

    ADK Runner 需要一个已存在的 session 才能运行，
    所以首次请求时需要创建 session。
    """
    existing = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )
    if existing is not None:
        return existing

    return await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={
            "app:model_name": custom_model.model,
            "user:language": "zh-CN",
        },
    )


# ── Step 4: invoke_agent —— 核心 Server 处理函数 ─────────────


async def invoke_agent(req: AgentRequest):
    """将 AgentRequest 转换为 ADK 调用并流式输出文本。

    流程：
    1. 从 header 提取 session_id / user_id
    2. 获取或创建 ADK session（不存在则自动创建）
    3. 打印当前 session 状态（展示 OTS 持久化效果）
    4. 取最后一条用户消息，转为 ADK Content
    5. 调用 runner.run_async() 流式输出
    6. output_key="last_reply" 自动将回复写入状态
    7. 手动更新额外状态（turn_count / last_user_input）
    """
    session_id = _get_session_id(req)
    user_id = _get_user_id(req)

    # 获取或创建 session
    session = await _get_or_create_session(user_id, session_id)

    # ── 读取并展示当前 session 状态（体现 OTS 持久化能力） ────
    #
    # 首次请求时状态为初始值；后续请求可以看到上一轮的
    # turn_count、last_user_input 以及 output_key 自动写入的 last_reply。
    turn_count = session.state.get("turn_count", 0)
    print(
        f"[Session {session.id}] "
        f"turn_count={turn_count}, "
        f"last_user_input={session.state.get('last_user_input', '(无)')}, "
        f"last_reply={session.state.get('last_reply', '(无)')}"
    )

    # 提取最后一条用户消息
    last_user_text = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            last_user_text = msg.content or ""
            break

    if not last_user_text:
        yield "请输入您的问题。"
        return

    # 转换为 ADK Content 格式
    content = types.Content(
        role="user",
        parts=[types.Part(text=last_user_text)],
    )

    # 调用 ADK Runner 流式输出
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content,
        ):
            if (
                event.is_final_response()
                and event.content
                and event.content.parts
            ):
                for part in event.content.parts:
                    if part.text:
                        yield part.text

        # ── 更新额外的 session 状态 ──────────────────────────
        #
        # output_key="last_reply" 已由 ADK Runner 自动将 agent 回复
        # 写入 session.state["last_reply"]，此处额外记录：
        #   - turn_count: 对话轮数（递增）
        #   - last_user_input: 本轮用户输入
        await store.update_session_state_async(
            APP_NAME,
            user_id,
            session.id,
            {
                "turn_count": turn_count + 1,
                "last_user_input": last_user_text,
            },
        )
        print(f"[Session {session.id}] 状态已更新: turn_count={turn_count + 1}")

    except Exception as e:
        print(f"ADK Runner 执行异常: {e}")
        raise Exception("Internal Error")


# ── Step 5: 启动 Server ──────────────────────────────────────

if __name__ == "__main__":
    server = AgentRunServer(
        invoke_agent=invoke_agent, memory_collection_name=MEMORY_COLLECTION_NAME
    )
    print(f"App Name: {APP_NAME}")
    print(f"Memory Collection: {MEMORY_COLLECTION_NAME}")
    print("请求时通过 X-AgentRun-Session-ID header 指定会话 ID")
    server.start(port=9000)
