"""Google ADK Agent —— 使用 OTSSessionService 持久化会话。

集成步骤：
  Step 1: 初始化 SessionStore（OTS 后端）
  Step 2: 创建 OTSSessionService
  Step 3: 创建 ADK Agent + Runner，传入 session_service
  Step 4: 通过 runner.run_async() 对话（自动持久化）

使用方式：
  export MEMORY_COLLECTION_NAME="your-collection-name"
  export DASHSCOPE_API_KEY="your-dashscope-api-key"
  uv run python examples/conversation_service_adk_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent  # type: ignore[import-untyped]
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner  # type: ignore[import-untyped]
from google.genai import types  # type: ignore[import-untyped]

from agentrun.conversation_service import SessionStore
from agentrun.conversation_service.adapters import OTSSessionService

load_dotenv()

APP_NAME = "adk_chat_demo"
USER_ID = "demo_user"
# ADK 通过 LiteLLM 调用 DashScope OpenAI 兼容接口
# 需设置: DASHSCOPE_API_KEY


# ── 工具定义 ──────────────────────────────────────────────────


def get_weather(city: str) -> dict[str, Any]:
    """查询指定城市的天气信息。"""
    data = {
        "北京": {"weather": "晴", "temperature": "5~15°C"},
        "上海": {"weather": "多云", "temperature": "12~20°C"},
    }
    return data.get(city, {"error": "暂无该城市数据"})


# ── Step 1: 初始化 SessionStore ──────────────────────────────

memory_collection_name = os.environ.get("MEMORY_COLLECTION_NAME", "")
if not memory_collection_name:
    print("ERROR: 请设置环境变量 MEMORY_COLLECTION_NAME")
    sys.exit(1)

store = SessionStore.from_memory_collection(memory_collection_name)
store.init_tables()

# ── Step 2: 创建 OTSSessionService ──────────────────────────

session_service = OTSSessionService(session_store=store)

# ── Step 3: 创建 Agent + Runner ─────────────────────────────

custom_model = LiteLlm(
    model="openai/qwen3-max",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
agent = Agent(
    name="smart_assistant",
    model=custom_model,
    instruction="你是一个友好的中文智能助手，用户问天气时调用 get_weather。",
    tools=[get_weather],
)

runner = Runner(
    agent=agent,
    app_name=APP_NAME,
    session_service=session_service,
)


# ── Step 4: 对话（自动持久化到 OTS） ────────────────────────


async def chat(session_id: str, text: str) -> str:
    """发送消息并返回 Agent 回复。"""
    content = types.Content(
        role="user",
        parts=[types.Part(text=text)],
    )
    reply_parts: list[str] = []
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    reply_parts.append(part.text)
    return "\n".join(reply_parts)


async def main() -> None:
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        state={"app:model_name": custom_model.model, "user:language": "zh-CN"},
    )
    print(f"会话已创建: {session.id}\n输入 /quit 退出\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input == "/quit":
            break
        reply = await chat(session.id, user_input)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
