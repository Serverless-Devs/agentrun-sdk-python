"""AgentRun Server + LangChain Agent 示例

本示例展示了如何使用 AgentRunServer 配合 LangChain Agent 创建一个支持 OpenAI 和 AG-UI 协议的服务。

主要特性:
- 支持 OpenAI Chat Completions 协议 (POST /openai/v1/chat/completions)
- 支持 AG-UI 协议 (POST /agui/v1/run)
- 使用 LangChain Agent 进行对话
- 支持生命周期钩子（步骤事件、工具调用事件等）
- 流式和非流式响应
- **同步代码**：直接 yield hooks.on_xxx() 发送事件

使用方法:
1. 运行: python examples/a.py
2. 测试 OpenAI 协议:
   curl 127.0.0.1:9000/openai/v1/chat/completions -XPOST \
       -H "content-type: application/json" \
       -d '{"messages": [{"role": "user", "content": "现在几点了?"}], "stream": true}'

3. 测试 AG-UI 协议:
   curl 127.0.0.1:9000/agui/v1/run -XPOST \
       -H "content-type: application/json" \
       -d '{"messages": [{"role": "user", "content": "现在几点了?"}]}'
"""

from typing import Any

from langchain.agents import create_agent
import pydash

from agentrun.integration.langchain import model, sandbox_toolset
from agentrun.sandbox import TemplateType
from agentrun.server import AgentRequest, AgentRunServer
from agentrun.utils.log import logger

# 请替换为您已经创建的 模型 和 沙箱 名称
MODEL_NAME = "sdk-test-model-service"
SANDBOX_NAME = "<your-sandbox-name>"

if MODEL_NAME.startswith("<"):
    raise ValueError("请将 MODEL_NAME 替换为您已经创建的模型名称")

code_interpreter_tools = []
if SANDBOX_NAME and not SANDBOX_NAME.startswith("<"):
    code_interpreter_tools = sandbox_toolset(
        template_name=SANDBOX_NAME,
        template_type=TemplateType.CODE_INTERPRETER,
        sandbox_idle_timeout_seconds=300,
    )
else:
    logger.warning("SANDBOX_NAME 未设置或未替换，跳过加载沙箱工具。")


def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间

    Args:
        timezone: 时区，默认为 Asia/Shanghai

    Returns:
        当前时间的字符串表示
    """
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


agent = create_agent(
    model=model(MODEL_NAME),
    tools=[*code_interpreter_tools, get_current_time],
    system_prompt="你是一个 AgentRun 的 AI 专家，可以通过沙箱运行代码来回答用户的问题。",
)


def invoke_agent(request: AgentRequest):
    """Agent 调用处理函数（同步版本）

    Args:
        request: AgentRequest 对象，包含：
            - messages: 对话历史消息列表
            - stream: 是否流式输出
            - raw_headers: 原始 HTTP 请求头
            - raw_body: 原始 HTTP 请求体
            - hooks: 生命周期钩子

    Yields:
        流式输出的内容字符串或事件
    """
    hooks = request.hooks
    content = request.messages[0].content
    input_data: Any = {"messages": [{"role": "user", "content": content}]}

    try:
        # 发送步骤开始事件（直接 yield，AG-UI 会发送 STEP_STARTED 事件）
        yield hooks.on_step_start("langchain_agent")

        if request.stream:
            # 流式响应
            result = agent.stream(input_data, stream_mode="messages")
            for chunk in result:
                # 处理工具调用事件
                tool_calls = pydash.get(chunk, "[0].tool_calls", [])
                for tool_call in tool_calls:
                    tool_call_id = tool_call.get("id")
                    tool_name = pydash.get(tool_call, "function.name")
                    tool_args = pydash.get(tool_call, "function.arguments")

                    if tool_call_id and tool_name:
                        # 发送工具调用事件
                        yield hooks.on_tool_call_start(
                            id=tool_call_id, name=tool_name
                        )
                    if tool_call_id and tool_args:
                        yield hooks.on_tool_call_args(
                            id=tool_call_id, args=tool_args
                        )

                # 处理文本内容
                chunk_content = pydash.get(chunk, "[0].content")
                if chunk_content:
                    yield chunk_content
        else:
            # 非流式响应
            result = agent.invoke(input_data)
            response = pydash.get(result, "messages.-1.content")
            if response:
                yield response

        # 发送步骤结束事件
        yield hooks.on_step_finish("langchain_agent")

    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error("调用出错: %s", e)

        # 发送错误事件
        yield hooks.on_run_error(str(e), "AGENT_ERROR")

        raise e


# 启动服务器
AgentRunServer(invoke_agent=invoke_agent).start()

"""
# 测试 OpenAI 协议（流式）
curl 127.0.0.1:9000/openai/v1/chat/completions -XPOST \
    -H "content-type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "写一段代码,查询现在是几点?"}], 
        "stream": true
    }'

# 测试 AG-UI 协议
curl 127.0.0.1:9000/agui/v1/run -XPOST \
    -H "content-type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "现在几点了?"}]
    }' -N

# 测试健康检查
curl 127.0.0.1:9000/agui/v1/health
curl 127.0.0.1:9000/openai/v1/models
"""
