"""AgentRun Server 快速开始示例

curl http://127.0.0.1:9000/openai/v1/chat/completions -X POST \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "写一段代码,查询现在是几点?"}], "stream": true}'
"""

import os
from typing import Any

from langchain.agents import create_agent
import pydash

from agentrun.integration.langchain import model, sandbox_toolset
from agentrun.integration.langgraph.agent_converter import AgentRunConverter
from agentrun.sandbox import TemplateType
from agentrun.server import AgentRequest, AgentRunServer
from agentrun.server.model import ServerConfig
from agentrun.utils.log import logger

# 请替换为您已经创建的 模型 和 沙箱 名称
AGENTRUN_MODEL_NAME = os.getenv("AGENTRUN_MODEL_NAME", "<your-model-name>")
SANDBOX_NAME = "<your-sandbox-name>"

if AGENTRUN_MODEL_NAME.startswith("<") or not AGENTRUN_MODEL_NAME:
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


def get_weather_tool():
    """
    获取天气工具"""
    import time

    logger.debug("调用获取天气工具")
    time.sleep(5)
    return {"weather": "晴天，25度"}


agent = create_agent(
    model=model(AGENTRUN_MODEL_NAME),
    tools=[
        *code_interpreter_tools,
        get_weather_tool,
    ],
    system_prompt="你是一个 AgentRun 的 AI 专家，可以通过沙箱运行代码来回答用户的问题。",
)


async def invoke_agent(request: AgentRequest):
    input: Any = {
        "messages": [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
    }

    converter = AgentRunConverter()
    if request.stream:

        async def async_generator():
            async for event in agent.astream(input, stream_mode="updates"):
                for item in converter.convert(event):
                    yield item

        return async_generator()
    else:
        result = await agent.ainvoke(input)
        return pydash.get(result, "messages[-1].content", "")


AgentRunServer(
    invoke_agent=invoke_agent,
    config=ServerConfig(
        cors_origins=[
            "*"
        ]  # 部署在 AgentRun 上时，AgentRun 已经自动为你处理了跨域问题，可以省略这一行
    ),
).start()
