"""AgentRun Server 快速开始示例

curl http://127.0.0.1:9000/openai/v1/chat/completions -X POST \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "什么是Serverless?"}], "stream": true}'
"""

import json
import os
from typing import Any

from langchain.agents import create_agent
import pydash

from agentrun import Config
from agentrun.integration.langchain import (
    knowledgebase_toolset,
    model,
    sandbox_toolset,
)
from agentrun.integration.langgraph.agent_converter import AgentRunConverter
from agentrun.knowledgebase import KnowledgeBase
from agentrun.sandbox import TemplateType
from agentrun.server import AgentRequest, AgentRunServer
from agentrun.server.model import ServerConfig
from agentrun.utils.log import logger

# 请替换为您已经创建的 模型 和 沙箱 名称
AGENTRUN_MODEL_SERVICE = os.getenv(
    "AGENTRUN_MODEL_SERVICE", "<your-model-service>"
)
AGENTRUN_MODEL_NAME = os.getenv("AGENTRUN_MODEL_NAME", "<your-model-name>")
SANDBOX_NAME = "<your-sandbox-name>"
KNOWLEDGE_BASES = os.getenv(
    "AGENTRUN_KNOWLEDGE_BASES", "<your-knowledge-bases>"
).split(",")

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

## 加载知识库工具，知识库可以以工具的方式供Agent进行调用
knowledgebase_tools = []
if KNOWLEDGE_BASES and not KNOWLEDGE_BASES[0].startswith("<"):
    knowledgebase_tools = knowledgebase_toolset(
        knowledge_base_names=KNOWLEDGE_BASES,
    )
else:
    logger.warning("KNOWLEDGE_BASES 未设置或未替换，跳过加载知识库工具。")


def get_weather_tool():
    """
    获取天气工具"""
    import time

    logger.debug("调用获取天气工具")
    time.sleep(5)
    return {"weather": "晴天，25度"}


agent = create_agent(
    model=model(
        AGENTRUN_MODEL_SERVICE,
        model=AGENTRUN_MODEL_NAME,
        config=Config(timeout=180),
    ),
    tools=[
        *code_interpreter_tools,
        *knowledgebase_tools,  ## 通过工具集成知识库查询能力
        get_weather_tool,
    ],
    system_prompt=(
        "你是一个 AgentRun 的 AI 专家，"
        "可以通过沙箱运行代码和查询知识库文档来回答用户的问题。"
    ),
)


async def invoke_agent(request: AgentRequest):
    messages = [
        {"role": msg.role, "content": msg.content} for msg in request.messages
    ]

    # 如果配置了知识库，查询知识库并将结果添加到上下文
    if KNOWLEDGE_BASES and not KNOWLEDGE_BASES[0].startswith("<"):
        # 获取用户最新的消息内容作为查询
        user_query = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content
                break

        if user_query:
            try:
                retrieve_result = await KnowledgeBase.multi_retrieve_async(
                    query=user_query,
                    knowledge_base_names=KNOWLEDGE_BASES,
                )
                # 直接将检索结果添加到上下文
                if retrieve_result:
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps(
                            retrieve_result, ensure_ascii=False
                        ),
                    })
            except Exception as e:
                logger.warning(f"知识库检索失败: {e}")

    input: Any = {"messages": messages}

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
    config=ServerConfig(cors_origins=["*"]),
).start()
