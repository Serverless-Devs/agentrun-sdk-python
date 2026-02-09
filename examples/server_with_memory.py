"""AgentRun Server with Memory Integration Example / 带记忆集成的 AgentRun Server 示例

演示如何使用 MemoryIntegration 自动存储对话历史到 TableStore。

运行前需要设置环境变量：
    export AGENTRUN_ACCESS_KEY_ID="your-access-key-id"
    export AGENTRUN_ACCESS_KEY_SECRET="your-access-key-secret"
    export AGENTRUN_REGION="cn-hangzhou"
    export MODEL_SERVICE="your-model-service"
    export MODEL_NAME="qwen3-max"
    export SANDBOX_NAME="your-sandbox"
    export MEMORY_COLLECTION_NAME="your-memory-collection"

运行示例：
    uv run python examples/server_with_memory.py
"""

import os

from langchain.agents import create_agent

from agentrun import AgentRequest
from agentrun.integration.langchain import (
    AgentRunConverter,
    model,
    sandbox_toolset,
)
from agentrun.memory_collection import MemoryConversation
from agentrun.sandbox import TemplateType
from agentrun.server import AgentRunServer

# 配置参数
MODEL_SERVICE = os.getenv("MODEL_SERVICE", "qwen3-max")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-max")
SANDBOX_NAME = os.getenv("SANDBOX_NAME", "sandbox-browser-BmUyyD")
MEMORY_COLLECTION_NAME = os.getenv("MEMORY_COLLECTION_NAME", "mem-ots0129")

# 创建 Agent
agent = create_agent(
    # 使用 AgentRun 注册的模型
    model=model(MODEL_SERVICE, model=MODEL_NAME),
    system_prompt="""
你是 AgentRun 的 AI 助手，可以通过网络搜索帮助用户解决问题


你的工作流程如下
- 当用户向你提问概念性问题时，不要直接回答，而是先进行网络搜索
- 使用 Browser 工具打开百度搜索。如果要搜索 AgentRun，对应的搜索链接为: `https://www.baidu.com/s?ie=utf-8&wd=agentrun`。为了节省 token 使用，不要使用 `snapshot` 获取完整页面内容，而是通过 `evaluate` 获取你需要的部分
- 获取百度搜索的结果，根据相关性分别打开子页面获取内容
    - 如果子页面的相关度较低，则可以直接忽略
    - 如果子页面的相关度较高，则将其记录为可参考的资料，记录页面标题和实时的 url
- 当你获得至少 3 条网络信息后，可以结束搜索，并根据搜索到的结果回答用户的问题。
- 如果某一部分回答引用了网络的信息，需要进行标注，并在回答的最后给出跳转链接
""",
    # 使用 AgentRun 的 Sandbox 工具
    tools=[*sandbox_toolset(SANDBOX_NAME, template_type=TemplateType.BROWSER)],
)

# 初始化 Memory Integration
memory = MemoryConversation(memory_collection_name=MEMORY_COLLECTION_NAME)


async def invoke_agent(req: AgentRequest):
    """Agent 调用函数"""
    try:
        converter = AgentRunConverter()
        result = agent.astream_events(
            {
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in req.messages
                ]
            },
            config={"recursion_limit": 1000},
        )
        async for event in result:
            for agentrun_event in converter.convert(event):
                yield agentrun_event
    except Exception as e:
        print(e)
        raise Exception("Internal Error")


# 创建并启动 Server
if __name__ == "__main__":
    server = AgentRunServer(
        invoke_agent=invoke_agent, memory_collection_name=MEMORY_COLLECTION_NAME
    )
    print(f"Server starting with memory collection: {MEMORY_COLLECTION_NAME}")
    print("Memory will be automatically saved to TableStore")
    server.start(port=9000)
