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
SANDBOX_NAME = os.getenv("SANDBOX_NAME", "")
MEMORY_COLLECTION_NAME = os.getenv("MEMORY_COLLECTION_NAME", "mem-ots0129")

# 创建 Agent
agent = create_agent(
    # 使用 AgentRun 注册的模型
    model=model(MODEL_SERVICE, model=MODEL_NAME),
    system_prompt="""
你是一个诗人，根据用户输入内容写一个20字以内的诗文
""",
    # 使用 AgentRun 的 Sandbox 工具
    # tools=[*sandbox_toolset(SANDBOX_NAME, template_type=TemplateType.BROWSER)],
)

# 初始化 Memory Integration
memory = MemoryConversation(memory_collection_name=MEMORY_COLLECTION_NAME)


async def invoke_agent(req: AgentRequest):
    """Agent 调用函数，集成了记忆存储功能"""
    try:
        converter = AgentRunConverter()

        # 定义原始的 agent 处理函数
        async def agent_handler(request: AgentRequest):
            result = agent.astream_events(
                {
                    "messages": [
                        {"role": msg.role, "content": msg.content}
                        for msg in request.messages
                    ]
                },
                config={"recursion_limit": 1000},
            )
            async for event in result:
                for agentrun_event in converter.convert(event):
                    yield agentrun_event

        # 使用 MemoryIntegration 包装，自动存储对话历史
        async for event in memory.wrap_invoke_agent(req, agent_handler):
            yield event

    except Exception as e:
        print(f"Error in invoke_agent: {e}")
        raise Exception("Internal Error")


# 创建并启动 Server
if __name__ == "__main__":
    server = AgentRunServer(invoke_agent=invoke_agent)
    print(f"Server starting with memory collection: {MEMORY_COLLECTION_NAME}")
    print("Memory will be automatically saved to TableStore")
    server.start(port=9000)
