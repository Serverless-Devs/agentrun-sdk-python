"""使用 HTTP 请求头临时凭证的 Agent Server 示例

本示例演示如何在 AgentRunServer 中从每次请求的 HTTP header 获取临时凭证，
替代环境变量中可能过期的 token，确保 Agent 调用知识库等资源时凭证始终有效。

适用场景：
- 部署在函数计算（FC）上的 Agent，FC 平台通过请求头注入临时凭证
- 需要长期运行的 Agent 服务，环境变量中的 STS Token 会过期

FC 平台注入的请求头：
- x-fc-access-key-id: 临时 AccessKey ID
- x-fc-access-key-secret: 临时 AccessKey Secret
- x-fc-security-token: 临时 Security Token

启动服务后测试：
    curl http://127.0.0.1:9000/openai/v1/chat/completions -X POST \\
        -H "Content-Type: application/json" \\
        -H "x-fc-access-key-id: <your-ak>" \\
        -H "x-fc-access-key-secret: <your-sk>" \\
        -H "x-fc-security-token: <your-token>" \\
        -d '{"messages": [{"role": "user", "content": "什么是Serverless?"}], "stream": true}'
"""

import os

from agentrun.knowledgebase import KnowledgeBase
from agentrun.server import AgentRequest, AgentRunServer
from agentrun.server.model import ServerConfig
from agentrun.utils.log import logger

KNOWLEDGE_BASE_NAME = os.getenv("AGENTRUN_KNOWLEDGE_BASE", "my-knowledge-base")


async def invoke_agent(request: AgentRequest):
    user_query = request.messages[-1].content if request.messages else ""
    if not user_query:
        yield "请输入您的问题。"
        return

    try:
        kb = KnowledgeBase.get_by_name(KNOWLEDGE_BASE_NAME, config=request.config)
        result = kb.retrieve(query=user_query, config=request.config)

        nodes = result.get("nodes", result.get("data", []))
        if not nodes:
            yield f"未在知识库 {KNOWLEDGE_BASE_NAME} 中找到相关内容。"
            return

        chunks = [
            node.get("content", node.get("text", ""))
            for node in nodes
            if node.get("content") or node.get("text")
        ]
        context = "\n---\n".join(chunks)
        yield f"根据知识库检索结果：\n\n{context}"

    except Exception as e:
        logger.error("知识库检索失败: %s", e)
        yield f"知识库检索失败: {e}"


AgentRunServer(
    invoke_agent=invoke_agent,
    config=ServerConfig(cors_origins=["*"]),
).start(port=9000)
