"""AgentRun Server 模块 / AgentRun Server Module

提供 HTTP Server 集成能力,支持符合 AgentRun 规范的 Agent 调用接口。

Example (基本使用):
>>> from agentrun.server import AgentRunServer, AgentRequest, AgentResponse
>>>
>>> def invoke_agent(request: AgentRequest) -> AgentResponse:
...     # 实现你的 Agent 逻辑
...     return AgentResponse(...)
>>>
>>> server = AgentRunServer(invoke_agent=invoke_agent)
>>> server.start(host="0.0.0.0", port=8080)

Example (异步处理):
>>> async def invoke_agent(request: AgentRequest) -> AgentResponse:
...     # 异步实现你的 Agent 逻辑
...     return AgentResponse(...)
>>>
>>> server = AgentRunServer(invoke_agent=invoke_agent)
>>> server.start()

Example (流式响应):
>>> async def invoke_agent(request: AgentRequest):
...     if request.stream:
...         async def stream():
...             for chunk in generate_chunks():
...                 yield AgentStreamResponse(...)
...         return stream()
...     return AgentResponse(...)"""

from .model import (
    AgentRequest,
    AgentResponse,
    AgentResponseChoice,
    AgentResponseUsage,
    AgentResult,
    AgentRunResult,
    AgentStreamIterator,
    AgentStreamResponse,
    AgentStreamResponseChoice,
    AgentStreamResponseDelta,
    Message,
    MessageRole,
    Tool,
    ToolCall,
)
from .openai_protocol import OpenAIProtocolHandler
from .protocol import (
    AsyncInvokeAgentHandler,
    InvokeAgentHandler,
    ProtocolHandler,
    SyncInvokeAgentHandler,
)
from .server import AgentRunServer

__all__ = [
    # Server
    "AgentRunServer",
    # Request/Response Models
    "AgentRequest",
    "AgentResponse",
    "AgentResponseChoice",
    "AgentResponseUsage",
    "AgentRunResult",
    "AgentStreamResponse",
    "AgentStreamResponseChoice",
    "AgentStreamResponseDelta",
    "Message",
    "MessageRole",
    "Tool",
    "ToolCall",
    # Type Aliases
    "AgentResult",
    "AgentStreamIterator",
    "InvokeAgentHandler",
    "AsyncInvokeAgentHandler",
    "SyncInvokeAgentHandler",
    # Protocol
    "ProtocolHandler",
    "OpenAIProtocolHandler",
]
