"""AgentRun Server 模块 / AgentRun Server Module

提供 HTTP Server 集成能力，支持符合 AgentRun 规范的 Agent 调用接口。
支持 OpenAI Chat Completions 和 AG-UI 两种协议。

Example (基本使用 - 返回字符串):
>>> from agentrun.server import AgentRunServer, AgentRequest
>>>
>>> def invoke_agent(request: AgentRequest):
...     return "Hello, world!"
>>>
>>> server = AgentRunServer(invoke_agent=invoke_agent)
>>> server.start(port=9000)

Example (流式输出):
>>> def invoke_agent(request: AgentRequest):
...     for word in ["Hello", ", ", "world", "!"]:
...         yield word
>>>
>>> AgentRunServer(invoke_agent=invoke_agent).start()

Example (使用事件):
>>> from agentrun.server import AgentResult, EventType
>>>
>>> async def invoke_agent(request: AgentRequest):
...     # 发送步骤开始事件
...     yield AgentResult(
...         event=EventType.STEP_STARTED,
...         data={"step_name": "processing"}
...     )
...
...     # 流式输出内容
...     yield "Hello, "
...     yield "world!"
...
...     # 发送步骤结束事件
...     yield AgentResult(
...         event=EventType.STEP_FINISHED,
...         data={"step_name": "processing"}
...     )

Example (工具调用事件):
>>> async def invoke_agent(request: AgentRequest):
...     # 工具调用开始
...     yield AgentResult(
...         event=EventType.TOOL_CALL_START,
...         data={"tool_call_id": "call_1", "tool_call_name": "get_time"}
...     )
...     yield AgentResult(
...         event=EventType.TOOL_CALL_ARGS,
...         data={"tool_call_id": "call_1", "delta": '{"timezone": "UTC"}'}
...     )
...
...     # 执行工具
...     result = "2024-01-01 12:00:00"
...
...     # 工具调用结果
...     yield AgentResult(
...         event=EventType.TOOL_CALL_RESULT,
...         data={"tool_call_id": "call_1", "result": result}
...     )
...     yield AgentResult(
...         event=EventType.TOOL_CALL_END,
...         data={"tool_call_id": "call_1"}
...     )
...
...     yield f"当前时间: {result}"

Example (访问原始请求):
>>> def invoke_agent(request: AgentRequest):
...     # 访问原始请求头
...     auth = request.headers.get("Authorization")
...
...     # 访问原始请求体
...     custom_field = request.body.get("custom_field")
...
...     return "Hello, world!"
"""

from .agui_protocol import AGUIProtocolHandler
from .model import (
    AdditionMode,
    AgentRequest,
    AgentResult,
    AgentResultItem,
    AgentReturnType,
    AsyncAgentResultGenerator,
    EventType,
    Message,
    MessageRole,
    OpenAIProtocolConfig,
    ProtocolConfig,
    ServerConfig,
    SyncAgentResultGenerator,
    Tool,
    ToolCall,
)
from .openai_protocol import OpenAIProtocolHandler
from .protocol import (
    AsyncInvokeAgentHandler,
    BaseProtocolHandler,
    InvokeAgentHandler,
    ProtocolHandler,
    SyncInvokeAgentHandler,
)
from .server import AgentRunServer

__all__ = [
    # Server
    "AgentRunServer",
    # Config
    "ServerConfig",
    "ProtocolConfig",
    "OpenAIProtocolConfig",
    # Request/Response Models
    "AgentRequest",
    "AgentResult",
    "Message",
    "MessageRole",
    "Tool",
    "ToolCall",
    # Event Types
    "EventType",
    "AdditionMode",
    # Type Aliases
    "AgentResultItem",
    "AgentReturnType",
    "SyncAgentResultGenerator",
    "AsyncAgentResultGenerator",
    "InvokeAgentHandler",
    "AsyncInvokeAgentHandler",
    "SyncInvokeAgentHandler",
    # Protocol Base
    "ProtocolHandler",
    "BaseProtocolHandler",
    # Protocol - OpenAI
    "OpenAIProtocolHandler",
    # Protocol - AG-UI
    "AGUIProtocolHandler",
]
