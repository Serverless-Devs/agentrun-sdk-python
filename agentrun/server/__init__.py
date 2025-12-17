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
>>> from agentrun.server import AgentEvent, EventType
>>>
>>> async def invoke_agent(request: AgentRequest):
...     # 发送自定义事件（如步骤开始）
...     yield AgentEvent(
...         event=EventType.CUSTOM,
...         data={"name": "step_started", "value": {"step": "processing"}}
...     )
...
...     # 流式输出内容
...     yield "Hello, "
...     yield "world!"
...
...     # 发送步骤结束事件
...     yield AgentEvent(
...         event=EventType.CUSTOM,
...         data={"name": "step_finished", "value": {"step": "processing"}}
...     )

Example (工具调用事件):
>>> async def invoke_agent(request: AgentRequest):
...     # 完整工具调用
...     yield AgentEvent(
...         event=EventType.TOOL_CALL,
...         data={"id": "call_1", "name": "get_time", "args": '{"timezone": "UTC"}'}
...     )
...
...     # 执行工具
...     result = "2024-01-01 12:00:00"
...
...     # 工具调用结果
...     yield AgentEvent(
...         event=EventType.TOOL_RESULT,
...         data={"id": "call_1", "result": result}
...     )
...
...     yield f"当前时间: {result}"

Example (流式工具输出):
>>> async def invoke_agent(request: AgentRequest):
...     # 发起工具调用
...     yield AgentEvent(
...         event=EventType.TOOL_CALL,
...         data={"id": "call_1", "name": "run_code", "args": '{"code": "..."}'}
...     )
...
...     # 流式输出执行过程
...     yield AgentEvent(
...         event=EventType.TOOL_RESULT_CHUNK,
...         data={"id": "call_1", "delta": "Step 1: Compiling...\\n"}
...     )
...     yield AgentEvent(
...         event=EventType.TOOL_RESULT_CHUNK,
...         data={"id": "call_1", "delta": "Step 2: Running...\\n"}
...     )
...
...     # 最终结果（标识流式输出结束）
...     yield AgentEvent(
...         event=EventType.TOOL_RESULT,
...         data={"id": "call_1", "result": "Execution completed."}
...     )

Example (HITL - 请求人类介入):
>>> async def invoke_agent(request: AgentRequest):
...     # 请求用户确认
...     yield AgentEvent(
...         event=EventType.HITL,
...         data={
...             "id": "hitl_1",
...             "tool_call_id": "call_delete",  # 可选
...             "type": "confirmation",
...             "prompt": "确认删除文件?",
...             "options": ["确认", "取消"]
...         }
...     )
...     # 用户响应将通过下一轮对话的 messages 传回

Example (访问原始请求):
>>> async def invoke_agent(request: AgentRequest):
...     # 访问当前协议
...     protocol = request.protocol  # "openai" 或 "agui"
...
...     # 访问原始请求头
...     auth = request.raw_request.headers.get("Authorization")
...
...     # 访问查询参数
...     params = request.raw_request.query_params
...
...     # 访问客户端 IP
...     client_ip = request.raw_request.client.host if request.raw_request.client else None
...
...     return "Hello, world!"
"""

from ..utils.helper import MergeOptions
from .agui_normalizer import AguiEventNormalizer
from .agui_protocol import AGUIProtocolHandler
from .model import (
    AgentEvent,
    AgentEventItem,
    AgentRequest,
    AgentResult,
    AgentResultItem,
    AgentReturnType,
    AGUIProtocolConfig,
    AsyncAgentEventGenerator,
    AsyncAgentResultGenerator,
    EventType,
    Message,
    MessageRole,
    OpenAIProtocolConfig,
    ProtocolConfig,
    ServerConfig,
    SyncAgentEventGenerator,
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
    "AGUIProtocolConfig",
    # Request/Response Models
    "AgentRequest",
    "AgentEvent",
    "AgentResult",  # 兼容别名
    "Message",
    "MessageRole",
    "Tool",
    "ToolCall",
    # Event Types
    "EventType",
    # Type Aliases
    "AgentEventItem",
    "AgentResultItem",  # 兼容别名
    "AgentReturnType",
    "SyncAgentEventGenerator",
    "SyncAgentResultGenerator",  # 兼容别名
    "AsyncAgentEventGenerator",
    "AsyncAgentResultGenerator",  # 兼容别名
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
    # Event Normalizer
    "AguiEventNormalizer",
    # Helpers
    "MergeOptions",
]
