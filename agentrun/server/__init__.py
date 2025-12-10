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

Example (使用生命周期钩子):
>>> def invoke_agent(request: AgentRequest):
...     hooks = request.hooks
...
...     # 发送步骤开始事件
...     yield hooks.on_step_start("processing")
...
...     # 流式输出内容
...     yield "Hello, "
...     yield "world!"
...
...     # 发送步骤结束事件
...     yield hooks.on_step_finish("processing")

Example (工具调用事件):
>>> def invoke_agent(request: AgentRequest):
...     hooks = request.hooks
...
...     # 工具调用开始
...     yield hooks.on_tool_call_start(id="call_1", name="get_time")
...     yield hooks.on_tool_call_args(id="call_1", args={"timezone": "UTC"})
...
...     # 执行工具
...     result = "2024-01-01 12:00:00"
...
...     # 工具调用结果
...     yield hooks.on_tool_call_result(id="call_1", result=result)
...     yield hooks.on_tool_call_end(id="call_1")
...
...     yield f"当前时间: {result}"

Example (访问原始请求):
>>> def invoke_agent(request: AgentRequest):
...     # 访问原始请求头
...     auth = request.raw_headers.get("Authorization")
...
...     # 访问原始请求体
...     custom_field = request.raw_body.get("custom_field")
...
...     return "Hello, world!"
"""

from .agui_protocol import (
    AGUIBaseEvent,
    AGUICustomEvent,
    AGUIEvent,
    AGUIEventType,
    AGUILifecycleHooks,
    AGUIMessage,
    AGUIMessagesSnapshotEvent,
    AGUIProtocolHandler,
    AGUIRawEvent,
    AGUIRole,
    AGUIRunAgentInput,
    AGUIRunErrorEvent,
    AGUIRunFinishedEvent,
    AGUIRunStartedEvent,
    AGUIStateDeltaEvent,
    AGUIStateSnapshotEvent,
    AGUIStepFinishedEvent,
    AGUIStepStartedEvent,
    AGUITextMessageContentEvent,
    AGUITextMessageEndEvent,
    AGUITextMessageStartEvent,
    AGUIToolCallArgsEvent,
    AGUIToolCallEndEvent,
    AGUIToolCallResultEvent,
    AGUIToolCallStartEvent,
    create_agui_event,
)
from .model import (
    AgentEvent,
    AgentLifecycleHooks,
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
from .openai_protocol import OpenAILifecycleHooks, OpenAIProtocolHandler
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
    # Lifecycle Hooks & Events
    "AgentLifecycleHooks",
    "AgentEvent",
    # Protocol Base
    "ProtocolHandler",
    "BaseProtocolHandler",
    # Protocol - OpenAI
    "OpenAIProtocolHandler",
    "OpenAILifecycleHooks",
    # Protocol - AG-UI
    "AGUIProtocolHandler",
    "AGUILifecycleHooks",
    "AGUIEventType",
    "AGUIRole",
    "AGUIBaseEvent",
    "AGUIEvent",
    "AGUIRunStartedEvent",
    "AGUIRunFinishedEvent",
    "AGUIRunErrorEvent",
    "AGUIStepStartedEvent",
    "AGUIStepFinishedEvent",
    "AGUITextMessageStartEvent",
    "AGUITextMessageContentEvent",
    "AGUITextMessageEndEvent",
    "AGUIToolCallStartEvent",
    "AGUIToolCallArgsEvent",
    "AGUIToolCallEndEvent",
    "AGUIToolCallResultEvent",
    "AGUIStateSnapshotEvent",
    "AGUIStateDeltaEvent",
    "AGUIMessagesSnapshotEvent",
    "AGUIRawEvent",
    "AGUICustomEvent",
    "AGUIMessage",
    "AGUIRunAgentInput",
    "create_agui_event",
]
