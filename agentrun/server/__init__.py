"""AgentRun Server 模块 / AgentRun Server Module

提供 HTTP Server 集成能力,支持符合 AgentRun 规范的 Agent 调用接口。
支持 OpenAI Chat Completions 和 AG-UI 两种协议。

Example (基本使用 - 同步):
>>> from agentrun.server import AgentRunServer, AgentRequest
>>>
>>> def invoke_agent(request: AgentRequest):
...     # 实现你的 Agent 逻辑
...     return "Hello, world!"
>>>
>>> server = AgentRunServer(invoke_agent=invoke_agent)
>>> server.start(host="0.0.0.0", port=8080)

Example (使用生命周期钩子 - 同步，推荐):
>>> def invoke_agent(request: AgentRequest):
...     hooks = request.hooks
...
...     # 发送步骤开始事件 (使用 emit_* 同步方法)
...     yield hooks.emit_step_start("processing")
...
...     # 处理逻辑...
...     yield "Hello, "
...     yield "world!"
...
...     # 发送步骤结束事件
...     yield hooks.emit_step_finish("processing")

Example (使用生命周期钩子 - 异步):
>>> async def invoke_agent(request: AgentRequest):
...     hooks = request.hooks
...
...     # 发送步骤开始事件 (使用 on_* 异步方法)
...     async for event in hooks.on_step_start("processing"):
...         yield event
...
...     # 处理逻辑...
...     yield "Hello, world!"
...
...     # 发送步骤结束事件
...     async for event in hooks.on_step_finish("processing"):
...         yield event

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
