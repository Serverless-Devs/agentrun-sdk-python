"""AG-UI (Agent-User Interaction Protocol) 协议实现

AG-UI 是一种开源、轻量级、基于事件的协议，用于标准化 AI Agent 与前端应用之间的交互。
参考: https://docs.ag-ui.com/

基于 Router 的设计:
- 协议自己创建 FastAPI Router
- 定义所有端点和处理逻辑
- Server 只需挂载 Router

生命周期钩子:
- AG-UI 完整支持所有生命周期事件
- 每个钩子映射到对应的 AG-UI 事件类型
"""

from enum import Enum
import json
import time
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    TYPE_CHECKING,
    Union,
)
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .model import (
    AgentEvent,
    AgentLifecycleHooks,
    AgentRequest,
    AgentResponse,
    AgentResult,
    AgentRunResult,
    Message,
    MessageRole,
)
from .protocol import BaseProtocolHandler

if TYPE_CHECKING:
    from .invoker import AgentInvoker


# ============================================================================
# AG-UI 事件类型定义
# ============================================================================


class AGUIEventType(str, Enum):
    """AG-UI 事件类型

    参考: https://docs.ag-ui.com/concepts/events
    """

    # Lifecycle Events (生命周期事件)
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"

    # Text Message Events (文本消息事件)
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"

    # Tool Call Events (工具调用事件)
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"

    # State Events (状态事件)
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"

    # Message Events (消息事件)
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"

    # Special Events (特殊事件)
    RAW = "RAW"
    CUSTOM = "CUSTOM"


class AGUIRole(str, Enum):
    """AG-UI 消息角色"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


# ============================================================================
# AG-UI 事件模型
# ============================================================================


class AGUIBaseEvent(BaseModel):
    """AG-UI 基础事件"""

    type: AGUIEventType
    timestamp: Optional[int] = Field(
        default_factory=lambda: int(time.time() * 1000)
    )
    rawEvent: Optional[Dict[str, Any]] = None


class AGUIRunStartedEvent(AGUIBaseEvent):
    """运行开始事件"""

    type: AGUIEventType = AGUIEventType.RUN_STARTED
    threadId: Optional[str] = None
    runId: Optional[str] = None


class AGUIRunFinishedEvent(AGUIBaseEvent):
    """运行结束事件"""

    type: AGUIEventType = AGUIEventType.RUN_FINISHED
    threadId: Optional[str] = None
    runId: Optional[str] = None


class AGUIRunErrorEvent(AGUIBaseEvent):
    """运行错误事件"""

    type: AGUIEventType = AGUIEventType.RUN_ERROR
    message: str
    code: Optional[str] = None


class AGUIStepStartedEvent(AGUIBaseEvent):
    """步骤开始事件"""

    type: AGUIEventType = AGUIEventType.STEP_STARTED
    stepName: Optional[str] = None


class AGUIStepFinishedEvent(AGUIBaseEvent):
    """步骤结束事件"""

    type: AGUIEventType = AGUIEventType.STEP_FINISHED
    stepName: Optional[str] = None


class AGUITextMessageStartEvent(AGUIBaseEvent):
    """文本消息开始事件"""

    type: AGUIEventType = AGUIEventType.TEXT_MESSAGE_START
    messageId: str
    role: AGUIRole = AGUIRole.ASSISTANT


class AGUITextMessageContentEvent(AGUIBaseEvent):
    """文本消息内容事件"""

    type: AGUIEventType = AGUIEventType.TEXT_MESSAGE_CONTENT
    messageId: str
    delta: str


class AGUITextMessageEndEvent(AGUIBaseEvent):
    """文本消息结束事件"""

    type: AGUIEventType = AGUIEventType.TEXT_MESSAGE_END
    messageId: str


class AGUIToolCallStartEvent(AGUIBaseEvent):
    """工具调用开始事件"""

    type: AGUIEventType = AGUIEventType.TOOL_CALL_START
    toolCallId: str
    toolCallName: str
    parentMessageId: Optional[str] = None


class AGUIToolCallArgsEvent(AGUIBaseEvent):
    """工具调用参数事件"""

    type: AGUIEventType = AGUIEventType.TOOL_CALL_ARGS
    toolCallId: str
    delta: str


class AGUIToolCallEndEvent(AGUIBaseEvent):
    """工具调用结束事件"""

    type: AGUIEventType = AGUIEventType.TOOL_CALL_END
    toolCallId: str


class AGUIToolCallResultEvent(AGUIBaseEvent):
    """工具调用结果事件"""

    type: AGUIEventType = AGUIEventType.TOOL_CALL_RESULT
    toolCallId: str
    result: str


class AGUIStateSnapshotEvent(AGUIBaseEvent):
    """状态快照事件"""

    type: AGUIEventType = AGUIEventType.STATE_SNAPSHOT
    snapshot: Dict[str, Any]


class AGUIStateDeltaEvent(AGUIBaseEvent):
    """状态增量事件"""

    type: AGUIEventType = AGUIEventType.STATE_DELTA
    delta: List[Dict[str, Any]]  # JSON Patch 格式


class AGUIMessage(BaseModel):
    """AG-UI 消息格式"""

    id: str
    role: AGUIRole
    content: Optional[str] = None
    name: Optional[str] = None
    toolCalls: Optional[List[Dict[str, Any]]] = None
    toolCallId: Optional[str] = None


class AGUIMessagesSnapshotEvent(AGUIBaseEvent):
    """消息快照事件"""

    type: AGUIEventType = AGUIEventType.MESSAGES_SNAPSHOT
    messages: List[AGUIMessage]


class AGUIRawEvent(AGUIBaseEvent):
    """原始事件"""

    type: AGUIEventType = AGUIEventType.RAW
    event: Dict[str, Any]


class AGUICustomEvent(AGUIBaseEvent):
    """自定义事件"""

    type: AGUIEventType = AGUIEventType.CUSTOM
    name: str
    value: Any


# 事件联合类型
AGUIEvent = Union[
    AGUIRunStartedEvent,
    AGUIRunFinishedEvent,
    AGUIRunErrorEvent,
    AGUIStepStartedEvent,
    AGUIStepFinishedEvent,
    AGUITextMessageStartEvent,
    AGUITextMessageContentEvent,
    AGUITextMessageEndEvent,
    AGUIToolCallStartEvent,
    AGUIToolCallArgsEvent,
    AGUIToolCallEndEvent,
    AGUIToolCallResultEvent,
    AGUIStateSnapshotEvent,
    AGUIStateDeltaEvent,
    AGUIMessagesSnapshotEvent,
    AGUIRawEvent,
    AGUICustomEvent,
]


# ============================================================================
# AG-UI 请求模型
# ============================================================================


class AGUIRunAgentInput(BaseModel):
    """AG-UI 运行 Agent 请求"""

    threadId: Optional[str] = None
    runId: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    context: Optional[List[Dict[str, Any]]] = None
    forwardedProps: Optional[Dict[str, Any]] = None


# ============================================================================
# AG-UI 协议生命周期钩子实现
# ============================================================================


class AGUILifecycleHooks(AgentLifecycleHooks):
    """AG-UI 协议的生命周期钩子实现

    AG-UI 完整支持所有生命周期事件，每个钩子映射到对应的 AG-UI 事件类型。

    所有 on_* 方法直接返回 AgentEvent，可以直接 yield。

    Example:
        >>> def invoke_agent(request):
        ...     hooks = request.hooks
        ...     yield hooks.on_step_start("processing")
        ...     yield hooks.on_tool_call_start(id="call_1", name="get_time")
        ...     yield hooks.on_tool_call_args(id="call_1", args={"tz": "UTC"})
        ...     result = get_time()
        ...     yield hooks.on_tool_call_result(id="call_1", result=result)
        ...     yield hooks.on_tool_call_end(id="call_1")
        ...     yield f"时间: {result}"
        ...     yield hooks.on_step_finish("processing")
    """

    def __init__(self, context: Dict[str, Any]):
        """初始化钩子

        Args:
            context: 运行上下文，包含 threadId, runId 等
        """
        self.context = context
        self.thread_id = context.get("threadId", str(uuid.uuid4()))
        self.run_id = context.get("runId", str(uuid.uuid4()))

    def _create_event(self, event: AGUIBaseEvent) -> AgentEvent:
        """创建 AgentEvent

        Args:
            event: AG-UI 事件对象

        Returns:
            AgentEvent 对象
        """
        json_str = event.model_dump_json(exclude_none=True)
        raw_sse = f"data: {json_str}\n\n"
        return AgentEvent(
            event_type=event.type.value
            if hasattr(event.type, "value")
            else str(event.type),
            data=event.model_dump(exclude_none=True),
            raw_sse=raw_sse,
        )

    # =========================================================================
    # 生命周期事件方法 (on_*) - 直接返回 AgentEvent
    # =========================================================================

    def on_run_start(self) -> AgentEvent:
        """发送 RUN_STARTED 事件"""
        return self._create_event(
            AGUIRunStartedEvent(threadId=self.thread_id, runId=self.run_id)
        )

    def on_run_finish(self) -> AgentEvent:
        """发送 RUN_FINISHED 事件"""
        return self._create_event(
            AGUIRunFinishedEvent(threadId=self.thread_id, runId=self.run_id)
        )

    def on_run_error(
        self, error: str, code: Optional[str] = None
    ) -> AgentEvent:
        """发送 RUN_ERROR 事件"""
        return self._create_event(AGUIRunErrorEvent(message=error, code=code))

    def on_step_start(self, step_name: Optional[str] = None) -> AgentEvent:
        """发送 STEP_STARTED 事件"""
        return self._create_event(AGUIStepStartedEvent(stepName=step_name))

    def on_step_finish(self, step_name: Optional[str] = None) -> AgentEvent:
        """发送 STEP_FINISHED 事件"""
        return self._create_event(AGUIStepFinishedEvent(stepName=step_name))

    def on_text_message_start(
        self, message_id: str, role: str = "assistant"
    ) -> AgentEvent:
        """发送 TEXT_MESSAGE_START 事件"""
        try:
            agui_role = AGUIRole(role)
        except ValueError:
            agui_role = AGUIRole.ASSISTANT
        return self._create_event(
            AGUITextMessageStartEvent(messageId=message_id, role=agui_role)
        )

    def on_text_message_content(
        self, message_id: str, delta: str
    ) -> Optional[AgentEvent]:
        """发送 TEXT_MESSAGE_CONTENT 事件"""
        if not delta:
            return None
        return self._create_event(
            AGUITextMessageContentEvent(messageId=message_id, delta=delta)
        )

    def on_text_message_end(self, message_id: str) -> AgentEvent:
        """发送 TEXT_MESSAGE_END 事件"""
        return self._create_event(AGUITextMessageEndEvent(messageId=message_id))

    def on_tool_call_start(
        self,
        id: str,
        name: str,
        parent_message_id: Optional[str] = None,
    ) -> AgentEvent:
        """发送 TOOL_CALL_START 事件"""
        return self._create_event(
            AGUIToolCallStartEvent(
                toolCallId=id,
                toolCallName=name,
                parentMessageId=parent_message_id,
            )
        )

    def on_tool_call_args_delta(
        self, id: str, delta: str
    ) -> Optional[AgentEvent]:
        """发送 TOOL_CALL_ARGS 事件（增量）"""
        if not delta:
            return None
        return self._create_event(
            AGUIToolCallArgsEvent(toolCallId=id, delta=delta)
        )

    def on_tool_call_args(
        self, id: str, args: Union[str, Dict[str, Any]]
    ) -> AgentEvent:
        """发送完整的 TOOL_CALL_ARGS 事件"""
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        return self._create_event(
            AGUIToolCallArgsEvent(toolCallId=id, delta=args)
        )

    def on_tool_call_result_delta(
        self, id: str, delta: str
    ) -> Optional[AgentEvent]:
        """发送 TOOL_CALL_RESULT 事件（增量）"""
        if not delta:
            return None
        return self._create_event(
            AGUIToolCallResultEvent(toolCallId=id, result=delta)
        )

    def on_tool_call_result(self, id: str, result: str) -> AgentEvent:
        """发送 TOOL_CALL_RESULT 事件"""
        return self._create_event(
            AGUIToolCallResultEvent(toolCallId=id, result=result)
        )

    def on_tool_call_end(self, id: str) -> AgentEvent:
        """发送 TOOL_CALL_END 事件"""
        return self._create_event(AGUIToolCallEndEvent(toolCallId=id))

    def on_state_snapshot(self, snapshot: Dict[str, Any]) -> AgentEvent:
        """发送 STATE_SNAPSHOT 事件"""
        return self._create_event(AGUIStateSnapshotEvent(snapshot=snapshot))

    def on_state_delta(self, delta: List[Dict[str, Any]]) -> AgentEvent:
        """发送 STATE_DELTA 事件"""
        return self._create_event(AGUIStateDeltaEvent(delta=delta))

    def on_custom_event(self, name: str, value: Any) -> AgentEvent:
        """发送 CUSTOM 事件"""
        return self._create_event(AGUICustomEvent(name=name, value=value))


# ============================================================================
# AG-UI 协议处理器
# ============================================================================


class AGUIProtocolHandler(BaseProtocolHandler):
    """AG-UI 协议处理器

    实现 AG-UI (Agent-User Interaction Protocol) 兼容接口
    参考: https://docs.ag-ui.com/

    特点:
    - 基于事件的流式通信
    - 完整支持所有生命周期事件
    - 支持状态同步
    - 支持工具调用

    Example:
        >>> from agentrun.server import AgentRunServer, AGUIProtocolHandler
        >>>
        >>> server = AgentRunServer(
        ...     invoke_agent=my_agent,
        ...     protocols=[AGUIProtocolHandler()]
        ... )
        >>> server.start(port=8000)
        # 可访问: POST http://localhost:8000/agui/v1/run
    """

    def get_prefix(self) -> str:
        """AG-UI 协议建议使用 /agui/v1 前缀"""
        return "/agui/v1"

    def create_hooks(self, context: Dict[str, Any]) -> AgentLifecycleHooks:
        """创建 AG-UI 协议的生命周期钩子"""
        return AGUILifecycleHooks(context)

    def as_fastapi_router(self, agent_invoker: "AgentInvoker") -> APIRouter:
        """创建 AG-UI 协议的 FastAPI Router"""
        router = APIRouter()

        @router.post("/run")
        async def run_agent(request: Request):
            """AG-UI 运行 Agent 端点

            接收 AG-UI 格式的请求,返回 SSE 事件流。
            """
            # SSE 响应头，禁用缓冲
            sse_headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
            }

            try:
                # 1. 解析请求
                request_data = await request.json()
                agent_request, context = await self.parse_request(
                    request, request_data
                )

                # 2. 调用 Agent
                agent_result = await agent_invoker.invoke(agent_request)

                # 3. 格式化为 AG-UI 事件流
                event_stream = self.format_response(
                    agent_result, agent_request, context
                )

                # 4. 返回 SSE 流
                return StreamingResponse(
                    event_stream,
                    media_type="text/event-stream",
                    headers=sse_headers,
                )

            except ValueError as e:
                # 返回错误事件流
                return StreamingResponse(
                    self._error_stream(str(e)),
                    media_type="text/event-stream",
                    headers=sse_headers,
                )
            except Exception as e:
                return StreamingResponse(
                    self._error_stream(f"Internal error: {str(e)}"),
                    media_type="text/event-stream",
                    headers=sse_headers,
                )

        @router.get("/health")
        async def health_check():
            """健康检查端点"""
            return {"status": "ok", "protocol": "ag-ui", "version": "1.0"}

        return router

    async def parse_request(
        self,
        request: Request,
        request_data: Dict[str, Any],
    ) -> tuple[AgentRequest, Dict[str, Any]]:
        """解析 AG-UI 格式的请求

        Args:
            request: FastAPI Request 对象
            request_data: HTTP 请求体 JSON 数据

        Returns:
            tuple: (AgentRequest, context)

        Raises:
            ValueError: 请求格式不正确
        """
        # 创建上下文
        context = {
            "threadId": request_data.get("threadId") or str(uuid.uuid4()),
            "runId": request_data.get("runId") or str(uuid.uuid4()),
        }

        # 创建钩子
        hooks = self.create_hooks(context)

        # 解析消息列表
        messages = []
        raw_messages = request_data.get("messages", [])

        for msg_data in raw_messages:
            if not isinstance(msg_data, dict):
                continue

            role_str = msg_data.get("role", "user")
            try:
                role = MessageRole(role_str)
            except ValueError:
                role = MessageRole.USER

            messages.append(
                Message(
                    role=role,
                    content=msg_data.get("content"),
                    name=msg_data.get("name"),
                    tool_calls=msg_data.get("toolCalls"),
                    tool_call_id=msg_data.get("toolCallId"),
                )
            )

        # 提取原始请求头
        raw_headers = dict(request.headers)

        # 构建 AgentRequest
        agent_request = AgentRequest(
            messages=messages,
            stream=True,  # AG-UI 总是流式
            tools=request_data.get("tools"),
            raw_headers=raw_headers,
            raw_body=request_data,
            hooks=hooks,
        )

        # 保存额外参数
        agent_request.extra = {
            "threadId": context["threadId"],
            "runId": context["runId"],
            "context": request_data.get("context"),
            "forwardedProps": request_data.get("forwardedProps"),
        }

        return agent_request, context

    async def format_response(
        self,
        result: AgentResult,
        request: AgentRequest,
        context: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """格式化响应为 AG-UI 事件流

        Agent 可以 yield 三种类型的内容:
        1. 普通字符串 - 会被包装成 TEXT_MESSAGE_CONTENT 事件
        2. AgentEvent - 直接输出其 raw_sse
        3. None - 忽略

        Args:
            result: Agent 执行结果
            request: 原始请求
            context: 运行上下文

        Yields:
            SSE 格式的事件数据
        """
        hooks = request.hooks
        message_id = str(uuid.uuid4())
        text_message_started = False

        # 1. 发送 RUN_STARTED 事件
        if hooks:
            event = hooks.on_run_start()
            if event and event.raw_sse:
                yield event.raw_sse

        try:
            # 2. 处理 Agent 结果
            content = self._extract_content(result)

            # 3. 流式发送内容
            if self._is_iterator(content):
                async for chunk in self._iterate_content(content):
                    if chunk is None:
                        continue

                    # 检查是否是 AgentEvent
                    if isinstance(chunk, AgentEvent):
                        if chunk.raw_sse:
                            yield chunk.raw_sse
                    elif isinstance(chunk, str) and chunk:
                        # 普通文本内容，包装成 TEXT_MESSAGE_CONTENT
                        if not text_message_started and hooks:
                            # 延迟发送 TEXT_MESSAGE_START，只在有文本内容时才发送
                            event = hooks.on_text_message_start(message_id)
                            if event and event.raw_sse:
                                yield event.raw_sse
                            text_message_started = True

                        if hooks:
                            event = hooks.on_text_message_content(
                                message_id, chunk
                            )
                            if event and event.raw_sse:
                                yield event.raw_sse
            else:
                # 非迭代器内容
                if isinstance(content, AgentEvent):
                    if content.raw_sse:
                        yield content.raw_sse
                elif content:
                    content_str = str(content)
                    if hooks:
                        event = hooks.on_text_message_start(message_id)
                        if event and event.raw_sse:
                            yield event.raw_sse
                        text_message_started = True
                        event = hooks.on_text_message_content(
                            message_id, content_str
                        )
                        if event and event.raw_sse:
                            yield event.raw_sse

            # 4. 发送 TEXT_MESSAGE_END 事件（如果有文本消息）
            if text_message_started and hooks:
                event = hooks.on_text_message_end(message_id)
                if event and event.raw_sse:
                    yield event.raw_sse

            # 5. 发送 RUN_FINISHED 事件
            if hooks:
                event = hooks.on_run_finish()
                if event and event.raw_sse:
                    yield event.raw_sse

        except Exception as e:
            # 发送错误事件
            if hooks:
                event = hooks.on_run_error(str(e), "AGENT_ERROR")
                if event and event.raw_sse:
                    yield event.raw_sse

    async def _error_stream(self, message: str) -> AsyncIterator[str]:
        """生成错误事件流"""
        context = {
            "threadId": str(uuid.uuid4()),
            "runId": str(uuid.uuid4()),
        }
        hooks = self.create_hooks(context)

        event = hooks.on_run_start()
        if event and event.raw_sse:
            yield event.raw_sse
        event = hooks.on_run_error(message, "REQUEST_ERROR")
        if event and event.raw_sse:
            yield event.raw_sse

    def _extract_content(self, result: AgentResult) -> Any:
        """从结果中提取内容"""
        if isinstance(result, AgentRunResult):
            return result.content
        if isinstance(result, AgentResponse):
            return result.content
        if isinstance(result, str):
            return result
        return result

    async def _iterate_content(
        self, content: Union[Iterator, AsyncIterator]
    ) -> AsyncIterator:
        """统一迭代同步和异步迭代器

        支持迭代包含字符串或 AgentEvent 的迭代器。
        对于同步迭代器，每次 next() 调用都在线程池中执行，避免阻塞事件循环。
        """
        import asyncio

        if hasattr(content, "__aiter__"):
            # 异步迭代器
            async for chunk in content:  # type: ignore
                yield chunk
        else:
            # 同步迭代器 - 在线程池中迭代，避免阻塞
            loop = asyncio.get_event_loop()
            iterator = iter(content)  # type: ignore

            while True:
                try:
                    # 在线程池中执行 next()，避免 time.sleep 阻塞事件循环
                    chunk = await loop.run_in_executor(None, next, iterator)
                    yield chunk
                except StopIteration:
                    break


# ============================================================================
# 辅助函数 - 用于用户自定义 AG-UI 事件
# ============================================================================


def create_agui_event(event_type: AGUIEventType, **kwargs) -> AGUIBaseEvent:
    """创建 AG-UI 事件的辅助函数

    Args:
        event_type: 事件类型
        **kwargs: 事件参数

    Returns:
        对应类型的事件对象

    Example:
        >>> event = create_agui_event(
        ...     AGUIEventType.TEXT_MESSAGE_CONTENT,
        ...     messageId="msg-123",
        ...     delta="Hello"
        ... )
    """
    event_classes = {
        AGUIEventType.RUN_STARTED: AGUIRunStartedEvent,
        AGUIEventType.RUN_FINISHED: AGUIRunFinishedEvent,
        AGUIEventType.RUN_ERROR: AGUIRunErrorEvent,
        AGUIEventType.STEP_STARTED: AGUIStepStartedEvent,
        AGUIEventType.STEP_FINISHED: AGUIStepFinishedEvent,
        AGUIEventType.TEXT_MESSAGE_START: AGUITextMessageStartEvent,
        AGUIEventType.TEXT_MESSAGE_CONTENT: AGUITextMessageContentEvent,
        AGUIEventType.TEXT_MESSAGE_END: AGUITextMessageEndEvent,
        AGUIEventType.TOOL_CALL_START: AGUIToolCallStartEvent,
        AGUIEventType.TOOL_CALL_ARGS: AGUIToolCallArgsEvent,
        AGUIEventType.TOOL_CALL_END: AGUIToolCallEndEvent,
        AGUIEventType.TOOL_CALL_RESULT: AGUIToolCallResultEvent,
        AGUIEventType.STATE_SNAPSHOT: AGUIStateSnapshotEvent,
        AGUIEventType.STATE_DELTA: AGUIStateDeltaEvent,
        AGUIEventType.MESSAGES_SNAPSHOT: AGUIMessagesSnapshotEvent,
        AGUIEventType.RAW: AGUIRawEvent,
        AGUIEventType.CUSTOM: AGUICustomEvent,
    }

    event_class = event_classes.get(event_type, AGUIBaseEvent)
    return event_class(type=event_type, **kwargs)
