"""AG-UI (Agent-User Interaction Protocol) 协议实现

AG-UI 是一种开源、轻量级、基于事件的协议，用于标准化 AI Agent 与前端应用之间的交互。
参考: https://docs.ag-ui.com/

本实现使用 ag-ui-protocol 包提供的事件类型和编码器，
将 AgentResult 事件转换为 AG-UI SSE 格式。
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    TYPE_CHECKING,
)
import uuid

if TYPE_CHECKING:
    from ag_ui.core import (
        Message as AguiMessage,
    )
    from ag_ui.encoder import EventEncoder

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import pydash

from ..utils.helper import merge, MergeOptions
from .model import (
    AgentEvent,
    AgentRequest,
    EventType,
    Message,
    MessageRole,
    ServerConfig,
    Tool,
    ToolCall,
)
from .protocol import BaseProtocolHandler

if TYPE_CHECKING:
    from .invoker import AgentInvoker


# ============================================================================
# AG-UI 协议处理器
# ============================================================================

DEFAULT_PREFIX = "/ag-ui/agent"


@dataclass
class TextState:
    started: bool = False
    ended: bool = False
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ToolCallState:
    name: str = ""
    started: bool = False
    ended: bool = False
    has_result: bool = False
    is_hitl: bool = False


@dataclass
class StreamStateMachine:
    text: TextState = field(default_factory=TextState)
    tool_call_states: Dict[str, ToolCallState] = field(default_factory=dict)
    tool_result_chunks: Dict[str, List[str]] = field(default_factory=dict)
    run_errored: bool = False

    def end_all_tools(
        self, encoder: "EventEncoder", exclude: Optional[str] = None
    ) -> Iterator[str]:
        from ag_ui.core import ToolCallEndEvent

        for tool_id, state in self.tool_call_states.items():
            if exclude and tool_id == exclude:
                continue
            if state.started and not state.ended:
                yield encoder.encode(ToolCallEndEvent(tool_call_id=tool_id))
                state.ended = True

    def ensure_text_started(self, encoder: "EventEncoder") -> Iterator[str]:
        from ag_ui.core import TextMessageStartEvent

        if not self.text.started or self.text.ended:
            if self.text.ended:
                self.text = TextState()
            yield encoder.encode(
                TextMessageStartEvent(
                    message_id=self.text.message_id,
                    role="assistant",
                )
            )
            self.text.started = True
            self.text.ended = False

    def end_text_if_open(self, encoder: "EventEncoder") -> Iterator[str]:
        from ag_ui.core import TextMessageEndEvent

        if self.text.started and not self.text.ended:
            yield encoder.encode(
                TextMessageEndEvent(message_id=self.text.message_id)
            )
            self.text.ended = True

    def cache_tool_result_chunk(self, tool_id: str, delta: str) -> None:
        if not tool_id or delta is None:
            return
        if delta:
            self.tool_result_chunks.setdefault(tool_id, []).append(delta)

    def pop_tool_result_chunks(self, tool_id: str) -> str:
        return "".join(self.tool_result_chunks.pop(tool_id, []))


class AGUIProtocolHandler(BaseProtocolHandler):
    """AG-UI 协议处理器

    实现 AG-UI (Agent-User Interaction Protocol) 兼容接口。
    参考: https://docs.ag-ui.com/

    使用 ag-ui-protocol 包提供的事件类型和编码器。

    特点:
    - 基于事件的流式通信
    - 完整支持所有 AG-UI 事件类型
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
        # 可访问: POST http://localhost:8000/ag-ui/agent
    """

    name = "ag-ui"

    def __init__(self, config: Optional[ServerConfig] = None):
        from ag_ui.encoder import EventEncoder

        self._config = config.agui if config else None
        self._encoder = EventEncoder()

    def get_prefix(self) -> str:
        """AG-UI 协议建议使用 /ag-ui/agent 前缀"""
        return pydash.get(self._config, "prefix", DEFAULT_PREFIX)

    def as_fastapi_router(self, agent_invoker: "AgentInvoker") -> APIRouter:
        """创建 AG-UI 协议的 FastAPI Router"""
        router = APIRouter()

        @router.post("")
        async def run_agent(request: Request):
            """AG-UI 运行 Agent 端点

            接收 AG-UI 格式的请求，返回 SSE 事件流。
            """
            sse_headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }

            try:
                request_data = await request.json()
                agent_request, context = await self.parse_request(
                    request, request_data
                )

                # 使用 invoke_stream 获取流式结果
                event_stream = self._format_stream(
                    agent_invoker.invoke_stream(agent_request),
                    context,
                )

                return StreamingResponse(
                    event_stream,
                    media_type=self._encoder.get_content_type(),
                    headers=sse_headers,
                )

            except ValueError as e:
                return StreamingResponse(
                    self._error_stream(str(e)),
                    media_type=self._encoder.get_content_type(),
                    headers=sse_headers,
                )
            except Exception as e:
                return StreamingResponse(
                    self._error_stream(f"Internal error: {str(e)}"),
                    media_type=self._encoder.get_content_type(),
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
        """
        # 创建上下文
        context = {
            "thread_id": request_data.get("threadId") or str(uuid.uuid4()),
            "run_id": request_data.get("runId") or str(uuid.uuid4()),
        }

        # 解析消息列表
        messages = self._parse_messages(request_data.get("messages", []))

        # 解析工具列表
        tools = self._parse_tools(request_data.get("tools"))

        # 构建 AgentRequest
        agent_request = AgentRequest(
            protocol="agui",  # 设置协议名称
            messages=messages,
            stream=True,  # AG-UI 总是流式
            tools=tools,
            raw_request=request,  # 保留原始请求对象
        )

        return agent_request, context

    def _parse_messages(
        self, raw_messages: List[Dict[str, Any]]
    ) -> List[Message]:
        """解析消息列表

        Args:
            raw_messages: 原始消息数据

        Returns:
            标准化的消息列表
        """
        messages = []

        for msg_data in raw_messages:
            if not isinstance(msg_data, dict):
                continue

            role_str = msg_data.get("role", "user")
            try:
                role = MessageRole(role_str)
            except ValueError:
                role = MessageRole.USER

            # 解析 tool_calls
            tool_calls = None
            if msg_data.get("toolCalls"):
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=tc.get("function", {}),
                    )
                    for tc in msg_data["toolCalls"]
                ]

            messages.append(
                Message(
                    id=msg_data.get("id"),
                    role=role,
                    content=msg_data.get("content"),
                    name=msg_data.get("name"),
                    tool_calls=tool_calls,
                    tool_call_id=msg_data.get("toolCallId"),
                )
            )

        return messages

    def _parse_tools(
        self, raw_tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Tool]]:
        """解析工具列表

        Args:
            raw_tools: 原始工具数据

        Returns:
            标准化的工具列表
        """
        if not raw_tools:
            return None

        tools = []
        for tool_data in raw_tools:
            if not isinstance(tool_data, dict):
                continue

            tools.append(
                Tool(
                    type=tool_data.get("type", "function"),
                    function=tool_data.get("function", {}),
                )
            )

        return tools if tools else None

    async def _format_stream(
        self,
        event_stream: AsyncIterator[AgentEvent],
        context: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """将 AgentEvent 流转换为 AG-UI SSE 格式

        自动生成边界事件：
        - RUN_STARTED / RUN_FINISHED（生命周期）
        - TEXT_MESSAGE_START / TEXT_MESSAGE_END（文本边界）
        - TOOL_CALL_START / TOOL_CALL_END（工具调用边界）

        注意：RUN_ERROR 之后不能再发送任何事件（包括 RUN_FINISHED）

        Args:
            event_stream: AgentEvent 流
            context: 上下文信息

        Yields:
            SSE 格式的字符串
        """
        from ag_ui.core import RunFinishedEvent, RunStartedEvent

        state = StreamStateMachine()

        # 发送 RUN_STARTED
        yield self._encoder.encode(
            RunStartedEvent(
                thread_id=context.get("thread_id"),
                run_id=context.get("run_id"),
            )
        )

        async for event in event_stream:
            # RUN_ERROR 后不再处理任何事件
            if state.run_errored:
                continue

            # 检查是否是错误事件
            if event.event == EventType.ERROR:
                state.run_errored = True

            # 处理边界事件注入
            for sse_data in self._process_event_with_boundaries(
                event,
                context,
                state,
            ):
                if sse_data:
                    yield sse_data

        # RUN_ERROR 后不发送任何清理事件
        if state.run_errored:
            return

        # 结束所有未结束的工具调用
        for sse_data in state.end_all_tools(self._encoder):
            yield sse_data

        # 发送 TEXT_MESSAGE_END（如果有文本消息且未结束）
        for sse_data in state.end_text_if_open(self._encoder):
            yield sse_data

        # 发送 RUN_FINISHED
        yield self._encoder.encode(
            RunFinishedEvent(
                thread_id=context.get("thread_id"),
                run_id=context.get("run_id"),
            )
        )

    def _process_event_with_boundaries(
        self,
        event: AgentEvent,
        context: Dict[str, Any],
        state: StreamStateMachine,
    ) -> Iterator[str]:
        """处理事件并注入边界事件"""
        import json

        from ag_ui.core import CustomEvent as AguiCustomEvent
        from ag_ui.core import (
            RunErrorEvent,
            StateDeltaEvent,
            StateSnapshotEvent,
            TextMessageContentEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
            ToolCallStartEvent,
        )

        # RAW 事件直接透传
        if event.event == EventType.RAW:
            raw_data = event.data.get("raw", "")
            if raw_data:
                if not raw_data.endswith("\n\n"):
                    raw_data = raw_data.rstrip("\n") + "\n\n"
                yield raw_data
            return

        # TEXT 事件：在首个 TEXT 前注入 TEXT_MESSAGE_START
        # AG-UI 协议要求：发送 TEXT_MESSAGE_START 前必须先结束所有未结束的 TOOL_CALL
        if event.event == EventType.TEXT:
            for sse_data in state.end_all_tools(self._encoder):
                yield sse_data

            for sse_data in state.ensure_text_started(self._encoder):
                yield sse_data

            agui_event = TextMessageContentEvent(
                message_id=state.text.message_id,
                delta=event.data.get("delta", ""),
            )
            if event.addition:
                event_dict = agui_event.model_dump(
                    by_alias=True, exclude_none=True
                )
                event_dict = self._apply_addition(
                    event_dict,
                    event.addition,
                    event.addition_merge_options,
                )
                json_str = json.dumps(event_dict, ensure_ascii=False)
                yield f"data: {json_str}\n\n"
            else:
                yield self._encoder.encode(agui_event)
            return

        # TOOL_CALL_CHUNK 事件：在首个 CHUNK 前注入 TOOL_CALL_START
        if event.event == EventType.TOOL_CALL_CHUNK:
            tool_id = event.data.get("id", "")
            tool_name = event.data.get("name", "")

            for sse_data in state.end_text_if_open(self._encoder):
                yield sse_data

            need_start = False
            current_state = state.tool_call_states.get(tool_id)
            if tool_id:
                if current_state is None or current_state.ended:
                    need_start = True

            if need_start:
                yield self._encoder.encode(
                    ToolCallStartEvent(
                        tool_call_id=tool_id,
                        tool_call_name=tool_name,
                    )
                )
                state.tool_call_states[tool_id] = ToolCallState(
                    name=tool_name,
                    started=True,
                    ended=False,
                )

            yield self._encoder.encode(
                ToolCallArgsEvent(
                    tool_call_id=tool_id,
                    delta=event.data.get("args_delta", ""),
                )
            )
            return

        # TOOL_CALL 事件：完整的工具调用事件
        if event.event == EventType.TOOL_CALL:
            tool_id = event.data.get("id", "")
            tool_name = event.data.get("name", "")
            tool_args = event.data.get("args", "")

            for sse_data in state.end_text_if_open(self._encoder):
                yield sse_data

            need_start = False
            current_state = state.tool_call_states.get(tool_id)
            if tool_id:
                if current_state is None or current_state.ended:
                    need_start = True

            if need_start:
                yield self._encoder.encode(
                    ToolCallStartEvent(
                        tool_call_id=tool_id,
                        tool_call_name=tool_name,
                    )
                )
                state.tool_call_states[tool_id] = ToolCallState(
                    name=tool_name,
                    started=True,
                    ended=False,
                )

            # 发送工具参数（如果存在）
            if tool_args:
                yield self._encoder.encode(
                    ToolCallArgsEvent(
                        tool_call_id=tool_id,
                        delta=tool_args,
                    )
                )
            return

        # TOOL_RESULT_CHUNK 事件：工具执行过程中的流式输出
        if event.event == EventType.TOOL_RESULT_CHUNK:
            tool_id = event.data.get("id", "")
            delta = event.data.get("delta", "")
            state.cache_tool_result_chunk(tool_id, delta)
            return

        # HITL 事件：请求人类介入
        if event.event == EventType.HITL:
            hitl_id = event.data.get("id", "")
            tool_call_id = event.data.get("tool_call_id", "")
            hitl_type = event.data.get("type", "confirmation")
            prompt = event.data.get("prompt", "")
            options = event.data.get("options")
            default = event.data.get("default")
            timeout = event.data.get("timeout")
            schema = event.data.get("schema")

            for sse_data in state.end_text_if_open(self._encoder):
                yield sse_data

            if tool_call_id and tool_call_id in state.tool_call_states:
                tool_state = state.tool_call_states[tool_call_id]
                if tool_state.started and not tool_state.ended:
                    yield self._encoder.encode(
                        ToolCallEndEvent(tool_call_id=tool_call_id)
                    )
                    tool_state.ended = True
                tool_state.is_hitl = True
                tool_state.has_result = False
                return

            import json as json_module

            args_dict: Dict[str, Any] = {
                "type": hitl_type,
                "prompt": prompt,
            }
            if options:
                args_dict["options"] = options
            if default is not None:
                args_dict["default"] = default
            if timeout is not None:
                args_dict["timeout"] = timeout
            if schema:
                args_dict["schema"] = schema

            args_json = json_module.dumps(args_dict, ensure_ascii=False)
            actual_id = tool_call_id or hitl_id

            yield self._encoder.encode(
                ToolCallStartEvent(
                    tool_call_id=actual_id,
                    tool_call_name=f"hitl_{hitl_type}",
                )
            )
            yield self._encoder.encode(
                ToolCallArgsEvent(
                    tool_call_id=actual_id,
                    delta=args_json,
                )
            )
            yield self._encoder.encode(ToolCallEndEvent(tool_call_id=actual_id))

            state.tool_call_states[actual_id] = ToolCallState(
                name=f"hitl_{hitl_type}",
                started=True,
                ended=True,
                has_result=False,
                is_hitl=True,
            )
            return

        # TOOL_RESULT 事件：确保当前工具调用已结束
        if event.event == EventType.TOOL_RESULT:
            tool_id = event.data.get("id", "")
            tool_name = event.data.get("name", "")

            for sse_data in state.end_text_if_open(self._encoder):
                yield sse_data

            tool_state = (
                state.tool_call_states.get(tool_id) if tool_id else None
            )
            if tool_id and tool_state is None:
                yield self._encoder.encode(
                    ToolCallStartEvent(
                        tool_call_id=tool_id,
                        tool_call_name=tool_name or "",
                    )
                )
                tool_state = ToolCallState(
                    name=tool_name, started=True, ended=False
                )
                state.tool_call_states[tool_id] = tool_state

            if tool_state and tool_state.started and not tool_state.ended:
                yield self._encoder.encode(
                    ToolCallEndEvent(tool_call_id=tool_id)
                )
                tool_state.ended = True

            final_result = event.data.get("content") or event.data.get(
                "result", ""
            )
            if tool_id:
                cached_chunks = state.pop_tool_result_chunks(tool_id)
                if cached_chunks:
                    final_result = cached_chunks + final_result

            yield self._encoder.encode(
                ToolCallResultEvent(
                    message_id=event.data.get(
                        "message_id", f"tool-result-{tool_id}"
                    ),
                    tool_call_id=tool_id,
                    content=final_result,
                    role="tool",
                )
            )
            return

        # ERROR 事件
        if event.event == EventType.ERROR:
            yield self._encoder.encode(
                RunErrorEvent(
                    message=event.data.get("message", ""),
                    code=event.data.get("code"),
                )
            )
            return

        # STATE 事件
        if event.event == EventType.STATE:
            if "snapshot" in event.data:
                yield self._encoder.encode(
                    StateSnapshotEvent(snapshot=event.data.get("snapshot", {}))
                )
            elif "delta" in event.data:
                yield self._encoder.encode(
                    StateDeltaEvent(delta=event.data.get("delta", []))
                )
            else:
                yield self._encoder.encode(
                    StateSnapshotEvent(snapshot=event.data)
                )
            return

        # CUSTOM 事件
        if event.event == EventType.CUSTOM:
            yield self._encoder.encode(
                AguiCustomEvent(
                    name=event.data.get("name", "custom"),
                    value=event.data.get("value"),
                )
            )
            return

        # 其他未知事件
        event_name = (
            event.event.value
            if hasattr(event.event, "value")
            else str(event.event)
        )
        yield self._encoder.encode(
            AguiCustomEvent(
                name=event_name,
                value=event.data,
            )
        )

    def _convert_messages_for_snapshot(
        self, messages: List[Dict[str, Any]]
    ) -> List["AguiMessage"]:
        """将消息列表转换为 ag-ui-protocol 格式

        Args:
            messages: 消息字典列表

        Returns:
            ag-ui-protocol 消息列表
        """
        from ag_ui.core import AssistantMessage, SystemMessage
        from ag_ui.core import ToolMessage as AguiToolMessage
        from ag_ui.core import UserMessage

        result = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "user")
            content = msg.get("content", "")
            msg_id = msg.get("id", str(uuid.uuid4()))

            if role == "user":
                result.append(
                    UserMessage(id=msg_id, role="user", content=content)
                )
            elif role == "assistant":
                result.append(
                    AssistantMessage(
                        id=msg_id,
                        role="assistant",
                        content=content,
                    )
                )
            elif role == "system":
                result.append(
                    SystemMessage(id=msg_id, role="system", content=content)
                )
            elif role == "tool":
                result.append(
                    AguiToolMessage(
                        id=msg_id,
                        role="tool",
                        content=content,
                        tool_call_id=msg.get("tool_call_id", ""),
                    )
                )

        return result

    def _apply_addition(
        self,
        event_data: Dict[str, Any],
        addition: Optional[Dict[str, Any]],
        merge_options: Optional[MergeOptions] = None,
    ) -> Dict[str, Any]:
        """应用 addition 字段

        Args:
            event_data: 原始事件数据
            addition: 附加字段
            merge_options: 合并选项，透传给 utils.helper.merge

        Returns:
            合并后的事件数据
        """
        if not addition:
            return event_data

        return merge(event_data, addition, **(merge_options or {}))

    async def _error_stream(self, message: str) -> AsyncIterator[str]:
        """生成错误事件流

        Args:
            message: 错误消息

        Yields:
            SSE 格式的错误事件
        """
        from ag_ui.core import RunErrorEvent, RunStartedEvent

        thread_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())

        # 生命周期开始
        yield self._encoder.encode(
            RunStartedEvent(thread_id=thread_id, run_id=run_id)
        )

        # 错误事件
        yield self._encoder.encode(
            RunErrorEvent(message=message, code="REQUEST_ERROR")
        )
