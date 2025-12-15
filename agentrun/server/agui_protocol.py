"""AG-UI (Agent-User Interaction Protocol) 协议实现

AG-UI 是一种开源、轻量级、基于事件的协议，用于标准化 AI Agent 与前端应用之间的交互。
参考: https://docs.ag-ui.com/

本实现使用 ag-ui-protocol 包提供的事件类型和编码器，
将 AgentResult 事件转换为 AG-UI SSE 格式。
"""

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

from ag_ui.core import AssistantMessage
from ag_ui.core import CustomEvent as AguiCustomEvent
from ag_ui.core import EventType as AguiEventType
from ag_ui.core import Message as AguiMessage
from ag_ui.core import MessagesSnapshotEvent
from ag_ui.core import RawEvent as AguiRawEvent
from ag_ui.core import (
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    SystemMessage,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.core import Tool as AguiTool
from ag_ui.core import ToolCall as AguiToolCall
from ag_ui.core import (
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.core import ToolMessage as AguiToolMessage
from ag_ui.core import UserMessage
from ag_ui.encoder import EventEncoder
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import pydash

from ..utils.helper import merge
from .model import (
    AdditionMode,
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

    name = "agui"

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config.openai if config else None
        self._encoder = EventEncoder()

    def get_prefix(self) -> str:
        """AG-UI 协议建议使用 /ag-ui/agent 前缀"""
        return pydash.get(self.config, "prefix", DEFAULT_PREFIX)

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

        # 提取原始请求头
        raw_headers = dict(request.headers)

        # 构建 AgentRequest
        agent_request = AgentRequest(
            protocol="agui",  # 设置协议名称
            messages=messages,
            stream=True,  # AG-UI 总是流式
            tools=tools,
            body=request_data,
            headers=raw_headers,
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

        Args:
            event_stream: AgentEvent 流
            context: 上下文信息

        Yields:
            SSE 格式的字符串
        """
        message_id = str(uuid.uuid4())

        # 状态追踪
        text_started = False
        # 工具调用状态：{tool_id: {"started": bool, "ended": bool}}
        tool_call_states: Dict[str, Dict[str, bool]] = {}

        # 发送 RUN_STARTED
        yield self._encoder.encode(
            RunStartedEvent(
                thread_id=context.get("thread_id"),
                run_id=context.get("run_id"),
            )
        )

        async for event in event_stream:
            # 处理边界事件注入
            for sse_data in self._process_event_with_boundaries(
                event, context, message_id, text_started, tool_call_states
            ):
                if sse_data:
                    yield sse_data

            # 更新状态
            if event.event == EventType.TEXT:
                text_started = True
            elif event.event == EventType.TOOL_CALL_CHUNK:
                tool_id = event.data.get("id", "")
                if tool_id:
                    if tool_id not in tool_call_states:
                        tool_call_states[tool_id] = {
                            "started": True,
                            "ended": False,
                        }

        # 结束所有未结束的工具调用
        for tool_id, state in tool_call_states.items():
            if state["started"] and not state["ended"]:
                yield self._encoder.encode(
                    ToolCallEndEvent(tool_call_id=tool_id)
                )

        # 发送 TEXT_MESSAGE_END（如果有文本消息）
        if text_started:
            yield self._encoder.encode(
                TextMessageEndEvent(message_id=message_id)
            )

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
        message_id: str,
        text_started: bool,
        tool_call_states: Dict[str, Dict[str, bool]],
    ) -> Iterator[str]:
        """处理事件并注入边界事件

        Args:
            event: 用户事件
            context: 上下文
            message_id: 消息 ID
            text_started: 文本是否已开始
            tool_call_states: 工具调用状态

        Yields:
            SSE 格式的字符串
        """
        import json

        # RAW 事件直接透传
        if event.event == EventType.RAW:
            raw_data = event.data.get("raw", "")
            if raw_data:
                if not raw_data.endswith("\n\n"):
                    raw_data = raw_data.rstrip("\n") + "\n\n"
                yield raw_data
            return

        # TEXT 事件：在首个 TEXT 前注入 TEXT_MESSAGE_START
        if event.event == EventType.TEXT:
            if not text_started:
                yield self._encoder.encode(
                    TextMessageStartEvent(
                        message_id=message_id,
                        role="assistant",
                    )
                )

            # 发送 TEXT_MESSAGE_CONTENT
            agui_event = TextMessageContentEvent(
                message_id=message_id,
                delta=event.data.get("delta", ""),
            )
            if event.addition:
                event_dict = agui_event.model_dump(
                    by_alias=True, exclude_none=True
                )
                event_dict = self._apply_addition(
                    event_dict, event.addition, event.addition_mode
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

            if tool_id and tool_id not in tool_call_states:
                # 首次见到这个工具调用，发送 TOOL_CALL_START
                yield self._encoder.encode(
                    ToolCallStartEvent(
                        tool_call_id=tool_id,
                        tool_call_name=tool_name,
                    )
                )
                tool_call_states[tool_id] = {"started": True, "ended": False}

            # 发送 TOOL_CALL_ARGS
            yield self._encoder.encode(
                ToolCallArgsEvent(
                    tool_call_id=tool_id,
                    delta=event.data.get("args_delta", ""),
                )
            )
            return

        # TOOL_RESULT 事件：确保工具调用已结束
        if event.event == EventType.TOOL_RESULT:
            tool_id = event.data.get("id", "")

            # 如果工具调用未开始，先补充 START
            if tool_id and tool_id not in tool_call_states:
                yield self._encoder.encode(
                    ToolCallStartEvent(
                        tool_call_id=tool_id,
                        tool_call_name="",
                    )
                )
                tool_call_states[tool_id] = {"started": True, "ended": False}

            # 如果工具调用未结束，先补充 END
            if (
                tool_id
                and tool_call_states.get(tool_id, {}).get("started")
                and not tool_call_states.get(tool_id, {}).get("ended")
            ):
                yield self._encoder.encode(
                    ToolCallEndEvent(tool_call_id=tool_id)
                )
                tool_call_states[tool_id]["ended"] = True

            # 发送 TOOL_CALL_RESULT
            yield self._encoder.encode(
                ToolCallResultEvent(
                    message_id=event.data.get(
                        "message_id", f"tool-result-{tool_id}"
                    ),
                    tool_call_id=tool_id,
                    content=event.data.get("content")
                    or event.data.get("result", ""),
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
        yield self._encoder.encode(
            AguiCustomEvent(
                name=event.event.value,
                value=event.data,
            )
        )

    def _convert_messages_for_snapshot(
        self, messages: List[Dict[str, Any]]
    ) -> List[AguiMessage]:
        """将消息列表转换为 ag-ui-protocol 格式

        Args:
            messages: 消息字典列表

        Returns:
            ag-ui-protocol 消息列表
        """
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
        addition: Dict[str, Any],
        mode: AdditionMode,
    ) -> Dict[str, Any]:
        """应用 addition 字段

        Args:
            event_data: 原始事件数据
            addition: 附加字段
            mode: 合并模式

        Returns:
            合并后的事件数据
        """
        if mode == AdditionMode.REPLACE:
            # 完全覆盖
            event_data.update(addition)

        elif mode == AdditionMode.MERGE:
            # 深度合并
            event_data = merge(event_data, addition)

        elif mode == AdditionMode.PROTOCOL_ONLY:
            # 仅覆盖原有字段
            event_data = merge(event_data, addition, no_new_field=True)

        return event_data

    async def _error_stream(self, message: str) -> AsyncIterator[str]:
        """生成错误事件流

        Args:
            message: 错误消息

        Yields:
            SSE 格式的错误事件
        """
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
