"""AG-UI (Agent-User Interaction Protocol) 协议实现

AG-UI 是一种开源、轻量级、基于事件的协议，用于标准化 AI Agent 与前端应用之间的交互。
参考: https://docs.ag-ui.com/

本实现使用 ag-ui-protocol 包提供的事件类型和编码器，
将 AgentResult 事件转换为 AG-UI SSE 格式。
"""

from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING
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
    AgentRequest,
    AgentResult,
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
        result_stream: AsyncIterator[AgentResult],
        context: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """将 AgentResult 流转换为 AG-UI SSE 格式

        Args:
            result_stream: AgentResult 流
            context: 上下文信息

        Yields:
            SSE 格式的字符串
        """
        async for result in result_stream:
            sse_data = self._format_event(result, context)
            if sse_data:
                yield sse_data

    def _format_event(
        self,
        result: AgentResult,
        context: Dict[str, Any],
    ) -> str:
        """将 AgentResult 转换为 SSE 格式

        Args:
            result: AgentResult 事件
            context: 上下文信息

        Returns:
            SSE 格式的字符串
        """
        import json

        # 统一将字符串或 dict 标准化为 AgentResult
        if isinstance(result, str):
            result = AgentResult(
                event=EventType.TEXT_MESSAGE_CHUNK,
                data={"delta": result},
            )
        elif isinstance(result, dict):
            ev = self._parse_event_type(result.get("event"))
            result = AgentResult(event=ev, data=result.get("data", result))

        # 特殊处理 STREAM_DATA 事件 - 直接返回原始 SSE 数据
        if result.event == EventType.STREAM_DATA:
            raw_data = result.data.get("raw", "")
            if raw_data:
                # 如果已经是 SSE 格式，直接返回
                if raw_data.startswith("data:"):
                    # 确保以 \n\n 结尾
                    if not raw_data.endswith("\n\n"):
                        raw_data = raw_data.rstrip("\n") + "\n\n"
                    return raw_data
                else:
                    # 包装为 SSE 格式
                    return f"data: {raw_data}\n\n"
            return ""

        # 创建 ag-ui-protocol 事件对象
        agui_event = self._create_agui_event(result, context)

        if agui_event is None:
            return ""

        # 处理 addition - 需要将事件转为 dict，应用 addition，然后重新序列化
        if result.addition:
            # ag-ui-protocol 事件是 pydantic model，需要转为 dict 处理 addition
            # 使用 by_alias=True 确保字段名使用 camelCase（与编码器一致）
            event_dict = agui_event.model_dump(by_alias=True, exclude_none=True)
            event_dict = self._apply_addition(
                event_dict, result.addition, result.addition_mode
            )
            # 使用与 EventEncoder 相同的格式
            json_str = json.dumps(event_dict, ensure_ascii=False)
            return f"data: {json_str}\n\n"

        # 使用 ag-ui-protocol 的编码器
        return self._encoder.encode(agui_event)

    def _parse_event_type(self, evt: Any) -> EventType:
        """解析事件类型

        Args:
            evt: 事件类型值

        Returns:
            EventType 枚举值
        """
        if isinstance(evt, EventType):
            return evt

        if isinstance(evt, str):
            try:
                return EventType(evt)
            except ValueError:
                try:
                    return EventType[evt]
                except KeyError:
                    pass

        return EventType.TEXT_MESSAGE_CHUNK

    def _create_agui_event(
        self,
        result: AgentResult,
        context: Dict[str, Any],
    ) -> Any:
        """根据 AgentResult 创建对应的 ag-ui-protocol 事件对象

        Args:
            result: AgentResult 事件
            context: 上下文信息

        Returns:
            ag-ui-protocol 事件对象
        """
        data = result.data
        event_type = result.event

        # 生命周期事件
        if event_type == EventType.RUN_STARTED:
            return RunStartedEvent(
                thread_id=data.get("thread_id") or context.get("thread_id"),
                run_id=data.get("run_id") or context.get("run_id"),
            )

        elif event_type == EventType.RUN_FINISHED:
            return RunFinishedEvent(
                thread_id=data.get("thread_id") or context.get("thread_id"),
                run_id=data.get("run_id") or context.get("run_id"),
            )

        elif event_type == EventType.RUN_ERROR:
            return RunErrorEvent(
                message=data.get("message", ""),
                code=data.get("code"),
            )

        elif event_type == EventType.STEP_STARTED:
            return StepStartedEvent(
                step_name=data.get("step_name", ""),
            )

        elif event_type == EventType.STEP_FINISHED:
            return StepFinishedEvent(
                step_name=data.get("step_name", ""),
            )

        # 文本消息事件
        elif event_type == EventType.TEXT_MESSAGE_START:
            return TextMessageStartEvent(
                message_id=data.get("message_id", str(uuid.uuid4())),
                role=data.get("role", "assistant"),
            )

        elif event_type == EventType.TEXT_MESSAGE_CONTENT:
            return TextMessageContentEvent(
                message_id=data.get("message_id", ""),
                delta=data.get("delta", ""),
            )

        elif event_type == EventType.TEXT_MESSAGE_END:
            return TextMessageEndEvent(
                message_id=data.get("message_id", ""),
            )

        elif event_type == EventType.TEXT_MESSAGE_CHUNK:
            # TEXT_MESSAGE_CHUNK 需要转换为 TEXT_MESSAGE_CONTENT
            return TextMessageContentEvent(
                message_id=data.get("message_id", ""),
                delta=data.get("delta", ""),
            )

        # 工具调用事件
        elif event_type == EventType.TOOL_CALL_START:
            return ToolCallStartEvent(
                tool_call_id=data.get("tool_call_id", ""),
                tool_call_name=data.get("tool_call_name", ""),
                parent_message_id=data.get("parent_message_id"),
            )

        elif event_type == EventType.TOOL_CALL_ARGS:
            return ToolCallArgsEvent(
                tool_call_id=data.get("tool_call_id", ""),
                delta=data.get("delta", ""),
            )

        elif event_type == EventType.TOOL_CALL_END:
            return ToolCallEndEvent(
                tool_call_id=data.get("tool_call_id", ""),
            )

        elif event_type == EventType.TOOL_CALL_RESULT:
            return ToolCallResultEvent(
                message_id=data.get(
                    "message_id", f"tool-result-{data.get('tool_call_id', '')}"
                ),
                tool_call_id=data.get("tool_call_id", ""),
                content=data.get("content") or data.get("result", ""),
                role="tool",
            )

        elif event_type == EventType.TOOL_CALL_CHUNK:
            # TOOL_CALL_CHUNK 需要转换为 TOOL_CALL_ARGS
            return ToolCallArgsEvent(
                tool_call_id=data.get("tool_call_id", ""),
                delta=data.get("delta", ""),
            )

        # 状态管理事件
        elif event_type == EventType.STATE_SNAPSHOT:
            return StateSnapshotEvent(
                snapshot=data.get("snapshot", {}),
            )

        elif event_type == EventType.STATE_DELTA:
            return StateDeltaEvent(
                delta=data.get("delta", []),
            )

        # 消息快照事件
        elif event_type == EventType.MESSAGES_SNAPSHOT:
            # 需要转换消息格式
            messages = self._convert_messages_for_snapshot(
                data.get("messages", [])
            )
            return MessagesSnapshotEvent(
                messages=messages,
            )

        # Reasoning 事件（ag-ui-protocol 使用 Thinking 命名）
        # 这些事件在 ag-ui-protocol 中可能使用不同的名称，
        # 需要映射到对应的事件类型或使用 CustomEvent
        elif event_type in (
            EventType.REASONING_START,
            EventType.REASONING_MESSAGE_START,
            EventType.REASONING_MESSAGE_CONTENT,
            EventType.REASONING_MESSAGE_END,
            EventType.REASONING_MESSAGE_CHUNK,
            EventType.REASONING_END,
        ):
            # 使用 CustomEvent 来包装 Reasoning 事件
            return AguiCustomEvent(
                name=event_type.value,
                value=data,
            )

        # Activity 事件 - ag-ui-protocol 有对应的事件但格式不同
        elif event_type == EventType.ACTIVITY_SNAPSHOT:
            return AguiCustomEvent(
                name="ACTIVITY_SNAPSHOT",
                value=data.get("snapshot", {}),
            )

        elif event_type == EventType.ACTIVITY_DELTA:
            return AguiCustomEvent(
                name="ACTIVITY_DELTA",
                value=data.get("delta", []),
            )

        # Meta 事件
        elif event_type == EventType.META_EVENT:
            return AguiCustomEvent(
                name=data.get("name", "meta"),
                value=data.get("value"),
            )

        # RAW 事件
        elif event_type == EventType.RAW:
            return AguiRawEvent(
                event=data.get("event", {}),
            )

        # CUSTOM 事件
        elif event_type == EventType.CUSTOM:
            return AguiCustomEvent(
                name=data.get("name", ""),
                value=data.get("value"),
            )

        # STREAM_DATA 在 _format_event 中已特殊处理，这里不应该到达
        # 但如果到达了，返回 None 表示跳过
        elif event_type == EventType.STREAM_DATA:
            return None

        # 默认使用 CustomEvent
        return AguiCustomEvent(
            name=event_type.value,
            value=data,
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
        context = {
            "thread_id": str(uuid.uuid4()),
            "run_id": str(uuid.uuid4()),
        }

        # RUN_STARTED
        yield self._format_event(
            AgentResult(
                event=EventType.RUN_STARTED,
                data=context,
            ),
            context,
        )

        # RUN_ERROR
        yield self._format_event(
            AgentResult(
                event=EventType.RUN_ERROR,
                data={"message": message, "code": "REQUEST_ERROR"},
            ),
            context,
        )
