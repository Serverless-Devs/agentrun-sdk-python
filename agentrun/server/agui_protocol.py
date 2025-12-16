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

    name = "ag-ui"

    def __init__(self, config: Optional[ServerConfig] = None):
        self._config = config.agui if config else None
        self._encoder = EventEncoder()
        # 是否串行化工具调用（兼容 CopilotKit 等前端）
        self._copilotkit_compatibility = pydash.get(
            self._config, "copilotkit_compatibility", False
        )

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
        # 状态追踪（使用可变容器以便在 _process_event_with_boundaries 中更新）
        # text_state: {"started": bool, "ended": bool, "message_id": str}
        text_state: Dict[str, Any] = {
            "started": False,
            "ended": False,
            "message_id": str(uuid.uuid4()),
        }
        # 工具调用状态：{tool_id: {"started": bool, "ended": bool, "name": str, "has_result": bool}}
        tool_call_states: Dict[str, Dict[str, Any]] = {}
        # 错误状态：RUN_ERROR 后不能再发送任何事件
        run_errored = False
        # 当前活跃的工具调用 ID（仅在 copilotkit_compatibility=True 时使用）
        # 用于实现严格的工具调用序列化
        active_tool_id: Optional[str] = None
        # 待发送的事件队列（仅在 copilotkit_compatibility=True 时使用）
        # 当一个工具调用正在进行时，其他工具的事件会被放入队列
        pending_events: List[AgentEvent] = []

        # 发送 RUN_STARTED
        yield self._encoder.encode(
            RunStartedEvent(
                thread_id=context.get("thread_id"),
                run_id=context.get("run_id"),
            )
        )

        # 辅助函数：处理队列中的所有事件
        def process_pending_queue() -> Iterator[str]:
            """处理队列中的所有待处理事件"""
            nonlocal active_tool_id
            while pending_events:
                pending_event = pending_events.pop(0)
                pending_tool_id = (
                    pending_event.data.get("id", "")
                    if pending_event.data
                    else ""
                )

                # 如果是新的工具调用，设置为活跃
                if (
                    pending_event.event == EventType.TOOL_CALL_CHUNK
                    and active_tool_id is None
                ):
                    active_tool_id = pending_tool_id

                for sse_data in self._process_event_with_boundaries(
                    pending_event,
                    context,
                    text_state,
                    tool_call_states,
                    self._copilotkit_compatibility,
                ):
                    if sse_data:
                        yield sse_data

                # 如果处理的是 TOOL_RESULT，检查是否需要继续处理队列
                if pending_event.event == EventType.TOOL_RESULT:
                    if pending_tool_id == active_tool_id:
                        active_tool_id = None

        async for event in event_stream:
            # RUN_ERROR 后不再处理任何事件
            if run_errored:
                continue

            # 检查是否是错误事件
            if event.event == EventType.ERROR:
                run_errored = True

            # 在 copilotkit_compatibility=True 模式下，实现严格的工具调用序列化
            # 当一个工具调用正在进行时，其他工具的事件会被放入队列
            if self._copilotkit_compatibility and not run_errored:
                tool_id = event.data.get("id", "") if event.data else ""

                # 处理 TOOL_CALL_CHUNK 事件
                if event.event == EventType.TOOL_CALL_CHUNK:
                    if active_tool_id is None:
                        # 没有活跃的工具调用，直接处理
                        active_tool_id = tool_id
                    elif tool_id != active_tool_id:
                        # 有其他活跃的工具调用，放入队列
                        pending_events.append(event)
                        continue
                    # 如果是同一个工具调用，继续处理

                # 处理 TOOL_RESULT 事件
                elif event.event == EventType.TOOL_RESULT:
                    # 检查是否是 UUID 格式的 ID，如果是，尝试映射到 call_xxx ID
                    actual_tool_id = tool_id
                    tool_name = event.data.get("name", "") if event.data else ""
                    is_uuid_format = (
                        tool_id
                        and not tool_id.startswith("call_")
                        and "-" in tool_id
                    )
                    if is_uuid_format:
                        # 尝试找到一个已存在的、相同工具名称的调用（使用 call_xxx ID）
                        for existing_id, state in tool_call_states.items():
                            if existing_id.startswith("call_") and (
                                state.get("name") == tool_name or not tool_name
                            ):
                                actual_tool_id = existing_id
                                break

                    # 如果不是当前活跃工具的结果，放入队列
                    if (
                        active_tool_id is not None
                        and actual_tool_id != active_tool_id
                    ):
                        pending_events.append(event)
                        continue

                    # 标记工具调用已有结果
                    if actual_tool_id and actual_tool_id in tool_call_states:
                        tool_call_states[actual_tool_id]["has_result"] = True

                    # 处理当前事件
                    for sse_data in self._process_event_with_boundaries(
                        event,
                        context,
                        text_state,
                        tool_call_states,
                        self._copilotkit_compatibility,
                    ):
                        if sse_data:
                            yield sse_data

                    # 如果这是当前活跃工具的结果，处理队列中的事件
                    if actual_tool_id == active_tool_id:
                        active_tool_id = None
                        # 处理队列中的事件
                        for sse_data in process_pending_queue():
                            yield sse_data
                    continue

                # 处理非工具相关事件（如 TEXT）
                # 需要先处理队列中的所有事件
                elif event.event == EventType.TEXT:
                    # 先处理队列中的所有事件
                    for sse_data in process_pending_queue():
                        yield sse_data
                    # 清除活跃工具 ID（因为我们要处理文本了）
                    active_tool_id = None

            # 处理边界事件注入
            for sse_data in self._process_event_with_boundaries(
                event,
                context,
                text_state,
                tool_call_states,
                self._copilotkit_compatibility,
            ):
                if sse_data:
                    yield sse_data

        # RUN_ERROR 后不发送任何清理事件
        if run_errored:
            return

        # 结束所有未结束的工具调用
        for tool_id, state in tool_call_states.items():
            if state["started"] and not state["ended"]:
                yield self._encoder.encode(
                    ToolCallEndEvent(tool_call_id=tool_id)
                )

        # 发送 TEXT_MESSAGE_END（如果有文本消息且未结束）
        if text_state["started"] and not text_state["ended"]:
            yield self._encoder.encode(
                TextMessageEndEvent(message_id=text_state["message_id"])
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
        text_state: Dict[str, Any],
        tool_call_states: Dict[str, Dict[str, bool]],
        copilotkit_compatibility: bool = False,
    ) -> Iterator[str]:
        """处理事件并注入边界事件

        Args:
            event: 用户事件
            context: 上下文
            text_state: 文本状态 {"started": bool, "ended": bool, "message_id": str}
            tool_call_states: 工具调用状态
            copilotkit_compatibility: CopilotKit 兼容模式（启用工具调用串行化）

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
        # AG-UI 协议要求：发送 TEXT_MESSAGE_START 前必须先结束所有未结束的 TOOL_CALL
        if event.event == EventType.TEXT:
            # 结束所有未结束的工具调用
            for tool_id, state in tool_call_states.items():
                if state["started"] and not state["ended"]:
                    yield self._encoder.encode(
                        ToolCallEndEvent(tool_call_id=tool_id)
                    )
                    state["ended"] = True

            # 如果文本消息未开始，或者之前已结束（需要重新开始新消息）
            if not text_state["started"] or text_state.get("ended", False):
                # 每个新文本消息需要新的 messageId
                if text_state.get("ended", False):
                    text_state["message_id"] = str(uuid.uuid4())
                yield self._encoder.encode(
                    TextMessageStartEvent(
                        message_id=text_state["message_id"],
                        role="assistant",
                    )
                )
                text_state["started"] = True
                text_state["ended"] = False

            # 发送 TEXT_MESSAGE_CONTENT
            agui_event = TextMessageContentEvent(
                message_id=text_state["message_id"],
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
        # 注意：
        # 1. AG-UI 协议要求在 TOOL_CALL_START 前必须先结束 TEXT_MESSAGE
        # 2. 当 copilotkit_compatibility=True 时，某些前端实现（如 CopilotKit）
        #    要求串行化工具调用，即在发送新的 TOOL_CALL_START 前必须先结束其他所有
        #    活跃的工具调用
        # 3. 如果一个工具调用已经结束，但收到了它的 ARGS 事件（LangChain 交错输出），
        #    需要重新开始该工具调用
        # 4. LangChain 的 on_tool_start 事件使用 run_id（UUID 格式），而流式 chunk
        #    使用 call_xxx ID。如果收到一个 UUID 格式的 ID，且已有相同工具名称的
        #    调用正在进行，则认为这是重复事件，使用已有的 ID
        if event.event == EventType.TOOL_CALL_CHUNK:
            tool_id = event.data.get("id", "")
            tool_name = event.data.get("name", "")

            # 如果文本消息未结束，先结束文本消息
            if text_state["started"] and not text_state.get("ended", False):
                yield self._encoder.encode(
                    TextMessageEndEvent(message_id=text_state["message_id"])
                )
                text_state["ended"] = True

            # 检查是否是 LangChain on_tool_start 的重复事件
            # 仅在 copilotkit_compatibility=True（兼容模式）下启用此检测
            # LangChain 的流式 chunk 使用 call_xxx ID，on_tool_start 使用 UUID
            # 如果收到 UUID 格式的 ID，且已有相同工具名称的调用（使用 call_xxx ID），
            # 则认为是重复事件
            # 注意：UUID 格式通常是 8-4-4-4-12 的格式，或者其他非 call_ 开头的长字符串
            # 我们只检测那些看起来像 UUID 的 ID（包含 - 且不是 call_ 开头）
            if copilotkit_compatibility:
                is_uuid_format = (
                    tool_id
                    and not tool_id.startswith("call_")
                    and "-" in tool_id
                )
                if is_uuid_format and tool_name:
                    for existing_id, state in tool_call_states.items():
                        # 只有当已有的调用使用 call_xxx ID 时，才认为是重复
                        if (
                            existing_id.startswith("call_")
                            and state.get("name") == tool_name
                            and state["started"]
                        ):
                            # 已有相同工具名称的调用（使用 call_xxx ID），这是重复事件
                            # 如果工具调用未结束，使用已有的 ID 发送 ARGS
                            # 如果工具调用已结束，忽略这个事件（ARGS 已经发送过了）
                            if not state["ended"]:
                                args_delta = event.data.get("args_delta", "")
                                if args_delta:
                                    yield self._encoder.encode(
                                        ToolCallArgsEvent(
                                            tool_call_id=existing_id,
                                            delta=args_delta,
                                        )
                                    )
                            # 无论是否结束，都认为这是重复事件，直接返回
                            return

            # 检查是否需要发送 TOOL_CALL_START
            need_start = False
            if tool_id:
                if tool_id not in tool_call_states:
                    # 首次见到这个工具调用
                    need_start = True
                elif tool_call_states[tool_id].get("ended", False):
                    # 工具调用已结束，但收到了新的 ARGS 事件
                    # 这种情况在 LangChain 交错输出时可能发生
                    # 需要重新开始该工具调用
                    need_start = True

            if need_start:
                # 当 copilotkit_compatibility=True 时，先结束所有其他活跃的工具调用
                if copilotkit_compatibility:
                    for other_tool_id, state in tool_call_states.items():
                        if state["started"] and not state["ended"]:
                            yield self._encoder.encode(
                                ToolCallEndEvent(tool_call_id=other_tool_id)
                            )
                            state["ended"] = True

                # 发送 TOOL_CALL_START
                yield self._encoder.encode(
                    ToolCallStartEvent(
                        tool_call_id=tool_id,
                        tool_call_name=tool_name,
                    )
                )
                tool_call_states[tool_id] = {
                    "started": True,
                    "ended": False,
                    "name": tool_name,  # 存储工具名称，用于检测重复
                }

            # 发送 TOOL_CALL_ARGS
            yield self._encoder.encode(
                ToolCallArgsEvent(
                    tool_call_id=tool_id,
                    delta=event.data.get("args_delta", ""),
                )
            )
            return

        # TOOL_RESULT 事件：确保当前工具调用已结束
        if event.event == EventType.TOOL_RESULT:
            tool_id = event.data.get("id", "")
            tool_name = event.data.get("name", "")

            # 如果文本消息未结束，先结束文本消息
            if text_state["started"] and not text_state.get("ended", False):
                yield self._encoder.encode(
                    TextMessageEndEvent(message_id=text_state["message_id"])
                )
                text_state["ended"] = True

            # 检查是否是 LangChain on_tool_end 的事件（使用 UUID 格式的 ID）
            # 仅在 copilotkit_compatibility=True（兼容模式）下启用此检测
            # 如果是，尝试找到对应的 call_xxx ID
            # UUID 格式通常是 8-4-4-4-12 的格式，或者其他非 call_ 开头且包含 - 的字符串
            actual_tool_id = tool_id
            if copilotkit_compatibility:
                is_uuid_format = (
                    tool_id
                    and not tool_id.startswith("call_")
                    and "-" in tool_id
                )
                if is_uuid_format:
                    # 尝试找到一个已存在的、相同工具名称的调用（使用 call_xxx ID）
                    for existing_id, state in tool_call_states.items():
                        if existing_id.startswith("call_") and (
                            state.get("name") == tool_name or not tool_name
                        ):
                            actual_tool_id = existing_id
                            break

            # 当 serialize_tool_calls=True 时，先结束所有其他活跃的工具调用
            if copilotkit_compatibility:
                for other_tool_id, state in tool_call_states.items():
                    if (
                        other_tool_id != actual_tool_id
                        and state["started"]
                        and not state["ended"]
                    ):
                        yield self._encoder.encode(
                            ToolCallEndEvent(tool_call_id=other_tool_id)
                        )
                        state["ended"] = True

            # 如果工具调用未开始，先补充 START
            if actual_tool_id and actual_tool_id not in tool_call_states:
                yield self._encoder.encode(
                    ToolCallStartEvent(
                        tool_call_id=actual_tool_id,
                        tool_call_name=tool_name or "",
                    )
                )
                tool_call_states[actual_tool_id] = {
                    "started": True,
                    "ended": False,
                    "name": tool_name,
                }

            # 如果当前工具调用未结束，先补充 END
            if (
                actual_tool_id
                and tool_call_states.get(actual_tool_id, {}).get("started")
                and not tool_call_states.get(actual_tool_id, {}).get("ended")
            ):
                yield self._encoder.encode(
                    ToolCallEndEvent(tool_call_id=actual_tool_id)
                )
                tool_call_states[actual_tool_id]["ended"] = True

            # 发送 TOOL_CALL_RESULT
            yield self._encoder.encode(
                ToolCallResultEvent(
                    message_id=event.data.get(
                        "message_id", f"tool-result-{actual_tool_id}"
                    ),
                    tool_call_id=actual_tool_id,
                    content=event.data.get("content")
                    or event.data.get("result", ""),
                    role="tool",
                )
            )
            return

        # ERROR 事件
        # 注意：AG-UI 协议允许 RUN_ERROR 在任何时候发送，不需要先结束其他事件
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
        # 注意：event.event 可能是字符串（Pydantic 序列化后）或枚举对象
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

        else:  # AdditionMode.PROTOCOL_ONLY
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
