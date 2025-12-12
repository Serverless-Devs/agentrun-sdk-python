"""AG-UI (Agent-User Interaction Protocol) 协议实现

AG-UI 是一种开源、轻量级、基于事件的协议，用于标准化 AI Agent 与前端应用之间的交互。
参考: https://docs.ag-ui.com/

本实现将 AgentResult 事件转换为 AG-UI SSE 格式。
"""

import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING
import uuid

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
# AG-UI 事件类型映射
# ============================================================================


# EventType 到 AG-UI 事件类型名的映射
AGUI_EVENT_TYPE_MAP = {
    EventType.RUN_STARTED: "RUN_STARTED",
    EventType.RUN_FINISHED: "RUN_FINISHED",
    EventType.RUN_ERROR: "RUN_ERROR",
    EventType.STEP_STARTED: "STEP_STARTED",
    EventType.STEP_FINISHED: "STEP_FINISHED",
    EventType.TEXT_MESSAGE_START: "TEXT_MESSAGE_START",
    EventType.TEXT_MESSAGE_CONTENT: "TEXT_MESSAGE_CONTENT",
    EventType.TEXT_MESSAGE_END: "TEXT_MESSAGE_END",
    EventType.TEXT_MESSAGE_CHUNK: "TEXT_MESSAGE_CHUNK",
    EventType.TOOL_CALL_START: "TOOL_CALL_START",
    EventType.TOOL_CALL_ARGS: "TOOL_CALL_ARGS",
    EventType.TOOL_CALL_END: "TOOL_CALL_END",
    EventType.TOOL_CALL_RESULT: "TOOL_CALL_RESULT",
    EventType.TOOL_CALL_CHUNK: "TOOL_CALL_CHUNK",
    EventType.STATE_SNAPSHOT: "STATE_SNAPSHOT",
    EventType.STATE_DELTA: "STATE_DELTA",
    EventType.MESSAGES_SNAPSHOT: "MESSAGES_SNAPSHOT",
    EventType.ACTIVITY_SNAPSHOT: "ACTIVITY_SNAPSHOT",
    EventType.ACTIVITY_DELTA: "ACTIVITY_DELTA",
    EventType.REASONING_START: "REASONING_START",
    EventType.REASONING_MESSAGE_START: "REASONING_MESSAGE_START",
    EventType.REASONING_MESSAGE_CONTENT: "REASONING_MESSAGE_CONTENT",
    EventType.REASONING_MESSAGE_END: "REASONING_MESSAGE_END",
    EventType.REASONING_MESSAGE_CHUNK: "REASONING_MESSAGE_CHUNK",
    EventType.REASONING_END: "REASONING_END",
    EventType.META_EVENT: "META_EVENT",
    EventType.RAW: "RAW",
    EventType.CUSTOM: "CUSTOM",
}


# ============================================================================
# AG-UI 协议处理器
# ============================================================================

DEFAULT_PREFIX = "/ag-ui/agent"


class AGUIProtocolHandler(BaseProtocolHandler):
    """AG-UI 协议处理器

    实现 AG-UI (Agent-User Interaction Protocol) 兼容接口。
    参考: https://docs.ag-ui.com/

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
        # 可访问: POST http://localhost:8000/agui/v1/run
    """

    name = "agui"

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config.openai if config else None

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
                    media_type="text/event-stream",
                    headers=sse_headers,
                )

            except ValueError as e:
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

    def _format_event(self, result, context):
        # 统一将字符串或 dict 标准化为 AgentResult，后续代码可安全访问 result.event 等属性
        if isinstance(result, str):
            # 选择合适的文本事件类型（优先使用 TEXT_MESSAGE_CHUNK，否则回退到 TEXT_MESSAGE_START）
            ev_key = None
            try:
                members = getattr(EventType, "__members__", None)
                if members and "TEXT_MESSAGE_CHUNK" in members:
                    ev_key = "TEXT_MESSAGE_CHUNK"
                elif members and "TEXT_MESSAGE_START" in members:
                    ev_key = "TEXT_MESSAGE_START"
            except Exception:
                ev_key = None

            try:
                ev = EventType[ev_key] if ev_key else list(EventType)[0]
            except Exception:
                ev = list(EventType)[0]

            result = AgentResult(event=ev, data={"text": result})

        elif isinstance(result, dict):
            # 尝试从 dict 中解析 event 字段为 EventType
            ev = None
            evt = result.get("event")
            try:
                members = getattr(EventType, "__members__", None)
                if isinstance(evt, str) and members and evt in members:
                    ev = EventType[evt]
                else:
                    # 尝试按 value 匹配
                    for e in list(EventType):
                        if str(getattr(e, "value", e)) == str(evt):
                            ev = e
                            break
            except Exception:
                ev = None

            if ev is None:
                ev = list(EventType)[0]

            result = AgentResult(event=ev, data=result.get("data", result))

        # 之后的逻辑可以安全地认为 result 是 AgentResult 对象
        timestamp = int(time.time() * 1000)

        # 基础事件数据
        event_data: Dict[str, Any] = {
            "type": result.event,
            "timestamp": timestamp,
        }

        # 根据事件类型添加特定字段
        event_data = self._add_event_fields(result, event_data, context)

        # 处理 addition
        if result.addition:
            event_data = self._apply_addition(
                event_data, result.addition, result.addition_mode
            )

        # 转换为 SSE 格式
        json_str = json.dumps(event_data, ensure_ascii=False)
        return f"data: {json_str}\n\n"

    def _add_event_fields(
        self,
        result: AgentResult,
        event_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """根据事件类型添加特定字段

        Args:
            result: AgentResult 事件
            event_data: 基础事件数据
            context: 上下文信息

        Returns:
            完整的事件数据
        """
        data = result.data

        # 生命周期事件
        if result.event in (EventType.RUN_STARTED, EventType.RUN_FINISHED):
            event_data["threadId"] = data.get("thread_id") or context.get(
                "thread_id"
            )
            event_data["runId"] = data.get("run_id") or context.get("run_id")

        elif result.event == EventType.RUN_ERROR:
            event_data["message"] = data.get("message", "")
            event_data["code"] = data.get("code")

        elif result.event in (EventType.STEP_STARTED, EventType.STEP_FINISHED):
            event_data["stepName"] = data.get("step_name")

        # 文本消息事件
        elif result.event == EventType.TEXT_MESSAGE_START:
            event_data["messageId"] = data.get("message_id", str(uuid.uuid4()))
            event_data["role"] = data.get("role", "assistant")

        elif result.event == EventType.TEXT_MESSAGE_CONTENT:
            event_data["messageId"] = data.get("message_id", "")
            event_data["delta"] = data.get("delta", "")

        elif result.event == EventType.TEXT_MESSAGE_END:
            event_data["messageId"] = data.get("message_id", "")

        elif result.event == EventType.TEXT_MESSAGE_CHUNK:
            event_data["messageId"] = data.get("message_id")
            event_data["role"] = data.get("role")
            event_data["delta"] = data.get("delta", "")

        # 工具调用事件
        elif result.event == EventType.TOOL_CALL_START:
            event_data["toolCallId"] = data.get("tool_call_id", "")
            event_data["toolCallName"] = data.get("tool_call_name", "")
            if data.get("parent_message_id"):
                event_data["parentMessageId"] = data["parent_message_id"]

        elif result.event == EventType.TOOL_CALL_ARGS:
            event_data["toolCallId"] = data.get("tool_call_id", "")
            event_data["delta"] = data.get("delta", "")

        elif result.event == EventType.TOOL_CALL_END:
            event_data["toolCallId"] = data.get("tool_call_id", "")

        elif result.event == EventType.TOOL_CALL_RESULT:
            event_data["toolCallId"] = data.get("tool_call_id", "")
            event_data["result"] = data.get("result", "")

        elif result.event == EventType.TOOL_CALL_CHUNK:
            event_data["toolCallId"] = data.get("tool_call_id")
            event_data["toolCallName"] = data.get("tool_call_name")
            event_data["delta"] = data.get("delta", "")
            if data.get("parent_message_id"):
                event_data["parentMessageId"] = data["parent_message_id"]

        # 状态管理事件
        elif result.event == EventType.STATE_SNAPSHOT:
            event_data["snapshot"] = data.get("snapshot", {})

        elif result.event == EventType.STATE_DELTA:
            event_data["delta"] = data.get("delta", [])

        # 消息快照事件
        elif result.event == EventType.MESSAGES_SNAPSHOT:
            event_data["messages"] = data.get("messages", [])

        # Activity 事件
        elif result.event == EventType.ACTIVITY_SNAPSHOT:
            event_data["snapshot"] = data.get("snapshot", {})

        elif result.event == EventType.ACTIVITY_DELTA:
            event_data["delta"] = data.get("delta", [])

        # Reasoning 事件
        elif result.event == EventType.REASONING_START:
            event_data["reasoningId"] = data.get(
                "reasoning_id", str(uuid.uuid4())
            )

        elif result.event == EventType.REASONING_MESSAGE_START:
            event_data["messageId"] = data.get("message_id", str(uuid.uuid4()))
            event_data["reasoningId"] = data.get("reasoning_id", "")

        elif result.event == EventType.REASONING_MESSAGE_CONTENT:
            event_data["messageId"] = data.get("message_id", "")
            event_data["delta"] = data.get("delta", "")

        elif result.event == EventType.REASONING_MESSAGE_END:
            event_data["messageId"] = data.get("message_id", "")

        elif result.event == EventType.REASONING_MESSAGE_CHUNK:
            event_data["messageId"] = data.get("message_id")
            event_data["delta"] = data.get("delta", "")

        elif result.event == EventType.REASONING_END:
            event_data["reasoningId"] = data.get("reasoning_id", "")

        # Meta 事件
        elif result.event == EventType.META_EVENT:
            event_data["name"] = data.get("name", "")
            event_data["value"] = data.get("value")

        # RAW 事件
        elif result.event == EventType.RAW:
            event_data["event"] = data.get("event", {})

        # CUSTOM 事件
        elif result.event == EventType.CUSTOM:
            event_data["name"] = data.get("name", "")
            event_data["value"] = data.get("value")

        return event_data

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
