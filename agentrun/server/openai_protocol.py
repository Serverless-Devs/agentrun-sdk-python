"""OpenAI Completions API 协议实现 / OpenAI Completions API Protocol Implementation

实现 OpenAI Chat Completions API 兼容接口。
参考: https://platform.openai.com/docs/api-reference/chat/create

本实现将 AgentResult 事件转换为 OpenAI 流式响应格式。
"""

import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
import pydash

from ..utils.helper import merge, MergeOptions
from .model import (
    AgentEvent,
    AgentRequest,
    EventType,
    Message,
    MessageRole,
    OpenAIProtocolConfig,
    ServerConfig,
    Tool,
    ToolCall,
)
from .protocol import BaseProtocolHandler

if TYPE_CHECKING:
    from .invoker import AgentInvoker


# ============================================================================
# OpenAI 协议处理器
# ============================================================================


DEFAULT_PREFIX = "/openai/v1"


class OpenAIProtocolHandler(BaseProtocolHandler):
    """OpenAI Completions API 协议处理器

    实现 OpenAI Chat Completions API 兼容接口。
    参考: https://platform.openai.com/docs/api-reference/chat/create

    特点:
    - 完全兼容 OpenAI API 格式
    - 支持流式和非流式响应
    - 支持工具调用
    - AgentResult 事件自动转换为 OpenAI 格式

    支持的事件映射:
    - TEXT_MESSAGE_* → delta.content
    - TOOL_CALL_* → delta.tool_calls
    - RUN_FINISHED → [DONE]
    - 其他事件 → 忽略

    Example:
        >>> from agentrun.server import AgentRunServer
        >>>
        >>> def my_agent(request):
        ...     return "Hello, world!"
        >>>
        >>> server = AgentRunServer(invoke_agent=my_agent)
        >>> server.start(port=8000)
        # 可访问: POST http://localhost:8000/openai/v1/chat/completions
    """

    name = "openai_chat_completions"

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config.openai if config else None

    def get_prefix(self) -> str:
        """OpenAI 协议建议使用 /openai/v1 前缀"""
        return pydash.get(self.config, "prefix", DEFAULT_PREFIX)

    def get_model_name(self) -> str:
        """获取默认模型名称"""
        return pydash.get(self.config, "model_name", "agentrun")

    def as_fastapi_router(self, agent_invoker: "AgentInvoker") -> APIRouter:
        """创建 OpenAI 协议的 FastAPI Router"""
        router = APIRouter()

        @router.post("/chat/completions")
        async def chat_completions(request: Request):
            """OpenAI Chat Completions 端点"""
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

                if agent_request.stream:
                    # 流式响应
                    event_stream = self._format_stream(
                        agent_invoker.invoke_stream(agent_request),
                        context,
                    )
                    return StreamingResponse(
                        event_stream,
                        media_type="text/event-stream",
                        headers=sse_headers,
                    )
                else:
                    # 非流式响应
                    results = await agent_invoker.invoke(agent_request)
                    if hasattr(results, "__aiter__"):
                        # 收集流式结果
                        result_list = []
                        async for r in results:
                            result_list.append(r)
                        results = result_list

                    formatted = self._format_non_stream(results, context)
                    return JSONResponse(formatted)

            except ValueError as e:
                return JSONResponse(
                    {
                        "error": {
                            "message": str(e),
                            "type": "invalid_request_error",
                        }
                    },
                    status_code=400,
                )
            except Exception as e:
                return JSONResponse(
                    {"error": {"message": str(e), "type": "internal_error"}},
                    status_code=500,
                )

        @router.get("/models")
        async def list_models():
            """列出可用模型"""
            return {
                "object": "list",
                "data": [{
                    "id": self.get_model_name(),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "agentrun",
                }],
            }

        return router

    async def parse_request(
        self,
        request: Request,
        request_data: Dict[str, Any],
    ) -> tuple[AgentRequest, Dict[str, Any]]:
        """解析 OpenAI 格式的请求

        Args:
            request: FastAPI Request 对象
            request_data: HTTP 请求体 JSON 数据

        Returns:
            tuple: (AgentRequest, context)
        """
        # 验证必需字段
        if "messages" not in request_data:
            raise ValueError("Missing required field: messages")

        # 创建上下文
        context = {
            "response_id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "model": request_data.get("model", self.get_model_name()),
            "created": int(time.time()),
        }

        # 解析消息列表
        messages = self._parse_messages(request_data["messages"])

        # 解析工具列表
        tools = self._parse_tools(request_data.get("tools"))

        # 构建 AgentRequest
        agent_request = AgentRequest(
            protocol="openai",  # 设置协议名称
            messages=messages,
            stream=request_data.get("stream", False),
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
                raise ValueError(f"Invalid message format: {msg_data}")

            if "role" not in msg_data:
                raise ValueError("Message missing 'role' field")

            try:
                role = MessageRole(msg_data["role"])
            except ValueError as e:
                raise ValueError(
                    f"Invalid message role: {msg_data['role']}"
                ) from e

            # 解析 tool_calls
            tool_calls = None
            if msg_data.get("tool_calls"):
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=tc.get("function", {}),
                    )
                    for tc in msg_data["tool_calls"]
                ]

            messages.append(
                Message(
                    role=role,
                    content=msg_data.get("content"),
                    name=msg_data.get("name"),
                    tool_calls=tool_calls,
                    tool_call_id=msg_data.get("tool_call_id"),
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
        """将 AgentEvent 流转换为 OpenAI SSE 格式

        自动生成边界事件：
        - 首个 TEXT 事件前发送 role: assistant
        - 工具调用自动追踪索引
        - 流结束发送 finish_reason 和 [DONE]

        Args:
            event_stream: AgentEvent 流
            context: 上下文信息

        Yields:
            SSE 格式的字符串
        """
        # 状态追踪
        sent_role = False
        has_text = False
        tool_call_index = -1  # 从 -1 开始，第一个工具调用时变为 0
        # 工具调用状态：{tool_id: {"started": bool, "index": int}}
        tool_call_states: Dict[str, Dict[str, Any]] = {}
        has_tool_calls = False

        async for event in event_stream:
            # RAW 事件直接透传
            if event.event == EventType.RAW:
                raw = event.data.get("raw", "")
                if raw:
                    if not raw.endswith("\n\n"):
                        raw = raw.rstrip("\n") + "\n\n"
                    yield raw
                continue

            # TEXT 事件
            if event.event == EventType.TEXT:
                delta: Dict[str, Any] = {}
                # 首个 TEXT 事件，发送 role
                if not sent_role:
                    delta["role"] = "assistant"
                    sent_role = True

                content = event.data.get("delta", "")
                if content:
                    delta["content"] = content
                    has_text = True

                    # 应用 addition
                    if event.addition:
                        delta = self._apply_addition(
                            delta,
                            event.addition,
                            event.addition_merge_options,
                        )

                    yield self._build_chunk(context, delta)
                continue

            # TOOL_CALL_CHUNK 事件
            if event.event == EventType.TOOL_CALL_CHUNK:
                tool_id = event.data.get("id", "")
                tool_name = event.data.get("name", "")
                args_delta = event.data.get("args_delta", "")

                delta = {}

                # 首次见到这个工具调用
                if tool_id and tool_id not in tool_call_states:
                    tool_call_index += 1
                    tool_call_states[tool_id] = {
                        "started": True,
                        "index": tool_call_index,
                    }
                    has_tool_calls = True

                    # 发送工具调用开始（包含 id, name）
                    delta["tool_calls"] = [{
                        "index": tool_call_index,
                        "id": tool_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": ""},
                    }]
                    yield self._build_chunk(context, delta)
                    delta = {}

                # 发送参数增量
                if args_delta:
                    current_index = tool_call_states.get(tool_id, {}).get(
                        "index", tool_call_index
                    )
                    delta["tool_calls"] = [{
                        "index": current_index,
                        "function": {"arguments": args_delta},
                    }]

                    # 应用 addition
                    if event.addition:
                        delta = self._apply_addition(
                            delta,
                            event.addition,
                            event.addition_merge_options,
                        )

                    yield self._build_chunk(context, delta)
                continue

            # TOOL_RESULT 事件：OpenAI 协议通常不在流中输出工具结果
            if event.event == EventType.TOOL_RESULT:
                continue

            # TOOL_RESULT_CHUNK 事件：OpenAI 协议不支持流式工具输出
            if event.event == EventType.TOOL_RESULT_CHUNK:
                continue

            # HITL 事件：OpenAI 协议不支持
            if event.event == EventType.HITL:
                continue

            # 其他事件忽略
            # (ERROR, STATE, CUSTOM 等不直接映射到 OpenAI 格式)

        # 流结束后发送 finish_reason 和 [DONE]
        if has_tool_calls:
            yield self._build_chunk(context, {}, finish_reason="tool_calls")
        elif has_text:
            yield self._build_chunk(context, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    def _build_chunk(
        self,
        context: Dict[str, Any],
        delta: Dict[str, Any],
        finish_reason: Optional[str] = None,
    ) -> str:
        """构建 OpenAI 流式响应块

        Args:
            context: 上下文信息
            delta: delta 数据
            finish_reason: 结束原因

        Returns:
            SSE 格式的字符串
        """
        chunk = {
            "id": context.get(
                "response_id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
            ),
            "object": "chat.completion.chunk",
            "created": context.get("created", int(time.time())),
            "model": context.get("model", "agentrun"),
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        json_str = json.dumps(chunk, ensure_ascii=False)
        return f"data: {json_str}\n\n"

    def _format_non_stream(
        self,
        events: List[AgentEvent],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """将 AgentEvent 列表转换为 OpenAI 非流式响应

        自动追踪工具调用状态。

        Args:
            events: AgentEvent 列表
            context: 上下文信息

        Returns:
            OpenAI 格式的响应字典
        """
        content_parts: List[str] = []
        # 工具调用状态：{tool_id: {id, name, arguments}}
        tool_call_map: Dict[str, Dict[str, Any]] = {}
        has_tool_calls = False

        for event in events:
            if event.event == EventType.TEXT:
                content_parts.append(event.data.get("delta", ""))

            elif event.event == EventType.TOOL_CALL_CHUNK:
                tool_id = event.data.get("id", "")
                tool_name = event.data.get("name", "")
                args_delta = event.data.get("args_delta", "")

                if tool_id:
                    if tool_id not in tool_call_map:
                        tool_call_map[tool_id] = {
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": ""},
                        }
                        has_tool_calls = True

                    if args_delta:
                        tool_call_map[tool_id]["function"][
                            "arguments"
                        ] += args_delta

        # 构建响应
        content = "".join(content_parts) if content_parts else None
        finish_reason = "tool_calls" if has_tool_calls else "stop"

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }

        if tool_call_map:
            message["tool_calls"] = list(tool_call_map.values())

        response = {
            "id": context.get(
                "response_id", f"chatcmpl-{uuid.uuid4().hex[:12]}"
            ),
            "object": "chat.completion",
            "created": context.get("created", int(time.time())),
            "model": context.get("model", "agentrun"),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
        }

        return response

    def _apply_addition(
        self,
        delta: Dict[str, Any],
        addition: Optional[Dict[str, Any]],
        merge_options: Optional[MergeOptions] = None,
    ) -> Dict[str, Any]:
        """应用 addition 字段

        Args:
            delta: 原始 delta 数据
            addition: 附加字段
            merge_options: 合并选项，透传给 utils.helper.merge

        Returns:
            合并后的 delta 数据
        """
        if not addition:
            return delta

        return merge(delta, addition, **(merge_options or {}))
