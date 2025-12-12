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

from ..utils.helper import merge
from .model import (
    AdditionMode,
    AgentRequest,
    AgentResult,
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

        # 提取原始请求头
        raw_headers = dict(request.headers)

        # 构建 AgentRequest
        agent_request = AgentRequest(
            messages=messages,
            stream=request_data.get("stream", False),
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
        result_stream: AsyncIterator[AgentResult],
        context: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """将 AgentResult 流转换为 OpenAI SSE 格式

        Args:
            result_stream: AgentResult 流
            context: 上下文信息

        Yields:
            SSE 格式的字符串
        """
        tool_call_index = -1  # 从 -1 开始，第一个工具调用时变为 0
        sent_role = False

        async for result in result_stream:
            # 在格式化之前更新 tool_call_index
            if result.event == EventType.TOOL_CALL_START:
                tool_call_index += 1

            sse_data = self._format_event(
                result, context, tool_call_index, sent_role
            )

            if sse_data:
                # 更新状态
                if result.event == EventType.TEXT_MESSAGE_START:
                    sent_role = True

                yield sse_data

    def _format_event(
        self,
        result: AgentResult,
        context: Dict[str, Any],
        tool_call_index: int = 0,
        sent_role: bool = False,
    ) -> Optional[str]:
        """将单个 AgentResult 转换为 OpenAI SSE 事件

        Args:
            result: AgentResult 事件
            context: 上下文信息
            tool_call_index: 当前工具调用索引
            sent_role: 是否已发送 role

        Returns:
            SSE 格式的字符串，如果不需要输出则返回 None
        """
        # STREAM_DATA 直接输出原始数据
        if result.event == EventType.STREAM_DATA:
            raw = result.data.get("raw", "")
            return raw if raw else None

        # RUN_FINISHED 发送 [DONE]
        if result.event == EventType.RUN_FINISHED:
            return "data: [DONE]\n\n"

        # 忽略不支持的事件
        if result.event not in (
            EventType.TEXT_MESSAGE_START,
            EventType.TEXT_MESSAGE_CONTENT,
            EventType.TEXT_MESSAGE_END,
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_ARGS,
            EventType.TOOL_CALL_END,
        ):
            return None

        # 构建 delta
        delta: Dict[str, Any] = {}

        if result.event == EventType.TEXT_MESSAGE_START:
            delta["role"] = result.data.get("role", "assistant")

        elif result.event == EventType.TEXT_MESSAGE_CONTENT:
            content = result.data.get("delta", "")
            if content:
                delta["content"] = content
            else:
                return None

        elif result.event == EventType.TEXT_MESSAGE_END:
            # 发送 finish_reason
            return self._build_chunk(context, {}, finish_reason="stop")

        elif result.event == EventType.TOOL_CALL_START:
            tc_id = result.data.get("tool_call_id", "")
            tc_name = result.data.get("tool_call_name", "")
            delta["tool_calls"] = [{
                "index": tool_call_index,
                "id": tc_id,
                "type": "function",
                "function": {"name": tc_name, "arguments": ""},
            }]

        elif result.event == EventType.TOOL_CALL_ARGS:
            args_delta = result.data.get("delta", "")
            if args_delta:
                delta["tool_calls"] = [{
                    "index": tool_call_index,
                    "function": {"arguments": args_delta},
                }]
            else:
                return None

        elif result.event == EventType.TOOL_CALL_END:
            # 发送 finish_reason
            return self._build_chunk(context, {}, finish_reason="tool_calls")

        # 应用 addition
        if result.addition:
            delta = self._apply_addition(
                delta, result.addition, result.addition_mode
            )

        return self._build_chunk(context, delta)

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
        results: List[AgentResult],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """将 AgentResult 列表转换为 OpenAI 非流式响应

        Args:
            results: AgentResult 列表
            context: 上下文信息

        Returns:
            OpenAI 格式的响应字典
        """
        content_parts = []
        tool_calls = []
        finish_reason = "stop"

        for result in results:
            if result.event == EventType.TEXT_MESSAGE_CONTENT:
                content_parts.append(result.data.get("delta", ""))

            elif result.event == EventType.TOOL_CALL_START:
                tc_id = result.data.get("tool_call_id", "")
                tc_name = result.data.get("tool_call_name", "")
                tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": tc_name, "arguments": ""},
                })

            elif result.event == EventType.TOOL_CALL_ARGS:
                if tool_calls:
                    args = result.data.get("delta", "")
                    tool_calls[-1]["function"]["arguments"] += args

            elif result.event == EventType.TOOL_CALL_END:
                finish_reason = "tool_calls"

        # 构建响应
        content = "".join(content_parts) if content_parts else None
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }

        if tool_calls:
            message["tool_calls"] = tool_calls
            if not content:
                finish_reason = "tool_calls"

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
        addition: Dict[str, Any],
        mode: AdditionMode,
    ) -> Dict[str, Any]:
        """应用 addition 字段

        Args:
            delta: 原始 delta 数据
            addition: 附加字段
            mode: 合并模式

        Returns:
            合并后的 delta 数据
        """
        if mode == AdditionMode.REPLACE:
            delta.update(addition)

        elif mode == AdditionMode.MERGE:
            delta = merge(delta, addition)

        elif mode == AdditionMode.PROTOCOL_ONLY:
            delta = merge(delta, addition, no_new_field=True)

        return delta
