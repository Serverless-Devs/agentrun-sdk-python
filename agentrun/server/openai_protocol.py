"""OpenAI Completions API 协议实现 / OpenAI Completions API Protocol Implementation

基于 Router 的设计:
- 协议自己创建 FastAPI Router
- 定义所有端点和处理逻辑
- Server 只需挂载 Router

生命周期钩子:
- OpenAI 协议支持部分钩子（主要是文本消息和工具调用）
- 不支持的钩子返回空迭代器
"""

import inspect
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
from fastapi.responses import JSONResponse, StreamingResponse

from .model import (
    AgentEvent,
    AgentLifecycleHooks,
    AgentRequest,
    AgentResponse,
    AgentResult,
    AgentRunResult,
    AgentStreamResponse,
    AgentStreamResponseChoice,
    AgentStreamResponseDelta,
    Message,
    MessageRole,
)
from .protocol import BaseProtocolHandler

if TYPE_CHECKING:
    from .invoker import AgentInvoker


# ============================================================================
# OpenAI 协议生命周期钩子实现
# ============================================================================


class OpenAILifecycleHooks(AgentLifecycleHooks):
    """OpenAI 协议的生命周期钩子实现

    OpenAI Chat Completions API 支持的事件有限，主要是：
    - 文本消息流式输出（通过 delta.content）
    - 工具调用流式输出（通过 delta.tool_calls）

    不支持的事件（如 step、state 等）返回 None。

    所有 on_* 方法直接返回 AgentEvent，可以直接 yield。
    """

    def __init__(self, context: Dict[str, Any]):
        """初始化钩子

        Args:
            context: 运行上下文，包含 response_id, model 等
        """
        self.context = context
        self.response_id = context.get(
            "response_id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
        )
        self.model = context.get("model", "agentrun-model")
        self.created = context.get("created", int(time.time()))

    def _create_event(
        self,
        delta: Dict[str, Any],
        finish_reason: Optional[str] = None,
        event_type: str = "text_message",
    ) -> AgentEvent:
        """创建 AgentEvent

        Args:
            delta: delta 内容
            finish_reason: 结束原因
            event_type: 事件类型

        Returns:
            AgentEvent 对象
        """
        chunk = {
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        raw_sse = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        return AgentEvent(event_type=event_type, data=chunk, raw_sse=raw_sse)

    # =========================================================================
    # 生命周期事件方法 (on_*) - 直接返回 AgentEvent 或 None
    # =========================================================================

    def on_run_start(self) -> Optional[AgentEvent]:
        """OpenAI 不支持 run_start 事件"""
        return None

    def on_run_finish(self) -> AgentEvent:
        """OpenAI 发送 [DONE] 标记"""
        return AgentEvent(event_type="run_finish", raw_sse="data: [DONE]\n\n")

    def on_run_error(
        self, error: str, code: Optional[str] = None
    ) -> Optional[AgentEvent]:
        """OpenAI 错误通过 HTTP 状态码返回"""
        return None

    def on_step_start(
        self, step_name: Optional[str] = None
    ) -> Optional[AgentEvent]:
        """OpenAI 不支持 step 事件"""
        return None

    def on_step_finish(
        self, step_name: Optional[str] = None
    ) -> Optional[AgentEvent]:
        """OpenAI 不支持 step 事件"""
        return None

    def on_text_message_start(
        self, message_id: str, role: str = "assistant"
    ) -> AgentEvent:
        """发送消息开始，包含 role"""
        return self._create_event(
            {"role": role}, event_type="text_message_start"
        )

    def on_text_message_content(
        self, message_id: str, delta: str
    ) -> Optional[AgentEvent]:
        """发送消息内容增量"""
        if not delta:
            return None
        return self._create_event(
            {"content": delta}, event_type="text_message_content"
        )

    def on_text_message_end(self, message_id: str) -> AgentEvent:
        """发送消息结束，包含 finish_reason"""
        return self._create_event(
            {}, finish_reason="stop", event_type="text_message_end"
        )

    def on_tool_call_start(
        self,
        id: str,
        name: str,
        parent_message_id: Optional[str] = None,
    ) -> AgentEvent:
        """发送工具调用开始"""
        # 记录当前工具调用索引
        if "tool_call_index" not in self.context:
            self.context["tool_call_index"] = 0
        else:
            self.context["tool_call_index"] += 1

        index = self.context["tool_call_index"]

        return self._create_event(
            {
                "tool_calls": [{
                    "index": index,
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }]
            },
            event_type="tool_call_start",
        )

    def on_tool_call_args_delta(
        self, id: str, delta: str
    ) -> Optional[AgentEvent]:
        """发送工具调用参数增量"""
        if not delta:
            return None
        index = self.context.get("tool_call_index", 0)
        return self._create_event(
            {
                "tool_calls": [{
                    "index": index,
                    "function": {"arguments": delta},
                }]
            },
            event_type="tool_call_args_delta",
        )

    def on_tool_call_args(
        self, id: str, args: Union[str, Dict[str, Any]]
    ) -> Optional[AgentEvent]:
        """工具调用参数完成 - OpenAI 通过增量累积"""
        return None

    def on_tool_call_result_delta(
        self, id: str, delta: str
    ) -> Optional[AgentEvent]:
        """工具调用结果增量 - OpenAI 不直接支持"""
        return None

    def on_tool_call_result(self, id: str, result: str) -> Optional[AgentEvent]:
        """工具调用结果 - OpenAI 需要作为 tool role 消息返回"""
        return None

    def on_tool_call_end(self, id: str) -> Optional[AgentEvent]:
        """工具调用结束"""
        return None

    def on_state_snapshot(
        self, snapshot: Dict[str, Any]
    ) -> Optional[AgentEvent]:
        """OpenAI 不支持状态事件"""
        return None

    def on_state_delta(
        self, delta: List[Dict[str, Any]]
    ) -> Optional[AgentEvent]:
        """OpenAI 不支持状态事件"""
        return None

    def on_custom_event(self, name: str, value: Any) -> Optional[AgentEvent]:
        """OpenAI 不支持自定义事件"""
        return None


# ============================================================================
# OpenAI 协议处理器
# ============================================================================


class OpenAIProtocolHandler(BaseProtocolHandler):
    """OpenAI Completions API 协议处理器

    实现 OpenAI Chat Completions API 兼容接口
    参考: https://platform.openai.com/docs/api-reference/chat/create

    特点:
    - 完全兼容 OpenAI API 格式
    - 支持流式和非流式响应
    - 支持工具调用
    - 提供生命周期钩子（部分支持）
    """

    def get_prefix(self) -> str:
        """OpenAI 协议建议使用 /openai/v1 前缀"""
        return "/openai/v1"

    def create_hooks(self, context: Dict[str, Any]) -> AgentLifecycleHooks:
        """创建 OpenAI 协议的生命周期钩子"""
        return OpenAILifecycleHooks(context)

    def as_fastapi_router(self, agent_invoker: "AgentInvoker") -> APIRouter:
        """创建 OpenAI 协议的 FastAPI Router"""
        router = APIRouter()

        @router.post("/chat/completions")
        async def chat_completions(request: Request):
            """OpenAI Chat Completions 端点"""
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

                # 3. 格式化响应
                is_stream = agent_request.stream or self._is_iterator(
                    agent_result
                )

                if is_stream:
                    # 流式响应
                    response_stream = self.format_response(
                        agent_result, agent_request, context
                    )
                    if inspect.isawaitable(response_stream):
                        response_stream = await response_stream
                    return StreamingResponse(
                        response_stream,
                        media_type="text/event-stream",
                        headers=sse_headers,
                    )
                else:
                    # 非流式响应
                    formatted_result = await self._format_non_stream_response(
                        agent_result, agent_request, context
                    )
                    return JSONResponse(formatted_result)

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
                    "id": "agentrun-model",
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

        Raises:
            ValueError: 请求格式不正确
        """
        # 验证必需字段
        if "messages" not in request_data:
            raise ValueError("Missing required field: messages")

        # 解析消息列表
        messages = []
        for msg_data in request_data["messages"]:
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

            messages.append(
                Message(
                    role=role,
                    content=msg_data.get("content"),
                    name=msg_data.get("name"),
                    tool_calls=msg_data.get("tool_calls"),
                    tool_call_id=msg_data.get("tool_call_id"),
                )
            )

        # 创建上下文
        context = {
            "response_id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "model": request_data.get("model", "agentrun-model"),
            "created": int(time.time()),
        }

        # 创建钩子
        hooks = self.create_hooks(context)

        # 提取原始请求头
        raw_headers = dict(request.headers)

        # 构建 AgentRequest（只包含协议无关的核心字段）
        # OpenAI 特定参数（temperature、top_p、max_tokens 等）保留在 raw_body 中
        agent_request = AgentRequest(
            messages=messages,
            stream=request_data.get("stream", False),
            tools=request_data.get("tools"),
            raw_headers=raw_headers,
            raw_body=request_data,
            hooks=hooks,
        )

        return agent_request, context

    async def format_response(
        self,
        result: AgentResult,
        request: AgentRequest,
        context: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """格式化流式响应为 OpenAI SSE 格式

        Agent 可以 yield 三种类型的内容:
        1. 普通字符串 - 会被包装成 OpenAI 流式响应格式
        2. AgentEvent - 直接输出其 raw_sse（如果是 OpenAI 格式）
        3. None - 忽略

        Args:
            result: Agent 执行结果
            request: 原始请求
            context: 运行上下文

        Yields:
            SSE 格式的数据行
        """
        hooks = request.hooks
        message_id = str(uuid.uuid4())
        text_message_started = False

        # 处理内容
        content = self._extract_content(result)

        if self._is_iterator(content):
            # 流式内容
            async for chunk in self._iterate_content(content):
                if chunk is None:
                    continue

                # 检查是否是 AgentEvent
                if isinstance(chunk, AgentEvent):
                    # 只输出有 raw_sse 且是 OpenAI 格式的事件
                    if chunk.raw_sse and chunk.event_type.startswith(
                        ("text_message", "tool_call", "run_finish")
                    ):
                        yield chunk.raw_sse
                    continue

                # 普通文本内容
                if isinstance(chunk, str) and chunk:
                    if not text_message_started and hooks:
                        # 延迟发送消息开始
                        event = hooks.on_text_message_start(message_id)
                        if event and event.raw_sse:
                            yield event.raw_sse
                        text_message_started = True

                    if hooks:
                        event = hooks.on_text_message_content(message_id, chunk)
                        if event and event.raw_sse:
                            yield event.raw_sse
        else:
            # 非流式内容转换为单个 chunk
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

        # 发送消息结束（如果有文本消息）
        if text_message_started and hooks:
            event = hooks.on_text_message_end(message_id)
            if event and event.raw_sse:
                yield event.raw_sse

        # 发送运行结束
        if hooks:
            event = hooks.on_run_finish()
            if event and event.raw_sse:
                yield event.raw_sse

    async def _format_non_stream_response(
        self,
        result: AgentResult,
        request: AgentRequest,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """格式化非流式响应

        Args:
            result: Agent 执行结果
            request: 原始请求
            context: 运行上下文

        Returns:
            OpenAI 格式的响应字典
        """
        # 检测 ModelResponse (来自 Model Service)
        if self._is_model_response(result):
            return self._format_model_response(result, request)

        # 处理 AgentRunResult
        if isinstance(result, AgentRunResult):
            content = result.content
            if isinstance(content, str):
                return self._build_completion_response(content, context)
            raise TypeError(
                "AgentRunResult.content must be str for non-stream, got"
                f" {type(content)}"
            )

        # 处理字符串
        if isinstance(result, str):
            return self._build_completion_response(result, context)

        # 处理 AgentResponse
        if isinstance(result, AgentResponse):
            return self._ensure_openai_format(result, request, context)

        raise TypeError(
            "Expected AgentRunResult, AgentResponse, or str, got"
            f" {type(result)}"
        )

    def _build_completion_response(
        self, content: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构建完整的 OpenAI completion 响应"""
        return {
            "id": context.get(
                "response_id", f"chatcmpl-{uuid.uuid4().hex[:12]}"
            ),
            "object": "chat.completion",
            "created": context.get("created", int(time.time())),
            "model": context.get("model", "agentrun-model"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }],
        }

    def _extract_content(self, result: AgentResult) -> Any:
        """从结果中提取内容"""
        if isinstance(result, AgentRunResult):
            return result.content
        if isinstance(result, AgentResponse):
            return result.content
        if isinstance(result, str):
            return result
        # 可能是迭代器
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

            # 使用哨兵值来检测迭代结束，避免 StopIteration 传播到 Future
            _STOP = object()

            def _safe_next():
                try:
                    return next(iterator)
                except StopIteration:
                    return _STOP

            while True:
                chunk = await loop.run_in_executor(None, _safe_next)
                if chunk is _STOP:
                    break
                yield chunk

    def _is_model_response(self, obj: Any) -> bool:
        """检查对象是否是 Model Service 的 ModelResponse"""
        if isinstance(obj, (str, AgentResponse, AgentRunResult, dict)):
            return False
        return (
            hasattr(obj, "choices")
            and hasattr(obj, "model")
            and (hasattr(obj, "usage") or hasattr(obj, "created"))
        )

    def _format_model_response(
        self, response: Any, request: AgentRequest
    ) -> Dict[str, Any]:
        """格式化 ModelResponse 为 OpenAI 格式"""
        if hasattr(response, "model_dump"):
            return response.model_dump(exclude_none=True)
        if hasattr(response, "dict"):
            return response.dict(exclude_none=True)

        # 手动转换
        result = {
            "id": getattr(
                response, "id", f"chatcmpl-{int(time.time() * 1000)}"
            ),
            "object": getattr(response, "object", "chat.completion"),
            "created": getattr(response, "created", int(time.time())),
            "model": getattr(
                response, "model", request.model or "agentrun-model"
            ),
            "choices": [],
        }

        if hasattr(response, "choices"):
            for choice in response.choices:
                choice_dict = {
                    "index": getattr(choice, "index", 0),
                    "finish_reason": getattr(choice, "finish_reason", None),
                }
                if hasattr(choice, "message"):
                    msg = choice.message
                    choice_dict["message"] = {
                        "role": getattr(msg, "role", "assistant"),
                        "content": getattr(msg, "content", None),
                    }
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        choice_dict["message"]["tool_calls"] = msg.tool_calls
                result["choices"].append(choice_dict)

        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            result["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

        return result

    def _ensure_openai_format(
        self,
        response: AgentResponse,
        request: AgentRequest,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """确保 AgentResponse 符合 OpenAI 格式"""
        if response.content and not response.choices:
            return self._build_completion_response(response.content, context)

        json_str = response.model_dump_json(exclude_none=True)
        result = json.loads(json_str)

        if "id" not in result:
            result["id"] = context.get(
                "response_id", f"chatcmpl-{uuid.uuid4().hex[:12]}"
            )
        if "object" not in result:
            result["object"] = "chat.completion"
        if "created" not in result:
            result["created"] = context.get("created", int(time.time()))
        if "model" not in result:
            result["model"] = context.get(
                "model", request.model or "agentrun-model"
            )

        result.pop("content", None)
        result.pop("extra", None)

        return result
