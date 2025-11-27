"""OpenAI Completions API 协议实现 / OpenAI Completions API 协议Implements

基于 Router 的设计:
- 协议自己创建 FastAPI Router
- 定义所有端点和处理逻辑
- Server 只需挂载 Router"""

import json
import time
from typing import Any, AsyncIterator, Dict, Iterator, TYPE_CHECKING, Union

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .model import (
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
from .protocol import ProtocolHandler

if TYPE_CHECKING:
    from .invoker import AgentInvoker


class OpenAIProtocolHandler(ProtocolHandler):
    """OpenAI Completions API 协议处理器

    实现 OpenAI Chat Completions API 兼容接口
    参考: https://platform.openai.com/docs/api-reference/chat/create
    """

    def get_prefix(self) -> str:
        """OpenAI 协议建议使用 /v1 前缀"""
        return "/openai/v1"

    def as_fastapi_router(self, agent_invoker: "AgentInvoker") -> APIRouter:
        """创建 OpenAI 协议的 FastAPI Router"""
        router = APIRouter()

        @router.post("/chat/completions")
        async def chat_completions(request: Request):
            """OpenAI Chat Completions 端点"""
            try:
                # 1. 解析请求
                request_data = await request.json()
                agent_request = await self.parse_request(request_data)

                # 2. 调用 Agent
                agent_result = await agent_invoker.invoke(agent_request)

                # 3. 格式化响应
                formatted_result = await self.format_response(
                    agent_result, agent_request
                )

                # 4. 返回响应
                # 自动检测是否为流式响应
                if hasattr(formatted_result, "__aiter__"):
                    return StreamingResponse(
                        formatted_result, media_type="text/event-stream"
                    )
                else:
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

        # 可以添加更多端点
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

    async def parse_request(self, request_data: Dict[str, Any]) -> AgentRequest:
        """解析 OpenAI 格式的请求

        Args:
            request_data: HTTP 请求体 JSON 数据

        Returns:
            AgentRequest: 标准化的请求对象

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

            # 转换消息
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

        # 提取标准参数
        agent_request = AgentRequest(
            messages=messages,
            model=request_data.get("model"),
            stream=request_data.get("stream", False),
            temperature=request_data.get("temperature"),
            top_p=request_data.get("top_p"),
            max_tokens=request_data.get("max_tokens"),
            tools=request_data.get("tools"),
            tool_choice=request_data.get("tool_choice"),
            user=request_data.get("user"),
        )

        # 保存其他额外参数
        standard_fields = {
            "messages",
            "model",
            "stream",
            "temperature",
            "top_p",
            "max_tokens",
            "tools",
            "tool_choice",
            "user",
        }
        agent_request.extra = {
            k: v for k, v in request_data.items() if k not in standard_fields
        }

        return agent_request

    async def format_response(
        self, result: AgentResult, request: AgentRequest
    ) -> Any:
        """格式化响应为 OpenAI 格式

        Args:
            result: Agent 执行结果,支持:
                - AgentRunResult: 核心数据结构 (推荐)
                - AgentResponse: 完整响应对象
                - ModelResponse: litellm 的 ModelResponse
                - CustomStreamWrapper: litellm 的流式响应
            request: 原始请求

        Returns:
            格式化后的响应(dict 或 AsyncIterator)
        """
        # 1. 检测 ModelResponse (来自 Model Service)
        if self._is_model_response(result):
            return self._format_model_response(result, request)

        # 2. 处理 AgentRunResult
        if isinstance(result, AgentRunResult):
            return await self._format_agent_run_result(result, request)

        # 3. 自动检测流式响应:
        #    - 请求明确指定 stream=true
        #    - 或返回值是迭代器/生成器
        is_stream = request.stream or self._is_iterator(result)

        if is_stream:
            return self._format_stream_response(result, request)

        # 4. 非流式响应
        # 如果是字符串,包装成 AgentResponse
        if isinstance(result, str):
            result = self._wrap_string_response(result, request)

        # 如果是 AgentResponse,补充 OpenAI 必需字段并序列化
        if isinstance(result, AgentResponse):
            return self._ensure_openai_format(result, request)

        raise TypeError(
            "Expected AgentRunResult, AgentResponse, or ModelResponse, "
            f"got {type(result)}"
        )

    async def _format_agent_run_result(
        self, result: AgentRunResult, request: AgentRequest
    ) -> Union[Dict[str, Any], AsyncIterator[str]]:
        """格式化 AgentRunResult 为 OpenAI 格式

        AgentRunResult 的 content 可以是:
        - string: 非流式响应
        - Iterator[str] 或 AsyncIterator[str]: 流式响应

        Args:
            result: AgentRunResult 对象
            request: 原始请求

        Returns:
            非流式: OpenAI 格式的字典
            流式: SSE 格式的异步迭代器
        """
        content = result.content

        # 检查 content 是否是迭代器
        if self._is_iterator(content):
            # 流式响应
            return self._format_stream_content(content, request)

        # 非流式响应
        if isinstance(content, str):
            return {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model or "agentrun-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }],
            }

        raise TypeError(
            "AgentRunResult.content must be str or Iterator[str], got"
            f" {type(content)}"
        )

    def _is_model_response(self, obj: Any) -> bool:
        """检查对象是否是 Model Service 的 ModelResponse

        ModelResponse 特征:
        - 有 choices 属性
        - 有 usage 属性 (或 created, id 等)
        - 不是 AgentResponse (AgentResponse 也有这些字段)

        Args:
            obj: 要检查的对象

        Returns:
            bool: 是否是 ModelResponse
        """
        # 排除已知类型
        if isinstance(obj, (str, AgentResponse, AgentRunResult, dict)):
            return False

        # 检查 ModelResponse 的特征属性
        # litellm 的 ModelResponse 有 choices 和 model 属性
        return (
            hasattr(obj, "choices")
            and hasattr(obj, "model")
            and (hasattr(obj, "usage") or hasattr(obj, "created"))
        )

    def _format_model_response(
        self, response: Any, request: AgentRequest
    ) -> Dict[str, Any]:
        """格式化 ModelResponse 为 OpenAI 格式

        ModelResponse 本身已经是 OpenAI 格式,直接转换为字典即可。

        Args:
            response: litellm 的 ModelResponse 对象
            request: 原始请求

        Returns:
            Dict: OpenAI 格式的响应字典
        """
        # 方式 1: 如果有 model_dump 方法 (Pydantic)
        if hasattr(response, "model_dump"):
            return response.model_dump(exclude_none=True)

        # 方式 2: 如果有 dict 方法
        if hasattr(response, "dict"):
            return response.dict(exclude_none=True)

        # 方式 3: 手动转换 (litellm ModelResponse)
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

        # 转换 choices
        if hasattr(response, "choices"):
            for choice in response.choices:
                choice_dict = {
                    "index": getattr(choice, "index", 0),
                    "finish_reason": getattr(choice, "finish_reason", None),
                }

                # 转换 message
                if hasattr(choice, "message"):
                    msg = choice.message
                    choice_dict["message"] = {
                        "role": getattr(msg, "role", "assistant"),
                        "content": getattr(msg, "content", None),
                    }
                    # 可选字段
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        choice_dict["message"]["tool_calls"] = msg.tool_calls

                result["choices"].append(choice_dict)

        # 转换 usage
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            result["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

        return result

    def _is_iterator(self, obj: Any) -> bool:
        """检查对象是否是迭代器

        Args:
            obj: 要检查的对象

        Returns:
            bool: 是否是迭代器
        """
        # 检查是否是迭代器或生成器
        return (
            hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict))
        ) or hasattr(obj, "__aiter__")

    async def _format_stream_content(
        self,
        content: Union[Iterator[str], AsyncIterator[str]],
        request: AgentRequest,
    ) -> AsyncIterator[str]:
        """格式化流式 content 为 OpenAI SSE 格式

        将字符串迭代器转换为 OpenAI 流式响应格式。

        Args:
            content: 字符串迭代器 (同步或异步)
            request: 原始请求

        Yields:
            SSE 格式的数据行
        """
        response_id = f"chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())
        model = request.model or "agentrun-model"

        # 发送第一个 chunk (包含 role)
        first_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

        # 检查是否是异步迭代器
        if hasattr(content, "__aiter__"):
            async for chunk in content:  # type: ignore
                if chunk:  # 跳过空字符串
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        else:
            # 同步迭代器
            for chunk in content:  # type: ignore
                if chunk:
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        # 发送结束 chunk
        final_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

        # 发送结束标记
        yield "data: [DONE]\n\n"

    def _wrap_string_response(
        self, content: str, request: AgentRequest
    ) -> AgentResponse:
        """将字符串包装成 AgentResponse

        Args:
            content: 响应内容字符串
            request: 原始请求

        Returns:
            AgentResponse: 包装后的响应对象
        """
        return AgentResponse(content=content)

    def _ensure_openai_format(
        self, response: AgentResponse, request: AgentRequest
    ) -> Dict[str, Any]:
        """确保 AgentResponse 符合 OpenAI 格式

        如果用户只填充了 content,自动补充 OpenAI 必需字段。
        如果用户已填充完整字段,直接使用。

        Args:
            response: Agent 返回的响应对象
            request: 原始请求

        Returns:
            Dict: OpenAI 格式的响应字典
        """
        # 如果用户只提供了 content,构造完整的 OpenAI 格式
        if response.content and not response.choices:
            return {
                "id": response.id or f"chatcmpl-{int(time.time() * 1000)}",
                "object": response.object or "chat.completion",
                "created": response.created or int(time.time()),
                "model": response.model or request.model or "agentrun-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.content,
                    },
                    "finish_reason": "stop",
                }],
                "usage": (
                    json.loads(response.usage.model_dump_json())
                    if response.usage
                    else None
                ),
            }

        # 用户提供了完整字段,使用 JSON 序列化避免对象嵌套问题
        json_str = response.model_dump_json(exclude_none=True)
        result = json.loads(json_str)

        # 确保必需字段存在
        if "id" not in result:
            result["id"] = f"chatcmpl-{int(time.time() * 1000)}"
        if "object" not in result:
            result["object"] = "chat.completion"
        if "created" not in result:
            result["created"] = int(time.time())
        if "model" not in result:
            result["model"] = request.model or "agentrun-model"

        # 移除 content 和 extra (OpenAI 格式中不需要)
        result.pop("content", None)
        result.pop("extra", None)

        return result

    def _is_custom_stream_wrapper(self, obj: Any) -> bool:
        """检查是否是 Model Service 的 CustomStreamWrapper"""
        # CustomStreamWrapper 的特征
        return (
            hasattr(obj, "__aiter__")
            and type(obj).__name__ == "CustomStreamWrapper"
        )

    async def _format_model_stream(
        self, stream_wrapper: Any, request: AgentRequest
    ) -> AsyncIterator[str]:
        """格式化 Model Service 的流式响应

        CustomStreamWrapper 返回的 chunk 已经是完整的 OpenAI 格式对象。
        """
        async for chunk in stream_wrapper:
            # chunk 是 litellm 的 ModelResponse 或字典
            if isinstance(chunk, dict):
                # 已经是字典,直接格式化为 SSE
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            elif hasattr(chunk, "model_dump"):
                # Pydantic 对象
                chunk_dict = chunk.model_dump(exclude_none=True)
                yield f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
            elif hasattr(chunk, "dict"):
                # 旧版 Pydantic
                chunk_dict = chunk.dict(exclude_none=True)
                yield f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
            else:
                # 手动转换对象为字典
                chunk_dict = {
                    "id": getattr(
                        chunk, "id", f"chatcmpl-{int(time.time() * 1000)}"
                    ),
                    "object": getattr(chunk, "object", "chat.completion.chunk"),
                    "created": getattr(chunk, "created", int(time.time())),
                    "model": getattr(
                        chunk, "model", request.model or "agentrun-model"
                    ),
                    "choices": [],
                }

                if hasattr(chunk, "choices"):
                    for choice in chunk.choices:
                        choice_dict = {
                            "index": getattr(choice, "index", 0),
                            "finish_reason": getattr(
                                choice, "finish_reason", None
                            ),
                        }

                        if hasattr(choice, "delta"):
                            delta = choice.delta
                            delta_dict = {}
                            if hasattr(delta, "role") and delta.role:
                                delta_dict["role"] = delta.role
                            if hasattr(delta, "content") and delta.content:
                                delta_dict["content"] = delta.content
                            if (
                                hasattr(delta, "tool_calls")
                                and delta.tool_calls
                            ):
                                delta_dict["tool_calls"] = delta.tool_calls
                            choice_dict["delta"] = delta_dict

                        chunk_dict["choices"].append(choice_dict)

                yield f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"

        # 发送结束标记
        yield "data: [DONE]\n\n"

    async def _format_stream_response(
        self, result: AgentResult, request: AgentRequest
    ) -> AsyncIterator[str]:
        """格式化流式响应

        Args:
            result: 流式迭代器,支持:
                - Iterator[str]/AsyncIterator[str]: 流式字符串
                - Iterator[AgentStreamResponse]: 流式响应对象
                - CustomStreamWrapper: Model Service 流式响应
            request: 原始请求

        Yields:
            SSE 格式的数据行
        """
        # 检查是否是 CustomStreamWrapper (Model Service 流式响应)
        if self._is_custom_stream_wrapper(result):
            async for chunk in self._format_model_stream(result, request):
                yield chunk
            return

        response_id = f"chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())
        model = request.model or "agentrun-model"

        # 检查是否是异步迭代器
        if hasattr(result, "__aiter__"):
            first_chunk = True
            async for chunk in result:  # type: ignore
                # 如果是字符串,包装成 AgentStreamResponse
                if isinstance(chunk, str):
                    if first_chunk:
                        # 第一个 chunk: 发送 role
                        yield self._format_sse_chunk(
                            AgentStreamResponse(
                                id=response_id,
                                created=created,
                                model=model,
                                choices=[
                                    AgentStreamResponseChoice(
                                        index=0,
                                        delta=AgentStreamResponseDelta(
                                            role=MessageRole.ASSISTANT,
                                        ),
                                        finish_reason=None,
                                    )
                                ],
                            )
                        )
                        first_chunk = False

                    # 发送内容 chunk
                    if chunk:  # 跳过空字符串
                        yield self._format_sse_chunk(
                            AgentStreamResponse(
                                id=response_id,
                                created=created,
                                model=model,
                                choices=[
                                    AgentStreamResponseChoice(
                                        index=0,
                                        delta=AgentStreamResponseDelta(
                                            content=chunk
                                        ),
                                        finish_reason=None,
                                    )
                                ],
                            )
                        )

                # 如果是 AgentStreamResponse,直接序列化
                elif isinstance(chunk, AgentStreamResponse):
                    yield self._format_sse_chunk(chunk)

            # 发送结束 chunk
            yield self._format_sse_chunk(
                AgentStreamResponse(
                    id=response_id,
                    created=created,
                    model=model,
                    choices=[
                        AgentStreamResponseChoice(
                            index=0,
                            delta=AgentStreamResponseDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
            )
            # 发送结束标记
            yield "data: [DONE]\n\n"

        # 同步迭代器
        elif hasattr(result, "__iter__"):
            first_chunk = True
            for chunk in result:  # type: ignore
                # 如果是字符串,包装成 AgentStreamResponse
                if isinstance(chunk, str):
                    if first_chunk:
                        yield self._format_sse_chunk(
                            AgentStreamResponse(
                                id=response_id,
                                created=created,
                                model=model,
                                choices=[
                                    AgentStreamResponseChoice(
                                        index=0,
                                        delta=AgentStreamResponseDelta(
                                            role=MessageRole.ASSISTANT,
                                        ),
                                        finish_reason=None,
                                    )
                                ],
                            )
                        )
                        first_chunk = False

                    if chunk:
                        yield self._format_sse_chunk(
                            AgentStreamResponse(
                                id=response_id,
                                created=created,
                                model=model,
                                choices=[
                                    AgentStreamResponseChoice(
                                        index=0,
                                        delta=AgentStreamResponseDelta(
                                            content=chunk
                                        ),
                                        finish_reason=None,
                                    )
                                ],
                            )
                        )

                elif isinstance(chunk, AgentStreamResponse):
                    yield self._format_sse_chunk(chunk)

            # 发送结束 chunk
            yield self._format_sse_chunk(
                AgentStreamResponse(
                    id=response_id,
                    created=created,
                    model=model,
                    choices=[
                        AgentStreamResponseChoice(
                            index=0,
                            delta=AgentStreamResponseDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
            )
            yield "data: [DONE]\n\n"

        else:
            raise TypeError(
                "Expected Iterator or AsyncIterator for stream response, "
                f"got {type(result)}"
            )

    def _format_sse_chunk(self, chunk: AgentStreamResponse) -> str:
        """格式化单个 SSE chunk

        Args:
            chunk: AgentStreamResponse 对象

        Returns:
            SSE 格式的字符串
        """
        # 使用 Pydantic 的 JSON 序列化,自动处理所有嵌套对象
        json_str = chunk.model_dump_json(exclude_none=True)
        json_data = json.loads(json_str)

        # 如果用户只提供了 content,转换为 OpenAI 格式
        if "content" in json_data and "choices" not in json_data:
            json_data = {
                "id": json_data.get(
                    "id", f"chatcmpl-{int(time.time() * 1000)}"
                ),
                "object": json_data.get("object", "chat.completion.chunk"),
                "created": json_data.get("created", int(time.time())),
                "model": json_data.get("model", "agentrun-model"),
                "choices": [{
                    "index": 0,
                    "delta": {"content": json_data["content"]},
                    "finish_reason": None,
                }],
            }
        else:
            # 移除不属于 OpenAI 格式的字段
            json_data.pop("content", None)
            json_data.pop("extra", None)

        return f"data: {json.dumps(json_data, ensure_ascii=False)}\n\n"
