"""PydanticAI 模型适配器 / PydanticAI Model Adapter"""

from contextlib import asynccontextmanager
import json
from typing import Any, AsyncIterator

from agentrun.integration.utils.adapter import ModelAdapter
from agentrun.integration.utils.model import CommonModel


class PydanticAIModelAdapter(ModelAdapter):
    """PydanticAI 模型适配器 / PydanticAI Model Adapter

    PydanticAI 支持 OpenAI 兼容的接口，我们提供一个轻量级包装。"""

    def wrap_model(self, common_model: CommonModel) -> Any:
        """将 CommonModel 包装为 PydanticAI 兼容的模型 / PydanticAI Model Adapter"""

        try:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
        except Exception as e:
            raise ImportError(
                "PydanticAI is not installed. "
                "Install it with: pip install pydantic-ai"
            ) from e

        from httpx import AsyncClient

        info = common_model.get_model_info()

        return OpenAIChatModel(
            info.model or "",
            provider=OpenAIProvider(
                base_url=info.base_url,
                api_key=info.api_key,
                http_client=AsyncClient(headers=info.headers),
            ),
        )

        try:
            from pydantic_ai.messages import (
                ModelMessage,
                ModelResponse,
                SystemPromptPart,
                TextPart,
                ToolCallPart,
                ToolReturnPart,
                UserPromptPart,
            )
            from pydantic_ai.models import Model, StreamedResponse
            from pydantic_ai.settings import ModelSettings
        except ImportError as e:
            raise ImportError(
                "PydanticAI is not installed. "
                "Install it with: pip install pydantic-ai"
            ) from e

        class AgentRunPydanticAIModel(Model):
            """PydanticAI Model 包装 AgentRun CommonModel / PydanticAI Model Adapter"""

            def __init__(self, model: CommonModel):
                self._model = model
                self._model_name = model.name
                self._function_tools: list[Any] = []

            @property
            def model_name(self) -> str:
                """返回模型名称（PydanticAI Model 抽象属性） / PydanticAI Model Adapter"""
                return self._model_name

            @property
            def system(self) -> str:
                """系统提示（PydanticAI Model 抽象属性） / PydanticAI Model Adapter"""
                return ""

            def name(self) -> str:
                """返回模型名称 / PydanticAI Model Adapter"""
                return self._model_name

            async def agent_model(
                self,
                *,
                function_tools: list[Any],
                allow_text_result: bool,
                result_tools: list[Any],
            ) -> Any:
                """返回用于 agent 的模型实例（可选实现） / PydanticAI Model Adapter"""
                # 存储工具信息以便在请求时使用
                self._function_tools = function_tools
                return self

            async def request(
                self,
                messages: list[ModelMessage],
                model_settings: ModelSettings | None = None,
                model_request_parameters: Any | None = None,
            ) -> ModelResponse:
                """处理非流式请求 / PydanticAI Model Adapter"""
                # 转换 PydanticAI 消息为 OpenAI 格式
                openai_messages = self._convert_messages(messages)

                # 准备请求参数
                kwargs = {}
                if model_settings:
                    # ModelSettings 是 TypedDict，使用字典访问
                    if "temperature" in model_settings:
                        kwargs["temperature"] = model_settings["temperature"]
                    if "max_tokens" in model_settings:
                        kwargs["max_tokens"] = model_settings["max_tokens"]
                    if "top_p" in model_settings:
                        kwargs["top_p"] = model_settings["top_p"]

                # 从 model_request_parameters 中提取工具
                tools_to_use = self._function_tools  # 默认使用存储的工具
                if model_request_parameters and hasattr(
                    model_request_parameters, "function_tools"
                ):
                    tools_to_use = model_request_parameters.function_tools

                # 转换工具为 OpenAI 格式
                if tools_to_use:
                    tools = []
                    for tool in tools_to_use:
                        # PydanticAI function tool 格式
                        if hasattr(tool, "name") and hasattr(
                            tool, "description"
                        ):
                            tool_def = {
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description or "",
                                },
                            }
                            # 添加参数 schema
                            if hasattr(tool, "parameters_json_schema"):
                                tool_def["function"][
                                    "parameters"
                                ] = tool.parameters_json_schema
                            tools.append(tool_def)

                    if tools:
                        kwargs["tools"] = tools

                # 调用底层模型（CommonModel.completions 返回字典）
                response = self._model.completions(
                    messages=openai_messages, **kwargs
                )

                # 如果是字典，直接转换
                if isinstance(response, dict):
                    return self._convert_response(response)

                # 否则可能是其他格式，尝试提取
                raise TypeError(f"Unexpected response type: {type(response)}")

            @asynccontextmanager
            async def request_stream(
                self,
                messages: list[ModelMessage],
                model_settings: ModelSettings | None = None,
                model_request_parameters: Any | None = None,
                run_context: Any | None = None,
            ) -> AsyncIterator[StreamedResponse]:
                """处理流式请求（暂不支持） / PydanticAI Model Adapter"""
                raise NotImplementedError(
                    "Streamed requests not supported by"
                    f" {self.__class__.__name__}"
                )
                yield  # pragma: no cover

            def _convert_messages(
                self, messages: list[ModelMessage]
            ) -> list[dict[str, Any]]:
                """将 PydanticAI 消息转换为 OpenAI 格式 / PydanticAI Model Adapter"""
                openai_messages = []

                for msg in messages:
                    # ModelMessage 可能是 ModelRequest 或 ModelResponse
                    # 检查消息的部分（parts）
                    if hasattr(msg, "parts") and msg.parts:
                        parts = msg.parts
                        message_dict: dict[str, Any] = {}

                        # 确定角色
                        if any(isinstance(p, SystemPromptPart) for p in parts):
                            message_dict["role"] = "system"
                        elif any(isinstance(p, UserPromptPart) for p in parts):
                            message_dict["role"] = "user"
                        elif any(isinstance(p, ToolReturnPart) for p in parts):
                            # 工具返回需要特殊处理
                            for part in parts:
                                if isinstance(part, ToolReturnPart):
                                    openai_messages.append({
                                        "role": "tool",
                                        "tool_call_id": part.tool_call_id,
                                        "content": (
                                            part.content
                                            if isinstance(part.content, str)
                                            else str(part.content)
                                        ),
                                    })
                            continue
                        elif any(isinstance(p, ToolCallPart) for p in parts):
                            message_dict["role"] = "assistant"
                        else:
                            message_dict["role"] = "assistant"

                        # 处理各种部分
                        content_parts = []
                        tool_calls = []

                        for part in parts:
                            if isinstance(
                                part,
                                (TextPart, UserPromptPart, SystemPromptPart),
                            ):
                                content = (
                                    part.content
                                    if hasattr(part, "content")
                                    else str(part)
                                )
                                content_parts.append(content)
                            elif isinstance(part, ToolCallPart):
                                # part.args 可能是 str、dict 或 Pydantic 模型
                                args = part.args
                                if isinstance(args, str):
                                    args_str = args
                                elif isinstance(args, dict):
                                    args_str = json.dumps(args)
                                elif hasattr(args, "model_dump"):
                                    # Pydantic 模型
                                    args_str = json.dumps(args.model_dump())  # type: ignore
                                else:
                                    args_str = json.dumps({})

                                tool_calls.append({
                                    "id": part.tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": part.tool_name,
                                        "arguments": args_str,
                                    },
                                })

                        if content_parts:
                            message_dict["content"] = " ".join(content_parts)
                        if tool_calls:
                            message_dict["tool_calls"] = tool_calls
                            if not content_parts:
                                message_dict["content"] = None

                        if message_dict:
                            openai_messages.append(message_dict)

                return openai_messages

            def _convert_response(
                self, response: dict[str, Any]
            ) -> ModelResponse:
                """将 OpenAI 格式响应转换为 PydanticAI ModelResponse / PydanticAI Model Adapter"""
                # 提取第一个选择
                choices = response.get("choices", [])
                if not choices:
                    return ModelResponse(parts=[TextPart(content="")])

                message = choices[0].get("message", {})
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])

                # 构建响应部分
                parts: list[Any] = []

                # 添加文本内容
                if content:
                    parts.append(TextPart(content=content))

                # 添加工具调用
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    args_str = function.get("arguments", "{}")

                    # 解析参数为字典
                    try:
                        args_dict = (
                            json.loads(args_str)
                            if isinstance(args_str, str)
                            else args_str
                        )
                    except Exception:
                        args_dict = {}

                    parts.append(
                        ToolCallPart(
                            tool_name=function.get("name", ""),
                            args=args_dict,
                            tool_call_id=tool_call.get("id", ""),
                        )
                    )

                # 如果没有任何内容，添加空文本
                if not parts:
                    parts.append(TextPart(content=""))

                return ModelResponse(parts=parts)

        return AgentRunPydanticAIModel(common_model)


__all__ = ["PydanticAIModelAdapter"]
