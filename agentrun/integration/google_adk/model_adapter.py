"""Google ADK 模型适配器 / Google ADK Model Adapter

将 CommonModel 包装为 Google ADK BaseLlm。"""

import json
from typing import Any

from litellm import api_base
from openai import api_key

from agentrun.integration.google_adk.message_adapter import (
    GoogleADKMessageAdapter,
)
from agentrun.integration.utils.adapter import ModelAdapter
from agentrun.integration.utils.model import CommonModel


class GoogleADKModelAdapter(ModelAdapter):
    """Google ADK 模型适配器 / Google ADK Model Adapter

    将 CommonModel 包装为 Google ADK BaseLlm。"""

    def __init__(self):
        """初始化适配器，创建内部的消息适配器 / Google ADK Message Adapter"""
        self._message_adapter = GoogleADKMessageAdapter()

    def wrap_model(self, common_model: CommonModel) -> Any:
        """包装 CommonModel 为 Google ADK BaseLlm / Google ADK Model Adapter"""

        try:
            from google.adk.models.lite_llm import LiteLlm  # type: ignore
        except ImportError as e:
            raise ImportError(
                "import google.adk.models.lite_llm failed."
                "Google ADK may not installed, "
                "Install it with: pip install google-adk"
            ) from e

        try:
            from agentscope.model import OpenAIChatModel
        except Exception as e:
            raise ImportError(
                "AgentScope is not installed. Install it with: pip install"
                " agentscope"
            ) from e

        from httpx import AsyncClient

        info = common_model.get_model_info()

        return LiteLlm(
            model=info.model or "",
            api_base=info.base_url,
            api_key=info.api_key,
            extra_headers=info.headers,
        )

        from google.adk.models.model_registry import ModelRegistry

        try:
            from google.adk.models.base_llm import BaseLlm
            from google.adk.models.llm_request import LlmRequest
            from google.adk.models.llm_response import LlmResponse
            from google.genai import types as genai_types
        except ImportError as e:
            raise ImportError(
                "Google ADK not installed. "
                "Install it with: pip install google-generativeai"
            ) from e

        # 获取全局 tool_adapter 实例（用于查找已注册的工具定义）
        from agentrun.integration.utils.converter import get_converter

        converter = get_converter()
        tool_adapter = converter._tool_adapters.get("google_adk")

        message_adapter = self._message_adapter
        model_instance = common_model

        class AgentRunLlm(BaseLlm):
            """AgentRun 模型适配为 Google ADK BaseLlm / Google ADK Model Adapter"""

            def __init__(self, **kwargs):
                super().__init__(model=model_instance.name, **kwargs)
                self._common_model = model_instance
                self._message_adapter = message_adapter
                self._tool_adapter = tool_adapter

            async def generate_content_async(
                self, llm_request: LlmRequest, stream: bool = False
            ):
                """实现 BaseLlm 的抽象方法 / Google ADK Model Adapter"""
                # 使用适配器转换消息
                canonical_messages = self._message_adapter.to_canonical(
                    llm_request
                )

                # 转换为 OpenAI 格式
                openai_messages = [msg.to_dict() for msg in canonical_messages]

                # 处理工具
                tools = None
                if (
                    hasattr(llm_request, "tools_dict")
                    and llm_request.tools_dict
                ):
                    tools = []
                    for tool_name, tool_obj in llm_request.tools_dict.items():
                        tool_def = {
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "description": getattr(
                                    tool_obj, "description", ""
                                ),
                            },
                        }
                        # 尝试获取参数 schema
                        input_schema = getattr(tool_obj, "input_schema", None)
                        if not input_schema and self._tool_adapter:
                            canonical_tool = (
                                self._tool_adapter.get_registered_tool(
                                    tool_name
                                )
                            )
                            if canonical_tool and canonical_tool.parameters:
                                input_schema = canonical_tool.parameters
                        if input_schema:
                            tool_def["function"]["parameters"] = input_schema
                        tools.append(tool_def)

                # 调用底层模型
                kwargs = {"messages": openai_messages, "stream": stream}
                if tools:
                    kwargs["tools"] = tools

                response = model_instance.completions(**kwargs)

                # 转换响应为 LlmResponse
                content = self._convert_response(response, genai_types)
                llm_response = LlmResponse(content=content)
                yield llm_response

            @staticmethod
            def _convert_response(response: Any, genai_types: Any):
                """转换模型响应为 Google ADK Content / Google ADK Model Adapter"""
                # 处理字典格式的响应
                if isinstance(response, dict):
                    choices = response.get("choices", [])
                    if choices:
                        message = choices[0].get("message", {})
                        return AgentRunLlm._build_content_from_message(
                            message, genai_types
                        )
                    return genai_types.Content(
                        parts=[genai_types.Part(text=str(response))],
                        role="model",
                    )

                # 处理对象格式的响应
                if hasattr(response, "choices") and response.choices:
                    message = response.choices[0].message
                    message_dict = {
                        "content": getattr(message, "content", None),
                        "tool_calls": getattr(message, "tool_calls", None),
                    }
                    return AgentRunLlm._build_content_from_message(
                        message_dict, genai_types
                    )

                # 非标准响应
                return genai_types.Content(
                    parts=[genai_types.Part(text=str(response))],
                    role="model",
                )

            @staticmethod
            def _build_content_from_message(message: dict, genai_types: Any):
                """从消息字典构建 Content / Google ADK Model Adapter"""
                parts = []

                # 处理文本内容
                content_text = message.get("content")
                if content_text:
                    parts.append(genai_types.Part(text=content_text))

                # 处理工具调用
                tool_calls = message.get("tool_calls", []) or []
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        function_info = tool_call.get("function", {})
                        tool_id = tool_call.get("id", "")
                    else:
                        function_info = {
                            "name": getattr(tool_call.function, "name", ""),
                            "arguments": getattr(
                                tool_call.function, "arguments", ""
                            ),
                        }
                        tool_id = getattr(tool_call, "id", "")

                    arguments = function_info.get("arguments", "")

                    # 解析参数
                    try:
                        if isinstance(arguments, str):
                            args_dict = (
                                json.loads(arguments)
                                if arguments.strip()
                                else {}
                            )
                        else:
                            args_dict = arguments or {}
                    except (json.JSONDecodeError, TypeError):
                        args_dict = {}

                    func_call = genai_types.FunctionCall(
                        name=function_info.get("name", ""),
                        args=args_dict,
                        id=tool_id,
                    )
                    parts.append(genai_types.Part(function_call=func_call))

                # 如果没有任何内容，添加空文本
                if not parts:
                    parts.append(genai_types.Part(text=""))

                return genai_types.Content(parts=parts, role="model")

        return AgentRunLlm()
