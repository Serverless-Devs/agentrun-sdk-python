"""LangChain 模型适配器 / LangChain Model Adapter

将 CommonModel 包装为 LangChain BaseChatModel。"""

import inspect
import json
from typing import Any, List, Optional

from agentrun.integration.langchain.message_adapter import (
    LangChainMessageAdapter,
)
from agentrun.integration.utils.adapter import ModelAdapter


class LangChainModelAdapter(ModelAdapter):
    """LangChain 模型适配器 / LangChain Model Adapter

    将 CommonModel 包装为 LangChain BaseChatModel。"""

    def __init__(self):
        """初始化适配器，创建内部的消息适配器 / LangChain Message Adapter"""
        self._message_adapter = LangChainMessageAdapter()

    def wrap_model(self, common_model: Any) -> Any:
        """包装 CommonModel 为 LangChain BaseChatModel / LangChain Model Adapter"""
        from httpx import AsyncClient
        from langchain_openai import ChatOpenAI

        info = common_model.get_model_info()  # 确保模型可用
        return ChatOpenAI(
            name=info.model,
            api_key=info.api_key,
            model=info.model,
            base_url=info.base_url,
            async_client=AsyncClient(headers=info.headers),
        )

        try:
            from langchain_core.callbacks.manager import (
                CallbackManagerForLLMRun,
            )
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_core.messages import AIMessage, BaseMessage
            from langchain_core.outputs import ChatGeneration, ChatResult
        except ImportError as e:
            raise ImportError(
                "LangChain is not installed. "
                "Install it with: pip install langchain-core"
            ) from e

        message_adapter = self._message_adapter

        class AgentRunLangChainChatModel(BaseChatModel):
            """LangChain ChatModel 封装 AgentRun CommonModel / LangChain Model Adapter"""

            model_name: str = common_model.name

            def __init__(
                self,
                model_name: str = None,
                _common_model: Any = None,
                _message_adapter: Any = None,
                **kwargs,
            ):
                super().__init__(**kwargs)
                if _common_model is not None:
                    # 从现有实例复制
                    self._common_model = _common_model
                    self._message_adapter = _message_adapter
                    if model_name:
                        self.model_name = model_name
                else:
                    # 新实例
                    self._common_model = common_model
                    self._message_adapter = message_adapter
                self._bound_tools = getattr(self, "_bound_tools", None)
                self._tool_choice = getattr(self, "_tool_choice", None)

            @property
            def _llm_type(self) -> str:
                return "agentrun-common-model"

            def bind_tools(
                self,
                tools: Any,
                *,
                tool_choice: Any = None,
                **kwargs: Any,
            ) -> Any:
                """绑定工具到模型 / LangChain Model Adapter

                Args:
                                    tools: 工具列表，可以是 StructuredTool、函数、字典等
                                    tool_choice: 工具选择策略
                                    **kwargs: 其他参数

                                Returns:
                                    绑定了工具的新模型实例"""
                from langchain_core.tools import BaseTool, StructuredTool

                # 转换工具为 OpenAI 格式
                openai_tools = []
                for tool in tools:
                    if isinstance(tool, dict):
                        # 已经是字典格式
                        openai_tools.append(tool)
                    elif isinstance(tool, (BaseTool, StructuredTool)):
                        # LangChain 工具，转换为 OpenAI 格式
                        tool_schema = {}
                        if hasattr(tool, "args_schema") and tool.args_schema:
                            try:
                                tool_schema = (
                                    tool.args_schema.model_json_schema()
                                )
                            except Exception:
                                tool_schema = {
                                    "type": "object",
                                    "properties": {},
                                }

                        openai_tools.append({
                            "type": "function",
                            "function": {
                                "name": getattr(tool, "name", ""),
                                "description": getattr(tool, "description", ""),
                                "parameters": tool_schema,
                            },
                        })
                    elif callable(tool):
                        # 函数，尝试提取信息
                        sig = inspect.signature(tool)
                        params = {}
                        for param_name, param in sig.parameters.items():
                            if param_name == "self":
                                continue
                            param_type = "string"
                            if param.annotation != inspect.Parameter.empty:
                                ann_str = str(param.annotation)
                                if "int" in ann_str:
                                    param_type = "integer"
                                elif "float" in ann_str or "number" in ann_str:
                                    param_type = "number"
                                elif "bool" in ann_str:
                                    param_type = "boolean"
                                elif "list" in ann_str or "List" in ann_str:
                                    param_type = "array"
                                elif "dict" in ann_str or "Dict" in ann_str:
                                    param_type = "object"

                            param_info = {"type": param_type}
                            if param.default != inspect.Parameter.empty:
                                param_info["default"] = param.default
                            params[param_name] = param_info

                        openai_tools.append({
                            "type": "function",
                            "function": {
                                "name": getattr(tool, "__name__", "unknown"),
                                "description": getattr(tool, "__doc__", ""),
                                "parameters": {
                                    "type": "object",
                                    "properties": params,
                                    "required": [
                                        name
                                        for name, param in sig.parameters.items()
                                        if param.default
                                        == inspect.Parameter.empty
                                        and name != "self"
                                    ],
                                },
                            },
                        })
                    else:
                        # 其他类型，尝试转换为字典
                        try:
                            if hasattr(tool, "to_openai_function"):
                                openai_tools.append(tool.to_openai_function())
                            elif hasattr(tool, "dict"):
                                openai_tools.append(tool.dict())
                            elif hasattr(tool, "model_dump"):
                                openai_tools.append(tool.model_dump())
                        except Exception:
                            # 如果无法转换，跳过
                            continue

                # 创建新的模型实例，保存工具信息
                bound_model = AgentRunLangChainChatModel(
                    model_name=self.model_name,
                    _common_model=self._common_model,
                    _message_adapter=self._message_adapter,
                )
                bound_model._bound_tools = openai_tools
                bound_model._tool_choice = tool_choice

                return bound_model

            def _generate(  # type: ignore[override]
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> ChatResult:
                # 使用适配器转换消息
                canonical_messages = self._message_adapter.to_canonical(
                    messages
                )

                # 转换为 OpenAI 格式（CommonModel 使用的格式）
                openai_messages = []
                for msg in canonical_messages:
                    msg_dict = msg.to_dict()
                    openai_messages.append(msg_dict)

                # 调用底层模型
                payload: dict = {"messages": openai_messages}
                if stop:
                    kwargs = {**kwargs, "stop": stop}

                # 如果有绑定的工具，添加到请求中
                if hasattr(self, "_bound_tools") and self._bound_tools:
                    kwargs["tools"] = self._bound_tools

                response = self._common_model.completions(**payload, **kwargs)

                # 转换响应
                ai_message = self._convert_response(response)
                generation = ChatGeneration(message=ai_message)
                llm_output = {"usage": getattr(response, "usage", None)}
                return ChatResult(
                    generations=[generation], llm_output=llm_output
                )

            @staticmethod
            def _convert_response(response: Any) -> AIMessage:
                """转换模型响应为 AIMessage / LangChain Model Adapter"""
                # 尝试提取响应信息
                response_dict = {}
                if hasattr(response, "model_dump"):
                    response_dict = response.model_dump()
                elif hasattr(response, "dict"):
                    response_dict = response.dict()
                elif isinstance(response, dict):
                    response_dict = response
                elif hasattr(response, "__dict__"):
                    response_dict = {
                        k: v
                        for k, v in response.__dict__.items()
                        if not k.startswith("_")
                    }

                choices = response_dict.get("choices") or []
                if choices:
                    first_choice = (
                        choices[0]
                        if isinstance(choices[0], dict)
                        else choices[0].__dict__
                    )
                    message_dict = first_choice.get("message", {})
                    if not isinstance(message_dict, dict):
                        message_dict = (
                            message_dict.__dict__
                            if hasattr(message_dict, "__dict__")
                            else {}
                        )

                    content = message_dict.get("content", "")
                    tool_calls_raw = message_dict.get("tool_calls") or []

                    tool_calls = []
                    for call in tool_calls_raw:
                        if isinstance(call, dict):
                            call_dict = call
                        else:
                            call_dict = (
                                call.__dict__
                                if hasattr(call, "__dict__")
                                else {}
                            )

                        function_dict = call_dict.get("function", {})
                        if not isinstance(function_dict, dict):
                            function_dict = (
                                function_dict.__dict__
                                if hasattr(function_dict, "__dict__")
                                else {}
                            )

                        args_str = function_dict.get("arguments", "{}")
                        if isinstance(args_str, str):
                            try:
                                args = json.loads(args_str)
                            except json.JSONDecodeError:
                                args = {}
                        else:
                            args = args_str

                        tool_calls.append({
                            "id": call_dict.get("id", ""),
                            "name": function_dict.get("name", ""),
                            "args": args,
                        })

                    # 构建 AIMessage，tool_calls 不能为 None
                    message_kwargs = {
                        "content": content or "",
                        "response_metadata": {
                            "usage": response_dict.get("usage")
                        },
                    }
                    if tool_calls:
                        message_kwargs["tool_calls"] = tool_calls

                    return AIMessage(**message_kwargs)

                return AIMessage(content=str(response))

        return AgentRunLangChainChatModel()
