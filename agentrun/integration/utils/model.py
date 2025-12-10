"""通用模型定义和转换模块 / Common Model Definition and Conversion Module

提供跨框架的通用模型定义和转换功能。
Provides cross-framework model definition and conversion capabilities.
"""

from typing import Any, Optional, Union

from agentrun.model import BackendType, ModelProxy, ModelService
from agentrun.utils.config import Config
from agentrun.utils.log import logger


class CommonModel:
    """通用模型定义

    封装 AgentRun 模型，提供跨框架转换能力。
    """

    def __init__(
        self,
        model_obj: Union[ModelService, ModelProxy],
        backend_type: Optional[BackendType] = None,
        specific_model: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.model_obj = model_obj
        self.backend_type = backend_type
        self.specific_model = specific_model
        self.config = config or Config()

    def completions(self, *args, **kwargs):
        """调用底层模型的 completions 方法"""
        return self.model_obj.completions(*args, **kwargs)

    def responses(self, *args, **kwargs):
        """调用底层模型的 responses 方法"""
        return self.model_obj.responses(*args, **kwargs)

    def get_model_info(self, config: Optional[Config] = None):
        """获取模型信息"""
        cfg = Config.with_configs(self.config, config)
        info = self.model_obj.model_info(config=cfg)
        if self.specific_model:
            info.model = self.specific_model
        return info

    def __convert_model(self, adapter_name: str):
        try:
            from agentrun.integration.utils.converter import get_converter

            converter = get_converter()
            adapter = converter.get_model_adapter(adapter_name)
            if adapter is not None:
                return adapter.wrap_model(self)
        except (ImportError, AttributeError, KeyError) as e:
            logger.warning("convert model failed, due to %s", e)
            pass

        logger.warning(
            f"Model adapter '{adapter_name}' not found. Falling back to default"
            " implementation."
        )

        return ""

    def to_google_adk(self) -> Any:
        """转换为 Google ADK BaseLlm

        优先使用适配器模式，如果适配器未注册则回退到旧实现。
        """
        # 尝试使用适配器模式
        return self.__convert_model("google_adk")

    def to_langchain(self) -> Any:
        """转换为 LangChain ChatModel

        优先使用适配器模式，如果适配器未注册则回退到旧实现。
        """
        # 尝试使用适配器模式
        return self.__convert_model("langchain")

    def __call__(self, messages: list, **kwargs) -> Any:
        """直接调用模型"""
        return self.completions(messages=messages, **kwargs)

    def to_langgraph(self) -> Any:
        """转换为 LangGraph 兼容的模型。

        LangGraph 与 LangChain 完全兼容，因此使用相同的接口。
        """
        return self.__convert_model("langgraph")

    def to_crewai(self) -> Any:
        """转换为 CrewAI 兼容的模型。

        CrewAI 内部使用 LangChain，因此使用相同的接口。
        """
        return self.__convert_model("crewai")

    def to_pydantic_ai(self) -> Any:
        """转换为 PydanticAI 兼容的模型。

        PydanticAI 支持 OpenAI 兼容的接口，返回一个包装对象。
        """
        return self.__convert_model("pydantic_ai")

    def to_agentscope(self) -> Any:
        """转换为 AgentScope ChatModelBase。"""
        return self.__convert_model("agentscope")
