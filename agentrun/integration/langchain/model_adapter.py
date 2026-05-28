"""LangChain 模型适配器 / LangChain Model Adapter

将 CommonModel 包装为 LangChain BaseChatModel。"""

from typing import Any

from agentrun.integration.langchain.message_adapter import (
    LangChainMessageAdapter,
)
from agentrun.integration.utils.adapter import ModelAdapter

_DEEPSEEK_PROVIDER = "deepseek"


class LangChainModelAdapter(ModelAdapter):
    """LangChain 模型适配器 / LangChain Model Adapter

    将 CommonModel 包装为 LangChain BaseChatModel。"""

    def __init__(self):
        """初始化适配器，创建内部的消息适配器 / LangChain Message Adapter"""
        self._message_adapter = LangChainMessageAdapter()

    def wrap_model(self, common_model: Any) -> Any:
        """包装 CommonModel 为 LangChain BaseChatModel / LangChain Model Adapter"""
        info = common_model.get_model_info()  # 确保模型可用
        provider = (info.provider or "").lower()

        if provider == _DEEPSEEK_PROVIDER:
            return self._create_reasoning_model(info)
        return self._create_openai_model(info)

    def _create_reasoning_model(self, info: Any) -> Any:
        """创建支持 reasoning_content 的模型（使用 ChatDeepSeek）"""
        try:
            from langchain_deepseek import ChatDeepSeek
        except ImportError as e:
            raise ImportError(
                "import langchain_deepseek failed. "
                "Install it with: pip install 'agentrun-sdk[langchain]' "
                "or pip install 'agentrun-sdk[langgraph]'"
            ) from e

        return ChatDeepSeek(
            name=info.model,
            model=info.model,
            api_key=info.api_key,
            api_base=info.base_url,
            default_headers=info.headers,
            stream_usage=True,
            streaming=True,
        )

    def _create_openai_model(self, info: Any) -> Any:
        """创建标准 OpenAI 兼容模型"""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "import langchain_openai failed. "
                "Install it with: pip install 'agentrun-sdk[langchain]' "
                "or pip install 'agentrun-sdk[langgraph]'"
            ) from e

        return ChatOpenAI(
            name=info.model,
            api_key=info.api_key,
            model=info.model,
            base_url=info.base_url,
            default_headers=info.headers,
            stream_usage=True,
            streaming=True,
        )
