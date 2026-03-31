"""LangChain 模型适配器 / LangChain Model Adapter

将 CommonModel 包装为 LangChain BaseChatModel。"""

from typing import Any, Set

from agentrun.integration.langchain.message_adapter import (
    LangChainMessageAdapter,
)
from agentrun.integration.utils.adapter import ModelAdapter

OPENAI_COMPATIBLE_PROVIDERS: Set[str] = {
    "openai",
    "tongyi",
    "deepseek",
    "moonshot",
    "baichuan",
    "hunyuan",
    "minimax",
    "spark",
    "stepfun",
    "wenxin",
    "yi",
    "zhipuai",
    "custom",
}


class LangChainModelAdapter(ModelAdapter):
    """LangChain 模型适配器 / LangChain Model Adapter

    将 CommonModel 包装为 LangChain BaseChatModel。
    根据 provider 自动选择对应的 LangChain Chat Model 类。"""

    def __init__(self):
        """初始化适配器，创建内部的消息适配器 / LangChain Message Adapter"""
        self._message_adapter = LangChainMessageAdapter()

    def wrap_model(self, common_model: Any) -> Any:
        """包装 CommonModel 为 LangChain BaseChatModel / LangChain Model Adapter

        根据 BaseInfo.provider 分发到对应的 LangChain Chat Model 类：
        - anthropic -> ChatAnthropic
        - gemini / vertex_ai -> ChatGoogleGenerativeAI
        - 其他（openai 兼容） -> ChatOpenAI
        """
        info = common_model.get_model_info()
        provider = (info.provider or "").lower()

        if provider == "anthropic":
            return self._create_anthropic(info)
        elif provider in ("gemini", "vertex_ai"):
            return self._create_google(info)
        else:
            return self._create_openai(info)

    def _create_openai(self, info: Any) -> Any:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            name=info.model,
            api_key=info.api_key,
            model=info.model,
            base_url=info.base_url,
            default_headers=info.headers,
            stream_usage=True,
            streaming=True,
        )

    def _create_anthropic(self, info: Any) -> Any:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "langchain-anthropic is required for Anthropic models. "
                "Install it with: "
                'pip install "agentrun-sdk[langchain-anthropic]"'
            ) from e

        kwargs: dict[str, Any] = {
            "model": info.model or "",
            "anthropic_api_key": info.api_key,
            "default_headers": info.headers or {},
            "streaming": True,
        }
        if info.base_url:
            kwargs["anthropic_api_url"] = info.base_url

        return ChatAnthropic(**kwargs)

    def _create_google(self, info: Any) -> Any:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai is required for Google / "
                "Vertex AI models. Install it with: "
                'pip install "agentrun-sdk[langchain-google]"'
            ) from e

        kwargs: dict[str, Any] = {
            "model": info.model or "",
            "google_api_key": info.api_key,
            "default_headers": info.headers or {},
        }

        return ChatGoogleGenerativeAI(**kwargs)
