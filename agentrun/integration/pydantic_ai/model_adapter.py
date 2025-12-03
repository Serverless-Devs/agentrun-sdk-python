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


__all__ = ["PydanticAIModelAdapter"]
