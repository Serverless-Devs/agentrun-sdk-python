"""Google ADK 模型适配器 / Google ADK Model Adapter

将 CommonModel 包装为 Google ADK BaseLlm。"""

from typing import Any

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

        info = common_model.get_model_info()

        return LiteLlm(
            model=f"{info.provider or 'openai'}/{info.model}",
            api_base=info.base_url,
            api_key=info.api_key,
            extra_headers=info.headers,
            stream_options={"include_usage": True},
        )
