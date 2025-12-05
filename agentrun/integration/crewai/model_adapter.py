"""LangChain 模型适配器 / CrewAI Model Adapter

将 CommonModel 包装为 LangChain BaseChatModel。"""

from typing import Any

from agentrun.integration.utils.adapter import ModelAdapter


class CrewAIModelAdapter(ModelAdapter):
    """CrewAI 模型适配器 / CrewAI Model Adapter

    将 CommonModel 包装为 CrewAI BaseChatModel。"""

    def wrap_model(self, common_model: Any) -> Any:
        """包装 CommonModel 为 CrewAI BaseChatModel / CrewAI Model Adapter"""
        from crewai import LLM

        info = common_model.get_model_info()  # 确保模型可用
        return LLM(
            api_key=info.api_key,
            model=f"{info.provider or 'openai'}/{info.model}",
            base_url=info.base_url,
            default_headers=info.headers,
            stream_options={"include_usage": True},
            # async_client=AsyncClient(headers=info.headers),
        )
