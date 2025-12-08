"""AgentScope 模型适配器 / AgentScope Model Adapter

将 CommonModel 包装为 AgentScope ChatModelBase。"""

from __future__ import annotations

from typing import Any

from agentrun.integration.agentscope.message_adapter import (
    AgentScopeMessageAdapter,
)
from agentrun.integration.utils.adapter import ModelAdapter
from agentrun.integration.utils.model import CommonModel


def _ensure_agentscope_installed() -> None:
    try:
        import agentscope  # noqa: F401
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "AgentScope is not installed. Install it with: pip install"
            " agentscope"
        ) from exc


class AgentScopeModelAdapter(ModelAdapter):
    """AgentScope 模型适配器 / AgentScope Model Adapter

    将 CommonModel 包装为 AgentScope ChatModelBase。"""

    def __init__(self):
        """初始化适配器，创建内部的消息适配器 / AgentScope Message Adapter"""
        self._message_adapter = AgentScopeMessageAdapter()

    def wrap_model(self, common_model: CommonModel) -> Any:
        """包装 CommonModel 为 AgentScope ChatModelBase / AgentScope Model Adapter"""
        _ensure_agentscope_installed()

        try:
            from agentscope.model import OpenAIChatModel
        except Exception as e:
            raise ImportError(
                "AgentScope is not installed. Install it with: pip install"
                " agentscope"
            ) from e

        from httpx import AsyncClient

        info = common_model.get_model_info()

        return OpenAIChatModel(
            model_name=info.model or "",
            api_key=info.api_key,
            stream=True,
            client_args={
                "base_url": info.base_url,
                "http_client": AsyncClient(headers=info.headers),
            },
            generate_kwargs={"stream_options": {"include_usage": True}},
        )
