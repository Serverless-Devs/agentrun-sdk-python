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
        from langchain_openai import ChatOpenAI

        info = common_model.get_model_info()  # 确保模型可用
        return ChatOpenAI(
            name=info.model,
            api_key=info.api_key,
            model=info.model,
            base_url=info.base_url,
            default_headers=info.headers,
            stream_usage=True,
            streaming=True,  # 启用流式输出以支持 token by token
        )
