"""LangGraph 适配器 / LangGraph Adapters

LangGraph 与 LangChain 完全兼容，因此直接复用 LangChain 的适配器。"""

from agentrun.integration.langchain.adapter import (
    LangChainMessageAdapter,
    LangChainModelAdapter,
    LangChainToolAdapter,
)

# LangGraph 使用与 LangChain 相同的适配器
LangGraphMessageAdapter = LangChainMessageAdapter
LangGraphToolAdapter = LangChainToolAdapter
LangGraphModelAdapter = LangChainModelAdapter

__all__ = [
    "LangGraphMessageAdapter",
    "LangGraphToolAdapter",
    "LangGraphModelAdapter",
]
