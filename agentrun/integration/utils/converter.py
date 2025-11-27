"""框架转换器 / Framework Converter

提供统一的框架适配器注册中心。
Provides a unified registry for framework adapters.
"""

from typing import Dict, Optional

from agentrun.integration.utils.adapter import ModelAdapter, ToolAdapter
from agentrun.utils.log import logger


class FrameworkConverter:
    """框架适配器注册中心

    管理所有框架的工具和模型适配器。
    MessageAdapter 不再单独注册，而是作为 ModelAdapter 的内部组件。
    """

    def __init__(self):
        self._tool_adapters: Dict[str, ToolAdapter] = {}
        self._model_adapters: Dict[str, ModelAdapter] = {}

    def register_tool_adapter(
        self, framework: str, adapter: ToolAdapter
    ) -> None:
        """注册工具适配器"""
        self._tool_adapters[framework] = adapter

    def register_model_adapter(
        self, framework: str, adapter: ModelAdapter
    ) -> None:
        """注册模型适配器"""
        self._model_adapters[framework] = adapter

    def get_model_adapter(self, framework: str) -> Optional[ModelAdapter]:
        """获取模型适配器"""
        return self._model_adapters.get(framework)


# 全局转换器实例
_converter = FrameworkConverter()


def get_converter() -> FrameworkConverter:
    """获取全局转换器实例"""
    return _converter


def _auto_register_adapters() -> None:
    """自动注册所有可用的适配器

    延迟导入，避免循环依赖。
    MessageAdapter 不再单独注册，由 ModelAdapter 内部管理。
    """
    # LangChain 适配器
    try:
        from agentrun.integration.langchain.adapter import (
            LangChainModelAdapter,
            LangChainToolAdapter,
        )

        _converter.register_tool_adapter("langchain", LangChainToolAdapter())
        _converter.register_model_adapter("langchain", LangChainModelAdapter())
    except (ImportError, AttributeError) as e:
        logger.warning("failed to register LangChain adapters, due to %s", e)

    # Google ADK 适配器
    try:
        from agentrun.integration.google_adk.adapter import (
            GoogleADKModelAdapter,
            GoogleADKToolAdapter,
        )

        _converter.register_tool_adapter("google_adk", GoogleADKToolAdapter())
        _converter.register_model_adapter("google_adk", GoogleADKModelAdapter())
    except (ImportError, AttributeError) as e:
        logger.warning("failed to register Google ADK adapters, due to %s", e)

    # AgentScope 适配器
    try:
        from agentrun.integration.agentscope.adapter import (
            AgentScopeModelAdapter,
            AgentScopeToolAdapter,
        )

        _converter.register_tool_adapter("agentscope", AgentScopeToolAdapter())
        _converter.register_model_adapter(
            "agentscope", AgentScopeModelAdapter()
        )
    except (ImportError, AttributeError) as e:
        logger.warning("failed to register AgentScope adapters, due to %s", e)

    # LangGraph 适配器（复用 LangChain）
    try:
        from agentrun.integration.langgraph.adapter import (
            LangGraphModelAdapter,
            LangGraphToolAdapter,
        )

        _converter.register_tool_adapter("langgraph", LangGraphToolAdapter())
        _converter.register_model_adapter("langgraph", LangGraphModelAdapter())
    except (ImportError, AttributeError) as e:
        logger.warning("failed to register LangGraph adapters, due to %s", e)

    # CrewAI 适配器（复用 LangChain）
    try:
        from agentrun.integration.crewai.adapter import (
            CrewAIModelAdapter,
            CrewAIToolAdapter,
        )

        _converter.register_tool_adapter("crewai", CrewAIToolAdapter())
        _converter.register_model_adapter("crewai", CrewAIModelAdapter())
    except (ImportError, AttributeError) as e:
        logger.warning("failed to register CrewAI adapters, due to %s", e)

    # PydanticAI 适配器
    try:
        from agentrun.integration.pydantic_ai.adapter import (
            PydanticAIModelAdapter,
            PydanticAIToolAdapter,
        )

        _converter.register_tool_adapter("pydantic_ai", PydanticAIToolAdapter())
        _converter.register_model_adapter(
            "pydantic_ai", PydanticAIModelAdapter()
        )
    except (ImportError, AttributeError) as e:
        logger.warning("failed to register PydanticAI adapters, due to %s", e)


# 初始化时自动注册
_auto_register_adapters()
