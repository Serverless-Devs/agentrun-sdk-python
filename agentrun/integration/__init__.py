"""Integration 模块 / Integration Module

提供与主流 AI 框架的集成适配器。
Provides integration adapters for mainstream AI frameworks.

支持的框架 / Supported Frameworks:
- LangChain: 流行的 LLM 应用开发框架 / Popular LLM application development framework
- LangGraph: LangChain 的图形化工作流扩展 / Graph-based workflow extension for LangChain
- Google ADK: Google 的 AI Development Kit / Google's AI Development Kit
- AgentScope: 阿里云的 Agent 开发框架 / Alibaba Cloud's agent development framework
- PydanticAI: 基于 Pydantic 的 AI 框架 / Pydantic-based AI framework
- CrewAI: 多 Agent 协作框架 / Multi-agent collaboration framework

使用方法 / Usage:
>>> from agentrun.integration.langchain import wrap_model, wrap_tools
>>> # 或 / Or
>>> from agentrun.integration.utils.model import CommonModel
>>> from agentrun.integration.utils.tool import CommonToolSet
"""

__all__ = []
