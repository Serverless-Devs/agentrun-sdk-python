"""CrewAI 集成模块。 / CrewAI 集成 Module

提供 AgentRun 模型与沙箱工具的 CrewAI 适配入口。 / 提供 AgentRun 模型with沙箱工具的 CrewAI 适配入口。
CrewAI 与 LangChain 兼容，因此直接复用 LangChain 的转换逻辑。 / CrewAI with LangChain 兼容，因此直接复用 LangChain 的转换逻辑。
"""

from .builtin import model, sandbox_toolset

__all__ = [
    "model",
    "sandbox_toolset",
]
