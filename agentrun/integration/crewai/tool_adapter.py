"""LangChain 工具适配器 / CrewAI Tool Adapter

将标准工具定义转换为 LangChain StructuredTool 格式。"""

from typing import Any, List

from agentrun.integration.utils.adapter import ToolAdapter
from agentrun.integration.utils.canonical import CanonicalTool


class CrewAIToolAdapter(ToolAdapter):
    """CrewAI 工具适配器 / CrewAI Tool Adapter

    实现 CanonicalTool → CrewAI StructuredTool 的转换。"""

    def from_canonical(self, tools: List[CanonicalTool]) -> Any:
        """将标准格式转换为 CrewAI StructuredTool / CrewAI Tool Adapter"""
        try:
            from crewai.tools import tool
        except ImportError as e:
            raise ImportError(
                "LangChain is not installed. "
                "Install it with: pip install langchain-core"
            ) from e

        return [tool(t) for t in self.function_tools(tools)]
