"""PydanticAI 工具适配器 / PydanticAI Tool Adapter"""

from typing import Any, List

from agentrun.integration.utils.adapter import ToolAdapter
from agentrun.integration.utils.canonical import CanonicalTool


class PydanticAIToolAdapter(ToolAdapter):
    """PydanticAI 工具适配器 / PydanticAI Tool Adapter

    PydanticAI 使用函数作为工具，需要附加元数据信息。"""

    def from_canonical(self, tools: List[CanonicalTool]) -> List[Any]:
        """将标准工具转换为 PydanticAI 函数格式 / PydanticAI Tool Adapter"""
        return self.function_tools(tools)


__all__ = ["PydanticAIToolAdapter"]
