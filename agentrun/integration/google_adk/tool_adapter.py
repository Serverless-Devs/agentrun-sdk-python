"""Google ADK 工具适配器 / Google ADK Tool Adapter

将标准工具定义转换为 Google ADK 函数格式。"""

from typing import List, Optional

from agentrun.integration.utils.adapter import ToolAdapter
from agentrun.integration.utils.canonical import CanonicalTool


class GoogleADKToolAdapter(ToolAdapter):
    """Google ADK 工具适配器 / Google ADK Tool Adapter

    实现 CanonicalTool → Google ADK 函数的转换。
        Google ADK 直接使用 Python 函数作为工具。"""

    def get_registered_tool(self, name: str) -> Optional[CanonicalTool]:
        """根据名称获取最近注册的工具定义 / Google ADK Tool Adapter"""
        return self._registered_tools.get(name)

    def from_canonical(self, tools: List[CanonicalTool]):
        """将标准格式转换为 Google ADK 工具 / Google ADK Tool Adapter

        Google ADK 通过函数的类型注解推断参数，需要动态创建带注解的函数。"""
        return self.function_tools(tools)
