"""内置 ToolResource 集成函数 / Built-in ToolResource Integration Functions

提供快速创建通用工具集对象的便捷函数（基于新版 Tool 模块）。
Provides convenient functions for quickly creating common toolset objects (based on new Tool module).
"""

from typing import Optional, Union

from agentrun.integration.utils.tool import CommonToolSet
from agentrun.tool.client import ToolClient
from agentrun.tool.tool import Tool as ToolResourceType
from agentrun.utils.config import Config


def tool_resource(
    input: Union[str, ToolResourceType], config: Optional[Config] = None
) -> CommonToolSet:
    """将 ToolResource 封装为通用工具集 / Wrap ToolResource as CommonToolSet

    支持从工具名称或 ToolResource 实例创建通用工具集。
    Supports creating CommonToolSet from tool name or ToolResource instance.

    Args:
        input: 工具名称或 ToolResource 实例 / Tool name or ToolResource instance
        config: 配置对象 / Configuration object

    Returns:
        CommonToolSet: 通用工具集实例 / CommonToolSet instance

    Examples:
        >>> # 从工具名称创建 / Create from tool name
        >>> ts = tool_resource("my-tool")
        >>>
        >>> # 从 ToolResource 实例创建 / Create from ToolResource instance
        >>> tool = ToolClient().get(name="my-tool")
        >>> ts = tool_resource(tool)
        >>>
        >>> # 转换为 LangChain 工具 / Convert to LangChain tools
        >>> lc_tools = ts.to_langchain()
    """

    resource = (
        input
        if isinstance(input, ToolResourceType)
        else ToolClient().get(name=input, config=config)
    )

    return CommonToolSet.from_agentrun_tool(resource, config=config)
