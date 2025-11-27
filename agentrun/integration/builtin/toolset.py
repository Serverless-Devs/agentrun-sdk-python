"""内置工具集集成函数 / Built-in ToolSet Integration Functions

提供快速创建通用工具集对象的便捷函数。
Provides convenient functions for quickly creating common toolset objects.
"""

from typing import Optional, Union

from agentrun.integration.utils.tool import CommonToolSet
from agentrun.toolset import ToolSet, ToolSetClient
from agentrun.utils.config import Config


def toolset(
    input: Union[str, ToolSet], config: Optional[Config] = None
) -> CommonToolSet:
    """将内置工具集封装为通用工具集 / Wrap built-in toolset as CommonToolSet

    支持从工具集名称或 ToolSet 实例创建通用工具集。
    Supports creating CommonToolSet from toolset name or ToolSet instance.

    Args:
        input: 工具集名称或 ToolSet 实例 / Toolset name or ToolSet instance
        config: 配置对象 / Configuration object

    Returns:
        CommonToolSet: 通用工具集实例 / CommonToolSet instance

    Examples:
        >>> # 从工具集名称创建 / Create from toolset name
        >>> ts = toolset("my-toolset")
        >>>
        >>> # 从 ToolSet 实例创建 / Create from ToolSet instance
        >>> toolset_obj = ToolSetClient().get("my-toolset")
        >>> ts = toolset(toolset_obj)
        >>>
        >>> # 转换为 LangChain 工具 / Convert to LangChain tools
        >>> lc_tools = ts.to_langchain()
    """

    toolset = (
        input
        if isinstance(input, ToolSet)
        else ToolSetClient().get(name=input, config=config)
    )

    return CommonToolSet.from_agentrun_toolset(toolset, config=config)
