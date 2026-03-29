"""Tool 客户端 / Tool Client

此模块提供工具的客户端 API。
This module provides the client API for tools.
"""

from typing import Any, Dict, List, Optional

from agentrun.tool.api.control import ToolControlAPI
from agentrun.utils.config import Config
from agentrun.utils.exception import HTTPError

from .tool import Tool


class ToolClient:
    """Tool 客户端 / Tool Client

    提供工具的获取功能。
    Provides get function for tools.
    """

    def __init__(self, config: Optional[Config] = None):
        """初始化客户端 / Initialize client

        Args:
            config: 配置对象,可选 / Configuration object, optional
        """
        self.__control_api = ToolControlAPI(config)

    async def get_async(
        self,
        name: str,
        config: Optional[Config] = None,
    ) -> "Tool":
        """异步获取工具 / Get tool asynchronously

        Args:
            name: 工具名称 / Tool name
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            Tool: 工具资源对象 / Tool resource object
        """
        try:
            result = await self.__control_api.get_tool_async(
                name=name,
                config=config,
            )
        except HTTPError as e:
            raise e.to_resource_error("Tool", name) from e

        return Tool.from_inner_object(result)
