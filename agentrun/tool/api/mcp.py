"""Tool MCP 数据链路 / Tool MCP Data API

通过 MCP 协议与 Tool 的数据链路交互，支持 SSE 和 Streamable HTTP 两种传输方式。
Interacts with Tool data endpoints via MCP protocol, supporting SSE and Streamable HTTP transports.
"""

import asyncio
from typing import Any, Dict, List, Optional

from agentrun.tool.model import ToolInfo, ToolSchema
from agentrun.utils.log import logger


class ToolMCPSession:
    """Tool MCP 会话管理 / Tool MCP Session Manager

    独立实现的 MCP 会话管理，支持 SSE 和 Streamable HTTP 两种传输方式。
    Independent MCP session manager supporting SSE and Streamable HTTP transports.
    """

    def __init__(
        self,
        endpoint: str,
        session_affinity: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """初始化 MCP 会话 / Initialize MCP session

        Args:
            endpoint: MCP 数据链路 URL / MCP data endpoint URL
            session_affinity: 会话亲和性策略 / Session affinity strategy
            headers: 请求头 / Request headers
        """
        self.endpoint = endpoint
        self.session_affinity = session_affinity
        self.headers = headers or {}

    @property
    def is_streamable(self) -> bool:
        """是否使用 Streamable HTTP 传输 / Whether to use Streamable HTTP transport"""
        return self.session_affinity == "MCP_STREAMABLE"

    async def list_tools_async(self) -> List[ToolInfo]:
        """异步获取工具列表 / Get tool list asynchronously

        Returns:
            List[ToolInfo]: 工具信息列表 / List of tool information
        """
        try:
            from mcp import ClientSession

            if self.is_streamable:
                from mcp.client.streamable_http import streamablehttp_client

                async with streamablehttp_client(
                    self.endpoint, headers=self.headers
                ) as (read_stream, write_stream, _):
                    async with ClientSession(
                        read_stream, write_stream
                    ) as session:
                        await session.initialize()
                        result = await session.list_tools()
                        return [
                            ToolInfo.from_mcp_tool(tool)
                            for tool in result.tools
                        ]
            else:
                from mcp.client.sse import sse_client

                async with sse_client(self.endpoint, headers=self.headers) as (
                    read_stream,
                    write_stream,
                ):
                    async with ClientSession(
                        read_stream, write_stream
                    ) as session:
                        await session.initialize()
                        result = await session.list_tools()
                        return [
                            ToolInfo.from_mcp_tool(tool)
                            for tool in result.tools
                        ]
        except ImportError:
            logger.warning(
                "mcp package is not installed. Install it with: pip install mcp"
            )
            return []

    def list_tools(self) -> List[ToolInfo]:
        """同步获取工具列表 / Get tool list synchronously

        Returns:
            List[ToolInfo]: 工具信息列表 / List of tool information
        """
        return asyncio.get_event_loop().run_until_complete(
            self.list_tools_async()
        )

    async def call_tool_async(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """异步调用工具 / Call tool asynchronously

        Args:
            name: 子工具名称 / Sub-tool name
            arguments: 调用参数 / Call arguments

        Returns:
            Any: 工具执行结果 / Tool execution result
        """
        try:
            from mcp import ClientSession

            if self.is_streamable:
                from mcp.client.streamable_http import streamablehttp_client

                async with streamablehttp_client(
                    self.endpoint, headers=self.headers
                ) as (read_stream, write_stream, _):
                    async with ClientSession(
                        read_stream, write_stream
                    ) as session:
                        await session.initialize()
                        result = await session.call_tool(
                            name, arguments=arguments or {}
                        )
                        return result
            else:
                from mcp.client.sse import sse_client

                async with sse_client(self.endpoint, headers=self.headers) as (
                    read_stream,
                    write_stream,
                ):
                    async with ClientSession(
                        read_stream, write_stream
                    ) as session:
                        await session.initialize()
                        result = await session.call_tool(
                            name, arguments=arguments or {}
                        )
                        return result
        except ImportError:
            raise ImportError(
                "mcp package is required for MCP tool calls. "
                "Install it with: pip install mcp"
            )

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """同步调用工具 / Call tool synchronously

        Args:
            name: 子工具名称 / Sub-tool name
            arguments: 调用参数 / Call arguments

        Returns:
            Any: 工具执行结果 / Tool execution result
        """
        return asyncio.get_event_loop().run_until_complete(
            self.call_tool_async(name, arguments)
        )
