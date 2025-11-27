"""MCP协议处理 / MCP Protocol Handler

处理MCP(Model Context Protocol)协议的工具调用。
Handles tool invocations for MCP (Model Context Protocol).
"""

from typing import Any, Dict, Optional

from agentrun.utils.config import Config
from agentrun.utils.log import logger


class MCPSession:

    def __init__(self, url: str, config: Optional[Config] = None):
        self.url = url
        self.config = Config.with_configs(config)

    async def __aenter__(self):
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        timeout = self.config.get_timeout()
        self.client = sse_client(
            url=self.url,
            headers=self.config.get_headers(),
            timeout=timeout if timeout else 60,
        )
        (read, write) = await self.client.__aenter__()

        self.client_session = ClientSession(read, write)
        session = await self.client_session.__aenter__()
        await session.initialize()

        return session

    async def __aexit__(self, *args):
        await self.client_session.__aexit__(*args)
        await self.client.__aexit__(*args)

    def toolsets(self, config: Optional[Config] = None):
        return MCPToolSet(url=self.url + "/toolsets", config=config)


class MCPToolSet:

    def __init__(self, url: str, config: Optional[Config] = None):
        try:
            __import__("mcp")
        except ImportError:
            logger.warning(
                "MCPToolSet requires Python 3.10 or higher and install 'mcp'"
                " package."
            )

        self.url = url
        self.config = Config.with_configs(config)

    def new_session(self, config: Optional[Config] = None):
        cfg = Config.with_configs(self.config, config)
        return MCPSession(url=self.url, config=cfg)

    async def tools_async(self, config: Optional[Config] = None):
        async with self.new_session(config=config) as session:
            results = await session.list_tools()
            return results.tools

    def tools(self, config: Optional[Config] = None):
        import asyncio

        return asyncio.run(self.tools_async(config=config))

    async def call_tool_async(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
    ):
        async with self.new_session(config=config) as session:
            result = await session.call_tool(
                name=name,
                arguments=arguments,
            )
            return [item.model_dump() for item in result.content]

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
    ):
        import asyncio

        return asyncio.run(
            self.call_tool_async(
                name=name,
                arguments=arguments,
                config=config,
            )
        )
