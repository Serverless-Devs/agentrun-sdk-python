"""MCP协议处理 / MCP Protocol Handler

处理MCP(Model Context Protocol)协议的工具调用。
Handles tool invocations for MCP (Model Context Protocol).
"""

from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

from agentrun.utils.config import Config
from agentrun.utils.log import logger
from agentrun.utils.ram_signature.auth import AgentrunRamAuth


def _rewrite_to_ram_url(url: str) -> str:
    """将 agentrun-data 域名改写为 -ram 端点。"""
    parsed = urlparse(url)
    parts = parsed.netloc.split(".", 1)
    if len(parts) == 2:
        ram_netloc = parts[0] + "-ram." + parts[1]
        return urlunparse((
            parsed.scheme,
            ram_netloc,
            parsed.path or "",
            parsed.params,
            parsed.query,
            parsed.fragment,
        ))
    return url


class MCPSession:

    def __init__(self, url: str, config: Optional[Config] = None):
        self.url = url
        self.config = Config.with_configs(config)

    def _build_ram_auth(self, url: str) -> tuple:
        """当目标是 agentrun-data 域名时，改写 URL 并返回 httpx Auth handler。

        Returns:
            (rewritten_url, auth_or_none)
        """
        parsed = urlparse(url)
        if ".agentrun-data." not in (parsed.netloc or ""):
            return url, None

        cfg = self.config
        ak = cfg.get_access_key_id()
        sk = cfg.get_access_key_secret()
        if not ak or not sk:
            return url, None

        url = _rewrite_to_ram_url(url)

        auth = AgentrunRamAuth(config=cfg)
        return url, auth

    async def __aenter__(self):
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        timeout = self.config.get_timeout()
        headers = self.config.get_headers()
        url = self.url

        url, auth = self._build_ram_auth(url)

        self.client = sse_client(
            url=url,
            headers=headers,
            auth=auth,
            timeout=timeout if timeout else 60,
        )
        read, write = await self.client.__aenter__()

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
