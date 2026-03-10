"""MCP协议处理 / MCP Protocol Handler

处理MCP(Model Context Protocol)协议的工具调用。
Handles tool invocations for MCP (Model Context Protocol).
"""

from typing import Any, Dict, Generator, Optional
from urllib.parse import urlparse, urlunparse

import httpx

from agentrun.utils.config import Config
from agentrun.utils.log import logger
from agentrun.utils.ram_signature import get_agentrun_signed_headers


class _AgentrunRamAuth(httpx.Auth):
    """httpx Auth handler：为每次请求动态生成 RAM 签名。

    SSE 场景下同一个 httpx.AsyncClient 会发出 GET（SSE 连接）和
    POST（消息发送）请求，URL / method / body 各不相同，因此必须
    per-request 计算签名，不能在 client 初始化时一次性设置 headers。
    """

    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        region: str,
        security_token: Optional[str] = None,
    ):
        self._ak = access_key_id
        self._sk = access_key_secret
        self._region = region
        self._security_token = security_token

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        url = str(request.url)
        method = request.method

        body: Optional[bytes] = None
        if request.content:
            body = request.content

        content_type: Optional[str] = request.headers.get("content-type")

        try:
            signed = get_agentrun_signed_headers(
                url=url,
                method=method,
                access_key_id=self._ak,
                access_key_secret=self._sk,
                security_token=self._security_token,
                region=self._region,
                product="agentrun",
                body=body,
                content_type=content_type,
            )
            for k, v in signed.items():
                request.headers[k] = v
            logger.debug(
                "applied RAM signature for MCP %s request to %s",
                method,
                url[:80] + ("..." if len(url) > 80 else ""),
            )
        except ValueError as e:
            logger.warning("RAM signing skipped for MCP request: %s", e)

        yield request


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

        auth = _AgentrunRamAuth(
            access_key_id=ak,
            access_key_secret=sk,
            region=cfg.get_region_id(),
            security_token=cfg.get_security_token() or None,
        )
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
