"""Tool MCP 数据链路 / Tool MCP Data API

通过 MCP 协议与 Tool 的数据链路交互，支持 SSE 和 Streamable HTTP 两种传输方式。
Interacts with Tool data endpoints via MCP protocol, supporting SSE and Streamable HTTP transports.
"""

import asyncio
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from agentrun.tool.model import ToolInfo
from agentrun.utils.config import Config
from agentrun.utils.log import logger
from agentrun.utils.ram_signature.auth import AgentrunRamAuth

_MCP_METADATA_TIMEOUT_SECONDS = 30.0


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """获取当前线程的事件循环，如果不存在则创建一个新的。
    Get the event loop for the current thread, creating a new one if none exists.

    Python 3.10+ 在非主线程中调用 asyncio.get_event_loop() 时，
    如果该线程没有事件循环会抛出 RuntimeError。此函数安全地处理该情况。
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


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
        config: Optional[Config] = None,
        use_ram_auth: bool = True,
    ):
        """初始化 MCP 会话 / Initialize MCP session

        Args:
            endpoint: MCP 数据链路 URL / MCP data endpoint URL
            session_affinity: 会话亲和性策略 / Session affinity strategy
            headers: 请求头 / Request headers
            config: 配置对象 / Configuration object
            use_ram_auth: 是否启用 RAM 签名鉴权 / Whether to enable RAM signature auth.
                MCP_REMOTE + proxy_enabled=false 时设为 False（直连外部服务）。
                Set to False for MCP_REMOTE with proxy disabled (direct external connection).
        """
        self.endpoint = endpoint
        self.session_affinity = session_affinity
        self.headers = headers or {}
        self.config = Config.with_configs(config)
        self.use_ram_auth = use_ram_auth

    @property
    def is_streamable(self) -> bool:
        """是否使用 Streamable HTTP 传输 / Whether to use Streamable HTTP transport"""
        return self.session_affinity == "MCP_STREAMABLE"

    def _metadata_timeout_seconds(self) -> float:
        timeout = self.config.get_timeout()
        if timeout and timeout > 0:
            return min(float(timeout), _MCP_METADATA_TIMEOUT_SECONDS)
        return _MCP_METADATA_TIMEOUT_SECONDS

    def _invoke_timeout_seconds(self) -> float:
        timeout = self.config.get_timeout()
        if timeout and timeout > 0:
            return float(timeout)
        return 600.0

    async def _wait_for_mcp_request(
        self,
        awaitable: Any,
        operation: str,
        timeout: float,
    ) -> Any:
        try:
            return await asyncio.wait_for(awaitable, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"MCP {operation} timed out after {timeout:g}s for endpoint"
                f" {self.endpoint}"
            ) from exc

    def _find_mcp_timeout_error(
        self, exc: BaseException
    ) -> Optional[TimeoutError]:
        if isinstance(exc, TimeoutError) and str(exc).startswith("MCP "):
            return exc

        nested_exceptions = getattr(exc, "exceptions", None)
        if not nested_exceptions:
            return None

        for nested_exc in nested_exceptions:
            timeout_error = self._find_mcp_timeout_error(nested_exc)
            if timeout_error is not None:
                return timeout_error

        return None

    def _raise_mcp_timeout_if_present(self, exc: BaseException) -> None:
        timeout_error = self._find_mcp_timeout_error(exc)
        if timeout_error is not None:
            raise timeout_error

    def _build_ram_auth(self, url: str) -> tuple:
        """当目标是 agentrun-data 域名时，改写 URL 并返回 httpx Auth handler。

        Returns:
            (rewritten_url, auth_or_none)
        """
        # MCP_REMOTE + proxy_enabled=false 时不走 RAM 鉴权
        # Skip RAM auth for MCP_REMOTE with proxy disabled
        if not self.use_ram_auth:
            return url, None

        parsed = urlparse(url)
        # 只对 agentrun-data 和 funagent-data-pre 域名应用 RAM 签名
        if ".agentrun-data." not in (
            parsed.netloc or ""
        ) and ".funagent-data-pre." not in (parsed.netloc or ""):
            return url, None

        cfg = self.config
        ak = cfg.get_access_key_id()
        sk = cfg.get_access_key_secret()
        if not ak or not sk:
            return url, None

        url = _rewrite_to_ram_url(url)

        auth = AgentrunRamAuth(config=cfg)
        return url, auth

    async def list_tools_async(self) -> List[ToolInfo]:
        """异步获取工具列表 / Get tool list asynchronously

        Returns:
            List[ToolInfo]: 工具信息列表 / List of tool information
        """
        try:
            from mcp import ClientSession

            # 应用 RAM 签名
            url, auth = self._build_ram_auth(self.endpoint)

            if self.is_streamable:
                from mcp.client.streamable_http import streamablehttp_client

                async with streamablehttp_client(
                    url, headers=self.headers, auth=auth
                ) as (read_stream, write_stream, _):
                    async with ClientSession(
                        read_stream, write_stream
                    ) as session:
                        metadata_timeout = self._metadata_timeout_seconds()
                        await self._wait_for_mcp_request(
                            session.initialize(),
                            "initialize",
                            metadata_timeout,
                        )
                        result = await self._wait_for_mcp_request(
                            session.list_tools(),
                            "list_tools",
                            metadata_timeout,
                        )
                        return [
                            ToolInfo.from_mcp_tool(tool)
                            for tool in result.tools
                        ]
            else:
                from mcp.client.sse import sse_client

                async with sse_client(url, headers=self.headers, auth=auth) as (
                    read_stream,
                    write_stream,
                ):
                    async with ClientSession(
                        read_stream, write_stream
                    ) as session:
                        metadata_timeout = self._metadata_timeout_seconds()
                        await self._wait_for_mcp_request(
                            session.initialize(),
                            "initialize",
                            metadata_timeout,
                        )
                        result = await self._wait_for_mcp_request(
                            session.list_tools(),
                            "list_tools",
                            metadata_timeout,
                        )
                        return [
                            ToolInfo.from_mcp_tool(tool)
                            for tool in result.tools
                        ]
        except ImportError:
            logger.warning(
                "mcp package is not installed. Install it with: pip install mcp"
            )
            return []
        except Exception as exc:
            self._raise_mcp_timeout_if_present(exc)
            raise

    def list_tools(self) -> List[ToolInfo]:
        """同步获取工具列表 / Get tool list synchronously

        Returns:
            List[ToolInfo]: 工具信息列表 / List of tool information
        """
        return _get_or_create_event_loop().run_until_complete(
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

            # 应用 RAM 签名
            url, auth = self._build_ram_auth(self.endpoint)

            if self.is_streamable:
                from mcp.client.streamable_http import streamablehttp_client

                async with streamablehttp_client(
                    url, headers=self.headers, auth=auth
                ) as (read_stream, write_stream, _):
                    async with ClientSession(
                        read_stream, write_stream
                    ) as session:
                        await self._wait_for_mcp_request(
                            session.initialize(),
                            "initialize",
                            self._metadata_timeout_seconds(),
                        )
                        result = await self._wait_for_mcp_request(
                            session.call_tool(name, arguments=arguments or {}),
                            f"call_tool {name}",
                            self._invoke_timeout_seconds(),
                        )
                        return result
            else:
                from mcp.client.sse import sse_client

                async with sse_client(url, headers=self.headers, auth=auth) as (
                    read_stream,
                    write_stream,
                ):
                    async with ClientSession(
                        read_stream, write_stream
                    ) as session:
                        await self._wait_for_mcp_request(
                            session.initialize(),
                            "initialize",
                            self._metadata_timeout_seconds(),
                        )
                        result = await self._wait_for_mcp_request(
                            session.call_tool(name, arguments=arguments or {}),
                            f"call_tool {name}",
                            self._invoke_timeout_seconds(),
                        )
                        return result
        except ImportError:
            raise ImportError(
                "mcp package is required for MCP tool calls. "
                "Install it with: pip install mcp"
            )
        except Exception as exc:
            self._raise_mcp_timeout_if_present(exc)
            raise

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
        return _get_or_create_event_loop().run_until_complete(
            self.call_tool_async(name, arguments)
        )
