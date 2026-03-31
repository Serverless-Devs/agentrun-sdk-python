"""Tool 资源类 / Tool Resource Class

提供工具资源的面向对象封装和完整生命周期管理。
Provides object-oriented wrapper and complete lifecycle management for tool resources.
"""

import io
import os
import shutil
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import zipfile

import httpx
import pydash

from agentrun.utils.config import Config
from agentrun.utils.log import logger
from agentrun.utils.model import BaseModel
from agentrun.utils.ram_signature import get_agentrun_signed_headers

from .model import (
    McpConfig,
    ToolCodeConfiguration,
    ToolContainerConfiguration,
    ToolInfo,
    ToolLogConfiguration,
    ToolNetworkConfiguration,
    ToolOSSMountConfig,
    ToolSchema,
    ToolType,
)


class Tool(BaseModel):
    """工具资源 / Tool Resource

    提供工具的查询、调用等功能。
    Provides query, invocation and other functionality for tools.

    Attributes:
        code_configuration: 代码包配置 / Code configuration
        container_configuration: 容器配置 / Container configuration
        created_time: 创建时间 / Creation time
        data_endpoint: 数据链路端点 / Data endpoint
        description: 描述 / Description
        environment_variables: 环境变量 / Environment variables
        gpu: GPU 配置 / GPU configuration
        internet_access: 是否允许公网访问 / Whether internet access is allowed
        last_modified_time: 最后修改时间 / Last modified time
        log_configuration: 日志配置 / Log configuration
        mcp_config: MCP 配置 / MCP configuration
        memory: 内存大小(MB) / Memory size in MB
        name: 工具名称 / Tool name
        network_config: 网络配置 / Network configuration
        oss_mount_config: OSS 挂载配置 / OSS mount configuration
        protocol_spec: 协议规格(OpenAPI JSON) / Protocol spec (OpenAPI JSON)
        protocol_type: 协议类型 / Protocol type
        status: 状态 / Status
        timeout: 超时时间(秒) / Timeout in seconds
        tool_id: 工具 ID / Tool ID
        tool_name: 工具名称 / Tool name
        tool_type: 工具类型(MCP/FUNCTIONCALL) / Tool type
        version_id: 版本 ID / Version ID
    """

    code_configuration: Optional[ToolCodeConfiguration] = None
    """代码包配置 / Code configuration"""

    container_configuration: Optional[ToolContainerConfiguration] = None
    """容器配置 / Container configuration"""

    created_time: Optional[str] = None
    """创建时间 / Creation time"""

    data_endpoint: Optional[str] = None
    """数据链路端点 / Data endpoint"""

    description: Optional[str] = None
    """描述 / Description"""

    environment_variables: Optional[Dict[str, str]] = None
    """环境变量 / Environment variables"""

    gpu: Optional[str] = None
    """GPU 配置 / GPU configuration"""

    internet_access: Optional[bool] = None
    """是否允许公网访问 / Whether internet access is allowed"""

    last_modified_time: Optional[str] = None
    """最后修改时间 / Last modified time"""

    log_configuration: Optional[ToolLogConfiguration] = None
    """日志配置 / Log configuration"""

    mcp_config: Optional[McpConfig] = None
    """MCP 配置 / MCP configuration"""

    memory: Optional[int] = None
    """内存大小(MB) / Memory size in MB"""

    name: Optional[str] = None
    """工具名称 / Tool name"""

    network_config: Optional[ToolNetworkConfiguration] = None
    """网络配置 / Network configuration"""

    oss_mount_config: Optional[ToolOSSMountConfig] = None
    """OSS 挂载配置 / OSS mount configuration"""

    protocol_spec: Optional[str] = None
    """协议规格(OpenAPI JSON 字符串) / Protocol spec (OpenAPI JSON string)"""

    protocol_type: Optional[str] = None
    """协议类型 / Protocol type"""

    status: Optional[str] = None
    """状态 / Status"""

    timeout: Optional[int] = None
    """超时时间(秒) / Timeout in seconds"""

    tool_id: Optional[str] = None
    """工具 ID / Tool ID"""

    tool_name: Optional[str] = None
    """工具名称 / Tool name"""

    tool_type: Optional[str] = None
    """工具类型(MCP/FUNCTIONCALL) / Tool type (MCP/FUNCTIONCALL)"""

    create_method: Optional[str] = None
    """工具创建方式 / Tool create method
    MCP_REMOTE: 远程 MCP 服务器 / Remote MCP server
    MCP_LOCAL: 本地 MCP 标准输入输出 / Local MCP stdio
    MCP_BUNDLE: MCP 打包部署 / MCP bundle deployment
    CODE_PACKAGE: 代码包部署 / Code package deployment
    OPENAPI_IMPORT: OpenAPI 导入 / OpenAPI import
    """

    version_id: Optional[str] = None
    """版本 ID / Version ID"""

    _RAM_DATA_DOMAINS = ("agentrun-data", "funagent-data-pre")

    @classmethod
    def __get_client(cls, config: Optional[Config] = None):
        from .client import ToolClient

        return ToolClient(config)

    @classmethod
    async def get_by_name_async(
        cls, name: str, config: Optional[Config] = None
    ) -> "Tool":
        """异步通过名称获取工具 / Get tool by name asynchronously"""
        cli = cls.__get_client(config)
        return await cli.get_async(name=name)

    async def get_async(self, config: Optional[Config] = None) -> "Tool":
        """异步刷新工具信息 / Refresh tool info asynchronously"""
        effective_name = self.tool_name or self.name
        if effective_name is None:
            raise ValueError("Tool name is required to get the Tool.")

        result = await self.get_by_name_async(
            name=effective_name, config=config
        )
        return self.update_self(result)

    def _get_functioncall_server_url(
        self, config: Optional[Config] = None
    ) -> Optional[str]:
        """获取 FunctionCall 工具的 fallback server URL / Get fallback server URL for FunctionCall tools

        当 OpenAPI spec 中没有 servers 字段时，使用 data_endpoint 构造 URL。
        Constructs URL from data_endpoint when servers is not present in OpenAPI spec.
        """
        effective_name = self.tool_name or self.name
        data_endpoint = self.data_endpoint
        if not data_endpoint:
            cfg = Config.with_configs(config)
            data_endpoint = cfg.get_data_endpoint()
        if not data_endpoint or not effective_name:
            return None
        return f"{data_endpoint}/tools/{effective_name}"

    def _get_tool_type(self) -> Optional[ToolType]:
        """获取工具类型 / Get tool type"""
        raw_type = self.tool_type
        if raw_type:
            try:
                return ToolType(raw_type)
            except ValueError:
                return None
        return None

    def _get_mcp_endpoint(
        self, config: Optional[Config] = None
    ) -> Optional[str]:
        """获取 MCP 数据链路 URL / Get MCP data endpoint URL

        根据 session_affinity 决定使用 /mcp 还是 /sse 路径。
        如果 self.data_endpoint 为空，则从 Config 中获取。
        Determines /mcp or /sse path based on session_affinity.
        Falls back to Config if self.data_endpoint is not set.
        """
        effective_name = self.tool_name or self.name
        data_endpoint = self.data_endpoint
        if not data_endpoint:
            cfg = Config.with_configs(config)
            data_endpoint = cfg.get_data_endpoint()
        if not data_endpoint or not effective_name:
            return None

        session_affinity = pydash.get(
            self, "mcp_config.session_affinity", "MCP_SSE"
        )

        if session_affinity == "MCP_STREAMABLE":
            return f"{data_endpoint}/tools/{effective_name}/mcp"
        return f"{data_endpoint}/tools/{effective_name}/sse"

    async def list_tools_async(
        self, config: Optional[Config] = None
    ) -> List[ToolInfo]:
        """异步获取子工具列表 / Get sub-tool list asynchronously

        对于 MCP 类型，通过 MCP 协议获取工具列表。
        对于 FUNCTIONCALL 类型，解析 protocol_spec 获取工具列表。
        For MCP type, gets tool list via MCP protocol.
        For FUNCTIONCALL type, parses protocol_spec to get tool list.

        Returns:
            List[ToolInfo]: 子工具信息列表 / List of sub-tool information
        """
        tool_type = self._get_tool_type()

        if tool_type == ToolType.MCP:
            from .api.mcp import ToolMCPSession

            mcp_endpoint = self._get_mcp_endpoint(config)
            if not mcp_endpoint:
                logger.warning(
                    "MCP endpoint not available for tool %s", self.name
                )
                return []

            session_affinity = pydash.get(
                self, "mcp_config.session_affinity", "MCP_SSE"
            )

            # MCP_REMOTE + proxy_enabled=false 时直连外部服务，不走 RAM 鉴权
            # Only skip RAM auth for MCP_REMOTE with proxy disabled (direct external connection)
            is_mcp_remote_without_proxy = (
                self.create_method == "MCP_REMOTE"
                and not pydash.get(self, "mcp_config.proxy_enabled", False)
            )

            cfg = Config.with_configs(config)
            session = ToolMCPSession(
                endpoint=mcp_endpoint,
                session_affinity=session_affinity,
                headers=cfg.get_headers(),
                config=cfg,
                use_ram_auth=not is_mcp_remote_without_proxy,
            )
            return await session.list_tools_async()

        elif tool_type == ToolType.FUNCTIONCALL:
            from .api.openapi import ToolOpenAPIClient

            # OPENAPI_IMPORT 时 server 是外部服务，不走 RAM 鉴权
            # Skip RAM auth for OPENAPI_IMPORT since the server is an external service
            is_openapi_import = self.create_method == "OPENAPI_IMPORT"

            cfg = Config.with_configs(config)
            openapi_client = ToolOpenAPIClient(
                protocol_spec=self.protocol_spec,
                fallback_server_url=self._get_functioncall_server_url(config),
                config=cfg,
                use_ram_auth=not is_openapi_import,
            )
            return await openapi_client.list_tools_async()

        return []

    async def call_tool_async(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
    ) -> Any:
        """异步调用子工具 / Call sub-tool asynchronously

        Args:
            name: 子工具名称 / Sub-tool name
            arguments: 调用参数 / Call arguments
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            Any: 工具执行结果 / Tool execution result
        """
        tool_type = self._get_tool_type()

        logger.debug("invoke tool %s with arguments %s", name, arguments)

        if tool_type == ToolType.MCP:
            from .api.mcp import ToolMCPSession

            mcp_endpoint = self._get_mcp_endpoint(config)
            if not mcp_endpoint:
                raise ValueError(
                    f"MCP endpoint not available for tool {self.name}"
                )

            session_affinity = pydash.get(
                self, "mcp_config.session_affinity", "MCP_SSE"
            )

            # MCP_REMOTE + proxy_enabled=false 时直连外部服务，不走 RAM 鉴权
            # Only skip RAM auth for MCP_REMOTE with proxy disabled (direct external connection)
            is_mcp_remote_without_proxy = (
                self.create_method == "MCP_REMOTE"
                and not pydash.get(self, "mcp_config.proxy_enabled", False)
            )

            cfg = Config.with_configs(config)
            session = ToolMCPSession(
                endpoint=mcp_endpoint,
                session_affinity=session_affinity,
                headers=cfg.get_headers(),
                config=cfg,
                use_ram_auth=not is_mcp_remote_without_proxy,
            )
            result = await session.call_tool_async(name, arguments)
            logger.debug("invoke tool %s got result %s", name, result)
            return result

        elif tool_type == ToolType.FUNCTIONCALL:
            from .api.openapi import ToolOpenAPIClient

            # OPENAPI_IMPORT 时 server 是外部服务，不走 RAM 鉴权
            # Skip RAM auth for OPENAPI_IMPORT since the server is an external service
            is_openapi_import = self.create_method == "OPENAPI_IMPORT"

            cfg = Config.with_configs(config)
            openapi_client = ToolOpenAPIClient(
                protocol_spec=self.protocol_spec,
                headers=cfg.get_headers(),
                fallback_server_url=self._get_functioncall_server_url(config),
                config=cfg,
                use_ram_auth=not is_openapi_import,
            )
            result = await openapi_client.call_tool_async(name, arguments)
            logger.debug("invoke tool %s got result %s", name, result)
            return result

        raise ValueError(f"Unsupported tool type: {self.tool_type}")

    def _use_ram_auth(self, config: Optional[Config] = None) -> bool:
        """是否使用 RAM 签名鉴权（配置了 AK/SK 时使用）。
        Whether to use RAM signature authentication (when AK/SK is configured).
        """
        cfg = Config.with_configs(config)
        return bool(cfg.get_access_key_id() and cfg.get_access_key_secret())

    def _get_ram_data_endpoint(
        self, url: str, config: Optional[Config] = None
    ) -> str:
        """返回 RAM 鉴权用的 data endpoint（仅当 agentrun-data / funagent-data-pre 域名时在 host 前加 -ram）。
        Return RAM-authenticated endpoint (add -ram prefix for agentrun-data / funagent-data-pre domains).
        """
        parsed = urlparse(url)
        if not parsed.netloc or not any(
            f".{domain}." in parsed.netloc for domain in self._RAM_DATA_DOMAINS
        ):
            return url
        parts = parsed.netloc.split(".", 1)
        if len(parts) != 2:
            return url
        ram_netloc = parts[0] + "-ram." + parts[1]

        from urllib.parse import urlunparse

        return urlunparse((
            parsed.scheme,
            ram_netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        ))

    def _get_auth_headers(
        self, url: str, config: Optional[Config] = None
    ) -> Dict[str, str]:
        """获取认证请求头，支持 RAM 签名。
        Get authentication headers with RAM signature support.

        Args:
            url: 请求 URL / Request URL
            config: 配置对象 / Configuration object

        Returns:
            Dict[str, str]: 包含认证信息的请求头 / Headers with authentication
        """
        cfg = Config.with_configs(config)
        headers = cfg.get_headers()

        if self._use_ram_auth(cfg):
            # 使用 RAM 端点
            ram_url = self._get_ram_data_endpoint(url, cfg)
            try:
                signed = get_agentrun_signed_headers(
                    url=ram_url,
                    method="GET",
                    access_key_id=cfg.get_access_key_id(),
                    access_key_secret=cfg.get_access_key_secret(),
                    security_token=cfg.get_security_token() or None,
                    region=cfg.get_region_id(),
                    product="agentrun",
                    body=None,
                )
                headers = {
                    **signed,
                    **headers,
                }
                logger.debug(
                    "using RAM signature for skill download to %s",
                    ram_url[:80] + "..." if len(ram_url) > 80 else ram_url,
                )
            except ValueError as e:
                logger.warning("RAM signing skipped (missing AK/SK): %s", e)

        return headers

    def _get_skill_download_url(
        self, config: Optional[Config] = None
    ) -> Optional[str]:
        """获取 Skill 工具的下载 URL / Get download URL for Skill tools

        根据 data_endpoint 和 tool_name 构造下载地址。
        Constructs download URL from data_endpoint and tool_name.

        Returns:
            Optional[str]: 下载 URL / Download URL
        """
        effective_name = self.tool_name or self.name
        data_endpoint = self.data_endpoint
        if not data_endpoint:
            cfg = Config.with_configs(config)
            data_endpoint = cfg.get_data_endpoint()
        if not data_endpoint or not effective_name:
            return None
        return f"{data_endpoint}/tools/{effective_name}/download"

    async def download_skill_async(
        self,
        target_dir: str = ".skills",
        config: Optional[Config] = None,
    ) -> str:
        """异步下载 Skill 包并解压到本地目录 / Download skill package and extract to local directory asynchronously

        从数据链路下载 skill 的 zip 包，并解压到 {target_dir}/{tool_name}/ 目录下。
        Downloads skill zip package from data endpoint and extracts to {target_dir}/{tool_name}/ directory.

        Args:
            target_dir: 目标根目录,默认为 ".skills" / Target root directory, defaults to ".skills"
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            str: 解压后的 skill 目录路径 / Extracted skill directory path

        Raises:
            ValueError: 工具类型不是 SKILL 或缺少必要信息 / Tool type is not SKILL or missing required info
            httpx.HTTPStatusError: 下载失败 / Download failed
        """
        tool_type = self._get_tool_type()
        if tool_type != ToolType.SKILL:
            raise ValueError(
                "download_skill is only available for SKILL type tools,"
                f" got {self.tool_type}"
            )

        download_url = self._get_skill_download_url(config)
        if not download_url:
            raise ValueError(
                "Cannot construct download URL: data_endpoint or tool_name"
                " is missing"
            )

        effective_name = self.tool_name or self.name
        skill_dir = os.path.join(target_dir, effective_name or "unknown_skill")

        logger.debug("downloading skill from %s to %s", download_url, skill_dir)

        cfg = Config.with_configs(config)
        headers = self._get_auth_headers(download_url, cfg)

        async with httpx.AsyncClient(
            timeout=300, follow_redirects=True
        ) as http_client:
            response = await http_client.get(download_url, headers=headers)
            response.raise_for_status()

            if os.path.exists(skill_dir):
                shutil.rmtree(skill_dir)
            os.makedirs(skill_dir, exist_ok=True)

            zip_buffer = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_buffer, "r") as zip_file:
                zip_file.extractall(skill_dir)

        logger.info("skill downloaded and extracted to %s", skill_dir)
        return skill_dir
