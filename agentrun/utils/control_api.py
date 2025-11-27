"""控制 API 基类模块 / Control API Base Module

此模块定义控制链路 API 的基类。
This module defines the base class for control API.
"""

from typing import Optional

from alibabacloud_agentrun20250910.client import Client as AgentRunClient
from alibabacloud_devs20230714.client import Client as DevsClient
from alibabacloud_tea_openapi import utils_models as open_api_util_models

from agentrun.utils.config import Config


class ControlAPI:
    """控制链路客户端基类 / Control API Client Base Class

    提供控制链路和 DevS API 客户端的获取功能。
    Provides functionality to get control API and DevS API clients.
    """

    def __init__(self, config: Optional[Config] = None):
        """初始化 Control API 客户端 / Initialize Control API client

        Args:
            config: 全局配置对象,可选 / Global configuration object, optional
        """
        self.config = config

    def _get_client(self, config: Optional[Config] = None) -> "AgentRunClient":
        """获取 Control API 客户端实例 / Get Control API client instance

        Args:
            config: 配置对象,可选,用于覆盖默认配置 / Configuration object, optional, to override default config

        Returns:
            AgentRunClient: Control API 客户端实例 / Control API client instance
        """

        cfg = Config.with_configs(self.config, config)
        endpoint = cfg.get_control_endpoint()
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            endpoint = endpoint.split("://", 1)[1]
        return AgentRunClient(
            open_api_util_models.Config(
                access_key_id=cfg.get_access_key_id(),
                access_key_secret=cfg.get_access_key_secret(),
                security_token=cfg.get_security_token(),
                region_id=cfg.get_region_id(),
                endpoint=endpoint,
                connect_timeout=cfg.get_timeout(),  # type: ignore
            )
        )

    def _get_devs_client(self, config: Optional[Config] = None) -> "DevsClient":
        """
        获取 Devs API 客户端实例

        Returns:
            DevsClient: Devs API 客户端实例
        """

        cfg = Config.with_configs(self.config, config)
        endpoint = cfg.get_devs_endpoint()
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            endpoint = endpoint.split("://", 1)[1]
        return DevsClient(
            open_api_util_models.Config(
                access_key_id=cfg.get_access_key_id(),
                access_key_secret=cfg.get_access_key_secret(),
                security_token=cfg.get_security_token(),
                region_id=cfg.get_region_id(),
                endpoint=endpoint,
                connect_timeout=cfg.get_timeout(),  # type: ignore
                read_timeout=cfg.get_read_timeout(),  # type: ignore
            )
        )
