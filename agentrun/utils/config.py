"""配置管理模块 / Configuration Management Module

此模块提供 AgentRun SDK 的全局配置管理功能。
This module provides global configuration management for AgentRun SDK.
"""

import os
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()


def get_env_with_default(default: str, *key: str) -> str:
    """从环境变量获取值,支持多个候选键 / Get value from environment variables with multiple fallback keys

    Args:
        default: 默认值 / Default value
        *key: 候选环境变量名 / Candidate environment variable names

    Returns:
        str: 环境变量值或默认值 / Environment variable value or default value
    """
    for k in key:
        v = os.getenv(k)
        if v is not None:
            return v
    return default


class Config:
    """AgentRun SDK 全局配置类 / AgentRun SDK Global Configuration Class

    用于管理账号凭证和客户端配置。
    Used for managing account credentials and client configuration.

    支持从参数或环境变量读取配置。
    Supports reading configuration from parameters or environment variables.

    Examples:
        >>> # 从参数创建配置 / Create config from parameters
        >>> config = Config(
        ...     account_id="your-account-id",
        ...     access_key_id="your-key-id",
        ...     access_key_secret="your-secret"
        ... )
        >>> # 或从环境变量读取 / Or read from environment variables
        >>> config = Config()
    """

    __slots__ = (
        "_access_key_id",
        "_access_key_secret",
        "_security_token",
        "_account_id",
        "_token",
        "_region_id",
        "_timeout",
        "_read_timeout",
        "_control_endpoint",
        "_data_endpoint",
        "_devs_endpoint",
        "_headers",
        "__weakref__",
    )

    def __init__(
        self,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        security_token: Optional[str] = None,
        account_id: Optional[str] = None,
        token: Optional[str] = None,
        region_id: Optional[str] = None,
        timeout: Optional[int] = 600,
        read_timeout: Optional[int] = 100000,
        control_endpoint: Optional[str] = None,
        data_endpoint: Optional[str] = None,
        devs_endpoint: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """初始化配置 / Initialize configuration

        Args:
            access_key_id: Access Key ID / Access Key ID
                未提供时从环境变量读取: AGENTRUN_ACCESS_KEY_ID 或 ALIBABA_CLOUD_ACCESS_KEY_ID
                Read from env vars if not provided: AGENTRUN_ACCESS_KEY_ID or ALIBABA_CLOUD_ACCESS_KEY_ID
            access_key_secret: Access Key Secret / Access Key Secret
                未提供时从环境变量读取: AGENTRUN_ACCESS_KEY_SECRET 或 ALIBABA_CLOUD_ACCESS_KEY_SECRET
                Read from env vars if not provided: AGENTRUN_ACCESS_KEY_SECRET or ALIBABA_CLOUD_ACCESS_KEY_SECRET
            security_token: 安全令牌 / Security token
                未提供时从环境变量读取: AGENTRUN_SECURITY_TOKEN 或 ALIBABA_CLOUD_SECURITY_TOKEN
                Read from env vars if not provided: AGENTRUN_SECURITY_TOKEN or ALIBABA_CLOUD_SECURITY_TOKEN
            account_id: 账号 ID / Account ID
                未提供时从环境变量读取: AGENTRUN_ACCOUNT_ID 或 FC_ACCOUNT_ID
                Read from env vars if not provided: AGENTRUN_ACCOUNT_ID or FC_ACCOUNT_ID
            token: 自定义令牌,用于数据链路调用 / Custom token for data API calls
            region_id: 区域 ID,默认 cn-hangzhou / Region ID, defaults to cn-hangzhou
            timeout: 请求超时时间(秒),默认 600 / Request timeout in seconds, defaults to 600
            read_timeout: 读取超时时间(秒),默认 100000 / Read timeout in seconds, defaults to 100000
            control_endpoint: 自定义控制链路端点,可选 / Custom control endpoint, optional
            data_endpoint: 自定义数据链路端点,可选 / Custom data endpoint, optional
            devs_endpoint: 自定义 DevS 端点,可选 / Custom DevS endpoint, optional
            headers: 自定义请求头,可选 / Custom request headers, optional
        """

        if access_key_id is None:
            access_key_id = get_env_with_default(
                "", "AGENTRUN_ACCESS_KEY_ID", "ALIBABA_CLOUD_ACCESS_KEY_ID"
            )
        if access_key_secret is None:
            access_key_secret = get_env_with_default(
                "",
                "AGENTRUN_ACCESS_KEY_SECRET",
                "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
            )
        if security_token is None:
            security_token = get_env_with_default(
                "", "AGENTRUN_SECURITY_TOKEN", "ALIBABA_CLOUD_SECURITY_TOKEN"
            )
        if account_id is None:
            account_id = get_env_with_default(
                "", "AGENTRUN_ACCOUNT_ID", "FC_ACCOUNT_ID"
            )
        if region_id is None:
            region_id = get_env_with_default(
                "cn-hangzhou", "AGENTRUN_REGION", "FC_REGION"
            )
        if control_endpoint is None:
            control_endpoint = get_env_with_default(
                "", "AGENTRUN_CONTROL_ENDPOINT"
            )
        if data_endpoint is None:
            data_endpoint = get_env_with_default("", "AGENTRUN_DATA_ENDPOINT")
        if devs_endpoint is None:
            devs_endpoint = get_env_with_default("", "DEVS_ENDPOINT")

        self._access_key_id = access_key_id
        self._access_key_secret = access_key_secret
        self._security_token = security_token
        self._account_id = account_id
        self._token = token
        self._region_id = region_id
        self._timeout = timeout
        self._read_timeout = read_timeout
        self._control_endpoint = control_endpoint
        self._data_endpoint = data_endpoint
        self._devs_endpoint = devs_endpoint
        self._headers = headers or {}

    @classmethod
    def with_configs(cls, *configs: Optional["Config"]) -> "Config":
        return cls().update(*configs)

    def update(self, *configs: Optional["Config"]) -> "Config":
        """
        使用给定的配置对象,返回新的实例,优先使用靠后的值

        Args:
            configs: 要合并的配置对象

        Returns:
            合并后的新配置对象
        """

        for config in configs:
            if config is None:
                continue

            for attr in filter(
                lambda x: x != "__weakref__",
                self.__slots__,
            ):
                value = getattr(config, attr)
                if value is not None:
                    if type(value) is dict:
                        getattr(self, attr).update(getattr(config, attr) or {})
                    else:
                        setattr(self, attr, value)

        return self

    def __repr__(self) -> str:

        return "Config{%s}" % (
            ", ".join([
                f'"{key}": "{getattr(self, key)}"'
                for key in self.__slots__
                if key != "__weakref__"
            ])
        )

    def get_access_key_id(self) -> str:
        """获取 Access Key ID"""
        return self._access_key_id

    def get_access_key_secret(self) -> str:
        """获取 Access Key Secret"""
        return self._access_key_secret

    def get_security_token(self) -> str:
        """获取安全令牌"""
        return self._security_token

    def get_account_id(self) -> str:
        """获取账号 ID"""
        if not self._account_id:
            raise ValueError(
                "account id is not set, please add AGENTRUN_ACCOUNT_ID env"
                " variable or set it in code."
            )

        return self._account_id

    def get_token(self) -> Optional[str]:
        """获取自定义令牌"""
        return self._token

    def get_region_id(self) -> str:
        """获取区域 ID"""
        if self._region_id:
            return self._region_id

        return "cn-hangzhou"

    def get_timeout(self) -> Optional[int]:
        """获取请求超时时间"""
        return self._timeout or 600

    def get_read_timeout(self) -> Optional[int]:
        """获取请求超时时间"""
        return self._read_timeout or 100000

    def get_control_endpoint(self) -> str:
        """获取控制链路端点"""
        if self._control_endpoint:
            return self._control_endpoint

        return f"https://agentrun.{self.get_region_id()}.aliyuncs.com"

    def get_data_endpoint(self) -> str:
        """获取数据链路端点"""
        if self._data_endpoint:
            return self._data_endpoint

        return f"https://{self.get_account_id()}.agentrun-data.{self.get_region_id()}.aliyuncs.com"

    def get_devs_endpoint(self) -> str:
        """获取 Devs 端点"""
        if self._devs_endpoint:
            return self._devs_endpoint

        return f"https://devs.{self.get_region_id()}.aliyuncs.com"

    def get_headers(self) -> Dict[str, str]:
        """获取自定义请求头"""
        return self._headers or {}
