"""Credential 模型定义 / Credential Model Definitions

定义凭证相关的数据模型和枚举。
Defines data models and enumerations related to credentials.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from agentrun.utils.config import Config
from agentrun.utils.model import BaseModel, PageableInput


class CredentialAuthType(str, Enum):
    """凭证认证类型 / Credential Authentication Type"""

    JWT = "jwt"
    API_KEY = "api_key"
    BASIC = "basic"
    AKSK = "ak_sk"
    CUSTOM_HEADER = "custom_header"


class CredentialSourceType(str, Enum):
    """凭证来源类型 / Credential Source Type"""

    LLM = "external_llm"
    """模型凭证"""
    TOOL = "external_tool"
    """工具凭证"""
    INTERNAL = "internal"
    """自定义凭证"""


class CredentialBasicAuth(BaseModel):
    username: Optional[str]
    password: Optional[str]


class RelatedResource(BaseModel):
    resource_id: Optional[str] = None
    """资源 ID"""
    resource_name: Optional[str] = None
    """资源名称"""
    resource_type: Optional[str] = None
    """资源类型"""


class CredentialConfigInner(BaseModel):
    credential_auth_type: Optional[CredentialAuthType] = None
    """凭证认证类型"""
    credential_source_type: Optional[CredentialSourceType] = None
    """凭证来源类型"""
    credential_public_config: Optional[Dict[str, Any]] = None
    """凭证公共配置"""
    credential_secret: Optional[str] = None
    """凭证密钥"""


class CredentialConfig(CredentialConfigInner):
    """凭证配置"""

    @classmethod
    def inbound_api_key(cls, api_key: str, header_key: str = "Authorization"):
        """配置访问 AgentRun 的 api key 凭证"""
        return cls(
            credential_source_type=CredentialSourceType.INTERNAL,
            credential_auth_type=CredentialAuthType.API_KEY,
            credential_public_config={"headerKey": header_key},
            credential_secret=api_key,
        )

    @classmethod
    def inbound_static_jwt(cls, jwks: str):
        """配置访问 AgentRun 的静态 JWKS 凭证"""
        return cls(
            credential_source_type=CredentialSourceType.INTERNAL,
            credential_auth_type=CredentialAuthType.JWT,
            credential_public_config={"authType": "static_jwks", "jwks": jwks},
            credential_secret="",
        )

    @classmethod
    def inbound_remote_jwt(
        cls, uri: str, timeout: int = 3000, ttl: int = 30000, **kwargs
    ):
        """配置访问 AgentRun 的远程 JWT 凭证"""
        return cls(
            credential_source_type=CredentialSourceType.INTERNAL,
            credential_auth_type=CredentialAuthType.JWT,
            credential_public_config={
                "uri": uri,
                "timeout": timeout,
                "ttl": ttl,
                **kwargs,
            },
            credential_secret="",
        )

    @classmethod
    def inbound_basic(cls, users: List[CredentialBasicAuth]):
        """配置访问 AgentRun 的 Basic 凭证"""
        return cls(
            credential_source_type=CredentialSourceType.INTERNAL,
            credential_auth_type=CredentialAuthType.BASIC,
            credential_public_config={
                "users": [user.model_dump() for user in users]
            },
            credential_secret="",
        )

    @classmethod
    def outbound_llm_api_key(cls, api_key: str, provider: str):
        """配置访问第三方模型的 api key 凭证"""
        return cls(
            credential_source_type=CredentialSourceType.LLM,
            credential_auth_type=CredentialAuthType.API_KEY,
            credential_public_config={"provider": provider},
            credential_secret=api_key,
        )

    @classmethod
    def outbound_tool_api_key(cls, api_key: str):
        """配置访问第三方工具的 api key 凭证"""
        return cls(
            credential_source_type=CredentialSourceType.TOOL,
            credential_auth_type=CredentialAuthType.API_KEY,
            credential_public_config={},
            credential_secret=api_key,
        )

    @classmethod
    def outbound_tool_ak_sk(
        cls,
        provider: str,
        access_key_id: str,
        access_key_secret: str,
        account_id: str,
    ):
        """配置访问第三方工具的 ak/sk 凭证"""
        return cls(
            credential_source_type=CredentialSourceType.TOOL,
            credential_auth_type=CredentialAuthType.AKSK,
            credential_public_config={
                "provider": provider,
                "authConfig": {
                    "accessKey": access_key_id,
                    "accountId": account_id,
                },
            },
            credential_secret=access_key_secret,
        )

    @classmethod
    def outbound_tool_ak_sk_custom(cls, auth_config: Dict[str, str]):
        """配置访问第三方工具的自定义凭证"""
        return cls(
            credential_source_type=CredentialSourceType.TOOL,
            credential_auth_type=CredentialAuthType.AKSK,
            credential_public_config={
                "provider": "custom",
                "authConfig": auth_config,
            },
            credential_secret="",
        )

    @classmethod
    def outbound_tool_custom_header(cls, headers: Dict[str, str]):
        """配置访问第三方工具的自定义 Header 凭证"""
        return cls(
            credential_source_type=CredentialSourceType.TOOL,
            credential_auth_type=CredentialAuthType.CUSTOM_HEADER,
            credential_public_config={"authConfig": headers},
            credential_secret="",
        )


class CredentialMutableProps(BaseModel):
    """凭证公共配置"""

    # credential_secret: Optional[str] = None
    """凭证密钥"""
    description: Optional[str] = None
    """描述"""
    enabled: Optional[bool] = None
    """是否启用"""


class CredentialImmutableProps(BaseModel):
    credential_name: Optional[str] = None
    """凭证名称"""


class CredentialSystemProps(CredentialConfigInner):
    credential_id: Optional[str] = None
    """凭证 ID"""
    created_at: Optional[str] = None
    """创建时间"""
    updated_at: Optional[str] = None
    """更新时间"""
    related_resources: Optional[List[RelatedResource]] = None
    """关联资源"""


class CredentialCreateInput(CredentialImmutableProps, CredentialMutableProps):
    """凭证创建输入参数"""

    credential_config: CredentialConfig


class CredentialUpdateInput(CredentialMutableProps):
    credential_config: Optional[CredentialConfig] = None


class CredentialListInput(PageableInput):
    credential_auth_type: Optional[CredentialAuthType] = None
    """凭证认证类型"""
    credential_name: Optional[str] = None
    """凭证名称"""
    credential_source_type: Optional[CredentialSourceType] = None
    """凭证来源类型（必填）"""
    provider: Optional[str] = None
    """提供商"""


class CredentialListOutput(BaseModel):
    created_at: Optional[str] = None
    credential_auth_type: Optional[str] = None
    credential_id: Optional[str] = None
    credential_name: Optional[str] = None
    credential_source_type: Optional[str] = None
    enabled: Optional[bool] = None
    related_resource_count: Optional[int] = None
    updated_at: Optional[str] = None

    async def to_credential_async(self, config: Optional[Config] = None):
        from .client import CredentialClient

        return await CredentialClient(config).get_async(
            self.credential_name or "", config=config
        )

    def to_credential(self, config: Optional[Config] = None):
        from .client import CredentialClient

        return CredentialClient(config).get(
            self.credential_name or "", config=config
        )
