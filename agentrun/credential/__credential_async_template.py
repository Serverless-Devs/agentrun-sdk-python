"""Credential 高层 API / Credential High-Level API

此模块定义凭证资源的高级API。
This module defines the high-level API for credential resources.
"""

from typing import List, Optional

from agentrun.utils.config import Config
from agentrun.utils.model import PageableInput
from agentrun.utils.resource import ResourceBase

from .model import (
    CredentialAuthType,
    CredentialCreateInput,
    CredentialImmutableProps,
    CredentialListInput,
    CredentialListOutput,
    CredentialMutableProps,
    CredentialSourceType,
    CredentialSystemProps,
    CredentialUpdateInput,
)


class Credential(
    CredentialMutableProps,
    CredentialImmutableProps,
    CredentialSystemProps,
    ResourceBase,
):
    """凭证资源 / Credential Resource

    提供凭证的完整生命周期管理,包括创建、删除、更新、查询。
    Provides complete lifecycle management for credentials, including create, delete, update, and query.
    """

    @classmethod
    def __get_client(cls):
        """获取客户端实例 / Get client instance

        Returns:
            CredentialClient: 客户端实例 / Client instance
        """
        from .client import CredentialClient

        return CredentialClient()

    @classmethod
    async def create_async(
        cls, input: CredentialCreateInput, config: Optional[Config] = None
    ):
        """创建凭证（异步）

        Args:
            input: 凭证输入参数
            config: 配置

        Returns:
            Credential: 创建的凭证对象
        """
        return await cls.__get_client().create_async(input, config=config)

    @classmethod
    async def delete_by_name_async(
        cls, credential_name: str, config: Optional[Config] = None
    ):
        """根据名称删除凭证（异步）

        Args:
            credential_name: 凭证名称
            config: 配置
        """
        return await cls.__get_client().delete_async(
            credential_name, config=config
        )

    @classmethod
    async def update_by_name_async(
        cls,
        credential_name: str,
        input: CredentialUpdateInput,
        config: Optional[Config] = None,
    ):
        """根据名称更新凭证（异步）

        Args:
            credential_name: 凭证名称
            input: 凭证更新输入参数
            config: 配置

        Returns:
            Credential: 更新后的凭证对象
        """
        return await cls.__get_client().update_async(
            credential_name, input, config=config
        )

    @classmethod
    async def get_by_name_async(
        cls, credential_name: str, config: Optional[Config] = None
    ):
        """根据名称获取凭证（异步）

        Args:
            credential_name: 凭证名称
            config: 配置

        Returns:
            Credential: 凭证对象
        """
        return await cls.__get_client().get_async(
            credential_name, config=config
        )

    @classmethod
    async def _list_page_async(
        cls, page_input: PageableInput, config: Config | None = None, **kwargs
    ):
        return await cls.__get_client().list_async(
            input=CredentialListInput(
                **kwargs,
                **page_input.model_dump(),
            ),
            config=config,
        )

    @classmethod
    async def list_all_async(
        cls,
        *,
        credential_auth_type: Optional[CredentialAuthType] = None,
        credential_name: Optional[str] = None,
        credential_source_type: Optional[CredentialSourceType] = None,
        provider: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> List[CredentialListOutput]:
        return await cls._list_all_async(
            lambda cred: cred.credential_id or "",
            config=config,
            credential_auth_type=credential_auth_type,
            credential_name=credential_name,
            credential_source_type=credential_source_type,
            provider=provider,
        )

    async def update_async(
        self, input: CredentialUpdateInput, config: Optional[Config] = None
    ):
        """更新凭证（异步）

        Args:
            input: 凭证更新输入参数
            config: 配置

        Returns:
            Credential: 更新后的凭证对象
        """
        if self.credential_name is None:
            raise ValueError(
                "credential_name is required to update a Credential"
            )

        result = await self.update_by_name_async(
            self.credential_name, input, config=config
        )
        self.update_self(result)

        return self

    async def delete_async(self, config: Optional[Config] = None):
        """删除凭证（异步）

        Args:
            config: 配置
        """
        if self.credential_name is None:
            raise ValueError(
                "credential_name is required to delete a Credential"
            )

        return await self.delete_by_name_async(
            self.credential_name, config=config
        )

    async def get_async(self, config: Optional[Config] = None):
        """刷新凭证信息（异步）

        Args:
            config: 配置

        Returns:
            Credential: 刷新后的凭证对象
        """
        if self.credential_name is None:
            raise ValueError(
                "credential_name is required to refresh a Credential"
            )

        result = await self.get_by_name_async(
            self.credential_name, config=config
        )
        self.update_self(result)

        return self

    async def refresh_async(self, config: Optional[Config] = None):
        """刷新凭证信息（异步）

        Args:
            config: 配置

        Returns:
            Credential: 刷新后的凭证对象
        """
        return await self.get_async(config=config)
