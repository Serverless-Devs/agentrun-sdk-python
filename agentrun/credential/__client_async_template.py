"""Credential 客户端 / Credential Client

此模块提供凭证管理的客户端API。
This module provides the client API for credential management.
"""

from typing import Optional

from alibabacloud_agentrun20250910.models import (
    CreateCredentialInput,
    ListCredentialsRequest,
    UpdateCredentialInput,
)

from agentrun.utils.config import Config
from agentrun.utils.exception import HTTPError

from .api.control import CredentialControlAPI
from .credential import Credential
from .model import (
    CredentialCreateInput,
    CredentialListInput,
    CredentialListOutput,
    CredentialUpdateInput,
)


class CredentialClient:
    """Credential 客户端 / Credential Client

    提供凭证的创建、删除、更新和查询功能。
    Provides create, delete, update and query functions for credentials.
    """

    def __init__(self, config: Optional[Config] = None):
        """初始化客户端 / Initialize client

        Args:
            config: 配置对象,可选 / Configuration object, optional
        """
        self.__control_api = CredentialControlAPI(config)

    async def create_async(
        self, input: CredentialCreateInput, config: Optional[Config] = None
    ):
        """创建凭证(异步) / Create credential asynchronously

        Args:
            input: 凭证输入参数 / Credential input parameters
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            Credential: 创建的凭证对象 / Created credential object

        Raises:
            ResourceAlreadyExistError: 资源已存在 / Resource already exists
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        try:
            result = await self.__control_api.create_credential_async(
                CreateCredentialInput().from_map({
                    **input.model_dump(),
                    **input.credential_config.model_dump(),
                }),
                config=config,
            )

            return Credential.from_inner_object(result)
        except HTTPError as e:
            raise e.to_resource_error(
                "Credential", input.credential_name
            ) from e

    async def delete_async(
        self, credential_name: str, config: Optional[Config] = None
    ):
        """删除凭证（异步）

        Args:
            credential_name: 凭证名称
            config: 配置

        Raises:
            ResourceNotExistError: 凭证不存在
        """
        try:
            result = await self.__control_api.delete_credential_async(
                credential_name, config=config
            )

            return Credential.from_inner_object(result)

        except HTTPError as e:
            raise e.to_resource_error("Credential", credential_name) from e

    async def update_async(
        self,
        credential_name: str,
        input: CredentialUpdateInput,
        config: Optional[Config] = None,
    ):
        """更新凭证（异步）

        Args:
            credential_name: 凭证名称
            input: 凭证更新输入参数
            config: 配置

        Returns:
            Credential: 更新后的凭证对象

        Raises:
            ResourceNotExistError: 凭证不存在
        """
        try:
            result = await self.__control_api.update_credential_async(
                credential_name,
                UpdateCredentialInput().from_map({
                    **input.model_dump(),
                    **(
                        input.credential_config.model_dump()
                        if input.credential_config
                        else {}
                    ),
                }),
                config=config,
            )

            return Credential.from_inner_object(result)
        except HTTPError as e:
            raise e.to_resource_error("Credential", credential_name) from e

    async def get_async(
        self, credential_name: str, config: Optional[Config] = None
    ):
        """获取凭证（异步）

        Args:
            credential_name: 凭证名称
            config: 配置

        Returns:
            Credential: 凭证对象

        Raises:
            ResourceNotExistError: 凭证不存在
        """
        try:
            result = await self.__control_api.get_credential_async(
                credential_name, config=config
            )
            return Credential.from_inner_object(result)
        except HTTPError as e:
            raise e.to_resource_error("Credential", credential_name) from e

    async def list_async(
        self,
        input: Optional[CredentialListInput] = None,
        config: Optional[Config] = None,
    ):
        """列出凭证（异步）

        Args:
            input: 分页查询参数
            config: 配置

        Returns:
            List[Credential]: 凭证列表
        """
        if input is None:
            input = CredentialListInput()

        results = await self.__control_api.list_credentials_async(
            ListCredentialsRequest().from_map(input.model_dump()),
            config=config,
        )
        return [CredentialListOutput.from_inner_object(item) for item in results.items]  # type: ignore
