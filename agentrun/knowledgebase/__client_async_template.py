"""KnowledgeBase 客户端 / KnowledgeBase Client

此模块提供知识库管理的客户端API。
This module provides the client API for knowledge base management.
"""

from typing import Optional

from alibabacloud_agentrun20250910.models import (
    CreateKnowledgeBaseInput,
    ListKnowledgeBasesRequest,
    UpdateKnowledgeBaseInput,
)

from agentrun.utils.config import Config
from agentrun.utils.exception import HTTPError

from .api.control import KnowledgeBaseControlAPI
from .knowledgebase import KnowledgeBase
from .model import (
    KnowledgeBaseCreateInput,
    KnowledgeBaseListInput,
    KnowledgeBaseListOutput,
    KnowledgeBaseUpdateInput,
)


class KnowledgeBaseClient:
    """KnowledgeBase 客户端 / KnowledgeBase Client

    提供知识库的创建、删除、更新和查询功能。
    Provides create, delete, update and query functions for knowledge bases.
    """

    def __init__(self, config: Optional[Config] = None):
        """初始化客户端 / Initialize client

        Args:
            config: 配置对象,可选 / Configuration object, optional
        """
        self.__control_api = KnowledgeBaseControlAPI(config)

    async def create_async(
        self, input: KnowledgeBaseCreateInput, config: Optional[Config] = None
    ):
        """创建知识库(异步) / Create knowledge base asynchronously

        Args:
            input: 知识库输入参数 / KnowledgeBase input parameters
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            KnowledgeBase: 创建的知识库对象 / Created knowledge base object

        Raises:
            ResourceAlreadyExistError: 资源已存在 / Resource already exists
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        try:
            result = await self.__control_api.create_knowledge_base_async(
                CreateKnowledgeBaseInput().from_map(input.model_dump()),
                config=config,
            )

            return KnowledgeBase.from_inner_object(result)
        except HTTPError as e:
            raise e.to_resource_error(
                "KnowledgeBase", input.knowledge_base_name
            ) from e

    async def delete_async(
        self, knowledge_base_name: str, config: Optional[Config] = None
    ):
        """删除知识库（异步）/ Delete knowledge base asynchronously

        Args:
            knowledge_base_name: 知识库名称 / KnowledgeBase name
            config: 配置 / Configuration

        Raises:
            ResourceNotExistError: 知识库不存在 / KnowledgeBase not found
        """
        try:
            result = await self.__control_api.delete_knowledge_base_async(
                knowledge_base_name, config=config
            )

            return KnowledgeBase.from_inner_object(result)

        except HTTPError as e:
            raise e.to_resource_error(
                "KnowledgeBase", knowledge_base_name
            ) from e

    async def update_async(
        self,
        knowledge_base_name: str,
        input: KnowledgeBaseUpdateInput,
        config: Optional[Config] = None,
    ):
        """更新知识库（异步）/ Update knowledge base asynchronously

        Args:
            knowledge_base_name: 知识库名称 / KnowledgeBase name
            input: 知识库更新输入参数 / KnowledgeBase update input parameters
            config: 配置 / Configuration

        Returns:
            KnowledgeBase: 更新后的知识库对象 / Updated knowledge base object

        Raises:
            ResourceNotExistError: 知识库不存在 / KnowledgeBase not found
        """
        try:
            result = await self.__control_api.update_knowledge_base_async(
                knowledge_base_name,
                UpdateKnowledgeBaseInput().from_map(input.model_dump()),
                config=config,
            )

            return KnowledgeBase.from_inner_object(result)
        except HTTPError as e:
            raise e.to_resource_error(
                "KnowledgeBase", knowledge_base_name
            ) from e

    async def get_async(
        self, knowledge_base_name: str, config: Optional[Config] = None
    ):
        """获取知识库（异步）/ Get knowledge base asynchronously

        Args:
            knowledge_base_name: 知识库名称 / KnowledgeBase name
            config: 配置 / Configuration

        Returns:
            KnowledgeBase: 知识库对象 / KnowledgeBase object

        Raises:
            ResourceNotExistError: 知识库不存在 / KnowledgeBase not found
        """
        try:
            result = await self.__control_api.get_knowledge_base_async(
                knowledge_base_name, config=config
            )
            return KnowledgeBase.from_inner_object(result)
        except HTTPError as e:
            raise e.to_resource_error(
                "KnowledgeBase", knowledge_base_name
            ) from e

    async def list_async(
        self,
        input: Optional[KnowledgeBaseListInput] = None,
        config: Optional[Config] = None,
    ):
        """列出知识库（异步）/ List knowledge bases asynchronously

        Args:
            input: 分页查询参数 / Pagination query parameters
            config: 配置 / Configuration

        Returns:
            List[KnowledgeBaseListOutput]: 知识库列表 / KnowledgeBase list
        """
        if input is None:
            input = KnowledgeBaseListInput()

        results = await self.__control_api.list_knowledge_bases_async(
            ListKnowledgeBasesRequest().from_map(input.model_dump()),
            config=config,
        )
        return [KnowledgeBaseListOutput.from_inner_object(item) for item in results.items]  # type: ignore
