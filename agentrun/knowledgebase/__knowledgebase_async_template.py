"""KnowledgeBase 高层 API / KnowledgeBase High-Level API

此模块定义知识库资源的高级API。
This module defines the high-level API for knowledge base resources.
"""

import asyncio
from typing import Any, Dict, List, Optional

from agentrun.utils.config import Config
from agentrun.utils.log import logger
from agentrun.utils.model import PageableInput
from agentrun.utils.resource import ResourceBase

from .api.data import get_data_api
from .model import (
    BailianProviderSettings,
    BailianRetrieveSettings,
    KnowledgeBaseCreateInput,
    KnowledgeBaseImmutableProps,
    KnowledgeBaseListInput,
    KnowledgeBaseListOutput,
    KnowledgeBaseMutableProps,
    KnowledgeBaseProvider,
    KnowledgeBaseSystemProps,
    KnowledgeBaseUpdateInput,
    RagFlowProviderSettings,
    RagFlowRetrieveSettings,
    RetrieveInput,
)


class KnowledgeBase(
    KnowledgeBaseMutableProps,
    KnowledgeBaseImmutableProps,
    KnowledgeBaseSystemProps,
    ResourceBase,
):
    """知识库资源 / KnowledgeBase Resource

    提供知识库的完整生命周期管理,包括创建、删除、更新、查询。
    Provides complete lifecycle management for knowledge bases, including create, delete, update, and query.
    """

    @classmethod
    def __get_client(cls):
        """获取客户端实例 / Get client instance

        Returns:
            KnowledgeBaseClient: 客户端实例 / Client instance
        """
        from .client import KnowledgeBaseClient

        return KnowledgeBaseClient()

    @classmethod
    async def create_async(
        cls, input: KnowledgeBaseCreateInput, config: Optional[Config] = None
    ):
        """创建知识库（异步）/ Create knowledge base asynchronously

        Args:
            input: 知识库输入参数 / KnowledgeBase input parameters
            config: 配置 / Configuration

        Returns:
            KnowledgeBase: 创建的知识库对象 / Created knowledge base object
        """
        return await cls.__get_client().create_async(input, config=config)

    @classmethod
    async def delete_by_name_async(
        cls, knowledge_base_name: str, config: Optional[Config] = None
    ):
        """根据名称删除知识库（异步）/ Delete knowledge base by name asynchronously

        Args:
            knowledge_base_name: 知识库名称 / KnowledgeBase name
            config: 配置 / Configuration
        """
        return await cls.__get_client().delete_async(
            knowledge_base_name, config=config
        )

    @classmethod
    async def update_by_name_async(
        cls,
        knowledge_base_name: str,
        input: KnowledgeBaseUpdateInput,
        config: Optional[Config] = None,
    ):
        """根据名称更新知识库（异步）/ Update knowledge base by name asynchronously

        Args:
            knowledge_base_name: 知识库名称 / KnowledgeBase name
            input: 知识库更新输入参数 / KnowledgeBase update input parameters
            config: 配置 / Configuration

        Returns:
            KnowledgeBase: 更新后的知识库对象 / Updated knowledge base object
        """
        return await cls.__get_client().update_async(
            knowledge_base_name, input, config=config
        )

    @classmethod
    async def get_by_name_async(
        cls, knowledge_base_name: str, config: Optional[Config] = None
    ):
        """根据名称获取知识库（异步）/ Get knowledge base by name asynchronously

        Args:
            knowledge_base_name: 知识库名称 / KnowledgeBase name
            config: 配置 / Configuration

        Returns:
            KnowledgeBase: 知识库对象 / KnowledgeBase object
        """
        return await cls.__get_client().get_async(
            knowledge_base_name, config=config
        )

    @classmethod
    async def _list_page_async(
        cls, page_input: PageableInput, config: Config | None = None, **kwargs
    ):
        return await cls.__get_client().list_async(
            input=KnowledgeBaseListInput(
                **kwargs,
                **page_input.model_dump(),
            ),
            config=config,
        )

    @classmethod
    async def list_all_async(
        cls,
        *,
        provider: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> List[KnowledgeBaseListOutput]:
        """列出所有知识库（异步）/ List all knowledge bases asynchronously

        Args:
            provider: 提供商 / Provider
            config: 配置 / Configuration

        Returns:
            List[KnowledgeBaseListOutput]: 知识库列表 / KnowledgeBase list
        """
        return await cls._list_all_async(
            lambda kb: kb.knowledge_base_id or "",
            config=config,
            provider=provider,
        )

    async def update_async(
        self, input: KnowledgeBaseUpdateInput, config: Optional[Config] = None
    ):
        """更新知识库（异步）/ Update knowledge base asynchronously

        Args:
            input: 知识库更新输入参数 / KnowledgeBase update input parameters
            config: 配置 / Configuration

        Returns:
            KnowledgeBase: 更新后的知识库对象 / Updated knowledge base object
        """
        if self.knowledge_base_name is None:
            raise ValueError(
                "knowledge_base_name is required to update a KnowledgeBase"
            )

        result = await self.update_by_name_async(
            self.knowledge_base_name, input, config=config
        )
        self.update_self(result)

        return self

    async def delete_async(self, config: Optional[Config] = None):
        """删除知识库（异步）/ Delete knowledge base asynchronously

        Args:
            config: 配置 / Configuration
        """
        if self.knowledge_base_name is None:
            raise ValueError(
                "knowledge_base_name is required to delete a KnowledgeBase"
            )

        return await self.delete_by_name_async(
            self.knowledge_base_name, config=config
        )

    async def get_async(self, config: Optional[Config] = None):
        """刷新知识库信息（异步）/ Refresh knowledge base info asynchronously

        Args:
            config: 配置 / Configuration

        Returns:
            KnowledgeBase: 刷新后的知识库对象 / Refreshed knowledge base object
        """
        if self.knowledge_base_name is None:
            raise ValueError(
                "knowledge_base_name is required to refresh a KnowledgeBase"
            )

        result = await self.get_by_name_async(
            self.knowledge_base_name, config=config
        )
        self.update_self(result)

        return self

    async def refresh_async(self, config: Optional[Config] = None):
        """刷新知识库信息（异步）/ Refresh knowledge base info asynchronously

        Args:
            config: 配置 / Configuration

        Returns:
            KnowledgeBase: 刷新后的知识库对象 / Refreshed knowledge base object
        """
        return await self.get_async(config=config)

    # =========================================================================
    # 数据链路方法 / Data API Methods
    # =========================================================================

    def _get_data_api(self, config: Optional[Config] = None):
        """获取数据链路 API 实例 / Get data API instance

        根据当前知识库的 provider 类型返回对应的数据链路 API。
        Returns the corresponding data API based on current knowledge base provider type.

        Args:
            config: 配置 / Configuration

        Returns:
            KnowledgeBaseDataAPI: 数据链路 API 实例 / Data API instance

        Raises:
            ValueError: 如果 provider 未设置 / If provider is not set
        """
        if self.provider is None:
            raise ValueError("provider is required to get data API")

        provider = (
            self.provider
            if isinstance(self.provider, KnowledgeBaseProvider)
            else KnowledgeBaseProvider(self.provider)
        )

        # 转换 provider_settings 和 retrieve_settings 为正确的类型
        # Convert provider_settings and retrieve_settings to correct types
        converted_provider_settings = None
        converted_retrieve_settings = None

        if provider == KnowledgeBaseProvider.BAILIAN:
            # 百炼设置 / Bailian settings
            if self.provider_settings:
                if isinstance(self.provider_settings, BailianProviderSettings):
                    converted_provider_settings = self.provider_settings
                elif isinstance(self.provider_settings, dict):
                    converted_provider_settings = BailianProviderSettings(
                        **self.provider_settings
                    )

            if self.retrieve_settings:
                if isinstance(self.retrieve_settings, BailianRetrieveSettings):
                    converted_retrieve_settings = self.retrieve_settings
                elif isinstance(self.retrieve_settings, dict):
                    converted_retrieve_settings = BailianRetrieveSettings(
                        **self.retrieve_settings
                    )

        elif provider == KnowledgeBaseProvider.RAGFLOW:
            # RagFlow 设置 / RagFlow settings
            if self.provider_settings:
                if isinstance(self.provider_settings, RagFlowProviderSettings):
                    converted_provider_settings = self.provider_settings
                elif isinstance(self.provider_settings, dict):
                    converted_provider_settings = RagFlowProviderSettings(
                        **self.provider_settings
                    )

            if self.retrieve_settings:
                if isinstance(self.retrieve_settings, RagFlowRetrieveSettings):
                    converted_retrieve_settings = self.retrieve_settings
                elif isinstance(self.retrieve_settings, dict):
                    converted_retrieve_settings = RagFlowRetrieveSettings(
                        **self.retrieve_settings
                    )

        return get_data_api(
            provider=provider,
            knowledge_base_name=self.knowledge_base_name or "",
            config=config,
            provider_settings=converted_provider_settings,
            retrieve_settings=converted_retrieve_settings,
            credential_name=self.credential_name,
        )

    async def retrieve_async(
        self,
        query: str,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """检索知识库（异步）/ Retrieve from knowledge base asynchronously

        根据当前知识库的 provider 类型和配置执行检索。
        Executes retrieval based on current knowledge base provider type and configuration.

        Args:
            query: 查询文本 / Query text
            config: 配置 / Configuration

        Returns:
            Dict[str, Any]: 检索结果 / Retrieval results
        """
        data_api = self._get_data_api(config)
        return await data_api.retrieve_async(query, config=config)

    @classmethod
    async def _safe_get_kb_async(
        cls,
        kb_name: str,
        config: Optional[Config] = None,
    ) -> Any:
        """安全获取知识库（异步）/ Safely get knowledge base asynchronously

        Args:
            kb_name: 知识库名称 / Knowledge base name
            config: 配置 / Configuration

        Returns:
            Any: 知识库对象或异常 / Knowledge base object or exception
        """
        try:
            return await cls.get_by_name_async(kb_name, config=config)
        except Exception as e:
            return e

    @classmethod
    async def _safe_retrieve_kb_async(
        cls,
        kb_name: str,
        kb_or_error: Any,
        query: str,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """安全执行知识库检索（异步）/ Safely retrieve from knowledge base asynchronously

        Args:
            kb_name: 知识库名称 / Knowledge base name
            kb_or_error: 知识库对象或异常 / Knowledge base object or exception
            query: 查询文本 / Query text
            config: 配置 / Configuration

        Returns:
            Dict[str, Any]: 检索结果 / Retrieval results
        """
        if isinstance(kb_or_error, Exception):
            logger.warning(
                f"Failed to get knowledge base '{kb_name}': {kb_or_error}"
            )
            return {
                "data": f"Failed to retrieve: {kb_or_error}",
                "query": query,
                "knowledge_base_name": kb_name,
                "error": True,
            }
        try:
            return await kb_or_error.retrieve_async(query, config=config)
        except Exception as e:
            logger.warning(
                f"Failed to retrieve from knowledge base '{kb_name}': {e}"
            )
            return {
                "data": f"Failed to retrieve: {e}",
                "query": query,
                "knowledge_base_name": kb_name,
                "error": True,
            }

    @classmethod
    async def multi_retrieve_async(
        cls,
        query: str,
        knowledge_base_names: List[str],
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """多知识库检索（异步）/ Multi knowledge base retrieval asynchronously

        根据知识库名称列表进行检索，自动获取各知识库的配置并执行检索。
        如果某个知识库查询失败，不影响其他知识库的查询。
        Retrieves from multiple knowledge bases by name list, automatically fetching
        configuration for each knowledge base and executing retrieval.
        If one knowledge base fails, it won't affect other knowledge bases.

        Args:
            query: 查询文本 / Query text
            knowledge_base_names: 知识库名称列表 / List of knowledge base names
            config: 配置 / Configuration

        Returns:
            Dict[str, Any]: 检索结果，按知识库名称分组 / Retrieval results grouped by knowledge base name
        """
        # 1. 根据 knowledge_base_names 并发获取各知识库配置（安全方式）
        #    Fetch all knowledge bases concurrently by name (safely)
        knowledge_base_results = await asyncio.gather(*[
            cls._safe_get_kb_async(name, config=config)
            for name in knowledge_base_names
        ])

        # 2. 并发执行各知识库的检索（安全方式）
        #    Execute retrieval for each knowledge base concurrently (safely)
        retrieve_results = await asyncio.gather(*[
            cls._safe_retrieve_kb_async(
                kb_name, kb_or_error, query, config=config
            )
            for kb_name, kb_or_error in zip(
                knowledge_base_names, knowledge_base_results
            )
        ])

        # 3. 合并返回结果，按知识库名称分组
        #    Merge results, grouped by knowledge base name
        results: Dict[str, Any] = {}
        for kb_name, result in zip(knowledge_base_names, retrieve_results):
            results[kb_name] = result

        return {
            "results": results,
            "query": query,
        }
