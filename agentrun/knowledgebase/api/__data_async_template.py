"""KnowledgeBase 数据链路 API / KnowledgeBase Data API

提供知识库检索功能的数据链路 API。
Provides data API for knowledge base retrieval operations.

根据不同的 provider 类型（ragflow / bailian）分发到不同的实现。
Dispatches to different implementations based on provider type (ragflow / bailian).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from alibabacloud_bailian20231229 import models as bailian_models
import httpx

from agentrun.utils.config import Config
from agentrun.utils.control_api import ControlAPI
from agentrun.utils.data_api import DataAPI, ResourceType
from agentrun.utils.log import logger

from ..model import (
    BailianProviderSettings,
    BailianRetrieveSettings,
    KnowledgeBaseProvider,
    RagFlowProviderSettings,
    RagFlowRetrieveSettings,
)


class KnowledgeBaseDataAPI(ABC):
    """知识库数据链路 API 基类 / KnowledgeBase Data API Base Class

    定义知识库检索的抽象接口，由具体的 provider 实现。
    Defines abstract interface for knowledge base retrieval, implemented by specific providers.
    """

    def __init__(
        self,
        knowledge_base_name: str,
        config: Optional[Config] = None,
    ):
        """初始化知识库数据链路 API / Initialize KnowledgeBase Data API

        Args:
            knowledge_base_name: 知识库名称 / Knowledge base name
            config: 配置 / Configuration
        """
        self.knowledge_base_name = knowledge_base_name
        self.config = Config.with_configs(config)

    @abstractmethod
    async def retrieve_async(
        self,
        query: str,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """检索知识库（异步）/ Retrieve from knowledge base (async)

        Args:
            query: 查询文本 / Query text
            config: 配置 / Configuration

        Returns:
            Dict[str, Any]: 检索结果 / Retrieval results
        """
        raise NotImplementedError("Subclasses must implement retrieve_async")


class RagFlowDataAPI(KnowledgeBaseDataAPI):
    """RagFlow 知识库数据链路 API / RagFlow KnowledgeBase Data API

    实现 RagFlow 知识库的检索逻辑。
    Implements retrieval logic for RagFlow knowledge base.
    """

    def __init__(
        self,
        knowledge_base_name: str,
        config: Optional[Config] = None,
        provider_settings: Optional[RagFlowProviderSettings] = None,
        retrieve_settings: Optional[RagFlowRetrieveSettings] = None,
        credential_name: Optional[str] = None,
    ):
        """初始化 RagFlow 知识库数据链路 API / Initialize RagFlow KnowledgeBase Data API

        Args:
            knowledge_base_name: 知识库名称 / Knowledge base name
            config: 配置 / Configuration
            provider_settings: RagFlow 提供商设置 / RagFlow provider settings
            retrieve_settings: RagFlow 检索设置 / RagFlow retrieve settings
            credential_name: 凭证名称 / Credential name
        """
        super().__init__(knowledge_base_name, config)
        self.provider_settings = provider_settings
        self.retrieve_settings = retrieve_settings
        self.credential_name = credential_name

    async def _get_api_key_async(self, config: Optional[Config] = None) -> str:
        """获取 API Key（异步）/ Get API Key (async)

        Args:
            config: 配置 / Configuration

        Returns:
            str: API Key

        Raises:
            ValueError: 凭证名称未设置或凭证不存在 / Credential name not set or credential not found
        """
        if not self.credential_name:
            raise ValueError(
                "credential_name is required for RagFlow retrieval"
            )

        from agentrun.credential import Credential

        credential = await Credential.get_by_name_async(
            self.credential_name, config=config
        )
        if not credential.credential_secret:
            raise ValueError(
                f"Credential '{self.credential_name}' has no secret configured"
            )
        return credential.credential_secret

    def _build_request_body(self, query: str) -> Dict[str, Any]:
        """构建请求体 / Build request body

        Args:
            query: 查询文本 / Query text

        Returns:
            Dict[str, Any]: 请求体 / Request body
        """
        if self.provider_settings is None:
            raise ValueError(
                "provider_settings is required for RagFlow retrieval"
            )

        body: Dict[str, Any] = {
            "question": query,
            "dataset_ids": self.provider_settings.dataset_ids,
            "page": 1,
            "page_size": 30,
        }

        # 添加检索设置 / Add retrieve settings
        if self.retrieve_settings:
            if self.retrieve_settings.similarity_threshold is not None:
                body["similarity_threshold"] = (
                    self.retrieve_settings.similarity_threshold
                )
            if self.retrieve_settings.vector_similarity_weight is not None:
                body["vector_similarity_weight"] = (
                    self.retrieve_settings.vector_similarity_weight
                )
            if self.retrieve_settings.cross_languages is not None:
                body["cross_languages"] = self.retrieve_settings.cross_languages

        return body

    async def retrieve_async(
        self,
        query: str,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """RagFlow 检索（异步）/ RagFlow retrieval (async)

        Args:
            query: 查询文本 / Query text
            config: 配置 / Configuration

        Returns:
            Dict[str, Any]: 检索结果 / Retrieval results
        """
        try:
            if self.provider_settings is None:
                raise ValueError(
                    "provider_settings is required for RagFlow retrieval"
                )

            # 获取 API Key / Get API Key
            api_key = await self._get_api_key_async(config)

            # 构建请求 / Build request
            base_url = self.provider_settings.base_url.rstrip("/")
            url = f"{base_url}/api/v1/retrieval"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            body = self._build_request_body(query)

            # 发送请求 / Send request
            async with httpx.AsyncClient(
                timeout=self.config.get_timeout()
            ) as client:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                result = response.json()
                logger.debug(f"RagFlow retrieval result: {result}")

            # 返回结果 / Return result
            data = result.get("data", {})

            if data == False:
                raise Exception(f"RagFlow retrieval failed: {result}")

            return {
                "data": data,
                "query": query,
                "knowledge_base_name": self.knowledge_base_name,
            }
        except Exception as e:
            logger.warning(
                "Failed to retrieve from RagFlow knowledge base "
                f"'{self.knowledge_base_name}': {e}"
            )
            return {
                "data": f"Failed to retrieve: {e}",
                "query": query,
                "knowledge_base_name": self.knowledge_base_name,
                "error": True,
            }


class BailianDataAPI(KnowledgeBaseDataAPI, ControlAPI):
    """百炼知识库数据链路 API / Bailian KnowledgeBase Data API

    实现百炼知识库的检索逻辑。
    Implements retrieval logic for Bailian knowledge base.
    """

    def __init__(
        self,
        knowledge_base_name: str,
        config: Optional[Config] = None,
        provider_settings: Optional[BailianProviderSettings] = None,
        retrieve_settings: Optional[BailianRetrieveSettings] = None,
    ):
        """初始化百炼知识库数据链路 API / Initialize Bailian KnowledgeBase Data API

        Args:
            knowledge_base_name: 知识库名称 / Knowledge base name
            config: 配置 / Configuration
            provider_settings: 百炼提供商设置 / Bailian provider settings
            retrieve_settings: 百炼检索设置 / Bailian retrieve settings
        """
        KnowledgeBaseDataAPI.__init__(self, knowledge_base_name, config)
        ControlAPI.__init__(self, config)
        self.provider_settings = provider_settings
        self.retrieve_settings = retrieve_settings

    async def retrieve_async(
        self,
        query: str,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """百炼检索（异步）/ Bailian retrieval (async)

        Args:
            query: 查询文本 / Query text
            config: 配置 / Configuration

        Returns:
            Dict[str, Any]: 检索结果 / Retrieval results
        """
        try:
            if self.provider_settings is None:
                raise ValueError(
                    "provider_settings is required for Bailian retrieval"
                )

            workspace_id = self.provider_settings.workspace_id
            index_ids = self.provider_settings.index_ids

            # 构建检索请求 / Build retrieve request
            request_params: Dict[str, Any] = {
                "query": query,
            }

            # 添加检索设置 / Add retrieve settings
            if self.retrieve_settings:
                if self.retrieve_settings.dense_similarity_top_k is not None:
                    request_params["dense_similarity_top_k"] = (
                        self.retrieve_settings.dense_similarity_top_k
                    )
                if self.retrieve_settings.sparse_similarity_top_k is not None:
                    request_params["sparse_similarity_top_k"] = (
                        self.retrieve_settings.sparse_similarity_top_k
                    )
                if self.retrieve_settings.rerank_min_score is not None:
                    request_params["rerank_min_score"] = (
                        self.retrieve_settings.rerank_min_score
                    )
                if self.retrieve_settings.rerank_top_n is not None:
                    request_params["rerank_top_n"] = (
                        self.retrieve_settings.rerank_top_n
                    )

            # 获取百炼客户端 / Get Bailian client
            client = self._get_bailian_client(config)

            # 对每个 index_id 进行检索并合并结果 / Retrieve from each index and merge results
            all_nodes: List[Dict[str, Any]] = []
            for index_id in index_ids:
                request_params["index_id"] = index_id
                request = bailian_models.RetrieveRequest(**request_params)
                response = await client.retrieve_async(workspace_id, request)
                logger.debug(f"Bailian retrieve response: {response}")

                if (
                    response.body
                    and response.body.data
                    and response.body.data.nodes
                ):
                    for node in response.body.data.nodes:
                        all_nodes.append({
                            "text": (
                                node.text if hasattr(node, "text") else None
                            ),
                            "score": (
                                node.score if hasattr(node, "score") else None
                            ),
                            "metadata": (
                                node.metadata
                                if hasattr(node, "metadata")
                                else None
                            ),
                        })

            return {
                "data": all_nodes,
                "query": query,
                "knowledge_base_name": self.knowledge_base_name,
            }
        except Exception as e:
            logger.warning(
                "Failed to retrieve from Bailian knowledge base "
                f"'{self.knowledge_base_name}': {e}"
            )
            return {
                "data": f"Failed to retrieve: {e}",
                "query": query,
                "knowledge_base_name": self.knowledge_base_name,
                "error": True,
            }


def get_data_api(
    provider: KnowledgeBaseProvider,
    knowledge_base_name: str,
    config: Optional[Config] = None,
    provider_settings: Optional[
        Union[RagFlowProviderSettings, BailianProviderSettings]
    ] = None,
    retrieve_settings: Optional[
        Union[RagFlowRetrieveSettings, BailianRetrieveSettings]
    ] = None,
    credential_name: Optional[str] = None,
) -> KnowledgeBaseDataAPI:
    """根据 provider 类型获取对应的数据链路 API / Get data API by provider type

    Args:
        provider: 提供商类型 / Provider type
        knowledge_base_name: 知识库名称 / Knowledge base name
        config: 配置 / Configuration
        provider_settings: 提供商设置 / Provider settings
        retrieve_settings: 检索设置 / Retrieve settings
        credential_name: 凭证名称（RagFlow 需要）/ Credential name (required for RagFlow)

    Returns:
        KnowledgeBaseDataAPI: 对应的数据链路 API 实例 / Corresponding data API instance

    Raises:
        ValueError: 不支持的 provider 类型 / Unsupported provider type
    """
    if provider == KnowledgeBaseProvider.RAGFLOW or provider == "ragflow":
        ragflow_provider_settings = (
            provider_settings
            if isinstance(provider_settings, RagFlowProviderSettings)
            else None
        )
        ragflow_retrieve_settings = (
            retrieve_settings
            if isinstance(retrieve_settings, RagFlowRetrieveSettings)
            else None
        )
        return RagFlowDataAPI(
            knowledge_base_name,
            config,
            provider_settings=ragflow_provider_settings,
            retrieve_settings=ragflow_retrieve_settings,
            credential_name=credential_name,
        )
    elif provider == KnowledgeBaseProvider.BAILIAN or provider == "bailian":
        bailian_provider_settings = (
            provider_settings
            if isinstance(provider_settings, BailianProviderSettings)
            else None
        )
        bailian_retrieve_settings = (
            retrieve_settings
            if isinstance(retrieve_settings, BailianRetrieveSettings)
            else None
        )
        return BailianDataAPI(
            knowledge_base_name,
            config,
            provider_settings=bailian_provider_settings,
            retrieve_settings=bailian_retrieve_settings,
        )
    else:
        raise ValueError(f"Unsupported provider type: {provider}")
