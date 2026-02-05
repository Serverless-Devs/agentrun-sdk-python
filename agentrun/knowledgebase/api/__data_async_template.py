"""KnowledgeBase 数据链路 API / KnowledgeBase Data API

提供知识库检索功能的数据链路 API。
Provides data API for knowledge base retrieval operations.

根据不同的 provider 类型（ragflow / bailian / adb）分发到不同的实现。
Dispatches to different implementations based on provider type (ragflow / bailian / adb).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from alibabacloud_bailian20231229 import models as bailian_models
from alibabacloud_gpdb20160503 import models as gpdb_models
import httpx

from agentrun.utils.config import Config
from agentrun.utils.control_api import ControlAPI
from agentrun.utils.data_api import DataAPI, ResourceType
from agentrun.utils.log import logger

from ..model import (
    ADBProviderSettings,
    ADBRetrieveSettings,
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


class ADBDataAPI(KnowledgeBaseDataAPI, ControlAPI):
    """ADB (AnalyticDB for PostgreSQL) 知识库数据链路 API / ADB KnowledgeBase Data API

    实现 ADB 知识库的检索逻辑，通过 GPDB SDK 调用 QueryContent 接口。
    Implements retrieval logic for ADB knowledge base via GPDB SDK QueryContent API.
    """

    def __init__(
        self,
        knowledge_base_name: str,
        config: Optional[Config] = None,
        provider_settings: Optional[ADBProviderSettings] = None,
        retrieve_settings: Optional[ADBRetrieveSettings] = None,
    ):
        """初始化 ADB 知识库数据链路 API / Initialize ADB KnowledgeBase Data API

        Args:
            knowledge_base_name: 知识库名称 / Knowledge base name
            config: 配置 / Configuration
            provider_settings: ADB 提供商设置 / ADB provider settings
            retrieve_settings: ADB 检索设置 / ADB retrieve settings
        """
        KnowledgeBaseDataAPI.__init__(self, knowledge_base_name, config)
        ControlAPI.__init__(self, config)
        self.provider_settings = provider_settings
        self.retrieve_settings = retrieve_settings

    def _build_query_content_request(
        self, query: str, config: Optional[Config] = None
    ) -> gpdb_models.QueryContentRequest:
        """构建 QueryContent 请求 / Build QueryContent request

        Args:
            query: 查询文本 / Query text
            config: 配置 / Configuration

        Returns:
            QueryContentRequest: GPDB QueryContent 请求对象
        """
        if self.provider_settings is None:
            raise ValueError("provider_settings is required for ADB retrieval")

        cfg = Config.with_configs(self.config, config)

        # 构建基础请求参数 / Build base request parameters
        request_params: Dict[str, Any] = {
            "content": query,
            "dbinstance_id": self.provider_settings.db_instance_id,
            "namespace": self.provider_settings.namespace,
            "namespace_password": self.provider_settings.namespace_password,
            "collection": self.knowledge_base_name,
            "region_id": cfg.get_region_id(),
        }

        # 添加可选的提供商设置 / Add optional provider settings
        if self.provider_settings.metrics is not None:
            request_params["metrics"] = self.provider_settings.metrics

        # 添加检索设置 / Add retrieve settings
        if self.retrieve_settings:
            if self.retrieve_settings.top_k is not None:
                request_params["top_k"] = self.retrieve_settings.top_k
            if self.retrieve_settings.use_full_text_retrieval is not None:
                request_params["use_full_text_retrieval"] = (
                    self.retrieve_settings.use_full_text_retrieval
                )
            if self.retrieve_settings.rerank_factor is not None:
                request_params["rerank_factor"] = (
                    self.retrieve_settings.rerank_factor
                )
            if self.retrieve_settings.recall_window is not None:
                request_params["recall_window"] = (
                    self.retrieve_settings.recall_window
                )
            if self.retrieve_settings.hybrid_search is not None:
                request_params["hybrid_search"] = (
                    self.retrieve_settings.hybrid_search
                )
            if self.retrieve_settings.hybrid_search_args is not None:
                request_params["hybrid_search_args"] = (
                    self.retrieve_settings.hybrid_search_args
                )

        return gpdb_models.QueryContentRequest(**request_params)

    def _parse_query_content_response(
        self, response: gpdb_models.QueryContentResponse, query: str
    ) -> Dict[str, Any]:
        """解析 QueryContent 响应 / Parse QueryContent response

        Args:
            response: GPDB QueryContent 响应对象
            query: 原始查询文本 / Original query text

        Returns:
            Dict[str, Any]: 格式化的检索结果 / Formatted retrieval results
        """
        all_matches: List[Dict[str, Any]] = []

        if response.body and response.body.matches:
            match_list = response.body.matches.match_list or []
            for match in match_list:
                all_matches.append({
                    "content": (
                        match.content if hasattr(match, "content") else None
                    ),
                    "score": match.score if hasattr(match, "score") else None,
                    "id": match.id if hasattr(match, "id") else None,
                    "file_name": (
                        match.file_name if hasattr(match, "file_name") else None
                    ),
                    "file_url": (
                        match.file_url if hasattr(match, "file_url") else None
                    ),
                    "metadata": (
                        match.metadata if hasattr(match, "metadata") else None
                    ),
                    "rerank_score": (
                        match.rerank_score
                        if hasattr(match, "rerank_score")
                        else None
                    ),
                    "retrieval_source": (
                        match.retrieval_source
                        if hasattr(match, "retrieval_source")
                        else None
                    ),
                })

        return {
            "data": all_matches,
            "query": query,
            "knowledge_base_name": self.knowledge_base_name,
            "request_id": (
                response.body.request_id
                if response.body and hasattr(response.body, "request_id")
                else None
            ),
        }

    async def retrieve_async(
        self,
        query: str,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """ADB 检索（异步）/ ADB retrieval asynchronously

        通过 GPDB SDK 调用 QueryContent 接口进行知识库检索。
        Retrieves from ADB knowledge base via GPDB SDK QueryContent API.

        Args:
            query: 查询文本 / Query text
            config: 配置 / Configuration

        Returns:
            Dict[str, Any]: 检索结果 / Retrieval results
        """
        try:
            if self.provider_settings is None:
                raise ValueError(
                    "provider_settings is required for ADB retrieval"
                )

            # 获取 GPDB 客户端 / Get GPDB client
            client = self._get_gpdb_client(config)

            # 构建请求 / Build request
            request = self._build_query_content_request(query, config)
            logger.debug(f"ADB QueryContent request: {request}")

            # 调用 QueryContent API / Call QueryContent API
            response = await client.query_content_async(request)
            logger.debug(f"ADB QueryContent response: {response}")

            # 解析并返回结果 / Parse and return results
            return self._parse_query_content_response(response, query)

        except Exception as e:
            logger.warning(
                "Failed to retrieve from ADB knowledge base "
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
        Union[
            RagFlowProviderSettings,
            BailianProviderSettings,
            ADBProviderSettings,
        ]
    ] = None,
    retrieve_settings: Optional[
        Union[
            RagFlowRetrieveSettings,
            BailianRetrieveSettings,
            ADBRetrieveSettings,
        ]
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
    elif provider == KnowledgeBaseProvider.ADB or provider == "adb":
        adb_provider_settings = (
            provider_settings
            if isinstance(provider_settings, ADBProviderSettings)
            else None
        )
        adb_retrieve_settings = (
            retrieve_settings
            if isinstance(retrieve_settings, ADBRetrieveSettings)
            else None
        )
        return ADBDataAPI(
            knowledge_base_name,
            config,
            provider_settings=adb_provider_settings,
            retrieve_settings=adb_retrieve_settings,
        )
    else:
        raise ValueError(f"Unsupported provider type: {provider}")
