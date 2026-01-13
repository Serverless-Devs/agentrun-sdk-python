"""MemoryCollection 高层 API / MemoryCollection High-Level API

此模块定义记忆集合资源的高级API。
This module defines the high-level API for memory collection resources.
"""

from typing import Any, Dict, List, Optional, Tuple

from agentrun.utils.config import Config
from agentrun.utils.model import PageableInput
from agentrun.utils.resource import ResourceBase

from .model import (
    MemoryCollectionCreateInput,
    MemoryCollectionImmutableProps,
    MemoryCollectionListInput,
    MemoryCollectionListOutput,
    MemoryCollectionMutableProps,
    MemoryCollectionSystemProps,
    MemoryCollectionUpdateInput,
)


class MemoryCollection(
    MemoryCollectionMutableProps,
    MemoryCollectionImmutableProps,
    MemoryCollectionSystemProps,
    ResourceBase,
):
    """记忆集合资源 / MemoryCollection Resource

    提供记忆集合的完整生命周期管理,包括创建、删除、更新、查询。
    Provides complete lifecycle management for memory collections, including create, delete, update, and query.
    """

    @classmethod
    def __get_client(cls):
        """获取客户端实例 / Get client instance

        Returns:
            MemoryCollectionClient: 客户端实例 / Client instance
        """
        from .client import MemoryCollectionClient

        return MemoryCollectionClient()

    @classmethod
    async def create_async(
        cls, input: MemoryCollectionCreateInput, config: Optional[Config] = None
    ):
        """创建记忆集合（异步）

        Args:
            input: 记忆集合输入参数
            config: 配置

        Returns:
            MemoryCollection: 创建的记忆集合对象
        """
        return await cls.__get_client().create_async(input, config=config)

    @classmethod
    async def delete_by_name_async(
        cls, memory_collection_name: str, config: Optional[Config] = None
    ):
        """根据名称删除记忆集合（异步）

        Args:
            memory_collection_name: 记忆集合名称
            config: 配置
        """
        return await cls.__get_client().delete_async(
            memory_collection_name, config=config
        )

    @classmethod
    async def update_by_name_async(
        cls,
        memory_collection_name: str,
        input: MemoryCollectionUpdateInput,
        config: Optional[Config] = None,
    ):
        """根据名称更新记忆集合（异步）

        Args:
            memory_collection_name: 记忆集合名称
            input: 记忆集合更新输入参数
            config: 配置

        Returns:
            MemoryCollection: 更新后的记忆集合对象
        """
        return await cls.__get_client().update_async(
            memory_collection_name, input, config=config
        )

    @classmethod
    async def get_by_name_async(
        cls, memory_collection_name: str, config: Optional[Config] = None
    ):
        """根据名称获取记忆集合（异步）

        Args:
            memory_collection_name: 记忆集合名称
            config: 配置

        Returns:
            MemoryCollection: 记忆集合对象
        """
        return await cls.__get_client().get_async(
            memory_collection_name, config=config
        )

    @classmethod
    async def _list_page_async(
        cls, page_input: PageableInput, config: Config | None = None, **kwargs
    ):
        return await cls.__get_client().list_async(
            input=MemoryCollectionListInput(
                **kwargs,
                **page_input.model_dump(),
            ),
            config=config,
        )

    @classmethod
    async def list_all_async(
        cls,
        *,
        memory_collection_name: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> List[MemoryCollectionListOutput]:
        """列出所有记忆集合（异步）

        Args:
            memory_collection_name: 记忆集合名称（可选）
            config: 配置

        Returns:
            List[MemoryCollectionListOutput]: 记忆集合列表
        """
        return await cls._list_all_async(
            lambda mc: mc.memory_collection_id or "",
            config=config,
            memory_collection_name=memory_collection_name,
        )

    async def update_async(
        self,
        input: MemoryCollectionUpdateInput,
        config: Optional[Config] = None,
    ):
        """更新记忆集合（异步）

        Args:
            input: 记忆集合更新输入参数
            config: 配置

        Returns:
            MemoryCollection: 更新后的记忆集合对象
        """
        if self.memory_collection_name is None:
            raise ValueError(
                "memory_collection_name is required to update a"
                " MemoryCollection"
            )

        result = await self.update_by_name_async(
            self.memory_collection_name, input, config=config
        )
        self.update_self(result)

        return self

    async def delete_async(self, config: Optional[Config] = None):
        """删除记忆集合（异步）

        Args:
            config: 配置
        """
        if self.memory_collection_name is None:
            raise ValueError(
                "memory_collection_name is required to delete a"
                " MemoryCollection"
            )

        return await self.delete_by_name_async(
            self.memory_collection_name, config=config
        )

    async def get_async(self, config: Optional[Config] = None):
        """刷新记忆集合信息（异步）

        Args:
            config: 配置

        Returns:
            MemoryCollection: 刷新后的记忆集合对象
        """
        if self.memory_collection_name is None:
            raise ValueError(
                "memory_collection_name is required to refresh a"
                " MemoryCollection"
            )

        result = await self.get_by_name_async(
            self.memory_collection_name, config=config
        )
        self.update_self(result)

        return self

    async def refresh_async(self, config: Optional[Config] = None):
        """刷新记忆集合信息（异步）

        Args:
            config: 配置

        Returns:
            MemoryCollection: 刷新后的记忆集合对象
        """
        return await self.get_async(config=config)

    @classmethod
    async def to_mem0_memory_async(
        cls,
        memory_collection_name: str,
        config: Optional[Config] = None,
        history_db_path: Optional[str] = None,
    ):
        """将 MemoryCollection 转换为 agentrun-mem0ai Memory 客户端（异步）

        Args:
            memory_collection_name: 记忆集合名称
            config: AgentRun 配置
            history_db_path: mem0 历史数据库路径（可选）

        Returns:
            Memory: agentrun-mem0ai Memory 客户端实例

        Raises:
            ImportError: 如果未安装 agentrun-mem0ai 包
            ValueError: 如果配置信息不完整

        Example:
            >>> memory = await MemoryCollection.to_mem0_memory_async(
            ...     "memoryCollection010901",
            ...     config=config
            ... )
            >>> memory.add("用户喜欢吃苹果", user_id="user123")
        """
        try:
            from agentrun_mem0 import Memory
        except ImportError as e:
            raise ImportError(
                "agentrun-mem0ai package is required. Install it with: pip"
                " install agentrun-mem0ai"
            ) from e

        # 获取 MemoryCollection 配置
        memory_collection = await cls.get_by_name_async(
            memory_collection_name, config=config
        )

        # 构建 mem0 配置
        mem0_config = await cls._build_mem0_config_async(
            memory_collection, config, history_db_path
        )

        # 创建并返回 Memory 实例
        return Memory.from_config(mem0_config)

    @staticmethod
    def _convert_vpc_endpoint_to_public(endpoint: str) -> str:
        """将 VPC 内网地址转换为公网地址

        Args:
            endpoint: 原始 endpoint，可能是 VPC 内网地址

        Returns:
            str: 公网地址

        Example:
            >>> _convert_vpc_endpoint_to_public("https://jiuqing.cn-hangzhou.vpc.tablestore.aliyuncs.com")
            "https://jiuqing.cn-hangzhou.ots.aliyuncs.com"
        """
        if ".vpc.tablestore.aliyuncs.com" in endpoint:
            # 将 .vpc.tablestore.aliyuncs.com 替换为 .ots.aliyuncs.com
            return endpoint.replace(
                ".vpc.tablestore.aliyuncs.com", ".ots.aliyuncs.com"
            )
        return endpoint

    @classmethod
    async def _build_mem0_config_async(
        cls,
        memory_collection: "MemoryCollection",
        config: Optional[Config],
        history_db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建 mem0 配置字典（异步）

        Args:
            memory_collection: MemoryCollection 对象
            config: AgentRun 配置
            history_db_path: 历史数据库路径

        Returns:
            Dict[str, Any]: mem0 配置字典
        """
        mem0_config: Dict[str, Any] = {}

        # 构建 vector_store 配置
        if memory_collection.vector_store_config:
            vector_store_config = memory_collection.vector_store_config
            provider = vector_store_config.provider or ""

            if vector_store_config.config:
                vs_config = vector_store_config.config
                vector_store: Dict[str, Any] = {
                    "provider": provider,
                    "config": {},
                }

                # 根据不同的 provider 构建配置
                if provider == "aliyun_tablestore":
                    # 获取凭证信息
                    effective_config = config or Config()
                    # 将 VPC 内网地址转换为公网地址
                    public_endpoint = cls._convert_vpc_endpoint_to_public(
                        vs_config.endpoint or ""
                    )
                    vector_store["config"] = {
                        "vector_dimension": vs_config.vector_dimension,
                        "endpoint": public_endpoint,
                        "instance_name": vs_config.instance_name,
                        "collection_name": vs_config.collection_name,
                        "access_key_id": effective_config.get_access_key_id(),
                        "access_key_secret": (
                            effective_config.get_access_key_secret()
                        ),
                    }
                    # 如果有 security_token，添加它
                    security_token = effective_config.get_security_token()
                    if security_token:
                        vector_store["config"]["sts_token"] = security_token
                else:
                    # 其他 provider 的通用配置
                    vector_store["config"] = {
                        "endpoint": vs_config.endpoint,
                        "collection_name": vs_config.collection_name,
                    }
                    if vs_config.vector_dimension:
                        vector_store["config"][
                            "vector_dimension"
                        ] = vs_config.vector_dimension

                mem0_config["vector_store"] = vector_store

        # 构建 llm 配置
        if memory_collection.llm_config:
            llm_config = memory_collection.llm_config
            model_service_name = llm_config.model_service_name

            if model_service_name and llm_config.config:
                # 使用高层 API 获取 ModelService 配置
                base_url, api_key = (
                    await cls._resolve_model_service_config_async(
                        model_service_name, config
                    )
                )

                mem0_config["llm"] = {
                    "provider": "openai",  # mem0 使用 openai 兼容接口
                    "config": {
                        "model": llm_config.config.model,
                        "openai_base_url": base_url,
                        "api_key": api_key,
                    },
                }

        # 构建 embedder 配置
        if memory_collection.embedder_config:
            embedder_config = memory_collection.embedder_config
            model_service_name = embedder_config.model_service_name

            if model_service_name and embedder_config.config:
                # 使用高层 API 获取 ModelService 配置
                base_url, api_key = (
                    await cls._resolve_model_service_config_async(
                        model_service_name, config
                    )
                )

                mem0_config["embedder"] = {
                    "provider": "openai",  # mem0 使用 openai 兼容接口
                    "config": {
                        "model": embedder_config.config.model,
                        "openai_base_url": base_url,
                        "api_key": api_key,
                    },
                }

        # 添加历史数据库路径
        if history_db_path:
            mem0_config["history_db_path"] = history_db_path

        return mem0_config

    @staticmethod
    async def _resolve_model_service_config_async(
        model_service_name: str, config: Optional[Config]
    ) -> Tuple[str, str]:
        """解析 ModelService 配置获取 baseUrl 和 apiKey（异步）

        Args:
            model_service_name: ModelService 名称
            config: AgentRun 配置

        Returns:
            Tuple[str, str]: (base_url, api_key)

        Raises:
            ValueError: 如果配置信息不完整
        """
        from agentrun.credential import Credential
        from agentrun.model import ModelService

        # 使用高层 API 获取 ModelService
        model_service = await ModelService.get_by_name_async(
            model_service_name, config=config
        )

        # 获取 provider_settings
        if not model_service.provider_settings:
            raise ValueError(
                f"ModelService {model_service_name} providerSettings is empty"
            )

        base_url = model_service.provider_settings.base_url or ""
        api_key = model_service.provider_settings.api_key or ""

        # 如果有 credentialName，使用高层 API 获取 credential secret
        credential_name = model_service.credential_name
        if credential_name:
            credential = await Credential.get_by_name_async(
                credential_name, config=config
            )
            if credential.credential_secret:
                api_key = credential.credential_secret

        if not base_url:
            raise ValueError(
                f"ModelService {model_service_name} baseUrl is empty"
            )

        return base_url, api_key
