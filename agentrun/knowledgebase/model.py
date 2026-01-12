"""KnowledgeBase 模型定义 / KnowledgeBase Model Definitions

定义知识库相关的数据模型和枚举。
Defines data models and enumerations related to knowledge bases.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from agentrun.utils.config import Config
from agentrun.utils.model import BaseModel, PageableInput


class KnowledgeBaseProvider(str, Enum):
    """知识库提供商类型 / KnowledgeBase Provider Type"""

    RAGFLOW = "ragflow"
    """RagFlow 知识库 / RagFlow knowledge base"""
    BAILIAN = "bailian"
    """百炼知识库 / Bailian knowledge base"""


# =============================================================================
# RagFlow 配置模型 / RagFlow Configuration Models
# =============================================================================


class RagFlowProviderSettings(BaseModel):
    """RagFlow 提供商设置 / RagFlow Provider Settings"""

    base_url: str
    """RagFlow 服务地址，http或https开头，最后不能有/
    RagFlow service URL, starting with http or https, no trailing slash"""
    dataset_ids: List[str]
    """RagFlow 知识库 ID 列表，可以填写多个
    List of RagFlow dataset IDs, multiple values allowed"""


class RagFlowRetrieveSettings(BaseModel):
    """RagFlow 检索设置 / RagFlow Retrieve Settings"""

    similarity_threshold: Optional[float] = None
    """相似度阈值 / Similarity threshold"""
    vector_similarity_weight: Optional[float] = None
    """向量相似度权重 / Vector similarity weight"""
    cross_languages: Optional[List[str]] = None
    """跨语言检索语言列表，如 ["English", "Chinese"]
    Cross-language retrieval languages, e.g. ["English", "Chinese"]"""


# =============================================================================
# Bailian 配置模型 / Bailian Configuration Models
# =============================================================================


class BailianProviderSettings(BaseModel):
    """百炼提供商设置 / Bailian Provider Settings"""

    workspace_id: str
    """百炼工作空间 ID / Bailian workspace ID"""
    index_ids: List[str]
    """绑定的知识库索引列表 / List of bound knowledge base index IDs"""


class BailianRetrieveSettings(BaseModel):
    """百炼检索设置 / Bailian Retrieve Settings"""

    dense_similarity_top_k: Optional[int] = None
    """稠密向量检索返回的 Top K 数量 / Dense similarity top K"""
    sparse_similarity_top_k: Optional[int] = None
    """稀疏向量检索返回的 Top K 数量 / Sparse similarity top K"""
    rerank_min_score: Optional[float] = None
    """重排序最低分数阈值 / Rerank minimum score threshold"""
    rerank_top_n: Optional[int] = None
    """重排序返回的 Top N 数量 / Rerank top N"""


# =============================================================================
# 联合类型定义 / Union Type Definitions
# =============================================================================

ProviderSettings = Union[
    RagFlowProviderSettings, BailianProviderSettings, Dict[str, Any]
]
"""提供商设置联合类型 / Provider settings union type"""

RetrieveSettings = Union[
    RagFlowRetrieveSettings, BailianRetrieveSettings, Dict[str, Any]
]
"""检索设置联合类型 / Retrieve settings union type"""


# =============================================================================
# 知识库属性模型 / KnowledgeBase Property Models
# =============================================================================


class KnowledgeBaseMutableProps(BaseModel):
    """知识库可变属性 / KnowledgeBase Mutable Properties"""

    description: Optional[str] = None
    """描述 / Description"""
    credential_name: Optional[str] = None
    """凭证名称 / Credential name"""
    provider_settings: Optional[Union[ProviderSettings, Dict[str, Any]]] = None
    """提供商设置 / Provider settings"""
    retrieve_settings: Optional[Union[RetrieveSettings, Dict[str, Any]]] = None
    """检索设置 / Retrieve settings"""


class KnowledgeBaseImmutableProps(BaseModel):
    """知识库不可变属性 / KnowledgeBase Immutable Properties"""

    knowledge_base_name: Optional[str] = None
    """知识库名称 / KnowledgeBase name"""
    provider: Optional[Union[KnowledgeBaseProvider, str]] = None
    """提供商 / Provider"""


class KnowledgeBaseSystemProps(BaseModel):
    """知识库系统属性 / KnowledgeBase System Properties"""

    knowledge_base_id: Optional[str] = None
    """知识库 ID / KnowledgeBase ID"""
    created_at: Optional[str] = None
    """创建时间 / Created at"""
    last_updated_at: Optional[str] = None
    """最后更新时间 / Last updated at"""


# =============================================================================
# API 输入输出模型 / API Input/Output Models
# =============================================================================


class KnowledgeBaseCreateInput(
    KnowledgeBaseImmutableProps, KnowledgeBaseMutableProps
):
    """知识库创建输入参数 / KnowledgeBase Create Input"""

    knowledge_base_name: str  # type: ignore
    """知识库名称（必填）/ KnowledgeBase name (required)"""
    provider: Union[KnowledgeBaseProvider, str]  # type: ignore
    """提供商（必填）/ Provider (required)"""
    provider_settings: Union[ProviderSettings, Dict[str, Any]]  # type: ignore
    """提供商设置（必填）/ Provider settings (required)"""


class KnowledgeBaseUpdateInput(KnowledgeBaseMutableProps):
    """知识库更新输入参数 / KnowledgeBase Update Input"""

    pass


class KnowledgeBaseListInput(PageableInput):
    """知识库列表查询输入参数 / KnowledgeBase List Input"""

    provider: Optional[Union[KnowledgeBaseProvider, str]] = None
    """提供商 / Provider"""


class KnowledgeBaseListOutput(BaseModel):
    """知识库列表查询输出 / KnowledgeBase List Output"""

    knowledge_base_id: Optional[str] = None
    """知识库 ID / KnowledgeBase ID"""
    knowledge_base_name: Optional[str] = None
    """知识库名称 / KnowledgeBase name"""
    provider: Optional[Union[KnowledgeBaseProvider, str]] = None
    """提供商 / Provider"""
    description: Optional[str] = None
    """描述 / Description"""
    credential_name: Optional[str] = None
    """凭证名称 / Credential name"""
    provider_settings: Optional[Union[ProviderSettings, Dict[str, Any]]] = None
    """提供商设置 / Provider settings"""
    retrieve_settings: Optional[Union[RetrieveSettings, Dict[str, Any]]] = None
    """检索设置 / Retrieve settings"""
    created_at: Optional[str] = None
    """创建时间 / Created at"""
    last_updated_at: Optional[str] = None
    """最后更新时间 / Last updated at"""

    async def to_knowledge_base_async(self, config: Optional[Config] = None):
        """转换为知识库对象（异步）/ Convert to KnowledgeBase object (async)

        Args:
            config: 配置 / Configuration

        Returns:
            KnowledgeBase: 知识库对象 / KnowledgeBase object
        """
        from .client import KnowledgeBaseClient

        return await KnowledgeBaseClient(config).get_async(
            self.knowledge_base_name or "", config=config
        )

    def to_knowledge_base(self, config: Optional[Config] = None):
        """转换为知识库对象（同步）/ Convert to KnowledgeBase object (sync)

        Args:
            config: 配置 / Configuration

        Returns:
            KnowledgeBase: 知识库对象 / KnowledgeBase object
        """
        from .client import KnowledgeBaseClient

        return KnowledgeBaseClient(config).get(
            self.knowledge_base_name or "", config=config
        )


class RetrieveInput(BaseModel):
    """知识库检索输入参数 / KnowledgeBase Retrieve Input

    用于多知识库检索的输入参数。
    Input parameters for multi-knowledge base retrieval.
    """

    knowledge_base_names: List[str]
    """知识库名称列表 / List of knowledge base names"""
    query: str
    """查询文本 / Query text"""

    knowledge_base_id: Optional[str] = None
    """知识库 ID / KnowledgeBase ID"""
    knowledge_base_name: Optional[str] = None
    """知识库名称 / KnowledgeBase name"""
    provider: Optional[str] = None
    """提供商 / Provider"""
    description: Optional[str] = None
    """描述 / Description"""
    credential_name: Optional[str] = None
    """凭证名称 / Credential name"""
    created_at: Optional[str] = None
    """创建时间 / Created at"""
    last_updated_at: Optional[str] = None
    """最后更新时间 / Last updated at"""

    async def to_knowledge_base_async(self, config: Optional[Config] = None):
        """转换为知识库对象（异步）/ Convert to KnowledgeBase object (async)

        Args:
            config: 配置 / Configuration

        Returns:
            KnowledgeBase: 知识库对象 / KnowledgeBase object
        """
        from .client import KnowledgeBaseClient

        return await KnowledgeBaseClient(config).get_async(
            self.knowledge_base_name or "", config=config
        )

    def to_knowledge_base(self, config: Optional[Config] = None):
        """转换为知识库对象（同步）/ Convert to KnowledgeBase object (sync)

        Args:
            config: 配置 / Configuration

        Returns:
            KnowledgeBase: 知识库对象 / KnowledgeBase object
        """
        from .client import KnowledgeBaseClient

        return KnowledgeBaseClient(config).get(
            self.knowledge_base_name or "", config=config
        )
