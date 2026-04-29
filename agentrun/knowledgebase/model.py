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
    ADB = "adb"
    """ADB (AnalyticDB for PostgreSQL) 知识库 / ADB knowledge base"""
    OTS = "ots"
    """OTS (TableStore) 知识库 / OTS (TableStore) knowledge base"""


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
# ADB 配置模型 / ADB Configuration Models
# =============================================================================


class ADBProviderSettings(BaseModel):
    """ADB (AnalyticDB for PostgreSQL) 提供商设置 / ADB Provider Settings

    配置 ADB 知识库的连接和访问参数。
    Configure ADB knowledge base connection and access parameters.
    """

    db_instance_id: str
    """ADB 实例 ID / ADB instance ID"""
    namespace: str
    """命名空间，默认为 public / Namespace, defaults to public"""
    namespace_password: str
    """命名空间密码 / Namespace password"""
    embedding_model: Optional[str] = None
    """向量化模型名称，如 text-embedding-v3 / Embedding model name"""
    metrics: Optional[str] = None
    """相似度算法：l2（欧氏距离）、ip（内积）、cosine（余弦相似度）
    Similarity algorithm: l2 (Euclidean), ip (inner product), cosine"""
    metadata: Optional[str] = None
    """元数据配置，JSON 字符串格式 / Metadata configuration in JSON string format"""


class ADBRetrieveSettings(BaseModel):
    """ADB 检索设置 / ADB Retrieve Settings

    配置 ADB 知识库的检索参数，支持向量检索和全文检索的混合模式。
    Configure ADB knowledge base retrieval parameters, supporting hybrid
    vector and full-text retrieval modes.
    """

    top_k: Optional[int] = None
    """返回结果的数量 / Number of results to return"""
    use_full_text_retrieval: Optional[bool] = None
    """是否启用全文检索（双路召回），默认 false 仅使用向量检索
    Enable full-text retrieval (dual recall), defaults to false (vector only)"""
    rerank_factor: Optional[float] = None
    """重排序因子，取值范围 1 < RerankFactor <= 5
    Re-ranking factor, value range: 1 < RerankFactor <= 5"""
    recall_window: Optional[List[int]] = None
    """召回窗口，格式为 [A, B]，其中 -10 <= A <= 0，0 <= B <= 10
    Recall window, format [A, B] where -10 <= A <= 0, 0 <= B <= 10"""
    hybrid_search: Optional[str] = None
    """混合检索算法：RRF（倒数排名融合）、Weight（加权排序）、Cascaded（级联检索）
    Hybrid search algorithm: RRF, Weight, or Cascaded"""
    hybrid_search_args: Optional[Dict[str, Any]] = None
    """混合检索算法参数，如 {"RRF": {"k": 60}} 或 {"Weight": {"alpha": 0.5}}
    Hybrid search algorithm parameters"""


# =============================================================================
# OTS 配置模型 / OTS Configuration Models
# =============================================================================


class OTSMetadataField(BaseModel):
    """OTS 元数据字段定义 / OTS Metadata Field Definition

    支持的类型：string / long / double / boolean / date /
    string_list / long_list / double_list / boolean_list / date_list
    """

    name: str
    """字段名 / Field name"""
    type: str
    """字段类型 / Field type"""


class OTSEmbeddingConfiguration(BaseModel):
    """OTS 向量化配置 / OTS Embedding Configuration"""

    provider: str
    """向量化服务提供商，如 "bailian" / Embedding provider"""
    model: str
    """向量化模型名称，如 "text-embedding-v3" / Embedding model name"""
    dimension: int
    """向量维度，如 1024 / Vector dimension"""
    url: Optional[str] = None
    """向量化服务地址（可选）/ Embedding service URL (optional)"""
    api_key: Optional[str] = None
    """向量化服务 API Key（可选）/ Embedding API key (optional)"""


class OTSProviderSettings(BaseModel):
    """OTS (TableStore) 提供商设置 / OTS Provider Settings

    配置 OTS 知识库的连接和访问参数。
    Configure OTS knowledge base connection and access parameters.
    """

    ots_instance_name: str
    """OTS 实例名称 / OTS instance name"""
    tags: Optional[List[str]] = None
    """标签列表 / Tag list"""
    metadata: Optional[List[OTSMetadataField]] = None
    """元数据字段定义列表 / Metadata field definitions"""
    embedding_configuration: Optional[OTSEmbeddingConfiguration] = None
    """向量化配置 / Embedding configuration"""


class OTSDenseVectorSearchConfig(BaseModel):
    """OTS 向量检索配置 / OTS Dense Vector Search Configuration"""

    number_of_results: Optional[int] = None
    """向量检索返回结果数量 / Number of dense vector search results"""


class OTSFullTextSearchConfig(BaseModel):
    """OTS 全文检索配置 / OTS Full Text Search Configuration"""

    number_of_results: Optional[int] = None
    """全文检索返回结果数量 / Number of full text search results"""


class OTSRRFConfig(BaseModel):
    """OTS RRF 重排序配置 / OTS RRF Reranking Configuration"""

    dense_vector_search_weight: Optional[float] = None
    """向量检索权重，默认 1.0 / Dense vector search weight"""
    full_text_search_weight: Optional[float] = None
    """全文检索权重，默认 1.0 / Full text search weight"""
    k: Optional[int] = None
    """RRF 参数 k，默认 60 / RRF parameter k"""


class OTSWeightConfig(BaseModel):
    """OTS Weight 重排序配置 / OTS Weight Reranking Configuration"""

    dense_vector_search_weight: Optional[float] = None
    """向量检索权重 / Dense vector search weight"""
    full_text_search_weight: Optional[float] = None
    """全文检索权重 / Full text search weight"""


class OTSModelConfig(BaseModel):
    """OTS Model 重排序配置 / OTS Model Reranking Configuration"""

    provider: Optional[str] = None
    """重排序模型提供商 / Reranking model provider"""
    model: Optional[str] = None
    """重排序模型名称 / Reranking model name"""


class OTSRerankingConfig(BaseModel):
    """OTS 重排序配置 / OTS Reranking Configuration"""

    type: Optional[str] = None
    """重排序类型：RRF / WEIGHT / MODEL / Reranking type"""
    number_of_results: Optional[int] = None
    """重排序后返回结果数量 / Number of results after reranking"""
    rrf_configuration: Optional[OTSRRFConfig] = None
    """RRF 配置（当 type=RRF 时）/ RRF config (when type=RRF)"""
    weight_configuration: Optional[OTSWeightConfig] = None
    """Weight 配置（当 type=WEIGHT 时）/ Weight config (when type=WEIGHT)"""
    model_configuration: Optional[OTSModelConfig] = None
    """Model 配置（当 type=MODEL 时）/ Model config (when type=MODEL)"""


class OTSRetrieveSettings(BaseModel):
    """OTS 检索设置 / OTS Retrieve Settings

    配置 OTS 知识库的检索参数，支持向量检索、全文检索和混合检索。
    Configure OTS knowledge base retrieval parameters, supporting
    dense vector, full-text, and hybrid search.
    """

    search_type: Optional[List[str]] = None
    """检索类型列表，支持 DENSE_VECTOR 和 FULL_TEXT
    Search type list, supports DENSE_VECTOR and FULL_TEXT"""
    dense_vector_search_configuration: Optional[OTSDenseVectorSearchConfig] = (
        None
    )
    """向量检索配置 / Dense vector search configuration"""
    full_text_search_configuration: Optional[OTSFullTextSearchConfig] = None
    """全文检索配置 / Full text search configuration"""
    reranking_configuration: Optional[OTSRerankingConfig] = None
    """重排序配置 / Reranking configuration"""
    filter: Optional[Dict[str, Any]] = None
    """元数据过滤条件 / Metadata filter"""


# =============================================================================
# 联合类型定义 / Union Type Definitions
# =============================================================================

ProviderSettings = Union[
    RagFlowProviderSettings,
    BailianProviderSettings,
    ADBProviderSettings,
    OTSProviderSettings,
    Dict[str, Any],
]
"""提供商设置联合类型 / Provider settings union type"""

RetrieveSettings = Union[
    RagFlowRetrieveSettings,
    BailianRetrieveSettings,
    ADBRetrieveSettings,
    OTSRetrieveSettings,
    Dict[str, Any],
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
    workspace_id: Optional[str] = None
    """知识库所属的 AgentRun 工作空间标识符；可选项，不填则使用默认工作空间。
    注意：与 ``BailianProviderSettings.workspace_id`` 不同，后者指百炼侧的 workspace。
    / Workspace identifier the knowledge base belongs to in AgentRun; optional,
    defaults to the default workspace if not provided. Distinct from
    ``BailianProviderSettings.workspace_id`` which refers to the Bailian-side workspace."""


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
    workspace_id: Optional[str] = None
    """按 AgentRun 工作空间标识符过滤
    / Filter by AgentRun workspace identifier"""


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
    workspace_id: Optional[str] = None
    """知识库所属的 AgentRun 工作空间标识符
    / AgentRun workspace identifier the knowledge base belongs to"""

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
