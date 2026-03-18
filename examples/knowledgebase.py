"""
知识库模块示例 / KnowledgeBase Module Example

本示例演示如何使用 AgentRun SDK 管理知识库，包括百炼、RagFlow、ADB 和 OTS 四种类型：
This example demonstrates how to use the AgentRun SDK to manage knowledge bases,
including Bailian, RagFlow, ADB and OTS types:

1. 创建知识库 / Create knowledge base (Bailian, RagFlow, ADB & OTS)
2. 获取知识库信息 / Get knowledge base info
3. 查询知识库 / Query knowledge base
4. 更新知识库配置 / Update knowledge base configuration
5. 删除知识库 / Delete knowledge base

使用前请确保设置以下环境变量：
Before using, please set the following environment variables:
- AGENTRUN_ACCESS_KEY_ID: 阿里云 AccessKey ID
- AGENTRUN_ACCESS_KEY_SECRET: 阿里云 AccessKey Secret
- AGENTRUN_REGION: 区域（默认 cn-hangzhou）

百炼知识库额外配置 / Additional config for Bailian:
- BAILIAN_WORKSPACE_ID: 百炼工作空间 ID
- BAILIAN_INDEX_IDS: 百炼知识库索引 ID 列表（逗号分隔）

RagFlow 知识库额外配置 / Additional config for RagFlow:
- RAGFLOW_BASE_URL: RagFlow 服务地址
- RAGFLOW_DATASET_IDS: RagFlow 数据集 ID 列表（逗号分隔）
- RAGFLOW_CREDENTIAL_NAME: RagFlow API Key 凭证名称

ADB 知识库额外配置 / Additional config for ADB:
- ADB_INSTANCE_ID: ADB 实例 ID
- ADB_NAMESPACE: ADB 命名空间
- ADB_NAMESPACE_PASSWORD: ADB 命名空间密码
- ADB_COLLECTION: ADB 文档集合名称

OTS 知识库额外配置 / Additional config for OTS:
- OTS_INSTANCE_NAME: OTS (TableStore) 实例名称
- OTS_EMBEDDING_PROVIDER: 向量化服务提供商（默认 bailian）
- OTS_EMBEDDING_MODEL: 向量化模型名称（默认 text-embedding-v3）
- OTS_EMBEDDING_DIMENSION: 向量维度（默认 1024）
"""

import json
import os
import time

from agentrun.knowledgebase import (
    ADBProviderSettings,
    ADBRetrieveSettings,
    BailianProviderSettings,
    BailianRetrieveSettings,
    KnowledgeBase,
    KnowledgeBaseClient,
    KnowledgeBaseCreateInput,
    KnowledgeBaseProvider,
    KnowledgeBaseUpdateInput,
    OTSDenseVectorSearchConfig,
    OTSEmbeddingConfiguration,
    OTSFullTextSearchConfig,
    OTSMetadataField,
    OTSProviderSettings,
    OTSRerankingConfig,
    OTSRetrieveSettings,
    OTSRRFConfig,
    OTSWeightConfig,
    RagFlowProviderSettings,
    RagFlowRetrieveSettings,
)
from agentrun.utils.exception import (
    ResourceAlreadyExistError,
    ResourceNotExistError,
)
from agentrun.utils.log import logger

# ============================================================================
# 配置项 / Configuration
# ============================================================================

# 时间戳后缀，用于生成唯一名称
# Timestamp suffix for generating unique names
TIMESTAMP = time.strftime("%Y%m%d%H%M%S")

# -----------------------------------------------------------------------------
# 百炼知识库配置 / Bailian Knowledge Base Configuration
# -----------------------------------------------------------------------------

# 百炼知识库名称
# Bailian knowledge base name
BAILIAN_KB_NAME = f"sdk-test-bailian-kb-{TIMESTAMP}"

# 百炼工作空间 ID，请替换为您的实际值
# Bailian workspace ID, please replace with your actual value
BAILIAN_WORKSPACE_ID = os.getenv("BAILIAN_WORKSPACE_ID", "your-workspace-id")

# 百炼知识库索引 ID 列表，请替换为您的实际值
# Bailian knowledge base index ID list, please replace with your actual values
BAILIAN_INDEX_IDS = os.getenv(
    "BAILIAN_INDEX_IDS", "index-id-1,index-id-2"
).split(",")

# -----------------------------------------------------------------------------
# RagFlow 知识库配置 / RagFlow Knowledge Base Configuration
# -----------------------------------------------------------------------------

# RagFlow 知识库名称
# RagFlow knowledge base name
RAGFLOW_KB_NAME = f"sdk-test-ragflow-kb-{TIMESTAMP}"

# RagFlow 服务地址，请替换为您的实际值
# RagFlow service URL, please replace with your actual value
RAGFLOW_BASE_URL = os.getenv(
    "RAGFLOW_BASE_URL", "https://your-ragflow-server.com"
)

# RagFlow 数据集 ID 列表，请替换为您的实际值
# RagFlow dataset ID list, please replace with your actual values
RAGFLOW_DATASET_IDS = os.getenv(
    "RAGFLOW_DATASET_IDS", "dataset-id-1,dataset-id-2"
).split(",")

# RagFlow API Key 凭证名称（需要先在 AgentRun 中创建凭证）
# RagFlow API Key credential name (need to create credential in AgentRun first)
RAGFLOW_CREDENTIAL_NAME = os.getenv(
    "RAGFLOW_CREDENTIAL_NAME", "ragflow-api-key"
)

# -----------------------------------------------------------------------------
# ADB 知识库配置 / ADB Knowledge Base Configuration
# -----------------------------------------------------------------------------

# ADB 知识库名称
# ADB knowledge base name
ADB_KB_NAME = f"sdk-test-adb-kb-{TIMESTAMP}"

# ADB 实例 ID，请替换为您的实际值
# ADB instance ID, please replace with your actual value
ADB_INSTANCE_ID = os.getenv("ADB_INSTANCE_ID", "gp-your-instance-id")

# ADB 命名空间，默认为 public
# ADB namespace, defaults to public
ADB_NAMESPACE = os.getenv("ADB_NAMESPACE", "public")

# ADB 命名空间密码，请替换为您的实际值
# ADB namespace password, please replace with your actual value
ADB_NAMESPACE_PASSWORD = os.getenv("ADB_NAMESPACE_PASSWORD", "your-password")

# ADB 文档集合名称，请替换为您的实际值
# ADB collection name, please replace with your actual value
ADB_COLLECTION = os.getenv("ADB_COLLECTION", "your-collection")

# ADB 向量化模型名称（可选）
# ADB embedding model name (optional)
ADB_EMBEDDING_MODEL = os.getenv("ADB_EMBEDDING_MODEL", "text-embedding-v3")

# -----------------------------------------------------------------------------
# OTS 知识库配置 / OTS Knowledge Base Configuration
# -----------------------------------------------------------------------------

# OTS 知识库名称
# OTS knowledge base name
OTS_KB_NAME = os.getenv("OTS_KB_NAME", f"sdk-test-ots-kb-{TIMESTAMP}")

# OTS 实例名称，请替换为您的实际值
# OTS instance name, please replace with your actual value
OTS_INSTANCE_NAME = os.getenv("OTS_INSTANCE_NAME", "your-ots-instance-name")

# OTS 向量化服务提供商
# OTS embedding provider
OTS_EMBEDDING_PROVIDER = os.getenv("OTS_EMBEDDING_PROVIDER", "bailian")

# OTS 向量化模型名称
# OTS embedding model name
OTS_EMBEDDING_MODEL = os.getenv("OTS_EMBEDDING_MODEL", "text-embedding-v3")

# OTS 向量维度
# OTS embedding dimension
OTS_EMBEDDING_DIMENSION = int(os.getenv("OTS_EMBEDDING_DIMENSION", "1024"))

# ============================================================================
# 客户端初始化 / Client Initialization
# ============================================================================

client = KnowledgeBaseClient()


# ============================================================================
# 百炼知识库示例函数 / Bailian Knowledge Base Example Functions
# ============================================================================


def create_or_get_bailian_kb() -> KnowledgeBase:
    """创建或获取已有的百炼知识库 / Create or get existing Bailian knowledge base

    Returns:
        KnowledgeBase: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("创建或获取百炼知识库")
    logger.info("Create or get Bailian knowledge base")
    logger.info("=" * 60)

    try:
        # 创建百炼知识库 / Create Bailian knowledge base
        kb = KnowledgeBase.create(
            KnowledgeBaseCreateInput(
                knowledge_base_name=BAILIAN_KB_NAME,
                description=(
                    "通过 SDK 创建的百炼知识库示例 / Bailian KB example created"
                    " via SDK"
                ),
                provider=KnowledgeBaseProvider.BAILIAN,
                provider_settings=BailianProviderSettings(
                    workspace_id=BAILIAN_WORKSPACE_ID,
                    index_ids=BAILIAN_INDEX_IDS,
                ),
                retrieve_settings=BailianRetrieveSettings(
                    dense_similarity_top_k=50,
                    sparse_similarity_top_k=50,
                    rerank_min_score=0.3,
                    rerank_top_n=5,
                ),
            )
        )
        logger.info("✅ 百炼知识库创建成功 / Bailian KB created successfully")

    except ResourceAlreadyExistError:
        logger.info(
            "ℹ️  百炼知识库已存在，获取已有资源 / Bailian KB exists, getting"
            " existing"
        )
        kb = client.get(BAILIAN_KB_NAME)

    _log_kb_info(kb)
    return kb


def query_bailian_kb(kb: KnowledgeBase):
    """查询百炼知识库 / Query Bailian knowledge base

    Args:
        kb: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("查询百炼知识库")
    logger.info("Query Bailian knowledge base")
    logger.info("=" * 60)

    query_text = "什么是函数计算"
    logger.info("查询文本 / Query text: %s", query_text)

    try:
        results = kb.retrieve(query=query_text)
        logger.info("✅ 查询成功 / Query successful")
        logger.info("检索结果 / Retrieval results: %s", results)
        logger.info(
            "  - 结果数量 / Result count: %s", len(results.get("data", []))
        )
    except Exception as e:
        logger.warning("⚠️  查询失败（可能是凭证或索引配置问题）: %s", e)


def update_bailian_kb(kb: KnowledgeBase):
    """更新百炼知识库配置 / Update Bailian knowledge base configuration

    Args:
        kb: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("更新百炼知识库配置")
    logger.info("Update Bailian knowledge base configuration")
    logger.info("=" * 60)

    new_description = f"[Bailian] 更新于 {time.strftime('%Y-%m-%d %H:%M:%S')}"

    kb.update(
        KnowledgeBaseUpdateInput(
            description=new_description,
            retrieve_settings=BailianRetrieveSettings(
                dense_similarity_top_k=15,
                sparse_similarity_top_k=15,
                rerank_min_score=0.3,
                rerank_top_n=10,
            ),
        )
    )

    logger.info("✅ 百炼知识库更新成功 / Bailian KB updated successfully")
    logger.info("  - 新描述 / New description: %s", kb.description)


def delete_bailian_kb(kb: KnowledgeBase):
    """删除百炼知识库 / Delete Bailian knowledge base

    Args:
        kb: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("删除百炼知识库")
    logger.info("Delete Bailian knowledge base")
    logger.info("=" * 60)

    kb.delete()
    logger.info("✅ 百炼知识库删除请求已发送 / Bailian KB delete request sent")

    try:
        client.get(BAILIAN_KB_NAME)
        logger.warning("⚠️  百炼知识库仍然存在 / Bailian KB still exists")
    except ResourceNotExistError:
        logger.info("✅ 百炼知识库已成功删除 / Bailian KB deleted successfully")


# ============================================================================
# RagFlow 知识库示例函数 / RagFlow Knowledge Base Example Functions
# ============================================================================


def create_or_get_ragflow_kb() -> KnowledgeBase:
    """创建或获取已有的 RagFlow 知识库 / Create or get existing RagFlow knowledge base

    Returns:
        KnowledgeBase: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("创建或获取 RagFlow 知识库")
    logger.info("Create or get RagFlow knowledge base")
    logger.info("=" * 60)

    try:
        # 创建 RagFlow 知识库 / Create RagFlow knowledge base
        kb = KnowledgeBase.create(
            KnowledgeBaseCreateInput(
                knowledge_base_name=RAGFLOW_KB_NAME,
                description=(
                    "通过 SDK 创建的 RagFlow 知识库示例 / RagFlow KB example"
                    " created via SDK"
                ),
                provider=KnowledgeBaseProvider.RAGFLOW,
                provider_settings=RagFlowProviderSettings(
                    base_url=RAGFLOW_BASE_URL,
                    dataset_ids=RAGFLOW_DATASET_IDS,
                ),
                retrieve_settings=RagFlowRetrieveSettings(
                    similarity_threshold=0.5,
                    vector_similarity_weight=0.7,
                    cross_languages=["Chinese", "English"],
                ),
                credential_name=RAGFLOW_CREDENTIAL_NAME,
            )
        )
        logger.info(
            "✅ RagFlow 知识库创建成功 / RagFlow KB created successfully"
        )

    except ResourceAlreadyExistError:
        logger.info(
            "ℹ️  RagFlow 知识库已存在，获取已有资源 / RagFlow KB exists, getting"
            " existing"
        )
        kb = client.get(RAGFLOW_KB_NAME)

    _log_kb_info(kb)
    return kb


def query_ragflow_kb(kb: KnowledgeBase):
    """查询 RagFlow 知识库 / Query RagFlow knowledge base

    Args:
        kb: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("查询 RagFlow 知识库")
    logger.info("Query RagFlow knowledge base")
    logger.info("=" * 60)

    query_text = "What is serverless computing?"
    logger.info("查询文本 / Query text: %s", query_text)

    try:
        results = kb.retrieve(query=query_text)
        logger.info("✅ 查询成功 / Query successful")
        logger.info("检索结果 / Retrieval results: %s", results)

    except Exception as e:
        logger.warning("⚠️  查询失败（可能是凭证或服务配置问题）: %s", e)


def update_ragflow_kb(kb: KnowledgeBase):
    """更新 RagFlow 知识库配置 / Update RagFlow knowledge base configuration

    Args:
        kb: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("更新 RagFlow 知识库配置")
    logger.info("Update RagFlow knowledge base configuration")
    logger.info("=" * 60)

    new_description = f"[RagFlow] 更新于 {time.strftime('%Y-%m-%d %H:%M:%S')}"

    kb.update(
        KnowledgeBaseUpdateInput(
            description=new_description,
            retrieve_settings=RagFlowRetrieveSettings(
                similarity_threshold=0.3,  # 降低阈值 / Lower threshold
                vector_similarity_weight=0.8,  # 增加向量权重 / Increase vector weight
                cross_languages=["Chinese", "English", "Japanese"],
            ),
        )
    )

    logger.info("✅ RagFlow 知识库更新成功 / RagFlow KB updated successfully")
    logger.info("  - 新描述 / New description: %s", kb.description)


def delete_ragflow_kb(kb: KnowledgeBase):
    """删除 RagFlow 知识库 / Delete RagFlow knowledge base

    Args:
        kb: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("删除 RagFlow 知识库")
    logger.info("Delete RagFlow knowledge base")
    logger.info("=" * 60)

    kb.delete()
    logger.info(
        "✅ RagFlow 知识库删除请求已发送 / RagFlow KB delete request sent"
    )

    try:
        client.get(RAGFLOW_KB_NAME)
        logger.warning("⚠️  RagFlow 知识库仍然存在 / RagFlow KB still exists")
    except ResourceNotExistError:
        logger.info(
            "✅ RagFlow 知识库已成功删除 / RagFlow KB deleted successfully"
        )


# ============================================================================
# ADB 知识库示例函数 / ADB Knowledge Base Example Functions
# ============================================================================


def create_or_get_adb_kb() -> KnowledgeBase:
    """创建或获取已有的 ADB 知识库 / Create or get existing ADB knowledge base

    Returns:
        KnowledgeBase: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("创建或获取 ADB 知识库")
    logger.info("Create or get ADB knowledge base")
    logger.info("=" * 60)

    try:
        # 创建 ADB 知识库 / Create ADB knowledge base
        kb = KnowledgeBase.create(
            KnowledgeBaseCreateInput(
                knowledge_base_name=ADB_KB_NAME,
                description=(
                    "通过 SDK 创建的 ADB 知识库示例 / ADB KB example"
                    " created via SDK"
                ),
                provider=KnowledgeBaseProvider.ADB,
                provider_settings=ADBProviderSettings(
                    db_instance_id=ADB_INSTANCE_ID,
                    namespace=ADB_NAMESPACE,
                    namespace_password=ADB_NAMESPACE_PASSWORD,
                    embedding_model=ADB_EMBEDDING_MODEL,
                    metrics="cosine",  # 使用余弦相似度 / Use cosine similarity
                ),
                retrieve_settings=ADBRetrieveSettings(
                    top_k=10,
                    use_full_text_retrieval=False,  # 仅使用向量检索 / Vector only
                    rerank_factor=2.0,  # 重排序因子 / Rerank factor
                ),
            )
        )
        logger.info("✅ ADB 知识库创建成功 / ADB KB created successfully")

    except ResourceAlreadyExistError:
        logger.info(
            "ℹ️  ADB 知识库已存在，获取已有资源 / ADB KB exists, getting"
            " existing"
        )
        kb = client.get(ADB_KB_NAME)

    _log_kb_info(kb)
    return kb


def query_adb_kb(kb: KnowledgeBase):
    """查询 ADB 知识库 / Query ADB knowledge base

    Args:
        kb: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("查询 ADB 知识库")
    logger.info("Query ADB knowledge base")
    logger.info("=" * 60)

    query_text = "什么是云原生数据库"
    logger.info("查询文本 / Query text: %s", query_text)

    try:
        results = kb.retrieve(query=query_text)
        logger.info("✅ 查询成功 / Query successful")
        logger.info("检索结果 / Retrieval results: %s", results)
        logger.info(
            "  - 结果数量 / Result count: %s", len(results.get("data", []))
        )
    except Exception as e:
        logger.warning("⚠️  查询失败（可能是配置或连接问题）: %s", e)


def query_adb_kb_by_name(knowledgebase_name: str):
    """查询 ADB 知识库 / Query ADB knowledge base
    Args:
        knowledgebase_name: 知识库名称 / Knowledge base name
    """

    try:
        kb = KnowledgeBase.get_by_name(knowledgebase_name)
        results = kb.retrieve(query="什么是云原生数据库")
        logger.info("✅ 查询成功 / Query successful")
        logger.info("检索结果 / Retrieval results: %s", results)
        logger.info(
            "  - 结果数量 / Result count: %s", len(results.get("data", []))
        )
    except Exception as e:
        logger.warning("⚠️  查询失败（可能是配置或连接问题）: %s", e)


def update_adb_kb(kb: KnowledgeBase):
    """更新 ADB 知识库配置 / Update ADB knowledge base configuration

    Args:
        kb: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("更新 ADB 知识库配置")
    logger.info("Update ADB knowledge base configuration")
    logger.info("=" * 60)

    new_description = f"[ADB] 更新于 {time.strftime('%Y-%m-%d %H:%M:%S')}"

    kb.update(
        KnowledgeBaseUpdateInput(
            description=new_description,
            retrieve_settings=ADBRetrieveSettings(
                top_k=20,  # 增加返回数量 / Increase result count
                use_full_text_retrieval=True,  # 启用双路召回 / Enable dual recall
                rerank_factor=3.0,  # 调整重排序因子 / Adjust rerank factor
                hybrid_search="RRF",  # 使用 RRF 混合检索 / Use RRF hybrid search
                hybrid_search_args={"RRF": {"k": 60}},
            ),
        )
    )

    logger.info("✅ ADB 知识库更新成功 / ADB KB updated successfully")
    logger.info("  - 新描述 / New description: %s", kb.description)


def delete_adb_kb(kb: KnowledgeBase):
    """删除 ADB 知识库 / Delete ADB knowledge base

    Args:
        kb: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("删除 ADB 知识库")
    logger.info("Delete ADB knowledge base")
    logger.info("=" * 60)

    kb.delete()
    logger.info("✅ ADB 知识库删除请求已发送 / ADB KB delete request sent")

    try:
        client.get(ADB_KB_NAME)
        logger.warning("⚠️  ADB 知识库仍然存在 / ADB KB still exists")
    except ResourceNotExistError:
        logger.info("✅ ADB 知识库已成功删除 / ADB KB deleted successfully")


# ============================================================================
# OTS 知识库示例函数 / OTS Knowledge Base Example Functions
# ============================================================================


def create_or_get_ots_kb() -> KnowledgeBase:
    """创建或获取已有的 OTS 知识库 / Create or get existing OTS knowledge base

    Returns:
        KnowledgeBase: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("创建或获取 OTS 知识库")
    logger.info("Create or get OTS knowledge base")
    logger.info("=" * 60)

    try:
        kb = KnowledgeBase.create(
            KnowledgeBaseCreateInput(
                knowledge_base_name=OTS_KB_NAME,
                description=(
                    "通过 SDK 创建的 OTS 知识库示例 / OTS KB example"
                    " created via SDK"
                ),
                provider=KnowledgeBaseProvider.OTS,
                provider_settings=OTSProviderSettings(
                    ots_instance_name=OTS_INSTANCE_NAME,
                    tags=["sdk-test", "example"],
                    metadata=[
                        OTSMetadataField(name="source", type="string"),
                        OTSMetadataField(name="category", type="string"),
                        OTSMetadataField(name="score", type="double"),
                    ],
                    embedding_configuration=OTSEmbeddingConfiguration(
                        provider=OTS_EMBEDDING_PROVIDER,
                        model=OTS_EMBEDDING_MODEL,
                        dimension=OTS_EMBEDDING_DIMENSION,
                    ),
                ),
                retrieve_settings=OTSRetrieveSettings(
                    search_type=["DENSE_VECTOR", "FULL_TEXT"],
                    dense_vector_search_configuration=OTSDenseVectorSearchConfig(
                        number_of_results=10,
                    ),
                    full_text_search_configuration=OTSFullTextSearchConfig(
                        number_of_results=10,
                    ),
                    reranking_configuration=OTSRerankingConfig(
                        type="RRF",
                        number_of_results=5,
                        rrf_configuration=OTSRRFConfig(
                            dense_vector_search_weight=1.0,
                            full_text_search_weight=0.5,
                            k=60,
                        ),
                    ),
                ),
            )
        )
        logger.info("✅ OTS 知识库创建成功 / OTS KB created successfully")

    except ResourceAlreadyExistError:
        logger.info(
            "ℹ️  OTS 知识库已存在，获取已有资源 / OTS KB exists, getting"
            " existing"
        )
        kb = client.get(OTS_KB_NAME)

    _log_kb_info(kb)
    return kb


def query_ots_kb(kb: KnowledgeBase):
    """查询 OTS 知识库 / Query OTS knowledge base

    Args:
        kb: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("查询 OTS 知识库")
    logger.info("Query OTS knowledge base")
    logger.info("=" * 60)

    query_text = "什么是云原生"
    logger.info("查询文本 / Query text: %s", query_text)

    try:
        results = kb.retrieve(query=query_text)
        logger.info("✅ 查询成功 / Query successful")
        logger.info("检索结果 / Retrieval results: %s", results)
        logger.info(
            "  - 结果数量 / Result count: %s", len(results.get("data", []))
        )
    except Exception as e:
        logger.warning("⚠️  查询失败（可能是配置或连接问题）: %s", e)


def query_ots_kb_by_name(knowledgebase_name: str):
    """根据名称查询 OTS 知识库 / Query OTS knowledge base by name

    Args:
        knowledgebase_name: 知识库名称 / Knowledge base name
    """
    try:
        kb = KnowledgeBase.get_by_name(knowledgebase_name)
        results = kb.retrieve(query="什么是云原生")
        logger.info("✅ 查询成功 / Query successful")
        logger.info("检索结果 / Retrieval results: %s", results)
        logger.info(
            "  - 结果数量 / Result count: %s", len(results.get("data", []))
        )
    except Exception as e:
        logger.warning("⚠️  查询失败（可能是配置或连接问题）: %s", e)


def update_ots_kb(kb: KnowledgeBase):
    """更新 OTS 知识库配置 / Update OTS knowledge base configuration

    Args:
        kb: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("更新 OTS 知识库配置")
    logger.info("Update OTS knowledge base configuration")
    logger.info("=" * 60)

    new_description = f"[OTS] 更新于 {time.strftime('%Y-%m-%d %H:%M:%S')}"

    kb.update(
        KnowledgeBaseUpdateInput(
            description=new_description,
            retrieve_settings=OTSRetrieveSettings(
                search_type=["DENSE_VECTOR", "FULL_TEXT"],
                dense_vector_search_configuration=OTSDenseVectorSearchConfig(
                    number_of_results=20,
                ),
                full_text_search_configuration=OTSFullTextSearchConfig(
                    number_of_results=20,
                ),
                reranking_configuration=OTSRerankingConfig(
                    type="WEIGHT",
                    number_of_results=10,
                    weight_configuration=OTSWeightConfig(
                        dense_vector_search_weight=0.7,
                        full_text_search_weight=0.3,
                    ),
                ),
            ),
        )
    )

    logger.info("✅ OTS 知识库更新成功 / OTS KB updated successfully")
    logger.info("  - 新描述 / New description: %s", kb.description)


def delete_ots_kb(kb: KnowledgeBase):
    """删除 OTS 知识库 / Delete OTS knowledge base

    Args:
        kb: 知识库对象 / Knowledge base object
    """
    logger.info("=" * 60)
    logger.info("删除 OTS 知识库")
    logger.info("Delete OTS knowledge base")
    logger.info("=" * 60)

    kb.delete()
    logger.info("✅ OTS 知识库删除请求已发送 / OTS KB delete request sent")

    try:
        client.get(OTS_KB_NAME)
        logger.warning("⚠️  OTS 知识库仍然存在 / OTS KB still exists")
    except ResourceNotExistError:
        logger.info("✅ OTS 知识库已成功删除 / OTS KB deleted successfully")


# ============================================================================
# 通用工具函数 / Common Utility Functions
# ============================================================================


def _log_kb_info(kb: KnowledgeBase):
    """打印知识库信息 / Log knowledge base info"""
    logger.info("知识库信息 / Knowledge base info:")
    logger.info("  - 名称 / Name: %s", kb.knowledge_base_name)
    logger.info("  - ID: %s", kb.knowledge_base_id)
    logger.info("  - 提供商 / Provider: %s", kb.provider)
    logger.info("  - 描述 / Description: %s", kb.description)
    logger.info("  - 创建时间 / Created at: %s", kb.created_at)


def list_knowledge_bases():
    """列出所有知识库 / List all knowledge bases"""
    logger.info("=" * 60)
    logger.info("列出所有知识库")
    logger.info("List all knowledge bases")
    logger.info("=" * 60)

    # 列出所有知识库 / List all knowledge bases
    kb_list = client.list()
    logger.info(
        "共有 %d 个知识库 / Total %d knowledge bases:",
        len(kb_list),
        len(kb_list),
    )

    for kb in kb_list:
        logger.info(
            "  - %s (provider: %s)", kb.knowledge_base_name, kb.provider
        )

    # 按 provider 过滤 / Filter by provider
    bailian_list = KnowledgeBase.list_all(
        provider=KnowledgeBaseProvider.BAILIAN.value
    )
    ragflow_list = KnowledgeBase.list_all(
        provider=KnowledgeBaseProvider.RAGFLOW.value
    )
    adb_list = KnowledgeBase.list_all(provider=KnowledgeBaseProvider.ADB.value)
    ots_list = KnowledgeBase.list_all(provider=KnowledgeBaseProvider.OTS.value)
    logger.info("  - 百炼知识库 / Bailian KBs: %d 个", len(bailian_list))
    logger.info("  - RagFlow 知识库 / RagFlow KBs: %d 个", len(ragflow_list))
    logger.info("  - ADB 知识库 / ADB KBs: %d 个", len(adb_list))
    logger.info("  - OTS 知识库 / OTS KBs: %d 个", len(ots_list))


# ============================================================================
# 主示例函数 / Main Example Functions
# ============================================================================


def bailian_example():
    """百炼知识库完整示例 / Complete Bailian knowledge base example"""
    logger.info("")
    logger.info("🔷 百炼知识库示例 / Bailian Knowledge Base Example")
    logger.info("=" * 60)

    # 创建百炼知识库 / Create Bailian KB
    kb = create_or_get_bailian_kb()

    # 查询百炼知识库 / Query Bailian KB
    query_bailian_kb(kb)

    # 更新百炼知识库 / Update Bailian KB
    update_bailian_kb(kb)

    # 删除百炼知识库 / Delete Bailian KB
    delete_bailian_kb(kb)

    logger.info("🔷 百炼知识库示例完成 / Bailian KB Example Complete")
    logger.info("")


def ragflow_example():
    """RagFlow 知识库完整示例 / Complete RagFlow knowledge base example"""
    logger.info("")
    logger.info("🔶 RagFlow 知识库示例 / RagFlow Knowledge Base Example")
    logger.info("=" * 60)

    # 创建 RagFlow 知识库 / Create RagFlow KB
    kb = create_or_get_ragflow_kb()

    # 查询 RagFlow 知识库 / Query RagFlow KB
    query_ragflow_kb(kb)

    # 更新 RagFlow 知识库 / Update RagFlow KB
    update_ragflow_kb(kb)

    # 删除 RagFlow 知识库 / Delete RagFlow KB
    delete_ragflow_kb(kb)

    logger.info("🔶 RagFlow 知识库示例完成 / RagFlow KB Example Complete")
    logger.info("")


def adb_example():
    """ADB 知识库完整示例 / Complete ADB knowledge base example"""
    logger.info("")
    logger.info("🔹 ADB 知识库示例 / ADB Knowledge Base Example")
    logger.info("=" * 60)

    # 创建 ADB 知识库 / Create ADB KB
    kb = create_or_get_adb_kb()

    # 查询 ADB 知识库 / Query ADB KB
    query_adb_kb(kb)

    # 更新 ADB 知识库 / Update ADB KB
    update_adb_kb(kb)

    # 删除 ADB 知识库 / Delete ADB KB
    delete_adb_kb(kb)

    logger.info("🔹 ADB 知识库示例完成 / ADB KB Example Complete")
    logger.info("")


def ots_example():
    """OTS 知识库完整示例 / Complete OTS knowledge base example"""
    logger.info("")
    logger.info("🔸 OTS 知识库示例 / OTS Knowledge Base Example")
    logger.info("=" * 60)

    # 创建 OTS 知识库 / Create OTS KB
    kb = create_or_get_ots_kb()

    # 查询 OTS 知识库 / Query OTS KB
    query_ots_kb(kb)

    # 更新 OTS 知识库 / Update OTS KB
    update_ots_kb(kb)

    # # 删除 OTS 知识库 / Delete OTS KB
    delete_ots_kb(kb)

    logger.info("🔸 OTS 知识库示例完成 / OTS KB Example Complete")
    logger.info("")


def knowledgebase_example():
    """知识库模块完整示例 / Complete knowledge base module example

    演示百炼、RagFlow、ADB 和 OTS 四种知识库的完整操作流程。
    Demonstrates complete operation flow for Bailian, RagFlow, ADB and OTS knowledge bases.
    """
    logger.info("")
    logger.info("🚀 知识库模块示例开始 / KnowledgeBase Module Example Start")
    logger.info("=" * 60)

    # 列出现有知识库 / List existing knowledge bases
    list_knowledge_bases()

    # 百炼知识库示例 / Bailian KB example
    bailian_example()

    # RagFlow 知识库示例 / RagFlow KB example
    ragflow_example()

    # ADB 知识库示例 / ADB KB example
    adb_example()

    # OTS 知识库示例 / OTS KB example
    ots_example()

    # 最终列出知识库 / Final list
    list_knowledge_bases()

    logger.info("🎉 知识库模块示例完成 / KnowledgeBase Module Example Complete")
    logger.info("=" * 60)


def bailian_only_example():
    """仅运行百炼知识库示例 / Run Bailian knowledge base example only"""
    logger.info("🚀 百炼知识库示例 / Bailian KB Example")
    list_knowledge_bases()
    bailian_example()
    list_knowledge_bases()
    logger.info("🎉 完成 / Complete")


def ragflow_only_example():
    """仅运行 RagFlow 知识库示例 / Run RagFlow knowledge base example only"""
    logger.info("🚀 RagFlow 知识库示例 / RagFlow KB Example")
    list_knowledge_bases()
    ragflow_example()
    list_knowledge_bases()
    logger.info("🎉 完成 / Complete")


def adb_only_example():
    """仅运行 ADB 知识库示例 / Run ADB knowledge base example only"""
    logger.info("🚀 ADB 知识库示例 / ADB KB Example")
    list_knowledge_bases()
    adb_example()
    list_knowledge_bases()
    logger.info("🎉 完成 / Complete")


def ots_only_example():
    """仅运行 OTS 知识库示例 / Run OTS knowledge base example only"""
    logger.info("🚀 OTS 知识库示例 / OTS KB Example")
    list_knowledge_bases()
    ots_example()
    list_knowledge_bases()
    logger.info("🎉 完成 / Complete")


def multiple_knowledgebase_query():
    """多知识库检索 / Multi knowledge base retrieval
    根据知识库名称列表进行检索，自动获取各知识库的配置并执行检索。
    Retrieves from multiple knowledge bases by name list, automatically fetching
    configuration for each knowledge base and executing retrieval.
    """
    multi_query_result = KnowledgeBase.multi_retrieve(
        query="什么是Serverless",
        knowledge_base_names=["jingsu-bailian", "logantest"],
    )
    logger.info(
        "多知识库检索结果 / Multi knowledge base retrieval result:\n%s",
        json.dumps(multi_query_result, indent=2, ensure_ascii=False),
    )


def update_ragflow_kb_config():
    """更新 RagFlow 知识库配置 / Update RagFlow knowledge base configuration"""
    kb = KnowledgeBase.get_by_name("sdk-test-ragflow-kb-20260106174023")
    new_kb = kb.update(
        KnowledgeBaseUpdateInput(
            description="[RagFlow] 更新于 2023-01-06 10:00:00",
            retrieve_settings=RagFlowRetrieveSettings(
                similarity_threshold=0.3,  # 降低阈值 / Lower threshold
                vector_similarity_weight=0.8,  # 增加向量权重 / Increase vector weight
                cross_languages=["Chinese"],
            ),
        )
    )
    logger.info("更新后的 RagFlow 知识库 / Updated RagFlow KB:\n%s", new_kb)


if __name__ == "__main__":
    # bailian_only_example()
    # ragflow_only_example()
    # adb_only_example()
    # ots_only_example()
    multiple_knowledgebase_query()
    # query_adb_kb_by_name("")
    # query_ots_kb_by_name("")
    # update_ragflow_kb_config()
