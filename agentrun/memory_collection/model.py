"""MemoryCollection 模型定义 / MemoryCollection Model Definitions

定义记忆集合相关的数据模型和枚举。
Defines data models and enumerations related to memory collections.
"""

from typing import List, Optional

from agentrun.utils.config import Config
from agentrun.utils.model import BaseModel, PageableInput


class EmbedderConfigConfig(BaseModel):
    """嵌入模型内部配置 / Embedder Inner Configuration"""

    model: Optional[str] = None
    """模型名称"""


class EmbedderConfig(BaseModel):
    """嵌入模型配置 / Embedder Configuration"""

    config: Optional[EmbedderConfigConfig] = None
    """配置"""
    model_service_name: Optional[str] = None
    """模型服务名称"""


class LLMConfigConfig(BaseModel):
    """LLM 内部配置 / LLM Inner Configuration"""

    model: Optional[str] = None
    """模型名称"""


class LLMConfig(BaseModel):
    """LLM 配置 / LLM Configuration"""

    config: Optional[LLMConfigConfig] = None
    """配置"""
    model_service_name: Optional[str] = None
    """模型服务名称"""


class NetworkConfiguration(BaseModel):
    """网络配置 / Network Configuration"""

    vpc_id: Optional[str] = None
    """VPC ID"""
    vswitch_ids: Optional[List[str]] = None
    """交换机 ID 列表"""
    security_group_id: Optional[str] = None
    """安全组 ID"""
    network_mode: Optional[str] = None
    """网络模式"""


class VectorStoreConfigConfig(BaseModel):
    """向量存储内部配置 / Vector Store Inner Configuration"""

    endpoint: Optional[str] = None
    """端点"""
    instance_name: Optional[str] = None
    """实例名称"""
    collection_name: Optional[str] = None
    """集合名称"""
    vector_dimension: Optional[int] = None
    """向量维度"""


class VectorStoreConfig(BaseModel):
    """向量存储配置 / Vector Store Configuration"""

    provider: Optional[str] = None
    """提供商"""
    config: Optional[VectorStoreConfigConfig] = None
    """配置"""


class MemoryCollectionMutableProps(BaseModel):
    """MemoryCollection 可变属性"""

    description: Optional[str] = None
    """描述"""
    embedder_config: Optional[EmbedderConfig] = None
    """嵌入模型配置"""
    execution_role_arn: Optional[str] = None
    """执行角色 ARN"""
    llm_config: Optional[LLMConfig] = None
    """LLM 配置"""
    network_configuration: Optional[NetworkConfiguration] = None
    """网络配置"""
    vector_store_config: Optional[VectorStoreConfig] = None
    """向量存储配置"""


class MemoryCollectionImmutableProps(BaseModel):
    """MemoryCollection 不可变属性"""

    memory_collection_name: Optional[str] = None
    """Memory Collection 名称"""
    type: Optional[str] = None
    """类型"""


class MemoryCollectionSystemProps(BaseModel):
    """MemoryCollection 系统属性"""

    memory_collection_id: Optional[str] = None
    """Memory Collection ID"""
    created_at: Optional[str] = None
    """创建时间"""
    last_updated_at: Optional[str] = None
    """最后更新时间"""


class MemoryCollectionCreateInput(
    MemoryCollectionImmutableProps, MemoryCollectionMutableProps
):
    """MemoryCollection 创建输入参数"""

    pass


class MemoryCollectionUpdateInput(MemoryCollectionMutableProps):
    """MemoryCollection 更新输入参数"""

    pass


class MemoryCollectionListInput(PageableInput):
    """MemoryCollection 列表查询输入参数"""

    memory_collection_name: Optional[str] = None
    """Memory Collection 名称"""


class MemoryCollectionListOutput(BaseModel):
    """MemoryCollection 列表输出"""

    memory_collection_id: Optional[str] = None
    memory_collection_name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    created_at: Optional[str] = None
    last_updated_at: Optional[str] = None

    async def to_memory_collection_async(self, config: Optional[Config] = None):
        """转换为完整的 MemoryCollection 对象（异步）"""
        from .client import MemoryCollectionClient

        return await MemoryCollectionClient(config).get_async(
            self.memory_collection_name or "", config=config
        )

    def to_memory_collection(self, config: Optional[Config] = None):
        """转换为完整的 MemoryCollection 对象"""
        from .client import MemoryCollectionClient

        return MemoryCollectionClient(config).get(
            self.memory_collection_name or "", config=config
        )
