"""测试 agentrun.memory_collection.model 模块 / Test agentrun.memory_collection.model module"""

import pytest

from agentrun.memory_collection.model import (
    EmbedderConfig,
    EmbedderConfigConfig,
    LLMConfig,
    LLMConfigConfig,
    MemoryCollectionCreateInput,
    MemoryCollectionListInput,
    MemoryCollectionListOutput,
    MemoryCollectionUpdateInput,
    NetworkConfiguration,
    VectorStoreConfig,
    VectorStoreConfigConfig,
)


class TestEmbedderConfigConfig:
    """测试 EmbedderConfigConfig 模型"""

    def test_create_embedder_config_config(self):
        """测试创建嵌入模型内部配置"""
        config = EmbedderConfigConfig(model="text-embedding-3-small")
        assert config.model == "text-embedding-3-small"

    def test_embedder_config_config_optional(self):
        """测试嵌入模型内部配置可选字段"""
        config = EmbedderConfigConfig()
        assert config.model is None


class TestEmbedderConfig:
    """测试 EmbedderConfig 模型"""

    def test_create_embedder_config(self):
        """测试创建嵌入模型配置"""
        config = EmbedderConfig(
            model_service_name="test-embedder",
            config=EmbedderConfigConfig(model="text-embedding-3-small"),
        )
        assert config.model_service_name == "test-embedder"
        assert config.config is not None
        assert config.config.model == "text-embedding-3-small"

    def test_embedder_config_optional(self):
        """测试嵌入模型配置可选字段"""
        config = EmbedderConfig()
        assert config.model_service_name is None
        assert config.config is None


class TestLLMConfigConfig:
    """测试 LLMConfigConfig 模型"""

    def test_create_llm_config_config(self):
        """测试创建 LLM 内部配置"""
        config = LLMConfigConfig(model="gpt-4")
        assert config.model == "gpt-4"

    def test_llm_config_config_optional(self):
        """测试 LLM 内部配置可选字段"""
        config = LLMConfigConfig()
        assert config.model is None


class TestLLMConfig:
    """测试 LLMConfig 模型"""

    def test_create_llm_config(self):
        """测试创建 LLM 配置"""
        config = LLMConfig(
            model_service_name="test-llm",
            config=LLMConfigConfig(model="gpt-4"),
        )
        assert config.model_service_name == "test-llm"
        assert config.config is not None
        assert config.config.model == "gpt-4"

    def test_llm_config_optional(self):
        """测试 LLM 配置可选字段"""
        config = LLMConfig()
        assert config.model_service_name is None
        assert config.config is None


class TestNetworkConfiguration:
    """测试 NetworkConfiguration 模型"""

    def test_create_network_configuration(self):
        """测试创建网络配置"""
        config = NetworkConfiguration(
            vpc_id="vpc-123",
            vswitch_ids=["vsw-123", "vsw-456"],
            security_group_id="sg-123",
            network_mode="vpc",
        )
        assert config.vpc_id == "vpc-123"
        assert config.vswitch_ids == ["vsw-123", "vsw-456"]
        assert config.security_group_id == "sg-123"
        assert config.network_mode == "vpc"

    def test_network_configuration_optional(self):
        """测试网络配置可选字段"""
        config = NetworkConfiguration()
        assert config.vpc_id is None
        assert config.vswitch_ids is None
        assert config.security_group_id is None
        assert config.network_mode is None


class TestVectorStoreConfigConfig:
    """测试 VectorStoreConfigConfig 模型"""

    def test_create_vector_store_config_config(self):
        """测试创建向量存储内部配置"""
        config = VectorStoreConfigConfig(
            endpoint="https://test.dashvector.cn",
            instance_name="test-instance",
            collection_name="test-collection",
            vector_dimension=1536,
        )
        assert config.endpoint == "https://test.dashvector.cn"
        assert config.instance_name == "test-instance"
        assert config.collection_name == "test-collection"
        assert config.vector_dimension == 1536

    def test_vector_store_config_config_optional(self):
        """测试向量存储内部配置可选字段"""
        config = VectorStoreConfigConfig()
        assert config.endpoint is None
        assert config.instance_name is None
        assert config.collection_name is None
        assert config.vector_dimension is None


class TestVectorStoreConfig:
    """测试 VectorStoreConfig 模型"""

    def test_create_vector_store_config(self):
        """测试创建向量存储配置"""
        config = VectorStoreConfig(
            provider="dashvector",
            config=VectorStoreConfigConfig(
                endpoint="https://test.dashvector.cn",
                instance_name="test-instance",
                collection_name="test-collection",
                vector_dimension=1536,
            ),
        )
        assert config.provider == "dashvector"
        assert config.config is not None
        assert config.config.endpoint == "https://test.dashvector.cn"

    def test_vector_store_config_optional(self):
        """测试向量存储配置可选字段"""
        config = VectorStoreConfig()
        assert config.provider is None
        assert config.config is None


class TestMemoryCollectionCreateInput:
    """测试 MemoryCollectionCreateInput 模型"""

    def test_create_minimal_input(self):
        """测试创建最小输入参数"""
        input_obj = MemoryCollectionCreateInput(
            memory_collection_name="test-memory-collection",
            type="vector",
        )
        assert input_obj.memory_collection_name == "test-memory-collection"
        assert input_obj.type == "vector"

    def test_create_full_input(self):
        """测试创建完整输入参数"""
        input_obj = MemoryCollectionCreateInput(
            memory_collection_name="test-memory-collection",
            type="vector",
            description="Test memory collection",
            embedder_config=EmbedderConfig(
                model_service_name="test-embedder",
                config=EmbedderConfigConfig(model="text-embedding-3-small"),
            ),
            llm_config=LLMConfig(
                model_service_name="test-llm",
                config=LLMConfigConfig(model="gpt-4"),
            ),
            vector_store_config=VectorStoreConfig(
                provider="dashvector",
                config=VectorStoreConfigConfig(
                    endpoint="https://test.dashvector.cn",
                    instance_name="test-instance",
                    collection_name="test-collection",
                    vector_dimension=1536,
                ),
            ),
            network_configuration=NetworkConfiguration(
                vpc_id="vpc-123",
                vswitch_ids=["vsw-123"],
                security_group_id="sg-123",
                network_mode="vpc",
            ),
            execution_role_arn="acs:ram::123:role/test",
        )
        assert input_obj.memory_collection_name == "test-memory-collection"
        assert input_obj.description == "Test memory collection"
        assert input_obj.embedder_config is not None
        assert input_obj.llm_config is not None
        assert input_obj.vector_store_config is not None
        assert input_obj.network_configuration is not None
        assert input_obj.execution_role_arn == "acs:ram::123:role/test"

    def test_model_dump(self):
        """测试模型序列化"""
        input_obj = MemoryCollectionCreateInput(
            memory_collection_name="test-memory-collection",
            type="vector",
            description="Test",
        )
        data = input_obj.model_dump()
        # 检查序列化后的数据包含必要字段
        assert (
            "memory_collection_name" in data or "memoryCollectionName" in data
        )
        assert input_obj.memory_collection_name == "test-memory-collection"
        assert input_obj.type == "vector"
        assert input_obj.description == "Test"


class TestMemoryCollectionUpdateInput:
    """测试 MemoryCollectionUpdateInput 模型"""

    def test_create_update_input(self):
        """测试创建更新输入参数"""
        input_obj = MemoryCollectionUpdateInput(
            description="Updated description",
        )
        assert input_obj.description == "Updated description"

    def test_update_input_with_embedder_config(self):
        """测试更新输入参数（带嵌入模型配置）"""
        input_obj = MemoryCollectionUpdateInput(
            embedder_config=EmbedderConfig(
                model_service_name="new-embedder",
                config=EmbedderConfigConfig(model="text-embedding-ada-002"),
            )
        )
        assert input_obj.embedder_config is not None
        assert input_obj.embedder_config.model_service_name == "new-embedder"

    def test_update_input_with_llm_config(self):
        """测试更新输入参数（带 LLM 配置）"""
        input_obj = MemoryCollectionUpdateInput(
            llm_config=LLMConfig(
                model_service_name="new-llm",
                config=LLMConfigConfig(model="gpt-4-turbo"),
            )
        )
        assert input_obj.llm_config is not None
        assert input_obj.llm_config.config is not None
        assert input_obj.llm_config.config.model == "gpt-4-turbo"

    def test_update_input_optional(self):
        """测试更新输入参数可选字段"""
        input_obj = MemoryCollectionUpdateInput()
        assert input_obj.description is None
        assert input_obj.embedder_config is None
        assert input_obj.llm_config is None


class TestMemoryCollectionListInput:
    """测试 MemoryCollectionListInput 模型"""

    def test_create_list_input(self):
        """测试创建列表输入参数"""
        input_obj = MemoryCollectionListInput(
            page_number=1,
            page_size=10,
            memory_collection_name="test-memory-collection",
        )
        assert input_obj.page_number == 1
        assert input_obj.page_size == 10
        assert input_obj.memory_collection_name == "test-memory-collection"

    def test_list_input_default(self):
        """测试列表输入参数默认值"""
        input_obj = MemoryCollectionListInput()
        assert input_obj.memory_collection_name is None

    def test_list_input_with_pagination(self):
        """测试列表输入参数（带分页）"""
        input_obj = MemoryCollectionListInput(
            page_number=2,
            page_size=20,
        )
        assert input_obj.page_number == 2
        assert input_obj.page_size == 20


class TestMemoryCollectionListOutput:
    """测试 MemoryCollectionListOutput 模型"""

    def test_create_list_output(self):
        """测试创建列表输出"""
        output = MemoryCollectionListOutput(
            memory_collection_id="mc-123",
            memory_collection_name="test-memory-collection",
            description="Test memory collection",
            type="vector",
            created_at="2024-01-01T00:00:00Z",
            last_updated_at="2024-01-01T00:00:00Z",
        )
        assert output.memory_collection_id == "mc-123"
        assert output.memory_collection_name == "test-memory-collection"
        assert output.description == "Test memory collection"
        assert output.type == "vector"

    def test_list_output_optional(self):
        """测试列表输出可选字段"""
        output = MemoryCollectionListOutput()
        assert output.memory_collection_id is None
        assert output.memory_collection_name is None
        assert output.description is None
        assert output.type is None
        assert output.created_at is None
        assert output.last_updated_at is None
