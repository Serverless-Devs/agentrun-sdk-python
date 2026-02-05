"""测试 agentrun.knowledgebase.model 模块 / Test agentrun.knowledgebase.model module"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.knowledgebase.model import (
    ADBProviderSettings,
    ADBRetrieveSettings,
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


class TestKnowledgeBaseProvider:
    """测试 KnowledgeBaseProvider 枚举"""

    def test_ragflow_value(self):
        """测试 RAGFLOW 枚举值"""
        assert KnowledgeBaseProvider.RAGFLOW.value == "ragflow"

    def test_bailian_value(self):
        """测试 BAILIAN 枚举值"""
        assert KnowledgeBaseProvider.BAILIAN.value == "bailian"

    def test_adb_value(self):
        """测试 ADB 枚举值"""
        assert KnowledgeBaseProvider.ADB.value == "adb"

    def test_provider_is_string_enum(self):
        """测试 Provider 是字符串枚举"""
        assert isinstance(KnowledgeBaseProvider.RAGFLOW, str)
        assert KnowledgeBaseProvider.RAGFLOW == "ragflow"


class TestRagFlowProviderSettings:
    """测试 RagFlowProviderSettings 模型"""

    def test_create_ragflow_provider_settings(self):
        """测试创建 RagFlow 提供商设置"""
        settings = RagFlowProviderSettings(
            base_url="https://ragflow.example.com",
            dataset_ids=["ds-1", "ds-2"],
        )
        assert settings.base_url == "https://ragflow.example.com"
        assert settings.dataset_ids == ["ds-1", "ds-2"]

    def test_ragflow_provider_settings_required_fields(self):
        """测试 RagFlow 提供商设置必填字段"""
        with pytest.raises(Exception):  # Pydantic ValidationError
            RagFlowProviderSettings()  # type: ignore

    def test_ragflow_provider_settings_model_dump(self):
        """测试 RagFlow 提供商设置序列化"""
        settings = RagFlowProviderSettings(
            base_url="https://ragflow.example.com",
            dataset_ids=["ds-1"],
        )
        data = settings.model_dump()
        assert "base_url" in data or "baseUrl" in data


class TestRagFlowRetrieveSettings:
    """测试 RagFlowRetrieveSettings 模型"""

    def test_create_ragflow_retrieve_settings(self):
        """测试创建 RagFlow 检索设置"""
        settings = RagFlowRetrieveSettings(
            similarity_threshold=0.8,
            vector_similarity_weight=0.5,
            cross_languages=["English", "Chinese"],
        )
        assert settings.similarity_threshold == 0.8
        assert settings.vector_similarity_weight == 0.5
        assert settings.cross_languages == ["English", "Chinese"]

    def test_ragflow_retrieve_settings_optional(self):
        """测试 RagFlow 检索设置可选字段"""
        settings = RagFlowRetrieveSettings()
        assert settings.similarity_threshold is None
        assert settings.vector_similarity_weight is None
        assert settings.cross_languages is None

    def test_ragflow_retrieve_settings_partial(self):
        """测试 RagFlow 检索设置部分字段"""
        settings = RagFlowRetrieveSettings(similarity_threshold=0.7)
        assert settings.similarity_threshold == 0.7
        assert settings.vector_similarity_weight is None


class TestBailianProviderSettings:
    """测试 BailianProviderSettings 模型"""

    def test_create_bailian_provider_settings(self):
        """测试创建百炼提供商设置"""
        settings = BailianProviderSettings(
            workspace_id="ws-123",
            index_ids=["idx-1", "idx-2"],
        )
        assert settings.workspace_id == "ws-123"
        assert settings.index_ids == ["idx-1", "idx-2"]

    def test_bailian_provider_settings_required_fields(self):
        """测试百炼提供商设置必填字段"""
        with pytest.raises(Exception):
            BailianProviderSettings()  # type: ignore

    def test_bailian_provider_settings_single_index(self):
        """测试百炼提供商设置单个索引"""
        settings = BailianProviderSettings(
            workspace_id="ws-123",
            index_ids=["idx-1"],
        )
        assert len(settings.index_ids) == 1


class TestBailianRetrieveSettings:
    """测试 BailianRetrieveSettings 模型"""

    def test_create_bailian_retrieve_settings(self):
        """测试创建百炼检索设置"""
        settings = BailianRetrieveSettings(
            dense_similarity_top_k=10,
            sparse_similarity_top_k=5,
            rerank_min_score=0.5,
            rerank_top_n=3,
        )
        assert settings.dense_similarity_top_k == 10
        assert settings.sparse_similarity_top_k == 5
        assert settings.rerank_min_score == 0.5
        assert settings.rerank_top_n == 3

    def test_bailian_retrieve_settings_optional(self):
        """测试百炼检索设置可选字段"""
        settings = BailianRetrieveSettings()
        assert settings.dense_similarity_top_k is None
        assert settings.sparse_similarity_top_k is None
        assert settings.rerank_min_score is None
        assert settings.rerank_top_n is None

    def test_bailian_retrieve_settings_partial(self):
        """测试百炼检索设置部分字段"""
        settings = BailianRetrieveSettings(
            dense_similarity_top_k=20,
            rerank_top_n=5,
        )
        assert settings.dense_similarity_top_k == 20
        assert settings.rerank_top_n == 5
        assert settings.sparse_similarity_top_k is None


class TestADBProviderSettings:
    """测试 ADBProviderSettings 模型"""

    def test_create_adb_provider_settings(self):
        """测试创建 ADB 提供商设置"""
        settings = ADBProviderSettings(
            db_instance_id="gp-123456",
            namespace="public",
            namespace_password="password123",
            embedding_model="text-embedding-v3",
            metrics="cosine",
            metadata='{"key": "value"}',
        )
        assert settings.db_instance_id == "gp-123456"
        assert settings.namespace == "public"
        assert settings.namespace_password == "password123"
        assert settings.embedding_model == "text-embedding-v3"
        assert settings.metrics == "cosine"
        assert settings.metadata == '{"key": "value"}'

    def test_adb_provider_settings_required_fields(self):
        """测试 ADB 提供商设置必填字段"""
        with pytest.raises(Exception):
            ADBProviderSettings()  # type: ignore

    def test_adb_provider_settings_minimal(self):
        """测试 ADB 提供商设置最小配置"""
        settings = ADBProviderSettings(
            db_instance_id="gp-123456",
            namespace="public",
            namespace_password="password123",
        )
        assert settings.db_instance_id == "gp-123456"
        assert settings.embedding_model is None
        assert settings.metrics is None
        assert settings.metadata is None


class TestADBRetrieveSettings:
    """测试 ADBRetrieveSettings 模型"""

    def test_create_adb_retrieve_settings(self):
        """测试创建 ADB 检索设置"""
        settings = ADBRetrieveSettings(
            top_k=10,
            use_full_text_retrieval=True,
            rerank_factor=1.5,
            recall_window=[-5, 5],
            hybrid_search="RRF",
            hybrid_search_args={"RRF": {"k": 60}},
        )
        assert settings.top_k == 10
        assert settings.use_full_text_retrieval is True
        assert settings.rerank_factor == 1.5
        assert settings.recall_window == [-5, 5]
        assert settings.hybrid_search == "RRF"
        assert settings.hybrid_search_args == {"RRF": {"k": 60}}

    def test_adb_retrieve_settings_optional(self):
        """测试 ADB 检索设置可选字段"""
        settings = ADBRetrieveSettings()
        assert settings.top_k is None
        assert settings.use_full_text_retrieval is None
        assert settings.rerank_factor is None
        assert settings.recall_window is None
        assert settings.hybrid_search is None
        assert settings.hybrid_search_args is None

    def test_adb_retrieve_settings_weight_hybrid(self):
        """测试 ADB 检索设置加权混合检索"""
        settings = ADBRetrieveSettings(
            hybrid_search="Weight",
            hybrid_search_args={"Weight": {"alpha": 0.5}},
        )
        assert settings.hybrid_search == "Weight"
        assert settings.hybrid_search_args["Weight"]["alpha"] == 0.5


class TestKnowledgeBaseMutableProps:
    """测试 KnowledgeBaseMutableProps 模型"""

    def test_create_mutable_props(self):
        """测试创建可变属性"""
        props = KnowledgeBaseMutableProps(
            description="Test description",
            credential_name="test-credential",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            retrieve_settings=RagFlowRetrieveSettings(
                similarity_threshold=0.8,
            ),
        )
        assert props.description == "Test description"
        assert props.credential_name == "test-credential"
        assert isinstance(props.provider_settings, RagFlowProviderSettings)
        assert isinstance(props.retrieve_settings, RagFlowRetrieveSettings)

    def test_mutable_props_optional(self):
        """测试可变属性可选字段"""
        props = KnowledgeBaseMutableProps()
        assert props.description is None
        assert props.credential_name is None
        assert props.provider_settings is None
        assert props.retrieve_settings is None

    def test_mutable_props_with_typed_settings(self):
        """测试可变属性使用类型化设置"""
        provider_settings = RagFlowProviderSettings(
            base_url="https://ragflow.example.com",
            dataset_ids=["ds-1"],
        )
        props = KnowledgeBaseMutableProps(
            provider_settings=provider_settings,
        )
        assert isinstance(props.provider_settings, RagFlowProviderSettings)


class TestKnowledgeBaseImmutableProps:
    """测试 KnowledgeBaseImmutableProps 模型"""

    def test_create_immutable_props(self):
        """测试创建不可变属性"""
        props = KnowledgeBaseImmutableProps(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
        )
        assert props.knowledge_base_name == "test-kb"
        assert props.provider == KnowledgeBaseProvider.RAGFLOW

    def test_immutable_props_optional(self):
        """测试不可变属性可选字段"""
        props = KnowledgeBaseImmutableProps()
        assert props.knowledge_base_name is None
        assert props.provider is None

    def test_immutable_props_with_string_provider(self):
        """测试不可变属性使用字符串提供商"""
        props = KnowledgeBaseImmutableProps(
            knowledge_base_name="test-kb",
            provider="bailian",
        )
        assert props.provider == "bailian"


class TestKnowledgeBaseSystemProps:
    """测试 KnowledgeBaseSystemProps 模型"""

    def test_create_system_props(self):
        """测试创建系统属性"""
        props = KnowledgeBaseSystemProps(
            knowledge_base_id="kb-123",
            created_at="2024-01-01T00:00:00Z",
            last_updated_at="2024-01-02T00:00:00Z",
        )
        assert props.knowledge_base_id == "kb-123"
        assert props.created_at == "2024-01-01T00:00:00Z"
        assert props.last_updated_at == "2024-01-02T00:00:00Z"

    def test_system_props_optional(self):
        """测试系统属性可选字段"""
        props = KnowledgeBaseSystemProps()
        assert props.knowledge_base_id is None
        assert props.created_at is None
        assert props.last_updated_at is None


class TestKnowledgeBaseCreateInput:
    """测试 KnowledgeBaseCreateInput 模型"""

    def test_create_minimal_input(self):
        """测试创建最小输入参数"""
        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
        )
        assert input_obj.knowledge_base_name == "test-kb"
        assert input_obj.provider == KnowledgeBaseProvider.RAGFLOW
        assert isinstance(input_obj.provider_settings, RagFlowProviderSettings)

    def test_create_full_input(self):
        """测试创建完整输入参数"""
        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.BAILIAN,
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
            description="Test knowledge base",
            credential_name="test-credential",
            retrieve_settings=BailianRetrieveSettings(
                dense_similarity_top_k=10,
            ),
        )
        assert input_obj.knowledge_base_name == "test-kb"
        assert input_obj.description == "Test knowledge base"
        assert input_obj.credential_name == "test-credential"

    def test_create_input_with_adb(self):
        """测试创建 ADB 输入参数"""
        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="test-adb-kb",
            provider=KnowledgeBaseProvider.ADB,
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
        )
        assert input_obj.provider == KnowledgeBaseProvider.ADB

    def test_create_input_with_dict_settings(self):
        """测试创建输入参数（字典设置会自动转换为类型化对象）"""
        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="test-kb",
            provider="ragflow",
            provider_settings={
                "base_url": "https://ragflow.example.com",
                "dataset_ids": ["ds-1"],
            },
        )
        assert input_obj.provider == "ragflow"
        # Pydantic 会自动将字典转换为类型化对象
        assert isinstance(input_obj.provider_settings, RagFlowProviderSettings)
        assert (
            input_obj.provider_settings.base_url
            == "https://ragflow.example.com"
        )

    def test_model_dump(self):
        """测试模型序列化"""
        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings={
                "base_url": "https://example.com",
                "dataset_ids": [],
            },
        )
        data = input_obj.model_dump()
        assert input_obj.knowledge_base_name == "test-kb"


class TestKnowledgeBaseUpdateInput:
    """测试 KnowledgeBaseUpdateInput 模型"""

    def test_create_update_input(self):
        """测试创建更新输入参数"""
        input_obj = KnowledgeBaseUpdateInput(
            description="Updated description",
        )
        assert input_obj.description == "Updated description"

    def test_update_input_with_credential(self):
        """测试更新输入参数（带凭证）"""
        input_obj = KnowledgeBaseUpdateInput(
            credential_name="new-credential",
        )
        assert input_obj.credential_name == "new-credential"

    def test_update_input_with_provider_settings(self):
        """测试更新输入参数（带提供商设置）"""
        input_obj = KnowledgeBaseUpdateInput(
            provider_settings=RagFlowProviderSettings(
                base_url="https://new-ragflow.example.com",
                dataset_ids=["ds-new"],
            ),
        )
        assert input_obj.provider_settings is not None

    def test_update_input_with_retrieve_settings(self):
        """测试更新输入参数（带检索设置）"""
        input_obj = KnowledgeBaseUpdateInput(
            retrieve_settings=BailianRetrieveSettings(
                dense_similarity_top_k=20,
            ),
        )
        assert input_obj.retrieve_settings is not None

    def test_update_input_optional(self):
        """测试更新输入参数可选字段"""
        input_obj = KnowledgeBaseUpdateInput()
        assert input_obj.description is None
        assert input_obj.credential_name is None
        assert input_obj.provider_settings is None
        assert input_obj.retrieve_settings is None


class TestKnowledgeBaseListInput:
    """测试 KnowledgeBaseListInput 模型"""

    def test_create_list_input(self):
        """测试创建列表输入参数"""
        input_obj = KnowledgeBaseListInput(
            page_number=1,
            page_size=10,
            provider=KnowledgeBaseProvider.RAGFLOW,
        )
        assert input_obj.page_number == 1
        assert input_obj.page_size == 10
        assert input_obj.provider == KnowledgeBaseProvider.RAGFLOW

    def test_list_input_default(self):
        """测试列表输入参数默认值"""
        input_obj = KnowledgeBaseListInput()
        assert input_obj.provider is None

    def test_list_input_with_pagination(self):
        """测试列表输入参数（带分页）"""
        input_obj = KnowledgeBaseListInput(
            page_number=2,
            page_size=20,
        )
        assert input_obj.page_number == 2
        assert input_obj.page_size == 20

    def test_list_input_with_string_provider(self):
        """测试列表输入参数（字符串提供商）"""
        input_obj = KnowledgeBaseListInput(
            provider="bailian",
        )
        assert input_obj.provider == "bailian"


class TestKnowledgeBaseListOutput:
    """测试 KnowledgeBaseListOutput 模型"""

    def test_create_list_output(self):
        """测试创建列表输出"""
        output = KnowledgeBaseListOutput(
            knowledge_base_id="kb-123",
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            description="Test knowledge base",
            credential_name="test-credential",
            created_at="2024-01-01T00:00:00Z",
            last_updated_at="2024-01-01T00:00:00Z",
        )
        assert output.knowledge_base_id == "kb-123"
        assert output.knowledge_base_name == "test-kb"
        assert output.provider == KnowledgeBaseProvider.RAGFLOW
        assert output.description == "Test knowledge base"

    def test_list_output_optional(self):
        """测试列表输出可选字段"""
        output = KnowledgeBaseListOutput()
        assert output.knowledge_base_id is None
        assert output.knowledge_base_name is None
        assert output.provider is None
        assert output.description is None
        assert output.credential_name is None
        assert output.created_at is None
        assert output.last_updated_at is None

    def test_list_output_with_settings(self):
        """测试列表输出带设置（字典会自动转换为类型化对象）"""
        output = KnowledgeBaseListOutput(
            knowledge_base_id="kb-123",
            knowledge_base_name="test-kb",
            provider="bailian",
            provider_settings={
                "workspace_id": "ws-123",
                "index_ids": ["idx-1"],
            },
            retrieve_settings={"dense_similarity_top_k": 10},
        )
        # Pydantic 会自动将字典转换为类型化对象
        assert isinstance(output.provider_settings, BailianProviderSettings)
        assert output.provider_settings.workspace_id == "ws-123"
        assert isinstance(output.retrieve_settings, BailianRetrieveSettings)
        assert output.retrieve_settings.dense_similarity_top_k == 10

    @patch("agentrun.knowledgebase.client.KnowledgeBaseClient")
    def test_to_knowledge_base_sync(self, mock_client_class):
        """测试同步转换为 KnowledgeBase 对象"""
        mock_client = MagicMock()
        mock_kb = MagicMock()
        mock_client.get.return_value = mock_kb
        mock_client_class.return_value = mock_client

        output = KnowledgeBaseListOutput(
            knowledge_base_id="kb-123",
            knowledge_base_name="test-kb",
        )

        result = output.to_knowledge_base()
        assert result == mock_kb
        mock_client.get.assert_called_once()

    @patch("agentrun.knowledgebase.client.KnowledgeBaseClient")
    @pytest.mark.asyncio
    async def test_to_knowledge_base_async(self, mock_client_class):
        """测试异步转换为 KnowledgeBase 对象"""
        mock_client = MagicMock()
        mock_kb = MagicMock()
        mock_client.get_async = AsyncMock(return_value=mock_kb)
        mock_client_class.return_value = mock_client

        output = KnowledgeBaseListOutput(
            knowledge_base_id="kb-123",
            knowledge_base_name="test-kb",
        )

        result = await output.to_knowledge_base_async()
        assert result == mock_kb


class TestRetrieveInput:
    """测试 RetrieveInput 模型"""

    def test_create_retrieve_input(self):
        """测试创建检索输入参数"""
        input_obj = RetrieveInput(
            knowledge_base_names=["kb-1", "kb-2"],
            query="What is AI?",
        )
        assert input_obj.knowledge_base_names == ["kb-1", "kb-2"]
        assert input_obj.query == "What is AI?"

    def test_retrieve_input_with_optional_fields(self):
        """测试检索输入参数可选字段"""
        input_obj = RetrieveInput(
            knowledge_base_names=["kb-1"],
            query="Test query",
            knowledge_base_id="kb-123",
            knowledge_base_name="test-kb",
            provider="ragflow",
            description="Test description",
            credential_name="test-credential",
            created_at="2024-01-01T00:00:00Z",
            last_updated_at="2024-01-01T00:00:00Z",
        )
        assert input_obj.knowledge_base_id == "kb-123"
        assert input_obj.knowledge_base_name == "test-kb"
        assert input_obj.provider == "ragflow"

    def test_retrieve_input_default_optional(self):
        """测试检索输入参数默认可选值"""
        input_obj = RetrieveInput(
            knowledge_base_names=["kb-1"],
            query="Test query",
        )
        assert input_obj.knowledge_base_id is None
        assert input_obj.knowledge_base_name is None
        assert input_obj.provider is None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseClient")
    def test_retrieve_input_to_knowledge_base_sync(self, mock_client_class):
        """测试检索输入同步转换为 KnowledgeBase"""
        mock_client = MagicMock()
        mock_kb = MagicMock()
        mock_client.get.return_value = mock_kb
        mock_client_class.return_value = mock_client

        input_obj = RetrieveInput(
            knowledge_base_names=["kb-1"],
            query="Test query",
            knowledge_base_name="test-kb",
        )

        result = input_obj.to_knowledge_base()
        assert result == mock_kb

    @patch("agentrun.knowledgebase.client.KnowledgeBaseClient")
    @pytest.mark.asyncio
    async def test_retrieve_input_to_knowledge_base_async(
        self, mock_client_class
    ):
        """测试检索输入异步转换为 KnowledgeBase"""
        mock_client = MagicMock()
        mock_kb = MagicMock()
        mock_client.get_async = AsyncMock(return_value=mock_kb)
        mock_client_class.return_value = mock_client

        input_obj = RetrieveInput(
            knowledge_base_names=["kb-1"],
            query="Test query",
            knowledge_base_name="test-kb",
        )

        result = await input_obj.to_knowledge_base_async()
        assert result == mock_kb
