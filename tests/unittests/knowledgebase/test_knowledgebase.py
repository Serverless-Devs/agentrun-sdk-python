"""测试 agentrun.knowledgebase.knowledgebase 模块 / Test agentrun.knowledgebase.knowledgebase module"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.knowledgebase.knowledgebase import KnowledgeBase
from agentrun.knowledgebase.model import (
    ADBProviderSettings,
    ADBRetrieveSettings,
    BailianProviderSettings,
    BailianRetrieveSettings,
    KnowledgeBaseCreateInput,
    KnowledgeBaseProvider,
    KnowledgeBaseUpdateInput,
    RagFlowProviderSettings,
    RagFlowRetrieveSettings,
)
from agentrun.utils.config import Config


class MockKnowledgeBaseData:
    """模拟知识库数据"""

    def to_map(self):
        return {
            "knowledgeBaseId": "kb-123",
            "knowledgeBaseName": "test-kb",
            "provider": "ragflow",
            "description": "Test knowledge base",
            "credentialName": "test-credential",
            "providerSettings": {
                "baseUrl": "https://ragflow.example.com",
                "datasetIds": ["ds-1"],
            },
            "retrieveSettings": {
                "similarityThreshold": 0.8,
            },
            "createdAt": "2024-01-01T00:00:00Z",
            "lastUpdatedAt": "2024-01-01T00:00:00Z",
        }


class MockBailianKnowledgeBaseData:
    """模拟百炼知识库数据"""

    def to_map(self):
        return {
            "knowledgeBaseId": "kb-456",
            "knowledgeBaseName": "test-bailian-kb",
            "provider": "bailian",
            "description": "Test Bailian knowledge base",
            "providerSettings": {
                "workspaceId": "ws-123",
                "indexIds": ["idx-1"],
            },
            "retrieveSettings": {
                "denseSimilarityTopK": 10,
            },
            "createdAt": "2024-01-01T00:00:00Z",
            "lastUpdatedAt": "2024-01-01T00:00:00Z",
        }


class MockADBKnowledgeBaseData:
    """模拟 ADB 知识库数据"""

    def to_map(self):
        return {
            "knowledgeBaseId": "kb-789",
            "knowledgeBaseName": "test-adb-kb",
            "provider": "adb",
            "description": "Test ADB knowledge base",
            "providerSettings": {
                "DBInstanceId": "gp-123456",
                "Namespace": "public",
                "NamespacePassword": "password123",
            },
            "retrieveSettings": {
                "TopK": 10,
            },
            "createdAt": "2024-01-01T00:00:00Z",
            "lastUpdatedAt": "2024-01-01T00:00:00Z",
        }


class MockListResult:
    """模拟列表结果"""

    def __init__(self, items):
        self.items = items


class TestKnowledgeBaseCreate:
    """测试 KnowledgeBase.create 方法"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_create_sync(self, mock_control_api_class):
        """测试同步创建知识库"""
        mock_control_api = MagicMock()
        mock_control_api.create_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            description="Test knowledge base",
        )

        result = KnowledgeBase.create(input_obj)
        assert result.knowledge_base_name == "test-kb"
        assert result.knowledge_base_id == "kb-123"

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_create_async(self, mock_control_api_class):
        """测试异步创建知识库"""
        mock_control_api = MagicMock()
        mock_control_api.create_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
        )

        result = await KnowledgeBase.create_async(input_obj)
        assert result.knowledge_base_name == "test-kb"


class TestKnowledgeBaseDelete:
    """测试 KnowledgeBase.delete 方法"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_delete_by_name_sync(self, mock_control_api_class):
        """测试根据名称同步删除知识库"""
        mock_control_api = MagicMock()
        mock_control_api.delete_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = KnowledgeBase.delete_by_name("test-kb")
        assert result is not None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_delete_by_name_async(self, mock_control_api_class):
        """测试根据名称异步删除知识库"""
        mock_control_api = MagicMock()
        mock_control_api.delete_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = await KnowledgeBase.delete_by_name_async("test-kb")
        assert result is not None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_delete_instance_sync(self, mock_control_api_class):
        """测试实例同步删除"""
        mock_control_api = MagicMock()
        mock_control_api.delete_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api.get_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        # 先获取实例
        kb = KnowledgeBase.get_by_name("test-kb")

        # 删除实例
        result = kb.delete()
        assert result is not None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_delete_instance_async(self, mock_control_api_class):
        """测试实例异步删除"""
        mock_control_api = MagicMock()
        mock_control_api.delete_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api.get_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        # 先获取实例
        kb = await KnowledgeBase.get_by_name_async("test-kb")

        # 删除实例
        result = await kb.delete_async()
        assert result is not None

    def test_delete_instance_without_name(self):
        """测试实例删除（无名称）"""
        kb = KnowledgeBase()
        with pytest.raises(ValueError, match="knowledge_base_name is required"):
            kb.delete()

    @pytest.mark.asyncio
    async def test_delete_instance_async_without_name(self):
        """测试异步实例删除（无名称）"""
        kb = KnowledgeBase()
        with pytest.raises(ValueError, match="knowledge_base_name is required"):
            await kb.delete_async()


class TestKnowledgeBaseUpdate:
    """测试 KnowledgeBase.update 方法"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_update_by_name_sync(self, mock_control_api_class):
        """测试根据名称同步更新知识库"""
        mock_control_api = MagicMock()
        mock_control_api.update_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        input_obj = KnowledgeBaseUpdateInput(description="Updated description")
        result = KnowledgeBase.update_by_name("test-kb", input_obj)
        assert result is not None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_update_by_name_async(self, mock_control_api_class):
        """测试根据名称异步更新知识库"""
        mock_control_api = MagicMock()
        mock_control_api.update_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        input_obj = KnowledgeBaseUpdateInput(description="Updated")
        result = await KnowledgeBase.update_by_name_async("test-kb", input_obj)
        assert result is not None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_update_instance_sync(self, mock_control_api_class):
        """测试实例同步更新"""
        mock_control_api = MagicMock()
        mock_control_api.update_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api.get_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        # 先获取实例
        kb = KnowledgeBase.get_by_name("test-kb")

        # 更新实例
        input_obj = KnowledgeBaseUpdateInput(description="Updated")
        result = kb.update(input_obj)
        assert result is not None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_update_instance_async(self, mock_control_api_class):
        """测试实例异步更新"""
        mock_control_api = MagicMock()
        mock_control_api.update_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api.get_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        # 先获取实例
        kb = await KnowledgeBase.get_by_name_async("test-kb")

        # 更新实例
        input_obj = KnowledgeBaseUpdateInput(description="Updated")
        result = await kb.update_async(input_obj)
        assert result is not None

    def test_update_instance_without_name(self):
        """测试实例更新（无名称）"""
        kb = KnowledgeBase()
        input_obj = KnowledgeBaseUpdateInput(description="Updated")
        with pytest.raises(ValueError, match="knowledge_base_name is required"):
            kb.update(input_obj)

    @pytest.mark.asyncio
    async def test_update_instance_async_without_name(self):
        """测试异步实例更新（无名称）"""
        kb = KnowledgeBase()
        input_obj = KnowledgeBaseUpdateInput(description="Updated")
        with pytest.raises(ValueError, match="knowledge_base_name is required"):
            await kb.update_async(input_obj)


class TestKnowledgeBaseGet:
    """测试 KnowledgeBase.get_by_name 方法"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_get_by_name_sync(self, mock_control_api_class):
        """测试同步获取知识库"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = KnowledgeBase.get_by_name("test-kb")
        assert result.knowledge_base_name == "test-kb"
        assert result.knowledge_base_id == "kb-123"

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_get_by_name_async(self, mock_control_api_class):
        """测试异步获取知识库"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = await KnowledgeBase.get_by_name_async("test-kb")
        assert result.knowledge_base_name == "test-kb"


class TestKnowledgeBaseRefresh:
    """测试 KnowledgeBase.refresh 方法"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_refresh_sync(self, mock_control_api_class):
        """测试同步刷新知识库"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        # 先获取实例
        kb = KnowledgeBase.get_by_name("test-kb")

        # 刷新实例
        kb.refresh()
        assert kb.knowledge_base_name == "test-kb"

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_refresh_async(self, mock_control_api_class):
        """测试异步刷新知识库"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        # 先获取实例
        kb = await KnowledgeBase.get_by_name_async("test-kb")

        # 刷新实例
        await kb.refresh_async()
        assert kb.knowledge_base_name == "test-kb"

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_get_instance_sync(self, mock_control_api_class):
        """测试实例 get 方法同步"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        kb = KnowledgeBase.get_by_name("test-kb")
        result = kb.get()
        assert result.knowledge_base_name == "test-kb"

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_get_instance_async(self, mock_control_api_class):
        """测试实例 get 方法异步"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        kb = await KnowledgeBase.get_by_name_async("test-kb")
        result = await kb.get_async()
        assert result.knowledge_base_name == "test-kb"

    def test_get_instance_without_name(self):
        """测试实例获取（无名称）"""
        kb = KnowledgeBase()
        with pytest.raises(ValueError, match="knowledge_base_name is required"):
            kb.get()

    @pytest.mark.asyncio
    async def test_get_instance_async_without_name(self):
        """测试异步实例获取（无名称）"""
        kb = KnowledgeBase()
        with pytest.raises(ValueError, match="knowledge_base_name is required"):
            await kb.get_async()


class TestKnowledgeBaseList:
    """测试 KnowledgeBase.list_all 方法"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_list_all_sync(self, mock_control_api_class):
        """测试同步列出知识库"""
        mock_control_api = MagicMock()
        mock_control_api.list_knowledge_bases.return_value = MockListResult([
            MockKnowledgeBaseData(),
            MockBailianKnowledgeBaseData(),
        ])
        mock_control_api_class.return_value = mock_control_api

        result = KnowledgeBase.list_all()
        # list_all 会对结果去重，所以相同 ID 的记录只会返回一个
        assert len(result) >= 1

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_list_all_async(self, mock_control_api_class):
        """测试异步列出知识库"""
        mock_control_api = MagicMock()
        mock_control_api.list_knowledge_bases_async = AsyncMock(
            return_value=MockListResult([MockKnowledgeBaseData()])
        )
        mock_control_api_class.return_value = mock_control_api

        result = await KnowledgeBase.list_all_async()
        assert len(result) == 1

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_list_all_with_provider(self, mock_control_api_class):
        """测试按提供商列出知识库"""
        mock_control_api = MagicMock()
        mock_control_api.list_knowledge_bases.return_value = MockListResult(
            [MockKnowledgeBaseData()]
        )
        mock_control_api_class.return_value = mock_control_api

        result = KnowledgeBase.list_all(provider="ragflow")
        assert len(result) >= 1


class TestKnowledgeBaseFromInnerObject:
    """测试 KnowledgeBase.from_inner_object 方法"""

    def test_from_inner_object(self):
        """测试从内部对象创建知识库"""
        mock_data = MockKnowledgeBaseData()
        kb = KnowledgeBase.from_inner_object(mock_data)

        assert kb.knowledge_base_id == "kb-123"
        assert kb.knowledge_base_name == "test-kb"
        assert kb.provider == "ragflow"
        assert kb.description == "Test knowledge base"

    def test_from_inner_object_with_extra(self):
        """测试从内部对象创建知识库（带额外字段）"""
        mock_data = MockKnowledgeBaseData()
        extra = {"custom_field": "custom_value"}
        kb = KnowledgeBase.from_inner_object(mock_data, extra)

        assert kb.knowledge_base_name == "test-kb"

    def test_from_inner_object_bailian(self):
        """测试从内部对象创建百炼知识库"""
        mock_data = MockBailianKnowledgeBaseData()
        kb = KnowledgeBase.from_inner_object(mock_data)

        assert kb.knowledge_base_id == "kb-456"
        assert kb.knowledge_base_name == "test-bailian-kb"
        assert kb.provider == "bailian"

    def test_from_inner_object_adb(self):
        """测试从内部对象创建 ADB 知识库"""
        mock_data = MockADBKnowledgeBaseData()
        kb = KnowledgeBase.from_inner_object(mock_data)

        assert kb.knowledge_base_id == "kb-789"
        assert kb.knowledge_base_name == "test-adb-kb"
        assert kb.provider == "adb"


class TestKnowledgeBaseGetDataAPI:
    """测试 KnowledgeBase._get_data_api 方法"""

    def test_get_data_api_ragflow(self):
        """测试获取 RagFlow 数据链路 API"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            retrieve_settings=RagFlowRetrieveSettings(
                similarity_threshold=0.8,
            ),
            credential_name="test-credential",
        )

        from agentrun.knowledgebase.api.data import RagFlowDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, RagFlowDataAPI)

    def test_get_data_api_bailian(self):
        """测试获取百炼数据链路 API"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.BAILIAN,
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
            retrieve_settings=BailianRetrieveSettings(
                dense_similarity_top_k=10,
            ),
        )

        from agentrun.knowledgebase.api.data import BailianDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, BailianDataAPI)

    def test_get_data_api_adb(self):
        """测试获取 ADB 数据链路 API"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.ADB,
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
            retrieve_settings=ADBRetrieveSettings(
                top_k=10,
            ),
        )

        from agentrun.knowledgebase.api.data import ADBDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, ADBDataAPI)

    def test_get_data_api_with_dict_settings(self):
        """测试使用字典设置获取数据链路 API"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings={
                "base_url": "https://ragflow.example.com",
                "dataset_ids": ["ds-1"],
            },
            retrieve_settings={
                "similarity_threshold": 0.8,
            },
            credential_name="test-credential",
        )

        from agentrun.knowledgebase.api.data import RagFlowDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, RagFlowDataAPI)

    def test_get_data_api_bailian_with_dict_settings(self):
        """测试百炼使用字典设置获取数据链路 API"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.BAILIAN,
            provider_settings={
                "workspace_id": "ws-123",
                "index_ids": ["idx-1"],
            },
            retrieve_settings={
                "dense_similarity_top_k": 10,
            },
        )

        from agentrun.knowledgebase.api.data import BailianDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, BailianDataAPI)

    def test_get_data_api_adb_with_dict_settings(self):
        """测试 ADB 使用字典设置获取数据链路 API（PascalCase 键名）"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.ADB,
            provider_settings={
                "DBInstanceId": "gp-123456",
                "Namespace": "public",
                "NamespacePassword": "password123",
            },
            retrieve_settings={
                "TopK": 10,
            },
        )

        from agentrun.knowledgebase.api.data import ADBDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, ADBDataAPI)

    def test_get_data_api_bailian_with_raw_dict_settings(self):
        """测试百炼使用原始字典设置获取数据链路 API（绕过 Pydantic 转换）"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.BAILIAN,
        )
        # Bypass Pydantic validation to set raw dict
        object.__setattr__(
            kb,
            "provider_settings",
            {
                "workspace_id": "ws-123",
                "index_ids": ["idx-1"],
            },
        )
        object.__setattr__(
            kb,
            "retrieve_settings",
            {
                "dense_similarity_top_k": 10,
            },
        )

        from agentrun.knowledgebase.api.data import BailianDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, BailianDataAPI)

    def test_get_data_api_ragflow_with_raw_dict_settings(self):
        """测试 RagFlow 使用原始字典设置获取数据链路 API（绕过 Pydantic 转换）"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            credential_name="test-credential",
        )
        # Bypass Pydantic validation to set raw dict
        object.__setattr__(
            kb,
            "provider_settings",
            {
                "base_url": "https://ragflow.example.com",
                "dataset_ids": ["ds-1"],
            },
        )
        object.__setattr__(
            kb,
            "retrieve_settings",
            {
                "similarity_threshold": 0.8,
            },
        )

        from agentrun.knowledgebase.api.data import RagFlowDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, RagFlowDataAPI)

    def test_get_data_api_adb_with_raw_dict_settings(self):
        """测试 ADB 使用原始字典设置获取数据链路 API（绕过 Pydantic 转换，PascalCase 键名）"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.ADB,
        )
        # Bypass Pydantic validation to set raw dict with PascalCase keys
        object.__setattr__(
            kb,
            "provider_settings",
            {
                "DBInstanceId": "gp-123456",
                "Namespace": "public",
                "NamespacePassword": "password123",
                "EmbeddingModel": "text-embedding-v1",
                "Metrics": "cosine",
                "Metadata": '{"key": "value"}',
            },
        )
        object.__setattr__(
            kb,
            "retrieve_settings",
            {
                "TopK": 10,
                "UseFullTextRetrieval": True,
                "RerankFactor": 1.5,
                "RecallWindow": [-5, 5],
                "HybridSearch": "RRF",
                "HybridSearchArgs": {"RRF": {"k": 60}},
            },
        )

        from agentrun.knowledgebase.api.data import ADBDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, ADBDataAPI)

    def test_get_data_api_without_provider(self):
        """测试获取数据链路 API（无提供商）"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
        )

        with pytest.raises(ValueError, match="provider is required"):
            kb._get_data_api()

    def test_get_data_api_with_string_provider(self):
        """测试使用字符串提供商获取数据链路 API"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider="ragflow",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        from agentrun.knowledgebase.api.data import RagFlowDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, RagFlowDataAPI)

    def test_get_data_api_bailian_without_settings(self):
        """测试百炼无设置获取数据链路 API"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.BAILIAN,
        )

        from agentrun.knowledgebase.api.data import BailianDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, BailianDataAPI)
        assert data_api.provider_settings is None
        assert data_api.retrieve_settings is None

    def test_get_data_api_bailian_without_retrieve_settings(self):
        """测试百炼无检索设置获取数据链路 API"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.BAILIAN,
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
        )

        from agentrun.knowledgebase.api.data import BailianDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, BailianDataAPI)
        assert data_api.provider_settings is not None
        assert data_api.retrieve_settings is None

    def test_get_data_api_ragflow_without_settings(self):
        """测试 RagFlow 无设置获取数据链路 API"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            credential_name="test-credential",
        )

        from agentrun.knowledgebase.api.data import RagFlowDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, RagFlowDataAPI)
        assert data_api.provider_settings is None
        assert data_api.retrieve_settings is None

    def test_get_data_api_ragflow_without_retrieve_settings(self):
        """测试 RagFlow 无检索设置获取数据链路 API"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        from agentrun.knowledgebase.api.data import RagFlowDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, RagFlowDataAPI)
        assert data_api.provider_settings is not None
        assert data_api.retrieve_settings is None

    def test_get_data_api_adb_without_settings(self):
        """测试 ADB 无设置获取数据链路 API"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.ADB,
        )

        from agentrun.knowledgebase.api.data import ADBDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, ADBDataAPI)
        assert data_api.provider_settings is None
        assert data_api.retrieve_settings is None

    def test_get_data_api_adb_without_retrieve_settings(self):
        """测试 ADB 无检索设置获取数据链路 API"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.ADB,
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
        )

        from agentrun.knowledgebase.api.data import ADBDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, ADBDataAPI)
        assert data_api.provider_settings is not None
        assert data_api.retrieve_settings is None

    def test_get_data_api_bailian_with_invalid_provider_settings_type(self):
        """测试百炼使用无效类型的 provider_settings"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.BAILIAN,
        )
        # Set an invalid type (not BailianProviderSettings or dict)
        object.__setattr__(kb, "provider_settings", "invalid")

        from agentrun.knowledgebase.api.data import BailianDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, BailianDataAPI)
        # converted_provider_settings should be None as the type is invalid
        assert data_api.provider_settings is None

    def test_get_data_api_bailian_with_invalid_retrieve_settings_type(self):
        """测试百炼使用无效类型的 retrieve_settings"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.BAILIAN,
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
        )
        # Set an invalid type (not BailianRetrieveSettings or dict)
        object.__setattr__(kb, "retrieve_settings", "invalid")

        from agentrun.knowledgebase.api.data import BailianDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, BailianDataAPI)
        assert data_api.provider_settings is not None
        # converted_retrieve_settings should be None as the type is invalid
        assert data_api.retrieve_settings is None

    def test_get_data_api_ragflow_with_invalid_provider_settings_type(self):
        """测试 RagFlow 使用无效类型的 provider_settings"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            credential_name="test-credential",
        )
        # Set an invalid type (not RagFlowProviderSettings or dict)
        object.__setattr__(kb, "provider_settings", "invalid")

        from agentrun.knowledgebase.api.data import RagFlowDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, RagFlowDataAPI)
        # converted_provider_settings should be None as the type is invalid
        assert data_api.provider_settings is None

    def test_get_data_api_ragflow_with_invalid_retrieve_settings_type(self):
        """测试 RagFlow 使用无效类型的 retrieve_settings"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )
        # Set an invalid type (not RagFlowRetrieveSettings or dict)
        object.__setattr__(kb, "retrieve_settings", "invalid")

        from agentrun.knowledgebase.api.data import RagFlowDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, RagFlowDataAPI)
        assert data_api.provider_settings is not None
        # converted_retrieve_settings should be None as the type is invalid
        assert data_api.retrieve_settings is None

    def test_get_data_api_adb_with_invalid_provider_settings_type(self):
        """测试 ADB 使用无效类型的 provider_settings"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.ADB,
        )
        # Set an invalid type (not ADBProviderSettings or dict)
        object.__setattr__(kb, "provider_settings", "invalid")

        from agentrun.knowledgebase.api.data import ADBDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, ADBDataAPI)
        # converted_provider_settings should be None as the type is invalid
        assert data_api.provider_settings is None

    def test_get_data_api_adb_with_invalid_retrieve_settings_type(self):
        """测试 ADB 使用无效类型的 retrieve_settings"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.ADB,
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
        )
        # Set an invalid type (not ADBRetrieveSettings or dict)
        object.__setattr__(kb, "retrieve_settings", "invalid")

        from agentrun.knowledgebase.api.data import ADBDataAPI

        data_api = kb._get_data_api()
        assert isinstance(data_api, ADBDataAPI)
        assert data_api.provider_settings is not None
        # converted_retrieve_settings should be None as the type is invalid
        assert data_api.retrieve_settings is None

    def test_get_data_api_with_unknown_provider(self):
        """测试使用未知提供商获取数据链路 API"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
        )
        # Set an unknown provider that's not in KnowledgeBaseProvider enum
        object.__setattr__(kb, "provider", "unknown_provider")

        # get_data_api should raise an error for unsupported provider
        # The error comes from trying to convert to KnowledgeBaseProvider enum
        with pytest.raises(
            ValueError, match="is not a valid KnowledgeBaseProvider"
        ):
            kb._get_data_api()


class TestKnowledgeBaseRetrieve:
    """测试 KnowledgeBase.retrieve 方法"""

    @patch("agentrun.knowledgebase.api.data.RagFlowDataAPI.retrieve")
    def test_retrieve_sync(self, mock_retrieve):
        """测试同步检索"""
        mock_retrieve.return_value = {
            "data": [{"content": "test content", "score": 0.9}],
            "query": "test query",
            "knowledge_base_name": "test-kb",
        }

        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        result = kb.retrieve("test query")
        assert result["query"] == "test query"
        assert "data" in result

    @patch("agentrun.knowledgebase.api.data.RagFlowDataAPI.retrieve_async")
    @pytest.mark.asyncio
    async def test_retrieve_async(self, mock_retrieve_async):
        """测试异步检索"""
        mock_retrieve_async.return_value = {
            "data": [{"content": "test content", "score": 0.9}],
            "query": "test query",
            "knowledge_base_name": "test-kb",
        }

        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        result = await kb.retrieve_async("test query")
        assert result["query"] == "test query"
        assert "data" in result


class TestKnowledgeBaseSafeGetKB:
    """测试 KnowledgeBase._safe_get_kb 方法"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_safe_get_kb_success(self, mock_control_api_class):
        """测试安全获取知识库成功"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = KnowledgeBase._safe_get_kb("test-kb")
        assert isinstance(result, KnowledgeBase)

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_safe_get_kb_failure(self, mock_control_api_class):
        """测试安全获取知识库失败"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base.side_effect = Exception("Not found")
        mock_control_api_class.return_value = mock_control_api

        result = KnowledgeBase._safe_get_kb("test-kb")
        assert isinstance(result, Exception)

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_safe_get_kb_async_success(self, mock_control_api_class):
        """测试异步安全获取知识库成功"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = await KnowledgeBase._safe_get_kb_async("test-kb")
        assert isinstance(result, KnowledgeBase)

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_safe_get_kb_async_failure(self, mock_control_api_class):
        """测试异步安全获取知识库失败"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base_async = AsyncMock(
            side_effect=Exception("Not found")
        )
        mock_control_api_class.return_value = mock_control_api

        result = await KnowledgeBase._safe_get_kb_async("test-kb")
        assert isinstance(result, Exception)


class TestKnowledgeBaseSafeRetrieveKB:
    """测试 KnowledgeBase._safe_retrieve_kb 方法"""

    def test_safe_retrieve_kb_with_exception(self):
        """测试安全检索知识库（传入异常）"""
        error = Exception("Not found")
        result = KnowledgeBase._safe_retrieve_kb("test-kb", error, "test query")

        assert result["error"] is True
        assert "Failed to retrieve" in result["data"]
        assert result["query"] == "test query"
        assert result["knowledge_base_name"] == "test-kb"

    @patch("agentrun.knowledgebase.api.data.RagFlowDataAPI.retrieve")
    def test_safe_retrieve_kb_success(self, mock_retrieve):
        """测试安全检索知识库成功"""
        mock_retrieve.return_value = {
            "data": [{"content": "test"}],
            "query": "test query",
            "knowledge_base_name": "test-kb",
        }

        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        result = KnowledgeBase._safe_retrieve_kb("test-kb", kb, "test query")
        assert "data" in result

    @patch("agentrun.knowledgebase.api.data.RagFlowDataAPI.retrieve")
    def test_safe_retrieve_kb_failure(self, mock_retrieve):
        """测试安全检索知识库失败"""
        mock_retrieve.side_effect = Exception("Retrieve failed")

        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        result = KnowledgeBase._safe_retrieve_kb("test-kb", kb, "test query")
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_safe_retrieve_kb_async_with_exception(self):
        """测试异步安全检索知识库（传入异常）"""
        error = Exception("Not found")
        result = await KnowledgeBase._safe_retrieve_kb_async(
            "test-kb", error, "test query"
        )

        assert result["error"] is True
        assert "Failed to retrieve" in result["data"]

    @patch("agentrun.knowledgebase.api.data.RagFlowDataAPI.retrieve_async")
    @pytest.mark.asyncio
    async def test_safe_retrieve_kb_async_success(self, mock_retrieve_async):
        """测试异步安全检索知识库成功"""
        mock_retrieve_async.return_value = {
            "data": [{"content": "test"}],
            "query": "test query",
            "knowledge_base_name": "test-kb",
        }

        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        result = await KnowledgeBase._safe_retrieve_kb_async(
            "test-kb", kb, "test query"
        )
        assert "data" in result

    @patch("agentrun.knowledgebase.api.data.RagFlowDataAPI.retrieve_async")
    @pytest.mark.asyncio
    async def test_safe_retrieve_kb_async_failure(self, mock_retrieve_async):
        """测试异步安全检索知识库失败"""
        mock_retrieve_async.side_effect = Exception("Retrieve failed")

        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        result = await KnowledgeBase._safe_retrieve_kb_async(
            "test-kb", kb, "test query"
        )
        assert result["error"] is True


class TestKnowledgeBaseMultiRetrieve:
    """测试 KnowledgeBase.multi_retrieve 方法"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @patch("agentrun.knowledgebase.api.data.RagFlowDataAPI.retrieve")
    def test_multi_retrieve_sync(self, mock_retrieve, mock_control_api_class):
        """测试同步多知识库检索"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        mock_retrieve.return_value = {
            "data": [{"content": "test"}],
            "query": "test query",
            "knowledge_base_name": "test-kb",
        }

        result = KnowledgeBase.multi_retrieve(
            query="test query",
            knowledge_base_names=["kb-1", "kb-2"],
        )

        assert "results" in result
        assert "query" in result
        assert result["query"] == "test query"

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @patch("agentrun.knowledgebase.api.data.RagFlowDataAPI.retrieve_async")
    @pytest.mark.asyncio
    async def test_multi_retrieve_async(
        self, mock_retrieve_async, mock_control_api_class
    ):
        """测试异步多知识库检索"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        mock_retrieve_async.return_value = {
            "data": [{"content": "test"}],
            "query": "test query",
            "knowledge_base_name": "test-kb",
        }

        result = await KnowledgeBase.multi_retrieve_async(
            query="test query",
            knowledge_base_names=["kb-1", "kb-2"],
        )

        assert "results" in result
        assert "query" in result
        assert result["query"] == "test query"

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_multi_retrieve_with_partial_failure(self, mock_control_api_class):
        """测试同步多知识库检索（部分失败）"""
        mock_control_api = MagicMock()
        # 第一个成功，第二个失败
        mock_control_api.get_knowledge_base.side_effect = [
            MockKnowledgeBaseData(),
            Exception("Not found"),
        ]
        mock_control_api_class.return_value = mock_control_api

        with patch(
            "agentrun.knowledgebase.api.data.RagFlowDataAPI.retrieve"
        ) as mock_retrieve:
            mock_retrieve.return_value = {
                "data": [{"content": "test"}],
                "query": "test query",
                "knowledge_base_name": "kb-1",
            }

            result = KnowledgeBase.multi_retrieve(
                query="test query",
                knowledge_base_names=["kb-1", "kb-2"],
            )

            assert "results" in result
            # kb-2 应该有错误
            assert "kb-2" in result["results"]
            assert result["results"]["kb-2"]["error"] is True

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_multi_retrieve_async_with_partial_failure(
        self, mock_control_api_class
    ):
        """测试异步多知识库检索（部分失败）"""
        mock_control_api = MagicMock()

        # 创建一个返回不同结果的 side_effect 函数
        call_count = [0]

        async def mock_get_async(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockKnowledgeBaseData()
            else:
                raise Exception("Not found")

        mock_control_api.get_knowledge_base_async = mock_get_async
        mock_control_api_class.return_value = mock_control_api

        with patch(
            "agentrun.knowledgebase.api.data.RagFlowDataAPI.retrieve_async"
        ) as mock_retrieve_async:
            mock_retrieve_async.return_value = {
                "data": [{"content": "test"}],
                "query": "test query",
                "knowledge_base_name": "kb-1",
            }

            result = await KnowledgeBase.multi_retrieve_async(
                query="test query",
                knowledge_base_names=["kb-1", "kb-2"],
            )

            assert "results" in result
