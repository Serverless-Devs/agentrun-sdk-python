"""测试 agentrun.knowledgebase.client 模块 / Test agentrun.knowledgebase.client module"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.knowledgebase.client import KnowledgeBaseClient
from agentrun.knowledgebase.model import (
    ADBProviderSettings,
    BailianProviderSettings,
    KnowledgeBaseCreateInput,
    KnowledgeBaseListInput,
    KnowledgeBaseProvider,
    KnowledgeBaseUpdateInput,
    RagFlowProviderSettings,
)
from agentrun.utils.config import Config
from agentrun.utils.exception import (
    HTTPError,
    ResourceAlreadyExistError,
    ResourceNotExistError,
)


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
            "createdAt": "2024-01-01T00:00:00Z",
            "lastUpdatedAt": "2024-01-01T00:00:00Z",
        }


class MockListResult:
    """模拟列表结果"""

    def __init__(self, items):
        self.items = items


class TestKnowledgeBaseClientInit:
    """测试 KnowledgeBaseClient 初始化"""

    def test_init_without_config(self):
        """测试不带配置的初始化"""
        client = KnowledgeBaseClient()
        assert client is not None

    def test_init_with_config(self):
        """测试带配置的初始化"""
        config = Config(access_key_id="test-ak")
        client = KnowledgeBaseClient(config=config)
        assert client is not None


class TestKnowledgeBaseClientCreate:
    """测试 KnowledgeBaseClient.create 方法"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_create_sync(self, mock_control_api_class):
        """测试同步创建知识库"""
        mock_control_api = MagicMock()
        mock_control_api.create_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            description="Test knowledge base",
        )

        result = client.create(input_obj)
        assert result.knowledge_base_name == "test-kb"
        assert mock_control_api.create_knowledge_base.called

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_create_async(self, mock_control_api_class):
        """测试异步创建知识库"""
        mock_control_api = MagicMock()
        mock_control_api.create_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
        )

        result = await client.create_async(input_obj)
        assert result.knowledge_base_name == "test-kb"

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_create_bailian_kb(self, mock_control_api_class):
        """测试创建百炼知识库"""
        mock_control_api = MagicMock()
        mock_control_api.create_knowledge_base.return_value = (
            MockBailianKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="test-bailian-kb",
            provider=KnowledgeBaseProvider.BAILIAN,
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
        )

        result = client.create(input_obj)
        assert result.knowledge_base_name == "test-bailian-kb"

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_create_adb_kb(self, mock_control_api_class):
        """测试创建 ADB 知识库"""
        mock_control_api = MagicMock()
        mock_control_api.create_knowledge_base.return_value = (
            MockADBKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="test-adb-kb",
            provider=KnowledgeBaseProvider.ADB,
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
        )

        result = client.create(input_obj)
        assert result.knowledge_base_name == "test-adb-kb"

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_create_with_config(self, mock_control_api_class):
        """测试带配置创建知识库"""
        mock_control_api = MagicMock()
        mock_control_api.create_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
        )
        config = Config(access_key_id="custom-ak")

        result = client.create(input_obj, config=config)
        assert result is not None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_create_already_exists(self, mock_control_api_class):
        """测试创建已存在的知识库"""
        mock_control_api = MagicMock()
        mock_control_api.create_knowledge_base.side_effect = HTTPError(
            status_code=409,
            message="Resource already exists",
            request_id="req-1",
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="existing-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
        )

        with pytest.raises(ResourceAlreadyExistError):
            client.create(input_obj)

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_create_async_already_exists(self, mock_control_api_class):
        """测试异步创建已存在的知识库"""
        mock_control_api = MagicMock()
        mock_control_api.create_knowledge_base_async = AsyncMock(
            side_effect=HTTPError(
                status_code=409,
                message="Resource already exists",
                request_id="req-1",
            )
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseCreateInput(
            knowledge_base_name="existing-kb",
            provider=KnowledgeBaseProvider.RAGFLOW,
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
        )

        with pytest.raises(ResourceAlreadyExistError):
            await client.create_async(input_obj)


class TestKnowledgeBaseClientDelete:
    """测试 KnowledgeBaseClient.delete 方法"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_delete_sync(self, mock_control_api_class):
        """测试同步删除知识库"""
        mock_control_api = MagicMock()
        mock_control_api.delete_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        result = client.delete("test-kb")
        assert result is not None
        assert mock_control_api.delete_knowledge_base.called

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_delete_async(self, mock_control_api_class):
        """测试异步删除知识库"""
        mock_control_api = MagicMock()
        mock_control_api.delete_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        result = await client.delete_async("test-kb")
        assert result is not None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_delete_with_config(self, mock_control_api_class):
        """测试带配置删除知识库"""
        mock_control_api = MagicMock()
        mock_control_api.delete_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        config = Config(access_key_id="custom-ak")
        result = client.delete("test-kb", config=config)
        assert result is not None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_delete_not_exist(self, mock_control_api_class):
        """测试删除不存在的知识库"""
        mock_control_api = MagicMock()
        mock_control_api.delete_knowledge_base.side_effect = HTTPError(
            status_code=404,
            message="Resource does not exist",
            request_id="req-1",
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        with pytest.raises(ResourceNotExistError):
            client.delete("nonexistent-kb")

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_delete_async_not_exist(self, mock_control_api_class):
        """测试异步删除不存在的知识库"""
        mock_control_api = MagicMock()
        mock_control_api.delete_knowledge_base_async = AsyncMock(
            side_effect=HTTPError(
                status_code=404,
                message="Resource does not exist",
                request_id="req-1",
            )
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        with pytest.raises(ResourceNotExistError):
            await client.delete_async("nonexistent-kb")


class TestKnowledgeBaseClientUpdate:
    """测试 KnowledgeBaseClient.update 方法"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_update_sync(self, mock_control_api_class):
        """测试同步更新知识库"""
        mock_control_api = MagicMock()
        mock_control_api.update_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseUpdateInput(description="Updated description")
        result = client.update("test-kb", input_obj)
        assert result is not None
        assert mock_control_api.update_knowledge_base.called

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_update_async(self, mock_control_api_class):
        """测试异步更新知识库"""
        mock_control_api = MagicMock()
        mock_control_api.update_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseUpdateInput(description="Updated")
        result = await client.update_async("test-kb", input_obj)
        assert result is not None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_update_with_provider_settings(self, mock_control_api_class):
        """测试更新知识库（带提供商设置）"""
        mock_control_api = MagicMock()
        mock_control_api.update_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseUpdateInput(
            provider_settings=RagFlowProviderSettings(
                base_url="https://new-ragflow.example.com",
                dataset_ids=["ds-new"],
            ),
        )
        result = client.update("test-kb", input_obj)
        assert result is not None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_update_with_credential(self, mock_control_api_class):
        """测试更新知识库（带凭证）"""
        mock_control_api = MagicMock()
        mock_control_api.update_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseUpdateInput(
            credential_name="new-credential",
        )
        result = client.update("test-kb", input_obj)
        assert result is not None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_update_not_exist(self, mock_control_api_class):
        """测试更新不存在的知识库"""
        mock_control_api = MagicMock()
        mock_control_api.update_knowledge_base.side_effect = HTTPError(
            status_code=404,
            message="Resource does not exist",
            request_id="req-1",
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseUpdateInput(description="Updated")
        with pytest.raises(ResourceNotExistError):
            client.update("nonexistent-kb", input_obj)

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_update_async_not_exist(self, mock_control_api_class):
        """测试异步更新不存在的知识库"""
        mock_control_api = MagicMock()
        mock_control_api.update_knowledge_base_async = AsyncMock(
            side_effect=HTTPError(
                status_code=404,
                message="Resource does not exist",
                request_id="req-1",
            )
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseUpdateInput(description="Updated")
        with pytest.raises(ResourceNotExistError):
            await client.update_async("nonexistent-kb", input_obj)


class TestKnowledgeBaseClientGet:
    """测试 KnowledgeBaseClient.get 方法"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_get_sync(self, mock_control_api_class):
        """测试同步获取知识库"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        result = client.get("test-kb")
        assert result.knowledge_base_name == "test-kb"
        assert mock_control_api.get_knowledge_base.called

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_get_async(self, mock_control_api_class):
        """测试异步获取知识库"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base_async = AsyncMock(
            return_value=MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        result = await client.get_async("test-kb")
        assert result.knowledge_base_name == "test-kb"

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_get_with_config(self, mock_control_api_class):
        """测试带配置获取知识库"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base.return_value = (
            MockKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        config = Config(access_key_id="custom-ak")
        result = client.get("test-kb", config=config)
        assert result is not None

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_get_not_exist(self, mock_control_api_class):
        """测试获取不存在的知识库"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base.side_effect = HTTPError(
            status_code=404,
            message="Resource does not exist",
            request_id="req-1",
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        with pytest.raises(ResourceNotExistError):
            client.get("nonexistent-kb")

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_get_async_not_exist(self, mock_control_api_class):
        """测试异步获取不存在的知识库"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base_async = AsyncMock(
            side_effect=HTTPError(
                status_code=404,
                message="Resource does not exist",
                request_id="req-1",
            )
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        with pytest.raises(ResourceNotExistError):
            await client.get_async("nonexistent-kb")


class TestKnowledgeBaseClientList:
    """测试 KnowledgeBaseClient.list 方法"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_list_sync(self, mock_control_api_class):
        """测试同步列出知识库"""
        mock_control_api = MagicMock()
        mock_control_api.list_knowledge_bases.return_value = MockListResult([
            MockKnowledgeBaseData(),
            MockBailianKnowledgeBaseData(),
        ])
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        result = client.list()
        assert len(result) == 2
        assert mock_control_api.list_knowledge_bases.called

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_list_async(self, mock_control_api_class):
        """测试异步列出知识库"""
        mock_control_api = MagicMock()
        mock_control_api.list_knowledge_bases_async = AsyncMock(
            return_value=MockListResult([MockKnowledgeBaseData()])
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        result = await client.list_async()
        assert len(result) == 1

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_list_with_input(self, mock_control_api_class):
        """测试同步列出知识库（带输入参数）"""
        mock_control_api = MagicMock()
        mock_control_api.list_knowledge_bases.return_value = MockListResult(
            [MockKnowledgeBaseData()]
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseListInput(
            page_number=1,
            page_size=10,
            provider=KnowledgeBaseProvider.RAGFLOW,
        )
        result = client.list(input=input_obj)
        assert len(result) == 1

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @pytest.mark.asyncio
    async def test_list_async_with_input(self, mock_control_api_class):
        """测试异步列出知识库（带输入参数）"""
        mock_control_api = MagicMock()
        mock_control_api.list_knowledge_bases_async = AsyncMock(
            return_value=MockListResult([MockKnowledgeBaseData()])
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        input_obj = KnowledgeBaseListInput(page_number=1, page_size=10)
        result = await client.list_async(input=input_obj)
        assert len(result) == 1

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_list_empty(self, mock_control_api_class):
        """测试列出空知识库列表"""
        mock_control_api = MagicMock()
        mock_control_api.list_knowledge_bases.return_value = MockListResult([])
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        result = client.list()
        assert len(result) == 0

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_list_with_none_input(self, mock_control_api_class):
        """测试列出知识库（输入为 None）"""
        mock_control_api = MagicMock()
        mock_control_api.list_knowledge_bases.return_value = MockListResult(
            [MockKnowledgeBaseData()]
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        result = client.list(input=None)
        assert len(result) == 1

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    def test_list_with_config(self, mock_control_api_class):
        """测试带配置列出知识库"""
        mock_control_api = MagicMock()
        mock_control_api.list_knowledge_bases.return_value = MockListResult(
            [MockKnowledgeBaseData()]
        )
        mock_control_api_class.return_value = mock_control_api

        client = KnowledgeBaseClient()
        config = Config(access_key_id="custom-ak")
        result = client.list(config=config)
        assert len(result) == 1
