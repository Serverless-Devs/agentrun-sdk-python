"""测试 agentrun.memory_collection.client 模块 / Test agentrun.memory_collection.client module"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.memory_collection.client import MemoryCollectionClient
from agentrun.memory_collection.model import (
    EmbedderConfig,
    EmbedderConfigConfig,
    LLMConfig,
    LLMConfigConfig,
    MemoryCollectionCreateInput,
    MemoryCollectionListInput,
    MemoryCollectionUpdateInput,
    NetworkConfiguration,
    VectorStoreConfig,
    VectorStoreConfigConfig,
)
from agentrun.utils.config import Config
from agentrun.utils.exception import (
    HTTPError,
    ResourceAlreadyExistError,
    ResourceNotExistError,
)


class MockMemoryCollectionData:
    """模拟记忆集合数据"""

    def to_map(self):
        return {
            "memoryCollectionId": "mc-123",
            "memoryCollectionName": "test-memory-collection",
            "description": "Test memory collection",
            "type": "vector",
            "createdAt": "2024-01-01T00:00:00Z",
            "lastUpdatedAt": "2024-01-01T00:00:00Z",
            "embedderConfig": {
                "modelServiceName": "test-embedder",
                "config": {"model": "text-embedding-3-small"},
            },
            "llmConfig": {
                "modelServiceName": "test-llm",
                "config": {"model": "gpt-4"},
            },
            "vectorStoreConfig": {
                "provider": "dashvector",
                "config": {
                    "endpoint": "https://test.dashvector.cn",
                    "instanceName": "test-instance",
                    "collectionName": "test-collection",
                    "vectorDimension": 1536,
                },
            },
        }


class MockListResult:
    """模拟列表结果"""

    def __init__(self, items):
        self.items = items


class TestMemoryCollectionClientInit:
    """测试 MemoryCollectionClient 初始化"""

    def test_init_without_config(self):
        """测试不带配置的初始化"""
        client = MemoryCollectionClient()
        assert client is not None

    def test_init_with_config(self):
        """测试带配置的初始化"""
        config = Config(access_key_id="test-ak")
        client = MemoryCollectionClient(config=config)
        assert client is not None


class TestMemoryCollectionClientCreate:
    """测试 MemoryCollectionClient.create 方法"""

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_create_sync(self, mock_control_api_class):
        """测试同步创建记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.create_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        input_obj = MemoryCollectionCreateInput(
            memory_collection_name="test-memory-collection",
            type="vector",
            description="Test memory collection",
            embedder_config=EmbedderConfig(
                model_service_name="test-embedder",
                config=EmbedderConfigConfig(model="text-embedding-3-small"),
            ),
        )

        result = client.create(input_obj)
        assert result.memory_collection_name == "test-memory-collection"
        assert mock_control_api.create_memory_collection.called

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_create_async(self, mock_control_api_class):
        """测试异步创建记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.create_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        input_obj = MemoryCollectionCreateInput(
            memory_collection_name="test-memory-collection",
            type="vector",
            description="Test memory collection",
        )

        result = await client.create_async(input_obj)
        assert result.memory_collection_name == "test-memory-collection"

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_create_with_full_config(self, mock_control_api_class):
        """测试创建记忆集合（完整配置）"""
        mock_control_api = MagicMock()
        mock_control_api.create_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
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

        result = client.create(input_obj)
        assert result.memory_collection_name == "test-memory-collection"
        assert mock_control_api.create_memory_collection.called

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_create_already_exists(self, mock_control_api_class):
        """测试创建已存在的记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.create_memory_collection.side_effect = HTTPError(
            status_code=409,
            message="Resource already exists",
            request_id="req-1",
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        input_obj = MemoryCollectionCreateInput(
            memory_collection_name="existing-memory-collection",
            type="vector",
        )

        with pytest.raises(ResourceAlreadyExistError):
            client.create(input_obj)

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_create_async_already_exists(self, mock_control_api_class):
        """测试异步创建已存在的记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.create_memory_collection_async = AsyncMock(
            side_effect=HTTPError(
                status_code=409,
                message="Resource already exists",
                request_id="req-1",
            )
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        input_obj = MemoryCollectionCreateInput(
            memory_collection_name="existing-memory-collection",
            type="vector",
        )

        with pytest.raises(ResourceAlreadyExistError):
            await client.create_async(input_obj)


class TestMemoryCollectionClientDelete:
    """测试 MemoryCollectionClient.delete 方法"""

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_delete_sync(self, mock_control_api_class):
        """测试同步删除记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.delete_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        result = client.delete("test-memory-collection")
        assert result is not None
        assert mock_control_api.delete_memory_collection.called

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_delete_async(self, mock_control_api_class):
        """测试异步删除记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.delete_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        result = await client.delete_async("test-memory-collection")
        assert result is not None

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_delete_not_exist(self, mock_control_api_class):
        """测试删除不存在的记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.delete_memory_collection.side_effect = HTTPError(
            status_code=404,
            message="Resource does not exist",
            request_id="req-1",
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        with pytest.raises(ResourceNotExistError):
            client.delete("nonexistent-memory-collection")

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_delete_async_not_exist(self, mock_control_api_class):
        """测试异步删除不存在的记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.delete_memory_collection_async = AsyncMock(
            side_effect=HTTPError(
                status_code=404,
                message="Resource does not exist",
                request_id="req-1",
            )
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        with pytest.raises(ResourceNotExistError):
            await client.delete_async("nonexistent-memory-collection")


class TestMemoryCollectionClientUpdate:
    """测试 MemoryCollectionClient.update 方法"""

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_update_sync(self, mock_control_api_class):
        """测试同步更新记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.update_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        input_obj = MemoryCollectionUpdateInput(
            description="Updated description"
        )
        result = client.update("test-memory-collection", input_obj)
        assert result is not None
        assert mock_control_api.update_memory_collection.called

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_update_sync_with_embedder_config(self, mock_control_api_class):
        """测试同步更新记忆集合（带嵌入模型配置）"""
        mock_control_api = MagicMock()
        mock_control_api.update_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        input_obj = MemoryCollectionUpdateInput(
            description="Updated",
            embedder_config=EmbedderConfig(
                model_service_name="new-embedder",
                config=EmbedderConfigConfig(model="text-embedding-ada-002"),
            ),
        )
        result = client.update("test-memory-collection", input_obj)
        assert result is not None

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_update_async(self, mock_control_api_class):
        """测试异步更新记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.update_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        input_obj = MemoryCollectionUpdateInput(description="Updated")
        result = await client.update_async("test-memory-collection", input_obj)
        assert result is not None

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_update_async_with_llm_config(self, mock_control_api_class):
        """测试异步更新记忆集合（带 LLM 配置）"""
        mock_control_api = MagicMock()
        mock_control_api.update_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        input_obj = MemoryCollectionUpdateInput(
            llm_config=LLMConfig(
                model_service_name="new-llm",
                config=LLMConfigConfig(model="gpt-4-turbo"),
            )
        )
        result = await client.update_async("test-memory-collection", input_obj)
        assert result is not None

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_update_not_exist(self, mock_control_api_class):
        """测试更新不存在的记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.update_memory_collection.side_effect = HTTPError(
            status_code=404,
            message="Resource does not exist",
            request_id="req-1",
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        input_obj = MemoryCollectionUpdateInput(description="Updated")
        with pytest.raises(ResourceNotExistError):
            client.update("nonexistent-memory-collection", input_obj)

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_update_with_vector_store_config(self, mock_control_api_class):
        """测试更新记忆集合（带向量存储配置）"""
        mock_control_api = MagicMock()
        mock_control_api.update_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        input_obj = MemoryCollectionUpdateInput(
            vector_store_config=VectorStoreConfig(
                provider="dashvector",
                config=VectorStoreConfigConfig(
                    endpoint="https://new.dashvector.cn",
                    instance_name="new-instance",
                    collection_name="new-collection",
                    vector_dimension=3072,
                ),
            )
        )
        result = client.update("test-memory-collection", input_obj)
        assert result is not None


class TestMemoryCollectionClientGet:
    """测试 MemoryCollectionClient.get 方法"""

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_get_sync(self, mock_control_api_class):
        """测试同步获取记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.get_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        result = client.get("test-memory-collection")
        assert result.memory_collection_name == "test-memory-collection"
        assert mock_control_api.get_memory_collection.called

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_get_async(self, mock_control_api_class):
        """测试异步获取记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.get_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        result = await client.get_async("test-memory-collection")
        assert result.memory_collection_name == "test-memory-collection"

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_get_not_exist(self, mock_control_api_class):
        """测试获取不存在的记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.get_memory_collection.side_effect = HTTPError(
            status_code=404,
            message="Resource does not exist",
            request_id="req-1",
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        with pytest.raises(ResourceNotExistError):
            client.get("nonexistent-memory-collection")

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_get_async_not_exist(self, mock_control_api_class):
        """测试异步获取不存在的记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.get_memory_collection_async = AsyncMock(
            side_effect=HTTPError(
                status_code=404,
                message="Resource does not exist",
                request_id="req-1",
            )
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        with pytest.raises(ResourceNotExistError):
            await client.get_async("nonexistent-memory-collection")


class TestMemoryCollectionClientList:
    """测试 MemoryCollectionClient.list 方法"""

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_list_sync(self, mock_control_api_class):
        """测试同步列出记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.list_memory_collections.return_value = MockListResult([
            MockMemoryCollectionData(),
            MockMemoryCollectionData(),
        ])
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        result = client.list()
        assert len(result) == 2
        assert mock_control_api.list_memory_collections.called

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_list_sync_with_input(self, mock_control_api_class):
        """测试同步列出记忆集合（带输入参数）"""
        mock_control_api = MagicMock()
        mock_control_api.list_memory_collections.return_value = MockListResult(
            [MockMemoryCollectionData()]
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        input_obj = MemoryCollectionListInput(
            page_number=1,
            page_size=10,
            memory_collection_name="test-memory-collection",
        )
        result = client.list(input=input_obj)
        assert len(result) == 1

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_list_async(self, mock_control_api_class):
        """测试异步列出记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.list_memory_collections_async = AsyncMock(
            return_value=MockListResult([MockMemoryCollectionData()])
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        result = await client.list_async()
        assert len(result) == 1

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_list_async_with_input(self, mock_control_api_class):
        """测试异步列出记忆集合（带输入参数）"""
        mock_control_api = MagicMock()
        mock_control_api.list_memory_collections_async = AsyncMock(
            return_value=MockListResult([MockMemoryCollectionData()])
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        input_obj = MemoryCollectionListInput(page_number=1, page_size=10)
        result = await client.list_async(input=input_obj)
        assert len(result) == 1

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_list_empty(self, mock_control_api_class):
        """测试列出空记忆集合列表"""
        mock_control_api = MagicMock()
        mock_control_api.list_memory_collections.return_value = MockListResult(
            []
        )
        mock_control_api_class.return_value = mock_control_api

        client = MemoryCollectionClient()
        result = client.list()
        assert len(result) == 0
