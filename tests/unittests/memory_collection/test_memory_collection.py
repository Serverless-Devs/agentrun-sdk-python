"""测试 agentrun.memory_collection.memory_collection 模块 / Test agentrun.memory_collection.memory_collection module"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.memory_collection.memory_collection import MemoryCollection
from agentrun.memory_collection.model import (
    EmbedderConfig,
    EmbedderConfigConfig,
    MemoryCollectionCreateInput,
    MemoryCollectionUpdateInput,
)
from agentrun.utils.config import Config


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
        }


class MockListResult:
    """模拟列表结果"""

    def __init__(self, items):
        self.items = items


class TestMemoryCollectionCreate:
    """测试 MemoryCollection.create 方法"""

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_create_sync(self, mock_control_api_class):
        """测试同步创建记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.create_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        input_obj = MemoryCollectionCreateInput(
            memory_collection_name="test-memory-collection",
            type="vector",
            description="Test memory collection",
        )

        result = MemoryCollection.create(input_obj)
        assert result.memory_collection_name == "test-memory-collection"
        assert result.memory_collection_id == "mc-123"

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_create_async(self, mock_control_api_class):
        """测试异步创建记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.create_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        input_obj = MemoryCollectionCreateInput(
            memory_collection_name="test-memory-collection",
            type="vector",
        )

        result = await MemoryCollection.create_async(input_obj)
        assert result.memory_collection_name == "test-memory-collection"


class TestMemoryCollectionDelete:
    """测试 MemoryCollection.delete 方法"""

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_delete_by_name_sync(self, mock_control_api_class):
        """测试根据名称同步删除记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.delete_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = MemoryCollection.delete_by_name("test-memory-collection")
        assert result is not None

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_delete_by_name_async(self, mock_control_api_class):
        """测试根据名称异步删除记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.delete_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = await MemoryCollection.delete_by_name_async(
            "test-memory-collection"
        )
        assert result is not None

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_delete_instance_sync(self, mock_control_api_class):
        """测试实例同步删除"""
        mock_control_api = MagicMock()
        mock_control_api.delete_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api.get_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        # 先获取实例
        memory_collection = MemoryCollection.get_by_name(
            "test-memory-collection"
        )

        # 删除实例
        result = memory_collection.delete()
        assert result is not None

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_delete_instance_async(self, mock_control_api_class):
        """测试实例异步删除"""
        mock_control_api = MagicMock()
        mock_control_api.delete_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api.get_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        # 先获取实例
        memory_collection = await MemoryCollection.get_by_name_async(
            "test-memory-collection"
        )

        # 删除实例
        result = await memory_collection.delete_async()
        assert result is not None


class TestMemoryCollectionUpdate:
    """测试 MemoryCollection.update 方法"""

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_update_by_name_sync(self, mock_control_api_class):
        """测试根据名称同步更新记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.update_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        input_obj = MemoryCollectionUpdateInput(
            description="Updated description"
        )
        result = MemoryCollection.update_by_name(
            "test-memory-collection", input_obj
        )
        assert result is not None

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_update_by_name_async(self, mock_control_api_class):
        """测试根据名称异步更新记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.update_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        input_obj = MemoryCollectionUpdateInput(description="Updated")
        result = await MemoryCollection.update_by_name_async(
            "test-memory-collection", input_obj
        )
        assert result is not None

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_update_instance_sync(self, mock_control_api_class):
        """测试实例同步更新"""
        mock_control_api = MagicMock()
        mock_control_api.update_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api.get_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        # 先获取实例
        memory_collection = MemoryCollection.get_by_name(
            "test-memory-collection"
        )

        # 更新实例
        input_obj = MemoryCollectionUpdateInput(description="Updated")
        result = memory_collection.update(input_obj)
        assert result is not None

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_update_instance_async(self, mock_control_api_class):
        """测试实例异步更新"""
        mock_control_api = MagicMock()
        mock_control_api.update_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api.get_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        # 先获取实例
        memory_collection = await MemoryCollection.get_by_name_async(
            "test-memory-collection"
        )

        # 更新实例
        input_obj = MemoryCollectionUpdateInput(description="Updated")
        result = await memory_collection.update_async(input_obj)
        assert result is not None


class TestMemoryCollectionGet:
    """测试 MemoryCollection.get_by_name 方法"""

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_get_by_name_sync(self, mock_control_api_class):
        """测试同步获取记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.get_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = MemoryCollection.get_by_name("test-memory-collection")
        assert result.memory_collection_name == "test-memory-collection"
        assert result.memory_collection_id == "mc-123"

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_get_by_name_async(self, mock_control_api_class):
        """测试异步获取记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.get_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = await MemoryCollection.get_by_name_async(
            "test-memory-collection"
        )
        assert result.memory_collection_name == "test-memory-collection"


class TestMemoryCollectionList:
    """测试 MemoryCollection.list_all 方法"""

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_list_all_sync(self, mock_control_api_class):
        """测试同步列出记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.list_memory_collections.return_value = MockListResult([
            MockMemoryCollectionData(),
            MockMemoryCollectionData(),
        ])
        mock_control_api_class.return_value = mock_control_api

        result = MemoryCollection.list_all()
        # list_all 会对结果去重，所以相同 ID 的记录只会返回一个
        assert len(result) >= 1

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_list_all_async(self, mock_control_api_class):
        """测试异步列出记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.list_memory_collections_async = AsyncMock(
            return_value=MockListResult([MockMemoryCollectionData()])
        )
        mock_control_api_class.return_value = mock_control_api

        result = await MemoryCollection.list_all_async()
        assert len(result) == 1


class TestMemoryCollectionRefresh:
    """测试 MemoryCollection.refresh 方法"""

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_refresh_sync(self, mock_control_api_class):
        """测试同步刷新记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.get_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        # 先获取实例
        memory_collection = MemoryCollection.get_by_name(
            "test-memory-collection"
        )

        # 刷新实例
        memory_collection.refresh()
        assert (
            memory_collection.memory_collection_name == "test-memory-collection"
        )

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_refresh_async(self, mock_control_api_class):
        """测试异步刷新记忆集合"""
        mock_control_api = MagicMock()
        mock_control_api.get_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        # 先获取实例
        memory_collection = await MemoryCollection.get_by_name_async(
            "test-memory-collection"
        )

        # 刷新实例
        await memory_collection.refresh_async()
        assert (
            memory_collection.memory_collection_name == "test-memory-collection"
        )


class TestMemoryCollectionFromInnerObject:
    """测试 MemoryCollection.from_inner_object 方法"""

    def test_from_inner_object(self):
        """测试从内部对象创建记忆集合"""
        mock_data = MockMemoryCollectionData()
        memory_collection = MemoryCollection.from_inner_object(mock_data)

        assert memory_collection.memory_collection_id == "mc-123"
        assert (
            memory_collection.memory_collection_name == "test-memory-collection"
        )
        assert memory_collection.description == "Test memory collection"
        assert memory_collection.type == "vector"

    def test_from_inner_object_with_extra(self):
        """测试从内部对象创建记忆集合（带额外字段）"""
        mock_data = MockMemoryCollectionData()
        extra = {"custom_field": "custom_value"}
        memory_collection = MemoryCollection.from_inner_object(mock_data, extra)

        assert (
            memory_collection.memory_collection_name == "test-memory-collection"
        )
