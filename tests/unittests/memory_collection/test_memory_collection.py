"""测试 agentrun.memory_collection.memory_collection 模块 / Test agentrun.memory_collection.memory_collection module"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.memory_collection.memory_collection import MemoryCollection
from agentrun.memory_collection.model import (
    EmbedderConfig,
    EmbedderConfigConfig,
    MemoryCollectionCreateInput,
    MemoryCollectionUpdateInput,
    VectorStoreConfig,
    VectorStoreConfigConfig,
    VectorStoreConfigMysqlConfig,
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


class TestMemoryCollectionToMem0Memory:
    """测试 MemoryCollection.to_mem0_memory 和 to_mem0_memory_async 方法"""

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_to_mem0_memory_sync(self, mock_control_api_class):
        """测试同步转换为 mem0 Memory 客户端"""
        # Mock MemoryCollection 数据
        mock_control_api = MagicMock()
        mock_data = MockMemoryCollectionData()
        mock_control_api.get_memory_collection.return_value = mock_data
        mock_control_api_class.return_value = mock_control_api

        # Mock Memory.from_config
        with patch("agentrun_mem0.Memory") as mock_memory_class:
            mock_memory_instance = MagicMock()
            mock_memory_class.from_config.return_value = mock_memory_instance

            # 调用 to_mem0_memory
            result = MemoryCollection.to_mem0_memory(
                "test-memory-collection",
                config=Config(
                    access_key_id="test-key",
                    access_key_secret="test-secret",
                    region_id="cn-hangzhou",
                ),
            )

            # 验证返回的是 Memory 实例
            assert result == mock_memory_instance
            mock_memory_class.from_config.assert_called_once()

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_to_mem0_memory_async(self, mock_control_api_class):
        """测试异步转换为 mem0 AsyncMemory 客户端"""
        # Mock MemoryCollection 数据
        mock_control_api = MagicMock()
        mock_data = MockMemoryCollectionData()
        mock_control_api.get_memory_collection_async = AsyncMock(
            return_value=mock_data
        )
        mock_control_api_class.return_value = mock_control_api

        # Mock AsyncMemory.from_config
        with patch("agentrun_mem0.AsyncMemory") as mock_async_memory_class:
            mock_async_memory_instance = MagicMock()
            # from_config 是异步方法，需要返回 AsyncMock
            mock_async_memory_class.from_config = AsyncMock(
                return_value=mock_async_memory_instance
            )

            # 调用 to_mem0_memory_async
            result = await MemoryCollection.to_mem0_memory_async(
                "test-memory-collection",
                config=Config(
                    access_key_id="test-key",
                    access_key_secret="test-secret",
                    region_id="cn-hangzhou",
                ),
            )

            # 验证返回的是 AsyncMemory 实例
            assert result == mock_async_memory_instance
            mock_async_memory_class.from_config.assert_called_once()

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    def test_to_mem0_memory_import_error(self, mock_control_api_class):
        """测试 mem0 包未安装时的错误处理"""
        mock_control_api = MagicMock()
        mock_control_api.get_memory_collection.return_value = (
            MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        # Mock import 失败
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(ImportError) as exc_info:
                MemoryCollection.to_mem0_memory("test-memory-collection")

            assert "agentrun-mem0ai package is required" in str(exc_info.value)

    @patch("agentrun.memory_collection.client.MemoryCollectionControlAPI")
    @pytest.mark.asyncio
    async def test_to_mem0_memory_async_import_error(
        self, mock_control_api_class
    ):
        """测试异步方法 mem0 包未安装时的错误处理"""
        mock_control_api = MagicMock()
        mock_control_api.get_memory_collection_async = AsyncMock(
            return_value=MockMemoryCollectionData()
        )
        mock_control_api_class.return_value = mock_control_api

        # Mock import 失败
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(ImportError) as exc_info:
                await MemoryCollection.to_mem0_memory_async(
                    "test-memory-collection"
                )

            assert "agentrun-mem0ai package is required" in str(exc_info.value)


class TestMemoryCollectionMySQLSupport:
    """测试 MemoryCollection MySQL 向量数据库支持"""

    def test_get_mysql_public_host_with_env(self):
        """测试从环境变量获取 MySQL 公网地址"""
        # 设置环境变量
        os.environ["AGENTRUN_MYSQL_PUBLIC_HOST"] = "public.mysql.com"

        try:
            host = MemoryCollection._get_mysql_public_host("internal.mysql.com")
            assert host == "public.mysql.com"
        finally:
            # 清理环境变量
            del os.environ["AGENTRUN_MYSQL_PUBLIC_HOST"]

    def test_get_mysql_public_host_without_env(self):
        """测试未设置环境变量时使用内网地址"""
        # 确保环境变量不存在
        if "AGENTRUN_MYSQL_PUBLIC_HOST" in os.environ:
            del os.environ["AGENTRUN_MYSQL_PUBLIC_HOST"]

        host = MemoryCollection._get_mysql_public_host("internal.mysql.com")
        assert host == "internal.mysql.com"

    @patch("agentrun.credential.Credential.get_by_name")
    def test_get_credential_secret_sync(self, mock_get_credential):
        """测试同步获取 Credential 密钥"""
        # Mock credential
        mock_credential = MagicMock()
        mock_credential.credential_secret = "test-password"
        mock_get_credential.return_value = mock_credential

        secret = MemoryCollection._get_credential_secret(
            "test-credential", None
        )
        assert secret == "test-password"

    @patch("agentrun.credential.Credential.get_by_name")
    def test_get_credential_secret_empty(self, mock_get_credential):
        """测试 Credential 密钥为空时抛出异常"""
        # Mock credential with empty secret
        mock_credential = MagicMock()
        mock_credential.credential_secret = None
        mock_get_credential.return_value = mock_credential

        with pytest.raises(ValueError, match="secret is empty"):
            MemoryCollection._get_credential_secret("test-credential", None)

    @patch("agentrun.credential.Credential.get_by_name_async")
    @pytest.mark.asyncio
    async def test_get_credential_secret_async(self, mock_get_credential):
        """测试异步获取 Credential 密钥"""
        # Mock credential
        mock_credential = MagicMock()
        mock_credential.credential_secret = "test-password"
        mock_get_credential.return_value = mock_credential

        secret = await MemoryCollection._get_credential_secret_async(
            "test-credential", None
        )
        assert secret == "test-password"

    @patch("agentrun.credential.Credential.get_by_name_async")
    @pytest.mark.asyncio
    async def test_build_mem0_config_with_mysql_async(
        self, mock_get_credential
    ):
        """测试异步构建包含 MySQL 配置的 mem0 配置"""
        # Mock credential
        mock_credential = MagicMock()
        mock_credential.credential_secret = "test-password"
        mock_get_credential.return_value = mock_credential

        # 设置环境变量
        os.environ["AGENTRUN_MYSQL_PUBLIC_HOST"] = "public.mysql.com"

        try:
            # 创建带 MySQL 配置的 MemoryCollection
            memory_collection = MemoryCollection(
                memory_collection_name="test-mysql-memory",
                vector_store_config=VectorStoreConfig(
                    provider="alibabacloud_mysql",
                    mysql_config=VectorStoreConfigMysqlConfig(
                        host="internal.mysql.com",
                        port=3306,
                        db_name="memory",
                        user="test_user",
                        collection_name="mem0",
                        credential_name="test-credential",
                        vector_dimension=512,
                    ),
                ),
            )

            # 构建 mem0 配置
            config = await MemoryCollection._build_mem0_config_async(
                memory_collection, None, None
            )

            # 验证配置
            assert "vector_store" in config
            assert config["vector_store"]["provider"] == "alibabacloud_mysql"

            vs_config = config["vector_store"]["config"]
            assert vs_config["dbname"] == "memory"
            assert vs_config["collection_name"] == "mem0"
            assert vs_config["user"] == "test_user"
            assert vs_config["password"] == "test-password"
            assert vs_config["host"] == "public.mysql.com"  # 使用公网地址
            assert vs_config["port"] == 3306
            assert vs_config["embedding_model_dims"] == 512
            assert vs_config["distance_function"] == "cosine"
            assert vs_config["m_value"] == 16

        finally:
            # 清理环境变量
            del os.environ["AGENTRUN_MYSQL_PUBLIC_HOST"]

    @patch("agentrun.credential.Credential.get_by_name")
    def test_build_mem0_config_with_mysql_sync(self, mock_get_credential):
        """测试同步构建包含 MySQL 配置的 mem0 配置"""
        # Mock credential
        mock_credential = MagicMock()
        mock_credential.credential_secret = "test-password"
        mock_get_credential.return_value = mock_credential

        # 创建带 MySQL 配置的 MemoryCollection
        memory_collection = MemoryCollection(
            memory_collection_name="test-mysql-memory",
            vector_store_config=VectorStoreConfig(
                provider="alibabacloud_mysql",
                mysql_config=VectorStoreConfigMysqlConfig(
                    host="internal.mysql.com",
                    port=3307,
                    db_name="test_db",
                    user="admin",
                    collection_name="vectors",
                    credential_name="test-credential",
                    vector_dimension=1024,
                ),
            ),
        )

        # 构建 mem0 配置
        config = MemoryCollection._build_mem0_config(
            memory_collection, None, None
        )

        # 验证配置
        assert "vector_store" in config
        vs_config = config["vector_store"]["config"]
        assert vs_config["dbname"] == "test_db"
        assert vs_config["user"] == "admin"
        assert vs_config["password"] == "test-password"
        assert vs_config["port"] == 3307
        assert vs_config["embedding_model_dims"] == 1024

    @patch("agentrun.credential.Credential.get_by_name_async")
    @pytest.mark.asyncio
    async def test_build_mem0_config_mysql_default_values(
        self, mock_get_credential
    ):
        """测试 MySQL 配置默认值"""
        # Mock credential
        mock_credential = MagicMock()
        mock_credential.credential_secret = "test-password"
        mock_get_credential.return_value = mock_credential

        # 创建带 MySQL 配置的 MemoryCollection（使用默认值）
        memory_collection = MemoryCollection(
            memory_collection_name="test-mysql-memory",
            vector_store_config=VectorStoreConfig(
                provider="alibabacloud_mysql",
                mysql_config=VectorStoreConfigMysqlConfig(
                    host="mysql.com",
                    db_name="memory",
                    user="user",
                    collection_name="mem0",
                    credential_name="test-credential",
                    # port 和 vector_dimension 使用默认值
                ),
            ),
        )

        # 构建 mem0 配置
        config = await MemoryCollection._build_mem0_config_async(
            memory_collection, None, None
        )

        # 验证默认值
        vs_config = config["vector_store"]["config"]
        assert vs_config["port"] == 3306  # 默认端口
        assert vs_config["embedding_model_dims"] == 1536  # 默认向量维度
        assert vs_config["distance_function"] == "cosine"  # 默认距离函数
        assert vs_config["m_value"] == 16  # 默认 HNSW 参数

    @patch("agentrun.credential.Credential.get_by_name_async")
    @pytest.mark.asyncio
    async def test_build_mem0_config_mysql_credential_error(
        self, mock_get_credential
    ):
        """测试获取 Credential 失败时的错误处理"""
        # Mock credential 获取失败
        mock_get_credential.side_effect = Exception("Credential not found")

        # 创建带 MySQL 配置的 MemoryCollection
        memory_collection = MemoryCollection(
            memory_collection_name="test-mysql-memory",
            vector_store_config=VectorStoreConfig(
                provider="alibabacloud_mysql",
                mysql_config=VectorStoreConfigMysqlConfig(
                    host="mysql.com",
                    db_name="memory",
                    user="user",
                    collection_name="mem0",
                    credential_name="invalid-credential",
                ),
            ),
        )

        # 验证抛出正确的异常
        with pytest.raises(ValueError, match="Failed to get MySQL password"):
            await MemoryCollection._build_mem0_config_async(
                memory_collection, None, None
            )

    @patch("agentrun.credential.Credential.get_by_name_async")
    @pytest.mark.asyncio
    async def test_build_mem0_config_with_tablestore(self, mock_get_credential):
        """测试构建 TableStore 配置（确保不影响现有功能）"""
        # 创建带 TableStore 配置的 MemoryCollection
        memory_collection = MemoryCollection(
            memory_collection_name="test-tablestore-memory",
            vector_store_config=VectorStoreConfig(
                provider="aliyun_tablestore",
                config=VectorStoreConfigConfig(
                    endpoint="https://test.cn-hangzhou.ots.aliyuncs.com",
                    instance_name="test-instance",
                    collection_name="vectors",
                    vector_dimension=768,
                ),
            ),
        )

        # 构建 mem0 配置
        config = await MemoryCollection._build_mem0_config_async(
            memory_collection, Config(), None
        )

        # 验证 TableStore 配置仍然正常工作
        assert "vector_store" in config
        assert config["vector_store"]["provider"] == "aliyun_tablestore"
        vs_config = config["vector_store"]["config"]
        assert vs_config["instance_name"] == "test-instance"
        assert vs_config["collection_name"] == "vectors"
