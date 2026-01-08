"""Agent Runtime 高层 API 单元测试"""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.agent_runtime.model import (
    AgentRuntimeCode,
    AgentRuntimeCreateInput,
    AgentRuntimeEndpointCreateInput,
    AgentRuntimeEndpointUpdateInput,
    AgentRuntimeLanguage,
    AgentRuntimeUpdateInput,
)
from agentrun.agent_runtime.runtime import AgentRuntime
from agentrun.utils.config import Config
from agentrun.utils.model import Status

# Mock path for AgentRuntimeClient - 在使用处 mock
# AgentRuntime.__get_client 动态导入 AgentRuntimeClient
CLIENT_PATH = "agentrun.agent_runtime.client.AgentRuntimeClient"


class MockAgentRuntimeInstance:
    """模拟 AgentRuntime 实例（避免抽象类实例化问题）"""

    agent_runtime_id = "ar-123456"
    agent_runtime_name = "test-runtime"
    agent_runtime_arn = "arn:acs:agentrun:cn-hangzhou:123456:agent/test"
    status = "READY"
    cpu = 2
    memory = 4096


class MockAgentRuntimeEndpointInstance:
    """模拟 AgentRuntimeEndpoint 实例"""

    agent_runtime_endpoint_id = "are-123456"
    agent_runtime_endpoint_name = "test-endpoint"
    agent_runtime_id = "ar-123456"
    endpoint_public_url = "https://test.agentrun.cn-hangzhou.aliyuncs.com"
    status = "READY"


class MockVersionInstance:
    """模拟 Version 数据"""

    agent_runtime_version = "1"
    agent_runtime_id = "ar-123456"
    agent_runtime_name = "test-runtime"


class TestAgentRuntimeCreate:
    """AgentRuntime.create 方法测试"""

    @patch(CLIENT_PATH)
    def test_create(self, mock_client_class):
        """测试同步创建"""
        mock_client = MagicMock()
        mock_client.create.return_value = MockAgentRuntimeInstance()
        mock_client_class.return_value = mock_client

        input_obj = AgentRuntimeCreateInput(
            agent_runtime_name="test-runtime",
            code_configuration=AgentRuntimeCode(
                language=AgentRuntimeLanguage.PYTHON312,
                command=["python", "main.py"],
            ),
        )
        result = AgentRuntime.create(input_obj)

        assert result.agent_runtime_id == "ar-123456"

    @patch(CLIENT_PATH)
    def test_create_async(self, mock_client_class):
        """测试异步创建"""
        mock_client = MagicMock()
        mock_client.create_async = AsyncMock(
            return_value=MockAgentRuntimeInstance()
        )
        mock_client_class.return_value = mock_client

        input_obj = AgentRuntimeCreateInput(
            agent_runtime_name="test-runtime",
            code_configuration=AgentRuntimeCode(
                language=AgentRuntimeLanguage.PYTHON312,
                command=["python", "main.py"],
            ),
        )
        result = asyncio.run(AgentRuntime.create_async(input_obj))

        assert result.agent_runtime_id == "ar-123456"


class TestAgentRuntimeDelete:
    """AgentRuntime.delete 方法测试"""

    @patch(CLIENT_PATH)
    def test_delete_by_id(self, mock_client_class):
        """测试按 ID 同步删除"""
        mock_client = MagicMock()
        # 模拟没有 endpoints
        mock_client.list_endpoints.return_value = MagicMock(items=[])
        mock_client.delete.return_value = MockAgentRuntimeInstance()
        mock_client_class.return_value = mock_client

        result = AgentRuntime.delete_by_id("ar-123456")

        assert result.agent_runtime_id == "ar-123456"

    @patch(CLIENT_PATH)
    def test_delete_by_id_async(self, mock_client_class):
        """测试按 ID 异步删除"""
        mock_client = MagicMock()
        # 模拟没有 endpoints
        mock_client.list_endpoints_async = AsyncMock(
            return_value=MagicMock(items=[])
        )
        mock_client.delete_async = AsyncMock(
            return_value=MockAgentRuntimeInstance()
        )
        mock_client_class.return_value = mock_client

        result = asyncio.run(AgentRuntime.delete_by_id_async("ar-123456"))

        assert result.agent_runtime_id == "ar-123456"

    def test_delete_instance_without_id(self):
        """测试无 ID 实例删除抛出错误"""
        runtime = AgentRuntime()  # 没有设置 agent_runtime_id

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            runtime.delete()

    def test_delete_async_instance_without_id(self):
        """测试无 ID 实例异步删除抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            asyncio.run(runtime.delete_async())


class TestAgentRuntimeUpdate:
    """AgentRuntime.update 方法测试"""

    @patch(CLIENT_PATH)
    def test_update_by_id(self, mock_client_class):
        """测试按 ID 同步更新"""
        mock_client = MagicMock()
        mock_client.update.return_value = MockAgentRuntimeInstance()
        mock_client_class.return_value = mock_client

        input_obj = AgentRuntimeUpdateInput(description="Updated")
        result = AgentRuntime.update_by_id("ar-123456", input_obj)

        assert result.agent_runtime_id == "ar-123456"

    @patch(CLIENT_PATH)
    def test_update_by_id_async(self, mock_client_class):
        """测试按 ID 异步更新"""
        mock_client = MagicMock()
        mock_client.update_async = AsyncMock(
            return_value=MockAgentRuntimeInstance()
        )
        mock_client_class.return_value = mock_client

        input_obj = AgentRuntimeUpdateInput(description="Updated")
        result = asyncio.run(
            AgentRuntime.update_by_id_async("ar-123456", input_obj)
        )

        assert result.agent_runtime_id == "ar-123456"

    def test_update_instance_without_id(self):
        """测试无 ID 实例更新抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            runtime.update(AgentRuntimeUpdateInput())

    def test_update_async_instance_without_id(self):
        """测试无 ID 实例异步更新抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            asyncio.run(runtime.update_async(AgentRuntimeUpdateInput()))


class TestAgentRuntimeGet:
    """AgentRuntime.get 方法测试"""

    @patch(CLIENT_PATH)
    def test_get_by_id(self, mock_client_class):
        """测试按 ID 同步获取"""
        mock_client = MagicMock()
        mock_client.get.return_value = MockAgentRuntimeInstance()
        mock_client_class.return_value = mock_client

        result = AgentRuntime.get_by_id("ar-123456")

        assert result.agent_runtime_id == "ar-123456"

    @patch(CLIENT_PATH)
    def test_get_by_id_async(self, mock_client_class):
        """测试按 ID 异步获取"""
        mock_client = MagicMock()
        mock_client.get_async = AsyncMock(
            return_value=MockAgentRuntimeInstance()
        )
        mock_client_class.return_value = mock_client

        result = asyncio.run(AgentRuntime.get_by_id_async("ar-123456"))

        assert result.agent_runtime_id == "ar-123456"

    def test_get_instance_without_id(self):
        """测试无 ID 实例获取抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            runtime.get()

    def test_get_async_instance_without_id(self):
        """测试无 ID 实例异步获取抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            asyncio.run(runtime.get_async())


class TestAgentRuntimeList:
    """AgentRuntime.list 方法测试"""

    @patch(CLIENT_PATH)
    def test_list(self, mock_client_class):
        """测试同步列表"""
        mock_client = MagicMock()

        # AgentRuntimeClient.list 返回列表（不是 MagicMock(items=...)）
        mock_client.list.return_value = [MockAgentRuntimeInstance()]
        mock_client_class.return_value = mock_client

        result = AgentRuntime.list()

        assert len(result) >= 1

    @patch(CLIENT_PATH)
    def test_list_async(self, mock_client_class):
        """测试异步列表"""
        mock_client = MagicMock()

        # AgentRuntimeClient.list_async 返回列表
        mock_client.list_async = AsyncMock(
            return_value=[MockAgentRuntimeInstance()]
        )
        mock_client_class.return_value = mock_client

        result = asyncio.run(AgentRuntime.list_async())

        assert len(result) >= 1

    @patch(CLIENT_PATH)
    def test_list_with_deduplication(self, mock_client_class):
        """测试列表去重"""
        mock_client = MagicMock()

        # 返回重复的数据（两个相同 ID 的对象）
        mock_client.list.return_value = [
            MockAgentRuntimeInstance(),
            MockAgentRuntimeInstance(),
        ]
        mock_client_class.return_value = mock_client

        result = AgentRuntime.list()

        # 应该去重为 1 个
        assert len(result) == 1


class TestAgentRuntimeRefresh:
    """AgentRuntime.refresh 方法测试"""

    @patch(CLIENT_PATH)
    def test_refresh(self, mock_client_class):
        """测试同步刷新"""
        mock_client = MagicMock()
        mock_client.get.return_value = MockAgentRuntimeInstance()
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = runtime.refresh()

        assert result.agent_runtime_id == "ar-123456"

    @patch(CLIENT_PATH)
    def test_refresh_async(self, mock_client_class):
        """测试异步刷新"""
        mock_client = MagicMock()
        mock_client.get_async = AsyncMock(
            return_value=MockAgentRuntimeInstance()
        )
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = asyncio.run(runtime.refresh_async())

        assert result.agent_runtime_id == "ar-123456"


class TestAgentRuntimeEndpointOperations:
    """AgentRuntime 端点操作测试"""

    @patch(CLIENT_PATH)
    def test_create_endpoint_by_id(self, mock_client_class):
        """测试按 ID 创建端点"""
        mock_client = MagicMock()
        mock_client.create_endpoint.return_value = (
            MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        input_obj = AgentRuntimeEndpointCreateInput(
            agent_runtime_endpoint_name="test-endpoint"
        )
        result = AgentRuntime.create_endpoint_by_id("ar-123456", input_obj)

        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(CLIENT_PATH)
    def test_create_endpoint_instance(self, mock_client_class):
        """测试实例创建端点"""
        mock_client = MagicMock()
        mock_client.create_endpoint.return_value = (
            MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        input_obj = AgentRuntimeEndpointCreateInput(
            agent_runtime_endpoint_name="test-endpoint"
        )
        result = runtime.create_endpoint(input_obj)

        assert result.agent_runtime_endpoint_id == "are-123456"

    def test_create_endpoint_without_id(self):
        """测试无 ID 创建端点抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            runtime.create_endpoint(AgentRuntimeEndpointCreateInput())

    @patch(CLIENT_PATH)
    def test_delete_endpoint_by_id(self, mock_client_class):
        """测试按 ID 删除端点"""
        mock_client = MagicMock()
        mock_client.delete_endpoint.return_value = (
            MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        result = AgentRuntime.delete_endpoint_by_id("ar-123456", "are-123456")

        assert result.agent_runtime_endpoint_id == "are-123456"

    def test_delete_endpoint_without_id(self):
        """测试无 ID 删除端点抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            runtime.delete_endpoint("are-123456")

    @patch(CLIENT_PATH)
    def test_update_endpoint_by_id(self, mock_client_class):
        """测试按 ID 更新端点"""
        mock_client = MagicMock()
        mock_client.update_endpoint.return_value = (
            MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        input_obj = AgentRuntimeEndpointUpdateInput(description="Updated")
        result = AgentRuntime.update_endpoint_by_id(
            "ar-123456", "are-123456", input_obj
        )

        assert result.agent_runtime_endpoint_id == "are-123456"

    def test_update_endpoint_without_id(self):
        """测试无 ID 更新端点抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            runtime.update_endpoint(
                "are-123456", AgentRuntimeEndpointUpdateInput()
            )

    @patch(CLIENT_PATH)
    def test_get_endpoint_by_id(self, mock_client_class):
        """测试按 ID 获取端点"""
        mock_client = MagicMock()
        mock_client.get_endpoint.return_value = (
            MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        result = AgentRuntime.get_endpoint_by_id("ar-123456", "are-123456")

        assert result.agent_runtime_endpoint_id == "are-123456"

    def test_get_endpoint_without_id(self):
        """测试无 ID 获取端点抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            runtime.get_endpoint("are-123456")

    @patch(CLIENT_PATH)
    def test_list_endpoints_by_id(self, mock_client_class):
        """测试按 ID 列表端点"""
        mock_client = MagicMock()

        # AgentRuntimeClient.list_endpoints 返回列表
        mock_client.list_endpoints.return_value = [
            MockAgentRuntimeEndpointInstance()
        ]
        mock_client_class.return_value = mock_client

        result = AgentRuntime.list_endpoints_by_id("ar-123456")

        assert len(result) >= 1

    def test_list_endpoints_without_id(self):
        """测试无 ID 列表端点抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            runtime.list_endpoints()


class TestAgentRuntimeVersionOperations:
    """AgentRuntime 版本操作测试"""

    @patch(CLIENT_PATH)
    def test_list_versions_by_id(self, mock_client_class):
        """测试按 ID 列表版本"""
        mock_client = MagicMock()
        # AgentRuntimeClient.list_versions 返回列表
        mock_client.list_versions.return_value = [MockVersionInstance()]
        mock_client_class.return_value = mock_client

        result = AgentRuntime.list_versions_by_id("ar-123456")

        assert len(result) >= 1

    @patch(CLIENT_PATH)
    def test_list_versions_by_id_async(self, mock_client_class):
        """测试按 ID 异步列表版本"""
        mock_client = MagicMock()
        mock_client.list_versions_async = AsyncMock(
            return_value=[MockVersionInstance()]
        )
        mock_client_class.return_value = mock_client

        result = asyncio.run(
            AgentRuntime.list_versions_by_id_async("ar-123456")
        )

        assert len(result) >= 1

    @patch(CLIENT_PATH)
    def test_list_versions_instance(self, mock_client_class):
        """测试实例列表版本"""
        mock_client = MagicMock()
        mock_client.list_versions.return_value = [MockVersionInstance()]
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = runtime.list_versions()

        assert len(result) >= 1

    def test_list_versions_without_id(self):
        """测试无 ID 列表版本抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            runtime.list_versions()

    def test_list_versions_async_without_id(self):
        """测试无 ID 异步列表版本抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            asyncio.run(runtime.list_versions_async())


class TestAgentRuntimeListAll:
    """AgentRuntime.list_all 方法测试"""

    @patch(CLIENT_PATH)
    def test_list_all(self, mock_client_class):
        """测试 list_all"""
        mock_client = MagicMock()
        # AgentRuntimeClient.list 返回列表
        mock_client.list.return_value = [MockAgentRuntimeInstance()]
        mock_client_class.return_value = mock_client

        result = AgentRuntime.list_all()

        assert len(result) >= 1

    @patch(CLIENT_PATH)
    def test_list_all_with_filters(self, mock_client_class):
        """测试带过滤器的 list_all"""
        mock_client = MagicMock()
        mock_client.list.return_value = [MockAgentRuntimeInstance()]
        mock_client_class.return_value = mock_client

        result = AgentRuntime.list_all(
            agent_runtime_name="test",
            tags="env:prod",
            search_mode="prefix",
        )

        assert len(result) >= 1

    @patch(CLIENT_PATH)
    def test_list_all_async(self, mock_client_class):
        """测试异步 list_all"""
        mock_client = MagicMock()
        mock_client.list_async = AsyncMock(
            return_value=[MockAgentRuntimeInstance()]
        )
        mock_client_class.return_value = mock_client

        result = asyncio.run(AgentRuntime.list_all_async())

        assert len(result) >= 1


class TestAgentRuntimeInstanceMethods:
    """AgentRuntime 实例方法成功路径测试"""

    @patch(CLIENT_PATH)
    def test_delete_instance_success(self, mock_client_class):
        """测试实例删除成功路径"""
        mock_client = MagicMock()
        mock_client.list_endpoints.return_value = []
        mock_client.delete.return_value = MockAgentRuntimeInstance()
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = runtime.delete()

        assert result.agent_runtime_id == "ar-123456"
        mock_client.delete.assert_called_once()

    @patch(CLIENT_PATH)
    def test_delete_async_instance_success(self, mock_client_class):
        """测试实例异步删除成功路径"""
        mock_client = MagicMock()
        mock_client.list_endpoints_async = AsyncMock(return_value=[])
        mock_client.delete_async = AsyncMock(
            return_value=MockAgentRuntimeInstance()
        )
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = asyncio.run(runtime.delete_async())

        assert result.agent_runtime_id == "ar-123456"

    @patch(CLIENT_PATH)
    def test_update_instance_success(self, mock_client_class):
        """测试实例更新成功路径"""
        mock_client = MagicMock()
        mock_client.update.return_value = MockAgentRuntimeInstance()
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = runtime.update(AgentRuntimeUpdateInput(description="Updated"))

        assert result.agent_runtime_id == "ar-123456"
        mock_client.update.assert_called_once()

    @patch(CLIENT_PATH)
    def test_update_async_instance_success(self, mock_client_class):
        """测试实例异步更新成功路径"""
        mock_client = MagicMock()
        mock_client.update_async = AsyncMock(
            return_value=MockAgentRuntimeInstance()
        )
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = asyncio.run(
            runtime.update_async(AgentRuntimeUpdateInput(description="Updated"))
        )

        assert result.agent_runtime_id == "ar-123456"

    @patch(CLIENT_PATH)
    def test_get_instance_success(self, mock_client_class):
        """测试实例获取成功路径"""
        mock_client = MagicMock()
        mock_client.get.return_value = MockAgentRuntimeInstance()
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = runtime.get()

        assert result.agent_runtime_id == "ar-123456"
        mock_client.get.assert_called_once()

    @patch(CLIENT_PATH)
    def test_get_async_instance_success(self, mock_client_class):
        """测试实例异步获取成功路径"""
        mock_client = MagicMock()
        mock_client.get_async = AsyncMock(
            return_value=MockAgentRuntimeInstance()
        )
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = asyncio.run(runtime.get_async())

        assert result.agent_runtime_id == "ar-123456"

    @patch(CLIENT_PATH)
    def test_refresh_async_success(self, mock_client_class):
        """测试实例异步刷新成功路径"""
        mock_client = MagicMock()
        mock_client.get_async = AsyncMock(
            return_value=MockAgentRuntimeInstance()
        )
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = asyncio.run(runtime.refresh_async())

        assert result.agent_runtime_id == "ar-123456"

    @patch(CLIENT_PATH)
    def test_create_endpoint_async_success(self, mock_client_class):
        """测试实例异步创建端点成功路径"""
        mock_client = MagicMock()
        mock_client.create_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = asyncio.run(
            runtime.create_endpoint_async(AgentRuntimeEndpointCreateInput())
        )

        assert result.agent_runtime_endpoint_id == "are-123456"

    def test_create_endpoint_async_without_id(self):
        """测试无 ID 实例创建端点抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            asyncio.run(
                runtime.create_endpoint_async(AgentRuntimeEndpointCreateInput())
            )

    @patch(CLIENT_PATH)
    def test_delete_endpoint_async_success(self, mock_client_class):
        """测试实例异步删除端点成功路径"""
        mock_client = MagicMock()
        mock_client.delete_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = asyncio.run(runtime.delete_endpoint_async("are-123456"))

        assert result.agent_runtime_endpoint_id == "are-123456"

    def test_delete_endpoint_async_without_id(self):
        """测试无 ID 实例删除端点抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            asyncio.run(runtime.delete_endpoint_async("are-123456"))

    @patch(CLIENT_PATH)
    def test_update_endpoint_async_success(self, mock_client_class):
        """测试实例异步更新端点成功路径"""
        mock_client = MagicMock()
        mock_client.update_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = asyncio.run(
            runtime.update_endpoint_async(
                "are-123456", AgentRuntimeEndpointUpdateInput()
            )
        )

        assert result.agent_runtime_endpoint_id == "are-123456"

    def test_update_endpoint_async_without_id(self):
        """测试无 ID 实例更新端点抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            asyncio.run(
                runtime.update_endpoint_async(
                    "are-123456", AgentRuntimeEndpointUpdateInput()
                )
            )

    @patch(CLIENT_PATH)
    def test_get_endpoint_async_success(self, mock_client_class):
        """测试实例异步获取端点成功路径"""
        mock_client = MagicMock()
        mock_client.get_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = asyncio.run(runtime.get_endpoint_async("are-123456"))

        assert result.agent_runtime_endpoint_id == "are-123456"

    def test_get_endpoint_async_without_id(self):
        """测试无 ID 实例获取端点抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            asyncio.run(runtime.get_endpoint_async("are-123456"))

    @patch(CLIENT_PATH)
    def test_list_endpoints_async_success(self, mock_client_class):
        """测试实例异步列出端点成功路径"""
        mock_client = MagicMock()
        mock_client.list_endpoints_async = AsyncMock(
            return_value=[MockAgentRuntimeEndpointInstance()]
        )
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = asyncio.run(runtime.list_endpoints_async())

        assert len(result) == 1

    def test_list_endpoints_async_without_id(self):
        """测试无 ID 实例列出端点抛出错误"""
        runtime = AgentRuntime()

        with pytest.raises(ValueError, match="agent_runtime_id is required"):
            asyncio.run(runtime.list_endpoints_async())

    @patch(CLIENT_PATH)
    def test_list_versions_async_success(self, mock_client_class):
        """测试实例异步列出版本成功路径"""
        mock_client = MagicMock()
        mock_client.list_versions_async = AsyncMock(
            return_value=[MockVersionInstance()]
        )
        mock_client_class.return_value = mock_client

        runtime = AgentRuntime(agent_runtime_id="ar-123456")
        result = asyncio.run(runtime.list_versions_async())

        assert len(result) == 1
