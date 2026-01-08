"""Agent Runtime 端点资源单元测试"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.agent_runtime.endpoint import AgentRuntimeEndpoint
from agentrun.agent_runtime.model import (
    AgentRuntimeEndpointCreateInput,
    AgentRuntimeEndpointUpdateInput,
)
from agentrun.utils.config import Config

# Mock path for AgentRuntimeClient - 在使用处 mock
# AgentRuntimeEndpoint.__get_client 动态导入 AgentRuntimeClient
CLIENT_PATH = "agentrun.agent_runtime.client.AgentRuntimeClient"


class ConcreteAgentRuntimeEndpoint(AgentRuntimeEndpoint):
    """具体实现用于测试，因为 AgentRuntimeEndpoint 是抽象类"""

    @classmethod
    def _list_page(cls, page_input, config=None, **kwargs):
        return []

    @classmethod
    async def _list_page_async(cls, page_input, config=None, **kwargs):
        return []


class MockAgentRuntimeEndpointInstance:
    """模拟 AgentRuntimeEndpoint 实例（避免抽象类实例化问题）"""

    agent_runtime_endpoint_id = "are-123456"
    agent_runtime_endpoint_name = "test-endpoint"
    agent_runtime_id = "ar-123456"
    endpoint_public_url = "https://test.agentrun.cn-hangzhou.aliyuncs.com"
    status = "READY"


class MockAgentRuntimeInstance:
    """模拟 AgentRuntime 数据"""

    agent_runtime_id = "ar-123456"
    agent_runtime_name = "test-runtime"


class TestAgentRuntimeEndpointCreateById:
    """AgentRuntimeEndpoint.create_by_id 方法测试"""

    @patch(CLIENT_PATH)
    def test_create_by_id(self, mock_client_class):
        """测试同步创建"""
        mock_client = MagicMock()
        mock_client.create_endpoint.return_value = (
            MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        input_obj = AgentRuntimeEndpointCreateInput(
            agent_runtime_endpoint_name="test-endpoint"
        )
        result = AgentRuntimeEndpoint.create_by_id("ar-123456", input_obj)

        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(CLIENT_PATH)
    def test_create_by_id_async(self, mock_client_class):
        """测试异步创建"""
        mock_client = MagicMock()
        mock_client.create_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        input_obj = AgentRuntimeEndpointCreateInput(
            agent_runtime_endpoint_name="test-endpoint"
        )
        result = asyncio.run(
            AgentRuntimeEndpoint.create_by_id_async("ar-123456", input_obj)
        )

        assert result.agent_runtime_endpoint_id == "are-123456"


class TestAgentRuntimeEndpointDeleteById:
    """AgentRuntimeEndpoint.delete_by_id 方法测试"""

    @patch(CLIENT_PATH)
    def test_delete_by_id(self, mock_client_class):
        """测试同步删除"""
        mock_client = MagicMock()
        mock_client.delete_endpoint.return_value = (
            MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        result = AgentRuntimeEndpoint.delete_by_id("ar-123456", "are-123456")

        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(CLIENT_PATH)
    def test_delete_by_id_async(self, mock_client_class):
        """测试异步删除"""
        mock_client = MagicMock()
        mock_client.delete_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        result = asyncio.run(
            AgentRuntimeEndpoint.delete_by_id_async("ar-123456", "are-123456")
        )

        assert result.agent_runtime_endpoint_id == "are-123456"


class TestAgentRuntimeEndpointUpdateById:
    """AgentRuntimeEndpoint.update_by_id 方法测试"""

    @patch(CLIENT_PATH)
    def test_update_by_id(self, mock_client_class):
        """测试同步更新"""
        mock_client = MagicMock()
        mock_client.update_endpoint.return_value = (
            MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        input_obj = AgentRuntimeEndpointUpdateInput(description="Updated")
        result = AgentRuntimeEndpoint.update_by_id(
            "ar-123456", "are-123456", input_obj
        )

        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(CLIENT_PATH)
    def test_update_by_id_async(self, mock_client_class):
        """测试异步更新"""
        mock_client = MagicMock()
        mock_client.update_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        input_obj = AgentRuntimeEndpointUpdateInput(description="Updated")
        result = asyncio.run(
            AgentRuntimeEndpoint.update_by_id_async(
                "ar-123456", "are-123456", input_obj
            )
        )

        assert result.agent_runtime_endpoint_id == "are-123456"


class TestAgentRuntimeEndpointGetById:
    """AgentRuntimeEndpoint.get_by_id 方法测试"""

    @patch(CLIENT_PATH)
    def test_get_by_id(self, mock_client_class):
        """测试同步获取"""
        mock_client = MagicMock()
        mock_client.get_endpoint.return_value = (
            MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        result = AgentRuntimeEndpoint.get_by_id("ar-123456", "are-123456")

        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(CLIENT_PATH)
    def test_get_by_id_async(self, mock_client_class):
        """测试异步获取"""
        mock_client = MagicMock()
        mock_client.get_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        result = asyncio.run(
            AgentRuntimeEndpoint.get_by_id_async("ar-123456", "are-123456")
        )

        assert result.agent_runtime_endpoint_id == "are-123456"


class TestAgentRuntimeEndpointListById:
    """AgentRuntimeEndpoint.list_by_id 方法测试"""

    @patch(CLIENT_PATH)
    def test_list_by_id(self, mock_client_class):
        """测试同步列表"""
        mock_client = MagicMock()
        # AgentRuntimeClient.list_endpoints 返回列表
        mock_client.list_endpoints.return_value = [
            MockAgentRuntimeEndpointInstance()
        ]
        mock_client_class.return_value = mock_client

        result = AgentRuntimeEndpoint.list_by_id("ar-123456")

        assert len(result) >= 1

    @patch(CLIENT_PATH)
    def test_list_by_id_async(self, mock_client_class):
        """测试异步列表"""
        mock_client = MagicMock()
        mock_client.list_endpoints_async = AsyncMock(
            return_value=[MockAgentRuntimeEndpointInstance()]
        )
        mock_client_class.return_value = mock_client

        result = asyncio.run(AgentRuntimeEndpoint.list_by_id_async("ar-123456"))

        assert len(result) >= 1

    @patch(CLIENT_PATH)
    def test_list_by_id_with_deduplication(self, mock_client_class):
        """测试列表去重"""
        mock_client = MagicMock()
        # 返回重复数据（两个相同 ID 的端点）
        mock_client.list_endpoints.return_value = [
            MockAgentRuntimeEndpointInstance(),
            MockAgentRuntimeEndpointInstance(),
        ]
        mock_client_class.return_value = mock_client

        result = AgentRuntimeEndpoint.list_by_id("ar-123456")

        # 应该去重为 1 个
        assert len(result) == 1


class TestAgentRuntimeEndpointInstanceDelete:
    """AgentRuntimeEndpoint 实例 delete 方法测试"""

    @patch(CLIENT_PATH)
    def test_delete(self, mock_client_class):
        """测试实例同步删除"""
        mock_client = MagicMock()
        mock_client.delete_endpoint.return_value = (
            MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        endpoint = ConcreteAgentRuntimeEndpoint(
            agent_runtime_id="ar-123456",
            agent_runtime_endpoint_id="are-123456",
        )
        result = endpoint.delete()

        assert result is endpoint
        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(CLIENT_PATH)
    def test_delete_async(self, mock_client_class):
        """测试实例异步删除"""
        mock_client = MagicMock()
        mock_client.delete_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        endpoint = ConcreteAgentRuntimeEndpoint(
            agent_runtime_id="ar-123456",
            agent_runtime_endpoint_id="are-123456",
        )
        result = asyncio.run(endpoint.delete_async())

        assert result is endpoint

    def test_delete_without_ids(self):
        """测试无 ID 删除抛出错误"""
        endpoint = ConcreteAgentRuntimeEndpoint()

        with pytest.raises(
            ValueError,
            match="agent_runtime_id and agent_runtime_endpoint_id are required",
        ):
            endpoint.delete()

    def test_delete_async_without_ids(self):
        """测试无 ID 异步删除抛出错误"""
        endpoint = ConcreteAgentRuntimeEndpoint()

        with pytest.raises(
            ValueError,
            match="agent_runtime_id and agent_runtime_endpoint_id are required",
        ):
            asyncio.run(endpoint.delete_async())

    def test_delete_without_endpoint_id(self):
        """测试只有 runtime ID 删除抛出错误"""
        endpoint = ConcreteAgentRuntimeEndpoint(agent_runtime_id="ar-123456")

        with pytest.raises(
            ValueError,
            match="agent_runtime_id and agent_runtime_endpoint_id are required",
        ):
            endpoint.delete()


class TestAgentRuntimeEndpointInstanceUpdate:
    """AgentRuntimeEndpoint 实例 update 方法测试"""

    @patch(CLIENT_PATH)
    def test_update(self, mock_client_class):
        """测试实例同步更新"""
        mock_client = MagicMock()
        mock_client.update_endpoint.return_value = (
            MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        endpoint = ConcreteAgentRuntimeEndpoint(
            agent_runtime_id="ar-123456",
            agent_runtime_endpoint_id="are-123456",
        )
        input_obj = AgentRuntimeEndpointUpdateInput(description="Updated")
        result = endpoint.update(input_obj)

        assert result is endpoint

    @patch(CLIENT_PATH)
    def test_update_async(self, mock_client_class):
        """测试实例异步更新"""
        mock_client = MagicMock()
        mock_client.update_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        endpoint = ConcreteAgentRuntimeEndpoint(
            agent_runtime_id="ar-123456",
            agent_runtime_endpoint_id="are-123456",
        )
        input_obj = AgentRuntimeEndpointUpdateInput(description="Updated")
        result = asyncio.run(endpoint.update_async(input_obj))

        assert result is endpoint

    def test_update_without_ids(self):
        """测试无 ID 更新抛出错误"""
        endpoint = ConcreteAgentRuntimeEndpoint()

        with pytest.raises(
            ValueError,
            match="agent_runtime_id and agent_runtime_endpoint_id are required",
        ):
            endpoint.update(AgentRuntimeEndpointUpdateInput())


class TestAgentRuntimeEndpointInstanceGet:
    """AgentRuntimeEndpoint 实例 get 方法测试"""

    @patch(CLIENT_PATH)
    def test_get(self, mock_client_class):
        """测试实例同步获取"""
        mock_client = MagicMock()
        mock_client.get_endpoint.return_value = (
            MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        endpoint = ConcreteAgentRuntimeEndpoint(
            agent_runtime_id="ar-123456",
            agent_runtime_endpoint_id="are-123456",
        )
        result = endpoint.get()

        assert result is endpoint

    @patch(CLIENT_PATH)
    def test_get_async(self, mock_client_class):
        """测试实例异步获取"""
        mock_client = MagicMock()
        mock_client.get_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        endpoint = ConcreteAgentRuntimeEndpoint(
            agent_runtime_id="ar-123456",
            agent_runtime_endpoint_id="are-123456",
        )
        result = asyncio.run(endpoint.get_async())

        assert result is endpoint

    def test_get_without_ids(self):
        """测试无 ID 获取抛出错误"""
        endpoint = ConcreteAgentRuntimeEndpoint()

        with pytest.raises(
            ValueError,
            match="agent_runtime_id and agent_runtime_endpoint_id are required",
        ):
            endpoint.get()


class TestAgentRuntimeEndpointRefresh:
    """AgentRuntimeEndpoint refresh 方法测试"""

    @patch(CLIENT_PATH)
    def test_refresh(self, mock_client_class):
        """测试同步刷新"""
        mock_client = MagicMock()
        mock_client.get_endpoint.return_value = (
            MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        endpoint = ConcreteAgentRuntimeEndpoint(
            agent_runtime_id="ar-123456",
            agent_runtime_endpoint_id="are-123456",
        )
        result = endpoint.refresh()

        assert result is endpoint

    @patch(CLIENT_PATH)
    def test_refresh_async(self, mock_client_class):
        """测试异步刷新"""
        mock_client = MagicMock()
        mock_client.get_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointInstance()
        )
        mock_client_class.return_value = mock_client

        endpoint = ConcreteAgentRuntimeEndpoint(
            agent_runtime_id="ar-123456",
            agent_runtime_endpoint_id="are-123456",
        )
        result = asyncio.run(endpoint.refresh_async())

        assert result is endpoint


class TestAgentRuntimeEndpointInvokeOpenai:
    """AgentRuntimeEndpoint invoke_openai 方法测试

    注意：invoke_openai 方法使用私有属性 __data_api，这在测试中使用子类时会有问题。
    这些测试通过 test_data.py 中对 AgentRuntimeDataAPI 的测试来覆盖。
    """

    @pytest.mark.skip(
        reason=(
            "invoke_openai uses private __data_api which is complex to test"
            " with subclass"
        )
    )
    @patch("agentrun.agent_runtime.api.data.AgentRuntimeDataAPI")
    @patch(CLIENT_PATH)
    def test_invoke_openai(self, mock_client_class, mock_data_api_class):
        """测试 invoke_openai - 跳过因为私有属性问题"""
        pass

    @pytest.mark.skip(
        reason=(
            "invoke_openai_async uses private __data_api which is complex to"
            " test with subclass"
        )
    )
    @patch("agentrun.agent_runtime.api.data.AgentRuntimeDataAPI")
    @patch(CLIENT_PATH)
    def test_invoke_openai_async(self, mock_client_class, mock_data_api_class):
        """测试 invoke_openai_async - 跳过因为私有属性问题"""
        pass
