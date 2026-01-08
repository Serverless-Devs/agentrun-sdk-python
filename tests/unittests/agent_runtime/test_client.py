"""Agent Runtime 客户端单元测试"""

import asyncio
import os
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.agent_runtime.model import (
    AgentRuntimeArtifact,
    AgentRuntimeCode,
    AgentRuntimeContainer,
    AgentRuntimeCreateInput,
    AgentRuntimeEndpointCreateInput,
    AgentRuntimeEndpointListInput,
    AgentRuntimeEndpointUpdateInput,
    AgentRuntimeLanguage,
    AgentRuntimeListInput,
    AgentRuntimeUpdateInput,
    AgentRuntimeVersionListInput,
)
from agentrun.utils.config import Config
from agentrun.utils.exception import (
    HTTPError,
    ResourceAlreadyExistError,
    ResourceNotExistError,
)

# Mock path for AgentRuntimeControlAPI - 在使用处 mock
# 需要 mock client.py 中导入的引用
CONTROL_API_PATH = "agentrun.agent_runtime.client.AgentRuntimeControlAPI"
ENDPOINT_FROM_INNER_PATH = (
    "agentrun.agent_runtime.client.AgentRuntimeEndpoint.from_inner_object"
)


class MockAgentRuntimeData:
    """模拟 AgentRuntime 数据"""

    agent_runtime_id = "ar-123456"
    agent_runtime_name = "test-runtime"
    agent_runtime_arn = "arn:acs:agentrun:cn-hangzhou:123456:agent/test"
    status = "READY"

    def to_map(self):
        return {
            "agentRuntimeId": self.agent_runtime_id,
            "agentRuntimeName": self.agent_runtime_name,
            "agentRuntimeArn": self.agent_runtime_arn,
            "status": self.status,
        }


class MockAgentRuntimeEndpointData:
    """模拟 AgentRuntimeEndpoint 数据"""

    agent_runtime_endpoint_id = "are-123456"
    agent_runtime_endpoint_name = "test-endpoint"
    agent_runtime_id = "ar-123456"
    endpoint_public_url = "https://test.agentrun.cn-hangzhou.aliyuncs.com"
    status = "READY"

    def to_map(self):
        return {
            "agentRuntimeEndpointId": self.agent_runtime_endpoint_id,
            "agentRuntimeEndpointName": self.agent_runtime_endpoint_name,
            "agentRuntimeId": self.agent_runtime_id,
            "endpointPublicUrl": self.endpoint_public_url,
            "status": self.status,
        }


class MockListOutput:
    """模拟 List 输出"""

    def __init__(self, items):
        self.items = items


class TestAgentRuntimeClientInit:
    """AgentRuntimeClient 初始化测试"""

    @patch(CONTROL_API_PATH)
    def test_init_without_config(self, mock_control_api):
        from agentrun.agent_runtime.client import AgentRuntimeClient

        client = AgentRuntimeClient()
        assert client.config is None

    @patch(CONTROL_API_PATH)
    def test_init_with_config(self, mock_control_api):
        from agentrun.agent_runtime.client import AgentRuntimeClient

        config = Config(
            access_key_id="test-key",
            access_key_secret="test-secret",
        )
        client = AgentRuntimeClient(config=config)
        assert client.config == config


class TestAgentRuntimeClientCreate:
    """AgentRuntimeClient.create 方法测试"""

    @patch(CONTROL_API_PATH)
    def test_create_with_code_configuration(self, mock_control_api_class):
        """测试使用代码配置创建"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.create_agent_runtime.return_value = (
            MockAgentRuntimeData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeCreateInput(
            agent_runtime_name="test-runtime",
            code_configuration=AgentRuntimeCode(
                language=AgentRuntimeLanguage.PYTHON312,
                command=["python", "main.py"],
                zip_file="base64data",
            ),
        )
        result = client.create(input_obj)

        assert result.agent_runtime_id == "ar-123456"
        mock_control_api.create_agent_runtime.assert_called_once()

    @patch(CONTROL_API_PATH)
    def test_create_with_container_configuration(self, mock_control_api_class):
        """测试使用容器配置创建"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.create_agent_runtime.return_value = (
            MockAgentRuntimeData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeCreateInput(
            agent_runtime_name="test-runtime",
            container_configuration=AgentRuntimeContainer(
                image="registry.cn-hangzhou.aliyuncs.com/test/agent:v1",
                command=["python", "app.py"],
            ),
        )
        result = client.create(input_obj)

        assert result.agent_runtime_id == "ar-123456"

    @patch(CONTROL_API_PATH)
    def test_create_without_configuration_raises_error(
        self, mock_control_api_class
    ):
        """测试无配置时抛出错误"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeCreateInput(agent_runtime_name="test-runtime")

        with pytest.raises(
            ValueError, match="Either code_configuration or image_configuration"
        ):
            client.create(input_obj)

    @patch(CONTROL_API_PATH)
    def test_create_with_http_error(self, mock_control_api_class):
        """测试 HTTP 错误处理"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.create_agent_runtime.side_effect = HTTPError(
            409, "resource already exists"
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeCreateInput(
            agent_runtime_name="test-runtime",
            code_configuration=AgentRuntimeCode(
                language=AgentRuntimeLanguage.PYTHON312,
                command=["python", "main.py"],
            ),
        )

        with pytest.raises(ResourceAlreadyExistError):
            client.create(input_obj)

    @patch(CONTROL_API_PATH)
    def test_create_async(self, mock_control_api_class):
        """测试异步创建"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.create_agent_runtime_async = AsyncMock(
            return_value=MockAgentRuntimeData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeCreateInput(
            agent_runtime_name="test-runtime",
            code_configuration=AgentRuntimeCode(
                language=AgentRuntimeLanguage.PYTHON312,
                command=["python", "main.py"],
            ),
        )

        result = asyncio.run(client.create_async(input_obj))
        assert result.agent_runtime_id == "ar-123456"

    @patch(CONTROL_API_PATH)
    def test_create_async_http_error(self, mock_control_api_class):
        """测试异步创建时的 HTTP 错误"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.create_agent_runtime_async = AsyncMock(
            side_effect=HTTPError(409, "resource already exists")
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeCreateInput(
            agent_runtime_name="test-runtime",
            code_configuration=AgentRuntimeCode(
                language=AgentRuntimeLanguage.PYTHON312,
                command=["python", "main.py"],
            ),
        )

        with pytest.raises(ResourceAlreadyExistError):
            asyncio.run(client.create_async(input_obj))

    @patch(CONTROL_API_PATH)
    def test_create_async_no_configuration(self, mock_control_api_class):
        """测试异步创建时缺少配置"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeCreateInput(
            agent_runtime_name="test-runtime",
            # No code_configuration or container_configuration
        )

        with pytest.raises(ValueError, match="Either code_configuration"):
            asyncio.run(client.create_async(input_obj))


class TestAgentRuntimeClientDelete:
    """AgentRuntimeClient.delete 方法测试"""

    @patch(CONTROL_API_PATH)
    def test_delete(self, mock_control_api_class):
        """测试删除"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.delete_agent_runtime.return_value = (
            MockAgentRuntimeData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        result = client.delete("ar-123456")

        assert result.agent_runtime_id == "ar-123456"
        mock_control_api.delete_agent_runtime.assert_called_once_with(
            "ar-123456", config=None
        )

    @patch(CONTROL_API_PATH)
    def test_delete_not_found(self, mock_control_api_class):
        """测试删除不存在的资源"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.delete_agent_runtime.side_effect = HTTPError(
            404, "resource not found"
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        with pytest.raises(ResourceNotExistError):
            client.delete("ar-notfound")

    @patch(CONTROL_API_PATH)
    def test_delete_async(self, mock_control_api_class):
        """测试异步删除"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.delete_agent_runtime_async = AsyncMock(
            return_value=MockAgentRuntimeData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        result = asyncio.run(client.delete_async("ar-123456"))
        assert result.agent_runtime_id == "ar-123456"

    @patch(CONTROL_API_PATH)
    def test_delete_async_not_found(self, mock_control_api_class):
        """测试异步删除不存在的资源"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.delete_agent_runtime_async = AsyncMock(
            side_effect=HTTPError(404, "resource not found")
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()

        with pytest.raises(ResourceNotExistError):
            asyncio.run(client.delete_async("ar-notfound"))


class TestAgentRuntimeClientUpdate:
    """AgentRuntimeClient.update 方法测试"""

    @patch(CONTROL_API_PATH)
    def test_update(self, mock_control_api_class):
        """测试更新"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.update_agent_runtime.return_value = (
            MockAgentRuntimeData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeUpdateInput(description="Updated description")
        result = client.update("ar-123456", input_obj)

        assert result.agent_runtime_id == "ar-123456"

    @patch(CONTROL_API_PATH)
    def test_update_not_found(self, mock_control_api_class):
        """测试更新不存在的资源"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.update_agent_runtime.side_effect = HTTPError(
            404, "resource not found"
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeUpdateInput(description="Updated description")
        with pytest.raises(ResourceNotExistError):
            client.update("ar-notfound", input_obj)

    @patch(CONTROL_API_PATH)
    def test_update_async(self, mock_control_api_class):
        """测试异步更新"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.update_agent_runtime_async = AsyncMock(
            return_value=MockAgentRuntimeData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeUpdateInput(description="Updated")
        result = asyncio.run(client.update_async("ar-123456", input_obj))
        assert result.agent_runtime_id == "ar-123456"

    @patch(CONTROL_API_PATH)
    def test_update_async_not_found(self, mock_control_api_class):
        """测试异步更新不存在的资源"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.update_agent_runtime_async = AsyncMock(
            side_effect=HTTPError(404, "resource not found")
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeUpdateInput(description="Updated")

        with pytest.raises(ResourceNotExistError):
            asyncio.run(client.update_async("ar-notfound", input_obj))


class TestAgentRuntimeClientGet:
    """AgentRuntimeClient.get 方法测试"""

    @patch(CONTROL_API_PATH)
    def test_get(self, mock_control_api_class):
        """测试获取"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.get_agent_runtime.return_value = MockAgentRuntimeData()
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        result = client.get("ar-123456")

        assert result.agent_runtime_id == "ar-123456"

    @patch(CONTROL_API_PATH)
    def test_get_not_found(self, mock_control_api_class):
        """测试获取不存在的资源"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.get_agent_runtime.side_effect = HTTPError(
            404, "resource not found"
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        with pytest.raises(ResourceNotExistError):
            client.get("ar-notfound")

    @patch(CONTROL_API_PATH)
    def test_get_async(self, mock_control_api_class):
        """测试异步获取"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.get_agent_runtime_async = AsyncMock(
            return_value=MockAgentRuntimeData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        result = asyncio.run(client.get_async("ar-123456"))
        assert result.agent_runtime_id == "ar-123456"

    @patch(CONTROL_API_PATH)
    def test_get_async_not_found(self, mock_control_api_class):
        """测试异步获取不存在的资源"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.get_agent_runtime_async = AsyncMock(
            side_effect=HTTPError(404, "resource not found")
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()

        with pytest.raises(ResourceNotExistError):
            asyncio.run(client.get_async("ar-notfound"))


class TestAgentRuntimeClientList:
    """AgentRuntimeClient.list 方法测试"""

    @patch(CONTROL_API_PATH)
    def test_list(self, mock_control_api_class):
        """测试列表"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.list_agent_runtimes.return_value = MockListOutput(
            [MockAgentRuntimeData(), MockAgentRuntimeData()]
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        result = client.list()

        assert len(result) == 2

    @patch(CONTROL_API_PATH)
    def test_list_with_input(self, mock_control_api_class):
        """测试带参数列表"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.list_agent_runtimes.return_value = MockListOutput(
            [MockAgentRuntimeData()]
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeListInput(agent_runtime_name="test")
        result = client.list(input_obj)

        assert len(result) == 1

    @patch(CONTROL_API_PATH)
    def test_list_http_error(self, mock_control_api_class):
        """测试列表 HTTP 错误"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.list_agent_runtimes.side_effect = HTTPError(
            500, "server error"
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        with pytest.raises(HTTPError):
            client.list()

    @patch(CONTROL_API_PATH)
    def test_list_async(self, mock_control_api_class):
        """测试异步列表"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.list_agent_runtimes_async = AsyncMock(
            return_value=MockListOutput([MockAgentRuntimeData()])
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        result = asyncio.run(client.list_async())
        assert len(result) == 1


class MockEndpointInstance:
    """模拟 AgentRuntimeEndpoint 实例 (避免抽象类实例化问题)"""

    agent_runtime_endpoint_id = "are-123456"
    agent_runtime_endpoint_name = "test-endpoint"
    agent_runtime_id = "ar-123456"
    endpoint_public_url = "https://test.agentrun.cn-hangzhou.aliyuncs.com"
    status = "READY"


class TestAgentRuntimeClientEndpoint:
    """AgentRuntimeClient Endpoint 方法测试"""

    @patch(ENDPOINT_FROM_INNER_PATH)
    @patch(CONTROL_API_PATH)
    def test_create_endpoint(self, mock_control_api_class, mock_from_inner):
        """测试创建端点"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.create_agent_runtime_endpoint.return_value = (
            MockAgentRuntimeEndpointData()
        )
        mock_control_api_class.return_value = mock_control_api
        mock_from_inner.return_value = MockEndpointInstance()

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeEndpointCreateInput(
            agent_runtime_endpoint_name="test-endpoint"
        )
        result = client.create_endpoint("ar-123456", input_obj)

        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(CONTROL_API_PATH)
    def test_create_endpoint_http_error(self, mock_control_api_class):
        """测试创建端点 HTTP 错误"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.create_agent_runtime_endpoint.side_effect = HTTPError(
            409, "endpoint already exists"
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeEndpointCreateInput(
            agent_runtime_endpoint_name="test-endpoint"
        )
        with pytest.raises(ResourceAlreadyExistError):
            client.create_endpoint("ar-123456", input_obj)

    @patch(ENDPOINT_FROM_INNER_PATH)
    @patch(CONTROL_API_PATH)
    def test_create_endpoint_async(
        self, mock_control_api_class, mock_from_inner
    ):
        """测试异步创建端点"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.create_agent_runtime_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointData()
        )
        mock_control_api_class.return_value = mock_control_api
        mock_from_inner.return_value = MockEndpointInstance()

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeEndpointCreateInput(
            agent_runtime_endpoint_name="test-endpoint"
        )
        result = asyncio.run(
            client.create_endpoint_async("ar-123456", input_obj)
        )
        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(ENDPOINT_FROM_INNER_PATH)
    @patch(CONTROL_API_PATH)
    def test_delete_endpoint(self, mock_control_api_class, mock_from_inner):
        """测试删除端点"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.delete_agent_runtime_endpoint.return_value = (
            MockAgentRuntimeEndpointData()
        )
        mock_control_api_class.return_value = mock_control_api
        mock_from_inner.return_value = MockEndpointInstance()

        client = AgentRuntimeClient()
        result = client.delete_endpoint("ar-123456", "are-123456")

        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(CONTROL_API_PATH)
    def test_delete_endpoint_not_found(self, mock_control_api_class):
        """测试删除不存在的端点"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.delete_agent_runtime_endpoint.side_effect = HTTPError(
            404, "endpoint not found"
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        with pytest.raises(ResourceNotExistError):
            client.delete_endpoint("ar-123456", "are-notfound")

    @patch(ENDPOINT_FROM_INNER_PATH)
    @patch(CONTROL_API_PATH)
    def test_delete_endpoint_async(
        self, mock_control_api_class, mock_from_inner
    ):
        """测试异步删除端点"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.delete_agent_runtime_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointData()
        )
        mock_from_inner.return_value = MockEndpointInstance()
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        result = asyncio.run(
            client.delete_endpoint_async("ar-123456", "are-123456")
        )
        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(ENDPOINT_FROM_INNER_PATH)
    @patch(CONTROL_API_PATH)
    def test_update_endpoint(self, mock_control_api_class, mock_from_inner):
        """测试更新端点"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.update_agent_runtime_endpoint.return_value = (
            MockAgentRuntimeEndpointData()
        )
        mock_control_api_class.return_value = mock_control_api
        mock_from_inner.return_value = MockEndpointInstance()

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeEndpointUpdateInput(description="Updated")
        result = client.update_endpoint("ar-123456", "are-123456", input_obj)

        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(ENDPOINT_FROM_INNER_PATH)
    @patch(CONTROL_API_PATH)
    def test_update_endpoint_async(
        self, mock_control_api_class, mock_from_inner
    ):
        """测试异步更新端点"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.update_agent_runtime_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointData()
        )
        mock_control_api_class.return_value = mock_control_api
        mock_from_inner.return_value = MockEndpointInstance()

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeEndpointUpdateInput(description="Updated")
        result = asyncio.run(
            client.update_endpoint_async("ar-123456", "are-123456", input_obj)
        )
        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(ENDPOINT_FROM_INNER_PATH)
    @patch(CONTROL_API_PATH)
    def test_get_endpoint(self, mock_control_api_class, mock_from_inner):
        """测试获取端点"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.get_agent_runtime_endpoint.return_value = (
            MockAgentRuntimeEndpointData()
        )
        mock_control_api_class.return_value = mock_control_api
        mock_from_inner.return_value = MockEndpointInstance()

        client = AgentRuntimeClient()
        result = client.get_endpoint("ar-123456", "are-123456")

        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(ENDPOINT_FROM_INNER_PATH)
    @patch(CONTROL_API_PATH)
    def test_get_endpoint_async(self, mock_control_api_class, mock_from_inner):
        """测试异步获取端点"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.get_agent_runtime_endpoint_async = AsyncMock(
            return_value=MockAgentRuntimeEndpointData()
        )
        mock_control_api_class.return_value = mock_control_api
        mock_from_inner.return_value = MockEndpointInstance()

        client = AgentRuntimeClient()
        result = asyncio.run(
            client.get_endpoint_async("ar-123456", "are-123456")
        )
        assert result.agent_runtime_endpoint_id == "are-123456"

    @patch(ENDPOINT_FROM_INNER_PATH)
    @patch(CONTROL_API_PATH)
    def test_list_endpoints(self, mock_control_api_class, mock_from_inner):
        """测试列表端点"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.list_agent_runtime_endpoints.return_value = (
            MockListOutput([MockAgentRuntimeEndpointData()])
        )
        mock_control_api_class.return_value = mock_control_api
        mock_from_inner.return_value = MockEndpointInstance()

        client = AgentRuntimeClient()
        result = client.list_endpoints("ar-123456")

        assert len(result) == 1

    @patch(ENDPOINT_FROM_INNER_PATH)
    @patch(CONTROL_API_PATH)
    def test_list_endpoints_with_input(
        self, mock_control_api_class, mock_from_inner
    ):
        """测试带参数列表端点"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.list_agent_runtime_endpoints.return_value = (
            MockListOutput([MockAgentRuntimeEndpointData()])
        )
        mock_control_api_class.return_value = mock_control_api
        mock_from_inner.return_value = MockEndpointInstance()

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeEndpointListInput(endpoint_name="test")
        result = client.list_endpoints("ar-123456", input_obj)

        assert len(result) == 1

    @patch(ENDPOINT_FROM_INNER_PATH)
    @patch(CONTROL_API_PATH)
    def test_list_endpoints_async(
        self, mock_control_api_class, mock_from_inner
    ):
        """测试异步列表端点"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.list_agent_runtime_endpoints_async = AsyncMock(
            return_value=MockListOutput([MockAgentRuntimeEndpointData()])
        )
        mock_control_api_class.return_value = mock_control_api
        mock_from_inner.return_value = MockEndpointInstance()

        client = AgentRuntimeClient()
        result = asyncio.run(client.list_endpoints_async("ar-123456"))
        assert len(result) == 1


class MockVersionData:
    """模拟 AgentRuntimeVersion 数据"""

    agent_runtime_version = "1"
    agent_runtime_id = "ar-123456"
    agent_runtime_name = "test-runtime"

    def to_map(self):
        return {
            "agentRuntimeVersion": self.agent_runtime_version,
            "agentRuntimeId": self.agent_runtime_id,
            "agentRuntimeName": self.agent_runtime_name,
        }


class TestAgentRuntimeClientVersions:
    """AgentRuntimeClient 版本方法测试"""

    @patch(CONTROL_API_PATH)
    def test_list_versions(self, mock_control_api_class):
        """测试列表版本"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.list_agent_runtime_versions.return_value = (
            MockListOutput([MockVersionData(), MockVersionData()])
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        result = client.list_versions("ar-123456")

        assert len(result) == 2

    @patch(CONTROL_API_PATH)
    def test_list_versions_with_input(self, mock_control_api_class):
        """测试带参数列表版本"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.list_agent_runtime_versions.return_value = (
            MockListOutput([MockVersionData()])
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        input_obj = AgentRuntimeVersionListInput()
        result = client.list_versions("ar-123456", input_obj)

        assert len(result) == 1

    @patch(CONTROL_API_PATH)
    def test_list_versions_async(self, mock_control_api_class):
        """测试异步列表版本"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_control_api.list_agent_runtime_versions_async = AsyncMock(
            return_value=MockListOutput([MockVersionData()])
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        result = asyncio.run(client.list_versions_async("ar-123456"))
        assert len(result) == 1


class TestAgentRuntimeClientInvoke:
    """AgentRuntimeClient invoke 方法测试"""

    @patch("agentrun.agent_runtime.client.AgentRuntimeDataAPI")
    @patch(CONTROL_API_PATH)
    def test_invoke_openai(self, mock_control_api_class, mock_data_api_class):
        """测试 invoke_openai"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_data_api = MagicMock()
        mock_data_api.invoke_openai.return_value = {
            "choices": [{"message": {"content": "Hello"}}]
        }
        mock_data_api_class.return_value = mock_data_api

        client = AgentRuntimeClient()
        result = client.invoke_openai(
            agent_runtime_name="test-runtime",
            agent_runtime_endpoint_name="Default",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )

        assert result is not None

    @patch("agentrun.agent_runtime.client.AgentRuntimeDataAPI")
    @patch(CONTROL_API_PATH)
    def test_invoke_openai_async(
        self, mock_control_api_class, mock_data_api_class
    ):
        """测试 invoke_openai_async"""
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_data_api = MagicMock()
        mock_data_api.invoke_openai_async = AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello"}}]}
        )
        mock_data_api_class.return_value = mock_data_api

        client = AgentRuntimeClient()
        result = asyncio.run(
            client.invoke_openai_async(
                agent_runtime_name="test-runtime",
                agent_runtime_endpoint_name="Default",
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
            )
        )

        assert result is not None
