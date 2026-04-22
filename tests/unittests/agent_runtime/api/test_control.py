"""
测试 AgentRuntimeControlAPI 对底层 SDK client 的调用是否与 SDK 方法签名匹配。
"""

from unittest.mock import create_autospec, MagicMock, patch

from alibabacloud_agentrun20250910.client import Client as AgentRunClient
from alibabacloud_agentrun20250910.models import (
    CreateAgentRuntimeEndpointInput,
    CreateAgentRuntimeInput,
    GetAgentRuntimeRequest,
    ListAgentRuntimeEndpointsRequest,
    ListAgentRuntimesRequest,
    ListAgentRuntimeVersionsRequest,
    UpdateAgentRuntimeEndpointInput,
    UpdateAgentRuntimeInput,
)
import pytest

from agentrun.agent_runtime.api.control import AgentRuntimeControlAPI
from agentrun.utils.config import Config


@pytest.fixture
def mock_config():
    return Config(
        access_key_id="test-ak",
        access_key_secret="test-sk",
        region_id="cn-hangzhou",
        control_endpoint="https://agentrun.cn-hangzhou.aliyuncs.com",
    )


@pytest.fixture
def mock_response():
    response = MagicMock()
    response.body.request_id = "test-request-id"
    response.body.data = MagicMock()
    return response


@pytest.fixture
def api_and_client(mock_config, mock_response):
    api = AgentRuntimeControlAPI(config=mock_config)
    mock_client = create_autospec(AgentRunClient, instance=True)

    for attr in dir(AgentRunClient):
        if "with_options" in attr:
            getattr(mock_client, attr).return_value = mock_response

    with patch.object(api, "_get_client", return_value=mock_client):
        yield api, mock_client


class TestAgentRuntimeControlAPISignatures:

    def test_create_agent_runtime(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateAgentRuntimeInput)
        api.create_agent_runtime(input_data)
        client.create_agent_runtime_with_options.assert_called_once()

    def test_delete_agent_runtime(self, api_and_client):
        api, client = api_and_client
        api.delete_agent_runtime("agent-123")
        client.delete_agent_runtime_with_options.assert_called_once()

    def test_update_agent_runtime(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateAgentRuntimeInput)
        api.update_agent_runtime("agent-123", input_data)
        client.update_agent_runtime_with_options.assert_called_once()

    def test_get_agent_runtime(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=GetAgentRuntimeRequest)
        api.get_agent_runtime("agent-123", input_data)
        client.get_agent_runtime_with_options.assert_called_once()

    def test_list_agent_runtimes(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListAgentRuntimesRequest)
        api.list_agent_runtimes(input_data)
        client.list_agent_runtimes_with_options.assert_called_once()

    def test_create_agent_runtime_endpoint(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateAgentRuntimeEndpointInput)
        api.create_agent_runtime_endpoint("agent-123", input_data)
        client.create_agent_runtime_endpoint_with_options.assert_called_once()

    def test_delete_agent_runtime_endpoint(self, api_and_client):
        api, client = api_and_client
        api.delete_agent_runtime_endpoint("agent-123", "endpoint-456")
        client.delete_agent_runtime_endpoint_with_options.assert_called_once()

    def test_update_agent_runtime_endpoint(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateAgentRuntimeEndpointInput)
        api.update_agent_runtime_endpoint(
            "agent-123", "endpoint-456", input_data
        )
        client.update_agent_runtime_endpoint_with_options.assert_called_once()

    def test_get_agent_runtime_endpoint(self, api_and_client):
        api, client = api_and_client
        api.get_agent_runtime_endpoint("agent-123", "endpoint-456")
        client.get_agent_runtime_endpoint_with_options.assert_called_once()

    def test_list_agent_runtime_endpoints(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListAgentRuntimeEndpointsRequest)
        api.list_agent_runtime_endpoints("agent-123", input_data)
        client.list_agent_runtime_endpoints_with_options.assert_called_once()

    def test_list_agent_runtime_versions(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListAgentRuntimeVersionsRequest)
        api.list_agent_runtime_versions("agent-123", input_data)
        client.list_agent_runtime_versions_with_options.assert_called_once()


class TestAgentRuntimeControlAPIAsyncSignatures:

    @pytest.mark.asyncio
    async def test_create_agent_runtime_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateAgentRuntimeInput)
        await api.create_agent_runtime_async(input_data)
        client.create_agent_runtime_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_agent_runtime_async(self, api_and_client):
        api, client = api_and_client
        await api.delete_agent_runtime_async("agent-123")
        client.delete_agent_runtime_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_agent_runtime_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateAgentRuntimeInput)
        await api.update_agent_runtime_async("agent-123", input_data)
        client.update_agent_runtime_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_agent_runtime_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=GetAgentRuntimeRequest)
        await api.get_agent_runtime_async("agent-123", input_data)
        client.get_agent_runtime_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_agent_runtimes_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListAgentRuntimesRequest)
        await api.list_agent_runtimes_async(input_data)
        client.list_agent_runtimes_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_agent_runtime_endpoint_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateAgentRuntimeEndpointInput)
        await api.create_agent_runtime_endpoint_async("agent-123", input_data)
        client.create_agent_runtime_endpoint_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_agent_runtime_endpoint_async(self, api_and_client):
        api, client = api_and_client
        await api.delete_agent_runtime_endpoint_async(
            "agent-123", "endpoint-456"
        )
        client.delete_agent_runtime_endpoint_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_agent_runtime_endpoint_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateAgentRuntimeEndpointInput)
        await api.update_agent_runtime_endpoint_async(
            "agent-123", "endpoint-456", input_data
        )
        client.update_agent_runtime_endpoint_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_agent_runtime_endpoint_async(self, api_and_client):
        api, client = api_and_client
        await api.get_agent_runtime_endpoint_async("agent-123", "endpoint-456")
        client.get_agent_runtime_endpoint_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_agent_runtime_endpoints_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListAgentRuntimeEndpointsRequest)
        await api.list_agent_runtime_endpoints_async("agent-123", input_data)
        client.list_agent_runtime_endpoints_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_agent_runtime_versions_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListAgentRuntimeVersionsRequest)
        await api.list_agent_runtime_versions_async("agent-123", input_data)
        client.list_agent_runtime_versions_with_options_async.assert_called_once()
