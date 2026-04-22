"""
测试 ModelControlAPI 对底层 SDK client 的调用是否与 SDK 方法签名匹配。
"""

from unittest.mock import create_autospec, MagicMock, patch

from alibabacloud_agentrun20250910.client import Client as AgentRunClient
from alibabacloud_agentrun20250910.models import (
    CreateModelProxyInput,
    CreateModelServiceInput,
    ListModelProxiesRequest,
    ListModelServicesRequest,
    UpdateModelProxyInput,
    UpdateModelServiceInput,
)
import pytest

from agentrun.model.api.control import ModelControlAPI
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
    api = ModelControlAPI(config=mock_config)
    mock_client = create_autospec(AgentRunClient, instance=True)

    for attr in dir(AgentRunClient):
        if "with_options" in attr:
            getattr(mock_client, attr).return_value = mock_response

    with patch.object(api, "_get_client", return_value=mock_client):
        yield api, mock_client


class TestModelServiceControlAPISignatures:

    def test_create_model_service(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateModelServiceInput)
        api.create_model_service(input_data)
        client.create_model_service_with_options.assert_called_once()

    def test_delete_model_service(self, api_and_client):
        api, client = api_and_client
        api.delete_model_service("test-service")
        client.delete_model_service_with_options.assert_called_once()

    def test_update_model_service(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateModelServiceInput)
        api.update_model_service("test-service", input_data)
        client.update_model_service_with_options.assert_called_once()

    def test_get_model_service(self, api_and_client):
        api, client = api_and_client
        api.get_model_service("test-service")
        client.get_model_service_with_options.assert_called_once()

    def test_list_model_services(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListModelServicesRequest)
        api.list_model_services(input_data)
        client.list_model_services_with_options.assert_called_once()


class TestModelProxyControlAPISignatures:

    def test_create_model_proxy(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateModelProxyInput)
        api.create_model_proxy(input_data)
        client.create_model_proxy_with_options.assert_called_once()

    def test_delete_model_proxy(self, api_and_client):
        api, client = api_and_client
        api.delete_model_proxy("test-proxy")
        client.delete_model_proxy_with_options.assert_called_once()

    def test_update_model_proxy(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateModelProxyInput)
        api.update_model_proxy("test-proxy", input_data)
        client.update_model_proxy_with_options.assert_called_once()

    def test_get_model_proxy(self, api_and_client):
        api, client = api_and_client
        api.get_model_proxy("test-proxy")
        client.get_model_proxy_with_options.assert_called_once()

    def test_list_model_proxies(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListModelProxiesRequest)
        api.list_model_proxies(input_data)
        client.list_model_proxies_with_options.assert_called_once()


class TestModelControlAPIAsyncSignatures:

    @pytest.mark.asyncio
    async def test_create_model_service_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateModelServiceInput)
        await api.create_model_service_async(input_data)
        client.create_model_service_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_model_service_async(self, api_and_client):
        api, client = api_and_client
        await api.delete_model_service_async("test-service")
        client.delete_model_service_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_model_service_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateModelServiceInput)
        await api.update_model_service_async("test-service", input_data)
        client.update_model_service_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_model_service_async(self, api_and_client):
        api, client = api_and_client
        await api.get_model_service_async("test-service")
        client.get_model_service_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_model_services_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListModelServicesRequest)
        await api.list_model_services_async(input_data)
        client.list_model_services_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_model_proxy_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateModelProxyInput)
        await api.create_model_proxy_async(input_data)
        client.create_model_proxy_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_model_proxy_async(self, api_and_client):
        api, client = api_and_client
        await api.delete_model_proxy_async("test-proxy")
        client.delete_model_proxy_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_model_proxy_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateModelProxyInput)
        await api.update_model_proxy_async("test-proxy", input_data)
        client.update_model_proxy_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_model_proxy_async(self, api_and_client):
        api, client = api_and_client
        await api.get_model_proxy_async("test-proxy")
        client.get_model_proxy_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_model_proxies_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListModelProxiesRequest)
        await api.list_model_proxies_async(input_data)
        client.list_model_proxies_with_options_async.assert_called_once()
