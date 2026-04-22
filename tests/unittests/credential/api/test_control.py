"""
测试 CredentialControlAPI 对底层 SDK client 的调用是否与 SDK 方法签名匹配。
"""

from unittest.mock import create_autospec, MagicMock, patch

from alibabacloud_agentrun20250910.client import Client as AgentRunClient
from alibabacloud_agentrun20250910.models import (
    CreateCredentialInput,
    ListCredentialsRequest,
    UpdateCredentialInput,
)
import pytest

from agentrun.credential.api.control import CredentialControlAPI
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
    api = CredentialControlAPI(config=mock_config)
    mock_client = create_autospec(AgentRunClient, instance=True)

    for attr in dir(AgentRunClient):
        if "with_options" in attr:
            getattr(mock_client, attr).return_value = mock_response

    with patch.object(api, "_get_client", return_value=mock_client):
        yield api, mock_client


class TestCredentialControlAPISignatures:

    def test_create_credential(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateCredentialInput)
        api.create_credential(input_data)
        client.create_credential_with_options.assert_called_once()

    def test_delete_credential(self, api_and_client):
        api, client = api_and_client
        api.delete_credential("test-cred")
        client.delete_credential_with_options.assert_called_once()

    def test_update_credential(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateCredentialInput)
        api.update_credential("test-cred", input_data)
        client.update_credential_with_options.assert_called_once()

    def test_get_credential(self, api_and_client):
        api, client = api_and_client
        api.get_credential("test-cred")
        client.get_credential_with_options.assert_called_once()

    def test_list_credentials(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListCredentialsRequest)
        api.list_credentials(input_data)
        client.list_credentials_with_options.assert_called_once()


class TestCredentialControlAPIAsyncSignatures:

    @pytest.mark.asyncio
    async def test_create_credential_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateCredentialInput)
        await api.create_credential_async(input_data)
        client.create_credential_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_credential_async(self, api_and_client):
        api, client = api_and_client
        await api.delete_credential_async("test-cred")
        client.delete_credential_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_credential_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateCredentialInput)
        await api.update_credential_async("test-cred", input_data)
        client.update_credential_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_credential_async(self, api_and_client):
        api, client = api_and_client
        await api.get_credential_async("test-cred")
        client.get_credential_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_credentials_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListCredentialsRequest)
        await api.list_credentials_async(input_data)
        client.list_credentials_with_options_async.assert_called_once()
