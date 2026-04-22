"""
测试 ToolControlAPI (toolset) 对底层 SDK client 的调用是否与 SDK 方法签名匹配。

注意：toolset 使用 DevsClient 而非 AgentRunClient。
"""

from unittest.mock import create_autospec, MagicMock, patch

from alibabacloud_devs20230714.client import Client as DevsClient
from alibabacloud_devs20230714.models import ListToolsetsRequest
import pytest

from agentrun.toolset.api.control import ToolControlAPI
from agentrun.utils.config import Config


@pytest.fixture
def mock_config():
    return Config(
        access_key_id="test-ak",
        access_key_secret="test-sk",
        region_id="cn-hangzhou",
        devs_endpoint="https://devs.cn-hangzhou.aliyuncs.com",
    )


@pytest.fixture
def mock_response():
    response = MagicMock()
    response.body = MagicMock()
    return response


@pytest.fixture
def api_and_client(mock_config, mock_response):
    api = ToolControlAPI(config=mock_config)
    mock_client = create_autospec(DevsClient, instance=True)

    for attr in dir(DevsClient):
        if "with_options" in attr:
            getattr(mock_client, attr).return_value = mock_response

    with patch.object(api, "_get_devs_client", return_value=mock_client):
        yield api, mock_client


class TestToolsetControlAPISignatures:

    def test_get_toolset(self, api_and_client):
        api, client = api_and_client
        api.get_toolset("test-toolset")
        client.get_toolset_with_options.assert_called_once()

    def test_list_toolsets(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListToolsetsRequest)
        api.list_toolsets(input_data)
        client.list_toolsets_with_options.assert_called_once()


class TestToolsetControlAPIAsyncSignatures:

    @pytest.mark.asyncio
    async def test_get_toolset_async(self, api_and_client):
        api, client = api_and_client
        await api.get_toolset_async("test-toolset")
        client.get_toolset_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_toolsets_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListToolsetsRequest)
        await api.list_toolsets_async(input_data)
        client.list_toolsets_with_options_async.assert_called_once()
