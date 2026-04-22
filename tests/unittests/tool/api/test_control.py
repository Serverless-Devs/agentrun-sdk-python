"""
测试 ToolControlAPI (tool) 对底层 SDK client 的调用是否与 SDK 方法签名匹配。
"""

from unittest.mock import create_autospec, MagicMock, patch

from alibabacloud_agentrun20250910.client import Client as AgentRunClient
import pytest

from agentrun.tool.api.control import ToolControlAPI
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
    api = ToolControlAPI(config=mock_config)
    mock_client = create_autospec(AgentRunClient, instance=True)

    for attr in dir(AgentRunClient):
        if "with_options" in attr:
            getattr(mock_client, attr).return_value = mock_response

    with patch.object(api, "_get_client", return_value=mock_client):
        yield api, mock_client


class TestToolControlAPISignatures:

    def test_get_tool(self, api_and_client):
        api, client = api_and_client
        api.get_tool("test-tool")
        client.get_tool_with_options.assert_called_once()


class TestToolControlAPIAsyncSignatures:

    @pytest.mark.asyncio
    async def test_get_tool_async(self, api_and_client):
        api, client = api_and_client
        await api.get_tool_async("test-tool")
        client.get_tool_with_options_async.assert_called_once()
