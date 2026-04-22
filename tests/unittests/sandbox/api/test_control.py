"""
测试 SandboxControlAPI 对底层 SDK client 的调用是否与 SDK 方法签名匹配。

使用 create_autospec(AgentRunClient) 来 mock _get_client 的返回值，
autospec 会强制校验方法签名，确保 SDK 方法签名变更时能及时检测出来。
"""

from unittest.mock import create_autospec, MagicMock, patch

from alibabacloud_agentrun20250910.client import Client as AgentRunClient
from alibabacloud_agentrun20250910.models import (
    CreateSandboxInput,
    CreateTemplateInput,
    ListSandboxesRequest,
    ListTemplatesRequest,
    UpdateTemplateInput,
)
import pytest

from agentrun.sandbox.api.control import SandboxControlAPI
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
    """构造通用的 SDK 响应对象"""
    response = MagicMock()
    response.body.request_id = "test-request-id"
    response.body.data = MagicMock()
    return response


@pytest.fixture
def api_and_client(mock_config, mock_response):
    """返回 (api, mock_client)，mock_client 使用 create_autospec 强制校验方法签名"""
    api = SandboxControlAPI(config=mock_config)
    mock_client = create_autospec(AgentRunClient, instance=True)

    # 为所有 _with_options 方法设置默认返回值
    for attr in dir(AgentRunClient):
        if "with_options" in attr:
            getattr(mock_client, attr).return_value = mock_response

    with patch.object(api, "_get_client", return_value=mock_client):
        yield api, mock_client


class TestSandboxControlAPISignatures:
    """测试 SandboxControlAPI 同步方法调用是否匹配 SDK 签名"""

    def test_create_template(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateTemplateInput)
        api.create_template(input_data)
        client.create_template_with_options.assert_called_once()

    def test_delete_template(self, api_and_client):
        api, client = api_and_client
        api.delete_template("test-template")
        client.delete_template_with_options.assert_called_once()

    def test_update_template(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateTemplateInput)
        api.update_template("test-template", input_data)
        client.update_template_with_options.assert_called_once()

    def test_get_template(self, api_and_client):
        api, client = api_and_client
        api.get_template("test-template")
        client.get_template_with_options.assert_called_once()

    def test_list_templates(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListTemplatesRequest)
        api.list_templates(input_data)
        client.list_templates_with_options.assert_called_once()

    def test_create_sandbox(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateSandboxInput)
        api.create_sandbox(input_data)
        client.create_sandbox_with_options.assert_called_once()

    def test_stop_sandbox(self, api_and_client):
        api, client = api_and_client
        api.stop_sandbox("sandbox-123")
        client.stop_sandbox_with_options.assert_called_once()

    def test_get_sandbox(self, api_and_client):
        api, client = api_and_client
        api.get_sandbox("sandbox-123")
        client.get_sandbox_with_options.assert_called_once()

    def test_list_sandboxes(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListSandboxesRequest)
        api.list_sandboxes(input_data)
        client.list_sandboxes_with_options.assert_called_once()


class TestSandboxControlAPIAsyncSignatures:
    """测试 SandboxControlAPI 异步方法调用是否匹配 SDK 签名"""

    @pytest.mark.asyncio
    async def test_create_template_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateTemplateInput)
        await api.create_template_async(input_data)
        client.create_template_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_template_async(self, api_and_client):
        api, client = api_and_client
        await api.delete_template_async("test-template")
        client.delete_template_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_template_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateTemplateInput)
        await api.update_template_async("test-template", input_data)
        client.update_template_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_template_async(self, api_and_client):
        api, client = api_and_client
        await api.get_template_async("test-template")
        client.get_template_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_templates_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListTemplatesRequest)
        await api.list_templates_async(input_data)
        client.list_templates_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_sandbox_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateSandboxInput)
        await api.create_sandbox_async(input_data)
        client.create_sandbox_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_sandbox_async(self, api_and_client):
        api, client = api_and_client
        await api.stop_sandbox_async("sandbox-123")
        client.stop_sandbox_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_sandbox_async(self, api_and_client):
        api, client = api_and_client
        await api.get_sandbox_async("sandbox-123")
        client.get_sandbox_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_sandboxes_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListSandboxesRequest)
        await api.list_sandboxes_async(input_data)
        client.list_sandboxes_with_options_async.assert_called_once()
