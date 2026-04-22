"""
测试 KnowledgeBaseControlAPI 对底层 SDK client 的调用是否与 SDK 方法签名匹配。
"""

from unittest.mock import create_autospec, MagicMock, patch

from alibabacloud_agentrun20250910.client import Client as AgentRunClient
from alibabacloud_agentrun20250910.models import (
    CreateKnowledgeBaseInput,
    ListKnowledgeBasesRequest,
    UpdateKnowledgeBaseInput,
)
import pytest

from agentrun.knowledgebase.api.control import KnowledgeBaseControlAPI
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
    api = KnowledgeBaseControlAPI(config=mock_config)
    mock_client = create_autospec(AgentRunClient, instance=True)

    for attr in dir(AgentRunClient):
        if "with_options" in attr:
            getattr(mock_client, attr).return_value = mock_response

    with patch.object(api, "_get_client", return_value=mock_client):
        yield api, mock_client


class TestKnowledgeBaseControlAPISignatures:

    def test_create_knowledge_base(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateKnowledgeBaseInput)
        api.create_knowledge_base(input_data)
        client.create_knowledge_base_with_options.assert_called_once()

    def test_delete_knowledge_base(self, api_and_client):
        api, client = api_and_client
        api.delete_knowledge_base("test-kb")
        client.delete_knowledge_base_with_options.assert_called_once()

    def test_update_knowledge_base(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateKnowledgeBaseInput)
        api.update_knowledge_base("test-kb", input_data)
        client.update_knowledge_base_with_options.assert_called_once()

    def test_get_knowledge_base(self, api_and_client):
        api, client = api_and_client
        api.get_knowledge_base("test-kb")
        client.get_knowledge_base_with_options.assert_called_once()

    def test_list_knowledge_bases(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListKnowledgeBasesRequest)
        api.list_knowledge_bases(input_data)
        client.list_knowledge_bases_with_options.assert_called_once()


class TestKnowledgeBaseControlAPIAsyncSignatures:

    @pytest.mark.asyncio
    async def test_create_knowledge_base_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateKnowledgeBaseInput)
        await api.create_knowledge_base_async(input_data)
        client.create_knowledge_base_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_knowledge_base_async(self, api_and_client):
        api, client = api_and_client
        await api.delete_knowledge_base_async("test-kb")
        client.delete_knowledge_base_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_knowledge_base_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateKnowledgeBaseInput)
        await api.update_knowledge_base_async("test-kb", input_data)
        client.update_knowledge_base_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_knowledge_base_async(self, api_and_client):
        api, client = api_and_client
        await api.get_knowledge_base_async("test-kb")
        client.get_knowledge_base_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_knowledge_bases_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListKnowledgeBasesRequest)
        await api.list_knowledge_bases_async(input_data)
        client.list_knowledge_bases_with_options_async.assert_called_once()
