"""
测试 MemoryCollectionControlAPI 对底层 SDK client 的调用是否与 SDK 方法签名匹配。
"""

from unittest.mock import create_autospec, MagicMock, patch

from alibabacloud_agentrun20250910.client import Client as AgentRunClient
from alibabacloud_agentrun20250910.models import (
    CreateMemoryCollectionInput,
    ListMemoryCollectionsRequest,
    UpdateMemoryCollectionInput,
)
import pytest

from agentrun.memory_collection.api.control import MemoryCollectionControlAPI
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
    api = MemoryCollectionControlAPI(config=mock_config)
    mock_client = create_autospec(AgentRunClient, instance=True)

    for attr in dir(AgentRunClient):
        if "with_options" in attr:
            getattr(mock_client, attr).return_value = mock_response

    with patch.object(api, "_get_client", return_value=mock_client):
        yield api, mock_client


class TestMemoryCollectionControlAPISignatures:

    def test_create_memory_collection(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateMemoryCollectionInput)
        api.create_memory_collection(input_data)
        client.create_memory_collection_with_options.assert_called_once()

    def test_delete_memory_collection(self, api_and_client):
        api, client = api_and_client
        api.delete_memory_collection("test-collection")
        client.delete_memory_collection_with_options.assert_called_once()

    def test_update_memory_collection(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateMemoryCollectionInput)
        api.update_memory_collection("test-collection", input_data)
        client.update_memory_collection_with_options.assert_called_once()

    def test_get_memory_collection(self, api_and_client):
        api, client = api_and_client
        api.get_memory_collection("test-collection")
        client.get_memory_collection_with_options.assert_called_once()

    def test_list_memory_collections(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListMemoryCollectionsRequest)
        api.list_memory_collections(input_data)
        client.list_memory_collections_with_options.assert_called_once()


class TestMemoryCollectionControlAPIAsyncSignatures:

    @pytest.mark.asyncio
    async def test_create_memory_collection_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=CreateMemoryCollectionInput)
        await api.create_memory_collection_async(input_data)
        client.create_memory_collection_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_memory_collection_async(self, api_and_client):
        api, client = api_and_client
        await api.delete_memory_collection_async("test-collection")
        client.delete_memory_collection_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_memory_collection_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=UpdateMemoryCollectionInput)
        await api.update_memory_collection_async("test-collection", input_data)
        client.update_memory_collection_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_memory_collection_async(self, api_and_client):
        api, client = api_and_client
        await api.get_memory_collection_async("test-collection")
        client.get_memory_collection_with_options_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_memory_collections_async(self, api_and_client):
        api, client = api_and_client
        input_data = MagicMock(spec=ListMemoryCollectionsRequest)
        await api.list_memory_collections_async(input_data)
        client.list_memory_collections_with_options_async.assert_called_once()
