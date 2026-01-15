"""Tests for agentrun/model/model_proxy.py"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.model.model import (
    ModelProxyCreateInput,
    ModelProxyUpdateInput,
    ProxyConfig,
    ProxyConfigEndpoint,
    ProxyMode,
)
from agentrun.model.model_proxy import ModelProxy
from agentrun.utils.config import Config
from agentrun.utils.model import Status


class TestModelProxyCreate:
    """Tests for ModelProxy.create methods"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    def test_create(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_proxy = ModelProxy(model_proxy_name="test-proxy")
        mock_client.create.return_value = mock_proxy

        input_obj = ModelProxyCreateInput(
            model_proxy_name="test-proxy",
            proxy_mode=ProxyMode.SINGLE,
        )

        result = ModelProxy.create(input_obj)

        mock_client.create.assert_called_once()
        assert result == mock_proxy

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    @pytest.mark.asyncio
    async def test_create_async(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_proxy = ModelProxy(model_proxy_name="test-proxy")
        mock_client.create_async = AsyncMock(return_value=mock_proxy)

        input_obj = ModelProxyCreateInput(model_proxy_name="test-proxy")

        result = await ModelProxy.create_async(input_obj)

        mock_client.create_async.assert_called_once()
        assert result == mock_proxy


class TestModelProxyDelete:
    """Tests for ModelProxy.delete methods"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    def test_delete_by_name(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_proxy = ModelProxy(model_proxy_name="test-proxy")
        mock_client.delete.return_value = mock_proxy

        result = ModelProxy.delete_by_name("test-proxy")

        mock_client.delete.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    @pytest.mark.asyncio
    async def test_delete_by_name_async(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.delete_async = AsyncMock()

        await ModelProxy.delete_by_name_async("test-proxy")

        mock_client.delete_async.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    def test_delete_instance(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        proxy = ModelProxy(model_proxy_name="test-proxy")
        proxy.delete()

        mock_client.delete.assert_called_once()

    def test_delete_without_name_raises_error(self):
        proxy = ModelProxy()
        with pytest.raises(ValueError, match="model_Proxy_name is required"):
            proxy.delete()

    @pytest.mark.asyncio
    async def test_delete_async_without_name_raises_error(self):
        proxy = ModelProxy()
        with pytest.raises(ValueError, match="model_Proxy_name is required"):
            await proxy.delete_async()


class TestModelProxyUpdate:
    """Tests for ModelProxy.update methods"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    def test_update_by_name(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_proxy = ModelProxy(model_proxy_name="test-proxy")
        mock_client.update.return_value = mock_proxy

        input_obj = ModelProxyUpdateInput(description="Updated")

        result = ModelProxy.update_by_name("test-proxy", input_obj)

        mock_client.update.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    @pytest.mark.asyncio
    async def test_update_by_name_async(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_proxy = ModelProxy(model_proxy_name="test-proxy")
        mock_client.update_async = AsyncMock(return_value=mock_proxy)

        input_obj = ModelProxyUpdateInput(description="Updated")

        await ModelProxy.update_by_name_async("test-proxy", input_obj)

        mock_client.update_async.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    def test_update_instance(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        updated_proxy = ModelProxy(
            model_proxy_name="test-proxy", description="Updated"
        )
        mock_client.update.return_value = updated_proxy

        proxy = ModelProxy(model_proxy_name="test-proxy")
        input_obj = ModelProxyUpdateInput(description="Updated")

        result = proxy.update(input_obj)

        assert result.description == "Updated"

    def test_update_without_name_raises_error(self):
        proxy = ModelProxy()
        input_obj = ModelProxyUpdateInput(description="Test")
        with pytest.raises(ValueError, match="model_Proxy_name is required"):
            proxy.update(input_obj)

    @pytest.mark.asyncio
    async def test_update_async_without_name_raises_error(self):
        proxy = ModelProxy()
        input_obj = ModelProxyUpdateInput(description="Test")
        with pytest.raises(ValueError, match="model_Proxy_name is required"):
            await proxy.update_async(input_obj)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    @pytest.mark.asyncio
    async def test_update_async_instance(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        updated_proxy = ModelProxy(
            model_proxy_name="test-proxy", description="Updated"
        )
        mock_client.update_async = AsyncMock(return_value=updated_proxy)

        proxy = ModelProxy(model_proxy_name="test-proxy")
        input_obj = ModelProxyUpdateInput(description="Updated")

        result = await proxy.update_async(input_obj)

        assert result.description == "Updated"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    @pytest.mark.asyncio
    async def test_delete_async_instance(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.delete_async = AsyncMock()

        proxy = ModelProxy(model_proxy_name="test-proxy")
        await proxy.delete_async()

        mock_client.delete_async.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    @pytest.mark.asyncio
    async def test_get_async_instance(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_proxy = ModelProxy(
            model_proxy_name="test-proxy", status=Status.READY
        )
        mock_client.get_async = AsyncMock(return_value=mock_proxy)

        proxy = ModelProxy(model_proxy_name="test-proxy")
        result = await proxy.get_async()

        assert result.status == Status.READY


class TestModelProxyGet:
    """Tests for ModelProxy.get methods"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    def test_get_by_name(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_proxy = ModelProxy(model_proxy_name="test-proxy")
        mock_client.get.return_value = mock_proxy

        result = ModelProxy.get_by_name("test-proxy")

        mock_client.get.assert_called_once()
        assert result == mock_proxy

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    @pytest.mark.asyncio
    async def test_get_by_name_async(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_proxy = ModelProxy(model_proxy_name="test-proxy")
        mock_client.get_async = AsyncMock(return_value=mock_proxy)

        result = await ModelProxy.get_by_name_async("test-proxy")

        mock_client.get_async.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    def test_get_instance(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_proxy = ModelProxy(
            model_proxy_name="test-proxy", status=Status.READY
        )
        mock_client.get.return_value = mock_proxy

        proxy = ModelProxy(model_proxy_name="test-proxy")
        result = proxy.get()

        assert result.status == Status.READY

    def test_get_without_name_raises_error(self):
        proxy = ModelProxy()
        with pytest.raises(ValueError, match="model_Proxy_name is required"):
            proxy.get()

    @pytest.mark.asyncio
    async def test_get_async_without_name_raises_error(self):
        proxy = ModelProxy()
        with pytest.raises(ValueError, match="model_Proxy_name is required"):
            await proxy.get_async()


class TestModelProxyRefresh:
    """Tests for ModelProxy.refresh methods"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    def test_refresh(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_proxy = ModelProxy(
            model_proxy_name="test-proxy", status=Status.READY
        )
        mock_client.get.return_value = mock_proxy

        proxy = ModelProxy(model_proxy_name="test-proxy")
        result = proxy.refresh()

        mock_client.get.assert_called()
        assert result.status == Status.READY

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    @pytest.mark.asyncio
    async def test_refresh_async(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_proxy = ModelProxy(
            model_proxy_name="test-proxy", status=Status.READY
        )
        mock_client.get_async = AsyncMock(return_value=mock_proxy)

        proxy = ModelProxy(model_proxy_name="test-proxy")
        result = await proxy.refresh_async()

        mock_client.get_async.assert_called()


class TestModelProxyList:
    """Tests for ModelProxy.list methods"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    def test_list_all(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_proxies = [
            ModelProxy(model_proxy_name="proxy1", model_proxy_id="id1"),
            ModelProxy(model_proxy_name="proxy2", model_proxy_id="id2"),
        ]
        mock_client.list.return_value = mock_proxies

        result = ModelProxy.list_all()

        mock_client.list.assert_called()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    @pytest.mark.asyncio
    async def test_list_all_async(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_proxies = [
            ModelProxy(model_proxy_name="proxy1", model_proxy_id="id1"),
        ]
        mock_client.list_async = AsyncMock(return_value=mock_proxies)

        result = await ModelProxy.list_all_async()

        mock_client.list_async.assert_called()


class TestModelProxyModelInfo:
    """Tests for ModelProxy.model_info method"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.api.data.ModelDataAPI")
    def test_model_info_single_mode(self, mock_data_api_class):
        mock_data_api = MagicMock()
        mock_data_api_class.return_value = mock_data_api

        from agentrun.model.api.data import BaseInfo

        mock_info = BaseInfo(model="gpt-4", base_url="https://api.example.com")
        mock_data_api.model_info.return_value = mock_info

        proxy = ModelProxy(
            model_proxy_name="test-proxy",
            proxy_mode=ProxyMode.SINGLE,
            proxy_config=ProxyConfig(
                endpoints=[ProxyConfigEndpoint(model_names=["gpt-4"])]
            ),
        )

        result = proxy.model_info()

        assert result.model == "gpt-4"


class TestModelProxyCompletions:
    """Tests for ModelProxy.completions method"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("litellm.completion")
    def test_completions(self, mock_completion):
        from agentrun.model.api.data import BaseInfo

        mock_completion.return_value = {"choices": []}

        proxy = ModelProxy(model_proxy_name="test-proxy")

        # Create a mock _data_client to provide model_info
        mock_data_client = MagicMock()
        mock_info = BaseInfo(model="gpt-4", base_url="https://api.example.com")
        mock_data_client.model_info.return_value = mock_info

        # Set _data_client so model_info() returns our mock info
        proxy._data_client = mock_data_client

        proxy.completions(messages=[{"role": "user", "content": "Hello"}])

        mock_completion.assert_called_once()


class TestModelProxyResponses:
    """Tests for ModelProxy.responses method"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("litellm.responses")
    def test_responses(self, mock_responses):
        from agentrun.model.api.data import BaseInfo

        mock_responses.return_value = {}

        proxy = ModelProxy(model_proxy_name="test-proxy")

        # Create a mock _data_client to provide model_info
        mock_data_client = MagicMock()
        mock_info = BaseInfo(model="gpt-4", base_url="https://api.example.com")
        mock_data_client.model_info.return_value = mock_info

        # Set _data_client so model_info() returns our mock info
        proxy._data_client = mock_data_client

        # Note: The responses method expects 'input' parameter (not 'messages')
        # based on the ModelAPI.responses signature
        proxy.responses(input="Hello")

        mock_responses.assert_called_once()
