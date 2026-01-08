"""Tests for agentrun/model/client.py"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.model.client import ModelClient
from agentrun.model.model import (
    BackendType,
    ModelProxyCreateInput,
    ModelProxyListInput,
    ModelProxyUpdateInput,
    ModelServiceCreateInput,
    ModelServiceListInput,
    ModelServiceUpdateInput,
    ProxyConfig,
    ProxyConfigEndpoint,
    ProxyMode,
)
from agentrun.model.model_proxy import ModelProxy
from agentrun.model.model_service import ModelService
from agentrun.utils.config import Config
from agentrun.utils.exception import HTTPError, ResourceNotExistError


class TestModelClientInit:
    """Tests for ModelClient initialization"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_init_without_config(self, mock_control_api_class):
        client = ModelClient()
        mock_control_api_class.assert_called_once_with(None)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_init_with_config(self, mock_control_api_class):
        config = Config(
            access_key_id="custom-key",
            access_key_secret="custom-secret",
            account_id="custom-account",
        )
        client = ModelClient(config=config)
        mock_control_api_class.assert_called_once_with(config)


class TestModelClientCreate:
    """Tests for ModelClient.create methods"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_create_model_service(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_result.model_service_name = "test-service"
        mock_control_api.create_model_service.return_value = mock_result

        client = ModelClient()
        input_obj = ModelServiceCreateInput(
            model_service_name="test-service",
            provider="openai",
        )

        result = client.create(input_obj)

        mock_control_api.create_model_service.assert_called_once()
        assert isinstance(result, ModelService)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_create_model_proxy(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_result.model_proxy_name = "test-proxy"
        mock_control_api.create_model_proxy.return_value = mock_result

        client = ModelClient()
        input_obj = ModelProxyCreateInput(
            model_proxy_name="test-proxy",
            proxy_mode=ProxyMode.SINGLE,
        )

        result = client.create(input_obj)

        mock_control_api.create_model_proxy.assert_called_once()
        assert isinstance(result, ModelProxy)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_create_model_proxy_auto_mode_single(self, mock_control_api_class):
        """Test auto-detection of SINGLE proxy mode"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.create_model_proxy.return_value = mock_result

        client = ModelClient()
        input_obj = ModelProxyCreateInput(
            model_proxy_name="test-proxy",
            proxy_config=ProxyConfig(
                endpoints=[ProxyConfigEndpoint(model_names=["gpt-4"])]
            ),
        )

        client.create(input_obj)

        # Should auto-detect SINGLE mode when only 1 endpoint
        assert input_obj.proxy_mode == ProxyMode.SINGLE

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_create_model_proxy_auto_mode_multi(self, mock_control_api_class):
        """Test auto-detection of MULTI proxy mode"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.create_model_proxy.return_value = mock_result

        client = ModelClient()
        input_obj = ModelProxyCreateInput(
            model_proxy_name="test-proxy",
            proxy_config=ProxyConfig(
                endpoints=[
                    ProxyConfigEndpoint(model_names=["gpt-4"]),
                    ProxyConfigEndpoint(model_names=["gpt-3.5"]),
                ]
            ),
        )

        client.create(input_obj)

        # Should auto-detect MULTI mode when multiple endpoints
        assert input_obj.proxy_mode == ProxyMode.MULTI

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_create_raises_http_error(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_control_api.create_model_service.side_effect = HTTPError(
            status_code=400,
            message="Bad Request",
        )

        client = ModelClient()
        input_obj = ModelServiceCreateInput(model_service_name="test-service")

        with pytest.raises(Exception):
            client.create(input_obj)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_create_async_model_service(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_result.model_service_name = "test-service"
        mock_control_api.create_model_service_async = AsyncMock(
            return_value=mock_result
        )

        client = ModelClient()
        input_obj = ModelServiceCreateInput(
            model_service_name="test-service",
            provider="openai",
        )

        result = await client.create_async(input_obj)

        mock_control_api.create_model_service_async.assert_called_once()
        assert isinstance(result, ModelService)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_create_async_model_proxy(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_result.model_proxy_name = "test-proxy"
        mock_control_api.create_model_proxy_async = AsyncMock(
            return_value=mock_result
        )

        client = ModelClient()
        input_obj = ModelProxyCreateInput(
            model_proxy_name="test-proxy",
            proxy_mode=ProxyMode.SINGLE,
        )

        result = await client.create_async(input_obj)

        mock_control_api.create_model_proxy_async.assert_called_once()
        assert isinstance(result, ModelProxy)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_create_async_raises_http_error_service(
        self, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_control_api.create_model_service_async = AsyncMock(
            side_effect=HTTPError(status_code=400, message="Bad Request")
        )

        client = ModelClient()
        input_obj = ModelServiceCreateInput(model_service_name="test-service")

        with pytest.raises(Exception):
            await client.create_async(input_obj)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_create_async_raises_http_error_proxy(
        self, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_control_api.create_model_proxy_async = AsyncMock(
            side_effect=HTTPError(status_code=400, message="Bad Request")
        )

        client = ModelClient()
        input_obj = ModelProxyCreateInput(model_proxy_name="test-proxy")

        with pytest.raises(Exception):
            await client.create_async(input_obj)


class TestModelClientDelete:
    """Tests for ModelClient.delete methods"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_delete_proxy(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.delete_model_proxy.return_value = mock_result

        client = ModelClient()
        result = client.delete("test-proxy", backend_type=BackendType.PROXY)

        mock_control_api.delete_model_proxy.assert_called_once()
        assert isinstance(result, ModelProxy)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_delete_service(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.delete_model_service.return_value = mock_result

        client = ModelClient()
        result = client.delete("test-service", backend_type=BackendType.SERVICE)

        mock_control_api.delete_model_service.assert_called_once()
        assert isinstance(result, ModelService)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_delete_auto_detect_proxy(self, mock_control_api_class):
        """Test auto-detection of proxy type when backend_type is None"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.delete_model_proxy.return_value = mock_result

        client = ModelClient()
        result = client.delete("test")

        # Should try proxy first
        mock_control_api.delete_model_proxy.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_delete_auto_detect_falls_back_to_service(
        self, mock_control_api_class
    ):
        """Test fallback to service when proxy not found"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        # Proxy delete fails
        mock_control_api.delete_model_proxy.side_effect = HTTPError(
            status_code=404, message="Not found"
        )
        # Service delete succeeds
        mock_result = MagicMock()
        mock_control_api.delete_model_service.return_value = mock_result

        client = ModelClient()
        result = client.delete("test")

        mock_control_api.delete_model_proxy.assert_called_once()
        mock_control_api.delete_model_service.assert_called_once()
        assert isinstance(result, ModelService)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_delete_async(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.delete_model_proxy_async = AsyncMock(
            return_value=mock_result
        )

        client = ModelClient()
        result = await client.delete_async(
            "test-proxy", backend_type=BackendType.PROXY
        )

        mock_control_api.delete_model_proxy_async.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_delete_async_service(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.delete_model_service_async = AsyncMock(
            return_value=mock_result
        )

        client = ModelClient()
        result = await client.delete_async(
            "test-service", backend_type=BackendType.SERVICE
        )

        mock_control_api.delete_model_service_async.assert_called_once()
        assert isinstance(result, ModelService)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_delete_async_auto_detect_fallback(
        self, mock_control_api_class
    ):
        """Test fallback to service when proxy not found in async delete"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        # Proxy delete fails
        mock_control_api.delete_model_proxy_async = AsyncMock(
            side_effect=HTTPError(status_code=404, message="Not found")
        )
        # Service delete succeeds
        mock_result = MagicMock()
        mock_control_api.delete_model_service_async = AsyncMock(
            return_value=mock_result
        )

        client = ModelClient()
        result = await client.delete_async("test")

        mock_control_api.delete_model_proxy_async.assert_called_once()
        mock_control_api.delete_model_service_async.assert_called_once()
        assert isinstance(result, ModelService)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_delete_async_proxy_raises_error(
        self, mock_control_api_class
    ):
        """Test that proxy delete raises error when backend_type is PROXY"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_control_api.delete_model_proxy_async = AsyncMock(
            side_effect=HTTPError(status_code=404, message="Not found")
        )

        client = ModelClient()
        with pytest.raises(Exception):
            await client.delete_async("test", backend_type=BackendType.PROXY)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_delete_async_service_fallback_raises_error(
        self, mock_control_api_class
    ):
        """Test that service delete raises error after proxy fallback fails"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        # Both proxy and service delete fail
        mock_control_api.delete_model_proxy_async = AsyncMock(
            side_effect=HTTPError(status_code=404, message="Not found")
        )
        mock_control_api.delete_model_service_async = AsyncMock(
            side_effect=HTTPError(status_code=404, message="Not found")
        )

        client = ModelClient()
        with pytest.raises(Exception):
            await client.delete_async("test")

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_delete_proxy_raises_error(self, mock_control_api_class):
        """Test that proxy delete raises error when backend_type is PROXY"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_control_api.delete_model_proxy.side_effect = HTTPError(
            status_code=404, message="Not found"
        )

        client = ModelClient()
        with pytest.raises(Exception):
            client.delete("test", backend_type=BackendType.PROXY)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_delete_service_fallback_raises_error(self, mock_control_api_class):
        """Test that service delete raises error after proxy fallback fails"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        # Both proxy and service delete fail
        mock_control_api.delete_model_proxy.side_effect = HTTPError(
            status_code=404, message="Not found"
        )
        mock_control_api.delete_model_service.side_effect = HTTPError(
            status_code=404, message="Not found"
        )

        client = ModelClient()
        with pytest.raises(Exception):
            client.delete("test")


class TestModelClientUpdate:
    """Tests for ModelClient.update methods"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_update_service(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.update_model_service.return_value = mock_result

        client = ModelClient()
        input_obj = ModelServiceUpdateInput(description="Updated")

        result = client.update("test-service", input_obj)

        mock_control_api.update_model_service.assert_called_once()
        assert isinstance(result, ModelService)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_update_proxy(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.update_model_proxy.return_value = mock_result

        client = ModelClient()
        input_obj = ModelProxyUpdateInput(description="Updated")

        result = client.update("test-proxy", input_obj)

        mock_control_api.update_model_proxy.assert_called_once()
        assert isinstance(result, ModelProxy)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_update_proxy_auto_mode(self, mock_control_api_class):
        """Test auto-detection of proxy mode during update"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.update_model_proxy.return_value = mock_result

        client = ModelClient()
        input_obj = ModelProxyUpdateInput(
            proxy_config=ProxyConfig(
                endpoints=[
                    ProxyConfigEndpoint(model_names=["gpt-4"]),
                    ProxyConfigEndpoint(model_names=["gpt-3.5"]),
                ]
            ),
        )

        client.update("test-proxy", input_obj)

        assert input_obj.proxy_mode == ProxyMode.MULTI

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_update_async_service(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.update_model_service_async = AsyncMock(
            return_value=mock_result
        )

        client = ModelClient()
        input_obj = ModelServiceUpdateInput(description="Updated")

        result = await client.update_async("test-service", input_obj)

        mock_control_api.update_model_service_async.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_update_async_proxy(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.update_model_proxy_async = AsyncMock(
            return_value=mock_result
        )

        client = ModelClient()
        input_obj = ModelProxyUpdateInput(description="Updated")

        result = await client.update_async("test-proxy", input_obj)

        mock_control_api.update_model_proxy_async.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_update_async_service_raises_error(
        self, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_control_api.update_model_service_async = AsyncMock(
            side_effect=HTTPError(status_code=404, message="Not found")
        )

        client = ModelClient()
        input_obj = ModelServiceUpdateInput(description="Updated")

        with pytest.raises(Exception):
            await client.update_async("test-service", input_obj)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_update_async_proxy_raises_error(
        self, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_control_api.update_model_proxy_async = AsyncMock(
            side_effect=HTTPError(status_code=404, message="Not found")
        )

        client = ModelClient()
        input_obj = ModelProxyUpdateInput(description="Updated")

        with pytest.raises(Exception):
            await client.update_async("test-proxy", input_obj)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_update_service_raises_error(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_control_api.update_model_service.side_effect = HTTPError(
            status_code=404, message="Not found"
        )

        client = ModelClient()
        input_obj = ModelServiceUpdateInput(description="Updated")

        with pytest.raises(Exception):
            client.update("test-service", input_obj)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_update_proxy_raises_error(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_control_api.update_model_proxy.side_effect = HTTPError(
            status_code=404, message="Not found"
        )

        client = ModelClient()
        input_obj = ModelProxyUpdateInput(description="Updated")

        with pytest.raises(Exception):
            client.update("test-proxy", input_obj)


class TestModelClientGet:
    """Tests for ModelClient.get methods"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_get_proxy(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.get_model_proxy.return_value = mock_result

        client = ModelClient()
        result = client.get("test-proxy", backend_type=BackendType.PROXY)

        mock_control_api.get_model_proxy.assert_called_once()
        assert isinstance(result, ModelProxy)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_get_service(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.get_model_service.return_value = mock_result

        client = ModelClient()
        result = client.get("test-service", backend_type=BackendType.SERVICE)

        mock_control_api.get_model_service.assert_called_once()
        assert isinstance(result, ModelService)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_get_auto_detect(self, mock_control_api_class):
        """Test auto-detection when backend_type is None"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.get_model_proxy.return_value = mock_result

        client = ModelClient()
        result = client.get("test")

        # Should try proxy first
        mock_control_api.get_model_proxy.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_get_auto_detect_falls_back_to_service(
        self, mock_control_api_class
    ):
        """Test fallback to service when proxy not found"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        # Proxy get fails
        mock_control_api.get_model_proxy.side_effect = HTTPError(
            status_code=404, message="Not found"
        )
        # Service get succeeds
        mock_result = MagicMock()
        mock_control_api.get_model_service.return_value = mock_result

        client = ModelClient()
        result = client.get("test")

        mock_control_api.get_model_proxy.assert_called_once()
        mock_control_api.get_model_service.assert_called_once()
        assert isinstance(result, ModelService)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_get_async(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.get_model_proxy_async = AsyncMock(
            return_value=mock_result
        )

        client = ModelClient()
        result = await client.get_async(
            "test-proxy", backend_type=BackendType.PROXY
        )

        mock_control_api.get_model_proxy_async.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_get_async_service(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_control_api.get_model_service_async = AsyncMock(
            return_value=mock_result
        )

        client = ModelClient()
        result = await client.get_async(
            "test-service", backend_type=BackendType.SERVICE
        )

        mock_control_api.get_model_service_async.assert_called_once()
        assert isinstance(result, ModelService)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_get_async_auto_detect_fallback(self, mock_control_api_class):
        """Test fallback to service when proxy not found in async get"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        # Proxy get fails
        mock_control_api.get_model_proxy_async = AsyncMock(
            side_effect=HTTPError(status_code=404, message="Not found")
        )
        # Service get succeeds
        mock_result = MagicMock()
        mock_control_api.get_model_service_async = AsyncMock(
            return_value=mock_result
        )

        client = ModelClient()
        result = await client.get_async("test")

        mock_control_api.get_model_proxy_async.assert_called_once()
        mock_control_api.get_model_service_async.assert_called_once()
        assert isinstance(result, ModelService)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_get_async_proxy_raises_error(self, mock_control_api_class):
        """Test that proxy get raises error when backend_type is PROXY"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_control_api.get_model_proxy_async = AsyncMock(
            side_effect=HTTPError(status_code=404, message="Not found")
        )

        client = ModelClient()
        with pytest.raises(Exception):
            await client.get_async("test", backend_type=BackendType.PROXY)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_get_async_service_raises_error(self, mock_control_api_class):
        """Test that service get raises error"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_control_api.get_model_service_async = AsyncMock(
            side_effect=HTTPError(status_code=404, message="Not found")
        )

        client = ModelClient()
        with pytest.raises(Exception):
            await client.get_async("test", backend_type=BackendType.SERVICE)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_get_proxy_raises_error(self, mock_control_api_class):
        """Test that proxy get raises error when backend_type is PROXY"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_control_api.get_model_proxy.side_effect = HTTPError(
            status_code=404, message="Not found"
        )

        client = ModelClient()
        with pytest.raises(Exception):
            client.get("test", backend_type=BackendType.PROXY)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_get_service_fallback_raises_error(self, mock_control_api_class):
        """Test that service get raises error after proxy fallback fails"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        # Both proxy and service get fail
        mock_control_api.get_model_proxy.side_effect = HTTPError(
            status_code=404, message="Not found"
        )
        mock_control_api.get_model_service.side_effect = HTTPError(
            status_code=404, message="Not found"
        )

        client = ModelClient()
        with pytest.raises(Exception):
            client.get("test")


class TestModelClientList:
    """Tests for ModelClient.list methods"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_list_services(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_result.items = [MagicMock(), MagicMock()]
        mock_control_api.list_model_services.return_value = mock_result

        client = ModelClient()
        input_obj = ModelServiceListInput()

        result = client.list(input_obj)

        mock_control_api.list_model_services.assert_called_once()
        assert len(result) == 2
        assert all(isinstance(item, ModelService) for item in result)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_list_proxies(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_result.items = [MagicMock()]
        mock_control_api.list_model_proxies.return_value = mock_result

        client = ModelClient()
        input_obj = ModelProxyListInput()

        result = client.list(input_obj)

        mock_control_api.list_model_proxies.assert_called_once()
        assert len(result) == 1
        assert all(isinstance(item, ModelProxy) for item in result)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    def test_list_empty_items(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_result.items = None
        mock_control_api.list_model_services.return_value = mock_result

        client = ModelClient()
        input_obj = ModelServiceListInput()

        result = client.list(input_obj)

        assert result == []

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_list_async_services(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_result.items = [MagicMock()]
        mock_control_api.list_model_services_async = AsyncMock(
            return_value=mock_result
        )

        client = ModelClient()
        input_obj = ModelServiceListInput()

        result = await client.list_async(input_obj)

        mock_control_api.list_model_services_async.assert_called_once()
        assert len(result) == 1

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelControlAPI")
    @pytest.mark.asyncio
    async def test_list_async_proxies(self, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_result = MagicMock()
        mock_result.items = [MagicMock()]
        mock_control_api.list_model_proxies_async = AsyncMock(
            return_value=mock_result
        )

        client = ModelClient()
        input_obj = ModelProxyListInput()

        result = await client.list_async(input_obj)

        mock_control_api.list_model_proxies_async.assert_called_once()
        assert len(result) == 1
