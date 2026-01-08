"""Tests for agentrun/model/model_service.py"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.model.model import (
    ModelServiceCreateInput,
    ModelServiceUpdateInput,
    ModelType,
    ProviderSettings,
)
from agentrun.model.model_service import ModelService
from agentrun.utils.config import Config
from agentrun.utils.model import Status


class TestModelServiceCreate:
    """Tests for ModelService.create methods"""

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

        mock_service = ModelService(model_service_name="test-service")
        mock_client.create.return_value = mock_service

        input_obj = ModelServiceCreateInput(
            model_service_name="test-service",
            provider="openai",
        )

        result = ModelService.create(input_obj)

        mock_client.create.assert_called_once()
        assert result == mock_service

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

        mock_service = ModelService(model_service_name="test-service")
        mock_client.create_async = AsyncMock(return_value=mock_service)

        input_obj = ModelServiceCreateInput(model_service_name="test-service")

        result = await ModelService.create_async(input_obj)

        mock_client.create_async.assert_called_once()
        assert result == mock_service


class TestModelServiceDelete:
    """Tests for ModelService.delete methods"""

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

        mock_service = ModelService(model_service_name="test-service")
        mock_client.delete.return_value = mock_service

        result = ModelService.delete_by_name("test-service")

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

        await ModelService.delete_by_name_async("test-service")

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

        service = ModelService(model_service_name="test-service")
        service.delete()

        mock_client.delete.assert_called_once()

    def test_delete_without_name_raises_error(self):
        service = ModelService()
        with pytest.raises(ValueError, match="model_service_name is required"):
            service.delete()

    @pytest.mark.asyncio
    async def test_delete_async_without_name_raises_error(self):
        service = ModelService()
        with pytest.raises(ValueError, match="model_service_name is required"):
            await service.delete_async()


class TestModelServiceUpdate:
    """Tests for ModelService.update methods"""

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

        mock_service = ModelService(model_service_name="test-service")
        mock_client.update.return_value = mock_service

        input_obj = ModelServiceUpdateInput(description="Updated")

        result = ModelService.update_by_name("test-service", input_obj)

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

        mock_service = ModelService(model_service_name="test-service")
        mock_client.update_async = AsyncMock(return_value=mock_service)

        input_obj = ModelServiceUpdateInput(description="Updated")

        await ModelService.update_by_name_async("test-service", input_obj)

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

        updated_service = ModelService(
            model_service_name="test-service", description="Updated"
        )
        mock_client.update.return_value = updated_service

        service = ModelService(model_service_name="test-service")
        input_obj = ModelServiceUpdateInput(description="Updated")

        result = service.update(input_obj)

        assert result.description == "Updated"

    def test_update_without_name_raises_error(self):
        service = ModelService()
        input_obj = ModelServiceUpdateInput(description="Test")
        with pytest.raises(ValueError, match="model_service_name is required"):
            service.update(input_obj)

    @pytest.mark.asyncio
    async def test_update_async_without_name_raises_error(self):
        service = ModelService()
        input_obj = ModelServiceUpdateInput(description="Test")
        with pytest.raises(ValueError, match="model_service_name is required"):
            await service.update_async(input_obj)

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

        updated_service = ModelService(
            model_service_name="test-service", description="Updated"
        )
        mock_client.update_async = AsyncMock(return_value=updated_service)

        service = ModelService(model_service_name="test-service")
        input_obj = ModelServiceUpdateInput(description="Updated")

        result = await service.update_async(input_obj)

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

        service = ModelService(model_service_name="test-service")
        await service.delete_async()

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

        mock_service = ModelService(
            model_service_name="test-service", status=Status.READY
        )
        mock_client.get_async = AsyncMock(return_value=mock_service)

        service = ModelService(model_service_name="test-service")
        result = await service.get_async()

        assert result.status == Status.READY


class TestModelServiceGet:
    """Tests for ModelService.get methods"""

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

        mock_service = ModelService(model_service_name="test-service")
        mock_client.get.return_value = mock_service

        result = ModelService.get_by_name("test-service")

        mock_client.get.assert_called_once()
        assert result == mock_service

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

        mock_service = ModelService(model_service_name="test-service")
        mock_client.get_async = AsyncMock(return_value=mock_service)

        result = await ModelService.get_by_name_async("test-service")

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

        mock_service = ModelService(
            model_service_name="test-service", status=Status.READY
        )
        mock_client.get.return_value = mock_service

        service = ModelService(model_service_name="test-service")
        result = service.get()

        assert result.status == Status.READY

    def test_get_without_name_raises_error(self):
        service = ModelService()
        with pytest.raises(ValueError, match="model_service_name is required"):
            service.get()

    @pytest.mark.asyncio
    async def test_get_async_without_name_raises_error(self):
        service = ModelService()
        with pytest.raises(ValueError, match="model_service_name is required"):
            await service.get_async()


class TestModelServiceRefresh:
    """Tests for ModelService.refresh methods"""

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

        mock_service = ModelService(
            model_service_name="test-service", status=Status.READY
        )
        mock_client.get.return_value = mock_service

        service = ModelService(model_service_name="test-service")
        result = service.refresh()

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

        mock_service = ModelService(
            model_service_name="test-service", status=Status.READY
        )
        mock_client.get_async = AsyncMock(return_value=mock_service)

        service = ModelService(model_service_name="test-service")
        result = await service.refresh_async()

        mock_client.get_async.assert_called()


class TestModelServiceList:
    """Tests for ModelService.list methods"""

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

        mock_services = [
            ModelService(model_service_name="service1", model_service_id="id1"),
            ModelService(model_service_name="service2", model_service_id="id2"),
        ]
        mock_client.list.return_value = mock_services

        result = ModelService.list_all()

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

        mock_services = [
            ModelService(model_service_name="service1", model_service_id="id1"),
        ]
        mock_client.list_async = AsyncMock(return_value=mock_services)

        result = await ModelService.list_all_async()

        mock_client.list_async.assert_called()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.model.client.ModelClient")
    def test_list_all_with_filters(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_services = [
            ModelService(model_service_name="service1", model_service_id="id1"),
        ]
        mock_client.list.return_value = mock_services

        result = ModelService.list_all(
            model_type=ModelType.LLM,
            provider="openai",
        )

        mock_client.list.assert_called()


class TestModelServiceModelInfo:
    """Tests for ModelService.model_info method"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    def test_model_info(self):
        service = ModelService(
            model_service_name="test-service",
            provider_settings=ProviderSettings(
                api_key="test-key",
                base_url="https://api.example.com",
                model_names=["gpt-4"],
            ),
        )

        info = service.model_info()

        assert info.api_key == "test-key"
        assert info.base_url == "https://api.example.com"
        assert info.model == "gpt-4"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    def test_model_info_with_empty_model_names(self):
        service = ModelService(
            model_service_name="test-service",
            provider_settings=ProviderSettings(
                api_key="test-key",
                base_url="https://api.example.com",
                model_names=[],
            ),
        )

        info = service.model_info()

        assert info.model is None

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.credential.Credential")
    def test_model_info_with_credential_name(self, mock_credential):
        mock_cred_instance = MagicMock()
        mock_cred_instance.credential_secret = "secret-key"
        mock_credential.get_by_name.return_value = mock_cred_instance

        service = ModelService(
            model_service_name="test-service",
            credential_name="test-credential",
            provider_settings=ProviderSettings(
                base_url="https://api.example.com",
                model_names=["gpt-4"],
            ),
        )

        info = service.model_info()

        assert info.api_key == "secret-key"


class TestModelServiceCompletions:
    """Tests for ModelService.completions method"""

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
        mock_completion.return_value = {"choices": []}

        service = ModelService(
            model_service_name="test-service",
            provider_settings=ProviderSettings(
                api_key="test-key",
                base_url="https://api.example.com",
                model_names=["gpt-4"],
            ),
        )

        service.completions(messages=[{"role": "user", "content": "Hello"}])

        mock_completion.assert_called_once()


class TestModelServiceResponses:
    """Tests for ModelService.responses method"""

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
        mock_responses.return_value = {}

        service = ModelService(
            model_service_name="test-service",
            provider="openai",
            provider_settings=ProviderSettings(
                api_key="test-key",
                base_url="https://api.example.com",
                model_names=["gpt-4"],
            ),
        )

        # Note: The responses method expects 'messages' but ModelCompletionAPI.responses
        # expects 'input'. Using input parameter via kwargs to match the API signature.
        service.responses(
            messages=[{"role": "user", "content": "Hello"}],
            input="Hello",  # Required by ModelCompletionAPI.responses
        )

        mock_responses.assert_called_once()
