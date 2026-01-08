"""Tests for agentrun/model/api/data.py"""

import os
from unittest.mock import MagicMock, patch

import pytest

from agentrun.model.api.data import BaseInfo, ModelCompletionAPI, ModelDataAPI
from agentrun.utils.config import Config
from agentrun.utils.data_api import ResourceType


class TestBaseInfo:
    """Tests for BaseInfo model"""

    def test_default_values(self):
        info = BaseInfo()
        assert info.model is None
        assert info.api_key is None
        assert info.base_url is None
        assert info.headers is None
        assert info.provider is None

    def test_with_values(self):
        info = BaseInfo(
            model="gpt-4",
            api_key="test-key",
            base_url="https://api.example.com",
            headers={"Authorization": "Bearer token"},
            provider="openai",
        )
        assert info.model == "gpt-4"
        assert info.api_key == "test-key"
        assert info.base_url == "https://api.example.com"
        assert info.headers == {"Authorization": "Bearer token"}
        assert info.provider == "openai"

    def test_model_dump(self):
        info = BaseInfo(model="gpt-4", api_key="key")
        dumped = info.model_dump()
        assert "model" in dumped
        assert dumped["model"] == "gpt-4"


class TestModelCompletionAPI:
    """Tests for ModelCompletionAPI class"""

    def test_init(self):
        api = ModelCompletionAPI(
            api_key="test-key",
            base_url="https://api.example.com",
            model="gpt-4",
        )
        assert api.api_key == "test-key"
        assert api.base_url == "https://api.example.com"
        assert api.model == "gpt-4"
        assert api.provider == "openai"
        assert api.headers == {}

    def test_init_with_provider_and_headers(self):
        api = ModelCompletionAPI(
            api_key="test-key",
            base_url="https://api.example.com",
            model="claude-3",
            provider="anthropic",
            headers={"X-Custom": "value"},
        )
        assert api.provider == "anthropic"
        assert api.headers == {"X-Custom": "value"}

    @patch("litellm.completion")
    def test_completions(self, mock_completion):
        mock_completion.return_value = {
            "choices": [{"message": {"content": "Hello"}}]
        }

        api = ModelCompletionAPI(
            api_key="test-key",
            base_url="https://api.example.com",
            model="gpt-4",
        )

        result = api.completions(
            messages=[{"role": "user", "content": "Hello"}],
        )

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["base_url"] == "https://api.example.com"
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_kwargs["stream_options"]["include_usage"] is True

    @patch("litellm.completion")
    def test_completions_with_custom_model(self, mock_completion):
        mock_completion.return_value = {"choices": []}

        api = ModelCompletionAPI(
            api_key="test-key",
            base_url="https://api.example.com",
            model="gpt-4",
        )

        api.completions(
            messages=[{"role": "user", "content": "Test"}],
            model="gpt-3.5-turbo",
            custom_llm_provider="azure",
        )

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "gpt-3.5-turbo"
        assert call_kwargs["custom_llm_provider"] == "azure"

    @patch("litellm.completion")
    def test_completions_merges_headers(self, mock_completion):
        mock_completion.return_value = {"choices": []}

        api = ModelCompletionAPI(
            api_key="test-key",
            base_url="https://api.example.com",
            model="gpt-4",
            headers={"X-Default": "default"},
        )

        api.completions(
            messages=[],
            headers={"X-Custom": "custom"},
        )

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["headers"]["X-Default"] == "default"
        assert call_kwargs["headers"]["X-Custom"] == "custom"

    @patch("litellm.completion")
    def test_completions_with_existing_stream_options(self, mock_completion):
        """测试传入已存在的 stream_options 参数"""
        mock_completion.return_value = {"choices": []}

        api = ModelCompletionAPI(
            api_key="test-key",
            base_url="https://api.example.com",
            model="gpt-4",
        )

        api.completions(
            messages=[{"role": "user", "content": "Hello"}],
            stream_options={"custom_option": True},
        )

        call_kwargs = mock_completion.call_args[1]
        # 验证 stream_options 被保留并添加了 include_usage
        assert call_kwargs["stream_options"]["custom_option"] is True
        assert call_kwargs["stream_options"]["include_usage"] is True

    @patch("litellm.responses")
    def test_responses(self, mock_responses):
        mock_responses.return_value = {"output": "test"}

        api = ModelCompletionAPI(
            api_key="test-key",
            base_url="https://api.example.com",
            model="gpt-4",
        )

        api.responses(input="Hello, world!")

        mock_responses.assert_called_once()
        call_kwargs = mock_responses.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["base_url"] == "https://api.example.com"
        assert call_kwargs["input"] == "Hello, world!"
        assert call_kwargs["stream_options"]["include_usage"] is True

    @patch("litellm.responses")
    def test_responses_with_custom_model(self, mock_responses):
        mock_responses.return_value = {}

        api = ModelCompletionAPI(
            api_key="test-key",
            base_url="https://api.example.com",
            model="gpt-4",
        )

        api.responses(
            input="Test",
            model="gpt-3.5-turbo",
            custom_llm_provider="azure",
        )

        call_kwargs = mock_responses.call_args[1]
        assert call_kwargs["model"] == "gpt-3.5-turbo"
        assert call_kwargs["custom_llm_provider"] == "azure"

    @patch("litellm.responses")
    def test_responses_with_existing_stream_options(self, mock_responses):
        """测试 responses 传入已存在的 stream_options 参数"""
        mock_responses.return_value = {}

        api = ModelCompletionAPI(
            api_key="test-key",
            base_url="https://api.example.com",
            model="gpt-4",
        )

        api.responses(
            input="Hello, world!",
            stream_options={"custom_option": True},
        )

        call_kwargs = mock_responses.call_args[1]
        # 验证 stream_options 被保留并添加了 include_usage
        assert call_kwargs["stream_options"]["custom_option"] is True
        assert call_kwargs["stream_options"]["include_usage"] is True


class TestModelDataAPI:
    """Tests for ModelDataAPI class"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI")
    def test_init(self, mock_control_api):
        mock_control_api.return_value.get_data_endpoint.return_value = (
            "https://data.example.com"
        )

        api = ModelDataAPI(
            model_proxy_name="test-proxy",
            model_name="gpt-4",
        )

        assert api.model_proxy_name == "test-proxy"
        assert api.model_name == "gpt-4"
        assert api.namespace == "models/test-proxy"
        assert api.provider == "openai"
        assert api.access_token == ""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI")
    def test_init_with_credential_name(self, mock_control_api):
        mock_control_api.return_value.get_data_endpoint.return_value = (
            "https://data.example.com"
        )

        api = ModelDataAPI(
            model_proxy_name="test-proxy",
            credential_name="test-credential",
        )

        # When credential_name is provided, access_token is not set to empty
        assert api.access_token is None

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI")
    def test_update_model_name(self, mock_control_api):
        mock_control_api.return_value.get_data_endpoint.return_value = (
            "https://data.example.com"
        )

        api = ModelDataAPI(model_proxy_name="proxy1")
        api.update_model_name(
            model_proxy_name="proxy2",
            model_name="new-model",
            provider="anthropic",
        )

        assert api.model_proxy_name == "proxy2"
        assert api.model_name == "new-model"
        assert api.namespace == "models/proxy2"
        assert api.provider == "anthropic"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI")
    def test_model_info(self, mock_control_api):
        mock_control_api.return_value.get_data_endpoint.return_value = (
            "https://data.example.com"
        )

        api = ModelDataAPI(
            model_proxy_name="test-proxy",
            model_name="gpt-4",
            provider="openai",
        )

        # Mock the auth method
        api.auth = MagicMock(return_value=("token", {"X-Auth": "test"}, None))
        api.with_path = MagicMock(return_value="https://data.example.com/v1/")

        info = api.model_info()

        assert isinstance(info, BaseInfo)
        assert info.api_key == ""
        assert info.model == "gpt-4"
        assert info.provider == "openai"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI")
    @patch.object(ModelDataAPI, "model_info")
    @patch("agentrun.model.api.data.ModelCompletionAPI")
    def test_completions(
        self, mock_api_class, mock_model_info, mock_control_api
    ):
        mock_control_api.return_value.get_data_endpoint.return_value = (
            "https://data.example.com"
        )

        mock_info = BaseInfo(
            api_key="key",
            base_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        mock_model_info.return_value = mock_info

        mock_api_instance = MagicMock()
        mock_api_class.return_value = mock_api_instance

        api = ModelDataAPI(model_proxy_name="test-proxy")
        api.completions(messages=[{"role": "user", "content": "Hello"}])

        mock_api_class.assert_called_once_with(
            base_url="https://api.example.com",
            api_key="key",
            model="gpt-4",
            headers={},
        )
        mock_api_instance.completions.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-access-key",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
            "AGENTRUN_ACCOUNT_ID": "test-account",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI")
    @patch.object(ModelDataAPI, "model_info")
    @patch("agentrun.model.api.data.ModelCompletionAPI")
    def test_responses(self, mock_api_class, mock_model_info, mock_control_api):
        mock_control_api.return_value.get_data_endpoint.return_value = (
            "https://data.example.com"
        )

        mock_info = BaseInfo(
            api_key="key",
            base_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        mock_model_info.return_value = mock_info

        mock_api_instance = MagicMock()
        mock_api_class.return_value = mock_api_instance

        api = ModelDataAPI(model_proxy_name="test-proxy")
        api.responses(input="Hello, world!")

        mock_api_class.assert_called_once()
        mock_api_instance.responses.assert_called_once()
