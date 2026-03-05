"""Tests for agentrun.sandbox.custom_sandbox module."""

from unittest.mock import MagicMock, patch

import pytest

from agentrun.sandbox.custom_sandbox import CustomSandbox
from agentrun.sandbox.model import TemplateType


class TestCustomSandbox:

    def test_template_type(self):
        assert (
            CustomSandbox.__private_attributes__["_template_type"].default
            == TemplateType.CUSTOM
        )

    @patch("agentrun.sandbox.custom_sandbox.DataAPI")
    def test_get_base_url(self, mock_data_api_cls):
        mock_api = MagicMock()
        mock_api.with_path.return_value = "https://example.com/sandboxes"
        mock_data_api_cls.return_value = mock_api

        sb = CustomSandbox.model_construct(sandbox_id="sb-1")
        result = sb.get_base_url()
        assert result == "https://example.com/sandboxes"
        mock_api.with_path.assert_called_once_with("")

    @patch("agentrun.sandbox.custom_sandbox.DataAPI")
    def test_get_base_url_with_config(self, mock_data_api_cls):
        mock_api = MagicMock()
        mock_api.with_path.return_value = "https://custom.com"
        mock_data_api_cls.return_value = mock_api

        sb = CustomSandbox.model_construct(sandbox_id="sb-1")
        config = MagicMock()
        result = sb.get_base_url(config=config)
        assert result == "https://custom.com"
