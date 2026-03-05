"""测试 agentrun.sandbox.sandbox 模块 / Test agentrun.sandbox.sandbox module"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.sandbox.aio_sandbox import AioSandbox
from agentrun.sandbox.browser_sandbox import BrowserSandbox
from agentrun.sandbox.code_interpreter_sandbox import CodeInterpreterSandbox
from agentrun.sandbox.custom_sandbox import CustomSandbox
from agentrun.sandbox.model import (
    ListSandboxesInput,
    TemplateInput,
    TemplateType,
)
from agentrun.sandbox.sandbox import Sandbox
from agentrun.sandbox.template import Template
from agentrun.utils.config import Config


class MockTemplateData:

    def to_map(self):
        return {
            "templateId": "tmpl-123",
            "templateName": "test-template",
            "templateType": "CodeInterpreter",
            "status": "READY",
        }


class MockBrowserTemplateData:

    def to_map(self):
        return {
            "templateId": "tmpl-456",
            "templateName": "test-browser",
            "templateType": "Browser",
            "status": "READY",
        }


class MockAioTemplateData:

    def to_map(self):
        return {
            "templateId": "tmpl-789",
            "templateName": "test-aio",
            "templateType": "AllInOne",
            "status": "READY",
        }


class MockCustomTemplateData:

    def to_map(self):
        return {
            "templateId": "tmpl-000",
            "templateName": "test-custom",
            "templateType": "CustomImage",
            "status": "READY",
        }


class MockListTemplatesResult:

    def __init__(self, items):
        self.items = items


class MockListSandboxesResult:

    def __init__(self, items, next_token=None):
        self.items = items
        self.next_token = next_token


class MockSandboxListItem:

    def to_map(self):
        return {
            "sandboxId": "sandbox-123",
            "templateName": "test-template",
            "status": "RUNNING",
        }


# ==================== Sandbox.create 测试 ====================


class TestSandboxCreate:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_code_interpreter(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockTemplateData()
        mock_control_api.create_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.create_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {
                "sandboxId": "sandbox-ci-123",
                "templateName": "test-template",
            },
        }
        mock_data_api_class.return_value = mock_data_api

        result = Sandbox.create(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-template",
        )
        assert isinstance(result, CodeInterpreterSandbox)
        assert result.sandbox_id == "sandbox-ci-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_browser(self, mock_data_api_class, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockBrowserTemplateData()
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.create_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {
                "sandboxId": "sandbox-br-123",
                "templateName": "test-browser",
            },
        }
        mock_data_api_class.return_value = mock_data_api

        result = Sandbox.create(
            template_type=TemplateType.BROWSER,
            template_name="test-browser",
        )
        assert isinstance(result, BrowserSandbox)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_aio(self, mock_data_api_class, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockAioTemplateData()
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.create_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {
                "sandboxId": "sandbox-aio-123",
                "templateName": "test-aio",
            },
        }
        mock_data_api_class.return_value = mock_data_api

        result = Sandbox.create(
            template_type=TemplateType.AIO,
            template_name="test-aio",
        )
        assert isinstance(result, AioSandbox)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_custom(self, mock_data_api_class, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockCustomTemplateData()
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.create_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {
                "sandboxId": "sandbox-custom-123",
                "templateName": "test-custom",
            },
        }
        mock_data_api_class.return_value = mock_data_api

        result = Sandbox.create(
            template_type=TemplateType.CUSTOM,
            template_name="test-custom",
        )
        assert isinstance(result, CustomSandbox)

    def test_create_without_template_name(self):
        with pytest.raises(ValueError, match="template_name is required"):
            Sandbox.create(template_type=TemplateType.CODE_INTERPRETER)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_type_mismatch(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        with pytest.raises(ValueError, match="template_type of"):
            Sandbox.create(
                template_type=TemplateType.BROWSER,
                template_name="test-template",
            )

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_unsupported_type(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        unsupported = MagicMock()
        unsupported.to_map.return_value = {
            "templateName": "test",
            "templateType": "UnknownType",
            "status": "READY",
        }
        mock_control_api.get_template.return_value = unsupported
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.create_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sandbox-x"},
        }
        mock_data_api_class.return_value = mock_data_api

        with pytest.raises(ValueError, match="is not supported"):
            Sandbox.create(
                template_type="UnknownType",
                template_name="test",
            )


class TestSandboxCreateAsync:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_create_async_code_interpreter(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template_async = AsyncMock(
            return_value=MockTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.create_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sandbox-ci-123"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        result = await Sandbox.create_async(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-template",
        )
        assert isinstance(result, CodeInterpreterSandbox)

    @pytest.mark.asyncio
    async def test_create_async_without_template_name(self):
        with pytest.raises(ValueError, match="template_name is required"):
            await Sandbox.create_async(
                template_type=TemplateType.CODE_INTERPRETER
            )

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_create_async_type_mismatch(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template_async = AsyncMock(
            return_value=MockTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        with pytest.raises(ValueError, match="template_type of"):
            await Sandbox.create_async(
                template_type=TemplateType.BROWSER,
                template_name="test-template",
            )

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_create_async_browser(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template_async = AsyncMock(
            return_value=MockBrowserTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.create_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sandbox-br"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        result = await Sandbox.create_async(
            template_type=TemplateType.BROWSER,
            template_name="test-browser",
        )
        assert isinstance(result, BrowserSandbox)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_create_async_aio(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template_async = AsyncMock(
            return_value=MockAioTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.create_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sandbox-aio"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        result = await Sandbox.create_async(
            template_type=TemplateType.AIO,
            template_name="test-aio",
        )
        assert isinstance(result, AioSandbox)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_create_async_custom(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template_async = AsyncMock(
            return_value=MockCustomTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.create_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sandbox-custom"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        result = await Sandbox.create_async(
            template_type=TemplateType.CUSTOM,
            template_name="test-custom",
        )
        assert isinstance(result, CustomSandbox)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_create_async_unsupported_type(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        unsupported = MagicMock()
        unsupported.to_map.return_value = {
            "templateName": "test",
            "templateType": "UnknownType",
            "status": "READY",
        }
        mock_control_api.get_template_async = AsyncMock(
            return_value=unsupported
        )
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.create_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sandbox-x"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        with pytest.raises(ValueError, match="is not supported"):
            await Sandbox.create_async(
                template_type="UnknownType",
                template_name="test",
            )


# ==================== Sandbox.connect 测试 ====================


class TestSandboxConnect:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_connect_code_interpreter_with_type(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {
                "sandboxId": "sandbox-123",
                "templateName": "test-template",
            },
        }
        mock_data_api_class.return_value = mock_data_api

        result = Sandbox.connect(
            "sandbox-123",
            template_type=TemplateType.CODE_INTERPRETER,
        )
        assert isinstance(result, CodeInterpreterSandbox)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_connect_browser_with_type(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {
                "sandboxId": "sandbox-123",
                "templateName": "test-browser",
            },
        }
        mock_data_api_class.return_value = mock_data_api

        result = Sandbox.connect(
            "sandbox-123",
            template_type=TemplateType.BROWSER,
        )
        assert isinstance(result, BrowserSandbox)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_connect_aio_with_type(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sandbox-123"},
        }
        mock_data_api_class.return_value = mock_data_api

        result = Sandbox.connect(
            "sandbox-123",
            template_type=TemplateType.AIO,
        )
        assert isinstance(result, AioSandbox)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_connect_without_type_resolves(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.get_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {
                "sandboxId": "sandbox-123",
                "templateName": "test-template",
            },
        }
        mock_data_api_class.return_value = mock_data_api

        result = Sandbox.connect("sandbox-123")
        assert isinstance(result, CodeInterpreterSandbox)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_connect_unsupported_type(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sandbox-123"},
        }
        mock_data_api_class.return_value = mock_data_api

        with pytest.raises(ValueError, match="Unsupported template type"):
            Sandbox.connect(
                "sandbox-123",
                template_type=TemplateType.CUSTOM,
            )

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_connect_no_template_name(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sandbox-123", "templateName": None},
        }
        mock_data_api_class.return_value = mock_data_api

        with pytest.raises(ValueError, match="has no template_name"):
            Sandbox.connect("sandbox-123")


class TestSandboxConnectAsync:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_connect_async_code_interpreter(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sandbox-123"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        result = await Sandbox.connect_async(
            "sandbox-123",
            template_type=TemplateType.CODE_INTERPRETER,
        )
        assert isinstance(result, CodeInterpreterSandbox)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_connect_async_without_type(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template_async = AsyncMock(
            return_value=MockBrowserTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.get_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {
                    "sandboxId": "sandbox-123",
                    "templateName": "test-browser",
                },
            }
        )
        mock_data_api_class.return_value = mock_data_api

        result = await Sandbox.connect_async("sandbox-123")
        assert isinstance(result, BrowserSandbox)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_connect_async_no_template_name(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sandbox-123", "templateName": None},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        with pytest.raises(ValueError, match="has no template_name"):
            await Sandbox.connect_async("sandbox-123")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_connect_async_unsupported_type(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sandbox-123"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        with pytest.raises(ValueError, match="Unsupported template type"):
            await Sandbox.connect_async(
                "sandbox-123",
                template_type=TemplateType.CUSTOM,
            )

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_connect_async_aio(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sandbox-123"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        result = await Sandbox.connect_async(
            "sandbox-123", template_type=TemplateType.AIO
        )
        assert isinstance(result, AioSandbox)


# ==================== Sandbox 实例方法测试 ====================


class TestSandboxInstanceMethods:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_stop_by_id(self, mock_data_api_class, mock_control_api_class):
        mock_data_api = MagicMock()
        mock_data_api.stop_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sandbox-123"},
        }
        mock_data_api_class.return_value = mock_data_api

        result = Sandbox.stop_by_id("sandbox-123")
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_stop_by_id_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.stop_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sandbox-123"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        result = await Sandbox.stop_by_id_async("sandbox-123")
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_delete_by_id(self, mock_data_api_class, mock_control_api_class):
        mock_data_api = MagicMock()
        mock_data_api.delete_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sandbox-123"},
        }
        mock_data_api_class.return_value = mock_data_api

        result = Sandbox.delete_by_id("sandbox-123")
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_delete_by_id_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.delete_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sandbox-123"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        result = await Sandbox.delete_by_id_async("sandbox-123")
        assert result.sandbox_id == "sandbox-123"

    def test_get_without_sandbox_id(self):
        sandbox = Sandbox()
        with pytest.raises(ValueError, match="sandbox_id is required"):
            sandbox.get()

    @pytest.mark.asyncio
    async def test_get_async_without_sandbox_id(self):
        sandbox = Sandbox()
        with pytest.raises(ValueError, match="sandbox_id is required"):
            await sandbox.get_async()

    def test_delete_without_sandbox_id(self):
        sandbox = Sandbox()
        with pytest.raises(ValueError, match="sandbox_id is required"):
            sandbox.delete()

    @pytest.mark.asyncio
    async def test_delete_async_without_sandbox_id(self):
        sandbox = Sandbox()
        with pytest.raises(ValueError, match="sandbox_id is required"):
            await sandbox.delete_async()

    def test_stop_without_sandbox_id(self):
        sandbox = Sandbox()
        with pytest.raises(ValueError, match="sandbox_id is required"):
            sandbox.stop()

    @pytest.mark.asyncio
    async def test_stop_async_without_sandbox_id(self):
        sandbox = Sandbox()
        with pytest.raises(ValueError, match="sandbox_id is required"):
            await sandbox.stop_async()


# ==================== Sandbox.list 测试 ====================


class TestSandboxList:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_list_sync(self, mock_data_api_class, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api.list_sandboxes.return_value = MockListSandboxesResult(
            [MockSandboxListItem()]
        )
        mock_control_api_class.return_value = mock_control_api

        result = Sandbox.list()
        assert len(result.sandboxes) == 1

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_list_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.list_sandboxes_async = AsyncMock(
            return_value=MockListSandboxesResult([MockSandboxListItem()])
        )
        mock_control_api_class.return_value = mock_control_api

        result = await Sandbox.list_async()
        assert len(result.sandboxes) == 1


# ==================== Sandbox Template 类方法测试 ====================


class TestSandboxTemplateMethods:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_template(self, mock_data_api_class, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api.create_template.return_value = MockTemplateData()
        mock_control_api.get_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-template",
        )
        result = Sandbox.create_template(input_obj)
        assert result.template_name == "test-template"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_create_template_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.create_template_async = AsyncMock(
            return_value=MockTemplateData()
        )
        mock_control_api.get_template_async = AsyncMock(
            return_value=MockTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-template",
        )
        result = await Sandbox.create_template_async(input_obj)
        assert result.template_name == "test-template"

    def test_create_template_no_type(self):
        input_obj = MagicMock()
        input_obj.template_type = None
        with pytest.raises(ValueError, match="template_type is required"):
            Sandbox.create_template(input_obj)

    @pytest.mark.asyncio
    async def test_create_template_async_no_type(self):
        input_obj = MagicMock()
        input_obj.template_type = None
        with pytest.raises(ValueError, match="template_type is required"):
            await Sandbox.create_template_async(input_obj)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_get_template(self, mock_data_api_class, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        result = Sandbox.get_template("test-template")
        assert result.template_name == "test-template"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_get_template_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template_async = AsyncMock(
            return_value=MockTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = await Sandbox.get_template_async("test-template")
        assert result.template_name == "test-template"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_update_template(self, mock_data_api_class, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api.update_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        input_obj = TemplateInput(template_type=TemplateType.CODE_INTERPRETER)
        result = Sandbox.update_template("test-template", input_obj)
        assert result is not None

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_update_template_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.update_template_async = AsyncMock(
            return_value=MockTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        input_obj = TemplateInput(template_type=TemplateType.CODE_INTERPRETER)
        result = await Sandbox.update_template_async("test-template", input_obj)
        assert result is not None

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_delete_template(self, mock_data_api_class, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api.delete_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        result = Sandbox.delete_template("test-template")
        assert result is not None

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_delete_template_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.delete_template_async = AsyncMock(
            return_value=MockTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = await Sandbox.delete_template_async("test-template")
        assert result is not None

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_list_templates(self, mock_data_api_class, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api.list_templates.return_value = MockListTemplatesResult(
            [MockTemplateData()]
        )
        mock_control_api_class.return_value = mock_control_api

        result = Sandbox.list_templates()
        assert len(result) == 1

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_list_templates_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.list_templates_async = AsyncMock(
            return_value=MockListTemplatesResult([MockTemplateData()])
        )
        mock_control_api_class.return_value = mock_control_api

        result = await Sandbox.list_templates_async()
        assert len(result) == 1


# ==================== None-guard tests for class methods ====================


class TestSandboxNoneGuards:

    def test_stop_by_id_none(self):
        with pytest.raises(ValueError, match="sandbox_id is required"):
            Sandbox.stop_by_id(None)

    @pytest.mark.asyncio
    async def test_stop_by_id_async_none(self):
        with pytest.raises(ValueError, match="sandbox_id is required"):
            await Sandbox.stop_by_id_async(None)

    def test_delete_by_id_none(self):
        with pytest.raises(ValueError, match="sandbox_id is required"):
            Sandbox.delete_by_id(None)

    @pytest.mark.asyncio
    async def test_delete_by_id_async_none(self):
        with pytest.raises(ValueError, match="sandbox_id is required"):
            await Sandbox.delete_by_id_async(None)

    def test_connect_none_sandbox_id(self):
        with pytest.raises(ValueError, match="sandbox_id is required"):
            Sandbox.connect(None)

    @pytest.mark.asyncio
    async def test_connect_async_none_sandbox_id(self):
        with pytest.raises(ValueError, match="sandbox_id is required"):
            await Sandbox.connect_async(None)

    def test_get_template_none(self):
        with pytest.raises(ValueError, match="template_name is required"):
            Sandbox.get_template(None)

    @pytest.mark.asyncio
    async def test_get_template_async_none(self):
        with pytest.raises(ValueError, match="template_name is required"):
            await Sandbox.get_template_async(None)

    def test_update_template_none(self):
        with pytest.raises(ValueError, match="template_name is required"):
            Sandbox.update_template(
                None, TemplateInput(template_type=TemplateType.CODE_INTERPRETER)
            )

    @pytest.mark.asyncio
    async def test_update_template_async_none(self):
        with pytest.raises(ValueError, match="template_name is required"):
            await Sandbox.update_template_async(
                None, TemplateInput(template_type=TemplateType.CODE_INTERPRETER)
            )

    def test_delete_template_none(self):
        with pytest.raises(ValueError, match="template_name is required"):
            Sandbox.delete_template(None)

    @pytest.mark.asyncio
    async def test_delete_template_async_none(self):
        with pytest.raises(ValueError, match="template_name is required"):
            await Sandbox.delete_template_async(None)


# ==================== Instance method happy paths ====================


class TestSandboxInstanceHappyPath:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_get_calls_connect(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.get_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sb-1", "templateName": "tpl"},
        }
        mock_data_api_class.return_value = mock_data_api

        sb = Sandbox(sandbox_id="sb-1")
        result = sb.get()
        assert result.sandbox_id == "sb-1"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_get_async_calls_connect(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template_async = AsyncMock(
            return_value=MockTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        mock_data_api = MagicMock()
        mock_data_api.get_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sb-1", "templateName": "tpl"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        sb = Sandbox(sandbox_id="sb-1")
        result = await sb.get_async()
        assert result.sandbox_id == "sb-1"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_delete_calls_delete_by_id(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.delete_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sb-1"},
        }
        mock_data_api_class.return_value = mock_data_api

        sb = Sandbox(sandbox_id="sb-1")
        result = sb.delete()
        assert result.sandbox_id == "sb-1"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_delete_async_calls_delete_by_id(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.delete_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sb-1"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        sb = Sandbox(sandbox_id="sb-1")
        result = await sb.delete_async()
        assert result.sandbox_id == "sb-1"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_stop_calls_stop_by_id(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.stop_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sb-1"},
        }
        mock_data_api_class.return_value = mock_data_api

        sb = Sandbox(sandbox_id="sb-1")
        result = sb.stop()
        assert result.sandbox_id == "sb-1"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_stop_async_calls_stop_by_id(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.stop_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sb-1"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        sb = Sandbox(sandbox_id="sb-1")
        result = await sb.stop_async()
        assert result.sandbox_id == "sb-1"
