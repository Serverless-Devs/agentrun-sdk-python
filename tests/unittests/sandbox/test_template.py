"""测试 agentrun.sandbox.template 模块 / Test agentrun.sandbox.template module"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.sandbox.model import PageableInput, TemplateInput, TemplateType
from agentrun.sandbox.template import Template
from agentrun.utils.config import Config


class MockTemplateData:

    def to_map(self):
        return {
            "templateId": "tmpl-123",
            "templateName": "test-template",
            "templateType": "CodeInterpreter",
            "cpu": 2.0,
            "memory": 4096,
            "diskSize": 512,
            "status": "READY",
            "createdAt": "2024-01-01T00:00:00Z",
            "lastUpdatedAt": "2024-01-01T00:00:00Z",
        }


class MockListTemplatesResult:

    def __init__(self, items):
        self.items = items


# ==================== Template.create 测试 ====================


class TestTemplateCreate:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_sync(self, mock_data_api_class, mock_control_api_class):
        mock_control_api = MagicMock()
        mock_control_api.create_template.return_value = MockTemplateData()
        mock_control_api.get_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-template",
        )
        result = Template.create(input_obj)
        assert result.template_name == "test-template"
        assert result.status == "READY"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_create_async(
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
        result = await Template.create_async(input_obj)
        assert result.template_name == "test-template"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_with_config(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.create_template.return_value = MockTemplateData()
        mock_control_api.get_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        config = Config(access_key_id="custom-ak")
        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-template",
        )
        result = Template.create(input_obj, config=config)
        assert result is not None


# ==================== Template.delete_by_name 测试 ====================


class TestTemplateDelete:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_delete_by_name_sync(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.delete_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        result = Template.delete_by_name("test-template")
        assert result is not None

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_delete_by_name_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.delete_template_async = AsyncMock(
            return_value=MockTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = await Template.delete_by_name_async("test-template")
        assert result is not None

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_delete_with_config(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.delete_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        config = Config(access_key_id="custom-ak")
        result = Template.delete_by_name("test-template", config=config)
        assert result is not None


# ==================== Template.update_by_name 测试 ====================


class TestTemplateUpdate:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_update_by_name_sync(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.update_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-template",
        )
        result = Template.update_by_name("test-template", input_obj)
        assert result is not None

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_update_by_name_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.update_template_async = AsyncMock(
            return_value=MockTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-template",
        )
        result = await Template.update_by_name_async("test-template", input_obj)
        assert result is not None


# ==================== Template.get_by_name 测试 ====================


class TestTemplateGet:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_get_by_name_sync(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        result = Template.get_by_name("test-template")
        assert result.template_name == "test-template"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_get_by_name_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template_async = AsyncMock(
            return_value=MockTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        result = await Template.get_by_name_async("test-template")
        assert result.template_name == "test-template"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_get_by_name_with_config(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        config = Config(access_key_id="custom-ak")
        result = Template.get_by_name("test-template", config=config)
        assert result is not None


# ==================== Template.list_templates 测试 ====================


class TestTemplateList:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_list_templates_sync(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.list_templates.return_value = MockListTemplatesResult(
            [MockTemplateData()]
        )
        mock_control_api_class.return_value = mock_control_api

        result = Template.list_templates()
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

        result = await Template.list_templates_async()
        assert len(result) == 1

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_list_templates_with_input(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.list_templates.return_value = MockListTemplatesResult(
            [MockTemplateData()]
        )
        mock_control_api_class.return_value = mock_control_api

        input_obj = PageableInput(page_number=1, page_size=5)
        result = Template.list_templates(input=input_obj)
        assert len(result) == 1


# ==================== Template.create_sandbox 测试 ====================


class TestTemplateCreateSandbox:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_sandbox_sync(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.create_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {
                "sandboxId": "sandbox-123",
                "templateName": "test-template",
            },
        }
        mock_data_api_class.return_value = mock_data_api

        template = Template(
            template_name="test-template",
            template_type=TemplateType.CODE_INTERPRETER,
        )
        result = template.create_sandbox()
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_create_sandbox_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.create_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {
                    "sandboxId": "sandbox-123",
                    "templateName": "test-template",
                },
            }
        )
        mock_data_api_class.return_value = mock_data_api

        template = Template(
            template_name="test-template",
            template_type=TemplateType.CODE_INTERPRETER,
        )
        result = await template.create_sandbox_async()
        assert result.sandbox_id == "sandbox-123"

    def test_create_sandbox_no_template_name(self):
        template = Template()
        with pytest.raises(ValueError, match="Template name is required"):
            template.create_sandbox()

    @pytest.mark.asyncio
    async def test_create_sandbox_async_no_template_name(self):
        template = Template()
        with pytest.raises(ValueError, match="Template name is required"):
            await template.create_sandbox_async()

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_sandbox_with_timeout(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.create_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sandbox-456"},
        }
        mock_data_api_class.return_value = mock_data_api

        template = Template(template_name="test-template")
        result = template.create_sandbox(sandbox_idle_timeout_seconds=1200)
        assert result.sandbox_id == "sandbox-456"


# ==================== Template 属性测试 ====================


class TestTemplateProperties:

    def test_template_all_properties(self):
        template = Template(
            template_id="tmpl-123",
            template_name="test-template",
            template_version="v1",
            template_arn="arn:123",
            resource_name="res-123",
            template_type=TemplateType.CODE_INTERPRETER,
            cpu=2.0,
            memory=4096,
            disk_size=512,
            description="Test template",
            execution_role_arn="arn:role",
            sandbox_idle_timeout_in_seconds=1800,
            share_concurrency_limit_per_sandbox=200,
            template_configuration={"key": "value"},
            environment_variables={"ENV": "VAL"},
            allow_anonymous_manage=False,
            created_at="2024-01-01T00:00:00Z",
            last_updated_at="2024-01-02T00:00:00Z",
            status="READY",
            status_reason="OK",
        )
        assert template.template_id == "tmpl-123"
        assert template.template_name == "test-template"
        assert template.template_version == "v1"
        assert template.template_arn == "arn:123"
        assert template.resource_name == "res-123"
        assert template.template_type == TemplateType.CODE_INTERPRETER
        assert template.cpu == 2.0
        assert template.memory == 4096
        assert template.disk_size == 512
        assert template.description == "Test template"
        assert template.execution_role_arn == "arn:role"
        assert template.sandbox_idle_timeout_in_seconds == 1800
        assert template.share_concurrency_limit_per_sandbox == 200
        assert template.template_configuration == {"key": "value"}
        assert template.environment_variables == {"ENV": "VAL"}
        assert template.allow_anonymous_manage is False
        assert template.created_at == "2024-01-01T00:00:00Z"
        assert template.last_updated_at == "2024-01-02T00:00:00Z"
        assert template.status == "READY"
        assert template.status_reason == "OK"

    def test_template_optional_properties(self):
        template = Template()
        assert template.template_id is None
        assert template.template_name is None
        assert template.template_type is None
        assert template.cpu is None
        assert template.memory is None
        assert template.disk_size is None
        assert template.description is None
        assert template.mcp_options is None
        assert template.mcp_state is None
        assert template.oss_configuration is None
        assert template.log_configuration is None
        assert template.credential_configuration is None
        assert template.container_configuration is None
        assert template.network_configuration is None

    def test_template_from_inner_object(self):
        mock_data = MockTemplateData()
        template = Template.from_inner_object(mock_data)
        assert template.template_id == "tmpl-123"
        assert template.template_name == "test-template"
        assert template.template_type == TemplateType.CODE_INTERPRETER
        assert template.status == "READY"
