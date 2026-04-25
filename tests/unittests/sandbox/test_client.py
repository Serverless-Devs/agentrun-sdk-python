"""测试 agentrun.sandbox.client 模块 / Test agentrun.sandbox.client module"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.sandbox.client import SandboxClient
from agentrun.sandbox.model import (
    ListSandboxesInput,
    NASConfig,
    OSSMountConfig,
    PageableInput,
    PolarFsConfig,
    TemplateInput,
    TemplateType,
)
from agentrun.utils.config import Config
from agentrun.utils.exception import (
    AgentRunError,
    ClientError,
    ResourceNotExistError,
)


class MockTemplateData:
    """模拟 Template 数据"""

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


class MockTemplateCreatingData:
    """模拟正在创建中的 Template 数据"""

    def to_map(self):
        return {
            "templateId": "tmpl-123",
            "templateName": "test-template",
            "templateType": "CodeInterpreter",
            "status": "CREATING",
        }


class MockTemplateFailedData:
    """模拟创建失败的 Template 数据"""

    def to_map(self):
        return {
            "templateId": "tmpl-123",
            "templateName": "test-template",
            "templateType": "CodeInterpreter",
            "status": "CREATE_FAILED",
        }


class MockListTemplatesResult:
    """模拟 Template 列表结果"""

    def __init__(self, items):
        self.items = items


class MockListSandboxesResult:
    """模拟 Sandbox 列表结果"""

    def __init__(self, items, next_token=None):
        self.items = items
        self.next_token = next_token


class MockSandboxData:
    """模拟底层 SDK Sandbox 对象"""

    def to_map(self):
        return {
            "sandboxId": "sandbox-123",
            "templateName": "test-template",
            "templateId": "tmpl-123",
            "status": "RUNNING",
            "createdAt": "2024-01-01T00:00:00Z",
        }


# ==================== 初始化测试 ====================


class TestSandboxClientInit:

    def test_init_without_config(self):
        client = SandboxClient()
        assert client is not None

    def test_init_with_config(self):
        config = Config(access_key_id="test-ak")
        client = SandboxClient(config=config)
        assert client is not None


# ==================== Template CRUD 测试 ====================


class TestSandboxClientCreateTemplate:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_template_sync(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.create_template.return_value = MockTemplateData()
        mock_control_api.get_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-template",
        )
        result = client.create_template(input_obj)
        assert result.template_name == "test-template"
        assert result.status == "READY"
        assert mock_control_api.create_template.called

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

        client = SandboxClient()
        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-template",
        )
        result = await client.create_template_async(input_obj)
        assert result.template_name == "test-template"


class TestSandboxClientDeleteTemplate:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_delete_template_sync(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.delete_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        result = client.delete_template("test-template")
        assert result is not None
        assert mock_control_api.delete_template.called

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

        client = SandboxClient()
        result = await client.delete_template_async("test-template")
        assert result is not None

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_delete_template_not_exist(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.delete_template.side_effect = ClientError(
            status_code=404, message="Not found"
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        with pytest.raises(ResourceNotExistError):
            client.delete_template("nonexistent")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_delete_template_async_not_exist(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.delete_template_async = AsyncMock(
            side_effect=ClientError(status_code=404, message="Not found")
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        with pytest.raises(ResourceNotExistError):
            await client.delete_template_async("nonexistent")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_delete_template_other_error(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.delete_template.side_effect = ClientError(
            status_code=500, message="Internal error"
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        with pytest.raises(ClientError):
            client.delete_template("test")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_delete_template_async_other_error(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.delete_template_async = AsyncMock(
            side_effect=ClientError(status_code=500, message="Internal error")
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        with pytest.raises(ClientError):
            await client.delete_template_async("test")


class TestSandboxClientUpdateTemplate:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_update_template_sync(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.update_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-template",
        )
        result = client.update_template("test-template", input_obj)
        assert result is not None
        assert mock_control_api.update_template.called

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

        client = SandboxClient()
        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test-template",
        )
        result = await client.update_template_async("test-template", input_obj)
        assert result is not None

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_update_template_not_exist(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.update_template.side_effect = ClientError(
            status_code=404, message="Not found"
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        input_obj = TemplateInput(template_type=TemplateType.CODE_INTERPRETER)
        with pytest.raises(ResourceNotExistError):
            client.update_template("nonexistent", input_obj)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_update_template_async_not_exist(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.update_template_async = AsyncMock(
            side_effect=ClientError(status_code=404, message="Not found")
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        input_obj = TemplateInput(template_type=TemplateType.CODE_INTERPRETER)
        with pytest.raises(ResourceNotExistError):
            await client.update_template_async("nonexistent", input_obj)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_update_template_other_error(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.update_template.side_effect = ClientError(
            status_code=500, message="Internal error"
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        input_obj = TemplateInput(template_type=TemplateType.CODE_INTERPRETER)
        with pytest.raises(ClientError):
            client.update_template("test", input_obj)

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_update_template_async_other_error(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.update_template_async = AsyncMock(
            side_effect=ClientError(status_code=500, message="Internal error")
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        input_obj = TemplateInput(template_type=TemplateType.CODE_INTERPRETER)
        with pytest.raises(ClientError):
            await client.update_template_async("test", input_obj)


class TestSandboxClientGetTemplate:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_get_template_sync(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        result = client.get_template("test-template")
        assert result.template_name == "test-template"
        assert mock_control_api.get_template.called

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

        client = SandboxClient()
        result = await client.get_template_async("test-template")
        assert result.template_name == "test-template"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_get_template_not_exist(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template.side_effect = ClientError(
            status_code=404, message="Not found"
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        with pytest.raises(ResourceNotExistError):
            client.get_template("nonexistent")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_get_template_async_not_exist(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template_async = AsyncMock(
            side_effect=ClientError(status_code=404, message="Not found")
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        with pytest.raises(ResourceNotExistError):
            await client.get_template_async("nonexistent")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_get_template_other_error(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template.side_effect = ClientError(
            status_code=500, message="Internal error"
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        with pytest.raises(ClientError):
            client.get_template("test")


class TestSandboxClientListTemplates:

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

        client = SandboxClient()
        result = client.list_templates()
        assert len(result) == 1
        assert mock_control_api.list_templates.called

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

        client = SandboxClient()
        result = await client.list_templates_async()
        assert len(result) == 1

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_list_templates_empty(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.list_templates.return_value = MockListTemplatesResult(
            None
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        result = client.list_templates()
        assert len(result) == 0

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_list_templates_async_empty(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.list_templates_async = AsyncMock(
            return_value=MockListTemplatesResult(None)
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        result = await client.list_templates_async()
        assert len(result) == 0

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

        client = SandboxClient()
        input_obj = PageableInput(page_number=1, page_size=5)
        result = client.list_templates(input=input_obj)
        assert len(result) == 1

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_list_templates_none_input(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.list_templates.return_value = MockListTemplatesResult(
            [MockTemplateData()]
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        result = client.list_templates(input=None)
        assert len(result) == 1


# ==================== Sandbox CRUD 测试 ====================


class TestSandboxClientCreateSandbox:

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
                "status": "RUNNING",
            },
        }
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        result = client.create_sandbox(template_name="test-template")
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
                    "status": "RUNNING",
                },
            }
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        result = await client.create_sandbox_async(
            template_name="test-template"
        )
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_sandbox_with_storage_configs(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.create_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sandbox-123"},
        }
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        result = client.create_sandbox(
            template_name="test-template",
            sandbox_id="custom-id",
            nas_config=NASConfig(group_id=1000),
            oss_mount_config=OSSMountConfig(),
            polar_fs_config=PolarFsConfig(user_id=1000),
        )
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_create_sandbox_failure(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.create_sandbox.return_value = {
            "code": "FAILED",
            "message": "Something went wrong",
        }
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError, match="Failed to create sandbox"):
            client.create_sandbox(template_name="test-template")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_create_sandbox_async_failure(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.create_sandbox_async = AsyncMock(
            return_value={
                "code": "FAILED",
                "message": "Something went wrong",
            }
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError, match="Failed to create sandbox"):
            await client.create_sandbox_async(template_name="test-template")


class TestSandboxClientStopSandbox:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_stop_sandbox_sync(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.stop_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sandbox-123", "status": "STOPPED"},
        }
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        result = client.stop_sandbox("sandbox-123")
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_stop_sandbox_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.stop_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sandbox-123", "status": "STOPPED"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        result = await client.stop_sandbox_async("sandbox-123")
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_stop_sandbox_failure(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.stop_sandbox.return_value = {
            "code": "FAILED",
            "message": "Stop failed",
        }
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError, match="Failed to stop sandbox"):
            client.stop_sandbox("sandbox-123")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_stop_sandbox_not_exist(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.stop_sandbox.side_effect = ClientError(
            status_code=404, message="Not found"
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ResourceNotExistError):
            client.stop_sandbox("nonexistent")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_stop_sandbox_async_not_exist(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.stop_sandbox_async = AsyncMock(
            side_effect=ClientError(status_code=404, message="Not found")
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ResourceNotExistError):
            await client.stop_sandbox_async("nonexistent")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_stop_sandbox_other_error(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.stop_sandbox.side_effect = ClientError(
            status_code=500, message="Internal error"
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError):
            client.stop_sandbox("sandbox-123")


class TestSandboxClientDeleteSandbox:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_delete_sandbox_sync(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.delete_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sandbox-123"},
        }
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        result = client.delete_sandbox("sandbox-123")
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_delete_sandbox_async(
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

        client = SandboxClient()
        result = await client.delete_sandbox_async("sandbox-123")
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_delete_sandbox_failure(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.delete_sandbox.return_value = {
            "code": "FAILED",
            "message": "Delete failed",
        }
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError, match="Failed to stop sandbox"):
            client.delete_sandbox("sandbox-123")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_delete_sandbox_not_exist(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.delete_sandbox.side_effect = ClientError(
            status_code=404, message="Not found"
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ResourceNotExistError):
            client.delete_sandbox("nonexistent")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_delete_sandbox_not_found_in_response_is_idempotent(
        self, mock_data_api_class, mock_control_api_class
    ):
        """数据面返回 not found 时，delete_sandbox 应幂等成功

        When the data plane returns a non-SUCCESS response whose message
        contains "not found", the SDK should treat the delete as a success
        rather than raising an error.  This handles the case where the
        control-plane list API still shows a TERMINATED sandbox, but the
        data plane has already removed it.
        """
        mock_data_api = MagicMock()
        mock_data_api.delete_sandbox.return_value = {
            "code": "FAILED",
            "message": "sandbox not found",
        }
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        result = client.delete_sandbox("sandbox-123")
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_delete_sandbox_async_not_found_in_response_is_idempotent(
        self, mock_data_api_class, mock_control_api_class
    ):
        """数据面返回 not found 时，delete_sandbox_async 应幂等成功"""
        mock_data_api = MagicMock()
        mock_data_api.delete_sandbox_async = AsyncMock(
            return_value={
                "code": "FAILED",
                "message": "sandbox not found",
            }
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        result = await client.delete_sandbox_async("sandbox-123")
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_delete_sandbox_async_not_exist(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.delete_sandbox_async = AsyncMock(
            side_effect=ClientError(status_code=404, message="Not found")
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ResourceNotExistError):
            await client.delete_sandbox_async("nonexistent")


class TestSandboxClientGetSandbox:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_get_sandbox_sync(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox.return_value = {
            "code": "SUCCESS",
            "data": {"sandboxId": "sandbox-123", "templateName": "tmpl"},
        }
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        result = client.get_sandbox("sandbox-123")
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_get_sandbox_async(
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

        client = SandboxClient()
        result = await client.get_sandbox_async("sandbox-123")
        assert result.sandbox_id == "sandbox-123"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_get_sandbox_failure(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox.return_value = {
            "code": "FAILED",
            "message": "Get failed",
        }
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError, match="Failed to get sandbox"):
            client.get_sandbox("sandbox-123")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_get_sandbox_not_exist(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox.side_effect = ClientError(
            status_code=404, message="Not found"
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ResourceNotExistError):
            client.get_sandbox("nonexistent")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_get_sandbox_async_not_exist(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox_async = AsyncMock(
            side_effect=ClientError(status_code=404, message="Not found")
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ResourceNotExistError):
            await client.get_sandbox_async("nonexistent")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_get_sandbox_async_failure(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox_async = AsyncMock(
            return_value={"code": "FAILED", "message": "err"}
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError, match="Failed to get sandbox"):
            await client.get_sandbox_async("sandbox-123")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_get_sandbox_other_client_error(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox.side_effect = ClientError(
            status_code=500, message="Internal"
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError):
            client.get_sandbox("sandbox-123")


class TestSandboxClientListSandboxes:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_list_sandboxes_sync(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.list_sandboxes.return_value = MockListSandboxesResult(
            [MockSandboxData()]
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        result = client.list_sandboxes()
        assert len(result.sandboxes) == 1

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_list_sandboxes_async(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.list_sandboxes_async = AsyncMock(
            return_value=MockListSandboxesResult([MockSandboxData()])
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        result = await client.list_sandboxes_async()
        assert len(result.sandboxes) == 1

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_list_sandboxes_with_input(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.list_sandboxes.return_value = MockListSandboxesResult(
            [], next_token="token"
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        input_obj = ListSandboxesInput(max_results=5)
        result = client.list_sandboxes(input=input_obj)
        assert len(result.sandboxes) == 0
        assert result.next_token == "token"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_list_sandboxes_none_input(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.list_sandboxes.return_value = MockListSandboxesResult(
            []
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        result = client.list_sandboxes(input=None)
        assert len(result.sandboxes) == 0


# ==================== _wait_template_ready 测试 ====================


class TestWaitTemplateReady:

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_wait_ready_immediately(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockTemplateData()
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        result = client._wait_template_ready("test-template")
        assert result.status == "READY"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_wait_ready_fails(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockTemplateFailedData()
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        with pytest.raises(AgentRunError, match="creation failed"):
            client._wait_template_ready("test-template")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_wait_ready_timeout(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template.return_value = MockTemplateCreatingData()
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        with pytest.raises(TimeoutError, match="Timeout"):
            client._wait_template_ready(
                "test-template",
                interval_seconds=0.01,
                timeout_seconds=0.05,
            )

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_wait_ready_async_immediately(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template_async = AsyncMock(
            return_value=MockTemplateData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        result = await client._wait_template_ready_async("test-template")
        assert result.status == "READY"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_wait_ready_async_fails(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template_async = AsyncMock(
            return_value=MockTemplateFailedData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        with pytest.raises(AgentRunError, match="creation failed"):
            await client._wait_template_ready_async("test-template")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_wait_ready_async_timeout(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        mock_control_api.get_template_async = AsyncMock(
            return_value=MockTemplateCreatingData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        with pytest.raises(TimeoutError, match="Timeout"):
            await client._wait_template_ready_async(
                "test-template",
                interval_seconds=0.01,
                timeout_seconds=0.05,
            )

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_stop_sandbox_async_failure(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.stop_sandbox_async = AsyncMock(
            return_value={"code": "FAILED", "message": "err"}
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError, match="Failed to stop sandbox"):
            await client.stop_sandbox_async("sandbox-123")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_delete_sandbox_async_failure(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.delete_sandbox_async = AsyncMock(
            return_value={"code": "FAILED", "message": "err"}
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError, match="Failed to stop sandbox"):
            await client.delete_sandbox_async("sandbox-123")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_delete_sandbox_other_client_error(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.delete_sandbox.side_effect = ClientError(
            status_code=500, message="Internal"
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError):
            client.delete_sandbox("sandbox-123")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_delete_sandbox_async_other_client_error(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.delete_sandbox_async = AsyncMock(
            side_effect=ClientError(status_code=500, message="Internal")
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError):
            await client.delete_sandbox_async("sandbox-123")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_stop_sandbox_async_other_error(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.stop_sandbox_async = AsyncMock(
            side_effect=ClientError(status_code=500, message="Internal")
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError):
            await client.stop_sandbox_async("sandbox-123")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_get_sandbox_async_other_client_error(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.get_sandbox_async = AsyncMock(
            side_effect=ClientError(status_code=500, message="Internal")
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        with pytest.raises(ClientError):
            await client.get_sandbox_async("sandbox-123")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_create_sandbox_async_with_storage_configs(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_data_api = MagicMock()
        mock_data_api.create_sandbox_async = AsyncMock(
            return_value={
                "code": "SUCCESS",
                "data": {"sandboxId": "sandbox-456"},
            }
        )
        mock_data_api_class.return_value = mock_data_api

        client = SandboxClient()
        result = await client.create_sandbox_async(
            template_name="test-template",
            sandbox_id="custom-id",
            nas_config=NASConfig(group_id=1000),
            oss_mount_config=OSSMountConfig(),
            polar_fs_config=PolarFsConfig(user_id=1000),
        )
        assert result.sandbox_id == "sandbox-456"

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    def test_wait_update_failed(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        failed_data = MagicMock()
        failed_data.to_map.return_value = {
            "templateId": "tmpl-123",
            "templateName": "test-template",
            "status": "UPDATE_FAILED",
        }
        mock_control_api.get_template.return_value = failed_data
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        with pytest.raises(AgentRunError, match="creation failed"):
            client._wait_template_ready("test-template")

    @patch("agentrun.sandbox.client.SandboxControlAPI")
    @patch("agentrun.sandbox.client.SandboxDataAPI")
    @pytest.mark.asyncio
    async def test_wait_async_update_failed(
        self, mock_data_api_class, mock_control_api_class
    ):
        mock_control_api = MagicMock()
        failed_data = MagicMock()
        failed_data.to_map.return_value = {
            "templateId": "tmpl-123",
            "templateName": "test-template",
            "status": "UPDATE_FAILED",
        }
        mock_control_api.get_template_async = AsyncMock(
            return_value=failed_data
        )
        mock_control_api_class.return_value = mock_control_api

        client = SandboxClient()
        with pytest.raises(AgentRunError, match="creation failed"):
            await client._wait_template_ready_async("test-template")
