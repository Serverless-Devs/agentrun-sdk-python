"""Tests for agentrun.sandbox.api.sandbox_data module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.sandbox.api.sandbox_data import SandboxDataAPI


@pytest.fixture
def api():
    with patch.object(SandboxDataAPI, "__init__", lambda self, **kw: None):
        obj = SandboxDataAPI.__new__(SandboxDataAPI)
        obj.access_token_map = {}
        obj.access_token = None
        obj.resource_name = ""
        obj.resource_type = None
        obj.namespace = "sandboxes"
        obj.config = MagicMock()
        obj.get = MagicMock(return_value={"status": "ok"})
        obj.get_async = AsyncMock(return_value={"status": "ok"})
        obj.post = MagicMock(return_value={"code": "SUCCESS"})
        obj.post_async = AsyncMock(return_value={"code": "SUCCESS"})
        obj.delete = MagicMock(return_value={"code": "SUCCESS"})
        obj.delete_async = AsyncMock(return_value={"code": "SUCCESS"})
        obj.auth = MagicMock(return_value=("token", {}, None))
        return obj


class TestSandboxDataAPIInit:

    @patch("agentrun.sandbox.api.sandbox_data.DataAPI.__init__")
    def test_init_without_sandbox_id(self, mock_init):
        mock_init.return_value = None
        api = SandboxDataAPI()
        assert api.access_token_map == {}

    @patch("agentrun.sandbox.api.sandbox_data.DataAPI.__init__")
    @patch("agentrun.sandbox.api.sandbox_data.DataAPI.auth")
    def test_init_with_sandbox_id(self, mock_auth, mock_init):
        mock_init.return_value = None
        mock_auth.return_value = None
        api = SandboxDataAPI.__new__(SandboxDataAPI)
        api.config = None
        api.access_token = None
        SandboxDataAPI.__init__(api, sandbox_id="sb-1")
        assert api.resource_name == "sb-1"

    @patch("agentrun.sandbox.api.sandbox_data.DataAPI.__init__")
    @patch("agentrun.sandbox.api.sandbox_data.DataAPI.auth")
    def test_init_with_template_name(self, mock_auth, mock_init):
        mock_init.return_value = None
        mock_auth.return_value = None
        api = SandboxDataAPI.__new__(SandboxDataAPI)
        api.config = None
        api.access_token = None
        SandboxDataAPI.__init__(api, template_name="tpl-1")
        assert api.resource_name == "tpl-1"


class TestSandboxDataAPIRefreshToken:

    @patch("agentrun.sandbox.api.sandbox_data.DataAPI.__init__")
    @patch("agentrun.sandbox.api.sandbox_data.DataAPI.auth")
    def test_refresh_with_cached_token(self, mock_auth, mock_init):
        mock_init.return_value = None
        api = SandboxDataAPI()
        api.access_token_map = {"sb-1": "cached-token"}
        api.config = MagicMock()
        api._SandboxDataAPI__refresh_access_token(sandbox_id="sb-1")
        assert api.access_token == "cached-token"

    @patch("agentrun.sandbox.api.sandbox_data.DataAPI.__init__")
    @patch("agentrun.sandbox.api.sandbox_data.DataAPI.auth")
    def test_refresh_template_name(self, mock_auth, mock_init):
        mock_init.return_value = None
        mock_auth.return_value = None
        api = SandboxDataAPI()
        api.access_token_map = {}
        api.access_token = None
        api.config = MagicMock()
        api._SandboxDataAPI__refresh_access_token(template_name="tpl-1")
        assert api.resource_name == "tpl-1"
        assert api.namespace == "sandboxes"


class TestSandboxDataAPIHealthCheck:

    def test_check_health(self, api):
        result = api.check_health()
        api.get.assert_called_once_with("/health")
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_check_health_async(self, api):
        result = await api.check_health_async()
        api.get_async.assert_called_once_with("/health")
        assert result == {"status": "ok"}


class TestSandboxDataAPICreateSandbox:

    def test_create_sandbox_minimal(self, api):
        api._SandboxDataAPI__refresh_access_token = MagicMock()
        result = api.create_sandbox("tpl-1")
        api.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_sandbox_async_minimal(self, api):
        api._SandboxDataAPI__refresh_access_token = MagicMock()
        result = await api.create_sandbox_async("tpl-1")
        api.post_async.assert_called_once()

    def test_create_sandbox_with_all_options(self, api):
        api._SandboxDataAPI__refresh_access_token = MagicMock()
        api.create_sandbox(
            "tpl-1",
            sandbox_idle_timeout_seconds=1200,
            sandbox_id="sb-custom",
            nas_config={"groupId": 1000},
            oss_mount_config={"buckets": []},
            polar_fs_config={"userId": 1000},
        )
        call_data = api.post.call_args
        data = (
            call_data[1].get("data") or call_data[0][1]
            if len(call_data[0]) > 1
            else call_data[1]["data"]
        )
        assert "sandboxId" in data
        assert "nasConfig" in data
        assert "ossMountConfig" in data
        assert "polarFsConfig" in data

    @pytest.mark.asyncio
    async def test_create_sandbox_async_with_all_options(self, api):
        api._SandboxDataAPI__refresh_access_token = MagicMock()
        await api.create_sandbox_async(
            "tpl-1",
            sandbox_id="sb-custom",
            nas_config={"groupId": 1000},
            oss_mount_config={"buckets": []},
            polar_fs_config={"userId": 1000},
        )
        api.post_async.assert_called_once()


class TestSandboxDataAPICRUD:

    def test_delete_sandbox(self, api):
        api._SandboxDataAPI__refresh_access_token = MagicMock()
        api.delete_sandbox("sb-1")
        api.delete.assert_called_once_with("/", config=None)

    @pytest.mark.asyncio
    async def test_delete_sandbox_async(self, api):
        api._SandboxDataAPI__refresh_access_token = MagicMock()
        await api.delete_sandbox_async("sb-1")
        api.delete_async.assert_called_once_with("/", config=None)

    def test_stop_sandbox(self, api):
        api._SandboxDataAPI__refresh_access_token = MagicMock()
        api.stop_sandbox("sb-1")
        api.post.assert_called_once_with("/stop", config=None)

    @pytest.mark.asyncio
    async def test_stop_sandbox_async(self, api):
        api._SandboxDataAPI__refresh_access_token = MagicMock()
        await api.stop_sandbox_async("sb-1")
        api.post_async.assert_called_once_with("/stop", config=None)

    def test_get_sandbox(self, api):
        api._SandboxDataAPI__refresh_access_token = MagicMock()
        api.get_sandbox("sb-1")
        api.get.assert_called_once_with("/", config=None)

    @pytest.mark.asyncio
    async def test_get_sandbox_async(self, api):
        api._SandboxDataAPI__refresh_access_token = MagicMock()
        await api.get_sandbox_async("sb-1")
        api.get_async.assert_called_once_with("/", config=None)
