"""Tests for agentrun.sandbox.api.sandbox_data module."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from agentrun.sandbox.api.sandbox_data import SandboxDataAPI
from agentrun.utils.config import Config
from agentrun.utils.exception import ClientError, ServerError

DATA_ENDPOINT = "https://sandbox-data.example.com"


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


class TestSandboxDataAPIHTTPHandling:

    @staticmethod
    def make_api():
        config = Config(
            account_id="test-account",
            data_endpoint=DATA_ENDPOINT,
            access_key_id="",
            access_key_secret="",
        )
        return SandboxDataAPI(sandbox_id="sb-1", config=config)

    @respx.mock
    def test_sync_success_returns_business_body(self):
        api = self.make_api()
        body = {"executionId": "exec-1", "status": "completed"}
        respx.post(f"{DATA_ENDPOINT}/sandboxes/sb-1/processes/cmd").mock(
            return_value=httpx.Response(200, json=body)
        )

        result = api.post("/processes/cmd", data={"command": "ls"})

        assert result == body

    @respx.mock
    def test_sync_success_code_body_is_not_treated_as_error(self):
        api = self.make_api()
        body = {"code": "SUCCESS", "data": {"sandboxId": "sb-1"}}
        respx.get(f"{DATA_ENDPOINT}/sandboxes/sb-1/health").mock(
            return_value=httpx.Response(200, json=body)
        )

        result = api.get("/health")

        assert result == body

    @respx.mock
    def test_sync_success_non_json_body_raises_client_error(self):
        api = self.make_api()
        respx.get(f"{DATA_ENDPOINT}/sandboxes/sb-1/health").mock(
            return_value=httpx.Response(200, text="<html>ok</html>")
        )

        with pytest.raises(ClientError) as exc_info:
            api.get("/health")

        error = exc_info.value
        assert error.status_code == 200
        assert "Failed to parse JSON response" in error.message
        assert error.response_body == "<html>ok</html>"

    @respx.mock
    def test_sync_client_error_extracts_error_envelope(self):
        api = self.make_api()
        body = {
            "code": "ERR_FORBIDDEN",
            "requestId": "req-body",
            "message": "Signature verification failed",
        }
        respx.get(f"{DATA_ENDPOINT}/sandboxes/sb-1/health").mock(
            return_value=httpx.Response(
                403,
                json=body,
                headers={"x-acs-request-id": "req-header"},
            )
        )

        with pytest.raises(ClientError) as exc_info:
            api.get("/health")

        error = exc_info.value
        assert error.status_code == 403
        assert error.error_code == "ERR_FORBIDDEN"
        assert error.request_id == "req-body"
        assert error.message == "Signature verification failed"
        assert error.response_body == body
        assert error.response_headers is not None

    @respx.mock
    def test_sync_client_error_uses_text_body_and_header_request_id(self):
        api = self.make_api()
        respx.get(f"{DATA_ENDPOINT}/sandboxes/sb-1/missing").mock(
            return_value=httpx.Response(
                404,
                text="sandbox not found",
                headers={"x-request-id": "req-header"},
            )
        )

        with pytest.raises(ClientError) as exc_info:
            api.get("/missing")

        error = exc_info.value
        assert error.status_code == 404
        assert error.error_code is None
        assert error.request_id == "req-header"
        assert error.message == "sandbox not found"
        assert error.response_body == "sandbox not found"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_server_error_extracts_error_envelope(self):
        api = self.make_api()
        body = {
            "code": "ERR_INTERNAL_SERVER",
            "requestId": "req-500",
            "message": "Internal server error",
        }
        respx.get(f"{DATA_ENDPOINT}/sandboxes/sb-1/health").mock(
            return_value=httpx.Response(503, json=body)
        )

        with pytest.raises(ServerError) as exc_info:
            await api.get_async("/health")

        error = exc_info.value
        assert error.status_code == 503
        assert error.error_code == "ERR_INTERNAL_SERVER"
        assert error.request_id == "req-500"
        assert error.message == "Internal server error"
        assert error.response_body == body

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_request_error_still_raises_client_error(self):
        api = self.make_api()
        respx.get(f"{DATA_ENDPOINT}/sandboxes/sb-1/health").mock(
            side_effect=httpx.RequestError("Connection failed")
        )

        with pytest.raises(ClientError) as exc_info:
            await api.get_async("/health")

        assert exc_info.value.status_code == 0
        assert "Connection failed" in exc_info.value.message
