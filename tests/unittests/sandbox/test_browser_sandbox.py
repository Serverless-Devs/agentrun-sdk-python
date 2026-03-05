"""Tests for agentrun.sandbox.browser_sandbox module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.sandbox.browser_sandbox import BrowserSandbox
from agentrun.sandbox.model import TemplateType


def _make_sandbox(sandbox_id="sb-br-1"):
    sb = BrowserSandbox.model_construct(sandbox_id=sandbox_id)
    sb._data_api = MagicMock()
    return sb


class TestBrowserSandbox:

    def test_template_type(self):
        assert (
            BrowserSandbox.__private_attributes__["_template_type"].default
            == TemplateType.BROWSER
        )

    def test_data_api_lazy_init(self):
        sb = BrowserSandbox.model_construct(sandbox_id="sb-1")
        with patch(
            "agentrun.sandbox.browser_sandbox.BrowserDataAPI"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            api = sb.data_api
            assert api is not None
            assert sb.data_api is api

    def test_data_api_no_sandbox_id_raises(self):
        sb = BrowserSandbox.model_construct(sandbox_id=None)
        sb._data_api = None
        with pytest.raises(ValueError, match="Sandbox ID is not set"):
            _ = sb.data_api

    def test_check_health(self):
        sb = _make_sandbox()
        sb.data_api.check_health.return_value = {"status": "ok"}
        assert sb.check_health() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_check_health_async(self):
        sb = _make_sandbox()
        sb.data_api.check_health_async = AsyncMock(
            return_value={"status": "ok"}
        )
        assert await sb.check_health_async() == {"status": "ok"}

    def test_get_cdp_url(self):
        sb = _make_sandbox()
        sb.data_api.get_cdp_url.return_value = "ws://example.com/ws/automation"
        assert sb.get_cdp_url(record=True) == "ws://example.com/ws/automation"
        sb.data_api.get_cdp_url.assert_called_once_with(record=True)

    def test_get_vnc_url(self):
        sb = _make_sandbox()
        sb.data_api.get_vnc_url.return_value = "ws://example.com/ws/liveview"
        assert sb.get_vnc_url() == "ws://example.com/ws/liveview"

    def test_sync_playwright(self):
        sb = _make_sandbox()
        sb.data_api.sync_playwright.return_value = MagicMock()
        result = sb.sync_playwright(record=True)
        assert result is not None

    def test_async_playwright(self):
        sb = _make_sandbox()
        sb.data_api.async_playwright.return_value = MagicMock()
        result = sb.async_playwright()
        assert result is not None

    @pytest.mark.asyncio
    async def test_list_recordings_async(self):
        sb = _make_sandbox()
        sb.data_api.list_recordings_async = AsyncMock(
            return_value=[{"name": "r1"}]
        )
        assert await sb.list_recordings_async() == [{"name": "r1"}]

    def test_list_recordings(self):
        sb = _make_sandbox()
        sb.data_api.list_recordings.return_value = [{"name": "r1"}]
        assert sb.list_recordings() == [{"name": "r1"}]

    @pytest.mark.asyncio
    async def test_download_recording_async(self):
        sb = _make_sandbox()
        sb.data_api.download_recording_async = AsyncMock(
            return_value={"saved_path": "/x.mkv", "size": 1024}
        )
        result = await sb.download_recording_async("r1.mkv", "/x.mkv")
        assert result["size"] == 1024

    def test_download_recording(self):
        sb = _make_sandbox()
        sb.data_api.download_recording.return_value = {
            "saved_path": "/x.mkv",
            "size": 1024,
        }
        result = sb.download_recording("r1.mkv", "/x.mkv")
        assert result["size"] == 1024

    @pytest.mark.asyncio
    async def test_delete_recording_async(self):
        sb = _make_sandbox()
        sb.data_api.delete_recording_async = AsyncMock(
            return_value={"ok": True}
        )
        assert await sb.delete_recording_async("r1.mkv") == {"ok": True}

    def test_delete_recording(self):
        sb = _make_sandbox()
        sb.data_api.delete_recording.return_value = {"ok": True}
        assert sb.delete_recording("r1.mkv") == {"ok": True}


class TestBrowserSandboxContextManager:

    def test_enter_health_ok(self):
        sb = _make_sandbox()
        sb.data_api.check_health.return_value = {"status": "ok"}
        result = sb.__enter__()
        assert result is sb

    def test_enter_retries_then_ok(self):
        sb = _make_sandbox()
        sb.data_api.check_health.side_effect = [
            {"status": "not-ready"},
            {"status": "ok"},
        ]
        with patch("agentrun.sandbox.browser_sandbox.time.sleep"):
            result = sb.__enter__()
            assert result is sb

    def test_enter_exception_retries(self):
        sb = _make_sandbox()
        sb.data_api.check_health.side_effect = [
            Exception("network"),
            {"status": "ok"},
        ]
        with patch("agentrun.sandbox.browser_sandbox.time.sleep"):
            result = sb.__enter__()
            assert result is sb

    def test_enter_timeout(self):
        sb = _make_sandbox()
        sb.data_api.check_health.return_value = {"status": "not-ready"}
        with patch("agentrun.sandbox.browser_sandbox.time.sleep"):
            with pytest.raises(RuntimeError, match="Health check timeout"):
                sb.__enter__()

    @pytest.mark.asyncio
    async def test_aenter_health_ok(self):
        sb = _make_sandbox()
        sb.data_api.check_health_async = AsyncMock(
            return_value={"status": "ok"}
        )
        result = await sb.__aenter__()
        assert result is sb

    @pytest.mark.asyncio
    async def test_aenter_retries_then_ok(self):
        sb = _make_sandbox()
        sb.data_api.check_health_async = AsyncMock(
            side_effect=[{"status": "not-ready"}, {"status": "ok"}]
        )
        with patch(
            "agentrun.sandbox.browser_sandbox.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await sb.__aenter__()
            assert result is sb

    @pytest.mark.asyncio
    async def test_aenter_exception_retries(self):
        sb = _make_sandbox()
        sb.data_api.check_health_async = AsyncMock(
            side_effect=[Exception("err"), {"status": "ok"}]
        )
        with patch(
            "agentrun.sandbox.browser_sandbox.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await sb.__aenter__()
            assert result is sb

    @pytest.mark.asyncio
    async def test_aenter_timeout(self):
        sb = _make_sandbox()
        sb.data_api.check_health_async = AsyncMock(
            return_value={"status": "not-ready"}
        )
        with patch(
            "agentrun.sandbox.browser_sandbox.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            with pytest.raises(RuntimeError, match="Health check timeout"):
                await sb.__aenter__()

    def test_exit_calls_delete(self):
        sb = _make_sandbox()
        sb.delete = MagicMock()
        sb.__exit__(None, None, None)
        sb.delete.assert_called_once()

    def test_exit_no_sandbox_id_raises(self):
        sb = BrowserSandbox.model_construct(sandbox_id=None)
        with pytest.raises(ValueError, match="Sandbox ID is not set"):
            sb.__exit__(None, None, None)

    @pytest.mark.asyncio
    async def test_aexit_calls_delete(self):
        sb = _make_sandbox()
        sb.delete_async = AsyncMock()
        await sb.__aexit__(None, None, None)
        sb.delete_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexit_no_sandbox_id_raises(self):
        sb = BrowserSandbox.model_construct(sandbox_id=None)
        with pytest.raises(ValueError, match="Sandbox ID is not set"):
            await sb.__aexit__(None, None, None)
