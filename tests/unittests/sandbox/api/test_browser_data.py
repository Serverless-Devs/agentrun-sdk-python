"""Tests for agentrun.sandbox.api.browser_data module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.sandbox.api.browser_data import BrowserDataAPI


@pytest.fixture
def api():
    with patch.object(BrowserDataAPI, "__init__", lambda self, **kw: None):
        obj = BrowserDataAPI.__new__(BrowserDataAPI)
        obj.sandbox_id = "sb-1"
        obj.config = MagicMock()
        obj.config.get_account_id.return_value = "account123"
        obj.access_token = "tok"
        obj.access_token_map = {}
        obj.resource_name = "sb-1"
        obj.with_path = MagicMock(
            side_effect=lambda p: f"http://host.com/ns{p}?sig=abc"
        )
        obj.auth = MagicMock(
            return_value=("tok", {"Authorization": "Bearer tok"}, None)
        )
        obj.get = MagicMock(return_value=[])
        obj.get_async = AsyncMock(return_value=[])
        obj.delete = MagicMock(return_value={"ok": True})
        obj.delete_async = AsyncMock(return_value={"ok": True})
        obj.get_video = MagicMock(
            return_value={"saved_path": "/x.mkv", "size": 1024}
        )
        obj.get_video_async = AsyncMock(
            return_value={"saved_path": "/x.mkv", "size": 1024}
        )
        return obj


class TestBrowserDataAPIInit:

    @patch("agentrun.sandbox.api.browser_data.SandboxDataAPI.__init__")
    def test_init(self, mock_super_init):
        mock_super_init.return_value = None
        api = BrowserDataAPI(sandbox_id="sb-1")
        assert api.sandbox_id == "sb-1"
        mock_super_init.assert_called_once_with(sandbox_id="sb-1", config=None)


class TestCdpUrl:

    def test_get_cdp_url_no_record(self, api):
        url = api.get_cdp_url()
        api.with_path.assert_called_once_with("/ws/automation")
        assert "ws://" in url
        assert "tenantId=account123" in url
        assert "recording" not in url

    def test_get_cdp_url_with_record(self, api):
        url = api.get_cdp_url(record=True)
        assert "recording=true" in url
        assert "tenantId=account123" in url


class TestVncUrl:

    def test_get_vnc_url_no_record(self, api):
        url = api.get_vnc_url()
        api.with_path.assert_called_once_with("/ws/liveview")
        assert "ws://" in url
        assert "tenantId=account123" in url

    def test_get_vnc_url_with_record(self, api):
        url = api.get_vnc_url(record=True)
        assert "recording=true" in url


class TestPlaywright:

    def test_sync_playwright(self, api):
        with patch(
            "agentrun.sandbox.api.playwright_sync.BrowserPlaywrightSync"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            result = api.sync_playwright(record=True)
            assert result is not None
            mock_cls.assert_called_once()

    def test_async_playwright(self, api):
        with patch(
            "agentrun.sandbox.api.playwright_async.BrowserPlaywrightAsync"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            result = api.async_playwright(record=True)
            assert result is not None
            mock_cls.assert_called_once()


class TestRecordings:

    def test_list_recordings(self, api):
        api.list_recordings()
        api.get.assert_called_once_with("/recordings")

    @pytest.mark.asyncio
    async def test_list_recordings_async(self, api):
        await api.list_recordings_async()
        api.get_async.assert_called_once_with("/recordings")

    def test_delete_recording(self, api):
        api.delete_recording("file.mkv")
        api.delete.assert_called_once_with("/recordings/file.mkv")

    @pytest.mark.asyncio
    async def test_delete_recording_async(self, api):
        await api.delete_recording_async("file.mkv")
        api.delete_async.assert_called_once_with("/recordings/file.mkv")

    def test_download_recording(self, api):
        result = api.download_recording("file.mkv", "/local/file.mkv")
        api.get_video.assert_called_once_with(
            "/recordings/file.mkv", save_path="/local/file.mkv"
        )
        assert result["size"] == 1024

    @pytest.mark.asyncio
    async def test_download_recording_async(self, api):
        result = await api.download_recording_async(
            "file.mkv", "/local/file.mkv"
        )
        api.get_video_async.assert_called_once_with(
            "/recordings/file.mkv", save_path="/local/file.mkv"
        )
        assert result["size"] == 1024
