"""Tests for agentrun.sandbox.api.aio_data module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.sandbox.api.aio_data import AioDataAPI
from agentrun.sandbox.model import CodeLanguage


@pytest.fixture
def api():
    with patch.object(AioDataAPI, "__init__", lambda self, **kw: None):
        obj = AioDataAPI.__new__(AioDataAPI)
        obj.sandbox_id = "sb-aio-1"
        obj.config = MagicMock()
        obj.config.get_account_id.return_value = "account123"
        obj.access_token = "tok"
        obj.access_token_map = {}
        obj.resource_name = "sb-aio-1"
        obj.with_path = MagicMock(
            side_effect=lambda p: f"http://host.com/ns{p}?sig=abc"
        )
        obj.auth = MagicMock(
            return_value=("tok", {"Authorization": "Bearer tok"}, None)
        )
        obj.get = MagicMock(return_value={"ok": True})
        obj.get_async = AsyncMock(return_value={"ok": True})
        obj.post = MagicMock(return_value={"ok": True})
        obj.post_async = AsyncMock(return_value={"ok": True})
        obj.delete = MagicMock(return_value={"ok": True})
        obj.delete_async = AsyncMock(return_value={"ok": True})
        obj.post_file = MagicMock(return_value={"ok": True})
        obj.post_file_async = AsyncMock(return_value={"ok": True})
        obj.get_file = MagicMock(return_value={"saved_path": "/x", "size": 10})
        obj.get_file_async = AsyncMock(
            return_value={"saved_path": "/x", "size": 10}
        )
        obj.get_video = MagicMock(
            return_value={"saved_path": "/x.mkv", "size": 1024}
        )
        obj.get_video_async = AsyncMock(
            return_value={"saved_path": "/x.mkv", "size": 1024}
        )
        return obj


class TestAioDataAPIInit:

    @patch("agentrun.sandbox.api.aio_data.SandboxDataAPI.__init__")
    def test_init(self, mock_super_init):
        mock_super_init.return_value = None
        api = AioDataAPI(sandbox_id="sb-1")
        assert api.sandbox_id == "sb-1"
        mock_super_init.assert_called_once_with(sandbox_id="sb-1", config=None)


# ==================== Browser API ====================


class TestAioCdpUrl:

    def test_get_cdp_url_no_record(self, api):
        url = api.get_cdp_url()
        api.with_path.assert_called_once_with("/ws/automation")
        assert "ws://" in url
        assert "tenantId=account123" in url
        assert "recording" not in url

    def test_get_cdp_url_with_record(self, api):
        url = api.get_cdp_url(record=True)
        assert "recording=true" in url


class TestAioVncUrl:

    def test_get_vnc_url_no_record(self, api):
        url = api.get_vnc_url()
        api.with_path.assert_called_once_with("/ws/liveview")
        assert "ws://" in url
        assert "tenantId=account123" in url

    def test_get_vnc_url_with_record(self, api):
        url = api.get_vnc_url(record=True)
        assert "recording=true" in url


class TestAioPlaywright:

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


class TestAioRecordings:

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


# ==================== Code Interpreter API ====================


class TestAioListDirectory:

    def test_list_directory_no_params(self, api):
        api.list_directory()
        api.get.assert_called_once_with("/filesystem", query={})

    def test_list_directory_with_path(self, api):
        api.list_directory(path="/home")
        api.get.assert_called_once_with("/filesystem", query={"path": "/home"})

    def test_list_directory_with_depth(self, api):
        api.list_directory(depth=2)
        api.get.assert_called_once_with("/filesystem", query={"depth": 2})

    def test_list_directory_with_path_and_depth(self, api):
        api.list_directory(path="/home", depth=3)
        api.get.assert_called_once_with(
            "/filesystem", query={"path": "/home", "depth": 3}
        )

    @pytest.mark.asyncio
    async def test_list_directory_async(self, api):
        await api.list_directory_async(path="/tmp", depth=1)
        api.get_async.assert_called_once_with(
            "/filesystem", query={"path": "/tmp", "depth": 1}
        )


class TestAioStat:

    def test_stat(self, api):
        api.stat("/tmp/f.txt")
        api.get.assert_called_once_with(
            "/filesystem/stat", query={"path": "/tmp/f.txt"}
        )

    @pytest.mark.asyncio
    async def test_stat_async(self, api):
        await api.stat_async("/tmp/f.txt")
        api.get_async.assert_called_once_with(
            "/filesystem/stat", query={"path": "/tmp/f.txt"}
        )


class TestAioMkdir:

    def test_mkdir_defaults(self, api):
        api.mkdir("/tmp/dir")
        api.post.assert_called_once_with(
            "/filesystem/mkdir",
            data={"path": "/tmp/dir", "parents": True, "mode": "0755"},
        )

    @pytest.mark.asyncio
    async def test_mkdir_async(self, api):
        await api.mkdir_async("/tmp/dir")
        api.post_async.assert_called_once_with(
            "/filesystem/mkdir",
            data={"path": "/tmp/dir", "parents": True, "mode": "0755"},
        )


class TestAioMoveFile:

    def test_move_file(self, api):
        api.move_file("/a", "/b")
        api.post.assert_called_once_with(
            "/filesystem/move", data={"source": "/a", "destination": "/b"}
        )

    @pytest.mark.asyncio
    async def test_move_file_async(self, api):
        await api.move_file_async("/a", "/b")
        api.post_async.assert_called_once_with(
            "/filesystem/move", data={"source": "/a", "destination": "/b"}
        )


class TestAioRemoveFile:

    def test_remove_file(self, api):
        api.remove_file("/tmp/x")
        api.post.assert_called_once_with(
            "/filesystem/remove", data={"path": "/tmp/x"}
        )

    @pytest.mark.asyncio
    async def test_remove_file_async(self, api):
        await api.remove_file_async("/tmp/x")
        api.post_async.assert_called_once_with(
            "/filesystem/remove", data={"path": "/tmp/x"}
        )


class TestAioContexts:

    def test_list_contexts(self, api):
        api.list_contexts()
        api.get.assert_called_once_with("/contexts")

    @pytest.mark.asyncio
    async def test_list_contexts_async(self, api):
        await api.list_contexts_async()
        api.get_async.assert_called_once_with("/contexts")

    def test_create_context_default(self, api):
        api.create_context()
        api.post.assert_called_once_with(
            "/contexts",
            data={"cwd": "/home/user", "language": CodeLanguage.PYTHON},
        )

    def test_create_context_invalid_language(self, api):
        with pytest.raises(ValueError, match="language must be"):
            api.create_context(language="ruby")

    @pytest.mark.asyncio
    async def test_create_context_async_default(self, api):
        await api.create_context_async()
        api.post_async.assert_called_once_with(
            "/contexts",
            data={"cwd": "/home/user", "language": CodeLanguage.PYTHON},
        )

    @pytest.mark.asyncio
    async def test_create_context_async_invalid_language(self, api):
        with pytest.raises(ValueError, match="language must be"):
            await api.create_context_async(language="ruby")

    def test_get_context(self, api):
        api.get_context("ctx-1")
        api.get.assert_called_once_with("/contexts/ctx-1")

    @pytest.mark.asyncio
    async def test_get_context_async(self, api):
        await api.get_context_async("ctx-1")
        api.get_async.assert_called_once_with("/contexts/ctx-1")

    def test_delete_context(self, api):
        api.delete_context("ctx-1")
        api.delete.assert_called_once_with("/contexts/ctx-1")

    @pytest.mark.asyncio
    async def test_delete_context_async(self, api):
        await api.delete_context_async("ctx-1")
        api.delete_async.assert_called_once_with("/contexts/ctx-1")


class TestAioExecuteCode:

    def test_execute_code_minimal(self, api):
        api.execute_code("print(1)", context_id=None)
        api.post.assert_called_once_with(
            "/contexts/execute", data={"code": "print(1)", "timeout": 30}
        )

    def test_execute_code_with_all_params(self, api):
        api.execute_code(
            "print(1)",
            context_id="c1",
            language=CodeLanguage.PYTHON,
            timeout=60,
        )
        call_data = api.post.call_args[1]["data"]
        assert call_data["contextId"] == "c1"
        assert call_data["language"] == CodeLanguage.PYTHON
        assert call_data["timeout"] == 60

    def test_execute_code_no_timeout(self, api):
        api.execute_code("print(1)", context_id=None, timeout=None)
        call_data = api.post.call_args[1]["data"]
        assert "timeout" not in call_data

    def test_execute_code_invalid_language(self, api):
        with pytest.raises(ValueError, match="language must be"):
            api.execute_code("code", context_id=None, language="ruby")

    @pytest.mark.asyncio
    async def test_execute_code_async_minimal(self, api):
        await api.execute_code_async("print(1)", context_id=None)
        api.post_async.assert_called_once_with(
            "/contexts/execute", data={"code": "print(1)", "timeout": 30}
        )

    @pytest.mark.asyncio
    async def test_execute_code_async_invalid_language(self, api):
        with pytest.raises(ValueError, match="language must be"):
            await api.execute_code_async(
                "code", context_id=None, language="ruby"
            )


class TestAioFiles:

    def test_read_file(self, api):
        api.read_file("/tmp/f.txt")
        api.get.assert_called_once_with("/files", query={"path": "/tmp/f.txt"})

    @pytest.mark.asyncio
    async def test_read_file_async(self, api):
        await api.read_file_async("/tmp/f.txt")
        api.get_async.assert_called_once_with(
            "/files", query={"path": "/tmp/f.txt"}
        )

    def test_write_file_defaults(self, api):
        api.write_file("/tmp/f.txt", "content")
        api.post.assert_called_once_with(
            "/files",
            data={
                "path": "/tmp/f.txt",
                "content": "content",
                "mode": "644",
                "encoding": "utf-8",
                "createDir": True,
            },
        )

    @pytest.mark.asyncio
    async def test_write_file_async(self, api):
        await api.write_file_async("/tmp/f.txt", "content")
        api.post_async.assert_called_once()

    def test_upload_file(self, api):
        api.upload_file("/local/f", "/remote/f")
        api.post_file.assert_called_once_with(
            path="/filesystem/upload",
            local_file_path="/local/f",
            target_file_path="/remote/f",
        )

    @pytest.mark.asyncio
    async def test_upload_file_async(self, api):
        await api.upload_file_async("/local/f", "/remote/f")
        api.post_file_async.assert_called_once_with(
            path="/filesystem/upload",
            local_file_path="/local/f",
            target_file_path="/remote/f",
        )

    def test_download_file(self, api):
        api.download_file("/remote/f", "/local/f")
        api.get_file.assert_called_once_with(
            path="/filesystem/download",
            save_path="/local/f",
            query={"path": "/remote/f"},
        )

    @pytest.mark.asyncio
    async def test_download_file_async(self, api):
        await api.download_file_async("/remote/f", "/local/f")
        api.get_file_async.assert_called_once_with(
            path="/filesystem/download",
            save_path="/local/f",
            query={"path": "/remote/f"},
        )


class TestAioProcesses:

    def test_cmd(self, api):
        api.cmd("ls", "/home")
        api.post.assert_called_once_with(
            "/processes/cmd",
            data={"command": "ls", "cwd": "/home", "timeout": 30},
        )

    def test_cmd_no_timeout(self, api):
        api.cmd("ls", "/home", timeout=None)
        call_data = api.post.call_args[1]["data"]
        assert "timeout" not in call_data

    @pytest.mark.asyncio
    async def test_cmd_async(self, api):
        await api.cmd_async("ls", "/home")
        api.post_async.assert_called_once_with(
            "/processes/cmd",
            data={"command": "ls", "cwd": "/home", "timeout": 30},
        )

    @pytest.mark.asyncio
    async def test_cmd_async_no_timeout(self, api):
        await api.cmd_async("ls", "/home", timeout=None)
        call_data = api.post_async.call_args[1]["data"]
        assert "timeout" not in call_data

    def test_list_processes(self, api):
        api.list_processes()
        api.get.assert_called_once_with("/processes")

    @pytest.mark.asyncio
    async def test_list_processes_async(self, api):
        await api.list_processes_async()
        api.get_async.assert_called_once_with("/processes")

    def test_get_process(self, api):
        api.get_process("123")
        api.get.assert_called_once_with("/processes/123")

    @pytest.mark.asyncio
    async def test_get_process_async(self, api):
        await api.get_process_async("123")
        api.get_async.assert_called_once_with("/processes/123")

    def test_kill_process(self, api):
        api.kill_process("123")
        api.delete.assert_called_once_with("/processes/123")

    @pytest.mark.asyncio
    async def test_kill_process_async(self, api):
        await api.kill_process_async("123")
        api.delete_async.assert_called_once_with("/processes/123")
