"""Tests for agentrun.sandbox.api.code_interpreter_data module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.sandbox.api.code_interpreter_data import CodeInterpreterDataAPI
from agentrun.sandbox.model import CodeLanguage


@pytest.fixture
def api():
    with patch.object(
        CodeInterpreterDataAPI, "__init__", lambda self, **kw: None
    ):
        obj = CodeInterpreterDataAPI.__new__(CodeInterpreterDataAPI)
        obj.config = MagicMock()
        obj.access_token = None
        obj.access_token_map = {}
        obj.resource_name = "sb-1"
        obj.resource_type = None
        obj.namespace = "sandboxes/sb-1"
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
        return obj


class TestCodeInterpreterDataAPIInit:

    @patch("agentrun.sandbox.api.code_interpreter_data.SandboxDataAPI.__init__")
    def test_init(self, mock_super_init):
        mock_super_init.return_value = None
        api = CodeInterpreterDataAPI(sandbox_id="sb-1")
        mock_super_init.assert_called_once_with(sandbox_id="sb-1", config=None)


class TestListDirectory:

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
    async def test_list_directory_async_no_params(self, api):
        await api.list_directory_async()
        api.get_async.assert_called_once_with("/filesystem", query={})

    @pytest.mark.asyncio
    async def test_list_directory_async_with_path_and_depth(self, api):
        await api.list_directory_async(path="/tmp", depth=1)
        api.get_async.assert_called_once_with(
            "/filesystem", query={"path": "/tmp", "depth": 1}
        )


class TestStat:

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


class TestMkdir:

    def test_mkdir_defaults(self, api):
        api.mkdir("/tmp/dir")
        api.post.assert_called_once_with(
            "/filesystem/mkdir",
            data={"path": "/tmp/dir", "parents": True, "mode": "0755"},
        )

    def test_mkdir_custom(self, api):
        api.mkdir("/tmp/dir", parents=False, mode="0700")
        api.post.assert_called_once_with(
            "/filesystem/mkdir",
            data={"path": "/tmp/dir", "parents": False, "mode": "0700"},
        )

    @pytest.mark.asyncio
    async def test_mkdir_async(self, api):
        await api.mkdir_async("/tmp/dir")
        api.post_async.assert_called_once_with(
            "/filesystem/mkdir",
            data={"path": "/tmp/dir", "parents": True, "mode": "0755"},
        )


class TestMoveFile:

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


class TestRemoveFile:

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


class TestContexts:

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


class TestExecuteCode:

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
        assert call_data["code"] == "print(1)"
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
    async def test_execute_code_async_with_all_params(self, api):
        await api.execute_code_async(
            "print(1)",
            context_id="c1",
            language=CodeLanguage.PYTHON,
            timeout=60,
        )
        call_data = api.post_async.call_args[1]["data"]
        assert call_data["contextId"] == "c1"
        assert call_data["language"] == CodeLanguage.PYTHON

    @pytest.mark.asyncio
    async def test_execute_code_async_invalid_language(self, api):
        with pytest.raises(ValueError, match="language must be"):
            await api.execute_code_async(
                "code", context_id=None, language="ruby"
            )


class TestFiles:

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

    def test_write_file_custom(self, api):
        api.write_file(
            "/tmp/f.txt", "data", mode="755", encoding="ascii", create_dir=False
        )
        call_data = api.post.call_args[1]["data"]
        assert call_data["mode"] == "755"
        assert call_data["encoding"] == "ascii"
        assert call_data["createDir"] is False

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


class TestProcesses:

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
