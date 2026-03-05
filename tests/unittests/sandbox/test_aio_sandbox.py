"""Tests for agentrun.sandbox.aio_sandbox module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.sandbox.aio_sandbox import (
    AioSandbox,
    ContextOperations,
    FileOperations,
    FileSystemOperations,
    ProcessOperations,
)
from agentrun.sandbox.model import CodeLanguage, TemplateType
from agentrun.utils.exception import ServerError


def _make_sandbox(sandbox_id="sb-aio-1"):
    sb = AioSandbox.model_construct(sandbox_id=sandbox_id)
    sb._data_api = MagicMock()
    return sb


# ==================== FileOperations ====================


class TestAioFileOperations:

    def test_read(self):
        sb = _make_sandbox()
        sb.data_api.read_file.return_value = "data"
        ops = FileOperations(sb)
        assert ops.read("/f") == "data"

    @pytest.mark.asyncio
    async def test_read_async(self):
        sb = _make_sandbox()
        sb.data_api.read_file_async = AsyncMock(return_value="data")
        ops = FileOperations(sb)
        assert await ops.read_async("/f") == "data"

    def test_write(self):
        sb = _make_sandbox()
        sb.data_api.write_file.return_value = {"ok": True}
        ops = FileOperations(sb)
        assert ops.write(
            "/f", "content", mode="755", encoding="ascii", create_dir=False
        ) == {"ok": True}

    @pytest.mark.asyncio
    async def test_write_async(self):
        sb = _make_sandbox()
        sb.data_api.write_file_async = AsyncMock(return_value={"ok": True})
        ops = FileOperations(sb)
        assert await ops.write_async("/f", "content") == {"ok": True}


# ==================== FileSystemOperations ====================


class TestAioFileSystemOperations:

    def test_list(self):
        sb = _make_sandbox()
        sb.data_api.list_directory.return_value = []
        ops = FileSystemOperations(sb)
        assert ops.list(path="/home", depth=2) == []

    @pytest.mark.asyncio
    async def test_list_async(self):
        sb = _make_sandbox()
        sb.data_api.list_directory_async = AsyncMock(return_value=[])
        assert await FileSystemOperations(sb).list_async() == []

    def test_move(self):
        sb = _make_sandbox()
        sb.data_api.move_file.return_value = {"ok": True}
        assert FileSystemOperations(sb).move("/a", "/b") == {"ok": True}

    @pytest.mark.asyncio
    async def test_move_async(self):
        sb = _make_sandbox()
        sb.data_api.move_file_async = AsyncMock(return_value={"ok": True})
        assert await FileSystemOperations(sb).move_async("/a", "/b") == {
            "ok": True
        }

    def test_remove(self):
        sb = _make_sandbox()
        sb.data_api.remove_file.return_value = {"ok": True}
        assert FileSystemOperations(sb).remove("/x") == {"ok": True}

    @pytest.mark.asyncio
    async def test_remove_async(self):
        sb = _make_sandbox()
        sb.data_api.remove_file_async = AsyncMock(return_value={"ok": True})
        assert await FileSystemOperations(sb).remove_async("/x") == {"ok": True}

    def test_stat(self):
        sb = _make_sandbox()
        sb.data_api.stat.return_value = {"size": 10}
        assert FileSystemOperations(sb).stat("/x") == {"size": 10}

    @pytest.mark.asyncio
    async def test_stat_async(self):
        sb = _make_sandbox()
        sb.data_api.stat_async = AsyncMock(return_value={"size": 10})
        assert await FileSystemOperations(sb).stat_async("/x") == {"size": 10}

    def test_mkdir(self):
        sb = _make_sandbox()
        sb.data_api.mkdir.return_value = {"ok": True}
        assert FileSystemOperations(sb).mkdir(
            "/d", parents=False, mode="0700"
        ) == {"ok": True}

    @pytest.mark.asyncio
    async def test_mkdir_async(self):
        sb = _make_sandbox()
        sb.data_api.mkdir_async = AsyncMock(return_value={"ok": True})
        assert await FileSystemOperations(sb).mkdir_async("/d") == {"ok": True}

    def test_upload(self):
        sb = _make_sandbox()
        sb.data_api.upload_file.return_value = {"ok": True}
        assert FileSystemOperations(sb).upload("/l", "/r") == {"ok": True}

    @pytest.mark.asyncio
    async def test_upload_async(self):
        sb = _make_sandbox()
        sb.data_api.upload_file_async = AsyncMock(return_value={"ok": True})
        assert await FileSystemOperations(sb).upload_async("/l", "/r") == {
            "ok": True
        }

    def test_download(self):
        sb = _make_sandbox()
        sb.data_api.download_file.return_value = {"saved_path": "/x"}
        assert FileSystemOperations(sb).download("/r", "/l") == {
            "saved_path": "/x"
        }

    @pytest.mark.asyncio
    async def test_download_async(self):
        sb = _make_sandbox()
        sb.data_api.download_file_async = AsyncMock(
            return_value={"saved_path": "/x"}
        )
        assert await FileSystemOperations(sb).download_async("/r", "/l") == {
            "saved_path": "/x"
        }


# ==================== ProcessOperations ====================


class TestAioProcessOperations:

    def test_cmd(self):
        sb = _make_sandbox()
        sb.data_api.cmd.return_value = {"exit_code": 0}
        assert ProcessOperations(sb).cmd("ls", "/home") == {"exit_code": 0}

    @pytest.mark.asyncio
    async def test_cmd_async(self):
        sb = _make_sandbox()
        sb.data_api.cmd_async = AsyncMock(return_value={"exit_code": 0})
        assert await ProcessOperations(sb).cmd_async("ls", "/home") == {
            "exit_code": 0
        }

    def test_list(self):
        sb = _make_sandbox()
        sb.data_api.list_processes.return_value = []
        assert ProcessOperations(sb).list() == []

    @pytest.mark.asyncio
    async def test_list_async(self):
        sb = _make_sandbox()
        sb.data_api.list_processes_async = AsyncMock(return_value=[])
        assert await ProcessOperations(sb).list_async() == []

    def test_get(self):
        sb = _make_sandbox()
        sb.data_api.get_process.return_value = {"pid": "1"}
        assert ProcessOperations(sb).get("1") == {"pid": "1"}

    @pytest.mark.asyncio
    async def test_get_async(self):
        sb = _make_sandbox()
        sb.data_api.get_process_async = AsyncMock(return_value={"pid": "1"})
        assert await ProcessOperations(sb).get_async("1") == {"pid": "1"}

    def test_kill(self):
        sb = _make_sandbox()
        sb.data_api.kill_process.return_value = {"ok": True}
        assert ProcessOperations(sb).kill("1") == {"ok": True}

    @pytest.mark.asyncio
    async def test_kill_async(self):
        sb = _make_sandbox()
        sb.data_api.kill_process_async = AsyncMock(return_value={"ok": True})
        assert await ProcessOperations(sb).kill_async("1") == {"ok": True}


# ==================== ContextOperations ====================


class TestAioContextOperations:

    def _make_ctx_ops(self):
        sb = _make_sandbox()
        return ContextOperations(sb), sb

    def test_context_id_default_none(self):
        ops, _ = self._make_ctx_ops()
        assert ops.context_id is None

    def test_list(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.list_contexts.return_value = []
        assert ops.list() == []

    @pytest.mark.asyncio
    async def test_list_async(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.list_contexts_async = AsyncMock(return_value=[])
        assert await ops.list_async() == []

    def test_create_success(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.create_context.return_value = {
            "id": "c1",
            "cwd": "/home",
            "language": "python",
        }
        result = ops.create()
        assert result is ops
        assert ops.context_id == "c1"

    def test_create_failure(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.create_context.return_value = {}
        with pytest.raises(ServerError):
            ops.create()

    @pytest.mark.asyncio
    async def test_create_async_success(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.create_context_async = AsyncMock(
            return_value={"id": "c1", "cwd": "/home", "language": "python"}
        )
        result = await ops.create_async()
        assert result is ops

    @pytest.mark.asyncio
    async def test_create_async_failure(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.create_context_async = AsyncMock(return_value={})
        with pytest.raises(ServerError):
            await ops.create_async()

    def test_get_with_id(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.get_context.return_value = {
            "id": "c1",
            "cwd": "/x",
            "language": "python",
        }
        ops.get(context_id="c1")
        assert ops.context_id == "c1"

    def test_get_no_id_raises(self):
        ops, _ = self._make_ctx_ops()
        with pytest.raises(ValueError, match="context id is not set"):
            ops.get()

    def test_get_failure(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.get_context.return_value = {}
        with pytest.raises(ServerError):
            ops.get(context_id="c1")

    @pytest.mark.asyncio
    async def test_get_async_with_id(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.get_context_async = AsyncMock(
            return_value={"id": "c1", "cwd": "/x", "language": "python"}
        )
        await ops.get_async(context_id="c1")
        assert ops.context_id == "c1"

    @pytest.mark.asyncio
    async def test_get_async_no_id_raises(self):
        ops, _ = self._make_ctx_ops()
        with pytest.raises(ValueError, match="context id is not set"):
            await ops.get_async()

    @pytest.mark.asyncio
    async def test_get_async_failure(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.get_context_async = AsyncMock(return_value={})
        with pytest.raises(ServerError):
            await ops.get_async(context_id="c1")

    def test_execute_with_context_id(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.execute_code.return_value = {"result": "ok"}
        ops.execute("code")
        sb.data_api.execute_code.assert_called_once_with(
            context_id="c1", language=None, code="code", timeout=30
        )

    def test_execute_defaults_python(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.execute_code.return_value = {"result": "ok"}
        ops.execute("code")
        sb.data_api.execute_code.assert_called_once_with(
            context_id=None,
            language=CodeLanguage.PYTHON,
            code="code",
            timeout=30,
        )

    @pytest.mark.asyncio
    async def test_execute_async_with_context_id(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.execute_code_async = AsyncMock(
            return_value={"result": "ok"}
        )
        await ops.execute_async("code")
        sb.data_api.execute_code_async.assert_called_once_with(
            context_id="c1", language=None, code="code", timeout=30
        )

    @pytest.mark.asyncio
    async def test_execute_async_defaults_python(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.execute_code_async = AsyncMock(
            return_value={"result": "ok"}
        )
        await ops.execute_async("code")
        sb.data_api.execute_code_async.assert_called_once_with(
            context_id=None,
            language=CodeLanguage.PYTHON,
            code="code",
            timeout=30,
        )

    def test_delete_with_id(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.delete_context.return_value = {"ok": True}
        ops.delete()
        assert ops._context_id is None

    def test_delete_no_id_raises(self):
        ops, _ = self._make_ctx_ops()
        with pytest.raises(ValueError, match="context_id is required"):
            ops.delete()

    @pytest.mark.asyncio
    async def test_delete_async_with_id(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.delete_context_async = AsyncMock(return_value={"ok": True})
        await ops.delete_async()
        assert ops._context_id is None

    @pytest.mark.asyncio
    async def test_delete_async_no_id_raises(self):
        ops, _ = self._make_ctx_ops()
        with pytest.raises(ValueError, match="context_id is required"):
            await ops.delete_async()

    def test_enter_with_context(self):
        ops, _ = self._make_ctx_ops()
        ops._context_id = "c1"
        assert ops.__enter__() is ops

    def test_enter_no_context_raises(self):
        ops, _ = self._make_ctx_ops()
        with pytest.raises(ValueError, match="No context has been created"):
            ops.__enter__()

    @pytest.mark.asyncio
    async def test_aenter_with_context(self):
        ops, _ = self._make_ctx_ops()
        ops._context_id = "c1"
        assert await ops.__aenter__() is ops

    @pytest.mark.asyncio
    async def test_aenter_no_context_raises(self):
        ops, _ = self._make_ctx_ops()
        with pytest.raises(ValueError, match="No context has been created"):
            await ops.__aenter__()

    def test_exit_with_context(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.delete_context.return_value = {"ok": True}
        assert ops.__exit__(None, None, None) is False

    def test_exit_no_context(self):
        ops, _ = self._make_ctx_ops()
        assert ops.__exit__(None, None, None) is False

    def test_exit_delete_fails(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.delete_context.side_effect = Exception("fail")
        assert ops.__exit__(None, None, None) is False

    @pytest.mark.asyncio
    async def test_aexit_with_context(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.delete_context_async = AsyncMock(return_value={"ok": True})
        assert await ops.__aexit__(None, None, None) is False

    @pytest.mark.asyncio
    async def test_aexit_no_context(self):
        ops, _ = self._make_ctx_ops()
        assert await ops.__aexit__(None, None, None) is False

    @pytest.mark.asyncio
    async def test_aexit_delete_fails(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.delete_context_async = AsyncMock(
            side_effect=Exception("fail")
        )
        assert await ops.__aexit__(None, None, None) is False


# ==================== AioSandbox ====================


class TestAioSandbox:

    def test_template_type(self):
        assert (
            AioSandbox.__private_attributes__["_template_type"].default
            == TemplateType.AIO
        )

    def test_data_api_lazy_init(self):
        sb = AioSandbox.model_construct(sandbox_id="sb-1")
        with patch("agentrun.sandbox.aio_sandbox.AioDataAPI") as mock_cls:
            mock_cls.return_value = MagicMock()
            api = sb.data_api
            assert api is not None
            assert sb.data_api is api

    def test_data_api_no_sandbox_id_raises(self):
        sb = AioSandbox.model_construct(sandbox_id=None)
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
        sb.data_api.get_cdp_url.return_value = "ws://url"
        assert sb.get_cdp_url() == "ws://url"

    def test_get_vnc_url(self):
        sb = _make_sandbox()
        sb.data_api.get_vnc_url.return_value = "ws://vnc"
        assert sb.get_vnc_url() == "ws://vnc"

    def test_sync_playwright(self):
        sb = _make_sandbox()
        sb.data_api.sync_playwright.return_value = MagicMock()
        assert sb.sync_playwright() is not None

    def test_async_playwright(self):
        sb = _make_sandbox()
        sb.data_api.async_playwright.return_value = MagicMock()
        assert sb.async_playwright() is not None

    @pytest.mark.asyncio
    async def test_list_recordings_async(self):
        sb = _make_sandbox()
        sb.data_api.list_recordings_async = AsyncMock(return_value=[])
        assert await sb.list_recordings_async() == []

    def test_list_recordings(self):
        sb = _make_sandbox()
        sb.data_api.list_recordings.return_value = []
        assert sb.list_recordings() == []

    @pytest.mark.asyncio
    async def test_download_recording_async(self):
        sb = _make_sandbox()
        sb.data_api.download_recording_async = AsyncMock(
            return_value={"saved_path": "/x"}
        )
        assert await sb.download_recording_async("f.mkv", "/x") == {
            "saved_path": "/x"
        }

    def test_download_recording(self):
        sb = _make_sandbox()
        sb.data_api.download_recording.return_value = {"saved_path": "/x"}
        assert sb.download_recording("f.mkv", "/x") == {"saved_path": "/x"}

    @pytest.mark.asyncio
    async def test_delete_recording_async(self):
        sb = _make_sandbox()
        sb.data_api.delete_recording_async = AsyncMock(
            return_value={"ok": True}
        )
        assert await sb.delete_recording_async("f.mkv") == {"ok": True}

    def test_delete_recording(self):
        sb = _make_sandbox()
        sb.data_api.delete_recording.return_value = {"ok": True}
        assert sb.delete_recording("f.mkv") == {"ok": True}

    def test_file_property(self):
        sb = _make_sandbox()
        f = sb.file
        assert isinstance(f, FileOperations)
        assert sb.file is f

    def test_file_system_property(self):
        sb = _make_sandbox()
        fs = sb.file_system
        assert isinstance(fs, FileSystemOperations)
        assert sb.file_system is fs

    def test_context_property(self):
        sb = _make_sandbox()
        ctx = sb.context
        assert isinstance(ctx, ContextOperations)
        assert sb.context is ctx

    def test_process_property(self):
        sb = _make_sandbox()
        p = sb.process
        assert isinstance(p, ProcessOperations)
        assert sb.process is p


class TestAioSandboxContextManager:

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
        with patch("agentrun.sandbox.aio_sandbox.time.sleep"):
            result = sb.__enter__()
            assert result is sb

    def test_enter_exception_retries(self):
        sb = _make_sandbox()
        sb.data_api.check_health.side_effect = [
            Exception("err"),
            {"status": "ok"},
        ]
        with patch("agentrun.sandbox.aio_sandbox.time.sleep"):
            result = sb.__enter__()
            assert result is sb

    def test_enter_timeout(self):
        sb = _make_sandbox()
        sb.data_api.check_health.return_value = {"status": "not-ready"}
        with patch("agentrun.sandbox.aio_sandbox.time.sleep"):
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
            "agentrun.sandbox.aio_sandbox.asyncio.sleep", new_callable=AsyncMock
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
            "agentrun.sandbox.aio_sandbox.asyncio.sleep", new_callable=AsyncMock
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
            "agentrun.sandbox.aio_sandbox.asyncio.sleep", new_callable=AsyncMock
        ):
            with pytest.raises(RuntimeError, match="Health check timeout"):
                await sb.__aenter__()

    def test_exit_calls_delete(self):
        sb = _make_sandbox()
        sb.delete = MagicMock()
        sb.__exit__(None, None, None)
        sb.delete.assert_called_once()

    def test_exit_no_sandbox_id_raises(self):
        sb = AioSandbox.model_construct(sandbox_id=None)
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
        sb = AioSandbox.model_construct(sandbox_id=None)
        with pytest.raises(ValueError, match="Sandbox ID is not set"):
            await sb.__aexit__(None, None, None)
