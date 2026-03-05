"""Tests for agentrun.sandbox.code_interpreter_sandbox module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.sandbox.code_interpreter_sandbox import (
    CodeInterpreterSandbox,
    ContextOperations,
    FileOperations,
    FileSystemOperations,
    ProcessOperations,
)
from agentrun.sandbox.model import CodeLanguage, TemplateType
from agentrun.utils.exception import ServerError


def _make_sandbox(sandbox_id="sb-123"):
    sb = CodeInterpreterSandbox.model_construct(sandbox_id=sandbox_id)
    sb._data_api = MagicMock()
    return sb


# ==================== FileOperations ====================


class TestFileOperations:

    def test_read(self):
        sb = _make_sandbox()
        sb.data_api.read_file.return_value = "content"
        ops = FileOperations(sb)
        assert ops.read("/tmp/f.txt") == "content"
        sb.data_api.read_file.assert_called_once_with(path="/tmp/f.txt")

    @pytest.mark.asyncio
    async def test_read_async(self):
        sb = _make_sandbox()
        sb.data_api.read_file_async = AsyncMock(return_value="async-content")
        ops = FileOperations(sb)
        result = await ops.read_async("/tmp/f.txt")
        assert result == "async-content"

    def test_write(self):
        sb = _make_sandbox()
        sb.data_api.write_file.return_value = {"ok": True}
        ops = FileOperations(sb)
        result = ops.write(
            "/tmp/f.txt", "data", mode="755", encoding="ascii", create_dir=False
        )
        assert result == {"ok": True}
        sb.data_api.write_file.assert_called_once_with(
            path="/tmp/f.txt",
            content="data",
            mode="755",
            encoding="ascii",
            create_dir=False,
        )

    @pytest.mark.asyncio
    async def test_write_async(self):
        sb = _make_sandbox()
        sb.data_api.write_file_async = AsyncMock(return_value={"ok": True})
        ops = FileOperations(sb)
        result = await ops.write_async("/tmp/f.txt", "data")
        assert result == {"ok": True}


# ==================== FileSystemOperations ====================


class TestFileSystemOperations:

    def test_list(self):
        sb = _make_sandbox()
        sb.data_api.list_directory.return_value = [{"name": "a"}]
        ops = FileSystemOperations(sb)
        assert ops.list(path="/home", depth=2) == [{"name": "a"}]

    @pytest.mark.asyncio
    async def test_list_async(self):
        sb = _make_sandbox()
        sb.data_api.list_directory_async = AsyncMock(return_value=[])
        ops = FileSystemOperations(sb)
        assert await ops.list_async() == []

    def test_move(self):
        sb = _make_sandbox()
        sb.data_api.move_file.return_value = {"ok": True}
        ops = FileSystemOperations(sb)
        assert ops.move("/a", "/b") == {"ok": True}

    @pytest.mark.asyncio
    async def test_move_async(self):
        sb = _make_sandbox()
        sb.data_api.move_file_async = AsyncMock(return_value={"ok": True})
        ops = FileSystemOperations(sb)
        assert await ops.move_async("/a", "/b") == {"ok": True}

    def test_remove(self):
        sb = _make_sandbox()
        sb.data_api.remove_file.return_value = {"ok": True}
        ops = FileSystemOperations(sb)
        assert ops.remove("/tmp/x") == {"ok": True}

    @pytest.mark.asyncio
    async def test_remove_async(self):
        sb = _make_sandbox()
        sb.data_api.remove_file_async = AsyncMock(return_value={"ok": True})
        ops = FileSystemOperations(sb)
        assert await ops.remove_async("/tmp/x") == {"ok": True}

    def test_stat(self):
        sb = _make_sandbox()
        sb.data_api.stat.return_value = {"size": 100}
        ops = FileSystemOperations(sb)
        assert ops.stat("/tmp/x") == {"size": 100}

    @pytest.mark.asyncio
    async def test_stat_async(self):
        sb = _make_sandbox()
        sb.data_api.stat_async = AsyncMock(return_value={"size": 100})
        ops = FileSystemOperations(sb)
        assert await ops.stat_async("/tmp/x") == {"size": 100}

    def test_mkdir(self):
        sb = _make_sandbox()
        sb.data_api.mkdir.return_value = {"ok": True}
        ops = FileSystemOperations(sb)
        assert ops.mkdir("/tmp/dir", parents=False, mode="0700") == {"ok": True}

    @pytest.mark.asyncio
    async def test_mkdir_async(self):
        sb = _make_sandbox()
        sb.data_api.mkdir_async = AsyncMock(return_value={"ok": True})
        ops = FileSystemOperations(sb)
        assert await ops.mkdir_async("/tmp/dir") == {"ok": True}

    def test_upload(self):
        sb = _make_sandbox()
        sb.data_api.upload_file.return_value = {"ok": True}
        ops = FileSystemOperations(sb)
        assert ops.upload("/local/f", "/remote/f") == {"ok": True}

    @pytest.mark.asyncio
    async def test_upload_async(self):
        sb = _make_sandbox()
        sb.data_api.upload_file_async = AsyncMock(return_value={"ok": True})
        ops = FileSystemOperations(sb)
        assert await ops.upload_async("/local/f", "/remote/f") == {"ok": True}

    def test_download(self):
        sb = _make_sandbox()
        sb.data_api.download_file.return_value = {
            "saved_path": "/x",
            "size": 10,
        }
        ops = FileSystemOperations(sb)
        assert ops.download("/remote/f", "/local/f") == {
            "saved_path": "/x",
            "size": 10,
        }

    @pytest.mark.asyncio
    async def test_download_async(self):
        sb = _make_sandbox()
        sb.data_api.download_file_async = AsyncMock(
            return_value={"saved_path": "/x", "size": 10}
        )
        ops = FileSystemOperations(sb)
        assert await ops.download_async("/remote/f", "/local/f") == {
            "saved_path": "/x",
            "size": 10,
        }


# ==================== ProcessOperations ====================


class TestProcessOperations:

    def test_cmd(self):
        sb = _make_sandbox()
        sb.data_api.cmd.return_value = {"exit_code": 0}
        ops = ProcessOperations(sb)
        assert ops.cmd("ls", "/home", timeout=10) == {"exit_code": 0}

    @pytest.mark.asyncio
    async def test_cmd_async(self):
        sb = _make_sandbox()
        sb.data_api.cmd_async = AsyncMock(return_value={"exit_code": 0})
        ops = ProcessOperations(sb)
        assert await ops.cmd_async("ls", "/home") == {"exit_code": 0}

    def test_list(self):
        sb = _make_sandbox()
        sb.data_api.list_processes.return_value = [{"pid": "1"}]
        ops = ProcessOperations(sb)
        assert ops.list() == [{"pid": "1"}]

    @pytest.mark.asyncio
    async def test_list_async(self):
        sb = _make_sandbox()
        sb.data_api.list_processes_async = AsyncMock(return_value=[])
        ops = ProcessOperations(sb)
        assert await ops.list_async() == []

    def test_get(self):
        sb = _make_sandbox()
        sb.data_api.get_process.return_value = {"pid": "1"}
        ops = ProcessOperations(sb)
        assert ops.get("1") == {"pid": "1"}

    @pytest.mark.asyncio
    async def test_get_async(self):
        sb = _make_sandbox()
        sb.data_api.get_process_async = AsyncMock(return_value={"pid": "1"})
        ops = ProcessOperations(sb)
        assert await ops.get_async("1") == {"pid": "1"}

    def test_kill(self):
        sb = _make_sandbox()
        sb.data_api.kill_process.return_value = {"ok": True}
        ops = ProcessOperations(sb)
        assert ops.kill("1") == {"ok": True}

    @pytest.mark.asyncio
    async def test_kill_async(self):
        sb = _make_sandbox()
        sb.data_api.kill_process_async = AsyncMock(return_value={"ok": True})
        ops = ProcessOperations(sb)
        assert await ops.kill_async("1") == {"ok": True}


# ==================== ContextOperations ====================


class TestContextOperations:

    def _make_ctx_ops(self):
        sb = _make_sandbox()
        return ContextOperations(sb), sb

    def test_context_id_default_none(self):
        ops, _ = self._make_ctx_ops()
        assert ops.context_id is None

    def test_list(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.list_contexts.return_value = [{"id": "c1"}]
        assert ops.list() == [{"id": "c1"}]

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
        assert ops._language == "python"
        assert ops._cwd == "/home"

    def test_create_failure(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.create_context.return_value = {
            "id": None,
            "cwd": "/home",
            "language": "python",
        }
        with pytest.raises(ServerError):
            ops.create()

    @pytest.mark.asyncio
    async def test_create_async_success(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.create_context_async = AsyncMock(
            return_value={"id": "c2", "cwd": "/home", "language": "javascript"}
        )
        result = await ops.create_async(
            language=CodeLanguage.PYTHON, cwd="/work"
        )
        assert result is ops
        assert ops.context_id == "c2"

    @pytest.mark.asyncio
    async def test_create_async_failure(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.create_context_async = AsyncMock(
            return_value={"incomplete": True}
        )
        with pytest.raises(ServerError):
            await ops.create_async()

    def test_get_with_explicit_id(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.get_context.return_value = {
            "id": "c3",
            "cwd": "/x",
            "language": "python",
        }
        result = ops.get(context_id="c3")
        assert result is ops
        assert ops.context_id == "c3"

    def test_get_with_saved_id(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "saved-id"
        sb.data_api.get_context.return_value = {
            "id": "saved-id",
            "cwd": "/x",
            "language": "python",
        }
        ops.get()
        assert ops.context_id == "saved-id"

    def test_get_no_id_raises(self):
        ops, _ = self._make_ctx_ops()
        with pytest.raises(ValueError, match="context id is not set"):
            ops.get()

    def test_get_failure_raises(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.get_context.return_value = {"id": None}
        with pytest.raises(ServerError):
            ops.get(context_id="c1")

    @pytest.mark.asyncio
    async def test_get_async_with_id(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.get_context_async = AsyncMock(
            return_value={"id": "c3", "cwd": "/x", "language": "python"}
        )
        result = await ops.get_async(context_id="c3")
        assert result is ops

    @pytest.mark.asyncio
    async def test_get_async_no_id_raises(self):
        ops, _ = self._make_ctx_ops()
        with pytest.raises(ValueError, match="context id is not set"):
            await ops.get_async()

    @pytest.mark.asyncio
    async def test_get_async_failure_raises(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.get_context_async = AsyncMock(return_value={"id": None})
        with pytest.raises(ServerError):
            await ops.get_async(context_id="c1")

    def test_execute_with_context_id(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.execute_code.return_value = {"result": "ok"}
        result = ops.execute("print(1)")
        assert result == {"result": "ok"}
        sb.data_api.execute_code.assert_called_once_with(
            context_id="c1", language=None, code="print(1)", timeout=30
        )

    def test_execute_no_context_no_language_defaults_python(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.execute_code.return_value = {"result": "ok"}
        ops.execute("print(1)")
        sb.data_api.execute_code.assert_called_once_with(
            context_id=None,
            language=CodeLanguage.PYTHON,
            code="print(1)",
            timeout=30,
        )

    def test_execute_with_explicit_params(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.execute_code.return_value = {"result": "ok"}
        ops.execute(
            "code", language=CodeLanguage.PYTHON, context_id="x", timeout=60
        )
        sb.data_api.execute_code.assert_called_once_with(
            context_id="x",
            language=CodeLanguage.PYTHON,
            code="code",
            timeout=60,
        )

    @pytest.mark.asyncio
    async def test_execute_async_with_context_id(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.execute_code_async = AsyncMock(
            return_value={"result": "ok"}
        )
        result = await ops.execute_async("code")
        assert result == {"result": "ok"}

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

    def test_delete_with_context_id(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.delete_context.return_value = {"ok": True}
        result = ops.delete()
        assert result == {"ok": True}
        assert ops._context_id is None

    def test_delete_with_explicit_id(self):
        ops, sb = self._make_ctx_ops()
        sb.data_api.delete_context.return_value = {"ok": True}
        result = ops.delete(context_id="c2")
        assert result == {"ok": True}
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
        result = await ops.delete_async()
        assert result == {"ok": True}
        assert ops._context_id is None

    @pytest.mark.asyncio
    async def test_delete_async_no_id_raises(self):
        ops, _ = self._make_ctx_ops()
        with pytest.raises(ValueError, match="context_id is required"):
            await ops.delete_async()

    def test_enter_with_context_id(self):
        ops, _ = self._make_ctx_ops()
        ops._context_id = "c1"
        assert ops.__enter__() is ops

    def test_enter_no_context_id_raises(self):
        ops, _ = self._make_ctx_ops()
        with pytest.raises(ValueError, match="No context has been created"):
            ops.__enter__()

    @pytest.mark.asyncio
    async def test_aenter_with_context_id(self):
        ops, _ = self._make_ctx_ops()
        ops._context_id = "c1"
        assert await ops.__aenter__() is ops

    @pytest.mark.asyncio
    async def test_aenter_no_context_id_raises(self):
        ops, _ = self._make_ctx_ops()
        with pytest.raises(ValueError, match="No context has been created"):
            await ops.__aenter__()

    def test_exit_with_context_id(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.delete_context.return_value = {"ok": True}
        result = ops.__exit__(None, None, None)
        assert result is False

    def test_exit_no_context_id(self):
        ops, _ = self._make_ctx_ops()
        result = ops.__exit__(None, None, None)
        assert result is False

    def test_exit_delete_fails_logs_error(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.delete_context.side_effect = Exception("fail")
        result = ops.__exit__(None, None, None)
        assert result is False

    @pytest.mark.asyncio
    async def test_aexit_with_context_id(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.delete_context_async = AsyncMock(return_value={"ok": True})
        result = await ops.__aexit__(None, None, None)
        assert result is False

    @pytest.mark.asyncio
    async def test_aexit_no_context_id(self):
        ops, _ = self._make_ctx_ops()
        result = await ops.__aexit__(None, None, None)
        assert result is False

    @pytest.mark.asyncio
    async def test_aexit_delete_fails_logs_error(self):
        ops, sb = self._make_ctx_ops()
        ops._context_id = "c1"
        sb.data_api.delete_context_async = AsyncMock(
            side_effect=Exception("fail")
        )
        result = await ops.__aexit__(None, None, None)
        assert result is False


# ==================== CodeInterpreterSandbox ====================


class TestCodeInterpreterSandbox:

    def test_template_type(self):
        assert (
            CodeInterpreterSandbox.__private_attributes__[
                "_template_type"
            ].default
            == TemplateType.CODE_INTERPRETER
        )

    def test_data_api_lazy_init(self):
        sb = CodeInterpreterSandbox.model_construct(sandbox_id="sb-123")
        with patch(
            "agentrun.sandbox.code_interpreter_sandbox.CodeInterpreterDataAPI"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            api = sb.data_api
            assert api is not None
            assert sb.data_api is api  # cached

    def test_file_property(self):
        sb = _make_sandbox()
        f = sb.file
        assert isinstance(f, FileOperations)
        assert sb.file is f  # cached

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

    def test_enter_health_ok(self):
        sb = _make_sandbox()
        sb.data_api.check_health.return_value = {"status": "ok"}
        result = sb.__enter__()
        assert result is sb

    def test_enter_health_retries_then_ok(self):
        sb = _make_sandbox()
        sb.data_api.check_health.side_effect = [
            {"status": "not-ready"},
            {"status": "ok"},
        ]
        with patch("agentrun.sandbox.code_interpreter_sandbox.time.sleep"):
            result = sb.__enter__()
            assert result is sb

    def test_enter_health_exception_retries(self):
        sb = _make_sandbox()
        sb.data_api.check_health.side_effect = [
            Exception("network error"),
            {"status": "ok"},
        ]
        with patch("agentrun.sandbox.code_interpreter_sandbox.time.sleep"):
            result = sb.__enter__()
            assert result is sb

    def test_enter_timeout(self):
        sb = _make_sandbox()
        sb.data_api.check_health.return_value = {"status": "not-ready"}
        with patch("agentrun.sandbox.code_interpreter_sandbox.time.sleep"):
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
            "agentrun.sandbox.code_interpreter_sandbox.asyncio.sleep",
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
            "agentrun.sandbox.code_interpreter_sandbox.asyncio.sleep",
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
            "agentrun.sandbox.code_interpreter_sandbox.asyncio.sleep",
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
        sb = CodeInterpreterSandbox.model_construct(sandbox_id=None)
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
        sb = CodeInterpreterSandbox.model_construct(sandbox_id=None)
        with pytest.raises(ValueError, match="Sandbox ID is not set"):
            await sb.__aexit__(None, None, None)
