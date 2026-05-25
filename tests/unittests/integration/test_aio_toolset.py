import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentrun.integration.builtin import sandbox as sandbox_builtin
from agentrun.integration.builtin.sandbox import (
    BrowserToolSet,
    CodeInterpreterToolSet,
    sandbox_toolset,
)
from agentrun.sandbox.model import TemplateType


def _tool_names(toolset):
    return [tool.name for tool in toolset.tools()]


def test_aio_factory_returns_aio_toolset():
    ts = sandbox_toolset("aio-template", template_type=TemplateType.AIO)

    assert isinstance(ts, sandbox_builtin.AioToolSet)
    assert ts.template_type == TemplateType.AIO


def test_aio_factory_uses_aio_template_type_when_creating_sandbox():
    mock_sandbox = MagicMock()
    mock_sandbox.sandbox_id = "sandbox-aio-123"
    mock_sandbox.check_health.return_value = {"status": "ok"}

    with patch("agentrun.integration.builtin.sandbox.Sandbox") as sandbox_cls:
        sandbox_cls.create.return_value = mock_sandbox
        ts = sandbox_toolset("aio-template", template_type=TemplateType.AIO)

        assert ts.check_health() == {"status": "ok"}

    sandbox_cls.create.assert_called_once()
    assert sandbox_cls.create.call_args.kwargs["template_type"] == TemplateType.AIO


def test_aio_tool_names_include_browser_code_and_extension_tools_once():
    ts = sandbox_toolset("aio-template", template_type=TemplateType.AIO)

    names = _tool_names(ts)

    assert len(names) == len(set(names))
    assert names.count("health") == 1
    for name in [
        "browser_navigate",
        "run_code",
        "read_file",
        "file_system_list",
        "process_exec_cmd",
        "browser_get_cdp_url",
        "browser_get_vnc_url",
        "upload_file",
        "download_file",
        "browser_recordings_list",
        "browser_recording_download",
        "browser_recording_delete",
    ]:
        assert name in names


def test_aio_code_tools_accept_capability_object_not_concrete_code_sandbox():
    ts = sandbox_toolset("aio-template", template_type=TemplateType.AIO)
    mock_sandbox = MagicMock()
    mock_sandbox.file.read.return_value = "hello from aio"
    ts.sandbox = mock_sandbox
    ts.sandbox_id = "sandbox-aio-123"

    assert ts.read_file("/tmp/hello.txt") == {
        "path": "/tmp/hello.txt",
        "content": "hello from aio",
    }


def test_aio_browser_tools_accept_capability_object_not_concrete_browser_sandbox():
    ts = sandbox_toolset("aio-template", template_type=TemplateType.AIO)
    mock_sandbox = MagicMock()
    mock_playwright = MagicMock()
    mock_playwright.title.return_value = "AIO page"
    mock_sandbox.sync_playwright.return_value = mock_playwright
    ts.sandbox = mock_sandbox
    ts.sandbox_id = "sandbox-aio-123"

    assert ts.browser_get_title() == {"title": "AIO page"}
    mock_sandbox.sync_playwright.assert_called_once()


def test_aio_run_code_with_existing_context_returns_standard_result():
    ts = sandbox_toolset("aio-template", template_type=TemplateType.AIO)
    mock_sandbox = MagicMock()
    mock_sandbox.context.execute.return_value = {
        "stdout": "42\n",
        "stderr": "",
        "exitCode": 0,
    }
    ts.sandbox = mock_sandbox
    ts.sandbox_id = "sandbox-aio-123"

    result = ts.run_code("print(42)", context_id="ctx-1")

    assert result == {
        "stdout": "42\n",
        "stderr": "",
        "exit_code": 0,
        "result": {"stdout": "42\n", "stderr": "", "exitCode": 0},
    }


def test_custom_template_type_is_explicitly_unsupported():
    with pytest.raises(ValueError, match="TemplateType.CUSTOM"):
        sandbox_toolset("custom-template", template_type=TemplateType.CUSTOM)


def test_aio_local_artifact_dir_rejects_escape_paths(tmp_path):
    ts = sandbox_toolset(
        "aio-template",
        template_type=TemplateType.AIO,
        local_artifact_dir=str(tmp_path),
    )

    with pytest.raises(ValueError, match="local_artifact_path must be relative"):
        ts.download_file("/tmp/a.txt", "/tmp/out.txt")

    with pytest.raises(ValueError, match="local_artifact_path escapes"):
        ts.download_file("/tmp/a.txt", "../out.txt")


def test_aio_local_artifact_dir_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    ts = sandbox_toolset(
        "aio-template",
        template_type=TemplateType.AIO,
        local_artifact_dir=str(tmp_path),
    )

    with pytest.raises(ValueError, match="local_artifact_path escapes"):
        ts.download_file("/tmp/a.txt", "link/out.txt")


def test_aio_upload_download_use_file_system_operations(tmp_path, monkeypatch):
    local_source = tmp_path / "source.txt"
    local_source.write_text("upload", encoding="utf-8")

    upload = MagicMock(return_value={"uploaded": True})
    download = MagicMock(return_value={"downloaded": True})
    sb = SimpleNamespace(
        sandbox_id="sandbox-aio-123",
        check_health=lambda: {"status": "ok"},
        context=object(),
        file=SimpleNamespace(read=lambda path: "", write=lambda **kwargs: {}),
        file_system=SimpleNamespace(upload=upload, download=download),
        process=object(),
        sync_playwright=lambda: object(),
        get_cdp_url=lambda **kwargs: "",
        get_vnc_url=lambda **kwargs: "",
        list_recordings=lambda: [],
        download_recording=lambda **kwargs: {},
        delete_recording=lambda filename: {},
    )
    ts = sandbox_toolset(
        "aio-template",
        template_type=TemplateType.AIO,
        local_artifact_dir=str(tmp_path),
    )
    monkeypatch.setattr(ts, "_ensure_sandbox", lambda: sb)

    assert ts.upload_file("source.txt", "/tmp/source.txt")["success"] is True
    assert (
        ts.download_file("/tmp/source.txt", "downloaded.txt")["success"]
        is True
    )
    upload.assert_called_once_with(
        local_file_path=str(local_source.resolve()),
        target_file_path="/tmp/source.txt",
    )
    download.assert_called_once_with(
        path="/tmp/source.txt",
        save_path=str((tmp_path / "downloaded.txt").resolve()),
    )


def test_aio_auto_artifact_dir_is_cleaned_on_close():
    ts = sandbox_toolset("aio-template", template_type=TemplateType.AIO)
    artifact_dir = Path(ts.local_artifact_dir)

    assert artifact_dir.exists()

    ts.close()

    assert not artifact_dir.exists()


def test_aio_explicit_artifact_dir_is_not_cleaned_on_close(tmp_path):
    ts = sandbox_toolset(
        "aio-template",
        template_type=TemplateType.AIO,
        local_artifact_dir=str(tmp_path),
    )

    ts.close()

    assert tmp_path.exists()


def test_aio_remote_url_tools_do_not_expose_headers():
    fields = sandbox_builtin.AioToolSet.browser_get_cdp_url.args_schema.model_fields
    assert "with_headers" not in fields

    ts = sandbox_toolset("aio-template", template_type=TemplateType.AIO)
    mock_sandbox = MagicMock()
    mock_sandbox.get_cdp_url.return_value = "wss://example.invalid/cdp"
    mock_sandbox.get_vnc_url.return_value = "wss://example.invalid/vnc"
    ts.sandbox = mock_sandbox
    ts.sandbox_id = "sandbox-aio-123"

    assert ts.browser_get_cdp_url(record=True) == {
        "url": "wss://example.invalid/cdp",
        "record": True,
    }
    assert ts.browser_get_vnc_url(record=False) == {
        "url": "wss://example.invalid/vnc",
        "record": False,
    }
    mock_sandbox.get_cdp_url.assert_called_once_with(
        record=True, with_headers=False
    )
    mock_sandbox.get_vnc_url.assert_called_once_with(
        record=False, with_headers=False
    )


@pytest.mark.parametrize(
    ("module_name", "convert_method"),
    [
        ("agentrun.integration.langchain.builtin", "to_langchain"),
        ("agentrun.integration.langgraph.builtin", "to_langgraph"),
        ("agentrun.integration.google_adk.builtin", "to_google_adk"),
        ("agentrun.integration.pydantic_ai.builtin", "to_pydantic_ai"),
        ("agentrun.integration.crewai.builtin", "to_crewai"),
        ("agentrun.integration.agentscope.builtin", "to_agentscope"),
    ],
)
def test_public_sandbox_toolset_wrappers_pass_local_artifact_dir(
    module_name, convert_method, tmp_path
):
    module = importlib.import_module(module_name)
    mock_toolset = MagicMock()
    getattr(mock_toolset, convert_method).return_value = []

    with patch.object(module, "_sandbox_toolset", return_value=mock_toolset) as fn:
        module.sandbox_toolset(
            "aio-template",
            template_type=TemplateType.AIO,
            local_artifact_dir=str(tmp_path),
        )

    assert fn.call_args.kwargs["local_artifact_dir"] == str(tmp_path)


def test_existing_browser_and_code_interpreter_factory_behavior_stays_same():
    assert isinstance(
        sandbox_toolset("browser-template", template_type=TemplateType.BROWSER),
        BrowserToolSet,
    )
    assert isinstance(
        sandbox_toolset(
            "code-template", template_type=TemplateType.CODE_INTERPRETER
        ),
        CodeInterpreterToolSet,
    )
