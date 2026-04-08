"""CodeInterpreterToolSet read_file 工具单元测试

测试 read_file 工具的 base64 编码行为和 raw 参数控制。
Tests the read_file tool's base64 encoding behavior and the raw parameter control.
"""

import base64
import threading
from unittest.mock import MagicMock, patch

import pytest

from agentrun.integration.builtin.sandbox import CodeInterpreterToolSet


@pytest.fixture
def toolset():
    """创建 CodeInterpreterToolSet 实例，绕过 __init__ / Create instance bypassing __init__."""
    with patch.object(CodeInterpreterToolSet, "__init__", lambda self: None):
        ts = CodeInterpreterToolSet()
        ts.sandbox = None
        ts.sandbox_id = ""
        ts._lock = threading.Lock()
        ts.template_name = "test-tpl"
        ts.template_type = MagicMock()
        ts.sandbox_idle_timeout_seconds = 600
        ts.config = None
        ts.oss_mount_config = None
        ts.nas_config = None
        ts.polar_fs_config = None
        return ts


def _make_mock_sandbox(file_content: str):
    """构造一个模拟沙箱，其 file.read 返回指定内容 / Build mock sandbox with file.read returning given content."""
    from agentrun.sandbox.code_interpreter_sandbox import CodeInterpreterSandbox

    mock_sb = MagicMock(spec=CodeInterpreterSandbox)
    mock_sb.file.read.return_value = file_content
    return mock_sb


class TestReadFileBase64Default:
    """测试 read_file 默认返回 base64 编码内容 / Test that read_file returns base64 by default."""

    def test_returns_base64_encoded_content(self, toolset):
        """默认情况下内容应为 base64 编码 / Content should be base64 encoded by default."""
        file_content = "hello world"
        mock_sb = _make_mock_sandbox(file_content)

        with patch.object(toolset, "_run_in_sandbox", side_effect=lambda fn: fn(mock_sb)):
            result = toolset.read_file(path="/tmp/test.txt")

        expected_b64 = base64.b64encode(b"hello world").decode("ascii")
        assert result["content"] == expected_b64
        assert result["encoding"] == "base64"
        assert result["path"] == "/tmp/test.txt"

    def test_base64_roundtrip(self, toolset):
        """base64 解码后应等于原始内容 / Decoded base64 should equal original content."""
        file_content = "中文内容 line1\nline2"
        mock_sb = _make_mock_sandbox(file_content)

        with patch.object(toolset, "_run_in_sandbox", side_effect=lambda fn: fn(mock_sb)):
            result = toolset.read_file(path="/tmp/utf8.txt")

        decoded = base64.b64decode(result["content"]).decode("utf-8")
        assert decoded == file_content

    def test_bytes_content_also_base64_encoded(self, toolset):
        """当底层返回 bytes 时同样应 base64 编码 / Bytes content should also be base64 encoded."""
        file_bytes = b"\x00\x01\x02\x03"
        from agentrun.sandbox.code_interpreter_sandbox import CodeInterpreterSandbox

        mock_sb = MagicMock(spec=CodeInterpreterSandbox)
        mock_sb.file.read.return_value = file_bytes

        with patch.object(toolset, "_run_in_sandbox", side_effect=lambda fn: fn(mock_sb)):
            result = toolset.read_file(path="/tmp/binary.bin")

        expected_b64 = base64.b64encode(file_bytes).decode("ascii")
        assert result["content"] == expected_b64
        assert result["encoding"] == "base64"


class TestReadFileRawParam:
    """测试 raw=True 时返回原始内容 / Test that raw=True returns plain text content."""

    def test_raw_true_returns_plain_content(self, toolset):
        """raw=True 时应返回原始文本 / raw=True should return raw text."""
        file_content = "plain text content"
        mock_sb = _make_mock_sandbox(file_content)

        with patch.object(toolset, "_run_in_sandbox", side_effect=lambda fn: fn(mock_sb)):
            result = toolset.read_file(path="/tmp/plain.txt", raw=True)

        assert result["content"] == file_content
        assert result["encoding"] == "raw"
        assert result["path"] == "/tmp/plain.txt"

    def test_raw_false_same_as_default(self, toolset):
        """raw=False 应与默认行为一致 / raw=False should behave identically to default."""
        file_content = "some content"
        mock_sb = _make_mock_sandbox(file_content)

        with patch.object(toolset, "_run_in_sandbox", side_effect=lambda fn: fn(mock_sb)):
            result_explicit = toolset.read_file(path="/tmp/f.txt", raw=False)

        mock_sb2 = _make_mock_sandbox(file_content)
        with patch.object(toolset, "_run_in_sandbox", side_effect=lambda fn: fn(mock_sb2)):
            result_default = toolset.read_file(path="/tmp/f.txt")

        assert result_explicit == result_default
        assert result_explicit["encoding"] == "base64"
