"""Tool MCP 会话单元测试 / Tool MCP Session Unit Tests

测试 ToolMCPSession 的 MCP 协议交互功能。
Tests MCP protocol interaction functionality of ToolMCPSession.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from agentrun.tool.api.mcp import ToolMCPSession
from agentrun.tool.model import ToolInfo


class TestToolMCPSessionInit:
    """测试 ToolMCPSession 初始化 / Test ToolMCPSession initialization"""

    def test_init_with_defaults(self):
        """测试使用默认参数初始化"""
        session = ToolMCPSession(endpoint="http://example.com/mcp")
        assert session.endpoint == "http://example.com/mcp"
        assert session.session_affinity is None
        assert session.headers == {}

    def test_init_with_all_parameters(self):
        """测试使用所有参数初始化"""
        headers = {"Authorization": "Bearer token"}
        session = ToolMCPSession(
            endpoint="http://example.com/mcp",
            session_affinity="MCP_STREAMABLE",
            headers=headers,
        )
        assert session.endpoint == "http://example.com/mcp"
        assert session.session_affinity == "MCP_STREAMABLE"
        assert session.headers == headers


class TestToolMCPSessionIsStreamable:
    """测试 is_streamable 属性"""

    def test_is_streamable_returns_true_for_mcp_streamable(self):
        """测试 MCP_STREAMABLE 返回 True"""
        session = ToolMCPSession(
            endpoint="http://example.com/mcp",
            session_affinity="MCP_STREAMABLE",
        )
        assert session.is_streamable is True

    def test_is_streamable_returns_false_for_other_values(self):
        """测试其他值返回 False"""
        for value in [None, "MCP_SSE", "OTHER", ""]:
            session = ToolMCPSession(
                endpoint="http://example.com/mcp",
                session_affinity=value,
            )
            assert session.is_streamable is False


def _make_mock_mcp_tool(name: str, description: str) -> MagicMock:
    """创建 mock MCP tool 对象"""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = {"type": "object", "properties": {}}
    return tool


def _setup_mock_mcp_modules(
    mock_session: AsyncMock,
) -> dict:
    """设置 mock mcp 模块，返回需要注入到 sys.modules 的字典"""
    mock_client_session_cls = MagicMock()
    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__.return_value = mock_session
    mock_session_ctx.__aexit__.return_value = None
    mock_client_session_cls.return_value = mock_session_ctx

    # mock streamablehttp_client
    mock_streamable_fn = MagicMock()
    mock_streamable_ctx = AsyncMock()
    mock_streamable_ctx.__aenter__.return_value = (
        AsyncMock(),
        AsyncMock(),
        MagicMock(),
    )
    mock_streamable_ctx.__aexit__.return_value = None
    mock_streamable_fn.return_value = mock_streamable_ctx

    # mock sse_client
    mock_sse_fn = MagicMock()
    mock_sse_ctx = AsyncMock()
    mock_sse_ctx.__aenter__.return_value = (AsyncMock(), AsyncMock())
    mock_sse_ctx.__aexit__.return_value = None
    mock_sse_fn.return_value = mock_sse_ctx

    mock_mcp = MagicMock()
    mock_mcp.ClientSession = mock_client_session_cls

    mock_mcp_client_streamable = MagicMock()
    mock_mcp_client_streamable.streamablehttp_client = mock_streamable_fn

    mock_mcp_client_sse = MagicMock()
    mock_mcp_client_sse.sse_client = mock_sse_fn

    return {
        "mcp": mock_mcp,
        "mcp.client": MagicMock(),
        "mcp.client.streamable_http": mock_mcp_client_streamable,
        "mcp.client.sse": mock_mcp_client_sse,
    }


class TestToolMCPSessionListToolsAsync:
    """测试 list_tools_async 方法"""

    @pytest.mark.asyncio
    async def test_list_tools_async_streamable_mode(self):
        """测试 Streamable 模式下获取工具列表"""
        mock_tool = _make_mock_mcp_tool("tool1", "Test tool 1")
        mock_result = MagicMock()
        mock_result.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_result)

        mock_modules = _setup_mock_mcp_modules(mock_session)

        with patch.dict(sys.modules, mock_modules):
            session = ToolMCPSession(
                endpoint="http://example.com/mcp",
                session_affinity="MCP_STREAMABLE",
                headers={"Authorization": "Bearer token"},
            )
            tools = await session.list_tools_async()

        assert len(tools) == 1
        assert isinstance(tools[0], ToolInfo)
        assert tools[0].name == "tool1"

    @pytest.mark.asyncio
    async def test_list_tools_async_sse_mode(self):
        """测试 SSE 模式下获取工具列表"""
        mock_tool = _make_mock_mcp_tool("tool1", "Test tool 1")
        mock_result = MagicMock()
        mock_result.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_result)

        mock_modules = _setup_mock_mcp_modules(mock_session)

        with patch.dict(sys.modules, mock_modules):
            session = ToolMCPSession(
                endpoint="http://example.com/mcp",
                session_affinity="MCP_SSE",
            )
            tools = await session.list_tools_async()

        assert len(tools) == 1
        assert isinstance(tools[0], ToolInfo)

    @pytest.mark.asyncio
    async def test_list_tools_async_import_error(self):
        """测试 mcp 未安装时返回空列表"""
        saved_modules = {}
        modules_to_remove = [
            k for k in sys.modules if k == "mcp" or k.startswith("mcp.")
        ]
        for key in modules_to_remove:
            saved_modules[key] = sys.modules.pop(key)

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore

        def mock_import(name, *args, **kwargs):
            if name == "mcp" or name.startswith("mcp."):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            session = ToolMCPSession(endpoint="http://example.com/mcp")
            tools = await session.list_tools_async()

        sys.modules.update(saved_modules)
        assert tools == []


class TestToolMCPSessionListTools:
    """测试 list_tools 同步方法"""

    def test_list_tools_synchronous(self):
        """测试同步获取工具列表"""
        expected_tools = [ToolInfo(name="tool1", description="Test")]

        with patch.object(
            ToolMCPSession,
            "list_tools_async",
            new_callable=AsyncMock,
            return_value=expected_tools,
        ):
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_loop.run_until_complete.return_value = expected_tools
                mock_get_loop.return_value = mock_loop

                session = ToolMCPSession(endpoint="http://example.com/mcp")
                tools = session.list_tools()

                assert tools == expected_tools
                mock_loop.run_until_complete.assert_called_once()


class TestToolMCPSessionCallToolAsync:
    """测试 call_tool_async 方法"""

    @pytest.mark.asyncio
    async def test_call_tool_async_streamable_mode(self):
        """测试 Streamable 模式下调用工具"""
        mock_call_result = MagicMock()
        mock_call_result.content = [{"type": "text", "text": "result"}]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        mock_modules = _setup_mock_mcp_modules(mock_session)

        with patch.dict(sys.modules, mock_modules):
            session = ToolMCPSession(
                endpoint="http://example.com/mcp",
                session_affinity="MCP_STREAMABLE",
            )
            result = await session.call_tool_async(
                "test_tool", {"param": "value"}
            )

        assert result == mock_call_result

    @pytest.mark.asyncio
    async def test_call_tool_async_sse_mode(self):
        """测试 SSE 模式下调用工具"""
        mock_call_result = MagicMock()

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        mock_modules = _setup_mock_mcp_modules(mock_session)

        with patch.dict(sys.modules, mock_modules):
            session = ToolMCPSession(
                endpoint="http://example.com/mcp",
                session_affinity="MCP_SSE",
            )
            result = await session.call_tool_async("test_tool", {"key": "val"})

        assert result == mock_call_result

    @pytest.mark.asyncio
    async def test_call_tool_async_import_error(self):
        """测试 mcp 未安装时抛出 ImportError"""
        saved_modules = {}
        modules_to_remove = [
            k for k in sys.modules if k == "mcp" or k.startswith("mcp.")
        ]
        for key in modules_to_remove:
            saved_modules[key] = sys.modules.pop(key)

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore

        def mock_import(name, *args, **kwargs):
            if name == "mcp" or name.startswith("mcp."):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            session = ToolMCPSession(endpoint="http://example.com/mcp")
            with pytest.raises(ImportError):
                await session.call_tool_async("test_tool")

        sys.modules.update(saved_modules)


class TestToolMCPSessionCallTool:
    """测试 call_tool 同步方法"""

    def test_call_tool_synchronous(self):
        """测试同步调用工具"""
        expected_result = {"result": "success"}

        with patch.object(
            ToolMCPSession,
            "call_tool_async",
            new_callable=AsyncMock,
            return_value=expected_result,
        ):
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_loop.run_until_complete.return_value = expected_result
                mock_get_loop.return_value = mock_loop

                session = ToolMCPSession(endpoint="http://example.com/mcp")
                result = session.call_tool("test_tool", {"param": "value"})

                assert result == expected_result
                mock_loop.run_until_complete.assert_called_once()
