"""MCP 协议处理单元测试 / MCP Protocol Handler Unit Tests

测试 MCP 协议相关功能。
Tests MCP protocol functionality.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.toolset.api.mcp import MCPSession, MCPToolSet
from agentrun.utils.config import Config


class TestMCPToolSetInit:
    """测试 MCPToolSet 初始化"""

    def test_init_basic(self):
        """测试基本初始化"""
        # 正常情况下，mcp 包已安装，不会记录警告
        toolset = MCPToolSet(url="https://mcp.example.com")
        assert toolset.url == "https://mcp.example.com"

    def test_init_with_url(self):
        """测试带 URL 初始化"""
        toolset = MCPToolSet(url="https://mcp.example.com")
        assert toolset.url == "https://mcp.example.com"

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = Config(
            access_key_id="test-key",
            access_key_secret="test-secret",
        )
        toolset = MCPToolSet(
            url="https://mcp.example.com",
            config=config,
        )
        assert toolset.url == "https://mcp.example.com"
        assert toolset.config is not None


class TestMCPToolSetNewSession:
    """测试 MCPToolSet.new_session 方法"""

    def test_new_session(self):
        """测试创建新会话"""
        toolset = MCPToolSet(url="https://mcp.example.com")
        session = toolset.new_session()
        assert isinstance(session, MCPSession)
        assert session.url == "https://mcp.example.com"

    def test_new_session_with_config(self):
        """测试带配置创建新会话"""
        toolset = MCPToolSet(url="https://mcp.example.com")
        config = Config(timeout=120)
        session = toolset.new_session(config=config)
        assert isinstance(session, MCPSession)


class TestMCPSession:
    """测试 MCPSession 类"""

    def test_session_init(self):
        """测试会话初始化"""
        session = MCPSession(url="https://mcp.example.com")
        assert session.url == "https://mcp.example.com"
        assert session.config is not None

    def test_session_init_with_config(self):
        """测试带配置初始化会话"""
        config = Config(timeout=60)
        session = MCPSession(url="https://mcp.example.com", config=config)
        assert session.config is not None

    def test_toolsets_method(self):
        """测试 toolsets 方法"""
        session = MCPSession(url="https://mcp.example.com")
        toolset = session.toolsets()
        assert isinstance(toolset, MCPToolSet)
        assert toolset.url == "https://mcp.example.com/toolsets"


class TestMCPToolSetTools:
    """测试 MCPToolSet.tools 方法"""

    @patch("agentrun.toolset.api.mcp.MCPToolSet.tools_async")
    def test_tools_sync(self, mock_tools_async):
        """测试同步获取工具列表"""
        mock_tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]

        with patch("asyncio.run", return_value=mock_tools) as mock_asyncio_run:
            toolset = MCPToolSet(url="https://mcp.example.com")
            result = toolset.tools()

            assert result == mock_tools
            mock_asyncio_run.assert_called_once()


class TestMCPToolSetCallTool:
    """测试 MCPToolSet.call_tool 方法"""

    def test_call_tool_sync(self):
        """测试同步调用工具"""
        mock_result = [{"type": "text", "text": "result"}]

        with patch("asyncio.run", return_value=mock_result) as mock_asyncio_run:
            toolset = MCPToolSet(url="https://mcp.example.com")
            result = toolset.call_tool("my_tool", {"arg": "value"})

            assert result == mock_result
            mock_asyncio_run.assert_called_once()

    def test_call_tool_sync_with_config(self):
        """测试带配置同步调用工具"""
        mock_result = [{"type": "text", "text": "result"}]

        with patch("asyncio.run", return_value=mock_result) as mock_asyncio_run:
            config = Config(timeout=120)
            toolset = MCPToolSet(url="https://mcp.example.com")
            result = toolset.call_tool(
                "my_tool", {"arg": "value"}, config=config
            )

            assert result == mock_result
