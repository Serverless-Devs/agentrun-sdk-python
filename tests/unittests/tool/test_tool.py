"""Tool 资源类和客户端单元测试 / Tool Resource Class and Client Unit Tests

测试 Tool 资源类和 ToolClient 的功能。
Tests functionality of Tool resource class and ToolClient.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentrun.tool.client import ToolClient
from agentrun.tool.model import (
    McpConfig,
    ToolCodeConfiguration,
    ToolContainerConfiguration,
    ToolInfo,
    ToolLogConfiguration,
    ToolNetworkConfiguration,
    ToolOSSMountConfig,
    ToolSchema,
    ToolType,
)
from agentrun.tool.tool import Tool


class TestTool:
    """测试 Tool 资源类"""

    def test_tool_attributes_default(self):
        """测试 Tool 默认属性"""
        tool = Tool()
        assert tool.tool_id is None
        assert tool.name is None
        assert tool.tool_name is None
        assert tool.description is None
        assert tool.tool_type is None
        assert tool.status is None
        assert tool.code_configuration is None
        assert tool.container_configuration is None
        assert tool.mcp_config is None
        assert tool.log_configuration is None
        assert tool.network_config is None
        assert tool.oss_mount_config is None
        assert tool.data_endpoint is None
        assert tool.protocol_spec is None
        assert tool.protocol_type is None
        assert tool.memory is None
        assert tool.gpu is None
        assert tool.timeout is None
        assert tool.internet_access is None
        assert tool.environment_variables is None
        assert tool.created_time is None
        assert tool.last_modified_time is None
        assert tool.version_id is None

    def test_tool_attributes_with_values(self):
        """测试 Tool 带值创建"""
        tool = Tool(
            tool_id="tool-123",
            name="my-tool",
            tool_name="my-tool",
            description="A test tool",
            tool_type="MCP",
            status="READY",
            data_endpoint="https://example.com/data",
            memory=1024,
            gpu="T4",
            timeout=60,
            internet_access=True,
            environment_variables={"KEY": "value"},
        )
        assert tool.tool_id == "tool-123"
        assert tool.name == "my-tool"
        assert tool.tool_name == "my-tool"
        assert tool.description == "A test tool"
        assert tool.tool_type == "MCP"
        assert tool.status == "READY"
        assert tool.data_endpoint == "https://example.com/data"
        assert tool.memory == 1024
        assert tool.gpu == "T4"
        assert tool.timeout == 60
        assert tool.internet_access is True
        assert tool.environment_variables == {"KEY": "value"}

    def test_get_tool_type_mcp(self):
        """测试获取 MCP 工具类型"""
        tool = Tool(tool_type="MCP")
        assert tool._get_tool_type() == ToolType.MCP

    def test_get_tool_type_functioncall(self):
        """测试获取 FUNCTIONCALL 工具类型"""
        tool = Tool(tool_type="FUNCTIONCALL")
        assert tool._get_tool_type() == ToolType.FUNCTIONCALL

    def test_get_tool_type_invalid(self):
        """测试获取无效工具类型"""
        tool = Tool(tool_type="INVALID")
        assert tool._get_tool_type() is None

    def test_get_tool_type_none(self):
        """测试获取 None 工具类型"""
        tool = Tool()
        assert tool._get_tool_type() is None

    def test_get_mcp_endpoint_sse(self):
        """测试获取 MCP SSE endpoint"""
        tool = Tool(
            tool_name="my-tool",
            data_endpoint="https://example.com",
            mcp_config=McpConfig(session_affinity="MCP_SSE"),
        )
        endpoint = tool._get_mcp_endpoint()
        assert endpoint == "https://example.com/tools/my-tool/sse"

    def test_get_mcp_endpoint_streamable(self):
        """测试获取 MCP Streamable endpoint"""
        tool = Tool(
            tool_name="my-tool",
            data_endpoint="https://example.com",
            mcp_config=McpConfig(session_affinity="MCP_STREAMABLE"),
        )
        endpoint = tool._get_mcp_endpoint()
        assert endpoint == "https://example.com/tools/my-tool/mcp"

    def test_get_mcp_endpoint_default(self):
        """测试获取 MCP endpoint（默认 SSE）"""
        tool = Tool(
            tool_name="my-tool",
            data_endpoint="https://example.com",
        )
        endpoint = tool._get_mcp_endpoint()
        assert endpoint == "https://example.com/tools/my-tool/sse"

    def test_get_mcp_endpoint_no_name(self):
        """测试没有 name 时获取 MCP endpoint"""
        tool = Tool(
            data_endpoint="https://example.com",
        )
        endpoint = tool._get_mcp_endpoint()
        assert endpoint is None

    def test_get_mcp_endpoint_no_data_endpoint(self):
        """测试没有 data_endpoint 时获取 MCP endpoint"""
        tool = Tool(
            tool_name="my-tool",
        )
        endpoint = tool._get_mcp_endpoint()
        assert endpoint is None

    def test_from_inner_object(self):
        """测试从内部对象创建 Tool"""
        inner_tool = Mock()
        inner_tool.tool_id = "tool-123"
        inner_tool.name = "my-tool"
        inner_tool.description = "Test tool"
        inner_tool.tool_type = "MCP"
        inner_tool.status = "READY"
        inner_tool.data_endpoint = "https://example.com/data"
        inner_tool.memory = 1024
        inner_tool.gpu = "T4"
        inner_tool.timeout = 60
        inner_tool.internet_access = True
        inner_tool.environment_variables = {"KEY": "value"}
        inner_tool.created_time = "2024-01-01T00:00:00Z"
        inner_tool.last_modified_time = "2024-01-02T00:00:00Z"
        inner_tool.version_id = "version-123"
        inner_tool.protocol_spec = '{"openapi": "3.0.0"}'
        inner_tool.protocol_type = "openapi"

        # Mock configurations
        inner_tool.code_configuration = None
        inner_tool.container_configuration = None
        inner_tool.mcp_config = None
        inner_tool.log_configuration = None
        inner_tool.network_config = None
        inner_tool.oss_mount_config = None

        # Mock to_map method
        inner_tool.to_map = Mock(
            return_value={
                "toolId": "tool-123",
                "name": "my-tool",
                "description": "Test tool",
                "toolType": "MCP",
                "status": "READY",
                "dataEndpoint": "https://example.com/data",
                "memory": 1024,
                "gpu": "T4",
                "timeout": 60,
                "internetAccess": True,
                "environmentVariables": {"KEY": "value"},
                "createdTime": "2024-01-01T00:00:00Z",
                "lastModifiedTime": "2024-01-02T00:00:00Z",
                "versionId": "version-123",
                "protocolSpec": '{"openapi": "3.0.0"}',
                "protocolType": "openapi",
            }
        )

        tool = Tool.from_inner_object(inner_tool)

        assert tool.tool_id == "tool-123"
        assert tool.name == "my-tool"
        assert tool.description == "Test tool"
        assert tool.tool_type == "MCP"
        assert tool.status == "READY"
        assert tool.data_endpoint == "https://example.com/data"
        assert tool.memory == 1024
        assert tool.gpu == "T4"
        assert tool.timeout == 60
        assert tool.internet_access is True
        assert tool.environment_variables == {"KEY": "value"}
        assert tool.created_time == "2024-01-01T00:00:00Z"
        assert tool.last_modified_time == "2024-01-02T00:00:00Z"
        assert tool.version_id == "version-123"
        assert tool.protocol_spec == '{"openapi": "3.0.0"}'
        assert tool.protocol_type == "openapi"

    @patch("agentrun.tool.api.mcp.ToolMCPSession")
    @patch("agentrun.utils.config.Config")
    def test_list_tools_mcp(self, mock_config_class, mock_mcp_session_class):
        """测试获取 MCP 工具列表"""
        mock_session = Mock()
        mock_session.list_tools.return_value = [
            ToolInfo(name="tool1", description="Tool 1"),
            ToolInfo(name="tool2", description="Tool 2"),
        ]
        mock_mcp_session_class.return_value = mock_session

        mock_config = Mock()
        mock_config.get_headers.return_value = {}
        mock_config_class.with_configs.return_value = mock_config

        tool = Tool(
            tool_name="my-tool",
            tool_type="MCP",
            data_endpoint="https://example.com",
            mcp_config=McpConfig(session_affinity="MCP_SSE"),
        )

        tools = tool.list_tools()

        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"

    @patch("agentrun.tool.api.openapi.ToolOpenAPIClient")
    def test_list_tools_functioncall(self, mock_openapi_client_class):
        """测试获取 FUNCTIONCALL 工具列表"""
        mock_client = Mock()
        mock_client.list_tools.return_value = [
            ToolInfo(name="tool1", description="Tool 1"),
            ToolInfo(name="tool2", description="Tool 2"),
        ]
        mock_openapi_client_class.return_value = mock_client

        tool = Tool(
            tool_type="FUNCTIONCALL",
            protocol_spec='{"openapi": "3.0.0"}',
        )

        tools = tool.list_tools()

        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"

    def test_list_tools_no_type(self):
        """测试没有工具类型时获取工具列表"""
        tool = Tool()
        tools = tool.list_tools()
        assert tools == []

    @patch("agentrun.tool.api.mcp.ToolMCPSession")
    @patch("agentrun.utils.config.Config")
    def test_call_tool_mcp(self, mock_config_class, mock_mcp_session_class):
        """测试调用 MCP 工具"""
        mock_session = Mock()
        mock_session.call_tool.return_value = {"result": "success"}
        mock_mcp_session_class.return_value = mock_session

        mock_config = Mock()
        mock_config.get_headers.return_value = {}
        mock_config_class.with_configs.return_value = mock_config

        tool = Tool(
            tool_name="my-tool",
            tool_type="MCP",
            data_endpoint="https://example.com",
            mcp_config=McpConfig(session_affinity="MCP_SSE"),
        )

        result = tool.call_tool("tool1", {"param": "value"})

        assert result == {"result": "success"}

    @patch("agentrun.tool.api.openapi.ToolOpenAPIClient")
    @patch("agentrun.utils.config.Config")
    def test_call_tool_functioncall(
        self, mock_config_class, mock_openapi_client_class
    ):
        """测试调用 FUNCTIONCALL 工具"""
        mock_client = Mock()
        mock_client.call_tool.return_value = {"result": "success"}
        mock_openapi_client_class.return_value = mock_client

        mock_config = Mock()
        mock_config.get_headers.return_value = {}
        mock_config_class.with_configs.return_value = mock_config

        tool = Tool(
            tool_type="FUNCTIONCALL",
            protocol_spec='{"openapi": "3.0.0"}',
        )

        result = tool.call_tool("tool1", {"param": "value"})

        assert result == {"result": "success"}

    def test_call_tool_unsupported_type(self):
        """测试调用不支持的类型工具"""
        tool = Tool(tool_type="UNSUPPORTED")
        with pytest.raises(ValueError, match="Unsupported tool type"):
            tool.call_tool("tool1", {})

    @patch("agentrun.tool.api.mcp.ToolMCPSession")
    @patch("agentrun.utils.config.Config")
    async def test_list_tools_async_mcp(
        self, mock_config_class, mock_mcp_session_class
    ):
        """测试异步获取 MCP 工具列表"""
        mock_session = Mock()
        mock_session.list_tools_async = AsyncMock(
            return_value=[
                ToolInfo(name="tool1", description="Tool 1"),
            ]
        )
        mock_mcp_session_class.return_value = mock_session

        mock_config = Mock()
        mock_config.get_headers.return_value = {}
        mock_config_class.with_configs.return_value = mock_config

        tool = Tool(
            tool_name="my-tool",
            tool_type="MCP",
            data_endpoint="https://example.com",
            mcp_config=McpConfig(session_affinity="MCP_SSE"),
        )

        tools = await tool.list_tools_async()

        assert len(tools) == 1
        assert tools[0].name == "tool1"

    @patch("agentrun.tool.api.mcp.ToolMCPSession")
    @patch("agentrun.utils.config.Config")
    async def test_call_tool_async_mcp(
        self, mock_config_class, mock_mcp_session_class
    ):
        """测试异步调用 MCP 工具"""
        mock_session = Mock()
        mock_session.call_tool_async = AsyncMock(
            return_value={"result": "success"}
        )
        mock_mcp_session_class.return_value = mock_session

        mock_config = Mock()
        mock_config.get_headers.return_value = {}
        mock_config_class.with_configs.return_value = mock_config

        tool = Tool(
            tool_name="my-tool",
            tool_type="MCP",
            data_endpoint="https://example.com",
            mcp_config=McpConfig(session_affinity="MCP_SSE"),
        )

        result = await tool.call_tool_async("tool1", {"param": "value"})

        assert result == {"result": "success"}


class TestToolClient:
    """测试 ToolClient"""

    def test_client_init(self):
        """测试客户端初始化"""
        client = ToolClient()
        assert client is not None

    @patch("agentrun.tool.client.ToolControlAPI")
    def test_get(self, mock_control_api_class):
        """测试获取工具"""
        # Mock inner tool
        inner_tool = Mock()
        inner_tool.tool_id = "tool-123"
        inner_tool.name = "my-tool"
        inner_tool.description = "Test tool"
        inner_tool.tool_type = "MCP"
        inner_tool.status = "READY"
        inner_tool.data_endpoint = "https://example.com/data"
        inner_tool.memory = 1024
        inner_tool.gpu = None
        inner_tool.timeout = 60
        inner_tool.internet_access = True
        inner_tool.environment_variables = None
        inner_tool.created_time = None
        inner_tool.last_modified_time = None
        inner_tool.version_id = None
        inner_tool.protocol_spec = None
        inner_tool.protocol_type = None
        inner_tool.code_configuration = None
        inner_tool.container_configuration = None
        inner_tool.mcp_config = None
        inner_tool.log_configuration = None
        inner_tool.network_config = None
        inner_tool.oss_mount_config = None

        # Mock to_map method
        inner_tool.to_map = Mock(
            return_value={
                "toolId": "tool-123",
                "name": "my-tool",
                "description": "Test tool",
                "toolType": "MCP",
                "status": "READY",
                "dataEndpoint": "https://example.com/data",
                "memory": 1024,
                "timeout": 60,
                "internetAccess": True,
            }
        )

        mock_api = Mock()
        mock_api.get_tool.return_value = inner_tool
        mock_control_api_class.return_value = mock_api

        client = ToolClient()
        tool = client.get(name="my-tool")

        assert tool.tool_id == "tool-123"
        assert tool.name == "my-tool"
        assert tool.tool_type == "MCP"
        mock_api.get_tool.assert_called_once_with(name="my-tool", config=None)

    @patch("agentrun.tool.client.ToolControlAPI")
    async def test_get_async(self, mock_control_api_class):
        """测试异步获取工具"""
        # Mock inner tool
        inner_tool = Mock()
        inner_tool.tool_id = "tool-123"
        inner_tool.name = "my-tool"
        inner_tool.description = "Test tool"
        inner_tool.tool_type = "MCP"
        inner_tool.status = "READY"
        inner_tool.data_endpoint = "https://example.com/data"
        inner_tool.memory = 1024
        inner_tool.gpu = None
        inner_tool.timeout = 60
        inner_tool.internet_access = True
        inner_tool.environment_variables = None
        inner_tool.created_time = None
        inner_tool.last_modified_time = None
        inner_tool.version_id = None
        inner_tool.protocol_spec = None
        inner_tool.protocol_type = None
        inner_tool.code_configuration = None
        inner_tool.container_configuration = None
        inner_tool.mcp_config = None
        inner_tool.log_configuration = None
        inner_tool.network_config = None
        inner_tool.oss_mount_config = None

        # Mock to_map method
        inner_tool.to_map = Mock(
            return_value={
                "toolId": "tool-123",
                "name": "my-tool",
                "description": "Test tool",
                "toolType": "MCP",
                "status": "READY",
                "dataEndpoint": "https://example.com/data",
                "memory": 1024,
                "timeout": 60,
                "internetAccess": True,
            }
        )

        mock_api = Mock()
        mock_api.get_tool_async = AsyncMock(return_value=inner_tool)
        mock_control_api_class.return_value = mock_api

        client = ToolClient()
        tool = await client.get_async(name="my-tool")

        assert tool.tool_id == "tool-123"
        assert tool.name == "my-tool"
        assert tool.tool_type == "MCP"
        mock_api.get_tool_async.assert_called_once_with(
            name="my-tool", config=None
        )

    @patch("agentrun.tool.client.ToolControlAPI")
    def test_get_http_error(self, mock_control_api_class):
        """测试 get() 遇到 HTTPError 时的异常转换"""
        from agentrun.utils.exception import HTTPError

        mock_resource_error = Exception("Resource not found")
        mock_resource_error.message = "Resource not found"  # type: ignore
        mock_resource_error.error_code = "ResourceNotFound"  # type: ignore

        mock_http_error = HTTPError.__new__(HTTPError)
        mock_http_error.to_resource_error = Mock(return_value=mock_resource_error)  # type: ignore

        mock_api = Mock()
        mock_api.get_tool.side_effect = mock_http_error
        mock_control_api_class.return_value = mock_api

        client = ToolClient()

        with pytest.raises(Exception) as exc_info:
            client.get(name="my-tool")
        assert exc_info.value.message == "Resource not found"  # type: ignore

    @patch("agentrun.tool.client.ToolControlAPI")
    async def test_get_async_http_error(self, mock_control_api_class):
        """测试 get_async() 遇到 HTTPError 时的异常转换"""
        from agentrun.utils.exception import HTTPError

        mock_resource_error = Exception("Resource not found")
        mock_resource_error.message = "Resource not found"  # type: ignore
        mock_resource_error.error_code = "ResourceNotFound"  # type: ignore

        mock_http_error = HTTPError.__new__(HTTPError)
        mock_http_error.to_resource_error = Mock(return_value=mock_resource_error)  # type: ignore

        mock_api = Mock()
        mock_api.get_tool_async = AsyncMock(side_effect=mock_http_error)
        mock_control_api_class.return_value = mock_api

        client = ToolClient()

        with pytest.raises(Exception) as exc_info:
            await client.get_async(name="my-tool")
        assert exc_info.value.message == "Resource not found"  # type: ignore

    @patch("agentrun.tool.tool.Tool._Tool__get_client")
    def test_get_by_name(self, mock_get_client):
        """测试类方法 get_by_name"""
        mock_client = Mock()
        mock_tool = Tool(tool_id="tool-123", name="my-tool", tool_type="MCP")
        mock_client.get.return_value = mock_tool
        mock_get_client.return_value = mock_client

        tool = Tool.get_by_name("my-tool")

        assert tool.tool_id == "tool-123"
        assert tool.name == "my-tool"
        mock_client.get.assert_called_once_with(name="my-tool")

    @patch("agentrun.tool.tool.Tool._Tool__get_client")
    async def test_get_by_name_async(self, mock_get_client):
        """测试类方法 get_by_name_async"""
        mock_client = Mock()
        mock_tool = Tool(tool_id="tool-123", name="my-tool", tool_type="MCP")
        mock_client.get_async = AsyncMock(return_value=mock_tool)
        mock_get_client.return_value = mock_client

        tool = await Tool.get_by_name_async("my-tool")

        assert tool.tool_id == "tool-123"
        assert tool.name == "my-tool"
        mock_client.get_async.assert_called_once_with(name="my-tool")

    @patch("agentrun.tool.tool.Tool.get_by_name")
    def test_get_sync(self, mock_get_by_name):
        """测试实例方法 get()"""
        mock_tool = Tool(tool_id="tool-123", name="my-tool", tool_type="MCP")
        mock_get_by_name.return_value = mock_tool

        tool = Tool(tool_name="my-tool")
        result = tool.get()

        assert result.tool_id == "tool-123"
        mock_get_by_name.assert_called_once_with(name="my-tool", config=None)

    def test_get_sync_no_name(self):
        """测试 get() 没有 name 时抛出 ValueError"""
        tool = Tool()

        with pytest.raises(ValueError, match="Tool name is required"):
            tool.get()

    @patch("agentrun.tool.tool.Tool.get_by_name_async")
    async def test_get_async_method(self, mock_get_by_name_async):
        """测试实例方法 get_async()"""
        mock_tool = Tool(tool_id="tool-123", name="my-tool", tool_type="MCP")
        mock_get_by_name_async.return_value = mock_tool

        tool = Tool(tool_name="my-tool")
        result = await tool.get_async()

        assert result.tool_id == "tool-123"
        mock_get_by_name_async.assert_called_once_with(
            name="my-tool", config=None
        )

    def test_get_async_no_name(self):
        """测试 get_async() 没有 name 时抛出 ValueError"""
        tool = Tool()

        with pytest.raises(ValueError, match="Tool name is required"):
            import asyncio

            asyncio.run(tool.get_async())

    def test_get_functioncall_server_url(self):
        """测试 _get_functioncall_server_url 有 data_endpoint"""
        tool = Tool(
            tool_name="my-tool", data_endpoint="https://example.com/data"
        )
        url = tool._get_functioncall_server_url()

        assert url == "https://example.com/data/tools/my-tool"

    def test_get_functioncall_server_url_no_endpoint(self):
        """测试 _get_functioncall_server_url 没有 data_endpoint 和 name 时返回 None"""
        tool = Tool()
        url = tool._get_functioncall_server_url()

        assert url is None

    @patch("agentrun.utils.config.Config")
    async def test_list_tools_async_mcp_no_endpoint(self, mock_config_class):
        """测试 MCP 类型但没有 endpoint 时返回空列表"""
        tool = Tool(tool_name="my-tool", tool_type="MCP")

        tools = await tool.list_tools_async()

        assert tools == []

    @patch("agentrun.tool.api.openapi.ToolOpenAPIClient")
    async def test_list_tools_async_functioncall(
        self, mock_openapi_client_class
    ):
        """测试 FUNCTIONCALL 类型的 list_tools_async"""
        mock_client = Mock()
        mock_client.list_tools_async = AsyncMock(
            return_value=[
                ToolInfo(name="tool1", description="Tool 1"),
                ToolInfo(name="tool2", description="Tool 2"),
            ]
        )
        mock_openapi_client_class.return_value = mock_client

        tool = Tool(
            tool_type="FUNCTIONCALL",
            protocol_spec='{"openapi": "3.0.0"}',
        )

        tools = await tool.list_tools_async()

        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"

    async def test_list_tools_async_no_type(self):
        """测试没有类型时 list_tools_async 返回空列表"""
        tool = Tool()
        tools = await tool.list_tools_async()
        assert tools == []

    @patch("agentrun.tool.api.openapi.ToolOpenAPIClient")
    @patch("agentrun.utils.config.Config")
    async def test_call_tool_async_functioncall(
        self, mock_config_class, mock_openapi_client_class
    ):
        """测试 FUNCTIONCALL 类型的 call_tool_async"""
        mock_client = Mock()
        mock_client.call_tool_async = AsyncMock(
            return_value={"result": "success"}
        )
        mock_openapi_client_class.return_value = mock_client

        mock_config = Mock()
        mock_config.get_headers.return_value = {}
        mock_config_class.with_configs.return_value = mock_config

        tool = Tool(
            tool_type="FUNCTIONCALL",
            protocol_spec='{"openapi": "3.0.0"}',
        )

        result = await tool.call_tool_async("tool1", {"param": "value"})

        assert result == {"result": "success"}

    async def test_call_tool_async_mcp_no_endpoint(self):
        """测试 MCP 类型但没有 endpoint 时 call_tool_async 抛出 ValueError"""
        tool = Tool(tool_name="my-tool", tool_type="MCP")

        with pytest.raises(ValueError, match="MCP endpoint not available"):
            await tool.call_tool_async("tool1", {"param": "value"})

    @patch("agentrun.tool.api.openapi.ToolOpenAPIClient")
    @patch("agentrun.utils.config.Config")
    def test_call_tool_functioncall(
        self, mock_config_class, mock_openapi_client_class
    ):
        """测试 FUNCTIONCALL 类型的 call_tool（同步）"""
        mock_client = Mock()
        mock_client.call_tool.return_value = {"result": "success"}
        mock_openapi_client_class.return_value = mock_client

        mock_config = Mock()
        mock_config.get_headers.return_value = {}
        mock_config_class.with_configs.return_value = mock_config

        tool = Tool(
            tool_type="FUNCTIONCALL",
            protocol_spec='{"openapi": "3.0.0"}',
        )

        result = tool.call_tool("tool1", {"param": "value"})

        assert result == {"result": "success"}

    def test_call_tool_mcp_no_endpoint(self):
        """测试 MCP 类型但没有 endpoint 时 call_tool 抛出 ValueError"""
        tool = Tool(tool_name="my-tool", tool_type="MCP")

        with pytest.raises(ValueError, match="MCP endpoint not available"):
            tool.call_tool("tool1", {"param": "value"})

    @patch("agentrun.utils.config.Config")
    def test_list_tools_mcp_no_endpoint(self, mock_config_class):
        """测试 MCP 类型但没有 endpoint 时 list_tools 返回空列表"""
        tool = Tool(tool_name="my-tool", tool_type="MCP")

        tools = tool.list_tools()

        assert tools == []
