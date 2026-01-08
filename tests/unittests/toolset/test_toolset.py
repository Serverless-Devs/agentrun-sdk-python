"""ToolSet 资源类单元测试 / ToolSet Resource Class Unit Tests

测试 ToolSet 资源类的相关功能。
Tests ToolSet resource class functionality.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.toolset.model import (
    APIKeyAuthParameter,
    Authorization,
    AuthorizationParameters,
    MCPServerConfig,
    OpenAPIToolMeta,
    SchemaType,
    ToolMeta,
    ToolSetSchema,
    ToolSetSpec,
    ToolSetStatus,
    ToolSetStatusOutputs,
    ToolSetStatusOutputsUrls,
)
from agentrun.toolset.toolset import ToolSet
from agentrun.utils.config import Config


class TestToolSetBasic:
    """测试 ToolSet 基本功能"""

    def test_create_empty_toolset(self):
        """测试创建空 ToolSet"""
        toolset = ToolSet()
        assert toolset.name is None
        assert toolset.uid is None
        assert toolset.spec is None
        assert toolset.status is None

    def test_create_toolset_with_values(self):
        """测试创建带值的 ToolSet"""
        toolset = ToolSet(
            name="test-toolset",
            uid="uid-123",
            description="A test toolset",
            generation=1,
            kind="ToolSet",
            labels={"env": "prod"},
        )
        assert toolset.name == "test-toolset"
        assert toolset.uid == "uid-123"
        assert toolset.description == "A test toolset"
        assert toolset.generation == 1
        assert toolset.kind == "ToolSet"
        assert toolset.labels == {"env": "prod"}


class TestToolSetType:
    """测试 ToolSet.type 方法"""

    def test_type_mcp(self):
        """测试 MCP 类型"""
        toolset = ToolSet(
            spec=ToolSetSpec(tool_schema=ToolSetSchema(type=SchemaType.MCP))
        )
        assert toolset.type() == SchemaType.MCP

    def test_type_openapi(self):
        """测试 OpenAPI 类型"""
        toolset = ToolSet(
            spec=ToolSetSpec(tool_schema=ToolSetSchema(type=SchemaType.OpenAPI))
        )
        assert toolset.type() == SchemaType.OpenAPI

    def test_type_none(self):
        """测试类型为空"""
        toolset = ToolSet()
        # 当 spec 为空时，调用 type() 会抛出异常因为空字符串不是有效的 SchemaType
        with pytest.raises(ValueError, match="is not a valid SchemaType"):
            toolset.type()


class TestToolSetGetByName:
    """测试 ToolSet.get_by_name 方法"""

    @patch("agentrun.toolset.client.ToolSetClient")
    def test_get_by_name(self, mock_client_class):
        """测试通过名称获取 ToolSet"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_toolset = ToolSet(name="test-toolset")
        mock_client.get.return_value = mock_toolset

        result = ToolSet.get_by_name("test-toolset")

        mock_client_class.assert_called_once_with(None)
        mock_client.get.assert_called_once_with(name="test-toolset")
        assert result.name == "test-toolset"

    @patch("agentrun.toolset.client.ToolSetClient")
    def test_get_by_name_with_config(self, mock_client_class):
        """测试通过名称和配置获取 ToolSet"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = Config(access_key_id="key", access_key_secret="secret")
        mock_toolset = ToolSet(name="test-toolset")
        mock_client.get.return_value = mock_toolset

        result = ToolSet.get_by_name("test-toolset", config=config)

        mock_client_class.assert_called_once_with(config)


class TestToolSetGetByNameAsync:
    """测试 ToolSet.get_by_name_async 方法"""

    @pytest.mark.asyncio
    @patch("agentrun.toolset.client.ToolSetClient")
    async def test_get_by_name_async(self, mock_client_class):
        """测试异步通过名称获取 ToolSet"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_toolset = ToolSet(name="test-toolset")
        mock_client.get_async = AsyncMock(return_value=mock_toolset)

        result = await ToolSet.get_by_name_async("test-toolset")

        mock_client.get_async.assert_called_once_with(name="test-toolset")
        assert result.name == "test-toolset"


class TestToolSetGet:
    """测试 ToolSet.get 实例方法"""

    @patch("agentrun.toolset.client.ToolSetClient")
    def test_get_instance(self, mock_client_class):
        """测试实例 get 方法"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_result = ToolSet(name="test-toolset", uid="updated-uid")
        mock_client.get.return_value = mock_result

        toolset = ToolSet(name="test-toolset")
        result = toolset.get()

        mock_client.get.assert_called_once_with(name="test-toolset")
        assert result.uid == "updated-uid"

    def test_get_instance_no_name(self):
        """测试没有名称时 get 方法抛出异常"""
        toolset = ToolSet()
        with pytest.raises(ValueError, match="ToolSet name is required"):
            toolset.get()


class TestToolSetGetAsync:
    """测试 ToolSet.get_async 实例方法"""

    @pytest.mark.asyncio
    @patch("agentrun.toolset.client.ToolSetClient")
    async def test_get_async_instance(self, mock_client_class):
        """测试异步实例 get 方法"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_result = ToolSet(name="test-toolset", uid="updated-uid")
        mock_client.get_async = AsyncMock(return_value=mock_result)

        toolset = ToolSet(name="test-toolset")
        result = await toolset.get_async()

        mock_client.get_async.assert_called_once_with(name="test-toolset")
        assert result.uid == "updated-uid"

    @pytest.mark.asyncio
    async def test_get_async_instance_no_name(self):
        """测试没有名称时异步 get 方法抛出异常"""
        toolset = ToolSet()
        with pytest.raises(ValueError, match="ToolSet name is required"):
            await toolset.get_async()


class TestToolSetOpenAPIAuthDefaults:
    """测试 ToolSet._get_openapi_auth_defaults 方法"""

    def test_no_auth_config(self):
        """测试没有认证配置"""
        toolset = ToolSet()
        headers, query = toolset._get_openapi_auth_defaults()
        assert headers == {}
        assert query == {}

    def test_apikey_header_auth(self):
        """测试 API Key Header 认证"""
        toolset = ToolSet(
            spec=ToolSetSpec(
                auth_config=Authorization(
                    type="APIKey",
                    parameters=AuthorizationParameters(
                        api_key_parameter=APIKeyAuthParameter(
                            in_="header",
                            key="X-API-Key",
                            value="secret-key",
                        )
                    ),
                )
            )
        )
        headers, query = toolset._get_openapi_auth_defaults()
        assert headers == {"X-API-Key": "secret-key"}
        assert query == {}

    def test_apikey_query_auth(self):
        """测试 API Key Query 认证"""
        toolset = ToolSet(
            spec=ToolSetSpec(
                auth_config=Authorization(
                    type="APIKey",
                    parameters=AuthorizationParameters(
                        api_key_parameter=APIKeyAuthParameter(
                            in_="query",
                            key="api_key",
                            value="secret-key",
                        )
                    ),
                )
            )
        )
        headers, query = toolset._get_openapi_auth_defaults()
        assert headers == {}
        assert query == {"api_key": "secret-key"}

    def test_apikey_no_location(self):
        """测试 API Key 没有指定位置"""
        toolset = ToolSet(
            spec=ToolSetSpec(
                auth_config=Authorization(
                    type="APIKey",
                    parameters=AuthorizationParameters(
                        api_key_parameter=APIKeyAuthParameter(
                            key="api_key",
                            value="secret-key",
                        )
                    ),
                )
            )
        )
        headers, query = toolset._get_openapi_auth_defaults()
        assert headers == {}
        assert query == {}

    def test_non_apikey_auth(self):
        """测试非 APIKey 认证类型"""
        toolset = ToolSet(
            spec=ToolSetSpec(
                auth_config=Authorization(
                    type="Basic",
                )
            )
        )
        headers, query = toolset._get_openapi_auth_defaults()
        assert headers == {}
        assert query == {}


class TestToolSetGetOpenAPIBaseUrl:
    """测试 ToolSet._get_openapi_base_url 方法"""

    def test_no_urls(self):
        """测试没有 URL"""
        toolset = ToolSet()
        assert toolset._get_openapi_base_url() is None

    def test_intranet_url_preferred(self):
        """测试优先使用内网 URL"""
        toolset = ToolSet(
            status=ToolSetStatus(
                outputs=ToolSetStatusOutputs(
                    urls=ToolSetStatusOutputsUrls(
                        internet_url="https://public.example.com",
                        intranet_url="https://internal.example.com",
                    )
                )
            )
        )
        assert toolset._get_openapi_base_url() == "https://internal.example.com"

    def test_internet_url_fallback(self):
        """测试公网 URL 作为回退"""
        toolset = ToolSet(
            status=ToolSetStatus(
                outputs=ToolSetStatusOutputs(
                    urls=ToolSetStatusOutputsUrls(
                        internet_url="https://public.example.com",
                    )
                )
            )
        )
        assert toolset._get_openapi_base_url() == "https://public.example.com"


class TestToolSetListTools:
    """测试 ToolSet.list_tools 方法"""

    def test_list_tools_mcp(self):
        """测试列出 MCP 工具"""
        toolset = ToolSet(
            spec=ToolSetSpec(tool_schema=ToolSetSchema(type=SchemaType.MCP)),
            status=ToolSetStatus(
                outputs=ToolSetStatusOutputs(
                    tools=[
                        ToolMeta(
                            name="tool1",
                            description="Tool 1",
                            input_schema={
                                "type": "object",
                                "properties": {"arg": {"type": "string"}},
                            },
                        ),
                        ToolMeta(
                            name="tool2",
                            description="Tool 2",
                        ),
                    ]
                )
            ),
        )
        tools = toolset.list_tools()
        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"

    @patch("agentrun.toolset.toolset.ToolSet.to_apiset")
    def test_list_tools_openapi(self, mock_to_apiset):
        """测试列出 OpenAPI 工具"""
        mock_apiset = MagicMock()
        mock_apiset.tools.return_value = [
            MagicMock(name="api_tool1"),
            MagicMock(name="api_tool2"),
        ]
        mock_to_apiset.return_value = mock_apiset

        toolset = ToolSet(
            spec=ToolSetSpec(tool_schema=ToolSetSchema(type=SchemaType.OpenAPI))
        )
        tools = toolset.list_tools()
        assert len(tools) == 2
        mock_to_apiset.assert_called_once()

    def test_list_tools_empty_tools(self):
        """测试 MCP 类型但没有工具返回空列表"""
        toolset = ToolSet(
            spec=ToolSetSpec(tool_schema=ToolSetSchema(type=SchemaType.MCP)),
            status=ToolSetStatus(
                outputs=ToolSetStatusOutputs(tools=[])  # 显式设置为空列表
            ),
        )
        # 当 tools 为空列表时，返回空列表
        tools = toolset.list_tools()
        assert tools == []


class TestToolSetListToolsAsync:
    """测试 ToolSet.list_tools_async 方法"""

    @pytest.mark.asyncio
    async def test_list_tools_async_mcp(self):
        """测试异步列出 MCP 工具"""
        toolset = ToolSet(
            spec=ToolSetSpec(tool_schema=ToolSetSchema(type=SchemaType.MCP)),
            status=ToolSetStatus(
                outputs=ToolSetStatusOutputs(
                    tools=[
                        ToolMeta(name="tool1", description="Tool 1"),
                    ]
                )
            ),
        )
        tools = await toolset.list_tools_async()
        assert len(tools) == 1

    @pytest.mark.asyncio
    @patch("agentrun.toolset.toolset.ToolSet.to_apiset")
    async def test_list_tools_async_openapi(self, mock_to_apiset):
        """测试异步列出 OpenAPI 工具"""
        mock_apiset = MagicMock()
        mock_apiset.tools.return_value = [MagicMock(name="api_tool")]
        mock_to_apiset.return_value = mock_apiset

        toolset = ToolSet(
            spec=ToolSetSpec(tool_schema=ToolSetSchema(type=SchemaType.OpenAPI))
        )
        tools = await toolset.list_tools_async()
        assert len(tools) == 1

    @pytest.mark.asyncio
    async def test_list_tools_async_empty_tools(self):
        """测试异步 MCP 类型但没有工具返回空列表"""
        toolset = ToolSet(
            spec=ToolSetSpec(tool_schema=ToolSetSchema(type=SchemaType.MCP)),
            status=ToolSetStatus(
                outputs=ToolSetStatusOutputs(tools=[])  # 显式设置为空列表
            ),
        )
        tools = await toolset.list_tools_async()
        assert tools == []


class TestToolSetCallTool:
    """测试 ToolSet.call_tool 方法"""

    @patch("agentrun.toolset.toolset.ToolSet.to_apiset")
    def test_call_tool_mcp(self, mock_to_apiset):
        """测试调用 MCP 工具"""
        mock_apiset = MagicMock()
        mock_apiset.invoke.return_value = {"result": "success"}
        mock_to_apiset.return_value = mock_apiset

        toolset = ToolSet(
            spec=ToolSetSpec(tool_schema=ToolSetSchema(type=SchemaType.MCP))
        )
        result = toolset.call_tool("tool1", {"arg": "value"})

        assert result == {"result": "success"}
        mock_apiset.invoke.assert_called_once()

    @patch("agentrun.toolset.toolset.ToolSet.to_apiset")
    def test_call_tool_openapi_found(self, mock_to_apiset):
        """测试调用 OpenAPI 工具（找到工具）"""
        mock_tool = MagicMock()
        mock_apiset = MagicMock()
        mock_apiset.get_tool.return_value = mock_tool
        mock_apiset.invoke.return_value = {"result": "success"}
        mock_to_apiset.return_value = mock_apiset

        toolset = ToolSet(
            spec=ToolSetSpec(tool_schema=ToolSetSchema(type=SchemaType.OpenAPI))
        )
        result = toolset.call_tool("listUsers", {"limit": 10})

        assert result == {"result": "success"}
        mock_apiset.get_tool.assert_called_once_with("listUsers")

    @patch("agentrun.toolset.toolset.ToolSet.to_apiset")
    def test_call_tool_openapi_by_tool_id(self, mock_to_apiset):
        """测试通过 tool_id 调用 OpenAPI 工具

        注意：由于 Pydantic 会将字典转换为 OpenAPIToolMeta 对象，
        然后 model_dump() 返回驼峰命名 (toolId, toolName)，
        而代码使用 snake_case (tool_id, tool_name) 查找，
        所以 tool_id 匹配不会成功，name 保持原样。
        这是当前代码的实际行为。
        """
        mock_apiset = MagicMock()
        mock_apiset.get_tool.return_value = None
        mock_apiset.invoke.return_value = {"result": "success"}
        mock_to_apiset.return_value = mock_apiset

        toolset = ToolSet(
            spec=ToolSetSpec(
                tool_schema=ToolSetSchema(type=SchemaType.OpenAPI)
            ),
            status=ToolSetStatus(
                outputs=ToolSetStatusOutputs(
                    open_api_tools=[
                        {"tool_id": "tool_001", "tool_name": "actualToolName"},
                    ]
                )
            ),
        )
        result = toolset.call_tool("tool_001", {})

        # 由于 model_dump() 返回驼峰命名，匹配不成功，name 保持为 "tool_001"
        mock_apiset.invoke.assert_called_once()
        call_args = mock_apiset.invoke.call_args
        assert call_args.kwargs["name"] == "tool_001"

    @patch("agentrun.toolset.toolset.ToolSet.to_apiset")
    def test_call_tool_openapi_tool_meta_with_model_dump(self, mock_to_apiset):
        """测试 OpenAPI 工具 meta 有 model_dump 方法

        注意：当前代码中存在一个问题 - model_dump() 返回的是驼峰命名 (toolId, toolName)，
        但代码使用 snake_case 查找 (tool_id, tool_name)，所以实际上这个分支不会匹配成功。
        这个测试验证了当前的行为。
        """
        mock_apiset = MagicMock()
        mock_apiset.get_tool.return_value = None
        mock_apiset.invoke.return_value = {"result": "success"}
        mock_to_apiset.return_value = mock_apiset

        # 使用 OpenAPIToolMeta 对象，它有 model_dump 方法
        # 但由于 model_dump() 返回驼峰命名，tool_meta.get("tool_id") 会返回 None
        toolset = ToolSet(
            spec=ToolSetSpec(
                tool_schema=ToolSetSchema(type=SchemaType.OpenAPI)
            ),
            status=ToolSetStatus(
                outputs=ToolSetStatusOutputs(
                    open_api_tools=[
                        OpenAPIToolMeta(
                            tool_id="tool_002",
                            tool_name="mappedToolName",
                        )
                    ]
                )
            ),
        )
        # 由于 model_dump() 返回驼峰命名，tool_id 匹配不上，name 保持原样
        result = toolset.call_tool("tool_002", {})

        call_args = mock_apiset.invoke.call_args
        # name 保持为 "tool_002" 因为 toolId != tool_id
        assert call_args.kwargs["name"] == "tool_002"

    @patch("agentrun.toolset.toolset.ToolSet.to_apiset")
    def test_call_tool_openapi_skip_none_meta(self, mock_to_apiset):
        """测试跳过 None 的工具 meta"""
        mock_apiset = MagicMock()
        mock_apiset.get_tool.return_value = None
        mock_apiset.invoke.return_value = {"result": "success"}
        mock_to_apiset.return_value = mock_apiset

        # 使用字典列表，其中包含 None 和无效项
        toolset = ToolSet(
            spec=ToolSetSpec(
                tool_schema=ToolSetSchema(type=SchemaType.OpenAPI)
            ),
            status=ToolSetStatus(
                outputs=ToolSetStatusOutputs(
                    open_api_tools=None  # type: ignore
                )
            ),
        )
        # 不应该崩溃
        result = toolset.call_tool("unknown_tool", {})
        assert result == {"result": "success"}


class TestToolSetCallToolAsync:
    """测试 ToolSet.call_tool_async 方法"""

    @pytest.mark.asyncio
    @patch("agentrun.toolset.toolset.ToolSet.to_apiset")
    async def test_call_tool_async(self, mock_to_apiset):
        """测试异步调用工具"""
        mock_apiset = MagicMock()
        mock_apiset.invoke_async = AsyncMock(
            return_value={"result": "async_success"}
        )
        mock_to_apiset.return_value = mock_apiset

        toolset = ToolSet(
            spec=ToolSetSpec(tool_schema=ToolSetSchema(type=SchemaType.MCP))
        )
        result = await toolset.call_tool_async("tool1", {"arg": "value"})

        assert result == {"result": "async_success"}

    @pytest.mark.asyncio
    @patch("agentrun.toolset.toolset.ToolSet.to_apiset")
    async def test_call_tool_async_openapi_by_tool_id(self, mock_to_apiset):
        """测试异步通过 tool_id 调用 OpenAPI 工具

        注意：由于 model_dump() 返回驼峰命名，匹配不成功，name 保持原样。
        """
        mock_apiset = MagicMock()
        mock_apiset.get_tool.return_value = None
        mock_apiset.invoke_async = AsyncMock(return_value={"result": "success"})
        mock_to_apiset.return_value = mock_apiset

        toolset = ToolSet(
            spec=ToolSetSpec(
                tool_schema=ToolSetSchema(type=SchemaType.OpenAPI)
            ),
            status=ToolSetStatus(
                outputs=ToolSetStatusOutputs(
                    open_api_tools=[
                        {
                            "tool_id": "async_tool_001",
                            "tool_name": "asyncToolName",
                        },
                    ]
                )
            ),
        )
        result = await toolset.call_tool_async("async_tool_001", {})

        call_args = mock_apiset.invoke_async.call_args
        # 由于 model_dump() 返回驼峰命名，匹配不成功
        assert call_args.kwargs["name"] == "async_tool_001"


class TestToolSetToApiset:
    """测试 ToolSet.to_apiset 方法"""

    @patch("agentrun.toolset.api.mcp.MCPToolSet")
    @patch("agentrun.toolset.api.openapi.ApiSet.from_mcp_tools")
    def test_to_apiset_mcp(self, mock_from_mcp_tools, mock_mcp_toolset):
        """测试转换 MCP ToolSet 为 ApiSet"""
        mock_apiset = MagicMock()
        mock_from_mcp_tools.return_value = mock_apiset

        toolset = ToolSet(
            spec=ToolSetSpec(tool_schema=ToolSetSchema(type=SchemaType.MCP)),
            status=ToolSetStatus(
                outputs=ToolSetStatusOutputs(
                    mcp_server_config=MCPServerConfig(
                        url="https://mcp.example.com",
                        headers={"Authorization": "Bearer token"},
                    ),
                    tools=[
                        ToolMeta(name="mcp_tool1"),
                    ],
                )
            ),
        )
        result = toolset.to_apiset()

        assert result == mock_apiset
        mock_mcp_toolset.assert_called_once()
        mock_from_mcp_tools.assert_called_once()

    @patch("agentrun.toolset.api.openapi.ApiSet.from_openapi_schema")
    def test_to_apiset_openapi(self, mock_from_openapi_schema):
        """测试转换 OpenAPI ToolSet 为 ApiSet"""
        mock_apiset = MagicMock()
        mock_from_openapi_schema.return_value = mock_apiset

        toolset = ToolSet(
            spec=ToolSetSpec(
                tool_schema=ToolSetSchema(
                    type=SchemaType.OpenAPI,
                    detail='{"openapi": "3.0.0"}',
                ),
                auth_config=Authorization(
                    type="APIKey",
                    parameters=AuthorizationParameters(
                        api_key_parameter=APIKeyAuthParameter(
                            in_="header",
                            key="X-API-Key",
                            value="secret",
                        )
                    ),
                ),
            ),
            status=ToolSetStatus(
                outputs=ToolSetStatusOutputs(
                    urls=ToolSetStatusOutputsUrls(
                        internet_url="https://api.example.com",
                    )
                )
            ),
        )
        result = toolset.to_apiset()

        assert result == mock_apiset
        mock_from_openapi_schema.assert_called_once()
        call_kwargs = mock_from_openapi_schema.call_args.kwargs
        assert call_kwargs["schema"] == '{"openapi": "3.0.0"}'
        assert call_kwargs["base_url"] == "https://api.example.com"
        assert call_kwargs["headers"] == {"X-API-Key": "secret"}

    def test_to_apiset_unsupported_type(self):
        """测试不支持的类型抛出异常"""
        # 当 type() 被调用时，空字符串会抛出 ValueError
        toolset = ToolSet()
        # 由于 type() 会将空字符串传给 SchemaType，会抛出异常
        with pytest.raises(ValueError):
            toolset.to_apiset()

    def test_to_apiset_mcp_missing_url(self):
        """测试 MCP 类型缺少 URL 抛出异常"""
        toolset = ToolSet(
            spec=ToolSetSpec(tool_schema=ToolSetSchema(type=SchemaType.MCP)),
            status=ToolSetStatus(
                outputs=ToolSetStatusOutputs(
                    mcp_server_config=MCPServerConfig(),
                )
            ),
        )
        with pytest.raises(AssertionError, match="MCP server URL is missing"):
            toolset.to_apiset()
