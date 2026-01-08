"""OpenAPI 协议处理扩展单元测试 / OpenAPI Protocol Handler Extended Unit Tests

测试 OpenAPI 协议处理的更多边界情况。
Tests more edge cases for OpenAPI protocol handling.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from agentrun.toolset.api.openapi import ApiSet, OpenAPI
from agentrun.toolset.model import ToolInfo, ToolSchema
from agentrun.utils.config import Config


class TestOpenAPIInit:
    """测试 OpenAPI 初始化"""

    def test_init_with_dict_schema(self):
        """测试使用字典 schema 初始化"""
        schema = {
            "openapi": "3.0.0",
            "paths": {},
        }
        openapi = OpenAPI(schema=schema, base_url="http://test")
        assert openapi._schema == schema

    def test_init_with_bytes_schema(self):
        """测试使用 bytes schema 初始化"""
        schema = b'{"openapi": "3.0.0", "paths": {}}'
        openapi = OpenAPI(schema=schema, base_url="http://test")
        assert openapi._schema["openapi"] == "3.0.0"

    def test_init_with_bytearray_schema(self):
        """测试使用 bytearray schema 初始化"""
        schema = bytearray(b'{"openapi": "3.0.0", "paths": {}}')
        openapi = OpenAPI(schema=schema, base_url="http://test")
        assert openapi._schema["openapi"] == "3.0.0"

    def test_init_with_empty_schema_raises(self):
        """测试空 schema 抛出异常"""
        with pytest.raises(
            ValueError, match="OpenAPI schema detail is required"
        ):
            OpenAPI(schema="", base_url="http://test")

    def test_init_with_base_url_from_servers(self):
        """测试从 servers 获取 base_url"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {},
        }
        openapi = OpenAPI(schema=json.dumps(schema))
        assert openapi._base_url == "https://api.example.com"

    def test_init_with_server_variables(self):
        """测试 servers 带变量"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{
                "url": "https://{env}.example.com",
                "variables": {"env": {"default": "api"}},
            }],
            "paths": {},
        }
        openapi = OpenAPI(schema=json.dumps(schema))
        assert openapi._base_url == "https://api.example.com"

    def test_init_with_timeout_from_config(self):
        """测试从配置获取超时"""
        schema = {"openapi": "3.0.0", "paths": {}}
        config = Config(timeout=120)
        openapi = OpenAPI(
            schema=json.dumps(schema),
            base_url="http://test",
            config=config,
        )
        assert openapi._default_timeout == 120

    def test_init_with_timeout_override(self):
        """测试超时覆盖"""
        schema = {"openapi": "3.0.0", "paths": {}}
        openapi = OpenAPI(
            schema=json.dumps(schema),
            base_url="http://test",
            timeout=30,
        )
        assert openapi._default_timeout == 30


class TestOpenAPIListTools:
    """测试 OpenAPI.list_tools 方法"""

    def test_list_tools_all(self):
        """测试列出所有工具"""
        schema = {
            "openapi": "3.0.0",
            "paths": {
                "/users": {
                    "get": {"operationId": "listUsers"},
                    "post": {"operationId": "createUser"},
                },
            },
        }
        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")
        tools = openapi.list_tools()
        assert len(tools) == 2

    def test_list_tools_by_name(self):
        """测试按名称获取工具"""
        schema = {
            "openapi": "3.0.0",
            "paths": {
                "/users": {
                    "get": {"operationId": "listUsers"},
                },
            },
        }
        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")
        tools = openapi.list_tools(name="listUsers")
        assert len(tools) == 1
        assert tools[0]["operationId"] == "listUsers"

    def test_list_tools_not_found(self):
        """测试工具不存在"""
        schema = {
            "openapi": "3.0.0",
            "paths": {},
        }
        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")
        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            openapi.list_tools(name="nonexistent")


class TestOpenAPIHasTool:
    """测试 OpenAPI.has_tool 方法"""

    def test_has_tool_true(self):
        """测试工具存在"""
        schema = {
            "openapi": "3.0.0",
            "paths": {
                "/users": {"get": {"operationId": "listUsers"}},
            },
        }
        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")
        assert openapi.has_tool("listUsers") is True

    def test_has_tool_false(self):
        """测试工具不存在"""
        schema = {
            "openapi": "3.0.0",
            "paths": {},
        }
        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")
        assert openapi.has_tool("nonexistent") is False


class TestOpenAPIInvokeTool:
    """测试 OpenAPI.invoke_tool 方法"""

    def test_invoke_tool_not_found(self):
        """测试调用不存在的工具"""
        schema = {
            "openapi": "3.0.0",
            "paths": {},
        }
        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")
        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            openapi.invoke_tool("nonexistent")

    def test_invoke_tool_no_base_url(self):
        """测试没有 base_url 抛出异常"""
        schema = {
            "openapi": "3.0.0",
            "paths": {
                "/users": {"get": {"operationId": "listUsers"}},
            },
        }
        openapi = OpenAPI(schema=json.dumps(schema))
        with pytest.raises(ValueError, match="Base URL is required"):
            openapi.invoke_tool("listUsers")

    @respx.mock
    def test_invoke_tool_with_files(self):
        """测试带文件上传的请求"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/upload": {
                    "post": {"operationId": "uploadFile"},
                },
            },
        }
        route = respx.post("https://api.example.com/upload").mock(
            return_value=httpx.Response(200, json={"uploaded": True})
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        # 模拟文件上传（实际使用 httpx 文件格式）
        result = openapi.invoke_tool(
            "uploadFile",
            {"files": {"file": ("test.txt", b"content")}},
        )
        assert route.called
        assert result["status_code"] == 200

    @respx.mock
    def test_invoke_tool_with_data(self):
        """测试带 form data 的请求"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/form": {
                    "post": {"operationId": "submitForm"},
                },
            },
        }
        route = respx.post("https://api.example.com/form").mock(
            return_value=httpx.Response(200, json={"submitted": True})
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        result = openapi.invoke_tool(
            "submitForm",
            {"data": {"field1": "value1", "field2": "value2"}},
        )
        assert route.called
        assert result["status_code"] == 200

    @respx.mock
    def test_invoke_tool_raise_for_status_false(self):
        """测试禁用 raise_for_status"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/error": {"get": {"operationId": "getError"}},
            },
        }
        route = respx.get("https://api.example.com/error").mock(
            return_value=httpx.Response(
                500, json={"error": "Internal Server Error"}
            )
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        result = openapi.invoke_tool("getError", {"raise_for_status": False})
        assert result["status_code"] == 500

    @respx.mock
    def test_invoke_tool_with_timeout_override(self):
        """测试超时覆盖"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/slow": {"get": {"operationId": "getSlow"}},
            },
        }
        route = respx.get("https://api.example.com/slow").mock(
            return_value=httpx.Response(200, json={})
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        result = openapi.invoke_tool("getSlow", {"timeout": 5})
        assert result["status_code"] == 200

    @respx.mock
    def test_invoke_tool_with_json_body(self):
        """测试 json 参数"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/data": {"post": {"operationId": "postData"}},
            },
        }
        route = respx.post("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json={"received": True})
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        result = openapi.invoke_tool(
            "postData",
            {"json": {"key": "value"}},
        )
        assert route.called
        body = json.loads(route.calls.last.request.content)
        assert body["key"] == "value"

    @respx.mock
    def test_invoke_tool_with_payload(self):
        """测试 payload 参数"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/data": {"post": {"operationId": "postData"}},
            },
        }
        route = respx.post("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json={"received": True})
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        result = openapi.invoke_tool(
            "postData",
            {"payload": {"key": "value"}},
        )
        assert route.called


class TestOpenAPIInvokeToolAsync:
    """测试 OpenAPI.invoke_tool_async 方法"""

    @pytest.mark.asyncio
    @respx.mock
    async def test_invoke_tool_async(self):
        """测试异步调用工具"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users": {"get": {"operationId": "listUsers"}},
            },
        }
        route = respx.get("https://api.example.com/users").mock(
            return_value=httpx.Response(200, json={"users": []})
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        result = await openapi.invoke_tool_async("listUsers")

        assert route.called
        assert result["status_code"] == 200


class TestOpenAPIPickServerUrl:
    """测试 OpenAPI._pick_server_url 方法"""

    def test_pick_server_url_empty(self):
        """测试空 servers"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        assert openapi._pick_server_url(None) is None
        assert openapi._pick_server_url([]) is None

    def test_pick_server_url_string(self):
        """测试字符串 server"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._pick_server_url(["https://api.example.com"])
        assert result == "https://api.example.com"

    def test_pick_server_url_dict(self):
        """测试字典 server"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._pick_server_url([{"url": "https://api.example.com"}])
        assert result == "https://api.example.com"

    def test_pick_server_url_dict_single(self):
        """测试单个字典 server"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._pick_server_url({"url": "https://api.example.com"})
        assert result == "https://api.example.com"

    def test_pick_server_url_invalid_type(self):
        """测试无效类型"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._pick_server_url("invalid")
        assert result is None

    def test_pick_server_url_skip_invalid_entry(self):
        """测试跳过无效条目"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._pick_server_url(
            [123, {"url": "https://api.example.com"}]
        )
        assert result == "https://api.example.com"


class TestOpenAPIBuildOperations:
    """测试 OpenAPI._build_operations 方法"""

    def test_build_operations_with_path_servers(self):
        """测试带路径级别 servers"""
        schema = {
            "openapi": "3.0.0",
            "paths": {
                "/users": {
                    "servers": [{"url": "https://users.example.com"}],
                    "get": {"operationId": "listUsers"},
                },
            },
        }
        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")
        assert (
            openapi._operations["listUsers"]["server_url"]
            == "https://users.example.com"
        )

    def test_build_operations_with_operation_servers(self):
        """测试带操作级别 servers"""
        schema = {
            "openapi": "3.0.0",
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "servers": [{"url": "https://get-users.example.com"}],
                    },
                },
            },
        }
        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")
        assert (
            openapi._operations["listUsers"]["server_url"]
            == "https://get-users.example.com"
        )


class TestOpenAPIConvertToNative:
    """测试 OpenAPI._convert_to_native 方法"""

    def test_convert_none(self):
        """测试转换 None"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        assert openapi._convert_to_native(None) is None

    def test_convert_primitives(self):
        """测试转换基本类型"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        assert openapi._convert_to_native("string") == "string"
        assert openapi._convert_to_native(123) == 123
        assert openapi._convert_to_native(1.5) == 1.5
        assert openapi._convert_to_native(True) is True

    def test_convert_list(self):
        """测试转换列表"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._convert_to_native([1, 2, 3])
        assert result == [1, 2, 3]

    def test_convert_dict(self):
        """测试转换字典"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._convert_to_native({"key": "value"})
        assert result == {"key": "value"}

    def test_convert_pydantic_model(self):
        """测试转换 Pydantic 模型"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )

        class MockModel:

            def model_dump(self, mode=None, exclude_unset=False):
                return {"field": "value"}

        result = openapi._convert_to_native(MockModel())
        assert result == {"field": "value"}

    def test_convert_pydantic_v1_model(self):
        """测试转换 Pydantic v1 模型"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )

        class MockV1Model:

            def dict(self, exclude_none=False):
                return {"field": "v1_value"}

        result = openapi._convert_to_native(MockV1Model())
        assert result == {"field": "v1_value"}

    def test_convert_to_dict_method(self):
        """测试转换 to_dict 方法对象"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )

        class MockWithToDict:

            def to_dict(self):
                return {"field": "to_dict_value"}

        result = openapi._convert_to_native(MockWithToDict())
        assert result == {"field": "to_dict_value"}

    def test_convert_object_with_dict(self):
        """测试转换带 __dict__ 的对象"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )

        class MockObject:

            def __init__(self):
                self.field = "object_value"

        result = openapi._convert_to_native(MockObject())
        assert result["field"] == "object_value"


class TestOpenAPIRenderPath:
    """测试 OpenAPI._render_path 方法"""

    def test_render_path_success(self):
        """测试成功渲染路径"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._render_path(
            "/users/{userId}/posts/{postId}",
            ["userId", "postId"],
            {"userId": "123", "postId": "456"},
        )
        assert result == "/users/123/posts/456"

    def test_render_path_missing_param(self):
        """测试缺少路径参数"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        with pytest.raises(ValueError, match="Missing path parameters"):
            openapi._render_path(
                "/users/{userId}",
                ["userId"],
                {},
            )


class TestOpenAPIJoinUrl:
    """测试 OpenAPI._join_url 方法"""

    def test_join_url(self):
        """测试拼接 URL"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._join_url("https://api.example.com", "/users")
        assert result == "https://api.example.com/users"

    def test_join_url_trailing_slash(self):
        """测试 base_url 带尾部斜杠"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._join_url("https://api.example.com/", "/users")
        assert result == "https://api.example.com/users"

    def test_join_url_empty_base(self):
        """测试空 base_url"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        with pytest.raises(ValueError, match="Base URL cannot be empty"):
            openapi._join_url("", "/users")

    def test_join_url_empty_path(self):
        """测试空路径"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._join_url("https://api.example.com", "")
        assert result == "https://api.example.com"


class TestOpenAPIExtractDict:
    """测试 OpenAPI._extract_dict 方法"""

    def test_extract_dict_found(self):
        """测试成功提取字典"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        source = {"path": {"id": "123"}, "other": "value"}
        result = openapi._extract_dict(source, ["path"])
        assert result == {"id": "123"}
        assert "path" not in source

    def test_extract_dict_not_found(self):
        """测试未找到返回空字典"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        source = {"other": "value"}
        result = openapi._extract_dict(source, ["path"])
        assert result == {}

    def test_extract_dict_non_dict_value(self):
        """测试非字典值发出警告"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        source = {"path": "not-a-dict"}
        with patch("agentrun.toolset.api.openapi.logger") as mock_logger:
            result = openapi._extract_dict(source, ["path"])
            mock_logger.warning.assert_called()
        assert result == {}


class TestOpenAPIMergeDicts:
    """测试 OpenAPI._merge_dicts 方法"""

    def test_merge_dicts_both(self):
        """测试合并两个字典"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_merge_dicts_override(self):
        """测试覆盖"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._merge_dicts({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_merge_dicts_none_base(self):
        """测试 base 为 None"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._merge_dicts(None, {"b": 2})
        assert result == {"b": 2}

    def test_merge_dicts_none_override(self):
        """测试 override 为 None"""
        openapi = OpenAPI(
            schema='{"openapi": "3.0.0", "paths": {}}', base_url="http://test"
        )
        result = openapi._merge_dicts({"a": 1}, None)
        assert result == {"a": 1}


class TestApiSetFromMCPTools:
    """测试 ApiSet.from_mcp_tools 方法"""

    def test_from_mcp_tools_list(self):
        """测试从工具列表创建"""
        tools = [
            {
                "name": "tool1",
                "description": "Tool 1",
                "inputSchema": {"type": "object"},
            },
            {"name": "tool2", "description": "Tool 2"},
        ]
        mock_mcp_client = MagicMock()
        apiset = ApiSet.from_mcp_tools(tools, mock_mcp_client)

        assert len(apiset.tools()) == 2

    def test_from_mcp_tools_single(self):
        """测试从单个工具创建"""
        tool = {"name": "single_tool", "description": "Single Tool"}
        mock_mcp_client = MagicMock()
        apiset = ApiSet.from_mcp_tools(tool, mock_mcp_client)

        assert len(apiset.tools()) == 1

    def test_from_mcp_tools_empty(self):
        """测试从空列表创建"""
        mock_mcp_client = MagicMock()
        apiset = ApiSet.from_mcp_tools(None, mock_mcp_client)
        assert len(apiset.tools()) == 0

    def test_from_mcp_tools_with_object_tool(self):
        """测试从对象格式工具创建"""

        class MockTool:
            name = "object_tool"
            description = "Object Tool"
            inputSchema = {"type": "object"}

        mock_mcp_client = MagicMock()
        apiset = ApiSet.from_mcp_tools([MockTool()], mock_mcp_client)
        assert len(apiset.tools()) == 1

    def test_from_mcp_tools_skip_invalid(self):
        """测试跳过无效工具"""
        tools = [
            "invalid",
            {"name": "valid_tool"},
            {"description": "no name"},
        ]
        mock_mcp_client = MagicMock()
        apiset = ApiSet.from_mcp_tools(tools, mock_mcp_client)
        assert len(apiset.tools()) == 1

    def test_from_mcp_tools_with_model_dump_input_schema(self):
        """测试 inputSchema 有 model_dump 方法"""

        class MockInputSchema:

            def model_dump(self):
                return {
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                }

        class MockTool:
            name = "tool_with_schema"
            description = "Tool with schema"
            inputSchema = MockInputSchema()

        mock_mcp_client = MagicMock()
        apiset = ApiSet.from_mcp_tools([MockTool()], mock_mcp_client)
        tool = apiset.get_tool("tool_with_schema")
        assert tool.parameters.type == "object"


class TestApiSetInvoke:
    """测试 ApiSet.invoke 方法"""

    def test_invoke_tool_not_found(self):
        """测试调用不存在的工具"""
        apiset = ApiSet(tools=[], invoker=MagicMock())
        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            apiset.invoke("nonexistent")

    def test_invoke_with_invoke_tool_method(self):
        """测试使用 invoke_tool 方法的 invoker"""
        mock_invoker = MagicMock()
        mock_invoker.invoke_tool.return_value = {"result": "success"}

        tools = [ToolInfo(name="my_tool", description="Test")]
        apiset = ApiSet(tools=tools, invoker=mock_invoker)
        result = apiset.invoke("my_tool", {"arg": "value"})

        assert result == {"result": "success"}

    def test_invoke_with_call_tool_method(self):
        """测试使用 call_tool 方法的 invoker"""
        mock_invoker = MagicMock(spec=["call_tool"])
        mock_invoker.call_tool.return_value = {"result": "success"}

        tools = [ToolInfo(name="my_tool", description="Test")]
        apiset = ApiSet(tools=tools, invoker=mock_invoker)
        result = apiset.invoke("my_tool", {"arg": "value"})

        assert result == {"result": "success"}

    def test_invoke_with_callable(self):
        """测试使用可调用对象的 invoker"""

        def mock_invoker(name, arguments):
            return {"name": name, "args": arguments}

        tools = [ToolInfo(name="my_tool", description="Test")]
        apiset = ApiSet(tools=tools, invoker=mock_invoker)
        result = apiset.invoke("my_tool", {"arg": "value"})

        assert result["name"] == "my_tool"
        assert result["args"]["arg"] == "value"

    def test_invoke_with_invalid_invoker(self):
        """测试无效的 invoker"""
        mock_invoker = "not-callable"

        tools = [ToolInfo(name="my_tool", description="Test")]
        apiset = ApiSet(tools=tools, invoker=mock_invoker)
        with pytest.raises(ValueError, match="Invalid invoker provided"):
            apiset.invoke("my_tool")


class TestApiSetInvokeAsync:
    """测试 ApiSet.invoke_async 方法"""

    @pytest.mark.asyncio
    async def test_invoke_async_tool_not_found(self):
        """测试异步调用不存在的工具"""
        apiset = ApiSet(tools=[], invoker=MagicMock())
        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            await apiset.invoke_async("nonexistent")

    @pytest.mark.asyncio
    async def test_invoke_async_with_invoke_tool_async(self):
        """测试使用 invoke_tool_async 方法"""
        mock_invoker = MagicMock()
        mock_invoker.invoke_tool_async = AsyncMock(
            return_value={"result": "async_success"}
        )

        tools = [ToolInfo(name="my_tool", description="Test")]
        apiset = ApiSet(tools=tools, invoker=mock_invoker)
        result = await apiset.invoke_async("my_tool", {"arg": "value"})

        assert result == {"result": "async_success"}

    @pytest.mark.asyncio
    async def test_invoke_async_with_call_tool_async(self):
        """测试使用 call_tool_async 方法"""
        mock_invoker = MagicMock(spec=["call_tool_async"])
        mock_invoker.call_tool_async = AsyncMock(
            return_value={"result": "success"}
        )

        tools = [ToolInfo(name="my_tool", description="Test")]
        apiset = ApiSet(tools=tools, invoker=mock_invoker)
        result = await apiset.invoke_async("my_tool", {"arg": "value"})

        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_invoke_async_no_async_invoker(self):
        """测试没有异步 invoker"""
        mock_invoker = MagicMock(spec=[])

        tools = [ToolInfo(name="my_tool", description="Test")]
        apiset = ApiSet(tools=tools, invoker=mock_invoker)
        with pytest.raises(ValueError, match="Async invoker not available"):
            await apiset.invoke_async("my_tool")


class TestApiSetConvertArguments:
    """测试 ApiSet._convert_arguments 方法"""

    def test_convert_arguments_none(self):
        """测试转换 None"""
        apiset = ApiSet(tools=[], invoker=MagicMock())
        assert apiset._convert_arguments(None) is None

    def test_convert_arguments_non_dict(self):
        """测试转换非字典"""
        apiset = ApiSet(tools=[], invoker=MagicMock())
        assert apiset._convert_arguments("not-dict") == "not-dict"

    def test_convert_arguments_dict(self):
        """测试转换字典"""
        apiset = ApiSet(tools=[], invoker=MagicMock())
        result = apiset._convert_arguments({"key": "value"})
        assert result == {"key": "value"}


class TestApiSetSchemaTypeToPhythonType:
    """测试 ApiSet._schema_type_to_python_type 方法"""

    def test_known_types(self):
        """测试已知类型"""
        apiset = ApiSet(tools=[], invoker=MagicMock())
        assert apiset._schema_type_to_python_type("string") == str
        assert apiset._schema_type_to_python_type("integer") == int
        assert apiset._schema_type_to_python_type("number") == float
        assert apiset._schema_type_to_python_type("boolean") == bool
        assert apiset._schema_type_to_python_type("object") == dict
        assert apiset._schema_type_to_python_type("array") == list

    def test_unknown_type(self):
        """测试未知类型"""
        from typing import Any

        apiset = ApiSet(tools=[], invoker=MagicMock())
        assert apiset._schema_type_to_python_type("unknown") == Any
