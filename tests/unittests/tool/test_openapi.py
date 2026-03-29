"""Tool OpenAPI 客户端单元测试 / Tool OpenAPI Client Unit Tests

测试 ToolOpenAPIClient 的 OpenAPI Schema 解析和 HTTP 调用功能。
Tests OpenAPI Schema parsing and HTTP call functionality of ToolOpenAPIClient.
"""

import json
from unittest.mock import Mock, patch

import httpx
import pytest

from agentrun.tool.api.openapi import ToolOpenAPIClient
from agentrun.tool.model import ToolInfo, ToolSchema


class TestToolOpenAPIClient:
    """测试 ToolOpenAPIClient"""

    @pytest.fixture
    def sample_openapi_spec(self):
        """示例 OpenAPI Spec"""
        return json.dumps({
            "openapi": "3.0.0",
            "info": {
                "title": "Test API",
                "version": "1.0.0",
            },
            "servers": [
                {"url": "https://api.example.com/v1"},
            ],
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "summary": "List all users",
                        "description": "Get a list of users",
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "required": False,
                                "schema": {"type": "integer"},
                            },
                        ],
                    },
                    "post": {
                        "operationId": "createUser",
                        "summary": "Create a user",
                        "description": "Create a new user",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "email": {"type": "string"},
                                        },
                                        "required": ["name"],
                                    },
                                }
                            }
                        },
                    },
                },
                "/users/{id}": {
                    "get": {
                        "operationId": "getUser",
                        "summary": "Get user by ID",
                    },
                    "put": {
                        "operationId": "updateUser",
                        "summary": "Update user",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                        },
                                    },
                                }
                            }
                        },
                    },
                    "delete": {
                        "operationId": "deleteUser",
                        "summary": "Delete user",
                    },
                },
            },
        })

    @pytest.fixture
    def sample_openapi_spec_no_servers(self):
        """没有 servers 的 OpenAPI Spec"""
        return json.dumps({
            "openapi": "3.0.0",
            "info": {
                "title": "Test API",
                "version": "1.0.0",
            },
            "paths": {},
        })

    def test_init_with_valid_json(self, sample_openapi_spec):
        """测试使用有效 JSON 初始化"""
        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        assert client._spec is not None
        assert client._spec["openapi"] == "3.0.0"

    def test_init_with_invalid_json(self):
        """测试使用无效 JSON 初始化"""
        client = ToolOpenAPIClient(protocol_spec="invalid json")
        assert client._spec is None

    def test_init_with_none(self):
        """测试使用 None 初始化"""
        client = ToolOpenAPIClient(protocol_spec=None)
        assert client._spec is None

    def test_init_with_headers(self, sample_openapi_spec):
        """测试带 headers 初始化"""
        headers = {"Authorization": "Bearer token"}
        client = ToolOpenAPIClient(
            protocol_spec=sample_openapi_spec,
            headers=headers,
        )
        assert client.headers == headers

    def test_server_url(self, sample_openapi_spec):
        """测试获取 server URL"""
        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        assert client.server_url == "https://api.example.com/v1"

    def test_server_url_no_servers(self, sample_openapi_spec_no_servers):
        """测试没有 servers 时的 server URL"""
        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec_no_servers)
        assert client.server_url is None

    def test_server_url_no_spec(self):
        """测试没有 spec 时的 server URL"""
        client = ToolOpenAPIClient(protocol_spec=None)
        assert client.server_url is None

    def test_parse_operations_get_method(self, sample_openapi_spec):
        """测试解析 GET 方法"""
        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        operations = client._parse_operations()

        get_operation = next(
            (op for op in operations if op["operation_id"] == "listUsers"),
            None,
        )
        assert get_operation is not None
        assert get_operation["method"] == "GET"
        assert get_operation["path"] == "/users"
        assert get_operation["summary"] == "List all users"
        assert get_operation["input_schema"] is not None
        assert "properties" in get_operation["input_schema"]

    def test_parse_operations_post_method_with_request_body(
        self, sample_openapi_spec
    ):
        """测试解析 POST 方法（带 requestBody）"""
        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        operations = client._parse_operations()

        post_operation = next(
            (op for op in operations if op["operation_id"] == "createUser"),
            None,
        )
        assert post_operation is not None
        assert post_operation["method"] == "POST"
        assert post_operation["path"] == "/users"
        assert post_operation["input_schema"] is not None
        assert post_operation["input_schema"]["type"] == "object"

    def test_parse_operations_multiple_methods(self, sample_openapi_spec):
        """测试解析多个 HTTP 方法"""
        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        operations = client._parse_operations()

        operation_ids = [op["operation_id"] for op in operations]
        assert "listUsers" in operation_ids
        assert "createUser" in operation_ids
        assert "getUser" in operation_ids
        assert "updateUser" in operation_ids
        assert "deleteUser" in operation_ids

    def test_parse_operations_parameters(self, sample_openapi_spec):
        """测试解析 parameters"""
        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        operations = client._parse_operations()

        list_users_op = next(
            (op for op in operations if op["operation_id"] == "listUsers"),
            None,
        )
        assert list_users_op is not None
        assert list_users_op["input_schema"] is not None
        assert "properties" in list_users_op["input_schema"]
        assert "limit" in list_users_op["input_schema"]["properties"]

    def test_parse_operations_no_spec(self):
        """测试没有 spec 时的解析"""
        client = ToolOpenAPIClient(protocol_spec=None)
        operations = client._parse_operations()
        assert operations == []

    def test_list_tools(self, sample_openapi_spec):
        """测试获取工具列表"""
        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        tools = client.list_tools()

        assert len(tools) > 0
        assert all(isinstance(tool, ToolInfo) for tool in tools)

        # 检查特定工具
        list_users_tool = next(
            (t for t in tools if t.name == "listUsers"),
            None,
        )
        assert list_users_tool is not None
        assert list_users_tool.description == "List all users"
        assert list_users_tool.parameters is not None

    def test_list_tools_empty_spec(self):
        """测试空 spec 时的工具列表"""
        client = ToolOpenAPIClient(protocol_spec='{"paths": {}}')
        tools = client.list_tools()
        assert tools == []

    @patch("agentrun.tool.api.openapi.httpx.Client")
    def test_call_tool_post_method(
        self, mock_client_class, sample_openapi_spec
    ):
        """测试调用 POST 方法"""
        # Mock httpx response
        mock_response = Mock()
        mock_response.json.return_value = {"id": 123, "name": "Test User"}
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status = Mock()

        mock_client_instance = Mock()
        mock_client_instance.request.return_value = mock_response
        mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client_instance

        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        result = client.call_tool(
            "createUser", {"name": "Test User", "email": "test@example.com"}
        )

        assert result == {"id": 123, "name": "Test User"}
        mock_client_instance.request.assert_called_once()
        call_args = mock_client_instance.request.call_args
        assert call_args[0][0] == "POST"
        assert "https://api.example.com/v1/users" in call_args[0][1]

    @patch("agentrun.tool.api.openapi.httpx.Client")
    def test_call_tool_get_method(self, mock_client_class, sample_openapi_spec):
        """测试调用 GET 方法"""
        # Mock httpx response
        mock_response = Mock()
        mock_response.json.return_value = {"id": 123, "name": "Test User"}
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status = Mock()

        mock_client_instance = Mock()
        mock_client_instance.request.return_value = mock_response
        mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client_instance

        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        result = client.call_tool("listUsers", {"limit": 10})

        assert result == {"id": 123, "name": "Test User"}
        mock_client_instance.request.assert_called_once()
        call_args = mock_client_instance.request.call_args
        assert call_args[0][0] == "GET"
        assert "limit" in call_args[1]["params"]

    @patch("agentrun.tool.api.openapi.httpx.Client")
    def test_call_tool_text_response(
        self, mock_client_class, sample_openapi_spec
    ):
        """测试调用工具返回文本响应"""
        # Mock httpx response
        mock_response = Mock()
        mock_response.text = "Plain text response"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.raise_for_status = Mock()

        mock_client_instance = Mock()
        mock_client_instance.request.return_value = mock_response
        mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client_instance

        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        result = client.call_tool("listUsers", {})

        assert result == "Plain text response"

    def test_call_tool_operation_not_found(self, sample_openapi_spec):
        """测试调用不存在的 operation"""
        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        with pytest.raises(
            ValueError, match="Operation 'nonExistent' not found"
        ):
            client.call_tool("nonExistent", {})

    def test_call_tool_no_server_url(self):
        """测试没有 server URL 时调用工具"""
        spec_without_server = json.dumps({
            "openapi": "3.0.0",
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "testOp",
                    },
                },
            },
        })
        client = ToolOpenAPIClient(protocol_spec=spec_without_server)
        with pytest.raises(ValueError, match="No server URL found"):
            client.call_tool("testOp", {})

    @patch("httpx.AsyncClient")
    async def test_call_tool_async_post_method(
        self, mock_async_client_class, sample_openapi_spec
    ):
        """测试异步调用 POST 方法"""
        # Mock async httpx response
        mock_response = Mock()
        mock_response.json = Mock(return_value={"id": 123})
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status = Mock()

        # Create a proper async context manager mock
        mock_client_instance = AsyncMock()
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(
            return_value=mock_client_instance
        )
        mock_client_instance.__aexit__ = AsyncMock()
        mock_async_client_class.return_value = mock_client_instance

        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        result = await client.call_tool_async("createUser", {"name": "Test"})

        assert result == {"id": 123}

    @patch("agentrun.tool.api.openapi.httpx.AsyncClient")
    async def test_call_tool_async_operation_not_found(
        self, mock_async_client_class, sample_openapi_spec
    ):
        """测试异步调用不存在的 operation"""
        mock_client_instance = AsyncMock()
        mock_async_client_class.return_value = mock_client_instance

        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        with pytest.raises(
            ValueError, match="Operation 'nonExistent' not found"
        ):
            await client.call_tool_async("nonExistent", {})

    @patch("agentrun.tool.api.openapi.httpx.AsyncClient")
    async def test_call_tool_async_no_server_url(self, mock_async_client_class):
        """测试异步调用没有 server URL"""
        spec_without_server = json.dumps({
            "openapi": "3.0.0",
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "testOp",
                    },
                },
            },
        })
        mock_client_instance = AsyncMock()
        mock_async_client_class.return_value = mock_client_instance

        client = ToolOpenAPIClient(protocol_spec=spec_without_server)
        with pytest.raises(ValueError, match="No server URL found"):
            await client.call_tool_async("testOp", {})

    async def test_list_tools_async(self, sample_openapi_spec):
        """测试异步获取工具列表"""
        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        tools = await client.list_tools_async()

        assert len(tools) > 0
        assert all(isinstance(tool, ToolInfo) for tool in tools)

    def test_resolve_ref(self):
        """测试 _resolve_ref 解析 $ref 引用"""
        spec = json.dumps({
            "openapi": "3.0.0",
            "components": {
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                        },
                    }
                }
            },
        })
        client = ToolOpenAPIClient(protocol_spec=spec)
        ref = "#/components/schemas/User"
        resolved = client._resolve_ref(ref)
        assert resolved is not None
        assert resolved["type"] == "object"
        assert "name" in resolved["properties"]

    def test_resolve_ref_invalid(self):
        """测试 _resolve_ref 无效引用"""
        spec = json.dumps({
            "openapi": "3.0.0",
            "components": {"schemas": {"User": {"type": "object"}}},
        })
        client = ToolOpenAPIClient(protocol_spec=spec)
        ref = "#/components/schemas/NonExistent"
        resolved = client._resolve_ref(ref)
        assert resolved == {}

    def test_resolve_ref_no_spec(self):
        """测试 _resolve_ref 没有 spec"""
        client = ToolOpenAPIClient(protocol_spec=None)
        ref = "#/components/schemas/User"
        resolved = client._resolve_ref(ref)
        assert resolved == {}

    def test_resolve_schema_with_ref(self):
        """测试 _resolve_schema 递归解析 $ref"""
        spec = json.dumps({
            "openapi": "3.0.0",
            "components": {
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    }
                }
            },
        })
        client = ToolOpenAPIClient(protocol_spec=spec)
        schema = {"$ref": "#/components/schemas/User"}
        resolved = client._resolve_schema(schema)
        assert resolved is not None
        assert resolved["type"] == "object"

    def test_resolve_schema_none(self):
        """测试 _resolve_schema 传入 None"""
        spec = json.dumps({"openapi": "3.0.0"})
        client = ToolOpenAPIClient(protocol_spec=spec)
        resolved = client._resolve_schema(None)
        assert resolved is None

    def test_resolve_schema_with_items(self):
        """测试 _resolve_schema 解析 items 中的 $ref"""
        spec = json.dumps({
            "openapi": "3.0.0",
            "components": {"schemas": {"Item": {"type": "string"}}},
        })
        client = ToolOpenAPIClient(protocol_spec=spec)
        schema = {
            "type": "array",
            "items": {"$ref": "#/components/schemas/Item"},
        }
        resolved = client._resolve_schema(schema)
        assert resolved is not None
        assert resolved["type"] == "array"
        assert resolved["items"]["type"] == "string"

    def test_resolve_schema_with_anyof(self):
        """测试 _resolve_schema 解析 anyOf 中的 $ref"""
        spec = json.dumps({
            "openapi": "3.0.0",
            "components": {
                "schemas": {
                    "StringType": {"type": "string"},
                    "NumberType": {"type": "number"},
                }
            },
        })
        client = ToolOpenAPIClient(protocol_spec=spec)
        schema = {
            "anyOf": [
                {"$ref": "#/components/schemas/StringType"},
                {"$ref": "#/components/schemas/NumberType"},
            ]
        }
        resolved = client._resolve_schema(schema)
        assert resolved is not None
        assert "anyOf" in resolved
        assert len(resolved["anyOf"]) == 2
        assert resolved["anyOf"][0]["type"] == "string"
        assert resolved["anyOf"][1]["type"] == "number"

    def test_server_url_fallback(self):
        """测试 server_url 使用 fallback_server_url"""
        spec = json.dumps(
            {"openapi": "3.0.0", "info": {"title": "Test API"}, "paths": {}}
        )
        client = ToolOpenAPIClient(
            protocol_spec=spec,
            fallback_server_url="https://fallback.example.com",
        )
        assert client.server_url == "https://fallback.example.com"

    def test_server_url_empty_servers_list(self):
        """测试 servers 为空列表时使用 fallback"""
        spec = json.dumps({
            "openapi": "3.0.0",
            "info": {"title": "Test API"},
            "servers": [],
            "paths": {},
        })
        client = ToolOpenAPIClient(
            protocol_spec=spec,
            fallback_server_url="https://fallback.example.com",
        )
        assert client.server_url == "https://fallback.example.com"

    @patch("agentrun.tool.api.openapi.httpx.Client")
    def test_call_tool_put_method(self, mock_client_class, sample_openapi_spec):
        """测试 PUT 方法调用（走 POST/PUT/PATCH 分支）"""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status = Mock()

        mock_client_instance = Mock()
        mock_client_instance.request.return_value = mock_response
        mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client_instance

        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        result = client.call_tool("updateUser", {"name": "Updated Name"})

        assert result == {"success": True}
        mock_client_instance.request.assert_called_once()
        call_args = mock_client_instance.request.call_args
        assert call_args[0][0] == "PUT"

    @patch("agentrun.tool.api.openapi.httpx.Client")
    def test_call_tool_delete_method(
        self, mock_client_class, sample_openapi_spec
    ):
        """测试 DELETE 方法调用（走 GET/DELETE 分支）"""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status = Mock()

        mock_client_instance = Mock()
        mock_client_instance.request.return_value = mock_response
        mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client_instance

        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        result = client.call_tool("deleteUser", {})

        assert result == {"success": True}
        mock_client_instance.request.assert_called_once()
        call_args = mock_client_instance.request.call_args
        assert call_args[0][0] == "DELETE"

    @patch("agentrun.tool.api.openapi.httpx.AsyncClient")
    async def test_call_tool_async_get_method(
        self, mock_async_client_class, sample_openapi_spec
    ):
        """测试异步 GET 方法调用"""
        mock_response = Mock()
        mock_response.json.return_value = {"id": 123}
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status = Mock()

        mock_client_instance = Mock()
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(
            return_value=mock_client_instance
        )
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_async_client_class.return_value = mock_client_instance

        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        result = await client.call_tool_async("getUser", {"id": "123"})

        assert result == {"id": 123}
        mock_client_instance.request.assert_called_once()
        call_args = mock_client_instance.request.call_args
        assert call_args[0][0] == "GET"

    @patch("agentrun.tool.api.openapi.httpx.AsyncClient")
    async def test_call_tool_async_text_response(
        self, mock_async_client_class, sample_openapi_spec
    ):
        """测试异步调用返回 text 响应"""
        mock_response = Mock()
        mock_response.text = "plain text response"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.json.side_effect = ValueError("No JSON")
        mock_response.raise_for_status = Mock()

        mock_client_instance = Mock()
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(
            return_value=mock_client_instance
        )
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_async_client_class.return_value = mock_client_instance

        client = ToolOpenAPIClient(protocol_spec=sample_openapi_spec)
        result = await client.call_tool_async("listUsers", {"limit": 10})

        assert result == "plain text response"

    def test_parse_operations_no_operation_id(self):
        """测试没有 operationId 时使用默认值"""
        spec = json.dumps({
            "openapi": "3.0.0",
            "info": {"title": "Test API"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/test": {"get": {"summary": "Test without operationId"}}
            },
        })
        client = ToolOpenAPIClient(protocol_spec=spec)
        operations = client._parse_operations()

        assert len(operations) == 1
        assert operations[0]["operation_id"] is not None
        assert operations[0]["method"] == "GET"

    def test_parse_operations_invalid_path_item(self):
        """测试无效的 path_item（非 dict）"""
        spec = json.dumps({
            "openapi": "3.0.0",
            "info": {"title": "Test API"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {"/test": "invalid"},
        })
        client = ToolOpenAPIClient(protocol_spec=spec)
        operations = client._parse_operations()

        assert operations == []

    def test_parse_operations_required_parameters(self):
        """测试 required 参数的解析"""
        spec = json.dumps({
            "openapi": "3.0.0",
            "info": {"title": "Test API"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users/{id}": {
                    "get": {
                        "operationId": "getUserById",
                        "parameters": [{
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }],
                    }
                }
            },
        })
        client = ToolOpenAPIClient(protocol_spec=spec)
        operations = client._parse_operations()

        assert len(operations) == 1
        op = operations[0]
        assert op["operation_id"] == "getUserById"
        assert op["input_schema"] is not None
        assert "id" in op["input_schema"]["properties"]
        assert "id" in op["input_schema"]["required"]


class AsyncMock(Mock):
    """Async mock helper"""

    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
