"""OpenAPI $ref 解析与请求构建单元测试

测试内容：
1. $ref 解析是否正确展开内部引用
2. Mock 服务端验证请求参数是否符合预期
"""

import json

import httpx
import respx

from agentrun.toolset.api.openapi import ApiSet, OpenAPI
from agentrun.toolset.model import ToolInfo, ToolSchema


class TestOpenAPIRefResolution:
    """测试 $ref 解析功能"""

    def test_resolve_simple_ref(self):
        """测试简单的 $ref 解析"""
        schema = {
            "openapi": "3.0.0",
            "components": {
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                    }
                }
            },
            "paths": {
                "/users": {
                    "post": {
                        "operationId": "createUser",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/User"
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }

        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")

        # 验证 $ref 已被展开
        resolved_schema = openapi._schema["paths"]["/users"]["post"][
            "requestBody"
        ]["content"]["application/json"]["schema"]

        assert "$ref" not in resolved_schema
        assert resolved_schema["type"] == "object"
        assert "name" in resolved_schema["properties"]
        assert "age" in resolved_schema["properties"]

    def test_resolve_nested_ref(self):
        """测试嵌套 $ref 解析"""
        schema = {
            "openapi": "3.0.0",
            "components": {
                "schemas": {
                    "Address": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                    "Person": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "address": {"$ref": "#/components/schemas/Address"},
                        },
                    },
                }
            },
            "paths": {
                "/person": {
                    "get": {
                        "operationId": "getPerson",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "$ref": (
                                                "#/components/schemas/Person"
                                            )
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }

        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")

        resolved = openapi._schema["paths"]["/person"]["get"]["responses"][
            "200"
        ]["content"]["application/json"]["schema"]

        # Person 已展开
        assert resolved["type"] == "object"
        assert "address" in resolved["properties"]

        # Address 也已展开
        address = resolved["properties"]["address"]
        assert "$ref" not in address
        assert address["type"] == "object"
        assert "city" in address["properties"]

    def test_resolve_ref_with_sibling_properties(self):
        """测试 $ref 与同级属性合并"""
        schema = {
            "openapi": "3.0.0",
            "components": {
                "schemas": {
                    "Base": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                    }
                }
            },
            "paths": {
                "/items": {
                    "get": {
                        "operationId": "getItems",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "$ref": "#/components/schemas/Base",
                                            "description": "Custom description",
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }

        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")

        resolved = openapi._schema["paths"]["/items"]["get"]["responses"][
            "200"
        ]["content"]["application/json"]["schema"]

        # $ref 已展开且保留同级属性
        assert "$ref" not in resolved
        assert resolved["type"] == "object"
        assert resolved["description"] == "Custom description"

    def test_resolve_ref_in_array_items(self):
        """测试数组 items 中的 $ref"""
        schema = {
            "openapi": "3.0.0",
            "components": {
                "schemas": {
                    "Item": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    }
                }
            },
            "paths": {
                "/items": {
                    "get": {
                        "operationId": "listItems",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "$ref": (
                                                    "#/components/schemas/Item"
                                                )
                                            },
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }

        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")

        resolved = openapi._schema["paths"]["/items"]["get"]["responses"][
            "200"
        ]["content"]["application/json"]["schema"]

        assert resolved["type"] == "array"
        assert "$ref" not in resolved["items"]
        assert resolved["items"]["type"] == "object"

    def test_resolve_ref_in_parameters(self):
        """测试参数中的 $ref"""
        schema = {
            "openapi": "3.0.0",
            "components": {
                "parameters": {
                    "PageSize": {
                        "name": "pageSize",
                        "in": "query",
                        "schema": {"type": "integer", "default": 10},
                    }
                }
            },
            "paths": {
                "/items": {
                    "get": {
                        "operationId": "listItems",
                        "parameters": [
                            {"$ref": "#/components/parameters/PageSize"}
                        ],
                    }
                }
            },
        }

        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")

        params = openapi._schema["paths"]["/items"]["get"]["parameters"]

        assert len(params) == 1
        assert "$ref" not in params[0]
        assert params[0]["name"] == "pageSize"
        assert params[0]["in"] == "query"


class TestOpenAPIRequestBuilding:
    """测试请求构建和 Mock 服务端验证"""

    @respx.mock
    def test_get_request_with_query_params(self):
        """测试 GET 请求带查询参数"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "parameters": [
                            {
                                "name": "page",
                                "in": "query",
                                "schema": {"type": "integer"},
                            },
                            {
                                "name": "limit",
                                "in": "query",
                                "schema": {"type": "integer"},
                            },
                        ],
                    }
                }
            },
        }

        # Mock 服务端
        route = respx.get("https://api.example.com/users").mock(
            return_value=httpx.Response(200, json={"users": [], "total": 0})
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        result = openapi.invoke_tool("listUsers", {"page": 1, "limit": 20})

        # 验证请求
        assert route.called
        request = route.calls.last.request
        assert "page=1" in str(request.url)
        assert "limit=20" in str(request.url)

        # 验证响应
        assert result["status_code"] == 200
        assert result["body"]["users"] == []

    @respx.mock
    def test_post_request_with_json_body(self):
        """测试 POST 请求带 JSON body"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users": {
                    "post": {
                        "operationId": "createUser",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "email": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    }
                }
            },
        }

        # Mock 服务端
        route = respx.post("https://api.example.com/users").mock(
            return_value=httpx.Response(
                201,
                json={"id": "123", "name": "Test", "email": "test@example.com"},
            )
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        result = openapi.invoke_tool(
            "createUser",
            {"body": {"name": "Test", "email": "test@example.com"}},
        )

        # 验证请求
        assert route.called
        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["name"] == "Test"
        assert body["email"] == "test@example.com"

        # 验证响应
        assert result["status_code"] == 201
        assert result["body"]["id"] == "123"

    @respx.mock
    def test_request_with_path_params(self):
        """测试路径参数"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users/{userId}": {
                    "get": {
                        "operationId": "getUser",
                        "parameters": [{
                            "name": "userId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }],
                    }
                }
            },
        }

        # Mock 服务端
        route = respx.get("https://api.example.com/users/abc123").mock(
            return_value=httpx.Response(
                200, json={"id": "abc123", "name": "Test User"}
            )
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        result = openapi.invoke_tool("getUser", {"userId": "abc123"})

        # 验证请求
        assert route.called
        assert (
            str(route.calls.last.request.url)
            == "https://api.example.com/users/abc123"
        )

        # 验证响应
        assert result["status_code"] == 200
        assert result["body"]["id"] == "abc123"

    @respx.mock
    def test_request_with_headers(self):
        """测试自定义请求头"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/protected": {
                    "get": {
                        "operationId": "getProtected",
                        "parameters": [{
                            "name": "X-API-Key",
                            "in": "header",
                            "required": True,
                            "schema": {"type": "string"},
                        }],
                    }
                }
            },
        }

        # Mock 服务端
        route = respx.get("https://api.example.com/protected").mock(
            return_value=httpx.Response(200, json={"data": "secret"})
        )

        openapi = OpenAPI(
            schema=json.dumps(schema),
            headers={"Authorization": "Bearer token123"},
        )
        openapi.invoke_tool("getProtected", {"X-API-Key": "my-api-key"})

        # 验证请求头
        assert route.called
        request = route.calls.last.request
        assert request.headers.get("Authorization") == "Bearer token123"
        assert request.headers.get("X-API-Key") == "my-api-key"

    @respx.mock
    def test_put_request_with_path_and_body(self):
        """测试 PUT 请求带路径参数和 body"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users/{userId}": {
                    "put": {
                        "operationId": "updateUser",
                        "parameters": [{
                            "name": "userId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"}
                                        },
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }

        # Mock 服务端
        route = respx.put("https://api.example.com/users/user123").mock(
            return_value=httpx.Response(
                200, json={"id": "user123", "name": "Updated"}
            )
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        openapi.invoke_tool(
            "updateUser",
            {"userId": "user123", "body": {"name": "Updated"}},
        )

        # 验证请求
        assert route.called
        request = route.calls.last.request
        assert "/users/user123" in str(request.url)
        body = json.loads(request.content)
        assert body["name"] == "Updated"

    @respx.mock
    def test_delete_request(self):
        """测试 DELETE 请求"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users/{userId}": {
                    "delete": {
                        "operationId": "deleteUser",
                        "parameters": [{
                            "name": "userId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }],
                    }
                }
            },
        }

        # Mock 服务端
        route = respx.delete("https://api.example.com/users/user456").mock(
            return_value=httpx.Response(204, content=b"")
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        result = openapi.invoke_tool("deleteUser", {"userId": "user456"})

        # 验证请求
        assert route.called
        assert result["status_code"] == 204


class TestOpenAPIWithRefRequestBuilding:
    """测试包含 $ref 的 schema 请求构建"""

    @respx.mock
    def test_post_with_ref_schema(self):
        """测试 POST 请求使用 $ref 定义的 schema"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "components": {
                "schemas": {
                    "CreateOrderRequest": {
                        "type": "object",
                        "required": ["product_id", "quantity"],
                        "properties": {
                            "product_id": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "notes": {"type": "string"},
                        },
                    }
                }
            },
            "paths": {
                "/orders": {
                    "post": {
                        "operationId": "createOrder",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/CreateOrderRequest"
                                    }
                                }
                            },
                        },
                    }
                }
            },
        }

        # Mock 服务端
        route = respx.post("https://api.example.com/orders").mock(
            return_value=httpx.Response(
                201, json={"order_id": "ord_123", "status": "pending"}
            )
        )

        openapi = OpenAPI(schema=json.dumps(schema))

        # 验证 $ref 已解析
        resolved = openapi._schema["paths"]["/orders"]["post"]["requestBody"][
            "content"
        ]["application/json"]["schema"]
        assert "$ref" not in resolved
        assert resolved["type"] == "object"

        # 发起请求
        result = openapi.invoke_tool(
            "createOrder",
            {"body": {"product_id": "prod_1", "quantity": 5}},
        )

        # 验证请求
        assert route.called
        body = json.loads(route.calls.last.request.content)
        assert body["product_id"] == "prod_1"
        assert body["quantity"] == 5

        # 验证响应
        assert result["status_code"] == 201
        assert result["body"]["order_id"] == "ord_123"

    @respx.mock
    def test_get_with_ref_parameters(self):
        """测试使用 $ref 定义的参数"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "components": {
                "parameters": {
                    "PaginationPage": {
                        "name": "page",
                        "in": "query",
                        "schema": {"type": "integer", "default": 1},
                    },
                    "PaginationLimit": {
                        "name": "limit",
                        "in": "query",
                        "schema": {"type": "integer", "default": 10},
                    },
                }
            },
            "paths": {
                "/products": {
                    "get": {
                        "operationId": "listProducts",
                        "parameters": [
                            {"$ref": "#/components/parameters/PaginationPage"},
                            {"$ref": "#/components/parameters/PaginationLimit"},
                        ],
                    }
                }
            },
        }

        # Mock 服务端
        route = respx.get("https://api.example.com/products").mock(
            return_value=httpx.Response(200, json={"products": []})
        )

        openapi = OpenAPI(schema=json.dumps(schema))

        # 验证参数 $ref 已解析
        params = openapi._schema["paths"]["/products"]["get"]["parameters"]
        assert len(params) == 2
        assert all("$ref" not in p for p in params)

        # 发起请求
        openapi.invoke_tool("listProducts", {"page": 2, "limit": 50})

        # 验证请求
        assert route.called
        url = str(route.calls.last.request.url)
        assert "page=2" in url
        assert "limit=50" in url


class TestApiSetFromOpenAPI:
    """测试 ApiSet.from_openapi_schema"""

    def test_create_apiset_from_schema(self):
        """测试从 OpenAPI schema 创建 ApiSet"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "summary": "List all users",
                    },
                    "post": {
                        "operationId": "createUser",
                        "summary": "Create a new user",
                    },
                }
            },
        }

        apiset = ApiSet.from_openapi_schema(
            schema=json.dumps(schema),
            base_url="https://api.example.com",
        )

        tools = apiset.tools()
        assert len(tools) == 2

        tool_names = {t.name for t in tools}
        assert "listUsers" in tool_names
        assert "createUser" in tool_names

    def test_apiset_get_tool(self):
        """测试 ApiSet.get_tool"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "description": "List all users",
                    }
                }
            },
        }

        apiset = ApiSet.from_openapi_schema(schema=json.dumps(schema))

        tool = apiset.get_tool("listUsers")
        assert tool is not None
        assert tool.name == "listUsers"
        assert tool.description == "List all users"

        # 不存在的工具
        assert apiset.get_tool("nonexistent") is None

    @respx.mock
    def test_apiset_invoke(self):
        """测试 ApiSet.invoke"""
        schema = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/echo": {
                    "post": {
                        "operationId": "echo",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "message": {"type": "string"}
                                        },
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }

        route = respx.post("https://api.example.com/echo").mock(
            return_value=httpx.Response(200, json={"echo": "Hello World"})
        )

        apiset = ApiSet.from_openapi_schema(schema=json.dumps(schema))
        result = apiset.invoke("echo", {"body": {"message": "Hello World"}})

        assert route.called
        assert result["status_code"] == 200
        assert result["body"]["echo"] == "Hello World"


class TestOpenAPIEdgeCases:
    """测试边界情况"""

    def test_invalid_ref_gracefully_handled(self):
        """测试无效 $ref 不会导致崩溃"""
        schema = {
            "openapi": "3.0.0",
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "test",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "$ref": "#/components/schemas/NonExistent"
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }

        # 应该不会抛出异常
        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")

        # 无效的 $ref 保持原样
        resolved = openapi._schema["paths"]["/test"]["get"]["responses"]["200"][
            "content"
        ]["application/json"]["schema"]
        assert "$ref" in resolved

    def test_external_ref_not_resolved(self):
        """测试外部 $ref 不会被解析"""
        schema = {
            "openapi": "3.0.0",
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "test",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "$ref": "https://example.com/schemas/external.json"
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }

        openapi = OpenAPI(schema=json.dumps(schema), base_url="http://test")

        # 外部 $ref 保持原样
        resolved = openapi._schema["paths"]["/test"]["get"]["responses"]["200"][
            "content"
        ]["application/json"]["schema"]
        assert "$ref" in resolved
        assert resolved["$ref"] == "https://example.com/schemas/external.json"

    def test_empty_schema(self):
        """测试空 schema"""
        openapi = OpenAPI(schema="{}", base_url="http://test")
        assert openapi._operations == {}

    def test_yaml_schema(self):
        """测试 YAML 格式的 schema"""
        yaml_schema = """
openapi: "3.0.0"
servers:
  - url: https://api.example.com
paths:
  /hello:
    get:
      operationId: sayHello
      summary: Say hello
"""
        openapi = OpenAPI(schema=yaml_schema)
        assert "sayHello" in openapi._operations


class TestCoffeeShopSchema:
    """测试希希咖啡店 OpenAPI Schema（复杂嵌套 $ref 场景）"""

    @staticmethod
    def _get_coffee_shop_schema():
        """获取咖啡店 schema"""
        return {
            "openapi": "3.1.0",
            "servers": [{"url": "http://127.0.0.1:8001"}],
            "components": {
                "schemas": {
                    "CreateOrderRequest": {
                        "properties": {
                            "items": {
                                "items": {
                                    "$ref": "#/components/schemas/OrderItem"
                                },
                                "type": "array",
                                "title": "Items",
                            },
                            "customer_name": {
                                "type": "string",
                                "title": "Customer Name",
                            },
                            "customer_phone": {
                                "type": "string",
                                "title": "Customer Phone",
                            },
                            "notes": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "title": "Notes",
                            },
                        },
                        "type": "object",
                        "required": [
                            "items",
                            "customer_name",
                            "customer_phone",
                        ],
                        "title": "CreateOrderRequest",
                    },
                    "OrderItem": {
                        "properties": {
                            "product_id": {"type": "integer"},
                            "name": {"type": "string"},
                            "price": {"type": "number"},
                            "quantity": {"type": "integer"},
                        },
                        "type": "object",
                        "required": ["product_id", "name", "price", "quantity"],
                        "title": "OrderItem",
                    },
                    "UpdateOrderStatusRequest": {
                        "properties": {"status": {"type": "string"}},
                        "type": "object",
                        "required": ["status"],
                    },
                }
            },
            "paths": {
                "/api/coffee/products": {
                    "get": {
                        "operationId": "get_products_api_coffee_products_get",
                        "parameters": [{
                            "name": "category",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                        }],
                    }
                },
                "/api/coffee/orders": {
                    "post": {
                        "operationId": "create_order_api_coffee_orders_post",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/CreateOrderRequest"
                                    }
                                }
                            },
                        },
                    },
                    "get": {
                        "operationId": "get_orders_api_coffee_orders_get",
                        "parameters": [
                            {
                                "name": "status",
                                "in": "query",
                                "schema": {"type": "string"},
                            },
                            {
                                "name": "limit",
                                "in": "query",
                                "schema": {"type": "integer", "default": 50},
                            },
                        ],
                    },
                },
                "/api/coffee/orders/{order_id}/status": {
                    "put": {
                        "operationId": "update_order_status_put",
                        "parameters": [{
                            "name": "order_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                        }],
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/UpdateOrderStatusRequest"
                                    }
                                }
                            },
                        },
                    }
                },
            },
        }

    def test_resolve_nested_ref_in_array(self):
        """测试数组中嵌套的 $ref 解析（OrderItem）"""
        schema = self._get_coffee_shop_schema()
        openapi = OpenAPI(schema=json.dumps(schema))

        # 验证 CreateOrderRequest 中的 items 数组的 $ref 已解析
        resolved = openapi._schema["paths"]["/api/coffee/orders"]["post"][
            "requestBody"
        ]["content"]["application/json"]["schema"]

        assert "$ref" not in resolved
        assert resolved["type"] == "object"
        assert "items" in resolved["properties"]

        items_schema = resolved["properties"]["items"]
        assert items_schema["type"] == "array"
        assert "$ref" not in items_schema["items"]
        assert items_schema["items"]["type"] == "object"
        assert "product_id" in items_schema["items"]["properties"]

    def test_apiset_resolves_request_body_ref(self):
        """测试 ApiSet 正确解析 requestBody 中的 $ref"""
        schema = self._get_coffee_shop_schema()
        apiset = ApiSet.from_openapi_schema(schema=json.dumps(schema))

        tool = apiset.get_tool("create_order_api_coffee_orders_post")
        assert tool is not None
        assert tool.parameters is not None
        assert tool.parameters.properties is not None

        # 验证属性已正确展开
        prop_names = list(tool.parameters.properties.keys())
        assert "items" in prop_names
        assert "customer_name" in prop_names
        assert "customer_phone" in prop_names

        # 验证 required 字段
        assert tool.parameters.required is not None
        assert "items" in tool.parameters.required
        assert "customer_name" in tool.parameters.required

    @respx.mock
    def test_create_order_request(self):
        """测试创建订单请求构建"""
        schema = self._get_coffee_shop_schema()

        route = respx.post("http://127.0.0.1:8001/api/coffee/orders").mock(
            return_value=httpx.Response(
                200,
                json={
                    "order_id": 1,
                    "status": "pending",
                    "total": 68.0,
                },
            )
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        result = openapi.invoke_tool(
            "create_order_api_coffee_orders_post",
            {
                "body": {
                    "items": [
                        {
                            "product_id": 1,
                            "name": "拿铁",
                            "price": 28.0,
                            "quantity": 2,
                        },
                        {
                            "product_id": 2,
                            "name": "美式",
                            "price": 12.0,
                            "quantity": 1,
                        },
                    ],
                    "customer_name": "张三",
                    "customer_phone": "13800138000",
                    "notes": "少糖",
                }
            },
        )

        # 验证请求
        assert route.called
        request = route.calls.last.request
        body = json.loads(request.content)

        assert len(body["items"]) == 2
        assert body["items"][0]["product_id"] == 1
        assert body["items"][0]["name"] == "拿铁"
        assert body["customer_name"] == "张三"
        assert body["notes"] == "少糖"

        # 验证响应
        assert result["status_code"] == 200
        assert result["body"]["order_id"] == 1

    @respx.mock
    def test_update_order_status_with_path_and_body(self):
        """测试更新订单状态（路径参数 + requestBody $ref）"""
        schema = self._get_coffee_shop_schema()

        route = respx.put(
            "http://127.0.0.1:8001/api/coffee/orders/123/status"
        ).mock(
            return_value=httpx.Response(
                200, json={"order_id": 123, "status": "completed"}
            )
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        result = openapi.invoke_tool(
            "update_order_status_put",
            {"order_id": 123, "body": {"status": "completed"}},
        )

        # 验证请求
        assert route.called
        request = route.calls.last.request
        assert "/orders/123/status" in str(request.url)
        body = json.loads(request.content)
        assert body["status"] == "completed"

        # 验证响应
        assert result["status_code"] == 200

    @respx.mock
    def test_get_products_with_query_param(self):
        """测试获取商品列表（query 参数）"""
        schema = self._get_coffee_shop_schema()

        route = respx.get("http://127.0.0.1:8001/api/coffee/products").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"id": 1, "name": "拿铁", "price": 28.0, "category": "咖啡"}
                ],
            )
        )

        openapi = OpenAPI(schema=json.dumps(schema))
        result = openapi.invoke_tool(
            "get_products_api_coffee_products_get",
            {"category": "咖啡"},
        )

        # 验证请求
        assert route.called
        assert "category=" in str(route.calls.last.request.url)

        # 验证响应
        assert result["status_code"] == 200
        assert len(result["body"]) == 1

    @respx.mock
    def test_apiset_invoke_create_order(self):
        """测试通过 ApiSet 调用创建订单"""
        schema = self._get_coffee_shop_schema()

        route = respx.post("http://127.0.0.1:8001/api/coffee/orders").mock(
            return_value=httpx.Response(200, json={"order_id": 99})
        )

        apiset = ApiSet.from_openapi_schema(schema=json.dumps(schema))
        result = apiset.invoke(
            "create_order_api_coffee_orders_post",
            {
                "body": {
                    "items": [{
                        "product_id": 1,
                        "name": "卡布奇诺",
                        "price": 30.0,
                        "quantity": 1,
                    }],
                    "customer_name": "李四",
                    "customer_phone": "13900139000",
                }
            },
        )

        assert route.called
        body = json.loads(route.calls.last.request.content)
        assert body["customer_name"] == "李四"
        assert result["body"]["order_id"] == 99

    def test_tool_schema(self):
        """测试工具的参数 schema"""
        schema = self._get_coffee_shop_schema()
        apiset = ApiSet.from_openapi_schema(schema=json.dumps(schema))

        tool = apiset.get_tool("create_order_api_coffee_orders_post")

        assert tool is not None
        assert tool.name == "create_order_api_coffee_orders_post"
        assert tool.parameters is not None
        assert tool.parameters.type == "object"
        assert tool.parameters.required == [
            "items",
            "customer_name",
            "customer_phone",
        ]

        # 验证 properties 存在且包含所有字段
        props = tool.parameters.properties
        assert props is not None
        assert set(props.keys()) == {
            "items",
            "customer_name",
            "customer_phone",
            "notes",
        }

        # 验证 items 数组结构
        items_schema = props["items"]
        assert items_schema.type == "array"
        assert items_schema.items is not None
        assert items_schema.items.type == "object"
        assert items_schema.items.properties is not None
        assert set(items_schema.items.properties.keys()) == {
            "product_id",
            "name",
            "price",
            "quantity",
        }
        assert items_schema.items.required == [
            "product_id",
            "name",
            "price",
            "quantity",
        ]

        # 验证 items 中的字段类型
        item_props = items_schema.items.properties
        assert item_props["product_id"].type == "integer"
        assert item_props["name"].type == "string"
        assert item_props["price"].type == "number"
        assert item_props["quantity"].type == "integer"

        # 验证简单字段
        assert props["customer_name"].type == "string"
        assert props["customer_phone"].type == "string"

        # 验证 anyOf 类型（nullable）
        notes_schema = props["notes"]
        assert notes_schema.any_of is not None
        assert len(notes_schema.any_of) == 2
        any_of_types = {s.type for s in notes_schema.any_of}
        assert any_of_types == {"string", "null"}

        # 验证 to_json_schema 能正确转换回去
        json_schema = tool.parameters.to_json_schema()
        assert json_schema["type"] == "object"
        assert "items" in json_schema["properties"]
        assert json_schema["properties"]["items"]["type"] == "array"
