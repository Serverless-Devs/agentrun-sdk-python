"""ToolSet 模型单元测试 / ToolSet Model Unit Tests

测试 toolset 模块中数据模型和工具 schema 的相关功能。
Tests data models and tool schema functionality in the toolset module.
"""

import pytest

from agentrun.toolset.model import (
    APIKeyAuthParameter,
    Authorization,
    AuthorizationParameters,
    MCPServerConfig,
    OpenAPIToolMeta,
    SchemaType,
    ToolInfo,
    ToolMeta,
    ToolSchema,
    ToolSetListInput,
    ToolSetSchema,
    ToolSetSpec,
    ToolSetStatus,
    ToolSetStatusOutputs,
    ToolSetStatusOutputsUrls,
)


class TestSchemaType:
    """测试 SchemaType 枚举"""

    def test_mcp_type(self):
        """测试 MCP 类型"""
        assert SchemaType.MCP == "MCP"
        assert SchemaType.MCP.value == "MCP"

    def test_openapi_type(self):
        """测试 OpenAPI 类型"""
        assert SchemaType.OpenAPI == "OpenAPI"
        assert SchemaType.OpenAPI.value == "OpenAPI"


class TestToolSetStatusOutputsUrls:
    """测试 ToolSetStatusOutputsUrls 模型"""

    def test_default_values(self):
        """测试默认值"""
        urls = ToolSetStatusOutputsUrls()
        assert urls.internet_url is None
        assert urls.intranet_url is None

    def test_with_values(self):
        """测试带值创建"""
        urls = ToolSetStatusOutputsUrls(
            internet_url="https://public.example.com",
            intranet_url="https://internal.example.com",
        )
        assert urls.internet_url == "https://public.example.com"
        assert urls.intranet_url == "https://internal.example.com"


class TestMCPServerConfig:
    """测试 MCPServerConfig 模型"""

    def test_default_values(self):
        """测试默认值"""
        config = MCPServerConfig()
        assert config.headers is None
        assert config.transport_type is None
        assert config.url is None

    def test_with_values(self):
        """测试带值创建"""
        config = MCPServerConfig(
            headers={"Authorization": "Bearer token"},
            transport_type="sse",
            url="https://mcp.example.com",
        )
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.transport_type == "sse"
        assert config.url == "https://mcp.example.com"


class TestToolMeta:
    """测试 ToolMeta 模型"""

    def test_default_values(self):
        """测试默认值"""
        meta = ToolMeta()
        assert meta.description is None
        assert meta.input_schema is None
        assert meta.name is None

    def test_with_values(self):
        """测试带值创建"""
        meta = ToolMeta(
            name="my_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {"arg1": {"type": "string"}},
            },
        )
        assert meta.name == "my_tool"
        assert meta.description == "A test tool"
        assert meta.input_schema["type"] == "object"


class TestOpenAPIToolMeta:
    """测试 OpenAPIToolMeta 模型"""

    def test_default_values(self):
        """测试默认值"""
        meta = OpenAPIToolMeta()
        assert meta.method is None
        assert meta.path is None
        assert meta.tool_id is None
        assert meta.tool_name is None

    def test_with_values(self):
        """测试带值创建"""
        meta = OpenAPIToolMeta(
            method="POST",
            path="/api/users",
            tool_id="create_user_001",
            tool_name="createUser",
        )
        assert meta.method == "POST"
        assert meta.path == "/api/users"
        assert meta.tool_id == "create_user_001"
        assert meta.tool_name == "createUser"


class TestToolSetStatusOutputs:
    """测试 ToolSetStatusOutputs 模型"""

    def test_default_values(self):
        """测试默认值"""
        outputs = ToolSetStatusOutputs()
        assert outputs.function_arn is None
        assert outputs.mcp_server_config is None
        assert outputs.open_api_tools is None
        assert outputs.tools is None
        assert outputs.urls is None

    def test_with_nested_values(self):
        """测试带嵌套值创建"""
        outputs = ToolSetStatusOutputs(
            function_arn="arn:aws:lambda:region:account:function:name",
            mcp_server_config=MCPServerConfig(url="https://mcp.example.com"),
            open_api_tools=[
                OpenAPIToolMeta(method="GET", path="/api/users"),
            ],
            tools=[
                ToolMeta(name="tool1", description="Tool 1"),
            ],
            urls=ToolSetStatusOutputsUrls(
                internet_url="https://public.example.com"
            ),
        )
        assert outputs.function_arn is not None
        assert outputs.mcp_server_config is not None
        assert outputs.mcp_server_config.url == "https://mcp.example.com"
        assert len(outputs.open_api_tools) == 1
        assert len(outputs.tools) == 1
        assert outputs.urls.internet_url == "https://public.example.com"


class TestAPIKeyAuthParameter:
    """测试 APIKeyAuthParameter 模型"""

    def test_default_values(self):
        """测试默认值"""
        param = APIKeyAuthParameter()
        assert param.encrypted is None
        assert param.in_ is None
        assert param.key is None
        assert param.value is None

    def test_with_values(self):
        """测试带值创建"""
        param = APIKeyAuthParameter(
            encrypted=True,
            in_="header",
            key="X-API-Key",
            value="secret-key-123",
        )
        assert param.encrypted is True
        assert param.in_ == "header"
        assert param.key == "X-API-Key"
        assert param.value == "secret-key-123"


class TestAuthorization:
    """测试 Authorization 模型"""

    def test_default_values(self):
        """测试默认值"""
        auth = Authorization()
        assert auth.parameters is None
        assert auth.type is None

    def test_with_api_key_auth(self):
        """测试带 API Key 认证"""
        auth = Authorization(
            type="APIKey",
            parameters=AuthorizationParameters(
                api_key_parameter=APIKeyAuthParameter(
                    in_="header",
                    key="X-API-Key",
                    value="my-secret-key",
                )
            ),
        )
        assert auth.type == "APIKey"
        assert auth.parameters.api_key_parameter.key == "X-API-Key"


class TestToolSetSchema:
    """测试 ToolSetSchema 模型"""

    def test_default_values(self):
        """测试默认值"""
        schema = ToolSetSchema()
        assert schema.detail is None
        assert schema.type is None

    def test_with_mcp_type(self):
        """测试 MCP 类型"""
        schema = ToolSetSchema(
            type=SchemaType.MCP,
            detail='{"mcp": "config"}',
        )
        assert schema.type == SchemaType.MCP
        assert schema.detail == '{"mcp": "config"}'

    def test_with_openapi_type(self):
        """测试 OpenAPI 类型"""
        schema = ToolSetSchema(
            type=SchemaType.OpenAPI,
            detail='{"openapi": "3.0.0"}',
        )
        assert schema.type == SchemaType.OpenAPI


class TestToolSetSpec:
    """测试 ToolSetSpec 模型"""

    def test_default_values(self):
        """测试默认值"""
        spec = ToolSetSpec()
        assert spec.auth_config is None
        assert spec.tool_schema is None

    def test_with_values(self):
        """测试带值创建"""
        spec = ToolSetSpec(
            auth_config=Authorization(type="APIKey"),
            tool_schema=ToolSetSchema(type=SchemaType.OpenAPI),
        )
        assert spec.auth_config.type == "APIKey"
        assert spec.tool_schema.type == SchemaType.OpenAPI


class TestToolSetStatus:
    """测试 ToolSetStatus 模型"""

    def test_default_values(self):
        """测试默认值"""
        status = ToolSetStatus()
        assert status.observed_generation is None
        assert status.observed_time is None
        assert status.outputs is None
        assert status.phase is None

    def test_with_values(self):
        """测试带值创建"""
        status = ToolSetStatus(
            observed_generation=1,
            observed_time="2024-01-01T00:00:00Z",
            phase="Ready",
            outputs=ToolSetStatusOutputs(
                urls=ToolSetStatusOutputsUrls(
                    internet_url="https://example.com"
                )
            ),
        )
        assert status.observed_generation == 1
        assert status.phase == "Ready"
        assert status.outputs.urls.internet_url == "https://example.com"


class TestToolSetListInput:
    """测试 ToolSetListInput 模型"""

    def test_default_values(self):
        """测试默认值"""
        input_obj = ToolSetListInput()
        assert input_obj.keyword is None
        assert input_obj.label_selector is None

    def test_with_values(self):
        """测试带值创建"""
        input_obj = ToolSetListInput(
            keyword="my-tool",
            label_selector=["env=prod", "team=backend"],
        )
        assert input_obj.keyword == "my-tool"
        assert len(input_obj.label_selector) == 2


class TestToolSchema:
    """测试 ToolSchema 模型"""

    def test_default_values(self):
        """测试默认值"""
        schema = ToolSchema()
        assert schema.type is None
        assert schema.description is None
        assert schema.properties is None

    def test_object_schema(self):
        """测试对象类型 schema"""
        schema = ToolSchema(
            type="object",
            description="A user object",
            properties={
                "name": ToolSchema(type="string", description="User name"),
                "age": ToolSchema(type="integer", description="User age"),
            },
            required=["name"],
            additional_properties=False,
        )
        assert schema.type == "object"
        assert "name" in schema.properties
        assert schema.required == ["name"]
        assert schema.additional_properties is False

    def test_array_schema(self):
        """测试数组类型 schema"""
        schema = ToolSchema(
            type="array",
            items=ToolSchema(type="string"),
            min_items=1,
            max_items=10,
        )
        assert schema.type == "array"
        assert schema.items.type == "string"
        assert schema.min_items == 1
        assert schema.max_items == 10

    def test_string_schema_with_constraints(self):
        """测试带约束的字符串 schema"""
        schema = ToolSchema(
            type="string",
            pattern=r"^[a-z]+$",
            min_length=1,
            max_length=100,
            format="email",
            enum=["a", "b", "c"],
        )
        assert schema.pattern == r"^[a-z]+$"
        assert schema.min_length == 1
        assert schema.max_length == 100
        assert schema.format == "email"
        assert schema.enum == ["a", "b", "c"]

    def test_number_schema_with_constraints(self):
        """测试带约束的数值 schema"""
        schema = ToolSchema(
            type="number",
            minimum=0,
            maximum=100,
            exclusive_minimum=0.0,
            exclusive_maximum=100.0,
        )
        assert schema.minimum == 0
        assert schema.maximum == 100
        assert schema.exclusive_minimum == 0.0
        assert schema.exclusive_maximum == 100.0

    def test_union_types(self):
        """测试联合类型"""
        schema = ToolSchema(
            any_of=[
                ToolSchema(type="string"),
                ToolSchema(type="null"),
            ],
            one_of=[
                ToolSchema(type="integer"),
                ToolSchema(type="number"),
            ],
            all_of=[
                ToolSchema(properties={"a": ToolSchema(type="string")}),
                ToolSchema(properties={"b": ToolSchema(type="integer")}),
            ],
        )
        assert len(schema.any_of) == 2
        assert len(schema.one_of) == 2
        assert len(schema.all_of) == 2

    def test_default_value(self):
        """测试默认值"""
        schema = ToolSchema(type="string", default="hello")
        assert schema.default == "hello"

    def test_from_any_openapi_schema_simple(self):
        """测试从简单 OpenAPI schema 创建"""
        openapi_schema = {
            "type": "string",
            "description": "A simple string",
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert schema.type == "string"
        assert schema.description == "A simple string"

    def test_from_any_openapi_schema_none(self):
        """测试从 None 创建"""
        schema = ToolSchema.from_any_openapi_schema(None)
        assert schema.type == "string"

    def test_from_any_openapi_schema_empty_dict(self):
        """测试从空字典创建"""
        schema = ToolSchema.from_any_openapi_schema({})
        # 空字典被视为 falsy，所以返回默认的 string 类型
        assert schema.type == "string"

    def test_from_any_openapi_schema_non_dict(self):
        """测试从非字典 schema 创建"""
        schema = ToolSchema.from_any_openapi_schema("invalid")
        assert schema.type == "string"

    def test_from_any_openapi_schema_with_properties(self):
        """测试从带 properties 的 schema 创建"""
        openapi_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
            "additionalProperties": False,
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert schema.type == "object"
        assert "name" in schema.properties
        assert "age" in schema.properties
        assert schema.properties["name"].type == "string"
        assert schema.required == ["name"]
        assert schema.additional_properties is False

    def test_from_any_openapi_schema_with_items(self):
        """测试从带 items 的数组 schema 创建"""
        openapi_schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert schema.type == "array"
        assert schema.items.type == "string"
        assert schema.min_items == 1
        assert schema.max_items == 10

    def test_from_any_openapi_schema_with_anyof(self):
        """测试从带 anyOf 的 schema 创建"""
        openapi_schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert len(schema.any_of) == 2
        assert schema.any_of[0].type == "string"
        assert schema.any_of[1].type == "null"

    def test_from_any_openapi_schema_with_oneof(self):
        """测试从带 oneOf 的 schema 创建"""
        openapi_schema = {
            "oneOf": [
                {"type": "integer"},
                {"type": "number"},
            ]
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert len(schema.one_of) == 2

    def test_from_any_openapi_schema_with_allof(self):
        """测试从带 allOf 的 schema 创建"""
        openapi_schema = {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "integer"}}},
            ]
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert len(schema.all_of) == 2

    def test_from_any_openapi_schema_with_string_constraints(self):
        """测试从带字符串约束的 schema 创建"""
        openapi_schema = {
            "type": "string",
            "pattern": "^[a-z]+$",
            "minLength": 1,
            "maxLength": 100,
            "format": "email",
            "enum": ["a", "b", "c"],
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert schema.pattern == "^[a-z]+$"
        assert schema.min_length == 1
        assert schema.max_length == 100
        assert schema.format == "email"
        assert schema.enum == ["a", "b", "c"]

    def test_from_any_openapi_schema_with_number_constraints(self):
        """测试从带数值约束的 schema 创建"""
        openapi_schema = {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "exclusiveMinimum": 0.0,
            "exclusiveMaximum": 100.0,
            "default": 50,
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert schema.minimum == 0
        assert schema.maximum == 100
        assert schema.exclusive_minimum == 0.0
        assert schema.exclusive_maximum == 100.0
        assert schema.default == 50

    def test_to_json_schema_simple(self):
        """测试简单 schema 转换为 JSON Schema"""
        schema = ToolSchema(type="string", description="A string")
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "string"
        assert json_schema["description"] == "A string"

    def test_to_json_schema_object(self):
        """测试对象 schema 转换为 JSON Schema"""
        schema = ToolSchema(
            type="object",
            title="User",
            properties={
                "name": ToolSchema(type="string"),
                "age": ToolSchema(type="integer"),
            },
            required=["name"],
            additional_properties=False,
        )
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "object"
        assert json_schema["title"] == "User"
        assert "properties" in json_schema
        assert json_schema["properties"]["name"]["type"] == "string"
        assert json_schema["required"] == ["name"]
        assert json_schema["additionalProperties"] is False

    def test_to_json_schema_array(self):
        """测试数组 schema 转换为 JSON Schema"""
        schema = ToolSchema(
            type="array",
            items=ToolSchema(type="string"),
            min_items=1,
            max_items=10,
        )
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "array"
        assert json_schema["items"]["type"] == "string"
        assert json_schema["minItems"] == 1
        assert json_schema["maxItems"] == 10

    def test_to_json_schema_string_constraints(self):
        """测试带字符串约束的 schema 转换"""
        schema = ToolSchema(
            type="string",
            pattern="^[a-z]+$",
            min_length=1,
            max_length=100,
            format="email",
            enum=["a", "b", "c"],
        )
        json_schema = schema.to_json_schema()
        assert json_schema["pattern"] == "^[a-z]+$"
        assert json_schema["minLength"] == 1
        assert json_schema["maxLength"] == 100
        assert json_schema["format"] == "email"
        assert json_schema["enum"] == ["a", "b", "c"]

    def test_to_json_schema_number_constraints(self):
        """测试带数值约束的 schema 转换"""
        schema = ToolSchema(
            type="number",
            minimum=0,
            maximum=100,
            exclusive_minimum=0.0,
            exclusive_maximum=100.0,
            default=50,
        )
        json_schema = schema.to_json_schema()
        assert json_schema["minimum"] == 0
        assert json_schema["maximum"] == 100
        assert json_schema["exclusiveMinimum"] == 0.0
        assert json_schema["exclusiveMaximum"] == 100.0
        assert json_schema["default"] == 50

    def test_to_json_schema_union_types(self):
        """测试联合类型 schema 转换"""
        schema = ToolSchema(
            any_of=[ToolSchema(type="string"), ToolSchema(type="null")],
            one_of=[ToolSchema(type="integer"), ToolSchema(type="number")],
            all_of=[
                ToolSchema(type="object"),
                ToolSchema(type="object"),
            ],
        )
        json_schema = schema.to_json_schema()
        assert len(json_schema["anyOf"]) == 2
        assert len(json_schema["oneOf"]) == 2
        assert len(json_schema["allOf"]) == 2


class TestToolInfo:
    """测试 ToolInfo 模型"""

    def test_default_values(self):
        """测试默认值"""
        info = ToolInfo()
        assert info.name is None
        assert info.description is None
        assert info.parameters is None

    def test_with_values(self):
        """测试带值创建"""
        info = ToolInfo(
            name="my_tool",
            description="A test tool",
            parameters=ToolSchema(type="object"),
        )
        assert info.name == "my_tool"
        assert info.description == "A test tool"
        assert info.parameters.type == "object"

    def test_from_mcp_tool_with_object(self):
        """测试从 MCP Tool 对象创建 ToolInfo"""

        class MockMCPTool:
            name = "test_tool"
            description = "A test tool"
            inputSchema = {
                "type": "object",
                "properties": {"arg1": {"type": "string"}},
            }

        tool = MockMCPTool()
        info = ToolInfo.from_mcp_tool(tool)
        assert info.name == "test_tool"
        assert info.description == "A test tool"
        assert info.parameters.type == "object"
        assert "arg1" in info.parameters.properties

    def test_from_mcp_tool_with_input_schema_snake_case(self):
        """测试从带 input_schema 的 MCP Tool 对象创建"""

        class MockMCPTool:
            name = "test_tool"
            description = "A test tool"
            input_schema = {
                "type": "object",
                "properties": {"arg1": {"type": "string"}},
            }

        tool = MockMCPTool()
        info = ToolInfo.from_mcp_tool(tool)
        assert info.name == "test_tool"
        assert info.parameters.type == "object"

    def test_from_mcp_tool_with_dict(self):
        """测试从字典格式创建 ToolInfo"""
        tool_dict = {
            "name": "dict_tool",
            "description": "A dict tool",
            "inputSchema": {
                "type": "object",
                "properties": {"param1": {"type": "integer"}},
            },
        }
        info = ToolInfo.from_mcp_tool(tool_dict)
        assert info.name == "dict_tool"
        assert info.description == "A dict tool"
        assert info.parameters.type == "object"

    def test_from_mcp_tool_with_dict_snake_case(self):
        """测试从带 input_schema 的字典格式创建"""
        tool_dict = {
            "name": "dict_tool",
            "description": "A dict tool",
            "input_schema": {
                "type": "object",
                "properties": {"param1": {"type": "integer"}},
            },
        }
        info = ToolInfo.from_mcp_tool(tool_dict)
        assert info.name == "dict_tool"
        assert info.parameters.type == "object"

    def test_from_mcp_tool_no_input_schema(self):
        """测试没有 input_schema 的情况"""
        tool_dict = {
            "name": "simple_tool",
            "description": "A simple tool",
        }
        info = ToolInfo.from_mcp_tool(tool_dict)
        assert info.name == "simple_tool"
        assert info.parameters.type == "object"
        assert info.parameters.properties == {}

    def test_from_mcp_tool_with_model_dump(self):
        """测试 input_schema 有 model_dump 方法的情况"""

        class MockInputSchema:

            def model_dump(self):
                return {
                    "type": "object",
                    "properties": {"field": {"type": "string"}},
                }

        class MockMCPTool:
            name = "tool_with_model"
            description = "Tool with model"
            inputSchema = MockInputSchema()

        tool = MockMCPTool()
        info = ToolInfo.from_mcp_tool(tool)
        assert info.name == "tool_with_model"
        assert info.parameters.type == "object"
        assert "field" in info.parameters.properties

    def test_from_mcp_tool_unsupported_format(self):
        """测试不支持的格式"""
        with pytest.raises(ValueError, match="Unsupported MCP tool format"):
            ToolInfo.from_mcp_tool("invalid")

    def test_from_mcp_tool_missing_name_object(self):
        """测试缺少 name 的对象"""

        class MockMCPToolNoName:
            # 必须有 name 属性才能被识别为 MCP Tool 对象
            name = None
            description = "No name"

        with pytest.raises(ValueError, match="MCP tool must have a name"):
            ToolInfo.from_mcp_tool(MockMCPToolNoName())

    def test_from_mcp_tool_missing_name_dict(self):
        """测试缺少 name 的字典"""
        with pytest.raises(ValueError, match="MCP tool must have a name"):
            ToolInfo.from_mcp_tool({"description": "No name"})

    def test_from_mcp_tool_none_name_dict(self):
        """测试 name 为 None 的字典"""
        with pytest.raises(ValueError, match="MCP tool must have a name"):
            ToolInfo.from_mcp_tool({"name": None, "description": "Null name"})

    def test_from_mcp_tool_no_description(self):
        """测试没有 description 的情况"""

        class MockMCPToolNoDesc:
            name = "no_desc_tool"

        tool = MockMCPToolNoDesc()
        info = ToolInfo.from_mcp_tool(tool)
        assert info.name == "no_desc_tool"
        assert info.description is None
