"""Tool 模型单元测试 / Tool Model Unit Tests

测试 tool 模块中数据模型和工具 schema 的相关功能。
Tests data models and tool schema functionality in the tool module.
"""

import pytest

from agentrun.tool.model import (
    McpConfig,
    ToolCodeConfiguration,
    ToolContainerConfiguration,
    ToolInfo,
    ToolLogConfiguration,
    ToolNASConfig,
    ToolNetworkConfiguration,
    ToolOSSMountConfig,
    ToolSchema,
    ToolType,
)


class TestToolType:
    """测试 ToolType 枚举"""

    def test_mcp_type(self):
        """测试 MCP 类型"""
        assert ToolType.MCP == "MCP"
        assert ToolType.MCP.value == "MCP"

    def test_functioncall_type(self):
        """测试 FUNCTIONCALL 类型"""
        assert ToolType.FUNCTIONCALL == "FUNCTIONCALL"
        assert ToolType.FUNCTIONCALL.value == "FUNCTIONCALL"


class TestMcpConfig:
    """测试 McpConfig 模型"""

    def test_default_values(self):
        """测试默认值"""
        config = McpConfig()
        assert config.session_affinity is None
        assert config.session_affinity_config is None
        assert config.proxy_enabled is None
        assert config.bound_configuration is None
        assert config.mcp_proxy_configuration is None

    def test_with_values(self):
        """测试带值创建"""
        config = McpConfig(
            session_affinity="MCP_SSE",
            session_affinity_config='{"key": "value"}',
            proxy_enabled=True,
            bound_configuration={"key": "value"},
            mcp_proxy_configuration={"proxy": "config"},
        )
        assert config.session_affinity == "MCP_SSE"
        assert config.session_affinity_config == '{"key": "value"}'
        assert config.proxy_enabled is True
        assert config.bound_configuration == {"key": "value"}
        assert config.mcp_proxy_configuration == {"proxy": "config"}


class TestToolCodeConfiguration:
    """测试 ToolCodeConfiguration 模型"""

    def test_default_values(self):
        """测试默认值"""
        config = ToolCodeConfiguration()
        assert config.code_checksum is None
        assert config.code_size is None
        assert config.command is None
        assert config.language is None
        assert config.oss_bucket_name is None
        assert config.oss_object_name is None

    def test_with_values(self):
        """测试带值创建"""
        config = ToolCodeConfiguration(
            code_checksum="abc123",
            code_size=1024,
            command=["python", "app.py"],
            language="python3.10",
            oss_bucket_name="my-bucket",
            oss_object_name="code.zip",
        )
        assert config.code_checksum == "abc123"
        assert config.code_size == 1024
        assert config.command == ["python", "app.py"]
        assert config.language == "python3.10"
        assert config.oss_bucket_name == "my-bucket"
        assert config.oss_object_name == "code.zip"


class TestToolContainerConfiguration:
    """测试 ToolContainerConfiguration 模型"""

    def test_default_values(self):
        """测试默认值"""
        config = ToolContainerConfiguration()
        assert config.args is None
        assert config.command is None
        assert config.image is None
        assert config.port is None

    def test_with_values(self):
        """测试带值创建"""
        config = ToolContainerConfiguration(
            args=["--arg1", "value1"],
            command=["python", "app.py"],
            image="registry.example.com/tool:latest",
            port=8080,
        )
        assert config.args == ["--arg1", "value1"]
        assert config.command == ["python", "app.py"]
        assert config.image == "registry.example.com/tool:latest"
        assert config.port == 8080


class TestToolLogConfiguration:
    """测试 ToolLogConfiguration 模型"""

    def test_default_values(self):
        """测试默认值"""
        config = ToolLogConfiguration()
        assert config.log_store is None
        assert config.project is None

    def test_with_values(self):
        """测试带值创建"""
        config = ToolLogConfiguration(
            log_store="my-log-store",
            project="my-project",
        )
        assert config.log_store == "my-log-store"
        assert config.project == "my-project"


class TestToolNASConfig:
    """测试 ToolNASConfig 模型"""

    def test_default_values(self):
        """测试默认值"""
        config = ToolNASConfig()
        assert config.group_id is None
        assert config.mount_points is None
        assert config.user_id is None

    def test_with_values(self):
        """测试带值创建"""
        config = ToolNASConfig(
            group_id=1001,
            mount_points=[{"path": "/mnt/nas", "nas_id": "nas-123"}],
            user_id=1000,
        )
        assert config.group_id == 1001
        assert config.mount_points == [
            {"path": "/mnt/nas", "nas_id": "nas-123"}
        ]
        assert config.user_id == 1000


class TestToolNetworkConfiguration:
    """测试 ToolNetworkConfiguration 模型"""

    def test_default_values(self):
        """测试默认值"""
        config = ToolNetworkConfiguration()
        assert config.security_group_id is None
        assert config.vpc_id is None
        assert config.vswitch_ids is None

    def test_with_values(self):
        """测试带值创建"""
        config = ToolNetworkConfiguration(
            security_group_id="sg-123",
            vpc_id="vpc-456",
            vswitch_ids=["vsw-789", "vsw-012"],
        )
        assert config.security_group_id == "sg-123"
        assert config.vpc_id == "vpc-456"
        assert config.vswitch_ids == ["vsw-789", "vsw-012"]


class TestToolOSSMountConfig:
    """测试 ToolOSSMountConfig 模型"""

    def test_default_values(self):
        """测试默认值"""
        config = ToolOSSMountConfig()
        assert config.mount_points is None

    def test_with_values(self):
        """测试带值创建"""
        config = ToolOSSMountConfig(
            mount_points=[{
                "bucket": "my-bucket",
                "endpoint": "oss-cn-hangzhou.aliyuncs.com",
            }]
        )
        assert config.mount_points == [
            {"bucket": "my-bucket", "endpoint": "oss-cn-hangzhou.aliyuncs.com"}
        ]


class TestToolSchema:
    """测试 ToolSchema 模型"""

    def test_default_values(self):
        """测试默认值"""
        schema = ToolSchema()
        assert schema.type is None
        assert schema.description is None
        assert schema.properties is None
        assert schema.required is None
        assert schema.items is None
        assert schema.any_of is None
        assert schema.one_of is None
        assert schema.all_of is None

    def test_from_any_openapi_schema_simple(self):
        """测试从简单 OpenAPI Schema 创建"""
        openapi_schema = {
            "type": "object",
            "description": "A simple object",
            "properties": {
                "name": {"type": "string", "description": "Name field"},
                "age": {"type": "integer", "description": "Age field"},
            },
            "required": ["name"],
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert schema.type == "object"
        assert schema.description == "A simple object"
        assert schema.properties is not None
        assert "name" in schema.properties
        assert "age" in schema.properties
        assert schema.required == ["name"]

    def test_from_any_openapi_schema_nested(self):
        """测试从嵌套 OpenAPI Schema 创建"""
        openapi_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                }
            },
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert schema.type == "object"
        assert schema.properties is not None
        assert "user" in schema.properties
        assert schema.properties["user"].type == "object"
        assert schema.properties["user"].properties is not None
        assert "name" in schema.properties["user"].properties

    def test_from_any_openapi_schema_empty(self):
        """测试从空 Schema 创建"""
        schema = ToolSchema.from_any_openapi_schema(None)
        assert schema.type == "string"

        # Empty dict creates a ToolSchema with None type (pydash_get returns None for missing keys)
        schema = ToolSchema.from_any_openapi_schema({})
        # Actually returns "string" due to the check at the beginning of the method
        assert schema.type == "string"

    def test_from_any_openapi_schema_non_dict(self):
        """测试从非 dict 输入创建"""
        schema = ToolSchema.from_any_openapi_schema("invalid")
        assert schema.type == "string"

        schema = ToolSchema.from_any_openapi_schema(123)
        assert schema.type == "string"

    def test_from_any_openapi_schema_array(self):
        """测试从数组 Schema 创建"""
        openapi_schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert schema.type == "array"
        assert schema.items is not None
        assert schema.items.type == "string"
        assert schema.min_items == 1
        assert schema.max_items == 10

    def test_from_any_openapi_schema_anyof(self):
        """测试 anyOf 支持"""
        openapi_schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
            ]
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert schema.any_of is not None
        assert len(schema.any_of) == 2
        assert schema.any_of[0].type == "string"
        assert schema.any_of[1].type == "integer"

    def test_from_any_openapi_schema_oneof(self):
        """测试 oneOf 支持"""
        openapi_schema = {
            "oneOf": [
                {"type": "string"},
                {"type": "boolean"},
            ]
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert schema.one_of is not None
        assert len(schema.one_of) == 2
        assert schema.one_of[0].type == "string"
        assert schema.one_of[1].type == "boolean"

    def test_from_any_openapi_schema_allof(self):
        """测试 allOf 支持"""
        openapi_schema = {
            "allOf": [
                {"type": "object", "properties": {"name": {"type": "string"}}},
                {"type": "object", "properties": {"age": {"type": "integer"}}},
            ]
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        assert schema.all_of is not None
        assert len(schema.all_of) == 2
        assert schema.all_of[0].type == "object"
        assert schema.all_of[1].type == "object"

    def test_to_json_schema_simple(self):
        """测试转换为 JSON Schema - 简单情况"""
        schema = ToolSchema(
            type="string",
            description="A string field",
            min_length=1,
            max_length=100,
        )
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "string"
        assert json_schema["description"] == "A string field"
        assert json_schema["minLength"] == 1
        assert json_schema["maxLength"] == 100

    def test_to_json_schema_nested(self):
        """测试转换为 JSON Schema - 嵌套情况"""
        schema = ToolSchema(
            type="object",
            properties={
                "user": ToolSchema(
                    type="object",
                    properties={
                        "name": ToolSchema(type="string"),
                    },
                )
            },
        )
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "object"
        assert "properties" in json_schema
        assert "user" in json_schema["properties"]
        assert json_schema["properties"]["user"]["type"] == "object"
        assert "properties" in json_schema["properties"]["user"]

    def test_to_json_schema_roundtrip(self):
        """测试完整往返转换"""
        openapi_schema = {
            "type": "object",
            "description": "Test schema",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name",
                    "minLength": 1,
                },
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 150,
                },
            },
            "required": ["name"],
        }
        schema = ToolSchema.from_any_openapi_schema(openapi_schema)
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "object"
        assert json_schema["description"] == "Test schema"
        assert "properties" in json_schema
        assert "name" in json_schema["properties"]
        assert "age" in json_schema["properties"]
        assert json_schema["required"] == ["name"]

    def test_to_json_schema_with_anyof(self):
        """测试转换包含 anyOf 的 schema"""
        schema = ToolSchema(
            any_of=[
                ToolSchema(type="string"),
                ToolSchema(type="integer"),
            ]
        )
        json_schema = schema.to_json_schema()
        assert "anyOf" in json_schema
        assert len(json_schema["anyOf"]) == 2
        assert json_schema["anyOf"][0]["type"] == "string"
        assert json_schema["anyOf"][1]["type"] == "integer"

    def test_recursive_properties(self):
        """测试递归嵌套 properties"""
        schema = ToolSchema(
            type="object",
            properties={
                "level1": ToolSchema(
                    type="object",
                    properties={
                        "level2": ToolSchema(
                            type="object",
                            properties={
                                "level3": ToolSchema(type="string"),
                            },
                        ),
                    },
                ),
            },
        )
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "object"
        assert json_schema["properties"]["level1"]["type"] == "object"
        assert (
            json_schema["properties"]["level1"]["properties"]["level2"]["type"]
            == "object"
        )
        assert (
            json_schema["properties"]["level1"]["properties"]["level2"][
                "properties"
            ]["level3"]["type"]
            == "string"
        )

    def test_to_json_schema_with_string_constraints(self):
        """测试 pattern, min_length, max_length, format"""
        schema = ToolSchema(
            type="string",
            pattern="^[a-zA-Z]+$",
            min_length=1,
            max_length=100,
            format="email",
        )
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "string"
        assert json_schema["pattern"] == "^[a-zA-Z]+$"
        assert json_schema["minLength"] == 1
        assert json_schema["maxLength"] == 100
        assert json_schema["format"] == "email"

    def test_to_json_schema_with_number_constraints(self):
        """测试 minimum, maximum, exclusive_minimum, exclusive_maximum"""
        schema = ToolSchema(
            type="number",
            minimum=0,
            maximum=100,
            exclusive_minimum=0,
            exclusive_maximum=100,
        )
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "number"
        assert json_schema["minimum"] == 0
        assert json_schema["maximum"] == 100
        assert json_schema["exclusiveMinimum"] == 0
        assert json_schema["exclusiveMaximum"] == 100

    def test_to_json_schema_with_enum(self):
        """测试 enum 字段"""
        schema = ToolSchema(
            type="string",
            enum=["red", "green", "blue"],
        )
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "string"
        assert json_schema["enum"] == ["red", "green", "blue"]

    def test_to_json_schema_with_additional_properties(self):
        """测试 additionalProperties"""
        schema = ToolSchema(
            type="object",
            additional_properties=True,
        )
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "object"
        assert json_schema["additionalProperties"] is True

    def test_to_json_schema_with_default(self):
        """测试 default 字段"""
        schema = ToolSchema(
            type="string",
            default="default_value",
        )
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "string"
        assert json_schema["default"] == "default_value"

    def test_to_json_schema_with_title(self):
        """测试 title 字段"""
        schema = ToolSchema(
            type="string",
            title="String Field",
        )
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "string"
        assert json_schema["title"] == "String Field"

    def test_to_json_schema_with_one_of(self):
        """测试 oneOf 序列化"""
        schema = ToolSchema(
            one_of=[
                ToolSchema(type="string"),
                ToolSchema(type="integer"),
            ],
        )
        json_schema = schema.to_json_schema()
        assert "oneOf" in json_schema
        assert len(json_schema["oneOf"]) == 2
        assert json_schema["oneOf"][0]["type"] == "string"
        assert json_schema["oneOf"][1]["type"] == "integer"

    def test_to_json_schema_with_all_of(self):
        """测试 allOf 序列化"""
        schema = ToolSchema(
            all_of=[
                ToolSchema(
                    type="object",
                    properties={"name": ToolSchema(type="string")},
                ),
                ToolSchema(
                    type="object",
                    properties={"age": ToolSchema(type="integer")},
                ),
            ],
        )
        json_schema = schema.to_json_schema()
        assert "allOf" in json_schema
        assert len(json_schema["allOf"]) == 2
        assert json_schema["allOf"][0]["type"] == "object"
        assert json_schema["allOf"][1]["type"] == "object"


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
            name="test_tool",
            description="A test tool",
            parameters=ToolSchema(
                type="object",
                properties={
                    "input": ToolSchema(type="string"),
                },
            ),
        )
        assert info.name == "test_tool"
        assert info.description == "A test tool"
        assert info.parameters is not None
        assert info.parameters.type == "object"

    def test_from_mcp_tool_with_object(self):
        """测试从 MCP 工具对象创建"""
        mcp_tool = {
            "name": "mcp_tool",
            "description": "An MCP tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "param": {"type": "string"},
                },
            },
        }
        info = ToolInfo.from_mcp_tool(mcp_tool)
        assert info.name == "mcp_tool"
        assert info.description == "An MCP tool"
        assert info.parameters is not None
        assert info.parameters.type == "object"

    def test_from_mcp_tool_with_dict(self):
        """测试从 MCP 工具字典创建"""
        mcp_tool = {
            "name": "dict_tool",
            "description": "Dict tool",
            "inputSchema": {
                "type": "string",
            },
        }
        info = ToolInfo.from_mcp_tool(mcp_tool)
        assert info.name == "dict_tool"
        assert info.description == "Dict tool"
        assert info.parameters is not None
        assert info.parameters.type == "string"

    def test_from_mcp_tool_without_name(self):
        """测试从没有 name 的 MCP 工具创建"""
        mcp_tool = {
            "description": "Tool without name",
            "inputSchema": {"type": "string"},
        }
        with pytest.raises(ValueError, match="name"):
            ToolInfo.from_mcp_tool(mcp_tool)

    def test_from_mcp_tool_with_empty_schema(self):
        """测试从空 schema 的 MCP 工具创建"""
        mcp_tool = {
            "name": "empty_schema_tool",
            "description": "Tool with empty schema",
        }
        info = ToolInfo.from_mcp_tool(mcp_tool)
        assert info.name == "empty_schema_tool"
        assert info.description == "Tool with empty schema"
        assert info.parameters is not None
        assert info.parameters.type == "object"

    def test_from_mcp_tool_with_model_dump(self):
        """测试 from_mcp_tool 当 input_schema 有 model_dump 方法时"""

        class MockInputSchema:

            def model_dump(self):
                return {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "integer"},
                    },
                    "required": ["param1"],
                }

        mcp_tool = {
            "name": "tool_with_model_dump",
            "description": "Tool with model_dump input schema",
            "inputSchema": MockInputSchema(),
        }
        info = ToolInfo.from_mcp_tool(mcp_tool)
        assert info.name == "tool_with_model_dump"
        assert info.description == "Tool with model_dump input schema"
        assert info.parameters is not None
        assert info.parameters.type == "object"
        assert "param1" in info.parameters.properties
        assert "param2" in info.parameters.properties
        assert info.parameters.required == ["param1"]
