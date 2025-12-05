"""ToolSet 模型定义 / ToolSet Model Definitions

定义工具集相关的数据模型和枚举。
Defines data models and enumerations related to toolsets.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from agentrun.utils.model import BaseModel, Field, PageableInput


class SchemaType(str, Enum):
    """Schema 类型 / Schema Type"""

    MCP = "MCP"
    """MCP 协议 / MCP Protocol"""
    OpenAPI = "OpenAPI"
    """OpenAPI 规范 / OpenAPI Specification"""


class ToolSetStatusOutputsUrls(BaseModel):
    internet_url: Optional[str] = None
    intranet_url: Optional[str] = None


class MCPServerConfig(BaseModel):
    headers: Optional[Dict[str, str]] = None
    transport_type: Optional[str] = None
    url: Optional[str] = None


class ToolMeta(BaseModel):
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    name: Optional[str] = None


class OpenAPIToolMeta(BaseModel):
    method: Optional[str] = None
    path: Optional[str] = None
    tool_id: Optional[str] = None
    tool_name: Optional[str] = None


class ToolSetStatusOutputs(BaseModel):
    function_arn: Optional[str] = None
    mcp_server_config: Optional[MCPServerConfig] = None
    open_api_tools: Optional[List[OpenAPIToolMeta]] = None
    tools: Optional[List[ToolMeta]] = None
    urls: Optional[ToolSetStatusOutputsUrls] = None


class APIKeyAuthParameter(BaseModel):
    encrypted: Optional[bool] = None
    in_: Optional[str] = None
    key: Optional[str] = None
    value: Optional[str] = None


class AuthorizationParameters(BaseModel):
    api_key_parameter: Optional[APIKeyAuthParameter] = None


class Authorization(BaseModel):
    parameters: Optional[AuthorizationParameters] = None
    type: Optional[str] = None


class ToolSetSchema(BaseModel):
    detail: Optional[str] = None
    type: Optional[SchemaType] = None


class ToolSetSpec(BaseModel):
    auth_config: Optional[Authorization] = None
    tool_schema: Optional[ToolSetSchema] = Field(alias="schema", default=None)


class ToolSetStatus(BaseModel):
    observed_generation: Optional[int] = None
    observed_time: Optional[str] = None
    outputs: Optional[ToolSetStatusOutputs] = None
    phase: Optional[str] = None


class ToolSetListInput(PageableInput):
    keyword: Optional[str] = None
    label_selector: Optional[List[str]] = None


class ToolSchema(BaseModel):
    """JSON Schema 兼容的工具参数描述

    支持完整的 JSON Schema 字段，能够描述复杂的嵌套数据结构。
    """

    # 基本字段
    type: Optional[str] = None
    description: Optional[str] = None
    title: Optional[str] = None

    # 对象类型字段
    properties: Optional[Dict[str, "ToolSchema"]] = None
    required: Optional[List[str]] = None
    additional_properties: Optional[bool] = None

    # 数组类型字段
    items: Optional["ToolSchema"] = None
    min_items: Optional[int] = None
    max_items: Optional[int] = None

    # 字符串类型字段
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    format: Optional[str] = None  # date, date-time, email, uri 等
    enum: Optional[List[Any]] = None

    # 数值类型字段
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    exclusive_minimum: Optional[float] = None
    exclusive_maximum: Optional[float] = None

    # 联合类型
    any_of: Optional[List["ToolSchema"]] = None
    one_of: Optional[List["ToolSchema"]] = None
    all_of: Optional[List["ToolSchema"]] = None

    # 默认值
    default: Optional[Any] = None

    @classmethod
    def from_any_openapi_schema(cls, schema: Any) -> "ToolSchema":
        """从任意 OpenAPI/JSON Schema 创建 ToolSchema

        递归解析所有嵌套结构，保留完整的 schema 信息。
        """
        if not schema or not isinstance(schema, dict):
            return cls(type="string")

        from pydash import get as pg

        # 解析 properties
        properties_raw = pg(schema, "properties", {})
        properties = (
            {
                key: cls.from_any_openapi_schema(value)
                for key, value in properties_raw.items()
            }
            if properties_raw
            else None
        )

        # 解析 items
        items_raw = pg(schema, "items")
        items = cls.from_any_openapi_schema(items_raw) if items_raw else None

        # 解析联合类型
        any_of_raw = pg(schema, "anyOf")
        any_of = (
            [cls.from_any_openapi_schema(s) for s in any_of_raw]
            if any_of_raw
            else None
        )

        one_of_raw = pg(schema, "oneOf")
        one_of = (
            [cls.from_any_openapi_schema(s) for s in one_of_raw]
            if one_of_raw
            else None
        )

        all_of_raw = pg(schema, "allOf")
        all_of = (
            [cls.from_any_openapi_schema(s) for s in all_of_raw]
            if all_of_raw
            else None
        )

        return cls(
            # 基本字段
            type=pg(schema, "type"),
            description=pg(schema, "description"),
            title=pg(schema, "title"),
            # 对象类型
            properties=properties,
            required=pg(schema, "required"),
            additional_properties=pg(schema, "additionalProperties"),
            # 数组类型
            items=items,
            min_items=pg(schema, "minItems"),
            max_items=pg(schema, "maxItems"),
            # 字符串类型
            pattern=pg(schema, "pattern"),
            min_length=pg(schema, "minLength"),
            max_length=pg(schema, "maxLength"),
            format=pg(schema, "format"),
            enum=pg(schema, "enum"),
            # 数值类型
            minimum=pg(schema, "minimum"),
            maximum=pg(schema, "maximum"),
            exclusive_minimum=pg(schema, "exclusiveMinimum"),
            exclusive_maximum=pg(schema, "exclusiveMaximum"),
            # 联合类型
            any_of=any_of,
            one_of=one_of,
            all_of=all_of,
            # 默认值
            default=pg(schema, "default"),
        )

    def to_json_schema(self) -> Dict[str, Any]:
        """转换为标准 JSON Schema 格式"""
        result: Dict[str, Any] = {}

        # 基本字段
        if self.type:
            result["type"] = self.type
        if self.description:
            result["description"] = self.description
        if self.title:
            result["title"] = self.title

        # 对象类型
        if self.properties:
            result["properties"] = {
                k: v.to_json_schema() for k, v in self.properties.items()
            }
        if self.required:
            result["required"] = self.required
        if self.additional_properties is not None:
            result["additionalProperties"] = self.additional_properties

        # 数组类型
        if self.items:
            result["items"] = self.items.to_json_schema()
        if self.min_items is not None:
            result["minItems"] = self.min_items
        if self.max_items is not None:
            result["maxItems"] = self.max_items

        # 字符串类型
        if self.pattern:
            result["pattern"] = self.pattern
        if self.min_length is not None:
            result["minLength"] = self.min_length
        if self.max_length is not None:
            result["maxLength"] = self.max_length
        if self.format:
            result["format"] = self.format
        if self.enum:
            result["enum"] = self.enum

        # 数值类型
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.exclusive_minimum is not None:
            result["exclusiveMinimum"] = self.exclusive_minimum
        if self.exclusive_maximum is not None:
            result["exclusiveMaximum"] = self.exclusive_maximum

        # 联合类型
        if self.any_of:
            result["anyOf"] = [s.to_json_schema() for s in self.any_of]
        if self.one_of:
            result["oneOf"] = [s.to_json_schema() for s in self.one_of]
        if self.all_of:
            result["allOf"] = [s.to_json_schema() for s in self.all_of]

        # 默认值
        if self.default is not None:
            result["default"] = self.default

        return result


class ToolInfo(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[ToolSchema] = None

    @classmethod
    def from_mcp_tool(cls, tool: Any):
        """从 MCP tool 创建 ToolInfo"""
        if hasattr(tool, "name"):
            # MCP Tool 对象
            tool_name = tool.name
            tool_description = getattr(tool, "description", None)
            input_schema = getattr(tool, "inputSchema", None) or getattr(
                tool, "input_schema", None
            )
        elif isinstance(tool, dict):
            # 字典格式
            tool_name = tool.get("name")
            tool_description = tool.get("description")
            input_schema = tool.get("inputSchema") or tool.get("input_schema")
        else:
            raise ValueError(f"Unsupported MCP tool format: {type(tool)}")

        if not tool_name:
            raise ValueError("MCP tool must have a name")

        # 构建 parameters schema
        parameters = None
        if input_schema:
            if isinstance(input_schema, dict):
                parameters = ToolSchema.from_any_openapi_schema(input_schema)
            elif hasattr(input_schema, "model_dump"):
                parameters = ToolSchema.from_any_openapi_schema(
                    input_schema.model_dump()
                )

        return cls(
            name=tool_name,
            description=tool_description,
            parameters=parameters or ToolSchema(type="object", properties={}),
        )
