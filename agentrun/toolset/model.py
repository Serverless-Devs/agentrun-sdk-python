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
    type: Optional[str] = None
    properties: Optional[Dict[str, "ToolSchema"]] = None
    required: Optional[List[str]] = None
    description: Optional[str] = None

    @classmethod
    def from_any_openapi_schema(cls, schema: Any):
        """从任意 OpenAPI schema 创建 ToolSchema"""
        from pydash import get as pg

        schema_type = pg(schema, "type", "string")
        properties = pg(schema, "properties", {})
        return cls(
            type=schema_type,
            properties=(
                {
                    key: cls.from_any_openapi_schema(value)
                    for key, value in properties.items()
                }
                if properties
                else None
            ),
            required=pg(schema, "required"),
            description=pg(schema, "description"),
        )


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
