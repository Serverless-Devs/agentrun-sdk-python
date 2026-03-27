"""Tool 模型定义 / Tool Model Definitions

定义工具相关的数据模型和枚举。
Defines data models and enumerations related to tools.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from agentrun.utils.model import BaseModel


class ToolType(str, Enum):
    """工具类型 / Tool Type"""

    MCP = "MCP"
    """MCP 协议工具 / MCP Protocol Tool"""
    FUNCTIONCALL = "FUNCTIONCALL"
    """函数调用工具 / Function Call Tool"""


class McpConfig(BaseModel):
    """MCP 工具配置 / MCP Tool Configuration

    包含 MCP 工具的会话亲和性、代理配置等信息。
    Contains session affinity, proxy configuration, etc. for MCP tools.
    """

    session_affinity: Optional[str] = None
    """会话亲和性策略 / Session affinity strategy
    NONE: 无亲和性 / No affinity
    MCP_SSE: 基于 SSE 的会话亲和性 / SSE-based session affinity
    MCP_STREAMABLE: 基于流式 HTTP 的会话亲和性 / Streamable HTTP-based session affinity
    """

    session_affinity_config: Optional[str] = None
    """会话亲和性的详细配置，JSON 格式字符串 / Session affinity config, JSON string"""

    proxy_enabled: Optional[bool] = None
    """是否启用 MCP 代理 / Whether MCP proxy is enabled"""

    bound_configuration: Optional[Dict[str, Any]] = None
    """工具的绑定配置 / Tool binding configuration"""

    mcp_proxy_configuration: Optional[Dict[str, Any]] = None
    """MCP 代理的详细配置 / MCP proxy detailed configuration"""


class ToolCodeConfiguration(BaseModel):
    """代码包配置 / Code Configuration

    代码包类型工具的配置信息。
    Configuration for code-package type tools.
    """

    code_checksum: Optional[str] = None
    """代码校验和 / Code checksum"""

    code_size: Optional[int] = None
    """代码大小（字节）/ Code size in bytes"""

    command: Optional[List[str]] = None
    """启动命令 / Startup command"""

    language: Optional[str] = None
    """编程语言 / Programming language"""

    oss_bucket_name: Optional[str] = None
    """OSS 存储桶名称 / OSS bucket name"""

    oss_object_name: Optional[str] = None
    """OSS 对象名称 / OSS object name"""


class ToolContainerConfiguration(BaseModel):
    """容器配置 / Container Configuration

    容器类型工具的配置信息。
    Configuration for container type tools.
    """

    args: Optional[List[str]] = None
    """容器启动参数 / Container startup arguments"""

    command: Optional[List[str]] = None
    """容器启动命令 / Container startup command"""

    image: Optional[str] = None
    """容器镜像地址 / Container image URL"""

    port: Optional[int] = None
    """容器端口 / Container port"""


class ToolLogConfiguration(BaseModel):
    """日志配置 / Log Configuration

    工具的日志配置信息。
    Log configuration for tools.
    """

    log_store: Optional[str] = None
    """SLS 日志库 / SLS log store"""

    project: Optional[str] = None
    """SLS 项目 / SLS project"""


class ToolNASConfig(BaseModel):
    """NAS 文件存储配置 / NAS Configuration

    工具访问 NAS 文件系统的配置。
    Configuration for tool access to NAS file system.
    """

    group_id: Optional[int] = None
    """组 ID / Group ID"""

    mount_points: Optional[List[Dict[str, Any]]] = None
    """挂载点列表 / Mount points list"""

    user_id: Optional[int] = None
    """用户 ID / User ID"""


class ToolNetworkConfiguration(BaseModel):
    """网络配置 / Network Configuration

    工具的网络配置信息。
    Network configuration for tools.
    """

    security_group_id: Optional[str] = None
    """安全组 ID / Security group ID"""

    vpc_id: Optional[str] = None
    """VPC ID"""

    vswitch_ids: Optional[List[str]] = None
    """交换机 ID 列表 / VSwitch IDs"""


class ToolOSSMountConfig(BaseModel):
    """OSS 挂载配置 / OSS Mount Configuration

    工具访问 OSS 存储的挂载配置。
    Configuration for tool access to OSS storage.
    """

    mount_points: Optional[List[Dict[str, Any]]] = None
    """挂载点列表 / Mount points list"""


class ToolSchema(BaseModel):
    """JSON Schema 兼容的工具参数描述 / JSON Schema Compatible Tool Parameter Description

    支持完整的 JSON Schema 字段，能够描述复杂的嵌套数据结构。
    Supports full JSON Schema fields for describing complex nested data structures.
    """

    type: Optional[str] = None
    """数据类型 / Data type"""

    description: Optional[str] = None
    """描述信息 / Description"""

    title: Optional[str] = None
    """标题 / Title"""

    properties: Optional[Dict[str, "ToolSchema"]] = None
    """对象属性 / Object properties"""

    required: Optional[List[str]] = None
    """必填字段 / Required fields"""

    additional_properties: Optional[bool] = None
    """是否允许额外属性 / Whether additional properties are allowed"""

    items: Optional["ToolSchema"] = None
    """数组元素类型 / Array item type"""

    min_items: Optional[int] = None
    """数组最小长度 / Minimum array length"""

    max_items: Optional[int] = None
    """数组最大长度 / Maximum array length"""

    pattern: Optional[str] = None
    """字符串正则模式 / String regex pattern"""

    min_length: Optional[int] = None
    """字符串最小长度 / Minimum string length"""

    max_length: Optional[int] = None
    """字符串最大长度 / Maximum string length"""

    format: Optional[str] = None
    """字符串格式 / String format (date, date-time, email, uri, etc.)"""

    enum: Optional[List[Any]] = None
    """枚举值 / Enum values"""

    minimum: Optional[float] = None
    """数值最小值 / Minimum numeric value"""

    maximum: Optional[float] = None
    """数值最大值 / Maximum numeric value"""

    exclusive_minimum: Optional[float] = None
    """数值排他最小值 / Exclusive minimum numeric value"""

    exclusive_maximum: Optional[float] = None
    """数值排他最大值 / Exclusive maximum numeric value"""

    any_of: Optional[List["ToolSchema"]] = None
    """任一匹配 / Any of"""

    one_of: Optional[List["ToolSchema"]] = None
    """唯一匹配 / One of"""

    all_of: Optional[List["ToolSchema"]] = None
    """全部匹配 / All of"""

    default: Optional[Any] = None
    """默认值 / Default value"""

    @classmethod
    def from_any_openapi_schema(cls, schema: Any) -> "ToolSchema":
        """从任意 OpenAPI/JSON Schema 创建 ToolSchema / Create ToolSchema from any OpenAPI/JSON Schema

        递归解析所有嵌套结构，保留完整的 schema 信息。
        Recursively parses all nested structures, preserving complete schema information.
        """
        if not schema or not isinstance(schema, dict):
            return cls(type="string")

        from pydash import get as pydash_get

        properties_raw = pydash_get(schema, "properties", {})
        properties = (
            {
                key: cls.from_any_openapi_schema(value)
                for key, value in properties_raw.items()
            }
            if properties_raw
            else None
        )

        items_raw = pydash_get(schema, "items")
        items = cls.from_any_openapi_schema(items_raw) if items_raw else None

        any_of_raw = pydash_get(schema, "anyOf")
        any_of = (
            [cls.from_any_openapi_schema(s) for s in any_of_raw]
            if any_of_raw
            else None
        )

        one_of_raw = pydash_get(schema, "oneOf")
        one_of = (
            [cls.from_any_openapi_schema(s) for s in one_of_raw]
            if one_of_raw
            else None
        )

        all_of_raw = pydash_get(schema, "allOf")
        all_of = (
            [cls.from_any_openapi_schema(s) for s in all_of_raw]
            if all_of_raw
            else None
        )

        return cls(
            type=pydash_get(schema, "type"),
            description=pydash_get(schema, "description"),
            title=pydash_get(schema, "title"),
            properties=properties,
            required=pydash_get(schema, "required"),
            additional_properties=pydash_get(schema, "additionalProperties"),
            items=items,
            min_items=pydash_get(schema, "minItems"),
            max_items=pydash_get(schema, "maxItems"),
            pattern=pydash_get(schema, "pattern"),
            min_length=pydash_get(schema, "minLength"),
            max_length=pydash_get(schema, "maxLength"),
            format=pydash_get(schema, "format"),
            enum=pydash_get(schema, "enum"),
            minimum=pydash_get(schema, "minimum"),
            maximum=pydash_get(schema, "maximum"),
            exclusive_minimum=pydash_get(schema, "exclusiveMinimum"),
            exclusive_maximum=pydash_get(schema, "exclusiveMaximum"),
            any_of=any_of,
            one_of=one_of,
            all_of=all_of,
            default=pydash_get(schema, "default"),
        )

    def to_json_schema(self) -> Dict[str, Any]:
        """转换为标准 JSON Schema 格式 / Convert to standard JSON Schema format"""
        result: Dict[str, Any] = {}

        if self.type:
            result["type"] = self.type
        if self.description:
            result["description"] = self.description
        if self.title:
            result["title"] = self.title

        if self.properties:
            result["properties"] = {
                k: v.to_json_schema() for k, v in self.properties.items()
            }
        if self.required:
            result["required"] = self.required
        if self.additional_properties is not None:
            result["additionalProperties"] = self.additional_properties

        if self.items:
            result["items"] = self.items.to_json_schema()
        if self.min_items is not None:
            result["minItems"] = self.min_items
        if self.max_items is not None:
            result["maxItems"] = self.max_items

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

        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.exclusive_minimum is not None:
            result["exclusiveMinimum"] = self.exclusive_minimum
        if self.exclusive_maximum is not None:
            result["exclusiveMaximum"] = self.exclusive_maximum

        if self.any_of:
            result["anyOf"] = [s.to_json_schema() for s in self.any_of]
        if self.one_of:
            result["oneOf"] = [s.to_json_schema() for s in self.one_of]
        if self.all_of:
            result["allOf"] = [s.to_json_schema() for s in self.all_of]

        if self.default is not None:
            result["default"] = self.default

        return result


class ToolInfo(BaseModel):
    """工具信息 / Tool Information

    描述单个工具的名称、描述和参数 schema。
    Describes a single tool's name, description, and parameter schema.
    """

    name: Optional[str] = None
    """工具名称 / Tool name"""

    description: Optional[str] = None
    """工具描述 / Tool description"""

    parameters: Optional[ToolSchema] = None
    """工具参数 schema / Tool parameter schema"""

    @classmethod
    def from_mcp_tool(cls, tool: Any) -> "ToolInfo":
        """从 MCP tool 创建 ToolInfo / Create ToolInfo from MCP tool"""
        if hasattr(tool, "name"):
            tool_name = tool.name
            tool_description = getattr(tool, "description", None)
            input_schema = getattr(tool, "inputSchema", None) or getattr(
                tool, "input_schema", None
            )
        elif isinstance(tool, dict):
            tool_name = tool.get("name")
            tool_description = tool.get("description")
            input_schema = tool.get("inputSchema") or tool.get("input_schema")
        else:
            raise ValueError(f"Unsupported MCP tool format: {type(tool)}")

        if not tool_name:
            raise ValueError("MCP tool must have a name")

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
