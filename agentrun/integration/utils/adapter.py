"""适配器接口定义 / Adapter Interface Definition

定义统一的适配器接口,所有框架适配器都实现这些接口。
Defines unified adapter interfaces that all framework adapters implement.

这样可以确保一致的转换行为,并最大化代码复用。
This ensures consistent conversion behavior and maximizes code reuse.
"""

from abc import ABC, abstractmethod
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import create_model, Field

from agentrun.integration.utils.canonical import CanonicalMessage, CanonicalTool
from agentrun.integration.utils.model import CommonModel
from agentrun.integration.utils.tool import normalize_tool_name

# 用于缓存动态创建的 Pydantic 模型，避免重复创建
_dynamic_models_cache: Dict[str, type] = {}


def _resolve_schema_ref(
    ref: str, root_schema: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """解析 JSON Schema $ref 引用

    Args:
        ref: $ref 字符串，如 "#/$defs/MyType" 或 "#/definitions/MyType"
        root_schema: 根 schema，包含 $defs 或 definitions

    Returns:
        解析后的 schema，如果无法解析则返回 None
    """
    if not ref or not ref.startswith("#/"):
        return None

    # 解析路径，如 "#/$defs/MyType" -> ["$defs", "MyType"]
    path_parts = ref[2:].split("/")
    current = root_schema

    for part in path_parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]

    return current if isinstance(current, dict) else None


def _generate_model_name(
    schema: Dict[str, Any], prefix: str = "Dynamic"
) -> str:
    """为 schema 生成唯一的模型名称"""
    # 优先使用 schema 中的 title
    title = schema.get("title")
    if title:
        # 清理名称，只保留字母数字
        clean_name = re.sub(r"[^a-zA-Z0-9]", "", title)
        if clean_name:
            return clean_name

    # 使用属性名生成名称
    properties = schema.get("properties", {})
    if properties:
        prop_names = sorted(properties.keys())[:3]  # 最多取3个属性名
        name = "".join(n.title() for n in prop_names)
        return f"{prefix}{name}Model"

    return f"{prefix}Model"


def _schema_to_type_description(
    schema: Dict[str, Any],
    root_schema: Optional[Dict[str, Any]] = None,
    indent: int = 0,
) -> str:
    """将 JSON Schema 转换为人类可读的类型描述

    用于在函数文档字符串中描述复杂类型结构，
    帮助不支持精确类型的框架（如 Google ADK）理解参数类型。

    Args:
        schema: JSON Schema 定义
        root_schema: 根 schema，用于解析 $ref 引用
        indent: 缩进级别

    Returns:
        类型描述字符串
    """
    if not schema or not isinstance(schema, dict):
        return "any"

    if root_schema is None:
        root_schema = schema

    indent_prefix = "  " * indent

    # 处理 $ref 引用
    if "$ref" in schema:
        resolved = _resolve_schema_ref(schema["$ref"], root_schema)
        if resolved:
            return _schema_to_type_description(resolved, root_schema, indent)
        return "object"

    schema_type = schema.get("type")

    # 处理 anyOf / oneOf
    if "anyOf" in schema:
        types = [
            _schema_to_type_description(s, root_schema, indent)
            for s in schema["anyOf"]
        ]
        return " | ".join(types)

    if "oneOf" in schema:
        types = [
            _schema_to_type_description(s, root_schema, indent)
            for s in schema["oneOf"]
        ]
        return " | ".join(types)

    # null 类型
    if schema_type == "null":
        return "null"

    # 数组类型
    if schema_type == "array":
        items_schema = schema.get("items")
        if items_schema:
            item_desc = _schema_to_type_description(
                items_schema, root_schema, indent
            )
            return f"array[{item_desc}]"
        return "array"

    # 对象类型 - 详细描述属性
    if schema_type == "object":
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        if not properties:
            return "object"

        # 获取标题
        title = schema.get("title", "object")
        lines = [f"{title} {{"]

        for prop_name, prop_schema in properties.items():
            prop_desc = _schema_to_type_description(
                prop_schema, root_schema, indent + 1
            )
            description = prop_schema.get("description", "")
            optional_marker = "" if prop_name in required else "?"

            prop_line = (
                f"{indent_prefix}  {prop_name}{optional_marker}: {prop_desc}"
            )
            if description:
                prop_line += f"  # {description}"
            lines.append(prop_line)

        lines.append(f"{indent_prefix}}}")
        return "\n".join(lines)

    # 基本类型
    type_names = {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
    }

    if schema_type in type_names:
        # 添加枚举值信息
        enum_values = schema.get("enum")
        if enum_values:
            enum_str = ", ".join(repr(v) for v in enum_values)
            return f"{type_names[schema_type]}({enum_str})"
        return type_names[schema_type]

    return "any"


def _generate_params_docstring(
    parameters_schema: Dict[str, Any],
) -> str:
    """生成参数的详细文档字符串

    为复杂类型生成详细的类型描述，用于不支持精确类型的框架。

    Args:
        parameters_schema: 参数的 JSON Schema

    Returns:
        参数文档字符串
    """
    properties = parameters_schema.get("properties", {})
    required = set(parameters_schema.get("required", []))

    if not properties:
        return ""

    lines = ["\n\nArgs:"]

    for param_name, param_schema in properties.items():
        # 基础描述
        description = param_schema.get("description", "")
        optional_marker = " (optional)" if param_name not in required else ""

        # 获取类型描述
        type_desc = _schema_to_type_description(param_schema, parameters_schema)

        # 检查是否是复杂类型（包含换行的描述）
        if "\n" in type_desc:
            # 复杂类型，使用多行格式
            lines.append(f"    {param_name}{optional_marker}:")
            if description:
                lines.append(f"        {description}")
            lines.append("        Type:")
            # 缩进复杂类型描述
            for type_line in type_desc.split("\n"):
                lines.append(f"            {type_line}")
        else:
            # 简单类型
            param_line = f"    {param_name} ({type_desc}){optional_marker}"
            if description:
                param_line += f": {description}"
            lines.append(param_line)

    return "\n".join(lines)


def _schema_to_pydantic_model(
    schema: Dict[str, Any],
    root_schema: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> type:
    """将 JSON Schema 转换为 Pydantic 模型

    Args:
        schema: 对象类型的 JSON Schema
        root_schema: 根 schema，用于解析 $ref
        model_name: 可选的模型名称

    Returns:
        动态创建的 Pydantic 模型类
    """
    if root_schema is None:
        root_schema = schema

    # 生成模型名称
    if not model_name:
        model_name = _generate_model_name(schema)

    # 检查缓存
    cache_key = f"{model_name}_{id(schema)}"
    if cache_key in _dynamic_models_cache:
        return _dynamic_models_cache[cache_key]

    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    if not properties:
        # 无属性的对象，返回基础 dict
        return dict

    fields: Dict[str, Any] = {}

    for field_name, field_schema in properties.items():
        # 递归获取字段类型
        field_type = schema_to_python_type(field_schema, root_schema)

        # 获取字段描述
        description = field_schema.get("description", "")
        default = field_schema.get("default")

        # 构建字段定义
        if field_name in required_fields:
            fields[field_name] = (field_type, Field(description=description))
        else:
            # 可选字段
            fields[field_name] = (
                Optional[field_type],
                Field(default=default, description=description),
            )

    # 动态创建 Pydantic 模型
    try:
        model = create_model(model_name, **fields)
        _dynamic_models_cache[cache_key] = model
        return model
    except Exception:
        # 如果创建失败，回退到 dict
        return dict


def schema_to_python_type(
    schema: Dict[str, Any],
    root_schema: Optional[Dict[str, Any]] = None,
) -> Any:
    """将 JSON Schema 转换为 Python 类型注解

    支持复杂类型：嵌套对象、数组元素类型、联合类型、$ref 引用等。
    对于嵌套对象，会动态创建 Pydantic 模型以保留完整的类型信息。

    Args:
        schema: JSON Schema 定义
        root_schema: 根 schema，用于解析 $ref 引用。
                     如果未提供，则使用 schema 本身作为根。

    Returns:
        Python 类型注解（可能是基本类型、List、Optional、Union 或 Pydantic 模型）
    """
    if not schema or not isinstance(schema, dict):
        return Any

    # 如果没有提供 root_schema，使用当前 schema 作为根
    if root_schema is None:
        root_schema = schema

    # 处理 $ref 引用
    if "$ref" in schema:
        resolved = _resolve_schema_ref(schema["$ref"], root_schema)
        if resolved:
            return schema_to_python_type(resolved, root_schema)
        # 无法解析 $ref，返回 dict（对象类型的默认）
        return dict

    schema_type = schema.get("type")

    # 处理 anyOf / oneOf（联合类型）
    if "anyOf" in schema:
        types = [schema_to_python_type(s, root_schema) for s in schema["anyOf"]]
        # 过滤掉 None 类型，用于构造 Optional
        non_none_types = [t for t in types if t is not type(None)]
        has_none = any(t is type(None) for t in types)

        if len(non_none_types) == 1 and has_none:
            return Optional[non_none_types[0]]
        elif len(non_none_types) > 1:
            return Union[tuple(non_none_types)]
        elif non_none_types:
            return non_none_types[0]
        return Any

    if "oneOf" in schema:
        types = [schema_to_python_type(s, root_schema) for s in schema["oneOf"]]
        non_none_types = [t for t in types if t is not type(None)]
        if len(non_none_types) > 1:
            return Union[tuple(non_none_types)]
        elif non_none_types:
            return non_none_types[0]
        return Any

    # 处理 null 类型
    if schema_type == "null":
        return type(None)

    # 处理数组类型
    if schema_type == "array":
        items_schema = schema.get("items")
        if items_schema:
            item_type = schema_to_python_type(items_schema, root_schema)
            return List[item_type]
        return List[Any]

    # 处理对象类型 - 动态创建 Pydantic 模型
    if schema_type == "object":
        properties = schema.get("properties")
        if properties:
            # 有属性定义，创建 Pydantic 模型
            return _schema_to_pydantic_model(schema, root_schema)
        # 无属性的通用对象
        return dict

    # 基本类型映射
    type_mapping: Dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    if schema_type and schema_type in type_mapping:
        return type_mapping[schema_type]

    return Any


class MessageAdapter(ABC):
    """消息格式适配器接口

    用于在 ModelAdapter 内部进行消息格式转换。
    只需要将框架消息转换为标准 OpenAI 格式。

    转换流程：
    - 框架消息 → to_canonical() → CanonicalMessage（OpenAI 格式）
    """

    @abstractmethod
    def to_canonical(self, messages: Any) -> List[CanonicalMessage]:
        """将框架消息转换为标准格式（供 ModelAdapter 内部使用）

        Args:
            messages: 框架特定的消息格式

        Returns:
            标准格式消息列表
        """
        pass


class ToolAdapter(ABC):
    """工具格式适配器接口 / Utils Adapters

    用于将标准工具定义转换为框架特定格式。
        单向转换：CanonicalTool → 框架工具"""

    def __init__(self) -> None:
        super().__init__()

        # 记录工具定义，便于模型适配器回溯参数 schema
        self._registered_tools: Dict[str, CanonicalTool] = {}

    @abstractmethod
    def from_canonical(self, tools: List[CanonicalTool]) -> Any:
        """将标准工具转换为框架特定格式 / 将标准工具Converts为框架特定格式

        Args:
                    tools: 标准格式工具列表

                Returns:
                    框架特定的工具格式"""
        pass

    def function_tools(
        self,
        tools: List[CanonicalTool],
        modify_func: Optional[Callable[..., Any]] = None,
        include_type_docstring: bool = False,
    ):
        """将标准格式转换为可调用函数工具

        通过函数的类型注解推断参数，动态创建带注解的函数。

        Args:
            tools: 标准格式工具列表
            modify_func: 可选的函数调用修改器
            include_type_docstring: 是否在文档字符串中包含详细的类型描述
                对于不支持精确类型的框架（如 Google ADK），可以设为 True
                以便在函数文档中包含复杂类型的结构描述"""
        result = []

        for tool in tools:
            # 记录工具定义
            self._registered_tools[tool.name] = tool

            # 从 parameters schema 构建函数签名
            parameters_schema = tool.parameters or {
                "type": "object",
                "properties": {},
            }
            properties = parameters_schema.get("properties", {})
            required = set(parameters_schema.get("required", []))

            # 构建函数参数
            params = []
            annotations = {}

            for param_name, param_schema in properties.items():
                # 使用递归类型转换处理复杂类型
                # 传入 parameters_schema 作为根 schema 以解析 $ref 引用
                param_type = schema_to_python_type(
                    param_schema, parameters_schema
                )

                # 设置默认值
                default = (
                    inspect.Parameter.empty if param_name in required else None
                )

                params.append(
                    inspect.Parameter(
                        param_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=default,
                        annotation=param_type,
                    )
                )
                annotations[param_name] = param_type

            # 创建带正确签名的函数
            def make_tool_function(
                canonical_tool: CanonicalTool,
                sig: inspect.Signature,
                annots: dict,
                params_schema: dict,
                add_type_doc: bool,
            ):
                def tool_func(**kwargs):
                    if canonical_tool.func is None:
                        raise NotImplementedError(
                            f"Tool function for '{canonical_tool.name}' "
                            "is not implemented."
                        )

                    if modify_func:
                        return modify_func(canonical_tool, **kwargs)

                    return canonical_tool.func(**kwargs)

                # 设置函数元数据（函数名也需要标准化以防第三方工具名限制）
                tool_func.__name__ = normalize_tool_name(canonical_tool.name)

                # 生成文档字符串
                base_doc = canonical_tool.description or ""
                if add_type_doc:
                    # 对于不支持精确类型的框架，在文档中包含详细的类型描述
                    params_doc = _generate_params_docstring(params_schema)
                    tool_func.__doc__ = base_doc + params_doc
                else:
                    tool_func.__doc__ = base_doc

                tool_func.__annotations__ = annots
                object.__setattr__(tool_func, "__signature__", sig)

                return tool_func

            # 创建签名
            signature = inspect.Signature(params)
            wrapped_func = make_tool_function(
                tool,
                signature,
                annotations,
                parameters_schema,
                include_type_docstring,
            )

            result.append(wrapped_func)

        return result


class ModelAdapter(ABC):
    """模型适配器接口 / Utils Model Adapter

    用于包装框架模型，使其能够与 CommonModel 协同工作。"""

    @abstractmethod
    def wrap_model(self, common_model: CommonModel) -> Any:
        """包装 CommonModel 为框架特定的模型格式 / 包装 CommonModel 为framework特定的模型格式

        Args:
                    common_model: CommonModel 实例

                Returns:
                    框架特定的模型对象"""
        pass
