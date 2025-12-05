"""Google ADK 工具适配器 / Google ADK Tool Adapter

将标准工具定义转换为 Google ADK 函数格式。"""

from typing import Any, Dict, List, Optional

from agentrun.integration.utils.adapter import ToolAdapter
from agentrun.integration.utils.canonical import CanonicalTool
from agentrun.integration.utils.tool import normalize_tool_name


def _json_schema_to_google_schema(
    schema: Dict[str, Any],
    root_schema: Optional[Dict[str, Any]] = None,
) -> Any:
    """将 JSON Schema 转换为 Google ADK Schema

    Google ADK 不支持 $ref 和 $defs，所以需要内联所有引用。

    Args:
        schema: JSON Schema 定义
        root_schema: 根 schema，用于解析 $ref 引用

    Returns:
        Google ADK types.Schema 对象
    """
    from google.genai import types

    if not schema or not isinstance(schema, dict):
        return types.Schema(type=types.Type.OBJECT)

    if root_schema is None:
        root_schema = schema

    # 处理 $ref 引用
    if "$ref" in schema:
        ref = schema["$ref"]
        resolved = _resolve_schema_ref(ref, root_schema)
        if resolved:
            return _json_schema_to_google_schema(resolved, root_schema)
        return types.Schema(type=types.Type.OBJECT)

    schema_type = schema.get("type")
    description = schema.get("description")

    # 类型映射
    type_mapping = {
        "string": types.Type.STRING,
        "integer": types.Type.INTEGER,
        "number": types.Type.NUMBER,
        "boolean": types.Type.BOOLEAN,
        "array": types.Type.ARRAY,
        "object": types.Type.OBJECT,
    }

    google_type = type_mapping.get(
        str(schema_type or "object"), types.Type.OBJECT
    )

    # 处理数组类型
    if schema_type == "array":
        items_schema = schema.get("items")
        items = None
        if items_schema:
            items = _json_schema_to_google_schema(items_schema, root_schema)
        return types.Schema(
            type=google_type,
            description=description,
            items=items,
        )

    # 处理对象类型
    if schema_type == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])

        google_props = {}
        for prop_name, prop_schema in props.items():
            google_props[prop_name] = _json_schema_to_google_schema(
                prop_schema, root_schema
            )

        return types.Schema(
            type=google_type,
            description=description,
            properties=google_props if google_props else None,
            required=required if required else None,
        )

    # 基本类型
    return types.Schema(
        type=google_type,
        description=description,
    )


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


def _create_custom_function_tool_class():
    """创建自定义 FunctionTool 类

    延迟创建以避免在模块导入时依赖 google.adk。
    """
    from google.adk.tools.base_tool import BaseTool

    class CustomFunctionTool(BaseTool):
        """自定义 Google ADK 工具类

        允许手动指定 FunctionDeclaration，避免 Pydantic 模型的 $ref 问题。
        继承自 google.adk.tools.BaseTool 以确保与 ADK Agent 兼容。
        """

        def __init__(
            self,
            func,
            declaration,
        ):
            # 调用父类 __init__，传递 name 和 description
            super().__init__(
                name=declaration.name,
                description=declaration.description or "",
            )
            self._func = func
            self._declaration = declaration

        def _get_declaration(self):
            return self._declaration

        async def run_async(self, *, args: Dict[str, Any], tool_context):
            """异步执行工具函数"""
            return self._func(**args)

    return CustomFunctionTool


# 缓存类，避免重复创建
_CustomFunctionToolClass = None


def _get_custom_function_tool_class():
    """获取 CustomFunctionTool 类（延迟加载）"""
    global _CustomFunctionToolClass
    if _CustomFunctionToolClass is None:
        _CustomFunctionToolClass = _create_custom_function_tool_class()
    return _CustomFunctionToolClass


class GoogleADKToolAdapter(ToolAdapter):
    """Google ADK 工具适配器 / Google ADK Tool Adapter

    实现 CanonicalTool → Google ADK 函数的转换。
    由于 Google ADK 不支持复杂的 JSON Schema（包含 $ref 和 $defs），
    此适配器会手动构建 FunctionDeclaration 并使用自定义工具类。
    """

    def get_registered_tool(self, name: str) -> Optional[CanonicalTool]:
        """根据名称获取最近注册的工具定义 / Google ADK Tool Adapter"""
        return self._registered_tools.get(normalize_tool_name(name))

    def from_canonical(self, tools: List[CanonicalTool]):
        """将标准格式转换为 Google ADK 工具 / Google ADK Tool Adapter

        为每个工具创建自定义的 FunctionTool，手动指定参数 schema，
        以避免 Pydantic 模型产生的 $ref 问题。
        """
        from google.genai import types

        result = []

        for tool in tools:
            # 记录工具定义
            self._registered_tools[tool.name] = tool

            # 从 parameters schema 构建 Google ADK Schema
            parameters_schema = tool.parameters or {
                "type": "object",
                "properties": {},
            }

            google_schema = _json_schema_to_google_schema(
                parameters_schema, parameters_schema
            )

            # 创建 FunctionDeclaration
            declaration = types.FunctionDeclaration(
                name=normalize_tool_name(tool.name),
                description=tool.description or "",
                parameters=google_schema,
            )

            # 创建包装函数
            def make_wrapper(canonical_tool: CanonicalTool):
                def wrapper(**kwargs):
                    if canonical_tool.func is None:
                        raise NotImplementedError(
                            f"Tool function for '{canonical_tool.name}' "
                            "is not implemented."
                        )
                    return canonical_tool.func(**kwargs)

                return wrapper

            wrapper_func = make_wrapper(tool)

            # 创建自定义工具
            CustomFunctionTool = _get_custom_function_tool_class()
            custom_tool = CustomFunctionTool(wrapper_func, declaration)
            result.append(custom_tool)

        return result
