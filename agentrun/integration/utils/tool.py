"""通用工具定义和转换模块

提供跨框架的通用工具定义和转换功能。

支持多种定义方式：
1. 使用 @tool 装饰器 - 最简单，推荐
2. 使用 Pydantic BaseModel - 类型安全
3. 使用 ToolParameter 列表 - 灵活但繁琐

支持转换为以下框架格式：
- OpenAI Function Calling
- Anthropic Claude Tools
- LangChain Tools
- Google ADK Tools
- AgentScope Tools
"""

from __future__ import annotations

from copy import deepcopy
from functools import wraps
import hashlib
import inspect
import json
import re
from typing import (
    Any,
    Callable,
    Dict,
    get_args,
    get_origin,
    get_type_hints,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
)

from pydantic import (
    AliasChoices,
    BaseModel,
    create_model,
    Field,
    ValidationError,
)

if TYPE_CHECKING:
    from agentrun.toolset import ToolSet

from agentrun.utils.log import logger

# Tool name constraints for external providers like OpenAI
MAX_TOOL_NAME_LEN = 64
TOOL_NAME_HEAD_LEN = 32


def normalize_tool_name(name: str) -> str:
    """Normalize a tool name to fit provider limits.

    If `name` length is <= MAX_TOOL_NAME_LEN, return it unchanged.
    Otherwise, return the first TOOL_NAME_HEAD_LEN characters + md5(full_name)
    (32 hex chars), resulting in a 64-char string.
    """
    if not isinstance(name, str):
        name = str(name)
    if len(name) <= MAX_TOOL_NAME_LEN:
        return name
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return name[:TOOL_NAME_HEAD_LEN] + digest


class ToolParameter:
    """工具参数定义

    Attributes:
        name: 参数名称
        param_type: 参数类型（string, integer, number, boolean, array, object）
        description: 参数描述
        required: 是否必填
        default: 默认值
        enum: 枚举值列表
        items: 数组元素类型（当 param_type 为 array 时使用）
        properties: 对象属性（当 param_type 为 object 时使用）
        format: 额外的格式限定（如 int32、int64 等）
        nullable: 是否允许为空
    """

    def __init__(
        self,
        name: str,
        param_type: str,
        description: str = "",
        required: bool = False,
        default: Any = None,
        enum: Optional[List[Any]] = None,
        items: Optional[Dict[str, Any]] = None,
        properties: Optional[Dict[str, Any]] = None,
        format: Optional[str] = None,
        nullable: bool = False,
    ):
        self.name = name
        self.param_type = param_type
        self.description = description
        self.required = required
        self.default = default
        self.enum = enum
        self.items = items
        self.properties = properties
        self.format = format
        self.nullable = nullable

    def to_json_schema(self) -> Dict[str, Any]:
        """转换为 JSON Schema 格式"""
        schema: Dict[str, Any] = {
            "type": self.param_type,
        }

        if self.description:
            schema["description"] = self.description

        if self.default is not None:
            schema["default"] = self.default

        if self.enum is not None:
            schema["enum"] = self.enum

        if self.format:
            schema["format"] = self.format

        if self.nullable:
            schema["nullable"] = True

        if self.param_type == "array" and self.items:
            schema["items"] = deepcopy(self.items)

        if self.param_type == "object" and self.properties:
            schema["properties"] = deepcopy(self.properties)

        return schema


def _merge_schema_dicts(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """递归合并 JSON Schema 片段，override 拥有更高优先级"""

    if not base:
        return deepcopy(override)
    if not override:
        return deepcopy(base)

    merged: Dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_schema_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_ref_schema(
    ref: Optional[str], root_schema: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """解析 $ref 指针"""

    if not ref or not isinstance(ref, str) or not ref.startswith("#/"):
        return None

    target: Any = root_schema
    # 处理 JSON Pointer 转义
    parts = ref[2:].split("/")
    for raw_part in parts:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, dict) or part not in target:
            return None
        target = target[part]

    if isinstance(target, dict):
        return deepcopy(target)
    return None


def _extract_core_schema(
    field_schema: Dict[str, Any],
    root_schema: Dict[str, Any],
    seen_refs: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Any], bool]:
    """展开 anyOf/oneOf/allOf/$ref，返回核心 schema 及 nullable 标记"""

    schema = deepcopy(field_schema)
    nullable = False

    if seen_refs is None:
        seen_refs = set()

    def _pick_from_union(options: List[Dict[str, Any]]) -> Dict[str, Any]:
        nonlocal nullable
        chosen: Optional[Dict[str, Any]] = None
        for option in options:
            if not isinstance(option, dict):
                continue
            if option.get("type") == "null":
                nullable = True
                continue
            resolved, is_nullable = _extract_core_schema(
                option, root_schema, seen_refs
            )
            nullable = nullable or is_nullable
            if chosen is None:
                chosen = resolved
        return chosen or {}

    union_key = None
    if "anyOf" in schema:
        union_key = "anyOf"
    elif "oneOf" in schema:
        union_key = "oneOf"

    if union_key:
        options = schema.pop(union_key, [])
        schema = _pick_from_union(options)

    if "allOf" in schema:
        merged: Dict[str, Any] = {}
        for option in schema.pop("allOf", []):
            if not isinstance(option, dict):
                continue
            resolved, is_nullable = _extract_core_schema(
                option, root_schema, seen_refs
            )
            nullable = nullable or is_nullable
            merged = _merge_schema_dicts(merged, resolved)
        schema = _merge_schema_dicts(merged, schema)

    ref = schema.get("$ref")
    ref_schema = _resolve_ref_schema(ref, root_schema)
    if ref_schema and isinstance(ref, str) and ref not in seen_refs:
        seen_refs.add(ref)
        schema.pop("$ref", None)
        resolved, is_nullable = _extract_core_schema(
            ref_schema, root_schema, seen_refs
        )
        seen_refs.discard(ref)
        nullable = nullable or is_nullable
        schema = _merge_schema_dicts(resolved, schema)

    return schema, nullable


class Tool:
    """通用工具定义

    支持多种参数定义方式：
    1. 使用 parameters 列表（ToolParameter）
    2. 使用 args_schema（Pydantic BaseModel）

    Attributes:
        name: 工具名称
        description: 工具描述
        parameters: 参数列表（使用 ToolParameter 定义）
        args_schema: 参数模型（使用 Pydantic BaseModel 定义）
        func: 工具执行函数
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        parameters: Optional[List[ToolParameter]] = None,
        args_schema: Optional[Type[BaseModel]] = None,
        func: Optional[Callable] = None,
    ):
        # Normalize tool name to avoid external provider limits (e.g. OpenAI 64 chars)
        # If name length > 64, keep first 32 chars and append 32-char md5 sum of full name.
        self.name = normalize_tool_name(name)
        self.description = description
        self.parameters = list(parameters or [])
        self.args_schema = args_schema or _build_args_model_from_parameters(
            name, self.parameters
        )
        if self.args_schema is None:
            model_name = f"{self.name.title().replace('_', '')}Args"
            self.args_schema = create_model(model_name)  # type: ignore
        elif not self.parameters:
            self.parameters = self._generate_parameters_from_schema(
                self.args_schema
            )
        self.func = func

    def _generate_parameters_from_schema(
        self, schema: Type[BaseModel]
    ) -> List[ToolParameter]:
        """从 Pydantic schema 生成参数列表"""
        parameters = []
        json_schema = schema.model_json_schema()

        properties = json_schema.get("properties", {})
        required_fields = set(json_schema.get("required", []))

        for field_name, field_info in properties.items():
            normalized_schema, nullable = _extract_core_schema(
                field_info, json_schema
            )

            param_type = normalized_schema.get("type")
            if not param_type:
                if "properties" in normalized_schema:
                    param_type = "object"
                elif "items" in normalized_schema:
                    param_type = "array"
                elif "enum" in normalized_schema:
                    param_type = "string"
                else:
                    param_type = field_info.get("type", "string")

            description = (
                field_info.get("description")
                or normalized_schema.get("description")
                or ""
            )

            if "default" in field_info:
                default = field_info.get("default")
            else:
                default = normalized_schema.get("default")

            if "enum" in field_info:
                enum = field_info.get("enum")
            else:
                enum = normalized_schema.get("enum")

            items = normalized_schema.get("items")
            if items is not None:
                items = deepcopy(items)

            properties_schema = normalized_schema.get("properties")
            if properties_schema is not None:
                properties_schema = deepcopy(properties_schema)

            fmt = normalized_schema.get("format")

            if (
                not nullable
                and field_name not in required_fields
                and "default" in field_info
                and field_info.get("default") is None
            ):
                nullable = True

            parameters.append(
                ToolParameter(
                    name=field_name,
                    param_type=param_type,
                    description=description,
                    required=field_name in required_fields,
                    default=default,
                    enum=enum,
                    items=items,
                    properties=properties_schema,
                    format=fmt,
                    nullable=nullable,
                )
            )

        return parameters

    def get_parameters_schema(self) -> Dict[str, Any]:
        """获取参数的 JSON Schema"""
        if self.args_schema is None:
            return {"type": "object", "properties": {}}
        raw_schema = self.args_schema.model_json_schema()
        schema = deepcopy(raw_schema)

        properties = raw_schema.get("properties")
        if not isinstance(properties, dict):
            return schema

        normalized_properties: Dict[str, Any] = {}
        required_fields = set(schema.get("required", []))
        meta_keys = (
            "default",
            "description",
            "examples",
            "title",
            "deprecated",
            "enum",
            "const",
            "format",
        )
        for field_name, field_schema in properties.items():
            normalized, nullable = _extract_core_schema(
                field_schema, raw_schema
            )
            enriched = deepcopy(normalized) if normalized else {}

            for meta_key in meta_keys:
                if meta_key in field_schema:
                    enriched[meta_key] = deepcopy(field_schema[meta_key])

            # 确保 description 字段总是存在（即使为空字符串）
            if "description" not in enriched:
                enriched["description"] = ""

            default_is_none = (
                field_name not in required_fields
                and "default" in field_schema
                and field_schema.get("default") is None
            )

            if nullable or field_schema.get("nullable") or default_is_none:
                enriched["nullable"] = True

            normalized_properties[field_name] = enriched

        schema["properties"] = normalized_properties
        return schema

    def to_openai_function(self) -> Dict[str, Any]:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema(),
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """转换为 Anthropic Claude Tools 格式"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_parameters_schema(),
        }

    def to_langchain(self) -> Any:
        """转换为 LangChain Tool 格式

        优先使用适配器模式，如果适配器未注册则回退到旧实现。
        """
        # 尝试使用适配器模式
        try:
            from agentrun.integration.utils.canonical import CanonicalTool
            from agentrun.integration.utils.converter import get_converter

            converter = get_converter()
            adapter = converter._tool_adapters.get("langchain")
            if adapter is not None:
                # 转换为中间格式
                canonical_tool = CanonicalTool(
                    name=self.name,
                    description=self.description,
                    parameters=self.get_parameters_schema(),
                    func=self.func,
                )
                # 转换为 LangChain 格式
                result = adapter.from_canonical([canonical_tool])
                return result[0] if result else None
        except (ImportError, AttributeError, KeyError):
            pass

        # 回退到旧实现（保持向后兼容）
        try:
            from langchain_core.tools import StructuredTool  # type: ignore
        except ImportError as e:
            raise ImportError(
                "LangChain is not installed. "
                "Install it with: pip install langchain-core"
            ) from e

        return StructuredTool.from_function(
            func=self.func,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )

    def to_google_adk(self) -> Callable:
        """转换为 Google ADK Tool 格式

        优先使用适配器模式，如果适配器未注册则回退到旧实现。
        Google ADK 直接使用 Python 函数作为工具。
        """
        # 尝试使用适配器模式
        try:
            from agentrun.integration.utils.canonical import CanonicalTool
            from agentrun.integration.utils.converter import get_converter

            converter = get_converter()
            adapter = converter._tool_adapters.get("google_adk")
            if adapter is not None:
                # 转换为中间格式
                canonical_tool = CanonicalTool(
                    name=self.name,
                    description=self.description,
                    parameters=self.get_parameters_schema(),
                    func=self.func,
                )
                # 转换为 Google ADK 格式
                result = adapter.from_canonical([canonical_tool])
                if result:
                    return result[0]  # type: ignore
        except (ImportError, AttributeError, KeyError):
            pass

        # 回退到旧实现（保持向后兼容）
        if self.func is None:
            raise ValueError(
                f"Tool '{self.name}' has no function implementation"
            )
        return self.func

    def to_agentscope(self) -> Dict[str, Any]:
        """转换为 AgentScope Tool 格式"""
        try:
            from agentrun.integration.utils.canonical import CanonicalTool
            from agentrun.integration.utils.converter import get_converter

            converter = get_converter()
            adapter = converter._tool_adapters.get("agentscope")
            if adapter is not None:
                canonical_tool = CanonicalTool(
                    name=self.name,
                    description=self.description,
                    parameters=self.get_parameters_schema(),
                    func=self.func,
                )
                result = adapter.from_canonical([canonical_tool])
                return result[0] if result else {}
        except (ImportError, AttributeError, KeyError):
            pass

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema(),
            },
        }

    def to_langgraph(self) -> Any:
        """转换为 LangGraph Tool 格式

        LangGraph 与 LangChain 完全兼容，因此使用相同的接口。
        """
        return self.to_langchain()

    def to_crewai(self) -> Any:
        """转换为 CrewAI Tool 格式

        CrewAI 内部使用 LangChain，因此使用相同的接口。
        """
        return self.to_langchain()

    def to_pydanticai(self) -> Any:
        """转换为 PydanticAI Tool 格式"""
        # PydanticAI 使用函数装饰器定义工具
        # 返回一个包装函数，使其符合 PydanticAI 的工具接口
        if self.func is None:
            raise ValueError(
                f"Tool '{self.name}' has no function implementation"
            )

        # 创建一个包装函数，添加必要的元数据
        func = self.func

        # 添加函数文档字符串
        if not func.__doc__:
            # 使用 setattr 绕过类型检查
            object.__setattr__(func, "__doc__", self.description)

        # 添加函数名称
        if not hasattr(func, "__name__"):
            object.__setattr__(func, "__name__", self.name)

        # 为 PydanticAI 添加参数 schema 信息
        # 使用 setattr 绕过类型检查
        object.__setattr__(
            func,
            "_tool_schema",
            {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema(),
            },
        )

        return func

    def bind(self, instance: Any) -> "Tool":
        """绑定工具到实例，便于在类中定义工具方法"""

        if self.func is None:
            raise ValueError(
                f"Tool '{self.name}' has no function implementation"
            )

        # 复制当前工具的定义，覆写执行函数为绑定后的函数
        parameters = deepcopy(self.parameters) if self.parameters else None
        original_func = self.func

        @wraps(original_func)
        def bound(*args, **kwargs):
            return original_func(instance, *args, **kwargs)

        # 更新函数签名，移除 self 参数
        try:
            original_sig = inspect.signature(original_func)
            params = list(original_sig.parameters.values())
            if params and params[0].name == "self":
                params = params[1:]
            setattr(
                bound, "__signature__", original_sig.replace(parameters=params)
            )
        except (TypeError, ValueError):
            # 某些内置函数不支持签名操作，忽略即可
            pass

        # 清理 self 的注解，确保外部框架解析一致
        original_annotations = getattr(original_func, "__annotations__", {})
        if original_annotations:
            setattr(
                bound,
                "__annotations__",
                {
                    key: value
                    for key, value in original_annotations.items()
                    if key != "self"
                },
            )

        return Tool(
            name=self.name,
            description=self.description,
            parameters=parameters,
            args_schema=self.args_schema,
            func=bound,
        )

    def openai(self) -> Dict[str, Any]:
        """to_openai_function 的别名"""
        return self.to_openai_function()

    def langchain(self) -> Any:
        """to_langchain 的别名"""
        return self.to_langchain()

    def google_adk(self) -> Callable:
        """to_google_adk 的别名"""
        return self.to_google_adk()

    def pydanticai(self) -> Any:
        """to_pydanticai 的别名（暂未实现）"""
        # TODO: 实现 PydanticAI 转换
        raise NotImplementedError("PydanticAI conversion not implemented yet")

    def __call__(self, *args, **kwargs) -> Any:
        """直接调用工具函数"""
        if self.func is None:
            raise ValueError(
                f"Tool '{self.name}' has no function implementation"
            )
        return self.func(*args, **kwargs)


class CommonToolSet:
    """工具集

    管理多个工具，提供批量转换和过滤功能。

    默认会收集子类中定义的 `Tool` 属性，因此简单的继承即可完成
    工具集的声明。也可以通过传入 ``tools_list`` 或调用 ``register``
    方法来自定义工具集合。

    Attributes:
        tools_list: 工具列表
    """

    def __init__(self, tools_list: Optional[List[Tool]] = None):
        if tools_list is None:
            tools_list = self._collect_declared_tools()
        self._tools: List[Tool] = list(tools_list)

    def _collect_declared_tools(self) -> List[Tool]:
        """收集并绑定子类中定义的 Tool 对象"""

        tools: List[Tool] = []
        seen: set[str] = set()
        for cls in type(self).mro():
            if cls in {CommonToolSet, object}:
                continue
            class_dict = getattr(cls, "__dict__", {})
            for attr_name, attr_value in class_dict.items():
                if attr_name.startswith("_") or attr_name in seen:
                    continue
                if isinstance(attr_value, Tool):
                    seen.add(attr_name)
                    tools.append(attr_value.bind(self))
        return tools

    # def register(self, *tools: Tool) -> None:
    #     """动态注册工具"""

    #     for tool in tools:
    #         if not isinstance(tool, Tool):
    #             raise TypeError(
    #                 f"register() 只接受 Tool 实例, 收到类型: {type(tool)!r}"
    #             )
    #         self._tools.append(tool)

    def tools(
        self,
        prefix: Optional[str] = None,
        modify_tool_name: Optional[Callable[[Tool], Tool]] = None,
        filter_tools_by_name: Optional[Callable[[str], bool]] = None,
    ):
        """获取工具列表

        Args:
            prefix: 为所有工具名称添加前缀
            modify_tool_name: 自定义工具修改函数
            filter_tools_by_name: 工具名称过滤函数

        Returns:
            处理后的工具列表
        """
        tools = list(self._tools)

        # 应用过滤器
        if filter_tools_by_name is not None:
            tools = [t for t in tools if filter_tools_by_name(t.name)]

        # 应用前缀
        if prefix is not None:
            tools = [
                Tool(
                    name=f"{prefix}{t.name}",
                    description=t.description,
                    parameters=t.parameters,
                    args_schema=t.args_schema,
                    func=t.func,
                )
                for t in tools
            ]

        # 应用自定义修改函数
        if modify_tool_name is not None:
            tools = [modify_tool_name(t) for t in tools]

        from agentrun.integration.utils.canonical import CanonicalTool

        return [
            CanonicalTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.get_parameters_schema(),
                func=tool.func,
            )
            for tool in tools
        ]

    def __convert_tools(
        self,
        adapter_name: str,
        prefix: Optional[str] = None,
        modify_tool_name: Optional[Callable[[Tool], Tool]] = None,
        filter_tools_by_name: Optional[Callable[[str], bool]] = None,
    ):
        tools = self.tools(prefix, modify_tool_name, filter_tools_by_name)

        # 尝试使用适配器模式进行批量转换
        try:
            from agentrun.integration.utils.converter import get_converter

            converter = get_converter()
            adapter = converter._tool_adapters.get(adapter_name)
            if adapter is not None:
                return adapter.from_canonical(tools)
        except (ImportError, AttributeError, KeyError):
            pass

        logger.warning(
            f"adapter {adapter_name} not found, returning empty list"
        )
        return []

    @classmethod
    def from_agentrun_toolset(
        cls,
        toolset: ToolSet,
        config: Optional[Any] = None,
        refresh: bool = False,
    ) -> "CommonToolSet":
        """从 AgentRun ToolSet 创建通用工具集

        Args:
            toolset: agentrun.toolset.toolset.ToolSet 实例
            config: 额外的请求配置,调用工具时会自动合并
            refresh: 是否先刷新最新信息

        Returns:
            通用 ToolSet 实例,可直接调用 .to_openai_function()、.to_langchain() 等

        Example:
            >>> from agentrun.toolset.client import ToolSetClient
            >>> from agentrun.integration.common import from_agentrun_toolset
            >>>
            >>> client = ToolSetClient()
            >>> remote_toolset = client.get(name="my-toolset")
            >>> common_toolset = from_agentrun_toolset(remote_toolset)
            >>>
            >>> # 使用已有的转换方法
            >>> openai_tools = common_toolset.to_openai_function()
            >>> google_adk_tools = common_toolset.to_google_adk()
        """

        if refresh:
            toolset = toolset.get(config=config)

        # 获取所有工具元数据
        tools_meta = toolset.list_tools(config=config) or []
        integration_tools: List[Tool] = []
        seen_names = set()

        for meta in tools_meta:
            tool = _build_tool_from_meta(toolset, meta, config)
            if tool:
                # 检测工具名冲突
                if tool.name in seen_names:
                    logger.warning(
                        f"Duplicate tool name '{tool.name}' detected, "
                        "second occurrence will be skipped"
                    )
                    continue
                seen_names.add(tool.name)
                integration_tools.append(tool)

        return CommonToolSet(integration_tools)

    def to_openai_function(
        self,
        prefix: Optional[str] = None,
        modify_tool_name: Optional[Callable[[Tool], Tool]] = None,
        filter_tools_by_name: Optional[Callable[[str], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """批量转换为 OpenAI Function Calling 格式"""
        tools = self.tools(prefix, modify_tool_name, filter_tools_by_name)
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]

    def to_anthropic_tool(
        self,
        prefix: Optional[str] = None,
        modify_tool_name: Optional[Callable[[Tool], Tool]] = None,
        filter_tools_by_name: Optional[Callable[[str], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """批量转换为 Anthropic Claude Tools 格式"""
        tools = self.tools(prefix, modify_tool_name, filter_tools_by_name)
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def to_langchain(
        self,
        prefix: Optional[str] = None,
        modify_tool_name: Optional[Callable[[Tool], Tool]] = None,
        filter_tools_by_name: Optional[Callable[[str], bool]] = None,
    ) -> List[Any]:
        """批量转换为 LangChain Tool 格式

        优先使用适配器模式进行批量转换，提高效率。
        """
        return self.__convert_tools(
            "langchain", prefix, modify_tool_name, filter_tools_by_name
        )

    def to_google_adk(
        self,
        prefix: Optional[str] = None,
        modify_tool_name: Optional[Callable[[Tool], Tool]] = None,
        filter_tools_by_name: Optional[Callable[[str], bool]] = None,
    ):
        """批量转换为 Google ADK Tool 格式

        优先使用适配器模式进行批量转换，提高效率。
        """
        return self.__convert_tools(
            "google_adk", prefix, modify_tool_name, filter_tools_by_name
        )

    def to_agentscope(
        self,
        prefix: Optional[str] = None,
        modify_tool_name: Optional[Callable[[Tool], Tool]] = None,
        filter_tools_by_name: Optional[Callable[[str], bool]] = None,
    ):
        """批量转换为 AgentScope Tool 格式"""
        return self.__convert_tools(
            "agentscope", prefix, modify_tool_name, filter_tools_by_name
        )

    def to_langgraph(
        self,
        prefix: Optional[str] = None,
        modify_tool_name: Optional[Callable[[Tool], Tool]] = None,
        filter_tools_by_name: Optional[Callable[[str], bool]] = None,
    ) -> List[Any]:
        """批量转换为 LangGraph Tool 格式

        LangGraph 与 LangChain 完全兼容，因此使用相同的接口。
        """
        return self.__convert_tools(
            "langgraph", prefix, modify_tool_name, filter_tools_by_name
        )

    def to_crewai(
        self,
        prefix: Optional[str] = None,
        modify_tool_name: Optional[Callable[[Tool], Tool]] = None,
        filter_tools_by_name: Optional[Callable[[str], bool]] = None,
    ) -> List[Any]:
        """批量转换为 CrewAI Tool 格式

        CrewAI 内部使用 LangChain，因此使用相同的接口。
        """
        return self.__convert_tools(
            "crewai", prefix, modify_tool_name, filter_tools_by_name
        )

    def to_pydantic_ai(
        self,
        prefix: Optional[str] = None,
        modify_tool_name: Optional[Callable[[Tool], Tool]] = None,
        filter_tools_by_name: Optional[Callable[[str], bool]] = None,
    ) -> List[Any]:
        """批量转换为 PydanticAI Tool 格式"""
        return self.__convert_tools(
            "pydantic_ai", prefix, modify_tool_name, filter_tools_by_name
        )


def _extract_type_hints_from_function(
    func: Callable,
) -> Dict[str, Any]:
    """从函数签名提取类型提示"""
    try:
        hints = get_type_hints(func, include_extras=True)
    except TypeError:
        hints = get_type_hints(func)
    except Exception:
        # 如果无法获取类型提示，使用空字典
        hints = {}

    sig = inspect.signature(func)
    result = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param_name in hints:
            result[param_name] = hints[param_name]
        else:
            result[param_name] = str

    return result


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[Callable], Tool]:
    """工具装饰器

    从函数自动创建 Tool 实例。

    Args:
        name: 工具名称，默认使用函数名
        description: 工具描述，默认使用函数文档字符串

    Returns:
        装饰后的 Tool 实例

    Example:
        >>> @tool()
        ... def calculator(operation: Literal["add", "subtract"], a: float, b: float) -> float:
        ...     '''执行数学运算'''
        ...     if operation == "add":
        ...         return a + b
        ...     return a - b
    """

    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        # ensure tool name is normalized
        tool_name = normalize_tool_name(tool_name)
        tool_description = description or (
            func.__doc__.strip() if func.__doc__ else ""
        )

        # 获取函数签名
        sig = inspect.signature(func)
        type_hints = _extract_type_hints_from_function(func)

        # 构建 Pydantic 字段
        fields: Dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            param_type = type_hints.get(param_name, str)

            # 检查是否使用了 Annotated 类型
            origin = get_origin(param_type)
            if origin is not None:
                # 处理 Annotated[T, Field(...)] 形式
                args = get_args(param_type)
                if args:
                    actual_type = args[0]
                    # 查找 Field 对象
                    field_obj = None
                    for arg in args[1:]:
                        if isinstance(arg, type(Field())):
                            field_obj = arg
                            break

                    if field_obj is not None:
                        # 如果有默认值
                        if param.default is not inspect.Parameter.empty:
                            fields[param_name] = (
                                actual_type,
                                field_obj,
                            )
                        else:
                            fields[param_name] = (actual_type, field_obj)
                    else:
                        # 没有 Field，使用基本类型
                        if param.default is not inspect.Parameter.empty:
                            fields[param_name] = (
                                actual_type,
                                Field(default=param.default),
                            )
                        else:
                            fields[param_name] = (actual_type, ...)
                else:
                    # 无法解析，使用字符串
                    if param.default is not inspect.Parameter.empty:
                        fields[param_name] = (str, Field(default=param.default))
                    else:
                        fields[param_name] = (str, ...)
            else:
                # 普通类型
                if param.default is not inspect.Parameter.empty:
                    fields[param_name] = (
                        param_type,
                        Field(default=param.default),
                    )
                else:
                    fields[param_name] = (param_type, ...)

        # 创建 Pydantic 模型
        # 模型名称与工具名保持一致
        model_name = tool_name
        args_schema = create_model(model_name, **fields)  # type: ignore

        def _func(*args, **kwargs):
            logger.debug(
                "invoke tool %s with arguments %s %s", tool_name, args, kwargs
            )
            results = func(*args, **kwargs)
            logger.debug(
                "invoke tool %s finished with result %s", tool_name, results
            )

            # if type(results) is str:
            #     return {"result": results}

            return results

        return Tool(
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
            func=_func,
        )

    return decorator


def from_pydantic(
    name: str,
    description: str,
    args_schema: Type[BaseModel],
    func: Callable,
) -> Tool:
    """从 Pydantic BaseModel 创建 Tool

    Args:
        name: 工具名称
        description: 工具描述
        args_schema: Pydantic BaseModel 类
        func: 工具执行函数

    Returns:
        Tool 实例

    Example:
        >>> class SearchArgs(BaseModel):
        ...     query: str = Field(description="搜索关键词")
        ...     limit: int = Field(description="结果数量", default=10)
        >>>
        >>> search_tool = from_pydantic(
        ...     name="search",
        ...     description="搜索网络信息",
        ...     args_schema=SearchArgs,
        ...     func=lambda query, limit: f"搜索: {query}"
        ... )
    """
    return Tool(
        name=name,
        description=description,
        args_schema=args_schema,
        func=func,
    )


"""AgentRun ToolSet 适配器

将 AgentRun 控制面的 ToolSet 资源适配为通用的 Tool/ToolSet 抽象,
从而无缝集成到各个 AI 框架中。

核心思路：
1. 从 toolset.list_tools() 获取工具元数据
2. 为每个工具构建 Pydantic schema
3. 使用 from_pydantic 创建 Tool（自动处理所有转换逻辑）
"""


def _build_tool_from_meta(
    toolset: Any,
    meta: Any,
    config: Optional[Any],
) -> Optional[Tool]:
    """从工具元数据构建 Tool - 使用 from_pydantic 复用已有逻辑"""
    meta_dict = _to_dict(meta)

    # 提取工具名称
    tool_name = (
        meta_dict.get("name")
        or meta_dict.get("operationId")
        or meta_dict.get("tool_id")
    )
    if not tool_name:
        return None

    # 提取描述（注意：description 可能是 None，需要显式转换）
    description = (
        meta_dict.get("description")
        or meta_dict.get("summary")
        or f"{meta_dict.get('method', '')} {meta_dict.get('path', '')}".strip()
        or ""
    )

    # 构建 Pydantic schema
    args_schema = _build_args_schema(tool_name, meta_dict)
    if not args_schema:
        # 无参数工具或 schema 解析失败
        if (
            "input_schema" in meta_dict
            or "parameters" in meta_dict
            or "request_body" in meta_dict
        ):
            logger.debug(
                f"Failed to parse schema for tool '{tool_name}', using empty"
                " schema"
            )
        args_schema = create_model(f"{tool_name}_Args")

    # 创建带有正确签名的调用函数
    call_tool_func = _create_function_with_signature(
        tool_name, args_schema, toolset, config
    )

    # 设置函数的 docstring（Google ADK 可能会读取）
    if description:
        call_tool_func.__doc__ = description

    # 使用 from_pydantic 创建 Tool (利用已有的转换逻辑)
    return from_pydantic(
        name=tool_name,
        description=description,
        args_schema=args_schema,
        func=call_tool_func,
    )


def _normalize_tool_arguments(
    raw_kwargs: Dict[str, Any],
    args_schema: Optional[Type[BaseModel]],
) -> Dict[str, Any]:
    """根据 schema 对传入参数做简单归一化，提升 LLM 容错能力"""

    if not raw_kwargs or args_schema is None:
        return raw_kwargs

    normalized = dict(raw_kwargs)
    model_fields = list(args_schema.model_fields.keys())
    if not model_fields:
        return normalized

    alias_map: Dict[str, str] = getattr(
        args_schema, "__agentrun_argument_aliases__", {}
    )
    for alias, canonical in alias_map.items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized.pop(alias)

    if (
        len(model_fields) == 1
        and model_fields[0] not in normalized
        and len(normalized) == 1
    ):
        _, value = normalized.popitem()
        normalized[model_fields[0]] = value
        return normalized

    # 次要策略：忽略大小写和分隔符进行匹配
    def _simplify(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    field_map = {_simplify(name): name for name in model_fields}

    for key in list(normalized.keys()):
        if key in args_schema.model_fields:
            continue
        simplified = _simplify(key)
        target = field_map.get(simplified)
        if target and target not in normalized:
            normalized[target] = normalized.pop(key)

    return normalized


def _maybe_add_body_alias(
    field_name: str, field_schema: Dict[str, Any], total_fields: int
) -> Tuple[str, ...]:
    """为展开的 body 字段添加常见别名，方便 LLM 调用"""

    aliases: List[str] = []
    name_lower = field_name.lower()

    if total_fields == 1 and "query" not in name_lower:
        if "search" in name_lower or name_lower.endswith("_input"):
            aliases.append("query")

    if not aliases:
        return tuple()

    extra_key = "x-aliases"
    existing = field_schema.get(extra_key, [])
    if not isinstance(existing, list):
        existing = []

    for alias in aliases:
        if not alias or alias == field_name or alias in existing:
            continue
        existing.append(alias)

    if existing:
        field_schema[extra_key] = existing

    return tuple(alias for alias in aliases if alias)


def _create_function_with_signature(
    tool_name: str,
    args_schema: Type[BaseModel],
    toolset: Any,
    config: Optional[Any],
) -> Callable:
    """创建一个带有正确签名的函数

    从 Pydantic schema 提取参数信息，动态构建函数签名
    """
    from functools import wraps
    import inspect

    # 从 Pydantic model 构建参数列表
    parameters = []
    for field_name, field_info in args_schema.model_fields.items():
        annotation = field_info.annotation

        # 处理默认值
        from pydantic_core import PydanticUndefined

        if field_info.default is not PydanticUndefined:
            default = field_info.default
        elif not field_info.is_required():
            default = None
        else:
            default = inspect.Parameter.empty

        param = inspect.Parameter(
            field_name,
            inspect.Parameter.KEYWORD_ONLY,
            default=default,
            annotation=annotation,
        )
        parameters.append(param)

    alias_map: Dict[str, str] = getattr(
        args_schema, "__agentrun_argument_aliases__", {}
    )
    if alias_map:
        for alias, canonical in alias_map.items():
            canonical_field = args_schema.model_fields.get(canonical)
            alias_annotation = (
                canonical_field.annotation
                if canonical_field
                else inspect._empty
            )
            if (
                alias_annotation is not inspect._empty
                and alias_annotation is not None
            ):
                alias_annotation = Optional[alias_annotation]
            parameters.append(
                inspect.Parameter(
                    alias,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=alias_annotation,
                )
            )

    # 创建实际执行函数
    def impl(**kwargs):
        # 使用 Pydantic 进行参数校验,确保 LLM 传入的字段合法
        normalized_kwargs = _normalize_tool_arguments(kwargs, args_schema)

        if args_schema is not None:
            try:
                parsed = args_schema(**normalized_kwargs)
                payload = parsed.model_dump(mode="python", exclude_unset=True)
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid arguments for tool '{tool_name}': {exc}"
                ) from exc
        else:
            payload = normalized_kwargs

        # 转换类型并映射字段名（body -> json）
        converted_kwargs = {
            ("json" if k == "body" else k): _convert_to_native(v)
            for k, v in payload.items()
        }
        body_fields = getattr(args_schema, "__agentrun_body_fields__", None)
        if body_fields:
            body_payload = {}
            for field in body_fields:
                if field in converted_kwargs:
                    body_payload[field] = converted_kwargs.pop(field)
            if body_payload and "json" not in converted_kwargs:
                converted_kwargs["json"] = body_payload

        return toolset.call_tool(
            name=tool_name, arguments=converted_kwargs, config=config
        )

    # 创建包装函数并设置签名
    @wraps(impl)
    def wrapper(**kwargs):
        return impl(**kwargs)

    # 设置函数属性（清理特殊字符，确保是有效的 Python 标识符）
    clean_name = re.sub(r"[^0-9a-zA-Z_]", "_", tool_name)
    clean_name = normalize_tool_name(clean_name)
    wrapper.__name__ = clean_name
    wrapper.__qualname__ = clean_name
    if parameters:
        wrapper.__signature__ = inspect.Signature(parameters)  # type: ignore

    return wrapper


def _convert_to_native(value: Any) -> Any:
    """转换框架特定类型为 Python 原生类型

    主要处理 Google ADK 的 Nestedobject 等特殊类型
    """
    # 如果是基本类型，直接返回
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # 如果是列表，递归转换
    if isinstance(value, list):
        return [_convert_to_native(item) for item in value]

    # 如果是字典，递归转换
    if isinstance(value, dict):
        return {k: _convert_to_native(v) for k, v in value.items()}

    # 检查类型名称，处理 Pydantic BaseModel 或类似的对象
    type_name = type(value).__name__
    if "Object" in type_name or hasattr(value, "model_dump"):
        # Pydantic BaseModel 或 Google ADK NestedObject
        if hasattr(value, "model_dump"):
            try:
                return _convert_to_native(value.model_dump())
            except Exception:
                pass

        # Google ADK NestedObject 等有 to_dict 方法的对象
        if hasattr(value, "to_dict"):
            try:
                return _convert_to_native(value.to_dict())
            except Exception:
                pass

    # 尝试转换为字典（处理其他特殊类型）
    if hasattr(value, "to_dict"):
        try:
            return _convert_to_native(value.to_dict())
        except Exception:
            pass

    # 尝试使用 __dict__ 属性
    if hasattr(value, "__dict__"):
        try:
            return _convert_to_native(value.__dict__)
        except Exception:
            pass

    # 如果是 Pydantic BaseModel，尝试提取字段值
    if hasattr(value, "__fields__") or hasattr(value, "model_fields"):
        try:
            result = {}
            # 新版本 Pydantic v2
            if hasattr(value, "model_fields"):
                for field_name in value.model_fields:
                    if hasattr(value, field_name):
                        result[field_name] = _convert_to_native(
                            getattr(value, field_name)
                        )
            # 旧版本 Pydantic v1
            elif hasattr(value, "__fields__"):
                for field_name in value.__fields__:
                    if hasattr(value, field_name):
                        result[field_name] = _convert_to_native(
                            getattr(value, field_name)
                        )
            return result
        except Exception:
            pass

    # 如果无法转换，返回原值（可能会导致序列化错误）
    logger.debug(
        f"Unable to convert object of type {type(value)} to native Python type:"
        f" {value}"
    )
    return value


def _build_args_schema(
    tool_name: str,
    meta: Dict[str, Any],
) -> Optional[Type[BaseModel]]:
    """构建参数 schema"""
    # 新的统一格式：ToolInfo.parameters (ToolSchema 对象)
    # 当 meta 是从 toolset.list_tools() 返回的 ToolInfo 时
    if "parameters" in meta and isinstance(meta.get("parameters"), dict):
        # 检查是否是 ToolSchema 格式（有 type/properties/required）
        params = meta.get("parameters")
        if params and "type" in params:
            # 这是 ToolSchema 的 JSON 表示，直接使用
            return _json_schema_to_pydantic(f"{tool_name}_Args", params)

    # MCP 格式（旧格式，保持兼容）
    if "input_schema" in meta:
        schema = _load_json(meta.get("input_schema"))
        return _json_schema_to_pydantic(f"{tool_name}_Args", schema)

    # OpenAPI 格式（旧格式，保持兼容）
    # 注意：这里的 parameters 是数组格式，不同于上面的 ToolSchema
    if "parameters" in meta and isinstance(meta.get("parameters"), list):
        schema, body_info, alias_map = _build_openapi_schema(meta)
        model = _json_schema_to_pydantic(f"{tool_name}_Args", schema)
        if model is not None and body_info:
            setattr(model, "__agentrun_body_fields__", tuple(body_info))
        if model is not None and alias_map:
            setattr(model, "__agentrun_argument_aliases__", dict(alias_map))
        return model

    # OpenAPI 格式（有 request_body）
    if "request_body" in meta:
        schema, body_info, alias_map = _build_openapi_schema(meta)
        model = _json_schema_to_pydantic(f"{tool_name}_Args", schema)
        if model is not None and body_info:
            setattr(model, "__agentrun_body_fields__", tuple(body_info))
        if model is not None and alias_map:
            setattr(model, "__agentrun_argument_aliases__", dict(alias_map))
        return model

    return None


def _build_openapi_schema(
    meta: Dict[str, Any],
) -> Tuple[Dict[str, Any], Tuple[str, ...], Dict[str, str]]:
    """从 OpenAPI 元数据构建 JSON Schema

    Returns:
        schema: JSON Schema dict
        body_fields: 若 requestBody 为对象并被展开, 返回其字段名称元组
    """
    properties = {}
    required = []
    body_field_names: List[str] = []
    alias_map: Dict[str, str] = {}

    # 处理 parameters
    for param in meta.get("parameters", []):
        if not isinstance(param, dict):
            continue
        name = param.get("name")
        if not name:
            continue

        schema = param.get("schema", {})
        if isinstance(schema, dict):
            properties[name] = {
                **schema,
                "description": param.get("description") or schema.get(
                    "description", ""
                ),
            }
            if param.get("required"):
                required.append(name)

    # 处理 requestBody
    request_body = meta.get("request_body", {})
    if isinstance(request_body, dict):
        content = request_body.get("content", {})
        for content_type in [
            "application/json",
            "application/x-www-form-urlencoded",
        ]:
            if content_type in content:
                body_schema = content[content_type].get("schema", {})
                if body_schema:
                    if (
                        isinstance(body_schema, dict)
                        and body_schema.get("type") == "object"
                        and body_schema.get("properties")
                        and not any(
                            name in properties
                            for name in body_schema["properties"].keys()
                        )
                    ):
                        # 将 body 属性展开，提升 LLM 可用性
                        nested_required = set(body_schema.get("required", []))
                        total_body_fields = len(body_schema["properties"])
                        for field_name, field_schema in body_schema[
                            "properties"
                        ].items():
                            aliases = _maybe_add_body_alias(
                                field_name,
                                field_schema,
                                total_body_fields,
                            )
                            for alias in aliases:
                                alias_map[alias] = field_name
                            properties[field_name] = field_schema
                            if field_name in nested_required:
                                required.append(field_name)
                            body_field_names.append(field_name)
                    else:
                        # 使用 body 字段
                        properties["body"] = body_schema
                        if request_body.get("required"):
                            required.append("body")
                break

    schema = {
        "type": "object",
        "properties": properties,
        "required": list(dict.fromkeys(required)) if required else [],
    }

    return schema, tuple(body_field_names), alias_map


def _json_schema_to_pydantic(
    name: str,
    schema: Optional[Dict[str, Any]],
) -> Optional[Type[BaseModel]]:
    """将 JSON Schema 转换为 Pydantic 模型"""
    if not schema or not isinstance(schema, dict):
        return None

    properties = schema.get("properties", {})
    if not properties:
        return None

    required_fields = set(schema.get("required", []))
    fields = {}

    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            continue

        # 映射类型
        field_type = _json_type_to_python(field_schema)
        description = field_schema.get("description", "")
        default = field_schema.get("default")
        aliases = field_schema.get("x-aliases")
        field_kwargs: Dict[str, Any] = {"description": description}
        if aliases:
            if not isinstance(aliases, (list, tuple)):
                aliases = [aliases]
            field_kwargs["validation_alias"] = AliasChoices(
                field_name, *aliases
            )

        # 构建字段定义
        if field_name in required_fields:
            # 必填字段
            fields[field_name] = (field_type, Field(**field_kwargs))
        else:
            # 可选字段
            from typing import Optional as TypingOptional

            fields[field_name] = (
                TypingOptional[field_type],
                Field(default=default, **field_kwargs),
            )

    # 创建模型,清理名称
    model_name = re.sub(r"[^0-9a-zA-Z]", "", name.title())
    return create_model(model_name or "Args", **fields)  # type: ignore


def _json_type_to_python(field_schema: Dict[str, Any]) -> type:
    """映射 JSON Schema 类型到 Python 类型"""
    schema_type = field_schema.get("type", "string")

    # 处理嵌套对象和数组
    if schema_type == "object":
        # 如果有 properties,递归创建嵌套模型
        properties = field_schema.get("properties")
        if properties:
            nested_model = _json_schema_to_pydantic(
                "NestedObject", field_schema
            )
            return nested_model if nested_model else dict
        return dict

    if schema_type == "array":
        items_schema = field_schema.get("items", {})
        if isinstance(items_schema, dict):
            item_type = _json_type_to_python(items_schema)
            from typing import List as TypingList

            return TypingList[item_type]  # type: ignore
        return list

    # 基本类型映射
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    return mapping.get(schema_type, str)


def _build_args_model_from_parameters(
    tool_name: str, parameters: List[ToolParameter]
) -> Optional[Type[BaseModel]]:
    """根据 ToolParameter 定义构建 Pydantic 模型"""

    if not parameters:
        return None

    schema = {
        "type": "object",
        "properties": {
            param.name: param.to_json_schema() for param in parameters
        },
        "required": [
            param.name
            for param in parameters
            if getattr(param, "required", False)
        ],
    }

    return _json_schema_to_pydantic(f"{tool_name}_Args", schema)


def _load_json(data: Any) -> Optional[Dict[str, Any]]:
    """加载 JSON 数据"""
    if isinstance(data, dict):
        return data
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None
    return None


def _to_dict(obj: Any) -> Dict[str, Any]:
    """转换为字典"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)

    # 使用 __dict__ 更安全（避免获取方法和属性）
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}

    return {}
