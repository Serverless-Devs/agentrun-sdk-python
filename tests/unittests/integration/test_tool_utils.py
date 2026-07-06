"""工具定义和转换模块测试

测试 agentrun.integration.utils.tool 模块。
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from pydantic import BaseModel, Field
import pytest

from agentrun.integration.utils.tool import (
    _build_tool_from_meta,
    _create_function_with_signature,
    _extract_core_schema,
    _json_schema_to_pydantic,
    _load_json,
    _merge_schema_dicts,
    _normalize_tool_arguments,
    _reachable_refs_for_ref,
    _sanitize_python_identifier,
    _to_dict,
    CommonToolSet,
    from_pydantic,
    normalize_tool_name,
    Tool,
    tool,
    ToolParameter,
)


class TestNormalizeToolName:
    """测试 normalize_tool_name 函数"""

    def test_short_name_unchanged(self):
        """测试短名称保持不变"""
        name = "short_tool"
        result = normalize_tool_name(name)
        assert result == name

    def test_exact_max_length(self):
        """测试恰好等于最大长度的名称"""
        name = "a" * 64
        result = normalize_tool_name(name)
        assert result == name

    def test_long_name_normalized(self):
        """测试长名称被规范化"""
        name = "a" * 100
        result = normalize_tool_name(name)
        assert len(result) == 64
        assert result.startswith("a" * 32)

    def test_non_string_input(self):
        """测试非字符串输入"""
        result = normalize_tool_name(123)
        assert result == "123"


class TestToolParameter:
    """测试 ToolParameter 类"""

    def test_init_basic(self):
        """测试基本初始化"""
        param = ToolParameter(
            name="test_param",
            param_type="string",
            description="A test parameter",
        )
        assert param.name == "test_param"
        assert param.param_type == "string"
        assert param.description == "A test parameter"
        assert param.required is False
        assert param.default is None

    def test_init_with_required(self):
        """测试必需参数"""
        param = ToolParameter(
            name="required_param",
            param_type="integer",
            description="A required parameter",
            required=True,
        )
        assert param.required is True

    def test_init_with_default(self):
        """测试带默认值的参数"""
        param = ToolParameter(
            name="default_param",
            param_type="number",
            description="A parameter with default",
            default=42.0,
        )
        assert param.default == 42.0

    def test_init_with_enum(self):
        """测试枚举参数"""
        param = ToolParameter(
            name="enum_param",
            param_type="string",
            description="An enum parameter",
            enum=["option1", "option2", "option3"],
        )
        assert param.enum == ["option1", "option2", "option3"]


class TestTool:
    """测试 Tool 类"""

    def test_init_basic(self):
        """测试基本初始化"""

        def sample_func(x: int) -> int:
            return x * 2

        tool_obj = Tool(
            name="sample_tool",
            description="A sample tool",
            func=sample_func,
        )
        assert tool_obj.name == "sample_tool"
        assert tool_obj.description == "A sample tool"
        assert tool_obj.func is sample_func

    def test_init_with_parameters(self):
        """测试带参数的初始化"""
        params = [
            ToolParameter("x", "integer", "Input value", required=True),
        ]

        def sample_func(x: int) -> int:
            return x * 2

        tool_obj = Tool(
            name="sample_tool",
            description="A sample tool",
            parameters=params,
            func=sample_func,
        )
        assert len(tool_obj.parameters) == 1
        assert tool_obj.parameters[0].name == "x"

    def test_call_method(self):
        """测试 Tool 对象的 __call__ 方法（如果存在）"""

        def sample_func(x: int) -> int:
            return x * 2

        tool_obj = Tool(
            name="sample_tool",
            description="A sample tool",
            func=sample_func,
        )

        # 直接调用 func 属性
        assert tool_obj.func(5) == 10


class TestToolDecorator:
    """测试 @tool 装饰器"""

    def test_decorator_returns_tool_object(self):
        """测试装饰器返回 Tool 对象"""

        @tool()
        def my_tool(x: int) -> int:
            """Multiply x by 2"""
            return x * 2

        # 验证返回的是 Tool 对象
        assert isinstance(my_tool, Tool)
        assert my_tool.name == "my_tool"
        assert my_tool.description == "Multiply x by 2"

    def test_decorator_with_custom_name(self):
        """测试带自定义名称的装饰器"""

        @tool(name="custom_name")
        def my_tool(x: int) -> int:
            """A custom named tool"""
            return x * 2

        assert isinstance(my_tool, Tool)
        assert my_tool.name == "custom_name"

    def test_decorator_with_custom_description(self):
        """测试带自定义描述的装饰器"""

        @tool(description="Custom description")
        def my_tool(x: int) -> int:
            """Original docstring"""
            return x * 2

        assert isinstance(my_tool, Tool)
        assert my_tool.description == "Custom description"

    def test_decorator_uses_docstring_as_description(self):
        """测试装饰器使用文档字符串作为描述"""

        @tool()
        def my_tool(x: int) -> int:
            """This is the tool description from docstring."""
            return x * 2

        assert isinstance(my_tool, Tool)
        assert "docstring" in my_tool.description.lower()

    def test_decorator_without_parentheses(self):
        """测试不带括号的装饰器用法（如果支持）"""

        # 注意：根据装饰器实现，可能需要使用 @tool() 而不是 @tool
        @tool()
        def simple_tool(name: str) -> str:
            """Greet someone"""
            return f"Hello, {name}"

        assert isinstance(simple_tool, Tool)
        assert simple_tool.name == "simple_tool"

    def test_decorator_preserves_func(self):
        """测试装饰器保留函数引用"""

        @tool()
        def my_func(x: int) -> int:
            """Double x"""
            return x * 2

        # 验证 func 属性可用并可调用
        assert my_func.func is not None
        assert callable(my_func.func)
        assert my_func.func(5) == 10

    def test_decorator_with_multiple_params(self):
        """测试带多个参数的装饰器"""

        @tool()
        def add_numbers(a: float, b: float) -> float:
            """Add two numbers"""
            return a + b

        assert isinstance(add_numbers, Tool)
        assert add_numbers.name == "add_numbers"
        assert add_numbers.func(1.5, 2.5) == 4.0


class TestToolDefinitionWithPydanticModel:
    """测试使用 Pydantic 模型的工具定义"""

    def test_tool_with_pydantic_param(self):
        """测试使用 Pydantic 模型作为参数"""

        class UserInput(BaseModel):
            name: str = Field(description="User name")
            age: int = Field(description="User age")

        @tool()
        def greet_user(user: UserInput) -> str:
            """Greet a user"""
            return f"Hello, {user.name}! You are {user.age} years old."

        # 验证返回的是 Tool 对象
        assert isinstance(greet_user, Tool)
        assert greet_user.name == "greet_user"

        # 通过 func 属性调用函数
        user = UserInput(name="Alice", age=30)
        result = greet_user.func(user)
        assert "Alice" in result
        assert "30" in result


class TestToolDefinitionTypeHints:
    """测试工具定义的类型提示处理"""

    def test_tool_with_optional_param(self):
        """测试可选参数"""

        @tool()
        def optional_tool(name: str, age: Optional[int] = None) -> str:
            """A tool with optional parameter"""
            if age:
                return f"{name} is {age}"
            return name

        assert isinstance(optional_tool, Tool)
        # 通过 func 调用
        assert optional_tool.func("Alice") == "Alice"
        assert optional_tool.func("Bob", 25) == "Bob is 25"

    def test_tool_with_list_param(self):
        """测试列表参数"""

        @tool()
        def list_tool(items: List[str]) -> int:
            """Count items in list"""
            return len(items)

        assert isinstance(list_tool, Tool)
        assert list_tool.func(["a", "b", "c"]) == 3

    def test_tool_with_default_values(self):
        """测试带默认值的参数"""

        @tool()
        def default_tool(x: int = 10, y: int = 20) -> int:
            """Add two numbers with defaults"""
            return x + y

        assert isinstance(default_tool, Tool)
        assert default_tool.func() == 30
        assert default_tool.func(5) == 25
        assert default_tool.func(5, 5) == 10

    def test_tool_long_name_normalized(self):
        """测试长名称被自动规范化"""
        long_name = "a_very_long_tool_name_that_exceeds_the_maximum_length_of_sixty_four_characters"

        @tool(name=long_name)
        def my_func(x: int) -> int:
            """Test func"""
            return x

        assert isinstance(my_func, Tool)
        assert len(my_func.name) == 64


class TestFromPydantic:
    """测试 from_pydantic 函数"""

    def test_basic_usage(self):
        """测试基本使用"""

        class SearchArgs(BaseModel):
            query: str = Field(description="搜索关键词")
            limit: int = Field(description="结果数量", default=10)

        def search_func(query: str, limit: int = 10) -> str:
            return f"搜索: {query}, 限制: {limit}"

        search_tool = from_pydantic(
            name="search",
            description="搜索网络信息",
            args_schema=SearchArgs,
            func=search_func,
        )

        assert isinstance(search_tool, Tool)
        assert search_tool.name == "search"
        assert search_tool.description == "搜索网络信息"
        assert search_tool.args_schema is SearchArgs


class TestToolParameterToJsonSchema:
    """测试 ToolParameter.to_json_schema 方法"""

    def test_basic_schema(self):
        """测试基本 schema 转换"""
        param = ToolParameter(
            name="name",
            param_type="string",
            description="User name",
        )
        schema = param.to_json_schema()

        assert schema["type"] == "string"
        assert schema["description"] == "User name"

    def test_with_default(self):
        """测试带默认值的 schema"""
        param = ToolParameter(
            name="count",
            param_type="integer",
            description="Count",
            default=10,
        )
        schema = param.to_json_schema()

        assert schema["default"] == 10

    def test_with_enum(self):
        """测试带枚举的 schema"""
        param = ToolParameter(
            name="color",
            param_type="string",
            description="Color choice",
            enum=["red", "green", "blue"],
        )
        schema = param.to_json_schema()

        assert schema["enum"] == ["red", "green", "blue"]

    def test_with_format(self):
        """测试带格式的 schema"""
        param = ToolParameter(
            name="id",
            param_type="integer",
            description="ID",
            format="int64",
        )
        schema = param.to_json_schema()

        assert schema["format"] == "int64"

    def test_nullable(self):
        """测试可空 schema"""
        param = ToolParameter(
            name="optional",
            param_type="string",
            description="Optional field",
            nullable=True,
        )
        schema = param.to_json_schema()

        assert schema["nullable"] is True

    def test_array_with_items(self):
        """测试数组类型 schema"""
        param = ToolParameter(
            name="tags",
            param_type="array",
            description="Tags list",
            items={"type": "string"},
        )
        schema = param.to_json_schema()

        assert schema["type"] == "array"
        assert schema["items"] == {"type": "string"}

    def test_object_with_properties(self):
        """测试对象类型 schema"""
        param = ToolParameter(
            name="user",
            param_type="object",
            description="User object",
            properties={"name": {"type": "string"}, "age": {"type": "integer"}},
        )
        schema = param.to_json_schema()

        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]


class TestMergeSchemaDicts:
    """测试 _merge_schema_dicts 函数"""

    def test_merge_empty_base(self):
        """测试空 base 的合并"""
        override = {"type": "string", "description": "Test"}
        result = _merge_schema_dicts({}, override)

        assert result == override

    def test_merge_override_takes_priority(self):
        """测试 override 优先级更高"""
        base = {"type": "string", "description": "Base"}
        override = {"description": "Override"}
        result = _merge_schema_dicts(base, override)

        assert result["type"] == "string"
        assert result["description"] == "Override"

    def test_merge_nested(self):
        """测试嵌套合并"""
        base = {"type": "object", "properties": {"name": {"type": "string"}}}
        override = {"properties": {"age": {"type": "integer"}}}
        result = _merge_schema_dicts(base, override)

        assert result["type"] == "object"
        assert "name" in result["properties"]
        assert "age" in result["properties"]


class TestExtractCoreSchema:
    """测试 _extract_core_schema 函数"""

    def test_simple_schema(self):
        """测试简单 schema - 返回 (schema, nullable) 元组"""
        schema = {"type": "string", "description": "A string"}
        result_schema, nullable = _extract_core_schema(schema, schema)

        assert result_schema["type"] == "string"
        assert nullable is False

    def test_schema_with_allOf(self):
        """测试带 allOf 的 schema"""
        defs = {"StringType": {"type": "string"}}
        schema = {"allOf": [{"$ref": "#/$defs/StringType"}]}
        full_schema = {"$defs": defs}

        result_schema, nullable = _extract_core_schema(schema, full_schema)
        assert result_schema is not None


class TestLoadJson:
    """测试 _load_json 函数"""

    def test_load_dict(self):
        """测试加载字典"""
        data = {"key": "value"}
        result = _load_json(data)

        assert result == data

    def test_load_json_string(self):
        """测试加载 JSON 字符串"""
        data = '{"key": "value"}'
        result = _load_json(data)

        assert result == {"key": "value"}

    def test_load_invalid_json(self):
        """测试加载无效 JSON"""
        result = _load_json("not a json")

        assert result is None

    def test_load_none(self):
        """测试加载 None"""
        result = _load_json(None)

        assert result is None


class TestToDict:
    """测试 _to_dict 函数"""

    def test_dict_passthrough(self):
        """测试字典直接返回"""
        data = {"key": "value"}
        result = _to_dict(data)

        assert result == data

    def test_pydantic_model(self):
        """测试 Pydantic 模型转换"""

        class TestModel(BaseModel):
            name: str
            age: int

        model = TestModel(name="Alice", age=30)
        result = _to_dict(model)

        assert result["name"] == "Alice"
        assert result["age"] == 30

    def test_object_with_dict(self):
        """测试带 __dict__ 的普通对象"""

        class MockObj:

            def __init__(self):
                self.key = "value"
                self.name = "test"

        result = _to_dict(MockObj())

        assert result["key"] == "value"
        assert result["name"] == "test"


class TestNormalizeToolArguments:
    """测试 _normalize_tool_arguments 函数"""

    def test_basic_normalization(self):
        """测试基本参数规范化 - 参数顺序是 (raw_kwargs, args_schema)"""

        class TestArgs(BaseModel):
            name: str
            count: int

        args = {"name": "test", "count": "10"}
        # 参数顺序是 (raw_kwargs, args_schema)
        result = _normalize_tool_arguments(args, TestArgs)

        assert result["name"] == "test"
        # 注意：此函数可能不会做类型转换，只是简单规范化
        assert result["count"] == "10" or result["count"] == 10

    def test_with_none_schema(self):
        """测试 schema 为 None 时"""
        args = {"name": "test", "count": 10}
        result = _normalize_tool_arguments(args, None)

        assert result == args

    def test_with_empty_kwargs(self):
        """测试空参数"""

        class TestArgs(BaseModel):
            name: str

        result = _normalize_tool_arguments({}, TestArgs)

        assert result == {}


class TestCommonToolSet:
    """测试 CommonToolSet 类"""

    def test_init_with_tools_list(self):
        """测试用工具列表初始化"""

        def func1():
            return "result1"

        tool1 = Tool(name="tool1", description="Tool 1", func=func1)

        toolset = CommonToolSet(tools_list=[tool1])
        tools = toolset.tools()

        assert len(tools) == 1

    def test_tools_with_filter(self):
        """测试工具过滤"""

        def func1():
            return "result1"

        def func2():
            return "result2"

        tool1 = Tool(name="search_tool", description="Search", func=func1)
        tool2 = Tool(name="other_tool", description="Other", func=func2)

        toolset = CommonToolSet(tools_list=[tool1, tool2])

        # 只保留名称包含 "search" 的工具
        filtered_tools = toolset.tools(
            filter_tools_by_name=lambda name: "search" in name
        )

        assert len(filtered_tools) == 1
        assert filtered_tools[0].name == "search_tool"

    def test_tools_with_prefix(self):
        """测试工具名称前缀"""

        def func1():
            return "result1"

        tool1 = Tool(name="mytool", description="My Tool", func=func1)

        toolset = CommonToolSet(tools_list=[tool1])
        prefixed_tools = toolset.tools(prefix="prefix_")

        assert len(prefixed_tools) == 1
        assert prefixed_tools[0].name == "prefix_mytool"

    def test_subclass_auto_collect_tools(self):
        """测试子类自动收集工具"""

        class MyToolSet(CommonToolSet):
            my_tool = Tool(
                name="my_tool",
                description="My custom tool",
                func=lambda: "result",
            )

        toolset = MyToolSet()
        tools = toolset.tools()

        assert len(tools) >= 1


class TestToolGetParametersSchema:
    """测试 Tool.get_parameters_schema 方法"""

    def test_get_schema_from_args_schema(self):
        """测试从 args_schema 获取 schema"""

        class TestArgs(BaseModel):
            name: str = Field(description="Name")
            age: int = Field(description="Age")

        tool_obj = Tool(
            name="test",
            description="Test tool",
            args_schema=TestArgs,
            func=lambda name, age: f"{name}: {age}",
        )

        schema = tool_obj.get_parameters_schema()

        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_get_schema_from_parameters(self):
        """测试从 parameters 获取 schema"""
        params = [
            ToolParameter("name", "string", "Name", required=True),
            ToolParameter("age", "integer", "Age"),
        ]

        tool_obj = Tool(
            name="test",
            description="Test tool",
            parameters=params,
            func=lambda name, age=0: f"{name}: {age}",
        )

        schema = tool_obj.get_parameters_schema()

        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert "name" in schema.get("required", [])


class TestSanitizePythonIdentifier:
    """测试字段名 sanitizer"""

    def test_valid_identifier_unchanged(self):
        assert _sanitize_python_identifier("normal_name") == "normal_name"

    def test_hyphenated_name(self):
        assert _sanitize_python_identifier("x-access-id") == "x_access_id"

    def test_dotted_name(self):
        assert _sanitize_python_identifier("a.b.c") == "a_b_c"

    def test_leading_digit_prefixed(self):
        assert _sanitize_python_identifier("123abc") == "field_123abc"

    def test_keyword_suffixed(self):
        assert _sanitize_python_identifier("class") == "class_"

    def test_empty_string(self):
        assert _sanitize_python_identifier("") == "field"

    def test_only_invalid_chars(self):
        assert _sanitize_python_identifier("---") == "field"


class TestJsonSchemaToPydanticInvalidFieldNames:
    """覆盖 _json_schema_to_pydantic 对非法 Python 标识符字段名的处理"""

    def test_hyphenated_field_name_builds_model(self):
        """字段名含 '-' 时不应抛错, 且 JSON Schema 仍以原名暴露"""
        schema = {
            "type": "object",
            "properties": {
                "x-access-id": {"type": "string", "description": "id"},
            },
            "required": ["x-access-id"],
        }

        model = _json_schema_to_pydantic("Args", schema)

        assert model is not None
        assert "x_access_id" in model.model_fields
        json_schema = model.model_json_schema()
        assert "x-access-id" in json_schema["properties"]
        assert "x-access-id" in json_schema["required"]

    def test_keyword_field_name_sanitized(self):
        schema = {
            "type": "object",
            "properties": {
                "class": {"type": "string", "description": "py keyword"},
            },
        }

        model = _json_schema_to_pydantic("Args", schema)

        assert model is not None
        assert "class_" in model.model_fields
        assert "class" in model.model_json_schema()["properties"]

    def test_accepts_both_original_and_sanitized_name(self):
        schema = {
            "type": "object",
            "properties": {
                "x-access-id": {"type": "string"},
            },
            "required": ["x-access-id"],
        }

        model = _json_schema_to_pydantic("Args", schema)

        # 原名: 通过 alias
        m1 = model(**{"x-access-id": "v1"})
        assert m1.model_dump(by_alias=True) == {"x-access-id": "v1"}
        # 沙化名: 通过 populate_by_name
        m2 = model(x_access_id="v2")
        assert m2.model_dump(by_alias=True) == {"x-access-id": "v2"}


class TestCreateFunctionWithSignatureAliasSanitization:
    """覆盖 _create_function_with_signature 对非法 alias 名的防御处理"""

    def test_alias_with_hyphen_sanitized(self):
        """`__agentrun_argument_aliases__` 含非法标识符 alias 时不应崩溃"""
        from pydantic import BaseModel as _BM

        class _Args(_BM):
            query: str

        setattr(_Args, "__agentrun_argument_aliases__", {"x-alias": "query"})

        toolset = MagicMock()
        func = _create_function_with_signature("demo", _Args, toolset, None)

        import inspect as _inspect

        sig = _inspect.signature(func)
        # 主字段保留, alias 被 sanitize
        assert "query" in sig.parameters
        assert "x_alias" in sig.parameters

    def test_call_via_sanitized_alias_name_routes_to_canonical(self):
        """用签名暴露的 sanitized alias 名调用时也应翻译到 canonical 字段

        回归 Copilot review 提出的: 仅 sanitize 签名不够, 还要让
        _normalize_tool_arguments 认识 sanitized alias.
        """
        from pydantic import BaseModel as _BM
        from pydantic import Field as _Field

        class _Args(_BM):
            query: str = _Field()

        setattr(_Args, "__agentrun_argument_aliases__", {"x-alias": "query"})

        toolset = MagicMock()
        toolset.call_tool = MagicMock(return_value={"ok": True})
        func = _create_function_with_signature("demo", _Args, toolset, None)

        # 用沙化后的 alias 名 (签名暴露的形式) 调用
        func(x_alias="hello")

        call_kwargs = toolset.call_tool.call_args.kwargs
        assert call_kwargs["arguments"] == {"query": "hello"}
        # 同时验证: alias_map 已被扩展, 包含 sanitized 形式
        merged_map = getattr(_Args, "__agentrun_argument_aliases__")
        assert merged_map.get("x_alias") == "query"
        assert merged_map.get("x-alias") == "query"


class TestNestedObjectRoundTrip:
    """覆盖嵌套 object 参数经二次转换后仍保持 object 类型"""

    @staticmethod
    def _unwrap_anyof(node: Dict[str, Any]) -> Dict[str, Any]:
        """从 anyOf/oneOf 中挑出非 null 的核心 schema"""
        for key in ("anyOf", "oneOf"):
            options = node.get(key)
            if isinstance(options, list):
                for option in options:
                    if isinstance(option, dict) and option.get("type") != "null":
                        return option
        return node

    @staticmethod
    def _resolve_local_ref(
        schema: Dict[str, Any], node: Dict[str, Any]
    ) -> Dict[str, Any]:
        """解析当前 schema 内的 $defs 引用，便于断言实际结构"""
        ref = node.get("$ref")
        if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
            return node
        ref_name = ref.rsplit("/", 1)[-1]
        return schema.get("$defs", {}).get(ref_name, node)

    def test_nested_optional_object_survives_second_conversion(self):
        """嵌套可选 object 经 LangChain 二次转换后仍应是 object"""
        toolset = MagicMock()

        meta = {
            "name": "create_order",
            "description": "demo",
            "input_schema": {
                "type": "object",
                "properties": {
                    "attrs": {
                        "type": "object",
                        "description": "attributes",
                        "properties": {
                            "risk": {
                                "type": "object",
                                "description": "risk info",
                                "properties": {
                                    "level": {"type": "string"},
                                },
                            },
                            "refund": {
                                "type": "object",
                                "description": "refund info",
                                "properties": {
                                    "amount": {"type": "number"},
                                },
                            },
                        },
                    },
                },
                "required": [],
            },
        }

        tool_obj = _build_tool_from_meta(toolset, meta, None)
        assert tool_obj is not None

        params = tool_obj.get_parameters_schema()

        from langchain_core.utils.function_calling import (
            convert_to_openai_tool,
        )

        from agentrun.integration.langchain.tool_adapter import (
            LangChainToolAdapter,
        )
        from agentrun.integration.utils.canonical import CanonicalTool

        canonical = CanonicalTool(
            name=tool_obj.name,
            description=tool_obj.description,
            parameters=params,
            func=tool_obj.func,
        )
        lc_tool = LangChainToolAdapter().from_canonical([canonical])[0]
        openai_tool = convert_to_openai_tool(lc_tool)

        props = openai_tool["function"]["parameters"]["properties"]
        attrs = self._unwrap_anyof(props["attrs"])
        assert attrs.get("type") == "object"
        assert "properties" in attrs

        risk = self._unwrap_anyof(attrs["properties"]["risk"])
        refund = self._unwrap_anyof(attrs["properties"]["refund"])

        assert risk.get("type") == "object", risk
        assert "level" in risk.get("properties", {})
        assert refund.get("type") == "object", refund
        assert "amount" in refund.get("properties", {})

    def test_anyof_ref_object_survives_second_conversion(self):
        """anyOf/$ref 包装的嵌套 object 经二次转换后仍应是 object"""
        from langchain_core.utils.function_calling import (
            convert_to_openai_tool,
        )

        from agentrun.integration.langchain.tool_adapter import (
            LangChainToolAdapter,
        )
        from agentrun.integration.utils.canonical import CanonicalTool

        parameters = {
            "type": "object",
            "properties": {
                "attrs": {
                    "anyOf": [
                        {"$ref": "#/$defs/Attrs"},
                        {"type": "null"},
                    ],
                    "description": "attributes",
                }
            },
            "$defs": {
                "Attrs": {
                    "type": "object",
                    "properties": {
                        "risk": {
                            "anyOf": [
                                {"$ref": "#/$defs/Risk"},
                                {"type": "null"},
                            ],
                            "description": "risk info",
                        }
                    },
                },
                "Risk": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "string"},
                    },
                },
            },
        }

        canonical = CanonicalTool(
            name="create_order",
            description="demo",
            parameters=parameters,
            func=lambda **kwargs: kwargs,
        )
        lc_tool = LangChainToolAdapter().from_canonical([canonical])[0]
        openai_tool = convert_to_openai_tool(lc_tool)

        attrs = self._unwrap_anyof(
            openai_tool["function"]["parameters"]["properties"]["attrs"]
        )
        risk = self._unwrap_anyof(attrs["properties"]["risk"])

        assert risk.get("type") == "object", risk
        assert "level" in risk.get("properties", {})

    def test_recursive_ref_does_not_crash(self):
        """自引用 $ref 不应导致 Pydantic 模型构建无限递归"""
        toolset = MagicMock()
        meta = {
            "name": "tree_tool",
            "description": "demo",
            "input_schema": {
                "type": "object",
                "properties": {
                    "root": {"$ref": "#/$defs/Node"},
                },
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "child": {"$ref": "#/$defs/Node"},
                        },
                    }
                },
            },
        }

        tool_obj = _build_tool_from_meta(toolset, meta, None)

        assert tool_obj is not None
        schema = tool_obj.args_schema.model_json_schema()
        root = self._resolve_local_ref(
            schema, self._unwrap_anyof(schema["properties"]["root"])
        )
        assert root.get("type") == "object"
        assert "value" in root.get("properties", {})
        child = self._unwrap_anyof(root["properties"]["child"])
        assert child.get("type") == "object"

    def test_mutual_recursive_refs_do_not_crash(self):
        """互递归 $ref 不应导致 Pydantic 模型构建无限递归"""
        toolset = MagicMock()
        meta = {
            "name": "mutual_tool",
            "description": "demo",
            "input_schema": {
                "type": "object",
                "properties": {
                    "root": {"$ref": "#/$defs/A"},
                },
                "$defs": {
                    "A": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "b": {"$ref": "#/$defs/B"},
                        },
                    },
                    "B": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "a": {"$ref": "#/$defs/A"},
                        },
                    },
                },
            },
        }

        tool_obj = _build_tool_from_meta(toolset, meta, None)

        assert tool_obj is not None
        schema = tool_obj.args_schema.model_json_schema()
        root = self._resolve_local_ref(
            schema, self._unwrap_anyof(schema["properties"]["root"])
        )
        assert root.get("type") == "object"
        nested_b = self._resolve_local_ref(
            schema, self._unwrap_anyof(root["properties"]["b"])
        )
        assert nested_b.get("type") == "object"

    def test_recursive_array_items_ref_does_not_crash(self):
        """array items 自引用 $ref 不应导致无限递归"""
        toolset = MagicMock()
        meta = {
            "name": "tree_list_tool",
            "description": "demo",
            "input_schema": {
                "type": "object",
                "properties": {
                    "root": {"$ref": "#/$defs/Node"},
                },
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "children": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/Node"},
                            },
                        },
                    }
                },
            },
        }

        tool_obj = _build_tool_from_meta(toolset, meta, None)

        assert tool_obj is not None
        schema = tool_obj.args_schema.model_json_schema()
        root = self._resolve_local_ref(
            schema, self._unwrap_anyof(schema["properties"]["root"])
        )
        children = self._unwrap_anyof(root["properties"]["children"])
        assert children.get("type") == "array"
        assert children["items"].get("type") == "object"

    def test_allof_recursive_refs_do_not_crash(self):
        """allOf 多 $ref 合并时递归引用不应导致无限递归"""
        toolset = MagicMock()
        meta = {
            "name": "allof_tool",
            "description": "demo",
            "input_schema": {
                "type": "object",
                "properties": {
                    "root": {
                        "allOf": [
                            {"$ref": "#/$defs/A"},
                            {"$ref": "#/$defs/B"},
                        ]
                    },
                },
                "$defs": {
                    "A": {
                        "type": "object",
                        "properties": {
                            "a_name": {"type": "string"},
                            "b": {"$ref": "#/$defs/B"},
                        },
                    },
                    "B": {
                        "type": "object",
                        "properties": {
                            "b_name": {"type": "string"},
                            "a": {"$ref": "#/$defs/A"},
                        },
                    },
                },
            },
        }

        tool_obj = _build_tool_from_meta(toolset, meta, None)

        assert tool_obj is not None
        schema = tool_obj.args_schema.model_json_schema()
        root = self._resolve_local_ref(
            schema, self._unwrap_anyof(schema["properties"]["root"])
        )
        assert root.get("type") == "object"
        assert "a_name" in root.get("properties", {})
        assert "b_name" in root.get("properties", {})

    def test_shared_refs_are_not_reexpanded_exponentially(self):
        """非环共享 $ref 不应沿兄弟路径重复展开成指数级模型"""
        depth = 6
        defs = {}
        for index in range(depth):
            defs[f"N{index}"] = {
                "type": "object",
                "properties": {
                    "left": {"$ref": f"#/$defs/N{index + 1}"},
                    "right": {"$ref": f"#/$defs/N{index + 1}"},
                },
            }
        defs[f"N{depth}"] = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        }
        meta = {
            "name": "shared_ref_tool",
            "description": "demo",
            "input_schema": {
                "type": "object",
                "properties": {"root": {"$ref": "#/$defs/N0"}},
                "$defs": defs,
            },
        }

        tool_obj = _build_tool_from_meta(MagicMock(), meta, None)

        assert tool_obj is not None
        schema = tool_obj.args_schema.model_json_schema()
        assert len(schema.get("$defs", {})) <= depth + 1

    def test_shared_refs_with_recursive_child_stay_linear(
        self, monkeypatch
    ):
        """共享 DAG 即使包含递归子节点也不应指数级重复展开"""
        import importlib

        tool_mod = importlib.import_module("agentrun.integration.utils.tool")
        depth = 8
        defs = {
            "Meta": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "next": {"$ref": "#/$defs/Meta"},
                },
            }
        }
        for index in range(depth):
            defs[f"N{index}"] = {
                "type": "object",
                "properties": {
                    "left": {"$ref": f"#/$defs/N{index + 1}"},
                    "right": {"$ref": f"#/$defs/N{index + 1}"},
                    "meta": {"$ref": "#/$defs/Meta"},
                },
            }
        defs[f"N{depth}"] = {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "meta": {"$ref": "#/$defs/Meta"},
            },
        }
        meta = {
            "name": "shared_recursive_tool",
            "description": "demo",
            "input_schema": {
                "type": "object",
                "properties": {"root": {"$ref": "#/$defs/N0"}},
                "$defs": defs,
            },
        }
        calls = {"count": 0}
        original = tool_mod._json_type_to_python

        def wrapped_json_type_to_python(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            tool_mod, "_json_type_to_python", wrapped_json_type_to_python
        )

        tool_obj = _build_tool_from_meta(MagicMock(), meta, None)

        assert tool_obj is not None
        tool_obj.args_schema.model_json_schema()
        assert calls["count"] <= depth * 8

    def test_shared_refs_with_different_parent_paths_stay_linear(
        self, monkeypatch
    ):
        """不同父路径收敛到同一 ref 时不应因无关 active_refs 失去缓存"""
        import importlib

        tool_mod = importlib.import_module("agentrun.integration.utils.tool")
        depth = 8
        defs = {}
        for index in range(depth):
            next_left = {"$ref": f"#/$defs/A{index + 1}"}
            next_right = {"$ref": f"#/$defs/B{index + 1}"}
            defs[f"A{index}"] = {
                "type": "object",
                "properties": {
                    "left": next_left,
                    "right": next_right,
                },
            }
            defs[f"B{index}"] = {
                "type": "object",
                "properties": {
                    "left": next_left,
                    "right": next_right,
                },
            }
        defs[f"A{depth}"] = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        }
        defs[f"B{depth}"] = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        }
        meta = {
            "name": "lattice_tool",
            "description": "demo",
            "input_schema": {
                "type": "object",
                "properties": {"root": {"$ref": "#/$defs/A0"}},
                "$defs": defs,
            },
        }
        calls = {"count": 0}
        original = tool_mod._json_type_to_python

        def wrapped_json_type_to_python(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            tool_mod, "_json_type_to_python", wrapped_json_type_to_python
        )

        tool_obj = _build_tool_from_meta(MagicMock(), meta, None)

        assert tool_obj is not None
        tool_obj.args_schema.model_json_schema()
        assert calls["count"] <= depth * 6

    def test_ref_with_sibling_properties_does_not_poison_cache(self):
        """带 sibling properties 的 $ref 不应污染同一 ref 的裸引用缓存"""
        meta = {
            "name": "context_ref_tool",
            "description": "demo",
            "input_schema": {
                "type": "object",
                "properties": {
                    "extended": {
                        "$ref": "#/$defs/Base",
                        "properties": {"extra": {"type": "string"}},
                    },
                    "plain": {"$ref": "#/$defs/Base"},
                },
                "$defs": {
                    "Base": {
                        "type": "object",
                        "properties": {"base": {"type": "string"}},
                    }
                },
            },
        }

        tool_obj = _build_tool_from_meta(MagicMock(), meta, None)

        assert tool_obj is not None
        schema = tool_obj.args_schema.model_json_schema()
        extended = self._resolve_local_ref(
            schema, self._unwrap_anyof(schema["properties"]["extended"])
        )
        plain = self._resolve_local_ref(
            schema, self._unwrap_anyof(schema["properties"]["plain"])
        )
        assert "extra" in extended.get("properties", {})
        assert "extra" not in plain.get("properties", {})

    def test_cycle_broken_ref_result_does_not_poison_cache(self):
        """递归截断后的 ref 结果不应污染后续环外同 ref 引用"""
        meta = {
            "name": "mutual_order_tool",
            "description": "demo",
            "input_schema": {
                "type": "object",
                "properties": {
                    "root": {"$ref": "#/$defs/A"},
                    "standalone_b": {"$ref": "#/$defs/B"},
                },
                "$defs": {
                    "A": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "b": {"$ref": "#/$defs/B"},
                        },
                    },
                    "B": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "a": {"$ref": "#/$defs/A"},
                        },
                    },
                },
            },
        }

        tool_obj = _build_tool_from_meta(MagicMock(), meta, None)

        assert tool_obj is not None
        schema = tool_obj.args_schema.model_json_schema()
        standalone_b = self._resolve_local_ref(
            schema, self._unwrap_anyof(schema["properties"]["standalone_b"])
        )
        nested_a = self._resolve_local_ref(
            schema, self._unwrap_anyof(standalone_b["properties"]["a"])
        )
        assert "name" in nested_a.get("properties", {})

    def test_reachable_refs_for_cycle_are_complete(self):
        """互递归 ref 的可达集合不应因计算顺序被低估"""
        schema = {
            "type": "object",
            "$defs": {
                "A": {
                    "type": "object",
                    "properties": {"b": {"$ref": "#/$defs/B"}},
                },
                "B": {
                    "type": "object",
                    "properties": {"c": {"$ref": "#/$defs/C"}},
                },
                "C": {
                    "type": "object",
                    "properties": {"a": {"$ref": "#/$defs/A"}},
                },
            },
        }
        expected_refs = {"#/$defs/A", "#/$defs/B", "#/$defs/C"}
        ref_cache: Dict[str, set] = {}

        assert (
            _reachable_refs_for_ref("#/$defs/A", schema, ref_cache)
            == expected_refs
        )
        assert (
            _reachable_refs_for_ref("#/$defs/B", schema, ref_cache)
            == expected_refs
        )
        assert (
            _reachable_refs_for_ref("#/$defs/C", schema, ref_cache)
            == expected_refs
        )

    def test_reachable_refs_cache_intermediate_chain_refs(self):
        """可达集合缓存应覆盖中间 ref, 避免深链重复遍历"""
        depth = 6
        defs = {}
        for index in range(depth):
            defs[f"N{index}"] = {
                "type": "object",
                "properties": {
                    "next": {"$ref": f"#/$defs/N{index + 1}"}
                },
            }
        defs[f"N{depth}"] = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        }
        schema = {"type": "object", "$defs": defs}
        ref_cache: Dict[str, set] = {}

        reachable_refs = _reachable_refs_for_ref(
            "#/$defs/N0", schema, ref_cache
        )

        assert reachable_refs == {
            f"#/$defs/N{index}" for index in range(1, depth + 1)
        }
        assert set(ref_cache) == {
            f"#/$defs/N{index}" for index in range(depth + 1)
        }
        assert ref_cache["#/$defs/N3"] == {
            f"#/$defs/N{index}" for index in range(4, depth + 1)
        }

    def test_union_cycle_guard_follows_selected_branch(self):
        """防环 ref 应与 anyOf 实际展开分支一致"""
        meta = {
            "name": "selected_union_tool",
            "description": "demo",
            "input_schema": {
                "type": "object",
                "properties": {
                    "payload": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "nested": {"$ref": "#/$defs/Nested"}
                                },
                            },
                            {"$ref": "#/$defs/Nested"},
                        ]
                    }
                },
                "$defs": {
                    "Nested": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    }
                },
            },
        }

        tool_obj = _build_tool_from_meta(MagicMock(), meta, None)

        assert tool_obj is not None
        schema = tool_obj.args_schema.model_json_schema()
        payload = self._resolve_local_ref(
            schema, self._unwrap_anyof(schema["properties"]["payload"])
        )
        nested = self._resolve_local_ref(
            schema, self._unwrap_anyof(payload["properties"]["nested"])
        )
        assert "value" in nested.get("properties", {})


class TestBuildToolFromMetaInvalidFieldNames:
    """覆盖 _build_tool_from_meta 完整链路 (回归 'x-access-id' 加载失败)"""

    def test_mcp_input_schema_with_hyphen_field(self):
        """模拟 MCP 工具元数据包含 'x-access-id' 入参时仍可成功构造 Tool"""
        toolset = MagicMock()
        toolset.call_tool = MagicMock(return_value={"status": "ok"})

        meta = {
            "name": "demo-tool",
            "description": "demo",
            "input_schema": {
                "type": "object",
                "properties": {
                    "x-access-id": {
                        "type": "string",
                        "description": "id",
                    },
                    "value": {"type": "integer"},
                },
                "required": ["x-access-id"],
            },
        }

        tool_obj = _build_tool_from_meta(toolset, meta, None)

        assert tool_obj is not None
        # 调用工具时, MCP 应收到原始字段名 'x-access-id'
        tool_obj.func(**{"x-access-id": "abc", "value": 1})
        toolset.call_tool.assert_called_once()
        call_kwargs = toolset.call_tool.call_args.kwargs
        assert call_kwargs["arguments"]["x-access-id"] == "abc"
        assert call_kwargs["arguments"]["value"] == 1
