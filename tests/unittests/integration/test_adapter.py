"""测试适配器模块 / Test Adapter Module

测试 schema_to_python_type 函数和其他适配器工具。
"""

from typing import Any, get_args, get_origin, List, Optional, Union

from pydantic import BaseModel
import pytest

from agentrun.integration.utils.adapter import (
    _generate_params_docstring,
    _resolve_schema_ref,
    _schema_to_type_description,
    schema_to_python_type,
)
from agentrun.integration.utils.canonical import CanonicalTool


class TestResolveSchemaRef:
    """测试 _resolve_schema_ref 函数"""

    def test_resolve_simple_ref(self):
        """测试解析简单的 $defs 引用"""
        root_schema = {
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            }
        }

        result = _resolve_schema_ref("#/$defs/User", root_schema)
        assert result == {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

    def test_resolve_definitions_ref(self):
        """测试解析 definitions 引用（旧格式）"""
        root_schema = {
            "definitions": {
                "Address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }
            }
        }

        result = _resolve_schema_ref("#/definitions/Address", root_schema)
        assert result == {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        }

    def test_resolve_invalid_ref(self):
        """测试无效的引用返回 None"""
        root_schema = {"$defs": {"User": {"type": "object"}}}

        # 不存在的引用
        assert _resolve_schema_ref("#/$defs/NotExist", root_schema) is None

        # 非 # 开头的引用
        assert (
            _resolve_schema_ref("http://example.com/schema", root_schema)
            is None
        )

        # 空引用
        assert _resolve_schema_ref("", root_schema) is None
        assert _resolve_schema_ref(None, root_schema) is None  # type: ignore


class TestSchemaToPythonType:
    """测试 schema_to_python_type 函数"""

    def test_basic_types(self):
        """测试基本类型转换"""
        assert schema_to_python_type({"type": "string"}) is str
        assert schema_to_python_type({"type": "integer"}) is int
        assert schema_to_python_type({"type": "number"}) is float
        assert schema_to_python_type({"type": "boolean"}) is bool
        assert schema_to_python_type({"type": "null"}) is type(None)

    def test_object_type(self):
        """测试对象类型转换"""
        # 无属性的对象返回 dict
        assert schema_to_python_type({"type": "object"}) is dict

        # 有属性的对象返回动态创建的 Pydantic 模型
        result = schema_to_python_type(
            {"type": "object", "properties": {"name": {"type": "string"}}}
        )
        # 验证是 Pydantic 模型
        assert isinstance(result, type)
        assert issubclass(result, BaseModel)
        assert "name" in result.model_fields

    def test_array_type_without_items(self):
        """测试无 items 的数组类型"""
        result = schema_to_python_type({"type": "array"})
        assert result == List[Any]

    def test_array_type_with_simple_items(self):
        """测试有简单 items 的数组类型"""
        result = schema_to_python_type(
            {"type": "array", "items": {"type": "string"}}
        )
        assert result == List[str]

        result = schema_to_python_type(
            {"type": "array", "items": {"type": "integer"}}
        )
        assert result == List[int]

    def test_array_type_with_object_items(self):
        """测试有对象 items 的数组类型"""
        result = schema_to_python_type(
            {"type": "array", "items": {"type": "object"}}
        )
        assert result == List[dict]

    def test_array_type_with_ref_items(self):
        """测试有 $ref items 的数组类型 - 需要 root_schema"""
        schema = {"type": "array", "items": {"$ref": "#/$defs/OrderItem"}}
        root_schema = {
            "$defs": {
                "OrderItem": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "quantity": {"type": "integer"},
                    },
                }
            },
            "type": "array",
            "items": {"$ref": "#/$defs/OrderItem"},
        }

        # 不提供 root_schema 时，$ref 解析失败，返回 List[dict]
        result = schema_to_python_type(schema)
        assert get_origin(result) is list
        assert get_args(result)[0] is dict

        # 提供 root_schema 时，$ref 正确解析为 Pydantic 模型
        result = schema_to_python_type(schema, root_schema)
        assert get_origin(result) is list
        item_type = get_args(result)[0]
        # 验证是 Pydantic 模型并包含正确的字段
        assert issubclass(item_type, BaseModel)
        assert "name" in item_type.model_fields
        assert "quantity" in item_type.model_fields

    def test_anyof_with_null(self):
        """测试 anyOf 包含 null 的情况 -> Optional"""
        schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        result = schema_to_python_type(schema)
        assert result == Optional[str]

    def test_anyof_multiple_types(self):
        """测试 anyOf 多个类型 -> Union"""
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        result = schema_to_python_type(schema)
        assert result == Union[str, int]

    def test_oneof_multiple_types(self):
        """测试 oneOf 多个类型 -> Union"""
        schema = {"oneOf": [{"type": "boolean"}, {"type": "number"}]}
        result = schema_to_python_type(schema)
        assert result == Union[bool, float]

    def test_ref_resolution(self):
        """测试 $ref 引用解析"""
        root_schema = {
            "$defs": {"User": {"type": "object"}},
            "type": "object",
            "properties": {"user": {"$ref": "#/$defs/User"}},
        }

        user_schema = {"$ref": "#/$defs/User"}
        result = schema_to_python_type(user_schema, root_schema)
        # User 无属性，返回 dict
        assert result is dict

    def test_nested_ref_in_array(self):
        """测试数组中嵌套的 $ref 引用"""
        root_schema = {
            "$defs": {
                "OrderItem": {
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "price": {"type": "number"},
                    },
                    "required": ["product_name", "quantity", "price"],
                }
            },
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/OrderItem"},
                    "description": "订单商品列表",
                },
                "customer_name": {"type": "string"},
            },
            "required": ["items", "customer_name"],
        }

        # 测试 items 字段 - 应该返回 List[PydanticModel]
        items_schema = root_schema["properties"]["items"]
        result = schema_to_python_type(items_schema, root_schema)
        assert get_origin(result) is list
        item_type = get_args(result)[0]
        # 验证是 Pydantic 模型并包含正确的字段
        assert issubclass(item_type, BaseModel)
        assert set(item_type.model_fields.keys()) == {
            "product_name",
            "quantity",
            "price",
        }

        # 测试 customer_name 字段
        customer_schema = root_schema["properties"]["customer_name"]
        result = schema_to_python_type(customer_schema, root_schema)
        assert result is str

    def test_empty_or_invalid_schema(self):
        """测试空或无效的 schema"""
        assert schema_to_python_type(None) == Any  # type: ignore
        assert schema_to_python_type({}) == Any
        assert schema_to_python_type({"foo": "bar"}) == Any

    def test_unknown_type(self):
        """测试未知类型"""
        assert schema_to_python_type({"type": "unknown"}) == Any

    def test_pydantic_model_json_schema_format(self):
        """测试 Pydantic model_json_schema() 生成的格式

        Pydantic 使用 $defs 存储嵌套模型定义，并使用 $ref 引用。
        """
        # 模拟 Pydantic 生成的 JSON Schema
        pydantic_schema = {
            "$defs": {
                "Nestedobject": {
                    "properties": {
                        "product_name": {
                            "description": "",
                            "title": "Product Name",
                            "type": "string",
                        },
                        "quantity": {
                            "description": "",
                            "title": "Quantity",
                            "type": "integer",
                        },
                        "price": {
                            "description": "",
                            "title": "Price",
                            "type": "number",
                        },
                    },
                    "required": ["product_name", "quantity", "price"],
                    "title": "Nestedobject",
                    "type": "object",
                }
            },
            "properties": {
                "items": {
                    "description": "订单商品列表",
                    "items": {"$ref": "#/$defs/Nestedobject"},
                    "title": "Items",
                    "type": "array",
                },
                "customer_name": {
                    "description": "",
                    "title": "Customer Name",
                    "type": "string",
                },
            },
            "required": ["items", "customer_name"],
            "title": "CreateOrderApiCoffeeOrdersPostArgs",
            "type": "object",
        }

        # 测试 items 字段 - 应该返回 List[PydanticModel]
        items_schema = pydantic_schema["properties"]["items"]
        result = schema_to_python_type(items_schema, pydantic_schema)
        assert get_origin(result) is list
        item_type = get_args(result)[0]
        # 验证是 Pydantic 模型并包含正确的字段
        assert issubclass(item_type, BaseModel)
        assert set(item_type.model_fields.keys()) == {
            "product_name",
            "quantity",
            "price",
        }

        # 测试 customer_name 字段
        customer_schema = pydantic_schema["properties"]["customer_name"]
        result = schema_to_python_type(customer_schema, pydantic_schema)
        assert result is str


class TestGoogleADKToolAdapter:
    """测试 Google ADK 工具适配器"""

    @pytest.mark.skipif(
        not pytest.importorskip(
            "google.adk", reason="google-adk not installed"
        ),
        reason="google-adk not installed",
    )
    def test_nested_object_in_array_no_ref(self):
        """测试嵌套对象在数组中时，生成的 schema 不包含 $ref"""
        from agentrun.integration.google_adk.tool_adapter import (
            GoogleADKToolAdapter,
        )

        # 创建一个带嵌套对象的 parameters schema
        parameters_schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/OrderItem"},
                    "description": "订单列表",
                },
                "customer_name": {"type": "string", "description": "客户姓名"},
            },
            "$defs": {
                "OrderItem": {
                    "type": "object",
                    "properties": {
                        "product_name": {
                            "type": "string",
                            "description": "商品名称",
                        },
                        "quantity": {"type": "integer", "description": "数量"},
                        "price": {"type": "number", "description": "单价"},
                    },
                    "required": ["product_name", "quantity", "price"],
                }
            },
            "required": ["items", "customer_name"],
        }

        canonical_tool = CanonicalTool(
            name="create_order",
            description="创建订单",
            parameters=parameters_schema,
            func=lambda **kwargs: "OK",
        )

        adapter = GoogleADKToolAdapter()
        tools = adapter.from_canonical([canonical_tool])

        assert len(tools) == 1
        tool = tools[0]

        # 获取声明并验证
        decl = tool._get_declaration()
        assert decl.name == "create_order"
        assert decl.description == "创建订单"

        # 验证 items 属性
        items_schema = decl.parameters.properties["items"]
        assert items_schema.type.name == "ARRAY"
        assert items_schema.description == "订单列表"

        # 验证 items 内部的对象属性（不应包含 $ref）
        item_props = items_schema.items.properties
        assert "product_name" in item_props
        assert "quantity" in item_props
        assert "price" in item_props

        # 验证内部属性类型
        assert item_props["product_name"].type.name == "STRING"
        assert item_props["quantity"].type.name == "INTEGER"
        assert item_props["price"].type.name == "NUMBER"

    @pytest.mark.skipif(
        not pytest.importorskip(
            "google.adk", reason="google-adk not installed"
        ),
        reason="google-adk not installed",
    )
    def test_deeply_nested_objects(self):
        """测试多层嵌套对象"""
        from agentrun.integration.google_adk.tool_adapter import (
            GoogleADKToolAdapter,
        )

        parameters_schema = {
            "type": "object",
            "properties": {
                "orders": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "options": {
                                            "type": "object",
                                            "properties": {
                                                "size": {"type": "string"},
                                                "sugar": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                            "customer": {"type": "string"},
                        },
                    },
                },
            },
        }

        canonical_tool = CanonicalTool(
            name="batch_orders",
            description="批量创建订单",
            parameters=parameters_schema,
            func=lambda **kwargs: "OK",
        )

        adapter = GoogleADKToolAdapter()
        tools = adapter.from_canonical([canonical_tool])

        assert len(tools) == 1
        decl = tools[0]._get_declaration()

        # 验证多层嵌套结构
        orders_schema = decl.parameters.properties["orders"]
        assert orders_schema.type.name == "ARRAY"

        order_item = orders_schema.items
        assert order_item.type.name == "OBJECT"
        assert "items" in order_item.properties
        assert "customer" in order_item.properties

        inner_items = order_item.properties["items"]
        assert inner_items.type.name == "ARRAY"

        inner_item = inner_items.items
        assert "name" in inner_item.properties
        assert "options" in inner_item.properties

        options = inner_item.properties["options"]
        assert options.type.name == "OBJECT"
        assert "size" in options.properties
        assert "sugar" in options.properties


class TestTypeDocstringGeneration:
    """测试类型文档字符串生成功能"""

    def test_schema_to_type_description_basic_types(self):
        """测试基本类型的类型描述"""
        assert _schema_to_type_description({"type": "string"}) == "string"
        assert _schema_to_type_description({"type": "integer"}) == "integer"
        assert _schema_to_type_description({"type": "number"}) == "number"
        assert _schema_to_type_description({"type": "boolean"}) == "boolean"

    def test_schema_to_type_description_enum(self):
        """测试枚举类型的类型描述"""
        result = _schema_to_type_description(
            {"type": "string", "enum": ["small", "medium", "large"]}
        )
        assert "small" in result
        assert "medium" in result
        assert "large" in result

    def test_schema_to_type_description_array(self):
        """测试数组类型的类型描述"""
        result = _schema_to_type_description(
            {"type": "array", "items": {"type": "string"}}
        )
        assert result == "array[string]"

    def test_schema_to_type_description_object(self):
        """测试对象类型的类型描述"""
        result = _schema_to_type_description({
            "type": "object",
            "title": "User",
            "properties": {
                "name": {"type": "string", "description": "用户名"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        })
        assert "User" in result
        assert "name:" in result
        assert "age?" in result  # optional marker
        assert "用户名" in result  # description

    def test_schema_to_type_description_ref(self):
        """测试 $ref 引用的类型描述"""
        root_schema = {
            "$defs": {
                "Item": {
                    "type": "object",
                    "title": "Item",
                    "properties": {"name": {"type": "string"}},
                }
            }
        }
        schema = {"$ref": "#/$defs/Item"}
        result = _schema_to_type_description(schema, root_schema)
        assert "Item" in result
        assert "name" in result

    def test_generate_params_docstring_simple(self):
        """测试简单参数的文档字符串生成"""
        params = {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
            },
            "required": ["city"],
        }
        result = _generate_params_docstring(params)
        assert "Args:" in result
        assert "city" in result
        assert "string" in result
        assert "城市名称" in result

    def test_generate_params_docstring_complex(self):
        """测试复杂参数的文档字符串生成"""
        params = {
            "type": "object",
            "properties": {
                "order": {
                    "type": "object",
                    "title": "Order",
                    "description": "订单信息",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "title": "OrderItem",
                                "properties": {
                                    "name": {"type": "string"},
                                    "quantity": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
            },
            "required": ["order"],
        }
        result = _generate_params_docstring(params)
        assert "Args:" in result
        assert "order" in result
        assert "Order" in result
        assert "订单信息" in result

    def test_function_tools_without_type_docstring(self):
        """测试不包含类型文档的函数工具"""
        from agentrun.integration.google_adk.tool_adapter import (
            GoogleADKToolAdapter,
        )

        tool = CanonicalTool(
            name="test_tool",
            description="测试工具",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
            },
            func=lambda **kwargs: "OK",
        )

        adapter = GoogleADKToolAdapter()
        # 默认不包含类型文档
        func_tools = adapter.function_tools([tool])

        assert len(func_tools) == 1
        assert func_tools[0].__doc__ == "测试工具"

    def test_function_tools_with_type_docstring(self):
        """测试包含类型文档的函数工具"""
        from agentrun.integration.google_adk.tool_adapter import (
            GoogleADKToolAdapter,
        )

        tool = CanonicalTool(
            name="test_tool",
            description="测试工具",
            parameters={
                "type": "object",
                "properties": {
                    "order": {
                        "type": "object",
                        "title": "Order",
                        "properties": {
                            "name": {"type": "string"},
                        },
                    },
                },
                "required": ["order"],
            },
            func=lambda **kwargs: "OK",
        )

        adapter = GoogleADKToolAdapter()
        # 启用类型文档
        func_tools = adapter.function_tools([tool], include_type_docstring=True)

        assert len(func_tools) == 1
        doc = func_tools[0].__doc__
        assert "测试工具" in doc
        assert "Args:" in doc
        assert "order" in doc
        assert "Order" in doc
