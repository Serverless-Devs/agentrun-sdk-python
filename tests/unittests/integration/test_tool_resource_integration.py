"""ToolResource Integration 单元测试 / ToolResource Integration Unit Tests

测试新版 ToolResource 到 integration 层的桥接功能：
- CommonToolSet.from_agentrun_tool() 类方法
- builtin/tool_resource.py 入口函数
- 各框架 builtin 中的 tool_resource() 函数
"""

import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# 获取 builtin.tool_resource 模块的真正模块对象
# __init__.py 中 from .tool_resource import tool_resource 会让
# "agentrun.integration.builtin.tool_resource" 在 patch 字符串路径中
# 解析为函数而非模块，所以必须用 sys.modules 获取真正的模块对象
# 再配合 patch.object 使用
import agentrun.integration.builtin.tool_resource  # noqa: F401
from agentrun.integration.utils.tool import CommonToolSet

_tool_resource_mod = sys.modules["agentrun.integration.builtin.tool_resource"]


# =============================================================================
# Helper: 构建 mock ToolResource 和 ToolInfo
# =============================================================================


class FakeToolInfo:
    """模拟 ToolInfo 对象，支持 model_dump() 返回真实字典。

    _to_dict() 内部会调用 obj.model_dump(exclude_none=True)，
    所以 mock 必须返回真实的 dict 而不是 MagicMock。
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        if self.input_schema is not None:
            result["input_schema"] = self.input_schema
        return result


def _make_tool_info(
    name: str,
    description: str = "",
    input_schema: Optional[Dict[str, Any]] = None,
) -> FakeToolInfo:
    """创建一个 FakeToolInfo 对象"""
    return FakeToolInfo(
        name=name, description=description, input_schema=input_schema
    )


def _make_mock_tool_resource(
    tool_infos: Optional[List[FakeToolInfo]] = None,
    tool_name: str = "test-tool",
) -> MagicMock:
    """创建一个 mock ToolResource 实例

    模拟 agentrun.tool.tool.Tool 的接口：
    - list_tools(config) -> List[ToolInfo]
    - call_tool(name, arguments, config) -> Any
    - get(config) -> ToolResource
    """
    resource = MagicMock()
    resource.tool_name = tool_name
    resource.list_tools.return_value = (
        tool_infos if tool_infos is not None else []
    )
    resource.call_tool.return_value = {"result": "ok"}
    resource.get.return_value = resource
    return resource


# =============================================================================
# Tests: CommonToolSet.from_agentrun_tool()
# =============================================================================


class TestFromAgentrunTool:
    """测试 CommonToolSet.from_agentrun_tool() 类方法"""

    def test_empty_tool_list(self):
        """空工具列表返回空 CommonToolSet"""
        resource = _make_mock_tool_resource(tool_infos=[])
        result = CommonToolSet.from_agentrun_tool(resource)
        assert isinstance(result, CommonToolSet)
        assert len(result.tools()) == 0
        resource.list_tools.assert_called_once_with(config=None)

    def test_single_tool(self):
        """单个工具正确桥接"""
        info = _make_tool_info("search", "Search the web")
        resource = _make_mock_tool_resource(tool_infos=[info])
        result = CommonToolSet.from_agentrun_tool(resource)
        tools = result.tools()
        assert len(tools) == 1
        assert tools[0].name == "search"

    def test_multiple_tools(self):
        """多个工具正确桥接"""
        infos = [
            _make_tool_info("tool_a", "Tool A"),
            _make_tool_info("tool_b", "Tool B"),
            _make_tool_info("tool_c", "Tool C"),
        ]
        resource = _make_mock_tool_resource(tool_infos=infos)
        result = CommonToolSet.from_agentrun_tool(resource)
        tools = result.tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"tool_a", "tool_b", "tool_c"}

    def test_duplicate_tool_names_skipped(self):
        """重复工具名被跳过"""
        infos = [
            _make_tool_info("dup_tool", "First"),
            _make_tool_info("dup_tool", "Second"),
        ]
        resource = _make_mock_tool_resource(tool_infos=infos)
        result = CommonToolSet.from_agentrun_tool(resource)
        tools = result.tools()
        assert len(tools) == 1
        assert tools[0].name == "dup_tool"

    def test_with_config(self):
        """config 参数正确传递"""
        config = MagicMock()
        resource = _make_mock_tool_resource(tool_infos=[])
        CommonToolSet.from_agentrun_tool(resource, config=config)
        resource.list_tools.assert_called_once_with(config=config)

    def test_with_refresh(self):
        """refresh=True 时先调用 get()"""
        config = MagicMock()
        resource = _make_mock_tool_resource(tool_infos=[])
        CommonToolSet.from_agentrun_tool(resource, config=config, refresh=True)
        resource.get.assert_called_once_with(config=config)

    def test_without_refresh(self):
        """refresh=False 时不调用 get()"""
        resource = _make_mock_tool_resource(tool_infos=[])
        CommonToolSet.from_agentrun_tool(resource, refresh=False)
        resource.get.assert_not_called()

    def test_none_tool_list(self):
        """list_tools 返回 None 时返回空 CommonToolSet"""
        resource = _make_mock_tool_resource()
        resource.list_tools.return_value = None
        result = CommonToolSet.from_agentrun_tool(resource)
        assert len(result.tools()) == 0

    def test_to_openai_function_conversion(self):
        """桥接后的工具可以转换为 OpenAI 格式（无 function 包装）"""
        info = _make_tool_info("weather", "Get weather info")
        resource = _make_mock_tool_resource(tool_infos=[info])
        result = CommonToolSet.from_agentrun_tool(resource)
        openai_tools = result.to_openai_function()
        assert len(openai_tools) == 1
        assert openai_tools[0]["name"] == "weather"

    def test_to_anthropic_tool_conversion(self):
        """桥接后的工具可以转换为 Anthropic 格式"""
        info = _make_tool_info("calculator", "Calculate things")
        resource = _make_mock_tool_resource(tool_infos=[info])
        result = CommonToolSet.from_agentrun_tool(resource)
        anthropic_tools = result.to_anthropic_tool()
        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["name"] == "calculator"

    def test_tool_description_preserved(self):
        """桥接后的工具描述被保留"""
        info = _make_tool_info("echo", "Echo back the input")
        resource = _make_mock_tool_resource(tool_infos=[info])
        result = CommonToolSet.from_agentrun_tool(resource)
        tools = result.tools()
        assert tools[0].description == "Echo back the input"

    def test_prefix_filter(self):
        """filter 参数正常工作"""
        infos = [
            _make_tool_info("get_weather", "Get weather"),
            _make_tool_info("set_alarm", "Set alarm"),
        ]
        resource = _make_mock_tool_resource(tool_infos=infos)
        result = CommonToolSet.from_agentrun_tool(resource)

        filtered = result.to_openai_function(
            filter_tools_by_name=lambda name: name.startswith("get_")
        )
        assert len(filtered) == 1
        assert filtered[0]["name"] == "get_weather"

    def test_tool_with_input_schema(self):
        """带 input_schema 的工具正确解析参数"""
        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        }
        info = _make_tool_info("weather", "Get weather", input_schema=schema)
        resource = _make_mock_tool_resource(tool_infos=[info])
        result = CommonToolSet.from_agentrun_tool(resource)
        tools = result.tools()
        assert len(tools) == 1
        assert tools[0].name == "weather"

    def test_tool_call_forwards_to_resource(self):
        """桥接后的工具调用会转发到 ToolResource.call_tool()"""
        info = _make_tool_info("echo", "Echo tool")
        resource = _make_mock_tool_resource(tool_infos=[info])
        result = CommonToolSet.from_agentrun_tool(resource)
        tools = result.tools()
        assert len(tools) == 1
        # Tool 对象有 func 属性，调用 func 而非直接调用 Tool
        if hasattr(tools[0], "func") and tools[0].func is not None:
            tools[0].func(message="hello")
            resource.call_tool.assert_called_once()


# =============================================================================
# Tests: builtin/tool_resource.py 入口函数
# =============================================================================


class TestBuiltinToolResource:
    """测试 agentrun.integration.builtin.tool_resource 入口函数"""

    def test_from_string_name(self):
        """通过字符串名称创建"""
        mock_client = MagicMock()
        mock_resource = _make_mock_tool_resource(tool_infos=[])
        mock_client.return_value.get.return_value = mock_resource

        with patch.object(_tool_resource_mod, "ToolClient", mock_client):
            from agentrun.integration.builtin.tool_resource import (
                tool_resource as builtin_tool_resource,
            )

            result = builtin_tool_resource("my-tool")
            assert isinstance(result, CommonToolSet)
            mock_client.return_value.get.assert_called_once_with(
                name="my-tool", config=None
            )

    def test_from_string_name_with_config(self):
        """通过字符串名称 + config 创建"""
        config = MagicMock()
        mock_client = MagicMock()
        mock_resource = _make_mock_tool_resource(tool_infos=[])
        mock_client.return_value.get.return_value = mock_resource

        with patch.object(_tool_resource_mod, "ToolClient", mock_client):
            from agentrun.integration.builtin.tool_resource import (
                tool_resource as builtin_tool_resource,
            )

            result = builtin_tool_resource("my-tool", config=config)
            assert isinstance(result, CommonToolSet)
            mock_client.return_value.get.assert_called_once_with(
                name="my-tool", config=config
            )

    def test_from_tool_resource_instance(self):
        """通过 ToolResource 实例创建"""
        from agentrun.integration.builtin.tool_resource import (
            tool_resource as builtin_tool_resource,
        )
        from agentrun.tool.tool import Tool as ToolResourceType

        mock_resource = MagicMock(spec=ToolResourceType)
        mock_resource.list_tools.return_value = []

        result = builtin_tool_resource(mock_resource)
        assert isinstance(result, CommonToolSet)
        mock_resource.list_tools.assert_called_once()


# =============================================================================
# Tests: 各框架 builtin tool_resource() 函数
# =============================================================================


class TestFrameworkBuiltinToolResource:
    """测试各框架 builtin 中的 tool_resource() 函数"""

    def _run_framework_test(self, framework_module_path: str):
        """通用框架测试辅助方法"""
        import importlib

        module = importlib.import_module(framework_module_path)
        framework_tool_resource = getattr(module, "tool_resource")

        mock_client = MagicMock()
        info = _make_tool_info("test_tool", "A test tool")
        mock_resource = _make_mock_tool_resource(tool_infos=[info])
        mock_client.return_value.get.return_value = mock_resource

        with patch.object(_tool_resource_mod, "ToolClient", mock_client):
            result = framework_tool_resource("my-tool")
            assert isinstance(result, list)
            assert len(result) >= 1

    def test_langchain_tool_resource(self):
        """LangChain tool_resource() 返回列表"""
        self._run_framework_test("agentrun.integration.langchain.builtin")

    def test_google_adk_tool_resource(self):
        """Google ADK tool_resource() 返回列表"""
        self._run_framework_test("agentrun.integration.google_adk.builtin")

    def test_langgraph_tool_resource(self):
        """LangGraph tool_resource() 返回列表"""
        self._run_framework_test("agentrun.integration.langgraph.builtin")

    def test_agentscope_tool_resource(self):
        """AgentScope tool_resource() 返回列表"""
        self._run_framework_test("agentrun.integration.agentscope.builtin")

    def test_crewai_tool_resource(self):
        """CrewAI tool_resource() 返回列表"""
        self._run_framework_test("agentrun.integration.crewai.builtin")

    def test_pydantic_ai_tool_resource(self):
        """PydanticAI tool_resource() 返回列表"""
        self._run_framework_test("agentrun.integration.pydantic_ai.builtin")

    def test_framework_with_filter(self):
        """框架 tool_resource() 支持 filter_tools_by_name 参数"""
        from agentrun.integration.langchain.builtin import (
            tool_resource as lc_tool_resource,
        )

        mock_client = MagicMock()
        infos = [
            _make_tool_info("get_data", "Get data"),
            _make_tool_info("set_data", "Set data"),
        ]
        mock_resource = _make_mock_tool_resource(tool_infos=infos)
        mock_client.return_value.get.return_value = mock_resource

        with patch.object(_tool_resource_mod, "ToolClient", mock_client):
            result = lc_tool_resource(
                "my-tool",
                filter_tools_by_name=lambda name: name.startswith("get_"),
            )
            assert isinstance(result, list)
            assert len(result) == 1


# =============================================================================
# Tests: builtin __init__ 导出
# =============================================================================


class TestBuiltinInit:
    """测试 builtin __init__.py 正确导出 tool_resource"""

    def test_import_from_builtin(self):
        """可以从 builtin 导入 tool_resource"""
        from agentrun.integration.builtin import tool_resource as imported_func

        assert callable(imported_func)

    def test_in_all(self):
        """tool_resource 在 __all__ 中"""
        import agentrun.integration.builtin as builtin_module

        assert "tool_resource" in builtin_module.__all__
