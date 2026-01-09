"""Integration 测试基类和统一响应模型

为所有 integration 测试提供统一的基类，屏蔽框架特定逻辑，
提供类似 AgentServer.invoke 的统一调用接口。

使用方式:
    class TestLangChain(IntegrationTestBase):
        def create_agent(self, model, tools=None, system_prompt="..."):
            ...

        def invoke(self, agent, message):
            ...

        async def ainvoke(self, agent, message):
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from agentrun.integration.builtin.model import CommonModel
from agentrun.integration.utils.tool import CommonToolSet


@dataclass
class ToolCallInfo:
    """工具调用信息"""

    name: str
    arguments: Dict[str, Any]
    id: str
    result: Optional[str] = None


@dataclass
class IntegrationTestResult:
    """统一的 Integration 测试结果

    将不同框架的响应格式统一为标准格式，便于测试验证。
    """

    final_text: str
    """最终文本响应"""

    tool_calls: List[ToolCallInfo] = field(default_factory=list)
    """所有工具调用信息"""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    """完整消息历史（框架特定格式）"""

    raw_response: Any = None
    """原始框架响应"""

    def has_tool_calls(self) -> bool:
        """是否有工具调用"""
        return len(self.tool_calls) > 0

    def get_tool_call(self, name: str) -> Optional[ToolCallInfo]:
        """获取指定名称的工具调用"""
        for tc in self.tool_calls:
            if tc.name == name:
                return tc
        return None


@dataclass
class StreamChunk:
    """流式输出的单个块"""

    content: Optional[str] = None
    """文本内容"""

    tool_call_id: Optional[str] = None
    """工具调用 ID"""

    tool_call_name: Optional[str] = None
    """工具调用名称"""

    tool_call_args_delta: Optional[str] = None
    """工具调用参数增量"""

    is_final: bool = False
    """是否是最后一个块"""


class IntegrationTestBase(ABC):
    """Integration 测试基类

    每个框架的测试类需要继承此基类并实现以下抽象方法：
    - create_agent(): 创建框架特定的 Agent
    - invoke(): 同步调用 Agent
    - ainvoke(): 异步调用 Agent
    - stream(): 流式调用 Agent（可选）

    基类提供统一的测试方法和验证逻辑。
    """

    @abstractmethod
    def create_agent(
        self,
        model: CommonModel,
        tools: Optional[CommonToolSet] = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Any:
        """创建框架特定的 Agent

        Args:
            model: AgentRun 通用模型
            tools: 可选的工具集
            system_prompt: 系统提示词

        Returns:
            框架特定的 Agent 对象
        """
        pass

    @abstractmethod
    def invoke(self, agent: Any, message: str) -> IntegrationTestResult:
        """同步调用 Agent

        Args:
            agent: 框架特定的 Agent 对象
            message: 用户消息

        Returns:
            统一的测试结果
        """
        pass

    @abstractmethod
    async def ainvoke(self, agent: Any, message: str) -> IntegrationTestResult:
        """异步调用 Agent

        Args:
            agent: 框架特定的 Agent 对象
            message: 用户消息

        Returns:
            统一的测试结果
        """
        pass

    def stream(self, agent: Any, message: str) -> Iterator[StreamChunk]:
        """流式调用 Agent（可选实现）

        Args:
            agent: 框架特定的 Agent 对象
            message: 用户消息

        Yields:
            流式输出块

        Raises:
            NotImplementedError: 如果框架不支持流式调用
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming"
        )

    async def astream(self, agent: Any, message: str) -> Iterator[StreamChunk]:
        """异步流式调用 Agent（可选实现）

        Args:
            agent: 框架特定的 Agent 对象
            message: 用户消息

        Yields:
            流式输出块

        Raises:
            NotImplementedError: 如果框架不支持流式调用
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async streaming"
        )

    # =========================================================================
    # 验证辅助方法
    # =========================================================================

    def assert_final_text(self, result: IntegrationTestResult, expected: str):
        """验证最终文本"""
        assert (
            result.final_text == expected
        ), f"Expected '{expected}', got '{result.final_text}'"

    def assert_final_text_contains(
        self, result: IntegrationTestResult, substring: str
    ):
        """验证最终文本包含指定字符串"""
        assert (
            substring in result.final_text
        ), f"Expected '{substring}' in '{result.final_text}'"

    def assert_tool_called(
        self,
        result: IntegrationTestResult,
        tool_name: str,
        expected_args: Optional[Dict[str, Any]] = None,
    ):
        """验证工具被调用"""
        tool_call = result.get_tool_call(tool_name)
        assert tool_call is not None, (
            f"Tool '{tool_name}' was not called. Called tools:"
            f" {[tc.name for tc in result.tool_calls]}"
        )

        if expected_args is not None:
            assert (
                tool_call.arguments == expected_args
            ), f"Expected args {expected_args}, got {tool_call.arguments}"

    def assert_tool_not_called(
        self, result: IntegrationTestResult, tool_name: str
    ):
        """验证工具未被调用"""
        tool_call = result.get_tool_call(tool_name)
        assert tool_call is None, f"Tool '{tool_name}' was unexpectedly called"

    def assert_no_tool_calls(self, result: IntegrationTestResult):
        """验证没有工具调用"""
        assert not result.has_tool_calls(), (
            "Expected no tool calls, "
            f"got {[tc.name for tc in result.tool_calls]}"
        )

    def assert_tool_call_count(
        self, result: IntegrationTestResult, expected_count: int
    ):
        """验证工具调用次数"""
        actual_count = len(result.tool_calls)
        assert actual_count == expected_count, (
            f"Expected {expected_count} tool calls, got {actual_count}. "
            f"Tools: {[tc.name for tc in result.tool_calls]}"
        )
