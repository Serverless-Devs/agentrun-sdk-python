"""LangGraph Integration 测试

测试 LangGraph 框架与 AgentRun 的集成：
- 简单对话（无工具调用）
- 单次工具调用
- 多工具同时调用
- 工作流执行
- stream_options 验证
"""

from typing import Any, List, Optional

import pydash
import pytest

from agentrun.integration.builtin.model import CommonModel
from agentrun.integration.utils.tool import CommonToolSet, tool
from agentrun.model.model_proxy import ModelProxy

from .base import IntegrationTestBase, IntegrationTestResult, ToolCallInfo
from .mock_llm_server import MockLLMServer
from .scenarios import Scenarios


class TestToolSet(CommonToolSet):
    """测试用工具集"""

    def __init__(self, timezone: str = "UTC"):
        self.time_zone = timezone
        self.call_history: List[Any] = []
        super().__init__()

    @tool(description="查询城市天气")
    def weather_lookup(self, city: str) -> str:
        result = f"{city} 天气晴朗"
        self.call_history.append(result)
        return result

    @tool()
    def get_time_now(self) -> dict:
        """返回当前时间"""
        result = {
            "time": "2025-01-02 15:04:05",
            "timezone": self.time_zone,
        }
        self.call_history.append(result)
        return result


class LangGraphTestMixin(IntegrationTestBase):
    """LangGraph 测试混入类

    实现 IntegrationTestBase 的抽象方法。
    """

    def create_agent(
        self,
        model: CommonModel,
        tools: Optional[CommonToolSet] = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Any:
        """创建 LangGraph 工作流"""
        from langgraph.graph import MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode

        llm = model.to_langgraph()
        langgraph_tools = list(tools.to_langgraph()) if tools else []

        # 绑定工具到模型
        if langgraph_tools:
            llm_with_tools = llm.bind_tools(langgraph_tools)
        else:
            llm_with_tools = llm

        def call_model(state: MessagesState) -> dict:
            messages = state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        # 构建图
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", call_model)

        if langgraph_tools:
            workflow.add_node("tools", ToolNode(langgraph_tools))

            def should_continue(state: MessagesState) -> str:
                messages = state["messages"]
                last_message = messages[-1]
                if hasattr(last_message, "tool_calls") and getattr(
                    last_message, "tool_calls", None
                ):
                    return "tools"
                return "end"

            workflow.add_conditional_edges(
                "agent",
                should_continue,
                {"tools": "tools", "end": "__end__"},
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", "__end__")

        workflow.set_entry_point("agent")

        return workflow.compile()

    def invoke(self, agent: Any, message: str) -> IntegrationTestResult:
        """同步调用 LangGraph 工作流"""
        from langchain.messages import AIMessage, HumanMessage, ToolMessage

        result = agent.invoke({"messages": [HumanMessage(content=message)]})

        messages = result.get("messages", [])

        # 提取最终文本
        final_text = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_text = msg.content
                break

        # 提取工具调用
        tool_calls: List[ToolCallInfo] = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        ToolCallInfo(
                            name=tc.get("name", ""),
                            arguments=tc.get("args", {}),
                            id=tc.get("id", ""),
                        )
                    )

        # 提取工具结果
        for msg in messages:
            if isinstance(msg, ToolMessage):
                for tc in tool_calls:
                    if tc.id == msg.tool_call_id:
                        tc.result = msg.content
                        break

        return IntegrationTestResult(
            final_text=final_text,
            tool_calls=tool_calls,
            messages=[self._msg_to_dict(m) for m in messages],
            raw_response=result,
        )

    async def ainvoke(self, agent: Any, message: str) -> IntegrationTestResult:
        """异步调用 LangGraph 工作流"""
        from langchain.messages import AIMessage, HumanMessage, ToolMessage

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]}
        )

        messages = result.get("messages", [])

        # 提取最终文本
        final_text = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_text = msg.content
                break

        # 提取工具调用
        tool_calls: List[ToolCallInfo] = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        ToolCallInfo(
                            name=tc.get("name", ""),
                            arguments=tc.get("args", {}),
                            id=tc.get("id", ""),
                        )
                    )

        # 提取工具结果
        for msg in messages:
            if isinstance(msg, ToolMessage):
                for tc in tool_calls:
                    if tc.id == msg.tool_call_id:
                        tc.result = msg.content
                        break

        return IntegrationTestResult(
            final_text=final_text,
            tool_calls=tool_calls,
            messages=[self._msg_to_dict(m) for m in messages],
            raw_response=result,
        )

    def _msg_to_dict(self, msg: Any) -> dict:
        """将 LangChain/LangGraph 消息转换为字典"""
        return {
            "type": getattr(msg, "type", "unknown"),
            "content": getattr(msg, "content", ""),
        }


class TestLangGraphIntegration(LangGraphTestMixin):
    """LangGraph Integration 测试类"""

    @pytest.fixture
    def mock_server(self, monkeypatch: Any, respx_mock: Any) -> MockLLMServer:
        """创建并安装 Mock LLM Server"""
        server = MockLLMServer(expect_tools=True, validate_tools=False)
        server.install(monkeypatch)
        server.add_default_scenarios()
        return server

    @pytest.fixture
    def mocked_model(
        self, mock_server: MockLLMServer, monkeypatch: Any
    ) -> CommonModel:
        """创建 mock 的模型"""
        from agentrun.integration.builtin.model import model

        mock_model_proxy = ModelProxy(model_proxy_name="mock-model-proxy")

        monkeypatch.setattr(
            "agentrun.model.client.ModelClient.get",
            lambda *args, **kwargs: mock_model_proxy,
        )
        return model("mock-model")

    @pytest.fixture
    def mocked_toolset(self) -> TestToolSet:
        """创建 mock 的工具集"""
        return TestToolSet(timezone="UTC")

    # =========================================================================
    # 测试：简单对话（无工具调用）
    # =========================================================================

    def test_simple_chat_no_tools(
        self,
        mock_server: MockLLMServer,
        mocked_model: CommonModel,
    ):
        """测试简单对话（无工具调用）"""
        # 配置场景
        mock_server.clear_scenarios()
        mock_server.add_scenario(
            Scenarios.simple_chat("你好", "你好！我是AI助手。")
        )

        # 创建无工具的工作流
        agent = self.create_agent(
            model=mocked_model,
            tools=None,
            system_prompt="你是一个友好的助手。",
        )

        # 执行调用
        result = self.invoke(agent, "你好")

        # 验证
        self.assert_final_text(result, "你好！我是AI助手。")
        self.assert_no_tool_calls(result)

    # =========================================================================
    # 测试：工具调用
    # =========================================================================

    def test_single_tool_call(
        self,
        mock_server: MockLLMServer,
        mocked_model: CommonModel,
        mocked_toolset: TestToolSet,
    ):
        """测试单次工具调用"""
        # 配置场景
        mock_server.clear_scenarios()
        mock_server.add_scenario(
            Scenarios.single_tool_call(
                trigger="北京天气",
                tool_name="weather_lookup",
                tool_args={"city": "北京"},
                final_response="北京今天晴天，温度 20°C。",
            )
        )

        # 创建工作流
        agent = self.create_agent(
            model=mocked_model,
            tools=mocked_toolset,
        )

        # 执行调用
        result = self.invoke(agent, "查询北京天气")

        # 验证
        self.assert_final_text(result, "北京今天晴天，温度 20°C。")
        self.assert_tool_called(result, "weather_lookup", {"city": "北京"})

    def test_multi_tool_calls(
        self,
        mock_server: MockLLMServer,
        mocked_model: CommonModel,
        mocked_toolset: TestToolSet,
    ):
        """测试多工具同时调用"""
        # 使用默认的多工具场景
        mock_server.clear_scenarios()
        mock_server.add_scenario(Scenarios.default_multi_tool_scenario())

        # 创建工作流
        agent = self.create_agent(
            model=mocked_model,
            tools=mocked_toolset,
        )

        # 执行调用
        result = self.invoke(agent, "查询上海天气")

        # 验证
        self.assert_final_text(result, "final result")
        self.assert_tool_called(result, "weather_lookup", {"city": "上海"})
        self.assert_tool_called(result, "get_time_now", {})
        self.assert_tool_call_count(result, 2)

    # =========================================================================
    # 测试：stream_options 验证
    # =========================================================================

    def test_stream_options_validation(
        self,
        mock_server: MockLLMServer,
        mocked_model: CommonModel,
        mocked_toolset: TestToolSet,
    ):
        """测试 stream_options 在请求中的正确性"""
        # 使用默认场景
        mock_server.clear_scenarios()
        mock_server.add_scenario(Scenarios.default_multi_tool_scenario())

        # 创建工作流
        agent = self.create_agent(
            model=mocked_model,
            tools=mocked_toolset,
        )

        # 执行调用
        self.invoke(agent, "查询上海天气")

        # 验证捕获的请求
        assert len(mock_server.captured_requests) > 0

        # 验证 stream_options 的正确使用
        for req in mock_server.captured_requests:
            if req.stream is False or req.stream is None:
                # 非流式请求不应该包含 stream_options.include_usage=True
                include_usage = pydash.get(req.stream_options, "include_usage")
                if include_usage is True:
                    pytest.fail(
                        "LangGraph: 非流式请求不应包含 "
                        "stream_options.include_usage=True，"
                        f"stream={req.stream}, "
                        f"stream_options={req.stream_options}"
                    )

    # =========================================================================
    # 测试：异步调用
    # =========================================================================

    @pytest.mark.asyncio
    async def test_async_invoke(
        self,
        mock_server: MockLLMServer,
        mocked_model: CommonModel,
        mocked_toolset: TestToolSet,
    ):
        """测试异步调用"""
        # 使用默认场景
        mock_server.clear_scenarios()
        mock_server.add_scenario(Scenarios.default_multi_tool_scenario())

        # 创建工作流
        agent = self.create_agent(
            model=mocked_model,
            tools=mocked_toolset,
        )

        # 异步执行调用
        result = await self.ainvoke(agent, "查询上海天气")

        # 验证
        self.assert_final_text(result, "final result")
        self.assert_tool_called(result, "weather_lookup")
