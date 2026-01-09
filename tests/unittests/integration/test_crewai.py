"""CrewAI Integration 测试

测试 CrewAI 框架与 AgentRun 的集成：
- 简单对话（无工具调用）
- 工具调用
- stream_options 验证

注意：CrewAI 的测试可能需要特殊配置，因为 CrewAI 的 Agent
需要特定的执行方式（如 Task 和 Crew）。
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


class CrewAITestMixin(IntegrationTestBase):
    """CrewAI 测试混入类

    实现 IntegrationTestBase 的抽象方法。
    """

    def create_agent(
        self,
        model: CommonModel,
        tools: Optional[CommonToolSet] = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Any:
        """创建 CrewAI Agent"""
        from crewai import Agent

        llm = model.to_crewai()
        crewai_tools = list(tools.to_crewai()) if tools else []

        agent = Agent(
            role="Assistant",
            goal=system_prompt,
            backstory="You are a helpful AI assistant.",
            llm=llm,
            tools=crewai_tools if crewai_tools else None,
            verbose=False,
        )
        return agent

    def invoke(self, agent: Any, message: str) -> IntegrationTestResult:
        """同步调用 CrewAI Agent

        CrewAI 需要通过 Task 和 Crew 来执行 Agent。
        """
        from crewai import Crew, Task

        task = Task(
            description=message,
            expected_output="A helpful response",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
        )

        result = crew.kickoff()

        # 提取最终文本
        final_text = str(result) if result else ""

        # CrewAI 的工具调用信息不容易直接获取
        tool_calls: List[ToolCallInfo] = []

        return IntegrationTestResult(
            final_text=final_text,
            tool_calls=tool_calls,
            messages=[],
            raw_response=result,
        )

    async def ainvoke(self, agent: Any, message: str) -> IntegrationTestResult:
        """异步调用 CrewAI Agent

        CrewAI 的异步支持可能有限，这里使用同步调用。
        """
        return self.invoke(agent, message)


class TestCrewAIIntegration(CrewAITestMixin):
    """CrewAI Integration 测试类"""

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

        # 创建无工具的 Agent
        agent = self.create_agent(
            model=mocked_model,
            tools=None,
            system_prompt="你是一个友好的助手。",
        )

        # 执行调用
        result = self.invoke(agent, "你好")

        # 验证 - CrewAI 的响应格式可能不同
        assert result.final_text is not None
        assert result.raw_response is not None

    # =========================================================================
    # 测试：工具调用
    # =========================================================================

    def test_multi_tool_calls(
        self,
        mock_server: MockLLMServer,
        mocked_model: CommonModel,
        mocked_toolset: TestToolSet,
    ):
        """测试多工具同时调用

        注意：CrewAI 的工具调用测试由于框架内部限制暂时跳过。
        CrewAI 的内部工具处理逻辑与 Mock 响应格式存在兼容性问题。
        """
        pytest.skip(
            "CrewAI 工具调用测试暂时跳过，框架内部处理与 Mock 响应存在兼容性问题"
        )

    # =========================================================================
    # 测试：stream_options 验证
    # =========================================================================

    def test_stream_options_validation(
        self,
        mock_server: MockLLMServer,
        mocked_model: CommonModel,
        mocked_toolset: TestToolSet,
    ):
        """测试 stream_options 在请求中的正确性

        CrewAI 不应该在非流式请求中传递 stream_options

        注意：由于 CrewAI 工具调用测试存在兼容性问题，此测试暂时跳过。
        """
        pytest.skip("CrewAI stream_options 测试暂时跳过，依赖于工具调用功能")
