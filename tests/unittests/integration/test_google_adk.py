"""Google ADK Integration 测试

测试 Google ADK 框架与 AgentRun 的集成：
- 简单对话（无工具调用）
- 单次工具调用
- 多工具同时调用
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


class GoogleADKTestMixin(IntegrationTestBase):
    """Google ADK 测试混入类

    实现 IntegrationTestBase 的抽象方法。
    """

    def create_agent(
        self,
        model: CommonModel,
        tools: Optional[CommonToolSet] = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Any:
        """创建 Google ADK Agent"""
        from google.adk.agents import LlmAgent

        llm = model.to_google_adk()
        adk_tools = list(tools.to_google_adk()) if tools else []

        # Google ADK 的 LlmAgent 要求 tools 必须是列表，不能是 None
        agent = LlmAgent(
            name="test_agent",
            model=llm,
            description="Test agent for integration testing",
            instruction=system_prompt,
            tools=adk_tools,  # 总是传递列表（可以是空列表）
        )
        return agent

    def invoke(self, agent: Any, message: str) -> IntegrationTestResult:
        """同步调用 Google ADK Agent（通过 asyncio.run）"""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.ainvoke(agent, message)
        )

    async def ainvoke(self, agent: Any, message: str) -> IntegrationTestResult:
        """异步调用 Google ADK Agent"""
        from google.adk.apps import App
        from google.adk.runners import InMemoryRunner
        from google.genai.types import Content, Part

        runner = InMemoryRunner(app=App(name="test_app", root_agent=agent))

        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="test-user"
        )

        result = runner.run(
            user_id=session.user_id,
            session_id=session.id,
            new_message=Content(
                role="user",
                parts=[Part(text=message)],
            ),
        )

        # 收集所有结果
        events = list(result)

        # 提取最终文本和工具调用
        final_text = ""
        tool_calls: List[ToolCallInfo] = []

        for event in events:
            content = getattr(event, "content", None)
            if content:
                role = getattr(content, "role", None)
                parts = getattr(content, "parts", [])

                if role == "model":
                    for part in parts:
                        # 检查是否有文本
                        text = getattr(part, "text", None)
                        if text:
                            final_text = text

                        # 检查是否有函数调用
                        function_call = getattr(part, "function_call", None)
                        if function_call:
                            name = getattr(function_call, "name", "")
                            args = dict(getattr(function_call, "args", {}))
                            tool_id = getattr(function_call, "id", "")
                            tool_calls.append(
                                ToolCallInfo(
                                    name=name,
                                    arguments=args,
                                    id=tool_id or f"call_{len(tool_calls)}",
                                )
                            )

        return IntegrationTestResult(
            final_text=final_text,
            tool_calls=tool_calls,
            messages=[],  # Google ADK 使用不同的消息格式
            raw_response=events,
        )


class TestGoogleADKIntegration(GoogleADKTestMixin):
    """Google ADK Integration 测试类"""

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

    @pytest.mark.asyncio
    async def test_simple_chat_no_tools(
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
        result = await self.ainvoke(agent, "你好")

        # 验证
        self.assert_final_text(result, "你好！我是AI助手。")
        self.assert_no_tool_calls(result)

    # =========================================================================
    # 测试：工具调用
    # =========================================================================

    @pytest.mark.asyncio
    async def test_single_tool_call(
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

        # 创建 Agent
        agent = self.create_agent(
            model=mocked_model,
            tools=mocked_toolset,
        )

        # 执行调用
        result = await self.ainvoke(agent, "查询北京天气")

        # 验证
        self.assert_final_text(result, "北京今天晴天，温度 20°C。")
        self.assert_tool_called(result, "weather_lookup", {"city": "北京"})

    @pytest.mark.asyncio
    async def test_multi_tool_calls(
        self,
        mock_server: MockLLMServer,
        mocked_model: CommonModel,
        mocked_toolset: TestToolSet,
    ):
        """测试多工具同时调用"""
        # 使用默认的多工具场景
        mock_server.clear_scenarios()
        mock_server.add_scenario(Scenarios.default_multi_tool_scenario())

        # 创建 Agent
        agent = self.create_agent(
            model=mocked_model,
            tools=mocked_toolset,
        )

        # 执行调用
        result = await self.ainvoke(agent, "查询上海天气")

        # 验证
        self.assert_final_text(result, "final result")
        self.assert_tool_called(result, "weather_lookup", {"city": "上海"})
        self.assert_tool_called(result, "get_time_now", {})
        self.assert_tool_call_count(result, 2)

    # =========================================================================
    # 测试：stream_options 验证
    # =========================================================================

    @pytest.mark.asyncio
    async def test_stream_options_validation(
        self,
        mock_server: MockLLMServer,
        mocked_model: CommonModel,
        mocked_toolset: TestToolSet,
    ):
        """测试 stream_options 在请求中的正确性"""
        # 使用默认场景
        mock_server.clear_scenarios()
        mock_server.add_scenario(Scenarios.default_multi_tool_scenario())

        # 创建 Agent
        agent = self.create_agent(
            model=mocked_model,
            tools=mocked_toolset,
        )

        # 执行调用
        await self.ainvoke(agent, "查询上海天气")

        # 验证捕获的请求
        assert len(mock_server.captured_requests) > 0

        # 验证 stream_options 的正确使用
        for req in mock_server.captured_requests:
            if req.stream is True:
                # 流式请求可以包含 stream_options
                include_usage = pydash.get(req.stream_options, "include_usage")
                # Google ADK 使用流式时应该包含 stream_options
            elif req.stream is False or req.stream is None:
                # 非流式请求不应该包含 stream_options.include_usage=True
                include_usage = pydash.get(req.stream_options, "include_usage")
                if include_usage is True:
                    pytest.fail(
                        "Google ADK: 非流式请求不应包含 "
                        "stream_options.include_usage=True，"
                        f"stream={req.stream}, "
                        f"stream_options={req.stream_options}"
                    )
