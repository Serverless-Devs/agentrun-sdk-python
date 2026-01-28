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

        # 设置一个安全的 LLM 调用限制，避免无限循环
        # 正常的工具调用场景不应该超过 10 次 LLM 调用
        from google.adk.agents.run_config import RunConfig

        run_config = RunConfig(max_llm_calls=10)

        # 关键修复：使用 run_async() 而不是 run()
        #
        # 问题背景：
        # - runner.run() 创建新线程执行异步代码（见 google/adk/runners.py）
        # - 新线程有独立的事件循环，respx_mock 无法跨线程工作
        # - 在 CI 环境（Linux）中，线程隔离更严格，导致 mock 完全失效
        # - Mock 失效后，真实 HTTP 请求被发送，失败后重试，达到 max_llm_calls 限制
        #
        # 解决方案：
        # - 使用 run_async() 在当前事件循环中运行，避免线程隔离问题
        # - 同时确保 respx_mock fixture 已传入 MockLLMServer（在 mock_server fixture 中）
        # - 这样 respx mock 能在所有环境（本地/CI）中一致生效
        result = runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=Content(
                role="user",
                parts=[Part(text=message)],
            ),
            run_config=run_config,
        )

        # 提取最终文本和工具调用
        final_text = ""
        tool_calls: List[ToolCallInfo] = []
        events = []

        # run_async() 返回异步生成器，使用 async for 遍历
        async for event in result:
            events.append(event)
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

    @pytest.fixture(autouse=True)
    def print_google_adk_version(self):
        """自动打印 Google ADK 版本(每个测试前)"""
        try:
            import google.adk

            version = getattr(google.adk, "__version__", "unknown")
            print(f"\n[INFO] Google ADK version: {version}")
        except Exception as e:
            print(f"\n[WARNING] Failed to get Google ADK version: {e}")

    @pytest.fixture
    def mock_server(self, monkeypatch: Any, respx_mock: Any) -> MockLLMServer:
        """创建并安装 Mock LLM Server

        关键修复：传入 respx_mock fixture 给 MockLLMServer
        - respx_mock 是 pytest-respx 提供的 fixture
        - 确保 HTTP mock 在所有环境（本地/CI）中一致生效
        - 解决了 CI 环境中 mock 不生效导致的测试失败问题
        """
        server = MockLLMServer(expect_tools=True, validate_tools=False)
        server.install(monkeypatch, respx_mock)  # 传入 respx_mock
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
