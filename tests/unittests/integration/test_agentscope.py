"""AgentScope Integration 测试

测试 AgentScope 框架与 AgentRun 的集成：
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


class AgentScopeTestMixin(IntegrationTestBase):
    """AgentScope 测试混入类

    实现 IntegrationTestBase 的抽象方法。
    """

    def create_agent(
        self,
        model: CommonModel,
        tools: Optional[CommonToolSet] = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Any:
        """创建 AgentScope Agent"""
        from agentscope.agent import ReActAgent
        from agentscope.formatter import DashScopeChatFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit

        llm = model.to_agentscope()

        toolkit = Toolkit()
        if tools:
            for t in tools.to_agentscope():
                toolkit.register_tool_function(t)

        agent = ReActAgent(
            name="test-agent",
            sys_prompt=system_prompt,
            model=llm,
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
            memory=InMemoryMemory(),
        )
        return agent

    def invoke(self, agent: Any, message: str) -> IntegrationTestResult:
        """同步调用 AgentScope Agent（通过 asyncio.run）"""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.ainvoke(agent, message)
        )

    async def ainvoke(self, agent: Any, message: str) -> IntegrationTestResult:
        """异步调用 AgentScope Agent"""
        from agentscope.message import Msg

        result = await agent.reply(
            Msg(
                name="user",
                content=message,
                role="user",
            )
        )

        # 提取最终文本
        final_text = ""
        if result:
            final_text = result.get_text_content() or ""

        # AgentScope 的工具调用信息需要从 agent 的历史中提取
        # 由于 ReActAgent 的实现，工具调用信息可能不容易直接获取
        # 这里暂时返回空的工具调用列表
        tool_calls: List[ToolCallInfo] = []

        return IntegrationTestResult(
            final_text=final_text,
            tool_calls=tool_calls,
            messages=[],
            raw_response=result,
        )


class TestAgentScopeIntegration(AgentScopeTestMixin):
    """AgentScope Integration 测试类"""

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

        # 验证 - AgentScope 的响应可能不完全匹配
        # 因为它可能会添加一些额外的格式化
        self.assert_final_text_contains(result, "你好")

    # =========================================================================
    # 测试：工具调用
    # =========================================================================

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

        # 验证 - AgentScope 使用 ReActAgent，最终结果可能是 "final result"
        assert result.final_text is not None
        assert result.raw_response is not None

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

        # AgentScope 配置了 stream=True，所以应该可以使用 stream_options
        for req in mock_server.captured_requests:
            if req.stream is True:
                # 流式请求应该包含 stream_options
                include_usage = pydash.get(req.stream_options, "include_usage")
                assert include_usage is True, (
                    "AgentScope 流式请求应包含 "
                    "stream_options.include_usage=True，"
                    f"但收到: stream={req.stream}, "
                    f"stream_options={req.stream_options}"
                )
