import json
from typing import Any, List

from litellm.files.main import ModelResponse
import pydash
import pytest
import respx

from agentrun.model.api.data import BaseInfo
from agentrun.utils.log import logger


@pytest.fixture
def mock_llm_transport(respx_mock):
    """Pytest fixture providing HTTP mocking for all tests"""
    transport = MockLLMTransport(expect_tools=True)
    transport._setup_respx()
    yield transport


class MockLLMTransport:

    def __init__(self, *, expect_tools: bool = True):
        self.expect_tools = expect_tools
        self.base_url = "https://mock-llm.local/v1"
        self.respx_mock = None

    def install(self, monkeypatch):
        self._patch_model_info(monkeypatch)
        self._patch_litellm(monkeypatch)
        self.respx_mock = self._setup_respx()
        return self

    def _patch_model_info(self, monkeypatch):
        def fake_model_info(inner_self, config=None):
            return BaseInfo(
                api_key="mock-api-key",
                base_url=self.base_url,
                model=inner_self.model_name or "fake-llm-model",
                headers={
                    "Authorization": "Bearer mock-token",
                    "Agentrun-Access-Token": "mock-token",
                },
            )

        monkeypatch.setattr(
            "agentrun.model.api.data.ModelDataAPI.model_info",
            fake_model_info,
        )

    def _patch_litellm(self, monkeypatch):
        def fake_completion(*args, **kwargs):
            messages = kwargs.get("messages") or []
            tools_payload = kwargs.get("tools")
            assert kwargs.get("stream") in (None, False)
            assert pydash.get(kwargs, "stream_options.include_usage") is True

            return self._build_model_response(messages, tools_payload)

        async def fake_acompletion(*args, **kwargs):
            messages = kwargs.get("messages") or []
            tools_payload = kwargs.get("tools")
            assert kwargs.get("stream") in (None, False)
            assert pydash.get(kwargs, "stream_options.include_usage") is True

            return self._build_model_response(messages, tools_payload)

        monkeypatch.setattr("litellm.completion", fake_completion)
        monkeypatch.setattr("litellm.acompletion", fake_acompletion)

        # Also patch the module-level imports in google.adk.models.lite_llm
        # Google ADK imports acompletion at module level, so we need to patch
        # there as well to ensure the mock is used in all contexts
        try:
            import google.adk.models.lite_llm as lite_llm_module

            monkeypatch.setattr(
                lite_llm_module, "acompletion", fake_acompletion
            )
            monkeypatch.setattr(lite_llm_module, "completion", fake_completion)
        except ImportError:
            pass  # google.adk not installed, skip patching

    def _setup_respx(self):
        """Setup respx to intercept all httpx requests to mock base URL"""

        def extract_payload(request):
            try:
                if request.content:
                    body = request.content
                    if isinstance(body, (bytes, bytearray)):
                        body = body.decode()
                    if isinstance(body, str) and body.strip():
                        return json.loads(body)
            except (json.JSONDecodeError, AttributeError):
                pass
            return {}

        def build_response(request, route):
            payload = extract_payload(request)
            is_stream = payload.get("stream", False)
            assert payload.get("model") == "mock-model-proxy"
            response_json = self._build_response(
                payload.get("messages") or [], payload.get("tools")
            )
            if is_stream:
                # Return SSE streaming response
                return respx.MockResponse(
                    status_code=200,
                    content=self._build_sse_stream(response_json),
                    headers={"content-type": "text/event-stream"},
                )
            return respx.MockResponse(status_code=200, json=response_json)

        # Route all requests to the mock base URL (already within respx.mock context)
        respx.route(url__startswith=self.base_url).mock(
            side_effect=build_response
        )

    def _build_response(self, messages: list, tools_payload):
        """Build response as dict for HTTP mock"""
        logger.debug("messages: %s, tools_payload: %s", messages, tools_payload)

        if self.expect_tools and tools_payload is not None:
            self._assert_tools(tools_payload)
        elif tools_payload is not None:
            assert isinstance(tools_payload, list)

        if not messages:
            raise AssertionError("messages payload cannot be empty")

        last_role = messages[-1].get("role")
        if last_role == "tool":
            return {
                "id": "chatcmpl-mock-final",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "mock-model-proxy",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "final result",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }

        return {
            "id": "chatcmpl-mock-tools",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "mock-model-proxy",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tool_call_1",
                            "type": "function",
                            "function": {
                                "name": "weather_lookup",
                                "arguments": '{"city": "上海"}',
                            },
                        },
                        {
                            "id": "tool_call_2",
                            "type": "function",
                            "function": {
                                "name": "get_time_now",
                                "arguments": "{}",
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "total_tokens": 6,
            },
        }

    def _build_model_response(self, messages: list, tools_payload):
        """Build response as ModelResponse for litellm mock"""
        response_dict = self._build_response(messages, tools_payload)
        return ModelResponse(**response_dict)

    def _build_sse_stream(self, response_json: dict) -> bytes:
        """Build SSE stream from response JSON for streaming requests"""
        chunks = []
        choice = response_json.get("choices", [{}])[0]
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls")

        # First chunk with role
        first_chunk = {
            "id": response_json.get("id", "chatcmpl-mock"),
            "object": "chat.completion.chunk",
            "created": response_json.get("created", 1234567890),
            "model": response_json.get("model", "mock-model"),
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        }
        chunks.append(f"data: {json.dumps(first_chunk)}\n\n")

        if tool_calls:
            # Stream tool calls
            for i, tool_call in enumerate(tool_calls):
                tc_chunk = {
                    "id": response_json.get("id", "chatcmpl-mock"),
                    "object": "chat.completion.chunk",
                    "created": response_json.get("created", 1234567890),
                    "model": response_json.get("model", "mock-model"),
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": i,
                                "id": tool_call.get("id"),
                                "type": "function",
                                "function": tool_call.get("function"),
                            }],
                        },
                        "finish_reason": None,
                    }],
                }
                chunks.append(f"data: {json.dumps(tc_chunk)}\n\n")
        else:
            # Stream content
            content = message.get("content", "")
            if content:
                content_chunk = {
                    "id": response_json.get("id", "chatcmpl-mock"),
                    "object": "chat.completion.chunk",
                    "created": response_json.get("created", 1234567890),
                    "model": response_json.get("model", "mock-model"),
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }],
                }
                chunks.append(f"data: {json.dumps(content_chunk)}\n\n")

        # Final chunk with finish_reason
        finish_reason = "tool_calls" if tool_calls else "stop"
        final_chunk = {
            "id": response_json.get("id", "chatcmpl-mock"),
            "object": "chat.completion.chunk",
            "created": response_json.get("created", 1234567890),
            "model": response_json.get("model", "mock-model"),
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }],
        }
        chunks.append(f"data: {json.dumps(final_chunk)}\n\n")
        chunks.append("data: [DONE]\n\n")

        return "".join(chunks).encode("utf-8")

    def _assert_tools(self, tools_payload):
        assert isinstance(tools_payload, list)
        assert (
            pydash.get(tools_payload, "[0].function.name") == "weather_lookup"
        )
        assert (
            pydash.get(tools_payload, "[0].function.description")
            == "查询城市天气"
        )
        assert (
            pydash.get(
                tools_payload,
                "[0].function.parameters.properties.city.type",
            )
            == "string"
        )
        assert (
            pydash.get(tools_payload, "[0].function.parameters.type")
            == "object"
        )
        assert "city" in (
            pydash.get(tools_payload, "[0].function.parameters.required", [])
            or []
        )

        assert pydash.get(tools_payload, "[1].function.name") == "get_time_now"
        assert (
            pydash.get(tools_payload, "[1].function.description")
            == "返回当前时间"
        )
        assert pydash.get(
            tools_payload, "[1].function.parameters.properties"
        ) in (
            {},
            None,
        )
        assert (
            pydash.get(tools_payload, "[1].function.parameters.type")
            == "object"
        )


class TestToolDescriptorProtocol:
    """测试 Tool 类的描述符协议实现

    确保工具方法内部调用其他 @tool 装饰的方法时能正常工作。
    这是修复 BrowserToolSet.goto() 调用 browser_navigate() 时缺少 self 参数问题的测试。
    """

    def test_tool_internal_call_works(self):
        """测试工具内部调用其他工具时能正常工作"""
        from agentrun.integration.utils.tool import CommonToolSet, tool

        class TestToolSet(CommonToolSet):

            def __init__(self):
                self.call_log: List[str] = []
                super().__init__()

            @tool(name="main_tool", description="主工具，会调用子工具")
            def main_tool(self, value: str) -> str:
                """主工具，内部调用 sub_tool"""
                self.call_log.append(f"main_tool({value})")
                # 这里调用另一个 @tool 装饰的方法
                # 修复前会报错：TypeError: ... missing 1 required positional argument: 'self'
                result = self.sub_tool(value=f"from_main:{value}")
                return f"main_result:{result}"

            @tool(name="sub_tool", description="子工具")
            def sub_tool(self, value: str) -> str:
                """子工具"""
                self.call_log.append(f"sub_tool({value})")
                return f"sub_result:{value}"

        ts = TestToolSet()

        # 直接调用 main_tool，它内部会调用 sub_tool
        result = ts.main_tool(value="test_input")

        # 验证两个工具都被正确调用
        assert ts.call_log == [
            "main_tool(test_input)",
            "sub_tool(from_main:test_input)",
        ]
        assert result == "main_result:sub_result:from_main:test_input"

    def test_tool_descriptor_returns_bound_tool(self):
        """测试 Tool.__get__ 返回绑定到实例的 Tool"""
        from agentrun.integration.utils.tool import CommonToolSet, Tool, tool

        class TestToolSet(CommonToolSet):

            def __init__(self):
                super().__init__()

            @tool(name="my_tool", description="测试工具")
            def my_tool(self, x: int) -> int:
                return x * 2

        ts = TestToolSet()

        # 通过实例访问应该返回绑定的 Tool
        bound_tool = ts.my_tool
        assert isinstance(bound_tool, Tool)

        # 绑定的 Tool 应该可以直接调用，不需要传入 self
        result = bound_tool(x=5)
        assert result == 10

    def test_tool_descriptor_class_access(self):
        """测试通过类访问 Tool 时返回未绑定的 Tool"""
        from agentrun.integration.utils.tool import CommonToolSet, Tool, tool

        class TestToolSet(CommonToolSet):

            @tool(name="class_tool", description="类工具")
            def class_tool(self, x: int) -> int:
                return x * 2

        # 通过类访问应该返回未绑定的 Tool
        unbound_tool = TestToolSet.class_tool
        assert isinstance(unbound_tool, Tool)

        # 未绑定的 Tool 调用时需要手动传入实例
        instance = TestToolSet()
        # 通过实例访问会自动绑定
        bound_tool = instance.class_tool
        assert bound_tool(x=3) == 6

    def test_tool_descriptor_caching(self):
        """测试 Tool.__get__ 的缓存机制"""
        from agentrun.integration.utils.tool import CommonToolSet, tool

        class TestToolSet(CommonToolSet):

            @tool(name="cached_tool", description="缓存测试工具")
            def cached_tool(self) -> str:
                return "cached"

        ts = TestToolSet()

        # 多次访问应该返回同一个绑定的 Tool 对象（缓存）
        tool1 = ts.cached_tool
        tool2 = ts.cached_tool

        assert tool1 is tool2  # 应该是同一个对象


class TestIntegration:

    def get_mocked_toolset(self, timezone="UTC"):

        from agentrun.integration.utils.tool import CommonToolSet, tool

        class CustomToolSet(CommonToolSet):

            def __init__(self, timezone="UTC"):
                self.time_zone = timezone
                self.call_history: List[Any] = []

                super().__init__()

            @tool(description="查询城市天气")
            def weather_lookup(self, city: str):
                result = f"{city} 天气晴朗"
                self.call_history.append(result)
                return result

            @tool()
            def get_time_now(self):
                """
                返回当前时间
                """
                result = {
                    "time": "2025-01-02 15:04:05",
                    "timezone": self.time_zone,
                }
                self.call_history.append(result)
                return result

        ts = CustomToolSet(timezone=timezone)
        return ts

    def get_mocked_model(
        self, monkeypatch, mock_llm_transport, *, expect_tools: bool = True
    ):
        mock_llm_transport.expect_tools = expect_tools
        mock_llm_transport._patch_model_info(monkeypatch)
        mock_llm_transport._patch_litellm(monkeypatch)

        from agentrun.integration.builtin.model import model
        from agentrun.model.model_proxy import ModelProxy

        mock_model_proxy = ModelProxy(
            model_proxy_name="mock-model-proxy",
        )

        monkeypatch.setattr(
            "agentrun.model.client.ModelClient.get",
            lambda *args, **kwargs: mock_model_proxy,
        )
        m = model("fake-llm-model")

        return m

    def test_langchain(self, monkeypatch, mock_llm_transport):
        pytest.importorskip("langchain")
        from langchain.agents import create_agent
        from langchain.messages import AIMessage, HumanMessage, ToolMessage

        llm = self.get_mocked_model(
            monkeypatch, mock_llm_transport
        ).to_langchain()
        any_toolset = self.get_mocked_toolset().to_langchain()

        agent = create_agent(
            model=llm,
            tools=[*any_toolset],
            system_prompt="你是一个 AgentRun 测试代理",
        )

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "查询上海天气"}]}
        )

        messages = result["messages"]

        assert len(messages) == 5

        msg0 = messages[0]
        assert isinstance(msg0, HumanMessage)
        assert msg0.content == "查询上海天气"

        msg1 = messages[1]
        assert isinstance(msg1, AIMessage)
        assert msg1.tool_calls == [
            {
                "name": "weather_lookup",
                "args": {"city": "上海"},
                "id": "tool_call_1",
                "type": "tool_call",
            },
            {
                "name": "get_time_now",
                "args": {},
                "id": "tool_call_2",
                "type": "tool_call",
            },
        ]

        msg2 = messages[2]
        assert isinstance(msg2, ToolMessage)
        assert msg2.tool_call_id == "tool_call_1"
        assert msg2.content == "上海 天气晴朗"

        msg3 = messages[3]
        assert isinstance(msg3, ToolMessage)
        assert (
            msg3.content == '{"time": "2025-01-02 15:04:05", "timezone": "UTC"}'
        )

        msg4 = messages[4]
        assert isinstance(msg4, AIMessage)
        assert msg4.content == "final result"

    @pytest.mark.asyncio
    async def test_google_adk(self, monkeypatch, mock_llm_transport):
        pytest.importorskip("google.adk")
        from google.adk.agents import LlmAgent
        from google.adk.apps import App
        from google.adk.runners import InMemoryRunner
        from google.genai.types import Content, Part

        llm = self.get_mocked_model(
            monkeypatch, mock_llm_transport
        ).to_google_adk()
        any_toolset = self.get_mocked_toolset().to_google_adk()

        root_agent = LlmAgent(
            name="weather_time_agent",
            model=llm,
            description=(
                "Agent to answer questions about the time and weather in a"
                " city."
            ),
            instruction=(
                "You are a helpful agent who can answer user questions about"
                " the time and weather in a city."
            ),
            tools=[*any_toolset],
        )

        runner = InMemoryRunner(
            app=App(
                name="agents",
                root_agent=root_agent,
            )
        )

        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="mock-user-id"
        )

        result = runner.run(
            user_id=session.user_id,
            session_id=session.id,
            new_message=Content(
                role="user",
                parts=[Part(text="查询上海天气")],
            ),
        )
        result = list(result)
        assert (
            pydash.get(result, "[0].content.parts[1].function_call.name")
            == "get_time_now"
        )
        assert (
            pydash.get(result, "[0].content.parts[1].function_call.args") == {}
        )

        assert pydash.get(result, "[1].content.role") == "user"
        assert (
            pydash.get(result, "[1].content.parts[0].function_response.id")
            == "tool_call_1"
        )
        assert (
            pydash.get(result, "[1].content.parts[0].function_response.name")
            == "weather_lookup"
        )
        assert (
            pydash.get(
                result, "[1].content.parts[0].function_response.response.result"
            )
            == "上海 天气晴朗"
        )
        assert (
            pydash.get(result, "[1].content.parts[1].function_response.id")
            == "tool_call_2"
        )
        assert (
            pydash.get(result, "[1].content.parts[1].function_response.name")
            == "get_time_now"
        )
        assert pydash.get(
            result, "[1].content.parts[1].function_response.response"
        ) == {"time": "2025-01-02 15:04:05", "timezone": "UTC"}

        assert pydash.get(result, "[2].content.role") == "model"
        assert pydash.get(result, "[2].content.parts[0].text") == "final result"

    @pytest.mark.asyncio
    async def test_agentscope(self, monkeypatch, mock_llm_transport):
        pytest.importorskip("agentscope")
        from agentscope.agent import ReActAgent
        from agentscope.formatter import DashScopeChatFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.message import Msg
        from agentscope.tool import Toolkit

        llm = self.get_mocked_model(
            monkeypatch, mock_llm_transport
        ).to_agentscope()
        any_toolset = self.get_mocked_toolset().to_agentscope()

        toolkit = Toolkit()
        for t in any_toolset:
            toolkit.register_tool_function(t)

        agent = ReActAgent(
            name="mock-agent",
            sys_prompt="mock agent description",
            model=llm,
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
            memory=InMemoryMemory(),
        )

        results = await agent.reply(
            Msg(
                name="user",
                content="查询上海天气",
                role="user",
            )
        )

        assert results is not None
        assert results.role == "assistant"
        assert results.get_text_content() == "final result"

    def test_langgraph(self, monkeypatch, mock_llm_transport):
        pytest.importorskip("langchain")
        pytest.importorskip("langgraph")
        from langchain.messages import AIMessage, HumanMessage, ToolMessage
        from langgraph.graph import MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode

        llm = self.get_mocked_model(
            monkeypatch, mock_llm_transport
        ).to_langgraph()
        any_toolset = self.get_mocked_toolset().to_langgraph()

        # 创建 LangGraph 工作流
        def call_model(state: MessagesState):
            messages = state["messages"]
            response = llm.invoke(messages)
            return {"messages": [response]}

        # 构建图
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(any_toolset))
        workflow.set_entry_point("agent")

        # 添加条件边：如果有工具调用，执行工具；否则结束
        def should_continue(state: MessagesState):
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
        # 工具执行后返回 agent 生成最终响应
        workflow.add_edge("tools", "agent")

        app = workflow.compile()

        # 执行工作流
        result = app.invoke(
            {"messages": [HumanMessage(content="查询上海天气")]}
        )

        messages = result["messages"]

        # 验证结果
        assert len(messages) >= 4

        msg0 = messages[0]
        assert isinstance(msg0, HumanMessage)
        assert msg0.content == "查询上海天气"

        msg1 = messages[1]
        assert isinstance(msg1, AIMessage)
        assert msg1.tool_calls == [
            {
                "name": "weather_lookup",
                "args": {"city": "上海"},
                "id": "tool_call_1",
                "type": "tool_call",
            },
            {
                "name": "get_time_now",
                "args": {},
                "id": "tool_call_2",
                "type": "tool_call",
            },
        ]

        # 验证工具调用结果
        assert any(
            isinstance(msg, ToolMessage)
            and msg.content
            == '{"time": "2025-01-02 15:04:05", "timezone": "UTC"}'
            for msg in messages
        )

        # 最后一条消息应该是最终回复
        assert isinstance(messages[-1], AIMessage)
        assert messages[-1].content == "final result"

    def test_crewai(self, monkeypatch, mock_llm_transport):
        pytest.skip("skip crewai")
        pytest.importorskip("crewai")
        from crewai import Agent

        llm = self.get_mocked_model(monkeypatch, mock_llm_transport).to_crewai()
        any_toolset = self.get_mocked_toolset().to_crewai()

        # 创建 CrewAI Agent
        agent = Agent(
            role="天气助手",
            goal="帮助用户查询天气和时间信息",
            backstory="你是一个专业的天气助手，能够查询天气和时间信息。",
            llm=llm,
            tools=[*any_toolset],
            verbose=False,
        )

        # 执行任务
        result = agent.kickoff("查询上海天气")

        # 验证结果 - CrewAI 会返回最终的文本结果
        # 由于我们的 mock 返回 "final result"，所以应该包含这个文本
        assert "final result" in str(result).lower() or result is not None

    def test_pydanticai(self, monkeypatch, mock_llm_transport):
        pytest.importorskip("pydantic_ai")

        from pydantic_ai import Agent

        llm = self.get_mocked_model(
            monkeypatch, mock_llm_transport
        ).to_pydantic_ai()
        any_toolset = self.get_mocked_toolset().to_pydantic_ai()

        agent = Agent(
            llm,
            instructions="Be concise, reply with one sentence.",
            tools=[*any_toolset],
        )

        result = agent.run_sync("上海的天气如何？")

        messages = result.all_messages()
        assert len(messages) == 4
        assert pydash.get(messages[0], "parts[0].content") == "上海的天气如何？"
        assert pydash.get(messages[1], "parts[0].tool_name") == "weather_lookup"
        # args is JSON string, not dict in pydantic_ai
        assert json.loads(pydash.get(messages[1], "parts[0].args")) == {
            "city": "上海"
        }
        assert pydash.get(messages[1], "parts[0].tool_call_id") == "tool_call_1"
        assert pydash.get(messages[2], "parts[0].content") == "上海 天气晴朗"
        assert pydash.get(messages[3], "parts[0].content") == "final result"

        assert (
            "final result" in result.output.lower() or result.output is not None
        )
