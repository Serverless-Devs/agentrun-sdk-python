"""Agent Invoker 单元测试

测试 AgentInvoker 的各种调用场景。

新设计：invoker 只输出核心事件（TEXT, TOOL_CALL_CHUNK 等），
边界事件（LIFECYCLE_START/END, TEXT_MESSAGE_START/END 等）由协议层自动生成。
"""

from typing import AsyncGenerator, List

import pytest

from agentrun.server.invoker import AgentInvoker
from agentrun.server.model import AgentEvent, AgentRequest, EventType


class TestInvokerBasic:
    """基本调用测试"""

    @pytest.fixture
    def req(self):
        return AgentRequest(
            messages=[],
            tools=[],
            stream=False,
            raw_request=None,
            protocol="unknown",
        )

    @pytest.mark.asyncio
    async def test_async_generator_returns_stream(self, req):
        """测试异步生成器返回流式结果"""

        async def invoke_agent(req: AgentRequest) -> AsyncGenerator[str, None]:
            yield "hello"
            yield " world"

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        # 结果应该是异步生成器
        assert hasattr(result, "__aiter__")

        # 收集所有结果
        items: List[AgentEvent] = []
        async for item in result:
            items.append(item)

        # 应该有 2 个 TEXT 事件（不再有边界事件）
        assert len(items) == 2

        content_events = [
            item for item in items if item.event == EventType.TEXT
        ]
        assert len(content_events) == 2
        assert content_events[0].data["delta"] == "hello"
        assert content_events[1].data["delta"] == " world"

    @pytest.mark.asyncio
    async def test_text_events_structure(self, req):
        """测试 TEXT 事件结构正确"""

        async def invoke_agent(req: AgentRequest) -> AsyncGenerator[str, None]:
            yield "Hello"
            yield " "
            yield "World"

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        items: List[AgentEvent] = []
        async for item in result:
            items.append(item)

        # 应该只有 TEXT 事件
        assert all(item.event == EventType.TEXT for item in items)
        assert len(items) == 3

        # 验证 delta 内容
        deltas = [item.data["delta"] for item in items]
        assert deltas == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_async_coroutine_returns_list(self, req):
        """测试异步协程返回列表结果"""

        async def invoke_agent(req: AgentRequest) -> str:
            return "world"

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        # 非流式返回应该是列表
        assert isinstance(result, list)

        # 应该只包含 TEXT 事件（无边界事件）
        assert len(result) == 1
        assert result[0].event == EventType.TEXT
        assert result[0].data["delta"] == "world"


class TestInvokerStream:
    """invoke_stream 方法测试"""

    @pytest.fixture
    def req(self):
        return AgentRequest(
            messages=[],
            tools=[],
            stream=False,
            raw_request=None,
            protocol="unknown",
        )

    @pytest.mark.asyncio
    async def test_invoke_stream_with_string(self, req):
        """测试 invoke_stream 返回核心事件"""

        async def invoke_agent(req: AgentRequest) -> str:
            return "hello"

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        # 应该只包含 TEXT 事件（边界事件由协议层生成）
        event_types = [item.event for item in items]
        assert EventType.TEXT in event_types
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_invoke_stream_with_agent_event(self, req):
        """测试返回 AgentEvent 事件"""

        async def invoke_agent(
            req: AgentRequest,
        ) -> AsyncGenerator[AgentEvent, None]:
            yield AgentEvent(
                event=EventType.CUSTOM,
                data={"name": "step_started", "value": {"step": "test"}},
            )
            yield AgentEvent(
                event=EventType.TEXT,
                data={"delta": "hello"},
            )
            yield AgentEvent(
                event=EventType.CUSTOM,
                data={"name": "step_finished", "value": {"step": "test"}},
            )

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        event_types = [item.event for item in items]

        # 应该包含用户返回的事件
        assert EventType.CUSTOM in event_types
        assert EventType.TEXT in event_types
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_invoke_stream_error_handling(self, req):
        """测试错误处理"""

        async def invoke_agent(req: AgentRequest) -> str:
            raise ValueError("Test error")

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        event_types = [item.event for item in items]

        # 应该包含 ERROR 事件
        assert EventType.ERROR in event_types

        # 检查错误信息
        error_event = next(
            item for item in items if item.event == EventType.ERROR
        )
        assert "Test error" in error_event.data["message"]
        assert error_event.data["code"] == "ValueError"


class TestInvokerSync:
    """同步调用测试"""

    @pytest.fixture
    def req(self):
        return AgentRequest(
            messages=[],
            tools=[],
            stream=False,
            raw_request=None,
            protocol="unknown",
        )

    @pytest.mark.asyncio
    async def test_sync_generator(self, req):
        """测试同步生成器"""

        def invoke_agent(req: AgentRequest):
            yield "hello"
            yield " world"

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        # 结果应该是异步生成器
        assert hasattr(result, "__aiter__")

        items: List[AgentEvent] = []
        async for item in result:
            items.append(item)

        content_events = [
            item for item in items if item.event == EventType.TEXT
        ]
        assert len(content_events) == 2

    @pytest.mark.asyncio
    async def test_sync_return(self):
        """测试同步函数返回字符串"""

        def invoke_agent(req: AgentRequest) -> str:
            return "sync result"

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(AgentRequest(messages=[]))

        assert isinstance(result, list)
        # 只有一个 TEXT 事件（无边界事件）
        assert len(result) == 1

        content_event = result[0]
        assert content_event.event == EventType.TEXT
        assert content_event.data["delta"] == "sync result"


class TestInvokerMixed:
    """混合内容测试"""

    @pytest.fixture
    def req(self):
        return AgentRequest(
            messages=[],
            tools=[],
            stream=False,
            raw_request=None,
            protocol="unknown",
        )

    @pytest.mark.asyncio
    async def test_mixed_string_and_events(self, req):
        """测试混合字符串和事件"""

        async def invoke_agent(req: AgentRequest):
            yield "Hello, "
            yield AgentEvent(
                event=EventType.TOOL_CALL,
                data={"id": "tc-1", "name": "test", "args": "{}"},
            )
            yield "world!"

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        event_types = [item.event for item in items]

        # 应该包含文本和工具调用事件
        # TOOL_CALL 被展开为 TOOL_CALL_CHUNK
        assert EventType.TEXT in event_types
        assert EventType.TOOL_CALL_CHUNK in event_types

        # 验证内容
        text_events = [i for i in items if i.event == EventType.TEXT]
        assert len(text_events) == 2
        assert text_events[0].data["delta"] == "Hello, "
        assert text_events[1].data["delta"] == "world!"

        tool_events = [i for i in items if i.event == EventType.TOOL_CALL_CHUNK]
        assert len(tool_events) == 1
        assert tool_events[0].data["id"] == "tc-1"
        assert tool_events[0].data["name"] == "test"

    @pytest.mark.asyncio
    async def test_empty_string_ignored(self, req):
        """测试空字符串被忽略"""

        async def invoke_agent(req: AgentRequest):
            yield ""
            yield "hello"
            yield ""
            yield "world"
            yield ""

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        content_events = [
            item for item in items if item.event == EventType.TEXT
        ]
        # 只有两个非空字符串
        assert len(content_events) == 2
        assert content_events[0].data["delta"] == "hello"
        assert content_events[1].data["delta"] == "world"


class TestInvokerNone:
    """None 值处理测试"""

    @pytest.fixture
    def req(self):
        return AgentRequest(
            messages=[],
            tools=[],
            stream=False,
            raw_request=None,
            protocol="unknown",
        )

    @pytest.mark.asyncio
    async def test_none_return(self, req):
        """测试返回 None"""

        async def invoke_agent(req: AgentRequest):
            return None

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_none_in_stream(self, req):
        """测试流中的 None 被忽略"""

        async def invoke_agent(req: AgentRequest):
            yield None
            yield "hello"
            yield None
            yield "world"

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        content_events = [
            item for item in items if item.event == EventType.TEXT
        ]
        assert len(content_events) == 2


class TestInvokerToolCall:
    """工具调用测试"""

    @pytest.fixture
    def req(self):
        return AgentRequest(
            messages=[],
            tools=[],
            stream=False,
            raw_request=None,
            protocol="unknown",
        )

    @pytest.mark.asyncio
    async def test_tool_call_expansion(self, req):
        """测试 TOOL_CALL 被展开为 TOOL_CALL_CHUNK"""

        async def invoke_agent(req: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL,
                data={
                    "id": "call-123",
                    "name": "get_weather",
                    "args": '{"city": "Beijing"}',
                },
            )

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        # TOOL_CALL 被展开为 TOOL_CALL_CHUNK
        assert len(items) == 1
        assert items[0].event == EventType.TOOL_CALL_CHUNK
        assert items[0].data["id"] == "call-123"
        assert items[0].data["name"] == "get_weather"
        assert items[0].data["args_delta"] == '{"city": "Beijing"}'

    @pytest.mark.asyncio
    async def test_tool_call_chunk_passthrough(self, req):
        """测试 TOOL_CALL_CHUNK 直接透传"""

        async def invoke_agent(req: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "call-456",
                    "name": "search",
                    "args_delta": '{"query":',
                },
            )
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "call-456",
                    "args_delta": '"hello"}',
                },
            )

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        assert len(items) == 2
        assert all(i.event == EventType.TOOL_CALL_CHUNK for i in items)
