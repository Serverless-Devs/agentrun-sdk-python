"""Invoker 扩展测试

测试 AgentInvoker 的更多边界情况。
"""

from typing import List

import pytest

from agentrun.server.invoker import AgentInvoker
from agentrun.server.model import AgentEvent, AgentRequest, EventType


class TestInvokerListReturn:
    """测试列表返回值处理"""

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
    async def test_list_of_agent_events(self, req):
        """测试返回 AgentEvent 列表"""

        def invoke_agent(req: AgentRequest):
            return [
                AgentEvent(event=EventType.TEXT, data={"delta": "Hello"}),
                AgentEvent(event=EventType.TEXT, data={"delta": " World"}),
            ]

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].event == EventType.TEXT
        assert result[0].data["delta"] == "Hello"
        assert result[1].data["delta"] == " World"

    @pytest.mark.asyncio
    async def test_list_of_strings(self, req):
        """测试返回字符串列表"""

        def invoke_agent(req: AgentRequest):
            return ["Hello", "", "World"]  # 空字符串被过滤

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        assert isinstance(result, list)
        # 注意：列表返回值中的空字符串不会被过滤，只有 `if item` 检查
        # 实际上 " " 不是空字符串，所以不会被过滤
        assert len(result) == 2  # 空字符串被过滤
        assert result[0].event == EventType.TEXT
        assert result[0].data["delta"] == "Hello"
        assert result[1].data["delta"] == "World"

    @pytest.mark.asyncio
    async def test_list_of_mixed_items(self, req):
        """测试返回混合列表"""

        def invoke_agent(req: AgentRequest):
            return [
                "Hello",
                AgentEvent(
                    event=EventType.TOOL_CALL,
                    data={"id": "tc-1", "name": "test", "args": "{}"},
                ),
                "World",
            ]

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0].event == EventType.TEXT
        assert result[1].event == EventType.TOOL_CALL_CHUNK  # TOOL_CALL 被展开
        assert result[2].event == EventType.TEXT

    @pytest.mark.asyncio
    async def test_list_with_empty_strings(self, req):
        """测试列表中的空字符串被过滤"""

        def invoke_agent(req: AgentRequest):
            return ["Hello", "", "World", ""]

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].data["delta"] == "Hello"
        assert result[1].data["delta"] == "World"


class TestInvokerAsyncHandler:
    """测试异步处理器的各种情况"""

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
    async def test_async_function_returning_non_awaitable(self, req):
        """测试异步函数返回非 awaitable 值"""

        # 这种情况在实际中不太可能发生，但代码中有处理
        async def invoke_agent(req: AgentRequest):
            # 直接返回字符串（不是 awaitable）
            return "Hello"

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].data["delta"] == "Hello"


class TestInvokerWrapStream:
    """测试 _wrap_stream 方法"""

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
    async def test_wrap_stream_with_none(self, req):
        """测试流中的 None 值被过滤"""

        async def invoke_agent(req: AgentRequest):
            yield None
            yield "Hello"
            yield None

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        items: List[AgentEvent] = []
        async for item in result:
            items.append(item)

        assert len(items) == 1
        assert items[0].data["delta"] == "Hello"

    @pytest.mark.asyncio
    async def test_wrap_stream_with_empty_string(self, req):
        """测试流中的空字符串被过滤"""

        async def invoke_agent(req: AgentRequest):
            yield ""
            yield "Hello"
            yield ""

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        items: List[AgentEvent] = []
        async for item in result:
            items.append(item)

        assert len(items) == 1
        assert items[0].data["delta"] == "Hello"

    @pytest.mark.asyncio
    async def test_wrap_stream_with_agent_event(self, req):
        """测试流中的 AgentEvent"""

        async def invoke_agent(req: AgentRequest):
            yield AgentEvent(event=EventType.TEXT, data={"delta": "Hello"})
            yield AgentEvent(
                event=EventType.TOOL_CALL,
                data={"id": "tc-1", "name": "test", "args": "{}"},
            )

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        items: List[AgentEvent] = []
        async for item in result:
            items.append(item)

        assert len(items) == 2
        assert items[0].event == EventType.TEXT
        assert items[1].event == EventType.TOOL_CALL_CHUNK  # TOOL_CALL 被展开


class TestInvokerIterateAsync:
    """测试 _iterate_async 方法"""

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
    async def test_iterate_sync_generator(self, req):
        """测试迭代同步生成器"""

        def invoke_agent(req: AgentRequest):
            yield "Hello"
            yield "World"

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        assert len(items) == 2
        assert items[0].data["delta"] == "Hello"
        assert items[1].data["delta"] == "World"

    @pytest.mark.asyncio
    async def test_iterate_async_generator(self, req):
        """测试迭代异步生成器"""

        async def invoke_agent(req: AgentRequest):
            yield "Hello"
            yield "World"

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        assert len(items) == 2
        assert items[0].data["delta"] == "Hello"
        assert items[1].data["delta"] == "World"


class TestInvokerIsIterator:
    """测试 _is_iterator 方法"""

    def test_is_iterator_with_agent_event(self):
        """测试 AgentEvent 不是迭代器"""

        def invoke_agent(req: AgentRequest):
            return "Hello"

        invoker = AgentInvoker(invoke_agent)

        event = AgentEvent(event=EventType.TEXT, data={"delta": "Hello"})
        assert invoker._is_iterator(event) is False

    def test_is_iterator_with_string(self):
        """测试字符串不是迭代器"""

        def invoke_agent(req: AgentRequest):
            return "Hello"

        invoker = AgentInvoker(invoke_agent)
        assert invoker._is_iterator("Hello") is False

    def test_is_iterator_with_bytes(self):
        """测试字节不是迭代器"""

        def invoke_agent(req: AgentRequest):
            return "Hello"

        invoker = AgentInvoker(invoke_agent)
        assert invoker._is_iterator(b"Hello") is False

    def test_is_iterator_with_dict(self):
        """测试字典不是迭代器"""

        def invoke_agent(req: AgentRequest):
            return "Hello"

        invoker = AgentInvoker(invoke_agent)
        assert invoker._is_iterator({"key": "value"}) is False

    def test_is_iterator_with_list(self):
        """测试列表不是迭代器"""

        def invoke_agent(req: AgentRequest):
            return "Hello"

        invoker = AgentInvoker(invoke_agent)
        assert invoker._is_iterator([1, 2, 3]) is False

    def test_is_iterator_with_generator(self):
        """测试生成器是迭代器"""

        def invoke_agent(req: AgentRequest):
            return "Hello"

        invoker = AgentInvoker(invoke_agent)

        def gen():
            yield 1

        assert invoker._is_iterator(gen()) is True

    def test_is_iterator_with_async_generator(self):
        """测试异步生成器是迭代器"""

        def invoke_agent(req: AgentRequest):
            return "Hello"

        invoker = AgentInvoker(invoke_agent)

        async def async_gen():
            yield 1

        assert invoker._is_iterator(async_gen()) is True


class TestInvokerProcessUserEvent:
    """测试 _process_user_event 方法"""

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
    async def test_tool_call_expansion_with_missing_id(self, req):
        """测试 TOOL_CALL 展开时缺少 id"""

        async def invoke_agent(req: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL,
                data={"name": "test", "args": "{}"},  # 没有 id
            )

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        assert len(items) == 1
        assert items[0].event == EventType.TOOL_CALL_CHUNK
        # id 应该被自动生成（UUID）
        assert items[0].data["id"] is not None
        assert len(items[0].data["id"]) > 0

    @pytest.mark.asyncio
    async def test_other_events_passthrough(self, req):
        """测试其他事件直接传递"""

        async def invoke_agent(req: AgentRequest):
            yield AgentEvent(
                event=EventType.CUSTOM,
                data={"name": "test", "value": {"data": 123}},
            )
            yield AgentEvent(
                event=EventType.STATE,
                data={"snapshot": {"key": "value"}},
            )
            yield AgentEvent(
                event=EventType.ERROR,
                data={"message": "Test error", "code": "TEST"},
            )

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        assert len(items) == 3
        assert items[0].event == EventType.CUSTOM
        assert items[1].event == EventType.STATE
        assert items[2].event == EventType.ERROR


class TestInvokerSyncHandler:
    """测试同步处理器"""

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
    async def test_sync_handler_returning_string(self, req):
        """测试同步处理器返回字符串"""

        def invoke_agent(req: AgentRequest) -> str:
            return "Hello from sync"

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].data["delta"] == "Hello from sync"

    @pytest.mark.asyncio
    async def test_sync_handler_in_invoke_stream(self, req):
        """测试同步处理器在 invoke_stream 中"""

        def invoke_agent(req: AgentRequest) -> str:
            return "Hello from sync"

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        assert len(items) == 1
        assert items[0].data["delta"] == "Hello from sync"


class TestInvokerAsyncNonAwaitable:
    """测试异步函数返回非 awaitable 值"""

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
    async def test_async_function_returning_list_directly(self, req):
        """测试异步函数直接返回列表（非 awaitable）

        这种情况在实际中不太可能发生，但代码中有处理
        """

        # 创建一个返回列表的异步函数
        async def invoke_agent(req: AgentRequest):
            # 直接返回列表（不是 awaitable）
            return [
                AgentEvent(event=EventType.TEXT, data={"delta": "Hello"}),
            ]

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        assert isinstance(result, list)
        assert len(result) == 1


class TestInvokerStreamError:
    """测试流式错误处理"""

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
    async def test_error_in_async_generator(self, req):
        """测试异步生成器中的错误"""

        async def invoke_agent(req: AgentRequest):
            yield "Hello"
            raise RuntimeError("Test error in generator")

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        # 应该有文本事件和错误事件
        assert len(items) == 2
        assert items[0].event == EventType.TEXT
        assert items[1].event == EventType.ERROR
        assert "Test error in generator" in items[1].data["message"]

    @pytest.mark.asyncio
    async def test_error_in_sync_generator(self, req):
        """测试同步生成器中的错误"""

        def invoke_agent(req: AgentRequest):
            yield "Hello"
            raise ValueError("Test error in sync generator")

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        # 应该有文本事件和错误事件
        assert len(items) == 2
        assert items[0].event == EventType.TEXT
        assert items[1].event == EventType.ERROR


class TestInvokerStreamAgentEvent:
    """测试流式返回 AgentEvent 的情况（覆盖 123->111 和 267->255 分支）"""

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
    async def test_stream_with_agent_event_in_async_generator(self, req):
        """测试异步生成器中返回 AgentEvent"""

        async def invoke_agent(req: AgentRequest):
            yield AgentEvent(event=EventType.TEXT, data={"delta": "Hello"})
            yield AgentEvent(event=EventType.TEXT, data={"delta": " World"})

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        assert len(items) == 2
        assert items[0].event == EventType.TEXT
        assert items[0].data["delta"] == "Hello"
        assert items[1].data["delta"] == " World"

    @pytest.mark.asyncio
    async def test_stream_with_mixed_types_in_async_generator(self, req):
        """测试异步生成器中混合返回字符串和 AgentEvent"""

        async def invoke_agent(req: AgentRequest):
            yield "Hello"
            yield AgentEvent(
                event=EventType.TOOL_CALL,
                data={"id": "tc-1", "name": "test", "args": "{}"},
            )
            yield " World"

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        assert len(items) == 3
        assert items[0].event == EventType.TEXT
        assert items[0].data["delta"] == "Hello"
        assert items[1].event == EventType.TOOL_CALL_CHUNK  # TOOL_CALL 被展开
        assert items[2].event == EventType.TEXT
        assert items[2].data["delta"] == " World"

    @pytest.mark.asyncio
    async def test_stream_with_agent_event_in_sync_generator(self, req):
        """测试同步生成器中返回 AgentEvent"""

        def invoke_agent(req: AgentRequest):
            yield AgentEvent(event=EventType.TEXT, data={"delta": "Hello"})
            yield AgentEvent(
                event=EventType.CUSTOM, data={"name": "test", "value": {}}
            )

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentEvent] = []
        async for item in invoker.invoke_stream(req):
            items.append(item)

        assert len(items) == 2
        assert items[0].event == EventType.TEXT
        assert items[1].event == EventType.CUSTOM


class TestInvokerNonStreamList:
    """测试非流式返回列表的情况（覆盖 230->242 分支）"""

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
    async def test_return_list_with_only_agent_events(self, req):
        """测试返回只包含 AgentEvent 的列表"""

        def invoke_agent(req: AgentRequest):
            return [
                AgentEvent(event=EventType.TEXT, data={"delta": "Hello"}),
                AgentEvent(event=EventType.TEXT, data={"delta": " World"}),
            ]

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_return_list_with_only_strings(self, req):
        """测试返回只包含字符串的列表"""

        def invoke_agent(req: AgentRequest):
            return ["Hello", "World"]

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].event == EventType.TEXT
        assert result[0].data["delta"] == "Hello"

    @pytest.mark.asyncio
    async def test_return_list_with_mixed_types(self, req):
        """测试返回混合类型的列表"""

        def invoke_agent(req: AgentRequest):
            return [
                "Hello",
                AgentEvent(event=EventType.CUSTOM, data={"name": "test"}),
                "World",
            ]

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0].event == EventType.TEXT
        assert result[1].event == EventType.CUSTOM
        assert result[2].event == EventType.TEXT


class TestInvokerAsyncNonIterator:
    """测试异步函数返回非迭代器值的情况（覆盖 196 分支）"""

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
    async def test_async_function_returning_list(self, req):
        """测试异步函数返回列表"""

        async def invoke_agent(req: AgentRequest):
            return [AgentEvent(event=EventType.TEXT, data={"delta": "Test"})]

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(req)

        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handler_detected_as_async_but_returns_non_awaitable(
        self, req
    ):
        """测试被检测为异步但返回非 awaitable 值的处理器（覆盖 196 行）

        通过 mock is_async 属性来模拟这种边界情况
        """
        from unittest.mock import patch

        def sync_handler(req: AgentRequest):
            return "Hello"

        invoker = AgentInvoker(sync_handler)

        # 强制设置 is_async 为 True，模拟边界情况
        with patch.object(invoker, "is_async", True):
            # 由于 sync_handler 返回的是字符串（不是 awaitable），
            # 代码会走到第 196 行的 else 分支
            result = await invoker.invoke(req)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].data["delta"] == "Hello"
