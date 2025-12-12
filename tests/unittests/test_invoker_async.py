"""Agent Invoker 单元测试

测试 AgentInvoker 的各种调用场景。
"""

from typing import AsyncGenerator, List

import pytest

from agentrun.server.invoker import AgentInvoker
from agentrun.server.model import AgentRequest, AgentResult, EventType


class TestInvokerBasic:
    """基本调用测试"""

    @pytest.mark.asyncio
    async def test_async_generator_returns_stream(self):
        """测试异步生成器返回流式结果"""

        async def invoke_agent(req: AgentRequest) -> AsyncGenerator[str, None]:
            yield "hello"
            yield " world"

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(AgentRequest(messages=[]))

        # 结果应该是异步生成器
        assert hasattr(result, "__aiter__")

        # 收集所有结果
        items: List[AgentResult] = []
        async for item in result:
            items.append(item)

        # 应该有 TEXT_MESSAGE_START + 2个 TEXT_MESSAGE_CONTENT
        assert len(items) >= 2

        content_events = [
            item
            for item in items
            if item.event == EventType.TEXT_MESSAGE_CONTENT
        ]
        assert len(content_events) == 2
        assert content_events[0].data["delta"] == "hello"
        assert content_events[1].data["delta"] == " world"

    @pytest.mark.asyncio
    async def test_async_coroutine_returns_list(self):
        """测试异步协程返回列表结果"""

        async def invoke_agent(req: AgentRequest) -> str:
            return "world"

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(AgentRequest(messages=[]))

        # 非流式返回应该是列表
        assert isinstance(result, list)

        # 应该包含 TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT, TEXT_MESSAGE_END
        assert len(result) == 3
        assert result[0].event == EventType.TEXT_MESSAGE_START
        assert result[1].event == EventType.TEXT_MESSAGE_CONTENT
        assert result[1].data["delta"] == "world"
        assert result[2].event == EventType.TEXT_MESSAGE_END


class TestInvokerStream:
    """invoke_stream 方法测试"""

    @pytest.mark.asyncio
    async def test_invoke_stream_with_string(self):
        """测试 invoke_stream 自动包装生命周期事件"""

        async def invoke_agent(req: AgentRequest) -> str:
            return "hello"

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentResult] = []
        async for item in invoker.invoke_stream(AgentRequest(messages=[])):
            items.append(item)

        # 应该包含 RUN_STARTED, TEXT_MESSAGE_*, RUN_FINISHED
        event_types = [item.event for item in items]
        assert EventType.RUN_STARTED in event_types
        assert EventType.RUN_FINISHED in event_types
        assert EventType.TEXT_MESSAGE_CONTENT in event_types
        assert EventType.TEXT_MESSAGE_START in event_types
        assert EventType.TEXT_MESSAGE_END in event_types

    @pytest.mark.asyncio
    async def test_invoke_stream_with_agent_result(self):
        """测试返回 AgentResult 事件"""

        async def invoke_agent(
            req: AgentRequest,
        ) -> AsyncGenerator[AgentResult, None]:
            yield AgentResult(
                event=EventType.STEP_STARTED, data={"step_name": "test"}
            )
            yield AgentResult(
                event=EventType.TEXT_MESSAGE_START,
                data={"message_id": "msg-1", "role": "assistant"},
            )
            yield AgentResult(
                event=EventType.TEXT_MESSAGE_CONTENT,
                data={"message_id": "msg-1", "delta": "hello"},
            )
            yield AgentResult(
                event=EventType.TEXT_MESSAGE_END,
                data={"message_id": "msg-1"},
            )
            yield AgentResult(
                event=EventType.STEP_FINISHED, data={"step_name": "test"}
            )

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentResult] = []
        async for item in invoker.invoke_stream(AgentRequest(messages=[])):
            items.append(item)

        event_types = [item.event for item in items]

        # 应该包含用户返回的事件
        assert EventType.STEP_STARTED in event_types
        assert EventType.STEP_FINISHED in event_types
        assert EventType.TEXT_MESSAGE_CONTENT in event_types

        # 以及自动添加的生命周期事件
        assert EventType.RUN_STARTED in event_types
        assert EventType.RUN_FINISHED in event_types

    @pytest.mark.asyncio
    async def test_invoke_stream_error_handling(self):
        """测试错误处理"""

        async def invoke_agent(req: AgentRequest) -> str:
            raise ValueError("Test error")

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentResult] = []
        async for item in invoker.invoke_stream(AgentRequest(messages=[])):
            items.append(item)

        event_types = [item.event for item in items]

        # 应该包含 RUN_STARTED 和 RUN_ERROR
        assert EventType.RUN_STARTED in event_types
        assert EventType.RUN_ERROR in event_types

        # 检查错误信息
        error_event = next(
            item for item in items if item.event == EventType.RUN_ERROR
        )
        assert "Test error" in error_event.data["message"]
        assert error_event.data["code"] == "ValueError"


class TestInvokerSync:
    """同步调用测试"""

    @pytest.mark.asyncio
    async def test_sync_generator(self):
        """测试同步生成器"""

        def invoke_agent(req: AgentRequest):
            yield "hello"
            yield " world"

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(AgentRequest(messages=[]))

        # 结果应该是异步生成器
        assert hasattr(result, "__aiter__")

        items: List[AgentResult] = []
        async for item in result:
            items.append(item)

        content_events = [
            item
            for item in items
            if item.event == EventType.TEXT_MESSAGE_CONTENT
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
        assert len(result) == 3

        content_event = result[1]
        assert content_event.event == EventType.TEXT_MESSAGE_CONTENT
        assert content_event.data["delta"] == "sync result"


class TestInvokerMixed:
    """混合内容测试"""

    @pytest.mark.asyncio
    async def test_mixed_string_and_events(self):
        """测试混合字符串和事件"""

        async def invoke_agent(req: AgentRequest):
            yield "Hello, "
            yield AgentResult(
                event=EventType.TOOL_CALL_START,
                data={"tool_call_id": "tc-1", "tool_call_name": "test"},
            )
            yield AgentResult(
                event=EventType.TOOL_CALL_END,
                data={"tool_call_id": "tc-1"},
            )
            yield "world!"

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentResult] = []
        async for item in invoker.invoke_stream(AgentRequest(messages=[])):
            items.append(item)

        event_types = [item.event for item in items]

        # 应该包含文本和工具调用事件
        assert EventType.TEXT_MESSAGE_CONTENT in event_types
        assert EventType.TOOL_CALL_START in event_types
        assert EventType.TOOL_CALL_END in event_types

    @pytest.mark.asyncio
    async def test_empty_string_ignored(self):
        """测试空字符串被忽略"""

        async def invoke_agent(req: AgentRequest):
            yield ""
            yield "hello"
            yield ""
            yield "world"
            yield ""

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentResult] = []
        async for item in invoker.invoke_stream(AgentRequest(messages=[])):
            items.append(item)

        content_events = [
            item
            for item in items
            if item.event == EventType.TEXT_MESSAGE_CONTENT
        ]
        # 只有两个非空字符串
        assert len(content_events) == 2
        assert content_events[0].data["delta"] == "hello"
        assert content_events[1].data["delta"] == "world"


class TestInvokerNone:
    """None 值处理测试"""

    @pytest.mark.asyncio
    async def test_none_return(self):
        """测试返回 None"""

        async def invoke_agent(req: AgentRequest):
            return None

        invoker = AgentInvoker(invoke_agent)
        result = await invoker.invoke(AgentRequest(messages=[]))

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_none_in_stream(self):
        """测试流中的 None 被忽略"""

        async def invoke_agent(req: AgentRequest):
            yield None
            yield "hello"
            yield None
            yield "world"

        invoker = AgentInvoker(invoke_agent)

        items: List[AgentResult] = []
        async for item in invoker.invoke_stream(AgentRequest(messages=[])):
            items.append(item)

        content_events = [
            item
            for item in items
            if item.event == EventType.TEXT_MESSAGE_CONTENT
        ]
        assert len(content_events) == 2
