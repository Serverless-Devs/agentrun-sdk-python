"""AG-UI 事件序列测试

基于 AG-UI 官方验证器 (verifyEvents) 的规则进行测试。

## AG-UI 官方验证规则

1. **RUN 生命周期**
   - 第一个事件必须是 RUN_STARTED（或 RUN_ERROR）
   - RUN_STARTED 不能在活跃 run 期间发送（必须先 RUN_FINISHED）
   - RUN_FINISHED 后可以发送新的 RUN_STARTED（新 run）
   - RUN_ERROR 后不能再发送任何事件
   - RUN_FINISHED 后不能再发送事件（除了新的 RUN_STARTED）
   - RUN_FINISHED 前必须结束所有活跃的 messages、tool calls、steps

2. **TEXT_MESSAGE 规则**（每个 messageId 独立跟踪）
   - TEXT_MESSAGE_START: 同一个 messageId 不能重复开始
   - TEXT_MESSAGE_CONTENT: 必须有对应的活跃 messageId
   - TEXT_MESSAGE_END: 必须有对应的活跃 messageId
   - TEXT_MESSAGE_END 必须在 TOOL_CALL_START 之前发送

3. **TOOL_CALL 规则**（每个 toolCallId 独立跟踪）
   - TOOL_CALL_START: 同一个 toolCallId 不能重复开始
   - TOOL_CALL_ARGS: 必须有对应的活跃 toolCallId
   - TOOL_CALL_END: 必须有对应的活跃 toolCallId
   - TOOL_CALL_START 前必须先结束其他活跃的工具调用（串行化）

4. **串行化要求**（兼容 CopilotKit 等前端）
   - TEXT_MESSAGE 和 TOOL_CALL 不能并行存在
   - 多个 TOOL_CALL 必须串行执行（在新工具开始前必须结束其他工具）
   - 注意：AG-UI 协议本身支持并行，但某些前端实现强制串行

5. **STEP 规则**
   - STEP_STARTED: 同一个 stepName 不能重复开始
   - STEP_FINISHED: 必须有对应的活跃 stepName

## 测试覆盖矩阵

| 规则 | 测试 |
|------|------|
| RUN_STARTED 是第一个事件 | test_run_started_is_first |
| RUN_FINISHED 是最后一个事件 | test_run_finished_is_last |
| RUN_ERROR 后不能发送事件 | test_no_events_after_run_error |
| TEXT_MESSAGE 后 TOOL_CALL | test_text_then_tool_call |
| 多个 TOOL_CALL 串行 | test_tool_calls_serialized |
| RUN_FINISHED 前结束所有 | test_run_finished_ends_all |
"""

import json
from typing import List

import pytest

from agentrun.server import (
    AgentEvent,
    AgentRequest,
    AgentRunServer,
    AGUIProtocolConfig,
    EventType,
    ServerConfig,
)


def parse_sse_line(line: str) -> dict:
    """解析 SSE 行"""
    if line.startswith("data: "):
        return json.loads(line[6:])
    return {}


def get_event_types(lines: List[str]) -> List[str]:
    """提取所有事件类型"""
    types = []
    for line in lines:
        if line.startswith("data: "):
            data = json.loads(line[6:])
            types.append(data.get("type", ""))
    return types


class TestAguiEventSequence:
    """AG-UI 事件序列测试"""

    # ==================== 基本序列测试 ====================

    @pytest.mark.asyncio
    async def test_pure_text_stream(self):
        """测试纯文本流的事件序列

        预期：RUN_STARTED → TEXT_MESSAGE_START → TEXT_MESSAGE_CONTENT* → TEXT_MESSAGE_END → RUN_FINISHED
        """

        async def invoke_agent(request: AgentRequest):
            yield "Hello "
            yield "World"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        assert types[0] == "RUN_STARTED"
        assert types[1] == "TEXT_MESSAGE_START"
        assert types[2] == "TEXT_MESSAGE_CONTENT"
        assert types[3] == "TEXT_MESSAGE_CONTENT"
        assert types[4] == "TEXT_MESSAGE_END"
        assert types[5] == "RUN_FINISHED"

    @pytest.mark.asyncio
    async def test_pure_tool_call(self):
        """测试纯工具调用的事件序列

        预期：RUN_STARTED → TOOL_CALL_START → TOOL_CALL_ARGS → TOOL_CALL_END → TOOL_CALL_RESULT → RUN_FINISHED
        """

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "done"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "call tool"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        assert types == [
            "RUN_STARTED",
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END",
            "TOOL_CALL_RESULT",
            "RUN_FINISHED",
        ]

    # ==================== 文本和工具调用交错测试 ====================

    @pytest.mark.asyncio
    async def test_text_then_tool_call(self):
        """测试 文本 → 工具调用

        AG-UI 协议要求：发送 TOOL_CALL_START 前必须先发送 TEXT_MESSAGE_END
        """

        async def invoke_agent(request: AgentRequest):
            yield "思考中..."
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "search", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "found"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "search"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证 TEXT_MESSAGE_END 在 TOOL_CALL_START 之前
        text_end_idx = types.index("TEXT_MESSAGE_END")
        tool_start_idx = types.index("TOOL_CALL_START")
        assert (
            text_end_idx < tool_start_idx
        ), "TEXT_MESSAGE_END must come before TOOL_CALL_START"

    @pytest.mark.asyncio
    async def test_tool_call_then_text(self):
        """测试 工具调用 → 文本

        关键点：工具调用后的文本需要新的 TEXT_MESSAGE_START
        """

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "calc", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "42"},
            )
            yield "答案是 42"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "calculate"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证工具调用后有新的 TEXT_MESSAGE_START
        assert "TEXT_MESSAGE_START" in types
        text_start_idx = types.index("TEXT_MESSAGE_START")
        tool_result_idx = types.index("TOOL_CALL_RESULT")
        assert (
            text_start_idx > tool_result_idx
        ), "TEXT_MESSAGE_START must come after TOOL_CALL_RESULT"

    @pytest.mark.asyncio
    async def test_text_tool_text(self):
        """测试 文本 → 工具调用 → 文本

        AG-UI 协议要求：
        1. 发送 TOOL_CALL_START 前必须先发送 TEXT_MESSAGE_END
        2. 工具调用后的新文本需要新的 TEXT_MESSAGE_START
        """

        async def invoke_agent(request: AgentRequest):
            yield "让我查一下..."
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "search", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "晴天"},
            )
            yield "今天是晴天。"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "weather"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证有两个 TEXT_MESSAGE_START 和两个 TEXT_MESSAGE_END
        assert types.count("TEXT_MESSAGE_START") == 2
        assert types.count("TEXT_MESSAGE_END") == 2

        # 验证 messageId 不同
        message_ids = []
        for line in lines:
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if data.get("type") == "TEXT_MESSAGE_START":
                    message_ids.append(data.get("messageId"))

        assert len(message_ids) == 2
        assert (
            message_ids[0] != message_ids[1]
        ), "Second text message should have different messageId"

    # ==================== 多工具调用测试 ====================

    @pytest.mark.asyncio
    async def test_sequential_tool_calls(self):
        """测试串行工具调用

        场景：工具1完成后再调用工具2
        """

        async def invoke_agent(request: AgentRequest):
            # 工具1
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "result1"},
            )
            # 工具2
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-2", "name": "tool2", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-2", "result": "result2"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "run tools"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证两个完整的工具调用序列
        assert types.count("TOOL_CALL_START") == 2
        assert types.count("TOOL_CALL_END") == 2
        assert types.count("TOOL_CALL_RESULT") == 2

    @pytest.mark.asyncio
    async def test_tool_chunk_then_text_without_result(self):
        """测试 工具调用（无结果）→ 文本

        AG-UI 协议要求：发送 TEXT_MESSAGE_START 前必须先发送 TOOL_CALL_END
        场景：发送工具调用 chunk 后直接输出文本，没有等待结果
        """

        async def invoke_agent(request: AgentRequest):
            # 发送工具调用
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "async_tool", "args_delta": "{}"},
            )
            # 直接输出文本（没有 TOOL_RESULT）
            yield "工具已触发，无需等待结果。"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "async"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证 TOOL_CALL_END 在 TEXT_MESSAGE_START 之前
        tool_end_idx = types.index("TOOL_CALL_END")
        text_start_idx = types.index("TEXT_MESSAGE_START")
        assert (
            tool_end_idx < text_start_idx
        ), "TOOL_CALL_END must come before TEXT_MESSAGE_START"

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self):
        """测试并行工具调用

        场景：同时开始多个工具调用，然后返回结果
        """

        async def invoke_agent(request: AgentRequest):
            # 两个工具同时开始
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-2", "name": "tool2", "args_delta": "{}"},
            )
            # 结果陆续返回
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "result1"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-2", "result": "result2"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "parallel"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证两个工具调用都正确关闭
        assert types.count("TOOL_CALL_START") == 2
        assert types.count("TOOL_CALL_END") == 2
        assert types.count("TOOL_CALL_RESULT") == 2

    # ==================== 状态和错误事件测试 ====================

    @pytest.mark.asyncio
    async def test_text_then_state(self):
        """测试 文本 → 状态更新

        问题：STATE 事件是否需要先关闭 TEXT_MESSAGE？
        """

        async def invoke_agent(request: AgentRequest):
            yield "处理中..."
            yield AgentEvent(
                event=EventType.STATE,
                data={"snapshot": {"progress": 50}},
            )
            yield "完成！"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "state"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证事件序列
        assert "STATE_SNAPSHOT" in types
        assert "RUN_STARTED" in types
        assert "RUN_FINISHED" in types

    @pytest.mark.asyncio
    async def test_text_then_error(self):
        """测试 文本 → 错误

        AG-UI 协议允许 RUN_ERROR 在任何时候发送，不需要先关闭 TEXT_MESSAGE
        RUN_ERROR 后不能再发送任何事件
        """

        async def invoke_agent(request: AgentRequest):
            yield "处理中..."
            yield AgentEvent(
                event=EventType.ERROR,
                data={"message": "出错了", "code": "ERR001"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "error"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证错误事件存在
        assert "RUN_ERROR" in types

        # RUN_ERROR 是最后一个事件
        assert types[-1] == "RUN_ERROR"

        # 没有 RUN_FINISHED（RUN_ERROR 后不能发送任何事件）
        assert "RUN_FINISHED" not in types

    @pytest.mark.asyncio
    async def test_tool_call_then_error(self):
        """测试 工具调用 → 错误

        AG-UI 协议允许 RUN_ERROR 在任何时候发送，不需要先发送 TOOL_CALL_END
        RUN_ERROR 后不能再发送任何事件
        """

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "risky_tool", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.ERROR,
                data={"message": "工具执行失败", "code": "TOOL_ERROR"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "error"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证错误事件存在
        assert "RUN_ERROR" in types

        # RUN_ERROR 是最后一个事件
        assert types[-1] == "RUN_ERROR"

        # 没有 RUN_FINISHED
        assert "RUN_FINISHED" not in types

    @pytest.mark.asyncio
    async def test_text_then_custom(self):
        """测试 文本 → 自定义事件

        问题：CUSTOM 事件是否需要先关闭 TEXT_MESSAGE？
        """

        async def invoke_agent(request: AgentRequest):
            yield "处理中..."
            yield AgentEvent(
                event=EventType.CUSTOM,
                data={"name": "progress", "value": {"percent": 50}},
            )
            yield "继续..."

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "custom"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证自定义事件存在
        assert "CUSTOM" in types

    # ==================== 边界情况测试 ====================

    @pytest.mark.asyncio
    async def test_empty_text_ignored(self):
        """测试空文本被忽略"""

        async def invoke_agent(request: AgentRequest):
            yield ""
            yield "Hello"
            yield ""

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "empty"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 只有一个 TEXT_MESSAGE_CONTENT（非空的那个）
        assert types.count("TEXT_MESSAGE_CONTENT") == 1

    @pytest.mark.asyncio
    async def test_tool_call_without_result(self):
        """测试没有结果的工具调用

        场景：只有 TOOL_CALL_CHUNK，没有 TOOL_RESULT
        """

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "tc-1",
                    "name": "fire_and_forget",
                    "args_delta": "{}",
                },
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "fire"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证工具调用被正确关闭
        assert "TOOL_CALL_START" in types
        assert "TOOL_CALL_END" in types

    @pytest.mark.asyncio
    async def test_complex_sequence(self):
        """测试复杂序列

        文本 → 工具1 → 文本 → 工具2 → 工具3（并行） → 文本
        AG-UI 允许 TEXT_MESSAGE 和 TOOL_CALL 并行，所以文本消息可以持续
        """

        async def invoke_agent(request: AgentRequest):
            yield "分析问题..."
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "analyze", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "分析完成"},
            )
            yield "开始搜索..."
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-2", "name": "search1", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-3", "name": "search2", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-2", "result": "搜索1完成"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-3", "result": "搜索2完成"},
            )
            yield "综合结果..."

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "complex"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证基本结构
        assert types[0] == "RUN_STARTED"
        assert types[-1] == "RUN_FINISHED"

        # AG-UI 允许并行，所以可能只有一个文本消息（持续）
        assert types.count("TEXT_MESSAGE_START") >= 1
        assert types.count("TEXT_MESSAGE_END") >= 1
        assert types.count("TEXT_MESSAGE_CONTENT") == 3  # 三段文本

        # 验证工具调用数量
        assert types.count("TOOL_CALL_START") == 3
        assert types.count("TOOL_CALL_END") == 3
        assert types.count("TOOL_CALL_RESULT") == 3

    @pytest.mark.asyncio
    async def test_tool_result_without_start(self):
        """测试直接发送 TOOL_RESULT（没有 TOOL_CALL_CHUNK）

        场景：用户直接发送 TOOL_RESULT，没有先发送 TOOL_CALL_CHUNK
        预期：系统自动补充 TOOL_CALL_START 和 TOOL_CALL_END
        """

        async def invoke_agent(request: AgentRequest):
            # 直接发送 TOOL_RESULT，没有 TOOL_CALL_CHUNK
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-orphan", "result": "孤立的结果"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "orphan"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证系统自动补充了 TOOL_CALL_START 和 TOOL_CALL_END
        assert "TOOL_CALL_START" in types
        assert "TOOL_CALL_END" in types
        assert "TOOL_CALL_RESULT" in types

        # 验证顺序：START -> END -> RESULT
        start_idx = types.index("TOOL_CALL_START")
        end_idx = types.index("TOOL_CALL_END")
        result_idx = types.index("TOOL_CALL_RESULT")
        assert start_idx < end_idx < result_idx

    @pytest.mark.asyncio
    async def test_text_then_tool_result_directly(self):
        """测试 文本 → 直接 TOOL_RESULT

        场景：先输出文本，然后直接发送 TOOL_RESULT（没有 TOOL_CALL_CHUNK）
        AG-UI 要求 TEXT_MESSAGE_END 在 TOOL_CALL_START 之前
        系统自动补充 TOOL_CALL_START 和 TOOL_CALL_END
        """

        async def invoke_agent(request: AgentRequest):
            yield "执行结果："
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-direct", "result": "直接结果"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "direct"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证 TEXT_MESSAGE_END 在 TOOL_CALL_START 之前
        text_end_idx = types.index("TEXT_MESSAGE_END")
        tool_start_idx = types.index("TOOL_CALL_START")
        assert text_end_idx < tool_start_idx

    @pytest.mark.asyncio
    async def test_multiple_parallel_tools_then_text(self):
        """测试多个并行工具调用后输出文本

        场景：同时开始多个工具调用，然后输出文本
        在 copilotkit_compatibility=True（默认）模式下，第二个工具的事件会被放入队列，
        等到第一个工具调用收到 RESULT 后再处理。

        由于没有 RESULT 事件，队列中的 tc-b 不会被处理，直到文本事件到来时
        需要先结束 tc-a，然后处理队列中的 tc-b，再结束 tc-b，最后输出文本。

        输入事件：
        - tc-a CHUNK (START)
        - tc-b CHUNK (放入队列，等待 tc-a 完成)
        - TEXT "工具已触发"

        预期输出：
        - tc-a START -> tc-a ARGS -> tc-a END -> tc-b START -> tc-b ARGS -> tc-b END
        - TEXT_MESSAGE_START -> TEXT_MESSAGE_CONTENT -> TEXT_MESSAGE_END
        """

        async def invoke_agent(request: AgentRequest):
            # 并行工具调用
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-a", "name": "tool_a", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-b", "name": "tool_b", "args_delta": "{}"},
            )
            # 直接输出文本（没有等待结果）
            yield "工具已触发"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "parallel"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证两个工具都被关闭了
        assert types.count("TOOL_CALL_END") == 2

        # 验证所有 TOOL_CALL_END 在 TEXT_MESSAGE_START 之前
        text_start_idx = types.index("TEXT_MESSAGE_START")
        for i, t in enumerate(types):
            if t == "TOOL_CALL_END":
                assert i < text_start_idx, (
                    f"TOOL_CALL_END at {i} must come before TEXT_MESSAGE_START"
                    f" at {text_start_idx}"
                )

    @pytest.mark.asyncio
    async def test_text_and_tool_interleaved_with_error(self):
        """测试文本和工具交错后发生错误

        场景：文本 → 工具调用（未完成）→ 错误
        AG-UI 允许 RUN_ERROR 在任何时候发送，不需要先结束其他事件
        """

        async def invoke_agent(request: AgentRequest):
            yield "开始处理..."
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "tc-fail",
                    "name": "failing_tool",
                    "args_delta": "{}",
                },
            )
            yield AgentEvent(
                event=EventType.ERROR,
                data={"message": "处理失败", "code": "PROCESS_ERROR"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "fail"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证错误事件存在
        assert "RUN_ERROR" in types

        # RUN_ERROR 是最后一个事件
        assert types[-1] == "RUN_ERROR"

        # 没有 RUN_FINISHED
        assert "RUN_FINISHED" not in types

    @pytest.mark.asyncio
    async def test_state_between_text_chunks(self):
        """测试在文本流中间发送状态事件

        场景：文本 → 状态 → 文本（同一个消息）
        预期：状态事件不会打断文本消息
        """

        async def invoke_agent(request: AgentRequest):
            yield "第一部分"
            yield AgentEvent(
                event=EventType.STATE,
                data={"snapshot": {"progress": 50}},
            )
            yield "第二部分"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "state"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证只有一个文本消息（状态事件没有打断）
        assert types.count("TEXT_MESSAGE_START") == 1
        assert types.count("TEXT_MESSAGE_END") == 1

        # 验证状态事件存在
        assert "STATE_SNAPSHOT" in types

    @pytest.mark.asyncio
    async def test_custom_between_text_chunks(self):
        """测试在文本流中间发送自定义事件

        场景：文本 → 自定义 → 文本（同一个消息）
        预期：自定义事件不会打断文本消息
        """

        async def invoke_agent(request: AgentRequest):
            yield "第一部分"
            yield AgentEvent(
                event=EventType.CUSTOM,
                data={"name": "metrics", "value": {"tokens": 100}},
            )
            yield "第二部分"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "custom"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证只有一个文本消息（自定义事件没有打断）
        assert types.count("TEXT_MESSAGE_START") == 1
        assert types.count("TEXT_MESSAGE_END") == 1

        # 验证自定义事件存在
        assert "CUSTOM" in types

    @pytest.mark.asyncio
    async def test_no_events_after_run_error(self):
        """测试 RUN_ERROR 后不再发送任何事件

        AG-UI 协议规则：RUN_ERROR 是终结事件，之后不能再发送任何事件
        （包括 TEXT_MESSAGE_START、TEXT_MESSAGE_END、RUN_FINISHED 等）
        """

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.ERROR,
                data={"message": "发生错误", "code": "TEST_ERROR"},
            )
            # 错误后继续输出文本（应该被忽略）
            yield "这段文本不应该出现"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "error"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证 RUN_ERROR 存在
        assert "RUN_ERROR" in types

        # 验证 RUN_ERROR 是最后一个事件
        assert types[-1] == "RUN_ERROR"

        # 验证没有 RUN_FINISHED（RUN_ERROR 后不应该有）
        assert "RUN_FINISHED" not in types

        # 验证没有 TEXT_MESSAGE_START（错误后的文本应该被忽略）
        assert "TEXT_MESSAGE_START" not in types

    @pytest.mark.asyncio
    async def test_text_error_text_ignored(self):
        """测试 文本 → 错误 → 文本（后续文本被忽略）

        场景：先输出文本，发生错误，然后继续输出文本
        预期：
        1. AG-UI 允许 RUN_ERROR 在任何时候发送
        2. 错误后的文本被忽略
        3. 没有 RUN_FINISHED
        """

        async def invoke_agent(request: AgentRequest):
            yield "处理中..."
            yield AgentEvent(
                event=EventType.ERROR,
                data={"message": "处理失败", "code": "PROCESS_ERROR"},
            )
            yield "这段不应该出现"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "error"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证基本结构
        assert "RUN_STARTED" in types
        assert "TEXT_MESSAGE_START" in types
        assert "TEXT_MESSAGE_CONTENT" in types
        assert "RUN_ERROR" in types

        # 验证 RUN_ERROR 是最后一个事件
        assert types[-1] == "RUN_ERROR"

        # 验证没有 RUN_FINISHED
        assert "RUN_FINISHED" not in types

        # 验证只有一个文本消息（错误后的不应该出现）
        assert types.count("TEXT_MESSAGE_START") == 1
        assert types.count("TEXT_MESSAGE_CONTENT") == 1

    @pytest.mark.asyncio
    async def test_tool_error_tool_ignored(self):
        """测试 工具调用 → 错误 → 工具调用（后续工具被忽略）

        场景：开始工具调用，发生错误，然后继续工具调用
        预期：
        1. TOOL_CALL_END 在 RUN_ERROR 之前
        2. 错误后的工具调用被忽略
        3. 没有 RUN_FINISHED
        """

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.ERROR,
                data={"message": "工具失败", "code": "TOOL_ERROR"},
            )
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-2", "name": "tool2", "args_delta": "{}"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "error"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证 RUN_ERROR 是最后一个事件
        assert types[-1] == "RUN_ERROR"

        # 验证没有 RUN_FINISHED
        assert "RUN_FINISHED" not in types

        # 验证只有一个工具调用（错误后的不应该出现）
        assert types.count("TOOL_CALL_START") == 1

    @pytest.mark.asyncio
    async def test_tool_calls_serialized_copilotkit_mode(self):
        """测试工具调用串行化（CopilotKit 兼容模式）

        场景：发送 tc-1 的 CHUNK，然后发送 tc-2 的 CHUNK（没有显式结束 tc-1）
        预期：在 copilotkit_compatibility=True 模式下，发送 tc-2 START 前会自动结束 tc-1

        注意：AG-UI 协议本身支持并行工具调用，但某些前端实现（如 CopilotKit）
        强制要求串行化。为了兼容性，我们提供 copilotkit_compatibility 模式。
        """
        from agentrun.server import AGUIProtocolConfig, ServerConfig

        async def invoke_agent(request: AgentRequest):
            # 第一个工具调用开始
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": '{"a": 1}'},
            )
            # 第二个工具调用开始（会自动先结束 tc-1）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-2", "name": "tool2", "args_delta": '{"b": 2}'},
            )
            # 结果（顺序返回）
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "result1"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-2", "result": "result2"},
            )

        # 启用 CopilotKit 兼容模式
        config = ServerConfig(
            agui=AGUIProtocolConfig(copilotkit_compatibility=True)
        )
        server = AgentRunServer(invoke_agent=invoke_agent, config=config)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "parallel"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证两个工具调用都存在
        assert types.count("TOOL_CALL_START") == 2
        assert types.count("TOOL_CALL_END") == 2
        assert types.count("TOOL_CALL_RESULT") == 2

        # 验证串行化工具调用的顺序：
        # tc-1 START -> tc-1 ARGS -> tc-1 END -> tc-2 START -> tc-2 ARGS -> tc-1 RESULT -> tc-2 END -> tc-2 RESULT
        # 关键验证：tc-1 END 在 tc-2 START 之前（串行化）
        tc1_end_idx = None
        tc2_start_idx = None
        for i, line in enumerate(lines):
            if "TOOL_CALL_END" in line and "tc-1" in line:
                tc1_end_idx = i
            if "TOOL_CALL_START" in line and "tc-2" in line:
                tc2_start_idx = i

        assert tc1_end_idx is not None, "tc-1 TOOL_CALL_END not found"
        assert tc2_start_idx is not None, "tc-2 TOOL_CALL_START not found"
        # 串行化工具调用：tc-1 END 应该在 tc-2 START 之前
        assert (
            tc1_end_idx < tc2_start_idx
        ), "Serialized tool calls: tc-1 END should come before tc-2 START"

    # ==================== AG-UI 官方验证器规则测试 ====================

    @pytest.mark.asyncio
    async def test_run_started_is_first(self):
        """AG-UI 规则：第一个事件必须是 RUN_STARTED"""

        async def invoke_agent(request: AgentRequest):
            yield "Hello"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 第一个事件必须是 RUN_STARTED
        assert types[0] == "RUN_STARTED"

    @pytest.mark.asyncio
    async def test_run_finished_is_last_normal(self):
        """AG-UI 规则：正常结束时 RUN_FINISHED 是最后一个事件"""

        async def invoke_agent(request: AgentRequest):
            yield "Hello"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 最后一个事件是 RUN_FINISHED
        assert types[-1] == "RUN_FINISHED"

    @pytest.mark.asyncio
    async def test_run_error_is_last_on_error(self):
        """AG-UI 规则：错误时 RUN_ERROR 是最后一个事件"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.ERROR,
                data={"message": "Error", "code": "ERR"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 最后一个事件是 RUN_ERROR
        assert types[-1] == "RUN_ERROR"
        # 没有 RUN_FINISHED
        assert "RUN_FINISHED" not in types

    @pytest.mark.asyncio
    async def test_run_finished_ends_all_messages(self):
        """AG-UI 规则：RUN_FINISHED 前必须结束所有 TEXT_MESSAGE"""

        async def invoke_agent(request: AgentRequest):
            # 只发送 TEXT，不显式结束
            yield "Hello"
            yield "World"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证 TEXT_MESSAGE_END 在 RUN_FINISHED 之前
        assert "TEXT_MESSAGE_END" in types
        text_end_idx = types.index("TEXT_MESSAGE_END")
        run_finished_idx = types.index("RUN_FINISHED")
        assert text_end_idx < run_finished_idx

    @pytest.mark.asyncio
    async def test_run_finished_ends_all_tool_calls(self):
        """AG-UI 规则：RUN_FINISHED 前必须结束所有 TOOL_CALL"""

        async def invoke_agent(request: AgentRequest):
            # 只发送 TOOL_CALL_CHUNK，不显式结束
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": "{}"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证 TOOL_CALL_END 在 RUN_FINISHED 之前
        assert "TOOL_CALL_END" in types
        tool_end_idx = types.index("TOOL_CALL_END")
        run_finished_idx = types.index("RUN_FINISHED")
        assert tool_end_idx < run_finished_idx

    @pytest.mark.asyncio
    async def test_text_message_id_consistency(self):
        """AG-UI 规则：TEXT_MESSAGE 的 messageId 必须一致"""

        async def invoke_agent(request: AgentRequest):
            yield "Hello"
            yield " World"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]

        # 提取所有 messageId
        message_ids = []
        for line in lines:
            data = parse_sse_line(line)
            if data and "messageId" in data:
                if data.get("type") in [
                    "TEXT_MESSAGE_START",
                    "TEXT_MESSAGE_CONTENT",
                    "TEXT_MESSAGE_END",
                ]:
                    message_ids.append(data["messageId"])

        # 所有 messageId 应该相同（同一个消息）
        assert len(set(message_ids)) == 1

    @pytest.mark.asyncio
    async def test_tool_call_id_consistency(self):
        """AG-UI 规则：TOOL_CALL 的 toolCallId 必须一致"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": '{"a":'},
            )
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": "1}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "done"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]

        # 提取所有 toolCallId
        tool_call_ids = []
        for line in lines:
            data = parse_sse_line(line)
            if data and "toolCallId" in data:
                tool_call_ids.append(data["toolCallId"])

        # 所有 toolCallId 应该相同（同一个工具调用）
        assert len(set(tool_call_ids)) == 1
        assert tool_call_ids[0] == "tc-1"

    @pytest.mark.asyncio
    async def test_text_tool_text_sequence(self):
        """AG-UI 规则：TEXT_MESSAGE 和 TOOL_CALL 不能并行

        文本 → 工具调用 → 文本 需要两个独立的 TEXT_MESSAGE
        """

        async def invoke_agent(request: AgentRequest):
            yield "开始..."
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": "{}"},
            )
            yield "继续..."  # 工具调用后的文本

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证有两个独立的 TEXT_MESSAGE
        assert types.count("TEXT_MESSAGE_START") == 2
        assert types.count("TEXT_MESSAGE_CONTENT") == 2
        assert types.count("TEXT_MESSAGE_END") == 2

        # 验证第一个 TEXT_MESSAGE_END 在 TOOL_CALL_START 之前
        first_text_end_idx = types.index("TEXT_MESSAGE_END")
        tool_start_idx = types.index("TOOL_CALL_START")
        assert first_text_end_idx < tool_start_idx

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_parallel(self):
        """AG-UI 规则：多个 TOOL_CALL 可以并行（但在 copilotkit_compatibility=True 时会串行）

        在 copilotkit_compatibility=True（默认）模式下，其他工具的事件会被放入队列，
        等到当前工具调用收到 RESULT 后再处理。

        输入事件：
        - tc-1 CHUNK (START)
        - tc-2 CHUNK (放入队列)
        - tc-3 CHUNK (放入队列)
        - tc-2 RESULT (放入队列，因为 tc-1 还没有 RESULT)
        - tc-1 RESULT (处理队列中的事件)
        - tc-3 RESULT

        预期输出：
        - tc-1 START -> tc-1 ARGS -> tc-1 END -> tc-1 RESULT
        - tc-2 START -> tc-2 ARGS -> tc-2 END -> tc-2 RESULT
        - tc-3 START -> tc-3 ARGS -> tc-3 END -> tc-3 RESULT
        """

        async def invoke_agent(request: AgentRequest):
            # 开始第一个工具
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": "{}"},
            )
            # 开始第二个工具（会被放入队列）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-2", "name": "tool2", "args_delta": "{}"},
            )
            # 开始第三个工具（会被放入队列）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-3", "name": "tool3", "args_delta": "{}"},
            )
            # 结果陆续返回
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-2", "result": "result2"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "result1"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-3", "result": "result3"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证三个工具调用都存在
        assert types.count("TOOL_CALL_START") == 3
        assert types.count("TOOL_CALL_END") == 3
        assert types.count("TOOL_CALL_RESULT") == 3

    @pytest.mark.asyncio
    async def test_same_tool_call_id_not_duplicated(self):
        """AG-UI 规则：同一个 toolCallId 不能重复 START"""

        async def invoke_agent(request: AgentRequest):
            # 发送多个相同 ID 的 CHUNK（应该只生成一个 START）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": '{"a":'},
            )
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": "1}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "done"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 只应该有一个 TOOL_CALL_START
        assert types.count("TOOL_CALL_START") == 1
        # 但有两个 TOOL_CALL_ARGS
        assert types.count("TOOL_CALL_ARGS") == 2

    @pytest.mark.asyncio
    async def test_interleaved_tool_calls_with_repeated_args(self):
        """测试交错的工具调用（带重复的 ARGS 事件）

        场景：模拟 LangChain 流式输出时可能产生的交错事件序列
        在 copilotkit_compatibility=True（默认）模式下，其他工具的事件会被放入队列，
        等到当前工具调用收到 RESULT 后再处理。

        输入事件：
        - tc-1 CHUNK (START)
        - tc-2 CHUNK (放入队列，等待 tc-1 完成)
        - tc-1 CHUNK (ARGS，同一个工具调用，继续处理)
        - tc-3 CHUNK (放入队列，等待 tc-1 完成)
        - tc-1 RESULT (处理队列中的 tc-2 和 tc-3)
        - tc-2 RESULT
        - tc-3 RESULT

        预期输出：
        - tc-1 START -> tc-1 ARGS -> tc-1 ARGS -> tc-1 END -> tc-1 RESULT
        - tc-2 START -> tc-2 ARGS -> tc-2 END -> tc-2 RESULT
        - tc-3 START -> tc-3 ARGS -> tc-3 END -> tc-3 RESULT
        """

        async def invoke_agent(request: AgentRequest):
            # 第一个工具调用
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "get_time", "args_delta": '{"tz":'},
            )
            # 第二个工具调用（会被放入队列）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-2", "name": "get_user", "args_delta": ""},
            )
            # tc-1 的额外 ARGS（同一个工具调用，继续处理）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "tc-1",
                    "name": "get_time",
                    "args_delta": '"Asia"}',
                },
            )
            # 第三个工具调用（会被放入队列）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "tc-3",
                    "name": "get_token",
                    "args_delta": '{"user":"test"}',
                },
            )
            # 结果
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "time result"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-2", "result": "user result"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-3", "result": "token result"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证事件序列正确
        assert types[0] == "RUN_STARTED"
        assert types[-1] == "RUN_FINISHED"

        # 每个工具调用只有一个 START
        assert types.count("TOOL_CALL_START") == 3
        assert types.count("TOOL_CALL_END") == 3
        assert types.count("TOOL_CALL_RESULT") == 3

        # 验证串行化：每个 ARGS 都有对应的活跃 START
        tool_states = {}
        for line in lines:
            data = parse_sse_line(line)
            event_type = data.get("type", "")
            tool_id = data.get("toolCallId", "")

            if event_type == "TOOL_CALL_START":
                tool_states[tool_id] = {"started": True, "ended": False}
            elif event_type == "TOOL_CALL_END":
                if tool_id in tool_states:
                    tool_states[tool_id]["ended"] = True
            elif event_type == "TOOL_CALL_ARGS":
                assert (
                    tool_id in tool_states
                ), f"ARGS for {tool_id} without START"
                assert not tool_states[tool_id][
                    "ended"
                ], f"ARGS for {tool_id} after END"

    @pytest.mark.asyncio
    async def test_text_tool_interleaved_complex(self):
        """测试复杂的文本和工具调用交错场景

        场景：模拟真实的 LLM 输出
        在 copilotkit_compatibility=True（默认）模式下，其他工具的事件会被放入队列，
        等到当前工具调用收到 RESULT 后再处理。

        输入事件：
        1. 文本 "让我查一下..."
        2. tc-a CHUNK (START)
        3. tc-b CHUNK (放入队列，等待 tc-a 完成)
        4. tc-a CHUNK (ARGS，同一个工具调用，继续处理)
        5. tc-a RESULT (处理队列中的 tc-b)
        6. tc-b RESULT
        7. 文本 "根据结果..."

        预期输出：
        - TEXT_MESSAGE_START -> TEXT_MESSAGE_CONTENT -> TEXT_MESSAGE_END
        - tc-a START -> tc-a ARGS -> tc-a ARGS -> tc-a END -> tc-a RESULT
        - tc-b START -> tc-b ARGS -> tc-b END -> tc-b RESULT
        - TEXT_MESSAGE_START -> TEXT_MESSAGE_CONTENT -> TEXT_MESSAGE_END
        """

        async def invoke_agent(request: AgentRequest):
            # 第一段文本
            yield "让我查一下..."
            # 工具调用 A
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-a", "name": "search", "args_delta": '{"q":'},
            )
            # 工具调用 B（会被放入队列）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "tc-b",
                    "name": "lookup",
                    "args_delta": '{"id": 1}',
                },
            )
            # A 的额外 ARGS（同一个工具调用，继续处理）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-a", "name": "search", "args_delta": '"test"}'},
            )
            # 结果
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-a", "result": "search result"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-b", "result": "lookup result"},
            )
            # 第二段文本
            yield "根据结果..."

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "complex"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证基本事件存在
        assert types[0] == "RUN_STARTED"
        assert types[-1] == "RUN_FINISHED"

        # 验证文本消息
        assert types.count("TEXT_MESSAGE_START") == 2
        assert types.count("TEXT_MESSAGE_END") == 2

        # 验证工具调用
        # 每个工具调用只有一个 START
        assert types.count("TOOL_CALL_START") == 2  # tc-a 一次, tc-b 一次
        assert types.count("TOOL_CALL_END") == 2
        assert types.count("TOOL_CALL_RESULT") == 2
        assert types.count("TOOL_CALL_ARGS") == 3  # tc-a 两次, tc-b 一次

    @pytest.mark.asyncio
    async def test_tool_call_args_after_end(self):
        """测试工具调用结束后收到 ARGS 事件

        场景：LangChain 交错输出时，可能在 tc-1 END 后收到 tc-1 的 ARGS
        在 copilotkit_compatibility=True（默认）模式下，其他工具的事件会被放入队列，
        等到当前工具调用收到 RESULT 后再处理。

        输入事件：
        - tc-1 CHUNK (START)
        - tc-2 CHUNK (放入队列，等待 tc-1 完成)
        - tc-1 CHUNK (ARGS，同一个工具调用，继续处理)
        - tc-1 RESULT (处理队列中的 tc-2)
        - tc-2 RESULT

        预期输出：
        - tc-1 START -> tc-1 ARGS -> tc-1 ARGS -> tc-1 END -> tc-1 RESULT
        - tc-2 START -> tc-2 ARGS -> tc-2 END -> tc-2 RESULT
        """

        async def invoke_agent(request: AgentRequest):
            # tc-1 开始
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": '{"a":'},
            )
            # tc-2（会被放入队列）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-2", "name": "tool2", "args_delta": '{"b": 2}'},
            )
            # tc-1 的额外 ARGS（同一个工具调用，继续处理）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": "1}"},
            )
            # 结果
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "result1"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-2", "result": "result2"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证事件序列
        assert types[0] == "RUN_STARTED"
        assert types[-1] == "RUN_FINISHED"

        # 每个工具调用只有一个 START
        assert types.count("TOOL_CALL_START") == 2
        assert types.count("TOOL_CALL_END") == 2
        assert types.count("TOOL_CALL_RESULT") == 2

        # 验证每个 ARGS 都有对应的活跃 START
        # 检查事件序列中没有 ARGS 出现在 END 之后（对于同一个 tool_id）
        tool_states = {}  # tool_id -> {"started": bool, "ended": bool}
        for line in lines:
            data = parse_sse_line(line)
            event_type = data.get("type", "")
            tool_id = data.get("toolCallId", "")

            if event_type == "TOOL_CALL_START":
                tool_states[tool_id] = {"started": True, "ended": False}
            elif event_type == "TOOL_CALL_END":
                if tool_id in tool_states:
                    tool_states[tool_id]["ended"] = True
            elif event_type == "TOOL_CALL_ARGS":
                # ARGS 必须在 START 之后，END 之前
                assert (
                    tool_id in tool_states
                ), f"ARGS for {tool_id} without START"
                assert not tool_states[tool_id][
                    "ended"
                ], f"ARGS for {tool_id} after END"

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_standard_mode(self):
        """测试标准模式下的并行工具调用

        场景：默认模式（copilotkit_compatibility=False）允许并行工具调用
        预期：tc-2 START 可以在 tc-1 END 之前发送
        """

        async def invoke_agent(request: AgentRequest):
            # 第一个工具调用开始
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": '{"a": 1}'},
            )
            # 第二个工具调用开始（并行）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-2", "name": "tool2", "args_delta": '{"b": 2}'},
            )
            # 结果
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "result1"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-2", "result": "result2"},
            )

        # 使用默认配置（标准模式）
        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "parallel"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证两个工具调用都存在
        assert types.count("TOOL_CALL_START") == 2
        assert types.count("TOOL_CALL_END") == 2
        assert types.count("TOOL_CALL_RESULT") == 2

        # 验证并行工具调用的顺序：
        # tc-1 START -> tc-1 ARGS -> tc-2 START -> tc-2 ARGS -> tc-1 END -> tc-1 RESULT -> tc-2 END -> tc-2 RESULT
        # 关键验证：tc-2 START 在 tc-1 END 之前（并行）
        tc1_end_idx = None
        tc2_start_idx = None
        for i, line in enumerate(lines):
            if "TOOL_CALL_END" in line and "tc-1" in line:
                tc1_end_idx = i
            if "TOOL_CALL_START" in line and "tc-2" in line:
                tc2_start_idx = i

        assert tc1_end_idx is not None, "tc-1 TOOL_CALL_END not found"
        assert tc2_start_idx is not None, "tc-2 TOOL_CALL_START not found"
        # 并行工具调用：tc-2 START 应该在 tc-1 END 之前
        assert (
            tc2_start_idx < tc1_end_idx
        ), "Parallel tool calls: tc-2 START should come before tc-1 END"

    @pytest.mark.asyncio
    async def test_langchain_duplicate_tool_call_with_uuid_id_copilotkit_mode(
        self,
    ):
        """测试 LangChain 重复工具调用（使用 UUID 格式 ID）- CopilotKit 兼容模式

        场景：模拟 LangChain 流式输出时可能产生的重复事件：
        - 流式 chunk 使用 call_xxx ID
        - on_tool_start 使用 UUID 格式的 run_id
        - 两者对应同一个逻辑工具调用

        预期：在 copilotkit_compatibility=True 模式下，UUID 格式的重复事件应该被忽略或合并到已有的 call_xxx ID
        """
        from agentrun.server import AGUIProtocolConfig, ServerConfig

        async def invoke_agent(request: AgentRequest):
            # 流式 chunk：使用 call_xxx ID
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "call_abc123",
                    "name": "get_weather",
                    "args_delta": '{"city": "Beijing"}',
                },
            )
            # on_tool_start：使用 UUID 格式的 run_id（同一个工具调用）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "name": "get_weather",
                    "args_delta": '{"city": "Beijing"}',
                },
            )
            # on_tool_end：使用 UUID 格式的 run_id
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={
                    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "name": "get_weather",
                    "result": "Sunny, 25°C",
                },
            )

        # 启用 CopilotKit 兼容模式
        config = ServerConfig(
            agui=AGUIProtocolConfig(copilotkit_compatibility=True)
        )
        server = AgentRunServer(invoke_agent=invoke_agent, config=config)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "weather"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证只有一个工具调用（UUID 重复事件被合并）
        assert (
            types.count("TOOL_CALL_START") == 1
        ), f"Expected 1 TOOL_CALL_START, got {types.count('TOOL_CALL_START')}"
        assert (
            types.count("TOOL_CALL_END") == 1
        ), f"Expected 1 TOOL_CALL_END, got {types.count('TOOL_CALL_END')}"
        assert (
            types.count("TOOL_CALL_RESULT") == 1
        ), f"Expected 1 TOOL_CALL_RESULT, got {types.count('TOOL_CALL_RESULT')}"

        # 验证使用的是 call_xxx ID，不是 UUID
        for line in lines:
            if "TOOL_CALL_START" in line:
                data = json.loads(line.replace("data: ", ""))
                assert (
                    data["toolCallId"] == "call_abc123"
                ), f"Expected call_abc123, got {data['toolCallId']}"
            if "TOOL_CALL_RESULT" in line:
                data = json.loads(line.replace("data: ", ""))
                assert (
                    data["toolCallId"] == "call_abc123"
                ), f"Expected call_abc123, got {data['toolCallId']}"

    # @pytest.mark.asyncio
    # async def test_langchain_multiple_tools_with_uuid_ids_copilotkit_mode(self):
    #     """测试 LangChain 多个工具调用（使用 UUID 格式 ID）- CopilotKit 兼容模式

    #     场景：模拟 LangChain 并行调用多个工具时的事件序列：
    #     - 流式 chunk 使用 call_xxx ID
    #     - on_tool_start/end 使用 UUID 格式的 run_id
    #     - 需要正确匹配每个工具的 ID

    #     预期：在 copilotkit_compatibility=True 模式下，每个工具调用应该使用正确的 ID，不会混淆
    #     """
    #     from agentrun.server import AGUIProtocolConfig, ServerConfig

    #     async def invoke_agent(request: AgentRequest):
    #         # 第一个工具的流式 chunk
    #         yield AgentEvent(
    #             event=EventType.TOOL_CALL_CHUNK,
    #             data={
    #                 "id": "call_tool1",
    #                 "name": "get_weather",
    #                 "args_delta": '{"city": "Beijing"}',
    #             },
    #         )
    #         # 第二个工具的流式 chunk
    #         yield AgentEvent(
    #             event=EventType.TOOL_CALL_CHUNK,
    #             data={
    #                 "id": "call_tool2",
    #                 "name": "get_time",
    #                 "args_delta": '{"timezone": "UTC"}',
    #             },
    #         )
    #         # 第一个工具的 on_tool_start（UUID）
    #         yield AgentEvent(
    #             event=EventType.TOOL_CALL,  # 改为 TOOL_CALL
    #             data={
    #                 "id": "uuid-weather-123",
    #                 "name": "get_weather",
    #                 "args": '{"city": "Beijing"}',
    #             },
    #         )
    #         # 第二个工具的 on_tool_start（UUID）
    #         yield AgentEvent(
    #             event=EventType.TOOL_CALL,  # 改为 TOOL_CALL
    #             data={
    #                 "id": "uuid-time-456",
    #                 "name": "get_time",
    #                 "args": '{"timezone": "UTC"}',
    #             },
    #         )
    #         # 第一个工具的 on_tool_end（UUID）
    #         yield AgentEvent(
    #             event=EventType.TOOL_RESULT,
    #             data={
    #                 "id": "uuid-weather-123",
    #                 "name": "get_weather",
    #                 "result": "Sunny, 25°C",
    #             },
    #         )
    #         # 第二个工具的 on_tool_end（UUID）
    #         yield AgentEvent(
    #             event=EventType.TOOL_RESULT,
    #             data={
    #                 "id": "uuid-time-456",
    #                 "name": "get_time",
    #                 "result": "12:00 UTC",
    #             },
    #         )

    #     # 启用 CopilotKit 兼容模式
    #     config = ServerConfig(
    #         agui=AGUIProtocolConfig(copilotkit_compatibility=True)
    #     )
    #     server = AgentRunServer(invoke_agent=invoke_agent, config=config)
    #     app = server.as_fastapi_app()
    #     from fastapi.testclient import TestClient

    #     client = TestClient(app)
    #     response = client.post(
    #         "/ag-ui/agent",
    #         json={"messages": [{"role": "user", "content": "tools"}]},
    #     )

    #     lines = [line async for line in response.aiter_lines() if line]
    #     types = get_event_types(lines)

    #     # 验证只有两个工具调用（UUID 重复事件被合并）
    #     assert (
    #         types.count("TOOL_CALL_START") == 2
    #     ), f"Expected 2 TOOL_CALL_START, got {types.count('TOOL_CALL_START')}"
    #     assert (
    #         types.count("TOOL_CALL_END") == 2
    #     ), f"Expected 2 TOOL_CALL_END, got {types.count('TOOL_CALL_END')}"
    #     assert (
    #         types.count("TOOL_CALL_RESULT") == 2
    #     ), f"Expected 2 TOOL_CALL_RESULT, got {types.count('TOOL_CALL_RESULT')}"

    #     # 验证使用的是 call_xxx ID，不是 UUID
    #     tool_call_ids = []
    #     for line in lines:
    #         if "TOOL_CALL_START" in line:
    #             data = json.loads(line.replace("data: ", ""))
    #             tool_call_ids.append(data["toolCallId"])

    #     assert (
    #         "call_tool1" in tool_call_ids or "call_tool2" in tool_call_ids
    #     ), f"Expected call_xxx IDs, got {tool_call_ids}"

    @pytest.mark.asyncio
    async def test_langchain_uuid_not_deduplicated_standard_mode(self):
        """测试标准模式下，UUID 格式 ID 不会被去重

        场景：默认模式（copilotkit_compatibility=False）下，UUID 格式的 ID 应该被视为独立的工具调用
        """

        async def invoke_agent(request: AgentRequest):
            # 流式 chunk：使用 call_xxx ID
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "call_abc123",
                    "name": "get_weather",
                    "args_delta": '{"city": "Beijing"}',
                },
            )
            # on_tool_start：使用 UUID 格式的 run_id（同一个工具调用）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "name": "get_weather",
                    "args_delta": '{"city": "Beijing"}',
                },
            )
            # 两个工具的结果
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={
                    "id": "call_abc123",
                    "name": "get_weather",
                    "result": "Sunny, 25°C",
                },
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={
                    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "name": "get_weather",
                    "result": "Sunny, 25°C",
                },
            )

        # 使用默认配置（标准模式）
        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "weather"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 禁用串行化时，UUID 格式的 ID 不会被去重，所以有两个工具调用
        assert (
            types.count("TOOL_CALL_START") == 2
        ), f"Expected 2 TOOL_CALL_START, got {types.count('TOOL_CALL_START')}"
        assert (
            types.count("TOOL_CALL_END") == 2
        ), f"Expected 2 TOOL_CALL_END, got {types.count('TOOL_CALL_END')}"
        assert (
            types.count("TOOL_CALL_RESULT") == 2
        ), f"Expected 2 TOOL_CALL_RESULT, got {types.count('TOOL_CALL_RESULT')}"

    @pytest.mark.asyncio
    async def test_tool_result_after_another_tool_started(self):
        """测试在另一个工具调用开始后收到之前工具的 RESULT

        场景：
        1. tc-1 START, ARGS, END
        2. tc-2 START, ARGS
        3. tc-1 ARGS (tc-1 已结束，会重新开始)
        4. tc-1 RESULT (此时 tc-1 是活跃的，需要先结束)

        预期：在发送 tc-1 RESULT 前，应该先结束所有活跃的工具调用
        """

        async def invoke_agent(request: AgentRequest):
            # tc-1 开始
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "call_tc1",
                    "name": "tool1",
                    "args_delta": '{"a": 1}',
                },
            )
            # tc-2 开始（会自动结束 tc-1）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "call_tc2",
                    "name": "tool2",
                    "args_delta": '{"b": 2}',
                },
            )
            # tc-1 的额外 ARGS（tc-1 已结束，会重新开始）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "call_tc1", "name": "tool1", "args_delta": ""},
            )
            # tc-1 的 RESULT（此时 tc-1 是活跃的）
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "call_tc1", "result": "result1"},
            )
            # tc-2 的 RESULT
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "call_tc2", "result": "result2"},
            )

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        lines = [line async for line in response.aiter_lines() if line]
        types = get_event_types(lines)

        # 验证事件序列
        assert types[0] == "RUN_STARTED"
        assert types[-1] == "RUN_FINISHED"

        # 验证没有连续的 TOOL_CALL_START 和 TOOL_CALL_RESULT（中间应该有 TOOL_CALL_END）
        for i in range(len(types) - 1):
            if types[i] == "TOOL_CALL_START":
                # 下一个不应该是 TOOL_CALL_RESULT
                assert types[i + 1] != "TOOL_CALL_RESULT", (
                    "TOOL_CALL_RESULT should not immediately follow"
                    f" TOOL_CALL_START at index {i}"
                )

        # 验证所有 TOOL_CALL_RESULT 之前都有对应的 TOOL_CALL_END
        assert types.count("TOOL_CALL_RESULT") == 2
        assert types.count("TOOL_CALL_END") >= types.count("TOOL_CALL_RESULT")
