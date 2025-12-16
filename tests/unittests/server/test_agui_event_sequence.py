"""AG-UI 事件序列测试

全面测试 AG-UI 协议的事件序列规则：

## 核心规则

1. **RUN 生命周期**
   - RUN_STARTED 必须是第一个事件
   - RUN_FINISHED 必须是最后一个事件

2. **TEXT_MESSAGE 规则**
   - 序列：START → CONTENT* → END
   - 发送 TOOL_CALL_START 前必须先 TEXT_MESSAGE_END
   - 发送 RUN_ERROR 前必须先 TEXT_MESSAGE_END
   - 工具调用后继续输出文本需要新的 TEXT_MESSAGE_START

3. **TOOL_CALL 规则**
   - 序列：START → ARGS* → END → RESULT
   - 发送 TEXT_MESSAGE_START 前必须先 TOOL_CALL_END
   - 发送 RUN_ERROR 前必须先 TOOL_CALL_END
   - TOOL_RESULT 前必须先 TOOL_CALL_END

## 测试覆盖矩阵

| 当前状态 | 下一事件 | 预处理 | 测试 |
|---------|----------|--------|------|
| - | TEXT | - | test_pure_text_stream |
| - | TOOL_CALL | - | test_pure_tool_call |
| TEXT_STARTED | TOOL_CALL | TEXT_END | test_text_then_tool_call |
| TOOL_STARTED | TEXT | TOOL_END | test_tool_chunk_then_text_without_result |
| TOOL_ENDED | TEXT | - | test_tool_call_then_text |
| TEXT_ENDED | TEXT | new START | test_text_tool_text |
| TEXT_STARTED | ERROR | TEXT_END | test_text_then_error |
| TOOL_STARTED | ERROR | TOOL_END | test_tool_call_then_error |
| TEXT_STARTED | STATE | - | test_text_then_state |
| TEXT_STARTED | CUSTOM | - | test_text_then_custom |
| - | TOOL_RESULT(直接) | TOOL_START+END | test_tool_result_without_start |
"""

import json
from typing import List

import pytest

from agentrun.server import AgentEvent, AgentRequest, AgentRunServer, EventType


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

        关键点：TEXT_MESSAGE_END 必须在 TOOL_CALL_START 之前
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

        关键点：
        1. 第一段文本在工具调用前关闭
        2. 第二段文本是新的消息（新 messageId）
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

        关键点：TOOL_CALL_END 必须在 TEXT_MESSAGE_START 之前
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

        关键点：RUN_ERROR 前必须先关闭 TEXT_MESSAGE
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

        # 验证 TEXT_MESSAGE_END 在 RUN_ERROR 之前
        text_end_idx = types.index("TEXT_MESSAGE_END")
        error_idx = types.index("RUN_ERROR")
        assert (
            text_end_idx < error_idx
        ), "TEXT_MESSAGE_END must come before RUN_ERROR"

    @pytest.mark.asyncio
    async def test_tool_call_then_error(self):
        """测试 工具调用 → 错误

        关键点：RUN_ERROR 前必须先发送 TOOL_CALL_END
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

        # 验证 TOOL_CALL_END 在 RUN_ERROR 之前
        tool_end_idx = types.index("TOOL_CALL_END")
        error_idx = types.index("RUN_ERROR")
        assert (
            tool_end_idx < error_idx
        ), "TOOL_CALL_END must come before RUN_ERROR"

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

        # 验证文本消息数量
        assert types.count("TEXT_MESSAGE_START") == 3
        assert types.count("TEXT_MESSAGE_END") == 3

        # 验证工具调用数量
        assert types.count("TOOL_CALL_START") == 3
        assert types.count("TOOL_CALL_END") == 3
        assert types.count("TOOL_CALL_RESULT") == 3

        # 验证每个 TEXT_MESSAGE_END 在对应的 TOOL_CALL_START 之前
        for i, t in enumerate(types):
            if t == "TOOL_CALL_START":
                # 找到之前最近的 TEXT_MESSAGE_START
                for j in range(i - 1, -1, -1):
                    if types[j] == "TEXT_MESSAGE_START":
                        # 确保在 TOOL_CALL_START 之前有 TEXT_MESSAGE_END
                        has_end = "TEXT_MESSAGE_END" in types[j:i]
                        assert has_end, (
                            "TEXT_MESSAGE_END must come before TOOL_CALL_START"
                            f" at index {i}"
                        )
                        break

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
        预期：
        1. TEXT_MESSAGE_END 在 TOOL_CALL_START 之前
        2. 系统自动补充 TOOL_CALL_START 和 TOOL_CALL_END
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
        预期：所有 TOOL_CALL_END 在 TEXT_MESSAGE_START 之前
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
        预期：TEXT_MESSAGE_END 和 TOOL_CALL_END 都在 RUN_ERROR 之前
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

        # 验证 TEXT_MESSAGE_END 在 RUN_ERROR 之前
        text_end_idx = types.index("TEXT_MESSAGE_END")
        error_idx = types.index("RUN_ERROR")
        assert text_end_idx < error_idx

        # 验证 TOOL_CALL_END 在 RUN_ERROR 之前
        tool_end_idx = types.index("TOOL_CALL_END")
        assert tool_end_idx < error_idx

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
        1. TEXT_MESSAGE_END 在 RUN_ERROR 之前
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
        assert "TEXT_MESSAGE_END" in types
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
