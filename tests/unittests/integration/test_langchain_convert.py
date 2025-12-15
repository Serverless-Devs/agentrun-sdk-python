"""测试 convert 函数 / Test convert Function

测试 convert 函数对不同 LangChain/LangGraph 调用方式返回事件格式的兼容性。
支持的格式：
- astream_events(version="v2") 格式
- stream/astream(stream_mode="updates") 格式
- stream/astream(stream_mode="values") 格式
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agentrun.integration.langgraph.agent_converter import (
    _is_astream_events_format,
    _is_stream_updates_format,
    _is_stream_values_format,
    convert,
)
from agentrun.server.model import AgentResult, EventType

# =============================================================================
# Mock 数据：模拟 LangChain/LangGraph 返回的事件格式
# =============================================================================


def create_mock_ai_message(
    content: str, tool_calls: List[Dict[str, Any]] = None
):
    """创建模拟的 AIMessage 对象"""
    msg = MagicMock()
    msg.content = content
    msg.type = "ai"
    msg.tool_calls = tool_calls or []
    return msg


def create_mock_ai_message_chunk(
    content: str, tool_call_chunks: List[Dict] = None
):
    """创建模拟的 AIMessageChunk 对象"""
    chunk = MagicMock()
    chunk.content = content
    chunk.tool_call_chunks = tool_call_chunks or []
    return chunk


def create_mock_tool_message(content: str, tool_call_id: str):
    """创建模拟的 ToolMessage 对象"""
    msg = MagicMock()
    msg.content = content
    msg.type = "tool"
    msg.tool_call_id = tool_call_id
    return msg


# =============================================================================
# 测试事件格式检测函数
# =============================================================================


class TestEventFormatDetection:
    """测试事件格式检测函数"""

    def test_is_astream_events_format(self):
        """测试 astream_events 格式检测"""
        # 正确的 astream_events 格式
        assert _is_astream_events_format(
            {"event": "on_chat_model_stream", "data": {}}
        )
        assert _is_astream_events_format({"event": "on_tool_start", "data": {}})
        assert _is_astream_events_format({"event": "on_tool_end", "data": {}})
        assert _is_astream_events_format(
            {"event": "on_chain_stream", "data": {}}
        )

        # 不是 astream_events 格式
        assert not _is_astream_events_format({"model": {"messages": []}})
        assert not _is_astream_events_format({"messages": []})
        assert not _is_astream_events_format({})
        assert not _is_astream_events_format(
            {"event": "custom_event"}
        )  # 不以 on_ 开头

    def test_is_stream_updates_format(self):
        """测试 stream(updates) 格式检测"""
        # 正确的 updates 格式
        assert _is_stream_updates_format({"model": {"messages": []}})
        assert _is_stream_updates_format({"agent": {"messages": []}})
        assert _is_stream_updates_format({"tools": {"messages": []}})
        assert _is_stream_updates_format(
            {"__end__": {}, "model": {"messages": []}}
        )

        # 不是 updates 格式
        assert not _is_stream_updates_format({"event": "on_chat_model_stream"})
        assert not _is_stream_updates_format(
            {"messages": []}
        )  # 这是 values 格式
        assert not _is_stream_updates_format({})

    def test_is_stream_values_format(self):
        """测试 stream(values) 格式检测"""
        # 正确的 values 格式
        assert _is_stream_values_format({"messages": []})
        assert _is_stream_values_format({"messages": [MagicMock()]})

        # 不是 values 格式
        assert not _is_stream_values_format({"event": "on_chat_model_stream"})
        assert not _is_stream_values_format({"model": {"messages": []}})
        assert not _is_stream_values_format({})


# =============================================================================
# 测试 astream_events 格式的转换
# =============================================================================


class TestConvertAstreamEventsFormat:
    """测试 astream_events 格式的事件转换"""

    def test_on_chat_model_stream_text_content(self):
        """测试 on_chat_model_stream 事件的文本内容提取"""
        chunk = create_mock_ai_message_chunk("你好")
        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": chunk},
        }

        results = list(convert(event))

        assert len(results) == 1
        assert results[0] == "你好"

    def test_on_chat_model_stream_empty_content(self):
        """测试 on_chat_model_stream 事件的空内容"""
        chunk = create_mock_ai_message_chunk("")
        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": chunk},
        }

        results = list(convert(event))
        assert len(results) == 0

    def test_on_chat_model_stream_with_tool_call_args(self):
        """测试 on_chat_model_stream 事件的工具调用参数"""
        chunk = create_mock_ai_message_chunk(
            "",
            tool_call_chunks=[{
                "id": "call_123",
                "name": "get_weather",
                "args": '{"city": "北京"}',
            }],
        )
        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": chunk},
        }

        results = list(convert(event))

        assert len(results) == 1
        assert isinstance(results[0], AgentResult)
        assert results[0].event == EventType.TOOL_CALL_ARGS
        assert results[0].data["tool_call_id"] == "call_123"
        assert results[0].data["delta"] == '{"city": "北京"}'

    def test_on_tool_start(self):
        """测试 on_tool_start 事件"""
        event = {
            "event": "on_tool_start",
            "name": "get_weather",
            "run_id": "run_456",
            "data": {"input": {"city": "北京"}},
        }

        results = list(convert(event))

        assert len(results) == 2

        # TOOL_CALL_START
        assert isinstance(results[0], AgentResult)
        assert results[0].event == EventType.TOOL_CALL_START
        assert results[0].data["tool_call_id"] == "run_456"
        assert results[0].data["tool_call_name"] == "get_weather"

        # TOOL_CALL_ARGS
        assert isinstance(results[1], AgentResult)
        assert results[1].event == EventType.TOOL_CALL_ARGS
        assert results[1].data["tool_call_id"] == "run_456"
        assert "city" in results[1].data["delta"]

    def test_on_tool_start_without_input(self):
        """测试 on_tool_start 事件（无输入参数）"""
        event = {
            "event": "on_tool_start",
            "name": "get_time",
            "run_id": "run_789",
            "data": {},
        }

        results = list(convert(event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_START
        assert results[0].data["tool_call_id"] == "run_789"
        assert results[0].data["tool_call_name"] == "get_time"

    def test_on_tool_end(self):
        """测试 on_tool_end 事件"""
        event = {
            "event": "on_tool_end",
            "run_id": "run_456",
            "data": {"output": {"weather": "晴天", "temperature": 25}},
        }

        results = list(convert(event))

        assert len(results) == 2

        # TOOL_CALL_RESULT
        assert results[0].event == EventType.TOOL_CALL_RESULT
        assert results[0].data["tool_call_id"] == "run_456"
        assert "晴天" in results[0].data["result"]

        # TOOL_CALL_END
        assert results[1].event == EventType.TOOL_CALL_END
        assert results[1].data["tool_call_id"] == "run_456"

    def test_on_tool_end_with_string_output(self):
        """测试 on_tool_end 事件（字符串输出）"""
        event = {
            "event": "on_tool_end",
            "run_id": "run_456",
            "data": {"output": "晴天，25度"},
        }

        results = list(convert(event))

        assert len(results) == 2
        assert results[0].event == EventType.TOOL_CALL_RESULT
        assert results[0].data["result"] == "晴天，25度"

    def test_on_tool_start_with_non_jsonable_args(self):
        """工具输入包含不可 JSON 序列化对象时也能正常转换"""

        class Dummy:

            def __str__(self):
                return "dummy_obj"

        event = {
            "event": "on_tool_start",
            "name": "get_weather",
            "run_id": "run_non_json",
            "data": {"input": {"obj": Dummy()}},
        }

        results = list(convert(event))

        # TOOL_CALL_START + TOOL_CALL_ARGS
        assert len(results) == 2
        assert results[0].event == EventType.TOOL_CALL_START
        assert results[0].data["tool_call_id"] == "run_non_json"
        assert results[1].event == EventType.TOOL_CALL_ARGS
        assert results[1].data["tool_call_id"] == "run_non_json"
        assert "dummy_obj" in results[1].data["delta"]

    def test_on_tool_start_filters_internal_runtime_field(self):
        """测试 on_tool_start 过滤 MCP 注入的 runtime 等内部字段"""

        class FakeToolRuntime:
            """模拟 MCP 的 ToolRuntime 对象"""

            def __str__(self):
                return "ToolRuntime(...huge internal state...)"

        event = {
            "event": "on_tool_start",
            "name": "maps_weather",
            "run_id": "run_mcp_tool",
            "data": {
                "input": {
                    "city": "北京",  # 用户实际参数
                    "runtime": FakeToolRuntime(),  # MCP 注入的内部字段
                    "config": {"internal": "state"},  # 另一个内部字段
                    "__pregel_runtime": "internal",  # LangGraph 内部字段
                }
            },
        }

        results = list(convert(event))

        # TOOL_CALL_START + TOOL_CALL_ARGS
        assert len(results) == 2
        assert results[0].event == EventType.TOOL_CALL_START
        assert results[0].data["tool_call_name"] == "maps_weather"

        assert results[1].event == EventType.TOOL_CALL_ARGS
        delta = results[1].data["delta"]
        # 应该只包含用户参数 city
        assert "北京" in delta
        # 不应该包含内部字段
        assert "runtime" not in delta.lower() or "ToolRuntime" not in delta
        assert "internal" not in delta
        assert "__pregel" not in delta

    def test_on_tool_start_uses_runtime_tool_call_id(self):
        """测试 on_tool_start 使用 runtime 中的原始 tool_call_id 而非 run_id

        MCP 工具会在 input.runtime 中注入 tool_call_id，这是 LLM 返回的原始 ID。
        应该优先使用这个 ID，以保证工具调用事件的 ID 一致性。
        """

        class FakeToolRuntime:
            """模拟 MCP 的 ToolRuntime 对象"""

            def __init__(self, tool_call_id: str):
                self.tool_call_id = tool_call_id

        original_tool_call_id = "call_original_from_llm_12345"

        event = {
            "event": "on_tool_start",
            "name": "get_weather",
            "run_id": (
                "run_id_different_from_tool_call_id"
            ),  # run_id 与 tool_call_id 不同
            "data": {
                "input": {
                    "city": "北京",
                    "runtime": FakeToolRuntime(original_tool_call_id),
                }
            },
        }

        results = list(convert(event))

        # TOOL_CALL_START + TOOL_CALL_ARGS
        assert len(results) == 2

        # 应该使用 runtime 中的原始 tool_call_id，而不是 run_id
        assert results[0].event == EventType.TOOL_CALL_START
        assert results[0].data["tool_call_id"] == original_tool_call_id
        assert results[0].data["tool_call_name"] == "get_weather"

        assert results[1].event == EventType.TOOL_CALL_ARGS
        assert results[1].data["tool_call_id"] == original_tool_call_id

    def test_on_tool_end_uses_runtime_tool_call_id(self):
        """测试 on_tool_end 使用 runtime 中的原始 tool_call_id 而非 run_id"""

        class FakeToolRuntime:
            """模拟 MCP 的 ToolRuntime 对象"""

            def __init__(self, tool_call_id: str):
                self.tool_call_id = tool_call_id

        original_tool_call_id = "call_original_from_llm_67890"

        event = {
            "event": "on_tool_end",
            "run_id": "run_id_different_from_tool_call_id",
            "data": {
                "output": {"weather": "晴天", "temp": 25},
                "input": {
                    "city": "北京",
                    "runtime": FakeToolRuntime(original_tool_call_id),
                },
            },
        }

        results = list(convert(event))

        # TOOL_CALL_RESULT + TOOL_CALL_END
        assert len(results) == 2

        # 应该使用 runtime 中的原始 tool_call_id
        assert results[0].event == EventType.TOOL_CALL_RESULT
        assert results[0].data["tool_call_id"] == original_tool_call_id

        assert results[1].event == EventType.TOOL_CALL_END
        assert results[1].data["tool_call_id"] == original_tool_call_id

    def test_on_tool_start_fallback_to_run_id(self):
        """测试当 runtime 中没有 tool_call_id 时，回退使用 run_id"""
        event = {
            "event": "on_tool_start",
            "name": "get_time",
            "run_id": "run_789",
            "data": {"input": {"timezone": "Asia/Shanghai"}},  # 没有 runtime
        }

        results = list(convert(event))

        assert len(results) == 2
        assert results[0].event == EventType.TOOL_CALL_START
        # 应该回退使用 run_id
        assert results[0].data["tool_call_id"] == "run_789"
        assert results[1].data["tool_call_id"] == "run_789"

    def test_streaming_tool_call_id_consistency_with_map(self):
        """测试流式工具调用的 tool_call_id 一致性（使用映射）

        在流式工具调用中：
        - 第一个 chunk 有 id 但可能没有 args（用于建立映射）
        - 后续 chunk 有 args 但 id 为空，只有 index（从映射查找 id）

        使用 tool_call_id_map 可以确保 ID 一致性。
        """
        # 模拟流式工具调用的多个 chunk
        events = [
            # 第一个 chunk: 有 id 和 name，没有 args（只用于建立映射）
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[{
                            "id": "call_abc123",
                            "name": "browser_navigate",
                            "args": "",
                            "index": 0,
                        }],
                    )
                },
            },
            # 第二个 chunk: id 为空，只有 index 和 args
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[{
                            "id": "",
                            "name": "",
                            "args": '{"url": "https://',
                            "index": 0,
                        }],
                    )
                },
            },
            # 第三个 chunk: id 为空，继续 args
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[{
                            "id": "",
                            "name": "",
                            "args": 'example.com"}',
                            "index": 0,
                        }],
                    )
                },
            },
        ]

        # 使用 tool_call_id_map 来确保 ID 一致性
        tool_call_id_map: Dict[int, str] = {}
        all_results = []

        for event in events:
            results = list(convert(event, tool_call_id_map=tool_call_id_map))
            all_results.extend(results)

        # 验证映射已建立
        assert 0 in tool_call_id_map
        assert tool_call_id_map[0] == "call_abc123"

        # 验证：所有 TOOL_CALL_ARGS 都使用相同的 tool_call_id
        args_events = [
            r
            for r in all_results
            if isinstance(r, AgentResult)
            and r.event == EventType.TOOL_CALL_ARGS
        ]

        # 应该有 2 个 TOOL_CALL_ARGS 事件（第一个没有 args 不生成事件）
        assert len(args_events) == 2

        # 所有事件应该使用相同的 tool_call_id（从映射获取）
        for event in args_events:
            assert event.data["tool_call_id"] == "call_abc123"

    def test_streaming_tool_call_id_without_map_uses_index(self):
        """测试不使用映射时，后续 chunk 回退到 index"""
        event = {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": MagicMock(
                    content="",
                    tool_call_chunks=[{
                        "id": "",
                        "name": "",
                        "args": '{"url": "test"}',
                        "index": 0,
                    }],
                )
            },
        }

        # 不传入 tool_call_id_map
        results = list(convert(event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_ARGS
        # 回退使用 index
        assert results[0].data["tool_call_id"] == "0"

    def test_streaming_multiple_concurrent_tool_calls(self):
        """测试多个并发工具调用（不同 index）的 ID 一致性"""
        # 模拟 LLM 同时调用两个工具
        events = [
            # 第一个 chunk: 两个工具调用的 ID
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[
                            {
                                "id": "call_tool1",
                                "name": "search",
                                "args": "",
                                "index": 0,
                            },
                            {
                                "id": "call_tool2",
                                "name": "weather",
                                "args": "",
                                "index": 1,
                            },
                        ],
                    )
                },
            },
            # 后续 chunk: 只有 index 和 args
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[
                            {
                                "id": "",
                                "name": "",
                                "args": '{"q": "test"',
                                "index": 0,
                            },
                        ],
                    )
                },
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[
                            {
                                "id": "",
                                "name": "",
                                "args": '{"city": "北京"',
                                "index": 1,
                            },
                        ],
                    )
                },
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[
                            {"id": "", "name": "", "args": "}", "index": 0},
                            {"id": "", "name": "", "args": "}", "index": 1},
                        ],
                    )
                },
            },
        ]

        tool_call_id_map: Dict[int, str] = {}
        all_results = []

        for event in events:
            results = list(convert(event, tool_call_id_map=tool_call_id_map))
            all_results.extend(results)

        # 验证映射正确建立
        assert tool_call_id_map[0] == "call_tool1"
        assert tool_call_id_map[1] == "call_tool2"

        # 验证所有事件使用正确的 ID
        args_events = [
            r
            for r in all_results
            if isinstance(r, AgentResult)
            and r.event == EventType.TOOL_CALL_ARGS
        ]

        # 应该有 4 个 TOOL_CALL_ARGS 事件
        assert len(args_events) == 4

        # 验证每个工具调用使用正确的 ID
        tool1_args = [
            e for e in args_events if e.data["tool_call_id"] == "call_tool1"
        ]
        tool2_args = [
            e for e in args_events if e.data["tool_call_id"] == "call_tool2"
        ]

        assert len(tool1_args) == 2  # '{"q": "test"' 和 '}'
        assert len(tool2_args) == 2  # '{"city": "北京"' 和 '}'

    def test_agentrun_converter_class(self):
        """测试 AgentRunConverter 类的完整功能"""
        from agentrun.integration.langchain import AgentRunConverter

        events = [
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[{
                            "id": "call_xyz",
                            "name": "test_tool",
                            "args": "",
                            "index": 0,
                        }],
                    )
                },
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[{
                            "id": "",
                            "name": "",
                            "args": '{"key": "value"}',
                            "index": 0,
                        }],
                    )
                },
            },
        ]

        converter = AgentRunConverter()
        all_results = []

        for event in events:
            results = list(converter.convert(event))
            all_results.extend(results)

        # 验证内部映射
        assert converter._tool_call_id_map[0] == "call_xyz"

        # 验证结果
        args_events = [
            r
            for r in all_results
            if isinstance(r, AgentResult)
            and r.event == EventType.TOOL_CALL_ARGS
        ]
        assert len(args_events) == 1
        assert args_events[0].data["tool_call_id"] == "call_xyz"

        # 测试 reset
        converter.reset()
        assert len(converter._tool_call_id_map) == 0

    def test_streaming_tool_call_with_first_chunk_having_args(self):
        """测试第一个 chunk 同时有 id 和 args 的情况"""
        # 有些模型可能在第一个 chunk 就返回完整的工具调用
        event = {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": MagicMock(
                    content="",
                    tool_call_chunks=[{
                        "id": "call_complete",
                        "name": "simple_tool",
                        "args": '{"done": true}',
                        "index": 0,
                    }],
                )
            },
        }

        tool_call_id_map: Dict[int, str] = {}
        results = list(convert(event, tool_call_id_map=tool_call_id_map))

        # 验证映射被建立
        assert tool_call_id_map[0] == "call_complete"

        # 验证 TOOL_CALL_ARGS 使用正确的 ID
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_ARGS
        assert results[0].data["tool_call_id"] == "call_complete"

    def test_streaming_tool_call_id_none_vs_empty_string(self):
        """测试 id 为 None 和空字符串的不同处理"""
        events = [
            # id 为 None（建立映射）
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[{
                            "id": "call_from_none",
                            "name": "tool",
                            "args": "",
                            "index": 0,
                        }],
                    )
                },
            },
            # id 为 None（应该从映射获取）
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[{
                            "id": None,
                            "name": "",
                            "args": '{"a": 1}',
                            "index": 0,
                        }],
                    )
                },
            },
        ]

        tool_call_id_map: Dict[int, str] = {}
        all_results = []

        for event in events:
            results = list(convert(event, tool_call_id_map=tool_call_id_map))
            all_results.extend(results)

        args_events = [
            r
            for r in all_results
            if isinstance(r, AgentResult)
            and r.event == EventType.TOOL_CALL_ARGS
        ]

        assert len(args_events) == 1
        # None 应该被当作 falsy，从映射获取 ID
        assert args_events[0].data["tool_call_id"] == "call_from_none"

    def test_full_tool_call_flow_id_consistency(self):
        """测试完整工具调用流程中的 ID 一致性

        模拟：
        1. on_chat_model_stream 产生 TOOL_CALL_ARGS
        2. on_tool_start 产生 TOOL_CALL_START
        3. on_tool_end 产生 TOOL_CALL_RESULT 和 TOOL_CALL_END

        验证所有事件使用相同的 tool_call_id
        """
        from agentrun.integration.langchain import AgentRunConverter

        # 模拟完整的工具调用流程
        events = [
            # 流式工具调用参数
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[{
                            "id": "call_full_flow",
                            "name": "test_tool",
                            "args": "",
                            "index": 0,
                        }],
                    )
                },
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[{
                            "id": "",
                            "name": "",
                            "args": '{"param": "value"}',
                            "index": 0,
                        }],
                    )
                },
            },
            # 工具开始（使用 runtime.tool_call_id）
            {
                "event": "on_tool_start",
                "name": "test_tool",
                "run_id": "run_123",
                "data": {
                    "input": {
                        "param": "value",
                        "runtime": MagicMock(tool_call_id="call_full_flow"),
                    }
                },
            },
            # 工具结束
            {
                "event": "on_tool_end",
                "run_id": "run_123",
                "data": {
                    "input": {
                        "param": "value",
                        "runtime": MagicMock(tool_call_id="call_full_flow"),
                    },
                    "output": "success",
                },
            },
        ]

        converter = AgentRunConverter()
        all_results = []

        for event in events:
            results = list(converter.convert(event))
            all_results.extend(results)

        # 获取所有工具调用相关事件
        tool_events = [
            r
            for r in all_results
            if isinstance(r, AgentResult)
            and r.event
            in [
                EventType.TOOL_CALL_ARGS,
                EventType.TOOL_CALL_START,
                EventType.TOOL_CALL_RESULT,
                EventType.TOOL_CALL_END,
            ]
        ]

        # 验证所有事件都使用相同的 tool_call_id
        for event in tool_events:
            assert event.data["tool_call_id"] == "call_full_flow", (
                f"Event {event.event} has wrong tool_call_id:"
                f" {event.data['tool_call_id']}"
            )

        # 验证事件顺序
        event_types = [e.event for e in tool_events]
        assert EventType.TOOL_CALL_ARGS in event_types
        assert EventType.TOOL_CALL_START in event_types
        assert EventType.TOOL_CALL_RESULT in event_types
        assert EventType.TOOL_CALL_END in event_types

    def test_on_chain_stream_model_node(self):
        """测试 on_chain_stream 事件（model 节点）"""
        msg = create_mock_ai_message("你好！有什么可以帮你的吗？")
        event = {
            "event": "on_chain_stream",
            "name": "model",
            "data": {"chunk": {"messages": [msg]}},
        }

        results = list(convert(event))

        assert len(results) == 1
        assert results[0] == "你好！有什么可以帮你的吗？"

    def test_on_chain_stream_non_model_node(self):
        """测试 on_chain_stream 事件（非 model 节点）"""
        event = {
            "event": "on_chain_stream",
            "name": "agent",  # 不是 "model"
            "data": {"chunk": {"messages": []}},
        }

        results = list(convert(event))
        assert len(results) == 0

    def test_on_chat_model_end_ignored(self):
        """测试 on_chat_model_end 事件被忽略（避免重复）"""
        event = {
            "event": "on_chat_model_end",
            "data": {"output": create_mock_ai_message("完成")},
        }

        results = list(convert(event))
        assert len(results) == 0


# =============================================================================
# 测试 stream/astream(stream_mode="updates") 格式的转换
# =============================================================================


class TestConvertStreamUpdatesFormat:
    """测试 stream(updates) 格式的事件转换"""

    def test_ai_message_text_content(self):
        """测试 AI 消息的文本内容"""
        msg = create_mock_ai_message("你好！")
        event = {"model": {"messages": [msg]}}

        results = list(convert(event))

        assert len(results) == 1
        assert results[0] == "你好！"

    def test_ai_message_empty_content(self):
        """测试 AI 消息的空内容"""
        msg = create_mock_ai_message("")
        event = {"model": {"messages": [msg]}}

        results = list(convert(event))
        assert len(results) == 0

    def test_ai_message_with_tool_calls(self):
        """测试 AI 消息包含工具调用"""
        msg = create_mock_ai_message(
            "",
            tool_calls=[{
                "id": "call_abc",
                "name": "get_weather",
                "args": {"city": "上海"},
            }],
        )
        event = {"agent": {"messages": [msg]}}

        results = list(convert(event))

        assert len(results) == 2

        # TOOL_CALL_START
        assert results[0].event == EventType.TOOL_CALL_START
        assert results[0].data["tool_call_id"] == "call_abc"
        assert results[0].data["tool_call_name"] == "get_weather"

        # TOOL_CALL_ARGS
        assert results[1].event == EventType.TOOL_CALL_ARGS
        assert results[1].data["tool_call_id"] == "call_abc"
        assert "上海" in results[1].data["delta"]

    def test_tool_message_result(self):
        """测试工具消息的结果"""
        msg = create_mock_tool_message('{"weather": "多云"}', "call_abc")
        event = {"tools": {"messages": [msg]}}

        results = list(convert(event))

        assert len(results) == 2

        # TOOL_CALL_RESULT
        assert results[0].event == EventType.TOOL_CALL_RESULT
        assert results[0].data["tool_call_id"] == "call_abc"
        assert "多云" in results[0].data["result"]

        # TOOL_CALL_END
        assert results[1].event == EventType.TOOL_CALL_END
        assert results[1].data["tool_call_id"] == "call_abc"

    def test_end_node_ignored(self):
        """测试 __end__ 节点被忽略"""
        event = {"__end__": {"messages": []}}

        results = list(convert(event))
        assert len(results) == 0

    def test_multiple_nodes_in_event(self):
        """测试一个事件中包含多个节点"""
        ai_msg = create_mock_ai_message("正在查询...")
        tool_msg = create_mock_tool_message("查询结果", "call_xyz")
        event = {
            "__end__": {},
            "model": {"messages": [ai_msg]},
            "tools": {"messages": [tool_msg]},
        }

        results = list(convert(event))

        # 应该有 3 个结果：1 个文本 + 1 个 RESULT + 1 个 END
        assert len(results) == 3
        assert results[0] == "正在查询..."
        assert results[1].event == EventType.TOOL_CALL_RESULT
        assert results[2].event == EventType.TOOL_CALL_END

    def test_custom_messages_key(self):
        """测试自定义 messages_key"""
        msg = create_mock_ai_message("自定义消息")
        event = {"model": {"custom_messages": [msg]}}

        # 使用默认 key 应该找不到消息
        results = list(convert(event, messages_key="messages"))
        assert len(results) == 0

        # 使用正确的 key
        results = list(convert(event, messages_key="custom_messages"))
        assert len(results) == 1
        assert results[0] == "自定义消息"


# =============================================================================
# 测试 stream/astream(stream_mode="values") 格式的转换
# =============================================================================


class TestConvertStreamValuesFormat:
    """测试 stream(values) 格式的事件转换"""

    def test_last_ai_message_content(self):
        """测试最后一条 AI 消息的内容"""
        msg1 = create_mock_ai_message("第一条消息")
        msg2 = create_mock_ai_message("最后一条消息")
        event = {"messages": [msg1, msg2]}

        results = list(convert(event))

        # 只处理最后一条消息
        assert len(results) == 1
        assert results[0] == "最后一条消息"

    def test_last_ai_message_with_tool_calls(self):
        """测试最后一条 AI 消息包含工具调用"""
        msg = create_mock_ai_message(
            "",
            tool_calls=[
                {"id": "call_def", "name": "search", "args": {"query": "天气"}}
            ],
        )
        event = {"messages": [msg]}

        results = list(convert(event))

        assert len(results) == 2
        assert results[0].event == EventType.TOOL_CALL_START
        assert results[1].event == EventType.TOOL_CALL_ARGS

    def test_last_tool_message_result(self):
        """测试最后一条工具消息的结果"""
        ai_msg = create_mock_ai_message("之前的消息")
        tool_msg = create_mock_tool_message("工具结果", "call_ghi")
        event = {"messages": [ai_msg, tool_msg]}

        results = list(convert(event))

        # 只处理最后一条消息（工具消息）
        assert len(results) == 2
        assert results[0].event == EventType.TOOL_CALL_RESULT
        assert results[1].event == EventType.TOOL_CALL_END

    def test_empty_messages(self):
        """测试空消息列表"""
        event = {"messages": []}

        results = list(convert(event))
        assert len(results) == 0


# =============================================================================
# 测试 StreamEvent 对象的转换
# =============================================================================


class TestConvertStreamEventObject:
    """测试 StreamEvent 对象（非 dict）的转换"""

    def test_stream_event_object(self):
        """测试 StreamEvent 对象自动转换为 dict"""
        # 模拟 StreamEvent 对象
        chunk = create_mock_ai_message_chunk("Hello")
        stream_event = MagicMock()
        stream_event.event = "on_chat_model_stream"
        stream_event.data = {"chunk": chunk}
        stream_event.name = "model"
        stream_event.run_id = "run_001"

        results = list(convert(stream_event))

        assert len(results) == 1
        assert results[0] == "Hello"


# =============================================================================
# 测试完整流程：模拟多个事件的序列
# =============================================================================


class TestConvertEventSequence:
    """测试完整的事件序列转换"""

    def test_astream_events_full_sequence(self):
        """测试 astream_events 格式的完整事件序列"""
        events = [
            # 1. 开始工具调用
            {
                "event": "on_tool_start",
                "name": "get_weather",
                "run_id": "tool_run_1",
                "data": {"input": {"city": "北京"}},
            },
            # 2. 工具结束
            {
                "event": "on_tool_end",
                "run_id": "tool_run_1",
                "data": {"output": {"weather": "晴天", "temp": 25}},
            },
            # 3. LLM 流式输出
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": create_mock_ai_message_chunk("北京")},
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": create_mock_ai_message_chunk("今天")},
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": create_mock_ai_message_chunk("晴天")},
            },
        ]

        all_results = []
        for event in events:
            all_results.extend(convert(event))

        # 验证结果
        assert len(all_results) == 7

        # 工具调用事件
        assert all_results[0].event == EventType.TOOL_CALL_START
        assert all_results[1].event == EventType.TOOL_CALL_ARGS
        assert all_results[2].event == EventType.TOOL_CALL_RESULT
        assert all_results[3].event == EventType.TOOL_CALL_END

        # 文本内容
        assert all_results[4] == "北京"
        assert all_results[5] == "今天"
        assert all_results[6] == "晴天"

    def test_stream_updates_full_sequence(self):
        """测试 stream(updates) 格式的完整事件序列"""
        events = [
            # 1. Agent 决定调用工具
            {
                "agent": {
                    "messages": [
                        create_mock_ai_message(
                            "",
                            tool_calls=[{
                                "id": "call_001",
                                "name": "get_weather",
                                "args": {"city": "上海"},
                            }],
                        )
                    ]
                }
            },
            # 2. 工具执行结果
            {
                "tools": {
                    "messages": [
                        create_mock_tool_message(
                            '{"weather": "多云"}', "call_001"
                        )
                    ]
                }
            },
            # 3. Agent 最终回复
            {"model": {"messages": [create_mock_ai_message("上海今天多云。")]}},
        ]

        all_results = []
        for event in events:
            all_results.extend(convert(event))

        # 验证结果
        assert len(all_results) == 5

        # 工具调用
        assert all_results[0].event == EventType.TOOL_CALL_START
        assert all_results[0].data["tool_call_name"] == "get_weather"
        assert all_results[1].event == EventType.TOOL_CALL_ARGS

        # 工具结果
        assert all_results[2].event == EventType.TOOL_CALL_RESULT
        assert all_results[3].event == EventType.TOOL_CALL_END

        # 最终回复
        assert all_results[4] == "上海今天多云。"


# =============================================================================
# 测试边界情况
# =============================================================================


class TestConvertEdgeCases:
    """测试边界情况"""

    def test_empty_event(self):
        """测试空事件"""
        results = list(convert({}))
        assert len(results) == 0

    def test_none_values(self):
        """测试 None 值"""
        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": None},
        }
        results = list(convert(event))
        assert len(results) == 0

    def test_invalid_message_type(self):
        """测试无效的消息类型"""
        msg = MagicMock()
        msg.type = "unknown"
        msg.content = "test"
        event = {"model": {"messages": [msg]}}

        results = list(convert(event))
        # unknown 类型不会产生输出
        assert len(results) == 0

    def test_tool_call_without_id(self):
        """测试没有 ID 的工具调用"""
        msg = create_mock_ai_message(
            "",
            tool_calls=[{"name": "test", "args": {}}],  # 没有 id
        )
        event = {"agent": {"messages": [msg]}}

        results = list(convert(event))
        # 没有 id 的工具调用应该被跳过
        assert len(results) == 0

    def test_tool_message_without_tool_call_id(self):
        """测试没有 tool_call_id 的工具消息"""
        msg = MagicMock()
        msg.type = "tool"
        msg.content = "result"
        msg.tool_call_id = None  # 没有 tool_call_id

        event = {"tools": {"messages": [msg]}}

        results = list(convert(event))
        # 没有 tool_call_id 的工具消息应该被跳过
        assert len(results) == 0

    def test_dict_message_format(self):
        """测试字典格式的消息（而非对象）"""
        event = {
            "model": {"messages": [{"type": "ai", "content": "字典格式消息"}]}
        }

        results = list(convert(event))

        assert len(results) == 1
        assert results[0] == "字典格式消息"

    def test_multimodal_content(self):
        """测试多模态内容（list 格式）"""
        chunk = MagicMock()
        chunk.content = [
            {"type": "text", "text": "这是"},
            {"type": "text", "text": "多模态内容"},
        ]
        chunk.tool_call_chunks = []

        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": chunk},
        }

        results = list(convert(event))

        assert len(results) == 1
        assert results[0] == "这是多模态内容"

    def test_output_with_content_attribute(self):
        """测试有 content 属性的工具输出"""
        output = MagicMock()
        output.content = "工具输出内容"

        event = {
            "event": "on_tool_end",
            "run_id": "run_123",
            "data": {"output": output},
        }

        results = list(convert(event))

        assert len(results) == 2
        assert results[0].event == EventType.TOOL_CALL_RESULT
        assert results[0].data["result"] == "工具输出内容"

    def test_unsupported_stream_mode_messages_format(self):
        """测试不支持的 stream_mode='messages' 格式（元组形式）

        stream_mode='messages' 返回 (AIMessageChunk, metadata) 元组，
        不是 dict 格式，to_agui_events 不支持此格式，应该不产生输出。
        """
        # 模拟 stream_mode="messages" 返回的元组格式
        chunk = create_mock_ai_message_chunk("测试内容")
        metadata = {"langgraph_node": "model"}
        event = (chunk, metadata)  # 元组格式

        # 元组格式会被 _event_to_dict 转换为空字典，因此不产生输出
        results = list(convert(event))
        assert len(results) == 0

    def test_unsupported_random_dict_format(self):
        """测试不支持的随机字典格式

        如果传入的 dict 不匹配任何已知格式，应该不产生输出。
        """
        event = {
            "random_key": "random_value",
            "another_key": {"nested": "data"},
        }

        results = list(convert(event))
        assert len(results) == 0
