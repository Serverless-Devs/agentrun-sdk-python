"""测试 convert 函数 / Test convert Function

测试 convert 函数对不同 LangChain/LangGraph 调用方式返回事件格式的兼容性。
支持的格式：
- astream_events(version="v2") 格式
- stream/astream(stream_mode="updates") 格式
- stream/astream(stream_mode="values") 格式
"""

from unittest.mock import MagicMock

import pytest

from agentrun.integration.langgraph.agent_converter import AgentRunConverter
from agentrun.server.model import AgentResult, EventType

# 使用 helpers.py 中的公共 mock 函数
from .helpers import (
    create_mock_ai_message,
    create_mock_ai_message_chunk,
    create_mock_tool_message,
)

# =============================================================================
# 测试事件格式检测函数
# =============================================================================


class TestEventFormatDetection:
    """测试事件格式检测函数"""

    def test_is_astream_events_format(self):
        """测试 astream_events 格式检测"""
        # 正确的 astream_events 格式
        assert AgentRunConverter.is_astream_events_format(
            {"event": "on_chat_model_stream", "data": {}}
        )
        assert AgentRunConverter.is_astream_events_format(
            {"event": "on_tool_start", "data": {}}
        )
        assert AgentRunConverter.is_astream_events_format(
            {"event": "on_tool_end", "data": {}}
        )
        assert AgentRunConverter.is_astream_events_format(
            {"event": "on_chain_stream", "data": {}}
        )

        # 不是 astream_events 格式
        assert not AgentRunConverter.is_astream_events_format(
            {"model": {"messages": []}}
        )
        assert not AgentRunConverter.is_astream_events_format({"messages": []})
        assert not AgentRunConverter.is_astream_events_format({})
        assert not AgentRunConverter.is_astream_events_format(
            {"event": "custom_event"}
        )  # 不以 on_ 开头

    def test_is_stream_updates_format(self):
        """测试 stream(updates) 格式检测"""
        # 正确的 updates 格式
        assert AgentRunConverter.is_stream_updates_format(
            {"model": {"messages": []}}
        )
        assert AgentRunConverter.is_stream_updates_format(
            {"agent": {"messages": []}}
        )
        assert AgentRunConverter.is_stream_updates_format(
            {"tools": {"messages": []}}
        )
        assert AgentRunConverter.is_stream_updates_format(
            {"__end__": {}, "model": {"messages": []}}
        )

        # 不是 updates 格式
        assert not AgentRunConverter.is_stream_updates_format(
            {"event": "on_chat_model_stream"}
        )
        assert not AgentRunConverter.is_stream_updates_format(
            {"messages": []}
        )  # 这是 values 格式
        assert not AgentRunConverter.is_stream_updates_format({})

    def test_is_stream_values_format(self):
        """测试 stream(values) 格式检测"""
        # 正确的 values 格式
        assert AgentRunConverter.is_stream_values_format({"messages": []})
        assert AgentRunConverter.is_stream_values_format(
            {"messages": [MagicMock()]}
        )

        # 不是 values 格式
        assert not AgentRunConverter.is_stream_values_format(
            {"event": "on_chat_model_stream"}
        )
        assert not AgentRunConverter.is_stream_values_format(
            {"model": {"messages": []}}
        )
        assert not AgentRunConverter.is_stream_values_format({})


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

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0] == "你好"

    def test_on_chat_model_stream_empty_content(self):
        """测试 on_chat_model_stream 事件的空内容"""
        chunk = create_mock_ai_message_chunk("")
        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": chunk},
        }

        results = list(AgentRunConverter.to_agui_events(event))
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

        results = list(AgentRunConverter.to_agui_events(event))

        # 第一个 chunk 有 id 和 name 时，发送完整的 TOOL_CALL_CHUNK
        assert len(results) == 1
        assert isinstance(results[0], AgentResult)
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "call_123"
        assert results[0].data["name"] == "get_weather"
        assert results[0].data["args_delta"] == '{"city": "北京"}'

    def test_on_tool_start(self):
        """测试 on_tool_start 事件"""
        event = {
            "event": "on_tool_start",
            "name": "get_weather",
            "run_id": "run_456",
            "data": {"input": {"city": "北京"}},
        }

        results = list(AgentRunConverter.to_agui_events(event))

        # 现在是单个 TOOL_CALL_CHUNK（边界事件由协议层自动处理）
        assert len(results) == 1
        assert isinstance(results[0], AgentResult)
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "run_456"
        assert results[0].data["name"] == "get_weather"
        assert "city" in results[0].data["args_delta"]

    def test_on_tool_start_without_input(self):
        """测试 on_tool_start 事件（无输入参数）"""
        event = {
            "event": "on_tool_start",
            "name": "get_time",
            "run_id": "run_789",
            "data": {},
        }

        results = list(AgentRunConverter.to_agui_events(event))

        # 现在是单个 TOOL_CALL_CHUNK（边界事件由协议层自动处理）
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "run_789"
        assert results[0].data["name"] == "get_time"

    def test_on_tool_end(self):
        """测试 on_tool_end 事件

        AG-UI 协议：TOOL_CALL_END 在 on_tool_start 中发送（参数传输完成时），
        TOOL_CALL_RESULT 在 on_tool_end 中发送（工具执行完成时）。
        """
        event = {
            "event": "on_tool_end",
            "run_id": "run_456",
            "data": {"output": {"weather": "晴天", "temperature": 25}},
        }

        results = list(AgentRunConverter.to_agui_events(event))

        # on_tool_end 只发送 TOOL_CALL_RESULT
        assert len(results) == 1

        # TOOL_CALL_RESULT
        assert results[0].event == EventType.TOOL_RESULT
        assert results[0].data["id"] == "run_456"
        assert "晴天" in results[0].data["result"]

    def test_on_tool_end_with_string_output(self):
        """测试 on_tool_end 事件（字符串输出）"""
        event = {
            "event": "on_tool_end",
            "run_id": "run_456",
            "data": {"output": "晴天，25度"},
        }

        results = list(AgentRunConverter.to_agui_events(event))

        # on_tool_end 只发送 TOOL_CALL_RESULT
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_RESULT
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

        results = list(AgentRunConverter.to_agui_events(event))

        # 现在是单个 TOOL_CALL_CHUNK
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "run_non_json"
        assert "dummy_obj" in results[0].data["args_delta"]

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

        results = list(AgentRunConverter.to_agui_events(event))

        # 现在是单个 TOOL_CALL_CHUNK
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["name"] == "maps_weather"

        delta = results[0].data["args_delta"]
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

        results = list(AgentRunConverter.to_agui_events(event))

        # 现在是单个 TOOL_CALL_CHUNK
        assert len(results) == 1

        # 应该使用 runtime 中的原始 tool_call_id，而不是 run_id
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == original_tool_call_id
        assert results[0].data["name"] == "get_weather"

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

        results = list(AgentRunConverter.to_agui_events(event))

        # on_tool_end 只发送 TOOL_CALL_RESULT（TOOL_CALL_END 在 on_tool_start 发送）
        assert len(results) == 1

        # 应该使用 runtime 中的原始 tool_call_id
        assert results[0].event == EventType.TOOL_RESULT
        assert results[0].data["id"] == original_tool_call_id

    def test_on_tool_start_fallback_to_run_id(self):
        """测试当 runtime 中没有 tool_call_id 时，回退使用 run_id"""
        event = {
            "event": "on_tool_start",
            "name": "get_time",
            "run_id": "run_789",
            "data": {"input": {"timezone": "Asia/Shanghai"}},  # 没有 runtime
        }

        results = list(AgentRunConverter.to_agui_events(event))

        # 现在是单个 TOOL_CALL_CHUNK
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        # 应该回退使用 run_id
        assert results[0].data["id"] == "run_789"

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
            results = list(
                AgentRunConverter.to_agui_events(
                    event, tool_call_id_map=tool_call_id_map
                )
            )
            all_results.extend(results)

        # 验证映射已建立
        assert 0 in tool_call_id_map
        assert tool_call_id_map[0] == "call_abc123"

        # 验证：所有 TOOL_CALL_CHUNK 都使用相同的 tool_call_id
        chunk_events = [
            r
            for r in all_results
            if isinstance(r, AgentResult)
            and r.event == EventType.TOOL_CALL_CHUNK
        ]

        # 应该有 3 个 TOOL_CALL_CHUNK 事件（每个 chunk 一个）
        assert len(chunk_events) == 3

        # 所有事件应该使用相同的 tool_call_id（从映射获取）
        for event in chunk_events:
            assert event.data["id"] == "call_abc123"

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
        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        # 回退使用 index
        assert results[0].data["id"] == "0"

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
            results = list(
                AgentRunConverter.to_agui_events(
                    event, tool_call_id_map=tool_call_id_map
                )
            )
            all_results.extend(results)

        # 验证映射正确建立
        assert tool_call_id_map[0] == "call_tool1"
        assert tool_call_id_map[1] == "call_tool2"

        # 验证所有事件使用正确的 ID
        chunk_events = [
            r
            for r in all_results
            if isinstance(r, AgentResult)
            and r.event == EventType.TOOL_CALL_CHUNK
        ]

        # 应该有 6 个 TOOL_CALL_CHUNK 事件
        # - 2 个初始 chunk（id + name）
        # - 4 个 args chunk
        assert len(chunk_events) == 6

        # 验证每个工具调用使用正确的 ID
        tool1_chunks = [e for e in chunk_events if e.data["id"] == "call_tool1"]
        tool2_chunks = [e for e in chunk_events if e.data["id"] == "call_tool2"]

        assert len(tool1_chunks) == 3  # 初始 + '{"q": "test"' + '}'
        assert len(tool2_chunks) == 3  # 初始 + '{"city": "北京"' + '}'

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
        chunk_events = [
            r
            for r in all_results
            if isinstance(r, AgentResult)
            and r.event == EventType.TOOL_CALL_CHUNK
        ]
        # 现在有 2 个 chunk 事件（每个 stream chunk 一个）
        assert len(chunk_events) == 2
        # 所有事件应该使用相同的 ID
        for event in chunk_events:
            assert event.data["id"] == "call_xyz"

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
        tool_call_started_set: set = set()
        results = list(
            AgentRunConverter.to_agui_events(
                event,
                tool_call_id_map=tool_call_id_map,
                tool_call_started_set=tool_call_started_set,
            )
        )

        # 验证映射被建立
        assert tool_call_id_map[0] == "call_complete"
        # 验证 START 已发送
        assert "call_complete" in tool_call_started_set

        # 现在是单个 TOOL_CALL_CHUNK（包含 id, name, args_delta）
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "call_complete"
        assert results[0].data["name"] == "simple_tool"
        assert results[0].data["args_delta"] == '{"done": true}'

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
            results = list(
                AgentRunConverter.to_agui_events(
                    event, tool_call_id_map=tool_call_id_map
                )
            )
            all_results.extend(results)

        chunk_events = [
            r
            for r in all_results
            if isinstance(r, AgentResult)
            and r.event == EventType.TOOL_CALL_CHUNK
        ]

        # 现在有 2 个 chunk 事件（每个 stream chunk 一个）
        assert len(chunk_events) == 2
        # 所有事件应该使用相同的 ID（从映射获取）
        for event in chunk_events:
            assert event.data["id"] == "call_from_none"

    def test_full_tool_call_flow_id_consistency(self):
        """测试完整工具调用流程中的 ID 一致性

        模拟：
        1. on_chat_model_stream 产生 TOOL_CALL_CHUNK
        2. on_tool_start 不产生事件（已在流式中处理）
        3. on_tool_end 产生 TOOL_RESULT

        验证所有事件使用相同的 tool_call_id
        """
        # 模拟完整的工具调用流程
        events = [
            # 流式工具调用参数（第一个 chunk 有 id 和 name）
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
            and r.event in [EventType.TOOL_CALL_CHUNK, EventType.TOOL_RESULT]
        ]

        # 验证所有事件都使用相同的 tool_call_id
        for event in tool_events:
            assert (
                event.data["id"] == "call_full_flow"
            ), f"Event {event.event} has wrong id: {event.data.get('id')}"

        # 验证所有事件类型都存在
        event_types = [e.event for e in tool_events]
        assert EventType.TOOL_CALL_CHUNK in event_types
        assert EventType.TOOL_RESULT in event_types

        # 验证顺序：TOOL_CALL_CHUNK 必须在 TOOL_RESULT 之前
        chunk_idx = event_types.index(EventType.TOOL_CALL_CHUNK)
        result_idx = event_types.index(EventType.TOOL_RESULT)
        assert (
            chunk_idx < result_idx
        ), "TOOL_CALL_CHUNK must come before TOOL_RESULT"

    def test_on_chain_stream_model_node(self):
        """测试 on_chain_stream 事件（model 节点）"""
        msg = create_mock_ai_message("你好！有什么可以帮你的吗？")
        event = {
            "event": "on_chain_stream",
            "name": "model",
            "data": {"chunk": {"messages": [msg]}},
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0] == "你好！有什么可以帮你的吗？"

    def test_on_chain_stream_non_model_node(self):
        """测试 on_chain_stream 事件（非 model 节点）"""
        event = {
            "event": "on_chain_stream",
            "name": "agent",  # 不是 "model"
            "data": {"chunk": {"messages": []}},
        }

        results = list(AgentRunConverter.to_agui_events(event))
        assert len(results) == 0

    def test_on_chat_model_end_ignored(self):
        """测试 on_chat_model_end 事件被忽略（避免重复）"""
        event = {
            "event": "on_chat_model_end",
            "data": {"output": create_mock_ai_message("完成")},
        }

        results = list(AgentRunConverter.to_agui_events(event))
        assert len(results) == 0

    def test_on_chain_stream_tool_call_no_duplicate_with_on_tool_start(self):
        """测试 on_chain_stream 中的 tool call 不会与 on_tool_start 重复

        这个测试验证：当 on_chain_stream 已经发送了 TOOL_CALL_CHUNK 后，
        后续的 on_tool_start 事件不应该再次发送 TOOL_CALL_CHUNK。
        """
        tool_call_id = "call_abc123"
        tool_name = "get_user_name"

        # 使用 converter 实例来维护状态
        converter = AgentRunConverter()

        # 1. 首先是 on_chain_stream 事件，包含 tool call
        msg = create_mock_ai_message(
            content="",
            tool_calls=[{"id": tool_call_id, "name": tool_name, "args": {}}],
        )
        chain_stream_event = {
            "event": "on_chain_stream",
            "name": "model",
            "data": {"chunk": {"messages": [msg]}},
        }

        results1 = list(converter.convert(chain_stream_event))

        # 应该产生一个 TOOL_CALL_CHUNK
        assert len(results1) == 1
        assert results1[0].event == EventType.TOOL_CALL_CHUNK
        assert results1[0].data["id"] == tool_call_id
        assert results1[0].data["name"] == tool_name

        # 2. 然后是 on_tool_start 事件（使用不同的 run_id，模拟真实场景）
        run_id = "run_xyz789"
        tool_start_event = {
            "event": "on_tool_start",
            "run_id": run_id,
            "name": tool_name,
            "data": {"input": {}},
        }

        results2 = list(converter.convert(tool_start_event))

        # 不应该产生 TOOL_CALL_CHUNK（因为已经在 on_chain_stream 中发送过了）
        # 通过 tool_name_to_call_ids 映射，on_tool_start 应该使用相同的 tool_call_id
        assert len(results2) == 0

    def test_on_chain_stream_tool_call_with_on_tool_end(self):
        """测试 on_chain_stream 中的 tool call 与 on_tool_end 的 ID 一致性

        这个测试验证：当 on_chain_stream 发送 TOOL_CALL_CHUNK 后，
        on_tool_end 应该使用相同的 tool_call_id 发送 TOOL_RESULT。
        """
        tool_call_id = "call_abc123"
        tool_name = "get_user_name"
        run_id = "run_xyz789"

        # 使用 converter 实例来维护状态
        converter = AgentRunConverter()

        # 1. on_chain_stream 事件
        msg = create_mock_ai_message(
            content="",
            tool_calls=[{"id": tool_call_id, "name": tool_name, "args": {}}],
        )
        chain_stream_event = {
            "event": "on_chain_stream",
            "name": "model",
            "data": {"chunk": {"messages": [msg]}},
        }
        list(converter.convert(chain_stream_event))

        # 2. on_tool_start 事件
        tool_start_event = {
            "event": "on_tool_start",
            "run_id": run_id,
            "name": tool_name,
            "data": {"input": {}},
        }
        list(converter.convert(tool_start_event))

        # 3. on_tool_end 事件
        tool_end_event = {
            "event": "on_tool_end",
            "run_id": run_id,
            "name": tool_name,
            "data": {"output": '{"user_name": "张三"}'},
        }
        results3 = list(converter.convert(tool_end_event))

        # 应该产生一个 TOOL_RESULT，使用原始的 tool_call_id
        assert len(results3) == 1
        assert results3[0].event == EventType.TOOL_RESULT
        assert results3[0].data["id"] == tool_call_id
        assert results3[0].data["result"] == '{"user_name": "张三"}'


# =============================================================================
# 测试 stream/astream(stream_mode="updates") 格式的转换
# =============================================================================


class TestConvertStreamUpdatesFormat:
    """测试 stream(updates) 格式的事件转换"""

    def test_ai_message_text_content(self):
        """测试 AI 消息的文本内容"""
        msg = create_mock_ai_message("你好！")
        event = {"model": {"messages": [msg]}}

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0] == "你好！"

    def test_ai_message_empty_content(self):
        """测试 AI 消息的空内容"""
        msg = create_mock_ai_message("")
        event = {"model": {"messages": [msg]}}

        results = list(AgentRunConverter.to_agui_events(event))
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

        results = list(AgentRunConverter.to_agui_events(event))

        # 现在是单个 TOOL_CALL_CHUNK
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "call_abc"
        assert results[0].data["name"] == "get_weather"
        assert "上海" in results[0].data["args_delta"]

    def test_tool_message_result(self):
        """测试工具消息的结果"""
        msg = create_mock_tool_message('{"weather": "多云"}', "call_abc")
        event = {"tools": {"messages": [msg]}}

        results = list(AgentRunConverter.to_agui_events(event))

        # 现在只有 TOOL_RESULT
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_RESULT
        assert results[0].data["id"] == "call_abc"
        assert "多云" in results[0].data["result"]

    def test_end_node_ignored(self):
        """测试 __end__ 节点被忽略"""
        event = {"__end__": {"messages": []}}

        results = list(AgentRunConverter.to_agui_events(event))
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

        results = list(AgentRunConverter.to_agui_events(event))

        # 应该有 2 个结果：1 个文本 + 1 个 TOOL_RESULT
        assert len(results) == 2
        assert results[0] == "正在查询..."
        assert results[1].event == EventType.TOOL_RESULT

    def test_custom_messages_key(self):
        """测试自定义 messages_key"""
        msg = create_mock_ai_message("自定义消息")
        event = {"model": {"custom_messages": [msg]}}

        # 使用默认 key 应该找不到消息
        results = list(
            AgentRunConverter.to_agui_events(event, messages_key="messages")
        )
        assert len(results) == 0

        # 使用正确的 key
        results = list(
            AgentRunConverter.to_agui_events(
                event, messages_key="custom_messages"
            )
        )
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

        results = list(AgentRunConverter.to_agui_events(event))

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

        results = list(AgentRunConverter.to_agui_events(event))

        # 现在是单个 TOOL_CALL_CHUNK
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK

    def test_last_tool_message_result(self):
        """测试最后一条工具消息的结果"""
        ai_msg = create_mock_ai_message("之前的消息")
        tool_msg = create_mock_tool_message("工具结果", "call_ghi")
        event = {"messages": [ai_msg, tool_msg]}

        results = list(AgentRunConverter.to_agui_events(event))

        # 只处理最后一条消息（工具消息），现在只有 TOOL_RESULT
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_RESULT

    def test_empty_messages(self):
        """测试空消息列表"""
        event = {"messages": []}

        results = list(AgentRunConverter.to_agui_events(event))
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

        results = list(AgentRunConverter.to_agui_events(stream_event))

        assert len(results) == 1
        assert results[0] == "Hello"


# =============================================================================
# 测试完整流程：模拟多个事件的序列
# =============================================================================


class TestConvertEventSequence:
    """测试完整的事件序列转换"""

    def test_astream_events_full_sequence(self):
        """测试 astream_events 格式的完整事件序列

        AG-UI 协议要求的事件顺序：
        TOOL_CALL_START → TOOL_CALL_ARGS → TOOL_CALL_END → TOOL_CALL_RESULT
        """
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
            all_results.extend(AgentRunConverter.to_agui_events(event))

        # 验证结果
        # on_tool_start: 1 TOOL_CALL_CHUNK
        # on_tool_end: 1 TOOL_RESULT
        # 3x on_chat_model_stream: 3 个文本
        assert len(all_results) == 5

        # 工具调用事件
        assert all_results[0].event == EventType.TOOL_CALL_CHUNK
        assert all_results[1].event == EventType.TOOL_RESULT

        # 文本内容
        assert all_results[2] == "北京"
        assert all_results[3] == "今天"
        assert all_results[4] == "晴天"

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
            all_results.extend(AgentRunConverter.to_agui_events(event))

        # 验证结果：
        # - 1 TOOL_CALL_CHUNK（工具调用）
        # - 1 TOOL_RESULT（工具结果）
        # - 1 文本回复
        assert len(all_results) == 3

        # 工具调用
        assert all_results[0].event == EventType.TOOL_CALL_CHUNK
        assert all_results[0].data["name"] == "get_weather"

        # 工具结果
        assert all_results[1].event == EventType.TOOL_RESULT

        # 最终回复
        assert all_results[2] == "上海今天多云。"


# =============================================================================
# 测试边界情况
# =============================================================================


class TestConvertEdgeCases:
    """测试边界情况"""

    def test_empty_event(self):
        """测试空事件"""
        results = list(AgentRunConverter.to_agui_events({}))
        assert len(results) == 0

    def test_none_values(self):
        """测试 None 值"""
        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": None},
        }
        results = list(AgentRunConverter.to_agui_events(event))
        assert len(results) == 0

    def test_invalid_message_type(self):
        """测试无效的消息类型"""
        msg = MagicMock()
        msg.type = "unknown"
        msg.content = "test"
        event = {"model": {"messages": [msg]}}

        results = list(AgentRunConverter.to_agui_events(event))
        # unknown 类型不会产生输出
        assert len(results) == 0

    def test_tool_call_without_id(self):
        """测试没有 ID 的工具调用"""
        msg = create_mock_ai_message(
            "",
            tool_calls=[{"name": "test", "args": {}}],  # 没有 id
        )
        event = {"agent": {"messages": [msg]}}

        results = list(AgentRunConverter.to_agui_events(event))
        # 没有 id 的工具调用应该被跳过
        assert len(results) == 0

    def test_tool_message_without_tool_call_id(self):
        """测试没有 tool_call_id 的工具消息"""
        msg = MagicMock()
        msg.type = "tool"
        msg.content = "result"
        msg.tool_call_id = None  # 没有 tool_call_id

        event = {"tools": {"messages": [msg]}}

        results = list(AgentRunConverter.to_agui_events(event))
        # 没有 tool_call_id 的工具消息应该被跳过
        assert len(results) == 0

    def test_dict_message_format(self):
        """测试字典格式的消息（而非对象）"""
        event = {
            "model": {"messages": [{"type": "ai", "content": "字典格式消息"}]}
        }

        results = list(AgentRunConverter.to_agui_events(event))

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

        results = list(AgentRunConverter.to_agui_events(event))

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

        results = list(AgentRunConverter.to_agui_events(event))

        # on_tool_end 只发送 TOOL_CALL_RESULT（TOOL_CALL_END 在 on_tool_start 发送）
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_RESULT
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
        results = list(AgentRunConverter.to_agui_events(event))
        assert len(results) == 0

    def test_unsupported_random_dict_format(self):
        """测试不支持的随机字典格式

        如果传入的 dict 不匹配任何已知格式，应该不产生输出。
        """
        event = {
            "random_key": "random_value",
            "another_key": {"nested": "data"},
        }

        results = list(AgentRunConverter.to_agui_events(event))
        assert len(results) == 0


# =============================================================================
# 测试 AG-UI 协议事件顺序
# =============================================================================


class TestAguiEventOrder:
    """测试事件顺序

    简化后的事件结构：
    - TOOL_CALL_CHUNK - 工具调用（包含 id, name, args_delta）
    - TOOL_RESULT - 工具执行结果

    边界事件（如 TOOL_CALL_START/END）由协议层自动处理。
    """

    def test_streaming_tool_call_order(self):
        """测试流式工具调用的事件顺序

        TOOL_CALL_CHUNK 应该在 TOOL_RESULT 之前
        """
        events = [
            # 第一个 chunk：包含 id、name，无 args
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[{
                            "id": "call_order_test",
                            "name": "test_tool",
                            "args": "",
                            "index": 0,
                        }],
                    )
                },
            },
            # 第二个 chunk：有 args
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
            # 工具开始执行
            {
                "event": "on_tool_start",
                "name": "test_tool",
                "run_id": "run_order",
                "data": {
                    "input": {
                        "key": "value",
                        "runtime": MagicMock(tool_call_id="call_order_test"),
                    }
                },
            },
            # 工具执行完成
            {
                "event": "on_tool_end",
                "run_id": "run_order",
                "data": {
                    "input": {
                        "key": "value",
                        "runtime": MagicMock(tool_call_id="call_order_test"),
                    },
                    "output": "success",
                },
            },
        ]

        converter = AgentRunConverter()
        all_results = []
        for event in events:
            all_results.extend(converter.convert(event))

        # 提取工具调用相关事件
        tool_events = [
            r
            for r in all_results
            if isinstance(r, AgentResult)
            and r.event in [EventType.TOOL_CALL_CHUNK, EventType.TOOL_RESULT]
        ]

        # 验证有这两种事件
        event_types = [e.event for e in tool_events]
        assert EventType.TOOL_CALL_CHUNK in event_types
        assert EventType.TOOL_RESULT in event_types

        # 找到第一个 TOOL_CALL_CHUNK 和 TOOL_RESULT 的索引
        chunk_idx = event_types.index(EventType.TOOL_CALL_CHUNK)
        result_idx = event_types.index(EventType.TOOL_RESULT)

        # 验证顺序：TOOL_CALL_CHUNK 必须在 TOOL_RESULT 之前
        assert chunk_idx < result_idx, (
            f"TOOL_CALL_CHUNK (idx={chunk_idx}) must come before "
            f"TOOL_RESULT (idx={result_idx})"
        )

    def test_streaming_tool_call_start_not_duplicated(self):
        """测试流式工具调用时 TOOL_CALL_START 不会重复发送"""
        events = [
            # 第一个 chunk：包含 id、name
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[{
                            "id": "call_no_dup",
                            "name": "test_tool",
                            "args": '{"a": 1}',
                            "index": 0,
                        }],
                    )
                },
            },
            # 工具开始执行（此时 START 已在上面发送，不应重复）
            {
                "event": "on_tool_start",
                "name": "test_tool",
                "run_id": "run_no_dup",
                "data": {
                    "input": {
                        "a": 1,
                        "runtime": MagicMock(tool_call_id="call_no_dup"),
                    }
                },
            },
            # 工具执行完成
            {
                "event": "on_tool_end",
                "run_id": "run_no_dup",
                "data": {
                    "input": {
                        "a": 1,
                        "runtime": MagicMock(tool_call_id="call_no_dup"),
                    },
                    "output": "done",
                },
            },
        ]

        converter = AgentRunConverter()
        all_results = []
        for event in events:
            all_results.extend(converter.convert(event))

        # 统计 TOOL_CALL_START 事件的数量
        start_events = [
            r
            for r in all_results
            if isinstance(r, AgentResult)
            and r.event == EventType.TOOL_CALL_CHUNK
        ]

        # 应该只有一个 TOOL_CALL_START
        assert (
            len(start_events) == 1
        ), f"Expected 1 TOOL_CALL_START, got {len(start_events)}"

    def test_non_streaming_tool_call_order(self):
        """测试非流式场景的工具调用事件顺序

        在没有 on_chat_model_stream 事件的场景下，
        事件顺序仍应正确：TOOL_CALL_CHUNK → TOOL_RESULT
        """
        events = [
            # 直接工具开始（无流式事件）
            {
                "event": "on_tool_start",
                "name": "weather",
                "run_id": "run_nonstream",
                "data": {"input": {"city": "北京"}},
            },
            # 工具执行完成
            {
                "event": "on_tool_end",
                "run_id": "run_nonstream",
                "data": {"output": "晴天"},
            },
        ]

        converter = AgentRunConverter()
        all_results = []
        for event in events:
            all_results.extend(converter.convert(event))

        tool_events = [r for r in all_results if isinstance(r, AgentResult)]

        event_types = [e.event for e in tool_events]

        # 验证顺序：TOOL_CALL_CHUNK → TOOL_RESULT
        assert event_types == [
            EventType.TOOL_CALL_CHUNK,
            EventType.TOOL_RESULT,
        ], f"Unexpected order: {event_types}"

    def test_multiple_concurrent_tool_calls_order(self):
        """测试多个并发工具调用时各自的事件顺序正确"""
        events = [
            # 两个工具调用的第一个 chunk
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[
                            {
                                "id": "call_a",
                                "name": "tool_a",
                                "args": "",
                                "index": 0,
                            },
                            {
                                "id": "call_b",
                                "name": "tool_b",
                                "args": "",
                                "index": 1,
                            },
                        ],
                    )
                },
            },
            # 两个工具的参数
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(
                        content="",
                        tool_call_chunks=[
                            {
                                "id": "",
                                "name": "",
                                "args": '{"x": 1}',
                                "index": 0,
                            },
                            {
                                "id": "",
                                "name": "",
                                "args": '{"y": 2}',
                                "index": 1,
                            },
                        ],
                    )
                },
            },
            # 工具 A 开始
            {
                "event": "on_tool_start",
                "name": "tool_a",
                "run_id": "run_a",
                "data": {
                    "input": {
                        "x": 1,
                        "runtime": MagicMock(tool_call_id="call_a"),
                    }
                },
            },
            # 工具 B 开始
            {
                "event": "on_tool_start",
                "name": "tool_b",
                "run_id": "run_b",
                "data": {
                    "input": {
                        "y": 2,
                        "runtime": MagicMock(tool_call_id="call_b"),
                    }
                },
            },
            # 工具 A 结束
            {
                "event": "on_tool_end",
                "run_id": "run_a",
                "data": {
                    "input": {
                        "x": 1,
                        "runtime": MagicMock(tool_call_id="call_a"),
                    },
                    "output": "result_a",
                },
            },
            # 工具 B 结束
            {
                "event": "on_tool_end",
                "run_id": "run_b",
                "data": {
                    "input": {
                        "y": 2,
                        "runtime": MagicMock(tool_call_id="call_b"),
                    },
                    "output": "result_b",
                },
            },
        ]

        converter = AgentRunConverter()
        all_results = []
        for event in events:
            all_results.extend(converter.convert(event))

        # 分别验证工具 A 和工具 B 的事件顺序
        for tool_id in ["call_a", "call_b"]:
            tool_events = [
                (i, r)
                for i, r in enumerate(all_results)
                if isinstance(r, AgentResult) and r.data.get("id") == tool_id
            ]

            event_types = [e.event for _, e in tool_events]

            # 验证包含所有必需事件
            assert (
                EventType.TOOL_CALL_CHUNK in event_types
            ), f"Tool {tool_id} missing TOOL_CALL_CHUNK"
            assert (
                EventType.TOOL_RESULT in event_types
            ), f"Tool {tool_id} missing TOOL_RESULT"

            # 验证顺序：TOOL_CALL_CHUNK 应该在 TOOL_RESULT 之前
            chunk_pos = event_types.index(EventType.TOOL_CALL_CHUNK)
            result_pos = event_types.index(EventType.TOOL_RESULT)

            assert (
                chunk_pos < result_pos
            ), f"Tool {tool_id}: CHUNK must come before RESULT"


# =============================================================================
# 集成测试：模拟完整流程
# =============================================================================


class TestConvertIntegration:
    """测试 convert 与完整流程的集成"""

    def test_astream_events_full_flow(self):
        """测试模拟的 astream_events 完整流程"""
        mock_events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": create_mock_ai_message_chunk("你好")},
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": create_mock_ai_message_chunk("，")},
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": create_mock_ai_message_chunk("世界！")},
            },
        ]

        results = []
        for event in mock_events:
            results.extend(AgentRunConverter.to_agui_events(event))

        assert len(results) == 3
        assert "".join(results) == "你好，世界！"

    def test_stream_updates_full_flow(self):
        """测试模拟的 stream(updates) 完整流程"""
        import json

        mock_events = [
            # Agent 决定调用工具
            {
                "agent": {
                    "messages": [
                        create_mock_ai_message(
                            "",
                            tool_calls=[{
                                "id": "tc_001",
                                "name": "get_weather",
                                "args": {"city": "北京"},
                            }],
                        )
                    ]
                }
            },
            # 工具执行结果
            {
                "tools": {
                    "messages": [
                        create_mock_tool_message(
                            json.dumps(
                                {"city": "北京", "weather": "晴天"},
                                ensure_ascii=False,
                            ),
                            "tc_001",
                        )
                    ]
                }
            },
            # Agent 最终回复
            {
                "model": {
                    "messages": [create_mock_ai_message("北京今天天气晴朗。")]
                }
            },
        ]

        results = []
        for event in mock_events:
            results.extend(AgentRunConverter.to_agui_events(event))

        # 验证事件顺序
        assert len(results) == 3

        # 工具调用
        assert isinstance(results[0], AgentResult)
        assert results[0].event == EventType.TOOL_CALL_CHUNK

        # 工具结果
        assert isinstance(results[1], AgentResult)
        assert results[1].event == EventType.TOOL_RESULT

        # 最终文本
        assert results[2] == "北京今天天气晴朗。"

    def test_stream_values_full_flow(self):
        """测试模拟的 stream(values) 完整流程"""
        mock_events = [
            {"messages": [create_mock_ai_message("")]},
            {
                "messages": [
                    create_mock_ai_message(
                        "",
                        tool_calls=[
                            {"id": "tc_002", "name": "get_time", "args": {}}
                        ],
                    )
                ]
            },
            {
                "messages": [
                    create_mock_ai_message(""),
                    create_mock_tool_message("2024-01-01 12:00:00", "tc_002"),
                ]
            },
            {
                "messages": [
                    create_mock_ai_message(""),
                    create_mock_tool_message("2024-01-01 12:00:00", "tc_002"),
                    create_mock_ai_message("现在是 2024年1月1日。"),
                ]
            },
        ]

        results = []
        for event in mock_events:
            results.extend(AgentRunConverter.to_agui_events(event))

        # 验证有工具调用事件
        tool_chunks = [
            r
            for r in results
            if isinstance(r, AgentResult)
            and r.event == EventType.TOOL_CALL_CHUNK
        ]
        assert len(tool_chunks) >= 1

        # 验证有最终文本
        text_results = [r for r in results if isinstance(r, str) and r]
        assert any("2024" in t for t in text_results)
