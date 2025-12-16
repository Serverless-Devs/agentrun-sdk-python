"""测试 LangGraph 事件到 AgentEvent 的转换

本文件专注于测试 LangGraph/LangChain 事件到 AgentEvent 的转换，
确保每种输入事件类型都有明确的预期输出。

简化后的事件结构：
- TOOL_CALL_CHUNK: 工具调用（包含 id, name, args_delta）
- TOOL_RESULT: 工具执行结果（包含 id, result）
- TEXT: 文本内容（字符串）

边界事件（如 TOOL_CALL_START/END）由协议层自动生成，转换器不再输出这些事件。
"""

from typing import Dict, List, Union
from unittest.mock import MagicMock

import pytest

from agentrun.integration.langgraph import AgentRunConverter
from agentrun.server.model import AgentEvent, EventType

# 使用 helpers.py 中的公共函数
from .helpers import convert_and_collect
from .helpers import create_mock_ai_message as create_ai_message
from .helpers import create_mock_ai_message_chunk as create_ai_message_chunk
from .helpers import create_mock_tool_message as create_tool_message
from .helpers import filter_agent_events

# =============================================================================
# 测试 on_chat_model_stream 事件（流式文本输出）
# =============================================================================


class TestOnChatModelStreamText:
    """测试 on_chat_model_stream 事件的文本内容输出"""

    def test_simple_text_content(self):
        """测试简单文本内容

        输入: on_chat_model_stream with content="你好"
        输出: "你好" (字符串)
        """
        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": create_ai_message_chunk("你好")},
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0] == "你好"

    def test_empty_content_no_output(self):
        """测试空内容不产生输出

        输入: on_chat_model_stream with content=""
        输出: (无)
        """
        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": create_ai_message_chunk("")},
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 0

    def test_multiple_stream_chunks(self):
        """测试多个流式 chunk

        输入: 多个 on_chat_model_stream
        输出: 多个字符串
        """
        events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": create_ai_message_chunk("你")},
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": create_ai_message_chunk("好")},
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": create_ai_message_chunk("！")},
            },
        ]

        results = convert_and_collect(events)

        assert len(results) == 3
        assert results[0] == "你"
        assert results[1] == "好"
        assert results[2] == "！"


# =============================================================================
# 测试 on_chat_model_stream 事件（流式工具调用）
# =============================================================================


class TestOnChatModelStreamToolCall:
    """测试 on_chat_model_stream 事件的工具调用输出"""

    def test_tool_call_first_chunk_with_id_and_name(self):
        """测试工具调用第一个 chunk（包含 id 和 name）

        输入: tool_call_chunk with id="call_123", name="get_weather", args=""
        输出: TOOL_CALL_CHUNK with id="call_123", name="get_weather", args_delta=""
        """
        event = {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": create_ai_message_chunk(
                    tool_call_chunks=[{
                        "id": "call_123",
                        "name": "get_weather",
                        "args": "",
                        "index": 0,
                    }]
                )
            },
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert isinstance(results[0], AgentEvent)
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "call_123"
        assert results[0].data["name"] == "get_weather"
        assert results[0].data["args_delta"] == ""

    def test_tool_call_subsequent_chunk_with_args(self):
        """测试工具调用后续 chunk（只有 args）

        输入: 后续 chunk with args='{"city": "北京"}'
        输出: TOOL_CALL_CHUNK with args_delta='{"city": "北京"}'
        """
        # 首先发送第一个 chunk 建立映射
        first_chunk = {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": create_ai_message_chunk(
                    tool_call_chunks=[{
                        "id": "call_123",
                        "name": "get_weather",
                        "args": "",
                        "index": 0,
                    }]
                )
            },
        }

        # 后续 chunk
        second_chunk = {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": create_ai_message_chunk(
                    tool_call_chunks=[{
                        "id": "",  # 后续 chunk 没有 id
                        "name": "",
                        "args": '{"city": "北京"}',
                        "index": 0,
                    }]
                )
            },
        }

        tool_call_id_map: Dict[int, str] = {}
        results1 = list(
            AgentRunConverter.to_agui_events(
                first_chunk, tool_call_id_map=tool_call_id_map
            )
        )
        results2 = list(
            AgentRunConverter.to_agui_events(
                second_chunk, tool_call_id_map=tool_call_id_map
            )
        )

        # 第一个 chunk 产生一个事件
        assert len(results1) == 1
        assert results1[0].data["id"] == "call_123"

        # 第二个 chunk 也产生一个事件，使用映射的 id
        assert len(results2) == 1
        assert results2[0].event == EventType.TOOL_CALL_CHUNK
        assert results2[0].data["id"] == "call_123"
        assert results2[0].data["args_delta"] == '{"city": "北京"}'

    def test_tool_call_complete_in_one_chunk(self):
        """测试完整工具调用在一个 chunk 中

        输入: chunk with id, name, and complete args
        输出: 单个 TOOL_CALL_CHUNK
        """
        event = {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": create_ai_message_chunk(
                    tool_call_chunks=[{
                        "id": "call_456",
                        "name": "get_time",
                        "args": '{"timezone": "UTC"}',
                        "index": 0,
                    }]
                )
            },
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "call_456"
        assert results[0].data["name"] == "get_time"
        assert results[0].data["args_delta"] == '{"timezone": "UTC"}'

    def test_multiple_concurrent_tool_calls(self):
        """测试多个并发工具调用

        输入: 一个 chunk 包含两个工具调用
        输出: 两个 TOOL_CALL_CHUNK
        """
        event = {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": create_ai_message_chunk(
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
                    ]
                )
            },
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 2
        assert results[0].data["id"] == "call_a"
        assert results[0].data["name"] == "tool_a"
        assert results[1].data["id"] == "call_b"
        assert results[1].data["name"] == "tool_b"


# =============================================================================
# 测试 on_tool_start 事件
# =============================================================================


class TestOnToolStart:
    """测试 on_tool_start 事件的转换"""

    def test_simple_tool_start(self):
        """测试简单的工具启动事件

        输入: on_tool_start with input
        输出: 单个 TOOL_CALL_CHUNK（如果未在流式中发送过）
        """
        event = {
            "event": "on_tool_start",
            "name": "get_weather",
            "run_id": "run_123",
            "data": {"input": {"city": "北京"}},
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "run_123"
        assert results[0].data["name"] == "get_weather"
        assert "北京" in results[0].data["args_delta"]

    def test_tool_start_with_runtime_tool_call_id(self):
        """测试使用 runtime.tool_call_id 的工具启动

        输入: on_tool_start with runtime.tool_call_id
        输出: TOOL_CALL_CHUNK 使用 runtime 中的 id
        """

        class FakeRuntime:
            tool_call_id = "call_original_id"

        event = {
            "event": "on_tool_start",
            "name": "get_weather",
            "run_id": "run_123",  # 这个不应该被使用
            "data": {"input": {"city": "北京", "runtime": FakeRuntime()}},
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert (
            results[0].data["id"] == "call_original_id"
        )  # 使用 runtime 中的 id

    def test_tool_start_no_duplicate_if_already_started(self):
        """测试如果工具已在流式中开始，不重复发送

        输入: on_tool_start after streaming chunk
        输出: 无（因为已在流式中发送过）
        """
        tool_call_started_set = {"call_123"}

        event = {
            "event": "on_tool_start",
            "name": "get_weather",
            "run_id": "run_123",
            "data": {
                "input": {
                    "city": "北京",
                    "runtime": MagicMock(tool_call_id="call_123"),
                }
            },
        }

        results = list(
            AgentRunConverter.to_agui_events(
                event, tool_call_started_set=tool_call_started_set
            )
        )

        # 已经在流式中发送过，不再发送
        assert len(results) == 0

    def test_tool_start_without_input(self):
        """测试无输入参数的工具启动

        输入: on_tool_start with empty input
        输出: TOOL_CALL_CHUNK with empty args_delta
        """
        event = {
            "event": "on_tool_start",
            "name": "get_time",
            "run_id": "run_456",
            "data": {},  # 无输入
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["name"] == "get_time"
        assert results[0].data["args_delta"] == ""


# =============================================================================
# 测试 on_tool_end 事件
# =============================================================================


class TestOnToolEnd:
    """测试 on_tool_end 事件的转换"""

    def test_simple_tool_end(self):
        """测试简单的工具结束事件

        输入: on_tool_end with output
        输出: TOOL_RESULT
        """
        event = {
            "event": "on_tool_end",
            "run_id": "run_123",
            "data": {"output": {"result": "晴天", "temp": 25}},
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_RESULT
        assert results[0].data["id"] == "run_123"
        assert "晴天" in results[0].data["result"]

    def test_tool_end_with_string_output(self):
        """测试字符串输出的工具结束

        输入: on_tool_end with string output
        输出: TOOL_RESULT with string result
        """
        event = {
            "event": "on_tool_end",
            "run_id": "run_456",
            "data": {"output": "操作成功"},
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_RESULT
        assert results[0].data["result"] == "操作成功"

    def test_tool_end_with_runtime_tool_call_id(self):
        """测试使用 runtime.tool_call_id 的工具结束

        输入: on_tool_end with runtime.tool_call_id
        输出: TOOL_RESULT 使用 runtime 中的 id
        """

        class FakeRuntime:
            tool_call_id = "call_original_id"

        event = {
            "event": "on_tool_end",
            "run_id": "run_123",
            "data": {
                "input": {"runtime": FakeRuntime()},
                "output": "结果",
            },
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].data["id"] == "call_original_id"


# =============================================================================
# 测试 stream_mode="updates" 格式
# =============================================================================


class TestStreamUpdatesFormat:
    """测试 stream(stream_mode="updates") 格式的事件转换"""

    def test_ai_message_with_text(self):
        """测试 AI 消息的文本内容

        输入: {model: {messages: [AIMessage(content="你好")]}}
        输出: "你好" (字符串)
        """
        msg = create_ai_message("你好")
        event = {"model": {"messages": [msg]}}

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0] == "你好"

    def test_ai_message_with_tool_calls(self):
        """测试 AI 消息的工具调用

        输入: {agent: {messages: [AIMessage with tool_calls]}}
        输出: TOOL_CALL_CHUNK
        """
        msg = create_ai_message(
            tool_calls=[{
                "id": "call_abc",
                "name": "search",
                "args": {"query": "天气"},
            }]
        )
        event = {"agent": {"messages": [msg]}}

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "call_abc"
        assert results[0].data["name"] == "search"

    def test_tool_message_result(self):
        """测试工具消息的结果

        输入: {tools: {messages: [ToolMessage]}}
        输出: TOOL_RESULT
        """
        msg = create_tool_message('{"weather": "晴天"}', "call_xyz")
        event = {"tools": {"messages": [msg]}}

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_RESULT
        assert results[0].data["id"] == "call_xyz"


# =============================================================================
# 测试 stream_mode="values" 格式
# =============================================================================


class TestStreamValuesFormat:
    """测试 stream(stream_mode="values") 格式的事件转换"""

    def test_values_format_last_message(self):
        """测试 values 格式只处理最后一条消息

        输入: {messages: [msg1, msg2, msg3]}
        输出: 只处理 msg3
        """
        msg1 = create_ai_message("第一条")
        msg2 = create_ai_message("第二条")
        msg3 = create_ai_message("第三条")
        event = {"messages": [msg1, msg2, msg3]}

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0] == "第三条"

    def test_values_format_tool_call(self):
        """测试 values 格式的工具调用

        输入: {messages: [AIMessage with tool_calls]}
        输出: TOOL_CALL_CHUNK
        """
        msg = create_ai_message(
            tool_calls=[{"id": "call_123", "name": "test", "args": {}}]
        )
        event = {"messages": [msg]}

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK


# =============================================================================
# 测试 AgentRunConverter 类
# =============================================================================


class TestAgentRunConverterClass:
    """测试 AgentRunConverter 类的功能"""

    def test_converter_maintains_state(self):
        """测试转换器维护状态（tool_call_id_map）"""
        converter = AgentRunConverter()

        # 第一个 chunk 建立映射
        event1 = {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": create_ai_message_chunk(
                    tool_call_chunks=[{
                        "id": "call_stateful",
                        "name": "test",
                        "args": "",
                        "index": 0,
                    }]
                )
            },
        }

        results1 = list(converter.convert(event1))
        assert converter._tool_call_id_map[0] == "call_stateful"

        # 第二个 chunk 使用映射
        event2 = {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": create_ai_message_chunk(
                    tool_call_chunks=[{
                        "id": "",
                        "name": "",
                        "args": '{"a": 1}',
                        "index": 0,
                    }]
                )
            },
        }

        results2 = list(converter.convert(event2))
        assert results2[0].data["id"] == "call_stateful"

    def test_converter_reset(self):
        """测试转换器重置"""
        converter = AgentRunConverter()

        # 建立一些状态
        event = {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": create_ai_message_chunk(
                    tool_call_chunks=[{
                        "id": "call_to_reset",
                        "name": "test",
                        "args": "",
                        "index": 0,
                    }]
                )
            },
        }
        list(converter.convert(event))

        assert len(converter._tool_call_id_map) > 0

        # 重置
        converter.reset()
        assert len(converter._tool_call_id_map) == 0
        assert len(converter._tool_call_started_set) == 0


# =============================================================================
# 测试完整流程
# =============================================================================


class TestCompleteFlow:
    """测试完整的工具调用流程"""

    def test_streaming_tool_call_complete_flow(self):
        """测试流式工具调用的完整流程

        流程:
        1. on_chat_model_stream (id + name)
        2. on_chat_model_stream (args)
        3. on_tool_start (不产生事件，因为已发送)
        4. on_tool_end (TOOL_RESULT)
        """
        converter = AgentRunConverter()

        events = [
            # 1. 流式工具调用开始
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": create_ai_message_chunk(
                        tool_call_chunks=[{
                            "id": "call_flow_test",
                            "name": "weather",
                            "args": "",
                            "index": 0,
                        }]
                    )
                },
            },
            # 2. 流式工具调用参数
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": create_ai_message_chunk(
                        tool_call_chunks=[{
                            "id": "",
                            "name": "",
                            "args": '{"city": "北京"}',
                            "index": 0,
                        }]
                    )
                },
            },
            # 3. 工具开始执行（已在流式中发送，不产生事件）
            {
                "event": "on_tool_start",
                "name": "weather",
                "run_id": "run_flow",
                "data": {
                    "input": {
                        "city": "北京",
                        "runtime": MagicMock(tool_call_id="call_flow_test"),
                    }
                },
            },
            # 4. 工具执行完成
            {
                "event": "on_tool_end",
                "run_id": "run_flow",
                "data": {
                    "input": {
                        "city": "北京",
                        "runtime": MagicMock(tool_call_id="call_flow_test"),
                    },
                    "output": "晴天",
                },
            },
        ]

        all_results = []
        for event in events:
            all_results.extend(converter.convert(event))

        # 预期结果:
        # - 2 个 TOOL_CALL_CHUNK（流式）
        # - 1 个 TOOL_RESULT
        chunk_events = filter_agent_events(
            all_results, EventType.TOOL_CALL_CHUNK
        )
        result_events = filter_agent_events(all_results, EventType.TOOL_RESULT)

        assert len(chunk_events) == 2
        assert len(result_events) == 1

        # 验证 ID 一致性
        for event in chunk_events + result_events:
            assert event.data["id"] == "call_flow_test"

    def test_non_streaming_tool_call_complete_flow(self):
        """测试非流式工具调用的完整流程

        流程:
        1. on_tool_start (TOOL_CALL_CHUNK)
        2. on_tool_end (TOOL_RESULT)
        """
        events = [
            {
                "event": "on_tool_start",
                "name": "calculator",
                "run_id": "run_calc",
                "data": {"input": {"a": 1, "b": 2}},
            },
            {
                "event": "on_tool_end",
                "run_id": "run_calc",
                "data": {"output": 3},
            },
        ]

        all_results = convert_and_collect(events)

        chunk_events = filter_agent_events(
            all_results, EventType.TOOL_CALL_CHUNK
        )
        result_events = filter_agent_events(all_results, EventType.TOOL_RESULT)

        assert len(chunk_events) == 1
        assert len(result_events) == 1

        # 验证顺序
        chunk_idx = all_results.index(chunk_events[0])
        result_idx = all_results.index(result_events[0])
        assert chunk_idx < result_idx


# =============================================================================
# 测试错误事件
# =============================================================================


class TestErrorEvents:
    """测试 LangChain 错误事件的转换"""

    def test_on_tool_error(self):
        """测试 on_tool_error 事件

        输入: on_tool_error with error
        输出: ERROR 事件
        """
        event = {
            "event": "on_tool_error",
            "name": "weather_tool",
            "run_id": "run_123",
            "data": {
                "error": ValueError("Invalid city name"),
                "input": {"city": "invalid"},
            },
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].event == EventType.ERROR
        assert "weather_tool" in results[0].data["message"]
        assert "ValueError" in results[0].data["message"]
        assert results[0].data["code"] == "TOOL_ERROR"
        assert results[0].data["tool_call_id"] == "run_123"

    def test_on_tool_error_with_runtime_tool_call_id(self):
        """测试 on_tool_error 使用 runtime 中的 tool_call_id"""

        class FakeRuntime:
            tool_call_id = "call_original_id"

        event = {
            "event": "on_tool_error",
            "name": "search_tool",
            "run_id": "run_456",
            "data": {
                "error": Exception("API timeout"),
                "input": {"query": "test", "runtime": FakeRuntime()},
            },
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].data["tool_call_id"] == "call_original_id"

    def test_on_tool_error_with_string_error(self):
        """测试 on_tool_error 使用字符串错误"""
        event = {
            "event": "on_tool_error",
            "name": "calc_tool",
            "run_id": "run_789",
            "data": {
                "error": "Division by zero",
                "input": {},
            },
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert "Division by zero" in results[0].data["message"]

    def test_on_llm_error(self):
        """测试 on_llm_error 事件

        输入: on_llm_error with error
        输出: ERROR 事件
        """
        event = {
            "event": "on_llm_error",
            "run_id": "run_llm",
            "data": {
                "error": RuntimeError("API rate limit exceeded"),
            },
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].event == EventType.ERROR
        assert "LLM error" in results[0].data["message"]
        assert "RuntimeError" in results[0].data["message"]
        assert results[0].data["code"] == "LLM_ERROR"

    def test_on_chain_error(self):
        """测试 on_chain_error 事件

        输入: on_chain_error with error
        输出: ERROR 事件
        """
        event = {
            "event": "on_chain_error",
            "name": "agent_chain",
            "run_id": "run_chain",
            "data": {
                "error": KeyError("missing_key"),
            },
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].event == EventType.ERROR
        assert "agent_chain" in results[0].data["message"]
        assert "KeyError" in results[0].data["message"]
        assert results[0].data["code"] == "CHAIN_ERROR"

    def test_on_retriever_error(self):
        """测试 on_retriever_error 事件

        输入: on_retriever_error with error
        输出: ERROR 事件
        """
        event = {
            "event": "on_retriever_error",
            "name": "vector_store",
            "run_id": "run_retriever",
            "data": {
                "error": ConnectionError("Database connection failed"),
            },
        }

        results = list(AgentRunConverter.to_agui_events(event))

        assert len(results) == 1
        assert results[0].event == EventType.ERROR
        assert "vector_store" in results[0].data["message"]
        assert "ConnectionError" in results[0].data["message"]
        assert results[0].data["code"] == "RETRIEVER_ERROR"

    def test_tool_error_in_complete_flow(self):
        """测试完整流程中的工具错误

        流程:
        1. on_tool_start (TOOL_CALL_CHUNK)
        2. on_tool_error (ERROR)
        """
        events = [
            {
                "event": "on_tool_start",
                "name": "risky_tool",
                "run_id": "run_risky",
                "data": {"input": {"param": "test"}},
            },
            {
                "event": "on_tool_error",
                "name": "risky_tool",
                "run_id": "run_risky",
                "data": {
                    "error": RuntimeError("Tool execution failed"),
                    "input": {"param": "test"},
                },
            },
        ]

        all_results = convert_and_collect(events)

        chunk_events = filter_agent_events(
            all_results, EventType.TOOL_CALL_CHUNK
        )
        error_events = filter_agent_events(all_results, EventType.ERROR)

        assert len(chunk_events) == 1
        assert len(error_events) == 1
        assert chunk_events[0].data["id"] == "run_risky"
        assert error_events[0].data["tool_call_id"] == "run_risky"
