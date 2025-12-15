"""测试 to_agui_events 函数 / Test to_agui_events Function

测试 to_agui_events 函数对不同 LangChain/LangGraph 调用方式返回事件格式的兼容性。
支持的格式：
- astream_events(version="v2") 格式
- stream/astream(stream_mode="updates") 格式
- stream/astream(stream_mode="values") 格式

本测试使用 Mock 模拟大模型返回值，无需真实模型即可测试。
"""

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agentrun.integration.langgraph.agent_converter import convert  # 别名，兼容旧代码
from agentrun.integration.langgraph.agent_converter import (
    _is_astream_events_format,
    _is_stream_updates_format,
    _is_stream_values_format,
    to_agui_events,
)
from agentrun.server.model import AgentResult, EventType

# =============================================================================
# Mock 数据：模拟 LangChain/LangGraph 返回的消息对象
# =============================================================================


def create_mock_ai_message(
    content: str, tool_calls: List[Dict[str, Any]] = None
) -> MagicMock:
    """创建模拟的 AIMessage 对象"""
    msg = MagicMock()
    msg.content = content
    msg.type = "ai"
    msg.tool_calls = tool_calls or []
    return msg


def create_mock_ai_message_chunk(
    content: str, tool_call_chunks: List[Dict] = None
) -> MagicMock:
    """创建模拟的 AIMessageChunk 对象"""
    chunk = MagicMock()
    chunk.content = content
    chunk.tool_call_chunks = tool_call_chunks or []
    return chunk


def create_mock_tool_message(content: str, tool_call_id: str) -> MagicMock:
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
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "call_123"
        assert results[0].data["args_delta"] == '{"city": "北京"}'

    def test_on_tool_start(self):
        """测试 on_tool_start 事件"""
        event = {
            "event": "on_tool_start",
            "name": "get_weather",
            "run_id": "run_456",
            "data": {"input": {"city": "北京"}},
        }

        results = list(convert(event))

        # 现在是单个 TOOL_CALL_CHUNK（包含 id, name, args_delta）
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

        results = list(convert(event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "run_789"
        assert results[0].data["name"] == "get_time"

    def test_on_tool_end(self):
        """测试 on_tool_end 事件"""
        event = {
            "event": "on_tool_end",
            "run_id": "run_456",
            "data": {"output": {"weather": "晴天", "temperature": 25}},
        }

        results = list(convert(event))

        # 现在只有 TOOL_RESULT（边界事件由协议层自动处理）
        assert len(results) == 1
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

        results = list(convert(event))

        # 现在只有 TOOL_RESULT
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_RESULT
        assert results[0].data["result"] == "晴天，25度"

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

        # 现在是单个 TOOL_CALL_CHUNK（包含 id, name, args_delta）
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "call_abc"
        assert results[0].data["name"] == "get_weather"
        assert "上海" in results[0].data["args_delta"]

    def test_tool_message_result(self):
        """测试工具消息的结果"""
        msg = create_mock_tool_message('{"weather": "多云"}', "call_abc")
        event = {"tools": {"messages": [msg]}}

        results = list(convert(event))

        # 现在只有 TOOL_RESULT
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_RESULT
        assert results[0].data["id"] == "call_abc"
        assert "多云" in results[0].data["result"]

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

        # 应该有 2 个结果：1 个文本 + 1 个 TOOL_RESULT
        assert len(results) == 2
        assert results[0] == "正在查询..."
        assert results[1].event == EventType.TOOL_RESULT

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

        # 现在是单个 TOOL_CALL_CHUNK
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK

    def test_last_tool_message_result(self):
        """测试最后一条工具消息的结果"""
        ai_msg = create_mock_ai_message("之前的消息")
        tool_msg = create_mock_tool_message("工具结果", "call_ghi")
        event = {"messages": [ai_msg, tool_msg]}

        results = list(convert(event))

        # 只处理最后一条消息（工具消息），只有 TOOL_RESULT
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_RESULT

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

        # 验证结果：
        # - 1 TOOL_CALL_CHUNK（工具开始）
        # - 1 TOOL_RESULT（工具结束）
        # - 3 个文本内容
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
            all_results.extend(convert(event))

        # 验证结果：
        # - 1 TOOL_CALL_CHUNK（工具调用）
        # - 1 TOOL_RESULT（工具结果）
        # - 1 文本内容
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

        # 现在只有 TOOL_RESULT
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_RESULT
        assert results[0].data["result"] == "工具输出内容"


# =============================================================================
# 测试与 AgentRunServer 集成（使用 Mock）
# =============================================================================


class TestConvertWithMockedServer:
    """测试 convert 与 AgentRunServer 集成（使用 Mock）"""

    def test_mock_astream_events_integration(self):
        """测试模拟的 astream_events 流程集成"""
        # 模拟 LLM 返回的事件流
        mock_events = [
            # LLM 开始生成
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

        # 收集转换后的结果
        results = []
        for event in mock_events:
            results.extend(convert(event))

        # 验证结果
        assert len(results) == 3
        assert results[0] == "你好"
        assert results[1] == "，"
        assert results[2] == "世界！"

        # 组合文本
        full_text = "".join(results)
        assert full_text == "你好，世界！"

    def test_mock_astream_updates_integration(self):
        """测试模拟的 astream(updates) 流程集成"""
        # 模拟工具调用场景
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
            # 工具执行
            {
                "tools": {
                    "messages": [
                        create_mock_tool_message(
                            json.dumps(
                                {"city": "北京", "weather": "晴天", "temp": 25},
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
                    "messages": [
                        create_mock_ai_message("北京今天天气晴朗，气温25度。")
                    ]
                }
            },
        ]

        # 收集转换后的结果
        results = []
        for event in mock_events:
            results.extend(convert(event))

        # 验证事件顺序：
        # - 1 TOOL_CALL_CHUNK（工具调用）
        # - 1 TOOL_RESULT（工具结果）
        # - 1 文本回复
        assert len(results) == 3

        # 工具调用
        assert isinstance(results[0], AgentResult)
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["name"] == "get_weather"

        # 工具结果
        assert isinstance(results[1], AgentResult)
        assert results[1].event == EventType.TOOL_RESULT
        assert "晴天" in results[1].data["result"]

        # 最终文本回复
        assert results[2] == "北京今天天气晴朗，气温25度。"

    def test_mock_stream_values_integration(self):
        """测试模拟的 stream(values) 流程集成"""
        # 模拟 values 模式的事件流（每次返回完整状态）
        mock_events = [
            # 初始状态
            {"messages": [create_mock_ai_message("")]},
            # 工具调用
            {
                "messages": [
                    create_mock_ai_message(
                        "",
                        tool_calls=[{
                            "id": "tc_002",
                            "name": "get_time",
                            "args": {},
                        }],
                    )
                ]
            },
            # 工具结果
            {
                "messages": [
                    create_mock_ai_message(""),
                    create_mock_tool_message("2024-01-01 12:00:00", "tc_002"),
                ]
            },
            # 最终回复
            {
                "messages": [
                    create_mock_ai_message(""),
                    create_mock_tool_message("2024-01-01 12:00:00", "tc_002"),
                    create_mock_ai_message("现在是 2024年1月1日 12:00:00。"),
                ]
            },
        ]

        # 收集转换后的结果
        results = []
        for event in mock_events:
            results.extend(convert(event))

        # values 模式只处理最后一条消息
        # 第一个事件：空内容，无输出
        # 第二个事件：工具调用
        # 第三个事件：工具结果
        # 第四个事件：最终文本

        # 过滤非空结果
        non_empty = [r for r in results if r]
        assert len(non_empty) >= 1

        # 验证有工具调用事件
        tool_starts = [
            r
            for r in results
            if isinstance(r, AgentResult)
            and r.event == EventType.TOOL_CALL_CHUNK
        ]
        assert len(tool_starts) >= 1

        # 验证有最终文本
        text_results = [r for r in results if isinstance(r, str) and r]
        assert any("2024" in t for t in text_results)
