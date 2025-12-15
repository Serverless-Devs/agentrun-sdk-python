"""测试 AG-UI 事件规范化器

测试 AguiEventNormalizer 类的功能：
- 自动补充 TOOL_CALL_START
- 忽略重复的 TOOL_CALL_START
- 在文本消息前自动发送 TOOL_CALL_END
- 使用 ag-ui-core 验证事件结构
"""

import pytest

from agentrun.server import AgentResult, AguiEventNormalizer, EventType


class TestAguiEventNormalizer:
    """测试 AguiEventNormalizer 类"""

    def test_pass_through_normal_events(self):
        """测试正常事件直接传递"""
        normalizer = AguiEventNormalizer()

        # 普通事件直接传递
        event = AgentResult(
            event=EventType.RUN_STARTED,
            data={"thread_id": "t1", "run_id": "r1"},
        )
        results = list(normalizer.normalize(event))

        assert len(results) == 1
        assert results[0].event == EventType.RUN_STARTED

    def test_auto_add_tool_call_start_before_args(self):
        """测试自动在 TOOL_CALL_ARGS 前补充 TOOL_CALL_START"""
        normalizer = AguiEventNormalizer()

        # 直接发送 ARGS，没有先发送 START
        event = AgentResult(
            event=EventType.TOOL_CALL_ARGS,
            data={"tool_call_id": "call_1", "delta": '{"x": 1}'},
        )
        results = list(normalizer.normalize(event))

        # 应该先发送 START，再发送 ARGS
        assert len(results) == 2
        assert results[0].event == EventType.TOOL_CALL_START
        assert results[0].data["tool_call_id"] == "call_1"
        assert results[1].event == EventType.TOOL_CALL_ARGS
        assert results[1].data["tool_call_id"] == "call_1"

    def test_ignore_duplicate_tool_call_start(self):
        """测试忽略重复的 TOOL_CALL_START"""
        normalizer = AguiEventNormalizer()

        # 第一次 START
        event1 = AgentResult(
            event=EventType.TOOL_CALL_START,
            data={"tool_call_id": "call_1", "tool_call_name": "test"},
        )
        results1 = list(normalizer.normalize(event1))
        assert len(results1) == 1

        # 重复的 START 应该被忽略
        event2 = AgentResult(
            event=EventType.TOOL_CALL_START,
            data={"tool_call_id": "call_1", "tool_call_name": "test"},
        )
        results2 = list(normalizer.normalize(event2))
        assert len(results2) == 0

    def test_auto_end_tool_calls_before_text_message(self):
        """测试在发送文本消息前自动结束工具调用"""
        normalizer = AguiEventNormalizer()

        # 开始工具调用
        start_event = AgentResult(
            event=EventType.TOOL_CALL_START,
            data={"tool_call_id": "call_1", "tool_call_name": "test"},
        )
        list(normalizer.normalize(start_event))

        # 发送参数
        args_event = AgentResult(
            event=EventType.TOOL_CALL_ARGS,
            data={"tool_call_id": "call_1", "delta": "{}"},
        )
        list(normalizer.normalize(args_event))

        # 工具调用应该是活跃的
        assert "call_1" in normalizer.get_active_tool_calls()

        # 发送文本消息
        text_event = AgentResult(
            event=EventType.TEXT_MESSAGE_CONTENT,
            data={"message_id": "msg_1", "delta": "Hello"},
        )
        results = list(normalizer.normalize(text_event))

        # 应该先发送 TOOL_CALL_END，再发送 TEXT_MESSAGE_CONTENT
        assert len(results) == 2
        assert results[0].event == EventType.TOOL_CALL_END
        assert results[0].data["tool_call_id"] == "call_1"
        assert results[1].event == EventType.TEXT_MESSAGE_CONTENT

        # 工具调用应该已结束
        assert len(normalizer.get_active_tool_calls()) == 0

    def test_auto_add_start_and_end_before_result(self):
        """测试在 TOOL_CALL_RESULT 前自动补充 START 和 END"""
        normalizer = AguiEventNormalizer()

        # 直接发送 RESULT，没有 START 和 END
        event = AgentResult(
            event=EventType.TOOL_CALL_RESULT,
            data={"tool_call_id": "call_1", "result": "success"},
        )
        results = list(normalizer.normalize(event))

        # 应该按顺序发送 START -> END -> RESULT
        assert len(results) == 3
        assert results[0].event == EventType.TOOL_CALL_START
        assert results[1].event == EventType.TOOL_CALL_END
        assert results[2].event == EventType.TOOL_CALL_RESULT

    def test_multiple_concurrent_tool_calls(self):
        """测试多个并发工具调用"""
        normalizer = AguiEventNormalizer()

        # 开始两个工具调用
        for tool_id in ["call_a", "call_b"]:
            event = AgentResult(
                event=EventType.TOOL_CALL_START,
                data={
                    "tool_call_id": tool_id,
                    "tool_call_name": f"tool_{tool_id}",
                },
            )
            list(normalizer.normalize(event))

        # 两个都应该是活跃的
        assert len(normalizer.get_active_tool_calls()) == 2
        assert "call_a" in normalizer.get_active_tool_calls()
        assert "call_b" in normalizer.get_active_tool_calls()

        # 结束其中一个
        end_event = AgentResult(
            event=EventType.TOOL_CALL_END,
            data={"tool_call_id": "call_a"},
        )
        list(normalizer.normalize(end_event))

        # call_a 应该已结束，call_b 仍然活跃
        assert len(normalizer.get_active_tool_calls()) == 1
        assert "call_b" in normalizer.get_active_tool_calls()

        # 发送文本消息应该结束 call_b
        text_event = AgentResult(
            event=EventType.TEXT_MESSAGE_CONTENT,
            data={"delta": "Done"},
        )
        results = list(normalizer.normalize(text_event))

        assert len(results) == 2
        assert results[0].event == EventType.TOOL_CALL_END
        assert results[0].data["tool_call_id"] == "call_b"

    def test_string_input_converted_to_text_message(self):
        """测试字符串输入自动转换为文本消息"""
        normalizer = AguiEventNormalizer()

        results = list(normalizer.normalize("Hello"))

        assert len(results) == 1
        assert results[0].event == EventType.TEXT_MESSAGE_CONTENT
        assert results[0].data["delta"] == "Hello"

    def test_dict_input_converted_to_agent_result(self):
        """测试字典输入自动转换为 AgentResult"""
        normalizer = AguiEventNormalizer()

        event_dict = {
            "event": EventType.TOOL_CALL_START,
            "data": {"tool_call_id": "call_1", "tool_call_name": "test"},
        }
        results = list(normalizer.normalize(event_dict))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_START

    def test_reset_clears_state(self):
        """测试 reset 清空状态"""
        normalizer = AguiEventNormalizer()

        # 添加一些状态
        event = AgentResult(
            event=EventType.TOOL_CALL_START,
            data={"tool_call_id": "call_1", "tool_call_name": "test"},
        )
        list(normalizer.normalize(event))
        assert len(normalizer.get_active_tool_calls()) == 1

        # 重置
        normalizer.reset()

        # 状态应该清空
        assert len(normalizer.get_active_tool_calls()) == 0

    def test_complete_tool_call_sequence(self):
        """测试完整的工具调用序列"""
        normalizer = AguiEventNormalizer()
        all_results = []

        # 正确顺序的事件
        events = [
            AgentResult(
                event=EventType.TOOL_CALL_START,
                data={"tool_call_id": "call_1", "tool_call_name": "get_time"},
            ),
            AgentResult(
                event=EventType.TOOL_CALL_ARGS,
                data={"tool_call_id": "call_1", "delta": '{"tz": "UTC"}'},
            ),
            AgentResult(
                event=EventType.TOOL_CALL_END,
                data={"tool_call_id": "call_1"},
            ),
            AgentResult(
                event=EventType.TOOL_CALL_RESULT,
                data={"tool_call_id": "call_1", "result": "12:00"},
            ),
        ]

        for event in events:
            all_results.extend(normalizer.normalize(event))

        # 应该保持原样（不需要补充）
        assert len(all_results) == 4
        event_types = [e.event for e in all_results]
        assert event_types == [
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_ARGS,
            EventType.TOOL_CALL_END,
            EventType.TOOL_CALL_RESULT,
        ]


class TestAguiEventNormalizerWithAguiProtocol:
    """使用 ag-ui-protocol 验证事件结构的测试

    需要安装 ag-ui-protocol: pip install ag-ui-protocol
    """

    @pytest.fixture
    def ag_ui_available(self):
        """检查 ag-ui-protocol 是否可用"""
        try:
            from ag_ui.core import (
                ToolCallArgsEvent,
                ToolCallEndEvent,
                ToolCallResultEvent,
                ToolCallStartEvent,
            )

            return True
        except ImportError:
            pytest.skip("ag-ui-protocol not installed")

    def test_normalized_events_are_valid_ag_ui_events(self, ag_ui_available):
        """测试规范化后的事件符合 AG-UI 协议"""
        from ag_ui.core import (
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
            ToolCallStartEvent,
        )

        normalizer = AguiEventNormalizer()

        # 模拟错误的事件顺序：直接发送 ARGS
        events = [
            AgentResult(
                event=EventType.TOOL_CALL_ARGS,
                data={"tool_call_id": "call_1", "delta": '{"x": 1}'},
            ),
            AgentResult(
                event=EventType.TOOL_CALL_RESULT,
                data={"tool_call_id": "call_1", "result": "success"},
            ),
        ]

        all_results = []
        for event in events:
            all_results.extend(normalizer.normalize(event))

        # 验证事件顺序
        event_types = [e.event for e in all_results]
        assert event_types == [
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_ARGS,
            EventType.TOOL_CALL_END,
            EventType.TOOL_CALL_RESULT,
        ]

        # 使用 ag-ui-protocol 验证每个事件
        # 注意：参数使用 camelCase，但属性访问使用 snake_case
        for result in all_results:
            if result.event == EventType.TOOL_CALL_START:
                event = ToolCallStartEvent(
                    toolCallId=result.data["tool_call_id"],
                    toolCallName=result.data.get("tool_call_name", ""),
                )
                assert event.tool_call_id == "call_1"
            elif result.event == EventType.TOOL_CALL_ARGS:
                event = ToolCallArgsEvent(
                    toolCallId=result.data["tool_call_id"],
                    delta=result.data["delta"],
                )
                assert event.tool_call_id == "call_1"
            elif result.event == EventType.TOOL_CALL_END:
                event = ToolCallEndEvent(
                    toolCallId=result.data["tool_call_id"],
                )
                assert event.tool_call_id == "call_1"
            elif result.event == EventType.TOOL_CALL_RESULT:
                # ToolCallResultEvent 需要 messageId 和 content
                event = ToolCallResultEvent(
                    messageId="msg_1",
                    toolCallId=result.data["tool_call_id"],
                    content=result.data["result"],
                )
                assert event.tool_call_id == "call_1"

    def test_event_sequence_validation(self, ag_ui_available):
        """测试事件序列验证"""
        normalizer = AguiEventNormalizer()

        # 发送完整的工具调用序列
        events = [
            AgentResult(
                event=EventType.TOOL_CALL_START,
                data={"tool_call_id": "call_1", "tool_call_name": "test"},
            ),
            AgentResult(
                event=EventType.TOOL_CALL_ARGS,
                data={"tool_call_id": "call_1", "delta": "{}"},
            ),
            AgentResult(
                event=EventType.TOOL_CALL_END,
                data={"tool_call_id": "call_1"},
            ),
            AgentResult(
                event=EventType.TOOL_CALL_RESULT,
                data={"tool_call_id": "call_1", "result": "done"},
            ),
        ]

        all_results = []
        for event in events:
            all_results.extend(normalizer.normalize(event))

        # 验证所有事件使用相同的 tool_call_id
        for result in all_results:
            assert result.data.get("tool_call_id") == "call_1"

        # 验证事件类型顺序
        expected_types = [
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_ARGS,
            EventType.TOOL_CALL_END,
            EventType.TOOL_CALL_RESULT,
        ]
        actual_types = [e.event for e in all_results]
        assert actual_types == expected_types
