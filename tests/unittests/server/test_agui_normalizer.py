"""测试 AG-UI 事件规范化器

测试 AguiEventNormalizer 类的功能：
- 追踪工具调用状态
- 字符串和字典输入转换
- 状态重置

注意：边界事件（TOOL_CALL_START/END、TEXT_MESSAGE_START/END）
现在由协议层自动生成，AguiEventNormalizer 主要用于状态追踪。
"""

import pytest

from agentrun.server import AgentEvent, AguiEventNormalizer, EventType


class TestAguiEventNormalizer:
    """测试 AguiEventNormalizer 类"""

    def test_pass_through_text_events(self):
        """测试文本事件直接传递"""
        normalizer = AguiEventNormalizer()

        event = AgentEvent(
            event=EventType.TEXT,
            data={"delta": "Hello"},
        )
        results = list(normalizer.normalize(event))

        assert len(results) == 1
        assert results[0].event == EventType.TEXT
        assert results[0].data["delta"] == "Hello"

    def test_pass_through_custom_events(self):
        """测试自定义事件直接传递"""
        normalizer = AguiEventNormalizer()

        event = AgentEvent(
            event=EventType.CUSTOM,
            data={"name": "step_started", "value": {"step": "test"}},
        )
        results = list(normalizer.normalize(event))

        assert len(results) == 1
        assert results[0].event == EventType.CUSTOM

    def test_tool_call_chunk_tracking(self):
        """测试 TOOL_CALL_CHUNK 状态追踪"""
        normalizer = AguiEventNormalizer()

        # 发送 CHUNK 事件
        event = AgentEvent(
            event=EventType.TOOL_CALL_CHUNK,
            data={"id": "call_1", "name": "test", "args_delta": '{"x": 1}'},
        )
        results = list(normalizer.normalize(event))

        # 事件直接传递
        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL_CHUNK
        assert results[0].data["id"] == "call_1"

        # 状态被追踪
        assert "call_1" in normalizer.get_seen_tool_calls()
        assert "call_1" in normalizer.get_active_tool_calls()

    def test_tool_call_tracking(self):
        """测试 TOOL_CALL 状态追踪"""
        normalizer = AguiEventNormalizer()

        event = AgentEvent(
            event=EventType.TOOL_CALL,
            data={"id": "call_2", "name": "search", "args": '{"q": "hello"}'},
        )
        results = list(normalizer.normalize(event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL

        # 状态被追踪
        assert "call_2" in normalizer.get_seen_tool_calls()
        assert "call_2" in normalizer.get_active_tool_calls()

    def test_tool_result_marks_tool_call_complete(self):
        """测试 TOOL_RESULT 标记工具调用完成"""
        normalizer = AguiEventNormalizer()

        # 先发送工具调用
        chunk_event = AgentEvent(
            event=EventType.TOOL_CALL_CHUNK,
            data={"id": "call_1", "name": "test", "args_delta": "{}"},
        )
        list(normalizer.normalize(chunk_event))
        assert "call_1" in normalizer.get_active_tool_calls()

        # 发送结果
        result_event = AgentEvent(
            event=EventType.TOOL_RESULT,
            data={"id": "call_1", "result": "success"},
        )
        results = list(normalizer.normalize(result_event))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_RESULT

        # 工具调用不再活跃（但仍在已见列表中）
        assert "call_1" not in normalizer.get_active_tool_calls()
        assert "call_1" in normalizer.get_seen_tool_calls()

    def test_multiple_concurrent_tool_calls(self):
        """测试多个并发工具调用追踪"""
        normalizer = AguiEventNormalizer()

        # 开始两个工具调用
        for tool_id in ["call_a", "call_b"]:
            event = AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": tool_id,
                    "name": f"tool_{tool_id}",
                    "args_delta": "{}",
                },
            )
            list(normalizer.normalize(event))

        # 两个都应该是活跃的
        assert len(normalizer.get_active_tool_calls()) == 2
        assert "call_a" in normalizer.get_active_tool_calls()
        assert "call_b" in normalizer.get_active_tool_calls()

        # 结束其中一个
        result_event = AgentEvent(
            event=EventType.TOOL_RESULT,
            data={"id": "call_a", "result": "done"},
        )
        list(normalizer.normalize(result_event))

        # call_a 不再活跃，call_b 仍然活跃
        assert len(normalizer.get_active_tool_calls()) == 1
        assert "call_a" not in normalizer.get_active_tool_calls()
        assert "call_b" in normalizer.get_active_tool_calls()

    def test_string_input_converted_to_text(self):
        """测试字符串输入自动转换为文本事件"""
        normalizer = AguiEventNormalizer()

        results = list(normalizer.normalize("Hello"))

        assert len(results) == 1
        assert results[0].event == EventType.TEXT
        assert results[0].data["delta"] == "Hello"

    def test_dict_input_converted_to_agent_event(self):
        """测试字典输入自动转换为 AgentEvent"""
        normalizer = AguiEventNormalizer()

        event_dict = {
            "event": EventType.TEXT,
            "data": {"delta": "Hello from dict"},
        }
        results = list(normalizer.normalize(event_dict))

        assert len(results) == 1
        assert results[0].event == EventType.TEXT
        assert results[0].data["delta"] == "Hello from dict"

    def test_dict_with_string_event_type(self):
        """测试字符串事件类型的字典转换"""
        normalizer = AguiEventNormalizer()

        event_dict = {
            "event": "CUSTOM",
            "data": {"name": "test"},
        }
        results = list(normalizer.normalize(event_dict))

        assert len(results) == 1
        assert results[0].event == EventType.CUSTOM

    def test_invalid_dict_returns_nothing(self):
        """测试无效字典不产生事件"""
        normalizer = AguiEventNormalizer()

        # 缺少 event 字段
        event_dict = {"data": {"delta": "Hello"}}
        results = list(normalizer.normalize(event_dict))

        assert len(results) == 0

    def test_dict_with_invalid_event_type_value(self):
        """测试字典中无效的事件类型值"""
        normalizer = AguiEventNormalizer()

        # 事件类型既不是有效的 EventType 值也不是有效的枚举名称
        event_dict = {
            "event": "INVALID_EVENT_TYPE",
            "data": {"delta": "Hello"},
        }
        results = list(normalizer.normalize(event_dict))

        # 无效事件类型应该返回空
        assert len(results) == 0

    def test_dict_with_invalid_event_type_key(self):
        """测试字典中无效的事件类型键（尝试通过枚举名称）"""
        normalizer = AguiEventNormalizer()

        # 尝试使用无效的枚举名称
        event_dict = {
            "event": "NONEXISTENT",
            "data": {"delta": "Hello"},
        }
        results = list(normalizer.normalize(event_dict))

        # 无效事件类型应该返回空
        assert len(results) == 0

    def test_normalize_with_non_standard_input(self):
        """测试非标准输入类型"""
        normalizer = AguiEventNormalizer()

        # 传入整数
        results = list(normalizer.normalize(123))
        assert len(results) == 0

        # 传入 None
        results = list(normalizer.normalize(None))
        assert len(results) == 0

        # 传入列表
        results = list(normalizer.normalize([1, 2, 3]))
        assert len(results) == 0

    def test_reset_clears_state(self):
        """测试 reset 清空状态"""
        normalizer = AguiEventNormalizer()

        # 添加一些状态
        event = AgentEvent(
            event=EventType.TOOL_CALL_CHUNK,
            data={"id": "call_1", "name": "test", "args_delta": "{}"},
        )
        list(normalizer.normalize(event))
        assert len(normalizer.get_active_tool_calls()) == 1
        assert len(normalizer.get_seen_tool_calls()) == 1

        # 重置
        normalizer.reset()

        # 状态应该清空
        assert len(normalizer.get_active_tool_calls()) == 0
        assert len(normalizer.get_seen_tool_calls()) == 0

    def test_tool_call_with_empty_id(self):
        """测试空 tool_call_id 的 TOOL_CALL 事件"""
        normalizer = AguiEventNormalizer()

        event = AgentEvent(
            event=EventType.TOOL_CALL,
            data={"id": "", "name": "test", "args": "{}"},  # 空 id
        )
        results = list(normalizer.normalize(event))

        assert len(results) == 1
        # 空 id 不会被添加到追踪列表
        assert len(normalizer.get_seen_tool_calls()) == 0
        assert len(normalizer.get_active_tool_calls()) == 0

    def test_tool_call_chunk_with_empty_id(self):
        """测试空 tool_call_id 的 TOOL_CALL_CHUNK 事件"""
        normalizer = AguiEventNormalizer()

        event = AgentEvent(
            event=EventType.TOOL_CALL_CHUNK,
            data={"id": "", "name": "test", "args_delta": "{}"},  # 空 id
        )
        results = list(normalizer.normalize(event))

        assert len(results) == 1
        # 空 id 不会被添加到追踪列表
        assert len(normalizer.get_seen_tool_calls()) == 0

    def test_tool_call_chunk_without_name(self):
        """测试没有 name 的 TOOL_CALL_CHUNK 事件"""
        normalizer = AguiEventNormalizer()

        event = AgentEvent(
            event=EventType.TOOL_CALL_CHUNK,
            data={"id": "call_1", "args_delta": "{}"},  # 没有 name
        )
        results = list(normalizer.normalize(event))

        assert len(results) == 1
        # id 会被追踪，但没有 name
        assert "call_1" in normalizer.get_seen_tool_calls()
        # 没有 name 时不会添加到 active_tool_calls 的名称映射
        assert "call_1" not in normalizer.get_active_tool_calls()

    def test_tool_result_with_empty_id(self):
        """测试空 tool_call_id 的 TOOL_RESULT 事件"""
        normalizer = AguiEventNormalizer()

        # 先添加一个工具调用
        chunk_event = AgentEvent(
            event=EventType.TOOL_CALL_CHUNK,
            data={"id": "call_1", "name": "test", "args_delta": "{}"},
        )
        list(normalizer.normalize(chunk_event))
        assert "call_1" in normalizer.get_active_tool_calls()

        # 发送空 id 的结果
        result_event = AgentEvent(
            event=EventType.TOOL_RESULT,
            data={"id": "", "result": "done"},  # 空 id
        )
        results = list(normalizer.normalize(result_event))

        assert len(results) == 1
        # call_1 仍然是活跃的（因为结果的 id 为空）
        assert "call_1" in normalizer.get_active_tool_calls()

    def test_dict_with_event_type_enum(self):
        """测试字典中直接使用 EventType 枚举（覆盖 106->115 分支）"""
        normalizer = AguiEventNormalizer()

        event_dict = {
            "event": EventType.TEXT,  # 直接使用枚举，不是字符串
            "data": {"delta": "Hello"},
        }
        results = list(normalizer.normalize(event_dict))

        assert len(results) == 1
        assert results[0].event == EventType.TEXT
        assert results[0].data["delta"] == "Hello"

    def test_dict_with_event_type_enum_custom(self):
        """测试字典中直接使用 EventType.CUSTOM 枚举"""
        normalizer = AguiEventNormalizer()

        event_dict = {
            "event": EventType.CUSTOM,  # 直接使用枚举
            "data": {"name": "test_event", "value": {"key": "value"}},
        }
        results = list(normalizer.normalize(event_dict))

        assert len(results) == 1
        assert results[0].event == EventType.CUSTOM
        assert results[0].data["name"] == "test_event"

    def test_dict_with_event_type_enum_tool_call(self):
        """测试字典中直接使用 EventType.TOOL_CALL 枚举"""
        normalizer = AguiEventNormalizer()

        event_dict = {
            "event": EventType.TOOL_CALL,  # 直接使用枚举
            "data": {"id": "tc-1", "name": "test", "args": "{}"},
        }
        results = list(normalizer.normalize(event_dict))

        assert len(results) == 1
        assert results[0].event == EventType.TOOL_CALL

    def test_complete_tool_call_sequence(self):
        """测试完整的工具调用序列追踪"""
        normalizer = AguiEventNormalizer()
        all_results = []

        # 完整的工具调用序列
        events = [
            AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "call_1",
                    "name": "get_time",
                    "args_delta": '{"tz":',
                },
            ),
            AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "call_1", "args_delta": '"UTC"}'},
            ),
            AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "call_1", "result": "12:00"},
            ),
        ]

        for event in events:
            all_results.extend(normalizer.normalize(event))

        # 事件保持原样传递
        assert len(all_results) == 3
        event_types = [e.event for e in all_results]
        assert event_types == [
            EventType.TOOL_CALL_CHUNK,
            EventType.TOOL_CALL_CHUNK,
            EventType.TOOL_RESULT,
        ]

        # 最终状态：工具调用完成
        assert "call_1" not in normalizer.get_active_tool_calls()
        assert "call_1" in normalizer.get_seen_tool_calls()


class TestAguiEventNormalizerWithAguiProtocol:
    """使用 ag-ui-protocol 验证事件结构的测试"""

    @pytest.fixture
    def ag_ui_available(self):
        """检查 ag-ui-protocol 是否可用"""
        try:
            from ag_ui.core import ToolCallArgsEvent, ToolCallResultEvent

            return True
        except ImportError:
            pytest.skip("ag-ui-protocol not installed")

    def test_tool_call_events_have_valid_structure(self, ag_ui_available):
        """测试工具调用事件结构有效"""
        from ag_ui.core import ToolCallArgsEvent, ToolCallResultEvent

        normalizer = AguiEventNormalizer()

        events = [
            AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "call_1", "name": "test", "args_delta": '{"x": 1}'},
            ),
            AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "call_1", "result": "success"},
            ),
        ]

        all_results = []
        for event in events:
            all_results.extend(normalizer.normalize(event))

        # 验证 TOOL_CALL_CHUNK 可以映射到 ag-ui
        chunk_result = all_results[0]
        args_event = ToolCallArgsEvent(
            tool_call_id=chunk_result.data["id"],
            delta=chunk_result.data["args_delta"],
        )
        assert args_event.tool_call_id == "call_1"

        # 验证 TOOL_RESULT 可以映射到 ag-ui
        result_result = all_results[1]
        result_event = ToolCallResultEvent(
            message_id="msg_1",
            tool_call_id=result_result.data["id"],
            content=result_result.data["result"],
        )
        assert result_event.tool_call_id == "call_1"
