"""conversation_service.adapters.langchain_adapter 单元测试。

通过 Mock SessionStore 测试 OTSChatMessageHistory 的核心逻辑：
- messages 属性（读取事件并反序列化为 LangChain BaseMessage）
- add_messages（写入消息）
- clear（清空事件）
- auto_create_session
- _message_to_dict / _event_to_message 序列化/反序列化
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agentrun.conversation_service.adapters.langchain_adapter import (
    _event_to_message,
    _message_to_dict,
    OTSChatMessageHistory,
)
from agentrun.conversation_service.model import (
    ConversationEvent,
    ConversationSession,
)
from agentrun.conversation_service.session_store import SessionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_store() -> MagicMock:
    """创建 Mock SessionStore。"""
    store = MagicMock(spec=SessionStore)
    store.get_session.return_value = ConversationSession(
        agent_id="a",
        user_id="u",
        session_id="s",
        created_at=100,
        updated_at=200,
    )
    store.get_events.return_value = []
    store.delete_events.return_value = 0
    return store


# ---------------------------------------------------------------------------
# _message_to_dict 序列化
# ---------------------------------------------------------------------------


class TestMessageToDict:
    """_message_to_dict 测试。"""

    def test_human_message(self) -> None:
        msg = HumanMessage(content="hello")
        result = _message_to_dict(msg)
        assert result["lc_type"] == "human"
        assert result["content"] == "hello"

    def test_ai_message(self) -> None:
        msg = AIMessage(content="response")
        result = _message_to_dict(msg)
        assert result["lc_type"] == "ai"
        assert result["content"] == "response"

    def test_system_message(self) -> None:
        msg = SystemMessage(content="you are a helper")
        result = _message_to_dict(msg)
        assert result["lc_type"] == "system"

    def test_tool_message(self) -> None:
        msg = ToolMessage(content="result", tool_call_id="tc-1")
        result = _message_to_dict(msg)
        assert result["lc_type"] == "tool"
        assert result["tool_call_id"] == "tc-1"

    def test_ai_message_with_tool_calls(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {"q": "test"},
                    "id": "tc-1",
                    "type": "tool_call",
                },
            ],
        )
        result = _message_to_dict(msg)
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1

    def test_with_additional_kwargs(self) -> None:
        msg = HumanMessage(
            content="hi",
            additional_kwargs={"extra": "data"},
        )
        result = _message_to_dict(msg)
        assert result["additional_kwargs"] == {"extra": "data"}

    def test_with_name_and_id(self) -> None:
        msg = HumanMessage(content="hi", name="user", id="msg-1")
        result = _message_to_dict(msg)
        assert result["name"] == "user"
        assert result["id"] == "msg-1"

    def test_minimal_fields(self) -> None:
        """空 additional_kwargs 等不应出现在结果中。"""
        msg = HumanMessage(content="hi")
        result = _message_to_dict(msg)
        assert "additional_kwargs" not in result
        assert "name" not in result

    def test_ai_message_with_response_metadata(self) -> None:
        msg = AIMessage(
            content="ok",
            response_metadata={"model": "gpt-4"},
        )
        result = _message_to_dict(msg)
        assert result["response_metadata"] == {"model": "gpt-4"}


# ---------------------------------------------------------------------------
# _event_to_message 反序列化
# ---------------------------------------------------------------------------


class TestEventToMessage:
    """_event_to_message 测试。"""

    def test_human_message(self) -> None:
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="message",
            content={"lc_type": "human", "content": "hello"},
        )
        msg = _event_to_message(event)
        assert isinstance(msg, HumanMessage)
        assert msg.content == "hello"

    def test_ai_message(self) -> None:
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="message",
            content={"lc_type": "ai", "content": "response"},
        )
        msg = _event_to_message(event)
        assert isinstance(msg, AIMessage)
        assert msg.content == "response"

    def test_system_message(self) -> None:
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="message",
            content={"lc_type": "system", "content": "be helpful"},
        )
        msg = _event_to_message(event)
        assert isinstance(msg, SystemMessage)

    def test_tool_message(self) -> None:
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="message",
            content={
                "lc_type": "tool",
                "content": "result",
                "tool_call_id": "tc-1",
            },
        )
        msg = _event_to_message(event)
        assert isinstance(msg, ToolMessage)
        assert msg.tool_call_id == "tc-1"

    def test_tool_message_without_tool_call_id(self) -> None:
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="message",
            content={"lc_type": "tool", "content": "result"},
        )
        msg = _event_to_message(event)
        assert isinstance(msg, ToolMessage)
        assert msg.tool_call_id == ""

    def test_unknown_type_fallback(self) -> None:
        """未知类型回退到 HumanMessage。"""
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="message",
            content={"lc_type": "unknown_type", "content": "hi"},
        )
        msg = _event_to_message(event)
        assert isinstance(msg, HumanMessage)

    def test_missing_lc_type(self) -> None:
        """无 lc_type 默认 human。"""
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="message",
            content={"content": "hi"},
        )
        msg = _event_to_message(event)
        assert isinstance(msg, HumanMessage)

    def test_ai_with_tool_calls(self) -> None:
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="message",
            content={
                "lc_type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "name": "fn",
                        "args": {},
                        "id": "tc-1",
                        "type": "tool_call",
                    },
                ],
            },
        )
        msg = _event_to_message(event)
        assert isinstance(msg, AIMessage)
        assert len(msg.tool_calls) == 1

    def test_with_additional_kwargs(self) -> None:
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="message",
            content={
                "lc_type": "human",
                "content": "hi",
                "additional_kwargs": {"extra": True},
            },
        )
        msg = _event_to_message(event)
        assert msg.additional_kwargs == {"extra": True}

    def test_with_name_and_id(self) -> None:
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="message",
            content={
                "lc_type": "ai",
                "content": "ok",
                "name": "assistant",
                "id": "msg-1",
            },
        )
        msg = _event_to_message(event)
        assert msg.name == "assistant"
        assert msg.id == "msg-1"

    def test_with_response_metadata(self) -> None:
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="message",
            content={
                "lc_type": "ai",
                "content": "ok",
                "response_metadata": {"model": "gpt-4"},
            },
        )
        msg = _event_to_message(event)
        assert msg.response_metadata == {"model": "gpt-4"}


# ---------------------------------------------------------------------------
# _message_to_dict + _event_to_message round-trip
# ---------------------------------------------------------------------------


class TestMessageRoundTrip:
    """消息序列化/反序列化 round-trip。"""

    def test_human_roundtrip(self) -> None:
        original = HumanMessage(content="hello world")
        data = _message_to_dict(original)
        event = ConversationEvent("a", "u", "s", 1, "message", content=data)
        restored = _event_to_message(event)
        assert isinstance(restored, HumanMessage)
        assert restored.content == "hello world"

    def test_ai_with_tool_calls_roundtrip(self) -> None:
        original = AIMessage(
            content="let me search",
            tool_calls=[
                {
                    "name": "search",
                    "args": {"q": "test"},
                    "id": "tc-1",
                    "type": "tool_call",
                },
            ],
        )
        data = _message_to_dict(original)
        event = ConversationEvent("a", "u", "s", 1, "message", content=data)
        restored = _event_to_message(event)
        assert isinstance(restored, AIMessage)
        assert len(restored.tool_calls) == 1

    def test_tool_message_roundtrip(self) -> None:
        original = ToolMessage(content="result data", tool_call_id="tc-1")
        data = _message_to_dict(original)
        event = ConversationEvent("a", "u", "s", 1, "message", content=data)
        restored = _event_to_message(event)
        assert isinstance(restored, ToolMessage)
        assert restored.tool_call_id == "tc-1"

    def test_system_roundtrip(self) -> None:
        original = SystemMessage(content="be helpful")
        data = _message_to_dict(original)
        event = ConversationEvent("a", "u", "s", 1, "message", content=data)
        restored = _event_to_message(event)
        assert isinstance(restored, SystemMessage)


# ---------------------------------------------------------------------------
# OTSChatMessageHistory
# ---------------------------------------------------------------------------


class TestOTSChatMessageHistoryInit:
    """OTSChatMessageHistory 初始化测试。"""

    def test_auto_create_session_new(self) -> None:
        """Session 不存在时自动创建。"""
        store = _make_mock_store()
        store.get_session.return_value = None  # Session 不存在

        history = OTSChatMessageHistory(
            session_store=store,
            agent_id="a",
            user_id="u",
            session_id="s",
        )

        store.create_session.assert_called_once_with(
            "a", "u", "s", framework="langchain"
        )

    def test_auto_create_session_exists(self) -> None:
        """Session 已存在时不创建。"""
        store = _make_mock_store()
        history = OTSChatMessageHistory(
            session_store=store,
            agent_id="a",
            user_id="u",
            session_id="s",
        )
        store.create_session.assert_not_called()

    def test_auto_create_disabled(self) -> None:
        """禁用自动创建。"""
        store = _make_mock_store()
        store.get_session.return_value = None

        history = OTSChatMessageHistory(
            session_store=store,
            agent_id="a",
            user_id="u",
            session_id="s",
            auto_create_session=False,
        )

        store.get_session.assert_not_called()
        store.create_session.assert_not_called()


class TestOTSChatMessageHistoryMessages:
    """messages 属性测试。"""

    def test_empty(self) -> None:
        store = _make_mock_store()
        history = OTSChatMessageHistory(
            session_store=store,
            agent_id="a",
            user_id="u",
            session_id="s",
        )
        assert history.messages == []

    def test_with_events(self) -> None:
        store = _make_mock_store()
        events = [
            ConversationEvent(
                "a",
                "u",
                "s",
                1,
                "message",
                content={"lc_type": "human", "content": "hi"},
            ),
            ConversationEvent(
                "a",
                "u",
                "s",
                2,
                "message",
                content={"lc_type": "ai", "content": "hello"},
            ),
        ]
        store.get_events.return_value = events

        history = OTSChatMessageHistory(
            session_store=store,
            agent_id="a",
            user_id="u",
            session_id="s",
        )
        messages = history.messages

        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)

    def test_skips_bad_events(self) -> None:
        """反序列化失败的事件应被跳过。"""
        store = _make_mock_store()
        events = [
            ConversationEvent(
                "a",
                "u",
                "s",
                1,
                "message",
                content={"lc_type": "human", "content": "hi"},
            ),
            ConversationEvent(
                "a",
                "u",
                "s",
                2,
                "bad_type",
                content={"invalid": True},  # 缺少 content 字段不会报错
            ),
        ]
        store.get_events.return_value = events

        history = OTSChatMessageHistory(
            session_store=store,
            agent_id="a",
            user_id="u",
            session_id="s",
        )
        messages = history.messages
        # 两个都应成功（第二个会 fallback 到 HumanMessage）
        assert len(messages) == 2


class TestOTSChatMessageHistoryAddMessages:
    """add_messages 测试。"""

    def test_add_single(self) -> None:
        store = _make_mock_store()
        history = OTSChatMessageHistory(
            session_store=store,
            agent_id="a",
            user_id="u",
            session_id="s",
        )

        history.add_messages([HumanMessage(content="hello")])

        store.append_event.assert_called_once()
        call_args = store.append_event.call_args
        assert call_args[0][0] == "a"  # agent_id
        assert call_args[0][1] == "u"  # user_id
        assert call_args[0][2] == "s"  # session_id

    def test_add_multiple(self) -> None:
        store = _make_mock_store()
        history = OTSChatMessageHistory(
            session_store=store,
            agent_id="a",
            user_id="u",
            session_id="s",
        )

        history.add_messages([
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
            SystemMessage(content="be kind"),
        ])

        assert store.append_event.call_count == 3


class TestOTSChatMessageHistoryClear:
    """clear 测试。"""

    def test_clear(self) -> None:
        store = _make_mock_store()
        history = OTSChatMessageHistory(
            session_store=store,
            agent_id="a",
            user_id="u",
            session_id="s",
        )

        history.clear()

        store.delete_events.assert_called_once_with("a", "u", "s")
