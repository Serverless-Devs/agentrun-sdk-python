"""conversation_service.model 单元测试。

覆盖 ConversationSession、ConversationEvent、StateData 数据类
以及 StateScope 枚举。
"""

from __future__ import annotations

import json

from agentrun.conversation_service.model import (
    ConversationEvent,
    ConversationSession,
    DEFAULT_APP_STATE_TABLE,
    DEFAULT_CONVERSATION_SEARCH_INDEX,
    DEFAULT_CONVERSATION_SECONDARY_INDEX,
    DEFAULT_CONVERSATION_TABLE,
    DEFAULT_EVENT_TABLE,
    DEFAULT_STATE_TABLE,
    DEFAULT_USER_STATE_TABLE,
    StateData,
    StateScope,
)

# ---------------------------------------------------------------------------
# 表名常量
# ---------------------------------------------------------------------------


class TestTableConstants:
    """表名常量校验。"""

    def test_default_table_names(self) -> None:
        assert DEFAULT_CONVERSATION_TABLE == "conversation"
        assert DEFAULT_EVENT_TABLE == "event"
        assert DEFAULT_STATE_TABLE == "state"
        assert DEFAULT_APP_STATE_TABLE == "app_state"
        assert DEFAULT_USER_STATE_TABLE == "user_state"
        assert (
            DEFAULT_CONVERSATION_SECONDARY_INDEX
            == "conversation_secondary_index"
        )
        assert DEFAULT_CONVERSATION_SEARCH_INDEX == "conversation_search_index"


# ---------------------------------------------------------------------------
# StateScope 枚举
# ---------------------------------------------------------------------------


class TestStateScope:
    """StateScope 枚举测试。"""

    def test_values(self) -> None:
        assert StateScope.APP.value == "app"
        assert StateScope.USER.value == "user"
        assert StateScope.SESSION.value == "session"

    def test_is_str_enum(self) -> None:
        assert isinstance(StateScope.APP, str)
        assert StateScope.APP == "app"


# ---------------------------------------------------------------------------
# ConversationSession
# ---------------------------------------------------------------------------


class TestConversationSession:
    """ConversationSession 数据类测试。"""

    def test_required_fields(self) -> None:
        session = ConversationSession(
            agent_id="agent1",
            user_id="user1",
            session_id="sess1",
            created_at=1000,
            updated_at=2000,
        )
        assert session.agent_id == "agent1"
        assert session.user_id == "user1"
        assert session.session_id == "sess1"
        assert session.created_at == 1000
        assert session.updated_at == 2000

    def test_default_values(self) -> None:
        session = ConversationSession(
            agent_id="a",
            user_id="u",
            session_id="s",
            created_at=0,
            updated_at=0,
        )
        assert session.is_pinned is False
        assert session.summary is None
        assert session.labels is None
        assert session.framework is None
        assert session.extensions is None
        assert session.version == 0

    def test_all_fields(self) -> None:
        session = ConversationSession(
            agent_id="a",
            user_id="u",
            session_id="s",
            created_at=100,
            updated_at=200,
            is_pinned=True,
            summary="hello",
            labels='["tag1"]',
            framework="adk",
            extensions={"key": "val"},
            version=3,
        )
        assert session.is_pinned is True
        assert session.summary == "hello"
        assert session.labels == '["tag1"]'
        assert session.framework == "adk"
        assert session.extensions == {"key": "val"}
        assert session.version == 3


# ---------------------------------------------------------------------------
# ConversationEvent
# ---------------------------------------------------------------------------


class TestConversationEvent:
    """ConversationEvent 数据类测试。"""

    def test_required_fields(self) -> None:
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="message",
        )
        assert event.agent_id == "a"
        assert event.seq_id == 1
        assert event.type == "message"

    def test_default_values(self) -> None:
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=None,
            type="test",
        )
        assert event.content == {}
        assert event.created_at == 0
        assert event.updated_at == 0
        assert event.version == 0
        assert event.raw_event is None

    def test_content_as_json(self) -> None:
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="msg",
            content={"key": "值", "num": 42},
        )
        result = event.content_as_json()
        parsed = json.loads(result)
        assert parsed == {"key": "值", "num": 42}
        # ensure_ascii=False 应保留中文
        assert "值" in result

    def test_content_from_json(self) -> None:
        raw = '{"key": "value", "nested": {"a": 1}}'
        result = ConversationEvent.content_from_json(raw)
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_content_as_json_empty(self) -> None:
        event = ConversationEvent(
            agent_id="a",
            user_id="u",
            session_id="s",
            seq_id=1,
            type="msg",
        )
        assert event.content_as_json() == "{}"

    def test_content_from_json_empty(self) -> None:
        result = ConversationEvent.content_from_json("{}")
        assert result == {}


# ---------------------------------------------------------------------------
# StateData
# ---------------------------------------------------------------------------


class TestStateData:
    """StateData 数据类测试。"""

    def test_default_values(self) -> None:
        sd = StateData()
        assert sd.state == {}
        assert sd.created_at == 0
        assert sd.updated_at == 0
        assert sd.version == 0

    def test_with_values(self) -> None:
        sd = StateData(
            state={"counter": 42},
            created_at=100,
            updated_at=200,
            version=3,
        )
        assert sd.state == {"counter": 42}
        assert sd.created_at == 100
        assert sd.updated_at == 200
        assert sd.version == 3

    def test_state_default_factory_isolation(self) -> None:
        """确保不同实例的 state 字典是独立的。"""
        sd1 = StateData()
        sd2 = StateData()
        sd1.state["key"] = "val"
        assert "key" not in sd2.state
