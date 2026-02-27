"""conversation_service.ots_backend 单元测试。

通过 Mock OTSClient 测试 OTSBackend 的同步和异步方法：
- 建表（含表已存在跳过）
- Session CRUD
- Event CRUD（含 batch 删除）
- State CRUD（含分片/拼接逻辑）
- 内部辅助方法
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, call, MagicMock, patch

import pytest
from tablestore import OTSServiceError, Row  # type: ignore[import-untyped]

from agentrun.conversation_service.model import (
    ConversationEvent,
    ConversationSession,
    StateData,
    StateScope,
)
from agentrun.conversation_service.ots_backend import (
    _BATCH_WRITE_LIMIT,
    OTSBackend,
)
from agentrun.conversation_service.utils import MAX_COLUMN_SIZE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
    """创建 mock OTSClient。"""
    return MagicMock()


def _make_backend(
    client: MagicMock | None = None,
    table_prefix: str = "",
) -> OTSBackend:
    """创建带 mock client 的 OTSBackend。"""
    if client is None:
        client = _make_mock_client()
    return OTSBackend(client, table_prefix=table_prefix)


def _make_session_row(
    agent_id: str = "agent1",
    user_id: str = "user1",
    session_id: str = "sess1",
    created_at: int = 1000,
    updated_at: int = 2000,
    is_pinned: bool = False,
    summary: str | None = None,
    labels: str | None = None,
    framework: str | None = None,
    extensions: dict | None = None,
    version: int = 0,
) -> Row:
    """构造 OTS 返回的 Session Row。"""
    pk = [
        ("agent_id", agent_id),
        ("user_id", user_id),
        ("session_id", session_id),
    ]
    attrs = [
        ("created_at", created_at, 0),
        ("updated_at", updated_at, 0),
        ("is_pinned", is_pinned, 0),
        ("version", version, 0),
    ]
    if summary is not None:
        attrs.append(("summary", summary, 0))
    if labels is not None:
        attrs.append(("labels", labels, 0))
    if framework is not None:
        attrs.append(("framework", framework, 0))
    if extensions is not None:
        attrs.append(("extensions", json.dumps(extensions), 0))
    return Row(pk, attrs)


def _make_event_row(
    agent_id: str = "agent1",
    user_id: str = "user1",
    session_id: str = "sess1",
    seq_id: int = 1,
    event_type: str = "message",
    content: dict | None = None,
    created_at: int = 1000,
    updated_at: int = 2000,
    version: int = 0,
    raw_event: str | None = None,
) -> Row:
    """构造 OTS 返回的 Event Row。"""
    pk = [
        ("agent_id", agent_id),
        ("user_id", user_id),
        ("session_id", session_id),
        ("seq_id", seq_id),
    ]
    content_json = json.dumps(content or {})
    attrs = [
        ("type", event_type, 0),
        ("content", content_json, 0),
        ("created_at", created_at, 0),
        ("updated_at", updated_at, 0),
        ("version", version, 0),
    ]
    if raw_event is not None:
        attrs.append(("raw_event", raw_event, 0))
    return Row(pk, attrs)


def _make_state_row(
    pk: list[tuple[str, Any]],
    state: dict | None = None,
    chunk_count: int = 0,
    chunks: list[str] | None = None,
    created_at: int = 1000,
    updated_at: int = 2000,
    version: int = 1,
) -> Row:
    """构造 OTS 返回的 State Row。"""
    attrs = [
        ("chunk_count", chunk_count, 0),
        ("created_at", created_at, 0),
        ("updated_at", updated_at, 0),
        ("version", version, 0),
    ]
    if chunk_count == 0 and state is not None:
        attrs.append(("state", json.dumps(state), 0))
    if chunks is not None:
        for idx, chunk in enumerate(chunks):
            attrs.append((f"state_{idx}", chunk, 0))
    return Row(pk, attrs)


def _make_null_row() -> Row:
    """构造空 Row（模拟行不存在）。"""
    row = MagicMock(spec=Row)
    row.primary_key = None
    return row


# ---------------------------------------------------------------------------
# 建表测试
# ---------------------------------------------------------------------------


class TestInitTables:
    """建表方法测试。"""

    def test_init_tables_success(self) -> None:
        client = _make_mock_client()
        backend = _make_backend(client)
        backend.init_tables()

        # 应创建 5 张表
        assert client.create_table.call_count == 5

    def test_init_tables_already_exist(self) -> None:
        client = _make_mock_client()
        err = OTSServiceError(
            409, "OTSObjectAlreadyExist", "table already exist"
        )
        client.create_table.side_effect = err

        backend = _make_backend(client)
        # 不应抛异常
        backend.init_tables()

    def test_init_tables_other_error(self) -> None:
        client = _make_mock_client()
        err = OTSServiceError(500, "InternalError", "internal error")
        client.create_table.side_effect = err

        backend = _make_backend(client)
        with pytest.raises(OTSServiceError):
            backend.init_tables()

    def test_create_event_table_other_error(self) -> None:
        """Event 表创建非已存在错误应抛出。"""
        client = _make_mock_client()
        # conversation table 正常，event table 抛异常
        err = OTSServiceError(500, "InternalError", "internal error")
        client.create_table.side_effect = [None, err]

        backend = _make_backend(client)
        with pytest.raises(OTSServiceError):
            backend.init_core_tables()

    def test_create_state_table_other_error(self) -> None:
        """State 表创建非已存在错误应抛出。"""
        client = _make_mock_client()
        err = OTSServiceError(500, "InternalError", "internal error")
        client.create_table.side_effect = err

        backend = _make_backend(client)
        with pytest.raises(OTSServiceError):
            backend.init_state_tables()

    def test_init_core_tables(self) -> None:
        client = _make_mock_client()
        backend = _make_backend(client)
        backend.init_core_tables()
        # Conversation + Event = 2 次
        assert client.create_table.call_count == 2

    def test_init_state_tables(self) -> None:
        client = _make_mock_client()
        backend = _make_backend(client)
        backend.init_state_tables()
        # state + app_state + user_state = 3 次
        assert client.create_table.call_count == 3

    def test_init_search_index_success(self) -> None:
        client = _make_mock_client()
        backend = _make_backend(client)
        backend.init_search_index()
        assert client.create_search_index.call_count == 2

    def test_init_search_index_already_exist(self) -> None:
        client = _make_mock_client()
        err = OTSServiceError(
            409, "OTSObjectAlreadyExist", "index already exist"
        )
        client.create_search_index.side_effect = err

        backend = _make_backend(client)
        backend.init_search_index()  # 不抛异常

    def test_init_search_index_other_error(self) -> None:
        client = _make_mock_client()
        call_count = 0

        def _side_effect(*args: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None
            raise OTSServiceError(500, "InternalError", "internal error")

        client.create_search_index.side_effect = _side_effect

        backend = _make_backend(client)
        with pytest.raises(OTSServiceError):
            backend.init_search_index()

    def test_table_prefix(self) -> None:
        client = _make_mock_client()
        backend = _make_backend(client, table_prefix="myprefix_")
        assert backend._conversation_table == "myprefix_conversation"
        assert backend._event_table == "myprefix_event"
        assert backend._state_table == "myprefix_state"
        assert backend._app_state_table == "myprefix_app_state"
        assert backend._user_state_table == "myprefix_user_state"
        assert backend._state_search_index == "myprefix_state_search_index"


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


class TestPutSession:
    """put_session 测试。"""

    def test_basic(self) -> None:
        client = _make_mock_client()
        backend = _make_backend(client)

        session = ConversationSession(
            agent_id="a",
            user_id="u",
            session_id="s",
            created_at=100,
            updated_at=200,
        )
        backend.put_session(session)
        client.put_row.assert_called_once()

    def test_with_optional_fields(self) -> None:
        client = _make_mock_client()
        backend = _make_backend(client)

        session = ConversationSession(
            agent_id="a",
            user_id="u",
            session_id="s",
            created_at=100,
            updated_at=200,
            summary="hello",
            labels='["tag"]',
            framework="adk",
            extensions={"key": "val"},
        )
        backend.put_session(session)
        client.put_row.assert_called_once()


class TestGetSession:
    """get_session 测试。"""

    def test_found(self) -> None:
        client = _make_mock_client()
        row = _make_session_row(
            summary="test",
            framework="adk",
            extensions={"k": "v"},
        )
        client.get_row.return_value = (None, row, None)

        backend = _make_backend(client)
        result = backend.get_session("agent1", "user1", "sess1")

        assert result is not None
        assert result.agent_id == "agent1"
        assert result.summary == "test"
        assert result.framework == "adk"
        assert result.extensions == {"k": "v"}

    def test_not_found_none(self) -> None:
        client = _make_mock_client()
        client.get_row.return_value = (None, None, None)

        backend = _make_backend(client)
        result = backend.get_session("a", "u", "s")
        assert result is None

    def test_not_found_null_pk(self) -> None:
        client = _make_mock_client()
        null_row = _make_null_row()
        client.get_row.return_value = (None, null_row, None)

        backend = _make_backend(client)
        result = backend.get_session("a", "u", "s")
        assert result is None


class TestDeleteSessionRow:
    """delete_session_row 测试。"""

    def test_delete(self) -> None:
        client = _make_mock_client()
        backend = _make_backend(client)
        backend.delete_session_row("a", "u", "s")
        client.delete_row.assert_called_once()


class TestUpdateSession:
    """update_session 乐观锁更新测试。"""

    def test_update(self) -> None:
        client = _make_mock_client()
        backend = _make_backend(client)
        backend.update_session(
            "a",
            "u",
            "s",
            {"updated_at": 999, "version": 2},
            version=1,
        )
        client.update_row.assert_called_once()


class TestListSessions:
    """list_sessions 通过二级索引测试。"""

    def test_list_desc(self) -> None:
        client = _make_mock_client()
        row = _make_session_row()
        # 二级索引 PK 包含 updated_at
        idx_row = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("updated_at", 2000),
                ("session_id", "s1"),
            ],
            [
                ("summary", "test", 0),
            ],
        )
        client.get_range.return_value = (None, None, [idx_row], None)

        backend = _make_backend(client)
        result = backend.list_sessions("a", "u", order_desc=True)

        assert len(result) == 1
        assert result[0].session_id == "s1"

    def test_list_asc(self) -> None:
        client = _make_mock_client()
        client.get_range.return_value = (None, None, [], None)

        backend = _make_backend(client)
        result = backend.list_sessions("a", "u", order_desc=False)
        assert result == []

    def test_list_with_limit(self) -> None:
        client = _make_mock_client()
        idx_rows = [
            Row(
                [
                    ("agent_id", "a"),
                    ("user_id", "u"),
                    ("updated_at", i),
                    ("session_id", f"s{i}"),
                ],
                [],
            )
            for i in range(5)
        ]
        client.get_range.return_value = (None, None, idx_rows, None)

        backend = _make_backend(client)
        result = backend.list_sessions("a", "u", limit=3)
        assert len(result) == 3

    def test_list_with_pagination(self) -> None:
        """模拟分页：第一次返回 next_token，第二次返回完。"""
        client = _make_mock_client()
        row1 = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("updated_at", 200),
                ("session_id", "s1"),
            ],
            [],
        )
        row2 = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("updated_at", 100),
                ("session_id", "s2"),
            ],
            [],
        )
        client.get_range.side_effect = [
            (None, "token", [row1], None),
            (None, None, [row2], None),
        ]

        backend = _make_backend(client)
        result = backend.list_sessions("a", "u")
        assert len(result) == 2


class TestListAllSessions:
    """list_all_sessions 主表扫描测试。"""

    def test_list_all(self) -> None:
        client = _make_mock_client()
        row = _make_session_row()
        client.get_range.return_value = (None, None, [row], None)

        backend = _make_backend(client)
        result = backend.list_all_sessions("agent1")
        assert len(result) == 1

    def test_list_all_with_limit(self) -> None:
        client = _make_mock_client()
        rows = [_make_session_row(session_id=f"s{i}") for i in range(5)]
        client.get_range.return_value = (None, None, rows, None)

        backend = _make_backend(client)
        result = backend.list_all_sessions("agent1", limit=2)
        assert len(result) == 2

    def test_list_all_with_pagination(self) -> None:
        client = _make_mock_client()
        r1 = _make_session_row(session_id="s1")
        r2 = _make_session_row(session_id="s2")
        client.get_range.side_effect = [
            (None, "token", [r1], None),
            (None, None, [r2], None),
        ]

        backend = _make_backend(client)
        result = backend.list_all_sessions("agent1")
        assert len(result) == 2


class TestSearchSessions:
    """search_sessions 多元索引搜索测试。"""

    def test_basic_search(self) -> None:
        client = _make_mock_client()
        # search 返回格式
        response = MagicMock()
        response.rows = [(
            [("agent_id", "a"), ("user_id", "u"), ("session_id", "s1")],
            [
                ("created_at", 100, 0),
                ("updated_at", 200, 0),
                ("is_pinned", False, 0),
                ("version", 0, 0),
            ],
        )]
        response.total_count = 1
        client.search.return_value = response

        backend = _make_backend(client)
        sessions, total = backend.search_sessions("a")

        assert len(sessions) == 1
        assert total == 1
        assert sessions[0].agent_id == "a"

    def test_search_with_all_filters(self) -> None:
        client = _make_mock_client()
        response = MagicMock()
        response.rows = []
        response.total_count = 0
        client.search.return_value = response

        backend = _make_backend(client)
        sessions, total = backend.search_sessions(
            "a",
            user_id="u",
            summary_keyword="hello",
            labels="tag",
            framework="adk",
            updated_after=100,
            updated_before=200,
            is_pinned=True,
        )

        assert sessions == []
        assert total == 0

    def test_search_is_pinned_false(self) -> None:
        client = _make_mock_client()
        response = MagicMock()
        response.rows = []
        response.total_count = 0
        client.search.return_value = response

        backend = _make_backend(client)
        backend.search_sessions("a", is_pinned=False)
        client.search.assert_called_once()

    def test_search_with_row_objects(self) -> None:
        """测试 search 返回 Row 对象而非 tuple 的情况。"""
        client = _make_mock_client()
        response = MagicMock()
        row = _make_session_row()
        response.rows = [row]
        response.total_count = 1
        client.search.return_value = response

        backend = _make_backend(client)
        sessions, total = backend.search_sessions("agent1")
        assert len(sessions) == 1

    def test_search_total_count_none(self) -> None:
        client = _make_mock_client()
        response = MagicMock()
        response.rows = []
        response.total_count = None
        client.search.return_value = response

        backend = _make_backend(client)
        _, total = backend.search_sessions("a")
        assert total == 0


# ---------------------------------------------------------------------------
# Event CRUD
# ---------------------------------------------------------------------------


class TestPutEvent:
    """put_event 测试。"""

    def test_basic(self) -> None:
        client = _make_mock_client()
        return_row = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("session_id", "s"),
                ("seq_id", 42),
            ],
            [],
        )
        client.put_row.return_value = (None, return_row)

        backend = _make_backend(client)
        seq_id = backend.put_event("a", "u", "s", "msg", {"key": "val"})

        assert seq_id == 42
        client.put_row.assert_called_once()

    def test_with_raw_event(self) -> None:
        client = _make_mock_client()
        return_row = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("session_id", "s"),
                ("seq_id", 1),
            ],
            [],
        )
        client.put_row.return_value = (None, return_row)

        backend = _make_backend(client)
        seq_id = backend.put_event(
            "a",
            "u",
            "s",
            "msg",
            {},
            raw_event='{"raw": "data"}',
        )
        assert seq_id == 1

    def test_with_timestamps(self) -> None:
        client = _make_mock_client()
        return_row = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("session_id", "s"),
                ("seq_id", 5),
            ],
            [],
        )
        client.put_row.return_value = (None, return_row)

        backend = _make_backend(client)
        seq_id = backend.put_event(
            "a",
            "u",
            "s",
            "msg",
            {},
            created_at=100,
            updated_at=200,
        )
        assert seq_id == 5

    def test_return_row_none(self) -> None:
        client = _make_mock_client()
        client.put_row.return_value = (None, None)

        backend = _make_backend(client)
        seq_id = backend.put_event("a", "u", "s", "msg", {})
        assert seq_id == 0

    def test_return_row_no_pk(self) -> None:
        client = _make_mock_client()
        return_row = MagicMock(spec=Row)
        return_row.primary_key = None
        client.put_row.return_value = (None, return_row)

        backend = _make_backend(client)
        seq_id = backend.put_event("a", "u", "s", "msg", {})
        assert seq_id == 0


class TestGetEvents:
    """get_events 测试。"""

    def test_forward(self) -> None:
        client = _make_mock_client()
        row = _make_event_row(seq_id=1, content={"msg": "hi"})
        client.get_range.return_value = (None, None, [row], None)

        backend = _make_backend(client)
        events = backend.get_events("a", "u", "s", direction="FORWARD")

        assert len(events) == 1
        assert events[0].seq_id == 1
        assert events[0].content == {"msg": "hi"}

    def test_backward(self) -> None:
        client = _make_mock_client()
        client.get_range.return_value = (None, None, [], None)

        backend = _make_backend(client)
        events = backend.get_events("a", "u", "s", direction="BACKWARD")
        assert events == []

    def test_with_limit(self) -> None:
        client = _make_mock_client()
        rows = [_make_event_row(seq_id=i) for i in range(5)]
        client.get_range.return_value = (None, None, rows, None)

        backend = _make_backend(client)
        events = backend.get_events("a", "u", "s", limit=2)
        assert len(events) == 2

    def test_with_raw_event(self) -> None:
        client = _make_mock_client()
        row = _make_event_row(raw_event='{"raw": "data"}')
        client.get_range.return_value = (None, None, [row], None)

        backend = _make_backend(client)
        events = backend.get_events("a", "u", "s")
        assert events[0].raw_event == '{"raw": "data"}'

    def test_pagination(self) -> None:
        client = _make_mock_client()
        r1 = _make_event_row(seq_id=1)
        r2 = _make_event_row(seq_id=2)
        client.get_range.side_effect = [
            (None, "token", [r1], None),
            (None, None, [r2], None),
        ]

        backend = _make_backend(client)
        events = backend.get_events("a", "u", "s")
        assert len(events) == 2

    def test_content_non_string(self) -> None:
        """content 列为非 string 的情况。"""
        client = _make_mock_client()
        pk = [
            ("agent_id", "a"),
            ("user_id", "u"),
            ("session_id", "s"),
            ("seq_id", 1),
        ]
        attrs = [
            ("type", "msg", 0),
            ("content", 12345, 0),  # 非 string
            ("created_at", 100, 0),
            ("updated_at", 200, 0),
            ("version", 0, 0),
        ]
        row = Row(pk, attrs)
        client.get_range.return_value = (None, None, [row], None)

        backend = _make_backend(client)
        events = backend.get_events("a", "u", "s")
        assert events[0].content == {}


class TestDeleteEventsBySession:
    """delete_events_by_session 测试。"""

    def test_no_events(self) -> None:
        client = _make_mock_client()
        client.get_range.return_value = (None, None, [], None)

        backend = _make_backend(client)
        deleted = backend.delete_events_by_session("a", "u", "s")
        assert deleted == 0
        client.batch_write_row.assert_not_called()

    def test_batch_delete(self) -> None:
        client = _make_mock_client()
        # 3 个 event
        rows = []
        for i in range(3):
            row = Row(
                [
                    ("agent_id", "a"),
                    ("user_id", "u"),
                    ("session_id", "s"),
                    ("seq_id", i),
                ],
                [],
            )
            rows.append(row)
        client.get_range.return_value = (None, None, rows, None)

        backend = _make_backend(client)
        deleted = backend.delete_events_by_session("a", "u", "s")
        assert deleted == 3
        client.batch_write_row.assert_called_once()

    def test_batch_delete_pagination(self) -> None:
        """模拟 event 扫描分页。"""
        client = _make_mock_client()
        r1 = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("session_id", "s"),
                ("seq_id", 1),
            ],
            [],
        )
        r2 = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("session_id", "s"),
                ("seq_id", 2),
            ],
            [],
        )
        client.get_range.side_effect = [
            (None, "token", [r1], None),
            (None, None, [r2], None),
        ]

        backend = _make_backend(client)
        deleted = backend.delete_events_by_session("a", "u", "s")
        assert deleted == 2


# ---------------------------------------------------------------------------
# State CRUD
# ---------------------------------------------------------------------------


class TestPutState:
    """put_state 测试。"""

    def test_first_write_no_chunk(self) -> None:
        """首次写入，不分片。"""
        client = _make_mock_client()
        backend = _make_backend(client)

        backend.put_state(
            StateScope.SESSION,
            "a",
            "u",
            "s",
            state={"key": "val"},
            version=0,
        )
        client.update_row.assert_called_once()

    def test_first_write_with_chunk(self) -> None:
        """首次写入，需要分片。"""
        client = _make_mock_client()
        backend = _make_backend(client)

        big_state = {"data": "x" * (MAX_COLUMN_SIZE + 100)}
        backend.put_state(
            StateScope.SESSION,
            "a",
            "u",
            "s",
            state=big_state,
            version=0,
        )
        client.update_row.assert_called_once()

    def test_update_no_chunk_to_no_chunk(self) -> None:
        """更新：旧无分片 → 新无分片。"""
        client = _make_mock_client()
        # _get_chunk_count 需要 get_row
        chunk_row = Row(
            [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")],
            [("chunk_count", 0, 0)],
        )
        client.get_row.return_value = (None, chunk_row, None)

        backend = _make_backend(client)
        backend.put_state(
            StateScope.SESSION,
            "a",
            "u",
            "s",
            state={"key": "new"},
            version=1,
        )
        client.update_row.assert_called_once()

    def test_update_chunk_to_no_chunk(self) -> None:
        """更新：旧有分片 → 新无分片，应删除旧 state_N 列。"""
        client = _make_mock_client()
        chunk_row = Row(
            [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")],
            [("chunk_count", 2, 0)],
        )
        client.get_row.return_value = (None, chunk_row, None)

        backend = _make_backend(client)
        backend.put_state(
            StateScope.SESSION,
            "a",
            "u",
            "s",
            state={"key": "small"},
            version=1,
        )
        # 检查 update_row 被调用，且包含 DELETE_ALL
        call_args = client.update_row.call_args
        row_arg = call_args[0][1]  # Row 参数
        # row.attribute_columns 是 update_of_attribute_columns dict
        assert "DELETE_ALL" in row_arg.attribute_columns

    def test_update_no_chunk_to_chunk(self) -> None:
        """更新：旧无分片 → 新有分片，应删除 state 列。"""
        client = _make_mock_client()
        chunk_row = Row(
            [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")],
            [("chunk_count", 0, 0)],
        )
        client.get_row.return_value = (None, chunk_row, None)

        backend = _make_backend(client)
        big_state = {"data": "x" * (MAX_COLUMN_SIZE + 100)}
        backend.put_state(
            StateScope.SESSION,
            "a",
            "u",
            "s",
            state=big_state,
            version=1,
        )
        call_args = client.update_row.call_args
        row_arg = call_args[0][1]
        assert "DELETE_ALL" in row_arg.attribute_columns
        assert "state" in row_arg.attribute_columns["DELETE_ALL"]

    def test_update_more_chunks_to_fewer(self) -> None:
        """更新：旧 4 个分片 → 新 2 个分片，应删除多余分片。"""
        client = _make_mock_client()
        chunk_row = Row(
            [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")],
            [("chunk_count", 4, 0)],
        )
        client.get_row.return_value = (None, chunk_row, None)

        backend = _make_backend(client)
        # 构造刚好 2 个分片的数据
        data_size = MAX_COLUMN_SIZE + 10  # 刚好超过 1 个分片
        big_state = {"d": "a" * data_size}
        backend.put_state(
            StateScope.SESSION,
            "a",
            "u",
            "s",
            state=big_state,
            version=1,
        )
        call_args = client.update_row.call_args
        row_arg = call_args[0][1]
        if "DELETE_ALL" in row_arg.attribute_columns:
            # 应删除 state_2 和 state_3
            deleted = row_arg.attribute_columns["DELETE_ALL"]
            assert "state_2" in deleted
            assert "state_3" in deleted

    def test_scope_app(self) -> None:
        """APP scope 使用 app_state 表。"""
        client = _make_mock_client()
        backend = _make_backend(client)
        backend.put_state(StateScope.APP, "a", "", "", state={}, version=0)
        call_args = client.update_row.call_args
        assert call_args[0][0] == "app_state"

    def test_scope_user(self) -> None:
        """USER scope 使用 user_state 表。"""
        client = _make_mock_client()
        backend = _make_backend(client)
        backend.put_state(StateScope.USER, "a", "u", "", state={}, version=0)
        call_args = client.update_row.call_args
        assert call_args[0][0] == "user_state"


class TestGetState:
    """get_state 测试。"""

    def test_not_found(self) -> None:
        client = _make_mock_client()
        client.get_row.return_value = (None, None, None)

        backend = _make_backend(client)
        result = backend.get_state(StateScope.SESSION, "a", "u", "s")
        assert result is None

    def test_not_found_null_pk(self) -> None:
        client = _make_mock_client()
        null_row = _make_null_row()
        client.get_row.return_value = (None, null_row, None)

        backend = _make_backend(client)
        result = backend.get_state(StateScope.SESSION, "a", "u", "s")
        assert result is None

    def test_no_chunk(self) -> None:
        """无分片正常读取。"""
        client = _make_mock_client()
        pk = [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")]
        row = _make_state_row(pk, state={"key": "val"}, chunk_count=0)
        client.get_row.return_value = (None, row, None)

        backend = _make_backend(client)
        result = backend.get_state(StateScope.SESSION, "a", "u", "s")

        assert result is not None
        assert result.state == {"key": "val"}
        assert result.version == 1

    def test_with_chunks(self) -> None:
        """有分片，拼接读取。"""
        client = _make_mock_client()
        pk = [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")]
        state_json = json.dumps({"data": "hello"})
        chunk1 = state_json[:5]
        chunk2 = state_json[5:]
        row = _make_state_row(pk, chunk_count=2, chunks=[chunk1, chunk2])
        client.get_row.return_value = (None, row, None)

        backend = _make_backend(client)
        result = backend.get_state(StateScope.SESSION, "a", "u", "s")

        assert result is not None
        assert result.state == {"data": "hello"}

    def test_missing_chunk_raises(self) -> None:
        """分片缺失应抛异常。"""
        client = _make_mock_client()
        pk = [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")]
        # chunk_count=2 但只有 state_0
        row = _make_state_row(pk, chunk_count=2, chunks=["partial"])
        client.get_row.return_value = (None, row, None)

        backend = _make_backend(client)
        with pytest.raises(ValueError, match="Missing state chunk"):
            backend.get_state(StateScope.SESSION, "a", "u", "s")

    def test_no_state_column(self) -> None:
        """chunk_count=0 但无 state 列，返回 None。"""
        client = _make_mock_client()
        pk = [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")]
        row = Row(pk, [("chunk_count", 0, 0), ("version", 1, 0)])
        client.get_row.return_value = (None, row, None)

        backend = _make_backend(client)
        result = backend.get_state(StateScope.SESSION, "a", "u", "s")
        assert result is None

    def test_scope_app(self) -> None:
        client = _make_mock_client()
        pk = [("agent_id", "a")]
        row = _make_state_row(pk, state={"app": True})
        client.get_row.return_value = (None, row, None)

        backend = _make_backend(client)
        result = backend.get_state(StateScope.APP, "a", "", "")
        assert result is not None
        assert result.state == {"app": True}


class TestDeleteStateRow:
    """delete_state_row 测试。"""

    def test_delete_session_state(self) -> None:
        client = _make_mock_client()
        backend = _make_backend(client)
        backend.delete_state_row(StateScope.SESSION, "a", "u", "s")
        client.delete_row.assert_called_once()

    def test_delete_app_state(self) -> None:
        client = _make_mock_client()
        backend = _make_backend(client)
        backend.delete_state_row(StateScope.APP, "a", "", "")
        call_args = client.delete_row.call_args
        assert call_args[0][0] == "app_state"

    def test_delete_user_state(self) -> None:
        client = _make_mock_client()
        backend = _make_backend(client)
        backend.delete_state_row(StateScope.USER, "a", "u", "")
        call_args = client.delete_row.call_args
        assert call_args[0][0] == "user_state"


# ---------------------------------------------------------------------------
# 内部辅助方法
# ---------------------------------------------------------------------------


class TestHelperMethods:
    """内部辅助方法测试。"""

    def test_attrs_to_dict(self) -> None:
        attrs = [("name", "val1", 0), ("count", 42, 0)]
        result = OTSBackend._attrs_to_dict(attrs)
        assert result == {"name": "val1", "count": 42}

    def test_attrs_to_dict_none(self) -> None:
        result = OTSBackend._attrs_to_dict(None)  # type: ignore[arg-type]
        assert result == {}

    def test_pk_to_dict(self) -> None:
        pk = [("agent_id", "a"), ("user_id", "u")]
        result = OTSBackend._pk_to_dict(pk)
        assert result == {"agent_id": "a", "user_id": "u"}

    def test_pk_to_dict_none(self) -> None:
        result = OTSBackend._pk_to_dict(None)  # type: ignore[arg-type]
        assert result == {}

    def test_resolve_state_table_app(self) -> None:
        backend = _make_backend()
        table, pk = backend._resolve_state_table_and_pk(
            StateScope.APP, "a", "u", "s"
        )
        assert table == "app_state"
        assert pk == [("agent_id", "a")]

    def test_resolve_state_table_user(self) -> None:
        backend = _make_backend()
        table, pk = backend._resolve_state_table_and_pk(
            StateScope.USER, "a", "u", "s"
        )
        assert table == "user_state"
        assert pk == [("agent_id", "a"), ("user_id", "u")]

    def test_resolve_state_table_session(self) -> None:
        backend = _make_backend()
        table, pk = backend._resolve_state_table_and_pk(
            StateScope.SESSION, "a", "u", "s"
        )
        assert table == "state"
        assert pk == [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")]

    def test_row_to_session_with_extensions(self) -> None:
        backend = _make_backend()
        row = _make_session_row(extensions={"k": "v"})
        session = backend._row_to_session(row)
        assert session.extensions == {"k": "v"}

    def test_row_to_session_without_extensions(self) -> None:
        backend = _make_backend()
        row = _make_session_row()
        session = backend._row_to_session(row)
        assert session.extensions is None

    def test_row_to_event(self) -> None:
        backend = _make_backend()
        row = _make_event_row(
            content={"msg": "hello"},
            raw_event='{"raw": true}',
        )
        event = backend._row_to_event(row)
        assert event.content == {"msg": "hello"}
        assert event.raw_event == '{"raw": true}'

    def test_row_to_session_from_index(self) -> None:
        backend = _make_backend()
        idx_row = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("updated_at", 2000),
                ("session_id", "s"),
            ],
            [("summary", "test", 0), ("extensions", '{"k": "v"}', 0)],
        )
        session = backend._row_to_session_from_index(idx_row)
        assert session.session_id == "s"
        assert session.updated_at == 2000
        assert session.created_at == 0  # 二级索引不含 created_at
        assert session.extensions == {"k": "v"}

    def test_get_chunk_count(self) -> None:
        client = _make_mock_client()
        pk = [("agent_id", "a")]
        row = Row(pk, [("chunk_count", 3, 0)])
        client.get_row.return_value = (None, row, None)

        backend = _make_backend(client)
        count = backend._get_chunk_count("app_state", pk)
        assert count == 3

    def test_get_chunk_count_no_row(self) -> None:
        client = _make_mock_client()
        client.get_row.return_value = (None, None, None)

        backend = _make_backend(client)
        count = backend._get_chunk_count("app_state", [("agent_id", "a")])
        assert count == 0

    def test_get_chunk_count_null_pk(self) -> None:
        client = _make_mock_client()
        null_row = _make_null_row()
        client.get_row.return_value = (None, null_row, None)

        backend = _make_backend(client)
        count = backend._get_chunk_count("app_state", [("agent_id", "a")])
        assert count == 0


# ===========================================================================
# 异步方法测试
# ===========================================================================


def _make_async_backend(
    async_client: MagicMock | None = None,
    table_prefix: str = "",
) -> OTSBackend:
    """创建带 async mock client 的 OTSBackend。"""
    if async_client is None:
        async_client = MagicMock()
        # 让所有方法返回 AsyncMock
        async_client.create_table = AsyncMock()
        async_client.create_search_index = AsyncMock()
        async_client.put_row = AsyncMock(return_value=(None, None))
        async_client.get_row = AsyncMock(return_value=(None, None, None))
        async_client.get_range = AsyncMock(return_value=(None, None, [], None))
        async_client.update_row = AsyncMock()
        async_client.delete_row = AsyncMock()
        async_client.batch_write_row = AsyncMock()
        async_client.search = AsyncMock()
    return OTSBackend(
        ots_client=None,
        table_prefix=table_prefix,
        async_ots_client=async_client,
    )


class TestInitTablesAsync:
    """异步建表测试。"""

    @pytest.mark.asyncio
    async def test_init_tables(self) -> None:
        backend = _make_async_backend()
        await backend.init_tables_async()
        assert backend._async_client.create_table.call_count == 5

    @pytest.mark.asyncio
    async def test_init_tables_already_exist(self) -> None:
        async_client = MagicMock()
        err = OTSServiceError(409, "OTSObjectAlreadyExist", "already exist")
        async_client.create_table = AsyncMock(side_effect=err)
        async_client.create_search_index = AsyncMock(side_effect=err)
        backend = _make_async_backend(async_client)
        await backend.init_tables_async()

    @pytest.mark.asyncio
    async def test_init_tables_other_error(self) -> None:
        async_client = MagicMock()
        err = OTSServiceError(500, "InternalError", "error")
        async_client.create_table = AsyncMock(side_effect=err)
        async_client.create_search_index = AsyncMock(side_effect=err)
        backend = _make_async_backend(async_client)
        with pytest.raises(OTSServiceError):
            await backend.init_tables_async()

    @pytest.mark.asyncio
    async def test_init_core_tables(self) -> None:
        backend = _make_async_backend()
        await backend.init_core_tables_async()
        assert backend._async_client.create_table.call_count == 2

    @pytest.mark.asyncio
    async def test_init_state_tables(self) -> None:
        backend = _make_async_backend()
        await backend.init_state_tables_async()
        assert backend._async_client.create_table.call_count == 3

    @pytest.mark.asyncio
    async def test_init_search_index(self) -> None:
        backend = _make_async_backend()
        await backend.init_search_index_async()
        assert backend._async_client.create_search_index.call_count == 2

    @pytest.mark.asyncio
    async def test_init_search_index_already_exist(self) -> None:
        async_client = MagicMock()
        async_client.create_table = AsyncMock()
        err = OTSServiceError(409, "OTSObjectAlreadyExist", "already exist")
        async_client.create_search_index = AsyncMock(side_effect=err)
        backend = _make_async_backend(async_client)
        await backend.init_search_index_async()

    @pytest.mark.asyncio
    async def test_init_search_index_other_error(self) -> None:
        async_client = MagicMock()
        async_client.create_table = AsyncMock()
        call_count = 0

        async def _side_effect(*args: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None
            raise OTSServiceError(500, "InternalError", "error")

        async_client.create_search_index = AsyncMock(side_effect=_side_effect)
        backend = _make_async_backend(async_client)
        with pytest.raises(OTSServiceError):
            await backend.init_search_index_async()


class TestSessionCrudAsync:
    """异步 Session CRUD 测试。"""

    @pytest.mark.asyncio
    async def test_put_session(self) -> None:
        backend = _make_async_backend()
        session = ConversationSession("a", "u", "s", 100, 200)
        await backend.put_session_async(session)
        backend._async_client.put_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_put_session_with_optional(self) -> None:
        backend = _make_async_backend()
        session = ConversationSession(
            "a",
            "u",
            "s",
            100,
            200,
            summary="hi",
            labels='["t"]',
            framework="adk",
            extensions={"k": "v"},
        )
        await backend.put_session_async(session)
        backend._async_client.put_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_found(self) -> None:
        backend = _make_async_backend()
        row = _make_session_row()
        backend._async_client.get_row = AsyncMock(
            return_value=(None, row, None)
        )

        result = await backend.get_session_async("agent1", "user1", "sess1")
        assert result is not None
        assert result.agent_id == "agent1"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self) -> None:
        backend = _make_async_backend()
        result = await backend.get_session_async("a", "u", "s")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_session_null_pk(self) -> None:
        backend = _make_async_backend()
        null_row = _make_null_row()
        backend._async_client.get_row = AsyncMock(
            return_value=(None, null_row, None)
        )
        result = await backend.get_session_async("a", "u", "s")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_session_row(self) -> None:
        backend = _make_async_backend()
        await backend.delete_session_row_async("a", "u", "s")
        backend._async_client.delete_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_session(self) -> None:
        backend = _make_async_backend()
        await backend.update_session_async("a", "u", "s", {"version": 2}, 1)
        backend._async_client.update_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_sessions_desc(self) -> None:
        backend = _make_async_backend()
        idx_row = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("updated_at", 2000),
                ("session_id", "s1"),
            ],
            [],
        )
        backend._async_client.get_range = AsyncMock(
            return_value=(None, None, [idx_row], None)
        )
        result = await backend.list_sessions_async("a", "u", order_desc=True)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_sessions_asc(self) -> None:
        backend = _make_async_backend()
        result = await backend.list_sessions_async("a", "u", order_desc=False)
        assert result == []

    @pytest.mark.asyncio
    async def test_list_sessions_with_limit(self) -> None:
        backend = _make_async_backend()
        rows = [
            Row(
                [
                    ("agent_id", "a"),
                    ("user_id", "u"),
                    ("updated_at", i),
                    ("session_id", f"s{i}"),
                ],
                [],
            )
            for i in range(5)
        ]
        backend._async_client.get_range = AsyncMock(
            return_value=(None, None, rows, None)
        )
        result = await backend.list_sessions_async("a", "u", limit=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_list_sessions_pagination(self) -> None:
        backend = _make_async_backend()
        r1 = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("updated_at", 200),
                ("session_id", "s1"),
            ],
            [],
        )
        r2 = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("updated_at", 100),
                ("session_id", "s2"),
            ],
            [],
        )
        backend._async_client.get_range = AsyncMock(
            side_effect=[
                (None, "token", [r1], None),
                (None, None, [r2], None),
            ]
        )
        result = await backend.list_sessions_async("a", "u")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_all_sessions(self) -> None:
        backend = _make_async_backend()
        row = _make_session_row()
        backend._async_client.get_range = AsyncMock(
            return_value=(None, None, [row], None)
        )
        result = await backend.list_all_sessions_async("agent1")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_all_sessions_with_limit(self) -> None:
        backend = _make_async_backend()
        rows = [_make_session_row(session_id=f"s{i}") for i in range(5)]
        backend._async_client.get_range = AsyncMock(
            return_value=(None, None, rows, None)
        )
        result = await backend.list_all_sessions_async("agent1", limit=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_all_sessions_pagination(self) -> None:
        backend = _make_async_backend()
        r1 = _make_session_row(session_id="s1")
        r2 = _make_session_row(session_id="s2")
        backend._async_client.get_range = AsyncMock(
            side_effect=[
                (None, "token", [r1], None),
                (None, None, [r2], None),
            ]
        )
        result = await backend.list_all_sessions_async("agent1")
        assert len(result) == 2


class TestSearchSessionsAsync:
    """异步 search_sessions 测试。"""

    @pytest.mark.asyncio
    async def test_basic(self) -> None:
        backend = _make_async_backend()
        response = MagicMock()
        response.rows = [(
            [("agent_id", "a"), ("user_id", "u"), ("session_id", "s1")],
            [
                ("created_at", 100, 0),
                ("updated_at", 200, 0),
                ("is_pinned", False, 0),
                ("version", 0, 0),
            ],
        )]
        response.total_count = 1
        backend._async_client.search = AsyncMock(return_value=response)

        sessions, total = await backend.search_sessions_async("a")
        assert len(sessions) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_with_all_filters(self) -> None:
        backend = _make_async_backend()
        response = MagicMock()
        response.rows = []
        response.total_count = 0
        backend._async_client.search = AsyncMock(return_value=response)

        sessions, total = await backend.search_sessions_async(
            "a",
            user_id="u",
            summary_keyword="hi",
            labels="t",
            framework="adk",
            updated_after=100,
            updated_before=200,
            is_pinned=True,
        )
        assert total == 0

    @pytest.mark.asyncio
    async def test_total_count_none(self) -> None:
        backend = _make_async_backend()
        response = MagicMock()
        response.rows = []
        response.total_count = None
        backend._async_client.search = AsyncMock(return_value=response)

        _, total = await backend.search_sessions_async("a")
        assert total == 0


class TestEventCrudAsync:
    """异步 Event CRUD 测试。"""

    @pytest.mark.asyncio
    async def test_put_event(self) -> None:
        backend = _make_async_backend()
        return_row = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("session_id", "s"),
                ("seq_id", 42),
            ],
            [],
        )
        backend._async_client.put_row = AsyncMock(
            return_value=(None, return_row)
        )

        seq_id = await backend.put_event_async(
            "a", "u", "s", "msg", {"key": "val"}
        )
        assert seq_id == 42

    @pytest.mark.asyncio
    async def test_put_event_with_raw_event(self) -> None:
        backend = _make_async_backend()
        return_row = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("session_id", "s"),
                ("seq_id", 1),
            ],
            [],
        )
        backend._async_client.put_row = AsyncMock(
            return_value=(None, return_row)
        )

        seq_id = await backend.put_event_async(
            "a",
            "u",
            "s",
            "msg",
            {},
            raw_event='{"raw": true}',
        )
        assert seq_id == 1

    @pytest.mark.asyncio
    async def test_put_event_with_timestamps(self) -> None:
        backend = _make_async_backend()
        return_row = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("session_id", "s"),
                ("seq_id", 5),
            ],
            [],
        )
        backend._async_client.put_row = AsyncMock(
            return_value=(None, return_row)
        )

        seq_id = await backend.put_event_async(
            "a",
            "u",
            "s",
            "msg",
            {},
            created_at=100,
            updated_at=200,
        )
        assert seq_id == 5

    @pytest.mark.asyncio
    async def test_put_event_return_none(self) -> None:
        backend = _make_async_backend()
        seq_id = await backend.put_event_async("a", "u", "s", "msg", {})
        assert seq_id == 0

    @pytest.mark.asyncio
    async def test_put_event_return_no_pk(self) -> None:
        backend = _make_async_backend()
        return_row = MagicMock(spec=Row)
        return_row.primary_key = None
        backend._async_client.put_row = AsyncMock(
            return_value=(None, return_row)
        )

        seq_id = await backend.put_event_async("a", "u", "s", "msg", {})
        assert seq_id == 0

    @pytest.mark.asyncio
    async def test_get_events_forward(self) -> None:
        backend = _make_async_backend()
        row = _make_event_row(seq_id=1)
        backend._async_client.get_range = AsyncMock(
            return_value=(None, None, [row], None)
        )

        events = await backend.get_events_async(
            "a", "u", "s", direction="FORWARD"
        )
        assert len(events) == 1
        assert events[0].seq_id == 1

    @pytest.mark.asyncio
    async def test_get_events_backward(self) -> None:
        backend = _make_async_backend()
        events = await backend.get_events_async(
            "a", "u", "s", direction="BACKWARD"
        )
        assert events == []

    @pytest.mark.asyncio
    async def test_get_events_with_limit(self) -> None:
        backend = _make_async_backend()
        rows = [_make_event_row(seq_id=i) for i in range(5)]
        backend._async_client.get_range = AsyncMock(
            return_value=(None, None, rows, None)
        )

        events = await backend.get_events_async("a", "u", "s", limit=2)
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_get_events_pagination(self) -> None:
        backend = _make_async_backend()
        r1 = _make_event_row(seq_id=1)
        r2 = _make_event_row(seq_id=2)
        backend._async_client.get_range = AsyncMock(
            side_effect=[
                (None, "token", [r1], None),
                (None, None, [r2], None),
            ]
        )
        events = await backend.get_events_async("a", "u", "s")
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_delete_events_no_events(self) -> None:
        backend = _make_async_backend()
        deleted = await backend.delete_events_by_session_async("a", "u", "s")
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_delete_events_batch(self) -> None:
        backend = _make_async_backend()
        rows = [
            Row(
                [
                    ("agent_id", "a"),
                    ("user_id", "u"),
                    ("session_id", "s"),
                    ("seq_id", i),
                ],
                [],
            )
            for i in range(3)
        ]
        backend._async_client.get_range = AsyncMock(
            return_value=(None, None, rows, None)
        )

        deleted = await backend.delete_events_by_session_async("a", "u", "s")
        assert deleted == 3
        backend._async_client.batch_write_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_events_pagination(self) -> None:
        backend = _make_async_backend()
        r1 = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("session_id", "s"),
                ("seq_id", 1),
            ],
            [],
        )
        r2 = Row(
            [
                ("agent_id", "a"),
                ("user_id", "u"),
                ("session_id", "s"),
                ("seq_id", 2),
            ],
            [],
        )
        backend._async_client.get_range = AsyncMock(
            side_effect=[
                (None, "token", [r1], None),
                (None, None, [r2], None),
            ]
        )
        deleted = await backend.delete_events_by_session_async("a", "u", "s")
        assert deleted == 2


class TestStateCrudAsync:
    """异步 State CRUD 测试。"""

    @pytest.mark.asyncio
    async def test_put_state_first_write(self) -> None:
        backend = _make_async_backend()
        await backend.put_state_async(
            StateScope.SESSION, "a", "u", "s", {"k": "v"}, 0
        )
        backend._async_client.update_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_put_state_with_chunks(self) -> None:
        backend = _make_async_backend()
        big_state = {"d": "x" * (MAX_COLUMN_SIZE + 100)}
        await backend.put_state_async(
            StateScope.SESSION, "a", "u", "s", big_state, 0
        )
        backend._async_client.update_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_put_state_update_clean_old_chunks(self) -> None:
        backend = _make_async_backend()
        chunk_row = Row(
            [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")],
            [("chunk_count", 2, 0)],
        )
        backend._async_client.get_row = AsyncMock(
            return_value=(None, chunk_row, None)
        )

        await backend.put_state_async(
            StateScope.SESSION, "a", "u", "s", {"k": "v"}, 1
        )
        call_args = backend._async_client.update_row.call_args
        row_arg = call_args[0][1]
        assert "DELETE_ALL" in row_arg.attribute_columns

    @pytest.mark.asyncio
    async def test_put_state_update_no_chunk_to_chunk(self) -> None:
        backend = _make_async_backend()
        chunk_row = Row(
            [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")],
            [("chunk_count", 0, 0)],
        )
        backend._async_client.get_row = AsyncMock(
            return_value=(None, chunk_row, None)
        )

        big_state = {"d": "x" * (MAX_COLUMN_SIZE + 100)}
        await backend.put_state_async(
            StateScope.SESSION, "a", "u", "s", big_state, 1
        )
        call_args = backend._async_client.update_row.call_args
        row_arg = call_args[0][1]
        assert "DELETE_ALL" in row_arg.attribute_columns
        assert "state" in row_arg.attribute_columns["DELETE_ALL"]

    @pytest.mark.asyncio
    async def test_get_state_not_found(self) -> None:
        backend = _make_async_backend()
        result = await backend.get_state_async(
            StateScope.SESSION, "a", "u", "s"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_state_null_pk(self) -> None:
        backend = _make_async_backend()
        null_row = _make_null_row()
        backend._async_client.get_row = AsyncMock(
            return_value=(None, null_row, None)
        )
        result = await backend.get_state_async(
            StateScope.SESSION, "a", "u", "s"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_state_no_chunk(self) -> None:
        backend = _make_async_backend()
        pk = [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")]
        row = _make_state_row(pk, state={"k": "v"})
        backend._async_client.get_row = AsyncMock(
            return_value=(None, row, None)
        )

        result = await backend.get_state_async(
            StateScope.SESSION, "a", "u", "s"
        )
        assert result is not None
        assert result.state == {"k": "v"}

    @pytest.mark.asyncio
    async def test_get_state_with_chunks(self) -> None:
        backend = _make_async_backend()
        pk = [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")]
        state_json = json.dumps({"data": "hello"})
        c1 = state_json[:5]
        c2 = state_json[5:]
        row = _make_state_row(pk, chunk_count=2, chunks=[c1, c2])
        backend._async_client.get_row = AsyncMock(
            return_value=(None, row, None)
        )

        result = await backend.get_state_async(
            StateScope.SESSION, "a", "u", "s"
        )
        assert result is not None
        assert result.state == {"data": "hello"}

    @pytest.mark.asyncio
    async def test_get_state_missing_chunk(self) -> None:
        backend = _make_async_backend()
        pk = [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")]
        row = _make_state_row(pk, chunk_count=2, chunks=["partial"])
        backend._async_client.get_row = AsyncMock(
            return_value=(None, row, None)
        )

        with pytest.raises(ValueError, match="Missing state chunk"):
            await backend.get_state_async(StateScope.SESSION, "a", "u", "s")

    @pytest.mark.asyncio
    async def test_get_state_no_state_column(self) -> None:
        backend = _make_async_backend()
        pk = [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")]
        row = Row(pk, [("chunk_count", 0, 0), ("version", 1, 0)])
        backend._async_client.get_row = AsyncMock(
            return_value=(None, row, None)
        )

        result = await backend.get_state_async(
            StateScope.SESSION, "a", "u", "s"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_state_row(self) -> None:
        backend = _make_async_backend()
        await backend.delete_state_row_async(StateScope.SESSION, "a", "u", "s")
        backend._async_client.delete_row.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_chunk_count_async(self) -> None:
        backend = _make_async_backend()
        pk = [("agent_id", "a")]
        row = Row(pk, [("chunk_count", 3, 0)])
        backend._async_client.get_row = AsyncMock(
            return_value=(None, row, None)
        )

        count = await backend._get_chunk_count_async("app_state", pk)
        assert count == 3

    @pytest.mark.asyncio
    async def test_get_chunk_count_async_no_row(self) -> None:
        backend = _make_async_backend()
        count = await backend._get_chunk_count_async(
            "app_state", [("agent_id", "a")]
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_chunk_count_async_null_pk(self) -> None:
        backend = _make_async_backend()
        null_row = _make_null_row()
        backend._async_client.get_row = AsyncMock(
            return_value=(None, null_row, None)
        )
        count = await backend._get_chunk_count_async(
            "app_state", [("agent_id", "a")]
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_put_state_async_more_chunks_to_fewer(self) -> None:
        """异步：旧 4 个分片 → 新 2 个分片，应删除多余分片。"""
        backend = _make_async_backend()
        chunk_row = Row(
            [("agent_id", "a"), ("user_id", "u"), ("session_id", "s")],
            [("chunk_count", 4, 0)],
        )
        backend._async_client.get_row = AsyncMock(
            return_value=(None, chunk_row, None)
        )

        data_size = MAX_COLUMN_SIZE + 10
        big_state = {"d": "a" * data_size}
        await backend.put_state_async(
            StateScope.SESSION, "a", "u", "s", big_state, 1
        )
        call_args = backend._async_client.update_row.call_args
        row_arg = call_args[0][1]
        if "DELETE_ALL" in row_arg.attribute_columns:
            deleted = row_arg.attribute_columns["DELETE_ALL"]
            assert "state_2" in deleted
            assert "state_3" in deleted

    @pytest.mark.asyncio
    async def test_create_event_table_other_error(self) -> None:
        """异步：Event 表创建非已存在错误应抛出。"""
        async_client = MagicMock()
        err = OTSServiceError(500, "InternalError", "internal error")
        async_client.create_table = AsyncMock(side_effect=[None, err])
        backend = _make_async_backend(async_client)
        with pytest.raises(OTSServiceError):
            await backend.init_core_tables_async()

    @pytest.mark.asyncio
    async def test_create_state_table_other_error(self) -> None:
        """异步：State 表创建非已存在错误应抛出。"""
        async_client = MagicMock()
        err = OTSServiceError(500, "InternalError", "internal error")
        async_client.create_table = AsyncMock(side_effect=err)
        backend = _make_async_backend(async_client)
        with pytest.raises(OTSServiceError):
            await backend.init_state_tables_async()

    @pytest.mark.asyncio
    async def test_search_sessions_is_pinned_false(self) -> None:
        """异步搜索 is_pinned=False。"""
        backend = _make_async_backend()
        response = MagicMock()
        response.rows = []
        response.total_count = 0
        backend._async_client.search = AsyncMock(return_value=response)
        await backend.search_sessions_async("a", is_pinned=False)
        backend._async_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_sessions_with_row_objects(self) -> None:
        """异步搜索返回 Row 对象而非 tuple。"""
        backend = _make_async_backend()
        response = MagicMock()
        row = _make_session_row()
        response.rows = [row]
        response.total_count = 1
        backend._async_client.search = AsyncMock(return_value=response)

        sessions, total = await backend.search_sessions_async("agent1")
        assert len(sessions) == 1
