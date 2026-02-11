"""conversation_service.session_store 单元测试。

通过 Mock OTSBackend 测试 SessionStore 的业务逻辑：
- Session CRUD（含级联删除）
- Event 追加/获取
- State 三级管理（app / user / session）
- 三级状态合并
- _apply_delta 增量更新逻辑
- from_memory_collection 工厂方法
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.conversation_service.model import (
    ConversationEvent,
    ConversationSession,
    StateData,
    StateScope,
)
from agentrun.conversation_service.ots_backend import OTSBackend
from agentrun.conversation_service.session_store import SessionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_backend() -> MagicMock:
    """创建 Mock OTSBackend。"""
    backend = MagicMock(spec=OTSBackend)

    # 同步方法默认返回值
    backend.get_session.return_value = None
    backend.list_sessions.return_value = []
    backend.list_all_sessions.return_value = []
    backend.get_events.return_value = []
    backend.get_state.return_value = None
    backend.put_event.return_value = 1
    backend.delete_events_by_session.return_value = 0
    backend.search_sessions.return_value = ([], 0)

    return backend


def _make_store(backend: MagicMock | None = None) -> SessionStore:
    """创建带 Mock backend 的 SessionStore。"""
    if backend is None:
        backend = _make_mock_backend()
    return SessionStore(backend)


# ---------------------------------------------------------------------------
# init 方法
# ---------------------------------------------------------------------------


class TestInitMethods:
    """init_tables / init_core_tables 等代理方法测试。"""

    def test_init_tables(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)
        store.init_tables()
        backend.init_tables.assert_called_once()

    def test_init_core_tables(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)
        store.init_core_tables()
        backend.init_core_tables.assert_called_once()

    def test_init_state_tables(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)
        store.init_state_tables()
        backend.init_state_tables.assert_called_once()

    def test_init_search_index(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)
        store.init_search_index()
        backend.init_search_index.assert_called_once()


# ---------------------------------------------------------------------------
# Session 管理
# ---------------------------------------------------------------------------


class TestCreateSession:
    """create_session 测试。"""

    def test_basic(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)

        session = store.create_session("agent", "user", "sess")

        assert session.agent_id == "agent"
        assert session.user_id == "user"
        assert session.session_id == "sess"
        assert session.created_at > 0
        assert session.updated_at == session.created_at
        assert session.version == 0
        backend.put_session.assert_called_once()

    def test_with_optional_fields(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)

        session = store.create_session(
            "a",
            "u",
            "s",
            is_pinned=True,
            summary="hello",
            labels='["tag"]',
            framework="adk",
            extensions={"key": "val"},
        )

        assert session.is_pinned is True
        assert session.summary == "hello"
        assert session.labels == '["tag"]'
        assert session.framework == "adk"
        assert session.extensions == {"key": "val"}


class TestGetSession:
    """get_session 测试。"""

    def test_found(self) -> None:
        backend = _make_mock_backend()
        expected = ConversationSession(
            agent_id="a",
            user_id="u",
            session_id="s",
            created_at=100,
            updated_at=200,
        )
        backend.get_session.return_value = expected

        store = _make_store(backend)
        result = store.get_session("a", "u", "s")

        assert result is expected
        backend.get_session.assert_called_once_with("a", "u", "s")

    def test_not_found(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)
        result = store.get_session("a", "u", "s")
        assert result is None


class TestListSessions:
    """list_sessions 测试。"""

    def test_basic(self) -> None:
        backend = _make_mock_backend()
        sessions = [
            ConversationSession("a", "u", "s1", 100, 200),
            ConversationSession("a", "u", "s2", 100, 300),
        ]
        backend.list_sessions.return_value = sessions

        store = _make_store(backend)
        result = store.list_sessions("a", "u")

        assert len(result) == 2
        backend.list_sessions.assert_called_once_with(
            "a", "u", limit=None, order_desc=True
        )

    def test_with_limit(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)
        store.list_sessions("a", "u", limit=5)
        backend.list_sessions.assert_called_once_with(
            "a", "u", limit=5, order_desc=True
        )


class TestListAllSessions:
    """list_all_sessions 测试。"""

    def test_basic(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)
        store.list_all_sessions("a")
        backend.list_all_sessions.assert_called_once_with("a", limit=None)

    def test_with_limit(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)
        store.list_all_sessions("a", limit=10)
        backend.list_all_sessions.assert_called_once_with("a", limit=10)


class TestSearchSessions:
    """search_sessions 测试。"""

    def test_basic(self) -> None:
        backend = _make_mock_backend()
        backend.search_sessions.return_value = ([], 0)

        store = _make_store(backend)
        sessions, total = store.search_sessions("a")

        assert sessions == []
        assert total == 0

    def test_with_all_filters(self) -> None:
        backend = _make_mock_backend()
        backend.search_sessions.return_value = ([], 0)

        store = _make_store(backend)
        store.search_sessions(
            "a",
            user_id="u",
            summary_keyword="hello",
            labels="tag",
            framework="adk",
            updated_after=100,
            updated_before=200,
            is_pinned=True,
            limit=10,
            offset=5,
        )

        backend.search_sessions.assert_called_once_with(
            "a",
            user_id="u",
            summary_keyword="hello",
            labels="tag",
            framework="adk",
            updated_after=100,
            updated_before=200,
            is_pinned=True,
            limit=10,
            offset=5,
        )


class TestUpdateSession:
    """update_session 乐观锁更新测试。"""

    def test_update_all_fields(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)

        store.update_session(
            "a",
            "u",
            "s",
            is_pinned=True,
            summary="new summary",
            labels='["new"]',
            extensions={"new": "ext"},
            version=1,
        )

        backend.update_session.assert_called_once()
        call_args = backend.update_session.call_args
        cols = call_args[0][3]  # columns_to_put
        assert cols["version"] == 2
        assert cols["is_pinned"] is True
        assert cols["summary"] == "new summary"
        assert cols["labels"] == '["new"]'
        assert json.loads(cols["extensions"]) == {"new": "ext"}

    def test_update_partial_fields(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)

        store.update_session("a", "u", "s", is_pinned=True, version=0)

        call_args = backend.update_session.call_args
        cols = call_args[0][3]
        assert "is_pinned" in cols
        assert "summary" not in cols
        assert "labels" not in cols
        assert "extensions" not in cols


class TestDeleteSession:
    """delete_session 级联删除测试。"""

    def test_cascade_delete(self) -> None:
        backend = _make_mock_backend()
        backend.delete_events_by_session.return_value = 3

        store = _make_store(backend)
        store.delete_session("a", "u", "s")

        # 1. 删除 Event
        backend.delete_events_by_session.assert_called_once_with("a", "u", "s")
        # 2. 删除 Session State
        backend.delete_state_row.assert_called_once_with(
            StateScope.SESSION, "a", "u", "s"
        )
        # 3. 删除 Session 行
        backend.delete_session_row.assert_called_once_with("a", "u", "s")


class TestDeleteEvents:
    """delete_events 测试。"""

    def test_basic(self) -> None:
        backend = _make_mock_backend()
        backend.delete_events_by_session.return_value = 5

        store = _make_store(backend)
        deleted = store.delete_events("a", "u", "s")

        assert deleted == 5
        backend.delete_events_by_session.assert_called_once_with("a", "u", "s")


# ---------------------------------------------------------------------------
# Event 管理
# ---------------------------------------------------------------------------


class TestAppendEvent:
    """append_event 测试。"""

    def test_basic(self) -> None:
        backend = _make_mock_backend()
        backend.put_event.return_value = 42
        backend.get_session.return_value = ConversationSession(
            "a",
            "u",
            "s",
            100,
            200,
            version=1,
        )

        store = _make_store(backend)
        event = store.append_event(
            "a",
            "u",
            "s",
            event_type="message",
            content={"msg": "hi"},
        )

        assert event.seq_id == 42
        assert event.type == "message"
        assert event.content == {"msg": "hi"}
        backend.put_event.assert_called_once()
        # 应更新 Session 的 updated_at
        backend.update_session.assert_called_once()

    def test_with_raw_event(self) -> None:
        backend = _make_mock_backend()
        backend.put_event.return_value = 1
        backend.get_session.return_value = None

        store = _make_store(backend)
        event = store.append_event(
            "a",
            "u",
            "s",
            event_type="adk_event",
            content={},
            raw_event='{"raw": true}',
        )

        assert event.raw_event == '{"raw": true}'

    def test_session_not_found_skips_update(self) -> None:
        """Session 不存在时不更新 updated_at。"""
        backend = _make_mock_backend()
        backend.put_event.return_value = 1
        backend.get_session.return_value = None

        store = _make_store(backend)
        store.append_event("a", "u", "s", "msg", {})

        backend.update_session.assert_not_called()

    def test_update_session_failure_ignored(self) -> None:
        """更新 Session 失败不应阻断事件写入。"""
        backend = _make_mock_backend()
        backend.put_event.return_value = 1
        backend.get_session.return_value = ConversationSession(
            "a",
            "u",
            "s",
            100,
            200,
            version=0,
        )
        backend.update_session.side_effect = Exception("OTS error")

        store = _make_store(backend)
        event = store.append_event("a", "u", "s", "msg", {})

        # 事件仍然返回
        assert event.seq_id == 1


class TestGetEvents:
    """get_events / get_recent_events 测试。"""

    def test_get_events(self) -> None:
        backend = _make_mock_backend()
        events = [
            ConversationEvent("a", "u", "s", 1, "msg", {"text": "1"}),
            ConversationEvent("a", "u", "s", 2, "msg", {"text": "2"}),
        ]
        backend.get_events.return_value = events

        store = _make_store(backend)
        result = store.get_events("a", "u", "s")

        assert len(result) == 2
        backend.get_events.assert_called_once_with(
            "a", "u", "s", direction="FORWARD"
        )

    def test_get_recent_events(self) -> None:
        backend = _make_mock_backend()
        # 倒序返回
        events = [
            ConversationEvent("a", "u", "s", 3, "msg"),
            ConversationEvent("a", "u", "s", 2, "msg"),
        ]
        backend.get_events.return_value = events

        store = _make_store(backend)
        result = store.get_recent_events("a", "u", "s", n=2)

        # 应翻转为正序
        assert result[0].seq_id == 2
        assert result[1].seq_id == 3
        backend.get_events.assert_called_once_with(
            "a", "u", "s", direction="BACKWARD", limit=2
        )


# ---------------------------------------------------------------------------
# State 管理
# ---------------------------------------------------------------------------


class TestGetSessionState:
    """get_session_state 测试。"""

    def test_exists(self) -> None:
        backend = _make_mock_backend()
        backend.get_state.return_value = StateData(state={"counter": 42})

        store = _make_store(backend)
        result = store.get_session_state("a", "u", "s")

        assert result == {"counter": 42}
        backend.get_state.assert_called_once_with(
            StateScope.SESSION, "a", "u", "s"
        )

    def test_not_exists(self) -> None:
        backend = _make_mock_backend()
        backend.get_state.return_value = None

        store = _make_store(backend)
        result = store.get_session_state("a", "u", "s")
        assert result == {}


class TestGetAppState:
    """get_app_state 测试。"""

    def test_exists(self) -> None:
        backend = _make_mock_backend()
        backend.get_state.return_value = StateData(state={"config": "val"})

        store = _make_store(backend)
        result = store.get_app_state("a")

        assert result == {"config": "val"}
        backend.get_state.assert_called_once_with(StateScope.APP, "a", "", "")

    def test_not_exists(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)
        assert store.get_app_state("a") == {}


class TestGetUserState:
    """get_user_state 测试。"""

    def test_exists(self) -> None:
        backend = _make_mock_backend()
        backend.get_state.return_value = StateData(state={"pref": "dark"})

        store = _make_store(backend)
        result = store.get_user_state("a", "u")

        assert result == {"pref": "dark"}
        backend.get_state.assert_called_once_with(StateScope.USER, "a", "u", "")

    def test_not_exists(self) -> None:
        backend = _make_mock_backend()
        store = _make_store(backend)
        assert store.get_user_state("a", "u") == {}


class TestUpdateSessionState:
    """update_session_state 增量更新测试。"""

    def test_first_write(self) -> None:
        """首次写入，过滤 None 值。"""
        backend = _make_mock_backend()
        backend.get_state.return_value = None

        store = _make_store(backend)
        store.update_session_state(
            "a", "u", "s", {"key": "val", "null_key": None}
        )

        backend.put_state.assert_called_once()
        call_args = backend.put_state.call_args
        # state 不应包含 null_key
        assert call_args.kwargs["state"] == {"key": "val"}
        assert call_args.kwargs["version"] == 0

    def test_merge_update(self) -> None:
        """增量合并已有 state。"""
        backend = _make_mock_backend()
        backend.get_state.return_value = StateData(
            state={"existing": "val", "to_delete": "old"},
            version=2,
        )

        store = _make_store(backend)
        store.update_session_state(
            "a",
            "u",
            "s",
            {"new_key": "new", "to_delete": None},
        )

        backend.put_state.assert_called_once()
        call_args = backend.put_state.call_args
        merged = call_args.kwargs["state"]
        assert merged == {"existing": "val", "new_key": "new"}
        assert "to_delete" not in merged
        assert call_args.kwargs["version"] == 2


class TestUpdateAppState:
    """update_app_state 测试。"""

    def test_first_write(self) -> None:
        backend = _make_mock_backend()
        backend.get_state.return_value = None

        store = _make_store(backend)
        store.update_app_state("a", {"config": "val"})

        backend.put_state.assert_called_once()
        call_args = backend.put_state.call_args
        assert call_args[0][0] == StateScope.APP
        assert call_args[0][1] == "a"


class TestUpdateUserState:
    """update_user_state 测试。"""

    def test_first_write(self) -> None:
        backend = _make_mock_backend()
        backend.get_state.return_value = None

        store = _make_store(backend)
        store.update_user_state("a", "u", {"pref": "dark"})

        backend.put_state.assert_called_once()
        call_args = backend.put_state.call_args
        assert call_args[0][0] == StateScope.USER
        assert call_args[0][1] == "a"
        assert call_args[0][2] == "u"


class TestGetMergedState:
    """get_merged_state 三级状态合并测试。"""

    def test_all_levels(self) -> None:
        backend = _make_mock_backend()
        # 模拟三级返回
        backend.get_state.side_effect = [
            StateData(state={"app_key": "app_val"}),  # APP
            StateData(state={"user_key": "user_val"}),  # USER
            StateData(state={"sess_key": "sess_val"}),  # SESSION
        ]

        store = _make_store(backend)
        result = store.get_merged_state("a", "u", "s")

        assert result == {
            "app_key": "app_val",
            "user_key": "user_val",
            "sess_key": "sess_val",
        }

    def test_override_order(self) -> None:
        """后者覆盖前者。"""
        backend = _make_mock_backend()
        backend.get_state.side_effect = [
            StateData(state={"key": "app"}),  # APP
            StateData(state={"key": "user"}),  # USER
            StateData(state={"key": "session"}),  # SESSION
        ]

        store = _make_store(backend)
        result = store.get_merged_state("a", "u", "s")
        assert result["key"] == "session"

    def test_missing_levels(self) -> None:
        """某级不存在视为空 dict。"""
        backend = _make_mock_backend()
        backend.get_state.side_effect = [
            None,  # APP
            StateData(state={"user_key": "val"}),  # USER
            None,  # SESSION
        ]

        store = _make_store(backend)
        result = store.get_merged_state("a", "u", "s")
        assert result == {"user_key": "val"}


# ---------------------------------------------------------------------------
# from_memory_collection 工厂方法
# ---------------------------------------------------------------------------


class TestFromMemoryCollection:
    """from_memory_collection 工厂方法测试。"""

    def test_import_error(self) -> None:
        """agentrun 主包未安装时抛 ImportError。"""
        with patch.dict(
            "sys.modules",
            {"agentrun.memory_collection": None, "agentrun.utils.config": None},
        ):
            with pytest.raises(ImportError, match="agentrun 主包未安装"):
                SessionStore.from_memory_collection("test-mc")

    def _make_mock_mc(
        self,
        endpoint: str = "https://inst.cn-hangzhou.ots.aliyuncs.com",
        instance_name: str = "test_instance",
        has_vs_config: bool = True,
    ) -> MagicMock:
        """构造 Mock MemoryCollection。"""
        mc = MagicMock()
        if not has_vs_config:
            mc.vector_store_config = None
        else:
            mc.vector_store_config = MagicMock()
            mc.vector_store_config.config = MagicMock()
            mc.vector_store_config.config.endpoint = endpoint
            mc.vector_store_config.config.instance_name = instance_name
        return mc

    @patch("tablestore.WriteRetryPolicy")
    @patch("tablestore.AsyncOTSClient")
    @patch("tablestore.OTSClient")
    def test_success(
        self,
        mock_ots_cls: MagicMock,
        mock_async_ots_cls: MagicMock,
        mock_wrp: MagicMock,
    ) -> None:
        """正常创建。"""
        mock_mc = self._make_mock_mc()

        with (
            patch(
                "agentrun.memory_collection.MemoryCollection.get_by_name",
                return_value=mock_mc,
            ),
            patch.dict(
                "os.environ",
                {
                    "AGENTRUN_ACCESS_KEY_ID": "ak_id",
                    "AGENTRUN_ACCESS_KEY_SECRET": "ak_secret",
                },
            ),
        ):
            store = SessionStore.from_memory_collection(
                "test-mc",
                table_prefix="p_",
            )

        assert isinstance(store, SessionStore)
        mock_ots_cls.assert_called_once()
        mock_async_ots_cls.assert_called_once()

    def test_missing_vector_store_config(self) -> None:
        mock_mc = self._make_mock_mc(has_vs_config=False)

        with patch(
            "agentrun.memory_collection.MemoryCollection.get_by_name",
            return_value=mock_mc,
        ):
            with pytest.raises(ValueError, match="缺少"):
                SessionStore.from_memory_collection("test-mc")

    def test_empty_endpoint(self) -> None:
        mock_mc = self._make_mock_mc(endpoint="")

        with patch(
            "agentrun.memory_collection.MemoryCollection.get_by_name",
            return_value=mock_mc,
        ):
            with pytest.raises(ValueError, match="endpoint 为空"):
                SessionStore.from_memory_collection("test-mc")

    def test_empty_instance_name(self) -> None:
        mock_mc = self._make_mock_mc(instance_name="")

        with patch(
            "agentrun.memory_collection.MemoryCollection.get_by_name",
            return_value=mock_mc,
        ):
            with pytest.raises(ValueError, match="instance_name 为空"):
                SessionStore.from_memory_collection("test-mc")

    def test_empty_credentials(self) -> None:
        mock_mc = self._make_mock_mc()

        with (
            patch(
                "agentrun.memory_collection.MemoryCollection.get_by_name",
                return_value=mock_mc,
            ),
            patch.dict(
                "os.environ",
                {
                    "AGENTRUN_ACCESS_KEY_ID": "",
                    "AGENTRUN_ACCESS_KEY_SECRET": "",
                },
                clear=False,
            ),
        ):
            with pytest.raises(ValueError, match="AK/SK 凭证为空"):
                SessionStore.from_memory_collection("test-mc")

    @patch("tablestore.WriteRetryPolicy")
    @patch("tablestore.AsyncOTSClient")
    @patch("tablestore.OTSClient")
    def test_with_sts_token(
        self,
        mock_ots_cls: MagicMock,
        mock_async_ots_cls: MagicMock,
        mock_wrp: MagicMock,
    ) -> None:
        """带 STS token。"""
        mock_mc = self._make_mock_mc()

        with (
            patch(
                "agentrun.memory_collection.MemoryCollection.get_by_name",
                return_value=mock_mc,
            ),
            patch.dict(
                "os.environ",
                {
                    "AGENTRUN_ACCESS_KEY_ID": "ak_id",
                    "AGENTRUN_ACCESS_KEY_SECRET": "ak_secret",
                    "AGENTRUN_SECURITY_TOKEN": "sts_token",
                },
            ),
        ):
            store = SessionStore.from_memory_collection("test-mc")

        assert isinstance(store, SessionStore)
        ots_kwargs = mock_ots_cls.call_args.kwargs
        assert ots_kwargs.get("sts_token") == "sts_token"

    @patch("tablestore.WriteRetryPolicy")
    @patch("tablestore.AsyncOTSClient")
    @patch("tablestore.OTSClient")
    def test_vpc_endpoint_conversion(
        self,
        mock_ots_cls: MagicMock,
        mock_async_ots_cls: MagicMock,
        mock_wrp: MagicMock,
    ) -> None:
        """VPC 地址转公网。"""
        mock_mc = self._make_mock_mc(
            endpoint="https://inst.cn-hangzhou.vpc.tablestore.aliyuncs.com",
        )

        with (
            patch(
                "agentrun.memory_collection.MemoryCollection.get_by_name",
                return_value=mock_mc,
            ),
            patch.dict(
                "os.environ",
                {
                    "AGENTRUN_ACCESS_KEY_ID": "ak_id",
                    "AGENTRUN_ACCESS_KEY_SECRET": "ak_secret",
                },
            ),
        ):
            SessionStore.from_memory_collection("test-mc")

        ots_call_args = mock_ots_cls.call_args[0]
        assert ots_call_args[0] == "https://inst.cn-hangzhou.ots.aliyuncs.com"


class TestFromMemoryCollectionAsync:
    """from_memory_collection_async 异步工厂方法测试。"""

    @pytest.mark.asyncio
    async def test_import_error(self) -> None:
        with patch.dict(
            "sys.modules",
            {"agentrun.memory_collection": None, "agentrun.utils.config": None},
        ):
            with pytest.raises(ImportError, match="agentrun 主包未安装"):
                await SessionStore.from_memory_collection_async("test-mc")

    def _make_mock_mc(
        self,
        endpoint: str = "https://inst.cn-hangzhou.ots.aliyuncs.com",
        instance_name: str = "inst",
        has_vs_config: bool = True,
    ) -> MagicMock:
        mc = MagicMock()
        if not has_vs_config:
            mc.vector_store_config = None
        else:
            mc.vector_store_config = MagicMock()
            mc.vector_store_config.config = MagicMock()
            mc.vector_store_config.config.endpoint = endpoint
            mc.vector_store_config.config.instance_name = instance_name
        return mc

    @pytest.mark.asyncio
    @patch("tablestore.WriteRetryPolicy")
    @patch("tablestore.AsyncOTSClient")
    @patch("tablestore.OTSClient")
    async def test_success(
        self,
        mock_ots_cls: MagicMock,
        mock_async_ots_cls: MagicMock,
        mock_wrp: MagicMock,
    ) -> None:
        mock_mc = self._make_mock_mc()

        with (
            patch(
                "agentrun.memory_collection.MemoryCollection.get_by_name_async",
                new=AsyncMock(return_value=mock_mc),
            ),
            patch.dict(
                "os.environ",
                {
                    "AGENTRUN_ACCESS_KEY_ID": "ak_id",
                    "AGENTRUN_ACCESS_KEY_SECRET": "ak_secret",
                },
            ),
        ):
            store = await SessionStore.from_memory_collection_async("test-mc")

        assert isinstance(store, SessionStore)

    @pytest.mark.asyncio
    async def test_missing_config(self) -> None:
        mock_mc = self._make_mock_mc(has_vs_config=False)

        with patch(
            "agentrun.memory_collection.MemoryCollection.get_by_name_async",
            new=AsyncMock(return_value=mock_mc),
        ):
            with pytest.raises(ValueError, match="缺少"):
                await SessionStore.from_memory_collection_async("test-mc")

    @pytest.mark.asyncio
    async def test_empty_endpoint(self) -> None:
        mock_mc = self._make_mock_mc(endpoint="")

        with patch(
            "agentrun.memory_collection.MemoryCollection.get_by_name_async",
            new=AsyncMock(return_value=mock_mc),
        ):
            with pytest.raises(ValueError, match="endpoint 为空"):
                await SessionStore.from_memory_collection_async("test-mc")

    @pytest.mark.asyncio
    async def test_empty_instance_name(self) -> None:
        mock_mc = self._make_mock_mc(instance_name="")

        with patch(
            "agentrun.memory_collection.MemoryCollection.get_by_name_async",
            new=AsyncMock(return_value=mock_mc),
        ):
            with pytest.raises(ValueError, match="instance_name 为空"):
                await SessionStore.from_memory_collection_async("test-mc")

    @pytest.mark.asyncio
    async def test_empty_credentials(self) -> None:
        mock_mc = self._make_mock_mc()

        with (
            patch(
                "agentrun.memory_collection.MemoryCollection.get_by_name_async",
                new=AsyncMock(return_value=mock_mc),
            ),
            patch.dict(
                "os.environ",
                {
                    "AGENTRUN_ACCESS_KEY_ID": "",
                    "AGENTRUN_ACCESS_KEY_SECRET": "",
                },
                clear=False,
            ),
        ):
            with pytest.raises(ValueError, match="AK/SK 凭证为空"):
                await SessionStore.from_memory_collection_async("test-mc")


# ---------------------------------------------------------------------------
# 异步方法测试
# ---------------------------------------------------------------------------


def _make_async_mock_backend() -> MagicMock:
    """创建带异步方法的 Mock OTSBackend。"""
    backend = MagicMock(spec=OTSBackend)

    # 异步方法
    backend.init_tables_async = AsyncMock()
    backend.init_core_tables_async = AsyncMock()
    backend.init_state_tables_async = AsyncMock()
    backend.init_search_index_async = AsyncMock()
    backend.put_session_async = AsyncMock()
    backend.get_session_async = AsyncMock(return_value=None)
    backend.list_sessions_async = AsyncMock(return_value=[])
    backend.list_all_sessions_async = AsyncMock(return_value=[])
    backend.search_sessions_async = AsyncMock(return_value=([], 0))
    backend.delete_session_row_async = AsyncMock()
    backend.update_session_async = AsyncMock()
    backend.put_event_async = AsyncMock(return_value=1)
    backend.get_events_async = AsyncMock(return_value=[])
    backend.delete_events_by_session_async = AsyncMock(return_value=0)
    backend.get_state_async = AsyncMock(return_value=None)
    backend.put_state_async = AsyncMock()
    backend.delete_state_row_async = AsyncMock()

    return backend


class TestInitMethodsAsync:
    """异步 init 方法测试。"""

    @pytest.mark.asyncio
    async def test_init_tables_async(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        await store.init_tables_async()
        backend.init_tables_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_core_tables_async(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        await store.init_core_tables_async()
        backend.init_core_tables_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_state_tables_async(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        await store.init_state_tables_async()
        backend.init_state_tables_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_search_index_async(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        await store.init_search_index_async()
        backend.init_search_index_async.assert_called_once()


class TestCreateSessionAsync:
    """create_session_async 测试。"""

    @pytest.mark.asyncio
    async def test_basic(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)

        session = await store.create_session_async("a", "u", "s")

        assert session.agent_id == "a"
        assert session.created_at > 0
        backend.put_session_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_optional_fields(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)

        session = await store.create_session_async(
            "a",
            "u",
            "s",
            is_pinned=True,
            summary="test",
            labels='["tag"]',
            framework="adk",
            extensions={"k": "v"},
        )

        assert session.is_pinned is True
        assert session.extensions == {"k": "v"}


class TestGetSessionAsync:
    """get_session_async 测试。"""

    @pytest.mark.asyncio
    async def test_found(self) -> None:
        backend = _make_async_mock_backend()
        expected = ConversationSession("a", "u", "s", 100, 200)
        backend.get_session_async.return_value = expected

        store = SessionStore(backend)
        result = await store.get_session_async("a", "u", "s")
        assert result is expected

    @pytest.mark.asyncio
    async def test_not_found(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        result = await store.get_session_async("a", "u", "s")
        assert result is None


class TestListSessionsAsync:
    """list_sessions_async / list_all_sessions_async 测试。"""

    @pytest.mark.asyncio
    async def test_list(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        await store.list_sessions_async("a", "u", limit=5)
        backend.list_sessions_async.assert_called_once_with(
            "a", "u", limit=5, order_desc=True
        )

    @pytest.mark.asyncio
    async def test_list_all(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        await store.list_all_sessions_async("a", limit=10)
        backend.list_all_sessions_async.assert_called_once_with("a", limit=10)


class TestSearchSessionsAsync:
    """search_sessions_async 测试。"""

    @pytest.mark.asyncio
    async def test_basic(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        sessions, total = await store.search_sessions_async("a")
        assert sessions == []
        assert total == 0


class TestUpdateSessionAsync:
    """update_session_async 测试。"""

    @pytest.mark.asyncio
    async def test_update_all_fields(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        await store.update_session_async(
            "a",
            "u",
            "s",
            is_pinned=True,
            summary="new",
            labels='["t"]',
            extensions={"e": 1},
            version=1,
        )
        backend.update_session_async.assert_called_once()
        call_args = backend.update_session_async.call_args
        cols = call_args[0][3]
        assert cols["version"] == 2
        assert cols["is_pinned"] is True
        assert cols["summary"] == "new"
        assert cols["labels"] == '["t"]'
        assert json.loads(cols["extensions"]) == {"e": 1}

    @pytest.mark.asyncio
    async def test_update_partial_fields(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        await store.update_session_async(
            "a", "u", "s", is_pinned=True, version=0
        )
        call_args = backend.update_session_async.call_args
        cols = call_args[0][3]
        assert "is_pinned" in cols
        assert "summary" not in cols
        assert "labels" not in cols
        assert "extensions" not in cols

    @pytest.mark.asyncio
    async def test_update_no_optional_fields(self) -> None:
        """不传任何可选字段。"""
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        await store.update_session_async("a", "u", "s", version=0)
        call_args = backend.update_session_async.call_args
        cols = call_args[0][3]
        assert "is_pinned" not in cols
        assert "summary" not in cols


class TestDeleteSessionAsync:
    """delete_session_async 级联删除测试。"""

    @pytest.mark.asyncio
    async def test_cascade(self) -> None:
        backend = _make_async_mock_backend()
        backend.delete_events_by_session_async.return_value = 3

        store = SessionStore(backend)
        await store.delete_session_async("a", "u", "s")

        backend.delete_events_by_session_async.assert_called_once()
        backend.delete_state_row_async.assert_called_once_with(
            StateScope.SESSION, "a", "u", "s"
        )
        backend.delete_session_row_async.assert_called_once_with("a", "u", "s")


class TestDeleteEventsAsync:
    """delete_events_async 测试。"""

    @pytest.mark.asyncio
    async def test_basic(self) -> None:
        backend = _make_async_mock_backend()
        backend.delete_events_by_session_async.return_value = 5

        store = SessionStore(backend)
        deleted = await store.delete_events_async("a", "u", "s")
        assert deleted == 5


class TestAppendEventAsync:
    """append_event_async 测试。"""

    @pytest.mark.asyncio
    async def test_basic(self) -> None:
        backend = _make_async_mock_backend()
        backend.put_event_async.return_value = 42
        backend.get_session_async.return_value = ConversationSession(
            "a",
            "u",
            "s",
            100,
            200,
            version=1,
        )

        store = SessionStore(backend)
        event = await store.append_event_async(
            "a",
            "u",
            "s",
            "msg",
            {"key": "val"},
        )

        assert event.seq_id == 42
        backend.update_session_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_not_found(self) -> None:
        backend = _make_async_mock_backend()
        backend.put_event_async.return_value = 1
        backend.get_session_async.return_value = None

        store = SessionStore(backend)
        event = await store.append_event_async("a", "u", "s", "msg", {})
        assert event.seq_id == 1
        backend.update_session_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_failure_ignored(self) -> None:
        backend = _make_async_mock_backend()
        backend.put_event_async.return_value = 1
        backend.get_session_async.return_value = ConversationSession(
            "a",
            "u",
            "s",
            100,
            200,
            version=0,
        )
        backend.update_session_async.side_effect = Exception("fail")

        store = SessionStore(backend)
        event = await store.append_event_async("a", "u", "s", "msg", {})
        assert event.seq_id == 1

    @pytest.mark.asyncio
    async def test_with_raw_event(self) -> None:
        backend = _make_async_mock_backend()
        backend.put_event_async.return_value = 1
        backend.get_session_async.return_value = None

        store = SessionStore(backend)
        event = await store.append_event_async(
            "a",
            "u",
            "s",
            "msg",
            {},
            raw_event='{"raw": true}',
        )
        assert event.raw_event == '{"raw": true}'


class TestGetEventsAsync:
    """get_events_async / get_recent_events_async 测试。"""

    @pytest.mark.asyncio
    async def test_get_events(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        await store.get_events_async("a", "u", "s")
        backend.get_events_async.assert_called_once_with(
            "a", "u", "s", direction="FORWARD"
        )

    @pytest.mark.asyncio
    async def test_get_recent_events(self) -> None:
        backend = _make_async_mock_backend()
        events = [
            ConversationEvent("a", "u", "s", 3, "msg"),
            ConversationEvent("a", "u", "s", 2, "msg"),
        ]
        backend.get_events_async.return_value = events

        store = SessionStore(backend)
        result = await store.get_recent_events_async("a", "u", "s", 2)

        assert result[0].seq_id == 2
        assert result[1].seq_id == 3


class TestStateAsync:
    """异步 State 管理测试。"""

    @pytest.mark.asyncio
    async def test_get_session_state(self) -> None:
        backend = _make_async_mock_backend()
        backend.get_state_async.return_value = StateData(state={"k": "v"})

        store = SessionStore(backend)
        result = await store.get_session_state_async("a", "u", "s")
        assert result == {"k": "v"}

    @pytest.mark.asyncio
    async def test_get_session_state_empty(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        result = await store.get_session_state_async("a", "u", "s")
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_app_state(self) -> None:
        backend = _make_async_mock_backend()
        backend.get_state_async.return_value = StateData(state={"app": True})

        store = SessionStore(backend)
        result = await store.get_app_state_async("a")
        assert result == {"app": True}

    @pytest.mark.asyncio
    async def test_get_app_state_empty(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        result = await store.get_app_state_async("a")
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_user_state(self) -> None:
        backend = _make_async_mock_backend()
        backend.get_state_async.return_value = StateData(state={"user": True})

        store = SessionStore(backend)
        result = await store.get_user_state_async("a", "u")
        assert result == {"user": True}

    @pytest.mark.asyncio
    async def test_get_user_state_empty(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        result = await store.get_user_state_async("a", "u")
        assert result == {}

    @pytest.mark.asyncio
    async def test_update_session_state(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        await store.update_session_state_async("a", "u", "s", {"k": "v"})
        backend.put_state_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_app_state(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        await store.update_app_state_async("a", {"k": "v"})
        backend.put_state_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_user_state(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)
        await store.update_user_state_async("a", "u", {"k": "v"})
        backend.put_state_async.assert_called_once()


class TestGetMergedStateAsync:
    """get_merged_state_async 测试。"""

    @pytest.mark.asyncio
    async def test_all_levels(self) -> None:
        backend = _make_async_mock_backend()
        backend.get_state_async.side_effect = [
            StateData(state={"app": 1}),
            StateData(state={"user": 2}),
            StateData(state={"sess": 3}),
        ]

        store = SessionStore(backend)
        result = await store.get_merged_state_async("a", "u", "s")
        assert result == {"app": 1, "user": 2, "sess": 3}


class TestApplyDeltaAsync:
    """_apply_delta_async 增量更新逻辑测试。"""

    @pytest.mark.asyncio
    async def test_first_write(self) -> None:
        backend = _make_async_mock_backend()
        store = SessionStore(backend)

        await store._apply_delta_async(
            StateScope.SESSION,
            "a",
            "u",
            "s",
            {"key": "val", "null": None},
        )

        backend.put_state_async.assert_called_once()
        call_args = backend.put_state_async.call_args
        assert call_args.kwargs["state"] == {"key": "val"}
        assert call_args.kwargs["version"] == 0

    @pytest.mark.asyncio
    async def test_merge_update(self) -> None:
        backend = _make_async_mock_backend()
        backend.get_state_async.return_value = StateData(
            state={"existing": "val", "to_remove": "old"},
            version=2,
        )

        store = SessionStore(backend)
        await store._apply_delta_async(
            StateScope.SESSION,
            "a",
            "u",
            "s",
            {"new": "val", "to_remove": None},
        )

        call_args = backend.put_state_async.call_args
        merged = call_args.kwargs["state"]
        assert merged == {"existing": "val", "new": "val"}
        assert call_args.kwargs["version"] == 2
