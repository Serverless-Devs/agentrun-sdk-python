"""ADK OTSSessionService 适配器单元测试。

通过 Mock SessionStore 测试 OTSSessionService 的核心逻辑：
- Event 序列化 round-trip（raw_event 列）
- 三级 state 映射（app / user / session）
- CRUD 操作
- list_sessions（含 user_id=None 场景）
"""

from __future__ import annotations

import json
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from google.adk.events.event import Event  # type: ignore[import-untyped]
from google.adk.events.event_actions import (
    EventActions,
)  # type: ignore[import-untyped]
from google.adk.sessions.base_session_service import (
    GetSessionConfig,
)  # type: ignore[import-untyped]
from google.adk.sessions.session import Session  # type: ignore[import-untyped]
from google.genai import types  # type: ignore[import-untyped]
import pytest

from agentrun.conversation_service.adapters.adk_adapter import (
    _extract_display_content,
    _extract_state_delta,
    OTSSessionService,
)
from agentrun.conversation_service.model import (
    ConversationEvent,
    ConversationSession,
)
from agentrun.conversation_service.session_store import SessionStore

# -------------------------------------------------------------------
# 工具函数测试
# -------------------------------------------------------------------


class TestExtractStateDelta:
    """_extract_state_delta 单元测试。"""

    def test_empty_state(self) -> None:
        result = _extract_state_delta({})
        assert result == {
            "app": {},
            "user": {},
            "session": {},
        }

    def test_session_only(self) -> None:
        result = _extract_state_delta({"key1": "val1", "key2": 42})
        assert result["session"] == {
            "key1": "val1",
            "key2": 42,
        }
        assert result["app"] == {}
        assert result["user"] == {}

    def test_app_prefix(self) -> None:
        result = _extract_state_delta({"app:config": "value"})
        assert result["app"] == {"config": "value"}
        assert result["session"] == {}

    def test_user_prefix(self) -> None:
        result = _extract_state_delta({"user:name": "Alice"})
        assert result["user"] == {"name": "Alice"}
        assert result["session"] == {}

    def test_temp_prefix_excluded(self) -> None:
        result = _extract_state_delta({"temp:cache": "data", "real_key": "val"})
        assert result["session"] == {"real_key": "val"}
        assert result["app"] == {}
        assert result["user"] == {}

    def test_mixed_prefixes(self) -> None:
        result = _extract_state_delta({
            "app:setting": True,
            "user:pref": "dark",
            "session_var": 123,
            "temp:scratch": "ignored",
        })
        assert result["app"] == {"setting": True}
        assert result["user"] == {"pref": "dark"}
        assert result["session"] == {"session_var": 123}


class TestExtractDisplayContent:
    """_extract_display_content 单元测试。"""

    def test_text_event(self) -> None:
        event = Event(
            author="user",
            content=types.Content(
                role="user",
                parts=[types.Part(text="Hello world")],
            ),
        )
        result = _extract_display_content(event)
        assert result["author"] == "user"
        assert result["text"] == "Hello world"

    def test_function_call_event(self) -> None:
        event = Event(
            author="agent",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="get_weather",
                            args={"city": "Shanghai"},
                        )
                    )
                ],
            ),
        )
        result = _extract_display_content(event)
        assert result["author"] == "agent"
        assert "[call:get_weather]" in result["text"]

    def test_function_response_event(self) -> None:
        event = Event(
            author="agent",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="get_weather",
                            response={"result": "sunny"},
                        )
                    )
                ],
            ),
        )
        result = _extract_display_content(event)
        assert "[response:get_weather]" in result["text"]

    def test_empty_content(self) -> None:
        event = Event(author="user")
        result = _extract_display_content(event)
        assert result["author"] == "user"
        assert "text" not in result


# -------------------------------------------------------------------
# Event 序列化 round-trip 测试
# -------------------------------------------------------------------


class TestEventSerialization:
    """ADK Event 的 model_dump_json / model_validate_json round-trip。"""

    def test_text_event_roundtrip(self) -> None:
        original = Event(
            invocation_id="inv-1",
            author="user",
            content=types.Content(
                role="user",
                parts=[types.Part(text="Hello, how are you?")],
            ),
        )
        json_str = original.model_dump_json(by_alias=False)
        restored = Event.model_validate_json(json_str)

        assert restored.author == "user"
        assert restored.invocation_id == "inv-1"
        assert restored.content is not None
        assert restored.content.parts is not None
        assert len(restored.content.parts) == 1
        assert restored.content.parts[0].text == "Hello, how are you?"

    def test_function_call_roundtrip(self) -> None:
        original = Event(
            invocation_id="inv-2",
            author="agent",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="search",
                            args={
                                "query": "weather",
                                "count": 5,
                            },
                        )
                    )
                ],
            ),
        )
        json_str = original.model_dump_json(by_alias=False)
        restored = Event.model_validate_json(json_str)

        assert restored.author == "agent"
        fc = restored.get_function_calls()
        assert len(fc) == 1
        assert fc[0].name == "search"
        assert fc[0].args == {
            "query": "weather",
            "count": 5,
        }

    def test_function_response_roundtrip(self) -> None:
        original = Event(
            invocation_id="inv-3",
            author="agent",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="search",
                            response={
                                "results": [
                                    "sunny",
                                    "warm",
                                ]
                            },
                        )
                    )
                ],
            ),
        )
        json_str = original.model_dump_json(by_alias=False)
        restored = Event.model_validate_json(json_str)

        fr = restored.get_function_responses()
        assert len(fr) == 1
        assert fr[0].name == "search"

    def test_event_with_state_delta_roundtrip(self) -> None:
        original = Event(
            invocation_id="inv-4",
            author="agent",
            actions=EventActions(
                state_delta={
                    "counter": 42,
                    "app:global_count": 100,
                    "user:preference": "dark",
                }
            ),
            content=types.Content(
                role="model",
                parts=[types.Part(text="Updated state")],
            ),
        )
        json_str = original.model_dump_json(by_alias=False)
        restored = Event.model_validate_json(json_str)

        assert restored.actions.state_delta == {
            "counter": 42,
            "app:global_count": 100,
            "user:preference": "dark",
        }

    def test_multipart_event_roundtrip(self) -> None:
        """多 Part 事件的 round-trip。"""
        original = Event(
            invocation_id="inv-5",
            author="model",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Let me search..."),
                    types.Part(
                        function_call=types.FunctionCall(
                            name="web_search",
                            args={"q": "test"},
                        )
                    ),
                ],
            ),
        )
        json_str = original.model_dump_json(by_alias=False)
        restored = Event.model_validate_json(json_str)

        assert restored.content is not None
        assert restored.content.parts is not None
        assert len(restored.content.parts) == 2
        assert restored.content.parts[0].text == "Let me search..."
        assert restored.content.parts[1].function_call.name == "web_search"


# -------------------------------------------------------------------
# OTSSessionService Mock 测试
# -------------------------------------------------------------------


def _make_mock_store() -> MagicMock:
    """创建 Mock SessionStore。

    同时设置同步方法（MagicMock）和异步方法（AsyncMock）的返回值。
    """
    store = MagicMock(spec=SessionStore)

    # 同步方法默认返回值
    store.get_app_state.return_value = {}
    store.get_user_state.return_value = {}
    store.get_session_state.return_value = {}

    # 异步方法使用 AsyncMock
    store.create_session_async = AsyncMock()
    store.get_session_async = AsyncMock(return_value=None)
    store.list_sessions_async = AsyncMock(return_value=[])
    store.list_all_sessions_async = AsyncMock(return_value=[])
    store.delete_session_async = AsyncMock()
    store.delete_events_async = AsyncMock()
    store.update_session_async = AsyncMock()
    store.append_event_async = AsyncMock()
    store.get_events_async = AsyncMock(return_value=[])
    store.get_recent_events_async = AsyncMock(return_value=[])
    store.get_app_state_async = AsyncMock(return_value={})
    store.get_user_state_async = AsyncMock(return_value={})
    store.get_session_state_async = AsyncMock(return_value={})
    store.update_app_state_async = AsyncMock()
    store.update_user_state_async = AsyncMock()
    store.update_session_state_async = AsyncMock()
    store.init_tables_async = AsyncMock()

    return store


class TestOTSSessionServiceCreateSession:
    """create_session 测试。"""

    @pytest.mark.asyncio
    async def test_create_basic(self) -> None:
        store = _make_mock_store()
        service = OTSSessionService(session_store=store)

        session = await service.create_session(
            app_name="test_app",
            user_id="user_1",
        )

        assert session.app_name == "test_app"
        assert session.user_id == "user_1"
        assert session.id  # 自动生成 UUID
        assert session.events == []
        store.create_session_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_with_session_id(self) -> None:
        store = _make_mock_store()
        service = OTSSessionService(session_store=store)

        session = await service.create_session(
            app_name="test_app",
            user_id="user_1",
            session_id="my-session-id",
        )

        assert session.id == "my-session-id"

    @pytest.mark.asyncio
    async def test_create_with_state(self) -> None:
        store = _make_mock_store()
        service = OTSSessionService(session_store=store)

        await service.create_session(
            app_name="test_app",
            user_id="user_1",
            state={
                "app:config": "val",
                "user:pref": "dark",
                "local": 123,
            },
        )

        store.update_app_state_async.assert_called_once_with(
            "test_app", {"config": "val"}
        )
        store.update_user_state_async.assert_called_once_with(
            "test_app", "user_1", {"pref": "dark"}
        )
        store.update_session_state_async.assert_called_once()

    def test_create_sync(self) -> None:
        store = _make_mock_store()
        service = OTSSessionService(session_store=store)

        session = service.create_session_sync(
            app_name="test_app",
            user_id="user_1",
        )

        assert session.app_name == "test_app"
        store.create_session.assert_called_once()


class TestOTSSessionServiceGetSession:
    """get_session 测试。"""

    @pytest.mark.asyncio
    async def test_get_nonexistent(self) -> None:
        store = _make_mock_store()
        store.get_session_async.return_value = None
        service = OTSSessionService(session_store=store)

        result = await service.get_session(
            app_name="test_app",
            user_id="user_1",
            session_id="nonexistent",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_with_events(self) -> None:
        store = _make_mock_store()

        ots_session = ConversationSession(
            agent_id="test_app",
            user_id="user_1",
            session_id="s1",
            created_at=1000000000,
            updated_at=2000000000,
        )
        store.get_session_async.return_value = ots_session

        # 构造一个 ADK Event 并序列化
        adk_event = Event(
            invocation_id="inv-1",
            author="user",
            content=types.Content(
                role="user",
                parts=[types.Part(text="Hello")],
            ),
        )
        raw_json = adk_event.model_dump_json(by_alias=False)

        ots_event = ConversationEvent(
            agent_id="test_app",
            user_id="user_1",
            session_id="s1",
            seq_id=1,
            type="adk_event",
            content={"author": "user", "text": "Hello"},
            raw_event=raw_json,
        )
        store.get_events_async.return_value = [ots_event]

        service = OTSSessionService(session_store=store)
        result = await service.get_session(
            app_name="test_app",
            user_id="user_1",
            session_id="s1",
        )

        assert result is not None
        assert len(result.events) == 1
        assert result.events[0].author == "user"
        parts = result.events[0].content.parts
        assert parts is not None
        assert parts[0].text == "Hello"

    @pytest.mark.asyncio
    async def test_get_skips_events_without_raw_event(
        self,
    ) -> None:
        """LangChain 事件（无 raw_event）应被跳过。"""
        store = _make_mock_store()

        ots_session = ConversationSession(
            agent_id="test_app",
            user_id="user_1",
            session_id="s1",
            created_at=1000000000,
            updated_at=2000000000,
        )
        store.get_session_async.return_value = ots_session

        lc_event = ConversationEvent(
            agent_id="test_app",
            user_id="user_1",
            session_id="s1",
            seq_id=1,
            type="message",
            content={"lc_type": "human", "content": "Hi"},
            raw_event=None,
        )
        store.get_events_async.return_value = [lc_event]

        service = OTSSessionService(session_store=store)
        result = await service.get_session(
            app_name="test_app",
            user_id="user_1",
            session_id="s1",
        )

        assert result is not None
        assert len(result.events) == 0

    @pytest.mark.asyncio
    async def test_get_with_num_recent_events(self) -> None:
        store = _make_mock_store()

        ots_session = ConversationSession(
            agent_id="test_app",
            user_id="user_1",
            session_id="s1",
            created_at=1000000000,
            updated_at=2000000000,
        )
        store.get_session_async.return_value = ots_session
        store.get_recent_events_async.return_value = []

        service = OTSSessionService(session_store=store)
        config = GetSessionConfig(num_recent_events=5)
        await service.get_session(
            app_name="test_app",
            user_id="user_1",
            session_id="s1",
            config=config,
        )

        store.get_recent_events_async.assert_called_once_with(
            "test_app", "user_1", "s1", 5
        )

    @pytest.mark.asyncio
    async def test_get_with_merged_state(self) -> None:
        store = _make_mock_store()

        ots_session = ConversationSession(
            agent_id="test_app",
            user_id="user_1",
            session_id="s1",
            created_at=1000000000,
            updated_at=2000000000,
        )
        store.get_session_async.return_value = ots_session
        store.get_events_async.return_value = []

        store.get_app_state_async.return_value = {"setting": "A"}
        store.get_user_state_async.return_value = {"pref": "dark"}
        store.get_session_state_async.return_value = {"counter": 42}

        service = OTSSessionService(session_store=store)
        result = await service.get_session(
            app_name="test_app",
            user_id="user_1",
            session_id="s1",
        )

        assert result is not None
        # session state: 无前缀
        assert result.state["counter"] == 42
        # user state: user: 前缀
        assert result.state["user:pref"] == "dark"
        # app state: app: 前缀
        assert result.state["app:setting"] == "A"


class TestOTSSessionServiceListSessions:
    """list_sessions 测试。"""

    @pytest.mark.asyncio
    async def test_list_with_user_id(self) -> None:
        store = _make_mock_store()
        store.list_sessions_async.return_value = [
            ConversationSession(
                agent_id="app",
                user_id="u1",
                session_id="s1",
                created_at=0,
                updated_at=1000000000,
            )
        ]
        service = OTSSessionService(session_store=store)

        response = await service.list_sessions(app_name="app", user_id="u1")

        assert len(response.sessions) == 1
        assert response.sessions[0].id == "s1"
        store.list_sessions_async.assert_called_once_with("app", "u1")

    @pytest.mark.asyncio
    async def test_list_all_users(self) -> None:
        store = _make_mock_store()
        store.list_all_sessions_async.return_value = [
            ConversationSession(
                agent_id="app",
                user_id="u1",
                session_id="s1",
                created_at=0,
                updated_at=1000000000,
            ),
            ConversationSession(
                agent_id="app",
                user_id="u2",
                session_id="s2",
                created_at=0,
                updated_at=2000000000,
            ),
        ]
        service = OTSSessionService(session_store=store)

        response = await service.list_sessions(app_name="app", user_id=None)

        assert len(response.sessions) == 2
        store.list_all_sessions_async.assert_called_once_with("app")


class TestOTSSessionServiceDeleteSession:
    """delete_session 测试。"""

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        store = _make_mock_store()
        service = OTSSessionService(session_store=store)

        await service.delete_session(
            app_name="app",
            user_id="u1",
            session_id="s1",
        )

        store.delete_session_async.assert_called_once_with("app", "u1", "s1")

    def test_delete_sync(self) -> None:
        store = _make_mock_store()
        service = OTSSessionService(session_store=store)

        service.delete_session_sync(
            app_name="app",
            user_id="u1",
            session_id="s1",
        )

        store.delete_session.assert_called_once_with("app", "u1", "s1")


class TestOTSSessionServiceAppendEvent:
    """append_event 测试。"""

    @pytest.mark.asyncio
    async def test_append_text_event(self) -> None:
        store = _make_mock_store()
        service = OTSSessionService(session_store=store)

        session = Session(
            id="s1",
            app_name="app",
            user_id="u1",
            state={},
            events=[],
        )

        event = Event(
            invocation_id="inv-1",
            author="user",
            content=types.Content(
                role="user",
                parts=[types.Part(text="Hello")],
            ),
        )

        result = await service.append_event(session, event)

        # 事件被追加到 session.events
        assert len(session.events) == 1
        assert session.events[0] is result

        # store.append_event_async 被调用，且传递了 raw_event
        store.append_event_async.assert_called_once()
        call_kwargs = store.append_event_async.call_args
        # raw_event 参数不为 None
        assert call_kwargs.kwargs.get("raw_event") is not None or (
            len(call_kwargs.args) > 5 and call_kwargs.args[5] is not None
        )

    @pytest.mark.asyncio
    async def test_append_skips_partial(self) -> None:
        store = _make_mock_store()
        service = OTSSessionService(session_store=store)

        session = Session(
            id="s1",
            app_name="app",
            user_id="u1",
            state={},
            events=[],
        )

        event = Event(
            invocation_id="inv-1",
            author="user",
            partial=True,
            content=types.Content(
                role="user",
                parts=[types.Part(text="partial")],
            ),
        )

        result = await service.append_event(session, event)

        assert result.partial is True
        store.append_event_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_append_persists_state_delta(self) -> None:
        store = _make_mock_store()
        service = OTSSessionService(session_store=store)

        session = Session(
            id="s1",
            app_name="app",
            user_id="u1",
            state={},
            events=[],
        )

        event = Event(
            invocation_id="inv-1",
            author="agent",
            actions=EventActions(
                state_delta={
                    "counter": 1,
                    "app:global": "yes",
                    "user:theme": "dark",
                }
            ),
            content=types.Content(
                role="model",
                parts=[types.Part(text="Done")],
            ),
        )

        await service.append_event(session, event)

        # 三级 state 分别被更新
        store.update_app_state_async.assert_called_once_with(
            "app", {"global": "yes"}
        )
        store.update_user_state_async.assert_called_once_with(
            "app", "u1", {"theme": "dark"}
        )
        store.update_session_state_async.assert_called_once_with(
            "app", "u1", "s1", {"counter": 1}
        )

    @pytest.mark.asyncio
    async def test_append_updates_session_state_in_memory(
        self,
    ) -> None:
        store = _make_mock_store()
        service = OTSSessionService(session_store=store)

        session = Session(
            id="s1",
            app_name="app",
            user_id="u1",
            state={"existing": "value"},
            events=[],
        )

        event = Event(
            invocation_id="inv-1",
            author="agent",
            actions=EventActions(state_delta={"new_key": "new_value"}),
            content=types.Content(
                role="model",
                parts=[types.Part(text="ok")],
            ),
        )

        await service.append_event(session, event)

        # 内存中的 session.state 应该已更新
        assert session.state["new_key"] == "new_value"
        assert session.state["existing"] == "value"

    @pytest.mark.asyncio
    async def test_append_raw_event_roundtrip(self) -> None:
        """验证 append_event 写入的 raw_event 可以被 get_session 还原。"""
        store = _make_mock_store()
        service = OTSSessionService(session_store=store)

        session = Session(
            id="s1",
            app_name="app",
            user_id="u1",
            state={},
            events=[],
        )

        original_event = Event(
            invocation_id="inv-roundtrip",
            author="agent",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Answer"),
                    types.Part(
                        function_call=types.FunctionCall(
                            name="tool1",
                            args={"x": 1},
                        )
                    ),
                ],
            ),
        )

        await service.append_event(session, original_event)

        # 获取 store.append_event_async 被调用时的 raw_event 参数
        call_args = store.append_event_async.call_args
        raw_event_str: str = call_args.kwargs["raw_event"]

        # 还原
        restored = Event.model_validate_json(raw_event_str)
        assert restored.invocation_id == "inv-roundtrip"
        assert restored.author == "agent"
        parts = restored.content.parts
        assert parts is not None
        assert parts[0].text == "Answer"
        assert parts[1].function_call.name == "tool1"
