"""Google ADK BaseSessionService 适配器。

将 Google ADK 的会话管理持久化到 OTS，通过 SessionStore 实现。

使用方式::

    from agentrun.conversation_service import SessionStore, OTSBackend
    from agentrun.conversation_service.adapters import OTSSessionService

    store = SessionStore(OTSBackend(ots_client, async_ots_client=async_ots_client))

    # 作为 ADK Runner 的 session_service
    from google.adk.runners import Runner

    runner = Runner(
        agent=my_agent,
        app_name="my_app",
        session_service=OTSSessionService(session_store=store),
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional
import uuid

from google.adk.events.event import Event  # type: ignore[import-untyped]
from google.adk.sessions.base_session_service import (  # type: ignore[import-untyped]
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.session import Session  # type: ignore[import-untyped]
from google.adk.sessions.state import State  # type: ignore[import-untyped]
from typing_extensions import override

from agentrun.conversation_service.session_store import SessionStore

logger = logging.getLogger(__name__)

# ADK 使用 key 前缀区分 state 作用域
_APP_PREFIX = State.APP_PREFIX  # "app:"
_USER_PREFIX = State.USER_PREFIX  # "user:"
_TEMP_PREFIX = State.TEMP_PREFIX  # "temp:"

# 事件类型标识
_EVENT_TYPE = "adk_event"


# -------------------------------------------------------------------
# 工具函数
# -------------------------------------------------------------------


def _extract_state_delta(
    state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """从 state 字典中按前缀拆分出 app / user / session 三级 delta。

    自行实现，避免依赖 google.adk.sessions._session_util（私有模块）。

    Args:
        state: 包含前缀标识的 state 字典。

    Returns:
        包含 'app'、'user'、'session' 三个 key 的字典。
    """
    deltas: dict[str, dict[str, Any]] = {
        "app": {},
        "user": {},
        "session": {},
    }
    if state:
        for key in state.keys():
            if key.startswith(_APP_PREFIX):
                deltas["app"][key.removeprefix(_APP_PREFIX)] = state[key]
            elif key.startswith(_USER_PREFIX):
                deltas["user"][key.removeprefix(_USER_PREFIX)] = state[key]
            elif not key.startswith(_TEMP_PREFIX):
                deltas["session"][key] = state[key]
    return deltas


def _extract_display_content(
    event: Event,
) -> dict[str, Any]:
    """从 ADK Event 提取用于展示的简化内容。

    存入 OTS Event 表的 content 列，供跨框架展示使用。
    """
    result: dict[str, Any] = {"author": event.author}
    if event.content and event.content.parts:
        texts: list[str] = []
        for part in event.content.parts:
            if part.text:
                texts.append(part.text)
            elif part.function_call:
                texts.append(f"[call:{part.function_call.name}]")
            elif part.function_response:
                texts.append(f"[response:{part.function_response.name}]")
        result["text"] = "\n".join(texts)
    return result


# -------------------------------------------------------------------
# OTSSessionService
# -------------------------------------------------------------------


class OTSSessionService(BaseSessionService):
    """基于 OTS 的 Google ADK SessionService 实现。

    async 公共方法使用原生 ``await self._store.xxx_async(...)`` 调用，
    sync ``_impl`` 方法使用 ``self._store.xxx(...)`` 调用。

    Args:
        session_store: SessionStore 实例。
    """

    def __init__(self, session_store: SessionStore) -> None:
        self._store = session_store

    # ---------------------------------------------------------------
    # create_session
    # ---------------------------------------------------------------

    @override
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        session_id = (
            session_id.strip()
            if session_id and session_id.strip()
            else str(uuid.uuid4())
        )

        # 1. 拆分初始 state 为三级
        state_deltas = _extract_state_delta(state or {})

        # 2. 创建 OTS session
        await self._store.create_session_async(
            app_name,
            user_id,
            session_id,
            framework="adk",
        )

        # 3. 持久化三级 state
        if state_deltas["app"]:
            await self._store.update_app_state_async(
                app_name, state_deltas["app"]
            )
        if state_deltas["user"]:
            await self._store.update_user_state_async(
                app_name, user_id, state_deltas["user"]
            )
        if state_deltas["session"]:
            await self._store.update_session_state_async(
                app_name,
                user_id,
                session_id,
                state_deltas["session"],
            )

        # 4. 构造 ADK Session 返回（含合并 state）
        return await self._build_adk_session_async(
            app_name, user_id, session_id, events=[]
        )

    def create_session_sync(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """同步版 create_session。"""
        return self._create_session_impl(
            app_name=app_name,
            user_id=user_id,
            state=state,
            session_id=session_id,
        )

    def _create_session_impl(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        session_id = (
            session_id.strip()
            if session_id and session_id.strip()
            else str(uuid.uuid4())
        )

        # 1. 拆分初始 state 为三级
        state_deltas = _extract_state_delta(state or {})

        # 2. 创建 OTS session
        self._store.create_session(
            app_name,
            user_id,
            session_id,
            framework="adk",
        )

        # 3. 持久化三级 state
        if state_deltas["app"]:
            self._store.update_app_state(app_name, state_deltas["app"])
        if state_deltas["user"]:
            self._store.update_user_state(
                app_name, user_id, state_deltas["user"]
            )
        if state_deltas["session"]:
            self._store.update_session_state(
                app_name,
                user_id,
                session_id,
                state_deltas["session"],
            )

        # 4. 构造 ADK Session 返回（含合并 state）
        return self._build_adk_session(app_name, user_id, session_id, events=[])

    # ---------------------------------------------------------------
    # get_session
    # ---------------------------------------------------------------

    @override
    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        # 1. 读 session 元数据
        ots_session = await self._store.get_session_async(
            app_name, user_id, session_id
        )
        if ots_session is None:
            return None

        # 2. 读 events（考虑 config.num_recent_events）
        if config and config.num_recent_events:
            ots_events = await self._store.get_recent_events_async(
                app_name,
                user_id,
                session_id,
                config.num_recent_events,
            )
        else:
            ots_events = await self._store.get_events_async(
                app_name, user_id, session_id
            )

        # 3. 从 raw_event 列反序列化为 ADK Event
        adk_events: list[Event] = []
        for e in ots_events:
            if e.raw_event is not None:
                try:
                    adk_events.append(Event.model_validate_json(e.raw_event))
                except Exception:
                    logger.warning(
                        "Failed to deserialize ADK Event seq_id=%s, skipping.",
                        e.seq_id,
                        exc_info=True,
                    )

        # 4. 如有 after_timestamp，过滤
        if config and config.after_timestamp:
            adk_events = [
                e for e in adk_events if e.timestamp >= config.after_timestamp
            ]

        # 5. 构造带 merged state 的 ADK Session
        return await self._build_adk_session_async(
            app_name,
            user_id,
            session_id,
            events=adk_events,
            updated_at=ots_session.updated_at,
        )

    def get_session_sync(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        """同步版 get_session。"""
        return self._get_session_impl(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            config=config,
        )

    def _get_session_impl(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        # 1. 读 session 元数据
        ots_session = self._store.get_session(app_name, user_id, session_id)
        if ots_session is None:
            return None

        # 2. 读 events（考虑 config.num_recent_events）
        if config and config.num_recent_events:
            ots_events = self._store.get_recent_events(
                app_name,
                user_id,
                session_id,
                config.num_recent_events,
            )
        else:
            ots_events = self._store.get_events(app_name, user_id, session_id)

        # 3. 从 raw_event 列反序列化为 ADK Event
        adk_events: list[Event] = []
        for e in ots_events:
            if e.raw_event is not None:
                try:
                    adk_events.append(Event.model_validate_json(e.raw_event))
                except Exception:
                    logger.warning(
                        "Failed to deserialize ADK Event seq_id=%s, skipping.",
                        e.seq_id,
                        exc_info=True,
                    )

        # 4. 如有 after_timestamp，过滤
        if config and config.after_timestamp:
            adk_events = [
                e for e in adk_events if e.timestamp >= config.after_timestamp
            ]

        # 5. 构造带 merged state 的 ADK Session
        return self._build_adk_session(
            app_name,
            user_id,
            session_id,
            events=adk_events,
            updated_at=ots_session.updated_at,
        )

    # ---------------------------------------------------------------
    # list_sessions
    # ---------------------------------------------------------------

    @override
    async def list_sessions(
        self,
        *,
        app_name: str,
        user_id: Optional[str] = None,
    ) -> ListSessionsResponse:
        if user_id is not None:
            ots_sessions = await self._store.list_sessions_async(
                app_name, user_id
            )
        else:
            ots_sessions = await self._store.list_all_sessions_async(app_name)

        sessions: list[Session] = []
        for s in ots_sessions:
            sessions.append(
                Session(
                    id=s.session_id,
                    app_name=app_name,
                    user_id=s.user_id,
                    state={},
                    events=[],
                    last_update_time=s.updated_at / 1_000_000_000.0,
                )
            )

        return ListSessionsResponse(sessions=sessions)

    def list_sessions_sync(
        self,
        *,
        app_name: str,
        user_id: Optional[str] = None,
    ) -> ListSessionsResponse:
        """同步版 list_sessions。"""
        return self._list_sessions_impl(app_name=app_name, user_id=user_id)

    def _list_sessions_impl(
        self,
        *,
        app_name: str,
        user_id: Optional[str] = None,
    ) -> ListSessionsResponse:
        if user_id is not None:
            ots_sessions = self._store.list_sessions(app_name, user_id)
        else:
            ots_sessions = self._store.list_all_sessions(app_name)

        sessions: list[Session] = []
        for s in ots_sessions:
            sessions.append(
                Session(
                    id=s.session_id,
                    app_name=app_name,
                    user_id=s.user_id,
                    state={},
                    events=[],
                    last_update_time=s.updated_at / 1_000_000_000.0,
                )
            )

        return ListSessionsResponse(sessions=sessions)

    # ---------------------------------------------------------------
    # delete_session
    # ---------------------------------------------------------------

    @override
    async def delete_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> None:
        await self._store.delete_session_async(app_name, user_id, session_id)

    def delete_session_sync(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> None:
        """同步版 delete_session。"""
        self._delete_session_impl(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

    def _delete_session_impl(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> None:
        self._store.delete_session(app_name, user_id, session_id)

    # ---------------------------------------------------------------
    # append_event
    # ---------------------------------------------------------------

    @override
    async def append_event(self, session: Session, event: Event) -> Event:
        if event.partial:
            return event

        # 1. 调用父类 sync 辅助方法更新内存 session
        #    （trim temp state delta + update session state）
        event = self._trim_temp_delta_state(event)
        self._update_session_state(session, event)
        session.events.append(event)
        session.last_update_time = event.timestamp

        # 2. 序列化 Event，写入 content（简化文本）和
        #    raw_event（完整 JSON）
        raw_event_str = event.model_dump_json(by_alias=False)
        content_dict = _extract_display_content(event)

        await self._store.append_event_async(
            session.app_name,
            session.user_id,
            session.id,
            event_type=_EVENT_TYPE,
            content=content_dict,
            raw_event=raw_event_str,
        )

        # 3. 持久化 state delta 到三级 state 表
        if event.actions and event.actions.state_delta:
            state_deltas = _extract_state_delta(event.actions.state_delta)
            if state_deltas["app"]:
                await self._store.update_app_state_async(
                    session.app_name, state_deltas["app"]
                )
            if state_deltas["user"]:
                await self._store.update_user_state_async(
                    session.app_name,
                    session.user_id,
                    state_deltas["user"],
                )
            if state_deltas["session"]:
                await self._store.update_session_state_async(
                    session.app_name,
                    session.user_id,
                    session.id,
                    state_deltas["session"],
                )

        return event

    def _append_event_impl(self, session: Session, event: Event) -> Event:
        """同步版 append_event 的内部实现。"""
        if event.partial:
            return event

        # 1. 调用父类 sync 辅助方法更新内存 session
        event = self._trim_temp_delta_state(event)
        self._update_session_state(session, event)
        session.events.append(event)
        session.last_update_time = event.timestamp

        # 2. 序列化 Event，写入 content（简化文本）和 raw_event（完整 JSON）
        raw_event_str = event.model_dump_json(by_alias=False)
        content_dict = _extract_display_content(event)

        self._store.append_event(
            session.app_name,
            session.user_id,
            session.id,
            event_type=_EVENT_TYPE,
            content=content_dict,
            raw_event=raw_event_str,
        )

        # 3. 持久化 state delta 到三级 state 表
        if event.actions and event.actions.state_delta:
            state_deltas = _extract_state_delta(event.actions.state_delta)
            if state_deltas["app"]:
                self._store.update_app_state(
                    session.app_name, state_deltas["app"]
                )
            if state_deltas["user"]:
                self._store.update_user_state(
                    session.app_name,
                    session.user_id,
                    state_deltas["user"],
                )
            if state_deltas["session"]:
                self._store.update_session_state(
                    session.app_name,
                    session.user_id,
                    session.id,
                    state_deltas["session"],
                )

        return event

    # ---------------------------------------------------------------
    # 内部辅助方法
    # ---------------------------------------------------------------

    async def _build_adk_session_async(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        events: list[Event],
        updated_at: Optional[int] = None,
    ) -> Session:
        """构造 ADK Session 对象，合并三级 state（异步）。

        Args:
            app_name: 应用名（对应 OTS agent_id）。
            user_id: 用户 ID。
            session_id: 会话 ID。
            events: ADK Event 列表。
            updated_at: OTS 中的 updated_at（纳秒），
                用于设置 last_update_time。
        """
        merged: dict[str, Any] = {}

        # session state（无前缀）
        session_state = await self._store.get_session_state_async(
            app_name, user_id, session_id
        )
        merged.update(session_state)

        # user state（加 user: 前缀）
        user_state = await self._store.get_user_state_async(app_name, user_id)
        for k, v in user_state.items():
            merged[_USER_PREFIX + k] = v

        # app state（加 app: 前缀）
        app_state = await self._store.get_app_state_async(app_name)
        for k, v in app_state.items():
            merged[_APP_PREFIX + k] = v

        last_update = (
            updated_at / 1_000_000_000.0
            if updated_at is not None
            else time.time()
        )

        return Session(
            id=session_id,
            app_name=app_name,
            user_id=user_id,
            state=merged,
            events=events,
            last_update_time=last_update,
        )

    def _build_adk_session(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        events: list[Event],
        updated_at: Optional[int] = None,
    ) -> Session:
        """构造 ADK Session 对象，合并三级 state（同步）。

        Args:
            app_name: 应用名（对应 OTS agent_id）。
            user_id: 用户 ID。
            session_id: 会话 ID。
            events: ADK Event 列表。
            updated_at: OTS 中的 updated_at（纳秒），
                用于设置 last_update_time。
        """
        merged: dict[str, Any] = {}

        # session state（无前缀）
        session_state = self._store.get_session_state(
            app_name, user_id, session_id
        )
        merged.update(session_state)

        # user state（加 user: 前缀）
        user_state = self._store.get_user_state(app_name, user_id)
        for k, v in user_state.items():
            merged[_USER_PREFIX + k] = v

        # app state（加 app: 前缀）
        app_state = self._store.get_app_state(app_name)
        for k, v in app_state.items():
            merged[_APP_PREFIX + k] = v

        last_update = (
            updated_at / 1_000_000_000.0
            if updated_at is not None
            else time.time()
        )

        return Session(
            id=session_id,
            app_name=app_name,
            user_id=user_id,
            state=merged,
            events=events,
            last_update_time=last_update,
        )
