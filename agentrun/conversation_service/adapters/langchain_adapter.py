"""LangChain BaseChatMessageHistory 适配器。

将 LangChain 的消息历史持久化到 OTS，通过 SessionStore 实现。

使用方式::

    from agentrun.conversation_service import SessionStore, OTSBackend
    from agentrun.conversation_service.adapters import OTSChatMessageHistory

    store = SessionStore(OTSBackend(ots_client))

    history = OTSChatMessageHistory(
        session_store=store,
        agent_id="my_agent",
        user_id="user_1",
        session_id="session_1",
    )

    # 配合 RunnableWithMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: OTSChatMessageHistory(
            session_store=store,
            agent_id="my_agent",
            user_id="user_1",
            session_id=session_id,
        ),
    )
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from langchain_core.chat_history import (
    BaseChatMessageHistory,
)  # type: ignore[import-untyped]
from langchain_core.messages import AIMessage  # type: ignore[import-untyped]
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agentrun.conversation_service.model import ConversationEvent
from agentrun.conversation_service.session_store import SessionStore

logger = logging.getLogger(__name__)

# LangChain message type → Message class 映射
_TYPE_TO_CLASS: dict[str, type[BaseMessage]] = {
    "human": HumanMessage,
    "ai": AIMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
}

# 统一的事件类型标识
_EVENT_TYPE = "message"


class OTSChatMessageHistory(BaseChatMessageHistory):
    """基于 OTS 的 LangChain 消息历史实现。

    将 LangChain 的 BaseMessage 序列化为 ConversationEvent
    存储到 TableStore，通过 SessionStore 进行读写。

    Attributes:
        session_store: SessionStore 实例。
        agent_id: 智能体 ID。
        user_id: 用户 ID。
        session_id: 会话 ID。
    """

    def __init__(
        self,
        session_store: SessionStore,
        agent_id: str,
        user_id: str,
        session_id: str,
        *,
        auto_create_session: bool = True,
    ) -> None:
        """初始化。

        Args:
            session_store: SessionStore 实例。
            agent_id: 智能体 ID。
            user_id: 用户 ID。
            session_id: 会话 ID。
            auto_create_session: 若 Session 不存在是否自动创建，
                默认 True。
        """
        self.session_store = session_store
        self.agent_id = agent_id
        self.user_id = user_id
        self.session_id = session_id

        if auto_create_session:
            existing = session_store.get_session(agent_id, user_id, session_id)
            if existing is None:
                session_store.create_session(
                    agent_id,
                    user_id,
                    session_id,
                    framework="langchain",
                )

    # ---------------------------------------------------------------
    # BaseChatMessageHistory 接口实现
    # ---------------------------------------------------------------

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """从 OTS 读取全部消息，按 seq_id 正序返回。"""
        events = self.session_store.get_events(
            self.agent_id, self.user_id, self.session_id
        )
        result: list[BaseMessage] = []
        for event in events:
            try:
                msg = _event_to_message(event)
                result.append(msg)
            except Exception:
                logger.warning(
                    "Failed to deserialize event seq_id=%s, skipping.",
                    event.seq_id,
                    exc_info=True,
                )
        return result

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """批量写入消息到 OTS。

        每条 LangChain BaseMessage 转为一个 ConversationEvent。
        """
        for message in messages:
            content = _message_to_dict(message)
            self.session_store.append_event(
                self.agent_id,
                self.user_id,
                self.session_id,
                event_type=_EVENT_TYPE,
                content=content,
            )

    def clear(self) -> None:
        """清空当前 Session 的所有消息。

        只删除 Event，不删除 Session 本身和 State。
        """
        self.session_store.delete_events(
            self.agent_id, self.user_id, self.session_id
        )


# -------------------------------------------------------------------
# 序列化 / 反序列化工具函数
# -------------------------------------------------------------------


def _message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """将 LangChain BaseMessage 序列化为可 JSON 存储的 dict。

    存储字段：
      - lc_type: 消息类型（human / ai / system / tool）
      - content: 消息内容
      - additional_kwargs: 额外参数
      - response_metadata: 响应元数据
      - name: 消息名称（可选）
      - id: 消息 ID（可选）
      - tool_calls: 工具调用列表（AIMessage 特有）
      - tool_call_id: 工具调用 ID（ToolMessage 特有）
    """
    data: dict[str, Any] = {
        "lc_type": message.type,
        "content": message.content,
    }

    # 只存非空字段，减少存储开销
    if message.additional_kwargs:
        data["additional_kwargs"] = message.additional_kwargs
    if getattr(message, "response_metadata", None):
        data["response_metadata"] = message.response_metadata
    if message.name is not None:
        data["name"] = message.name
    if message.id is not None:
        data["id"] = message.id

    # AIMessage 特有字段
    if isinstance(message, AIMessage):
        if message.tool_calls:
            data["tool_calls"] = message.tool_calls
        if message.invalid_tool_calls:
            data["invalid_tool_calls"] = message.invalid_tool_calls

    # ToolMessage 特有字段
    if isinstance(message, ToolMessage):
        data["tool_call_id"] = message.tool_call_id

    return data


def _event_to_message(
    event: ConversationEvent,
) -> BaseMessage:
    """将 ConversationEvent 反序列化为 LangChain BaseMessage。"""
    data = dict(event.content)
    lc_type = data.pop("lc_type", "human")

    cls = _TYPE_TO_CLASS.get(lc_type)
    if cls is None:
        logger.warning(
            "Unknown message type '%s', falling back to HumanMessage.",
            lc_type,
        )
        cls = HumanMessage

    # 构造参数：只传非空字段
    kwargs: dict[str, Any] = {
        "content": data.get("content", ""),
    }

    if "additional_kwargs" in data:
        kwargs["additional_kwargs"] = data["additional_kwargs"]
    if "response_metadata" in data:
        kwargs["response_metadata"] = data["response_metadata"]
    if "name" in data:
        kwargs["name"] = data["name"]
    if "id" in data:
        kwargs["id"] = data["id"]

    # AIMessage 特有
    if cls is AIMessage:
        if "tool_calls" in data:
            kwargs["tool_calls"] = data["tool_calls"]
        if "invalid_tool_calls" in data:
            kwargs["invalid_tool_calls"] = data["invalid_tool_calls"]

    # ToolMessage 特有
    if cls is ToolMessage:
        kwargs["tool_call_id"] = data.get("tool_call_id", "")

    return cls(**kwargs)
