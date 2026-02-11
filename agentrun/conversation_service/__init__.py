"""Conversation Service 模块。

为不同 Agent 开发框架提供会话状态持久化能力，
持久化数据库选用阿里云 TableStore（OTS，宽表模型）。

使用方式::

    # 方式一（推荐）：通过 MemoryCollection 自动获取 OTS 连接信息
    from agentrun.conversation_service import SessionStore

    store = SessionStore.from_memory_collection("your-memory-collection-name")
    store.init_tables()

    # 方式二：手动传入 OTSClient
    import tablestore
    from agentrun.conversation_service import SessionStore, OTSBackend

    ots_client = tablestore.OTSClient(
        endpoint, access_key_id, access_key_secret, instance_name,
    )
    backend = OTSBackend(ots_client)
    store = SessionStore(backend)
    store.init_tables()
"""

from agentrun.conversation_service.model import (
    ConversationEvent,
    ConversationSession,
    StateData,
    StateScope,
)
from agentrun.conversation_service.ots_backend import OTSBackend
from agentrun.conversation_service.session_store import SessionStore

__all__ = [
    # 核心服务
    "SessionStore",
    "OTSBackend",
    # 领域模型
    "ConversationSession",
    "ConversationEvent",
    "StateData",
    "StateScope",
]
