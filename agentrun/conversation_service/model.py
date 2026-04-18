"""Conversation Service 领域模型。

定义会话、事件、状态等核心数据结构，以及表名常量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Optional

# ---------------------------------------------------------------------------
# 表名常量（支持通过 table_prefix 自定义）
# ---------------------------------------------------------------------------

DEFAULT_CONVERSATION_TABLE = "conversation"
DEFAULT_EVENT_TABLE = "event"
DEFAULT_STATE_TABLE = "state"
DEFAULT_APP_STATE_TABLE = "app_state"
DEFAULT_USER_STATE_TABLE = "user_state"
DEFAULT_CONVERSATION_SECONDARY_INDEX = "conversation_secondary_index"
DEFAULT_CONVERSATION_SEARCH_INDEX = "conversation_search_index"
DEFAULT_STATE_SEARCH_INDEX = "state_search_index"

# LangGraph checkpoint 表
DEFAULT_CHECKPOINT_TABLE = "checkpoint"
DEFAULT_CHECKPOINT_WRITES_TABLE = "checkpoint_writes"
DEFAULT_CHECKPOINT_BLOBS_TABLE = "checkpoint_blobs"

# ---------------------------------------------------------------------------
# OTS Schema 版本管理
#
# 用于 SDK 写入端与 Core 读取端(funagent-core)的兼容性协调。
# 每次写入行（PutRow / UpdateRow / BatchWriteRow）时在
# attribute_columns 中携带 _schema_version 字段。
# Core 端读取时检查该字段，版本不匹配时打 WARN 日志并尽力解析。
# 历史数据（无此字段）视为 v0。
#
# 版本计数规则：
#   - 大部分表独立计数
#   - state / app_state / user_state 三张表共享 STATE_SCHEMA_VERSION
#
# 升级流程：
#   1. 递增对应表的 *_SCHEMA_VERSION 常量
#   2. 在 PR 描述中记录变更的列名/类型/语义
#   3. 通知 funagent-core 侧同步更新解析逻辑和版本常量
#   4. 如涉及 breaking change，提供数据迁移指引
#
# 兼容性规则：
#   - 只加不删：新增列允许，删除/重命名列视为 breaking change
#   - PK 不可变：主键结构永不改变
#   - 索引名不可变：Search Index 名称一旦确定不再修改
#   - 语义不可变：已有列的类型和含义不改变
# ---------------------------------------------------------------------------

SCHEMA_VERSION_COLUMN = "_schema_version"

CONVERSATION_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1  # state / app_state / user_state 共享
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_WRITES_SCHEMA_VERSION = 1
CHECKPOINT_BLOBS_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


class StateScope(str, Enum):
    """状态作用域。

    三级 State 是 ADK 的概念，其他框架按需使用。
    - APP: 应用级状态（agent_id 维度）
    - USER: 用户级状态（agent_id + user_id 维度）
    - SESSION: 会话级状态（agent_id + user_id + session_id 维度）
    """

    APP = "app"
    USER = "user"
    SESSION = "session"


# ---------------------------------------------------------------------------
# 领域对象
# ---------------------------------------------------------------------------


@dataclass
class ConversationSession:
    """会话对象。

    Attributes:
        agent_id: 智能体 ID（分区键）。
        user_id: 用户 ID。
        session_id: 会话 ID。
        created_at: 创建时间（纳秒时间戳）。
        updated_at: 最后更新时间（纳秒时间戳）。
        is_pinned: 是否置顶。
        summary: 会话摘要。
        labels: 会话标签（JSON 字符串）。
        framework: 框架标识，如 'adk' / 'langchain' / 'langgraph'。
        extensions: 框架扩展数据（JSON 序列化后存储）。
        version: 乐观锁版本号。
    """

    agent_id: str
    user_id: str
    session_id: str
    created_at: int
    updated_at: int
    is_pinned: bool = False
    summary: Optional[str] = None
    labels: Optional[str] = None
    framework: Optional[str] = None
    extensions: Optional[dict[str, Any]] = None
    version: int = 0


@dataclass
class ConversationEvent:
    """事件对象。

    统一用 Event 抽象，Message 是 Event 的子集。

    Attributes:
        agent_id: 智能体 ID（分区键）。
        user_id: 用户 ID。
        session_id: 会话 ID。
        seq_id: 事件序号（OTS AUTO_INCREMENT 生成，写入前为 None）。
        type: 事件类型。
        content: 事件数据（JSON 序列化后存储）。
        created_at: 创建时间（纳秒时间戳）。
        updated_at: 最后更新时间（纳秒时间戳）。
        version: 乐观锁版本号。
        raw_event: 框架原生 Event 的完整 JSON 序列化（可选）。
            用于精确还原框架特定的 Event 对象（如 ADK Event）。
            LangChain 等不使用此字段的框架默认为 None。
    """

    agent_id: str
    user_id: str
    session_id: str
    seq_id: Optional[int]
    type: str
    content: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
    updated_at: int = 0
    version: int = 0
    raw_event: Optional[str] = None

    def content_as_json(self) -> str:
        """将 content 序列化为 JSON 字符串。"""
        return json.dumps(self.content, ensure_ascii=False)

    @staticmethod
    def content_from_json(raw: str) -> dict[str, Any]:
        """从 JSON 字符串反序列化 content。"""
        result: dict[str, Any] = json.loads(raw)
        return result


@dataclass
class StateData:
    """状态数据对象。

    Attributes:
        state: 状态字典。
        created_at: 创建时间（纳秒时间戳）。
        updated_at: 最后更新时间（纳秒时间戳）。
        version: 乐观锁版本号。
    """

    state: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
    updated_at: int = 0
    version: int = 0
