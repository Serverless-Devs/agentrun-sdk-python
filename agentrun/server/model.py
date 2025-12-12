"""AgentRun Server 模型定义 / AgentRun Server Model Definitions

定义标准化的 AgentRequest 和 AgentResult 数据结构。
基于 AG-UI 协议进行扩展，支持多协议转换。

参考: https://docs.ag-ui.com/concepts/events
"""

from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Union,
)

from ..utils.model import BaseModel, Field

# ============================================================================
# 协议配置
# ============================================================================


class ProtocolConfig(BaseModel):
    prefix: Optional[str] = None
    enable: bool = True


class ServerConfig(BaseModel):
    openai: Optional["OpenAIProtocolConfig"] = None
    agui: Optional[ProtocolConfig] = None
    cors_origins: Optional[List[str]] = None


# ============================================================================
# 消息角色和消息体定义
# ============================================================================


class MessageRole(str, Enum):
    """消息角色 / Message Role"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """工具调用 / Tool Call"""

    id: str
    type: str = "function"
    function: Dict[str, Any]


class Message(BaseModel):
    """标准化消息体 / Standardized Message

    兼容 AG-UI 和 OpenAI 消息格式。

    Attributes:
        id: 消息唯一标识（AG-UI 格式）
        role: 消息角色
        content: 消息内容（字符串或多模态内容列表）
        name: 发送者名称（可选）
        tool_calls: 工具调用列表（assistant 消息）
        tool_call_id: 对应的工具调用 ID（tool 消息）
    """

    id: Optional[str] = None
    role: MessageRole
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class Tool(BaseModel):
    """工具定义 / Tool Definition

    兼容 AG-UI 和 OpenAI 工具格式。
    """

    type: str = "function"
    function: Dict[str, Any]


# ============================================================================
# AG-UI 事件类型定义（完整超集）
# ============================================================================


class EventType(str, Enum):
    """AG-UI 事件类型（完整超集）

    包含 AG-UI 协议的所有事件类型，以及扩展事件。
    参考: https://docs.ag-ui.com/concepts/events
    """

    # =========================================================================
    # Lifecycle Events（生命周期事件）
    # =========================================================================
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"

    # =========================================================================
    # Text Message Events（文本消息事件）
    # =========================================================================
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TEXT_MESSAGE_CHUNK = (
        "TEXT_MESSAGE_CHUNK"  # 简化事件（包含 start/content/end）
    )

    # =========================================================================
    # Tool Call Events（工具调用事件）
    # =========================================================================
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    TOOL_CALL_CHUNK = "TOOL_CALL_CHUNK"  # 简化事件（包含 start/args/end）

    # =========================================================================
    # State Management Events（状态管理事件）
    # =========================================================================
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"

    # =========================================================================
    # Message Snapshot Events（消息快照事件）
    # =========================================================================
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"

    # =========================================================================
    # Activity Events（活动事件）
    # =========================================================================
    ACTIVITY_SNAPSHOT = "ACTIVITY_SNAPSHOT"
    ACTIVITY_DELTA = "ACTIVITY_DELTA"

    # =========================================================================
    # Reasoning Events（推理事件）
    # =========================================================================
    REASONING_START = "REASONING_START"
    REASONING_MESSAGE_START = "REASONING_MESSAGE_START"
    REASONING_MESSAGE_CONTENT = "REASONING_MESSAGE_CONTENT"
    REASONING_MESSAGE_END = "REASONING_MESSAGE_END"
    REASONING_MESSAGE_CHUNK = "REASONING_MESSAGE_CHUNK"
    REASONING_END = "REASONING_END"

    # =========================================================================
    # Meta Events（元事件）
    # =========================================================================
    META_EVENT = "META_EVENT"

    # =========================================================================
    # Special Events（特殊事件）
    # =========================================================================
    RAW = "RAW"  # 原始事件
    CUSTOM = "CUSTOM"  # 自定义事件

    # =========================================================================
    # Extended Events（扩展事件 - 非 AG-UI 标准）
    # =========================================================================
    STREAM_DATA = "STREAM_DATA"  # 原始流数据（用户可直接发送任意 SSE 内容）


# ============================================================================
# Addition Mode（附加字段合并模式）
# ============================================================================


class AdditionMode(str, Enum):
    """附加字段合并模式

    控制 AgentResult.addition 如何与协议默认字段合并。
    """

    REPLACE = "replace"  # 完全覆盖协议默认值
    MERGE = "merge"  # 深度合并（使用 helper.merge）
    PROTOCOL_ONLY = "protocol_only"  # 仅覆盖协议原有字段，不添加新字段


# ============================================================================
# AgentResult（标准化返回值）
# ============================================================================


class AgentResult(BaseModel):
    """Agent 执行结果事件

    标准化的返回值结构，基于 AG-UI 事件模型。
    框架层会自动将 AgentResult 转换为对应协议的格式。

    Attributes:
        event: 事件类型（AG-UI 事件枚举）
        data: 事件数据
        addition: 额外附加字段（可选）
        addition_mode: 附加字段合并模式

    Example (文本消息):
        >>> yield AgentResult(
        ...     event=EventType.TEXT_MESSAGE_CONTENT,
        ...     data={"message_id": "msg-1", "delta": "Hello"}
        ... )

    Example (工具调用):
        >>> yield AgentResult(
        ...     event=EventType.TOOL_CALL_START,
        ...     data={"tool_call_id": "tc-1", "tool_call_name": "get_weather"}
        ... )

    Example (原始流数据):
        >>> yield AgentResult(
        ...     event=EventType.STREAM_DATA,
        ...     data={"raw": "data: {...}\\n\\n"}
        ... )

    Example (自定义事件):
        >>> yield AgentResult(
        ...     event=EventType.CUSTOM,
        ...     data={"name": "my_event", "value": {"foo": "bar"}}
        ... )
    """

    event: EventType
    data: Dict[str, Any] = Field(default_factory=dict)
    addition: Optional[Dict[str, Any]] = None
    addition_mode: AdditionMode = AdditionMode.MERGE


# ============================================================================
# AgentRequest（标准化请求）
# ============================================================================


class AgentRequest(BaseModel):
    """Agent 请求参数（协议无关）

    标准化的请求结构，统一了 OpenAI 和 AG-UI 协议的输入格式。

    Attributes:
        messages: 对话历史消息列表（标准化格式）
        stream: 是否使用流式输出
        tools: 可用的工具列表（AG-UI 格式）
        body: 原始 HTTP 请求体
        headers: 原始 HTTP 请求头

    Example (基本使用):
        >>> def invoke_agent(request: AgentRequest):
        ...     user_msg = request.messages[-1].content
        ...     return f"你说的是: {user_msg}"

    Example (流式输出):
        >>> async def invoke_agent(request: AgentRequest):
        ...     for word in ["Hello", " ", "World"]:
        ...         yield word

    Example (使用事件):
        >>> async def invoke_agent(request: AgentRequest):
        ...     yield AgentResult(
        ...         event=EventType.STEP_STARTED,
        ...         data={"step_name": "thinking"}
        ...     )
        ...     yield "I'm thinking..."
        ...     yield AgentResult(
        ...         event=EventType.STEP_FINISHED,
        ...         data={"step_name": "thinking"}
        ...     )

    Example (工具调用):
        >>> async def invoke_agent(request: AgentRequest):
        ...     yield AgentResult(
        ...         event=EventType.TOOL_CALL_START,
        ...         data={"tool_call_id": "tc-1", "tool_call_name": "search"}
        ...     )
        ...     yield AgentResult(
        ...         event=EventType.TOOL_CALL_ARGS,
        ...         data={"tool_call_id": "tc-1", "delta": '{"query": "weather"}'}
        ...     )
        ...     result = do_search("weather")
        ...     yield AgentResult(
        ...         event=EventType.TOOL_CALL_RESULT,
        ...         data={"tool_call_id": "tc-1", "result": result}
        ...     )
        ...     yield AgentResult(
        ...         event=EventType.TOOL_CALL_END,
        ...         data={"tool_call_id": "tc-1"}
        ...     )
    """

    model_config = {"arbitrary_types_allowed": True}

    # 标准化参数
    messages: List[Message] = Field(
        default_factory=list, description="对话历史消息列表"
    )
    stream: bool = Field(False, description="是否使用流式输出")
    tools: Optional[List[Tool]] = Field(None, description="可用的工具列表")

    # 原始请求信息
    body: Dict[str, Any] = Field(
        default_factory=dict, description="原始 HTTP 请求体"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict, description="原始 HTTP 请求头"
    )


# ============================================================================
# OpenAI 协议配置（前置声明）
# ============================================================================


class OpenAIProtocolConfig(ProtocolConfig):
    """OpenAI 协议配置"""

    enable: bool = True
    prefix: Optional[str] = "/openai/v1"
    model_name: Optional[str] = None


# ============================================================================
# 返回值类型别名
# ============================================================================


# 单个结果项：可以是字符串或 AgentResult
AgentResultItem = Union[str, AgentResult]

# 同步生成器
SyncAgentResultGenerator = Generator[AgentResultItem, None, None]

# 异步生成器
AsyncAgentResultGenerator = AsyncGenerator[AgentResultItem, None]

# Agent 函数返回值类型
AgentReturnType = Union[
    # 简单返回
    str,  # 直接返回字符串
    AgentResult,  # 返回单个事件
    List[AgentResult],  # 返回多个事件（非流式）
    Dict[str, Any],  # 返回字典（如 OpenAI/AG-UI 非流式响应）
    # 迭代器/生成器返回（流式）
    Iterator[AgentResultItem],
    AsyncIterator[AgentResultItem],
    SyncAgentResultGenerator,
    AsyncAgentResultGenerator,
]
