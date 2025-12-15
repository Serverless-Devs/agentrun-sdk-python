"""AgentRun Server 模型定义 / AgentRun Server Model Definitions

定义标准化的 AgentRequest 和 AgentEvent 数据结构。
采用协议无关的设计，支持多协议转换（OpenAI、AG-UI 等）。
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
# 事件类型定义（协议无关）
# ============================================================================


class EventType(str, Enum):
    """事件类型（协议无关）

    定义核心事件类型，框架会自动转换为对应协议格式（OpenAI、AG-UI 等）。
    用户只需关心语义，无需关心具体协议细节。

    边界事件（如消息开始/结束、生命周期开始/结束）由协议层自动处理，
    用户无需关心。
    """

    # =========================================================================
    # 核心事件（用户主要使用）
    # =========================================================================
    TEXT = "TEXT"  # 文本内容块
    TOOL_CALL = "TOOL_CALL"  # 完整工具调用（含 id, name, args）
    TOOL_CALL_CHUNK = "TOOL_CALL_CHUNK"  # 工具调用参数片段（流式场景）
    TOOL_RESULT = "TOOL_RESULT"  # 工具执行结果
    ERROR = "ERROR"  # 错误事件
    STATE = "STATE"  # 状态更新（快照或增量）

    # =========================================================================
    # 扩展事件
    # =========================================================================
    CUSTOM = "CUSTOM"  # 自定义事件（协议层会正确处理）
    RAW = "RAW"  # 原始协议数据（直接透传到响应流）


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
# AgentEvent（标准化事件）
# ============================================================================


class AgentEvent(BaseModel):
    """Agent 执行事件

    标准化的事件结构，协议无关设计。
    框架层会自动将 AgentEvent 转换为对应协议的格式（OpenAI、AG-UI 等）。

    Attributes:
        event: 事件类型
        data: 事件数据
        addition: 额外附加字段（可选，用于协议特定扩展）
        addition_mode: 附加字段合并模式

    Example (文本消息):
        >>> yield AgentEvent(
        ...     event=EventType.TEXT,
        ...     data={"delta": "Hello, world!"}
        ... )

    Example (完整工具调用):
        >>> yield AgentEvent(
        ...     event=EventType.TOOL_CALL,
        ...     data={
        ...         "id": "tc-1",
        ...         "name": "get_weather",
        ...         "args": '{"location": "Beijing"}'
        ...     }
        ... )

    Example (流式工具调用):
        >>> yield AgentEvent(
        ...     event=EventType.TOOL_CALL_CHUNK,
        ...     data={"id": "tc-1", "name": "search", "args_delta": '{"q":'}
        ... )
        >>> yield AgentEvent(
        ...     event=EventType.TOOL_CALL_CHUNK,
        ...     data={"id": "tc-1", "args_delta": '"test"}'}
        ... )

    Example (工具执行结果):
        >>> yield AgentEvent(
        ...     event=EventType.TOOL_RESULT,
        ...     data={"id": "tc-1", "result": "Sunny, 25°C"}
        ... )

    Example (自定义事件):
        >>> yield AgentEvent(
        ...     event=EventType.CUSTOM,
        ...     data={"name": "step_started", "value": {"step": "thinking"}}
        ... )

    Example (原始协议数据):
        >>> yield AgentEvent(
        ...     event=EventType.RAW,
        ...     data={"raw": "data: {...}\\n\\n"}
        ... )
    """

    event: EventType
    data: Dict[str, Any] = Field(default_factory=dict)
    addition: Optional[Dict[str, Any]] = None
    addition_mode: AdditionMode = AdditionMode.MERGE


# 兼容别名
AgentResult = AgentEvent


# ============================================================================
# AgentRequest（标准化请求）
# ============================================================================


class AgentRequest(BaseModel):
    """Agent 请求参数（协议无关）

    标准化的请求结构，统一了 OpenAI 和 AG-UI 协议的输入格式。

    Attributes:
        protocol: 当前交互协议名称（如 "openai", "agui"）
        messages: 对话历史消息列表（标准化格式）
        stream: 是否使用流式输出
        tools: 可用的工具列表
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
        ...     yield AgentEvent(
        ...         event=EventType.CUSTOM,
        ...         data={"name": "step_started", "value": {"step": "thinking"}}
        ...     )
        ...     yield "I'm thinking..."
        ...     yield AgentEvent(
        ...         event=EventType.CUSTOM,
        ...         data={"name": "step_finished", "value": {"step": "thinking"}}
        ...     )

    Example (工具调用):
        >>> async def invoke_agent(request: AgentRequest):
        ...     # 完整工具调用
        ...     yield AgentEvent(
        ...         event=EventType.TOOL_CALL,
        ...         data={
        ...             "id": "tc-1",
        ...             "name": "search",
        ...             "args": '{"query": "weather"}'
        ...         }
        ...     )
        ...     # 执行工具并返回结果
        ...     result = do_search("weather")
        ...     yield AgentEvent(
        ...         event=EventType.TOOL_RESULT,
        ...         data={"id": "tc-1", "result": result}
        ...     )

    Example (根据协议差异化处理):
        >>> async def invoke_agent(request: AgentRequest):
        ...     if request.protocol == "openai":
        ...         # OpenAI 特定处理
        ...         pass
        ...     elif request.protocol == "agui":
        ...         # AG-UI 特定处理
        ...         pass
    """

    model_config = {"arbitrary_types_allowed": True}

    # 协议信息
    protocol: str = Field("unknown", description="当前交互协议名称")

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


# 单个结果项：可以是字符串或 AgentEvent
AgentEventItem = Union[str, AgentEvent]

# 兼容别名
AgentResultItem = AgentEventItem

# 同步生成器
SyncAgentEventGenerator = Generator[AgentEventItem, None, None]
SyncAgentResultGenerator = SyncAgentEventGenerator  # 兼容别名

# 异步生成器
AsyncAgentEventGenerator = AsyncGenerator[AgentEventItem, None]
AsyncAgentResultGenerator = AsyncAgentEventGenerator  # 兼容别名

# Agent 函数返回值类型
AgentReturnType = Union[
    # 简单返回
    str,  # 直接返回字符串
    AgentEvent,  # 返回单个事件
    List[AgentEvent],  # 返回多个事件（非流式）
    Dict[str, Any],  # 返回字典（如 OpenAI/AG-UI 非流式响应）
    # 迭代器/生成器返回（流式）
    Iterator[AgentEventItem],
    AsyncIterator[AgentEventItem],
    SyncAgentEventGenerator,
    AsyncAgentEventGenerator,
]
