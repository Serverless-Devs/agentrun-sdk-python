"""AgentRun Server 模型定义 / AgentRun Server Model Definitions

定义 invokeAgent callback 的参数结构、响应类型和生命周期钩子。
Defines invokeAgent callback parameter structures, response types, and lifecycle hooks.
"""

from abc import ABC, abstractmethod
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
    TYPE_CHECKING,
    Union,
)

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    # 运行时不导入,避免依赖问题
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.types.utils import ModelResponse


class MessageRole(str, Enum):
    """消息角色"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """消息体"""

    role: MessageRole
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ToolCall(BaseModel):
    """工具调用"""

    id: str
    type: str = "function"
    function: Dict[str, Any]


class Tool(BaseModel):
    """工具定义 / 工具Defines"""

    type: str = "function"
    function: Dict[str, Any]


# ============================================================================
# 生命周期钩子类型定义 / Lifecycle Hook Type Definitions
# ============================================================================


class AgentLifecycleHooks(ABC):
    """Agent 生命周期钩子抽象基类

    定义 Agent 执行过程中的所有生命周期事件。
    不同协议（OpenAI、AG-UI 等）实现各自的钩子处理逻辑。

    所有 on_* 方法直接返回一个 AgentEvent 对象，可以直接 yield。
    对于不支持的事件，返回 None。

    Example (同步):
        >>> def invoke_agent(request: AgentRequest):
        ...     hooks = request.hooks
        ...     yield hooks.on_step_start("processing")
        ...     yield "Hello, world!"
        ...     yield hooks.on_step_finish("processing")

    Example (异步):
        >>> async def invoke_agent(request: AgentRequest):
        ...     hooks = request.hooks
        ...     yield hooks.on_step_start("processing")
        ...     yield "Hello, world!"
        ...     yield hooks.on_step_finish("processing")

    Example (工具调用):
        >>> def invoke_agent(request: AgentRequest):
        ...     hooks = request.hooks
        ...     yield hooks.on_tool_call_start(id="call_1", name="get_time")
        ...     yield hooks.on_tool_call_args(id="call_1", args='{"tz": "UTC"}')
        ...     result = get_time(tz="UTC")
        ...     yield hooks.on_tool_call_result(id="call_1", result=result)
        ...     yield hooks.on_tool_call_end(id="call_1")
        ...     yield f"当前时间: {result}"
    """

    # =========================================================================
    # 生命周期事件方法 (on_*) - 直接返回 AgentEvent，可以直接 yield
    # =========================================================================

    @abstractmethod
    def on_run_start(self) -> Optional["AgentEvent"]:
        """运行开始事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_run_finish(self) -> Optional["AgentEvent"]:
        """运行结束事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_run_error(
        self, error: str, code: Optional[str] = None
    ) -> Optional["AgentEvent"]:
        """运行错误事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_step_start(
        self, step_name: Optional[str] = None
    ) -> Optional["AgentEvent"]:
        """步骤开始事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_step_finish(
        self, step_name: Optional[str] = None
    ) -> Optional["AgentEvent"]:
        """步骤结束事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_text_message_start(
        self, message_id: str, role: str = "assistant"
    ) -> Optional["AgentEvent"]:
        """文本消息开始事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_text_message_content(
        self, message_id: str, delta: str
    ) -> Optional["AgentEvent"]:
        """文本消息内容事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_text_message_end(self, message_id: str) -> Optional["AgentEvent"]:
        """文本消息结束事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_tool_call_start(
        self,
        id: str,
        name: str,
        parent_message_id: Optional[str] = None,
    ) -> Optional["AgentEvent"]:
        """工具调用开始事件

        Args:
            id: 工具调用 ID
            name: 工具名称
            parent_message_id: 父消息 ID（可选）
        """
        return None  # pragma: no cover

    @abstractmethod
    def on_tool_call_args_delta(
        self, id: str, delta: str
    ) -> Optional["AgentEvent"]:
        """工具调用参数增量事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_tool_call_args(
        self, id: str, args: Union[str, Dict[str, Any]]
    ) -> Optional["AgentEvent"]:
        """工具调用参数完成事件

        Args:
            id: 工具调用 ID
            args: 参数，可以是 JSON 字符串或字典
        """
        return None  # pragma: no cover

    @abstractmethod
    def on_tool_call_result_delta(
        self, id: str, delta: str
    ) -> Optional["AgentEvent"]:
        """工具调用结果增量事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_tool_call_result(
        self, id: str, result: str
    ) -> Optional["AgentEvent"]:
        """工具调用结果完成事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_tool_call_end(self, id: str) -> Optional["AgentEvent"]:
        """工具调用结束事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_state_snapshot(
        self, snapshot: Dict[str, Any]
    ) -> Optional["AgentEvent"]:
        """状态快照事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_state_delta(
        self, delta: List[Dict[str, Any]]
    ) -> Optional["AgentEvent"]:
        """状态增量事件"""
        return None  # pragma: no cover

    @abstractmethod
    def on_custom_event(self, name: str, value: Any) -> Optional["AgentEvent"]:
        """自定义事件"""
        return None  # pragma: no cover


class AgentEvent:
    """Agent 事件

    表示一个生命周期事件，可以被 yield 给框架处理。
    框架会根据协议将其转换为相应的格式。

    Attributes:
        event_type: 事件类型
        data: 事件数据
        raw_sse: 原始 SSE 格式字符串（可选，用于直接输出）
    """

    def __init__(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        raw_sse: Optional[str] = None,
    ):
        self.event_type = event_type
        self.data = data or {}
        self.raw_sse = raw_sse

    def __repr__(self) -> str:
        return f"AgentEvent(type={self.event_type}, data={self.data})"

    def __bool__(self) -> bool:
        """允许在 if 语句中检查事件是否有效"""
        return self.raw_sse is not None or bool(self.data)


class AgentRequest(BaseModel):
    """Agent 请求参数（协议无关）

    invokeAgent callback 接收的参数结构。
    只包含协议无关的核心字段，协议特定参数（如 OpenAI 的 temperature、top_p 等）
    可通过 raw_body 访问。

    Attributes:
        messages: 对话历史消息列表
        stream: 是否使用流式输出
        tools: 可用的工具列表
        raw_headers: 原始 HTTP 请求头
        raw_body: 原始 HTTP 请求体（包含协议特定参数）
        hooks: 生命周期钩子，用于发送协议特定事件

    Example (基本使用):
        >>> def invoke_agent(request: AgentRequest):
        ...     # 获取用户消息
        ...     user_msg = request.messages[-1].content
        ...     return f"你说的是: {user_msg}"

    Example (访问协议特定参数):
        >>> def invoke_agent(request: AgentRequest):
        ...     # OpenAI 特定参数从 raw_body 获取
        ...     temperature = request.raw_body.get("temperature", 0.7)
        ...     top_p = request.raw_body.get("top_p")
        ...     max_tokens = request.raw_body.get("max_tokens")
        ...     return "Hello, world!"

    Example (使用生命周期钩子):
        >>> def invoke_agent(request: AgentRequest):
        ...     hooks = request.hooks
        ...     yield hooks.on_step_start("processing")
        ...     yield "Hello, world!"
        ...     yield hooks.on_step_finish("processing")

    Example (工具调用):
        >>> def invoke_agent(request: AgentRequest):
        ...     hooks = request.hooks
        ...     yield hooks.on_tool_call_start(id="call_1", name="get_time")
        ...     yield hooks.on_tool_call_args(id="call_1", args={"tz": "UTC"})
        ...     result = get_time(tz="UTC")
        ...     yield hooks.on_tool_call_result(id="call_1", result=result)
        ...     yield hooks.on_tool_call_end(id="call_1")
        ...     yield f"当前时间: {result}"
    """

    model_config = {"arbitrary_types_allowed": True}

    # 核心参数（协议无关）
    messages: List[Message] = Field(..., description="对话历史消息列表")
    stream: bool = Field(False, description="是否使用流式输出")
    tools: Optional[List[Tool]] = Field(None, description="可用的工具列表")

    # 原始请求信息（包含协议特定参数）
    raw_headers: Dict[str, str] = Field(
        default_factory=dict, description="原始 HTTP 请求头"
    )
    raw_body: Dict[str, Any] = Field(
        default_factory=dict,
        description="原始 HTTP 请求体，包含协议特定参数如 temperature、top_p 等",
    )

    # 生命周期钩子
    hooks: Optional[AgentLifecycleHooks] = Field(
        None, description="生命周期钩子，由协议层注入"
    )

    # 扩展参数（协议层解析后的额外信息）
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="协议层解析后的额外信息"
    )


class AgentResponseChoice(BaseModel):
    """响应选项"""

    index: int
    message: Message
    finish_reason: Optional[str] = None


class AgentResponseUsage(BaseModel):
    """Token 使用统计"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgentRunResult(BaseModel):
    """Agent 运行结果

    核心数据结构,用于表示 Agent 执行结果。
    content 字段支持字符串或字符串迭代器。

    Example:
        >>> # 返回字符串
        >>> AgentRunResult(content="Hello, world!")
        >>>
        >>> # 返回字符串迭代器(流式)
        >>> def stream():
        ...     yield "Hello, "
        ...     yield "world!"
        >>> AgentRunResult(content=stream())
    """

    model_config = {"arbitrary_types_allowed": True}

    content: Union[str, Iterator[str], AsyncIterator[str], Any]
    """响应内容,支持字符串或字符串迭代器 / 响应内容,Supports字符串或字符串迭代器"""


class AgentResponse(BaseModel):
    """Agent 响应(非流式)

    灵活的响应数据结构,所有字段都是可选的。
    用户可以只填充需要的字段,协议层会根据实际协议格式补充或跳过字段。

    Example:
        >>> # 最简单 - 只返回内容
        >>> AgentResponse(content="Hello")
        >>>
        >>> # OpenAI 格式 - 完整字段
        >>> AgentResponse(
        ...     id="chatcmpl-123",
        ...     model="gpt-4",
        ...     choices=[...]
        ... )
    """

    # 核心字段 - 协议无关
    content: Optional[str] = None
    """响应内容"""

    # OpenAI 协议字段 - 可选
    id: Optional[str] = Field(None, description="响应 ID")
    object: Optional[str] = Field(None, description="对象类型")
    created: Optional[int] = Field(None, description="创建时间戳")
    model: Optional[str] = Field(None, description="使用的模型")
    choices: Optional[List[AgentResponseChoice]] = Field(
        None, description="响应选项列表"
    )
    usage: Optional[AgentResponseUsage] = Field(
        None, description="Token 使用情况"
    )

    # 扩展字段 - 其他协议可能需要
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="协议特定的额外字段"
    )


class AgentStreamResponseDelta(BaseModel):
    """流式响应增量"""

    role: Optional[MessageRole] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class AgentStreamResponse(BaseModel):
    """流式响应块"""

    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choices: Optional[List["AgentStreamResponseChoice"]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class AgentStreamResponseChoice(BaseModel):
    """流式响应选项"""

    index: int
    delta: AgentStreamResponseDelta
    finish_reason: Optional[str] = None


# 类型别名 - 流式响应迭代器
AgentStreamIterator = Union[
    Iterator[AgentResponse],
    AsyncIterator[AgentResponse],
]

# Model Service 类型 - 直接返回 litellm 的 ModelResponse
if TYPE_CHECKING:
    ModelServiceResult = Union["ModelResponse", "CustomStreamWrapper"]
else:
    ModelServiceResult = Any  # 运行时使用 Any

# AgentResult - 支持多种返回形式
# 用户可以返回:
# 1. string 或 string 迭代器 - 自动转换为 AgentRunResult
# 2. AgentEvent - 生命周期事件
# 3. AgentRunResult - 核心数据结构
# 4. AgentResponse - 完整响应对象
# 5. ModelResponse - Model Service 响应
# 6. 混合迭代器/生成器 - 可以 yield AgentEvent、str 或 None
AgentResult = Union[
    str,  # 简化: 直接返回字符串
    AgentEvent,  # 事件: 生命周期事件
    Iterator[str],  # 简化: 字符串流
    AsyncIterator[str],  # 简化: 异步字符串流
    Generator[str, None, None],  # 生成器: 字符串流
    AsyncGenerator[str, None],  # 异步生成器: 字符串流
    Iterator[Union[AgentEvent, str, None]],  # 混合流: AgentEvent、str 或 None
    AsyncIterator[Union[AgentEvent, str, None]],  # 异步混合流
    Generator[Union[AgentEvent, str, None], None, None],  # 混合生成器
    AsyncGenerator[Union[AgentEvent, str, None], None],  # 异步混合生成器
    AgentRunResult,  # 核心: AgentRunResult 对象
    AgentResponse,  # 完整: AgentResponse 对象
    AgentStreamIterator,  # 流式: AgentResponse 流
    ModelServiceResult,  # Model Service: ModelResponse 或 CustomStreamWrapper
]
