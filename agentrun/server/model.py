"""AgentRun Server 模型定义 / AgentRun Server 模型Defines

定义 invokeAgent callback 的参数结构和响应类型"""

from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Dict,
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


class AgentRequest(BaseModel):
    """Agent 请求参数

    invokeAgent callback 接收的参数结构
    符合 OpenAI Completions API 格式
    """

    # 必需参数
    messages: List[Message] = Field(..., description="对话历史消息列表")

    # 可选参数
    model: Optional[str] = Field(None, description="模型名称")
    stream: bool = Field(False, description="是否使用流式输出")
    temperature: Optional[float] = Field(
        None, description="采样温度", ge=0.0, le=2.0
    )
    top_p: Optional[float] = Field(
        None, description="核采样参数", ge=0.0, le=1.0
    )
    max_tokens: Optional[int] = Field(
        None, description="最大生成 token 数", gt=0
    )
    tools: Optional[List[Tool]] = Field(None, description="可用的工具列表")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None, description="工具选择策略"
    )
    user: Optional[str] = Field(None, description="用户标识")

    # 扩展参数
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="其他自定义参数"
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
# 2. AgentRunResult - 核心数据结构
# 3. AgentResponse - 完整响应对象
# 4. ModelResponse - Model Service 响应
AgentResult = Union[
    str,  # 简化: 直接返回字符串
    Iterator[str],  # 简化: 字符串流
    AsyncIterator[str],  # 简化: 异步字符串流
    AgentRunResult,  # 核心: AgentRunResult 对象
    AgentResponse,  # 完整: AgentResponse 对象
    AgentStreamIterator,  # 流式: AgentResponse 流
    ModelServiceResult,  # Model Service: ModelResponse 或 CustomStreamWrapper
]
