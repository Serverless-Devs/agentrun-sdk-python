"""协议抽象层 / Protocol Abstraction Layer

定义协议接口,支持未来扩展多种协议格式(OpenAI, AG-UI, Anthropic, Google 等)。
Defines protocol interfaces, supporting future expansion of various protocol formats (OpenAI, AG-UI, Anthropic, Google, etc.).

基于 Router 的设计 / Router-based design:
- 每个协议提供自己的 FastAPI Router / Each protocol provides its own FastAPI Router
- Server 负责挂载 Router 并管理路由前缀 / Server mounts Routers and manages route prefixes
- 协议完全自治,无需向 Server 声明接口 / Protocols are fully autonomous, no need to declare interfaces to Server

生命周期钩子设计 / Lifecycle Hooks Design:
- 每个协议实现自己的 AgentLifecycleHooks 子类
- 钩子在请求解析时注入到 AgentRequest
- Agent 可以通过 hooks 发送协议特定的事件
"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    TYPE_CHECKING,
    Union,
)

from .model import AgentLifecycleHooks, AgentRequest, AgentResult

if TYPE_CHECKING:
    from fastapi import APIRouter, Request

    from .invoker import AgentInvoker


class ProtocolHandler(ABC):
    """协议处理器基类 / Protocol Handler Base Class

    基于 Router 的设计 / Router-based design:
    协议通过 as_fastapi_router() 方法提供完整的路由定义,包括所有端点、请求处理、响应格式化等。
    Protocol provides complete route definitions through as_fastapi_router() method, including all endpoints, request handling, response formatting, etc.

    Server 只需挂载 Router 并管理路由前缀,无需了解协议细节。
    Server only needs to mount Router and manage route prefixes, without knowing protocol details.
    """

    @abstractmethod
    def as_fastapi_router(self, agent_invoker: "AgentInvoker") -> "APIRouter":
        """
        将协议转换为 FastAPI Router

        协议自己决定:
        - 有哪些端点
        - 端点的路径
        - HTTP 方法
        - 请求/响应处理

        Args:
            agent_invoker: Agent 调用器,用于执行用户的 invoke_agent

        Returns:
            APIRouter: FastAPI 路由器,包含该协议的所有端点

        Example:
            ```python
            def as_fastapi_router(self, agent_invoker):
                router = APIRouter()

                @router.post("/chat/completions")
                async def chat_completions(request: Request):
                    data = await request.json()
                    agent_request = parse_request(data)
                    result = await agent_invoker.invoke(agent_request)
                    return format_response(result)

                return router
            ```
        """
        pass

    def get_prefix(self) -> str:
        """
        获取协议建议的路由前缀

        Server 会优先使用用户指定的前缀,如果没有指定则使用此建议值。

        Returns:
            str: 建议的前缀,如 "/v1" 或 ""

        Example:
            - OpenAI 协议: "/v1"
            - Anthropic 协议: "/anthropic"
            - 无前缀: ""
        """
        return ""


class BaseProtocolHandler(ProtocolHandler):
    """协议处理器扩展基类 / Extended Protocol Handler Base Class

    提供通用的请求解析、响应格式化和钩子创建逻辑。
    子类可以重写特定方法来实现协议特定的行为。

    主要职责:
    1. 创建协议特定的生命周期钩子
    2. 解析请求并注入钩子和原始请求信息
    3. 格式化响应为协议特定格式

    Example:
        >>> class MyProtocolHandler(BaseProtocolHandler):
        ...     def create_hooks(self, context):
        ...         return MyProtocolHooks(context)
        ...
        ...     async def parse_request(self, request):
        ...         # 自定义解析逻辑
        ...         pass
    """

    @abstractmethod
    def create_hooks(self, context: Dict[str, Any]) -> AgentLifecycleHooks:
        """创建协议特定的生命周期钩子

        Args:
            context: 运行上下文，包含 threadId, runId, messageId 等

        Returns:
            AgentLifecycleHooks: 协议特定的钩子实现
        """
        pass

    async def parse_request(
        self,
        request: "Request",
        request_data: Dict[str, Any],
    ) -> tuple[AgentRequest, Dict[str, Any]]:
        """解析 HTTP 请求为 AgentRequest

        子类应该重写此方法来实现协议特定的解析逻辑。
        基类提供通用的原始请求信息提取。

        Args:
            request: FastAPI Request 对象
            request_data: 请求体 JSON 数据

        Returns:
            tuple: (AgentRequest, context)
                - AgentRequest: 标准化的请求对象
                - context: 协议特定的上下文信息
        """
        # 提取原始请求头
        raw_headers = dict(request.headers)

        # 子类需要实现具体的解析逻辑
        raise NotImplementedError("Subclass must implement parse_request")

    async def format_response(
        self,
        result: AgentResult,
        request: AgentRequest,
        context: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """格式化 Agent 结果为协议特定的响应

        Args:
            result: Agent 执行结果
            request: 原始请求
            context: 协议特定的上下文

        Yields:
            协议特定格式的响应数据
        """
        raise NotImplementedError("Subclass must implement format_response")

    def _is_iterator(self, obj: Any) -> bool:
        """检查对象是否是迭代器

        Args:
            obj: 要检查的对象

        Returns:
            bool: 是否是迭代器
        """
        return (
            hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict))
        ) or hasattr(obj, "__aiter__")


# Handler 类型定义
# 同步 handler: 普通函数,直接返回 AgentResult
SyncInvokeAgentHandler = Callable[[AgentRequest], AgentResult]

# 异步 handler: 协程函数,返回 Awaitable[AgentResult]
AsyncInvokeAgentHandler = Callable[[AgentRequest], Awaitable[AgentResult]]

# 通用 handler: 可以是同步或异步
InvokeAgentHandler = Union[
    SyncInvokeAgentHandler,
    AsyncInvokeAgentHandler,
]
