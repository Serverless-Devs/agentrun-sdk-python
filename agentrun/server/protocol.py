"""协议抽象层 / Protocol Abstraction Layer

定义协议接口,支持未来扩展多种协议格式(OpenAI, Anthropic, Google 等)。
Defines protocol interfaces, supporting future expansion of various protocol formats (OpenAI, Anthropic, Google, etc.).

基于 Router 的设计 / Router-based design:
- 每个协议提供自己的 FastAPI Router / Each protocol provides its own FastAPI Router
- Server 负责挂载 Router 并管理路由前缀 / Server mounts Routers and manages route prefixes
- 协议完全自治,无需向 Server 声明接口 / Protocols are fully autonomous, no need to declare interfaces to Server
"""

from abc import ABC, abstractmethod
from typing import Awaitable, Callable, TYPE_CHECKING, Union

from .model import AgentRequest, AgentResult

if TYPE_CHECKING:
    from fastapi import APIRouter

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
