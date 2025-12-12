"""协议抽象层 / Protocol Abstraction Layer

定义协议接口，支持多种协议格式（OpenAI, AG-UI 等）。

基于 Router 的设计:
- 每个协议提供自己的 FastAPI Router
- Server 负责挂载 Router 并管理路由前缀
- 协议完全自治，无需向 Server 声明接口
"""

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, TYPE_CHECKING, Union

from .model import AgentRequest, AgentReturnType

if TYPE_CHECKING:
    from fastapi import APIRouter, Request

    from .invoker import AgentInvoker


# ============================================================================
# 协议处理器基类
# ============================================================================


class ProtocolHandler(ABC):
    """协议处理器基类 / Protocol Handler Base Class

    基于 Router 的设计:
    协议通过 as_fastapi_router() 方法提供完整的路由定义，
    包括所有端点、请求处理、响应格式化等。

    Server 只需挂载 Router 并管理路由前缀，无需了解协议细节。

    Example:
        >>> class MyProtocolHandler(ProtocolHandler):
        ...     name = "my_protocol"
        ...
        ...     def as_fastapi_router(self, agent_invoker):
        ...         router = APIRouter()
        ...
        ...         @router.post("/run")
        ...         async def run(request: Request):
        ...             ...
        ...
        ...         return router
    """

    name: str

    @abstractmethod
    def as_fastapi_router(self, agent_invoker: "AgentInvoker") -> "APIRouter":
        """将协议转换为 FastAPI Router

        协议自己决定:
        - 有哪些端点
        - 端点的路径
        - HTTP 方法
        - 请求/响应处理

        Args:
            agent_invoker: Agent 调用器，用于执行用户的 invoke_agent

        Returns:
            APIRouter: FastAPI 路由器，包含该协议的所有端点
        """
        pass

    def get_prefix(self) -> str:
        """获取协议建议的路由前缀

        Server 会优先使用用户指定的前缀，如果没有指定则使用此建议值。

        Returns:
            str: 建议的前缀，如 "/v1" 或 ""

        Example:
            - OpenAI 协议: "/openai/v1"
            - AG-UI 协议: "/agui/v1"
            - 无前缀: ""
        """
        return ""


class BaseProtocolHandler(ProtocolHandler):
    """协议处理器扩展基类 / Extended Protocol Handler Base Class

    提供通用的请求解析和响应格式化逻辑。
    子类需要实现具体的协议转换。
    """

    async def parse_request(
        self,
        request: "Request",
        request_data: Dict[str, Any],
    ) -> tuple[AgentRequest, Dict[str, Any]]:
        """解析 HTTP 请求为 AgentRequest

        子类应该重写此方法来实现协议特定的解析逻辑。

        Args:
            request: FastAPI Request 对象
            request_data: 请求体 JSON 数据

        Returns:
            tuple: (AgentRequest, context)
                - AgentRequest: 标准化的请求对象
                - context: 协议特定的上下文信息
        """
        raise NotImplementedError("Subclass must implement parse_request")

    def _is_iterator(self, obj: Any) -> bool:
        """检查对象是否是迭代器

        Args:
            obj: 要检查的对象

        Returns:
            bool: 是否是迭代器
        """
        return (
            hasattr(obj, "__iter__")
            and not isinstance(obj, (str, bytes, dict, list))
        ) or hasattr(obj, "__aiter__")


# ============================================================================
# Handler 类型定义
# ============================================================================


# 同步 handler: 返回 AgentReturnType
SyncInvokeAgentHandler = Callable[[AgentRequest], AgentReturnType]

# 异步 handler: 返回 Awaitable[AgentReturnType]
AsyncInvokeAgentHandler = Callable[[AgentRequest], Awaitable[AgentReturnType]]

# 通用 handler: 同步或异步
InvokeAgentHandler = Union[SyncInvokeAgentHandler, AsyncInvokeAgentHandler]
