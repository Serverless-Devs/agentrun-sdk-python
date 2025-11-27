"""AgentRun HTTP Server / AgentRun HTTP 服务器

基于 Router 的设计 / Router-based design:
- 每个协议提供自己的 Router / Each protocol provides its own Router
- Server 负责挂载 Router 并管理路由前缀 / Server mounts Routers and manages route prefixes
- 支持多协议同时运行 / Supports running multiple protocols simultaneously
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
import uvicorn

from agentrun.utils.log import logger

from .invoker import AgentInvoker
from .openai_protocol import OpenAIProtocolHandler
from .protocol import InvokeAgentHandler, ProtocolHandler


class AgentRunServer:
    """AgentRun HTTP Server / AgentRun HTTP 服务器

    基于 Router 的架构 / Router-based architecture:
    - 每个协议提供完整的 FastAPI Router / Each protocol provides a complete FastAPI Router
    - Server 只负责组装和前缀管理 / Server only handles assembly and prefix management
    - 易于扩展新协议 / Easy to extend with new protocols

    Example (默认 OpenAI 协议 / Default OpenAI protocol):
        >>> def invoke_agent(request: AgentRequest):
        ...     return "Hello, world!"
        >>>
        >>> server = AgentRunServer(invoke_agent=invoke_agent)
        >>> server.start(port=8000)
        # 可访问 / Accessible: POST http://localhost:8000/v1/chat/completions

    Example (自定义前缀 / Custom prefix):
        >>> server = AgentRunServer(
        ...     invoke_agent=invoke_agent,
        ...     prefix_overrides={"OpenAIProtocolHandler": "/api/v1"}
        ... )
        >>> server.start(port=8000)
        # 可访问 / Accessible: POST http://localhost:8000/api/v1/chat/completions

    Example (多协议 / Multiple protocols):
        >>> server = AgentRunServer(
        ...     invoke_agent=invoke_agent,
        ...     protocols=[
        ...         OpenAIProtocolHandler(),
        ...         CustomProtocolHandler(),
        ...     ]
        ... )
        >>> server.start(port=8000)

    Example (集成到现有 FastAPI 应用 / Integrate with existing FastAPI app):
        >>> from fastapi import FastAPI
        >>>
        >>> app = FastAPI()
        >>> agent_server = AgentRunServer(invoke_agent=invoke_agent)
        >>> app.mount("/agent", agent_server.as_fastapi_app())
        # 可访问 / Accessible: POST http://localhost:8000/agent/v1/chat/completions
    """

    def __init__(
        self,
        invoke_agent: InvokeAgentHandler,
        protocols: Optional[List[ProtocolHandler]] = None,
        prefix_overrides: Optional[Dict[str, str]] = None,
    ):
        """初始化 AgentRun Server / Initialize AgentRun Server

        Args:
            invoke_agent: Agent 调用回调函数 / Agent invocation callback function
                - 可以是同步或异步函数 / Can be synchronous or asynchronous function
                - 支持返回字符串、AgentResponse 或生成器 / Supports returning string, AgentResponse or generator

            protocols: 协议处理器列表 / List of protocol handlers
                - 默认使用 OpenAI 协议 / Default uses OpenAI protocol
                - 可以添加自定义协议 / Can add custom protocols

            prefix_overrides: 协议前缀覆盖 / Protocol prefix overrides
                - 格式 / Format: {协议类名 / protocol class name: 前缀 / prefix}
                - 例如 / Example: {"OpenAIProtocolHandler": "/api/v1"}
        """
        self.app = FastAPI(title="AgentRun Server")
        self.agent_invoker = AgentInvoker(invoke_agent)

        # 默认使用 OpenAI 协议
        if protocols is None:
            protocols = [OpenAIProtocolHandler()]

        self.prefix_overrides = prefix_overrides or {}

        # 挂载所有协议的 Router
        self._mount_protocols(protocols)

    def _mount_protocols(self, protocols: List[ProtocolHandler]):
        """挂载所有协议的路由

        Args:
            protocols: 协议处理器列表
        """
        for protocol in protocols:
            # 获取协议的 Router
            router = protocol.as_fastapi_router(self.agent_invoker)

            # 确定路由前缀
            prefix = self._get_protocol_prefix(protocol)

            # 挂载到主应用
            self.app.include_router(router, prefix=prefix)

            logger.info(
                f"✅ 已挂载协议: {protocol.__class__.__name__} ->"
                f" {prefix or '(无前缀)'}"
            )

    def _get_protocol_prefix(self, protocol: ProtocolHandler) -> str:
        """获取协议的路由前缀

        优先级:
        1. 用户指定的覆盖前缀
        2. 协议自己的建议前缀
        3. 基于协议类名的默认前缀

        Args:
            protocol: 协议处理器

        Returns:
            str: 路由前缀
        """
        protocol_name = protocol.__class__.__name__

        # 1. 检查用户覆盖
        if protocol_name in self.prefix_overrides:
            return self.prefix_overrides[protocol_name]

        # 2. 使用协议建议
        suggested_prefix = protocol.get_prefix()
        if suggested_prefix:
            return suggested_prefix

        # 3. 默认前缀(基于类名)
        # OpenAIProtocolHandler -> /openai
        name_without_handler = protocol_name.replace(
            "ProtocolHandler", ""
        ).replace("Handler", "")
        return f"/{name_without_handler.lower()}"

    def start(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        log_level: str = "info",
        **kwargs: Any,
    ):
        """启动 HTTP 服务器

        Args:
            host: 监听地址,默认 0.0.0.0
            port: 监听端口,默认 9000
            log_level: 日志级别,默认 info
            **kwargs: 传递给 uvicorn.run 的其他参数
        """
        logger.info(f"🚀 启动 AgentRun Server: http://{host}:{port}")

        # 打印路由信息
        # for route in self.app.routes:
        #     if hasattr(route, "methods") and hasattr(route, "path"):
        #         methods = ", ".join(route.methods)  # type: ignore
        # logger.info(f"   {methods:10} {route.path}")  # type: ignore

        uvicorn.run(
            self.app, host=host, port=port, log_level=log_level, **kwargs
        )

    def as_fastapi_app(self) -> FastAPI:
        """导出 FastAPI 应用

        用于集成到现有的 FastAPI 项目中。

        Returns:
            FastAPI: FastAPI 应用实例

        Example:
            >>> from fastapi import FastAPI
            >>>
            >>> app = FastAPI()
            >>> agent_server = AgentRunServer(invoke_agent=invoke_agent)
            >>> app.mount("/agent", agent_server.as_fastapi_app())
        """
        return self.app
