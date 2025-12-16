"""AgentRun HTTP Server / AgentRun HTTP 服务器

基于 Router 的设计:
- 每个协议提供自己的 Router
- Server 负责挂载 Router 并管理路由前缀
- 支持多协议同时运行（OpenAI + AG-UI）
"""

from typing import Any, List, Optional, Sequence

from fastapi import FastAPI
import uvicorn

from agentrun.utils.log import logger

from .agui_protocol import AGUIProtocolHandler
from .invoker import AgentInvoker
from .model import ServerConfig
from .openai_protocol import OpenAIProtocolHandler
from .protocol import InvokeAgentHandler, ProtocolHandler


class AgentRunServer:
    """AgentRun HTTP Server / AgentRun HTTP 服务器

    基于 Router 的架构:
    - 每个协议提供完整的 FastAPI Router
    - Server 只负责组装和前缀管理
    - 易于扩展新协议

    Example (最简单用法):
        >>> def invoke_agent(request: AgentRequest):
        ...     return "Hello, world!"
        >>>
        >>> server = AgentRunServer(invoke_agent=invoke_agent)
        >>> server.start(port=8000)
        # 可访问:
        #   POST http://localhost:8000/openai/v1/chat/completions (OpenAI)
        #   POST http://localhost:8000/agui/v1/run (AG-UI)

    Example (流式输出):
        >>> async def invoke_agent(request: AgentRequest):
        ...     yield "Hello, "
        ...     yield "world!"
        >>>
        >>> server = AgentRunServer(invoke_agent=invoke_agent)
        >>> server.start(port=8000)

    Example (使用事件):
        >>> from agentrun.server import AgentResult, EventType
        >>>
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
        >>>
        >>> server = AgentRunServer(invoke_agent=invoke_agent)
        >>> server.start(port=8000)

    Example (仅 OpenAI 协议):
        >>> server = AgentRunServer(
        ...     invoke_agent=invoke_agent,
        ...     protocols=[OpenAIProtocolHandler()]
        ... )
        >>> server.start(port=8000)

    Example (集成到现有 FastAPI 应用):
        >>> from fastapi import FastAPI
        >>>
        >>> app = FastAPI()
        >>> agent_server = AgentRunServer(invoke_agent=invoke_agent)
        >>> app.mount("/agent", agent_server.as_fastapi_app())
        # 可访问: POST http://localhost:8000/agent/openai/v1/chat/completions

    Example (配置 CORS):
        >>> server = AgentRunServer(
        ...     invoke_agent=invoke_agent,
        ...     config=ServerConfig(cors_origins=["http://localhost:3000"])
        ... )
    """

    def __init__(
        self,
        invoke_agent: InvokeAgentHandler,
        protocols: Optional[List[ProtocolHandler]] = None,
        config: Optional[ServerConfig] = None,
    ):
        """初始化 AgentRun Server

        Args:
            invoke_agent: Agent 调用回调函数
                - 可以是同步或异步函数
                - 支持返回字符串或 AgentResult
                - 支持使用 yield 进行流式输出

            protocols: 协议处理器列表
                - 默认使用 OpenAI + AG-UI 协议
                - 可以添加自定义协议

            config: 服务器配置
                - cors_origins: CORS 允许的源列表
                - openai: OpenAI 协议配置
                - agui: AG-UI 协议配置
        """
        self.app = FastAPI(title="AgentRun Server")
        self.agent_invoker = AgentInvoker(invoke_agent)

        # 配置 CORS
        self._setup_cors(config.cors_origins if config else None)

        # 默认使用 OpenAI 和 AG-UI 协议
        if protocols is None:
            protocols = [
                OpenAIProtocolHandler(config),
                AGUIProtocolHandler(config),
            ]

        # 挂载所有协议的 Router
        self._mount_protocols(protocols)

    def _setup_cors(self, cors_origins: Optional[Sequence[str]] = None):
        """配置 CORS 中间件

        Args:
            cors_origins: 允许的源列表，默认为 ["*"] 允许所有源
        """
        if not cors_origins:
            return

        from fastapi.middleware.cors import CORSMiddleware

        origins = list(cors_origins) if cors_origins else ["*"]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        logger.debug(f"CORS 已启用，允许的源: {origins}")

    def _mount_protocols(self, protocols: List[ProtocolHandler]):
        """挂载所有协议的路由

        Args:
            protocols: 协议处理器列表
        """
        for protocol in protocols:
            # 获取协议的 Router
            router = protocol.as_fastapi_router(self.agent_invoker)

            # 使用协议定义的前缀
            prefix = protocol.get_prefix()

            # 挂载到主应用
            self.app.include_router(router, prefix=prefix)

            logger.debug(
                f"已挂载协议: {protocol.__class__.__name__} ->"
                f" {prefix or '(无前缀)'}"
            )

    def start(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        log_level: str = "info",
        **kwargs: Any,
    ):
        """启动 HTTP 服务器

        Args:
            host: 监听地址，默认 0.0.0.0
            port: 监听端口，默认 9000
            log_level: 日志级别，默认 info
            **kwargs: 传递给 uvicorn.run 的其他参数
        """
        logger.info(f"启动 AgentRun Server: http://{host}:{port}")

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
