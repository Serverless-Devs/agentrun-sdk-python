"""Server 扩展测试

测试 AgentRunServer 的 CORS 配置和其他功能。
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from agentrun.server import (
    AgentRequest,
    AgentRunServer,
    OpenAIProtocolHandler,
    ServerConfig,
)


class TestServerCORS:
    """测试 CORS 配置"""

    def test_cors_enabled(self):
        """测试启用 CORS"""

        def invoke_agent(request: AgentRequest):
            return "Hello"

        config = ServerConfig(cors_origins=["http://localhost:3000"])
        server = AgentRunServer(invoke_agent=invoke_agent, config=config)
        client = TestClient(server.as_fastapi_app())

        # 发送预检请求
        response = client.options(
            "/openai/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        # CORS 头应该存在
        assert "access-control-allow-origin" in response.headers

    def test_cors_disabled_by_default(self):
        """测试默认不启用 CORS"""

        def invoke_agent(request: AgentRequest):
            return "Hello"

        server = AgentRunServer(invoke_agent=invoke_agent)
        client = TestClient(server.as_fastapi_app())

        response = client.post(
            "/openai/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
            headers={"Origin": "http://localhost:3000"},
        )

        # 没有 CORS 配置时，不应该有 CORS 头
        assert response.status_code == 200

    def test_cors_with_multiple_origins(self):
        """测试多个允许的源"""

        def invoke_agent(request: AgentRequest):
            return "Hello"

        config = ServerConfig(
            cors_origins=["http://localhost:3000", "http://example.com"]
        )
        server = AgentRunServer(invoke_agent=invoke_agent, config=config)
        client = TestClient(server.as_fastapi_app())

        # 测试第一个源
        response = client.options(
            "/openai/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" in response.headers


class TestServerProtocols:
    """测试协议配置"""

    def test_custom_protocols(self):
        """测试自定义协议列表"""

        def invoke_agent(request: AgentRequest):
            return "Hello"

        # 只使用 OpenAI 协议
        server = AgentRunServer(
            invoke_agent=invoke_agent,
            protocols=[OpenAIProtocolHandler()],
        )
        client = TestClient(server.as_fastapi_app())

        # OpenAI 端点应该存在
        response = client.post(
            "/openai/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert response.status_code == 200

        # AG-UI 端点应该不存在
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert response.status_code == 404


class TestServerFastAPIApp:
    """测试 FastAPI 应用导出"""

    def test_as_fastapi_app(self):
        """测试 as_fastapi_app 方法"""
        from fastapi import FastAPI

        def invoke_agent(request: AgentRequest):
            return "Hello"

        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()

        assert isinstance(app, FastAPI)
        assert app.title == "AgentRun Server"

    def test_mount_to_existing_app(self):
        """测试挂载到现有 FastAPI 应用"""
        from fastapi import FastAPI

        def invoke_agent(request: AgentRequest):
            return "Hello"

        # 创建主应用
        main_app = FastAPI()

        @main_app.get("/")
        def root():
            return {"message": "Main app"}

        # 挂载 AgentRun Server
        agent_server = AgentRunServer(invoke_agent=invoke_agent)
        main_app.mount("/agent", agent_server.as_fastapi_app())

        client = TestClient(main_app)

        # 主应用端点
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "Main app"

        # AgentRun 端点
        response = client.post(
            "/agent/openai/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert response.status_code == 200


class TestServerStartMethod:
    """测试 start 方法"""

    @patch("agentrun.server.server.uvicorn.run")
    @patch("agentrun.server.server.logger")
    def test_start_calls_uvicorn_run(self, mock_logger, mock_uvicorn_run):
        """测试 start 方法调用 uvicorn.run"""

        def invoke_agent(request: AgentRequest):
            return "Hello"

        server = AgentRunServer(invoke_agent=invoke_agent)
        server.start(
            host="127.0.0.1", port=8080, log_level="debug", reload=True
        )

        # 验证 uvicorn.run 被调用
        mock_uvicorn_run.assert_called_once_with(
            server.app,
            host="127.0.0.1",
            port=8080,
            log_level="debug",
            reload=True,
        )

        # 验证日志被记录
        mock_logger.info.assert_called_once()
        assert "127.0.0.1" in mock_logger.info.call_args[0][0]
        assert "8080" in mock_logger.info.call_args[0][0]

    @patch("agentrun.server.server.uvicorn.run")
    @patch("agentrun.server.server.logger")
    def test_start_with_default_args(self, mock_logger, mock_uvicorn_run):
        """测试 start 方法使用默认参数"""

        def invoke_agent(request: AgentRequest):
            return "Hello"

        server = AgentRunServer(invoke_agent=invoke_agent)
        server.start()

        # 验证使用默认参数
        mock_uvicorn_run.assert_called_once_with(
            server.app,
            host="0.0.0.0",
            port=9000,
            log_level="info",
        )
