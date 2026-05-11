"""
E2E 测试: Config.from_request_headers 通过真实 HTTP 请求提取临时凭证

测试覆盖:
- 通过 OpenAI 协议发送 x-fc-* header，invoke_agent 中使用 Config.from_request_headers 提取凭证
- 通过 AG-UI 协议发送 x-fc-* header，验证同样可以提取
- 部分 header 缺失时对应字段为空字符串
- 无 x-fc-* header 时所有字段为空字符串
"""

from agentrun.server import AgentRequest, AgentRunServer
from agentrun.utils.config import Config


def _make_client(invoke_agent):
    server = AgentRunServer(invoke_agent=invoke_agent)
    app = server.as_fastapi_app()
    from fastapi.testclient import TestClient

    return TestClient(app)


class TestHeaderCredentials:
    """通过真实 HTTP 请求验证 Config.from_request_headers 的 E2E 行为"""

    def test_openai_full_headers(self):
        """OpenAI 协议: 三个 x-fc-* header 都存在时正确提取"""
        captured = {}

        async def invoke_agent(request: AgentRequest):
            config = Config.from_request_headers(request)
            captured["ak"] = config.get_access_key_id()
            captured["sk"] = config.get_access_key_secret()
            captured["token"] = config.get_security_token()
            yield "ok"

        client = _make_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
            headers={
                "x-fc-access-key-id": "ak-e2e-test",
                "x-fc-access-key-secret": "sk-e2e-test",
                "x-fc-security-token": "token-e2e-test",
            },
        )
        assert response.status_code == 200
        assert captured["ak"] == "ak-e2e-test"
        assert captured["sk"] == "sk-e2e-test"
        assert captured["token"] == "token-e2e-test"

    def test_agui_full_headers(self):
        """AG-UI 协议: 三个 x-fc-* header 都存在时正确提取"""
        captured = {}

        async def invoke_agent(request: AgentRequest):
            config = Config.from_request_headers(request)
            captured["ak"] = config.get_access_key_id()
            captured["sk"] = config.get_access_key_secret()
            captured["token"] = config.get_security_token()
            yield "ok"

        client = _make_client(invoke_agent)
        response = client.post(
            "/ag-ui/agent",
            json={
                "messages": [{"role": "user", "content": "test"}],
            },
            headers={
                "x-fc-access-key-id": "ak-agui-test",
                "x-fc-access-key-secret": "sk-agui-test",
                "x-fc-security-token": "token-agui-test",
            },
        )
        assert response.status_code == 200
        assert captured["ak"] == "ak-agui-test"
        assert captured["sk"] == "sk-agui-test"
        assert captured["token"] == "token-agui-test"

    def test_partial_headers(self):
        """部分 x-fc-* header 缺失时，缺失字段为空字符串"""
        captured = {}

        async def invoke_agent(request: AgentRequest):
            config = Config.from_request_headers(request)
            captured["ak"] = config.get_access_key_id()
            captured["sk"] = config.get_access_key_secret()
            captured["token"] = config.get_security_token()
            yield "ok"

        client = _make_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
            headers={
                "x-fc-access-key-id": "ak-partial",
            },
        )
        assert response.status_code == 200
        assert captured["ak"] == "ak-partial"
        assert captured["sk"] == ""
        assert captured["token"] == ""

    def test_no_fc_headers(self):
        """无 x-fc-* header 时所有字段为空字符串"""
        captured = {}

        async def invoke_agent(request: AgentRequest):
            config = Config.from_request_headers(request)
            captured["ak"] = config.get_access_key_id()
            captured["sk"] = config.get_access_key_secret()
            captured["token"] = config.get_security_token()
            yield "ok"

        client = _make_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert captured["ak"] == ""
        assert captured["sk"] == ""
        assert captured["token"] == ""

    def test_config_merge_with_base(self):
        """从 header 提取的 Config 与 base Config 合并时 header 凭证优先"""
        captured = {}

        async def invoke_agent(request: AgentRequest):
            base = Config(
                access_key_id="base-ak",
                access_key_secret="base-sk",
                security_token="base-token",
            )
            header_config = Config.from_request_headers(request)
            merged = Config.with_configs(base, header_config)
            captured["ak"] = merged.get_access_key_id()
            captured["sk"] = merged.get_access_key_secret()
            captured["token"] = merged.get_security_token()
            yield "ok"

        client = _make_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
            headers={
                "x-fc-access-key-id": "header-ak",
                "x-fc-access-key-secret": "header-sk",
                "x-fc-security-token": "header-token",
            },
        )
        assert response.status_code == 200
        assert captured["ak"] == "header-ak"
        assert captured["sk"] == "header-sk"
        assert captured["token"] == "header-token"
