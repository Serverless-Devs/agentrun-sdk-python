"""Config.from_request_headers 单元测试"""

from agentrun.utils.config import Config


class FakeRequest:
    """模拟 Starlette Request"""

    def __init__(self, headers: dict):
        self.headers = headers


class FakeAgentRequest:
    """模拟 AgentRequest（含 raw_request）"""

    def __init__(self, headers: dict):
        self.raw_request = FakeRequest(headers)


class TestFromRequestHeaders:

    def test_full_headers(self):
        request = FakeRequest({
            "x-fc-access-key-id": "ak-123",
            "x-fc-access-key-secret": "sk-456",
            "x-fc-security-token": "token-789",
        })
        config = Config.from_request_headers(request)

        assert config.get_access_key_id() == "ak-123"
        assert config.get_access_key_secret() == "sk-456"
        assert config.get_security_token() == "token-789"

    def test_partial_headers_missing_token(self):
        request = FakeRequest({
            "x-fc-access-key-id": "ak-123",
            "x-fc-access-key-secret": "sk-456",
        })
        config = Config.from_request_headers(request)

        assert config.get_access_key_id() == "ak-123"
        assert config.get_access_key_secret() == "sk-456"
        assert config.get_security_token() == ""

    def test_partial_headers_only_key_id(self):
        request = FakeRequest({
            "x-fc-access-key-id": "ak-only",
        })
        config = Config.from_request_headers(request)

        assert config.get_access_key_id() == "ak-only"
        assert config.get_access_key_secret() == ""
        assert config.get_security_token() == ""

    def test_empty_headers(self):
        request = FakeRequest({})
        config = Config.from_request_headers(request)

        assert config.get_access_key_id() == ""
        assert config.get_access_key_secret() == ""
        assert config.get_security_token() == ""

    def test_agent_request_unwrap(self):
        agent_req = FakeAgentRequest({
            "x-fc-access-key-id": "ak-from-agent",
            "x-fc-access-key-secret": "sk-from-agent",
            "x-fc-security-token": "token-from-agent",
        })
        config = Config.from_request_headers(agent_req)

        assert config.get_access_key_id() == "ak-from-agent"
        assert config.get_access_key_secret() == "sk-from-agent"
        assert config.get_security_token() == "token-from-agent"

    def test_config_usable_as_resource_param(self):
        config = Config.from_request_headers(FakeRequest({
            "x-fc-access-key-id": "ak-new",
            "x-fc-access-key-secret": "sk-new",
            "x-fc-security-token": "token-new",
        }))
        assert config.get_access_key_id() == "ak-new"
        assert config.get_access_key_secret() == "sk-new"
        assert config.get_security_token() == "token-new"
        assert config.get_region_id() == "cn-hangzhou"

    def test_agent_request_with_none_raw_request(self):
        agent_req = FakeAgentRequest({})
        agent_req.raw_request = None
        config = Config.from_request_headers(agent_req)
        assert config.get_access_key_id() == ""
        assert config.get_access_key_secret() == ""
        assert config.get_security_token() == ""

    def test_extra_headers_ignored(self):
        request = FakeRequest({
            "x-fc-access-key-id": "ak-123",
            "x-fc-access-key-secret": "sk-456",
            "x-fc-security-token": "token-789",
            "authorization": "Bearer xxx",
            "content-type": "application/json",
        })
        config = Config.from_request_headers(request)
        assert config.get_access_key_id() == "ak-123"
        assert config.get_access_key_secret() == "sk-456"
        assert config.get_security_token() == "token-789"
