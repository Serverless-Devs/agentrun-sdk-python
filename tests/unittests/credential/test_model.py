"""测试 agentrun.credential.model 模块 / Test agentrun.credential.model module"""

import pytest

from agentrun.credential.model import (
    CredentialAuthType,
    CredentialBasicAuth,
    CredentialConfig,
    CredentialConfigInner,
    CredentialCreateInput,
    CredentialImmutableProps,
    CredentialListInput,
    CredentialListOutput,
    CredentialMutableProps,
    CredentialSourceType,
    CredentialSystemProps,
    CredentialUpdateInput,
    RelatedResource,
)


class TestCredentialAuthType:
    """测试 CredentialAuthType 枚举"""

    def test_jwt(self):
        assert CredentialAuthType.JWT.value == "jwt"

    def test_api_key(self):
        assert CredentialAuthType.API_KEY.value == "api_key"

    def test_basic(self):
        assert CredentialAuthType.BASIC.value == "basic"

    def test_aksk(self):
        assert CredentialAuthType.AKSK.value == "ak_sk"

    def test_custom_header(self):
        assert CredentialAuthType.CUSTOM_HEADER.value == "custom_header"


class TestCredentialSourceType:
    """测试 CredentialSourceType 枚举"""

    def test_llm(self):
        assert CredentialSourceType.LLM.value == "external_llm"

    def test_tool(self):
        assert CredentialSourceType.TOOL.value == "external_tool"

    def test_internal(self):
        assert CredentialSourceType.INTERNAL.value == "internal"


class TestCredentialBasicAuth:
    """测试 CredentialBasicAuth 模型"""

    def test_basic_auth(self):
        auth = CredentialBasicAuth(username="user", password="pass")
        assert auth.username == "user"
        assert auth.password == "pass"


class TestRelatedResource:
    """测试 RelatedResource 模型"""

    def test_related_resource(self):
        resource = RelatedResource(
            resource_id="res-123",
            resource_name="test-resource",
            resource_type="AgentRuntime",
        )
        assert resource.resource_id == "res-123"
        assert resource.resource_name == "test-resource"
        assert resource.resource_type == "AgentRuntime"

    def test_related_resource_defaults(self):
        resource = RelatedResource()
        assert resource.resource_id is None
        assert resource.resource_name is None
        assert resource.resource_type is None


class TestCredentialConfigInner:
    """测试 CredentialConfigInner 模型"""

    def test_config_inner(self):
        config = CredentialConfigInner(
            credential_auth_type=CredentialAuthType.API_KEY,
            credential_source_type=CredentialSourceType.LLM,
            credential_public_config={"provider": "openai"},
            credential_secret="sk-xxx",
        )
        assert config.credential_auth_type == CredentialAuthType.API_KEY
        assert config.credential_source_type == CredentialSourceType.LLM
        assert config.credential_public_config == {"provider": "openai"}
        assert config.credential_secret == "sk-xxx"


class TestCredentialConfig:
    """测试 CredentialConfig 类的工厂方法"""

    def test_inbound_api_key(self):
        """测试 inbound_api_key 工厂方法"""
        config = CredentialConfig.inbound_api_key("my-api-key")
        assert config.credential_source_type == CredentialSourceType.INTERNAL
        assert config.credential_auth_type == CredentialAuthType.API_KEY
        assert config.credential_public_config == {"headerKey": "Authorization"}
        assert config.credential_secret == "my-api-key"

    def test_inbound_api_key_custom_header(self):
        """测试 inbound_api_key 自定义 header"""
        config = CredentialConfig.inbound_api_key(
            "my-api-key", header_key="X-API-Key"
        )
        assert config.credential_public_config == {"headerKey": "X-API-Key"}

    def test_inbound_static_jwt(self):
        """测试 inbound_static_jwt 工厂方法"""
        config = CredentialConfig.inbound_static_jwt("jwks-content")
        assert config.credential_source_type == CredentialSourceType.INTERNAL
        assert config.credential_auth_type == CredentialAuthType.JWT
        assert config.credential_public_config["authType"] == "static_jwks"
        assert config.credential_public_config["jwks"] == "jwks-content"

    def test_inbound_remote_jwt(self):
        """测试 inbound_remote_jwt 工厂方法"""
        config = CredentialConfig.inbound_remote_jwt(
            uri="https://example.com/.well-known/jwks.json",
            timeout=5000,
            ttl=60000,
            extra_param="value",
        )
        assert config.credential_source_type == CredentialSourceType.INTERNAL
        assert config.credential_auth_type == CredentialAuthType.JWT
        assert (
            config.credential_public_config["uri"]
            == "https://example.com/.well-known/jwks.json"
        )
        assert config.credential_public_config["timeout"] == 5000
        assert config.credential_public_config["ttl"] == 60000
        assert config.credential_public_config["extra_param"] == "value"

    def test_inbound_basic(self):
        """测试 inbound_basic 工厂方法"""
        users = [
            CredentialBasicAuth(username="user1", password="pass1"),
            CredentialBasicAuth(username="user2", password="pass2"),
        ]
        config = CredentialConfig.inbound_basic(users)
        assert config.credential_source_type == CredentialSourceType.INTERNAL
        assert config.credential_auth_type == CredentialAuthType.BASIC
        assert len(config.credential_public_config["users"]) == 2

    def test_outbound_llm_api_key(self):
        """测试 outbound_llm_api_key 工厂方法"""
        config = CredentialConfig.outbound_llm_api_key(
            api_key="sk-xxx", provider="openai"
        )
        assert config.credential_source_type == CredentialSourceType.LLM
        assert config.credential_auth_type == CredentialAuthType.API_KEY
        assert config.credential_public_config == {"provider": "openai"}
        assert config.credential_secret == "sk-xxx"

    def test_outbound_tool_api_key(self):
        """测试 outbound_tool_api_key 工厂方法"""
        config = CredentialConfig.outbound_tool_api_key(api_key="tool-key")
        assert config.credential_source_type == CredentialSourceType.TOOL
        assert config.credential_auth_type == CredentialAuthType.API_KEY
        assert config.credential_public_config == {}
        assert config.credential_secret == "tool-key"

    def test_outbound_tool_ak_sk(self):
        """测试 outbound_tool_ak_sk 工厂方法"""
        config = CredentialConfig.outbound_tool_ak_sk(
            provider="aliyun",
            access_key_id="ak-id",
            access_key_secret="ak-secret",
            account_id="account-123",
        )
        assert config.credential_source_type == CredentialSourceType.TOOL
        assert config.credential_auth_type == CredentialAuthType.AKSK
        assert config.credential_public_config["provider"] == "aliyun"
        assert (
            config.credential_public_config["authConfig"]["accessKey"]
            == "ak-id"
        )
        assert (
            config.credential_public_config["authConfig"]["accountId"]
            == "account-123"
        )
        assert config.credential_secret == "ak-secret"

    def test_outbound_tool_ak_sk_custom(self):
        """测试 outbound_tool_ak_sk_custom 工厂方法"""
        auth_config = {"key1": "value1", "key2": "value2"}
        config = CredentialConfig.outbound_tool_ak_sk_custom(auth_config)
        assert config.credential_source_type == CredentialSourceType.TOOL
        assert config.credential_auth_type == CredentialAuthType.AKSK
        assert config.credential_public_config["provider"] == "custom"
        assert config.credential_public_config["authConfig"] == auth_config

    def test_outbound_tool_custom_header(self):
        """测试 outbound_tool_custom_header 工厂方法"""
        headers = {"X-Custom-1": "value1", "X-Custom-2": "value2"}
        config = CredentialConfig.outbound_tool_custom_header(headers)
        assert config.credential_source_type == CredentialSourceType.TOOL
        assert config.credential_auth_type == CredentialAuthType.CUSTOM_HEADER
        assert config.credential_public_config["authConfig"] == headers


class TestCredentialMutableProps:
    """测试 CredentialMutableProps 模型"""

    def test_mutable_props(self):
        props = CredentialMutableProps(
            description="Test description", enabled=True
        )
        assert props.description == "Test description"
        assert props.enabled is True

    def test_mutable_props_defaults(self):
        props = CredentialMutableProps()
        assert props.description is None
        assert props.enabled is None


class TestCredentialImmutableProps:
    """测试 CredentialImmutableProps 模型"""

    def test_immutable_props(self):
        props = CredentialImmutableProps(credential_name="my-credential")
        assert props.credential_name == "my-credential"


class TestCredentialSystemProps:
    """测试 CredentialSystemProps 模型"""

    def test_system_props(self):
        props = CredentialSystemProps(
            credential_id="cred-123",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
            related_resources=[
                RelatedResource(resource_id="res-1", resource_type="Agent")
            ],
        )
        assert props.credential_id == "cred-123"
        assert props.created_at == "2024-01-01T00:00:00Z"
        assert props.updated_at == "2024-01-02T00:00:00Z"
        assert len(props.related_resources) == 1


class TestCredentialCreateInput:
    """测试 CredentialCreateInput 模型"""

    def test_create_input(self):
        config = CredentialConfig.outbound_llm_api_key("sk-xxx", "openai")
        input_obj = CredentialCreateInput(
            credential_name="my-cred",
            description="Test credential",
            enabled=True,
            credential_config=config,
        )
        assert input_obj.credential_name == "my-cred"
        assert input_obj.description == "Test credential"
        assert input_obj.enabled is True
        assert input_obj.credential_config == config


class TestCredentialUpdateInput:
    """测试 CredentialUpdateInput 模型"""

    def test_update_input(self):
        input_obj = CredentialUpdateInput(
            description="Updated description", enabled=False
        )
        assert input_obj.description == "Updated description"
        assert input_obj.enabled is False

    def test_update_input_with_config(self):
        config = CredentialConfig.outbound_llm_api_key("new-key", "openai")
        input_obj = CredentialUpdateInput(credential_config=config)
        assert input_obj.credential_config == config


class TestCredentialListInput:
    """测试 CredentialListInput 模型"""

    def test_list_input(self):
        input_obj = CredentialListInput(
            page_number=1,
            page_size=20,
            credential_auth_type=CredentialAuthType.API_KEY,
            credential_name="test",
            credential_source_type=CredentialSourceType.LLM,
            provider="openai",
        )
        assert input_obj.page_number == 1
        assert input_obj.page_size == 20
        assert input_obj.credential_auth_type == CredentialAuthType.API_KEY
        assert input_obj.credential_name == "test"
        assert input_obj.credential_source_type == CredentialSourceType.LLM
        assert input_obj.provider == "openai"


class TestCredentialListOutput:
    """测试 CredentialListOutput 模型"""

    def test_list_output(self):
        output = CredentialListOutput(
            credential_id="cred-123",
            credential_name="my-cred",
            credential_auth_type="api_key",
            credential_source_type="external_llm",
            enabled=True,
            related_resource_count=3,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
        )
        assert output.credential_id == "cred-123"
        assert output.credential_name == "my-cred"
        assert output.credential_auth_type == "api_key"
        assert output.enabled is True
        assert output.related_resource_count == 3
