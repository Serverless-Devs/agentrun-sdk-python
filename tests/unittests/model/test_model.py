"""Tests for agentrun/model/model.py"""

import pytest

from agentrun.model.model import (
    BackendType,
    CommonModelImmutableProps,
    CommonModelMutableProps,
    CommonModelSystemProps,
    ModelFeatures,
    ModelInfoConfig,
    ModelParameterRule,
    ModelProperties,
    ModelProxyCreateInput,
    ModelProxyImmutableProps,
    ModelProxyListInput,
    ModelProxyMutableProps,
    ModelProxySystemProps,
    ModelProxyUpdateInput,
    ModelServiceCreateInput,
    ModelServiceImmutableProps,
    ModelServiceListInput,
    ModelServiceMutableProps,
    ModelServicesSystemProps,
    ModelServiceUpdateInput,
    ModelType,
    Provider,
    ProviderSettings,
    ProxyConfig,
    ProxyConfigAIGuardrailConfig,
    ProxyConfigEndpoint,
    ProxyConfigFallback,
    ProxyConfigPolicies,
    ProxyConfigTokenRateLimiter,
    ProxyMode,
)
from agentrun.utils.model import NetworkConfig, Status


class TestBackendType:
    """Tests for BackendType enum"""

    def test_backend_type_values(self):
        assert BackendType.PROXY == "proxy"
        assert BackendType.SERVICE == "service"

    def test_backend_type_is_str(self):
        assert isinstance(BackendType.PROXY, str)
        assert isinstance(BackendType.SERVICE, str)


class TestModelType:
    """Tests for ModelType enum"""

    def test_model_type_values(self):
        assert ModelType.LLM == "llm"
        assert ModelType.EMBEDDING == "text-embedding"
        assert ModelType.RERANK == "rerank"
        assert ModelType.SPEECH2TEXT == "speech2text"
        assert ModelType.TTS == "tts"
        assert ModelType.MODERATION == "moderation"


class TestProvider:
    """Tests for Provider enum"""

    def test_provider_values(self):
        assert Provider.OpenAI == "openai"
        assert Provider.Anthropic == "anthropic"
        assert Provider.DeepSeek == "deepseek"
        assert Provider.Tongyi == "tongyi"
        assert Provider.Custom == "custom"


class TestProxyMode:
    """Tests for ProxyMode enum"""

    def test_proxy_mode_values(self):
        assert ProxyMode.SINGLE == "single"
        assert ProxyMode.MULTI == "multi"


class TestProviderSettings:
    """Tests for ProviderSettings model"""

    def test_default_values(self):
        settings = ProviderSettings()
        assert settings.api_key is None
        assert settings.base_url is None
        assert settings.model_names is None

    def test_with_values(self):
        settings = ProviderSettings(
            api_key="test-key",
            base_url="https://api.example.com",
            model_names=["model1", "model2"],
        )
        assert settings.api_key == "test-key"
        assert settings.base_url == "https://api.example.com"
        assert settings.model_names == ["model1", "model2"]


class TestModelFeatures:
    """Tests for ModelFeatures model"""

    def test_default_values(self):
        features = ModelFeatures()
        assert features.agent_thought is None
        assert features.multi_tool_call is None
        assert features.stream_tool_call is None
        assert features.tool_call is None
        assert features.vision is None

    def test_with_values(self):
        features = ModelFeatures(
            agent_thought=True,
            multi_tool_call=True,
            stream_tool_call=False,
            tool_call=True,
            vision=True,
        )
        assert features.agent_thought is True
        assert features.multi_tool_call is True
        assert features.stream_tool_call is False
        assert features.tool_call is True
        assert features.vision is True


class TestModelProperties:
    """Tests for ModelProperties model"""

    def test_default_values(self):
        props = ModelProperties()
        assert props.context_size is None

    def test_with_value(self):
        props = ModelProperties(context_size=128000)
        assert props.context_size == 128000


class TestModelParameterRule:
    """Tests for ModelParameterRule model"""

    def test_default_values(self):
        rule = ModelParameterRule()
        assert rule.default is None
        assert rule.max is None
        assert rule.min is None
        assert rule.name is None
        assert rule.required is None
        assert rule.type is None

    def test_with_values(self):
        rule = ModelParameterRule(
            default=0.7,
            max=2.0,
            min=0.0,
            name="temperature",
            required=False,
            type="float",
        )
        assert rule.default == 0.7
        assert rule.max == 2.0
        assert rule.min == 0.0
        assert rule.name == "temperature"
        assert rule.required is False
        assert rule.type == "float"


class TestModelInfoConfig:
    """Tests for ModelInfoConfig model"""

    def test_default_values(self):
        config = ModelInfoConfig()
        assert config.model_name is None
        assert config.model_features is None
        assert config.model_properties is None
        assert config.model_parameter_rules is None

    def test_with_nested_values(self):
        config = ModelInfoConfig(
            model_name="gpt-4",
            model_features=ModelFeatures(tool_call=True),
            model_properties=ModelProperties(context_size=128000),
            model_parameter_rules=[
                ModelParameterRule(name="temperature", default=1.0)
            ],
        )
        assert config.model_name == "gpt-4"
        assert config.model_features is not None
        assert config.model_features.tool_call is True
        assert config.model_properties is not None
        assert config.model_properties.context_size == 128000
        assert config.model_parameter_rules is not None
        assert len(config.model_parameter_rules) == 1


class TestProxyConfigEndpoint:
    """Tests for ProxyConfigEndpoint model"""

    def test_default_values(self):
        endpoint = ProxyConfigEndpoint()
        assert endpoint.base_url is None
        assert endpoint.model_names is None
        assert endpoint.model_service_name is None
        assert endpoint.weight is None

    def test_with_values(self):
        endpoint = ProxyConfigEndpoint(
            base_url="https://api.example.com",
            model_names=["model1"],
            model_service_name="service1",
            weight=100,
        )
        assert endpoint.base_url == "https://api.example.com"
        assert endpoint.model_names == ["model1"]
        assert endpoint.model_service_name == "service1"
        assert endpoint.weight == 100


class TestProxyConfigFallback:
    """Tests for ProxyConfigFallback model"""

    def test_default_values(self):
        fallback = ProxyConfigFallback()
        assert fallback.model_name is None
        assert fallback.model_service_name is None

    def test_with_values(self):
        fallback = ProxyConfigFallback(
            model_name="fallback-model",
            model_service_name="fallback-service",
        )
        assert fallback.model_name == "fallback-model"
        assert fallback.model_service_name == "fallback-service"


class TestProxyConfigTokenRateLimiter:
    """Tests for ProxyConfigTokenRateLimiter model"""

    def test_default_values(self):
        limiter = ProxyConfigTokenRateLimiter()
        assert limiter.tps is None
        assert limiter.tpm is None
        assert limiter.tph is None
        assert limiter.tpd is None

    def test_with_values(self):
        limiter = ProxyConfigTokenRateLimiter(
            tps=10,
            tpm=100,
            tph=1000,
            tpd=10000,
        )
        assert limiter.tps == 10
        assert limiter.tpm == 100
        assert limiter.tph == 1000
        assert limiter.tpd == 10000


class TestProxyConfigAIGuardrailConfig:
    """Tests for ProxyConfigAIGuardrailConfig model"""

    def test_default_values(self):
        config = ProxyConfigAIGuardrailConfig()
        assert config.check_request is None
        assert config.check_response is None

    def test_with_values(self):
        config = ProxyConfigAIGuardrailConfig(
            check_request=True,
            check_response=False,
        )
        assert config.check_request is True
        assert config.check_response is False


class TestProxyConfigPolicies:
    """Tests for ProxyConfigPolicies model"""

    def test_default_values(self):
        policies = ProxyConfigPolicies()
        assert policies.cache is None
        assert policies.concurrency_limit is None
        assert policies.fallbacks is None
        assert policies.num_retries is None
        assert policies.request_timeout is None
        assert policies.ai_guardrail_config is None
        assert policies.token_rate_limiter is None

    def test_with_values(self):
        policies = ProxyConfigPolicies(
            cache=True,
            concurrency_limit=10,
            fallbacks=[ProxyConfigFallback(model_name="fallback")],
            num_retries=3,
            request_timeout=30,
            ai_guardrail_config=ProxyConfigAIGuardrailConfig(
                check_request=True
            ),
            token_rate_limiter=ProxyConfigTokenRateLimiter(tpm=100),
        )
        assert policies.cache is True
        assert policies.concurrency_limit == 10
        assert policies.fallbacks is not None
        assert len(policies.fallbacks) == 1
        assert policies.num_retries == 3
        assert policies.request_timeout == 30
        assert policies.ai_guardrail_config is not None
        assert policies.token_rate_limiter is not None


class TestProxyConfig:
    """Tests for ProxyConfig model"""

    def test_default_values(self):
        config = ProxyConfig()
        assert config.endpoints is None
        assert config.policies is None

    def test_with_values(self):
        config = ProxyConfig(
            endpoints=[ProxyConfigEndpoint(base_url="https://api.example.com")],
            policies=ProxyConfigPolicies(cache=True),
        )
        assert config.endpoints is not None
        assert len(config.endpoints) == 1
        assert config.policies is not None
        assert config.policies.cache is True


class TestCommonModelProps:
    """Tests for common model property classes"""

    def test_common_mutable_props(self):
        props = CommonModelMutableProps(
            credential_name="test-cred",
            description="Test description",
            network_configuration=NetworkConfig(),
        )
        assert props.credential_name == "test-cred"
        assert props.description == "Test description"
        assert props.network_configuration is not None

    def test_common_immutable_props(self):
        props = CommonModelImmutableProps(model_type=ModelType.LLM)
        assert props.model_type == ModelType.LLM

    def test_common_system_props(self):
        props = CommonModelSystemProps()
        props.created_at = "2024-01-01T00:00:00Z"
        props.last_updated_at = "2024-01-02T00:00:00Z"
        props.status = Status.READY
        assert props.created_at == "2024-01-01T00:00:00Z"
        assert props.last_updated_at == "2024-01-02T00:00:00Z"
        assert props.status == Status.READY


class TestModelServiceProps:
    """Tests for ModelService property classes"""

    def test_model_service_mutable_props(self):
        props = ModelServiceMutableProps(
            credential_name="cred",
            provider_settings=ProviderSettings(api_key="key"),
        )
        assert props.credential_name == "cred"
        assert props.provider_settings is not None
        assert props.provider_settings.api_key == "key"

    def test_model_service_immutable_props(self):
        props = ModelServiceImmutableProps(
            model_service_name="test-service",
            provider="openai",
            model_info_configs=[ModelInfoConfig(model_name="gpt-4")],
        )
        assert props.model_service_name == "test-service"
        assert props.provider == "openai"
        assert props.model_info_configs is not None
        assert len(props.model_info_configs) == 1

    def test_model_services_system_props(self):
        props = ModelServicesSystemProps()
        props.model_service_id = "service-123"
        assert props.model_service_id == "service-123"


class TestModelProxyProps:
    """Tests for ModelProxy property classes"""

    def test_model_proxy_mutable_props_defaults(self):
        props = ModelProxyMutableProps()
        assert props.cpu == 2
        assert props.memory == 4096
        assert props.litellm_version is None
        assert props.model_proxy_name is None
        assert props.proxy_mode is None
        assert props.service_region_id is None
        assert props.proxy_config is None
        assert props.execution_role_arn is None

    def test_model_proxy_mutable_props_with_values(self):
        props = ModelProxyMutableProps(
            cpu=4,
            memory=8192,
            model_proxy_name="test-proxy",
            proxy_mode=ProxyMode.SINGLE,
            proxy_config=ProxyConfig(
                endpoints=[
                    ProxyConfigEndpoint(base_url="https://api.example.com")
                ]
            ),
        )
        assert props.cpu == 4
        assert props.memory == 8192
        assert props.model_proxy_name == "test-proxy"
        assert props.proxy_mode == ProxyMode.SINGLE
        assert props.proxy_config is not None

    def test_model_proxy_immutable_props(self):
        props = ModelProxyImmutableProps()
        # ModelProxyImmutableProps inherits from CommonModelImmutableProps
        assert props.model_type is None

    def test_model_proxy_system_props(self):
        props = ModelProxySystemProps()
        props.endpoint = "https://proxy.example.com"
        props.function_name = "test-function"
        props.model_proxy_id = "proxy-123"
        assert props.endpoint == "https://proxy.example.com"
        assert props.function_name == "test-function"
        assert props.model_proxy_id == "proxy-123"


class TestModelServiceInputs:
    """Tests for ModelService input classes"""

    def test_model_service_create_input(self):
        input_obj = ModelServiceCreateInput(
            model_service_name="test-service",
            provider="openai",
            provider_settings=ProviderSettings(
                api_key="test-key",
                base_url="https://api.openai.com",
            ),
        )
        assert input_obj.model_service_name == "test-service"
        assert input_obj.provider == "openai"
        assert input_obj.provider_settings is not None

    def test_model_service_update_input(self):
        input_obj = ModelServiceUpdateInput(
            description="Updated description",
            provider_settings=ProviderSettings(api_key="new-key"),
        )
        assert input_obj.description == "Updated description"
        assert input_obj.provider_settings is not None

    def test_model_service_list_input(self):
        input_obj = ModelServiceListInput(
            model_type=ModelType.LLM,
            provider="openai",
            page_number=1,
            page_size=10,
        )
        assert input_obj.model_type == ModelType.LLM
        assert input_obj.provider == "openai"
        assert input_obj.page_number == 1
        assert input_obj.page_size == 10


class TestModelProxyInputs:
    """Tests for ModelProxy input classes"""

    def test_model_proxy_create_input(self):
        input_obj = ModelProxyCreateInput(
            model_proxy_name="test-proxy",
            proxy_mode=ProxyMode.SINGLE,
            proxy_config=ProxyConfig(
                endpoints=[
                    ProxyConfigEndpoint(
                        model_service_name="test-service",
                        model_names=["gpt-4"],
                    )
                ]
            ),
        )
        assert input_obj.model_proxy_name == "test-proxy"
        assert input_obj.proxy_mode == ProxyMode.SINGLE
        assert input_obj.proxy_config is not None

    def test_model_proxy_update_input(self):
        input_obj = ModelProxyUpdateInput(
            description="Updated proxy",
            cpu=4,
            memory=8192,
        )
        assert input_obj.description == "Updated proxy"
        assert input_obj.cpu == 4
        assert input_obj.memory == 8192

    def test_model_proxy_list_input(self):
        input_obj = ModelProxyListInput(
            proxy_mode="single",
            status=Status.READY,
            page_number=1,
            page_size=20,
        )
        assert input_obj.proxy_mode == "single"
        assert input_obj.status == Status.READY
        assert input_obj.page_number == 1
        assert input_obj.page_size == 20


class TestModelDump:
    """Tests for model serialization"""

    def test_model_service_create_input_dump(self):
        input_obj = ModelServiceCreateInput(
            model_service_name="test-service",
            provider="openai",
            model_type=ModelType.LLM,
        )
        dumped = input_obj.model_dump()
        # BaseModel uses camelCase by default
        assert "modelServiceName" in dumped
        assert dumped["modelServiceName"] == "test-service"
        assert dumped["provider"] == "openai"

    def test_model_proxy_create_input_dump(self):
        input_obj = ModelProxyCreateInput(
            model_proxy_name="test-proxy",
            proxy_mode=ProxyMode.SINGLE,
        )
        dumped = input_obj.model_dump()
        # BaseModel uses camelCase by default
        assert "modelProxyName" in dumped
        assert dumped["modelProxyName"] == "test-proxy"
        assert dumped["proxyMode"] == "single"
