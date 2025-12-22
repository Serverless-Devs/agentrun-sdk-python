"""Model Service 模型定义"""

from enum import Enum
from typing import Any, Dict, List, Optional

from agentrun.utils.model import BaseModel, NetworkConfig, PageableInput, Status


class BackendType(str, Enum):
    """后端类型"""

    PROXY = "proxy"
    """模型治理"""
    SERVICE = "service"
    """模型服务"""


class ModelType(str, Enum):
    """模型类型"""

    LLM = "llm"
    """大语言模型"""
    EMBEDDING = "text-embedding"
    """嵌入模型"""
    RERANK = "rerank"
    """重排序模型"""
    SPEECH2TEXT = "speech2text"
    TTS = "tts"
    MODERATION = "moderation"


class Provider(str, Enum):
    OpenAI = "openai"
    Anthropic = "anthropic"
    BaiChuan = "baichuan"
    DeepSeek = "deepseek"
    Gemini = "gemini"
    HunYuan = "hunyuan"
    MiniMax = "minimax"
    MoonShot = "moonshot"
    Spark = "spark"
    StepFun = "stepfun"
    Tongyi = "tongyi"
    VertexAI = "vertex_ai"
    WenXin = "wenxin"
    Yi = "yi"
    ZhiPuAI = "zhipuai"
    Custom = "custom"


class ProxyMode(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


class ProviderSettings(BaseModel):
    """提供商设置"""

    api_key: Optional[str] = None
    """API Key"""
    base_url: Optional[str] = None
    """基础 URL"""
    model_names: Optional[List[str]] = None
    """模型名称列表"""


class ModelFeatures(BaseModel):
    """模型特性"""

    agent_thought: Optional[bool] = None
    """是否支持智能体思维"""
    multi_tool_call: Optional[bool] = None
    """是否支持多工具调用"""
    stream_tool_call: Optional[bool] = None
    """是否支持流式工具调用"""
    tool_call: Optional[bool] = None
    """是否支持工具调用"""
    vision: Optional[bool] = None
    """是否支持视觉"""


class ModelProperties(BaseModel):
    """模型属性"""

    context_size: Optional[int] = None
    """上下文大小"""


class ModelParameterRule(BaseModel):
    """模型参数规则"""

    default: Optional[Any] = None
    max: Optional[float] = None
    min: Optional[float] = None
    name: Optional[str] = None
    required: Optional[bool] = None
    type: Optional[str] = None


class ModelInfoConfig(BaseModel):
    """模型信息配置"""

    model_name: Optional[str] = None
    """模型名称"""
    model_features: Optional[ModelFeatures] = None
    """模型特性"""
    model_properties: Optional[ModelProperties] = None
    """模型属性"""
    model_parameter_rules: Optional[List[ModelParameterRule]] = None
    """模型参数规则"""


class ProxyConfigEndpoint(BaseModel):
    base_url: Optional[str] = None
    model_names: Optional[List[str]] = None
    model_service_name: Optional[str] = None
    weight: Optional[str] = None


class ProxyConfigFallback(BaseModel):
    model_name: Optional[str] = None
    model_service_name: Optional[str] = None


class ProxyConfigTokenRateLimiter(BaseModel):
    tps: Optional[int] = None
    tpm: Optional[int] = None
    tph: Optional[int] = None
    tpd: Optional[int] = None


class ProxyConfigAIGuardrailConfig(BaseModel):
    """AI 防护配置"""

    check_request: Optional[bool] = None
    check_response: Optional[bool] = None


class ProxyConfigPolicies(BaseModel):
    cache: Optional[bool] = None
    concurrency_limit: Optional[int] = None
    fallbacks: Optional[List[ProxyConfigFallback]] = None
    num_retries: Optional[int] = None
    request_timeout: Optional[int] = None
    ai_guardrail_config: Optional[ProxyConfigAIGuardrailConfig] = None
    token_rate_limiter: Optional[ProxyConfigTokenRateLimiter] = None


class ProxyConfig(BaseModel):
    endpoints: Optional[List[ProxyConfigEndpoint]] = None
    """代理端点列表"""
    policies: Optional[ProxyConfigPolicies] = None


class CommonModelMutableProps(BaseModel):
    credential_name: Optional[str] = None
    description: Optional[str] = None
    network_configuration: Optional[NetworkConfig] = None


class CommonModelImmutableProps(BaseModel):
    model_type: Optional[ModelType] = None


class CommonModelSystemProps:
    created_at: Optional[str] = None
    last_updated_at: Optional[str] = None
    status: Optional[Status] = None


class ModelServiceMutableProps(CommonModelMutableProps):
    provider_settings: Optional[ProviderSettings] = None


class ModelServiceImmutableProps(CommonModelImmutableProps):
    model_info_configs: Optional[List[ModelInfoConfig]] = None
    model_service_name: Optional[str] = None
    provider: Optional[str] = None


class ModelServicesSystemProps(CommonModelSystemProps):
    model_service_id: Optional[str] = None


class ModelProxyMutableProps(CommonModelMutableProps):
    cpu: Optional[float] = 2
    litellm_version: Optional[str] = None
    memory: Optional[int] = 4096
    model_proxy_name: Optional[str] = None
    proxy_mode: Optional[ProxyMode] = None
    service_region_id: Optional[str] = None
    proxy_config: Optional[ProxyConfig] = None
    execution_role_arn: Optional[str] = None


class ModelProxyImmutableProps(CommonModelImmutableProps):
    pass


class ModelProxySystemProps(CommonModelSystemProps):
    endpoint: Optional[str] = None
    function_name: Optional[str] = None
    model_proxy_id: Optional[str] = None


class ModelServiceCreateInput(
    ModelServiceImmutableProps, ModelServiceMutableProps
):
    """模型服务创建输入参数"""

    pass


class ModelServiceUpdateInput(ModelServiceMutableProps):
    """模型服务更新输入参数"""

    pass


class ModelServiceListInput(PageableInput):
    model_type: Optional[ModelType] = None
    provider: Optional[str] = None


class ModelProxyCreateInput(ModelProxyMutableProps, ModelProxyImmutableProps):
    pass


class ModelProxyUpdateInput(ModelProxyMutableProps):
    pass


class ModelProxyListInput(PageableInput):
    proxy_mode: Optional[str] = None
    status: Optional[Status] = None
