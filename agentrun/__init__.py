"""AgentRun Python SDK / AgentRun Python SDK

AgentRun Python SDK 是阿里云 AgentRun 服务的 Python 客户端库。
AgentRun Python SDK is the Python client library for Alibaba Cloud AgentRun service.

提供简洁易用的 API 来管理 AI Agent 运行时环境、模型服务、沙箱环境等。
Provides simple and easy-to-use APIs for managing AI Agent runtime environments, model services, sandbox environments, etc.

主要功能 / Main Features:
- Agent Runtime: Agent 运行时管理 / Agent runtime management
- Model Service: 模型服务管理 / Model service management
- Sandbox: 沙箱环境管理 / Sandbox environment management
- ToolSet: 工具集管理 / Toolset management
- Credential: 凭证管理 / Credential management
- Server: HTTP 服务器 / HTTP server
- Integration: 框架集成 / Framework integration
"""

from typing import TYPE_CHECKING

__version__ = "0.0.8"

# Agent Runtime
from agentrun.agent_runtime import (
    AgentRuntime,
    AgentRuntimeArtifact,
    AgentRuntimeClient,
    AgentRuntimeCode,
    AgentRuntimeContainer,
    AgentRuntimeControlAPI,
    AgentRuntimeCreateInput,
    AgentRuntimeEndpoint,
    AgentRuntimeEndpointCreateInput,
    AgentRuntimeEndpointListInput,
    AgentRuntimeEndpointRoutingConfig,
    AgentRuntimeEndpointRoutingWeight,
    AgentRuntimeEndpointUpdateInput,
    AgentRuntimeHealthCheckConfig,
    AgentRuntimeLanguage,
    AgentRuntimeListInput,
    AgentRuntimeLogConfig,
    AgentRuntimeProtocolConfig,
    AgentRuntimeProtocolType,
    AgentRuntimeUpdateInput,
)
# Credential
from agentrun.credential import (
    Credential,
    CredentialBasicAuth,
    CredentialClient,
    CredentialConfig,
    CredentialControlAPI,
    CredentialCreateInput,
    CredentialListInput,
    CredentialUpdateInput,
    RelatedResource,
)
# Model Service
from agentrun.model import (
    BackendType,
    ModelClient,
    ModelCompletionAPI,
    ModelControlAPI,
    ModelDataAPI,
    ModelFeatures,
    ModelInfoConfig,
    ModelParameterRule,
    ModelProperties,
    ModelProxy,
    ModelProxyCreateInput,
    ModelProxyListInput,
    ModelProxyUpdateInput,
    ModelService,
    ModelServiceCreateInput,
    ModelServiceListInput,
    ModelServiceUpdateInput,
    ModelType,
    Provider,
    ProviderSettings,
    ProxyConfig,
    ProxyConfigEndpoint,
    ProxyConfigFallback,
    ProxyConfigPolicies,
)
# Sandbox
from agentrun.sandbox import (
    BrowserSandbox,
    CodeInterpreterSandbox,
    SandboxClient,
    Template,
)
# ToolSet
from agentrun.toolset import ToolSet, ToolSetClient
from agentrun.utils.config import Config
from agentrun.utils.exception import (
    ResourceAlreadyExistError,
    ResourceNotExistError,
)
from agentrun.utils.model import Status

# Server - 延迟导入以避免可选依赖问题
# Type hints for IDE and type checkers
if TYPE_CHECKING:
    from agentrun.server import (
        AgentEvent,
        AgentEventItem,
        AgentRequest,
        AgentResult,
        AgentResultItem,
        AgentReturnType,
        AgentRunServer,
        AguiEventNormalizer,
        AGUIProtocolConfig,
        AGUIProtocolHandler,
        AsyncAgentEventGenerator,
        AsyncAgentResultGenerator,
        AsyncInvokeAgentHandler,
        BaseProtocolHandler,
        EventType,
        InvokeAgentHandler,
        MergeOptions,
        Message,
        MessageRole,
        OpenAIProtocolConfig,
        OpenAIProtocolHandler,
        ProtocolConfig,
        ProtocolHandler,
        ServerConfig,
        SyncAgentEventGenerator,
        SyncAgentResultGenerator,
        SyncInvokeAgentHandler,
        Tool,
        ToolCall,
    )

__all__ = [
    ######## Agent Runtime ########
    # base
    "AgentRuntime",
    "AgentRuntimeEndpoint",
    "AgentRuntimeClient",
    "AgentRuntimeControlAPI",
    # enum
    "AgentRuntimeArtifact",
    "AgentRuntimeLanguage",
    "AgentRuntimeProtocolType",
    "Status",
    # inner model
    "AgentRuntimeCode",
    "AgentRuntimeContainer",
    "AgentRuntimeHealthCheckConfig",
    "AgentRuntimeLogConfig",
    "AgentRuntimeProtocolConfig",
    "AgentRuntimeEndpointRoutingConfig",
    "AgentRuntimeEndpointRoutingWeight",
    # api model
    "AgentRuntimeCreateInput",
    "AgentRuntimeUpdateInput",
    "AgentRuntimeListInput",
    "AgentRuntimeEndpointCreateInput",
    "AgentRuntimeEndpointUpdateInput",
    "AgentRuntimeEndpointListInput",
    ######## Credential ########
    # base
    "Credential",
    "CredentialClient",
    "CredentialControlAPI",
    # inner model
    "CredentialBasicAuth",
    "RelatedResource",
    "CredentialConfig",
    # api model
    "CredentialCreateInput",
    "CredentialUpdateInput",
    "CredentialListInput",
    ######## Model ########
    # base
    "ModelClient",
    "ModelService",
    "ModelProxy",
    "ModelControlAPI",
    "ModelCompletionAPI",
    "ModelDataAPI",
    # enum
    "BackendType",
    "ModelType",
    "Provider",
    # inner model
    "ProviderSettings",
    "ModelFeatures",
    "ModelProperties",
    "ModelParameterRule",
    "ModelInfoConfig",
    "ProxyConfigEndpoint",
    "ProxyConfigFallback",
    "ProxyConfigPolicies",
    "ProxyConfig",
    # api model
    "ModelServiceCreateInput",
    "ModelServiceUpdateInput",
    "ModelServiceListInput",
    "ModelProxyCreateInput",
    "ModelProxyUpdateInput",
    "ModelProxyListInput",
    ######## Sandbox ########
    "SandboxClient",
    "BrowserSandbox",
    "CodeInterpreterSandbox",
    "Template",
    ######## ToolSet ########
    "ToolSetClient",
    "ToolSet",
    ######## Server (延迟加载) ########
    # Server
    "AgentRunServer",
    # Config
    "ServerConfig",
    "ProtocolConfig",
    "OpenAIProtocolConfig",
    "AGUIProtocolConfig",
    # Request/Response Models
    "AgentRequest",
    "AgentEvent",
    "AgentResult",
    "Message",
    "MessageRole",
    "Tool",
    "ToolCall",
    # Event Types
    "EventType",
    # Type Aliases
    "AgentEventItem",
    "AgentResultItem",
    "AgentReturnType",
    "SyncAgentEventGenerator",
    "SyncAgentResultGenerator",
    "AsyncAgentEventGenerator",
    "AsyncAgentResultGenerator",
    "InvokeAgentHandler",
    "AsyncInvokeAgentHandler",
    "SyncInvokeAgentHandler",
    # Protocol Base
    "ProtocolHandler",
    "BaseProtocolHandler",
    # Protocol - OpenAI
    "OpenAIProtocolHandler",
    # Protocol - AG-UI
    "AGUIProtocolHandler",
    # Event Normalizer
    "AguiEventNormalizer",
    # Helpers
    "MergeOptions",
    ######## Others ########
    "ResourceNotExistError",
    "ResourceAlreadyExistError",
    "Config",
]

# Server 模块的所有导出
_SERVER_EXPORTS = {
    "AgentRunServer",
    "ServerConfig",
    "ProtocolConfig",
    "OpenAIProtocolConfig",
    "AGUIProtocolConfig",
    "AgentRequest",
    "AgentEvent",
    "AgentResult",
    "Message",
    "MessageRole",
    "Tool",
    "ToolCall",
    "EventType",
    "AgentEventItem",
    "AgentResultItem",
    "AgentReturnType",
    "SyncAgentEventGenerator",
    "SyncAgentResultGenerator",
    "AsyncAgentEventGenerator",
    "AsyncAgentResultGenerator",
    "InvokeAgentHandler",
    "AsyncInvokeAgentHandler",
    "SyncInvokeAgentHandler",
    "ProtocolHandler",
    "BaseProtocolHandler",
    "OpenAIProtocolHandler",
    "AGUIProtocolHandler",
    "AguiEventNormalizer",
    "MergeOptions",
}

# 可选依赖包映射：安装命令 -> 导入错误的包名列表
# Optional dependency mapping: installation command -> list of import error package names
# 将使用相同安装命令的包合并到一起 / Group packages with the same installation command
_OPTIONAL_PACKAGES = {
    "agentrun-sdk[server]": ["fastapi", "uvicorn", "ag_ui"],
}


def __getattr__(name: str):
    """延迟加载 server 模块的导出，避免可选依赖导致导入失败

    当用户访问 server 相关的类时，才尝试导入 server 模块。
    如果 server 可选依赖未安装，会抛出清晰的错误提示。
    """
    if name in _SERVER_EXPORTS:
        try:
            from agentrun import server

            return getattr(server, name)
        except ImportError as e:
            # 检查是否是缺少可选依赖导致的错误
            error_str = str(e)
            for install_cmd, package_names in _OPTIONAL_PACKAGES.items():
                for package_name in package_names:
                    if package_name in error_str:
                        raise ImportError(
                            f"'{name}' requires the 'server' optional dependencies. "
                            f"Install with: pip install {install_cmd}\n"
                            f"Original error: {e}"
                        ) from e
            # 其他导入错误继续抛出
            raise

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
