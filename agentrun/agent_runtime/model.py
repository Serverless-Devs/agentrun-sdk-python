"""Agent Runtime 数据模型 / Agent Runtime Data Models

此模块定义 Agent Runtime 相关的所有数据模型、枚举和输入输出对象。
This module defines all data models, enums, and input/output objects related to Agent Runtime.
"""

import base64
from enum import Enum
import os
import time
from typing import Dict, List, Optional
import zipfile

import crcmod

from agentrun.utils.model import (
    BaseModel,
    Field,
    NetworkConfig,
    PageableInput,
    Status,
)


class AgentRuntimeArtifact(str, Enum):
    """Agent Runtime 运行方式 / Agent Runtime Artifact Type

    定义 Agent Runtime 的运行方式,支持代码模式和容器模式。
    Defines the runtime mode of Agent Runtime, supporting code mode and container mode.
    """

    CODE = "Code"
    """代码直接运行模式 / Code execution mode"""
    CONTAINER = "Container"
    """容器镜像模式 / Container image mode"""


class AgentRuntimeLanguage(str, Enum):
    """Agent Runtime 运行时语言 / Agent Runtime Language

    支持的编程语言运行时。
    Supported programming language runtimes.
    """

    PYTHON310 = "python3.10"
    """Python 3.10 运行时 / Python 3.10 runtime"""
    PYTHON312 = "python3.12"
    """Python 3.12 运行时 / Python 3.12 runtime"""
    NODEJS18 = "nodejs18"
    """Node.js 18 运行时 / Node.js 18 runtime"""
    NODEJS20 = "nodejs20"
    """Node.js 20 运行时 / Node.js 20 runtime"""
    JAVA8 = "java8"
    """Java 8 运行时 / Java 8 runtime"""
    JAVA11 = "java11"
    """Java 11 运行时 / Java 11 runtime"""


class AgentRuntimeCode(BaseModel):
    """Agent Runtime 代码配置"""

    checksum: Optional[str] = None
    """代码包的 CRC-64校验值。如果提供了checksum，则函数计算会校验代码包的checksum是否和提供的一致"""
    command: Optional[List[str]] = None
    """在运行时中运行的命令（例如：["python"]）"""
    language: Optional[AgentRuntimeLanguage] = None
    """代码运行时的编程语言，如 python3、nodejs 等"""
    oss_bucket_name: Optional[str] = None
    """OSS存储桶名称"""
    oss_object_name: Optional[str] = None
    """OSS对象名称"""
    zip_file: Optional[str] = None
    """智能体代码ZIP包的Base64编码"""

    @classmethod
    def from_zip_file(
        cls,
        language: AgentRuntimeLanguage,
        command: List[str],
        zip_file_path: str,
    ) -> "AgentRuntimeCode":
        with open(zip_file_path, "rb") as f:
            data = f.read()

        crc64 = crcmod.mkCrcFun(
            0x142F0E1EBA9EA3693, initCrc=0, xorOut=0xFFFFFFFFFFFFFFFF, rev=True
        )

        checksum = crc64(data).__str__()
        return cls(
            language=language,
            command=command,
            zip_file=base64.b64encode(data).decode("utf-8"),
            checksum=checksum,
        )

    @classmethod
    def from_oss(
        cls,
        language: AgentRuntimeLanguage,
        command: List[str],
        bucket: str,
        object: str,
    ) -> "AgentRuntimeCode":
        return cls(
            language=language,
            command=command,
            oss_bucket_name=bucket,
            oss_object_name=object,
        )

    @classmethod
    def from_file(
        cls, language: AgentRuntimeLanguage, command: List[str], file_path: str
    ) -> "AgentRuntimeCode":
        # 如果是文件夹，则先将文件夹打包成 zip
        zip_file_path = os.path.join(
            os.path.dirname(file_path), str(int(time.time())) + ".zip"
        )

        if os.path.isdir(file_path):
            with zipfile.ZipFile(
                zip_file_path, "w", zipfile.ZIP_DEFLATED
            ) as zipf:
                for root, dirs, files in os.walk(file_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, file_path)
                        zipf.write(full_path, relative_path)
        else:
            with zipfile.ZipFile(
                zip_file_path, "w", zipfile.ZIP_DEFLATED
            ) as zipf:
                zipf.write(file_path, os.path.basename(file_path))

        c = cls.from_zip_file(language, command, zip_file_path)
        os.remove(zip_file_path)

        return c


class RegistryAuthConfig(BaseModel):
    """镜像仓库认证配置 / Registry Authentication Configuration"""

    password: Optional[str] = None
    """镜像仓库的登录密码 / Registry login password"""
    user_name: Optional[str] = None
    """镜像仓库的登录用户名 / Registry login username"""


class RegistryCertConfig(BaseModel):
    """镜像仓库证书配置 / Registry Certificate Configuration"""

    insecure: Optional[bool] = None
    """是否跳过 TLS 证书验证 / Whether to skip TLS certificate verification"""
    root_ca_cert_base_64: Optional[str] = None
    """镜像仓库根 CA 证书 Base64 编码 / Registry root CA certificate (Base64 encoded)"""


class RegistryNetworkConfig(BaseModel):
    """镜像仓库网络配置 / Registry Network Configuration"""

    security_group_id: Optional[str] = None
    """镜像仓库的安全组 ID / Registry security group ID"""
    v_switch_id: Optional[str] = None
    """镜像仓库所在的交换机 ID / Registry vSwitch ID"""
    vpc_id: Optional[str] = None
    """镜像仓库所在的 VPC ID / Registry VPC ID"""


class RegistryConfig(BaseModel):
    """自定义镜像仓库配置 / Custom Registry Configuration"""

    auth_config: Optional[RegistryAuthConfig] = None
    """镜像仓库的认证配置 / Registry authentication configuration"""
    cert_config: Optional[RegistryCertConfig] = None
    """镜像仓库的证书配置 / Registry certificate configuration"""
    network_config: Optional[RegistryNetworkConfig] = None
    """镜像仓库的网络配置 / Registry network configuration"""


class AgentRuntimeContainer(BaseModel):
    """Agent Runtime 容器配置"""

    acr_instance_id: Optional[str] = None
    """阿里云容器镜像服务（ACR）的实例 ID / Aliyun Container Registry (ACR) instance ID"""
    command: Optional[List[str]] = Field(alias="command", default=None)
    """在运行时中运行的命令（例如：["python"]）"""
    image: Optional[str] = Field(alias="image", default=None)
    """容器镜像地址"""
    image_registry_type: Optional[str] = None
    """容器镜像来源类型，支持 ACR / ACREE / CUSTOM
    / Container image registry type: ACR / ACREE / CUSTOM"""
    port: Optional[int] = None
    """容器内部监听端口 / Container internal port"""
    registry_config: Optional[RegistryConfig] = None
    """自定义镜像仓库配置（当 image_registry_type 为 CUSTOM 时使用）
    / Custom registry configuration (used when image_registry_type is CUSTOM)"""


class AgentRuntimeHealthCheckConfig(BaseModel):
    """Agent Runtime 健康检查配置"""

    failure_threshold: Optional[int] = Field(
        alias="failureThreshold", default=None
    )
    """在将容器视为不健康之前，连续失败的健康检查次数"""
    http_get_url: Optional[str] = Field(alias="httpGetUrl", default=None)
    """用于健康检查的HTTP GET请求的URL地址"""
    initial_delay_seconds: Optional[int] = Field(
        alias="initialDelaySeconds", default=None
    )
    """在容器启动后，首次执行健康检查前的延迟时间（秒）"""
    period_seconds: Optional[int] = Field(alias="periodSeconds", default=None)
    """执行健康检查的时间间隔（秒）"""
    success_threshold: Optional[int] = Field(
        alias="successThreshold", default=None
    )
    """在将容器视为健康之前，连续成功的健康检查次数"""
    timeout_seconds: Optional[int] = Field(alias="timeoutSeconds", default=None)
    """健康检查的超时时间（秒）"""


class AgentRuntimeLogConfig(BaseModel):
    """Agent Runtime 日志配置"""

    project: str = Field(alias="project")
    """SLS项目名称"""
    logstore: str = Field(alias="logstore")
    """SLS日志库名称"""


class AgentRuntimeProtocolType(str, Enum):
    """Agent Runtime 协议类型"""

    HTTP = "HTTP"
    MCP = "MCP"
    SUPER_AGENT = "SUPER_AGENT"


class ProtocolSettings(BaseModel):
    """详细协议配置项 / Detailed Protocol Settings

    用于配置单个协议的明细参数（路径、HTTP 方法、请求/响应 schema 等）。
    Used to configure detailed parameters for a single protocol (path, HTTP method,
    request/response schemas, etc.).
    """

    a_2aagent_card: Optional[str] = Field(alias="A2AAgentCard", default=None)
    """A2A Agent Card（兼容旧字段名 A2AAgentCard）/ A2A Agent Card (legacy field name)"""
    a_2a_agent_card: Optional[str] = Field(alias="a2aAgentCard", default=None)
    """A2A Agent Card / A2A Agent Card"""
    a_2a_agent_card_url: Optional[str] = Field(
        alias="a2aAgentCardUrl", default=None
    )
    """A2A Agent Card URL / A2A Agent Card URL"""
    config: Optional[str] = None
    """协议配置的 JSON 字符串 / Protocol configuration JSON string"""
    headers: Optional[str] = None
    """请求头 / Request headers"""
    input_body_json_schema: Optional[str] = None
    """请求体 JSON Schema / Request body JSON schema"""
    method: Optional[str] = None
    """HTTP 方法 / HTTP method"""
    name: Optional[str] = None
    """可选展示名 / Optional display name"""
    output_body_json_schema: Optional[str] = None
    """响应体 JSON Schema / Response body JSON schema"""
    path: Optional[str] = None
    """协议路径 / Protocol path"""
    path_prefix: Optional[str] = None
    """协议路径前缀 / Protocol path prefix"""
    request_content_type: Optional[str] = None
    """请求内容类型 / Request content type"""
    response_content_type: Optional[str] = None
    """响应内容类型 / Response content type"""
    type: Optional[str] = None
    """协议类型标识 / Protocol type identifier"""


class AgentRuntimeProtocolConfig(BaseModel):
    """Agent Runtime 协议配置"""

    protocol_settings: Optional[List[ProtocolSettings]] = None
    """详细的协议配置信息 / Detailed protocol settings"""
    type: AgentRuntimeProtocolType = Field(
        alias="type", default=AgentRuntimeProtocolType.HTTP
    )
    """协议类型"""


class NASMountConfig(BaseModel):
    """NAS 挂载点配置 / NAS Mount Configuration"""

    enable_tls: Optional[bool] = Field(alias="enableTLS", default=None)
    """是否启用 TLS / Whether to enable TLS"""
    mount_dir: Optional[str] = None
    """挂载目录 / Mount directory"""
    server_addr: Optional[str] = None
    """NAS 服务地址 / NAS server address"""


class NASConfig(BaseModel):
    """NAS 文件系统配置 / NAS Filesystem Configuration"""

    group_id: Optional[int] = None
    """用户组 ID / Group ID"""
    mount_points: Optional[List[NASMountConfig]] = None
    """挂载点列表 / Mount points"""
    user_id: Optional[int] = None
    """用户 ID / User ID"""


class OSSMountPoint(BaseModel):
    """OSS 挂载点 / OSS Mount Point"""

    bucket_name: Optional[str] = None
    """OSS Bucket 名称 / OSS bucket name"""
    bucket_path: Optional[str] = None
    """OSS Bucket 中要挂载的路径 / Path inside the bucket to mount"""
    endpoint: Optional[str] = None
    """OSS Endpoint / OSS endpoint"""
    mount_dir: Optional[str] = None
    """容器内挂载目录 / Mount directory inside the container"""
    read_only: Optional[bool] = None
    """是否只读挂载 / Whether to mount read-only"""


class OSSMountConfig(BaseModel):
    """OSS 挂载配置 / OSS Mount Configuration"""

    mount_points: Optional[List[OSSMountPoint]] = None
    """挂载点列表 / Mount points"""


class ScheduledPolicy(BaseModel):
    """端点弹性伸缩定时策略 / Endpoint Scaling Scheduled Policy"""

    end_time: Optional[str] = None
    """结束时间 / End time"""
    name: Optional[str] = None
    """策略名称 / Policy name"""
    schedule_expression: Optional[str] = None
    """定时表达式（cron） / Schedule expression (cron)"""
    start_time: Optional[str] = None
    """开始时间 / Start time"""
    target: Optional[int] = None
    """目标实例数 / Target instance count"""
    time_zone: Optional[str] = None
    """时区 / Time zone"""


class ScalingConfig(BaseModel):
    """端点弹性伸缩配置 / Endpoint Scaling Configuration"""

    min_instances: Optional[int] = None
    """最小实例数 / Minimum instance count"""
    scheduled_policies: Optional[List[ScheduledPolicy]] = None
    """定时扩缩容策略列表 / Scheduled scaling policies"""


class AgentRuntimeEndpointRoutingWeight(BaseModel):
    """智能体运行时端点路由配置"""

    version: Optional[str] = None
    weight: Optional[float] = None


class AgentRuntimeEndpointRoutingConfig(BaseModel):
    """智能体运行时端点路由配置"""

    version_weights: Optional[List[AgentRuntimeEndpointRoutingWeight]] = None
    """版本权重列表"""


class AgentRuntimeMutableProps(BaseModel):
    agent_runtime_name: Optional[str] = None
    """Agent Runtime 名称"""
    artifact_type: Optional[AgentRuntimeArtifact] = None
    """Agent Runtime 运行方式"""
    code_configuration: Optional[AgentRuntimeCode] = None
    """Agent Runtime 代码配置"""
    container_configuration: Optional[AgentRuntimeContainer] = None
    """Agent Runtime 容器配置"""
    cpu: Optional[float] = 2
    """Agent Runtime CPU 配置，单位：核"""
    credential_name: Optional[str] = None
    """Agent Runtime 凭证 ID"""
    description: Optional[str] = None
    """Agent Runtime 描述"""
    disk_size: Optional[int] = None
    """Agent Runtime 实例磁盘大小，单位：GB
    / Instance disk size in GB"""
    enable_session_isolation: Optional[bool] = None
    """是否启用会话隔离，启用后每个会话将在独立的环境中运行
    / Whether to enable session isolation; each session runs in an isolated environment when enabled"""
    environment_variables: Optional[Dict[str, str]] = None
    """环境变量"""
    execution_role_arn: Optional[str] = None
    """Agent Runtime 执行角色 ARN"""
    health_check_configuration: Optional[AgentRuntimeHealthCheckConfig] = None
    """健康检查配置"""
    # instance_idle_timeout_seconds: Optional[int] = 1
    # """实例空闲超时时间，单位：秒"""
    log_configuration: Optional[AgentRuntimeLogConfig] = None
    """日志配置"""
    memory: Optional[int] = 4096
    """Agent Runtime 内存配置，单位：MB"""
    nas_config: Optional[NASConfig] = None
    """NAS 文件系统挂载配置 / NAS filesystem mount configuration"""
    network_configuration: Optional[NetworkConfig] = None
    """Agent Runtime 网络配置"""
    oss_mount_config: Optional[OSSMountConfig] = None
    """OSS 挂载配置 / OSS mount configuration"""
    port: Optional[int] = 9000
    """Agent Runtime 端口配置"""
    protocol_configuration: Optional[AgentRuntimeProtocolConfig] = None
    """协议配置"""
    # request_timeout_seconds: Optional[int] = None
    # """请求超时时间，单位：秒"""
    session_concurrency_limit_per_instance: Optional[int] = None
    """每实例会话并发限制"""
    session_idle_timeout_seconds: Optional[int] = None
    """会话空闲超时时间，单位：秒"""
    system_tags: Optional[List[str]] = None
    """系统标签列表 (由平台内部使用, 例如 SuperAgent 用来标识下游 AgentRuntime)
    / System tags (used internally by the platform, e.g. by SuperAgent to mark downstream AgentRuntimes)"""


class AgentRuntimeImmutableProps(BaseModel):
    workspace_id: Optional[str] = None
    """Agent Runtime 所属的工作空间标识符；可选项，不填则使用默认工作空间
    / Workspace identifier the Agent Runtime belongs to; optional, defaults to the default workspace if not provided"""


class AgentRuntimeSystemProps(BaseModel):
    agent_runtime_arn: Optional[str] = None
    """全局唯一资源名称"""
    agent_runtime_id: Optional[str] = None
    """唯一标识符"""
    created_at: Optional[str] = None
    """创建时间"""
    last_updated_at: Optional[str] = None
    """最后更新时间"""
    resource_name: Optional[str] = None
    """资源名称"""
    status: Optional[Status] = None
    """运行状态"""
    status_reason: Optional[str] = None
    """状态原因"""
    agent_runtime_version: Optional[str] = None
    """版本号"""


class AgentRuntimeEndpointMutableProps(BaseModel):
    agent_runtime_endpoint_name: Optional[str] = None
    description: Optional[str] = None
    disable_public_network_access: Optional[bool] = None
    """是否禁用该端点的公网访问
    / Whether to disable public network access for this endpoint"""
    routing_configuration: Optional[AgentRuntimeEndpointRoutingConfig] = None
    """智能体运行时端点的路由配置，支持多版本权重分配"""
    scaling_config: Optional[ScalingConfig] = None
    """端点的弹性伸缩配置，包括最小实例数和定时扩容策略
    / Endpoint scaling configuration: min instances and scheduled policies"""
    target_version: Optional[str] = "LATEST"
    """智能体运行时的目标版本"""


class AgentRuntimeEndpointImmutableProps(BaseModel):
    pass


class AgentRuntimeEndpointSystemProps(BaseModel):
    agent_runtime_endpoint_arn: Optional[str] = None
    """智能体运行时端点的资源ARN"""
    agent_runtime_endpoint_id: Optional[str] = None
    """智能体运行时端点的唯一标识ID"""
    agent_runtime_id: Optional[str] = None
    """智能体运行时的ID"""
    endpoint_public_url: Optional[str] = None
    """智能体运行时端点的公网访问地址"""
    resource_name: Optional[str] = None
    """智能体运行时端点的资源名称"""
    status: Optional[Status] = None
    """智能体运行时端点的状态"""
    status_reason: Optional[str] = None
    """智能体运行时端点的状态原因"""


class AgentRuntimeCreateInput(
    AgentRuntimeMutableProps, AgentRuntimeImmutableProps
):
    pass


class AgentRuntimeUpdateInput(AgentRuntimeMutableProps):
    pass


class AgentRuntimeListInput(PageableInput):
    agent_runtime_name: Optional[str] = None
    """Agent Runtime 名称"""
    system_tags: Optional[str] = None
    """系统标签过滤, 多个标签用逗号分隔
    / Filter by system tags, comma separated"""
    search_mode: Optional[str] = None
    """搜索模式"""
    status: Optional[str] = None
    """按状态过滤，多个状态用逗号分隔，支持精确匹配
    / Filter by status, comma separated"""
    workspace_id: Optional[str] = None
    """按工作空间标识符过滤
    / Filter by workspace identifier"""
    workspace_ids: Optional[str] = None
    """按多个工作空间标识符过滤，逗号分隔
    / Filter by multiple workspace identifiers, comma separated"""


class AgentRuntimeEndpointCreateInput(
    AgentRuntimeEndpointMutableProps, AgentRuntimeEndpointImmutableProps
):
    pass


class AgentRuntimeEndpointUpdateInput(AgentRuntimeEndpointMutableProps):
    delete_scaling_config: Optional[bool] = None
    """为 true 时删除该端点的弹性伸缩配置
    / If true, delete the existing scaling configuration for this endpoint"""


class AgentRuntimeEndpointListInput(PageableInput):
    endpoint_name: Optional[str] = None
    """端点名称"""
    search_mode: Optional[str] = None
    """搜索模式"""


class AgentRuntimeVersion(BaseModel):
    agent_runtime_arn: Optional[str] = None
    """智能体运行时的ARN"""
    agent_runtime_id: Optional[str] = None
    """智能体运行时的ID"""
    agent_runtime_name: Optional[str] = None
    """智能体运行时的名称"""
    agent_runtime_version: Optional[str] = None
    """已发布版本的版本号"""
    description: Optional[str] = None
    """此版本的描述"""
    last_updated_at: Optional[str] = None
    """最后更新的时间戳"""


class AgentRuntimeVersionListInput(PageableInput):
    pass
