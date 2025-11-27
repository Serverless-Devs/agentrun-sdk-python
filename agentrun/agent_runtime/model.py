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


class AgentRuntimeContainer(BaseModel):
    """Agent Runtime 容器配置"""

    command: Optional[List[str]] = Field(alias="command", default=None)
    """在运行时中运行的命令（例如：["python"]）"""
    image: Optional[str] = Field(alias="image", default=None)
    """容器镜像地址"""


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


class AgentRuntimeProtocolConfig(BaseModel):
    """Agent Runtime 协议配置"""

    type: AgentRuntimeProtocolType = Field(
        alias="type", default=AgentRuntimeProtocolType.HTTP
    )
    """协议类型"""


class AgentRuntimeEndpointRoutingWeight(BaseModel):
    """智能体运行时端点路由配置"""

    version: Optional[str] = None
    weight: Optional[int] = None


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
    network_configuration: Optional[NetworkConfig] = None
    """Agent Runtime 网络配置"""
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
    tags: Optional[List[str]] = None
    """标签列表"""


class AgentRuntimeImmutableProps(BaseModel):
    pass


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
    routing_configuration: Optional[AgentRuntimeEndpointRoutingConfig] = None
    """智能体运行时端点的路由配置，支持多版本权重分配"""
    tags: Optional[List[str]] = None
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
    tags: Optional[str] = None
    """标签过滤，多个标签用逗号分隔"""
    search_mode: Optional[str] = None
    """搜索模式"""


class AgentRuntimeEndpointCreateInput(
    AgentRuntimeEndpointMutableProps, AgentRuntimeEndpointImmutableProps
):
    pass


class AgentRuntimeEndpointUpdateInput(AgentRuntimeEndpointMutableProps):
    pass


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
