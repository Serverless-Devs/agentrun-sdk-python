"""Sandbox 模型定义 / Sandbox Model Definitions

定义沙箱相关的数据模型和枚举。
Defines data models and enumerations related to sandboxes.
"""

from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING
import uuid

from pydantic import model_validator

from agentrun.utils.model import BaseModel

if TYPE_CHECKING:
    from agentrun.sandbox.sandbox import Sandbox


class TemplateOSSPermission(str, Enum):
    """Agent Runtime 网络访问模式 / Agent Runtime Network Access Mode"""

    READ_WRITE = "READ_WRITE"
    """读写模式 / Read-Write Mode"""
    READ_ONLY = "READ_ONLY"
    """只读模式 / Read-Only Mode"""


class TemplateType(str, Enum):
    """沙箱模板类型 / Sandbox Template Type"""

    CODE_INTERPRETER = "CodeInterpreter"
    """代码解释器 / Code Interpreter"""
    BROWSER = "Browser"
    """浏览器 / Browser"""
    AIO = "AllInOne"
    """All-in-One 沙箱 / All-in-One Sandbox"""


class TemplateNetworkMode(str, Enum):
    """Agent Runtime 网络访问模式 / Agent Runtime Network Access Mode"""

    PUBLIC = "PUBLIC"
    """公网模式 / Public Mode"""
    PRIVATE = "PRIVATE"
    """私网模式 / Private Mode"""
    PUBLIC_AND_PRIVATE = "PUBLIC_AND_PRIVATE"
    """公私网模式 / Public and Private Mode"""


class CodeLanguage(str, Enum):
    """Code Interpreter 代码语言 / Code Interpreter Programming Language"""

    PYTHON = "python"


class TemplateNetworkConfiguration(BaseModel):
    """沙箱模板网络配置 / Sandbox Template Network Configuration"""

    network_mode: TemplateNetworkMode = TemplateNetworkMode.PUBLIC
    """网络访问模式 / Network Access Mode"""
    security_group_id: Optional[str] = None
    """安全组 ID / Security Group ID"""
    vpc_id: Optional[str] = None
    """私有网络 ID / VPC ID"""
    vswitch_ids: Optional[List[str]] = None
    """私有网络交换机 ID 列表 / VSwitch ID List"""


class TemplateOssConfiguration(BaseModel):
    """沙箱模板 OSS 配置 / Sandbox Template OSS Configuration"""

    bucket_name: str
    """OSS 存储桶名称 / OSS Bucket Name"""
    mount_point: str
    """OSS 挂载点 / OSS Mount Point"""
    permission: Optional[TemplateOSSPermission] = (
        TemplateOSSPermission.READ_WRITE
    )
    """OSS 权限 / OSS Permission"""
    prefix: str
    """OSS 对象前缀/路径 / OSS Object Prefix/Path"""
    region: str
    """OSS 区域"""


class TemplateLogConfiguration(BaseModel):
    """沙箱模板日志配置 / Sandbox Template Log Configuration"""

    project: Optional[str] = None
    """SLS 日志项目 / SLS Log Project"""
    logstore: Optional[str] = None
    """SLS 日志库 / SLS Logstore"""


class TemplateCredentialConfiguration(BaseModel):
    """沙箱模板凭证配置 / Sandbox Template Credential Configuration"""

    credential_name: Optional[str] = None
    """凭证名称 / Credential Name"""


class TemplateArmsConfiguration(BaseModel):
    """沙箱模板 ARMS 监控配置 / Sandbox Template ARMS Monitoring Configuration"""

    arms_license_key: Optional[str] = None
    """ARMS 许可证密钥 / ARMS License Key"""

    enable_arms: bool
    """是否启用 ARMS 监控 / Whether to enable ARMS monitoring"""


class TemplateContainerConfiguration(BaseModel):
    """沙箱模板容器配置 / Sandbox Template Container Configuration"""

    image: Optional[str] = None
    """容器镜像地址 / Container Image Address"""
    command: Optional[List[str]] = None
    """容器启动命令 / Container Start Command"""


class TemplateMcpOptions(BaseModel):
    """沙箱模板 MCP 选项配置 / Sandbox Template MCP Options Configuration"""

    enabled_tools: Optional[List[str]] = None

    """启用的工具列表 / Enabled Tools List"""
    transport: Optional[str] = None
    """传输协议 / Transport Protocol"""


class TemplateMcpState(BaseModel):
    """沙箱模板 MCP 状态 / Sandbox Template MCP State"""

    access_endpoint: Optional[str] = None
    """访问端点 / Access Endpoint"""
    status: Optional[str] = None
    """状态 / Status"""
    status_reason: Optional[str] = None
    """状态原因 / Status Reason"""


class TemplateInput(BaseModel):
    """沙箱模板配置 / Sandbox Template Configuration"""

    template_name: Optional[str] = f"sandbox_template_{uuid.uuid4()}"
    """模板名称 / Template Name"""
    template_type: TemplateType
    """模板类型 / Template Type"""
    cpu: Optional[float] = 2.0
    """CPU 核数 / CPU Cores"""
    memory: Optional[int] = 4096
    """内存大小（MB） / Memory Size (MB)"""
    execution_role_arn: Optional[str] = None
    """执行角色 ARN / Execution Role ARN"""
    sandbox_idle_timeout_in_seconds: Optional[int] = 1800
    """沙箱空闲超时时间（秒） / Sandbox Idle Timeout (seconds)"""
    sandbox_ttlin_seconds: Optional[int] = 21600
    """沙箱存活时间（秒） / Sandbox TTL (seconds)"""
    share_concurrency_limit_per_sandbox: Optional[int] = 200
    """每个沙箱的最大并发会话数 / Max Concurrency Limit Per Sandbox"""
    template_configuration: Optional[Dict] = None
    """模板配置 / Template Configuration"""
    description: Optional[str] = None
    """描述 / Description"""
    environment_variables: Optional[Dict] = None
    """环境变量 / Environment Variables"""
    network_configuration: Optional[TemplateNetworkConfiguration] = (
        TemplateNetworkConfiguration(network_mode=TemplateNetworkMode.PUBLIC)
    )
    """网络配置 / Network Configuration"""
    oss_configuration: Optional[List[TemplateOssConfiguration]] = None
    """OSS 配置列表 / OSS Configuration List"""
    log_configuration: Optional[TemplateLogConfiguration] = None
    """日志配置 / Log Configuration"""
    credential_configuration: Optional[TemplateCredentialConfiguration] = None
    """凭证配置 / Credential Configuration"""
    arms_configuration: Optional[TemplateArmsConfiguration] = None
    """ARMS 监控配置 / ARMS Monitoring Configuration"""
    container_configuration: Optional[TemplateContainerConfiguration] = None
    """容器配置 / Container Configuration"""
    disk_size: Optional[int] = None
    """磁盘大小（GB） / Disk Size (GB)"""

    @model_validator(mode="before")
    @classmethod
    def set_disk_size_default(cls, values):
        """根据 template_type 设置 disk_size 的默认值 / Set default disk_size based on template_type"""
        # 如果 disk_size 已经被显式设置，则不修改
        if (
            values.get("disk_size") is not None
            or values.get("diskSize") is not None
        ):
            return values

        # 获取 template_type（支持两种命名格式）
        template_type = values.get("template_type") or values.get(
            "templateType"
        )

        # 根据 template_type 设置默认值
        if (
            template_type == TemplateType.CODE_INTERPRETER
            or template_type == "CodeInterpreter"
        ):
            values["disk_size"] = 512
        elif (
            template_type == TemplateType.BROWSER
            or template_type == "Browser"
            or template_type == TemplateType.AIO
            or template_type == "AllInOne"
        ):
            values["disk_size"] = 10240
        else:
            # 如果 template_type 未设置或为其他值，使用 512 作为默认值
            values["disk_size"] = 512

        # 如果 template_type 为 AIO，则设置 cpu 和 memory 的默认值4C8G
        if template_type == TemplateType.AIO or template_type == "AllInOne":
            values["cpu"] = 4.0
            values["memory"] = 8192

        return values

    @model_validator(mode="after")
    def validate_template_constraints(self):
        if (
            self.template_type == TemplateType.BROWSER
            or self.template_type == TemplateType.AIO
        ):
            if self.disk_size != 10240:
                raise ValueError(
                    "when template_type is BROWSER，disk_size should be 10240，"
                    f"the current disk size is {self.disk_size}"
                )

        if (
            self.template_type == TemplateType.CODE_INTERPRETER
            or self.template_type == TemplateType.AIO
        ):
            if (
                self.network_configuration
                and self.network_configuration.network_mode
                == TemplateNetworkMode.PRIVATE
            ):
                raise ValueError(
                    "when template_type is CODE_INTERPRETER，"
                    "network_mode cannot be PRIVATE"
                )
        return self


class SandboxInput(BaseModel):
    """Sandbox 创建配置 / Sandbox Creation Configuration"""

    template_name: str
    """模板名称 / Template Name"""
    sandbox_idle_timeout_seconds: Optional[int] = 600
    """沙箱空闲超时时间（秒） / Sandbox Idle Timeout (seconds)"""


class ListSandboxesInput(BaseModel):
    """Sandbox 列表查询配置 / Sandbox List Query Configuration"""

    max_results: Optional[int] = 10
    next_token: Optional[str] = None
    status: Optional[str] = None
    template_name: Optional[str] = None
    template_type: Optional[TemplateType] = None


class ListSandboxesOutput(BaseModel):
    """Sandbox 列表查询结果 / Sandbox List Query Result"""

    sandboxes: List["Sandbox"]
    next_token: Optional[str] = None


class PageableInput(BaseModel):
    """分页查询参数 / Pagination Query Parameters"""

    page_number: Optional[int] = 1
    """页码 / Page Number"""
    page_size: Optional[int] = 10
    """每页大小 / Page Size"""
    template_type: Optional[TemplateType] = None
