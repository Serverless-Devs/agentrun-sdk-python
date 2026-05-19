"""Quick Start: 通过容器镜像部署一个 Agent Runtime

演示如何使用本仓库 ``agentrun.agent_runtime`` 模块，通过 **镜像（Container）**
方式部署一个 Agent Runtime，并创建访问端点（Endpoint）以验证可调用。

涵盖最近一次模型对齐（feat: align SDK model with official agentrun-20250910）
新增的字段，重点演示：

- ``AgentRuntimeContainer``: ``image`` / ``port`` / ``image_registry_type``
  以及可选的 ``acr_instance_id`` / ``registry_config``
- ``AgentRuntimeMutableProps``: ``disk_size`` / ``enable_session_isolation``
  / ``system_tags`` 等新字段
- ``AgentRuntimeEndpointCreateInput``: ``disable_public_network_access``
  / ``scaling_config``

运行前请先在环境变量里准备好镜像地址等：

    export AGENTRUNTIME_IMAGE="registry.cn-hangzhou.aliyuncs.com/<ns>/<repo>:<tag>"
    # 可选：阿里云 ACR 个人版/企业版实例
    export AGENTRUNTIME_ACR_INSTANCE_ID=""              # 企业版 ACR 实例 ID，可空
    export AGENTRUNTIME_IMAGE_REGISTRY_TYPE="ACR"       # ACR / ACREE / CUSTOM
    # 可选：自定义镜像仓库 (CUSTOM) 的认证
    export AGENTRUNTIME_REGISTRY_USERNAME=""
    export AGENTRUNTIME_REGISTRY_PASSWORD=""
    # 可选：选择工作空间，SDK 会自动调用 list_workspaces 把名字解析为 ID；
    # 不填则使用账号默认工作空间。
    export AGENTRUNTIME_WORKSPACE_NAME="my-workspace"

执行（必须用项目根 .venv）：

    .venv/bin/python examples/quickstart_runtime.py
"""

import os
from pathlib import Path
import time

from dotenv import load_dotenv

from agentrun.agent_runtime import (
    AgentRuntime,
    AgentRuntimeArtifact,
    AgentRuntimeClient,
    AgentRuntimeContainer,
    AgentRuntimeCreateInput,
    AgentRuntimeEndpoint,
    AgentRuntimeEndpointCreateInput,
    AgentRuntimeListInput,
    AgentRuntimeProtocolConfig,
    AgentRuntimeProtocolType,
    Status,
)
from agentrun.agent_runtime.model import (
    RegistryAuthConfig,
    RegistryConfig,
    ScalingConfig,
)
from agentrun.utils.exception import (
    ResourceAlreadyExistError,
    ResourceNotExistError,
)
from agentrun.utils.log import logger

load_dotenv(Path(__file__).parent / ".env")

AGENT_RUNTIME_NAME = os.getenv(
    "AGENTRUNTIME_NAME", "sdk-quickstart-runtime-image"
)
ENDPOINT_NAME = os.getenv(
    "AGENTRUNTIME_ENDPOINT_NAME", "sdk-quickstart-runtime-image-endpoint"
)
IMAGE = os.getenv("AGENTRUNTIME_IMAGE", "")
IMAGE_REGISTRY_TYPE = os.getenv("AGENTRUNTIME_IMAGE_REGISTRY_TYPE", "ACR")
ACR_INSTANCE_ID = os.getenv("AGENTRUNTIME_ACR_INSTANCE_ID") or None
REGISTRY_USERNAME = os.getenv("AGENTRUNTIME_REGISTRY_USERNAME") or None
REGISTRY_PASSWORD = os.getenv("AGENTRUNTIME_REGISTRY_PASSWORD") or None

# 工作空间名；SDK 会自动调用 list_workspaces 把它解析为 workspace_id。
# Workspace name; the SDK resolves it to workspace_id by calling
# list_workspaces before invoking the AgentRun API.
WORKSPACE_NAME = os.getenv("AGENTRUNTIME_WORKSPACE_NAME") or None

# 容器内监听端口；与 Dockerfile / 业务代码中的端口保持一致。
CONTAINER_PORT = int(os.getenv("AGENTRUNTIME_CONTAINER_PORT", "9000"))


def _require_image() -> None:
    if not IMAGE:
        raise ValueError(
            "请先设置环境变量 AGENTRUNTIME_IMAGE 为可访问的容器镜像地址，"
            "示例：registry.cn-hangzhou.aliyuncs.com/<ns>/<repo>:<tag>"
        )


def _build_container_config() -> AgentRuntimeContainer:
    """根据环境变量组装容器配置。

    - 若使用阿里云 ACR / ACREE，可填写 ``acr_instance_id``；
    - 若使用自建/三方镜像仓库（CUSTOM），需提供 ``registry_config``。
    """

    registry_config = None
    if IMAGE_REGISTRY_TYPE.upper() == "CUSTOM" and (
        REGISTRY_USERNAME or REGISTRY_PASSWORD
    ):
        registry_config = RegistryConfig(
            auth_config=RegistryAuthConfig(
                user_name=REGISTRY_USERNAME,
                password=REGISTRY_PASSWORD,
            ),
        )

    return AgentRuntimeContainer(
        image=IMAGE,
        port=CONTAINER_PORT,
        # image_registry_type=IMAGE_REGISTRY_TYPE,
        # acr_instance_id=ACR_INSTANCE_ID,
        # registry_config=registry_config,
        # 如有需要可覆盖镜像默认 CMD/ENTRYPOINT：
        # command=["python", "-m", "agentrun.server"],
    )


client = AgentRuntimeClient()


def create_or_get_runtime() -> AgentRuntime:
    """通过镜像创建 AgentRuntime，已存在则直接获取。"""

    logger.info("==== 步骤 1: 通过镜像创建 / 获取 Agent Runtime ====")
    container = _build_container_config()

    create_input = AgentRuntimeCreateInput(
        agent_runtime_name=AGENT_RUNTIME_NAME,
        # 显式声明镜像模式；不传时 client 也会按 container_configuration 自动推断。
        artifact_type=AgentRuntimeArtifact.CONTAINER,
        container_configuration=container,
        description="quickstart_runtime example: deploy AgentRuntime via image",
        cpu=2,
        memory=4096,
        # 实例磁盘大小（MB），新字段：disk_size
        disk_size=10240,
        # 启用会话隔离，每个会话独立运行；适合需要按会话隔离上下文的 Agent 应用。
        enable_session_isolation=True,
        # 端口与容器内监听端口保持一致
        port=CONTAINER_PORT,
        # SDK 自动把 workspace_name 解析为 workspace_id；不填使用默认 workspace。
        workspace_name=WORKSPACE_NAME,
        protocol_configuration=AgentRuntimeProtocolConfig(
            type=AgentRuntimeProtocolType.HTTP,
        ),
        # 平台用的系统标签，便于过滤/聚合（替代旧的 tags 字段）
        system_tags=["quickstart", "image-deploy"],
    )

    try:
        runtime = client.create(create_input)
        logger.info("已发起创建，runtime_id=%s", runtime.agent_runtime_id)
    except ResourceAlreadyExistError:
        logger.info(
            "同名 Runtime 已存在，改为查询并复用：%s", AGENT_RUNTIME_NAME
        )
        runtime = next(
            r
            for r in client.list(
                AgentRuntimeListInput(agent_runtime_name=AGENT_RUNTIME_NAME)
            )
            if r.agent_runtime_name == AGENT_RUNTIME_NAME
        )

    runtime.wait_until_ready_or_failed()
    if runtime.status != Status.READY:
        raise RuntimeError(
            f"AgentRuntime 未能就绪，status={runtime.status}, "
            f"reason={runtime.status_reason}"
        )
    logger.info(
        "Runtime 就绪: arn=%s, version=%s",
        runtime.agent_runtime_arn,
        runtime.agent_runtime_version,
    )
    return runtime


def create_or_get_endpoint(runtime: AgentRuntime) -> AgentRuntimeEndpoint:
    """为 Runtime 创建一个端点（Endpoint）以供外部调用。"""

    logger.info("==== 步骤 2: 创建 / 获取 Endpoint ====")

    endpoint_input = AgentRuntimeEndpointCreateInput(
        agent_runtime_endpoint_name=ENDPOINT_NAME,
        description="quickstart_runtime example endpoint",
        target_version="LATEST",
        # 演示新字段：保持公网可访问；如需内网专属，置为 True。
        disable_public_network_access=False,
        # 演示新字段：最小实例数为 0（按需弹性），可按需扩展定时策略。
        scaling_config=ScalingConfig(min_instances=0),
    )

    try:
        endpoint = runtime.create_endpoint(endpoint_input)
        logger.info(
            "已发起 Endpoint 创建，endpoint_id=%s",
            endpoint.agent_runtime_endpoint_id,
        )
    except ResourceAlreadyExistError:
        logger.info("同名 Endpoint 已存在，改为查询复用：%s", ENDPOINT_NAME)
        endpoint = next(
            e
            for e in runtime.list_endpoints()
            if e.agent_runtime_endpoint_name == ENDPOINT_NAME
        )

    endpoint.wait_until_ready_or_failed()
    if endpoint.status != Status.READY:
        raise RuntimeError(
            f"Endpoint 未能就绪，status={endpoint.status}, "
            f"reason={endpoint.status_reason}"
        )
    logger.info(
        "Endpoint 就绪: %s -> %s",
        endpoint.agent_runtime_endpoint_arn,
        endpoint.endpoint_public_url,
    )
    return endpoint


def cleanup(runtime: AgentRuntime, endpoint: AgentRuntimeEndpoint) -> None:
    """清理资源；线上保留时可跳过此步。"""

    logger.info("==== 步骤 3: 清理资源 ====")
    try:
        endpoint.delete_and_wait_until_finished()
        logger.info("Endpoint 已删除")
    except ResourceNotExistError:
        logger.info("Endpoint 已不存在，跳过")

    try:
        runtime.delete_and_wait_until_finished()
        logger.info("Runtime 已删除")
    except ResourceNotExistError:
        logger.info("Runtime 已不存在，跳过")


def main() -> None:
    _require_image()
    logger.info(
        "镜像部署 AgentRuntime 示例开始：image=%s, registry_type=%s",
        IMAGE,
        IMAGE_REGISTRY_TYPE,
    )

    runtime = create_or_get_runtime()
    endpoint = create_or_get_endpoint(runtime)

    logger.info(
        "部署完成，可通过该地址调用：%s （等待 1s 确保权限生效）",
        endpoint.endpoint_public_url,
    )
    time.sleep(1)

    # 默认会在最后清理；若想保留资源做后续验证，把 cleanup 注释掉即可。
    # if os.getenv("AGENTRUNTIME_KEEP_RESOURCES", "").lower() not in (
    #     "1",
    #     "true",
    #     "yes",
    # ):
    # cleanup(runtime, endpoint)


if __name__ == "__main__":
    main()
