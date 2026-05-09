"""指定 workspace_id 创建 / 查询 / 过滤资源的端到端 Demo.

End-to-end demo: create / get / list resources with an explicit workspace_id.

本 demo 演示新版 SDK 在 Credential 和 Sandbox Template 两类资源上
对 workspace_id 字段的支持：
- 创建资源时显式指定 workspace_id，资源会落在该工作空间下
- get 接口能回读 workspace_id
- list 接口能按 workspace_id 过滤

Demonstrates how the new SDK exposes ``workspace_id`` on resource creation,
read-back, and list-filter for both Credential and Sandbox Template.

环境变量 / Environment variables:
- ``AGENTRUN_ACCESS_KEY_ID`` / ``AGENTRUN_ACCESS_KEY_SECRET``: AccessKey 凭据
- ``AGENTRUN_REGION_ID``: 地域 (默认 cn-hangzhou)
- ``AGENTRUN_WORKSPACE_ID``: 目标工作空间 ID。未配置时本 demo 直接报错退出，
  防止资源被误创建在默认工作空间。
"""

import os
import time
import uuid

from agentrun.credential import (
    Credential,
    CredentialClient,
    CredentialConfig,
    CredentialCreateInput,
    CredentialListInput,
)
from agentrun.sandbox import Template
from agentrun.sandbox.model import PageableInput, TemplateInput, TemplateType
from agentrun.utils.exception import ResourceNotExistError
from agentrun.utils.log import logger


def _require_workspace_id() -> str:
    """读取 workspace_id 环境变量，缺失则直接报错退出，避免误用默认工作空间."""
    workspace_id = os.getenv("AGENTRUN_WORKSPACE_ID")
    if not workspace_id:
        raise SystemExit(
            "AGENTRUN_WORKSPACE_ID is required for this demo. "
            "Set it to a real workspace ID before running."
        )
    return workspace_id


def demo_credential_with_workspace(workspace_id: str) -> None:
    """凭证：在指定 workspace_id 下创建 → 回读 → 列举过滤 → 清理."""
    logger.info("=" * 60)
    logger.info("Credential demo · workspace_id=%s", workspace_id)
    logger.info("=" * 60)

    client = CredentialClient()
    suffix = uuid.uuid4().hex[:8]
    credential_name = f"sdk-ws-demo-cred-{suffix}"

    cred: Credential | None = None
    try:
        # 1. 创建带 workspace_id 的凭证
        cred = Credential.create(
            CredentialCreateInput(
                credential_name=credential_name,
                description="workspace_id demo (safe to delete)",
                credential_config=CredentialConfig.inbound_api_key(
                    "sk-demo-workspace-id"
                ),
                workspace_id=workspace_id,
            )
        )
        logger.info("✓ created credential: %s", cred.credential_name)
        logger.info(
            "  workspace_id (from create response): %s", cred.workspace_id
        )
        assert cred.workspace_id == workspace_id, (
            f"workspace_id mismatch on create: expected={workspace_id!r}, "
            f"got={cred.workspace_id!r}"
        )

        # 2. get 接口回读 workspace_id
        cred_fetched = client.get(credential_name=credential_name)
        logger.info(
            "✓ get returned workspace_id: %s", cred_fetched.workspace_id
        )
        assert cred_fetched.workspace_id == workspace_id, (
            f"workspace_id mismatch on get: expected={workspace_id!r}, "
            f"got={cred_fetched.workspace_id!r}"
        )

        # 3. list 按 workspace_id 过滤
        results = client.list(CredentialListInput(workspace_id=workspace_id))
        names = [item.credential_name for item in results]
        logger.info(
            "✓ list(workspace_id=%s) returned %d items",
            workspace_id,
            len(names),
        )
        assert credential_name in names, (
            f"credential {credential_name!r} not found in workspace-filtered"
            f" list; got {names!r}"
        )
        match = next(
            (
                item
                for item in results
                if item.credential_name == credential_name
            ),
            None,
        )
        assert match is not None
        assert match.workspace_id == workspace_id, (
            f"list item workspace_id mismatch: expected={workspace_id!r}, "
            f"got={match.workspace_id!r}"
        )
        logger.info("✓ workspace_id round-trip verified for credential")
    finally:
        if cred is not None:
            try:
                cred.delete()
                logger.info("✓ cleaned up credential: %s", credential_name)
            except ResourceNotExistError:
                pass


def demo_template_with_workspace(workspace_id: str) -> None:
    """Sandbox Template：在指定 workspace_id 下创建 → 回读 → 列举过滤 → 清理."""
    logger.info("=" * 60)
    logger.info("Sandbox Template demo · workspace_id=%s", workspace_id)
    logger.info("=" * 60)

    suffix = uuid.uuid4().hex[:8]
    template_name = f"sdk-ws-demo-tpl-{suffix}"

    template: Template | None = None
    try:
        # 1. 创建带 workspace_id 的 Template
        template = Template.create(
            TemplateInput(
                template_name=template_name,
                template_type=TemplateType.CODE_INTERPRETER,
                description="workspace_id demo (safe to delete)",
                cpu=2.0,
                memory=4096,
                disk_size=512,
                sandbox_idle_timeout_in_seconds=600,
                sandbox_ttlin_seconds=600,
                workspace_id=workspace_id,
            )
        )
        logger.info("✓ created template: %s", template.template_name)
        logger.info(
            "  workspace_id (from create response): %s", template.workspace_id
        )
        assert template.workspace_id == workspace_id, (
            f"workspace_id mismatch on create: expected={workspace_id!r}, "
            f"got={template.workspace_id!r}"
        )

        # 2. get 接口回读 workspace_id
        fetched = Template.get_by_name(template_name)
        logger.info("✓ get returned workspace_id: %s", fetched.workspace_id)
        assert fetched.workspace_id == workspace_id, (
            f"workspace_id mismatch on get: expected={workspace_id!r}, "
            f"got={fetched.workspace_id!r}"
        )

        # 3. list 按 workspace_id 过滤
        results = Template.list_templates(
            PageableInput(workspace_id=workspace_id, page_size=100)
        )
        names = [t.template_name for t in results or []]
        logger.info(
            "✓ list_templates(workspace_id=%s) returned %d items",
            workspace_id,
            len(names),
        )
        assert template_name in names, (
            f"template {template_name!r} not found in workspace-filtered list; "
            f"got {names!r}"
        )
        logger.info("✓ workspace_id round-trip verified for template")
    finally:
        if template is not None:
            try:
                Template.delete_by_name(template_name)
                logger.info("✓ cleaned up template: %s", template_name)
            except ResourceNotExistError:
                pass


def main() -> None:
    workspace_id = _require_workspace_id()
    logger.info(
        "Running workspace_id demo · region=%s · ws=%s",
        os.getenv("AGENTRUN_REGION_ID", "cn-hangzhou"),
        workspace_id,
    )

    started_at = time.time()
    demo_credential_with_workspace(workspace_id)
    demo_template_with_workspace(workspace_id)
    logger.info("=" * 60)
    logger.info(
        "✓ All workspace_id demos passed in %.1fs", time.time() - started_at
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
