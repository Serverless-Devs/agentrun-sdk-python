"""
workspace_id 跨模块 E2E 测试 / Cross-module workspace_id E2E test

验证 SDK 在 create / get / list 接口上正确传递和回填 ``workspace_id``。
Verifies the SDK correctly passes and back-fills ``workspace_id`` across
create / get / list interfaces for resource modules.

环境变量 / Environment variables:
- ``AGENTRUN_TEST_WORKSPACE_ID``：用于本测试的工作空间 ID。未配置则跳过整个文件。
  Workspace ID to use for this test; the entire file is skipped if not set.
"""

import os

import pytest

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

WORKSPACE_ID = os.getenv("AGENTRUN_TEST_WORKSPACE_ID")

pytestmark = pytest.mark.skipif(
    not WORKSPACE_ID,
    reason=(
        "AGENTRUN_TEST_WORKSPACE_ID not configured; skipping workspace_id E2E"
    ),
)


class TestWorkspaceId:
    """workspace_id 跨模块 E2E 测试"""

    @pytest.fixture
    def credential_name(self, unique_name: str) -> str:
        return f"{unique_name}-ws-cred"

    @pytest.fixture
    def template_name(self, unique_name: str) -> str:
        return f"{unique_name}-ws-tpl"

    async def test_credential_with_workspace_id_async(
        self, credential_name: str
    ):
        """凭证创建时指定 workspace_id，回读与列举均能拿到该 workspace_id"""
        client = CredentialClient()
        ws = WORKSPACE_ID  # type: ignore[assignment]
        assert ws is not None

        cred: Credential | None = None
        try:
            # 1. 创建带 workspace_id 的凭证
            cred = await Credential.create_async(
                CredentialCreateInput(
                    credential_name=credential_name,
                    description="E2E workspace_id test",
                    credential_config=CredentialConfig.inbound_api_key(
                        "sk-test-ws-e2e"
                    ),
                    workspace_id=ws,
                )
            )
            assert cred.credential_name == credential_name
            assert (
                cred.workspace_id == ws
            ), f"create 返回的 workspace_id 不匹配: {cred.workspace_id!r}"

            # 2. get 接口回读 workspace_id
            cred_fetched = await client.get_async(
                credential_name=credential_name
            )
            assert (
                cred_fetched.workspace_id == ws
            ), f"get 返回的 workspace_id 不匹配: {cred_fetched.workspace_id!r}"

            # 3. list 接口按 workspace_id 过滤，本次创建的资源应在结果中
            list_results = await client.list_async(
                CredentialListInput(workspace_id=ws)
            )
            names = [item.credential_name for item in list_results]
            assert credential_name in names, (
                f"list(workspace_id={ws!r}) 未返回刚创建的凭证"
                f" {credential_name!r}，"
                f"实际返回 {names!r}"
            )
            # 列表项的 workspace_id 也应该是同一个
            for item in list_results:
                if item.credential_name == credential_name:
                    assert item.workspace_id == ws
        finally:
            if cred is not None:
                try:
                    await cred.delete_async()
                except ResourceNotExistError:
                    pass

    async def test_template_with_workspace_id_async(self, template_name: str):
        """Sandbox Template 创建时指定 workspace_id，回读与列举均能拿到该 workspace_id"""
        ws = WORKSPACE_ID  # type: ignore[assignment]
        assert ws is not None

        template: Template | None = None
        try:
            # 1. 创建带 workspace_id 的 Template
            template = await Template.create_async(
                TemplateInput(
                    template_name=template_name,
                    template_type=TemplateType.CODE_INTERPRETER,
                    description="E2E workspace_id test",
                    cpu=2.0,
                    memory=4096,
                    disk_size=512,
                    sandbox_idle_timeout_in_seconds=600,
                    sandbox_ttlin_seconds=600,
                    workspace_id=ws,
                )
            )
            assert template.template_name == template_name
            assert (
                template.workspace_id == ws
            ), f"create 返回的 workspace_id 不匹配: {template.workspace_id!r}"

            # 2. get 接口回读 workspace_id
            template_fetched = await Template.get_by_name_async(template_name)
            assert (
                template_fetched.workspace_id == ws
            ), f"get 返回的 workspace_id 不匹配: {template_fetched.workspace_id!r}"

            # 3. list 接口按 workspace_id 过滤
            list_results = await Template.list_templates_async(
                PageableInput(workspace_id=ws, page_size=100)
            )
            names = [t.template_name for t in list_results or []]
            assert template_name in names, (
                f"list_templates(workspace_id={ws!r}) 未返回刚创建的"
                f" Template {template_name!r}，实际返回 {names!r}"
            )
        finally:
            if template is not None:
                try:
                    await Template.delete_by_name_async(template_name)
                except ResourceNotExistError:
                    pass
