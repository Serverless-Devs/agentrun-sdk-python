"""
Credential 模块的 E2E 测试

测试覆盖:
- 创建凭证
- 获取凭证
- 列举凭证
- 更新凭证
- 删除凭证
"""

import datetime
import time

import pytest

from agentrun.credential import (
    Credential,
    CredentialClient,
    CredentialConfig,
    CredentialCreateInput,
    CredentialUpdateInput,
)
from agentrun.utils.exception import (
    ResourceAlreadyExistError,
    ResourceNotExistError,
)


class TestCredential:
    """凭证模块 E2E 测试"""

    @pytest.fixture
    def credential_name(self, unique_name: str) -> str:
        """生成凭证名称"""
        return f"{unique_name}-cred"

    async def test_credential_lifecycle_async(self, credential_name: str):
        """测试凭证的完整生命周期"""
        client = CredentialClient()

        time1 = datetime.datetime.now(datetime.timezone.utc)

        # 创建 credential
        cred = await Credential.create_async(
            CredentialCreateInput(
                credential_name=credential_name,
                description="原始描述",
                credential_config=CredentialConfig.inbound_api_key(
                    "sk-test-e2e-123456"
                ),
            )
        )

        cred2 = await client.get_async(credential_name=credential_name)

        # 检查返回的内容是否符合预期
        pre_created_at: datetime.datetime

        def assert_cred(cred: Credential):
            assert cred.status is None
            assert cred.credential_auth_type == "api_key"
            assert cred.credential_source_type == "internal"
            assert cred.credential_public_config is not None
            assert cred.credential_public_config["headerKey"] == "Authorization"
            assert cred.credential_public_config["users"] == []
            assert cred.credential_secret == "sk-test-e2e-123456"
            assert type(cred.credential_id) is str and cred.credential_id != ""
            assert cred.created_at is not None
            created_at = datetime.datetime.strptime(
                cred.created_at, "%Y-%m-%dT%H:%M:%S.%f%z"
            )
            assert created_at > time1
            assert cred.updated_at is not None
            updated_at = datetime.datetime.strptime(
                cred.updated_at, "%Y-%m-%dT%H:%M:%S.%f%z"
            )
            assert updated_at == created_at
            assert cred.credential_name == credential_name
            assert cred.description == "原始描述"
            assert cred.enabled is True

            nonlocal pre_created_at
            pre_created_at = created_at

        assert_cred(cred)
        assert_cred(cred2)
        assert cred is not cred2
        cred3 = cred

        # 更新 credential
        new_description = f"更新后的描述 - {time.time()}"
        await cred.update_async(
            CredentialUpdateInput(
                description=new_description,
                enabled=False,
                credential_config=CredentialConfig.inbound_api_key(
                    "sk-test-654321"
                ),
            )
        )

        # 检查返回的内容是否符合预期
        def assert_cred2(cred: Credential):
            nonlocal pre_created_at
            assert cred.status is None
            assert cred.credential_auth_type == "api_key"
            assert cred.credential_source_type == "internal"
            assert cred.credential_public_config is not None
            assert cred.credential_public_config["headerKey"] == "Authorization"
            assert cred.credential_public_config["users"] == []
            assert cred.credential_secret == "sk-test-654321"
            assert type(cred.credential_id) is str and cred.credential_id != ""
            assert cred.created_at is not None
            created_at = datetime.datetime.strptime(
                cred.created_at, "%Y-%m-%dT%H:%M:%S.%f%z"
            )
            assert pre_created_at == created_at
            assert created_at > time1
            assert cred.updated_at is not None
            updated_at = datetime.datetime.strptime(
                cred.updated_at, "%Y-%m-%dT%H:%M:%S.%f%z"
            )
            assert updated_at > created_at
            assert cred.credential_name == credential_name
            assert cred.description == new_description
            assert cred.enabled is False

        assert_cred2(cred)
        assert_cred2(cred3)
        assert_cred(cred2)
        assert cred3 is cred

        # 获取 credential
        await cred2.refresh_async()
        assert_cred2(cred2)

        # 列举 credentials
        credentials = await Credential.list_all_async()
        assert len(credentials) > 0
        matched_cred = 0
        for c in credentials:
            if c.credential_name == credential_name:
                matched_cred += 1
                assert_cred2(await c.to_credential_async())
        assert matched_cred == 1

        # 尝试重复创建
        with pytest.raises(ResourceAlreadyExistError):
            await client.create_async(
                CredentialCreateInput(
                    credential_name=credential_name,
                    description="重复的凭证",
                    credential_config=CredentialConfig.inbound_api_key(
                        "sk-test-duplicate"
                    ),
                )
            )

        # 删除
        await cred.delete_async()

        # 尝试重复删除
        with pytest.raises(ResourceNotExistError):
            await cred.delete_async()

        # 验证删除
        with pytest.raises(ResourceNotExistError):
            await client.get_async(credential_name=credential_name)
