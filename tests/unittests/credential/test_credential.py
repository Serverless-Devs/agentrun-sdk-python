"""测试 agentrun.credential.credential 模块 / Test agentrun.credential.credential module"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.credential.credential import Credential
from agentrun.credential.model import (
    CredentialConfig,
    CredentialCreateInput,
    CredentialUpdateInput,
)
from agentrun.utils.config import Config


class MockCredentialData:
    """模拟凭证数据"""

    def to_map(self):
        return {
            "credentialId": "cred-123",
            "credentialName": "test-cred",
            "credentialAuthType": "api_key",
            "credentialSourceType": "external_llm",
            "enabled": True,
        }


class MockListResult:
    """模拟列表结果"""

    def __init__(self, items):
        self.items = items


# CredentialClient 是在 Credential 的方法内部延迟导入的，所以需要 patch 正确的路径
CREDENTIAL_CLIENT_PATH = "agentrun.credential.client.CredentialClient"


class TestCredentialClassMethods:
    """测试 Credential 类方法"""

    @patch(CREDENTIAL_CLIENT_PATH)
    def test_create_sync(self, mock_client_class):
        """测试同步创建凭证"""
        mock_client = MagicMock()
        mock_credential = Credential(
            credential_name="test-cred", credential_id="cred-123"
        )
        mock_client.create.return_value = mock_credential
        mock_client_class.return_value = mock_client

        input_obj = CredentialCreateInput(
            credential_name="test-cred",
            credential_config=CredentialConfig.outbound_llm_api_key(
                "sk-xxx", "openai"
            ),
        )
        result = Credential.create(input_obj)
        assert result.credential_name == "test-cred"

    @patch(CREDENTIAL_CLIENT_PATH)
    @pytest.mark.asyncio
    async def test_create_async(self, mock_client_class):
        """测试异步创建凭证"""
        mock_client = MagicMock()
        mock_credential = Credential(
            credential_name="test-cred", credential_id="cred-123"
        )
        mock_client.create_async = AsyncMock(return_value=mock_credential)
        mock_client_class.return_value = mock_client

        input_obj = CredentialCreateInput(
            credential_name="test-cred",
            credential_config=CredentialConfig.outbound_llm_api_key(
                "sk-xxx", "openai"
            ),
        )
        result = await Credential.create_async(input_obj)
        assert result.credential_name == "test-cred"

    @patch(CREDENTIAL_CLIENT_PATH)
    def test_delete_by_name_sync(self, mock_client_class):
        """测试同步按名称删除凭证"""
        mock_client = MagicMock()
        mock_client.delete.return_value = None
        mock_client_class.return_value = mock_client

        Credential.delete_by_name("test-cred")
        mock_client.delete.assert_called_once()

    @patch(CREDENTIAL_CLIENT_PATH)
    @pytest.mark.asyncio
    async def test_delete_by_name_async(self, mock_client_class):
        """测试异步按名称删除凭证"""
        mock_client = MagicMock()
        mock_client.delete_async = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        await Credential.delete_by_name_async("test-cred")
        mock_client.delete_async.assert_called_once()

    @patch(CREDENTIAL_CLIENT_PATH)
    def test_update_by_name_sync(self, mock_client_class):
        """测试同步按名称更新凭证"""
        mock_client = MagicMock()
        mock_credential = Credential(credential_name="test-cred")
        mock_client.update.return_value = mock_credential
        mock_client_class.return_value = mock_client

        input_obj = CredentialUpdateInput(description="Updated")
        result = Credential.update_by_name("test-cred", input_obj)
        assert result is not None

    @patch(CREDENTIAL_CLIENT_PATH)
    @pytest.mark.asyncio
    async def test_update_by_name_async(self, mock_client_class):
        """测试异步按名称更新凭证"""
        mock_client = MagicMock()
        mock_credential = Credential(credential_name="test-cred")
        mock_client.update_async = AsyncMock(return_value=mock_credential)
        mock_client_class.return_value = mock_client

        input_obj = CredentialUpdateInput(description="Updated")
        result = await Credential.update_by_name_async("test-cred", input_obj)
        assert result is not None

    @patch(CREDENTIAL_CLIENT_PATH)
    def test_get_by_name_sync(self, mock_client_class):
        """测试同步按名称获取凭证"""
        mock_client = MagicMock()
        mock_credential = Credential(
            credential_name="test-cred", credential_id="cred-123"
        )
        mock_client.get.return_value = mock_credential
        mock_client_class.return_value = mock_client

        result = Credential.get_by_name("test-cred")
        assert result.credential_name == "test-cred"

    @patch(CREDENTIAL_CLIENT_PATH)
    @pytest.mark.asyncio
    async def test_get_by_name_async(self, mock_client_class):
        """测试异步按名称获取凭证"""
        mock_client = MagicMock()
        mock_credential = Credential(
            credential_name="test-cred", credential_id="cred-123"
        )
        mock_client.get_async = AsyncMock(return_value=mock_credential)
        mock_client_class.return_value = mock_client

        result = await Credential.get_by_name_async("test-cred")
        assert result.credential_name == "test-cred"


class TestCredentialInstanceMethods:
    """测试 Credential 实例方法"""

    @patch(CREDENTIAL_CLIENT_PATH)
    def test_update_sync(self, mock_client_class):
        """测试同步更新凭证实例"""
        mock_client = MagicMock()
        mock_updated = Credential(
            credential_name="test-cred", description="Updated"
        )
        mock_client.update.return_value = mock_updated
        mock_client_class.return_value = mock_client

        credential = Credential(credential_name="test-cred")
        input_obj = CredentialUpdateInput(description="Updated")
        result = credential.update(input_obj)
        assert result is credential
        assert credential.description == "Updated"

    @patch(CREDENTIAL_CLIENT_PATH)
    @pytest.mark.asyncio
    async def test_update_async(self, mock_client_class):
        """测试异步更新凭证实例"""
        mock_client = MagicMock()
        mock_updated = Credential(
            credential_name="test-cred", description="Updated"
        )
        mock_client.update_async = AsyncMock(return_value=mock_updated)
        mock_client_class.return_value = mock_client

        credential = Credential(credential_name="test-cred")
        input_obj = CredentialUpdateInput(description="Updated")
        result = await credential.update_async(input_obj)
        assert result is credential

    def test_update_without_name_raises(self):
        """测试没有名称时更新抛出异常"""
        credential = Credential()
        input_obj = CredentialUpdateInput(description="Updated")
        with pytest.raises(ValueError) as exc_info:
            credential.update(input_obj)
        assert "credential_name is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_async_without_name_raises(self):
        """测试没有名称时异步更新抛出异常"""
        credential = Credential()
        input_obj = CredentialUpdateInput(description="Updated")
        with pytest.raises(ValueError) as exc_info:
            await credential.update_async(input_obj)
        assert "credential_name is required" in str(exc_info.value)

    @patch(CREDENTIAL_CLIENT_PATH)
    def test_delete_sync(self, mock_client_class):
        """测试同步删除凭证实例"""
        mock_client = MagicMock()
        mock_client.delete.return_value = None
        mock_client_class.return_value = mock_client

        credential = Credential(credential_name="test-cred")
        credential.delete()
        mock_client.delete.assert_called_once()

    @patch(CREDENTIAL_CLIENT_PATH)
    @pytest.mark.asyncio
    async def test_delete_async(self, mock_client_class):
        """测试异步删除凭证实例"""
        mock_client = MagicMock()
        mock_client.delete_async = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        credential = Credential(credential_name="test-cred")
        await credential.delete_async()
        mock_client.delete_async.assert_called_once()

    def test_delete_without_name_raises(self):
        """测试没有名称时删除抛出异常"""
        credential = Credential()
        with pytest.raises(ValueError) as exc_info:
            credential.delete()
        assert "credential_name is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_async_without_name_raises(self):
        """测试没有名称时异步删除抛出异常"""
        credential = Credential()
        with pytest.raises(ValueError) as exc_info:
            await credential.delete_async()
        assert "credential_name is required" in str(exc_info.value)

    @patch(CREDENTIAL_CLIENT_PATH)
    def test_get_sync(self, mock_client_class):
        """测试同步刷新凭证实例"""
        mock_client = MagicMock()
        mock_refreshed = Credential(credential_name="test-cred", enabled=True)
        mock_client.get.return_value = mock_refreshed
        mock_client_class.return_value = mock_client

        credential = Credential(credential_name="test-cred", enabled=False)
        result = credential.get()
        assert result is credential
        assert credential.enabled is True

    @patch(CREDENTIAL_CLIENT_PATH)
    @pytest.mark.asyncio
    async def test_get_async(self, mock_client_class):
        """测试异步刷新凭证实例"""
        mock_client = MagicMock()
        mock_refreshed = Credential(credential_name="test-cred", enabled=True)
        mock_client.get_async = AsyncMock(return_value=mock_refreshed)
        mock_client_class.return_value = mock_client

        credential = Credential(credential_name="test-cred", enabled=False)
        result = await credential.get_async()
        assert result is credential

    def test_get_without_name_raises(self):
        """测试没有名称时刷新抛出异常"""
        credential = Credential()
        with pytest.raises(ValueError) as exc_info:
            credential.get()
        assert "credential_name is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_async_without_name_raises(self):
        """测试没有名称时异步刷新抛出异常"""
        credential = Credential()
        with pytest.raises(ValueError) as exc_info:
            await credential.get_async()
        assert "credential_name is required" in str(exc_info.value)

    @patch(CREDENTIAL_CLIENT_PATH)
    def test_refresh_sync(self, mock_client_class):
        """测试同步 refresh 方法"""
        mock_client = MagicMock()
        mock_refreshed = Credential(credential_name="test-cred")
        mock_client.get.return_value = mock_refreshed
        mock_client_class.return_value = mock_client

        credential = Credential(credential_name="test-cred")
        result = credential.refresh()
        assert result is credential

    @patch(CREDENTIAL_CLIENT_PATH)
    @pytest.mark.asyncio
    async def test_refresh_async(self, mock_client_class):
        """测试异步 refresh 方法"""
        mock_client = MagicMock()
        mock_refreshed = Credential(credential_name="test-cred")
        mock_client.get_async = AsyncMock(return_value=mock_refreshed)
        mock_client_class.return_value = mock_client

        credential = Credential(credential_name="test-cred")
        result = await credential.refresh_async()
        assert result is credential


class TestCredentialListMethods:
    """测试 Credential 列表方法"""

    @patch(CREDENTIAL_CLIENT_PATH)
    def test_list_page_sync(self, mock_client_class):
        """测试同步列表分页"""
        mock_client = MagicMock()
        mock_client.list.return_value = [
            Credential(credential_name="cred-1", credential_id="id-1"),
            Credential(credential_name="cred-2", credential_id="id-2"),
        ]
        mock_client_class.return_value = mock_client

        from agentrun.utils.model import PageableInput

        result = Credential._list_page(
            PageableInput(page_number=1, page_size=10)
        )
        assert len(result) == 2

    @patch(CREDENTIAL_CLIENT_PATH)
    @pytest.mark.asyncio
    async def test_list_page_async(self, mock_client_class):
        """测试异步列表分页"""
        mock_client = MagicMock()
        mock_client.list_async = AsyncMock(
            return_value=[
                Credential(credential_name="cred-1", credential_id="id-1"),
            ]
        )
        mock_client_class.return_value = mock_client

        from agentrun.utils.model import PageableInput

        result = await Credential._list_page_async(
            PageableInput(page_number=1, page_size=10)
        )
        assert len(result) == 1

    @patch(CREDENTIAL_CLIENT_PATH)
    def test_list_all_sync(self, mock_client_class):
        """测试同步列出所有凭证"""
        mock_client = MagicMock()
        # 第一页返回数据，第二页返回空
        mock_client.list.side_effect = [
            [Credential(credential_name="cred-1", credential_id="id-1")],
            [],
        ]
        mock_client_class.return_value = mock_client

        result = Credential.list_all()
        assert len(result) == 1

    @patch(CREDENTIAL_CLIENT_PATH)
    @pytest.mark.asyncio
    async def test_list_all_async(self, mock_client_class):
        """测试异步列出所有凭证"""
        mock_client = MagicMock()
        mock_client.list_async = AsyncMock(
            side_effect=[
                [Credential(credential_name="cred-1", credential_id="id-1")],
                [],
            ]
        )
        mock_client_class.return_value = mock_client

        result = await Credential.list_all_async()
        assert len(result) == 1
