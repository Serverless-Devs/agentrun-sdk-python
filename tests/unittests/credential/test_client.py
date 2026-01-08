"""测试 agentrun.credential.client 模块 / Test agentrun.credential.client module"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.credential.client import CredentialClient
from agentrun.credential.model import (
    CredentialConfig,
    CredentialCreateInput,
    CredentialListInput,
    CredentialSourceType,
    CredentialUpdateInput,
)
from agentrun.utils.config import Config
from agentrun.utils.exception import (
    HTTPError,
    ResourceAlreadyExistError,
    ResourceNotExistError,
)


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


class TestCredentialClientInit:
    """测试 CredentialClient 初始化"""

    def test_init_without_config(self):
        """测试不带配置的初始化"""
        client = CredentialClient()
        assert client is not None

    def test_init_with_config(self):
        """测试带配置的初始化"""
        config = Config(access_key_id="test-ak")
        client = CredentialClient(config=config)
        assert client is not None


class TestCredentialClientCreate:
    """测试 CredentialClient.create 方法"""

    @patch("agentrun.credential.client.CredentialControlAPI")
    def test_create_sync(self, mock_control_api_class):
        """测试同步创建凭证"""
        mock_control_api = MagicMock()
        mock_control_api.create_credential.return_value = MockCredentialData()
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        input_obj = CredentialCreateInput(
            credential_name="test-cred",
            credential_config=CredentialConfig.outbound_llm_api_key(
                "sk-xxx", "openai"
            ),
        )

        result = client.create(input_obj)
        assert result.credential_name == "test-cred"
        assert mock_control_api.create_credential.called

    @patch("agentrun.credential.client.CredentialControlAPI")
    @pytest.mark.asyncio
    async def test_create_async(self, mock_control_api_class):
        """测试异步创建凭证"""
        mock_control_api = MagicMock()
        mock_control_api.create_credential_async = AsyncMock(
            return_value=MockCredentialData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        input_obj = CredentialCreateInput(
            credential_name="test-cred",
            credential_config=CredentialConfig.outbound_llm_api_key(
                "sk-xxx", "openai"
            ),
        )

        result = await client.create_async(input_obj)
        assert result.credential_name == "test-cred"

    @patch("agentrun.credential.client.CredentialControlAPI")
    def test_create_already_exists(self, mock_control_api_class):
        """测试创建已存在的凭证"""
        mock_control_api = MagicMock()
        mock_control_api.create_credential.side_effect = HTTPError(
            status_code=409,
            message="Resource already exists",
            request_id="req-1",
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        input_obj = CredentialCreateInput(
            credential_name="existing-cred",
            credential_config=CredentialConfig.outbound_llm_api_key(
                "sk-xxx", "openai"
            ),
        )

        with pytest.raises(ResourceAlreadyExistError):
            client.create(input_obj)

    @patch("agentrun.credential.client.CredentialControlAPI")
    @pytest.mark.asyncio
    async def test_create_async_already_exists(self, mock_control_api_class):
        """测试异步创建已存在的凭证"""
        mock_control_api = MagicMock()
        mock_control_api.create_credential_async = AsyncMock(
            side_effect=HTTPError(
                status_code=409,
                message="Resource already exists",
                request_id="req-1",
            )
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        input_obj = CredentialCreateInput(
            credential_name="existing-cred",
            credential_config=CredentialConfig.outbound_llm_api_key(
                "sk-xxx", "openai"
            ),
        )

        with pytest.raises(ResourceAlreadyExistError):
            await client.create_async(input_obj)


class TestCredentialClientDelete:
    """测试 CredentialClient.delete 方法"""

    @patch("agentrun.credential.client.CredentialControlAPI")
    def test_delete_sync(self, mock_control_api_class):
        """测试同步删除凭证"""
        mock_control_api = MagicMock()
        mock_control_api.delete_credential.return_value = MockCredentialData()
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        result = client.delete("test-cred")
        assert result is not None
        assert mock_control_api.delete_credential.called

    @patch("agentrun.credential.client.CredentialControlAPI")
    @pytest.mark.asyncio
    async def test_delete_async(self, mock_control_api_class):
        """测试异步删除凭证"""
        mock_control_api = MagicMock()
        mock_control_api.delete_credential_async = AsyncMock(
            return_value=MockCredentialData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        result = await client.delete_async("test-cred")
        assert result is not None

    @patch("agentrun.credential.client.CredentialControlAPI")
    def test_delete_not_exist(self, mock_control_api_class):
        """测试删除不存在的凭证"""
        mock_control_api = MagicMock()
        mock_control_api.delete_credential.side_effect = HTTPError(
            status_code=404,
            message="Resource does not exist",
            request_id="req-1",
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        with pytest.raises(ResourceNotExistError):
            client.delete("nonexistent-cred")

    @patch("agentrun.credential.client.CredentialControlAPI")
    @pytest.mark.asyncio
    async def test_delete_async_not_exist(self, mock_control_api_class):
        """测试异步删除不存在的凭证"""
        mock_control_api = MagicMock()
        mock_control_api.delete_credential_async = AsyncMock(
            side_effect=HTTPError(
                status_code=404,
                message="Resource does not exist",
                request_id="req-1",
            )
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        with pytest.raises(ResourceNotExistError):
            await client.delete_async("nonexistent-cred")


class TestCredentialClientUpdate:
    """测试 CredentialClient.update 方法"""

    @patch("agentrun.credential.client.CredentialControlAPI")
    def test_update_sync(self, mock_control_api_class):
        """测试同步更新凭证"""
        mock_control_api = MagicMock()
        mock_control_api.update_credential.return_value = MockCredentialData()
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        input_obj = CredentialUpdateInput(description="Updated", enabled=False)
        result = client.update("test-cred", input_obj)
        assert result is not None
        assert mock_control_api.update_credential.called

    @patch("agentrun.credential.client.CredentialControlAPI")
    def test_update_sync_with_config(self, mock_control_api_class):
        """测试同步更新凭证（带 credential_config）"""
        mock_control_api = MagicMock()
        mock_control_api.update_credential.return_value = MockCredentialData()
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        input_obj = CredentialUpdateInput(
            description="Updated",
            credential_config=CredentialConfig.outbound_llm_api_key(
                "new-key", "openai"
            ),
        )
        result = client.update("test-cred", input_obj)
        assert result is not None

    @patch("agentrun.credential.client.CredentialControlAPI")
    @pytest.mark.asyncio
    async def test_update_async(self, mock_control_api_class):
        """测试异步更新凭证"""
        mock_control_api = MagicMock()
        mock_control_api.update_credential_async = AsyncMock(
            return_value=MockCredentialData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        input_obj = CredentialUpdateInput(description="Updated")
        result = await client.update_async("test-cred", input_obj)
        assert result is not None

    @patch("agentrun.credential.client.CredentialControlAPI")
    @pytest.mark.asyncio
    async def test_update_async_with_config(self, mock_control_api_class):
        """测试异步更新凭证（带 credential_config）"""
        mock_control_api = MagicMock()
        mock_control_api.update_credential_async = AsyncMock(
            return_value=MockCredentialData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        input_obj = CredentialUpdateInput(
            credential_config=CredentialConfig.outbound_llm_api_key(
                "new-key", "openai"
            )
        )
        result = await client.update_async("test-cred", input_obj)
        assert result is not None

    @patch("agentrun.credential.client.CredentialControlAPI")
    def test_update_not_exist(self, mock_control_api_class):
        """测试更新不存在的凭证"""
        mock_control_api = MagicMock()
        mock_control_api.update_credential.side_effect = HTTPError(
            status_code=404,
            message="Resource does not exist",
            request_id="req-1",
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        input_obj = CredentialUpdateInput(description="Updated")
        with pytest.raises(ResourceNotExistError):
            client.update("nonexistent-cred", input_obj)


class TestCredentialClientGet:
    """测试 CredentialClient.get 方法"""

    @patch("agentrun.credential.client.CredentialControlAPI")
    def test_get_sync(self, mock_control_api_class):
        """测试同步获取凭证"""
        mock_control_api = MagicMock()
        mock_control_api.get_credential.return_value = MockCredentialData()
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        result = client.get("test-cred")
        assert result.credential_name == "test-cred"
        assert mock_control_api.get_credential.called

    @patch("agentrun.credential.client.CredentialControlAPI")
    @pytest.mark.asyncio
    async def test_get_async(self, mock_control_api_class):
        """测试异步获取凭证"""
        mock_control_api = MagicMock()
        mock_control_api.get_credential_async = AsyncMock(
            return_value=MockCredentialData()
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        result = await client.get_async("test-cred")
        assert result.credential_name == "test-cred"

    @patch("agentrun.credential.client.CredentialControlAPI")
    def test_get_not_exist(self, mock_control_api_class):
        """测试获取不存在的凭证"""
        mock_control_api = MagicMock()
        mock_control_api.get_credential.side_effect = HTTPError(
            status_code=404,
            message="Resource does not exist",
            request_id="req-1",
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        with pytest.raises(ResourceNotExistError):
            client.get("nonexistent-cred")

    @patch("agentrun.credential.client.CredentialControlAPI")
    @pytest.mark.asyncio
    async def test_get_async_not_exist(self, mock_control_api_class):
        """测试异步获取不存在的凭证"""
        mock_control_api = MagicMock()
        mock_control_api.get_credential_async = AsyncMock(
            side_effect=HTTPError(
                status_code=404,
                message="Resource does not exist",
                request_id="req-1",
            )
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        with pytest.raises(ResourceNotExistError):
            await client.get_async("nonexistent-cred")


class TestCredentialClientList:
    """测试 CredentialClient.list 方法"""

    @patch("agentrun.credential.client.CredentialControlAPI")
    def test_list_sync(self, mock_control_api_class):
        """测试同步列出凭证"""
        mock_control_api = MagicMock()
        mock_control_api.list_credentials.return_value = MockListResult([
            MockCredentialData(),
            MockCredentialData(),
        ])
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        result = client.list()
        assert len(result) == 2
        assert mock_control_api.list_credentials.called

    @patch("agentrun.credential.client.CredentialControlAPI")
    def test_list_sync_with_input(self, mock_control_api_class):
        """测试同步列出凭证（带输入参数）"""
        mock_control_api = MagicMock()
        mock_control_api.list_credentials.return_value = MockListResult(
            [MockCredentialData()]
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        input_obj = CredentialListInput(
            page_number=1,
            page_size=10,
            credential_source_type=CredentialSourceType.LLM,
        )
        result = client.list(input=input_obj)
        assert len(result) == 1

    @patch("agentrun.credential.client.CredentialControlAPI")
    @pytest.mark.asyncio
    async def test_list_async(self, mock_control_api_class):
        """测试异步列出凭证"""
        mock_control_api = MagicMock()
        mock_control_api.list_credentials_async = AsyncMock(
            return_value=MockListResult([MockCredentialData()])
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        result = await client.list_async()
        assert len(result) == 1

    @patch("agentrun.credential.client.CredentialControlAPI")
    @pytest.mark.asyncio
    async def test_list_async_with_input(self, mock_control_api_class):
        """测试异步列出凭证（带输入参数）"""
        mock_control_api = MagicMock()
        mock_control_api.list_credentials_async = AsyncMock(
            return_value=MockListResult([MockCredentialData()])
        )
        mock_control_api_class.return_value = mock_control_api

        client = CredentialClient()
        input_obj = CredentialListInput(page_number=1, page_size=10)
        result = await client.list_async(input=input_obj)
        assert len(result) == 1
