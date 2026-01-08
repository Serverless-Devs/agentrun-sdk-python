"""ToolSet 客户端单元测试 / ToolSet Client Unit Tests

测试 ToolSetClient 的相关功能。
Tests ToolSetClient functionality.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.toolset.client import ToolSetClient
from agentrun.toolset.model import ToolSetListInput
from agentrun.utils.config import Config
from agentrun.utils.exception import HTTPError, ResourceNotExistError


class TestToolSetClientInit:
    """测试 ToolSetClient 初始化"""

    @patch("agentrun.toolset.client.ToolControlAPI")
    def test_init_without_config(self, mock_control_api):
        """测试不带配置初始化"""
        client = ToolSetClient()
        mock_control_api.assert_called_once_with(None)

    @patch("agentrun.toolset.client.ToolControlAPI")
    def test_init_with_config(self, mock_control_api):
        """测试带配置初始化"""
        config = Config(
            access_key_id="test-key",
            access_key_secret="test-secret",
        )
        client = ToolSetClient(config)
        mock_control_api.assert_called_once_with(config)


class TestToolSetClientGet:
    """测试 ToolSetClient.get 方法"""

    @patch("agentrun.toolset.client.ToolControlAPI")
    def test_get_success(self, mock_control_api_class):
        """测试成功获取 ToolSet"""
        # 设置 mock
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_toolset = MagicMock()
        mock_toolset.name = "test-toolset"
        mock_toolset.uid = "uid-123"
        mock_control_api.get_toolset.return_value = mock_toolset

        # 执行测试
        client = ToolSetClient()
        result = client.get(name="test-toolset")

        # 验证
        mock_control_api.get_toolset.assert_called_once_with(
            name="test-toolset",
            config=None,
        )
        assert result is not None

    @patch("agentrun.toolset.client.ToolControlAPI")
    def test_get_with_config(self, mock_control_api_class):
        """测试带配置获取 ToolSet"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_toolset = MagicMock()
        mock_toolset.name = "test-toolset"
        mock_control_api.get_toolset.return_value = mock_toolset

        config = Config(
            access_key_id="test-key",
            access_key_secret="test-secret",
        )

        client = ToolSetClient()
        result = client.get(name="test-toolset", config=config)

        mock_control_api.get_toolset.assert_called_once_with(
            name="test-toolset",
            config=config,
        )

    @patch("agentrun.toolset.client.ToolControlAPI")
    def test_get_not_found(self, mock_control_api_class):
        """测试 ToolSet 不存在"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        # 模拟 HTTPError 并转换为资源错误
        # message 需要包含 "not found"（小写）才能被 to_resource_error 识别
        http_error = HTTPError(404, "resource not found")
        mock_control_api.get_toolset.side_effect = http_error

        client = ToolSetClient()
        with pytest.raises(ResourceNotExistError):
            client.get(name="non-existent-toolset")


class TestToolSetClientGetAsync:
    """测试 ToolSetClient.get_async 方法"""

    @pytest.mark.asyncio
    @patch("agentrun.toolset.client.ToolControlAPI")
    async def test_get_async_success(self, mock_control_api_class):
        """测试异步成功获取 ToolSet"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_toolset = MagicMock()
        mock_toolset.name = "test-toolset"
        mock_toolset.uid = "uid-123"
        mock_control_api.get_toolset_async = AsyncMock(
            return_value=mock_toolset
        )

        client = ToolSetClient()
        result = await client.get_async(name="test-toolset")

        mock_control_api.get_toolset_async.assert_called_once_with(
            name="test-toolset",
            config=None,
        )
        assert result is not None

    @pytest.mark.asyncio
    @patch("agentrun.toolset.client.ToolControlAPI")
    async def test_get_async_not_found(self, mock_control_api_class):
        """测试异步 ToolSet 不存在"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        http_error = HTTPError(404, "resource not found")
        mock_control_api.get_toolset_async = AsyncMock(side_effect=http_error)

        client = ToolSetClient()
        with pytest.raises(ResourceNotExistError):
            await client.get_async(name="non-existent-toolset")


class TestToolSetClientList:
    """测试 ToolSetClient.list 方法"""

    @patch("agentrun.toolset.client.ToolControlAPI")
    def test_list_without_input(self, mock_control_api_class):
        """测试不带输入列表 ToolSets"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_response = MagicMock()
        mock_response.data = [MagicMock(), MagicMock()]
        mock_control_api.list_toolsets.return_value = mock_response

        client = ToolSetClient()
        result = client.list()

        assert len(result) == 2
        mock_control_api.list_toolsets.assert_called_once()

    @patch("agentrun.toolset.client.ToolControlAPI")
    def test_list_with_input(self, mock_control_api_class):
        """测试带输入列表 ToolSets"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_control_api.list_toolsets.return_value = mock_response

        input_obj = ToolSetListInput(keyword="test")
        client = ToolSetClient()
        result = client.list(input=input_obj)

        assert len(result) == 1
        mock_control_api.list_toolsets.assert_called_once()

    @patch("agentrun.toolset.client.ToolControlAPI")
    def test_list_with_config(self, mock_control_api_class):
        """测试带配置列表 ToolSets"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_response = MagicMock()
        mock_response.data = []
        mock_control_api.list_toolsets.return_value = mock_response

        config = Config(
            access_key_id="test-key",
            access_key_secret="test-secret",
        )

        client = ToolSetClient()
        result = client.list(config=config)

        assert result == []


class TestToolSetClientListAsync:
    """测试 ToolSetClient.list_async 方法"""

    @pytest.mark.asyncio
    @patch("agentrun.toolset.client.ToolControlAPI")
    async def test_list_async_without_input(self, mock_control_api_class):
        """测试异步不带输入列表 ToolSets"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_response = MagicMock()
        mock_response.data = [MagicMock(), MagicMock()]
        mock_control_api.list_toolsets_async = AsyncMock(
            return_value=mock_response
        )

        client = ToolSetClient()
        result = await client.list_async()

        assert len(result) == 2

    @pytest.mark.asyncio
    @patch("agentrun.toolset.client.ToolControlAPI")
    async def test_list_async_with_input(self, mock_control_api_class):
        """测试异步带输入列表 ToolSets"""
        mock_control_api = MagicMock()
        mock_control_api_class.return_value = mock_control_api

        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_control_api.list_toolsets_async = AsyncMock(
            return_value=mock_response
        )

        input_obj = ToolSetListInput(keyword="test")
        client = ToolSetClient()
        result = await client.list_async(input=input_obj)

        assert len(result) == 1
