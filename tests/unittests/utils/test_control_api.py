"""测试 agentrun.utils.control_api 模块 / Test agentrun.utils.control_api module"""

import os
from unittest.mock import MagicMock, patch

import pytest

from agentrun.utils.config import Config
from agentrun.utils.control_api import ControlAPI


class TestControlAPIInit:
    """测试 ControlAPI 初始化"""

    def test_init_without_config(self):
        """测试不带配置的初始化"""
        api = ControlAPI()
        assert api.config is None

    def test_init_with_config(self):
        """测试带配置的初始化"""
        config = Config(access_key_id="test-ak")
        api = ControlAPI(config=config)
        assert api.config is config


class TestControlAPIGetClient:
    """测试 ControlAPI._get_client"""

    @patch("agentrun.utils.control_api.AgentRunClient")
    def test_get_client_basic(self, mock_client_class):
        """测试获取基本客户端"""
        config = Config(
            access_key_id="ak",
            access_key_secret="sk",
            region_id="cn-hangzhou",
            control_endpoint="https://agentrun.cn-hangzhou.aliyuncs.com",
        )
        api = ControlAPI(config=config)

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        result = api._get_client()

        assert mock_client_class.called
        call_args = mock_client_class.call_args
        config_arg = call_args[0][0]
        assert config_arg.access_key_id == "ak"
        assert config_arg.access_key_secret == "sk"
        assert config_arg.region_id == "cn-hangzhou"

    @patch("agentrun.utils.control_api.AgentRunClient")
    def test_get_client_strips_http_prefix(self, mock_client_class):
        """测试获取客户端时去除 http:// 前缀"""
        config = Config(
            access_key_id="ak",
            access_key_secret="sk",
            control_endpoint="http://custom.endpoint.com",
        )
        api = ControlAPI(config=config)

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        api._get_client()

        call_args = mock_client_class.call_args
        config_arg = call_args[0][0]
        assert config_arg.endpoint == "custom.endpoint.com"

    @patch("agentrun.utils.control_api.AgentRunClient")
    def test_get_client_strips_https_prefix(self, mock_client_class):
        """测试获取客户端时去除 https:// 前缀"""
        config = Config(
            access_key_id="ak",
            access_key_secret="sk",
            control_endpoint="https://custom.endpoint.com",
        )
        api = ControlAPI(config=config)

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        api._get_client()

        call_args = mock_client_class.call_args
        config_arg = call_args[0][0]
        assert config_arg.endpoint == "custom.endpoint.com"

    @patch("agentrun.utils.control_api.AgentRunClient")
    def test_get_client_with_override_config(self, mock_client_class):
        """测试使用覆盖配置获取客户端"""
        base_config = Config(
            access_key_id="base-ak",
            access_key_secret="base-sk",
            region_id="cn-hangzhou",
        )
        override_config = Config(
            access_key_id="override-ak",
            region_id="cn-shanghai",
        )
        api = ControlAPI(config=base_config)

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        api._get_client(config=override_config)

        call_args = mock_client_class.call_args
        config_arg = call_args[0][0]
        assert config_arg.access_key_id == "override-ak"
        assert config_arg.region_id == "cn-shanghai"

    @patch("agentrun.utils.control_api.AgentRunClient")
    def test_get_client_without_protocol_prefix(self, mock_client_class):
        """测试获取客户端时 endpoint 不带协议前缀"""
        config = Config(
            access_key_id="ak",
            access_key_secret="sk",
            control_endpoint="custom.endpoint.com",
        )
        api = ControlAPI(config=config)

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        api._get_client()

        call_args = mock_client_class.call_args
        config_arg = call_args[0][0]
        # endpoint 不带协议时应该保持原样
        assert config_arg.endpoint == "custom.endpoint.com"

    @patch("agentrun.utils.control_api.AgentRunClient")
    def test_get_client_with_security_token(self, mock_client_class):
        """测试使用安全令牌获取客户端"""
        config = Config(
            access_key_id="ak",
            access_key_secret="sk",
            security_token="sts-token",
        )
        api = ControlAPI(config=config)

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        api._get_client()

        call_args = mock_client_class.call_args
        config_arg = call_args[0][0]
        assert config_arg.security_token == "sts-token"


class TestControlAPIGetDevsClient:
    """测试 ControlAPI._get_devs_client"""

    @patch("agentrun.utils.control_api.DevsClient")
    def test_get_devs_client_basic(self, mock_client_class):
        """测试获取基本 Devs 客户端"""
        config = Config(
            access_key_id="ak",
            access_key_secret="sk",
            region_id="cn-hangzhou",
            devs_endpoint="https://devs.cn-hangzhou.aliyuncs.com",
        )
        api = ControlAPI(config=config)

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        result = api._get_devs_client()

        assert mock_client_class.called
        call_args = mock_client_class.call_args
        config_arg = call_args[0][0]
        assert config_arg.access_key_id == "ak"
        assert config_arg.access_key_secret == "sk"
        assert config_arg.region_id == "cn-hangzhou"

    @patch("agentrun.utils.control_api.DevsClient")
    def test_get_devs_client_strips_http_prefix(self, mock_client_class):
        """测试获取 Devs 客户端时去除 http:// 前缀"""
        config = Config(
            access_key_id="ak",
            access_key_secret="sk",
            devs_endpoint="http://devs.custom.com",
        )
        api = ControlAPI(config=config)

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        api._get_devs_client()

        call_args = mock_client_class.call_args
        config_arg = call_args[0][0]
        assert config_arg.endpoint == "devs.custom.com"

    @patch("agentrun.utils.control_api.DevsClient")
    def test_get_devs_client_strips_https_prefix(self, mock_client_class):
        """测试获取 Devs 客户端时去除 https:// 前缀"""
        config = Config(
            access_key_id="ak",
            access_key_secret="sk",
            devs_endpoint="https://devs.custom.com",
        )
        api = ControlAPI(config=config)

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        api._get_devs_client()

        call_args = mock_client_class.call_args
        config_arg = call_args[0][0]
        assert config_arg.endpoint == "devs.custom.com"

    @patch("agentrun.utils.control_api.DevsClient")
    def test_get_devs_client_with_override_config(self, mock_client_class):
        """测试使用覆盖配置获取 Devs 客户端"""
        base_config = Config(
            access_key_id="base-ak",
            access_key_secret="base-sk",
        )
        override_config = Config(
            access_key_id="override-ak",
        )
        api = ControlAPI(config=base_config)

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        api._get_devs_client(config=override_config)

        call_args = mock_client_class.call_args
        config_arg = call_args[0][0]
        assert config_arg.access_key_id == "override-ak"

    @patch("agentrun.utils.control_api.DevsClient")
    def test_get_devs_client_without_protocol_prefix(self, mock_client_class):
        """测试获取 Devs 客户端时 endpoint 不带协议前缀"""
        config = Config(
            access_key_id="ak",
            access_key_secret="sk",
            devs_endpoint="devs.custom.com",
        )
        api = ControlAPI(config=config)

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        api._get_devs_client()

        call_args = mock_client_class.call_args
        config_arg = call_args[0][0]
        # endpoint 不带协议时应该保持原样
        assert config_arg.endpoint == "devs.custom.com"

    @patch("agentrun.utils.control_api.DevsClient")
    def test_get_devs_client_with_read_timeout(self, mock_client_class):
        """测试 Devs 客户端使用 read_timeout"""
        config = Config(
            access_key_id="ak",
            access_key_secret="sk",
            timeout=300,
            read_timeout=60000,
        )
        api = ControlAPI(config=config)

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        api._get_devs_client()

        call_args = mock_client_class.call_args
        config_arg = call_args[0][0]
        assert config_arg.connect_timeout == 300
        assert config_arg.read_timeout == 60000
