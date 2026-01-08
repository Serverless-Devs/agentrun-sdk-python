"""扩展的 Config 测试 / Extended Config tests"""

import os
from unittest.mock import patch

import pytest

from agentrun.utils.config import Config, get_env_with_default


class TestGetEnvWithDefault:
    """测试 get_env_with_default 函数"""

    def test_returns_first_available_env(self):
        """测试返回第一个可用的环境变量值"""
        with patch.dict(
            os.environ,
            {"KEY_A": "value_a", "KEY_B": "value_b"},
            clear=False,
        ):
            result = get_env_with_default("default", "KEY_A", "KEY_B")
            assert result == "value_a"

    def test_returns_second_if_first_not_set(self):
        """测试如果第一个不存在则返回第二个"""
        with patch.dict(os.environ, {"KEY_B": "value_b"}, clear=False):
            # 确保 KEY_A 不存在
            env = os.environ.copy()
            env.pop("KEY_A", None)
            env["KEY_B"] = "value_b"
            with patch.dict(os.environ, env, clear=True):
                result = get_env_with_default("default", "KEY_A", "KEY_B")
                assert result == "value_b"

    def test_returns_default_if_none_set(self):
        """测试如果都不存在则返回默认值"""
        with patch.dict(os.environ, {}, clear=True):
            result = get_env_with_default(
                "my_default", "NONEXISTENT_A", "NONEXISTENT_B"
            )
            assert result == "my_default"


class TestConfigExtended:
    """扩展的 Config 测试"""

    def test_init_with_all_parameters(self):
        """测试使用所有参数初始化"""
        config = Config(
            access_key_id="ak_id",
            access_key_secret="ak_secret",
            security_token="token",
            account_id="account",
            token="custom_token",
            region_id="cn-shanghai",
            timeout=300,
            read_timeout=50000,
            control_endpoint="https://custom-control.com",
            data_endpoint="https://custom-data.com",
            devs_endpoint="https://custom-devs.com",
            headers={"X-Custom": "value"},
        )
        assert config.get_access_key_id() == "ak_id"
        assert config.get_access_key_secret() == "ak_secret"
        assert config.get_security_token() == "token"
        assert config.get_account_id() == "account"
        assert config.get_token() == "custom_token"
        assert config.get_region_id() == "cn-shanghai"
        assert config.get_timeout() == 300
        assert config.get_read_timeout() == 50000
        assert config.get_control_endpoint() == "https://custom-control.com"
        assert config.get_data_endpoint() == "https://custom-data.com"
        assert config.get_devs_endpoint() == "https://custom-devs.com"
        assert config.get_headers() == {"X-Custom": "value"}

    def test_init_from_env_alibaba_cloud_vars(self):
        """测试从阿里云环境变量读取配置"""
        with patch.dict(
            os.environ,
            {
                "ALIBABA_CLOUD_ACCESS_KEY_ID": "alibaba_ak_id",
                "ALIBABA_CLOUD_ACCESS_KEY_SECRET": "alibaba_ak_secret",
                "ALIBABA_CLOUD_SECURITY_TOKEN": "alibaba_token",
                "FC_ACCOUNT_ID": "fc_account",
                "FC_REGION": "cn-beijing",
            },
            clear=True,
        ):
            config = Config()
            assert config.get_access_key_id() == "alibaba_ak_id"
            assert config.get_access_key_secret() == "alibaba_ak_secret"
            assert config.get_security_token() == "alibaba_token"
            assert config.get_account_id() == "fc_account"
            assert config.get_region_id() == "cn-beijing"

    def test_with_configs_class_method(self):
        """测试 with_configs 类方法"""
        config1 = Config(access_key_id="id1", region_id="cn-hangzhou")
        config2 = Config(access_key_id="id2", timeout=200)

        result = Config.with_configs(config1, config2)
        assert result.get_access_key_id() == "id2"
        assert result.get_region_id() == "cn-hangzhou"
        assert result.get_timeout() == 200

    def test_update_with_none_config(self):
        """测试 update 方法处理 None 配置"""
        config = Config(access_key_id="original")
        result = config.update(None)
        assert result.get_access_key_id() == "original"

    def test_update_merges_headers(self):
        """测试 update 方法合并 headers"""
        config1 = Config(headers={"Key1": "Value1"})
        config2 = Config(headers={"Key2": "Value2"})

        result = config1.update(config2)
        headers = result.get_headers()
        assert headers.get("Key1") == "Value1"
        assert headers.get("Key2") == "Value2"

    def test_repr(self):
        """测试 __repr__ 方法"""
        config = Config(access_key_id="test_id")
        result = repr(config)
        assert "Config{" in result
        assert "test_id" in result

    def test_get_account_id_raises_when_not_set(self):
        """测试 get_account_id 在未设置时抛出异常"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            with pytest.raises(ValueError) as exc_info:
                config.get_account_id()
            assert "account id is not set" in str(exc_info.value)

    def test_get_region_id_default(self):
        """测试 get_region_id 默认值"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.get_region_id() == "cn-hangzhou"

    def test_get_timeout_default(self):
        """测试 get_timeout 默认值"""
        config = Config(timeout=None)
        assert config.get_timeout() == 600

    def test_get_read_timeout_default(self):
        """测试 get_read_timeout 默认值"""
        config = Config(read_timeout=None)
        assert config.get_read_timeout() == 100000

    def test_get_control_endpoint_default(self):
        """测试 get_control_endpoint 默认值"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            result = config.get_control_endpoint()
            assert "agentrun.cn-hangzhou.aliyuncs.com" in result

    def test_get_data_endpoint_default(self):
        """测试 get_data_endpoint 默认值"""
        with patch.dict(
            os.environ, {"AGENTRUN_ACCOUNT_ID": "test-account"}, clear=True
        ):
            config = Config()
            result = config.get_data_endpoint()
            assert "test-account" in result
            assert "agentrun-data" in result

    def test_get_devs_endpoint_default(self):
        """测试 get_devs_endpoint 默认值"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            result = config.get_devs_endpoint()
            assert "devs.cn-hangzhou.aliyuncs.com" in result

    def test_get_headers_default_empty(self):
        """测试 get_headers 默认返回空字典"""
        config = Config()
        assert config.get_headers() == {}

    def test_control_endpoint_from_env(self):
        """测试从环境变量读取 control_endpoint"""
        with patch.dict(
            os.environ,
            {"AGENTRUN_CONTROL_ENDPOINT": "https://custom-endpoint.com"},
            clear=True,
        ):
            config = Config()
            assert (
                config.get_control_endpoint() == "https://custom-endpoint.com"
            )

    def test_data_endpoint_from_env(self):
        """测试从环境变量读取 data_endpoint"""
        with patch.dict(
            os.environ,
            {"AGENTRUN_DATA_ENDPOINT": "https://custom-data.com"},
            clear=True,
        ):
            config = Config()
            assert config.get_data_endpoint() == "https://custom-data.com"

    def test_devs_endpoint_from_env(self):
        """测试从环境变量读取 devs_endpoint"""
        with patch.dict(
            os.environ,
            {"DEVS_ENDPOINT": "https://custom-devs.com"},
            clear=True,
        ):
            config = Config()
            assert config.get_devs_endpoint() == "https://custom-devs.com"
