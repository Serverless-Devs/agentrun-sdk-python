import os
from unittest.mock import patch

from agentrun.utils.config import Config


class TestConfig:

    def test_init_without_parameters(self):
        with patch.dict(
            os.environ,
            {
                "AGENTRUN_ACCESS_KEY_ID": "mock-access-key-id",
                "AGENTRUN_ACCESS_KEY_SECRET": "mock-access-key-secret",
                "AGENTRUN_ACCOUNT_ID": "mock-account-id",
            },
        ):
            config = Config()
            assert config._access_key_id == "mock-access-key-id"
            assert config._access_key_secret == "mock-access-key-secret"
            assert config._account_id == "mock-account-id"
            assert config._use_vpc_endpoint is False

    def test_use_vpc_endpoint_from_env(self):
        with patch.dict(os.environ, {"AGENTRUN_KB_USE_VPC": "true"}, clear=False):
            config = Config()
            assert config.get_use_vpc_endpoint() is True

    def test_kb_endpoints_default_public(self):
        config = Config(region_id="cn-hangzhou")
        assert config.get_bailian_endpoint() == "https://bailian.cn-beijing.aliyuncs.com"
        assert config.get_gpdb_endpoint() == "gpdb.aliyuncs.com"
        assert config.get_ots_endpoint("my-instance") == (
            "http://ots-cn-hangzhou.aliyuncs.com"
        )

    def test_kb_endpoints_vpc_mode(self):
        config = Config(region_id="cn-hangzhou", use_vpc_endpoint=True)
        assert config.get_bailian_endpoint() == (
            "bailian-vpc.cn-hangzhou.aliyuncs.com"
        )
        assert config.get_gpdb_endpoint() == "gpdb-vpc.cn-hangzhou.aliyuncs.com"
        assert config.get_ots_endpoint("my-instance") == (
            "https://my-instance.cn-hangzhou.vpc.tablestore.aliyuncs.com"
        )

    def test_bailian_endpoint_override_takes_precedence(self):
        config = Config(
            use_vpc_endpoint=True,
            bailian_endpoint="custom.bailian.example.com",
        )
        assert config.get_bailian_endpoint() == "custom.bailian.example.com"
