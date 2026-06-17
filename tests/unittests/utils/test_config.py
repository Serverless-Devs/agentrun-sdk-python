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
            # 凭证改为懒解析：未显式传入时私有字段保持 None（ambient），
            # 实际值经 getter（overlay 优先 -> 环境变量）解析。
            assert config._access_key_id is None
            assert config._access_key_secret is None
            assert config.get_access_key_id() == "mock-access-key-id"
            assert config.get_access_key_secret() == "mock-access-key-secret"
            assert config._account_id == "mock-account-id"
