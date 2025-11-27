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
