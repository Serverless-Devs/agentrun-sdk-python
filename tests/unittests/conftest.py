"""Unit test fixtures / 单元测试公共 fixture.

本模块统一为 tests/unittests/ 目录提供测试隔离保障。

Why:
    agentrun.utils.config 在模块导入时调用 load_dotenv(), 会把仓库根目录
    的 .env 注入到 os.environ。Config() 在构造时会用 get_env_with_default
    读取这些环境变量作为默认值, 导致：
      - respx mock 的 URL 是基于 SDK 默认 account_id / region 构造,
        而实际请求打到开发者本地 .env 里的真实 account / region,
        触发 "AllMockedAssertionError: ... not mocked!"
      - 硬编码期望默认 account_id / region 的断言直接 AssertionError

How:
    autouse fixture 在每个 test 进入前, 通过 monkeypatch.delenv 清理
    Config.__init__ 读取的那一组 env。monkeypatch 会在 test 结束时
    自动恢复原值, 不影响仓库 .env 文件本身, 也不影响显式使用
    patch.dict(os.environ, ...) 设置 test 用凭据的测试 (patch.dict
    会覆盖 delenv 的结果)。
"""

from typing import List

import pytest

# Config.__init__ 默认值会读取的环境变量白名单
# 仅清理这些 key, 不动用户测试自己 set 的其他 AGENTRUN_* 变量
# (例如 memory_collection test 里的 AGENTRUN_MYSQL_PUBLIC_HOST)
_SDK_CONFIG_ENV_KEYS: List[str] = [
    # 凭据类 / Credentials
    "AGENTRUN_ACCESS_KEY_ID",
    "ALIBABA_CLOUD_ACCESS_KEY_ID",
    "AGENTRUN_ACCESS_KEY_SECRET",
    "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
    "AGENTRUN_SECURITY_TOKEN",
    "ALIBABA_CLOUD_SECURITY_TOKEN",
    # 账号与区域 / Account & Region
    "AGENTRUN_ACCOUNT_ID",
    "FC_ACCOUNT_ID",
    "AGENTRUN_REGION",
    "FC_REGION",
    # Endpoint 类 / Endpoints
    "AGENTRUN_CONTROL_ENDPOINT",
    "AGENTRUN_DATA_ENDPOINT",
    "DEVS_ENDPOINT",
    "BAILIAN_ENDPOINT",
]


@pytest.fixture(autouse=True)
def _isolate_sdk_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """自动为每个 unit test 清理 SDK Config 相关 env, 避免被本地 .env 污染."""
    for key in _SDK_CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
