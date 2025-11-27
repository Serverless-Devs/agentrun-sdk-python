"""
E2E 测试的公共配置和 fixtures
"""

import os
from pathlib import Path
import time
from typing import Generator

from dotenv import load_dotenv
import pytest

from agentrun.utils.helper import mask_password


def auto_load_env():
    folder = Path(__file__).parent
    while folder != "/":
        dotfile = folder / ".env"
        if dotfile.exists():
            load_dotenv(dotfile)
            print("load .env:", dotfile)
            break
        folder = folder.parent


auto_load_env()


@pytest.fixture(scope="session")
def test_prefix() -> str:
    """生成测试资源名称前缀，用于标识测试资源"""
    timestamp = time.strftime("%Y%m%d%H%M%S")
    return f"e2e-test-{timestamp}"


@pytest.fixture(scope="session")
def check_credentials() -> None:
    """检查必要的环境变量是否已配置"""
    pass


@pytest.fixture(scope="session", autouse=True)
def test_environment(check_credentials) -> Generator[None, None, None]:
    """设置和清理测试环境"""
    # 设置测试环境

    print("\n=== 开始 E2E 测试 ===")
    print(f"Region: {os.getenv('AGENTRUN_REGION', 'cn-hangzhou')}")
    print(
        "AccessKeyId:"
        f" {mask_password( os.getenv('AGENTRUN_ACCESS_KEY_ID') or os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'))}"
    )
    print(f"API Key: {mask_password( os.getenv('API_KEY'))}")

    yield

    # 清理测试环境
    print("\n=== E2E 测试完成 ===")


@pytest.fixture
def unique_name(test_prefix: str) -> Generator[str, None, None]:
    """为每个测试生成唯一的资源名称"""
    import uuid

    name = f"{test_prefix}-{uuid.uuid4().hex[:8]}"
    yield name
