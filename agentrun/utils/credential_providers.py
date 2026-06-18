"""动态凭证 Provider / Dynamic credential providers.

把 :class:`agentrun.utils.config.Config` 接入阿里云 SDK 的"每次请求动态取证"
机制，使长生命周期 client 也能在每次请求时拿到最新 STS（请求级 overlay 优先，
再回退环境变量）。

- :class:`OpenApiCredentialsProvider` 适配 ``alibabacloud_credentials`` 的
  ``ICredentialsProvider``，供 ``alibabacloud_tea_openapi`` 控制面 client 使用。
- TableStore 的 ``CredentialsProvider`` 适配见
  :func:`agentrun.conversation_service.utils.build_ots_credentials_provider`
  （延迟导入 ``tablestore``，避免在非会话场景引入该可选依赖）。

约定 / Convention:
    所有阿里云 / TableStore client 一律通过 provider 注入凭证，**不要**再传静态
    ak/sk/sts —— 否则凭证会在 client 构造时被冻结，STS 过期后请求全部失败。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alibabacloud_credentials.models import CredentialModel
from alibabacloud_credentials_api import ICredentialsProvider

if TYPE_CHECKING:
    from agentrun.utils.config import Config

PROVIDER_NAME = "agentrun_context"


class OpenApiCredentialsProvider(ICredentialsProvider):
    """从 ``Config`` 实时解析凭证的 alibabacloud OpenAPI provider。

    ``alibabacloud_tea_openapi`` 的 client 在**每个请求方法内部**调用
    ``credential.get_credential()``，因此这里返回的凭证总是当前最新值（请求级
    overlay 优先，再回退环境变量）。
    """

    def __init__(self, config: "Config"):
        self._config = config

    def get_provider_name(self) -> str:
        return PROVIDER_NAME

    def get_credentials(self) -> CredentialModel:
        cfg = self._config
        security_token = cfg.get_security_token() or None
        return CredentialModel(
            access_key_id=cfg.get_access_key_id(),
            access_key_secret=cfg.get_access_key_secret(),
            security_token=security_token,
            # 语义化取值，仅在直接使用本 provider 时有意义。注意：经
            # build_openapi_credential 包进 alibabacloud Client 后，
            # client.get_credential().type 实际报告为 provider_name
            # （"agentrun_context"）——任意非 bearer/id_token 的 type 都会进入
            # AK/STS 签名分支并自动附带 security_token，故此处取值不影响签名。
            type="sts" if security_token else "access_key",
            provider_name=PROVIDER_NAME,
        )

    async def get_credentials_async(self) -> CredentialModel:
        return self.get_credentials()


def build_openapi_credential(config: "Config"):
    """构造可直接传给 ``open_api_util_models.Config(credential=...)`` 的凭证 client。

    Returns:
        ``alibabacloud_credentials.client.Client`` 实例，内部包裹
        :class:`OpenApiCredentialsProvider`，每次请求动态取证。
    """
    from alibabacloud_credentials.client import Client as CredentialsClient

    return CredentialsClient(provider=OpenApiCredentialsProvider(config))
