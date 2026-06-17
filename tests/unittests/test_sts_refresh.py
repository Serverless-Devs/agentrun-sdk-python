"""STS 静默刷新机制单元测试 / Unit tests for silent STS refresh.

覆盖：
- 请求级凭证 overlay（contextvars）
- Config getter 的解析优先级（显式 > overlay(仅 ambient) > 环境变量）
- alibabacloud OpenAPI 动态 credential provider
- TableStore 动态 credentials_provider
- 服务端 StsRefreshMiddleware 端到端（async / sync / streaming / 隔离）
"""

from __future__ import annotations

import pytest

from agentrun.utils.config import Config
from agentrun.utils.credential_context import (
    StsCredential,
    get_request_sts,
    reset_request_sts,
    set_request_sts,
    use_sts_credentials,
    use_sts_from_headers,
)


@pytest.fixture
def overlay():
    """在 with 块内设置 overlay，退出自动复位。"""
    from contextlib import contextmanager

    @contextmanager
    def _set(ak=None, sk=None, sts=None):
        token = set_request_sts(
            StsCredential(
                access_key_id=ak, access_key_secret=sk, security_token=sts
            )
        )
        try:
            yield
        finally:
            reset_request_sts(token)

    return _set


# --------------------------------------------------------------------------- #
# overlay 基础行为
# --------------------------------------------------------------------------- #
def test_overlay_default_is_none():
    assert get_request_sts() is None


def test_overlay_set_reset():
    token = set_request_sts(StsCredential("a", "b", "c"))
    try:
        cur = get_request_sts()
        assert cur is not None and cur.access_key_id == "a"
    finally:
        reset_request_sts(token)
    assert get_request_sts() is None


# --------------------------------------------------------------------------- #
# Config getter 解析优先级
# --------------------------------------------------------------------------- #
def test_ambient_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_ID", "ENV_AK")
    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_SECRET", "ENV_SK")
    cfg = Config()
    assert cfg.get_access_key_id() == "ENV_AK"
    assert cfg.get_access_key_secret() == "ENV_SK"
    assert cfg.get_security_token() == ""


def test_ambient_prefers_overlay(overlay):
    cfg = Config()
    with overlay(ak="OV_AK", sk="OV_SK", sts="OV_STS"):
        assert cfg.get_access_key_id() == "OV_AK"
        assert cfg.get_access_key_secret() == "OV_SK"
        assert cfg.get_security_token() == "OV_STS"


def test_explicit_not_overridden_by_overlay(overlay):
    cfg = Config(access_key_id="USER_AK", access_key_secret="USER_SK")
    with overlay(ak="OV_AK", sk="OV_SK", sts="OV_STS"):
        assert cfg.get_access_key_id() == "USER_AK"
        assert cfg.get_access_key_secret() == "USER_SK"
        # 显式设置了 ak/sk -> 非 ambient -> sts 不得从 overlay 取（避免混用）
        assert cfg.get_security_token() == ""


def test_overlay_dropped_after_reset(monkeypatch, overlay):
    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_ID", "ENV_AK")
    cfg = Config()
    with overlay(ak="OV_AK"):
        assert cfg.get_access_key_id() == "OV_AK"
    assert cfg.get_access_key_id() == "ENV_AK"


def test_with_configs_stays_overlay_aware(overlay):
    merged = Config.with_configs(Config(), Config())
    with overlay(ak="OV_AK", sts="OV_STS"):
        assert merged.get_access_key_id() == "OV_AK"
        assert merged.get_security_token() == "OV_STS"


def test_with_configs_preserves_explicit(overlay):
    user = Config(access_key_id="USER_AK", access_key_secret="USER_SK")
    merged = Config.with_configs(Config(), user)
    with overlay(ak="OV_AK"):
        assert merged.get_access_key_id() == "USER_AK"


# --------------------------------------------------------------------------- #
# OpenAPI provider（alibabacloud）
# --------------------------------------------------------------------------- #
def test_openapi_provider_per_request_fresh(overlay):
    from agentrun.utils.credential_providers import build_openapi_credential

    cred_client = build_openapi_credential(Config())
    with overlay(ak="OV_AK", sk="OV_SK", sts="OV_STS"):
        model = cred_client.get_credential()
        assert model.access_key_id == "OV_AK"
        assert model.access_key_secret == "OV_SK"
        assert model.security_token == "OV_STS"


def test_openapi_provider_reflects_change_between_calls(overlay):
    from agentrun.utils.credential_providers import build_openapi_credential

    cred_client = build_openapi_credential(Config())
    with overlay(ak="AK1", sk="SK1", sts="STS1"):
        assert cred_client.get_credential().access_key_id == "AK1"
    with overlay(ak="AK2", sk="SK2", sts="STS2"):
        assert cred_client.get_credential().access_key_id == "AK2"


# --------------------------------------------------------------------------- #
# TableStore provider
# --------------------------------------------------------------------------- #
def test_ots_provider_per_request_fresh(overlay):
    from agentrun.conversation_service.utils import (
        build_ots_credentials_provider,
    )

    provider = build_ots_credentials_provider(Config())
    with overlay(ak="OV_AK", sk="OV_SK", sts="OV_STS"):
        creds = provider.get_credentials()
        assert creds.get_access_key_id() == "OV_AK"
        assert creds.get_access_key_secret() == "OV_SK"
        assert creds.get_security_token() == "OV_STS"


def test_ots_provider_empty_sts_is_none(monkeypatch):
    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_ID", "ENV_AK")
    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_SECRET", "ENV_SK")
    from agentrun.conversation_service.utils import (
        build_ots_credentials_provider,
    )

    creds = build_ots_credentials_provider(Config()).get_credentials()
    assert creds.get_access_key_id() == "ENV_AK"
    assert creds.get_security_token() is None


# --------------------------------------------------------------------------- #
# 服务端中间件端到端
# --------------------------------------------------------------------------- #
def _build_app():
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    from agentrun.server.sts_middleware import StsRefreshMiddleware

    app = FastAPI()
    # 显式启用，避免受 dev 环境 AGENTRUN_STS_REFRESH_ENABLED 影响。
    app.add_middleware(StsRefreshMiddleware, enabled=True)

    @app.get("/async")
    async def _async():
        return {
            "ak": Config().get_access_key_id(),
            "sts": Config().get_security_token(),
        }

    @app.get("/sync")
    def _sync():  # 在 threadpool 中执行
        return {"ak": Config().get_access_key_id()}

    @app.get("/stream")
    async def _stream():
        cfg = Config()

        async def gen():
            yield (
                f"ak={cfg.get_access_key_id()};"
                f"sts={cfg.get_security_token()}"
            ).encode()

        return StreamingResponse(gen())

    return app


_HEADERS = {
    "x-fc-access-key-id": "H_AK",
    "x-fc-access-key-secret": "H_SK",
    "x-fc-security-token": "H_STS",
}


def test_middleware_async_overlay():
    from fastapi.testclient import TestClient

    client = TestClient(_build_app())
    assert client.get("/async", headers=_HEADERS).json() == {
        "ak": "H_AK",
        "sts": "H_STS",
    }


def test_middleware_sync_threadpool_overlay():
    from fastapi.testclient import TestClient

    client = TestClient(_build_app())
    assert client.get("/sync", headers=_HEADERS).json() == {"ak": "H_AK"}


def test_middleware_streaming_overlay():
    from fastapi.testclient import TestClient

    client = TestClient(_build_app())
    assert client.get("/stream", headers=_HEADERS).text == "ak=H_AK;sts=H_STS"


def test_middleware_no_header_falls_back_to_env(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_ID", "ENV_AK")
    client = TestClient(_build_app())
    assert client.get("/async").json() == {"ak": "ENV_AK", "sts": ""}


def test_middleware_resets_overlay_after_request():
    from fastapi.testclient import TestClient

    client = TestClient(_build_app())
    client.get("/async", headers=_HEADERS)
    assert get_request_sts() is None


@pytest.mark.parametrize(
    "import_path",
    [
        "agentrun.toolset.api.mcp",
        "agentrun.tool.api.mcp",
        "agentrun.tool.api.openapi",
    ],
)
def test_ram_auth_helper_signs_with_live_sts(import_path, overlay):
    """长生命周期的 _AgentrunRamAuth 实例应在每次 auth_flow 读到最新 STS。"""
    import importlib

    import httpx

    module = importlib.import_module(import_path)
    # 构造一次（模拟长连接 SSE 会话只在打开时建一次 auth）
    # 三个模块均复用共享的 AgentrunRamAuth。
    auth = module.AgentrunRamAuth(config=Config())

    def signed_token() -> str:
        request = httpx.Request(
            "POST",
            "https://x.agentrun-data.cn-hangzhou.aliyuncs.com/m",
            content=b"{}",
        )
        flow = auth.auth_flow(request)
        next(flow)  # 触发签名，写入 request.headers
        return request.headers.get("x-acs-security-token")

    with overlay(ak="AK1", sk="SK1", sts="STS1"):
        assert signed_token() == "STS1"
    # 同一个 auth 实例：overlay 切换后应签出新的 sts
    with overlay(ak="AK2", sk="SK2", sts="STS2"):
        assert signed_token() == "STS2"


def test_middleware_custom_header_names(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from agentrun.server.sts_middleware import StsRefreshMiddleware

    app = FastAPI()
    app.add_middleware(
        StsRefreshMiddleware,
        enabled=True,
        access_key_id_header="x-custom-ak",
        access_key_secret_header="x-custom-sk",
        security_token_header="x-custom-sts",
    )

    @app.get("/c")
    async def _c():
        return {
            "ak": Config().get_access_key_id(),
            "sts": Config().get_security_token(),
        }

    client = TestClient(app)
    resp = client.get(
        "/c",
        headers={
            "x-custom-ak": "C_AK",
            "x-custom-sk": "C_SK",
            "x-custom-sts": "C_STS",
        },
    ).json()
    assert resp == {"ak": "C_AK", "sts": "C_STS"}


def test_middleware_partial_headers_ignored(monkeypatch):
    """H1：部分头集合（缺一）不应设置 overlay，避免与 env 混用。"""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_ID", "ENV_AK")
    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_SECRET", "ENV_SK")
    client = TestClient(_build_app())
    # 只发 ak + sts（缺 sk）→ 不构成完整三元组 → 回退 env，不混用。
    resp = client.get(
        "/async",
        headers={"x-fc-access-key-id": "H_AK", "x-fc-security-token": "H_STS"},
    ).json()
    assert resp == {"ak": "ENV_AK", "sts": ""}


def test_middleware_enabled_by_default(monkeypatch):
    """默认启用（不看 FC 环境，无需任何开关）。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from agentrun.server.sts_middleware import StsRefreshMiddleware

    monkeypatch.delenv("FC_REGION", raising=False)
    monkeypatch.delenv("AGENTRUN_STS_REFRESH_ENABLED", raising=False)

    app = FastAPI()
    app.add_middleware(StsRefreshMiddleware)  # enabled=None → 默认启用

    @app.get("/x")
    async def _x():
        return {"ak": Config().get_access_key_id()}

    client = TestClient(app)
    # 默认开：携带完整 x-fc-* 头即生效。
    assert client.get("/x", headers=_HEADERS).json() == {"ak": "H_AK"}


def test_middleware_disabled_via_env(monkeypatch):
    """仅在 AGENTRUN_STS_REFRESH_ENABLED 设为假值时关闭。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from agentrun.server.sts_middleware import StsRefreshMiddleware

    monkeypatch.setenv("AGENTRUN_STS_REFRESH_ENABLED", "false")
    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_ID", "ENV_AK")

    app = FastAPI()
    app.add_middleware(StsRefreshMiddleware)  # enabled=None → 读环境变量 -> 关闭

    @app.get("/x")
    async def _x():
        return {"ak": Config().get_access_key_id()}

    client = TestClient(app)
    # 已关闭：即便携带完整 x-fc-* 头也不覆盖 env 凭证。
    assert client.get("/x", headers=_HEADERS).json() == {"ak": "ENV_AK"}


def test_middleware_disabled_via_constructor(monkeypatch):
    """构造参数 enabled=False 显式关闭。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from agentrun.server.sts_middleware import StsRefreshMiddleware

    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_ID", "ENV_AK")

    app = FastAPI()
    app.add_middleware(StsRefreshMiddleware, enabled=False)

    @app.get("/x")
    async def _x():
        return {"ak": Config().get_access_key_id()}

    client = TestClient(app)
    assert client.get("/x", headers=_HEADERS).json() == {"ak": "ENV_AK"}


def test_invoker_sync_path_sees_overlay():
    """C2：AgentInvoker 的同步 handler + 同步生成器路径应读到请求级 overlay。

    生产路径为 async 端点 -> AgentInvoker.invoke_stream -> run_in_executor，
    必须拷贝 context，否则 sync 路径取不到最新 STS（回退陈旧 env）。
    """
    import asyncio

    from agentrun.server.invoker import AgentInvoker
    from agentrun.server.model import AgentRequest, EventType

    def sync_handler(_request):
        # 同步 handler 返回同步生成器：两段都经 run_in_executor。
        def gen():
            cfg = Config()
            yield (
                f"ak={cfg.get_access_key_id()};"
                f"sts={cfg.get_security_token()}"
            )

        return gen()

    invoker = AgentInvoker(sync_handler)

    async def run():
        token = set_request_sts(StsCredential("OV_AK", "OV_SK", "OV_STS"))
        try:
            return [
                event
                async for event in invoker.invoke_stream(AgentRequest())
            ]
        finally:
            reset_request_sts(token)

    events = asyncio.run(run())
    text = "".join(
        e.data.get("delta", "")
        for e in events
        if e.event == EventType.TEXT
    )
    assert text == "ak=OV_AK;sts=OV_STS", text


# --------------------------------------------------------------------------- #
# 公开 API：非 server 场景手动注入
# --------------------------------------------------------------------------- #
def test_use_sts_credentials_context_manager(monkeypatch):
    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_ID", "ENV_AK")
    cfg = Config()
    assert cfg.get_access_key_id() == "ENV_AK"
    with use_sts_credentials("OV_AK", "OV_SK", "OV_STS"):
        assert cfg.get_access_key_id() == "OV_AK"
        assert cfg.get_access_key_secret() == "OV_SK"
        assert cfg.get_security_token() == "OV_STS"
    # 退出自动复位
    assert cfg.get_access_key_id() == "ENV_AK"
    assert get_request_sts() is None


def test_use_sts_from_headers_complete_case_insensitive():
    cfg = Config()
    headers = {
        "X-Fc-Access-Key-Id": "H_AK",  # 大小写不敏感
        "x-fc-access-key-secret": "H_SK",
        "X-FC-SECURITY-TOKEN": "H_STS",
    }
    with use_sts_from_headers(headers) as cred:
        assert cred is not None
        assert cfg.get_access_key_id() == "H_AK"
        assert cfg.get_security_token() == "H_STS"
    assert get_request_sts() is None


def test_use_sts_from_headers_partial_no_override(monkeypatch):
    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_ID", "ENV_AK")
    cfg = Config()
    # 只有 sts、缺 ak/sk -> 不构成完整三元组 -> 不覆盖
    with use_sts_from_headers({"x-fc-security-token": "H_STS"}) as cred:
        assert cred is None
        assert cfg.get_access_key_id() == "ENV_AK"
        assert cfg.get_security_token() == ""
    assert get_request_sts() is None


def test_sts_from_headers_helper():
    from agentrun.utils.credential_context import sts_from_headers

    assert sts_from_headers({"x-fc-access-key-id": "a"}) is None  # 不齐全
    cred = sts_from_headers({
        "x-fc-access-key-id": "a",
        "x-fc-access-key-secret": "b",
        "x-fc-security-token": "c",
    })
    assert cred is not None
    assert (
        cred.access_key_id,
        cred.access_key_secret,
        cred.security_token,
    ) == ("a", "b", "c")


def test_public_exports_available():
    import agentrun

    for name in (
        "StsCredential",
        "use_sts_credentials",
        "use_sts_from_headers",
    ):
        assert hasattr(agentrun, name), f"{name} not exported"
        assert name in agentrun.__all__, f"{name} missing from __all__"


def test_middleware_background_task_sees_overlay():
    """纯 ASGI：响应后的 background task 仍能读到请求级 overlay。

    BaseHTTPMiddleware 下 background task 可能在 overlay 复位后才运行；纯 ASGI
    中间件持有 overlay 直到 app（含 background）整体结束，故覆盖到位。
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    from starlette.background import BackgroundTask

    from agentrun.server.sts_middleware import StsRefreshMiddleware

    captured: dict = {}

    def _bg():
        cfg = Config()
        captured["ak"] = cfg.get_access_key_id()
        captured["sts"] = cfg.get_security_token()

    app = FastAPI()
    app.add_middleware(StsRefreshMiddleware, enabled=True)

    @app.get("/bg")
    async def _ep():
        return JSONResponse({"ok": True}, background=BackgroundTask(_bg))

    client = TestClient(app)
    client.get("/bg", headers=_HEADERS)
    assert captured == {"ak": "H_AK", "sts": "H_STS"}, captured
