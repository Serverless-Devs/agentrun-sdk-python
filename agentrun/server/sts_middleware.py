"""STS 刷新中间件 / STS refresh middleware.

部署在函数计算（FC）时，每次请求的 HTTP 头会携带最新轮转的 STS 临时凭证。
此中间件在请求进入时解析这些头，写入请求级 overlay
（:mod:`agentrun.utils.credential_context`），使本次请求内所有 ``Config`` /
client 的取证都拿到最新 STS；请求结束时复位。

为何用中间件 / Why middleware:
    在 ``call_next`` 之前 ``set`` 的 contextvar 会被拷贝进下游 endpoint 子任务，
    并对 ``StreamingResponse`` 的 body 生成器、``run_in_threadpool`` 中的同步
    处理器同样可见，因此能覆盖整条请求（含 SSE 流）。已在 Starlette 0.48 +
    FastAPI 0.118 上实测验证。

头名可配置 / Configurable header names:
    构造参数 > 环境变量 > 默认值(``x-fc-*``)。头名大小写不敏感。

信任边界 / Trust boundary:
    overlay 会覆盖运维方在环境变量里配置的凭证，因此只在**可信来源**提供
    ``x-fc-*`` 头时才能启用。函数计算（FC）拥有该头命名空间并会剥离客户端伪造的
    同名头，故默认仅在检测到 FC 环境（``FC_REGION`` 存在）时生效；非 FC 环境
    （如裸 uvicorn / 自有网关）默认关闭，避免不可信客户端注入 ``x-fc-*`` 头冒用
    身份。可用构造参数 ``enabled`` 或环境变量 ``AGENTRUN_STS_REFRESH_ENABLED``
    显式开关（开启前请确保已有前置鉴权或网关剥离这些头）。
"""

from __future__ import annotations

import os
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from agentrun.utils.credential_context import use_sts_from_headers


def _detect_enabled() -> bool:
    """决定是否启用 overlay：显式环境变量优先，否则按是否在 FC 环境自动判断。"""
    flag = os.getenv("AGENTRUN_STS_REFRESH_ENABLED")
    if flag is not None:
        return flag.strip().lower() in ("1", "true", "yes", "on")
    # FC 环境下 FC_REGION 必然存在；非 FC 默认关闭以防 x-fc-* 头注入。
    return os.getenv("FC_REGION") is not None


class StsRefreshMiddleware(BaseHTTPMiddleware):
    """从请求头解析最新 STS 并注入请求级 overlay。"""

    def __init__(
        self,
        app: ASGIApp,
        *,
        enabled: Optional[bool] = None,
        access_key_id_header: Optional[str] = None,
        access_key_secret_header: Optional[str] = None,
        security_token_header: Optional[str] = None,
    ) -> None:
        super().__init__(app)
        # enabled=None 时自动探测（FC 环境或显式环境变量开关）。
        self._enabled = _detect_enabled() if enabled is None else enabled
        # 头名解析（参数 > 环境变量 > 默认）交由 sts_from_headers 处理，这里只存原值。
        self._ak_header = access_key_id_header
        self._sk_header = access_key_secret_header
        self._sts_header = security_token_header

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._enabled:
            return await call_next(request)

        # 直接复用公开上下文管理器：解析请求头 -> 注入 overlay -> 退出复位。
        # 三元组不齐全时 use_sts_from_headers 不覆盖（透传），与手动注入完全一致。
        with use_sts_from_headers(
            request.headers,
            access_key_id_header=self._ak_header,
            access_key_secret_header=self._sk_header,
            security_token_header=self._sts_header,
        ):
            return await call_next(request)
