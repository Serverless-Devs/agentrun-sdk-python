"""STS 刷新中间件 / STS refresh middleware.

部署在函数计算（FC）时，每次请求的 HTTP 头会携带最新轮转的 STS 临时凭证。
此中间件在请求进入时解析这些头，写入请求级 overlay
（:mod:`agentrun.utils.credential_context`），使本次请求内所有 ``Config`` /
client 的取证都拿到最新 STS；请求结束时复位。

为何用纯 ASGI 中间件 / Why a plain ASGI middleware:
    本中间件只做一件事——设置 / 复位一个 contextvar，故用**纯 ASGI** 实现
    （``__call__`` 包裹 ``await self.app(scope, receive, send)``），而非
    ``BaseHTTPMiddleware``。优点：overlay 与请求**同任务、同生命周期**——对
    endpoint、``StreamingResponse`` 的 body、``run_in_threadpool`` 的同步处理器、
    以及**响应后的 background task** 全程可见，并在 app 完全结束后于 ``finally``
    复位；同时避免 ``BaseHTTPMiddleware`` 的额外 task/stream 包装及其在流式 /
    断连 / 异常传播上的已知坑。

注入时机与有效期 / Injection lifetime:
    STS 在请求入口注入一份、整条请求固定不变（头只到达一次）。流式响应全程使用
    这份入口 STS；仅当**单条请求持续时间超过 STS 有效期**时才会中途过期——属按
    请求头注入模型的固有上限，正常请求 / 流远短于有效期，不受影响。

头名可配置 / Configurable header names:
    构造参数 > 环境变量 > 默认值(``x-fc-*``)。头名大小写不敏感。

启用开关 / Enable switch:
    中间件**默认启用**。仅在特定情况下关闭：构造参数 ``enabled=False``，或
    环境变量 ``AGENTRUN_STS_REFRESH_ENABLED`` 设为假值（``0`` / ``false`` /
    ``no`` / ``off``）。

信任边界 / Trust boundary:
    overlay 仅在 ``x-fc-*`` 头**齐全**时注入（覆盖运维方 env 凭证），否则透传。
    函数计算（FC）拥有该头命名空间并会剥离客户端伪造的同名头，FC 内安全。
    **注意**：若部署在非 FC 环境（裸 uvicorn / 自有网关）且服务可被不可信客户端
    直达，攻击者可注入 ``x-fc-*`` 头冒用身份——此类场景请前置鉴权 / 由网关剥离
    这些头，或按上面的开关关闭本中间件。
"""

from __future__ import annotations

import os
from typing import Optional

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from agentrun.utils.credential_context import use_sts_from_headers


def _detect_enabled() -> bool:
    """是否启用 overlay：**默认启用**，仅环境变量显式设为假值时关闭。

    ``AGENTRUN_STS_REFRESH_ENABLED`` 未设置 -> 启用；设为
    ``0`` / ``false`` / ``no`` / ``off`` -> 关闭；其余真值 -> 启用。
    """
    flag = os.getenv("AGENTRUN_STS_REFRESH_ENABLED")
    if flag is None:
        return True
    return flag.strip().lower() in ("1", "true", "yes", "on")


class StsRefreshMiddleware:
    """纯 ASGI 中间件：从请求头解析最新 STS 并注入请求级 overlay。"""

    def __init__(
        self,
        app: ASGIApp,
        *,
        enabled: Optional[bool] = None,
        access_key_id_header: Optional[str] = None,
        access_key_secret_header: Optional[str] = None,
        security_token_header: Optional[str] = None,
    ) -> None:
        self.app = app
        # enabled=None 时按环境变量决定（默认启用，
        # AGENTRUN_STS_REFRESH_ENABLED 设为假值时关闭）。
        self._enabled = _detect_enabled() if enabled is None else enabled
        # 头名解析（参数 > 环境变量 > 默认）交由 sts_from_headers 处理，这里只存原值。
        self._ak_header = access_key_id_header
        self._sk_header = access_key_secret_header
        self._sts_header = security_token_header

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] != "http" or not self._enabled:
            await self.app(scope, receive, send)
            return

        # 复用公开上下文管理器：解析请求头 -> 注入 overlay -> app 整体跑完后复位。
        # 三元组不齐全时 use_sts_from_headers 不覆盖（透传），与手动注入完全一致。
        # 纯 ASGI：overlay 在同一任务内对 endpoint / 流式 body / 同步处理器 /
        # 响应后的 background task 全程可见，``with`` 在 app 结束后才退出复位。
        with use_sts_from_headers(
            Headers(scope=scope),
            access_key_id_header=self._ak_header,
            access_key_secret_header=self._sk_header,
            security_token_header=self._sts_header,
        ):
            await self.app(scope, receive, send)
