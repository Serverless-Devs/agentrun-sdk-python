"""共享的 httpx RAM 签名 Auth handler / Shared httpx RAM-signing auth handler."""

from __future__ import annotations

from typing import Generator, Optional, TYPE_CHECKING

import httpx

from agentrun.utils.log import logger
from agentrun.utils.ram_signature.signer import get_agentrun_signed_headers

if TYPE_CHECKING:
    from agentrun.utils.config import Config


class AgentrunRamAuth(httpx.Auth):
    """httpx Auth handler：为每次请求动态生成 RAM 签名。

    SSE 场景下同一个 ``httpx.AsyncClient`` 会发出 GET（SSE 连接）和 POST
    （消息发送）等不同请求，URL / method / body 各异，因此必须 per-request
    计算签名，不能在 client 初始化时一次性设置 headers。

    持有 ``Config`` 而非快照凭证：``auth_flow`` 每次请求实时取 ak/sk/sts，
    使长连接（一次建连、多请求复用）也能拿到请求级 overlay 注入的最新 STS。
    """

    def __init__(self, config: "Config"):
        self._config = config

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        url = str(request.url)
        method = request.method

        body: Optional[bytes] = None
        if request.content:
            body = request.content

        content_type: Optional[str] = request.headers.get("content-type")

        cfg = self._config
        try:
            signed = get_agentrun_signed_headers(
                url=url,
                method=method,
                access_key_id=cfg.get_access_key_id(),
                access_key_secret=cfg.get_access_key_secret(),
                security_token=cfg.get_security_token() or None,
                region=cfg.get_region_id(),
                product="agentrun",
                body=body,
                content_type=content_type,
            )
            for k, v in signed.items():
                request.headers[k] = v
            logger.debug(
                "applied RAM signature for %s request to %s",
                method,
                url[:80] + ("..." if len(url) > 80 else ""),
            )
        except ValueError as e:
            logger.warning("RAM signing skipped: %s", e)

        yield request
