"""请求级凭证上下文 / Per-request credential overlay.

此模块提供一个进程级、按请求隔离的"最新 STS 凭证"覆盖层（overlay）。

背景 / Background:
    所有凭证（ak/sk/sts）默认来自环境变量，在 ``Config`` 构造时被读取。但 STS
    临时凭证会过期；部署在函数计算（FC）时，最新轮转后的 STS 通过**每次请求的
    HTTP 头**下发，而非进程级环境变量。因此需要一个按请求设置、所有 ``Config``/
    client 都能优先读取的"当前凭证"覆盖层。

    The overlay is backed by a :class:`contextvars.ContextVar`, so it is:

    - **任务隔离 / task-isolated**: 并发请求各自拥有独立的副本，互不串号；
    - **线程安全 / thread-safe**: ``run_in_threadpool`` 启动的同步处理器会拷贝
      当前 context，因此也能读到；
    - **流式安全 / streaming-safe**: ``StreamingResponse`` 的 body 生成器在请求
      协程的 context 中创建，整条 SSE 流可见同一份凭证。

    默认值为 ``None`` —— 未设置时（非 server 场景、本地调用）overlay 完全不参与，
    行为与历史完全一致。
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from dataclasses import dataclass
from typing import Iterator, Mapping, Optional

# FC 注入 STS 的默认头名（可经构造参数或环境变量覆盖）。
DEFAULT_ACCESS_KEY_ID_HEADER = "x-fc-access-key-id"
DEFAULT_ACCESS_KEY_SECRET_HEADER = "x-fc-access-key-secret"
DEFAULT_SECURITY_TOKEN_HEADER = "x-fc-security-token"


@dataclass(frozen=True)
class StsCredential:
    """一组完整的 STS 临时凭证 / An atomic STS credential triple.

    STS 轮转时 ak/sk/sts 三者一起更新，必须作为整体提供，绝不能把新 sts 与旧
    ak/sk 混用。某字段为 ``None`` 表示该来源未提供，调用方应回退到下一来源。
    """

    access_key_id: Optional[str] = None
    access_key_secret: Optional[str] = None
    security_token: Optional[str] = None

    def is_complete(self) -> bool:
        """三个字段是否齐全。

        STS 轮转时 ak/sk/sts 同时更新，必须作为完整三元组才可作为 overlay，
        避免把新 sts 与陈旧/环境变量里的 ak/sk 混用。
        """
        return bool(
            self.access_key_id
            and self.access_key_secret
            and self.security_token
        )


# 默认 None：未在 server 场景注入时 overlay 不参与，getter 回退到 env 快照。
_current_sts: contextvars.ContextVar[Optional[StsCredential]] = (
    contextvars.ContextVar("agentrun_current_sts", default=None)
)


def set_request_sts(cred: Optional[StsCredential]) -> contextvars.Token:
    """设置当前请求的 STS 覆盖层，返回用于复位的 token。

    Args:
        cred: 本次请求的最新 STS 三元组；传 ``None`` 表示清除覆盖。

    Returns:
        contextvars.Token: 传给 :func:`reset_request_sts` 以恢复上一状态。
    """
    return _current_sts.set(cred)


def reset_request_sts(token: contextvars.Token) -> None:
    """恢复 :func:`set_request_sts` 之前的覆盖状态。"""
    _current_sts.reset(token)


def get_request_sts() -> Optional[StsCredential]:
    """获取当前请求的 STS 覆盖层；未设置时返回 ``None``。"""
    return _current_sts.get()


def _resolve_header_name(
    explicit: Optional[str], env_key: str, default: str
) -> str:
    """解析头名：构造参数 > 环境变量 > 默认值；统一转小写。"""
    return (explicit or os.getenv(env_key) or default).lower()


def sts_from_headers(
    headers: Mapping[str, str],
    *,
    access_key_id_header: Optional[str] = None,
    access_key_secret_header: Optional[str] = None,
    security_token_header: Optional[str] = None,
) -> Optional[StsCredential]:
    """从请求头映射解析 STS 三元组；不齐全则返回 ``None``。

    仅当 ak/sk/sts 三者齐全才视为有效刷新（避免把新 sts 与陈旧/环境变量里的
    ak/sk 混用）。``headers`` 可为任意 Mapping（如 ``dict`` 或 Starlette
    ``Headers``），按头名**大小写不敏感**查找。头名优先级：参数 > 环境变量
    （``AGENTRUN_STS_HEADER_*``）> 默认（``x-fc-*``）。
    """
    ak_name = _resolve_header_name(
        access_key_id_header,
        "AGENTRUN_STS_HEADER_ACCESS_KEY_ID",
        DEFAULT_ACCESS_KEY_ID_HEADER,
    )
    sk_name = _resolve_header_name(
        access_key_secret_header,
        "AGENTRUN_STS_HEADER_ACCESS_KEY_SECRET",
        DEFAULT_ACCESS_KEY_SECRET_HEADER,
    )
    sts_name = _resolve_header_name(
        security_token_header,
        "AGENTRUN_STS_HEADER_SECURITY_TOKEN",
        DEFAULT_SECURITY_TOKEN_HEADER,
    )

    lower = {str(k).lower(): v for k, v in headers.items()}
    cred = StsCredential(
        access_key_id=lower.get(ak_name),
        access_key_secret=lower.get(sk_name),
        security_token=lower.get(sts_name),
    )
    return cred if cred.is_complete() else None


@contextlib.contextmanager
def use_sts_credentials(
    access_key_id: Optional[str] = None,
    access_key_secret: Optional[str] = None,
    security_token: Optional[str] = None,
) -> Iterator[StsCredential]:
    """在 ``with`` 块内临时使用给定 STS 临时凭证（请求级 overlay），退出自动复位。

    适用于**不经过 agentrun server** 的场景：自有 FastAPI / Flask / Django，或
    非 HTTP 的任务里，从上游 / 请求头拿到最新 STS 后注入——块内所有 SDK 调用
    （以及其内创建的 asyncio 任务）即使用这组凭证。

    Examples:
        >>> with use_sts_credentials(ak, sk, sts):
        ...     knowledgebase.retrieve(...)  # 使用最新 STS

    Note:
        基于 ``contextvars``，按当前任务/线程隔离；用户自行 ``threading.Thread``
        起的裸线程不会继承（``asyncio.create_task`` 会）。
    """
    cred = StsCredential(access_key_id, access_key_secret, security_token)
    token = set_request_sts(cred)
    try:
        yield cred
    finally:
        reset_request_sts(token)


@contextlib.contextmanager
def use_sts_from_headers(
    headers: Mapping[str, str],
    *,
    access_key_id_header: Optional[str] = None,
    access_key_secret_header: Optional[str] = None,
    security_token_header: Optional[str] = None,
) -> Iterator[Optional[StsCredential]]:
    """从请求头映射解析 STS 并在 ``with`` 块内生效；三元组不齐全则不覆盖（透传）。

    与 :class:`agentrun.server.sts_middleware.StsRefreshMiddleware` 共用同一套
    解析逻辑。适用于在自有 Web 框架里手动接入：

        >>> with use_sts_from_headers(request.headers):
        ...     await invoke_agent(request)
    """
    cred = sts_from_headers(
        headers,
        access_key_id_header=access_key_id_header,
        access_key_secret_header=access_key_secret_header,
        security_token_header=security_token_header,
    )
    if cred is None:
        yield None
        return
    token = set_request_sts(cred)
    try:
        yield cred
    finally:
        reset_request_sts(token)
