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

import contextvars
from dataclasses import dataclass
from typing import Optional


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
