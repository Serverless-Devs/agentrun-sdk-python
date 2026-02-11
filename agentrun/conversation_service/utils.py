"""Conversation Service 工具函数。

提供状态序列化/反序列化、字符串分片/拼接、时间戳生成等工具。
"""

from __future__ import annotations

import json
import time
from typing import Any

# OTS 单个属性列值上限为 2MB，留 0.5MB 余量（按字符数计）
MAX_COLUMN_SIZE: int = 1_500_000  # 1.5M 字符


def convert_vpc_endpoint_to_public(endpoint: str) -> str:
    """将 OTS VPC 内网地址转换为公网地址。

    Args:
        endpoint: 原始 endpoint，可能是 VPC 内网地址。

    Returns:
        公网地址。若非 VPC 地址则原样返回。

    Example::

        >>> convert_vpc_endpoint_to_public(
        ...     "https://inst.cn-hangzhou.vpc.tablestore.aliyuncs.com"
        ... )
        'https://inst.cn-hangzhou.ots.aliyuncs.com'
    """
    if ".vpc.tablestore.aliyuncs.com" in endpoint:
        return endpoint.replace(
            ".vpc.tablestore.aliyuncs.com", ".ots.aliyuncs.com"
        )
    return endpoint


def nanoseconds_timestamp() -> int:
    """返回当前时间的纳秒时间戳。"""
    return int(time.time() * 1_000_000_000)


def serialize_state(state: dict[str, Any]) -> str:
    """将状态字典序列化为 JSON 字符串。

    Args:
        state: 状态字典。

    Returns:
        JSON 字符串。
    """
    return json.dumps(state, ensure_ascii=False)


def deserialize_state(data: str) -> dict[str, Any]:
    """将 JSON 字符串反序列化为状态字典。

    Args:
        data: JSON 字符串。

    Returns:
        状态字典。
    """
    result: dict[str, Any] = json.loads(data)
    return result


def to_chunks(data: str, max_size: int = MAX_COLUMN_SIZE) -> list[str]:
    """将字符串按指定长度切分为多个分片。

    Args:
        data: 待切分的字符串。
        max_size: 每个分片的最大字符数，默认 1.5M。

    Returns:
        分片列表。若数据小于 max_size，返回包含单个元素的列表。
    """
    if max_size <= 0:
        raise ValueError("max_size must be positive")

    chunks: list[str] = []
    offset = 0
    while offset < len(data):
        chunks.append(data[offset : offset + max_size])
        offset += max_size
    return chunks


def from_chunks(chunks: list[str]) -> str:
    """将多个分片拼接为完整字符串。

    Args:
        chunks: 分片列表。

    Returns:
        拼接后的完整字符串。
    """
    return "".join(chunks)
