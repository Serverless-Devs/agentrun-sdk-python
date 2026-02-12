"""conversation_service.utils 单元测试。

覆盖 convert_vpc_endpoint_to_public、nanoseconds_timestamp、
serialize_state、deserialize_state、to_chunks、from_chunks。
"""

from __future__ import annotations

import time

import pytest

from agentrun.conversation_service.utils import (
    convert_vpc_endpoint_to_public,
    deserialize_state,
    from_chunks,
    MAX_COLUMN_SIZE,
    nanoseconds_timestamp,
    serialize_state,
    to_chunks,
)

# ---------------------------------------------------------------------------
# convert_vpc_endpoint_to_public
# ---------------------------------------------------------------------------


class TestConvertVpcEndpoint:
    """VPC 地址转公网地址。"""

    def test_vpc_endpoint(self) -> None:
        result = convert_vpc_endpoint_to_public(
            "https://inst.cn-hangzhou.vpc.tablestore.aliyuncs.com"
        )
        assert result == "https://inst.cn-hangzhou.ots.aliyuncs.com"

    def test_non_vpc_endpoint(self) -> None:
        ep = "https://inst.cn-hangzhou.ots.aliyuncs.com"
        assert convert_vpc_endpoint_to_public(ep) == ep

    def test_empty_string(self) -> None:
        assert convert_vpc_endpoint_to_public("") == ""

    def test_other_domain(self) -> None:
        ep = "https://example.com"
        assert convert_vpc_endpoint_to_public(ep) == ep


# ---------------------------------------------------------------------------
# nanoseconds_timestamp
# ---------------------------------------------------------------------------


class TestNanosecondsTimestamp:
    """纳秒时间戳。"""

    def test_returns_int(self) -> None:
        ts = nanoseconds_timestamp()
        assert isinstance(ts, int)

    def test_roughly_correct(self) -> None:
        before = int(time.time() * 1_000_000_000)
        ts = nanoseconds_timestamp()
        after = int(time.time() * 1_000_000_000)
        assert before <= ts <= after


# ---------------------------------------------------------------------------
# serialize_state / deserialize_state
# ---------------------------------------------------------------------------


class TestStateSerialization:
    """状态序列化/反序列化。"""

    def test_roundtrip(self) -> None:
        state = {"key": "value", "num": 42, "nested": {"a": [1, 2]}}
        serialized = serialize_state(state)
        deserialized = deserialize_state(serialized)
        assert deserialized == state

    def test_unicode(self) -> None:
        state = {"中文": "值"}
        serialized = serialize_state(state)
        assert "中文" in serialized
        assert deserialize_state(serialized) == state

    def test_empty(self) -> None:
        serialized = serialize_state({})
        assert deserialize_state(serialized) == {}


# ---------------------------------------------------------------------------
# to_chunks / from_chunks
# ---------------------------------------------------------------------------


class TestChunking:
    """字符串分片/拼接。"""

    def test_small_data_single_chunk(self) -> None:
        data = "hello"
        chunks = to_chunks(data, max_size=100)
        assert chunks == ["hello"]
        assert from_chunks(chunks) == data

    def test_exact_size(self) -> None:
        data = "abcdef"
        chunks = to_chunks(data, max_size=6)
        assert chunks == ["abcdef"]
        assert from_chunks(chunks) == data

    def test_split_into_multiple_chunks(self) -> None:
        data = "abcdefghij"  # 10 chars
        chunks = to_chunks(data, max_size=3)
        assert chunks == ["abc", "def", "ghi", "j"]
        assert from_chunks(chunks) == data

    def test_empty_string(self) -> None:
        assert to_chunks("", max_size=10) == []
        assert from_chunks([]) == ""

    def test_max_size_one(self) -> None:
        data = "abc"
        chunks = to_chunks(data, max_size=1)
        assert chunks == ["a", "b", "c"]
        assert from_chunks(chunks) == data

    def test_invalid_max_size(self) -> None:
        with pytest.raises(ValueError, match="max_size must be positive"):
            to_chunks("data", max_size=0)
        with pytest.raises(ValueError, match="max_size must be positive"):
            to_chunks("data", max_size=-1)

    def test_default_max_size(self) -> None:
        """默认使用 MAX_COLUMN_SIZE。"""
        data = "x" * 10
        chunks = to_chunks(data)
        assert len(chunks) == 1
        assert MAX_COLUMN_SIZE == 1_500_000

    def test_large_data(self) -> None:
        """模拟大数据分片场景。"""
        data = "a" * 100
        chunks = to_chunks(data, max_size=30)
        assert len(chunks) == 4  # 30+30+30+10
        assert from_chunks(chunks) == data
