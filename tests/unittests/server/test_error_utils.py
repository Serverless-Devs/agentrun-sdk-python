"""Tests for server error helpers."""

from agentrun.server.error_utils import _get_header


def test_get_header_matches_name_case_insensitively():
    headers = {"x-trace-id": "trace-123"}

    assert _get_header(headers, "X-Trace-ID") == "trace-123"
