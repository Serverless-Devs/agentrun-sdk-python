"""Tests for server error helpers."""

from agentrun.utils.error_utils import _get_header, is_rate_limited_error


def test_get_header_matches_name_case_insensitively():
    headers = {"x-trace-id": "trace-123"}

    assert _get_header(headers, "X-Trace-ID") == "trace-123"


def test_explanatory_code_429_text_is_not_rate_limited():
    error = RuntimeError("validation failed for field code 429")

    assert not is_rate_limited_error(error)


def test_explanatory_http_429_text_is_not_rate_limited():
    error = RuntimeError(
        "docs mention HTTP 429 means rate limit; actual error is 401"
    )

    assert not is_rate_limited_error(error)


def test_explicit_throttling_text_is_rate_limited():
    assert is_rate_limited_error(RuntimeError("Throttling: model overloaded"))


def test_explicit_throttled_text_is_rate_limited():
    assert is_rate_limited_error(RuntimeError("request throttled by provider"))


def test_structured_status_429_is_rate_limited():
    class RateLimitError(RuntimeError):
        status_code = 429

    assert is_rate_limited_error(RateLimitError("model overloaded"))
