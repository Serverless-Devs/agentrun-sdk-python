"""Tests for model rate-limit helpers."""

from agentrun.utils.error_utils import (
    build_error_event_data,
    is_rate_limited_error,
)


def test_text_429_rate_limit_is_rate_limited():
    assert is_rate_limited_error("Error code: 429 - rate limit exceeded")


def test_structured_status_429_is_rate_limited():
    class RateLimitError(RuntimeError):
        status_code = 429

    assert is_rate_limited_error(RateLimitError("provider overloaded"))


def test_non_rate_limit_text_is_not_rate_limited():
    assert not is_rate_limited_error("normal response")


def test_rate_limit_event_uses_original_message():
    data = build_error_event_data(
        "Error code: 429 - rate limit exceeded",
        fallback_code="str",
        fallback_message="Error code: 429 - rate limit exceeded",
    )

    assert data == {
        "message": "Error code: 429 - rate limit exceeded",
        "code": "RATE_LIMITED",
        "retryable": True,
        "retryAfterMs": 2000,
    }
