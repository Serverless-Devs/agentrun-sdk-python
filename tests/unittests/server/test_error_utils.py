"""Tests for model-side error helpers."""

from types import SimpleNamespace

import pytest

from agentrun.utils.error_utils import (
    build_error_event_data,
    classify_model_error,
    is_model_error,
    is_rate_limited_error,
)


def test_text_429_rate_limit_is_rate_limited():
    assert is_rate_limited_error("Error code: 429 - rate limit exceeded")


@pytest.mark.parametrize(
    ("text", "code", "status_code", "retryable"),
    [
        (
            "Error code: 400 - Arrearage: account is not in good standing",
            "MODEL_ARREARAGE",
            400,
            False,
        ),
        (
            "Error code: 400 - data_inspection_failed: unsafe content",
            "MODEL_DATA_INSPECTION_FAILED",
            400,
            False,
        ),
        (
            "Error code: 401 - InvalidApiKey: invalid api key",
            "MODEL_AUTHENTICATION_ERROR",
            401,
            False,
        ),
        (
            "Error code: 403 - AccessDenied.Unpurchased: model disabled",
            "MODEL_ACCESS_DENIED",
            403,
            False,
        ),
        (
            "Error code: 429 - Throttling.RateQuota: too many requests",
            "RATE_LIMITED",
            429,
            True,
        ),
        (
            "Error code: 500 - InternalError.Algo: backend failed",
            "MODEL_INTERNAL_ERROR",
            500,
            True,
        ),
        (
            "Error code: 503 - ModelServingError: model overloaded",
            "MODEL_SERVICE_UNAVAILABLE",
            503,
            True,
        ),
    ],
)
def test_common_bailian_text_errors_are_model_errors(
    text,
    code,
    status_code,
    retryable,
):
    model_error = classify_model_error(text)

    assert model_error is not None
    assert model_error.code == code
    assert model_error.status_code == status_code
    assert model_error.retryable is retryable


def test_structured_status_429_is_rate_limited():
    class RateLimitError(RuntimeError):
        status_code = 429

    assert is_rate_limited_error(RateLimitError("provider overloaded"))


def test_structured_response_body_code_is_model_error():
    error = SimpleNamespace(
        response=SimpleNamespace(
            status_code=403,
            json=lambda: {
                "code": "WorkspaceAccessDenied",
                "requestId": "req-123",
            },
        )
    )

    data = build_error_event_data(
        error,
        fallback_code="RuntimeError",
        fallback_message="workspace denied",
    )

    assert data == {
        "message": "workspace denied",
        "code": "MODEL_ACCESS_DENIED",
        "statusCode": 403,
        "providerCode": "WorkspaceAccessDenied",
        "requestId": "req-123",
    }


@pytest.mark.parametrize(
    ("text", "provider_code"),
    [
        (
            "Error code: 403 - AccessDenied.Unpurchased: model disabled",
            "AccessDenied.Unpurchased",
        ),
        (
            "Error code: 429 - Throttling.RateQuota: too many requests",
            "Throttling.RateQuota",
        ),
        (
            "Error code: 500 - InternalError.Algo: backend failed",
            "InternalError.Algo",
        ),
    ],
)
def test_text_provider_code_preserves_dotted_suffix(text, provider_code):
    data = build_error_event_data(
        text,
        fallback_code="str",
        fallback_message=text,
    )

    assert data["providerCode"] == provider_code


def test_non_rate_limit_text_is_not_rate_limited():
    assert not is_rate_limited_error("normal response")


def test_plain_internal_error_is_not_model_error():
    assert not is_model_error("Internal error")


@pytest.mark.parametrize(
    "text",
    [
        "Error code: 400 - validation failed",
        "HTTP 500 - Internal Server Error means the service failed",
    ],
)
def test_status_code_only_text_is_not_model_error(text):
    assert not is_model_error(text)


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
        "statusCode": 429,
    }
