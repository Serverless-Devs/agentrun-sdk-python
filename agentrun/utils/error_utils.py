"""Small helpers for model rate-limit errors."""

import re
from typing import Any, Dict, Optional

RATE_LIMITED_CODE = "RATE_LIMITED"
RATE_LIMITED_RETRY_AFTER_MS = 2000
_RATE_LIMIT_TEXT_RE = re.compile(
    r"429|too[-_\s]*many[-_\s]*requests|rate[-_\s]*limit|throttl",
    re.IGNORECASE,
)
_RATE_LIMIT_CODES = {
    "ratelimitexceeded",
    "ratelimited",
    "throttling",
    "toomanyrequests",
}


def build_error_event_data(
    error: Any,
    *,
    fallback_code: str,
    fallback_message: str,
) -> Dict[str, Any]:
    """Keep the original message; add rate-limit metadata only when matched."""
    if not is_rate_limited_error(error):
        return {"message": fallback_message, "code": fallback_code}

    data: Dict[str, Any] = {
        "message": fallback_message,
        "code": RATE_LIMITED_CODE,
        "retryable": True,
        "retryAfterMs": RATE_LIMITED_RETRY_AFTER_MS,
    }
    trace_id = _get_value(error, "trace_id") or _get_value(error, "traceId")
    if trace_id:
        data["traceId"] = str(trace_id)
    return data


def is_rate_limited_error(error: Any) -> bool:
    if error is None:
        return False
    if _status_code(error) == 429 or _status_code(_get_value(error, "response")) == 429:
        return True
    if _rate_limit_code(error) or _rate_limit_code(_get_value(error, "response")):
        return True
    return bool(_RATE_LIMIT_TEXT_RE.search(str(error)))


def _status_code(obj: Any) -> Optional[int]:
    for name in ("status_code", "status", "http_status", "statusCode"):
        value = _get_value(obj, name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _rate_limit_code(obj: Any) -> bool:
    for name in ("code", "error_code", "errorCode"):
        code = _get_value(obj, name)
        if code and _normalize_code(code) in _RATE_LIMIT_CODES:
            return True
    return False


def _get_value(obj: Any, name: str) -> Optional[Any]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _normalize_code(code: Any) -> str:
    value = "".join(ch for ch in str(code).lower() if ch.isalnum())
    for suffix in ("exception", "error"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value
