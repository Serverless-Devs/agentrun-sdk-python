"""Error helpers for AgentRun event streams."""

import re
from typing import Any, Dict, Optional

RATE_LIMITED_CODE = "RATE_LIMITED"
RATE_LIMITED_RETRY_AFTER_MS = 2000

_RATE_LIMIT_CODES = {
    "ratelimitexceeded",
    "ratelimited",
    "resourcethrottled",
    "throttling",
    "throttlingquota",
    "throttlingratequota",
    "throttlingexception",
    "toomanyrequests",
}

_RATE_LIMIT_TEXT_PATTERNS = [
    re.compile(r"\btoo[-_\s]*many[-_\s]*requests\b", re.IGNORECASE),
    re.compile(
        r"\brate[-_\s]*limit(?:ed|[-_\s]*exceeded)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bresource[-_\s]*throttled\b", re.IGNORECASE),
    re.compile(
        r"\b(?:throttling|throttlingexception|throttled)\b",
        re.IGNORECASE,
    ),
]


def format_error_message(error: Any) -> str:
    """Format errors consistently with existing LangGraph conversion."""
    if error is None:
        return ""
    if isinstance(error, Exception):
        return f"{type(error).__name__}: {str(error)}"
    return str(error)


def build_error_event_data(
    error: Any,
    *,
    fallback_code: str,
    fallback_message: str,
) -> Dict[str, Any]:
    """Build AgentEvent ERROR data, normalizing model rate limits."""
    if not is_rate_limited_error(error):
        return {"message": fallback_message, "code": fallback_code}

    data: Dict[str, Any] = {
        "message": fallback_message,
        "code": RATE_LIMITED_CODE,
        "retryable": True,
        "retryAfterMs": RATE_LIMITED_RETRY_AFTER_MS,
    }
    trace_id = _extract_trace_id(error)
    if trace_id:
        data["traceId"] = str(trace_id)
    return data


def is_rate_limited_error(error: Any) -> bool:
    """Return whether an error carries an explicit rate-limit signal."""
    if _extract_status_code(error) == 429:
        return True

    if _has_rate_limit_code(error):
        return True

    message = str(error or "")
    return any(pattern.search(message) for pattern in _RATE_LIMIT_TEXT_PATTERNS)


def _extract_status_code(error: Any) -> Optional[int]:
    fallback = None
    for obj in (error, _get_value(error, "response")):
        if obj is None:
            continue
        for name in ("status_code", "status", "http_status", "statusCode"):
            status_code = _to_int(_get_value(obj, name))
            if status_code == 429:
                return status_code
            if fallback is None and status_code is not None:
                fallback = status_code
    return fallback


def _has_rate_limit_code(error: Any) -> bool:
    for obj in (error, _get_value(error, "response")):
        if obj is None:
            continue
        for name in ("code", "error_code", "errorCode"):
            error_code = _get_value(obj, name)
            if (
                error_code is not None
                and _normalize_code(error_code) in _RATE_LIMIT_CODES
            ):
                return True
    return False


def _extract_trace_id(error: Any) -> Optional[Any]:
    for name in ("trace_id", "traceId", "request_id", "requestId"):
        trace_id = _get_value(error, name)
        if trace_id:
            return trace_id

    response = _get_value(error, "response")
    headers = _get_value(response, "headers")
    if not headers:
        return None

    for name in ("x-acs-request-id", "x-request-id", "x-trace-id", "trace-id"):
        trace_id = _get_header(headers, name)
        if trace_id:
            return trace_id
    return None


def _get_value(obj: Any, name: str) -> Optional[Any]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _get_header(headers: Any, name: str) -> Optional[Any]:
    target = str(name).lower()
    if isinstance(headers, dict):
        for key, value in headers.items():
            if str(key).lower() == target:
                return value
        return None
    get = getattr(headers, "get", None)
    if callable(get):
        return get(name)
    return None


def _normalize_code(code: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", str(code).lower())
    for suffix in ("exception", "error"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
