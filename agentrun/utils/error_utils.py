"""Small helpers for model-side errors."""

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterator, Optional, Tuple

RATE_LIMITED_CODE = "RATE_LIMITED"
RATE_LIMITED_RETRY_AFTER_MS = 2000
_KNOWN_STATUS_CODES = (400, 401, 403, 429, 500, 503)
_HTTP_STATUS_TEXT_RE = re.compile(
    r"(?:error\s+code|status\s+code|http\s+status|http)"
    r"\D{0,20}(400|401|403|429|500|503)\b"
    r"|\b(400|401|403|429|500|503)\s*(?:[-:—]|$)",
    re.IGNORECASE,
)
_RATE_LIMIT_TEXT_RE = re.compile(
    r"429|too[-_\s]*many[-_\s]*requests|rate[-_\s]*limit|throttl",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ModelErrorSpec:
    code: str
    status_code: int
    retryable: bool = False
    retry_after_ms: Optional[int] = None


@dataclass(frozen=True)
class ModelErrorInfo:
    code: str
    status_code: Optional[int] = None
    provider_code: Optional[str] = None
    retryable: bool = False
    retry_after_ms: Optional[int] = None


_RATE_LIMIT_SPEC = _ModelErrorSpec(
    RATE_LIMITED_CODE,
    429,
    retryable=True,
    retry_after_ms=RATE_LIMITED_RETRY_AFTER_MS,
)
_MODEL_ERROR_SPECS = {
    "arrearage": _ModelErrorSpec("MODEL_ARREARAGE", 400),
    "datainspectionfailed": _ModelErrorSpec(
        "MODEL_DATA_INSPECTION_FAILED", 400
    ),
    "invalidapikey": _ModelErrorSpec("MODEL_AUTHENTICATION_ERROR", 401),
    "authentication": _ModelErrorSpec("MODEL_AUTHENTICATION_ERROR", 401),
    "accessdenied": _ModelErrorSpec("MODEL_ACCESS_DENIED", 403),
    "accessdeniedunpurchased": _ModelErrorSpec("MODEL_ACCESS_DENIED", 403),
    "modelaccessdenied": _ModelErrorSpec("MODEL_ACCESS_DENIED", 403),
    "workspaceaccessdenied": _ModelErrorSpec("MODEL_ACCESS_DENIED", 403),
    "throttling": _RATE_LIMIT_SPEC,
    "throttlingratequota": _RATE_LIMIT_SPEC,
    "throttlingallocationquota": _RATE_LIMIT_SPEC,
    "limitrequests": _RATE_LIMIT_SPEC,
    "ratelimit": _RATE_LIMIT_SPEC,
    "ratelimitexceeded": _RATE_LIMIT_SPEC,
    "ratelimited": _RATE_LIMIT_SPEC,
    "toomanyrequests": _RATE_LIMIT_SPEC,
    "internal": _ModelErrorSpec("MODEL_INTERNAL_ERROR", 500, retryable=True),
    "internalerroralgo": _ModelErrorSpec(
        "MODEL_INTERNAL_ERROR", 500, retryable=True
    ),
    "system": _ModelErrorSpec("MODEL_INTERNAL_ERROR", 500, retryable=True),
    "modelserving": _ModelErrorSpec(
        "MODEL_SERVICE_UNAVAILABLE", 503, retryable=True
    ),
    "serviceunavailable": _ModelErrorSpec(
        "MODEL_SERVICE_UNAVAILABLE", 503, retryable=True
    ),
}
_STATUS_CODE_SPECS = {
    400: _ModelErrorSpec("MODEL_BAD_REQUEST", 400),
    401: _ModelErrorSpec("MODEL_AUTHENTICATION_ERROR", 401),
    403: _ModelErrorSpec("MODEL_ACCESS_DENIED", 403),
    429: _RATE_LIMIT_SPEC,
    500: _ModelErrorSpec("MODEL_INTERNAL_ERROR", 500, retryable=True),
    503: _ModelErrorSpec("MODEL_SERVICE_UNAVAILABLE", 503, retryable=True),
}
_TEXT_PROVIDER_PATTERNS: Tuple[Tuple[re.Pattern[str], Optional[str]], ...] = (
    (re.compile(r"\bArrearage\b", re.IGNORECASE), "Arrearage"),
    (
        re.compile(
            r"access\s+denied.*account.*good\s+standing",
            re.IGNORECASE,
        ),
        "Arrearage",
    ),
    (
        re.compile(
            r"\bDataInspectionFailed\b|\bdata[_-]inspection[_-]failed\b",
            re.IGNORECASE,
        ),
        "DataInspectionFailed",
    ),
    (re.compile(r"\bInvalidApiKey\b", re.IGNORECASE), None),
    (
        re.compile(r"\bAuthenticationError\b", re.IGNORECASE),
        None,
    ),
    (
        re.compile(r"\bModel\.AccessDenied\b", re.IGNORECASE),
        None,
    ),
    (
        re.compile(r"\bAccessDenied\.Unpurchased\b", re.IGNORECASE),
        None,
    ),
    (re.compile(r"\bAccessDenied\b", re.IGNORECASE), None),
    (
        re.compile(r"\bWorkspaceAccessDenied\b", re.IGNORECASE),
        None,
    ),
    (
        re.compile(
            r"\bThrottling(?:\.(?:RateQuota|AllocationQuota))?\b",
            re.IGNORECASE,
        ),
        None,
    ),
    (re.compile(r"\bLimitRequests\b", re.IGNORECASE), None),
    (re.compile(r"\bRateLimit\b", re.IGNORECASE), None),
    (
        re.compile(r"\bInternalError(?:\.Algo)?\b", re.IGNORECASE),
        None,
    ),
    (re.compile(r"\bSystemError\b", re.IGNORECASE), None),
    (
        re.compile(r"\bModelServingError\b", re.IGNORECASE),
        None,
    ),
    (
        re.compile(r"\bServiceUnavailable\b", re.IGNORECASE),
        None,
    ),
)


def build_error_event_data(
    error: Any,
    *,
    fallback_code: str,
    fallback_message: str,
) -> Dict[str, Any]:
    """Keep the original message; add model-error metadata when matched."""
    model_error = classify_model_error(error)
    if model_error is None:
        return {"message": fallback_message, "code": fallback_code}

    data: Dict[str, Any] = {
        "message": fallback_message,
        "code": model_error.code,
    }
    if model_error.retryable:
        data["retryable"] = True
    if model_error.retry_after_ms is not None:
        data["retryAfterMs"] = model_error.retry_after_ms
    if model_error.status_code is not None:
        data["statusCode"] = model_error.status_code
    if model_error.provider_code:
        data["providerCode"] = model_error.provider_code
    trace_id = _first_value(error, "trace_id", "traceId")
    if trace_id:
        data["traceId"] = str(trace_id)
    request_id = _first_value(error, "request_id", "requestId")
    if request_id:
        data["requestId"] = str(request_id)
    return data


def classify_model_error(error: Any) -> Optional[ModelErrorInfo]:
    if error is None:
        return None

    status_code = _status_code(error)
    provider_code = _provider_code(error)
    spec = _spec_from_provider_code(provider_code)
    if spec is not None:
        return _to_model_error_info(spec, status_code, provider_code)

    text = str(error)
    provider_code = _provider_code_from_text(text)
    spec = _spec_from_provider_code(provider_code)
    if spec is not None:
        return _to_model_error_info(
            spec,
            status_code or _status_code_from_text(text),
            provider_code,
        )

    if status_code in _STATUS_CODE_SPECS:
        return _to_model_error_info(
            _STATUS_CODE_SPECS[status_code], status_code
        )

    if _RATE_LIMIT_TEXT_RE.search(text):
        return _to_model_error_info(_RATE_LIMIT_SPEC, status_code)

    return None


def is_model_error(error: Any) -> bool:
    return classify_model_error(error) is not None


def is_rate_limited_error(error: Any) -> bool:
    model_error = classify_model_error(error)
    return bool(model_error and model_error.code == RATE_LIMITED_CODE)


def _to_model_error_info(
    spec: _ModelErrorSpec,
    status_code: Optional[int],
    provider_code: Optional[str] = None,
) -> ModelErrorInfo:
    return ModelErrorInfo(
        code=spec.code,
        status_code=status_code or spec.status_code,
        provider_code=provider_code,
        retryable=spec.retryable,
        retry_after_ms=spec.retry_after_ms,
    )


def _spec_from_provider_code(
    provider_code: Optional[str],
) -> Optional[_ModelErrorSpec]:
    if not provider_code:
        return None
    return _MODEL_ERROR_SPECS.get(_normalize_code(provider_code))


def _status_code(obj: Any) -> Optional[int]:
    for part in _iter_error_parts(obj):
        for name in ("status_code", "status", "http_status", "statusCode"):
            value = _get_value(part, name)
            if value is None:
                continue
            try:
                status_code = int(value)
            except (TypeError, ValueError):
                continue
            if status_code in _KNOWN_STATUS_CODES:
                return status_code
    return None


def _status_code_from_text(text: str) -> Optional[int]:
    match = _HTTP_STATUS_TEXT_RE.search(text)
    if not match:
        return None
    for value in match.groups():
        if value:
            return int(value)
    return None


def _provider_code(obj: Any) -> Optional[str]:
    for part in _iter_error_parts(obj):
        for name in ("code", "error_code", "errorCode"):
            code = _get_value(part, name)
            if code is None or isinstance(code, (dict, list, tuple)):
                continue
            value = str(code)
            if not value.isdigit():
                return value
    return None


def _provider_code_from_text(text: str) -> Optional[str]:
    for pattern, provider_code in _TEXT_PROVIDER_PATTERNS:
        match = pattern.search(text)
        if match:
            return provider_code or match.group(0)
    return None


def _iter_error_parts(obj: Any) -> Iterator[Any]:
    parts = [obj]
    seen = set()
    index = 0
    while index < len(parts) and index < 20:
        part = parts[index]
        index += 1
        if part is None or id(part) in seen:
            continue
        seen.add(id(part))
        yield part
        for name in ("response", "body", "data", "error"):
            value = _get_value(part, name)
            if value is not None and value is not part:
                parts.append(value)
        json_body = _json_body(part)
        if json_body is not None and json_body is not part:
            parts.append(json_body)


def _json_body(obj: Any) -> Optional[Any]:
    json_method = getattr(obj, "json", None)
    if not callable(json_method):
        return None
    try:
        return json_method()
    except Exception:
        return None


def _get_value(obj: Any, name: str) -> Optional[Any]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        value = obj.get(name)
        if value is not None:
            return value
        lower_name = name.lower()
        for key, value in obj.items():
            if isinstance(key, str) and key.lower() == lower_name:
                return value
        return None
    return getattr(obj, name, None)


def _first_value(obj: Any, *names: str) -> Optional[Any]:
    for part in _iter_error_parts(obj):
        for name in names:
            value = _get_value(part, name)
            if value:
                return value
    return None


def _normalize_code(code: Any) -> str:
    value = "".join(ch for ch in str(code).lower() if ch.isalnum())
    for suffix in ("exception", "error"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value
