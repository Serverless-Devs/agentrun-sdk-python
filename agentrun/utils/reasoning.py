"""Utilities for reasoning content extraction and gating."""

import json
import os
from collections.abc import Mapping
from typing import Any, Optional


def parse_bool(value: Any) -> Optional[bool]:
    """Parse loose boolean values used by env-provided model parameters."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return None


def is_thinking_enabled_from_env(
    environ: Mapping[str, str] = os.environ,
) -> bool:
    """Return whether MODEL_PARAMETER_RULES enables thinking."""
    return get_thinking_value_from_env(environ) is True


def get_thinking_value_from_env(
    environ: Mapping[str, str] = os.environ,
) -> Optional[bool]:
    """Return the optional thinking value from MODEL_PARAMETER_RULES."""
    raw_rules = environ.get("MODEL_PARAMETER_RULES")
    if not raw_rules:
        return None

    try:
        rules = json.loads(raw_rules)
    except (TypeError, ValueError):
        return None

    return _extract_thinking_value(rules)


def get_reasoning_content(chunk_or_message: Any) -> Optional[str]:
    """Extract reasoning_content from common model chunk/message shapes."""
    value = _read_attr_or_key(chunk_or_message, "reasoning_content")
    if value is not None:
        return value

    additional_kwargs = _read_attr_or_key(
        chunk_or_message, "additional_kwargs"
    )
    if isinstance(additional_kwargs, Mapping):
        value = additional_kwargs.get("reasoning_content")
        if value is not None:
            return value

    return None


def _extract_thinking_value(value: Any) -> Optional[bool]:
    if isinstance(value, Mapping):
        direct = parse_bool(value.get("thinking"))
        if direct is not None:
            return direct

        for nested_key in ("model_parameter_rules", "parameters", "rules"):
            nested = value.get(nested_key)
            nested_value = _extract_thinking_value(nested)
            if nested_value is not None:
                return nested_value

        if value.get("name") == "thinking":
            for candidate_key in ("value", "default", "enabled"):
                parsed = parse_bool(value.get(candidate_key))
                if parsed is not None:
                    return parsed

    if isinstance(value, list):
        for item in value:
            parsed = _extract_thinking_value(item)
            if parsed is not None:
                return parsed

    return None


def _read_attr_or_key(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)
