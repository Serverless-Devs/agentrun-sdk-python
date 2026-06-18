"""Unit tests for AgentRun SDK logging."""

import logging
import time

import pytest

from agentrun.utils.log import CustomFormatter, _utc8_converter


def make_record(level: int, level_name: str) -> logging.LogRecord:
    record = logging.LogRecord(
        name="agentrun-logger",
        level=level,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.created = 0
    record.levelname = level_name
    return record


def test_utc8_converter_is_independent_from_local_timezone():
    assert _utc8_converter(0) == time.gmtime(8 * 60 * 60)


def test_custom_formatter_uses_utc8_for_all_inner_formatters():
    formatter = CustomFormatter()
    expected = time.gmtime(8 * 60 * 60)

    for inner in formatter._formatters.values():
        assert inner.converter(0) == expected
    assert formatter._default.converter(0) == expected


@pytest.mark.parametrize(
    ("level", "level_name"),
    [
        (logging.DEBUG, "DEBUG"),
        (logging.INFO, "INFO"),
        (logging.WARNING, "WARNING"),
        (logging.ERROR, "ERROR"),
        (logging.CRITICAL, "CRITICAL"),
    ],
)
def test_custom_formatter_formats_known_levels_in_utc8(
    level: int, level_name: str
):
    formatter = CustomFormatter()

    output = formatter.format(make_record(level, level_name))

    assert "1970-01-01 08:00:00" in output
    assert "1970-01-01 00:00:00" not in output


def test_custom_formatter_formats_fallback_level_in_utc8():
    formatter = CustomFormatter()

    output = formatter.format(make_record(5, "TRACE"))

    assert "1970-01-01 08:00:00" in output
    assert "1970-01-01 00:00:00" not in output
