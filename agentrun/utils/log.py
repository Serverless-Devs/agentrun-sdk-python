"""日志模块 / Logging Module

此模块配置 AgentRun SDK 的日志系统。
This module configures the logging system for AgentRun SDK.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()


class CustomFormatter(logging.Formatter):
    """自定义日志格式化器 / Custom Log Formatter

    提供带颜色的日志输出格式。
    Provides colorful log output format.
    """

    FORMATS = {
        "DEBUG": (
            "\n\x1b[1;36m%(levelname)s\x1b[0m \x1b[36m[%(name)s] %(asctime)s"
            " \x1b[2;3m%(pathname)s:%(lineno)s\x1b[0m\n\x1b[2m%(message)s\x1b[0m\n"
        ),
        "INFO": (
            "\n\x1b[1;34m%(levelname)s\x1b[0m \x1b[34m[%(name)s] %(asctime)s"
            " \x1b[2;3m%(pathname)s:%(lineno)s\x1b[0m\n%(message)s\n"
        ),
        "WARNING": (
            "\n\x1b[1;33m%(levelname)s\x1b[0m \x1b[33m[%(name)s] %(asctime)s"
            " \x1b[2;3m%(pathname)s:%(lineno)s\x1b[0m\n%(message)s\n"
        ),
        "ERROR": (
            "\n\x1b[1;31m%(levelname)s\x1b[0m \x1b[31m[%(name)s] %(asctime)s"
            " \x1b[2;3m%(pathname)s:%(lineno)s\x1b[0m\n%(message)s\n"
        ),
        "CRITICAL": (
            "\n\x1b[1;31m%(levelname)s\x1b[0m \x1b[31m[%(name)s] %(asctime)s"
            " \x1b[2;3m%(pathname)s:%(lineno)s\x1b[0m\n%(message)s\n"
        ),
        "DEFAULT": (
            "\n%(levelname)s [%(name)s] %(asctime)s"
            " \x1b[2;3m%(pathname)s:%(lineno)s\x1b[0m\n%(message)s\n"
        ),
    }

    def format(self, record):
        formatter = logging.Formatter(
            self.FORMATS.get(record.levelname, self.FORMATS["DEFAULT"])
        )
        return formatter.format(record)


logger = logging.getLogger("agentrun-logger")

logger.setLevel(level=logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)


if os.getenv("AGENTRUN_SDK_DEBUG") not in [
    None,
    "",
    "False",
    "FALSE",
    "false",
    "0",
]:
    logger.setLevel(logging.DEBUG)
    logger.warning(
        "启用 AgentRun SDK 调试日志， 移除 AGENTRUN_SDK_DEBUG 环境变量以关闭"
    )
else:
    logger.setLevel(logging.INFO)
