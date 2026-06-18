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

    COLORS: dict[str, str] = {
        "DEBUG": "\x1b[36m",
        "INFO": "\x1b[34m",
        "WARNING": "\x1b[33m",
        "ERROR": "\x1b[31m",
        "CRITICAL": "\x1b[1;31m",
    }
    RESET = "\x1b[0m"
    DIM = "\x1b[2;3m"
    DIM_ONLY = "\x1b[2m"

    def __init__(self) -> None:
        super().__init__()
        self._formatters: dict[str, logging.Formatter] = {}
        for level, color in self.COLORS.items():
            if level == "DEBUG":
                fmt = (
                    f"\n{color}%(levelname)s{self.RESET} {color}[%(name)s]"
                    " %(asctime)s"
                    f" {self.DIM}%(pathname)s:%(lineno)s{self.RESET}"
                    f"\n{self.DIM_ONLY}%(message)s{self.RESET}"
                )
            else:
                fmt = (
                    f"\n{color}%(levelname)s{self.RESET} {color}[%(name)s]"
                    " %(asctime)s"
                    f" {self.DIM}%(pathname)s:%(lineno)s{self.RESET}"
                    "\n%(message)s"
                )
            self._formatters[level] = logging.Formatter(fmt)
        self._default = logging.Formatter(
            "\n%(levelname)s [%(name)s] %(asctime)s"
            " %(pathname)s:%(lineno)s\n%(message)s"
        )

    def format(self, record: logging.LogRecord) -> str:
        return self._formatters.get(record.levelname, self._default).format(
            record
        )


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
