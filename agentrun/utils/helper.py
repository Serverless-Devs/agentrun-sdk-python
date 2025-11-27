"""辅助工具模块 / Helper Utilities Module

此模块提供一些通用的辅助函数。
This module provides general utility functions.
"""

from typing import Optional


def mask_password(password: Optional[str]) -> str:
    """遮蔽密码用于日志记录 / Mask password for logging purposes

    将密码部分字符替换为星号,用于安全地记录日志。
    Replaces part of the password characters with asterisks for safe logging.

    Args:
        password: 原始密码,可选 / Original password, optional

    Returns:
        str: 遮蔽后的密码 / Masked password

    Examples:
        >>> mask_password("password123")
        'pa******23'
        >>> mask_password("abc")
        'a*c'
    """
    if not password:
        return ""
    if len(password) <= 2:
        return "*" * len(password)
    if len(password) <= 4:
        return password[0] + "*" * (len(password) - 2) + password[-1]
    return password[0:2] + "*" * (len(password) - 4) + password[-2:]
