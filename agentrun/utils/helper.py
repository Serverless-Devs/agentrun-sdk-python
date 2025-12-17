"""辅助工具模块 / Helper Utilities Module

此模块提供一些通用的辅助函数。
This module provides general utility functions.
"""

from typing import Any, Optional

from typing_extensions import NotRequired, TypedDict, Unpack


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


class MergeOptions(TypedDict):
    concat_list: NotRequired[bool]
    no_new_field: NotRequired[bool]
    ignore_empty_list: NotRequired[bool]


def merge(a: Any, b: Any, **args: Unpack[MergeOptions]) -> Any:
    """通用合并函数 / Generic deep merge helper.

    合并规则概览:
    - 若 ``b`` 为 ``None``: 返回 ``a``
    - 若 ``a`` 为 ``None``: 返回 ``b``
    - ``dict``: 递归按 key 深度合并
    - ``list``: 连接列表 ``a + b``
    - ``tuple``: 连接元组 ``a + b``
    - ``set``/``frozenset``: 取并集
    - 具有 ``__dict__`` 的同类型对象: 按属性字典递归合并后构造新实例
    - 其他类型: 直接返回 ``b`` (视为覆盖)
    """

    # None 合并: 保留非 None 一方
    if b is None:
        return a
    if a is None:
        return b

    # dict 深度合并
    if isinstance(a, dict) and isinstance(b, dict):
        result: dict[Any, Any] = dict(a)
        for key, value in b.items():
            if key in result:
                result[key] = merge(result[key], value, **args)
            else:
                if args.get("no_new_field", False):
                    continue
                result[key] = value
        return result

    # list 合并: 连接
    if isinstance(a, list) and isinstance(b, list):
        if args.get("concat_list", False):
            return [*a, *b]
        if args.get("ignore_empty_list", False):
            if len(b) == 0:
                return a
        return b

    # tuple 合并: 连接
    if isinstance(a, tuple) and isinstance(b, tuple):
        return (*a, *b)

    # set / frozenset: 并集
    if isinstance(a, set) and isinstance(b, set):
        return a | b
    if isinstance(a, frozenset) and isinstance(b, frozenset):
        return a | b

    # 同类型且具备 __dict__ 的对象: 按属性递归合并, 就地更新 a
    if type(a) is type(b) and hasattr(a, "__dict__") and hasattr(b, "__dict__"):
        for key, value in b.__dict__.items():
            if key in a.__dict__:
                setattr(a, key, merge(getattr(a, key), value, **args))
            else:
                if args.get("no_new_field", False):
                    continue
                setattr(a, key, value)
        return a

    # 其他情况: 视为覆盖, 返回 b
    return b
