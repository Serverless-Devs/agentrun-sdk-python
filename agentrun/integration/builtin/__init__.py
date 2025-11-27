"""Builtin Integration 模块 / Builtin Integration Module

此模块提供内置的集成函数,用于快速创建模型和工具。
This module provides built-in integration functions for quickly creating models and tools.
"""

from .model import model, ModelArgs
from .sandbox import sandbox_toolset
from .toolset import toolset

__all__ = [
    "model",
    "ModelArgs",
    "toolset",
    "sandbox_toolset",
]
