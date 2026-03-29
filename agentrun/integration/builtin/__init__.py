"""Builtin Integration 模块 / Builtin Integration Module

此模块提供内置的集成函数,用于快速创建模型和工具。
This module provides built-in integration functions for quickly creating models and tools.
"""

from .knowledgebase import knowledgebase_toolset
from .model import model, ModelArgs
from .sandbox import sandbox_toolset
from .skill import skill_tools
from .tool_resource import tool_resource
from .toolset import toolset

__all__ = [
    "model",
    "ModelArgs",
    "toolset",
    "tool_resource",
    "sandbox_toolset",
    "knowledgebase_toolset",
    "skill_tools",
]
