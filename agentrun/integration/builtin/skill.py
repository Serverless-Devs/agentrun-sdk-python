"""内置 Skill 集成函数 / Built-in Skill Integration Functions

提供快速创建 Skill 工具集对象的便捷函数。
Provides convenient functions for quickly creating Skill toolset objects.
"""

from typing import List, Optional, Union

from agentrun.integration.utils.skill_loader import skill_tools as _skill_tools
from agentrun.integration.utils.tool import CommonToolSet
from agentrun.utils.config import Config

# Re-export for convenience
skill_tools = _skill_tools

__all__ = ["skill_tools"]
