"""CrewAI 内置集成函数 / CrewAI Built-in Integration Functions

提供快速创建 CrewAI 兼容模型和工具的便捷函数。
Provides convenient functions for quickly creating CrewAI-compatible models and tools.
"""

from typing import Any, Callable, List, Optional, Union

from typing_extensions import Unpack

from agentrun.integration.builtin import model as _model
from agentrun.integration.builtin import ModelArgs
from agentrun.integration.builtin import sandbox_toolset as _sandbox_toolset
from agentrun.integration.builtin import toolset as _toolset
from agentrun.integration.utils.tool import Tool
from agentrun.model import ModelProxy, ModelService
from agentrun.sandbox import TemplateType
from agentrun.toolset import ToolSet
from agentrun.utils.config import Config


def model(
    name: Union[str, ModelProxy, ModelService],
    **kwargs: Unpack[ModelArgs],
):
    """获取 AgentRun 模型并转换为 LangChain ``BaseChatModel``。 / CrewAI Built-in Integration Functions"""

    m = _model(input=name, **kwargs)  # type: ignore
    return m.to_crewai()


def toolset(
    name: Union[str, ToolSet],
    *,
    prefix: Optional[str] = None,
    modify_tool_name: Optional[Callable[[Tool], Tool]] = None,
    filter_tools_by_name: Optional[Callable[[str], bool]] = None,
    config: Optional[Config] = None,
) -> List[Any]:
    """将内置工具集封装为 LangChain ``StructuredTool`` 列表。 / CrewAI Built-in Integration Functions"""

    ts = _toolset(input=name, config=config)
    return ts.to_crewai(
        prefix=prefix,
        modify_tool_name=modify_tool_name,
        filter_tools_by_name=filter_tools_by_name,
    )


def sandbox_toolset(
    template_name: str,
    *,
    template_type: TemplateType = TemplateType.CODE_INTERPRETER,
    config: Optional[Config] = None,
    sandbox_idle_timeout_seconds: int = 600,
    prefix: Optional[str] = None,
) -> List[Any]:
    """将沙箱模板封装为 LangChain ``StructuredTool`` 列表。 / CrewAI Built-in Integration Functions"""

    return _sandbox_toolset(
        template_name=template_name,
        template_type=template_type,
        config=config,
        sandbox_idle_timeout_seconds=sandbox_idle_timeout_seconds,
    ).to_crewai(prefix=prefix)
