"""LangGraph 集成模块

Example:
    >>> from langgraph.prebuilt import create_react_agent
    >>> from agentrun.integration.langgraph import convert
    >>>
    >>> agent = create_react_agent(llm, tools)
    >>>
    >>> async def invoke_agent(request: AgentRequest):
    ...     input_data = {"messages": [...]}
    ...     async for event in agent.astream_events(input_data, version="v2"):
    ...         for item in convert(event, request.hooks):
    ...             yield item
"""

from .agent_converter import convert
from .builtin import model, sandbox_toolset, toolset

__all__ = [
    "convert",
    "model",
    "toolset",
    "sandbox_toolset",
]
