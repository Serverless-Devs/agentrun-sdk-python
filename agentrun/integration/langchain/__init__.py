"""LangChain 集成模块

Example:
    >>> from langchain.agents import create_agent
    >>> from agentrun.integration.langchain import convert, model, toolset
    >>>
    >>> agent = create_agent(model=model("my-model"), tools=toolset("my-tools"))
    >>>
    >>> async def invoke_agent(request: AgentRequest):
    ...     input_data = {"messages": [...]}
    ...     async for event in agent.astream_events(input_data, version="v2"):
    ...         for item in convert(event, request.hooks):
    ...             yield item
"""

from agentrun.integration.langgraph.agent_converter import convert

from .builtin import model, sandbox_toolset, toolset

__all__ = [
    "convert",
    "model",
    "toolset",
    "sandbox_toolset",
]
