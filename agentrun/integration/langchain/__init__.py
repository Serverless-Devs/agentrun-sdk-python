"""LangChain 集成模块

使用 AgentRunConverter 将 LangChain 事件转换为 AG-UI 协议事件：

    >>> from agentrun.integration.langchain import AgentRunConverter
    >>>
    >>> async def invoke_agent(request: AgentRequest):
    ...     converter = AgentRunConverter()
    ...     async for event in agent.astream_events(input_data, version="v2"):
    ...         for item in converter.convert(event):
    ...             yield item

支持多种调用方式：
- agent.astream_events(input, version="v2") - 支持 token by token
- agent.stream(input, stream_mode="updates") - 按节点输出
- agent.astream(input, stream_mode="updates") - 异步按节点输出
"""

from agentrun.integration.langgraph.agent_converter import (
    AgentRunConverter,
)  # 向后兼容

from .builtin import model, sandbox_toolset, toolset

__all__ = [
    "AgentRunConverter",
    "model",
    "toolset",
    "sandbox_toolset",
]
