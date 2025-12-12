"""LangGraph 集成模块

使用 to_agui_events 将 LangGraph 事件转换为 AG-UI 协议事件：

    >>> from agentrun.integration.langgraph import to_agui_events
    >>>
    >>> async def invoke_agent(request: AgentRequest):
    ...     input_data = {"messages": [...]}
    ...     async for event in agent.astream_events(input_data, version="v2"):
    ...         for item in to_agui_events(event):
    ...             yield item

支持多种调用方式：
- agent.astream_events(input, version="v2") - 支持 token by token
- agent.stream(input, stream_mode="updates") - 按节点输出
- agent.astream(input, stream_mode="updates") - 异步按节点输出
"""

from .agent_converter import convert, to_agui_events
from .builtin import model, sandbox_toolset, toolset

__all__ = [
    "to_agui_events",
    "convert",  # 兼容旧代码
    "model",
    "toolset",
    "sandbox_toolset",
]
