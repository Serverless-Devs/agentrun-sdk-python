import asyncio
from typing import AsyncGenerator

import pytest

from agentrun.server.invoker import AgentInvoker
from agentrun.server.model import AgentRequest, AgentRunResult


async def test_invoke_with_async_generator_returns_runresult():
    async def invoke_agent(req: AgentRequest) -> AsyncGenerator[str, None]:
        yield "hello"

    invoker = AgentInvoker(invoke_agent)
    result = await invoker.invoke(AgentRequest(messages=[]))
    assert isinstance(result, AgentRunResult)
    # content should be an async iterator
    assert hasattr(result.content, "__aiter__")


async def test_invoke_with_async_coroutine_returns_runresult():
    async def invoke_agent(req: AgentRequest) -> str:
        return "world"

    invoker = AgentInvoker(invoke_agent)
    result = await invoker.invoke(AgentRequest(messages=[]))
    assert isinstance(result, AgentRunResult)
    assert result.content == "world"
