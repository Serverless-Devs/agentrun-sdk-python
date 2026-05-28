from types import SimpleNamespace

import pytest

from agentrun.server import AgentRequest, EventType, MessageRole
from agentrun.server.invoker import AgentInvoker
from agentrun.server.model import Message
from agentrun.utils.reasoning import (
    get_reasoning_content,
    get_thinking_value_from_env,
    is_thinking_enabled_from_env,
)


def test_model_parameter_rules_object_enables_thinking():
    env = {"MODEL_PARAMETER_RULES": '{"thinking": true}'}

    assert is_thinking_enabled_from_env(env) is True


def test_model_parameter_rules_list_enables_thinking():
    env = {
        "MODEL_PARAMETER_RULES": (
            '[{"name": "temperature", "default": 0.1}, '
            '{"name": "thinking", "default": "true"}]'
        )
    }

    assert is_thinking_enabled_from_env(env) is True
    assert get_thinking_value_from_env(env) is True


def test_model_parameter_rules_invalid_json_disables_thinking():
    env = {"MODEL_PARAMETER_RULES": "not json"}

    assert is_thinking_enabled_from_env(env) is False


def test_get_reasoning_content_from_additional_kwargs():
    chunk = {"additional_kwargs": {"reasoning_content": "thinking"}}

    assert get_reasoning_content(chunk) == "thinking"


@pytest.mark.asyncio
async def test_invoker_converts_chunk_additional_kwargs_to_reasoning_event():
    chunk = SimpleNamespace(
        content="answer",
        additional_kwargs={"reasoning_content": "thinking"},
    )

    async def invoke_agent(request):
        yield chunk

    request = AgentRequest(
        protocol="openai",
        messages=[Message(role=MessageRole.USER, content="hello")],
        stream=True,
        raw_request=None,
    )

    events = [event async for event in AgentInvoker(invoke_agent).invoke_stream(request)]

    assert events[0].event == EventType.REASONING
    assert events[0].data["delta"] == "thinking"
    assert events[1].event == EventType.TEXT
    assert events[1].data["delta"] == "answer"
