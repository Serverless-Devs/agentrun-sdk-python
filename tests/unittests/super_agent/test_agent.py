"""Unit tests for ``agentrun.super_agent.agent.SuperAgent``."""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.super_agent.agent import SuperAgent
from agentrun.super_agent.model import InvokeResponseData
from agentrun.super_agent.stream import InvokeStream


def _make_agent() -> SuperAgent:
    return SuperAgent(name="demo")


def _mock_data_api(invoke_result: InvokeResponseData = None):
    """Return a MagicMock replacing SuperAgentDataAPI constructor."""
    instance = MagicMock()
    instance.invoke_async = AsyncMock(return_value=invoke_result)
    instance.stream_async = MagicMock()
    instance.get_conversation_async = AsyncMock()
    instance.delete_conversation_async = AsyncMock()
    factory = MagicMock(return_value=instance)
    return factory, instance


# ─── invoke_async ───────────────────────────────────────────


async def test_invoke_async_returns_invoke_stream_with_conversation_id():
    resp = InvokeResponseData(
        conversation_id="c1",
        stream_url="https://stream/",
        stream_headers={"X-Super-Agent-Session-Id": "sess"},
    )
    factory, _ = _mock_data_api(resp)
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        agent = _make_agent()
        stream = await agent.invoke_async([{"role": "user", "content": "hi"}])
    assert stream.conversation_id == "c1"
    assert stream.session_id == "sess"
    assert stream.stream_url == "https://stream/"
    assert stream.stream_headers == {"X-Super-Agent-Session-Id": "sess"}


async def test_invoke_async_no_tools_kwarg_raises():
    agent = _make_agent()
    with pytest.raises(TypeError):
        await agent.invoke_async([], tools=["t"])  # type: ignore[call-arg]


async def test_invoke_async_forwards_business_fields():
    """SuperAgent 实例字段 MUST 作为 forwarded_extras 透传给 DataAPI."""
    resp = InvokeResponseData(
        conversation_id="c",
        stream_url="https://x/",
        stream_headers={},
    )
    invoke_mock = AsyncMock(return_value=resp)
    instance = MagicMock()
    instance.invoke_async = invoke_mock
    factory = MagicMock(return_value=instance)
    agent = SuperAgent(
        name="demo",
        prompt="p",
        agents=["a1"],
        tools=["t1", "t2"],
        skills=["s1"],
        sandboxes=["sb1"],
        workspaces=["w1"],
        model_service_name="svc",
        model_name="mod",
    )
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        await agent.invoke_async([{"role": "user", "content": "hi"}])

    extras = invoke_mock.await_args.kwargs["forwarded_extras"]
    assert extras == {
        "prompt": "p",
        "agents": ["a1"],
        "tools": ["t1", "t2"],
        "skills": ["s1"],
        "sandboxes": ["sb1"],
        "workspaces": ["w1"],
        "modelServiceName": "svc",
        "modelName": "mod",
    }


async def test_invoke_async_forwards_business_fields_defaults():
    """没设置的 scalar 字段保留 None, list 字段为 []."""
    resp = InvokeResponseData(
        conversation_id="c", stream_url="https://x/", stream_headers={}
    )
    invoke_mock = AsyncMock(return_value=resp)
    instance = MagicMock()
    instance.invoke_async = invoke_mock
    factory = MagicMock(return_value=instance)
    agent = SuperAgent(name="demo")
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        await agent.invoke_async([])
    extras = invoke_mock.await_args.kwargs["forwarded_extras"]
    assert extras == {
        "prompt": None,
        "agents": [],
        "tools": [],
        "skills": [],
        "sandboxes": [],
        "workspaces": [],
        "modelServiceName": None,
        "modelName": None,
    }


async def test_invoke_async_concurrent_streams_independent():
    responses = [
        InvokeResponseData(
            conversation_id=f"c{i}",
            stream_url=f"https://s{i}",
            stream_headers={},
        )
        for i in range(2)
    ]
    invoke_mock = AsyncMock(side_effect=responses)
    instance = MagicMock()
    instance.invoke_async = invoke_mock
    factory = MagicMock(return_value=instance)
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        agent = _make_agent()
        s0, s1 = await asyncio.gather(
            agent.invoke_async([]),
            agent.invoke_async([]),
        )
    ids = {s0.conversation_id, s1.conversation_id}
    assert ids == {"c0", "c1"}


async def test_invoke_async_phase2_lazy():
    resp = InvokeResponseData(
        conversation_id="c",
        stream_url="https://x/",
        stream_headers={},
    )
    invoke_mock = AsyncMock(return_value=resp)
    stream_mock = MagicMock()
    instance = MagicMock()
    instance.invoke_async = invoke_mock
    instance.stream_async = stream_mock
    factory = MagicMock(return_value=instance)
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        agent = _make_agent()
        stream = await agent.invoke_async([])
    # At this point Phase 2 must NOT have been called
    stream_mock.assert_not_called()
    # The stream factory stored inside InvokeStream should only invoke stream_async when iteration starts
    assert isinstance(stream, InvokeStream)


# ─── get_conversation_async ─────────────────────────────────


async def test_get_conversation_async_returns_conversation_info():
    instance = MagicMock()
    instance.get_conversation_async = AsyncMock(
        return_value={
            "conversationId": "c1",
            "agentId": "ag",
            "title": "t",
            "mainUserId": "u1",
            "subUserId": "u2",
            "createdAt": 100,
            "updatedAt": 200,
            "errorMessage": None,
            "invokeInfo": {"foo": "bar"},
            "messages": [
                {"role": "user", "content": "hi", "messageId": "m1"},
                {"role": "assistant", "content": "hello"},
            ],
            "params": {"a": 1},
        }
    )
    factory = MagicMock(return_value=instance)
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        info = await _make_agent().get_conversation_async("c1")
    assert info.conversation_id == "c1"
    assert info.agent_id == "ag"
    assert info.title == "t"
    assert info.main_user_id == "u1"
    assert info.sub_user_id == "u2"
    assert info.created_at == 100
    assert info.updated_at == 200
    assert info.invoke_info == {"foo": "bar"}
    assert len(info.messages) == 2
    assert info.messages[0].message_id == "m1"
    assert info.params == {"a": 1}


async def test_get_conversation_async_partial_fields():
    instance = MagicMock()
    instance.get_conversation_async = AsyncMock(return_value={"agentId": "x"})
    factory = MagicMock(return_value=instance)
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        info = await _make_agent().get_conversation_async("c1")
    assert info.conversation_id == "c1"  # fallback from argument
    assert info.title is None
    assert info.main_user_id is None
    assert info.created_at == 0


async def test_get_conversation_async_empty_messages():
    instance = MagicMock()
    instance.get_conversation_async = AsyncMock(return_value={"messages": []})
    factory = MagicMock(return_value=instance)
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        info = await _make_agent().get_conversation_async("c1")
    assert info.messages == []


async def test_delete_conversation_async_returns_none():
    instance = MagicMock()
    instance.delete_conversation_async = AsyncMock(return_value=None)
    factory = MagicMock(return_value=instance)
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        assert await _make_agent().delete_conversation_async("c") is None


# ─── list_conversations_async ──────────────────────────────


async def test_list_conversations_async_default_filters_by_agent_name():
    """不传 metadata 时, 默认按 ``{"agentRuntimeName": self.name}`` 过滤."""
    instance = MagicMock()
    instance.list_conversations_async = AsyncMock(return_value=[])
    factory = MagicMock(return_value=instance)
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        agent = SuperAgent(name="my-agent")
        result = await agent.list_conversations_async()
    assert result == []
    assert instance.list_conversations_async.await_args.kwargs["metadata"] == {
        "agentRuntimeName": "my-agent"
    }


async def test_list_conversations_async_explicit_metadata_passthrough():
    instance = MagicMock()
    instance.list_conversations_async = AsyncMock(return_value=[])
    factory = MagicMock(return_value=instance)
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        agent = _make_agent()
        await agent.list_conversations_async(metadata={"foo": "bar"})
    assert instance.list_conversations_async.await_args.kwargs["metadata"] == {
        "foo": "bar"
    }


async def test_list_conversations_async_empty_metadata_overrides_default():
    """传入空 dict 明确表示「不按 agent 过滤」, SDK MUST 不再注入默认值."""
    instance = MagicMock()
    instance.list_conversations_async = AsyncMock(return_value=[])
    factory = MagicMock(return_value=instance)
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        agent = _make_agent()
        await agent.list_conversations_async(metadata={})
    assert instance.list_conversations_async.await_args.kwargs["metadata"] == {}


async def test_list_conversations_async_returns_conversation_info_list():
    instance = MagicMock()
    instance.list_conversations_async = AsyncMock(
        return_value=[
            {
                "conversationId": "c1",
                "agentId": "ag",
                "title": "first",
                "createdAt": 1,
                "updatedAt": 2,
                "messages": [{"role": "user", "content": "hi"}],
            },
            {
                "conversationId": "c2",
                "agentId": "ag",
                "title": "second",
                "messages": [],
            },
        ]
    )
    factory = MagicMock(return_value=instance)
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        result = await _make_agent().list_conversations_async()
    assert [c.conversation_id for c in result] == ["c1", "c2"]
    assert result[0].title == "first"
    assert len(result[0].messages) == 1
    assert result[0].messages[0].content == "hi"
    assert result[1].title == "second"


async def test_list_conversations_async_empty_list():
    instance = MagicMock()
    instance.list_conversations_async = AsyncMock(return_value=[])
    factory = MagicMock(return_value=instance)
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        assert await _make_agent().list_conversations_async() == []


async def test_list_conversations_async_item_missing_conversation_id_uses_empty_fallback():
    """对单条会话里 ``conversationId`` 缺失的情况, fallback 保持空串 (不会用 agent name)."""
    instance = MagicMock()
    instance.list_conversations_async = AsyncMock(
        return_value=[{"agentId": "ag"}]
    )
    factory = MagicMock(return_value=instance)
    with patch("agentrun.super_agent.agent.SuperAgentDataAPI", factory):
        result = await _make_agent().list_conversations_async()
    assert len(result) == 1
    assert result[0].conversation_id == ""


# ─── sync methods → NotImplementedError ─────────────────────


def test_sync_methods_not_implemented():
    agent = _make_agent()
    with pytest.raises(NotImplementedError):
        agent.invoke([])
    with pytest.raises(NotImplementedError):
        agent.get_conversation("c")
    with pytest.raises(NotImplementedError):
        agent.delete_conversation("c")
    with pytest.raises(NotImplementedError):
        agent.list_conversations()


def test_invoke_async_signature_only_messages_and_conversation_id():
    sig = inspect.signature(SuperAgent.invoke_async)
    params = list(sig.parameters.keys())
    # self, messages, then KEYWORD_ONLY: conversation_id, config
    assert params[:2] == ["self", "messages"]
    assert "conversation_id" in params
    assert "config" in params
    assert "tools" not in params
