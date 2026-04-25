"""Unit tests for ``agentrun.super_agent.api.data.SuperAgentDataAPI``."""

import json
import re

import httpx
import pytest
import respx

from agentrun.super_agent.api.data import SuperAgentDataAPI
from agentrun.utils.config import Config


def _auth_cfg(**overrides) -> Config:
    """Config with RAM AK/SK so ``DataAPI.auth`` actually signs."""
    base = dict(
        access_key_id="AK",
        access_key_secret="SK",
        account_id="123",
        region_id="cn-hangzhou",
    )
    base.update(overrides)
    return Config(**base)


# ─── URL construction (production / pre / custom gateway) ────


@respx.mock
async def test_invoke_async_phase1_url_includes_version_production():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("agent-prod", config=cfg)
    route = respx.post(
        re.compile(
            r"https://123-ram\.agentrun-data\.cn-hangzhou\.aliyuncs\.com"
            r"/2025-09-10/super-agents/__SUPER_AGENT__/invoke"
        )
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "conversationId": "c1",
                    "url": "https://x/stream",
                    "headers": {},
                }
            },
        )
    )
    await api.invoke_async([{"role": "user", "content": "hi"}])
    assert route.called


@respx.mock
async def test_invoke_async_phase1_url_pre_environment():
    cfg = _auth_cfg(
        data_endpoint=(
            "http://1431999136518149.funagent-data-pre.cn-hangzhou.aliyuncs.com"
        )
    )
    api = SuperAgentDataAPI("agent-pre", config=cfg)
    route = respx.post(
        re.compile(
            r"http://1431999136518149-ram\.funagent-data-pre\.cn-hangzhou\.aliyuncs\.com"
            r"/2025-09-10/super-agents/__SUPER_AGENT__/invoke"
        )
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "conversationId": "c",
                    "url": "https://stream",
                    "headers": {},
                }
            },
        )
    )
    await api.invoke_async([{"role": "user", "content": "hi"}])
    assert route.called


@respx.mock
async def test_invoke_async_phase1_url_custom_gateway_no_ram():
    cfg = _auth_cfg(data_endpoint="https://my-gateway.example.com")
    api = SuperAgentDataAPI("agent-cust", config=cfg)
    route = respx.post(
        "https://my-gateway.example.com/2025-09-10/super-agents/__SUPER_AGENT__/invoke"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "conversationId": "c",
                    "url": "https://s",
                    "headers": {},
                }
            },
        )
    )
    await api.invoke_async([{"role": "user", "content": "hi"}])
    assert route.called


@respx.mock
async def test_list_conversations_async_url_pre_environment():
    cfg = _auth_cfg(
        data_endpoint="http://111.funagent-data-pre.cn-hangzhou.aliyuncs.com"
    )
    api = SuperAgentDataAPI("n", config=cfg)
    route = respx.get(
        re.compile(
            r"http://111-ram\.funagent-data-pre\.cn-hangzhou\.aliyuncs\.com"
            r"/2025-09-10/super-agents/__SUPER_AGENT__/conversations(\?.*)?$"
        )
    ).mock(
        return_value=httpx.Response(200, json={"data": {"conversations": []}})
    )
    await api.list_conversations_async()
    assert route.called


@respx.mock
async def test_get_conversation_async_url_pre_environment():
    cfg = _auth_cfg(
        data_endpoint="http://111.funagent-data-pre.cn-hangzhou.aliyuncs.com"
    )
    api = SuperAgentDataAPI("n", config=cfg)
    route = respx.get(
        "http://111-ram.funagent-data-pre.cn-hangzhou.aliyuncs.com"
        "/2025-09-10/super-agents/__SUPER_AGENT__/conversations/cid"
    ).mock(return_value=httpx.Response(200, json={"data": {}}))
    await api.get_conversation_async("cid")
    assert route.called


@respx.mock
async def test_delete_conversation_async_url_pre_environment():
    cfg = _auth_cfg(
        data_endpoint="http://111.funagent-data-pre.cn-hangzhou.aliyuncs.com"
    )
    api = SuperAgentDataAPI("n", config=cfg)
    route = respx.delete(
        "http://111-ram.funagent-data-pre.cn-hangzhou.aliyuncs.com"
        "/2025-09-10/super-agents/__SUPER_AGENT__/conversations/cid"
    ).mock(return_value=httpx.Response(200))
    await api.delete_conversation_async("cid")
    assert route.called


# ─── body shape ───────────────────────────────────────────────


@respx.mock
async def test_invoke_async_body_new_conversation():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": {
                    "conversationId": "c",
                    "url": "https://s",
                    "headers": {},
                }
            },
        )

    respx.post(re.compile(r".*/invoke$")).mock(side_effect=_responder)
    await api.invoke_async([{"role": "user", "content": "hi"}])
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["body"]["forwardedProps"]["metadata"] == {
        "agentRuntimeName": "demo"
    }
    assert "conversationId" not in captured["body"]["forwardedProps"]


@respx.mock
async def test_invoke_async_body_continue_conversation():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": {
                    "conversationId": "abc",
                    "url": "https://s",
                    "headers": {},
                }
            },
        )

    respx.post(re.compile(r".*/invoke$")).mock(side_effect=_responder)
    await api.invoke_async(
        [{"role": "user", "content": "hi"}], conversation_id="abc"
    )
    assert captured["body"]["forwardedProps"]["conversationId"] == "abc"


@respx.mock
async def test_invoke_async_body_forwarded_extras_passthrough():
    """``forwarded_extras`` 业务字段 MUST 出现在 forwardedProps 顶层."""
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": {
                    "conversationId": "c",
                    "url": "https://s",
                    "headers": {},
                }
            },
        )

    respx.post(re.compile(r".*/invoke$")).mock(side_effect=_responder)
    await api.invoke_async(
        [{"role": "user", "content": "hi"}],
        forwarded_extras={
            "prompt": "p",
            "agents": ["a1"],
            "modelServiceName": "svc",
            "modelName": "mod",
        },
    )
    fp = captured["body"]["forwardedProps"]
    assert fp["prompt"] == "p"
    assert fp["agents"] == ["a1"]
    assert fp["modelServiceName"] == "svc"
    assert fp["modelName"] == "mod"
    # SDK 托管字段不受 extras 影响
    assert fp["metadata"] == {"agentRuntimeName": "demo"}


@respx.mock
async def test_invoke_async_body_extras_cannot_override_sdk_fields():
    """extras 里带 metadata/conversationId 也不能覆盖 SDK 托管字段."""
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": {
                    "conversationId": "real",
                    "url": "https://s",
                    "headers": {},
                }
            },
        )

    respx.post(re.compile(r".*/invoke$")).mock(side_effect=_responder)
    await api.invoke_async(
        [{"role": "user", "content": "hi"}],
        conversation_id="real",
        forwarded_extras={
            "metadata": {"agentRuntimeName": "SPOOFED"},
            "conversationId": "SPOOFED",
            "prompt": "p",
        },
    )
    fp = captured["body"]["forwardedProps"]
    assert fp["metadata"] == {"agentRuntimeName": "demo"}
    assert fp["conversationId"] == "real"
    assert fp["prompt"] == "p"


@respx.mock
async def test_invoke_async_body_prunes_none_scalars_in_extras():
    """数据面: extras 中值为 None 的 scalar 字段 MUST NOT 进入 forwardedProps."""
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": {
                    "conversationId": "c",
                    "url": "https://s",
                    "headers": {},
                }
            },
        )

    respx.post(re.compile(r".*/invoke$")).mock(side_effect=_responder)
    await api.invoke_async(
        [{"role": "user", "content": "hi"}],
        forwarded_extras={
            "prompt": None,
            "modelServiceName": None,
            "modelName": None,
            "agents": [],
            "tools": ["t1"],
        },
    )
    fp = captured["body"]["forwardedProps"]
    assert "prompt" not in fp
    assert "modelServiceName" not in fp
    assert "modelName" not in fp
    assert "agents" not in fp
    assert fp["tools"] == ["t1"]
    # SDK 托管字段保留
    assert fp["metadata"] == {"agentRuntimeName": "demo"}


@respx.mock
async def test_invoke_async_body_metadata_preserved_even_if_none_in_extras():
    """extras 里带 metadata=None 也不应覆盖 SDK 管理的 metadata."""
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": {
                    "conversationId": "c",
                    "url": "https://s",
                    "headers": {},
                }
            },
        )

    respx.post(re.compile(r".*/invoke$")).mock(side_effect=_responder)
    await api.invoke_async(
        [{"role": "user", "content": "hi"}],
        forwarded_extras={"metadata": None},
    )
    fp = captured["body"]["forwardedProps"]
    assert fp["metadata"] == {"agentRuntimeName": "demo"}


@respx.mock
async def test_invoke_async_body_drops_leaked_conversation_id_from_extras():
    """extras 里若带 conversationId 但 kwarg 里 conversation_id=None, 最终 payload 不含 conversationId.

    回归保护 _build_invoke_body 里的防御性 pop 分支。
    """
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": {
                    "conversationId": "c",
                    "url": "https://s",
                    "headers": {},
                }
            },
        )

    respx.post(re.compile(r".*/invoke$")).mock(side_effect=_responder)
    await api.invoke_async(
        [{"role": "user", "content": "hi"}],
        forwarded_extras={"conversationId": "should-be-dropped"},
    )
    assert "conversationId" not in captured["body"]["forwardedProps"]


# ─── signing ──────────────────────────────────────────────────


@respx.mock
async def test_invoke_async_request_signed():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "data": {
                    "conversationId": "c",
                    "url": "https://s",
                    "headers": {},
                }
            },
        )

    respx.post(re.compile(r".*/invoke$")).mock(side_effect=_responder)
    await api.invoke_async([{"role": "user"}])
    h = captured["headers"]
    assert any(k.lower() == "agentrun-authorization" for k in h)
    assert any(k.lower() == "x-acs-date" for k in h)
    assert any(k.lower() == "x-acs-content-sha256" for k in h)
    ct = next((v for k, v in h.items() if k.lower() == "content-type"), None)
    assert ct == "application/json"


# ─── returns InvokeResponseData ──────────────────────────────


@respx.mock
async def test_invoke_async_returns_invoke_response_data():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    respx.post(re.compile(r".*/invoke$")).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "conversationId": "c",
                    "url": "https://stream",
                    "headers": {"X-Super-Agent-Session-Id": "s"},
                }
            },
        )
    )
    resp = await api.invoke_async([])
    assert resp.conversation_id == "c"
    assert resp.stream_url == "https://stream"
    assert resp.stream_headers == {"X-Super-Agent-Session-Id": "s"}


@respx.mock
async def test_invoke_async_missing_conversation_id_raises():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    respx.post(re.compile(r".*/invoke$")).mock(
        return_value=httpx.Response(
            200, json={"data": {"url": "https://s", "headers": {}}}
        )
    )
    with pytest.raises(ValueError) as exc:
        await api.invoke_async([])
    assert "missing" in str(exc.value)
    assert "conversationId" in str(exc.value)


@respx.mock
async def test_invoke_async_5xx_raises_http_error():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    respx.post(re.compile(r".*/invoke$")).mock(
        return_value=httpx.Response(500, text="boom")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await api.invoke_async([])


@respx.mock
async def test_invoke_async_user_headers_merged():
    cfg = _auth_cfg(headers={"X-Custom": "v"})
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "data": {
                    "conversationId": "c",
                    "url": "https://s",
                    "headers": {},
                }
            },
        )

    respx.post(re.compile(r".*/invoke$")).mock(side_effect=_responder)
    await api.invoke_async([])
    assert captured["headers"].get("x-custom") == "v"
    assert any(
        k.lower() == "agentrun-authorization" for k in captured["headers"]
    )


# ─── stream_async ────────────────────────────────────────────


@respx.mock
async def test_stream_async_yields_sse_events():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    sse_body = b"event: m\ndata: hello\n\nevent: m\ndata: world\n\n"
    respx.get("https://stream.example.com/flow").mock(
        return_value=httpx.Response(200, content=sse_body)
    )
    events = []
    async for ev in api.stream_async("https://stream.example.com/flow"):
        events.append(ev)
    assert [(e.event, e.data) for e in events] == [
        ("m", "hello"),
        ("m", "world"),
    ]


@respx.mock
async def test_stream_async_request_includes_phase1_headers_and_signed():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, content=b":keep\n\n")

    respx.get("https://stream.example.com/go").mock(side_effect=_responder)
    it = api.stream_async(
        "https://stream.example.com/go",
        stream_headers={"X-Super-Agent-Session-Id": "sess1"},
    )
    async for _ in it:
        pass
    h = captured["headers"]
    assert h.get("x-super-agent-session-id") == "sess1"
    assert any(k.lower() == "agentrun-authorization" for k in h)


# ─── get/delete conversation ─────────────────────────────────


@respx.mock
async def test_get_conversation_async_url_and_signed():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"data": {"conversationId": "c1"}})

    respx.get(
        re.compile(
            r".*/2025-09-10/super-agents/__SUPER_AGENT__/conversations/c1$"
        )
    ).mock(side_effect=_responder)
    result = await api.get_conversation_async("c1")
    assert result == {"conversationId": "c1"}
    assert any(
        k.lower() == "agentrun-authorization" for k in captured["headers"]
    )


@respx.mock
async def test_get_conversation_async_returns_empty_on_missing_data():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    respx.get(re.compile(r".*/conversations/.*")).mock(
        return_value=httpx.Response(200, json={"code": "ok"})
    )
    assert await api.get_conversation_async("c1") == {}


@respx.mock
async def test_delete_conversation_async_returns_none():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    route = respx.delete(re.compile(r".*/conversations/c1$")).mock(
        return_value=httpx.Response(200)
    )
    result = await api.delete_conversation_async("c1")
    assert result is None
    assert route.called


@respx.mock
async def test_delete_conversation_async_404_raises():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    respx.delete(re.compile(r".*/conversations/missing$")).mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(httpx.HTTPStatusError):
        await api.delete_conversation_async("missing")


# ─── list_conversations ──────────────────────────────────────


@respx.mock
async def test_list_conversations_async_parses_data_array():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    respx.get(re.compile(r".*/conversations(\?.*)?$")).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "conversations": [
                        {"conversationId": "c1", "title": "t1"},
                        {"conversationId": "c2", "title": "t2"},
                    ]
                },
                "success": True,
            },
        )
    )
    result = await api.list_conversations_async()
    assert [c["conversationId"] for c in result] == ["c1", "c2"]


@respx.mock
async def test_list_conversations_async_metadata_query_encoded():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"data": {"conversations": []}})

    respx.get(re.compile(r".*/conversations.*")).mock(side_effect=_responder)
    await api.list_conversations_async(metadata={"agentRuntimeName": "demo"})
    assert "metadata=" in captured["url"]
    # metadata is json-encoded then URL-encoded
    assert "agentRuntimeName" in captured["url"]


@respx.mock
async def test_list_conversations_async_without_metadata_no_query():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"data": {"conversations": []}})

    respx.get(re.compile(r".*/conversations.*")).mock(side_effect=_responder)
    await api.list_conversations_async()
    assert "metadata" not in captured["url"]


@respx.mock
async def test_list_conversations_async_request_signed():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    captured = {}

    def _responder(request):
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"data": {"conversations": []}})

    respx.get(re.compile(r".*/conversations.*")).mock(side_effect=_responder)
    await api.list_conversations_async()
    assert any(
        k.lower() == "agentrun-authorization" for k in captured["headers"]
    )


@respx.mock
async def test_list_conversations_async_missing_data_returns_empty():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    respx.get(re.compile(r".*/conversations.*")).mock(
        return_value=httpx.Response(200, json={"success": True})
    )
    assert await api.list_conversations_async() == []


@respx.mock
async def test_list_conversations_async_data_not_dict_returns_empty():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    respx.get(re.compile(r".*/conversations.*")).mock(
        return_value=httpx.Response(200, json={"data": "unexpected"})
    )
    assert await api.list_conversations_async() == []


@respx.mock
async def test_list_conversations_async_conversations_not_list_returns_empty():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    respx.get(re.compile(r".*/conversations.*")).mock(
        return_value=httpx.Response(
            200, json={"data": {"conversations": "bad"}}
        )
    )
    assert await api.list_conversations_async() == []


@respx.mock
async def test_list_conversations_async_filters_non_dict_items():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    respx.get(re.compile(r".*/conversations.*")).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "conversations": [
                        {"conversationId": "c1"},
                        "invalid",
                        None,
                        {"conversationId": "c2"},
                    ]
                }
            },
        )
    )
    result = await api.list_conversations_async()
    assert [c["conversationId"] for c in result] == ["c1", "c2"]


@respx.mock
async def test_list_conversations_async_payload_not_dict_returns_empty():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    respx.get(re.compile(r".*/conversations.*")).mock(
        return_value=httpx.Response(200, json=[1, 2, 3])
    )
    assert await api.list_conversations_async() == []


@respx.mock
async def test_list_conversations_async_empty_body_returns_empty():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    respx.get(re.compile(r".*/conversations.*")).mock(
        return_value=httpx.Response(200, content=b"")
    )
    assert await api.list_conversations_async() == []


@respx.mock
async def test_list_conversations_async_5xx_raises():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    respx.get(re.compile(r".*/conversations.*")).mock(
        return_value=httpx.Response(500, text="boom")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await api.list_conversations_async()


# ─── sync stubs are NotImplementedError ──────────────────────


def test_sync_methods_not_implemented():
    cfg = _auth_cfg()
    api = SuperAgentDataAPI("demo", config=cfg)
    for fn in (
        lambda: api.invoke([]),
        lambda: api.stream("url"),
        lambda: api.list_conversations(),
        lambda: api.get_conversation("c"),
        lambda: api.delete_conversation("c"),
    ):
        with pytest.raises(NotImplementedError):
            fn()
