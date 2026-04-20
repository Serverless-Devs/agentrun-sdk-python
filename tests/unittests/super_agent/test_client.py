"""Unit tests for ``agentrun.super_agent.client.SuperAgentClient``."""

import asyncio
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.super_agent.api.control import (
    SUPER_AGENT_PROTOCOL_TYPE,
    SUPER_AGENT_TAG,
)
from agentrun.super_agent.client import SuperAgentClient
from agentrun.utils.config import Config


def _client_config() -> Config:
    return Config(
        access_key_id="AK",
        access_key_secret="SK",
        account_id="123",
        region_id="cn-hangzhou",
    )


class _DaraResult:
    """Fake Dara-level AgentRuntime returned from the control API."""

    def __init__(self, map_dict: dict):
        self._map = map_dict

    def to_map(self):
        return self._map


def _rt_to_dara_result(rt: SimpleNamespace) -> _DaraResult:
    """Turn the internal ``_fake_rt`` into a Dara-style object with ``to_map``."""
    return _DaraResult({
        "agentRuntimeName": rt.agent_runtime_name,
        "agentRuntimeId": rt.agent_runtime_id,
        "agentRuntimeArn": rt.agent_runtime_arn,
        "status": rt.status,
        "createdAt": rt.created_at,
        "lastUpdatedAt": rt.last_updated_at,
        "description": rt.description,
        "protocolConfiguration": rt.protocol_configuration,
    })


def _fake_rt(
    *,
    name: str = "n",
    prompt: str = "old",
    tools=None,
    description=None,
    protocol_type: str = SUPER_AGENT_PROTOCOL_TYPE,
) -> SimpleNamespace:
    """Build a minimal AgentRuntime-like object for ``from_agent_runtime``."""
    cfg_dict = {
        "path": "/invoke",
        "headers": {},
        "body": {
            "forwardedProps": {
                "prompt": prompt,
                "agents": [],
                "tools": tools if tools is not None else [],
                "skills": [],
                "sandboxes": [],
                "workspaces": [],
                "modelServiceName": None,
                "modelName": None,
                "metadata": {"agentRuntimeName": name},
            }
        },
    }
    pc = {
        "type": protocol_type,
        "protocolSettings": [{
            "type": protocol_type,
            "name": name,
            "path": "/invoke",
            "config": json.dumps(cfg_dict),
        }],
        "externalEndpoint": "https://x.com/super-agents/__SUPER_AGENT__",
    }
    return SimpleNamespace(
        agent_runtime_name=name,
        agent_runtime_id="rid",
        agent_runtime_arn="arn",
        status="READY",
        created_at="t1",
        last_updated_at="t2",
        description=description,
        protocol_configuration=pc,
    )


# ─── create ──────────────────────────────────────────────────


async def test_create_async_calls_runtime_with_correct_input():
    captured_input = {}

    async def _create_async(dara_input, config=None):
        captured_input["dara"] = dara_input
        return _rt_to_dara_result(_fake_rt(name="alpha", prompt="new"))

    ctrl = MagicMock()
    ctrl.create_agent_runtime_async = _create_async
    with patch(
        "agentrun.super_agent.client.AgentRuntimeControlAPI",
        return_value=ctrl,
    ):
        client = SuperAgentClient(config=_client_config())
        agent = await client.create_async(name="alpha", prompt="new")

    dara = captured_input["dara"]
    # Dara-level model uses snake_case attributes
    assert dara.agent_runtime_name == "alpha"
    # 注: alibabacloud-agentrun20250910 的 Dara CreateAgentRuntimeInput 目前
    # 不包含 ``tags`` 字段, pydantic → Dara roundtrip 会丢弃. 校验 pydantic 侧
    # 的 rt_input 是否含 tags 在 test_control.py::test_to_create_input_tags_fixed
    # 已覆盖.
    pc = dara.protocol_configuration
    # externalEndpoint preserved via the additive Dara monkey-patch
    assert pc.external_endpoint.endswith("/super-agents/__SUPER_AGENT__")
    first = pc.protocol_settings[0]
    assert first.type == SUPER_AGENT_PROTOCOL_TYPE
    cfg_json = json.loads(first.config)
    assert cfg_json["body"]["forwardedProps"]["prompt"] == "new"
    assert agent.name == "alpha"


async def test_create_async_returns_super_agent_with_client_handle():
    ctrl = MagicMock()
    ctrl.create_agent_runtime_async = AsyncMock(
        return_value=_rt_to_dara_result(_fake_rt(name="alpha"))
    )
    with patch(
        "agentrun.super_agent.client.AgentRuntimeControlAPI",
        return_value=ctrl,
    ):
        client = SuperAgentClient(config=_client_config())
        agent = await client.create_async(name="alpha")
    assert agent._client is client


# ─── get ────────────────────────────────────────────────────


async def test_get_async_normal():
    rt_client = MagicMock()
    rt_client.list_async = AsyncMock(return_value=[_fake_rt(name="alpha")])
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        agent = await client.get_async("alpha")
    assert agent.name == "alpha"
    assert agent._client is client


async def test_get_async_not_super_agent_raises():
    rt_client = MagicMock()
    rt_client.list_async = AsyncMock(
        return_value=[_fake_rt(name="alpha", protocol_type="HTTP")]
    )
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        with pytest.raises(ValueError) as exc:
            await client.get_async("alpha")
    assert "is not a super agent" in str(exc.value)


# ─── update (read-merge-write) ───────────────────────────────


async def test_update_async_partial_modify_prompt_only():
    existing = _fake_rt(name="x", prompt="old", tools=["t1"])
    rt_client = MagicMock()
    rt_client.list_async = AsyncMock(return_value=[existing])
    rt_client.get_async = AsyncMock(return_value=existing)
    ctrl = MagicMock()
    captured = {}

    async def _update(agent_id, dara_input, config=None):
        captured["dara"] = dara_input
        return _rt_to_dara_result(
            _fake_rt(name="x", prompt="new", tools=["t1"])
        )

    ctrl.update_agent_runtime_async = _update
    with (
        patch(
            "agentrun.super_agent.client.AgentRuntimeClient",
            return_value=rt_client,
        ),
        patch(
            "agentrun.super_agent.client.AgentRuntimeControlAPI",
            return_value=ctrl,
        ),
    ):
        client = SuperAgentClient(config=_client_config())
        await client.update_async("x", prompt="new")
    cfg_json = json.loads(
        captured["dara"].protocol_configuration.protocol_settings[0].config
    )
    forwarded = cfg_json["body"]["forwardedProps"]
    assert forwarded["prompt"] == "new"
    assert forwarded["tools"] == ["t1"]


async def test_update_async_explicit_none_clears_field():
    existing = _fake_rt(name="x", prompt="old")
    rt_client = MagicMock()
    rt_client.list_async = AsyncMock(return_value=[existing])
    rt_client.get_async = AsyncMock(return_value=existing)
    ctrl = MagicMock()
    captured = {}

    async def _update(agent_id, dara_input, config=None):
        captured["dara"] = dara_input
        return _rt_to_dara_result(_fake_rt(name="x"))

    ctrl.update_agent_runtime_async = _update
    with (
        patch(
            "agentrun.super_agent.client.AgentRuntimeClient",
            return_value=rt_client,
        ),
        patch(
            "agentrun.super_agent.client.AgentRuntimeControlAPI",
            return_value=ctrl,
        ),
    ):
        client = SuperAgentClient(config=_client_config())
        await client.update_async("x", prompt=None)
    cfg_json = json.loads(
        captured["dara"].protocol_configuration.protocol_settings[0].config
    )
    assert cfg_json["body"]["forwardedProps"]["prompt"] is None


async def test_update_async_multiple_fields():
    existing = _fake_rt(name="x", prompt="old")
    rt_client = MagicMock()
    rt_client.list_async = AsyncMock(return_value=[existing])
    rt_client.get_async = AsyncMock(return_value=existing)
    ctrl = MagicMock()
    captured = {}

    async def _update(agent_id, dara_input, config=None):
        captured["dara"] = dara_input
        return _rt_to_dara_result(_fake_rt(name="x"))

    ctrl.update_agent_runtime_async = _update
    with (
        patch(
            "agentrun.super_agent.client.AgentRuntimeClient",
            return_value=rt_client,
        ),
        patch(
            "agentrun.super_agent.client.AgentRuntimeControlAPI",
            return_value=ctrl,
        ),
    ):
        client = SuperAgentClient(config=_client_config())
        await client.update_async(
            "x", prompt="p", tools=["a", "b"], description="d"
        )
    cfg_json = json.loads(
        captured["dara"].protocol_configuration.protocol_settings[0].config
    )
    forwarded = cfg_json["body"]["forwardedProps"]
    assert forwarded["prompt"] == "p"
    assert forwarded["tools"] == ["a", "b"]
    assert captured["dara"].description == "d"


async def test_update_async_target_not_super_agent_raises():
    rt_client = MagicMock()
    rt_client.list_async = AsyncMock(
        return_value=[_fake_rt(name="x", protocol_type="HTTP")]
    )
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        with pytest.raises(ValueError):
            await client.update_async("x", prompt="p")


# ─── delete ─────────────────────────────────────────────────


async def test_delete_async_calls_runtime():
    rt_client = MagicMock()
    rt_client.list_async = AsyncMock(return_value=[_fake_rt(name="alpha")])
    rt_client.delete_async = AsyncMock(return_value=None)
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        result = await client.delete_async("alpha")
    assert result is None
    rt_client.delete_async.assert_awaited_once()
    called_with = rt_client.delete_async.await_args.args
    # list_async returns rt with agent_runtime_id="rid"; delete_async 用 id 调用
    assert called_with[0] == "rid"


# ─── list ───────────────────────────────────────────────────


async def test_list_async_default_pagination():
    rt_client = MagicMock()
    captured = {}

    async def _list(inp=None, config=None):
        captured["inp"] = inp
        return []

    rt_client.list_async = _list
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        await client.list_async()
    assert captured["inp"].page_number == 1
    assert captured["inp"].page_size == 20
    assert captured["inp"].tags == SUPER_AGENT_TAG


async def test_list_async_custom_pagination():
    rt_client = MagicMock()
    captured = {}

    async def _list(inp=None, config=None):
        captured["inp"] = inp
        return []

    rt_client.list_async = _list
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        await client.list_async(page_number=2, page_size=50)
    assert captured["inp"].page_number == 2
    assert captured["inp"].page_size == 50


async def test_list_async_rejects_tags_kwarg():
    client = SuperAgentClient()
    with pytest.raises(TypeError):
        await client.list_async(tags=["x"])  # type: ignore[call-arg]


async def test_list_async_filters_non_super_agent():
    items = [
        _fake_rt(name="a"),
        _fake_rt(name="b", protocol_type="HTTP"),
        _fake_rt(name="c"),
    ]
    rt_client = MagicMock()
    rt_client.list_async = AsyncMock(return_value=items)
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        result = await client.list_async()
    assert [a.name for a in result] == ["a", "c"]


async def test_list_all_async_auto_pagination():
    # page_size=50 is the default for list_all_async; craft pages accordingly
    page1 = [_fake_rt(name=f"a{i}") for i in range(50)]
    page2 = [_fake_rt(name=f"a{i}") for i in range(50, 85)]
    pages = [page1, page2, []]
    rt_client = MagicMock()
    rt_client.list_async = AsyncMock(side_effect=pages)
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        result = await client.list_all_async()
    names = [a.name for a in result]
    assert len(names) == 85
    assert len(set(names)) == 85


# ─── sync mirrors async ─────────────────────────────────────


def test_sync_methods_exist_and_mirror_async():
    rt_client = MagicMock()
    rt_client.get = MagicMock(return_value=_fake_rt(name="alpha"))
    rt_client.delete = MagicMock(return_value=None)
    rt_client.list = MagicMock(return_value=[_fake_rt(name="alpha")])
    ctrl = MagicMock()
    ctrl.create_agent_runtime = MagicMock(
        return_value=_rt_to_dara_result(_fake_rt(name="alpha"))
    )
    ctrl.update_agent_runtime = MagicMock(
        return_value=_rt_to_dara_result(_fake_rt(name="alpha"))
    )
    with (
        patch(
            "agentrun.super_agent.client.AgentRuntimeClient",
            return_value=rt_client,
        ),
        patch(
            "agentrun.super_agent.client.AgentRuntimeControlAPI",
            return_value=ctrl,
        ),
    ):
        client = SuperAgentClient(config=_client_config())
        agent = client.create(name="alpha")
        assert agent.name == "alpha"
        assert client.get("alpha").name == "alpha"
        assert client.update("alpha", prompt="p").name == "alpha"
        assert client.delete("alpha") is None
        assert len(client.list()) == 1
        # list_all hits list() once, then empty page
        rt_client.list = MagicMock(side_effect=[[_fake_rt(name="a")], []])
        result = client.list_all()
        assert len(result) == 1


# ─── not-found / not-super error paths (async + sync) ──────


def _make_client_with_list(list_items, *, sync=False) -> SuperAgentClient:
    """Helper: build a SuperAgentClient where _rt.list(_async) returns ``list_items``."""
    rt_client = MagicMock()
    if sync:
        rt_client.list = MagicMock(return_value=list_items)
    else:
        rt_client.list_async = AsyncMock(return_value=list_items)
    patcher = patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    )
    patcher.start()
    client = SuperAgentClient(config=_client_config())
    client._patcher = patcher  # type: ignore[attr-defined]
    client._rt_client = rt_client  # type: ignore[attr-defined]
    return client


async def test_get_async_not_found_raises():
    client = _make_client_with_list([])
    try:
        with pytest.raises(ValueError, match="not found"):
            await client.get_async("missing")
    finally:
        client._patcher.stop()  # type: ignore[attr-defined]


def test_get_sync_not_found_raises():
    client = _make_client_with_list([], sync=True)
    try:
        with pytest.raises(ValueError, match="not found"):
            client.get("missing")
    finally:
        client._patcher.stop()  # type: ignore[attr-defined]


def test_get_sync_not_super_agent_raises():
    client = _make_client_with_list(
        [_fake_rt(name="x", protocol_type="HTTP")], sync=True
    )
    try:
        with pytest.raises(ValueError, match="is not a super agent"):
            client.get("x")
    finally:
        client._patcher.stop()  # type: ignore[attr-defined]


async def test_update_async_not_found_raises():
    client = _make_client_with_list([])
    try:
        with pytest.raises(ValueError, match="not found"):
            await client.update_async("missing", prompt="p")
    finally:
        client._patcher.stop()  # type: ignore[attr-defined]


def test_update_sync_not_found_raises():
    client = _make_client_with_list([], sync=True)
    try:
        with pytest.raises(ValueError, match="not found"):
            client.update("missing", prompt="p")
    finally:
        client._patcher.stop()  # type: ignore[attr-defined]


def test_update_sync_not_super_agent_raises():
    client = _make_client_with_list(
        [_fake_rt(name="x", protocol_type="HTTP")], sync=True
    )
    try:
        with pytest.raises(ValueError, match="is not a super agent"):
            client.update("x", prompt="p")
    finally:
        client._patcher.stop()  # type: ignore[attr-defined]


async def test_delete_async_not_found_raises():
    client = _make_client_with_list([])
    try:
        with pytest.raises(ValueError, match="not found"):
            await client.delete_async("missing")
    finally:
        client._patcher.stop()  # type: ignore[attr-defined]


def test_delete_sync_not_found_raises():
    client = _make_client_with_list([], sync=True)
    try:
        with pytest.raises(ValueError, match="not found"):
            client.delete("missing")
    finally:
        client._patcher.stop()  # type: ignore[attr-defined]


# ─── _find_rt_by_name_* 多页分页 ───────────────────────────────


async def test_find_rt_by_name_async_paginates_until_match():
    """page1 全是非匹配 item (满 50), page2 才有 target → 需要翻页."""
    page1 = [_fake_rt(name=f"other{i}") for i in range(50)]
    page2 = [_fake_rt(name="target")]
    rt_client = MagicMock()
    rt_client.list_async = AsyncMock(side_effect=[page1, page2])
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        agent = await client.get_async("target")
    assert agent.name == "target"
    assert rt_client.list_async.await_count == 2


def test_find_rt_by_name_sync_paginates_until_match():
    page1 = [_fake_rt(name=f"other{i}") for i in range(50)]
    page2 = [_fake_rt(name="target")]
    rt_client = MagicMock()
    rt_client.list = MagicMock(side_effect=[page1, page2])
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        agent = client.get("target")
    assert agent.name == "target"
    assert rt_client.list.call_count == 2


# ─── _wait_final / _raise_if_failed ──────────────────────────


def test_raise_if_failed_raises_on_failed_status():
    from agentrun.super_agent.client import _raise_if_failed

    rt = SimpleNamespace(
        status="CREATE_FAILED",
        status_reason="disk full",
        agent_runtime_name="x",
    )
    with pytest.raises(RuntimeError) as exc:
        _raise_if_failed(rt, action="create")
    assert "disk full" in str(exc.value)
    assert "CREATE_FAILED" in str(exc.value)


def test_raise_if_failed_noop_on_ready():
    from agentrun.super_agent.client import _raise_if_failed

    rt = SimpleNamespace(status="READY")
    _raise_if_failed(rt, action="update")  # no raise


async def test_wait_final_async_timeout():
    rt_pending = _fake_rt(name="x")
    rt_pending.status = "CREATING"
    rt_client = MagicMock()
    rt_client.get_async = AsyncMock(return_value=rt_pending)
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        with pytest.raises(TimeoutError):
            await client._wait_final_async(
                "rid", interval_seconds=0, timeout_seconds=-1
            )


def test_wait_final_sync_timeout():
    rt_pending = _fake_rt(name="x")
    rt_pending.status = "CREATING"
    rt_client = MagicMock()
    rt_client.get = MagicMock(return_value=rt_pending)
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        with pytest.raises(TimeoutError):
            client._wait_final("rid", interval_seconds=0, timeout_seconds=-1)


async def test_wait_final_async_retries_then_ready():
    """第一次 get 返回 CREATING → await asyncio.sleep → 第二次 READY."""
    pending = _fake_rt(name="x")
    pending.status = "CREATING"
    ready = _fake_rt(name="x")  # status=READY by default
    rt_client = MagicMock()
    rt_client.get_async = AsyncMock(side_effect=[pending, ready])
    with (
        patch(
            "agentrun.super_agent.client.AgentRuntimeClient",
            return_value=rt_client,
        ),
        patch("agentrun.super_agent.client.asyncio.sleep", AsyncMock()),
    ):
        client = SuperAgentClient(config=_client_config())
        result = await client._wait_final_async(
            "rid", interval_seconds=0, timeout_seconds=60
        )
    assert getattr(result, "status", None) == "READY"
    assert rt_client.get_async.await_count == 2


def test_wait_final_sync_retries_then_ready():
    pending = _fake_rt(name="x")
    pending.status = "CREATING"
    ready = _fake_rt(name="x")
    rt_client = MagicMock()
    rt_client.get = MagicMock(side_effect=[pending, ready])
    with (
        patch(
            "agentrun.super_agent.client.AgentRuntimeClient",
            return_value=rt_client,
        ),
        patch("agentrun.super_agent.client.time.sleep", MagicMock()),
    ):
        client = SuperAgentClient(config=_client_config())
        result = client._wait_final(
            "rid", interval_seconds=0, timeout_seconds=60
        )
    assert getattr(result, "status", None) == "READY"
    assert rt_client.get.call_count == 2


# ─── create: 非 final 状态触发 _wait_final ────────────────────


async def test_create_async_non_final_status_triggers_wait():
    creating = _fake_rt(name="alpha")
    creating.status = "CREATING"
    ready = _fake_rt(name="alpha")  # READY
    rt_client = MagicMock()
    rt_client.get_async = AsyncMock(return_value=ready)
    ctrl = MagicMock()
    ctrl.create_agent_runtime_async = AsyncMock(
        return_value=_rt_to_dara_result(creating)
    )
    with (
        patch(
            "agentrun.super_agent.client.AgentRuntimeClient",
            return_value=rt_client,
        ),
        patch(
            "agentrun.super_agent.client.AgentRuntimeControlAPI",
            return_value=ctrl,
        ),
    ):
        client = SuperAgentClient(config=_client_config())
        agent = await client.create_async(name="alpha")
    assert agent.name == "alpha"
    rt_client.get_async.assert_awaited()  # _wait_final_async 确实调了 get


# ─── sync list / list_all 未覆盖分支 ─────────────────────────


def test_list_sync_filters_non_super_agent():
    items = [
        _fake_rt(name="a"),
        _fake_rt(name="b", protocol_type="HTTP"),
        _fake_rt(name="c"),
    ]
    rt_client = MagicMock()
    rt_client.list = MagicMock(return_value=items)
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        result = client.list()
    assert [a.name for a in result] == ["a", "c"]


def test_list_all_sync_multi_page():
    page1 = [_fake_rt(name=f"a{i}") for i in range(50)]
    page2 = [_fake_rt(name=f"a{i}") for i in range(50, 85)]
    pages = [page1, page2]
    rt_client = MagicMock()
    rt_client.list = MagicMock(side_effect=pages)
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        result = client.list_all()
    assert len(result) == 85


def test_list_all_async_empty_first_page_breaks():
    """list_async 首页直接空, list_all 立刻 break."""

    async def _list(*args, **kwargs):
        return []

    rt_client = MagicMock()
    rt_client.list_async = _list
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        result = asyncio.run(client.list_all_async())
    assert result == []


def test_list_all_sync_empty_first_page_breaks():
    rt_client = MagicMock()
    rt_client.list = MagicMock(return_value=[])
    with patch(
        "agentrun.super_agent.client.AgentRuntimeClient",
        return_value=rt_client,
    ):
        client = SuperAgentClient(config=_client_config())
        result = client.list_all()
    assert result == []


def test_no_agent_runtime_in_public_signatures():
    """No public SuperAgentClient method exposes AgentRuntime-related types."""
    public_methods = [m for m in dir(SuperAgentClient) if not m.startswith("_")]
    for name in public_methods:
        attr = getattr(SuperAgentClient, name)
        if not callable(attr):
            continue
        sig = inspect.signature(attr)
        all_annotations = [p.annotation for p in sig.parameters.values()]
        all_annotations.append(sig.return_annotation)
        rendered = " ".join(str(a) for a in all_annotations)
        assert (
            "AgentRuntime" not in rendered
        ), f"{name} exposes AgentRuntime in its signature: {rendered}"
