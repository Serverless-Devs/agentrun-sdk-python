"""Unit tests for ``agentrun.super_agent.api.control``."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agentrun.super_agent.api.control import (
    _add_ram_prefix_to_host,
    API_VERSION,
    build_super_agent_endpoint,
    ensure_super_agent_patches_applied,
    EXTERNAL_TAG,
    from_agent_runtime,
    is_super_agent,
    parse_super_agent_config,
    SUPER_AGENT_PROTOCOL_TYPE,
    SUPER_AGENT_RESOURCE_PATH,
    SUPER_AGENT_TAG,
    to_create_input,
    to_update_input,
)
from agentrun.super_agent.api.data import SuperAgentDataAPI
from agentrun.utils.config import Config

# 本文件部分测试 (list request tags 补丁) 依赖 Dara SDK 已被打过补丁,
# 显式在模块加载时触发补丁 (幂等, 与 SuperAgentClient.__init__ 内触发点一致)。
ensure_super_agent_patches_applied()

# ─── build_super_agent_endpoint ────────────────────────────────


def test_build_super_agent_endpoint_production():
    cfg = Config(account_id="123", region_id="cn-hangzhou")
    ep = build_super_agent_endpoint(cfg)
    assert (
        ep
        == "https://123-ram.agentrun-data.cn-hangzhou.aliyuncs.com/super-agents/__SUPER_AGENT__"
    )


def test_build_super_agent_endpoint_pre_environment():
    cfg = Config(
        data_endpoint=(
            "http://1431999136518149.funagent-data-pre.cn-hangzhou.aliyuncs.com"
        )
    )
    ep = build_super_agent_endpoint(cfg)
    assert (
        ep
        == "http://1431999136518149-ram.funagent-data-pre.cn-hangzhou.aliyuncs.com/super-agents/__SUPER_AGENT__"
    )


def test_build_super_agent_endpoint_custom_gateway():
    cfg = Config(data_endpoint="https://my-gateway.example.com")
    ep = build_super_agent_endpoint(cfg)
    assert ep == "https://my-gateway.example.com/super-agents/__SUPER_AGENT__"


def test_build_super_agent_endpoint_unknown_first_segment():
    # `agentrun-data` in the first segment → no `-ram` rewrite
    cfg = Config(data_endpoint="https://agentrun-data.example.com")
    ep = build_super_agent_endpoint(cfg)
    assert (
        ep == "https://agentrun-data.example.com/super-agents/__SUPER_AGENT__"
    )


# ─── _add_ram_prefix_to_host ──────────────────────────────────


def test_add_ram_prefix_to_host_no_netloc():
    assert _add_ram_prefix_to_host("") == ""
    assert _add_ram_prefix_to_host("/path/only") == "/path/only"


def test_add_ram_prefix_to_host_single_segment_host():
    # Host has a single segment → no rewrite
    assert (
        _add_ram_prefix_to_host("https://localhost:8080")
        == "https://localhost:8080"
    )


def test_add_ram_prefix_to_host_unknown_domain():
    assert (
        _add_ram_prefix_to_host("https://foo.example.com")
        == "https://foo.example.com"
    )


# ─── SuperAgentDataAPI URL 含版本号 ──────────────────────────
def test_build_data_url_via_with_path_includes_version():
    cfg = Config(account_id="123", region_id="cn-hangzhou")
    api = SuperAgentDataAPI("demo", config=cfg)
    url = api.with_path("invoke")
    assert url.endswith(
        f"/{API_VERSION}/super-agents/{SUPER_AGENT_RESOURCE_PATH}/invoke"
    )


# ─── to_create_input ──────────────────────────────────────────


def test_to_create_input_minimal():
    cfg = Config(account_id="123", region_id="cn-hangzhou")
    inp = to_create_input("alpha", cfg=cfg)
    assert inp.agent_runtime_name == "alpha"
    assert inp.tags == [SUPER_AGENT_TAG]
    pc = inp.protocol_configuration
    assert pc.type == SUPER_AGENT_PROTOCOL_TYPE
    assert pc.external_endpoint.endswith("/super-agents/__SUPER_AGENT__")
    settings = pc.protocol_settings
    assert len(settings) == 1
    cfg_dict = json.loads(settings[0]["config"])
    assert cfg_dict["path"] == "/invoke"
    assert cfg_dict["agents"] == []
    assert cfg_dict["metadata"] == {"agentRuntimeName": "alpha"}


def test_to_create_input_full():
    cfg = Config(account_id="123", region_id="cn-hangzhou")
    inp = to_create_input(
        "bravo",
        prompt="hello",
        agents=["a1"],
        tools=["t1", "t2"],
        skills=["s1"],
        sandboxes=["sb1"],
        workspaces=["ws1"],
        model_service_name="foo",
        model_name="bar",
        cfg=cfg,
    )
    pc_dict = inp.model_dump()["protocolConfiguration"]
    settings_cfg = json.loads(pc_dict["protocolSettings"][0]["config"])
    assert settings_cfg["prompt"] == "hello"
    assert settings_cfg["agents"] == ["a1"]
    assert settings_cfg["tools"] == ["t1", "t2"]
    assert settings_cfg["skills"] == ["s1"]
    assert settings_cfg["sandboxes"] == ["sb1"]
    assert settings_cfg["workspaces"] == ["ws1"]
    assert settings_cfg["modelServiceName"] == "foo"
    assert settings_cfg["modelName"] == "bar"


def test_to_create_input_tags_fixed():
    cfg = Config(account_id="123", region_id="cn-hangzhou")
    inp = to_create_input("c", cfg=cfg)
    assert inp.tags == [SUPER_AGENT_TAG]


def test_to_create_input_metadata_only_agent_runtime_name():
    cfg = Config(account_id="123", region_id="cn-hangzhou")
    inp = to_create_input("d", cfg=cfg)
    settings_cfg = json.loads(
        inp.protocol_configuration.protocol_settings[0]["config"]
    )
    assert settings_cfg["metadata"] == {"agentRuntimeName": "d"}


def test_to_create_input_uses_pre_environment_endpoint():
    cfg = Config(
        data_endpoint=(
            "http://1431999136518149.funagent-data-pre.cn-hangzhou.aliyuncs.com"
        )
    )
    inp = to_create_input("pre-agent", cfg=cfg)
    ep = inp.protocol_configuration.external_endpoint
    assert "funagent-data-pre" in ep
    assert "-ram" in ep


# ─── is_super_agent / parse_super_agent_config ─────────────────


def _make_rt(**kwargs):
    """Minimal fake AgentRuntime-like object."""
    defaults = {
        "agent_runtime_name": "n",
        "agent_runtime_id": "rid",
        "agent_runtime_arn": "arn",
        "status": "READY",
        "created_at": "2026-01-01",
        "last_updated_at": "2026-01-02",
        "description": None,
        "protocol_configuration": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_from_agent_runtime():
    config_json = json.dumps({
        "prompt": "hi",
        "agents": ["a"],
        "tools": ["t"],
        "skills": [],
        "sandboxes": [],
        "workspaces": [],
        "modelServiceName": "svc",
        "modelName": "mod",
        "metadata": {"agentRuntimeName": "foo"},
    })
    pc = {
        "type": SUPER_AGENT_PROTOCOL_TYPE,
        "protocolSettings": [{
            "type": SUPER_AGENT_PROTOCOL_TYPE,
            "config": config_json,
            "name": "foo",
            "path": "/invoke",
        }],
        "externalEndpoint": "https://x.com/super-agents/__SUPER_AGENT__",
    }
    rt = _make_rt(agent_runtime_name="foo", protocol_configuration=pc)
    agent = from_agent_runtime(rt)
    assert agent.name == "foo"
    assert agent.prompt == "hi"
    assert agent.agents == ["a"]
    assert agent.tools == ["t"]
    assert agent.model_service_name == "svc"
    assert agent.model_name == "mod"
    assert (
        agent.external_endpoint == "https://x.com/super-agents/__SUPER_AGENT__"
    )


def test_is_super_agent_true():
    pc = {
        "type": SUPER_AGENT_PROTOCOL_TYPE,
        "protocolSettings": [{"type": SUPER_AGENT_PROTOCOL_TYPE}],
    }
    assert is_super_agent(_make_rt(protocol_configuration=pc))


def test_is_super_agent_false():
    for type_name in ("HTTP", "MCP", "OTHER"):
        pc = {"type": type_name, "protocolSettings": [{"type": type_name}]}
        assert not is_super_agent(_make_rt(protocol_configuration=pc))
    # No protocol_configuration
    assert not is_super_agent(_make_rt(protocol_configuration=None))


def test_parse_super_agent_config_invalid_json_returns_empty():
    pc = {
        "protocolSettings": [
            {"type": SUPER_AGENT_PROTOCOL_TYPE, "config": "not-json"}
        ]
    }
    assert parse_super_agent_config(_make_rt(protocol_configuration=pc)) == {}


def test_parse_super_agent_config_missing_config_returns_empty():
    pc = {"protocolSettings": [{"type": SUPER_AGENT_PROTOCOL_TYPE}]}
    assert parse_super_agent_config(_make_rt(protocol_configuration=pc)) == {}


# ─── to_update_input ──────────────────────────────────────────


def test_to_update_input_full_protocol_replace():
    cfg = Config(account_id="123", region_id="cn-hangzhou")
    inp = to_update_input(
        "alpha",
        {
            "description": "new",
            "prompt": "p",
            "agents": [],
            "tools": ["t"],
            "skills": [],
            "sandboxes": [],
            "workspaces": [],
            "model_service_name": None,
            "model_name": None,
        },
        cfg=cfg,
    )
    assert inp.description == "new"
    settings = inp.protocol_configuration.protocol_settings
    assert len(settings) == 1
    assert (
        json.loads(settings[0]["config"])["metadata"]["agentRuntimeName"]
        == "alpha"
    )


# ─── Dara ListAgentRuntimesRequest tags 补丁 ──────────────────
# 补丁已在模块顶部通过 ensure_super_agent_patches_applied() 显式触发。


def test_list_request_from_map_preserves_tags():
    from alibabacloud_agentrun20250910.models import ListAgentRuntimesRequest

    req = ListAgentRuntimesRequest().from_map({
        "tags": SUPER_AGENT_TAG,
        "pageNumber": 1,
        "pageSize": 20,
    })
    assert getattr(req, "tags", None) == SUPER_AGENT_TAG


def test_list_request_to_map_preserves_tags():
    from alibabacloud_agentrun20250910.models import ListAgentRuntimesRequest

    req = ListAgentRuntimesRequest()
    req.tags = SUPER_AGENT_TAG
    assert req.to_map().get("tags") == SUPER_AGENT_TAG


def _invoke_list_patch(tags_value):
    """调用打过补丁的 ``list_agent_runtimes_with_options``, 捕获 call_api 的 query."""
    from alibabacloud_agentrun20250910.client import Client as _DaraClient
    from alibabacloud_agentrun20250910.models import ListAgentRuntimesRequest
    from darabonba.runtime import RuntimeOptions

    captured = {}

    def _fake_call_api(self, params, req, rt):
        captured["query"] = dict(req.query) if req.query else {}
        raise RuntimeError("_stop_after_query_capture_")

    client = _DaraClient.__new__(_DaraClient)
    client._endpoint = "x"
    # 绑定实例级 call_api (优先于类方法)
    client.call_api = _fake_call_api.__get__(client, _DaraClient)

    req = ListAgentRuntimesRequest(page_number=1, page_size=20)
    req.tags = tags_value
    with pytest.raises(RuntimeError, match="_stop_after_query_capture_"):
        client.list_agent_runtimes_with_options(req, {}, RuntimeOptions())
    return captured["query"]


def test_list_with_options_injects_tags_str():
    query = _invoke_list_patch(SUPER_AGENT_TAG)
    assert query.get("tags") == SUPER_AGENT_TAG
    assert query.get("pageNumber") == "1"


def test_list_with_options_injects_tags_list_comma_join():
    query = _invoke_list_patch([EXTERNAL_TAG, SUPER_AGENT_TAG])
    assert query.get("tags") == f"{EXTERNAL_TAG},{SUPER_AGENT_TAG}"


def test_list_with_options_no_tags_no_injection():
    from alibabacloud_agentrun20250910.client import Client as _DaraClient
    from alibabacloud_agentrun20250910.models import ListAgentRuntimesRequest
    from darabonba.runtime import RuntimeOptions

    captured = {}

    def _fake_call_api(self, params, req, rt):
        captured["query"] = dict(req.query) if req.query else {}
        raise RuntimeError("_stop_")

    client = _DaraClient.__new__(_DaraClient)
    client._endpoint = "x"
    client.call_api = _fake_call_api.__get__(client, _DaraClient)

    req = ListAgentRuntimesRequest(page_number=1, page_size=20)
    with pytest.raises(RuntimeError, match="_stop_"):
        client.list_agent_runtimes_with_options(req, {}, RuntimeOptions())
    assert "tags" not in captured["query"]


@pytest.mark.asyncio
async def test_list_with_options_async_injects_tags():
    from alibabacloud_agentrun20250910.client import Client as _DaraClient
    from alibabacloud_agentrun20250910.models import ListAgentRuntimesRequest
    from darabonba.runtime import RuntimeOptions

    captured = {}

    async def _fake_call_api_async(self, params, req, rt):
        captured["query"] = dict(req.query) if req.query else {}
        raise RuntimeError("_stop_")

    client = _DaraClient.__new__(_DaraClient)
    client._endpoint = "x"
    client.call_api_async = _fake_call_api_async.__get__(client, _DaraClient)

    req = ListAgentRuntimesRequest(page_number=1, page_size=20)
    req.tags = SUPER_AGENT_TAG
    with pytest.raises(RuntimeError, match="_stop_"):
        await client.list_agent_runtimes_with_options_async(
            req, {}, RuntimeOptions()
        )
    assert captured["query"].get("tags") == SUPER_AGENT_TAG
