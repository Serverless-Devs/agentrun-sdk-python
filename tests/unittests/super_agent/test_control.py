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

# 本文件部分测试依赖 Dara ProtocolConfiguration 已被打过补丁 (externalEndpoint),
# 显式在模块加载时触发补丁 (幂等, 与 SuperAgentClient.__init__ 内触发点一致)。
ensure_super_agent_patches_applied()

# ─── _prune_forwarded_props ───────────────────────────────────


def test_prune_forwarded_props_removes_none_scalars():
    from agentrun.super_agent.api.control import _prune_forwarded_props

    out = _prune_forwarded_props({"a": None, "b": "x"})
    assert out == {"b": "x"}


def test_prune_forwarded_props_removes_empty_lists():
    from agentrun.super_agent.api.control import _prune_forwarded_props

    out = _prune_forwarded_props({"a": [], "b": ["x"]})
    assert out == {"b": ["x"]}


def test_prune_forwarded_props_keeps_keep_keys_even_when_none():
    from agentrun.super_agent.api.control import _prune_forwarded_props

    out = _prune_forwarded_props(
        {"metadata": None, "other": None}, keep_keys=("metadata",)
    )
    assert out == {"metadata": None}


def test_prune_forwarded_props_keeps_keep_keys_even_when_empty_list():
    from agentrun.super_agent.api.control import _prune_forwarded_props

    out = _prune_forwarded_props(
        {"metadata": [], "other": []}, keep_keys=("metadata",)
    )
    assert out == {"metadata": []}


def test_prune_forwarded_props_preserves_falsy_non_none_scalars():
    """0, False, "" 不应被剔除 (只有 None 或空 list)."""
    from agentrun.super_agent.api.control import _prune_forwarded_props

    out = _prune_forwarded_props({"n": 0, "b": False, "s": ""})
    assert out == {"n": 0, "b": False, "s": ""}


def test_prune_forwarded_props_preserves_non_empty_lists_and_dicts():
    from agentrun.super_agent.api.control import _prune_forwarded_props

    out = _prune_forwarded_props({
        "list": ["x"],
        "dict_empty": {},  # dict 不算 list, 保留
        "dict_full": {"k": "v"},
    })
    assert out == {"list": ["x"], "dict_empty": {}, "dict_full": {"k": "v"}}


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
    assert inp.system_tags == [SUPER_AGENT_TAG]
    pc = inp.protocol_configuration
    assert pc.type == SUPER_AGENT_PROTOCOL_TYPE
    assert pc.external_endpoint.endswith("/super-agents/__SUPER_AGENT__")
    settings = pc.protocol_settings
    assert len(settings) == 1
    cfg_dict = json.loads(settings[0]["config"])
    assert cfg_dict["path"] == "/invoke"
    assert cfg_dict["headers"] == {}
    # 具体字段缺席断言放到 test_to_create_input_minimal_omits_unset_scalar_and_empty_list_fields
    forwarded = cfg_dict["body"]["forwardedProps"]
    assert forwarded["metadata"] == {"agentRuntimeName": "alpha"}


def test_to_create_input_minimal_omits_unset_scalar_and_empty_list_fields():
    """create 时, 未设置的 scalar 字段和空 list 字段 MUST NOT 出现在 forwardedProps 里."""
    cfg = Config(account_id="123", region_id="cn-hangzhou")
    inp = to_create_input("alpha", cfg=cfg)
    cfg_dict = json.loads(
        inp.protocol_configuration.protocol_settings[0]["config"]
    )
    forwarded = cfg_dict["body"]["forwardedProps"]
    # metadata 永远保留
    assert forwarded["metadata"] == {"agentRuntimeName": "alpha"}
    # 未设置的 scalar 字段缺席
    assert "prompt" not in forwarded
    assert "modelServiceName" not in forwarded
    assert "modelName" not in forwarded
    # 空 list 字段缺席
    assert "agents" not in forwarded
    assert "tools" not in forwarded
    assert "skills" not in forwarded
    assert "sandboxes" not in forwarded
    assert "workspaces" not in forwarded


def test_to_create_input_partial_only_keeps_set_fields():
    """仅设置部分字段时, 未设置的字段不出现, 已设置的字段按原值出现."""
    cfg = Config(account_id="123", region_id="cn-hangzhou")
    inp = to_create_input(
        "bravo", prompt="hello", model_service_name="svc", cfg=cfg
    )
    cfg_dict = json.loads(
        inp.protocol_configuration.protocol_settings[0]["config"]
    )
    forwarded = cfg_dict["body"]["forwardedProps"]
    assert forwarded["prompt"] == "hello"
    assert forwarded["modelServiceName"] == "svc"
    assert "modelName" not in forwarded
    assert "agents" not in forwarded
    assert forwarded["metadata"] == {"agentRuntimeName": "bravo"}


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
    assert settings_cfg["path"] == "/invoke"
    assert settings_cfg["headers"] == {}
    forwarded = settings_cfg["body"]["forwardedProps"]
    assert forwarded["prompt"] == "hello"
    assert forwarded["agents"] == ["a1"]
    assert forwarded["tools"] == ["t1", "t2"]
    assert forwarded["skills"] == ["s1"]
    assert forwarded["sandboxes"] == ["sb1"]
    assert forwarded["workspaces"] == ["ws1"]
    assert forwarded["modelServiceName"] == "foo"
    assert forwarded["modelName"] == "bar"


def test_to_create_input_system_tags_fixed():
    cfg = Config(account_id="123", region_id="cn-hangzhou")
    inp = to_create_input("c", cfg=cfg)
    assert inp.system_tags == [SUPER_AGENT_TAG]


def test_to_create_input_metadata_only_agent_runtime_name():
    cfg = Config(account_id="123", region_id="cn-hangzhou")
    inp = to_create_input("d", cfg=cfg)
    settings_cfg = json.loads(
        inp.protocol_configuration.protocol_settings[0]["config"]
    )
    assert settings_cfg["body"]["forwardedProps"]["metadata"] == {
        "agentRuntimeName": "d"
    }


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
        "path": "/invoke",
        "headers": {},
        "body": {
            "forwardedProps": {
                "prompt": "hi",
                "agents": ["a"],
                "tools": ["t"],
                "skills": [],
                "sandboxes": [],
                "workspaces": [],
                "modelServiceName": "svc",
                "modelName": "mod",
                "metadata": {"agentRuntimeName": "foo"},
            }
        },
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


def test_from_agent_runtime_legacy_flat_config():
    """旧结构兼容: config 是扁平 dict, 业务字段直接在根 (历史 AgentRuntime)."""
    config_json = json.dumps({
        "prompt": "legacy",
        "agents": ["la"],
        "tools": ["lt"],
        "skills": [],
        "sandboxes": [],
        "workspaces": [],
        "modelServiceName": "legacy-svc",
        "modelName": "legacy-mod",
        "metadata": {"agentRuntimeName": "legacy"},
    })
    pc = {
        "type": SUPER_AGENT_PROTOCOL_TYPE,
        "protocolSettings": [{
            "type": SUPER_AGENT_PROTOCOL_TYPE,
            "config": config_json,
            "name": "legacy",
            "path": "/invoke",
        }],
        "externalEndpoint": "https://x.com/super-agents/__SUPER_AGENT__",
    }
    rt = _make_rt(agent_runtime_name="legacy", protocol_configuration=pc)
    agent = from_agent_runtime(rt)
    assert agent.prompt == "legacy"
    assert agent.agents == ["la"]
    assert agent.model_service_name == "legacy-svc"


def test_parse_super_agent_config_dict_config_new_structure():
    """config 已经是 dict (非字符串) 时也能拍平."""
    pc = {
        "type": SUPER_AGENT_PROTOCOL_TYPE,
        "protocolSettings": [{
            "type": SUPER_AGENT_PROTOCOL_TYPE,
            "config": {
                "path": "/invoke",
                "headers": {},
                "body": {
                    "forwardedProps": {
                        "prompt": "p",
                        "agents": [],
                    }
                },
            },
        }],
    }
    business = parse_super_agent_config(_make_rt(protocol_configuration=pc))
    assert business["prompt"] == "p"
    assert business["agents"] == []


def test_parse_super_agent_config_dict_config_legacy_flat():
    """config 是 dict + 旧扁平结构时走 fallback, 原样返回."""
    pc = {
        "type": SUPER_AGENT_PROTOCOL_TYPE,
        "protocolSettings": [{
            "type": SUPER_AGENT_PROTOCOL_TYPE,
            "config": {"prompt": "legacy-dict", "agents": ["la"]},
        }],
    }
    business = parse_super_agent_config(_make_rt(protocol_configuration=pc))
    assert business == {"prompt": "legacy-dict", "agents": ["la"]}


def test_flatten_protocol_config_non_dict_returns_empty():
    """非 dict 输入 (防御分支) 返回空 dict."""
    from agentrun.super_agent.api.control import _flatten_protocol_config

    assert _flatten_protocol_config(None) == {}
    assert _flatten_protocol_config("not-a-dict") == {}
    assert _flatten_protocol_config([1, 2, 3]) == {}


def test_flatten_protocol_config_body_without_forwarded_props():
    """body 存在但缺 forwardedProps → fallback 到整个 cfg (旧结构)."""
    from agentrun.super_agent.api.control import _flatten_protocol_config

    cfg = {"body": {"other": "x"}, "prompt": "flat"}
    assert _flatten_protocol_config(cfg) == cfg


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
    cfg_json = json.loads(settings[0]["config"])
    forwarded = cfg_json["body"]["forwardedProps"]
    assert forwarded["metadata"]["agentRuntimeName"] == "alpha"
    assert forwarded["prompt"] == "p"
    assert forwarded["tools"] == ["t"]


def test_to_update_input_keeps_null_for_none_scalars():
    """update 路径: 合并后为 None 的 scalar 字段 MUST 仍写 null (不剪除).

    保证 SDK 的 'update(model_name=None) 表示清空' 语义不被本次 PR 破坏。
    """
    cfg = Config(account_id="123", region_id="cn-hangzhou")
    merged = {
        "prompt": None,
        "agents": [],
        "tools": [],
        "skills": [],
        "sandboxes": [],
        "workspaces": [],
        "model_service_name": None,
        "model_name": None,
    }
    inp = to_update_input("u1", merged, cfg=cfg)
    cfg_dict = json.loads(
        inp.protocol_configuration.protocol_settings[0]["config"]
    )
    forwarded = cfg_dict["body"]["forwardedProps"]
    # 明确包含 null (未被剪除)
    assert "prompt" in forwarded and forwarded["prompt"] is None
    assert (
        "modelServiceName" in forwarded
        and forwarded["modelServiceName"] is None
    )
    assert "modelName" in forwarded and forwarded["modelName"] is None
    # 空 list 也保留 (update 语义下 [] 代表 "清空列表", 不能剪除)
    assert forwarded["agents"] == []
    assert forwarded["tools"] == []


def test_to_update_input_keeps_values_and_nulls_mixed():
    """update: 有些字段有值, 有些是 None, 都应完整出现在 payload."""
    cfg = Config(account_id="123", region_id="cn-hangzhou")
    merged = {
        "prompt": "new",
        "agents": ["a"],
        "model_service_name": None,
        "model_name": "m",
    }
    inp = to_update_input("u2", merged, cfg=cfg)
    cfg_dict = json.loads(
        inp.protocol_configuration.protocol_settings[0]["config"]
    )
    forwarded = cfg_dict["body"]["forwardedProps"]
    assert forwarded["prompt"] == "new"
    assert forwarded["agents"] == ["a"]
    assert (
        "modelServiceName" in forwarded
        and forwarded["modelServiceName"] is None
    )
    assert forwarded["modelName"] == "m"


# ─── Dara ListAgentRuntimesRequest systemTags 原生字段 ──────────────
# ``systemTags`` 已由 Dara SDK 原生支持, 无需补丁。以下测试只校验 pydantic →
# Dara roundtrip 能把 ``system_tags`` 保留到请求 query。


def test_list_request_from_map_preserves_system_tags():
    from alibabacloud_agentrun20250910.models import ListAgentRuntimesRequest

    req = ListAgentRuntimesRequest().from_map({
        "systemTags": SUPER_AGENT_TAG,
        "pageNumber": 1,
        "pageSize": 20,
    })
    assert req.system_tags == SUPER_AGENT_TAG


def test_list_request_to_map_preserves_system_tags():
    from alibabacloud_agentrun20250910.models import ListAgentRuntimesRequest

    req = ListAgentRuntimesRequest()
    req.system_tags = SUPER_AGENT_TAG
    assert req.to_map().get("systemTags") == SUPER_AGENT_TAG


def test_list_with_options_writes_system_tags_query():
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

    req = ListAgentRuntimesRequest(
        page_number=1, page_size=20, system_tags=SUPER_AGENT_TAG
    )
    with pytest.raises(RuntimeError, match="_stop_"):
        client.list_agent_runtimes_with_options(req, {}, RuntimeOptions())
    assert captured["query"].get("systemTags") == SUPER_AGENT_TAG
