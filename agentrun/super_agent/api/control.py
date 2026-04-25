"""Super Agent 控制面辅助函数 / Super Agent Control Plane Helpers

本模块包含:
- 常量: API 版本号 / 协议类型 / 标签 / 资源路径 / RAM 数据域名列表
- URL 工具: ``_add_ram_prefix_to_host`` / ``build_super_agent_endpoint``
- AgentRuntime ↔ SuperAgent 的双向转换:
  ``to_create_input`` / ``to_update_input`` / ``from_agent_runtime``
  / ``is_super_agent`` / ``parse_super_agent_config``
- 为承载 ``externalEndpoint`` 的 Pydantic 与 Dara 层扩展类

不使用模板生成,保持单一来源,避免同步/异步重复维护。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

if TYPE_CHECKING:
    from agentrun.super_agent.agent import SuperAgent

from alibabacloud_agentrun20250910.models import (
    ProtocolConfiguration as _DaraProtocolConfiguration,
)
from pydantic import Field

from agentrun.agent_runtime.model import (
    AgentRuntimeArtifact,
    AgentRuntimeContainer,
    AgentRuntimeCreateInput,
    AgentRuntimeProtocolConfig,
    AgentRuntimeUpdateInput,
)
from agentrun.agent_runtime.runtime import AgentRuntime
from agentrun.utils.config import Config
from agentrun.utils.model import NetworkConfig, NetworkMode

# ─── 常量 ─────────────────────────────────────────────
API_VERSION = "2025-09-10"
SUPER_AGENT_PROTOCOL_TYPE = "SUPER_AGENT"
# ``SUPER_AGENT_TAG`` 标识下游 AgentRuntime 是超级 Agent, 用于 list 过滤。
# 写入 ``systemTags`` 字段 (由服务端原生支持), create/update/list 的 system_tags 参数统一使用。
SUPER_AGENT_TAG = "x-agentrun-super"
# ``EXTERNAL_TAG`` 标识下游 AgentRuntime 由外部 (SuperAgent) 托管调用, 不由 AgentRun 直接托管。
# 保留常量以便外部消费者引用; 创建超级 Agent 时不再写入此 tag。
EXTERNAL_TAG = "x-agentrun-external"
# 创建下游 AgentRuntime 时固定写入的 systemTags 列表: ``[SUPER_AGENT_TAG]`` (仅一个)。
SUPER_AGENT_CREATE_TAGS = [SUPER_AGENT_TAG]
SUPER_AGENT_RESOURCE_PATH = "__SUPER_AGENT__"
SUPER_AGENT_INVOKE_PATH = "/invoke"
SUPER_AGENT_NAMESPACE = (
    f"{API_VERSION}/super-agents/{SUPER_AGENT_RESOURCE_PATH}"
)

_RAM_DATA_DOMAINS = ("agentrun-data", "funagent-data-pre")

# SUPER_AGENT 不跑用户 container/code, 但服务端强制要求 artifact/container_configuration 非空,
# 这里给一个占位镜像地址即可。region 取杭州仅为了格式合法, 服务端不会实际 pull。
_PLACEHOLDER_IMAGE = (
    "registry.cn-hangzhou.aliyuncs.com/agentrun/super-agent-placeholder:v1"
)


# ─── URL 工具 ──────────────────────────────────────────


def _add_ram_prefix_to_host(base_url: str) -> str:
    """给已知 data host 加 ``-ram`` 前缀.

    仅当 host 第二段命中 :data:`_RAM_DATA_DOMAINS` 时改写为
    ``<host前缀>-ram.<其余>``, 其他情况原样返回。
    与 :meth:`agentrun.utils.data_api.DataAPI._get_ram_data_endpoint` 同源。
    """
    parsed = urlparse(base_url)
    if not parsed.netloc:
        return base_url
    if not any(f".{d}." in parsed.netloc for d in _RAM_DATA_DOMAINS):
        return base_url
    parts = parsed.netloc.split(".", 1)
    if len(parts) != 2:
        return base_url
    ram_netloc = parts[0] + "-ram." + parts[1]
    return urlunparse((
        parsed.scheme,
        ram_netloc,
        parsed.path or "",
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))


def build_super_agent_endpoint(cfg: Optional[Config] = None) -> str:
    """构造 ``protocolConfiguration.externalEndpoint`` 的存储值 (不带版本号).

    基于 :meth:`Config.get_data_endpoint` + :func:`_add_ram_prefix_to_host`
    + 追加 ``/super-agents/__SUPER_AGENT__``, 自动适配生产 / 预发 / 自定义网关。
    """
    cfg = Config.with_configs(cfg)
    base = cfg.get_data_endpoint()
    ram_base = _add_ram_prefix_to_host(base)
    return f"{ram_base.rstrip('/')}/super-agents/{SUPER_AGENT_RESOURCE_PATH}"


# ─── Pydantic 扩展类 ────────────────────────────────────
class SuperAgentProtocolConfig(AgentRuntimeProtocolConfig):
    """承载 ``protocol_settings`` + ``external_endpoint`` 的 Pydantic 扩展.

    基类 ``AgentRuntimeProtocolConfig`` 的 ``type`` 字段是 ``HTTP / MCP`` 枚举,
    本子类通过 ``model_construct`` 绕过校验存入字符串 ``"SUPER_AGENT"``。
    """

    protocol_settings: Optional[List[Dict[str, Any]]] = Field(
        alias="protocolSettings", default=None
    )
    external_endpoint: Optional[str] = Field(
        alias="externalEndpoint", default=None
    )


class _SuperAgentCreateInput(AgentRuntimeCreateInput):
    """默认使用 ``serialize_as_any=True`` 的 create input, 保留子类 extras.

    ``external_agent_endpoint_url`` 是基类 ``AgentRuntimeMutableProps`` 没有覆盖
    的顶层字段, 这里显式补齐 (alias 由 BaseModel 的 ``to_camel_case`` 生成 →
    ``externalAgentEndpointUrl``), 用于承载超级 Agent 的数据面入口地址。
    """

    external_agent_endpoint_url: Optional[str] = None

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump(**kwargs)


class _SuperAgentUpdateInput(AgentRuntimeUpdateInput):
    """默认使用 ``serialize_as_any=True`` 的 update input, 保留子类 extras."""

    external_agent_endpoint_url: Optional[str] = None

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump(**kwargs)


# ─── Dara 模型猴补丁 ──────────────────────────────────────
# 当前版 Dara SDK 缺 ``ProtocolConfiguration.externalEndpoint`` 字段, 会在
# Pydantic ↔ Dara ``from_map / to_map`` roundtrip 中静默丢失。补丁延迟到
# ``SuperAgentClient`` 实例化时 (见 ``ensure_super_agent_patches_applied``) 才
# 触发, 避免仅 import 本模块的调用方被动承担全局副作用。补丁用哨兵属性保证
# 幂等, 重复调用安全。TODO: 等 Dara SDK 原生支持后删除。
#
# ``tags`` (原 hack 写入) 已由 SDK 原生 ``systemTags`` 字段替代, 不再需要任何
# 补丁; create/update/list 统一走 ``system_tags`` 参数。


def _patch_dara_protocol_configuration() -> None:
    """补齐 ``ProtocolConfiguration.externalEndpoint`` 的 from_map/to_map 读写."""
    if getattr(_DaraProtocolConfiguration, "_super_agent_patched", False):
        return

    _orig_to_map = _DaraProtocolConfiguration.to_map
    _orig_from_map = _DaraProtocolConfiguration.from_map

    def _patched_to_map(self: _DaraProtocolConfiguration) -> Dict[str, Any]:
        result = _orig_to_map(self)
        ee = getattr(self, "external_endpoint", None)
        if ee is not None:
            result["externalEndpoint"] = ee
        return result

    def _patched_from_map(
        self: _DaraProtocolConfiguration, m: Optional[Dict[str, Any]] = None
    ) -> _DaraProtocolConfiguration:
        _orig_from_map(self, m)
        if m and m.get("externalEndpoint") is not None:
            self.external_endpoint = m.get("externalEndpoint")
        return self

    _DaraProtocolConfiguration.to_map = _patched_to_map  # type: ignore[assignment]
    _DaraProtocolConfiguration.from_map = _patched_from_map  # type: ignore[assignment]
    _DaraProtocolConfiguration._super_agent_patched = True  # type: ignore[attr-defined]


def ensure_super_agent_patches_applied() -> None:
    """按需应用 Dara SDK 兼容补丁 (幂等)。

    由 ``SuperAgentClient.__init__`` 调用。如果调用方直接使用
    ``to_create_input`` / ``to_update_input`` 并自己构造 Dara 输入, 也应在
    Pydantic → Dara 转换前调用一次本函数。
    """
    _patch_dara_protocol_configuration()


# ─── AgentRuntime ↔ SuperAgent 转换 ────────────────────────
def _business_fields_from_args(
    *,
    prompt: Optional[str] = None,
    agents: Optional[List[str]] = None,
    tools: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    sandboxes: Optional[List[str]] = None,
    workspaces: Optional[List[str]] = None,
    model_service_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """把业务字段 (None 保留为 None) 收拢成 dict, 供 ``protocolSettings.config`` 使用."""
    return {
        "prompt": prompt,
        "agents": agents if agents is not None else [],
        "tools": tools if tools is not None else [],
        "skills": skills if skills is not None else [],
        "sandboxes": sandboxes if sandboxes is not None else [],
        "workspaces": workspaces if workspaces is not None else [],
        "modelServiceName": model_service_name,
        "modelName": model_name,
    }


def _prune_forwarded_props(
    props: Dict[str, Any],
    *,
    keep_keys: Iterable[str] = ("metadata",),
) -> Dict[str, Any]:
    """删除值为 None 的 scalar 字段和空 list 字段。

    ``keep_keys`` 里的 key 永远保留 (即便是 None 或空 list), 用来保护 SDK 托管
    的必要字段 (如 ``metadata`` / ``conversationId``)。

    语义上只处理两类:
    - scalar = None → 丢弃
    - 空 list → 丢弃

    其他 falsy 值 (0 / False / "" / 空 dict) 保留, 因为它们是业务显式值。
    """
    keep = set(keep_keys)
    out: Dict[str, Any] = {}
    for k, v in props.items():
        if k in keep:
            out[k] = v
            continue
        if v is None:
            continue
        if isinstance(v, list) and not v:
            continue
        out[k] = v
    return out


def _build_protocol_settings_config(
    *, name: str, business: Dict[str, Any], prune_props: bool = False
) -> str:
    """构造 ``protocolSettings[0].config`` 的 JSON 字符串.

    新结构: 顶层 ``path`` / ``headers`` / ``body``, 业务字段收拢到
    ``body.forwardedProps`` (开放字典, 语义 "any, merge")。

    ``prune_props=True`` 时, 对 forwardedProps 过一遍 :func:`_prune_forwarded_props`,
    丢弃 None scalar 和空 list 字段 (保留 ``metadata``)。create 路径使用; update
    路径使用 False, 仍写 null 以保留 "显式清空" 语义。
    """
    forwarded_props: Dict[str, Any] = {
        "prompt": business.get("prompt"),
        "agents": business.get("agents") or [],
        "tools": business.get("tools") or [],
        "skills": business.get("skills") or [],
        "sandboxes": business.get("sandboxes") or [],
        "workspaces": business.get("workspaces") or [],
        "modelServiceName": business.get("modelServiceName"),
        "modelName": business.get("modelName"),
        "metadata": {"agentRuntimeName": name},
    }
    if prune_props:
        forwarded_props = _prune_forwarded_props(
            forwarded_props, keep_keys=("metadata",)
        )
    cfg_dict: Dict[str, Any] = {
        "path": SUPER_AGENT_INVOKE_PATH,
        "headers": {},
        "body": {"forwardedProps": forwarded_props},
    }
    return json.dumps(cfg_dict, ensure_ascii=False)


def _build_protocol_configuration(
    *,
    name: str,
    business: Dict[str, Any],
    cfg: Optional[Config],
    prune_props: bool = False,
) -> SuperAgentProtocolConfig:
    """构造超级 Agent 的 ``protocolConfiguration`` Pydantic 模型.

    ``prune_props`` 透传到 :func:`_build_protocol_settings_config`。
    """
    config_json = _build_protocol_settings_config(
        name=name, business=business, prune_props=prune_props
    )
    settings: List[Dict[str, Any]] = [{
        "type": SUPER_AGENT_PROTOCOL_TYPE,
        "name": name,
        "path": SUPER_AGENT_INVOKE_PATH,
        "config": config_json,
    }]
    pc = SuperAgentProtocolConfig.model_construct(
        type=SUPER_AGENT_PROTOCOL_TYPE,
        protocol_settings=settings,
        external_endpoint=build_super_agent_endpoint(cfg),
    )
    return pc


def to_create_input(
    name: str,
    *,
    description: Optional[str] = None,
    prompt: Optional[str] = None,
    agents: Optional[List[str]] = None,
    tools: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
    sandboxes: Optional[List[str]] = None,
    workspaces: Optional[List[str]] = None,
    model_service_name: Optional[str] = None,
    model_name: Optional[str] = None,
    cfg: Optional[Config] = None,
) -> AgentRuntimeCreateInput:
    """把超级 Agent 业务字段转为 :class:`AgentRuntimeCreateInput`."""
    business = _business_fields_from_args(
        prompt=prompt,
        agents=agents,
        tools=tools,
        skills=skills,
        sandboxes=sandboxes,
        workspaces=workspaces,
        model_service_name=model_service_name,
        model_name=model_name,
    )
    pc = _build_protocol_configuration(
        name=name, business=business, cfg=cfg, prune_props=True
    )
    # SUPER_AGENT 是平台托管运行时, 不跑用户代码/容器, 但服务端仍要求
    # artifact_type / network_configuration 非空. 这里给占位默认值即可.
    return _SuperAgentCreateInput.model_construct(
        agent_runtime_name=name,
        description=description,
        protocol_configuration=pc,
        system_tags=list(SUPER_AGENT_CREATE_TAGS),
        # 超级 Agent 的数据面入口 (与 protocolConfiguration.externalEndpoint 同值)。
        external_agent_endpoint_url=build_super_agent_endpoint(cfg),
        # 占位 artifact: SUPER_AGENT 不跑用户 container/code, 但服务端要求非空。
        artifact_type=AgentRuntimeArtifact.CONTAINER,
        container_configuration=AgentRuntimeContainer(image=_PLACEHOLDER_IMAGE),
        network_configuration=NetworkConfig(network_mode=NetworkMode.PUBLIC),
    )


def to_update_input(
    name: str,
    merged: Dict[str, Any],
    cfg: Optional[Config] = None,
) -> AgentRuntimeUpdateInput:
    """把合并后的业务字段转为 :class:`AgentRuntimeUpdateInput` (全量替换)."""
    business = _business_fields_from_args(
        prompt=merged.get("prompt"),
        agents=merged.get("agents"),
        tools=merged.get("tools"),
        skills=merged.get("skills"),
        sandboxes=merged.get("sandboxes"),
        workspaces=merged.get("workspaces"),
        model_service_name=merged.get("model_service_name"),
        model_name=merged.get("model_name"),
    )
    pc = _build_protocol_configuration(name=name, business=business, cfg=cfg)
    return _SuperAgentUpdateInput.model_construct(
        agent_runtime_name=name,
        description=merged.get("description"),
        protocol_configuration=pc,
        system_tags=list(SUPER_AGENT_CREATE_TAGS),
        # 超级 Agent 的数据面入口 (与 protocolConfiguration.externalEndpoint 同值)。
        external_agent_endpoint_url=build_super_agent_endpoint(cfg),
        # 占位 artifact: SUPER_AGENT 不跑用户 container/code, 但服务端要求非空。
        artifact_type=AgentRuntimeArtifact.CONTAINER,
        container_configuration=AgentRuntimeContainer(image=_PLACEHOLDER_IMAGE),
        network_configuration=NetworkConfig(network_mode=NetworkMode.PUBLIC),
    )


def _extract_protocol_configuration(rt: AgentRuntime) -> Dict[str, Any]:
    """统一把 rt.protocol_configuration 转为 dict (兼容 dict / pydantic 两种形态)."""
    pc = getattr(rt, "protocol_configuration", None)
    if pc is None:
        return {}
    if isinstance(pc, dict):
        return pc
    # Pydantic 模型: 用 model_dump(serialize_as_any=True) 保留 extras
    try:
        return pc.model_dump(by_alias=True, serialize_as_any=True)
    except TypeError:
        return pc.model_dump(by_alias=True)


def _extract_protocol_settings(pc_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 protocolConfiguration dict 中取出 protocolSettings 列表."""
    for key in ("protocolSettings", "protocol_settings"):
        v = pc_dict.get(key)
        if isinstance(v, list):
            return v
    return []


def is_super_agent(rt: AgentRuntime) -> bool:
    """判断一个 AgentRuntime 是否为超级 Agent."""
    pc_dict = _extract_protocol_configuration(rt)
    if not pc_dict:
        return False
    settings = _extract_protocol_settings(pc_dict)
    if not settings:
        return False
    first = settings[0] if isinstance(settings[0], dict) else {}
    return first.get("type") == SUPER_AGENT_PROTOCOL_TYPE


def _flatten_protocol_config(cfg: Any) -> Dict[str, Any]:
    """把 ``protocolSettings[0].config`` 解析结果压平为扁平业务字段 dict.

    兼容两种物理布局:
    - 新: ``{"path": ..., "headers": ..., "body": {"forwardedProps": {...}}}``
    - 旧: 业务字段直接在根 (历史 AgentRuntime, 迁移前写入)

    两种结构都返回扁平的业务字段 dict, 上游
    :func:`from_agent_runtime` 无需感知物理布局差异。
    """
    if not isinstance(cfg, dict):
        return {}
    body = cfg.get("body")
    if isinstance(body, dict):
        forwarded = body.get("forwardedProps")
        if isinstance(forwarded, dict):
            return forwarded
    return cfg


def parse_super_agent_config(rt: AgentRuntime) -> Dict[str, Any]:
    """反解 ``protocolSettings[0].config`` 为扁平业务字段 dict.

    如果 config 缺失或非法 JSON, 返回空 dict (不抛异常)。
    新旧嵌套布局由 :func:`_flatten_protocol_config` 统一拍平。
    """
    pc_dict = _extract_protocol_configuration(rt)
    if not pc_dict:
        return {}
    settings = _extract_protocol_settings(pc_dict)
    if not settings:
        return {}
    first = settings[0] if isinstance(settings[0], dict) else {}
    raw_config = first.get("config")
    if not raw_config:
        return {}
    if isinstance(raw_config, dict):
        return _flatten_protocol_config(raw_config)
    if isinstance(raw_config, str):
        try:
            parsed = json.loads(raw_config)
        except (TypeError, ValueError):
            return {}
        return _flatten_protocol_config(parsed)
    return {}


def _get_external_endpoint(rt: AgentRuntime) -> str:
    pc_dict = _extract_protocol_configuration(rt)
    return (
        pc_dict.get("externalEndpoint")
        or pc_dict.get("external_endpoint", "")
        or ""
    )


def from_agent_runtime(rt: AgentRuntime) -> "SuperAgent":
    """反解 AgentRuntime → SuperAgent 实例 (不注入 ``_client``)."""
    # 延迟导入避免循环
    from agentrun.super_agent.agent import SuperAgent

    business = parse_super_agent_config(rt)
    return SuperAgent(
        name=getattr(rt, "agent_runtime_name", None) or "",
        description=getattr(rt, "description", None),
        prompt=business.get("prompt"),
        agents=list(business.get("agents") or []),
        tools=list(business.get("tools") or []),
        skills=list(business.get("skills") or []),
        sandboxes=list(business.get("sandboxes") or []),
        workspaces=list(business.get("workspaces") or []),
        model_service_name=business.get("modelServiceName"),
        model_name=business.get("modelName"),
        agent_runtime_id=getattr(rt, "agent_runtime_id", None) or "",
        arn=getattr(rt, "agent_runtime_arn", None) or "",
        status=str(getattr(rt, "status", "") or ""),
        created_at=getattr(rt, "created_at", None) or "",
        last_updated_at=getattr(rt, "last_updated_at", None) or "",
        external_endpoint=_get_external_endpoint(rt),
    )


__all__ = [
    "API_VERSION",
    "SUPER_AGENT_PROTOCOL_TYPE",
    "SUPER_AGENT_TAG",
    "EXTERNAL_TAG",
    "SUPER_AGENT_CREATE_TAGS",
    "SUPER_AGENT_RESOURCE_PATH",
    "SUPER_AGENT_INVOKE_PATH",
    "SUPER_AGENT_NAMESPACE",
    "SuperAgentProtocolConfig",
    "_SuperAgentCreateInput",
    "_SuperAgentUpdateInput",
    "build_super_agent_endpoint",
    "_add_ram_prefix_to_host",
    "to_create_input",
    "to_update_input",
    "from_agent_runtime",
    "is_super_agent",
    "parse_super_agent_config",
    "ensure_super_agent_patches_applied",
]
