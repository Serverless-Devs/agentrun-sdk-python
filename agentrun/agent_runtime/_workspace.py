"""Workspace 名称解析助手 / Workspace Name Resolution Helper

提供 ``workspace_name -> workspace_id`` 的解析能力，供 ``AgentRuntimeClient``
在 ``create`` / ``list`` 等场景下自动转换用户传入的工作空间名称。

The official AgentRun API only accepts ``workspace_id``. The SDK exposes a
convenience field ``workspace_name``; this module wraps ``list_workspaces``
to look the id up by name (exact match, with a simple in-memory cache).
"""

from typing import Dict, List, Optional, Tuple

from alibabacloud_agentrun20250910.models import (
    ListWorkspacesRequest,
    Workspace,
)
from alibabacloud_tea_openapi.exceptions._client import ClientException
from alibabacloud_tea_openapi.exceptions._server import ServerException
import pydash

from agentrun.utils.config import Config
from agentrun.utils.control_api import ControlAPI
from agentrun.utils.exception import (
    ClientError,
    ResourceNotExistError,
    ServerError,
)

# Cache key 为 (access_key_id, region_id, name)，避免不同账号/地域串号。
# Value 为解析得到的 workspace_id。
_RESOLVE_CACHE: Dict[Tuple[str, str, str], str] = {}


def _cache_key(cfg: Config, name: str) -> Tuple[str, str, str]:
    return (
        cfg.get_access_key_id() or "",
        cfg.get_region_id() or "",
        name,
    )


def _pick_exact_match(
    workspaces: List[Workspace], name: str
) -> Optional[Workspace]:
    matches = [w for w in workspaces if w.name == name]
    if len(matches) > 1:
        raise ValueError(
            f"Workspace name {name!r} is ambiguous: matched"
            f" {len(matches)} workspaces; please use workspace_id instead."
        )
    return matches[0] if matches else None


def _raise_for_tea_exception(e: Exception) -> None:
    if isinstance(e, ClientException):
        raise ClientError(
            e.status_code,
            pydash.get(e, "data.message", pydash.get(e, "message", "")),
            pydash.get(e, "data.requestId", ""),
            pydash.get(e, "data.code", ""),
        ) from e
    if isinstance(e, ServerException):
        raise ServerError(
            e.status_code,
            pydash.get(e, "data.message", pydash.get(e, "message", "")),
            pydash.get(e, "data.requestId", ""),
            pydash.get(e, "data.code", ""),
        ) from e


class _WorkspaceResolver(ControlAPI):
    """轻量封装：复用 ControlAPI 拿底层 AgentRun client。"""

    def resolve(self, name: str, config: Optional[Config] = None) -> str:
        if not name:
            raise ValueError("workspace_name must be non-empty")

        cfg = Config.with_configs(self.config, config)
        cache_key = _cache_key(cfg, name)
        if cache_key in _RESOLVE_CACHE:
            return _RESOLVE_CACHE[cache_key]

        ws = self._lookup_sync(name, config)
        if ws is None:
            raise ResourceNotExistError("Workspace", name)

        assert ws.workspace_id is not None
        _RESOLVE_CACHE[cache_key] = ws.workspace_id
        return ws.workspace_id

    async def resolve_async(
        self, name: str, config: Optional[Config] = None
    ) -> str:
        if not name:
            raise ValueError("workspace_name must be non-empty")

        cfg = Config.with_configs(self.config, config)
        cache_key = _cache_key(cfg, name)
        if cache_key in _RESOLVE_CACHE:
            return _RESOLVE_CACHE[cache_key]

        ws = await self._lookup_async(name, config)
        if ws is None:
            raise ResourceNotExistError("Workspace", name)

        assert ws.workspace_id is not None
        _RESOLVE_CACHE[cache_key] = ws.workspace_id
        return ws.workspace_id

    # --- internal -----------------------------------------------------------

    def _lookup_sync(
        self, name: str, config: Optional[Config] = None
    ) -> Optional[Workspace]:
        client = self._get_client(config)
        try:
            response = client.list_workspaces(
                ListWorkspacesRequest(name=name, page_size="50")
            )
        except (ClientException, ServerException) as e:
            _raise_for_tea_exception(e)
            raise
        workspaces = (
            getattr(getattr(response.body, "data", None), "workspaces", None)
            or []
        )
        return _pick_exact_match(workspaces, name)

    async def _lookup_async(
        self, name: str, config: Optional[Config] = None
    ) -> Optional[Workspace]:
        client = self._get_client(config)
        try:
            response = await client.list_workspaces_async(
                ListWorkspacesRequest(name=name, page_size="50")
            )
        except (ClientException, ServerException) as e:
            _raise_for_tea_exception(e)
            raise
        workspaces = (
            getattr(getattr(response.body, "data", None), "workspaces", None)
            or []
        )
        return _pick_exact_match(workspaces, name)


def resolve_workspace_id_by_name(
    name: str, config: Optional[Config] = None
) -> str:
    """同步：根据 workspace name 解析出 workspace_id。

    Raises:
        ValueError: ``name`` 为空，或在该账号下存在重名 workspace。
        ResourceNotExistError: 该账号下未找到同名 workspace。
    """

    return _WorkspaceResolver(config).resolve(name, config)


async def resolve_workspace_id_by_name_async(
    name: str, config: Optional[Config] = None
) -> str:
    """异步：根据 workspace name 解析出 workspace_id。"""

    return await _WorkspaceResolver(config).resolve_async(name, config)


def resolve_workspace_ids_by_names(
    names: str, config: Optional[Config] = None
) -> str:
    """同步：将逗号分隔的多个 workspace 名称解析为逗号分隔的 workspace_id 列表。"""

    return ",".join(
        resolve_workspace_id_by_name(n.strip(), config)
        for n in names.split(",")
        if n.strip()
    )


async def resolve_workspace_ids_by_names_async(
    names: str, config: Optional[Config] = None
) -> str:
    """异步：将逗号分隔的多个 workspace 名称解析为逗号分隔的 workspace_id 列表。"""

    out: List[str] = []
    for n in names.split(","):
        n = n.strip()
        if not n:
            continue
        out.append(await resolve_workspace_id_by_name_async(n, config))
    return ",".join(out)


def _clear_cache_for_tests() -> None:
    """仅供单测使用：清空内部解析缓存。"""

    _RESOLVE_CACHE.clear()


__all__ = [
    "resolve_workspace_id_by_name",
    "resolve_workspace_id_by_name_async",
    "resolve_workspace_ids_by_names",
    "resolve_workspace_ids_by_names_async",
]
