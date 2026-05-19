"""Workspace 名称解析 & 客户端集成单元测试"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.agent_runtime import _workspace as ws_mod
from agentrun.agent_runtime._workspace import (
    _clear_cache_for_tests,
    resolve_workspace_id_by_name,
    resolve_workspace_id_by_name_async,
    resolve_workspace_ids_by_names,
    resolve_workspace_ids_by_names_async,
)
from agentrun.agent_runtime.model import (
    AgentRuntimeContainer,
    AgentRuntimeCreateInput,
    AgentRuntimeListInput,
)
from agentrun.utils.exception import ResourceNotExistError

CONTROL_API_PATH = "agentrun.agent_runtime.client.AgentRuntimeControlAPI"


def _make_ws(name: str, ws_id: str):
    ws = MagicMock()
    ws.name = name
    ws.workspace_id = ws_id
    return ws


def _make_response(workspaces):
    resp = MagicMock()
    resp.body.data.workspaces = workspaces
    return resp


@pytest.fixture(autouse=True)
def _clear_ws_cache():
    _clear_cache_for_tests()
    yield
    _clear_cache_for_tests()


class TestResolveWorkspace:
    """resolve_workspace_id_by_name 行为测试"""

    def _patch_client(self, monkeypatch, workspaces):
        client = MagicMock()
        client.list_workspaces.return_value = _make_response(workspaces)
        client.list_workspaces_async = AsyncMock(
            return_value=_make_response(workspaces)
        )
        monkeypatch.setattr(
            ws_mod._WorkspaceResolver,
            "_get_client",
            lambda self, config=None: client,
        )
        return client

    def test_resolve_sync_exact_match(self, monkeypatch):
        self._patch_client(
            monkeypatch,
            [_make_ws("other", "ws-other"), _make_ws("my-ws", "ws-123")],
        )
        assert resolve_workspace_id_by_name("my-ws") == "ws-123"

    def test_resolve_sync_uses_cache(self, monkeypatch):
        client = self._patch_client(monkeypatch, [_make_ws("my-ws", "ws-123")])
        assert resolve_workspace_id_by_name("my-ws") == "ws-123"
        assert resolve_workspace_id_by_name("my-ws") == "ws-123"
        # 二次解析应命中缓存，避免重复 list_workspaces
        assert client.list_workspaces.call_count == 1

    def test_resolve_sync_not_found(self, monkeypatch):
        self._patch_client(monkeypatch, [])
        with pytest.raises(ResourceNotExistError):
            resolve_workspace_id_by_name("absent")

    def test_resolve_sync_ambiguous(self, monkeypatch):
        self._patch_client(
            monkeypatch,
            [_make_ws("dup", "ws-1"), _make_ws("dup", "ws-2")],
        )
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_workspace_id_by_name("dup")

    def test_resolve_sync_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            resolve_workspace_id_by_name("")

    def test_resolve_sync_none_workspaces_response(self, monkeypatch):
        # response.body.data.workspaces 为 None 时应正常报 not-exist
        self._patch_client(monkeypatch, None)
        with pytest.raises(ResourceNotExistError):
            resolve_workspace_id_by_name("absent")

    def test_resolve_sync_client_exception_raises_client_error(
        self, monkeypatch
    ):
        from alibabacloud_tea_openapi.exceptions._client import ClientException

        from agentrun.utils.exception import ClientError

        client = MagicMock()
        client.list_workspaces.side_effect = ClientException(
            status_code=403, data={"message": "denied"}
        )
        monkeypatch.setattr(
            ws_mod._WorkspaceResolver,
            "_get_client",
            lambda self, config=None: client,
        )
        with pytest.raises(ClientError):
            resolve_workspace_id_by_name("my-ws")

    def test_resolve_sync_server_exception_raises_server_error(
        self, monkeypatch
    ):
        from alibabacloud_tea_openapi.exceptions._server import ServerException

        from agentrun.utils.exception import ServerError

        client = MagicMock()
        client.list_workspaces.side_effect = ServerException(
            status_code=500, data={"message": "boom"}
        )
        monkeypatch.setattr(
            ws_mod._WorkspaceResolver,
            "_get_client",
            lambda self, config=None: client,
        )
        with pytest.raises(ServerError):
            resolve_workspace_id_by_name("my-ws")

    def test_resolve_async_exact_match(self, monkeypatch):
        self._patch_client(monkeypatch, [_make_ws("my-ws", "ws-async-1")])

        result = asyncio.run(resolve_workspace_id_by_name_async("my-ws"))
        assert result == "ws-async-1"

    def test_resolve_async_not_found(self, monkeypatch):
        self._patch_client(monkeypatch, [])
        with pytest.raises(ResourceNotExistError):
            asyncio.run(resolve_workspace_id_by_name_async("absent"))

    def test_resolve_async_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(resolve_workspace_id_by_name_async(""))

    def test_resolve_async_uses_cache(self, monkeypatch):
        client = self._patch_client(monkeypatch, [_make_ws("my-ws", "ws-123")])

        async def go():
            await resolve_workspace_id_by_name_async("my-ws")
            await resolve_workspace_id_by_name_async("my-ws")

        asyncio.run(go())
        assert client.list_workspaces_async.await_count == 1

    def test_resolve_async_ambiguous(self, monkeypatch):
        self._patch_client(
            monkeypatch,
            [_make_ws("dup", "ws-1"), _make_ws("dup", "ws-2")],
        )
        with pytest.raises(ValueError, match="ambiguous"):
            asyncio.run(resolve_workspace_id_by_name_async("dup"))

    def test_resolve_async_client_exception(self, monkeypatch):
        from alibabacloud_tea_openapi.exceptions._client import ClientException

        from agentrun.utils.exception import ClientError

        client = MagicMock()
        client.list_workspaces_async = AsyncMock(
            side_effect=ClientException(
                status_code=403, data={"message": "denied"}
            )
        )
        monkeypatch.setattr(
            ws_mod._WorkspaceResolver,
            "_get_client",
            lambda self, config=None: client,
        )
        with pytest.raises(ClientError):
            asyncio.run(resolve_workspace_id_by_name_async("my-ws"))

    def test_resolve_async_server_exception(self, monkeypatch):
        from alibabacloud_tea_openapi.exceptions._server import ServerException

        from agentrun.utils.exception import ServerError

        client = MagicMock()
        client.list_workspaces_async = AsyncMock(
            side_effect=ServerException(
                status_code=500, data={"message": "boom"}
            )
        )
        monkeypatch.setattr(
            ws_mod._WorkspaceResolver,
            "_get_client",
            lambda self, config=None: client,
        )
        with pytest.raises(ServerError):
            asyncio.run(resolve_workspace_id_by_name_async("my-ws"))

    def test_resolve_many_sync(self, monkeypatch):
        # 模拟两次连续调用返回不同的 workspace
        client = MagicMock()
        client.list_workspaces.side_effect = [
            _make_response([_make_ws("a", "ws-a")]),
            _make_response([_make_ws("b", "ws-b")]),
        ]
        monkeypatch.setattr(
            ws_mod._WorkspaceResolver,
            "_get_client",
            lambda self, config=None: client,
        )
        assert resolve_workspace_ids_by_names("a, b") == "ws-a,ws-b"

    def test_resolve_many_sync_strips_empty(self, monkeypatch):
        # 空字符串 / 仅有逗号空格 应被忽略
        self._patch_client(monkeypatch, [_make_ws("a", "ws-a")])
        assert resolve_workspace_ids_by_names("a,  ,") == "ws-a"

    def test_resolve_many_async(self, monkeypatch):
        client = MagicMock()
        client.list_workspaces_async = AsyncMock(
            side_effect=[
                _make_response([_make_ws("a", "ws-a")]),
                _make_response([_make_ws("b", "ws-b")]),
            ]
        )
        monkeypatch.setattr(
            ws_mod._WorkspaceResolver,
            "_get_client",
            lambda self, config=None: client,
        )
        result = asyncio.run(resolve_workspace_ids_by_names_async("a,b"))
        assert result == "ws-a,ws-b"


class TestClientCreateWorkspaceResolution:
    """create 集成测试：workspace_name 自动转换为 workspace_id"""

    @patch(CONTROL_API_PATH)
    @patch(
        "agentrun.agent_runtime.client.resolve_workspace_id_by_name",
        return_value="ws-resolved",
    )
    def test_create_with_workspace_name_resolves(
        self, mock_resolve, mock_control_api_class
    ):
        from agentrun.agent_runtime.client import AgentRuntimeClient

        # control_api 不在路径中暴露需要的真实 client，所以仅 mock 其 create 返回值
        mock_control_api = MagicMock()
        mock_data = MagicMock()
        mock_data.agent_runtime_id = "ar-1"
        mock_data.to_map.return_value = {
            "agentRuntimeId": "ar-1",
            "agentRuntimeName": "n",
            "status": "READY",
        }
        mock_control_api.create_agent_runtime.return_value = mock_data
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        inp = AgentRuntimeCreateInput(
            agent_runtime_name="n",
            workspace_name="my-ws",
            container_configuration=AgentRuntimeContainer(
                image="img", port=9000
            ),
        )
        client.create(inp)

        mock_resolve.assert_called_once_with("my-ws", None)
        # 调用 API 前，应把 workspace_name 清空且 workspace_id 写好
        assert inp.workspace_id == "ws-resolved"
        assert inp.workspace_name is None

    @patch(CONTROL_API_PATH)
    def test_create_rejects_both_workspace_fields(self, mock_control_api_class):
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api_class.return_value = MagicMock()
        client = AgentRuntimeClient()
        inp = AgentRuntimeCreateInput(
            agent_runtime_name="n",
            workspace_id="ws-1",
            workspace_name="my-ws",
            container_configuration=AgentRuntimeContainer(
                image="img", port=9000
            ),
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            client.create(inp)

    @patch(CONTROL_API_PATH)
    @patch(
        "agentrun.agent_runtime.client.resolve_workspace_id_by_name_async",
        new_callable=AsyncMock,
    )
    def test_create_async_resolves_workspace_name(
        self, mock_resolve_async, mock_control_api_class
    ):
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_resolve_async.return_value = "ws-resolved-async"
        mock_control_api = MagicMock()
        mock_data = MagicMock()
        mock_data.agent_runtime_id = "ar-1"
        mock_data.to_map.return_value = {
            "agentRuntimeId": "ar-1",
            "agentRuntimeName": "n",
            "status": "READY",
        }
        mock_control_api.create_agent_runtime_async = AsyncMock(
            return_value=mock_data
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        inp = AgentRuntimeCreateInput(
            agent_runtime_name="n",
            workspace_name="my-ws",
            container_configuration=AgentRuntimeContainer(
                image="img", port=9000
            ),
        )
        asyncio.run(client.create_async(inp))

        mock_resolve_async.assert_awaited_once_with("my-ws", None)
        assert inp.workspace_id == "ws-resolved-async"
        assert inp.workspace_name is None


class TestClientListWorkspaceResolution:
    """list 集成测试：workspace_name / workspace_names 自动转换"""

    @patch(CONTROL_API_PATH)
    @patch(
        "agentrun.agent_runtime.client.resolve_workspace_id_by_name",
        return_value="ws-1",
    )
    @patch(
        "agentrun.agent_runtime.client.resolve_workspace_ids_by_names",
        return_value="ws-1,ws-2",
    )
    def test_list_resolves_workspace_name_and_names(
        self,
        mock_resolve_many,
        mock_resolve_one,
        mock_control_api_class,
    ):
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api = MagicMock()
        mock_result = MagicMock()
        mock_result.items = []
        mock_control_api.list_agent_runtimes.return_value = mock_result
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        inp = AgentRuntimeListInput(
            workspace_name="my-ws",
            workspace_names="ws-a,ws-b",
        )
        client.list(inp)

        mock_resolve_one.assert_called_once_with("my-ws", None)
        mock_resolve_many.assert_called_once_with("ws-a,ws-b", None)
        assert inp.workspace_id == "ws-1"
        assert inp.workspace_ids == "ws-1,ws-2"
        assert inp.workspace_name is None
        assert inp.workspace_names is None

    @patch(CONTROL_API_PATH)
    def test_list_rejects_both_singular(self, mock_control_api_class):
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api_class.return_value = MagicMock()
        client = AgentRuntimeClient()
        inp = AgentRuntimeListInput(
            workspace_id="ws-1",
            workspace_name="my-ws",
        )
        with pytest.raises(ValueError, match="workspace_id and workspace_name"):
            client.list(inp)

    @patch(CONTROL_API_PATH)
    def test_list_rejects_both_plural(self, mock_control_api_class):
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_control_api_class.return_value = MagicMock()
        client = AgentRuntimeClient()
        inp = AgentRuntimeListInput(
            workspace_ids="ws-1,ws-2",
            workspace_names="a,b",
        )
        with pytest.raises(
            ValueError, match="workspace_ids and workspace_names"
        ):
            client.list(inp)

    @patch(CONTROL_API_PATH)
    @patch(
        "agentrun.agent_runtime.client.resolve_workspace_id_by_name_async",
        new_callable=AsyncMock,
    )
    @patch(
        "agentrun.agent_runtime.client.resolve_workspace_ids_by_names_async",
        new_callable=AsyncMock,
    )
    def test_list_async_resolves(
        self,
        mock_resolve_many_async,
        mock_resolve_one_async,
        mock_control_api_class,
    ):
        from agentrun.agent_runtime.client import AgentRuntimeClient

        mock_resolve_one_async.return_value = "ws-1"
        mock_resolve_many_async.return_value = "ws-1,ws-2"
        mock_control_api = MagicMock()
        mock_result = MagicMock()
        mock_result.items = []
        mock_control_api.list_agent_runtimes_async = AsyncMock(
            return_value=mock_result
        )
        mock_control_api_class.return_value = mock_control_api

        client = AgentRuntimeClient()
        inp = AgentRuntimeListInput(
            workspace_name="my-ws",
            workspace_names="ws-a,ws-b",
        )
        asyncio.run(client.list_async(inp))

        mock_resolve_one_async.assert_awaited_once_with("my-ws", None)
        mock_resolve_many_async.assert_awaited_once_with("ws-a,ws-b", None)
        assert inp.workspace_id == "ws-1"
        assert inp.workspace_ids == "ws-1,ws-2"
