"""测试 agentrun.utils.data_api 模块 / Test agentrun.utils.data_api module"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from agentrun.utils.config import Config
from agentrun.utils.data_api import DataAPI, ResourceType
from agentrun.utils.exception import ClientError


class TestResourceType:
    """测试 ResourceType 枚举"""

    def test_runtime(self):
        assert ResourceType.Runtime.value == "runtime"

    def test_litellm(self):
        assert ResourceType.LiteLLM.value == "litellm"

    def test_tool(self):
        assert ResourceType.Tool.value == "tool"

    def test_template(self):
        assert ResourceType.Template.value == "template"

    def test_sandbox(self):
        assert ResourceType.Sandbox.value == "sandbox"


class TestDataAPIInit:
    """测试 DataAPI 初始化"""

    def test_init_basic(self):
        """测试基本初始化"""
        with patch.dict(
            os.environ, {"AGENTRUN_ACCOUNT_ID": "test-account"}, clear=True
        ):
            api = DataAPI(
                resource_name="test-resource",
                resource_type=ResourceType.Runtime,
            )
            assert api.resource_name == "test-resource"
            assert api.resource_type == ResourceType.Runtime
            assert api.namespace == "agents"

    def test_init_with_custom_namespace(self):
        """测试自定义 namespace"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test-resource",
            resource_type=ResourceType.Runtime,
            config=config,
            namespace="custom",
        )
        assert api.namespace == "custom"


class TestDataAPIGetBaseUrl:
    """测试 DataAPI.get_base_url"""

    def test_get_base_url(self):
        """测试获取基础 URL"""
        config = Config(
            account_id="test-account",
            data_endpoint="https://custom-data.example.com",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )
        assert api.get_base_url() == "https://custom-data.example.com"


class TestDataAPIWithPath:
    """测试 DataAPI.with_path"""

    def test_simple_path(self):
        """测试简单路径"""
        config = Config(
            account_id="test-account",
            data_endpoint="https://example.com",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )
        result = api.with_path("resources")
        assert result == "https://example.com/agents/resources"

    def test_path_with_leading_slash(self):
        """测试带前导斜杠的路径"""
        config = Config(
            account_id="test-account",
            data_endpoint="https://example.com",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )
        result = api.with_path("/resources")
        assert result == "https://example.com/agents/resources"

    def test_path_with_query(self):
        """测试带查询参数的路径"""
        config = Config(
            account_id="test-account",
            data_endpoint="https://example.com",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )
        result = api.with_path("resources", query={"limit": 10})
        assert "limit=10" in result

    def test_path_with_existing_query(self):
        """测试已有查询参数的路径"""
        config = Config(
            account_id="test-account",
            data_endpoint="https://example.com",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )
        result = api.with_path("resources?page=1", query={"limit": 10})
        assert "page=1" in result
        assert "limit=10" in result

    def test_path_with_list_query_value(self):
        """测试列表类型的查询参数值"""
        config = Config(
            account_id="test-account",
            data_endpoint="https://example.com",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )
        result = api.with_path("resources", query={"ids": ["a", "b"]})
        assert "ids=a" in result
        assert "ids=b" in result


class TestDataAPIAuth:
    """测试 DataAPI.auth（仅 RAM 签名鉴权）"""

    def test_auth_without_ak_sk_returns_no_auth_header(self):
        """无 AK/SK 时 auth 不添加鉴权头"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )
        url, headers, query = api.auth("https://example.com", {}, None)
        assert "Agentrun-Access-Token" not in headers
        assert "Agentrun-Authorization" not in headers

    @patch("agentrun.utils.data_api.get_agentrun_signed_headers")
    def test_auth_uses_ram_signature_when_ak_sk_provided(
        self, mock_signed_headers
    ):
        """测试配置了 AK/SK 且无 token 时使用 RAM 签名鉴权"""
        mock_signed_headers.return_value = {
            "Agentrun-Authorization": "mock-sig",
            "x-acs-date": "2025-01-01T00:00:00Z",
            "x-acs-content-sha256": "UNSIGNED-PAYLOAD",
        }
        config = Config(
            access_key_id="ak",
            access_key_secret="sk",
            account_id="test-account",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        url = "https://test-account-ram.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        url, headers, query = api.auth(url, {}, None, method="GET", body=None)
        assert "Agentrun-Authorization" in headers
        assert headers["Agentrun-Authorization"] == "mock-sig"
        assert "x-acs-date" in headers
        assert "x-acs-content-sha256" in headers
        assert headers.get("x-acs-content-sha256") == "UNSIGNED-PAYLOAD"

    @patch("agentrun.utils.data_api.get_agentrun_signed_headers")
    def test_auth_with_ak_sk_returns_signed_headers(self, mock_signed_headers):
        """测试有 AK/SK 时 auth 返回签名头且不抛异常"""
        mock_signed_headers.return_value = {
            "Agentrun-Authorization": "mock-sig",
            "x-acs-date": "2025-01-01T00:00:00Z",
            "x-acs-content-sha256": "UNSIGNED-PAYLOAD",
        }
        config = Config(
            access_key_id="ak",
            access_key_secret="sk",
            account_id="test-account",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        url, headers, query = api.auth(
            "https://test-account-ram.agentrun-data.cn-hangzhou.aliyuncs.com/path",
            {},
            None,
            method="GET",
        )
        assert "Agentrun-Authorization" in headers
        assert headers["Agentrun-Authorization"] == "mock-sig"


class TestDataAPIPrepareRequest:
    """测试 DataAPI._prepare_request"""

    def test_prepare_request_with_dict_data(self):
        """测试使用字典数据准备请求"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )
        method, url, headers, json_data, content = api._prepare_request(
            "POST", "https://example.com", data={"key": "value"}
        )
        assert method == "POST"
        assert json_data == {"key": "value"}
        assert content is None

    def test_prepare_request_with_string_data(self):
        """测试使用字符串数据准备请求"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )
        method, url, headers, json_data, content = api._prepare_request(
            "POST", "https://example.com", data="raw string"
        )
        assert json_data is None
        assert content == "raw string"

    def test_prepare_request_with_query(self):
        """测试带查询参数的请求准备"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )
        method, url, headers, json_data, content = api._prepare_request(
            "GET", "https://example.com", query={"page": 1}
        )
        assert "page=1" in url

    def test_prepare_request_with_list_query(self):
        """测试带多值列表查询参数的请求准备"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )
        method, url, headers, json_data, content = api._prepare_request(
            "GET", "https://example.com", query={"ids": ["a", "b", "c"]}
        )
        # 验证多值列表被正确编码
        assert "ids=a" in url
        assert "ids=b" in url
        assert "ids=c" in url

    def test_prepare_request_with_non_standard_data(self):
        """测试使用非 dict/str 类型数据准备请求"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )
        # 使用数字作为数据，应该被转换为字符串
        method, url, headers, json_data, content = api._prepare_request(
            "POST", "https://example.com", data=12345
        )
        assert json_data is None
        assert content == "12345"


class TestDataAPIHTTPMethods:
    """测试 DataAPI 的 HTTP 方法"""

    @respx.mock
    def test_get(self):
        """测试 GET 请求"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(return_value=httpx.Response(200, json={"data": "value"}))

        result = api.get("resources")
        assert result == {"data": "value"}

    @respx.mock
    def test_post(self):
        """测试 POST 请求"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.post(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(return_value=httpx.Response(200, json={"id": "new-id"}))

        result = api.post("resources", data={"name": "test"})
        assert result == {"id": "new-id"}

    @respx.mock
    def test_put(self):
        """测试 PUT 请求"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.put(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources/1"
        ).mock(return_value=httpx.Response(200, json={"updated": True}))

        result = api.put("resources/1", data={"name": "updated"})
        assert result == {"updated": True}

    @respx.mock
    def test_patch(self):
        """测试 PATCH 请求"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.patch(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources/1"
        ).mock(return_value=httpx.Response(200, json={"patched": True}))

        result = api.patch("resources/1", data={"field": "value"})
        assert result == {"patched": True}

    @respx.mock
    def test_delete(self):
        """测试 DELETE 请求"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.delete(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources/1"
        ).mock(return_value=httpx.Response(200, json={"deleted": True}))

        result = api.delete("resources/1")
        assert result == {"deleted": True}

    @respx.mock
    def test_empty_response(self):
        """测试空响应"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(return_value=httpx.Response(204, text=""))

        result = api.get("resources")
        assert result == {}

    @respx.mock
    def test_bad_gateway_error(self):
        """测试 502 Bad Gateway 错误"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(
            return_value=httpx.Response(
                502, text="<html>502 Bad Gateway</html>"
            )
        )

        with pytest.raises(ClientError) as exc_info:
            api.get("resources")
        assert exc_info.value.status_code == 502

    @respx.mock
    def test_json_parse_error(self):
        """测试 JSON 解析错误"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(return_value=httpx.Response(200, text="not valid json"))

        with pytest.raises(ClientError) as exc_info:
            api.get("resources")
        assert "Failed to parse JSON" in exc_info.value.message

    @respx.mock
    def test_request_error(self):
        """测试请求错误"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(side_effect=httpx.RequestError("Connection failed"))

        with pytest.raises(ClientError) as exc_info:
            api.get("resources")
        assert exc_info.value.status_code == 0


class TestDataAPIAsyncMethods:
    """测试 DataAPI 的异步 HTTP 方法"""

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_async(self):
        """测试异步 GET 请求"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(return_value=httpx.Response(200, json={"data": "value"}))

        result = await api.get_async("resources")
        assert result == {"data": "value"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_post_async(self):
        """测试异步 POST 请求"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.post(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(return_value=httpx.Response(200, json={"id": "new-id"}))

        result = await api.post_async("resources", data={"name": "test"})
        assert result == {"id": "new-id"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_put_async(self):
        """测试异步 PUT 请求"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.put(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources/1"
        ).mock(return_value=httpx.Response(200, json={"updated": True}))

        result = await api.put_async("resources/1", data={"name": "updated"})
        assert result == {"updated": True}

    @respx.mock
    @pytest.mark.asyncio
    async def test_patch_async(self):
        """测试异步 PATCH 请求"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.patch(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources/1"
        ).mock(return_value=httpx.Response(200, json={"patched": True}))

        result = await api.patch_async("resources/1", data={"field": "value"})
        assert result == {"patched": True}

    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_async(self):
        """测试异步 DELETE 请求"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.delete(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources/1"
        ).mock(return_value=httpx.Response(200, json={"deleted": True}))

        result = await api.delete_async("resources/1")
        assert result == {"deleted": True}

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_empty_response(self):
        """测试异步空响应"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(return_value=httpx.Response(204, text=""))

        result = await api.get_async("resources")
        assert result == {}

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_request_error(self):
        """测试异步请求错误"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(side_effect=httpx.RequestError("Connection failed"))

        with pytest.raises(ClientError) as exc_info:
            await api.get_async("resources")
        assert exc_info.value.status_code == 0


class TestDataAPIFileOperations:
    """测试 DataAPI 的文件操作方法"""

    @respx.mock
    def test_post_file(self):
        """测试同步上传文件"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.post(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/files"
        ).mock(return_value=httpx.Response(200, json={"uploaded": True}))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            result = api.post_file("files", temp_path, "/remote/file.txt")
            assert result == {"uploaded": True}
        finally:
            os.unlink(temp_path)

    @respx.mock
    @pytest.mark.asyncio
    async def test_post_file_async(self):
        """测试异步上传文件"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.post(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/files"
        ).mock(return_value=httpx.Response(200, json={"uploaded": True}))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            result = await api.post_file_async(
                "files", temp_path, "/remote/file.txt"
            )
            assert result == {"uploaded": True}
        finally:
            os.unlink(temp_path)

    @respx.mock
    def test_get_file(self):
        """测试同步下载文件"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/files"
        ).mock(return_value=httpx.Response(200, content=b"file content here"))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            temp_path = f.name

        try:
            result = api.get_file(
                "files", temp_path, query={"path": "/remote/file.txt"}
            )
            assert result["saved_path"] == temp_path
            assert result["size"] == len(b"file content here")

            with open(temp_path, "rb") as f:
                assert f.read() == b"file content here"
        finally:
            os.unlink(temp_path)

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_file_async(self):
        """测试异步下载文件"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/files"
        ).mock(return_value=httpx.Response(200, content=b"async file content"))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            temp_path = f.name

        try:
            result = await api.get_file_async(
                "files", temp_path, query={"path": "/remote/file.txt"}
            )
            assert result["saved_path"] == temp_path
            assert result["size"] == len(b"async file content")
        finally:
            os.unlink(temp_path)

    @respx.mock
    def test_get_video(self):
        """测试同步下载视频"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/videos"
        ).mock(return_value=httpx.Response(200, content=b"video binary data"))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mkv") as f:
            temp_path = f.name

        try:
            result = api.get_video("videos", temp_path)
            assert result["saved_path"] == temp_path
            assert result["size"] == len(b"video binary data")
        finally:
            os.unlink(temp_path)

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_video_async(self):
        """测试异步下载视频"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/videos"
        ).mock(return_value=httpx.Response(200, content=b"async video data"))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mkv") as f:
            temp_path = f.name

        try:
            result = await api.get_video_async("videos", temp_path)
            assert result["saved_path"] == temp_path
            assert result["size"] == len(b"async video data")
        finally:
            os.unlink(temp_path)

    @respx.mock
    def test_post_file_http_error(self):
        """测试上传文件时的 HTTP 错误"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.post(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/files"
        ).mock(return_value=httpx.Response(500, text="Server Error"))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            with pytest.raises(ClientError):
                api.post_file("files", temp_path, "/remote/file.txt")
        finally:
            os.unlink(temp_path)

    @respx.mock
    def test_get_file_http_error(self):
        """测试下载文件时的 HTTP 错误"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/files"
        ).mock(return_value=httpx.Response(404, text="Not Found"))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            temp_path = f.name

        try:
            with pytest.raises(ClientError):
                api.get_file("files", temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDataAPIHTTPStatusError:
    """测试 DataAPI 的 HTTPStatusError 处理"""

    @respx.mock
    def test_http_status_error_with_response_text(self):
        """测试 HTTPStatusError 带响应文本"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        # 创建一个模拟的 HTTPStatusError
        mock_response = httpx.Response(
            status_code=400,
            text="Bad Request Error",
            request=httpx.Request("GET", "https://example.com"),
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(
            side_effect=httpx.HTTPStatusError(
                "Error", request=mock_response.request, response=mock_response
            )
        )

        with pytest.raises(ClientError) as exc_info:
            api.get("resources")
        assert exc_info.value.status_code == 400

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_http_status_error(self):
        """测试异步 HTTPStatusError"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        mock_response = httpx.Response(
            status_code=403,
            text="Forbidden",
            request=httpx.Request("GET", "https://example.com"),
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(
            side_effect=httpx.HTTPStatusError(
                "Error", request=mock_response.request, response=mock_response
            )
        )

        with pytest.raises(ClientError) as exc_info:
            await api.get_async("resources")
        assert exc_info.value.status_code == 403

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_bad_gateway_error(self):
        """测试异步 502 Bad Gateway 错误"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(
            return_value=httpx.Response(
                502, text="<html>502 Bad Gateway</html>"
            )
        )

        with pytest.raises(ClientError) as exc_info:
            await api.get_async("resources")
        assert exc_info.value.status_code == 502

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_json_parse_error(self):
        """测试异步 JSON 解析错误"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/resources"
        ).mock(return_value=httpx.Response(200, text="not valid json"))

        with pytest.raises(ClientError) as exc_info:
            await api.get_async("resources")
        assert "Failed to parse JSON" in exc_info.value.message


class TestDataAPIFileOperationsErrors:
    """测试 DataAPI 文件操作的错误处理"""

    @respx.mock
    @pytest.mark.asyncio
    async def test_post_file_async_http_error(self):
        """测试异步上传文件时的 HTTP 错误"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.post(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/files"
        ).mock(return_value=httpx.Response(500, text="Server Error"))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            with pytest.raises(ClientError):
                await api.post_file_async(
                    "files", temp_path, "/remote/file.txt"
                )
        finally:
            os.unlink(temp_path)

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_file_async_http_error(self):
        """测试异步下载文件时的 HTTP 错误"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/files"
        ).mock(return_value=httpx.Response(404, text="Not Found"))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            temp_path = f.name

        try:
            with pytest.raises(ClientError):
                await api.get_file_async("files", temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @respx.mock
    def test_get_video_http_error(self):
        """测试同步下载视频时的 HTTP 错误"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/videos"
        ).mock(return_value=httpx.Response(404, text="Not Found"))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mkv") as f:
            temp_path = f.name

        try:
            with pytest.raises(ClientError):
                api.get_video("videos", temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_video_async_http_error(self):
        """测试异步下载视频时的 HTTP 错误"""
        config = Config(
            account_id="test-account",
            access_key_id="",
            access_key_secret="",
        )
        api = DataAPI(
            resource_name="test",
            resource_type=ResourceType.Runtime,
            config=config,
        )

        respx.get(
            "https://test-account.agentrun-data.cn-hangzhou.aliyuncs.com/agents/videos"
        ).mock(return_value=httpx.Response(404, text="Not Found"))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mkv") as f:
            temp_path = f.name

        try:
            with pytest.raises(ClientError):
                await api.get_video_async("videos", temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDataAPIAuthWithSandbox:
    """测试 DataAPI 针对 Sandbox 资源类型的认证（RAM 鉴权下与其它资源类型一致）"""

    @patch("agentrun.utils.data_api.get_agentrun_signed_headers")
    def test_auth_with_sandbox_uses_ram_when_ak_sk_provided(
        self, mock_signed_headers
    ):
        """测试 Sandbox 资源类型在配置 AK/SK 时同样使用 RAM 签名"""
        mock_signed_headers.return_value = {
            "Agentrun-Authorization": "mock-sig",
            "x-acs-date": "2025-01-01T00:00:00Z",
            "x-acs-content-sha256": "UNSIGNED-PAYLOAD",
        }
        config = Config(
            access_key_id="ak",
            access_key_secret="sk",
            account_id="test-account",
        )
        api = DataAPI(
            resource_name="sandbox-123",
            resource_type=ResourceType.Sandbox,
            config=config,
        )

        url, headers, query = api.auth(
            "https://test-account-ram.agentrun-data.cn-hangzhou.aliyuncs.com/sandboxes/sandbox-123/health",
            {},
            None,
            method="GET",
        )
        assert "Agentrun-Authorization" in headers
        assert (
            api.get_base_url().startswith("https://")
            and "-ram." in api.get_base_url()
        )


class TestDataAPIConfigForwarding:
    """测试 DataAPI 的 config 透传到 with_path 的行为（修复 issue #75）

    当 DataAPI 实例初始化时没有提供 account_id（如通过 __get_client() 获取的无配置客户端），
    但在调用 HTTP 方法时通过 config 参数传入 account_id，URL 应该使用该 account_id 构造。
    """

    @respx.mock
    def test_get_uses_call_time_config_for_url(self):
        """测试 get() 使用调用时传入的 config 构造 URL（不依赖实例 config 的 account_id）"""
        # 实例创建时不提供 account_id（模拟 __get_client() 无配置的情况）
        api = DataAPI(
            resource_name="",
            resource_type=ResourceType.Template,
            config=None,
            namespace="sandboxes",
        )

        # 调用时传入带有 account_id 的 config
        call_config = Config(
            account_id="call-time-account",
            access_key_id="",
            access_key_secret="",
            region_id="cn-hangzhou",
        )

        respx.get(
            "https://call-time-account.agentrun-data.cn-hangzhou.aliyuncs.com/sandboxes"
        ).mock(return_value=httpx.Response(200, json={"code": "SUCCESS"}))

        result = api.get("/", config=call_config)
        assert result == {"code": "SUCCESS"}

    @respx.mock
    def test_post_uses_call_time_config_for_url(self):
        """测试 post() 使用调用时传入的 config 构造 URL"""
        api = DataAPI(
            resource_name="",
            resource_type=ResourceType.Template,
            config=None,
            namespace="sandboxes",
        )

        call_config = Config(
            account_id="call-time-account",
            access_key_id="",
            access_key_secret="",
            region_id="cn-hangzhou",
        )

        respx.post(
            "https://call-time-account.agentrun-data.cn-hangzhou.aliyuncs.com/sandboxes"
        ).mock(return_value=httpx.Response(200, json={"code": "SUCCESS"}))

        result = api.post("/", data={"key": "val"}, config=call_config)
        assert result == {"code": "SUCCESS"}

    @respx.mock
    def test_delete_uses_call_time_config_for_url(self):
        """测试 delete() 使用调用时传入的 config 构造 URL"""
        api = DataAPI(
            resource_name="",
            resource_type=ResourceType.Template,
            config=None,
            namespace="sandboxes",
        )

        call_config = Config(
            account_id="call-time-account",
            access_key_id="",
            access_key_secret="",
            region_id="cn-hangzhou",
        )

        respx.delete(
            "https://call-time-account.agentrun-data.cn-hangzhou.aliyuncs.com/sandboxes"
        ).mock(return_value=httpx.Response(200, json={"code": "SUCCESS"}))

        result = api.delete("/", config=call_config)
        assert result == {"code": "SUCCESS"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_post_async_uses_call_time_config_for_url(self):
        """测试 post_async() 使用调用时传入的 config 构造 URL"""
        api = DataAPI(
            resource_name="",
            resource_type=ResourceType.Template,
            config=None,
            namespace="sandboxes",
        )

        call_config = Config(
            account_id="call-time-account",
            access_key_id="",
            access_key_secret="",
            region_id="cn-hangzhou",
        )

        respx.post(
            "https://call-time-account.agentrun-data.cn-hangzhou.aliyuncs.com/sandboxes"
        ).mock(return_value=httpx.Response(200, json={"code": "SUCCESS"}))

        result = await api.post_async("/", data={"key": "val"}, config=call_config)
        assert result == {"code": "SUCCESS"}

    def test_no_account_id_without_call_time_config_raises(self):
        """测试未在实例 config 或调用 config 中设置 account_id 时抛出 ValueError"""
        api = DataAPI(
            resource_name="",
            resource_type=ResourceType.Template,
            config=None,
            namespace="sandboxes",
        )

        with pytest.raises(ValueError, match="account id is not set"):
            api.with_path("/")
