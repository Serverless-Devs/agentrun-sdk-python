"""测试 agentrun.knowledgebase.api.data 模块 / Test agentrun.knowledgebase.api.data module"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.knowledgebase.api.data import (
    ADBDataAPI,
    BailianDataAPI,
    get_data_api,
    KnowledgeBaseDataAPI,
    RagFlowDataAPI,
)
from agentrun.knowledgebase.model import (
    ADBProviderSettings,
    ADBRetrieveSettings,
    BailianProviderSettings,
    BailianRetrieveSettings,
    KnowledgeBaseProvider,
    RagFlowProviderSettings,
    RagFlowRetrieveSettings,
)
from agentrun.utils.config import Config


class TestKnowledgeBaseDataAPIBase:
    """测试 KnowledgeBaseDataAPI 基类"""

    def test_abstract_methods(self):
        """测试抽象方法"""
        # KnowledgeBaseDataAPI 是抽象类，不能直接实例化
        with pytest.raises(TypeError):
            KnowledgeBaseDataAPI("test-kb")  # type: ignore


class TestRagFlowDataAPIInit:
    """测试 RagFlowDataAPI 初始化"""

    def test_init(self):
        """测试初始化"""
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            retrieve_settings=RagFlowRetrieveSettings(
                similarity_threshold=0.8,
            ),
            credential_name="test-credential",
        )

        assert api.knowledge_base_name == "test-kb"
        assert api.provider_settings is not None
        assert api.retrieve_settings is not None
        assert api.credential_name == "test-credential"

    def test_init_minimal(self):
        """测试最小化初始化"""
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
        )

        assert api.knowledge_base_name == "test-kb"
        assert api.provider_settings is None
        assert api.retrieve_settings is None
        assert api.credential_name is None

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = Config(access_key_id="test-ak")
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            config=config,
        )

        assert api.knowledge_base_name == "test-kb"


class TestRagFlowDataAPIGetApiKey:
    """测试 RagFlowDataAPI._get_api_key 方法"""

    @patch("agentrun.credential.Credential.get_by_name")
    def test_get_api_key_sync(self, mock_get_credential):
        """测试同步获取 API Key"""
        mock_credential = MagicMock()
        mock_credential.credential_secret = "test-api-key"
        mock_get_credential.return_value = mock_credential

        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            credential_name="test-credential",
        )

        result = api._get_api_key()
        assert result == "test-api-key"

    @patch("agentrun.credential.Credential.get_by_name_async")
    @pytest.mark.asyncio
    async def test_get_api_key_async(self, mock_get_credential):
        """测试异步获取 API Key"""
        mock_credential = MagicMock()
        mock_credential.credential_secret = "test-api-key"
        mock_get_credential.return_value = mock_credential

        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            credential_name="test-credential",
        )

        result = await api._get_api_key_async()
        assert result == "test-api-key"

    def test_get_api_key_without_credential_name(self):
        """测试无凭证名称时获取 API Key"""
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
        )

        with pytest.raises(ValueError, match="credential_name is required"):
            api._get_api_key()

    @pytest.mark.asyncio
    async def test_get_api_key_async_without_credential_name(self):
        """测试异步无凭证名称时获取 API Key"""
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
        )

        with pytest.raises(ValueError, match="credential_name is required"):
            await api._get_api_key_async()

    @patch("agentrun.credential.Credential.get_by_name")
    def test_get_api_key_empty_secret(self, mock_get_credential):
        """测试凭证密钥为空时获取 API Key"""
        mock_credential = MagicMock()
        mock_credential.credential_secret = None
        mock_get_credential.return_value = mock_credential

        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            credential_name="test-credential",
        )

        with pytest.raises(ValueError, match="has no secret configured"):
            api._get_api_key()

    @patch("agentrun.credential.Credential.get_by_name_async")
    @pytest.mark.asyncio
    async def test_get_api_key_async_empty_secret(self, mock_get_credential):
        """测试异步凭证密钥为空时获取 API Key"""
        mock_credential = MagicMock()
        mock_credential.credential_secret = None
        mock_get_credential.return_value = mock_credential

        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            credential_name="test-credential",
        )

        with pytest.raises(ValueError, match="has no secret configured"):
            await api._get_api_key_async()


class TestRagFlowDataAPIBuildRequestBody:
    """测试 RagFlowDataAPI._build_request_body 方法"""

    def test_build_request_body(self):
        """测试构建请求体"""
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1", "ds-2"],
            ),
        )

        body = api._build_request_body("test query")
        assert body["question"] == "test query"
        assert body["dataset_ids"] == ["ds-1", "ds-2"]
        assert body["page"] == 1
        assert body["page_size"] == 30

    def test_build_request_body_with_retrieve_settings(self):
        """测试带检索设置构建请求体"""
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            retrieve_settings=RagFlowRetrieveSettings(
                similarity_threshold=0.8,
                vector_similarity_weight=0.5,
                cross_languages=["English", "Chinese"],
            ),
        )

        body = api._build_request_body("test query")
        assert body["similarity_threshold"] == 0.8
        assert body["vector_similarity_weight"] == 0.5
        assert body["cross_languages"] == ["English", "Chinese"]

    def test_build_request_body_without_provider_settings(self):
        """测试无提供商设置构建请求体"""
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
        )

        with pytest.raises(ValueError, match="provider_settings is required"):
            api._build_request_body("test query")

    def test_build_request_body_with_partial_retrieve_settings_only_threshold(
        self,
    ):
        """测试仅设置 similarity_threshold 的部分检索设置"""
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            retrieve_settings=RagFlowRetrieveSettings(
                similarity_threshold=0.8,
            ),
        )

        body = api._build_request_body("test query")
        assert body["similarity_threshold"] == 0.8
        assert "vector_similarity_weight" not in body
        assert "cross_languages" not in body

    def test_build_request_body_with_partial_retrieve_settings_only_weight(
        self,
    ):
        """测试仅设置 vector_similarity_weight 的部分检索设置"""
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            retrieve_settings=RagFlowRetrieveSettings(
                vector_similarity_weight=0.5,
            ),
        )

        body = api._build_request_body("test query")
        assert "similarity_threshold" not in body
        assert body["vector_similarity_weight"] == 0.5
        assert "cross_languages" not in body

    def test_build_request_body_with_partial_retrieve_settings_only_languages(
        self,
    ):
        """测试仅设置 cross_languages 的部分检索设置"""
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            retrieve_settings=RagFlowRetrieveSettings(
                cross_languages=["English"],
            ),
        )

        body = api._build_request_body("test query")
        assert "similarity_threshold" not in body
        assert "vector_similarity_weight" not in body
        assert body["cross_languages"] == ["English"]

    def test_build_request_body_without_retrieve_settings(self):
        """测试无检索设置构建请求体"""
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
        )

        body = api._build_request_body("test query")
        assert "similarity_threshold" not in body
        assert "vector_similarity_weight" not in body
        assert "cross_languages" not in body


class TestRagFlowDataAPIRetrieve:
    """测试 RagFlowDataAPI.retrieve 方法"""

    @patch("httpx.Client")
    @patch("agentrun.credential.Credential.get_by_name")
    def test_retrieve_sync(self, mock_get_credential, mock_httpx_client):
        """测试同步检索"""
        mock_credential = MagicMock()
        mock_credential.credential_secret = "test-api-key"
        mock_get_credential.return_value = mock_credential

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {"chunks": [{"content": "test content"}]}
        }
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(
            return_value=mock_client_instance
        )
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx_client.return_value = mock_client_instance

        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        result = api.retrieve("test query")
        assert result["query"] == "test query"
        assert result["knowledge_base_name"] == "test-kb"
        assert "data" in result

    @patch("httpx.AsyncClient")
    @patch("agentrun.credential.Credential.get_by_name_async")
    @pytest.mark.asyncio
    async def test_retrieve_async(self, mock_get_credential, mock_httpx_client):
        """测试异步检索"""
        mock_credential = MagicMock()
        mock_credential.credential_secret = "test-api-key"
        mock_get_credential.return_value = mock_credential

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {"chunks": [{"content": "test content"}]}
        }
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(
            return_value=mock_client_instance
        )
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_client.return_value = mock_client_instance

        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        result = await api.retrieve_async("test query")
        assert result["query"] == "test query"
        assert result["knowledge_base_name"] == "test-kb"

    def test_retrieve_without_provider_settings(self):
        """测试无提供商设置检索"""
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            credential_name="test-credential",
        )

        result = api.retrieve("test query")
        assert result["error"] is True

    @patch("httpx.Client")
    @patch("agentrun.credential.Credential.get_by_name")
    def test_retrieve_with_false_data(
        self, mock_get_credential, mock_httpx_client
    ):
        """测试检索返回 False 数据"""
        mock_credential = MagicMock()
        mock_credential.credential_secret = "test-api-key"
        mock_get_credential.return_value = mock_credential

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": False}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__enter__ = MagicMock(
            return_value=mock_client_instance
        )
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx_client.return_value = mock_client_instance

        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        result = api.retrieve("test query")
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_retrieve_async_without_provider_settings(self):
        """测试异步检索无提供商设置"""
        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            credential_name="test-credential",
        )

        result = await api.retrieve_async("test query")
        assert result["error"] is True

    @patch("httpx.AsyncClient")
    @patch("agentrun.credential.Credential.get_by_name_async")
    @pytest.mark.asyncio
    async def test_retrieve_async_with_false_data(
        self, mock_get_credential, mock_httpx_client
    ):
        """测试异步检索返回 False 数据"""
        mock_credential = MagicMock()
        mock_credential.credential_secret = "test-api-key"
        mock_get_credential.return_value = mock_credential

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": False}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(
            return_value=mock_client_instance
        )
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_client.return_value = mock_client_instance

        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        result = await api.retrieve_async("test query")
        assert result["error"] is True

    @patch("httpx.AsyncClient")
    @patch("agentrun.credential.Credential.get_by_name_async")
    @pytest.mark.asyncio
    async def test_retrieve_async_exception(
        self, mock_get_credential, mock_httpx_client
    ):
        """测试异步检索异常处理"""
        mock_credential = MagicMock()
        mock_credential.credential_secret = "test-api-key"
        mock_get_credential.return_value = mock_credential

        mock_client_instance = MagicMock()
        mock_client_instance.post = AsyncMock(
            side_effect=Exception("Connection error")
        )
        mock_client_instance.__aenter__ = AsyncMock(
            return_value=mock_client_instance
        )
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_client.return_value = mock_client_instance

        api = RagFlowDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        result = await api.retrieve_async("test query")
        assert result["error"] is True
        assert "Connection error" in result["data"]


class TestBailianDataAPIInit:
    """测试 BailianDataAPI 初始化"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_init(self):
        """测试初始化"""
        api = BailianDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
            retrieve_settings=BailianRetrieveSettings(
                dense_similarity_top_k=10,
            ),
        )

        assert api.knowledge_base_name == "test-kb"
        assert api.provider_settings is not None
        assert api.retrieve_settings is not None

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_init_minimal(self):
        """测试最小化初始化"""
        api = BailianDataAPI(
            knowledge_base_name="test-kb",
        )

        assert api.knowledge_base_name == "test-kb"
        assert api.provider_settings is None


class TestBailianDataAPIRetrieve:
    """测试 BailianDataAPI.retrieve 方法"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_bailian_client")
    def test_retrieve_sync(self, mock_get_bailian_client):
        """测试同步检索"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.data.nodes = [
            MagicMock(text="test content", score=0.9, metadata={}),
        ]
        mock_client.retrieve.return_value = mock_response
        mock_get_bailian_client.return_value = mock_client

        api = BailianDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
        )

        result = api.retrieve("test query")
        assert result["query"] == "test query"
        assert result["knowledge_base_name"] == "test-kb"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_bailian_client")
    @pytest.mark.asyncio
    async def test_retrieve_async(self, mock_get_bailian_client):
        """测试异步检索"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.data.nodes = [
            MagicMock(text="test content", score=0.9, metadata={}),
        ]
        mock_client.retrieve_async = AsyncMock(return_value=mock_response)
        mock_get_bailian_client.return_value = mock_client

        api = BailianDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
        )

        result = await api.retrieve_async("test query")
        assert result["query"] == "test query"
        assert result["knowledge_base_name"] == "test-kb"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_retrieve_without_provider_settings(self):
        """测试无提供商设置检索"""
        api = BailianDataAPI(
            knowledge_base_name="test-kb",
        )

        result = api.retrieve("test query")
        assert result["error"] is True

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @pytest.mark.asyncio
    async def test_retrieve_async_without_provider_settings(self):
        """测试异步检索无提供商设置"""
        api = BailianDataAPI(
            knowledge_base_name="test-kb",
        )

        result = await api.retrieve_async("test query")
        assert result["error"] is True

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_bailian_client")
    def test_retrieve_with_retrieve_settings(self, mock_get_bailian_client):
        """测试带检索设置检索"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.data.nodes = []
        mock_client.retrieve.return_value = mock_response
        mock_get_bailian_client.return_value = mock_client

        api = BailianDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
            retrieve_settings=BailianRetrieveSettings(
                dense_similarity_top_k=10,
                sparse_similarity_top_k=5,
                rerank_min_score=0.5,
                rerank_top_n=3,
            ),
        )

        result = api.retrieve("test query")
        assert result["query"] == "test query"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_bailian_client")
    def test_retrieve_with_partial_retrieve_settings_dense_only(
        self, mock_get_bailian_client
    ):
        """测试仅设置 dense_similarity_top_k 的部分检索设置"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.data.nodes = []
        mock_client.retrieve.return_value = mock_response
        mock_get_bailian_client.return_value = mock_client

        api = BailianDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
            retrieve_settings=BailianRetrieveSettings(
                dense_similarity_top_k=10,
            ),
        )

        result = api.retrieve("test query")
        assert result["query"] == "test query"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_bailian_client")
    def test_retrieve_with_partial_retrieve_settings_sparse_only(
        self, mock_get_bailian_client
    ):
        """测试仅设置 sparse_similarity_top_k 的部分检索设置"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.data.nodes = []
        mock_client.retrieve.return_value = mock_response
        mock_get_bailian_client.return_value = mock_client

        api = BailianDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
            retrieve_settings=BailianRetrieveSettings(
                sparse_similarity_top_k=5,
            ),
        )

        result = api.retrieve("test query")
        assert result["query"] == "test query"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_bailian_client")
    def test_retrieve_with_partial_retrieve_settings_rerank_only(
        self, mock_get_bailian_client
    ):
        """测试仅设置 rerank 相关的部分检索设置"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.data.nodes = []
        mock_client.retrieve.return_value = mock_response
        mock_get_bailian_client.return_value = mock_client

        api = BailianDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
            retrieve_settings=BailianRetrieveSettings(
                rerank_min_score=0.5,
                rerank_top_n=3,
            ),
        )

        result = api.retrieve("test query")
        assert result["query"] == "test query"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_bailian_client")
    @pytest.mark.asyncio
    async def test_retrieve_async_with_partial_settings(
        self, mock_get_bailian_client
    ):
        """测试异步检索带部分设置"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.data.nodes = []
        mock_client.retrieve_async = AsyncMock(return_value=mock_response)
        mock_get_bailian_client.return_value = mock_client

        api = BailianDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
            retrieve_settings=BailianRetrieveSettings(
                dense_similarity_top_k=10,
            ),
        )

        result = await api.retrieve_async("test query")
        assert result["query"] == "test query"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_bailian_client")
    @pytest.mark.asyncio
    async def test_retrieve_async_with_sparse_only_settings(
        self, mock_get_bailian_client
    ):
        """测试异步检索仅设置 sparse_similarity_top_k"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.data.nodes = []
        mock_client.retrieve_async = AsyncMock(return_value=mock_response)
        mock_get_bailian_client.return_value = mock_client

        api = BailianDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
            retrieve_settings=BailianRetrieveSettings(
                sparse_similarity_top_k=5,
            ),
        )

        result = await api.retrieve_async("test query")
        assert result["query"] == "test query"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_bailian_client")
    @pytest.mark.asyncio
    async def test_retrieve_async_with_rerank_settings(
        self, mock_get_bailian_client
    ):
        """测试异步检索仅设置 rerank 相关参数"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.data.nodes = []
        mock_client.retrieve_async = AsyncMock(return_value=mock_response)
        mock_get_bailian_client.return_value = mock_client

        api = BailianDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
            retrieve_settings=BailianRetrieveSettings(
                rerank_min_score=0.5,
                rerank_top_n=3,
            ),
        )

        result = await api.retrieve_async("test query")
        assert result["query"] == "test query"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_bailian_client")
    @pytest.mark.asyncio
    async def test_retrieve_async_exception(self, mock_get_bailian_client):
        """测试异步检索异常处理"""
        mock_client = MagicMock()
        mock_client.retrieve_async = AsyncMock(
            side_effect=Exception("API error")
        )
        mock_get_bailian_client.return_value = mock_client

        api = BailianDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
        )

        result = await api.retrieve_async("test query")
        assert result["error"] is True
        assert "API error" in result["data"]

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_bailian_client")
    @pytest.mark.asyncio
    async def test_retrieve_async_with_all_settings(
        self, mock_get_bailian_client
    ):
        """测试异步检索带完整设置"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.data.nodes = []
        mock_client.retrieve_async = AsyncMock(return_value=mock_response)
        mock_get_bailian_client.return_value = mock_client

        api = BailianDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
            retrieve_settings=BailianRetrieveSettings(
                dense_similarity_top_k=10,
                sparse_similarity_top_k=5,
                rerank_min_score=0.5,
                rerank_top_n=3,
            ),
        )

        result = await api.retrieve_async("test query")
        assert result["query"] == "test query"


class TestADBDataAPIInit:
    """测试 ADBDataAPI 初始化"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_init(self):
        """测试初始化"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
            retrieve_settings=ADBRetrieveSettings(
                top_k=10,
            ),
        )

        assert api.knowledge_base_name == "test-kb"
        assert api.provider_settings is not None
        assert api.retrieve_settings is not None

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_init_minimal(self):
        """测试最小化初始化"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
        )

        assert api.knowledge_base_name == "test-kb"
        assert api.provider_settings is None


class TestADBDataAPIBuildQueryContentRequest:
    """测试 ADBDataAPI._build_query_content_request 方法"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_build_query_content_request(self):
        """测试构建 QueryContent 请求"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
        )

        request = api._build_query_content_request("test query")
        assert request.content == "test query"
        assert request.dbinstance_id == "gp-123456"
        assert request.namespace == "public"
        assert request.collection == "test-kb"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_build_query_content_request_with_settings(self):
        """测试带设置构建 QueryContent 请求"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
                metrics="cosine",
            ),
            retrieve_settings=ADBRetrieveSettings(
                top_k=10,
                use_full_text_retrieval=True,
                rerank_factor=1.5,
                recall_window=[-5, 5],
                hybrid_search="RRF",
                hybrid_search_args={"RRF": {"k": 60}},
            ),
        )

        request = api._build_query_content_request("test query")
        assert request.metrics == "cosine"
        assert request.top_k == 10
        assert request.use_full_text_retrieval is True
        assert request.rerank_factor == 1.5

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_build_query_content_request_without_provider_settings(self):
        """测试无提供商设置构建 QueryContent 请求"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
        )

        with pytest.raises(ValueError, match="provider_settings is required"):
            api._build_query_content_request("test query")

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_build_query_content_request_with_partial_settings_top_k_only(self):
        """测试仅设置 top_k 的部分检索设置"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
            retrieve_settings=ADBRetrieveSettings(
                top_k=10,
            ),
        )

        request = api._build_query_content_request("test query")
        assert request.top_k == 10

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_build_query_content_request_with_partial_settings_full_text_only(
        self,
    ):
        """测试仅设置 use_full_text_retrieval 的部分检索设置"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
            retrieve_settings=ADBRetrieveSettings(
                use_full_text_retrieval=True,
            ),
        )

        request = api._build_query_content_request("test query")
        assert request.use_full_text_retrieval is True

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_build_query_content_request_with_partial_settings_rerank_only(
        self,
    ):
        """测试仅设置 rerank_factor 的部分检索设置"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
            retrieve_settings=ADBRetrieveSettings(
                rerank_factor=1.5,
            ),
        )

        request = api._build_query_content_request("test query")
        assert request.rerank_factor == 1.5

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_build_query_content_request_with_partial_settings_recall_window_only(
        self,
    ):
        """测试仅设置 recall_window 的部分检索设置"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
            retrieve_settings=ADBRetrieveSettings(
                recall_window=[-5, 5],
            ),
        )

        request = api._build_query_content_request("test query")
        assert request.recall_window == [-5, 5]

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_build_query_content_request_with_partial_settings_hybrid_only(
        self,
    ):
        """测试仅设置 hybrid_search 的部分检索设置"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
            retrieve_settings=ADBRetrieveSettings(
                hybrid_search="RRF",
            ),
        )

        request = api._build_query_content_request("test query")
        assert request.hybrid_search == "RRF"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_build_query_content_request_with_partial_settings_hybrid_args_only(
        self,
    ):
        """测试仅设置 hybrid_search_args 的部分检索设置"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
            retrieve_settings=ADBRetrieveSettings(
                hybrid_search_args={"RRF": {"k": 60}},
            ),
        )

        request = api._build_query_content_request("test query")
        assert request.hybrid_search_args == {"RRF": {"k": 60}}


class TestADBDataAPIParseQueryContentResponse:
    """测试 ADBDataAPI._parse_query_content_response 方法"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_parse_query_content_response(self):
        """测试解析 QueryContent 响应"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
        )

        mock_response = MagicMock()
        mock_response.body.matches.match_list = [
            MagicMock(
                content="test content",
                score=0.9,
                id="1",
                file_name="test.txt",
                metadata={},
                rerank_score=0.95,
                retrieval_source="vector",
            ),
        ]
        mock_response.body.request_id = "req-123"

        result = api._parse_query_content_response(mock_response, "test query")
        assert result["query"] == "test query"
        assert result["knowledge_base_name"] == "test-kb"
        assert result["request_id"] == "req-123"
        assert len(result["data"]) == 1

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_parse_query_content_response_empty(self):
        """测试解析空 QueryContent 响应"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
        )

        mock_response = MagicMock()
        mock_response.body.matches = None
        mock_response.body.request_id = "req-123"

        result = api._parse_query_content_response(mock_response, "test query")
        assert result["data"] == []


class TestADBDataAPIRetrieve:
    """测试 ADBDataAPI.retrieve 方法"""

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_gpdb_client")
    def test_retrieve_sync(self, mock_get_gpdb_client):
        """测试同步检索"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.matches.match_list = [
            MagicMock(
                content="test content",
                score=0.9,
                id="1",
                file_name="test.txt",
                metadata={},
                rerank_score=0.95,
                retrieval_source="vector",
            ),
        ]
        mock_response.body.request_id = "req-123"
        mock_client.query_content.return_value = mock_response
        mock_get_gpdb_client.return_value = mock_client

        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
        )

        result = api.retrieve("test query")
        assert result["query"] == "test query"
        assert result["knowledge_base_name"] == "test-kb"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_gpdb_client")
    @pytest.mark.asyncio
    async def test_retrieve_async(self, mock_get_gpdb_client):
        """测试异步检索"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.body.matches.match_list = [
            MagicMock(
                content="test content",
                score=0.9,
                id="1",
                file_name="test.txt",
                metadata={},
                rerank_score=0.95,
                retrieval_source="vector",
            ),
        ]
        mock_response.body.request_id = "req-123"
        mock_client.query_content_async = AsyncMock(return_value=mock_response)
        mock_get_gpdb_client.return_value = mock_client

        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
        )

        result = await api.retrieve_async("test query")
        assert result["query"] == "test query"
        assert result["knowledge_base_name"] == "test-kb"

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_retrieve_without_provider_settings(self):
        """测试无提供商设置检索"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
        )

        result = api.retrieve("test query")
        assert result["error"] is True

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_gpdb_client")
    def test_retrieve_exception(self, mock_get_gpdb_client):
        """测试检索异常处理"""
        mock_client = MagicMock()
        mock_client.query_content.side_effect = Exception("Query failed")
        mock_get_gpdb_client.return_value = mock_client

        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
        )

        result = api.retrieve("test query")
        assert result["error"] is True

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @patch("agentrun.utils.control_api.ControlAPI._get_gpdb_client")
    @pytest.mark.asyncio
    async def test_retrieve_async_exception(self, mock_get_gpdb_client):
        """测试异步检索异常处理"""
        mock_client = MagicMock()
        mock_client.query_content_async = AsyncMock(
            side_effect=Exception("Query failed")
        )
        mock_get_gpdb_client.return_value = mock_client

        api = ADBDataAPI(
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
        )

        result = await api.retrieve_async("test query")
        assert result["error"] is True
        assert "Query failed" in result["data"]

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    @pytest.mark.asyncio
    async def test_retrieve_async_without_provider_settings(self):
        """测试异步检索无提供商设置"""
        api = ADBDataAPI(
            knowledge_base_name="test-kb",
        )

        result = await api.retrieve_async("test query")
        assert result["error"] is True


class TestGetDataAPI:
    """测试 get_data_api 工厂函数"""

    def test_get_data_api_ragflow(self):
        """测试获取 RagFlow 数据链路 API"""
        api = get_data_api(
            provider=KnowledgeBaseProvider.RAGFLOW,
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            credential_name="test-credential",
        )

        assert isinstance(api, RagFlowDataAPI)

    def test_get_data_api_ragflow_string(self):
        """测试使用字符串获取 RagFlow 数据链路 API"""
        api = get_data_api(
            provider="ragflow",  # type: ignore
            knowledge_base_name="test-kb",
        )

        assert isinstance(api, RagFlowDataAPI)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_get_data_api_bailian(self):
        """测试获取百炼数据链路 API"""
        api = get_data_api(
            provider=KnowledgeBaseProvider.BAILIAN,
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
        )

        assert isinstance(api, BailianDataAPI)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_get_data_api_bailian_string(self):
        """测试使用字符串获取百炼数据链路 API"""
        api = get_data_api(
            provider="bailian",  # type: ignore
            knowledge_base_name="test-kb",
        )

        assert isinstance(api, BailianDataAPI)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_get_data_api_adb(self):
        """测试获取 ADB 数据链路 API"""
        api = get_data_api(
            provider=KnowledgeBaseProvider.ADB,
            knowledge_base_name="test-kb",
            provider_settings=ADBProviderSettings(
                db_instance_id="gp-123456",
                namespace="public",
                namespace_password="password123",
            ),
        )

        assert isinstance(api, ADBDataAPI)

    @patch.dict(
        os.environ,
        {
            "AGENTRUN_ACCESS_KEY_ID": "test-ak",
            "AGENTRUN_ACCESS_KEY_SECRET": "test-sk",
        },
    )
    def test_get_data_api_adb_string(self):
        """测试使用字符串获取 ADB 数据链路 API"""
        api = get_data_api(
            provider="adb",  # type: ignore
            knowledge_base_name="test-kb",
        )

        assert isinstance(api, ADBDataAPI)

    def test_get_data_api_unsupported_provider(self):
        """测试不支持的提供商"""
        with pytest.raises(ValueError, match="Unsupported provider type"):
            get_data_api(
                provider="unsupported",  # type: ignore
                knowledge_base_name="test-kb",
            )

    def test_get_data_api_with_wrong_settings_type(self):
        """测试使用错误类型的设置"""
        # 传入 Bailian 设置给 RagFlow
        api = get_data_api(
            provider=KnowledgeBaseProvider.RAGFLOW,
            knowledge_base_name="test-kb",
            provider_settings=BailianProviderSettings(
                workspace_id="ws-123",
                index_ids=["idx-1"],
            ),
        )

        # 应该返回 RagFlowDataAPI，但 provider_settings 会是 None
        assert isinstance(api, RagFlowDataAPI)
        assert api.provider_settings is None

    def test_get_data_api_with_config(self):
        """测试带配置获取数据链路 API"""
        config = Config(access_key_id="test-ak")
        api = get_data_api(
            provider=KnowledgeBaseProvider.RAGFLOW,
            knowledge_base_name="test-kb",
            config=config,
        )

        assert isinstance(api, RagFlowDataAPI)

    def test_get_data_api_with_retrieve_settings(self):
        """测试带检索设置获取数据链路 API"""
        api = get_data_api(
            provider=KnowledgeBaseProvider.RAGFLOW,
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="https://ragflow.example.com",
                dataset_ids=["ds-1"],
            ),
            retrieve_settings=RagFlowRetrieveSettings(
                similarity_threshold=0.8,
            ),
            credential_name="test-credential",
        )

        assert isinstance(api, RagFlowDataAPI)
        assert api.retrieve_settings is not None
