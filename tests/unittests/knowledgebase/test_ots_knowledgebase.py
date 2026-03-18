"""测试 OTS 知识库相关功能 / Test OTS KnowledgeBase functionality"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.knowledgebase.api.data import get_data_api, OTSDataAPI
from agentrun.knowledgebase.knowledgebase import KnowledgeBase
from agentrun.knowledgebase.model import (
    KnowledgeBaseProvider,
    OTSDenseVectorSearchConfig,
    OTSEmbeddingConfiguration,
    OTSFullTextSearchConfig,
    OTSMetadataField,
    OTSModelConfig,
    OTSProviderSettings,
    OTSRerankingConfig,
    OTSRetrieveSettings,
    OTSRRFConfig,
    OTSWeightConfig,
)
from agentrun.utils.config import Config

# =============================================================================
# OTS 模型测试 / OTS Model Tests
# =============================================================================


class TestOTSModels:
    """测试 OTS 模型定义"""

    def test_ots_provider_enum(self):
        """测试 OTS provider 枚举值"""
        assert KnowledgeBaseProvider.OTS == "ots"
        assert KnowledgeBaseProvider("ots") == KnowledgeBaseProvider.OTS

    def test_ots_metadata_field(self):
        """测试 OTSMetadataField 模型"""
        field = OTSMetadataField(name="author", type="string")
        assert field.name == "author"
        assert field.type == "string"

    def test_ots_embedding_configuration(self):
        """测试 OTSEmbeddingConfiguration 模型"""
        ec = OTSEmbeddingConfiguration(
            provider="bailian",
            model="text-embedding-v3",
            dimension=1024,
        )
        assert ec.provider == "bailian"
        assert ec.model == "text-embedding-v3"
        assert ec.dimension == 1024
        assert ec.url is None
        assert ec.api_key is None

    def test_ots_embedding_configuration_full(self):
        """测试 OTSEmbeddingConfiguration 完整字段"""
        ec = OTSEmbeddingConfiguration(
            provider="bailian",
            model="text-embedding-v3",
            dimension=1024,
            url="https://embedding.example.com",
            api_key="test-key",
        )
        assert ec.url == "https://embedding.example.com"
        assert ec.api_key == "test-key"

    def test_ots_provider_settings_required_only(self):
        """测试 OTSProviderSettings 仅必填字段"""
        ps = OTSProviderSettings(ots_instance_name="test-instance")
        assert ps.ots_instance_name == "test-instance"
        assert ps.tags is None
        assert ps.metadata is None
        assert ps.embedding_configuration is None

    def test_ots_provider_settings_full(self):
        """测试 OTSProviderSettings 完整字段"""
        ps = OTSProviderSettings(
            ots_instance_name="test-instance",
            tags=["demo", "test"],
            metadata=[
                OTSMetadataField(name="author", type="string"),
                OTSMetadataField(name="score", type="double"),
            ],
            embedding_configuration=OTSEmbeddingConfiguration(
                provider="bailian",
                model="text-embedding-v3",
                dimension=1024,
            ),
        )
        assert ps.ots_instance_name == "test-instance"
        assert ps.tags == ["demo", "test"]
        assert len(ps.metadata) == 2
        assert ps.metadata[0].name == "author"
        assert ps.embedding_configuration.provider == "bailian"

    def test_ots_dense_vector_search_config(self):
        """测试 OTSDenseVectorSearchConfig 模型"""
        c = OTSDenseVectorSearchConfig(number_of_results=10)
        assert c.number_of_results == 10

    def test_ots_full_text_search_config(self):
        """测试 OTSFullTextSearchConfig 模型"""
        c = OTSFullTextSearchConfig(number_of_results=20)
        assert c.number_of_results == 20

    def test_ots_rrf_config(self):
        """测试 OTSRRFConfig 模型"""
        c = OTSRRFConfig(
            dense_vector_search_weight=1.0,
            full_text_search_weight=1.0,
            k=60,
        )
        assert c.dense_vector_search_weight == 1.0
        assert c.full_text_search_weight == 1.0
        assert c.k == 60

    def test_ots_weight_config(self):
        """测试 OTSWeightConfig 模型"""
        c = OTSWeightConfig(
            dense_vector_search_weight=0.7,
            full_text_search_weight=0.3,
        )
        assert c.dense_vector_search_weight == 0.7
        assert c.full_text_search_weight == 0.3

    def test_ots_model_config(self):
        """测试 OTSModelConfig 模型"""
        c = OTSModelConfig(provider="bailian", model="gte-rerank-v2")
        assert c.provider == "bailian"
        assert c.model == "gte-rerank-v2"

    def test_ots_reranking_config_rrf(self):
        """测试 OTSRerankingConfig RRF 类型"""
        c = OTSRerankingConfig(
            type="RRF",
            number_of_results=10,
            rrf_configuration=OTSRRFConfig(
                dense_vector_search_weight=1.0,
                full_text_search_weight=1.0,
                k=60,
            ),
        )
        assert c.type == "RRF"
        assert c.rrf_configuration.k == 60

    def test_ots_reranking_config_weight(self):
        """测试 OTSRerankingConfig WEIGHT 类型"""
        c = OTSRerankingConfig(
            type="WEIGHT",
            number_of_results=10,
            weight_configuration=OTSWeightConfig(
                dense_vector_search_weight=0.7,
                full_text_search_weight=0.3,
            ),
        )
        assert c.type == "WEIGHT"
        assert c.weight_configuration.dense_vector_search_weight == 0.7

    def test_ots_reranking_config_model(self):
        """测试 OTSRerankingConfig MODEL 类型"""
        c = OTSRerankingConfig(
            type="MODEL",
            number_of_results=10,
            model_configuration=OTSModelConfig(
                provider="bailian", model="gte-rerank-v2"
            ),
        )
        assert c.type == "MODEL"
        assert c.model_configuration.model == "gte-rerank-v2"

    def test_ots_retrieve_settings_dense_only(self):
        """测试 OTSRetrieveSettings 仅向量检索"""
        rs = OTSRetrieveSettings(
            search_type=["DENSE_VECTOR"],
            dense_vector_search_configuration=OTSDenseVectorSearchConfig(
                number_of_results=10
            ),
            reranking_configuration=OTSRerankingConfig(
                type="RRF", number_of_results=10
            ),
        )
        assert rs.search_type == ["DENSE_VECTOR"]
        assert rs.dense_vector_search_configuration.number_of_results == 10
        assert rs.full_text_search_configuration is None

    def test_ots_retrieve_settings_hybrid(self):
        """测试 OTSRetrieveSettings 混合检索"""
        rs = OTSRetrieveSettings(
            search_type=["DENSE_VECTOR", "FULL_TEXT"],
            dense_vector_search_configuration=OTSDenseVectorSearchConfig(
                number_of_results=20
            ),
            full_text_search_configuration=OTSFullTextSearchConfig(
                number_of_results=20
            ),
        )
        assert len(rs.search_type) == 2
        assert rs.full_text_search_configuration.number_of_results == 20

    def test_ots_retrieve_settings_with_filter(self):
        """测试 OTSRetrieveSettings 带元数据过滤"""
        rs = OTSRetrieveSettings(
            search_type=["DENSE_VECTOR"],
            filter={
                "andAll": [
                    {"equals": {"key": "author", "value": "test"}},
                    {"greaterThan": {"key": "score", "value": 0.5}},
                ]
            },
        )
        assert rs.filter is not None
        assert "andAll" in rs.filter

    def test_ots_retrieve_settings_all_none(self):
        """测试 OTSRetrieveSettings 所有字段为 None"""
        rs = OTSRetrieveSettings()
        assert rs.search_type is None
        assert rs.dense_vector_search_configuration is None
        assert rs.full_text_search_configuration is None
        assert rs.reranking_configuration is None
        assert rs.filter is None


# =============================================================================
# OTSDataAPI 测试 / OTSDataAPI Tests
# =============================================================================


class TestOTSDataAPIBuildRetrievalConfiguration:
    """测试 OTSDataAPI._build_retrieval_configuration"""

    def test_none_retrieve_settings(self):
        """测试无 retrieve_settings"""
        api = OTSDataAPI("test-kb", provider_settings=None)
        assert api._build_retrieval_configuration() is None

    def test_empty_retrieve_settings(self):
        """测试空 retrieve_settings"""
        api = OTSDataAPI(
            "test-kb",
            retrieve_settings=OTSRetrieveSettings(),
        )
        assert api._build_retrieval_configuration() is None

    def test_dense_vector_only(self):
        """测试仅向量检索配置"""
        api = OTSDataAPI(
            "test-kb",
            retrieve_settings=OTSRetrieveSettings(
                search_type=["DENSE_VECTOR"],
                dense_vector_search_configuration=OTSDenseVectorSearchConfig(
                    number_of_results=10
                ),
            ),
        )
        config = api._build_retrieval_configuration()
        assert config["searchType"] == ["DENSE_VECTOR"]
        assert config["denseVectorSearchConfiguration"]["numberOfResults"] == 10

    def test_hybrid_search_with_rrf(self):
        """测试混合检索 + RRF 重排序"""
        api = OTSDataAPI(
            "test-kb",
            retrieve_settings=OTSRetrieveSettings(
                search_type=["DENSE_VECTOR", "FULL_TEXT"],
                dense_vector_search_configuration=OTSDenseVectorSearchConfig(
                    number_of_results=20
                ),
                full_text_search_configuration=OTSFullTextSearchConfig(
                    number_of_results=20
                ),
                reranking_configuration=OTSRerankingConfig(
                    type="RRF",
                    number_of_results=10,
                    rrf_configuration=OTSRRFConfig(
                        dense_vector_search_weight=1.0,
                        full_text_search_weight=1.0,
                        k=60,
                    ),
                ),
            ),
        )
        config = api._build_retrieval_configuration()
        assert config["searchType"] == ["DENSE_VECTOR", "FULL_TEXT"]
        assert config["rerankingConfiguration"]["type"] == "RRF"
        assert config["rerankingConfiguration"]["rrfConfiguration"]["k"] == 60

    def test_weight_reranking(self):
        """测试 Weight 重排序"""
        api = OTSDataAPI(
            "test-kb",
            retrieve_settings=OTSRetrieveSettings(
                reranking_configuration=OTSRerankingConfig(
                    type="WEIGHT",
                    number_of_results=10,
                    weight_configuration=OTSWeightConfig(
                        dense_vector_search_weight=0.7,
                        full_text_search_weight=0.3,
                    ),
                ),
            ),
        )
        config = api._build_retrieval_configuration()
        rr = config["rerankingConfiguration"]
        assert rr["type"] == "WEIGHT"
        assert rr["weightConfiguration"]["denseVectorSearchWeight"] == 0.7
        assert rr["weightConfiguration"]["fullTextSearchWeight"] == 0.3

    def test_model_reranking(self):
        """测试 Model 重排序"""
        api = OTSDataAPI(
            "test-kb",
            retrieve_settings=OTSRetrieveSettings(
                reranking_configuration=OTSRerankingConfig(
                    type="MODEL",
                    number_of_results=10,
                    model_configuration=OTSModelConfig(
                        provider="bailian", model="gte-rerank-v2"
                    ),
                ),
            ),
        )
        config = api._build_retrieval_configuration()
        rr = config["rerankingConfiguration"]
        assert rr["type"] == "MODEL"
        assert rr["modelConfiguration"]["provider"] == "bailian"
        assert rr["modelConfiguration"]["model"] == "gte-rerank-v2"

    def test_with_filter(self):
        """测试带过滤条件"""
        api = OTSDataAPI(
            "test-kb",
            retrieve_settings=OTSRetrieveSettings(
                search_type=["DENSE_VECTOR"],
                filter={
                    "andAll": [{"equals": {"key": "author", "value": "test"}}]
                },
            ),
        )
        config = api._build_retrieval_configuration()
        assert config["filter"]["andAll"][0]["equals"]["key"] == "author"


class TestOTSDataAPIParseResponse:
    """测试 OTSDataAPI._parse_retrieve_response"""

    def test_parse_normal_response(self):
        """测试解析正常响应"""
        api = OTSDataAPI("test-kb")
        response = {
            "code": "SUCCESS",
            "data": {
                "retrievalResults": [{
                    "ossKey": "oss://testbucket/xxx.pdf",
                    "docId": "96fb386e-44d5-40aa-aa4d-edc0762f867c",
                    "chunkId": 3,
                    "subspace": "test",
                    "score": 0.1,
                    "content": "test content",
                    "metadata": {"date": "2026-01-22"},
                }]
            },
            "message": "success",
        }
        result = api._parse_retrieve_response(response, "test query")
        assert result["query"] == "test query"
        assert result["knowledge_base_name"] == "test-kb"
        assert len(result["data"]) == 1
        assert result["data"][0]["content"] == "test content"
        assert result["data"][0]["score"] == 0.1
        assert (
            result["data"][0]["doc_id"]
            == "96fb386e-44d5-40aa-aa4d-edc0762f867c"
        )
        assert result["data"][0]["chunk_id"] == 3
        assert result["data"][0]["subspace"] == "test"
        assert result["data"][0]["oss_key"] == "oss://testbucket/xxx.pdf"
        assert result["data"][0]["metadata"]["date"] == "2026-01-22"

    def test_parse_empty_response(self):
        """测试解析空响应"""
        api = OTSDataAPI("test-kb")
        response = {"code": "SUCCESS", "data": {"retrievalResults": []}}
        result = api._parse_retrieve_response(response, "test query")
        assert result["data"] == []

    def test_parse_no_data(self):
        """测试解析无 data 字段的响应"""
        api = OTSDataAPI("test-kb")
        response = {"code": "SUCCESS"}
        result = api._parse_retrieve_response(response, "test query")
        assert result["data"] == []

    def test_parse_multiple_results(self):
        """测试解析多条结果"""
        api = OTSDataAPI("test-kb")
        response = {
            "code": "SUCCESS",
            "data": {
                "retrievalResults": [
                    {"content": "result 1", "score": 0.9},
                    {"content": "result 2", "score": 0.8},
                    {"content": "result 3", "score": 0.7},
                ]
            },
        }
        result = api._parse_retrieve_response(response, "query")
        assert len(result["data"]) == 3
        assert result["data"][0]["score"] == 0.9
        assert result["data"][2]["content"] == "result 3"


class TestOTSDataAPIRetrieve:
    """测试 OTSDataAPI.retrieve 方法"""

    @patch(
        "agentrun.knowledgebase.api.data.OTSDataAPI._build_agent_storage_client"
    )
    def test_retrieve_sync_success(self, mock_build_client):
        """测试同步检索成功"""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = {
            "code": "SUCCESS",
            "data": {
                "retrievalResults": [{"content": "test result", "score": 0.9}]
            },
        }
        mock_build_client.return_value = mock_client

        api = OTSDataAPI(
            "test-kb",
            provider_settings=OTSProviderSettings(
                ots_instance_name="test-instance"
            ),
            retrieve_settings=OTSRetrieveSettings(
                search_type=["DENSE_VECTOR"],
                dense_vector_search_configuration=OTSDenseVectorSearchConfig(
                    number_of_results=10
                ),
            ),
        )

        result = api.retrieve("test query")
        assert result["query"] == "test query"
        assert len(result["data"]) == 1
        assert result["data"][0]["content"] == "test result"
        mock_client.retrieve.assert_called_once()

    @patch(
        "agentrun.knowledgebase.api.data.OTSDataAPI._build_agent_storage_client"
    )
    @pytest.mark.asyncio
    async def test_retrieve_async_success(self, mock_build_client):
        """测试异步检索成功"""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = {
            "code": "SUCCESS",
            "data": {
                "retrievalResults": [{"content": "async result", "score": 0.85}]
            },
        }
        mock_build_client.return_value = mock_client

        api = OTSDataAPI(
            "test-kb",
            provider_settings=OTSProviderSettings(
                ots_instance_name="test-instance"
            ),
        )

        result = await api.retrieve_async("async query")
        assert result["query"] == "async query"
        assert result["data"][0]["content"] == "async result"

    @patch(
        "agentrun.knowledgebase.api.data.OTSDataAPI._build_agent_storage_client"
    )
    def test_retrieve_error_handling(self, mock_build_client):
        """测试检索错误处理"""
        mock_build_client.side_effect = Exception("Connection failed")

        api = OTSDataAPI(
            "test-kb",
            provider_settings=OTSProviderSettings(
                ots_instance_name="test-instance"
            ),
        )

        result = api.retrieve("test query")
        assert result["error"] is True
        assert "Failed to retrieve" in result["data"]
        assert result["query"] == "test query"
        assert result["knowledge_base_name"] == "test-kb"

    @patch(
        "agentrun.knowledgebase.api.data.OTSDataAPI._build_agent_storage_client"
    )
    @pytest.mark.asyncio
    async def test_retrieve_async_error_handling(self, mock_build_client):
        """测试异步检索错误处理"""
        mock_build_client.side_effect = Exception("Network error")

        api = OTSDataAPI(
            "test-kb",
            provider_settings=OTSProviderSettings(
                ots_instance_name="test-instance"
            ),
        )

        result = await api.retrieve_async("test query")
        assert result["error"] is True
        assert "Failed to retrieve" in result["data"]

    def test_retrieve_without_provider_settings(self):
        """测试无 provider_settings 时检索"""
        api = OTSDataAPI("test-kb")
        result = api.retrieve("test query")
        assert result["error"] is True
        assert "provider_settings is required" in result["data"]

    @pytest.mark.asyncio
    async def test_retrieve_async_without_provider_settings(self):
        """测试异步无 provider_settings 时检索"""
        api = OTSDataAPI("test-kb")
        result = await api.retrieve_async("test query")
        assert result["error"] is True
        assert "provider_settings is required" in result["data"]

    @patch(
        "agentrun.knowledgebase.api.data.OTSDataAPI._build_agent_storage_client"
    )
    def test_retrieve_without_retrieve_settings(self, mock_build_client):
        """测试无 retrieve_settings 时检索（使用默认配置）"""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = {
            "code": "SUCCESS",
            "data": {
                "retrievalResults": [{"content": "default", "score": 0.5}]
            },
        }
        mock_build_client.return_value = mock_client

        api = OTSDataAPI(
            "test-kb",
            provider_settings=OTSProviderSettings(
                ots_instance_name="test-instance"
            ),
        )

        result = api.retrieve("test query")
        assert len(result["data"]) == 1

        call_args = mock_client.retrieve.call_args[0][0]
        assert "retrievalConfiguration" not in call_args

    @patch(
        "agentrun.knowledgebase.api.data.OTSDataAPI._build_agent_storage_client"
    )
    def test_retrieve_request_structure(self, mock_build_client):
        """测试检索请求结构是否正确"""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = {
            "code": "SUCCESS",
            "data": {"retrievalResults": []},
        }
        mock_build_client.return_value = mock_client

        api = OTSDataAPI(
            "test-kb",
            provider_settings=OTSProviderSettings(
                ots_instance_name="test-instance"
            ),
            retrieve_settings=OTSRetrieveSettings(
                search_type=["DENSE_VECTOR"],
                dense_vector_search_configuration=OTSDenseVectorSearchConfig(
                    number_of_results=10
                ),
            ),
        )

        api.retrieve("test query")

        call_args = mock_client.retrieve.call_args[0][0]
        assert call_args["knowledgeBaseName"] == "test-kb"
        assert call_args["retrievalQuery"]["text"] == "test query"
        assert call_args["retrievalQuery"]["type"] == "TEXT"
        assert call_args["retrievalConfiguration"]["searchType"] == [
            "DENSE_VECTOR"
        ]


class TestOTSDataAPIBuildClient:
    """测试 OTSDataAPI._build_agent_storage_client"""

    @patch("agentrun.knowledgebase.api.data.AgentStorageClient")
    def test_build_client(self, mock_client_class):
        """测试构建客户端"""
        mock_config = MagicMock(spec=Config)
        mock_config.get_region_id.return_value = "cn-hangzhou"
        mock_config.get_access_key_id.return_value = "test-ak"
        mock_config.get_access_key_secret.return_value = "test-sk"

        with patch.object(Config, "with_configs", return_value=mock_config):
            api = OTSDataAPI(
                "test-kb",
                provider_settings=OTSProviderSettings(
                    ots_instance_name="test-instance"
                ),
            )
            api._build_agent_storage_client()

        mock_client_class.assert_called_once_with(
            access_key_id="test-ak",
            access_key_secret="test-sk",
            ots_endpoint="http://ots-cn-hangzhou.aliyuncs.com",
            ots_instance_name="test-instance",
        )

    def test_build_client_without_provider_settings(self):
        """测试无 provider_settings 时构建客户端"""
        api = OTSDataAPI("test-kb")
        with pytest.raises(ValueError, match="provider_settings is required"):
            api._build_agent_storage_client()


# =============================================================================
# get_data_api 工厂函数测试 / get_data_api Factory Tests
# =============================================================================


class TestGetDataAPIOTS:
    """测试 get_data_api 的 OTS 分支"""

    def test_get_data_api_ots_with_enum(self):
        """测试使用枚举获取 OTS DataAPI"""
        api = get_data_api(
            provider=KnowledgeBaseProvider.OTS,
            knowledge_base_name="test-kb",
            provider_settings=OTSProviderSettings(
                ots_instance_name="test-instance"
            ),
        )
        assert isinstance(api, OTSDataAPI)

    def test_get_data_api_ots_with_string(self):
        """测试使用字符串获取 OTS DataAPI"""
        api = get_data_api(
            provider="ots",
            knowledge_base_name="test-kb",
            provider_settings=OTSProviderSettings(
                ots_instance_name="test-instance"
            ),
        )
        assert isinstance(api, OTSDataAPI)

    def test_get_data_api_ots_with_settings(self):
        """测试获取带设置的 OTS DataAPI"""
        ps = OTSProviderSettings(ots_instance_name="test-instance")
        rs = OTSRetrieveSettings(search_type=["DENSE_VECTOR"])

        api = get_data_api(
            provider=KnowledgeBaseProvider.OTS,
            knowledge_base_name="test-kb",
            provider_settings=ps,
            retrieve_settings=rs,
        )
        assert isinstance(api, OTSDataAPI)
        assert api.provider_settings is ps
        assert api.retrieve_settings is rs

    def test_get_data_api_ots_without_settings(self):
        """测试获取无设置的 OTS DataAPI"""
        api = get_data_api(
            provider=KnowledgeBaseProvider.OTS,
            knowledge_base_name="test-kb",
        )
        assert isinstance(api, OTSDataAPI)
        assert api.provider_settings is None
        assert api.retrieve_settings is None

    def test_get_data_api_ots_wrong_settings_type(self):
        """测试传入非 OTS 类型的 settings"""
        from agentrun.knowledgebase.model import RagFlowProviderSettings

        api = get_data_api(
            provider=KnowledgeBaseProvider.OTS,
            knowledge_base_name="test-kb",
            provider_settings=RagFlowProviderSettings(
                base_url="http://example.com",
                dataset_ids=["ds-1"],
            ),
        )
        assert isinstance(api, OTSDataAPI)
        assert api.provider_settings is None


# =============================================================================
# KnowledgeBase._get_data_api OTS 分支测试
# =============================================================================


class TestKnowledgeBaseGetDataAPIOTS:
    """测试 KnowledgeBase._get_data_api 的 OTS 分支"""

    def test_get_data_api_ots_with_typed_settings(self):
        """测试 OTS 使用类型化设置"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.OTS,
            provider_settings=OTSProviderSettings(
                ots_instance_name="test-instance",
            ),
            retrieve_settings=OTSRetrieveSettings(
                search_type=["DENSE_VECTOR"],
            ),
        )
        data_api = kb._get_data_api()
        assert isinstance(data_api, OTSDataAPI)
        assert data_api.provider_settings.ots_instance_name == "test-instance"
        assert data_api.retrieve_settings.search_type == ["DENSE_VECTOR"]

    def test_get_data_api_ots_with_camelcase_dict(self):
        """测试 OTS 使用 camelCase dict 设置"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.OTS,
        )
        object.__setattr__(
            kb,
            "provider_settings",
            {
                "otsInstanceName": "jingsu-ots-test",
                "tags": ["demo", "test"],
                "metadata": [
                    {"name": "author", "type": "string"},
                    {"name": "date", "type": "date"},
                ],
                "embeddingConfiguration": {
                    "provider": "bailian",
                    "model": "text-embedding-v3",
                    "dimension": 1024,
                },
            },
        )
        object.__setattr__(
            kb,
            "retrieve_settings",
            {
                "searchType": ["DENSE_VECTOR", "FULL_TEXT"],
                "denseVectorSearchConfiguration": {"numberOfResults": 20},
                "fullTextSearchConfiguration": {"numberOfResults": 20},
                "rerankingConfiguration": {
                    "type": "RRF",
                    "numberOfResults": 10,
                    "rrfConfiguration": {
                        "denseVectorSearchWeight": 1.0,
                        "fullTextSearchWeight": 1.0,
                        "k": 60,
                    },
                },
            },
        )

        data_api = kb._get_data_api()
        assert isinstance(data_api, OTSDataAPI)

        ps = data_api.provider_settings
        assert ps.ots_instance_name == "jingsu-ots-test"
        assert ps.tags == ["demo", "test"]
        assert len(ps.metadata) == 2
        assert ps.metadata[0].name == "author"
        assert ps.embedding_configuration.provider == "bailian"
        assert ps.embedding_configuration.dimension == 1024

        rs = data_api.retrieve_settings
        assert rs.search_type == ["DENSE_VECTOR", "FULL_TEXT"]
        assert rs.dense_vector_search_configuration.number_of_results == 20
        assert rs.full_text_search_configuration.number_of_results == 20
        assert rs.reranking_configuration.type == "RRF"
        assert rs.reranking_configuration.rrf_configuration.k == 60

    def test_get_data_api_ots_with_weight_reranking_dict(self):
        """测试 OTS camelCase dict 带 Weight 重排序"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.OTS,
        )
        object.__setattr__(
            kb,
            "provider_settings",
            {"otsInstanceName": "test-instance"},
        )
        object.__setattr__(
            kb,
            "retrieve_settings",
            {
                "searchType": ["DENSE_VECTOR", "FULL_TEXT"],
                "rerankingConfiguration": {
                    "type": "WEIGHT",
                    "numberOfResults": 10,
                    "weightConfiguration": {
                        "denseVectorSearchWeight": 0.7,
                        "fullTextSearchWeight": 0.3,
                    },
                },
            },
        )

        data_api = kb._get_data_api()
        assert isinstance(data_api, OTSDataAPI)
        rs = data_api.retrieve_settings
        assert rs.reranking_configuration.type == "WEIGHT"
        assert (
            rs.reranking_configuration.weight_configuration.dense_vector_search_weight
            == 0.7
        )

    def test_get_data_api_ots_with_model_reranking_dict(self):
        """测试 OTS camelCase dict 带 Model 重排序"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.OTS,
        )
        object.__setattr__(
            kb,
            "provider_settings",
            {"otsInstanceName": "test-instance"},
        )
        object.__setattr__(
            kb,
            "retrieve_settings",
            {
                "rerankingConfiguration": {
                    "type": "MODEL",
                    "numberOfResults": 10,
                    "modelConfiguration": {
                        "provider": "bailian",
                        "model": "gte-rerank-v2",
                    },
                },
            },
        )

        data_api = kb._get_data_api()
        rs = data_api.retrieve_settings
        assert rs.reranking_configuration.type == "MODEL"
        assert (
            rs.reranking_configuration.model_configuration.provider == "bailian"
        )

    def test_get_data_api_ots_with_filter_dict(self):
        """测试 OTS camelCase dict 带 filter"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.OTS,
        )
        object.__setattr__(
            kb,
            "provider_settings",
            {"otsInstanceName": "test-instance"},
        )
        object.__setattr__(
            kb,
            "retrieve_settings",
            {
                "searchType": ["DENSE_VECTOR"],
                "filter": {
                    "andAll": [{"equals": {"key": "author", "value": "test"}}]
                },
            },
        )

        data_api = kb._get_data_api()
        rs = data_api.retrieve_settings
        assert rs.filter is not None
        assert rs.filter["andAll"][0]["equals"]["key"] == "author"

    def test_get_data_api_ots_minimal_dict(self):
        """测试 OTS 最小化 camelCase dict"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.OTS,
        )
        object.__setattr__(
            kb,
            "provider_settings",
            {"otsInstanceName": "test-instance"},
        )

        data_api = kb._get_data_api()
        assert isinstance(data_api, OTSDataAPI)
        assert data_api.provider_settings.ots_instance_name == "test-instance"
        assert data_api.retrieve_settings is None

    def test_get_data_api_ots_without_settings(self):
        """测试 OTS 无设置"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.OTS,
        )

        data_api = kb._get_data_api()
        assert isinstance(data_api, OTSDataAPI)
        assert data_api.provider_settings is None
        assert data_api.retrieve_settings is None

    def test_get_data_api_ots_with_invalid_provider_settings_type(self):
        """测试 OTS 使用无效类型的 provider_settings"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.OTS,
        )
        object.__setattr__(kb, "provider_settings", "invalid")

        data_api = kb._get_data_api()
        assert isinstance(data_api, OTSDataAPI)
        assert data_api.provider_settings is None

    def test_get_data_api_ots_with_invalid_retrieve_settings_type(self):
        """测试 OTS 使用无效类型的 retrieve_settings"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.OTS,
            provider_settings=OTSProviderSettings(
                ots_instance_name="test-instance"
            ),
        )
        object.__setattr__(kb, "retrieve_settings", "invalid")

        data_api = kb._get_data_api()
        assert isinstance(data_api, OTSDataAPI)
        assert data_api.provider_settings is not None
        assert data_api.retrieve_settings is None

    def test_get_data_api_ots_with_string_provider(self):
        """测试使用字符串 provider='ots'"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider="ots",
            provider_settings=OTSProviderSettings(
                ots_instance_name="test-instance"
            ),
        )
        data_api = kb._get_data_api()
        assert isinstance(data_api, OTSDataAPI)

    def test_get_data_api_ots_embedding_with_api_key(self):
        """测试 camelCase dict 带 apiKey 的 embeddingConfiguration"""
        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.OTS,
        )
        object.__setattr__(
            kb,
            "provider_settings",
            {
                "otsInstanceName": "test-instance",
                "embeddingConfiguration": {
                    "provider": "custom",
                    "model": "my-model",
                    "dimension": 512,
                    "url": "https://custom.embedding.com",
                    "apiKey": "secret-key",
                },
            },
        )

        data_api = kb._get_data_api()
        ec = data_api.provider_settings.embedding_configuration
        assert ec.provider == "custom"
        assert ec.url == "https://custom.embedding.com"
        assert ec.api_key == "secret-key"


# =============================================================================
# KnowledgeBase.retrieve OTS 测试
# =============================================================================


class TestKnowledgeBaseRetrieveOTS:
    """测试 KnowledgeBase.retrieve 的 OTS 分支"""

    @patch("agentrun.knowledgebase.api.data.OTSDataAPI.retrieve")
    def test_retrieve_ots_sync(self, mock_retrieve):
        """测试 OTS 同步检索"""
        mock_retrieve.return_value = {
            "data": [{"content": "ots result", "score": 0.9}],
            "query": "test query",
            "knowledge_base_name": "test-kb",
        }

        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.OTS,
            provider_settings=OTSProviderSettings(
                ots_instance_name="test-instance"
            ),
        )

        result = kb.retrieve("test query")
        assert result["query"] == "test query"
        assert result["data"][0]["content"] == "ots result"

    @patch("agentrun.knowledgebase.api.data.OTSDataAPI.retrieve_async")
    @pytest.mark.asyncio
    async def test_retrieve_ots_async(self, mock_retrieve_async):
        """测试 OTS 异步检索"""
        mock_retrieve_async.return_value = {
            "data": [{"content": "ots async result", "score": 0.85}],
            "query": "async query",
            "knowledge_base_name": "test-kb",
        }

        kb = KnowledgeBase(
            knowledge_base_name="test-kb",
            provider=KnowledgeBaseProvider.OTS,
            provider_settings=OTSProviderSettings(
                ots_instance_name="test-instance"
            ),
        )

        result = await kb.retrieve_async("async query")
        assert result["query"] == "async query"
        assert result["data"][0]["content"] == "ots async result"


# =============================================================================
# OTS multi_retrieve 测试
# =============================================================================


class MockOTSKnowledgeBaseData:
    """模拟 OTS 知识库数据"""

    def to_map(self):
        return {
            "knowledgeBaseId": "kb-ots-001",
            "knowledgeBaseName": "test-ots-kb",
            "provider": "ots",
            "description": "Test OTS knowledge base",
            "providerSettings": {
                "otsInstanceName": "test-instance",
            },
            "retrieveSettings": {
                "searchType": ["DENSE_VECTOR"],
            },
            "createdAt": "2024-01-01T00:00:00Z",
            "lastUpdatedAt": "2024-01-01T00:00:00Z",
        }


class TestKnowledgeBaseMultiRetrieveOTS:
    """测试 OTS 参与 multi_retrieve"""

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @patch("agentrun.knowledgebase.api.data.OTSDataAPI.retrieve")
    def test_multi_retrieve_with_ots(
        self, mock_retrieve, mock_control_api_class
    ):
        """测试 OTS 参与同步多知识库检索"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base.return_value = (
            MockOTSKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        mock_retrieve.return_value = {
            "data": [{"content": "ots content"}],
            "query": "test query",
            "knowledge_base_name": "test-ots-kb",
        }

        result = KnowledgeBase.multi_retrieve(
            query="test query",
            knowledge_base_names=["test-ots-kb"],
        )

        assert "results" in result
        assert "test-ots-kb" in result["results"]
        assert (
            result["results"]["test-ots-kb"]["data"][0]["content"]
            == "ots content"
        )

    @patch("agentrun.knowledgebase.client.KnowledgeBaseControlAPI")
    @patch("agentrun.knowledgebase.api.data.OTSDataAPI.retrieve_async")
    @pytest.mark.asyncio
    async def test_multi_retrieve_async_with_ots(
        self, mock_retrieve_async, mock_control_api_class
    ):
        """测试 OTS 参与异步多知识库检索"""
        mock_control_api = MagicMock()
        mock_control_api.get_knowledge_base_async = AsyncMock(
            return_value=MockOTSKnowledgeBaseData()
        )
        mock_control_api_class.return_value = mock_control_api

        mock_retrieve_async.return_value = {
            "data": [{"content": "ots async content"}],
            "query": "test query",
            "knowledge_base_name": "test-ots-kb",
        }

        result = await KnowledgeBase.multi_retrieve_async(
            query="test query",
            knowledge_base_names=["test-ots-kb"],
        )

        assert "results" in result
        assert "test-ots-kb" in result["results"]

    def test_from_inner_object_ots(self):
        """测试从内部对象创建 OTS 知识库"""
        mock_data = MockOTSKnowledgeBaseData()
        kb = KnowledgeBase.from_inner_object(mock_data)

        assert kb.knowledge_base_id == "kb-ots-001"
        assert kb.knowledge_base_name == "test-ots-kb"
        assert kb.provider == "ots"
