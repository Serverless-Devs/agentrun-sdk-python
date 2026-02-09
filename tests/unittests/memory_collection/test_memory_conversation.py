"""Tests for AgentRun Memory Conversation / AgentRun 记忆对话测试"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from agentrun.memory_collection import MemoryConversation
from agentrun.server.model import AgentRequest, Message, MessageRole


@pytest.fixture
def mock_memory_collection():
    """Mock MemoryCollection"""
    with patch("agentrun.memory_collection.MemoryCollection") as mock:
        # Mock get_by_name_async
        mock_collection = MagicMock()
        mock_collection.vector_store_config = MagicMock()
        mock_collection.vector_store_config.provider = "aliyun_tablestore"
        mock_collection.vector_store_config.config = MagicMock()
        mock_collection.vector_store_config.config.endpoint = (
            "https://test.cn-hangzhou.ots.aliyuncs.com"
        )
        mock_collection.vector_store_config.config.instance_name = (
            "test-instance"
        )

        mock.get_by_name_async = AsyncMock(return_value=mock_collection)
        yield mock


@pytest.fixture
def mock_memory_store():
    """Mock AsyncMemoryStore"""
    with patch(
        "tablestore_for_agent_memory.memory.async_memory_store.AsyncMemoryStore"
    ) as mock_store_class:
        mock_store = AsyncMock()
        mock_store.put_session = AsyncMock()
        mock_store.put_message = AsyncMock()
        mock_store.update_session = AsyncMock()
        mock_store.init_table = AsyncMock()
        mock_store.init_search_index = AsyncMock()
        mock_store_class.return_value = mock_store
        yield mock_store


@pytest.fixture
def mock_ots_client():
    """Mock AsyncOTSClient"""
    with patch("tablestore.AsyncOTSClient") as mock:
        mock_client = AsyncMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_request():
    """Create a mock Starlette Request"""
    mock_req = Mock()
    mock_headers = Mock()
    mock_headers.get = Mock(return_value="user123")
    mock_query = Mock()
    mock_query.get = Mock(return_value=None)

    mock_req.headers = mock_headers
    mock_req.query_params = mock_query
    mock_req.client = None

    return mock_req


class TestMemoryConversation:
    """Test MemoryConversation class"""

    def test_default_user_id_extractor(self, mock_request):
        """Test default user_id extraction"""
        # Test with X-User-ID header
        request = AgentRequest.model_construct(
            messages=[],
            raw_request=mock_request,
        )

        user_id = MemoryConversation._default_user_id_extractor(request)
        assert user_id == "user123"

    def test_default_user_id_extractor_fallback(self):
        """Test user_id extraction fallback to default"""
        request = AgentRequest(messages=[])

        user_id = MemoryConversation._default_user_id_extractor(request)
        assert user_id == "default_user"

    def test_default_session_id_extractor(self):
        """Test default session_id extraction"""
        # Test with X-Session-ID header
        mock_req = Mock()
        mock_headers = Mock()
        mock_headers.get = Mock(return_value="session456")
        mock_query = Mock()
        mock_query.get = Mock(return_value=None)

        mock_req.headers = mock_headers
        mock_req.query_params = mock_query

        request = AgentRequest.model_construct(
            messages=[],
            raw_request=mock_req,
        )

        session_id = MemoryConversation._default_session_id_extractor(request)
        assert session_id == "session456"

    def test_default_session_id_extractor_from_message(self):
        """Test session_id extraction from message ID"""
        request = AgentRequest(
            messages=[
                Message(id="msg123", role=MessageRole.USER, content="Hello")
            ]
        )

        session_id = MemoryConversation._default_session_id_extractor(request)
        assert session_id == "session_msg123"

    def test_default_session_id_extractor_generate(self):
        """Test session_id generation"""
        request = AgentRequest(messages=[])

        session_id = MemoryConversation._default_session_id_extractor(request)
        assert session_id.startswith("session_")

    def test_extract_message_content_string(self):
        """Test extracting string content"""
        content = "Hello, world!"
        result = MemoryConversation._extract_message_content(content)
        assert result == "Hello, world!"

    def test_extract_message_content_multimodal(self):
        """Test extracting multimodal content"""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "image", "url": "https://example.com/image.jpg"},
            {"type": "text", "text": "World"},
        ]
        result = MemoryConversation._extract_message_content(content)
        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_wrap_invoke_agent_basic(
        self, mock_memory_collection, mock_memory_store, mock_ots_client
    ):
        """Test basic wrap_invoke_agent functionality"""
        # Create MemoryConversation
        memory = MemoryConversation(memory_collection_name="test-memory")

        # Mock agent handler
        async def mock_agent(request: AgentRequest):
            yield "Hello"
            yield ", "
            yield "world!"

        # Create request
        request = AgentRequest(
            messages=[Message(role=MessageRole.USER, content="Hi there")]
        )

        # Wrap and collect results
        results = []
        async for event in memory.wrap_invoke_agent(request, mock_agent):
            results.append(event)

        # Verify results
        assert results == ["Hello", ", ", "world!"]

        # Verify memory store calls
        assert mock_memory_store.put_session.called
        assert mock_memory_store.put_message.called
        assert mock_memory_store.update_session.called

    @pytest.mark.asyncio
    async def test_wrap_invoke_agent_with_custom_extractors(
        self, mock_memory_collection, mock_memory_store, mock_ots_client
    ):
        """Test wrap_invoke_agent with custom extractors"""

        # Custom extractors
        def custom_user_extractor(req: AgentRequest) -> str:
            return "custom_user"

        def custom_session_extractor(req: AgentRequest) -> str:
            return "custom_session"

        # Create MemoryConversation with custom extractors
        memory = MemoryConversation(
            memory_collection_name="test-memory",
            user_id_extractor=custom_user_extractor,
            session_id_extractor=custom_session_extractor,
        )

        # Mock agent handler
        async def mock_agent(request: AgentRequest):
            yield "Response"

        # Create request
        request = AgentRequest(
            messages=[Message(role=MessageRole.USER, content="Test")]
        )

        # Wrap and collect results
        results = []
        async for event in memory.wrap_invoke_agent(request, mock_agent):
            results.append(event)

        # Verify results
        assert results == ["Response"]

    @pytest.mark.asyncio
    async def test_wrap_invoke_agent_handles_errors(
        self, mock_memory_collection, mock_memory_store, mock_ots_client
    ):
        """Test that memory errors don't break agent responses"""
        # Make memory store raise error
        mock_memory_store.put_session.side_effect = Exception("Storage error")

        # Create MemoryConversation
        memory = MemoryConversation(memory_collection_name="test-memory")

        # Mock agent handler
        async def mock_agent(request: AgentRequest):
            yield "Still works!"

        # Create request
        request = AgentRequest(
            messages=[Message(role=MessageRole.USER, content="Test")]
        )

        # Wrap and collect results - should still work
        results = []
        async for event in memory.wrap_invoke_agent(request, mock_agent):
            results.append(event)

        # Verify agent still responds
        assert results == ["Still works!"]

    @pytest.mark.asyncio
    async def test_wrap_invoke_agent_without_dependencies(self):
        """Test graceful fallback when dependencies not installed"""
        memory = MemoryConversation(memory_collection_name="test-memory")

        # Force _memory_store to None to simulate uninitialized state
        memory._memory_store = None

        # Mock _get_memory_store to raise ImportError
        async def mock_get_memory_store():
            raise ImportError("Module not found")

        memory._get_memory_store = mock_get_memory_store

        async def mock_agent(request: AgentRequest):
            yield "Response"

        request = AgentRequest(
            messages=[Message(role=MessageRole.USER, content="Test")]
        )

        # Should still work, just without storage
        results = []
        async for event in memory.wrap_invoke_agent(request, mock_agent):
            results.append(event)

        assert results == ["Response"]
