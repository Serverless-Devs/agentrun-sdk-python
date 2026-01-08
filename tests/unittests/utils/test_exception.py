"""测试 agentrun.utils.exception 模块 / Test agentrun.utils.exception module"""

import pytest

from agentrun.utils.exception import (
    AgentRunError,
    ClientError,
    DeleteResourceError,
    HTTPError,
    ResourceAlreadyExistError,
    ResourceNotExistError,
    ServerError,
)


class TestAgentRunError:
    """测试 AgentRunError 基类"""

    def test_init_with_message_only(self):
        """测试只传入消息的初始化"""
        error = AgentRunError("Test error message")
        assert error.message == "Test error message"
        assert str(error) == "Test error message"
        assert error.details == {}

    def test_init_with_kwargs(self):
        """测试带有额外参数的初始化"""
        error = AgentRunError("Test error", key1="value1", key2=123)
        assert error.message == "Test error"
        assert error.details == {"key1": "value1", "key2": 123}

    def test_kwargs_str_with_empty_kwargs(self):
        """测试空 kwargs 的字符串表示"""
        result = AgentRunError.kwargs_str()
        assert result == ""

    def test_kwargs_str_with_values(self):
        """测试带值的 kwargs 字符串表示"""
        result = AgentRunError.kwargs_str(name="test", count=5)
        # 验证是 JSON 格式
        import json

        parsed = json.loads(result)
        assert parsed["name"] == "test"
        assert parsed["count"] == 5

    def test_details_str(self):
        """测试 details_str 方法"""
        error = AgentRunError("Error", detail="info")
        result = error.details_str()
        import json

        parsed = json.loads(result)
        assert parsed["detail"] == "info"


class TestHTTPError:
    """测试 HTTPError 异常类"""

    def test_init(self):
        """测试初始化"""
        error = HTTPError(
            status_code=404,
            message="Not Found",
            request_id="req-123",
            extra="info",
        )
        assert error.status_code == 404
        assert error.message == "Not Found"
        assert error.request_id == "req-123"
        assert error.details["extra"] == "info"

    def test_str(self):
        """测试字符串表示"""
        error = HTTPError(
            status_code=500, message="Internal Error", request_id="req-456"
        )
        result = str(error)
        assert "HTTP 500" in result
        assert "Internal Error" in result
        assert "req-456" in result

    def test_to_resource_error_not_found(self):
        """测试转换为 ResourceNotExistError (does not exist)"""
        error = HTTPError(
            status_code=404,
            message="Resource does not exist",
            request_id="req-1",
        )
        result = error.to_resource_error("Agent", "agent-123")
        assert isinstance(result, ResourceNotExistError)

    def test_to_resource_error_not_found_alternative(self):
        """测试转换为 ResourceNotExistError (not found)"""
        error = HTTPError(
            status_code=404, message="Resource not found", request_id="req-1"
        )
        result = error.to_resource_error("Agent", "agent-123")
        assert isinstance(result, ResourceNotExistError)

    def test_to_resource_error_already_exists_400(self):
        """测试转换为 ResourceAlreadyExistError (400)"""
        error = HTTPError(
            status_code=400,
            message="Resource already exists",
            request_id="req-1",
        )
        result = error.to_resource_error("Agent", "agent-123")
        assert isinstance(result, ResourceAlreadyExistError)

    def test_to_resource_error_already_exists_409(self):
        """测试转换为 ResourceAlreadyExistError (409)"""
        error = HTTPError(
            status_code=409,
            message="Resource already exists",
            request_id="req-1",
        )
        result = error.to_resource_error("Agent", "agent-123")
        assert isinstance(result, ResourceAlreadyExistError)

    def test_to_resource_error_already_exists_500(self):
        """测试转换为 ResourceAlreadyExistError (500, for ModelProxy)"""
        error = HTTPError(
            status_code=500,
            message="Resource already exists",
            request_id="req-1",
        )
        result = error.to_resource_error("ModelProxy", "proxy-123")
        assert isinstance(result, ResourceAlreadyExistError)

    def test_to_resource_error_returns_self(self):
        """测试不匹配时返回自身"""
        error = HTTPError(
            status_code=500, message="Server error", request_id="req-1"
        )
        result = error.to_resource_error("Agent", "agent-123")
        assert result is error


class TestClientError:
    """测试 ClientError 异常类"""

    def test_init(self):
        """测试初始化"""
        error = ClientError(
            status_code=400,
            message="Bad Request",
            request_id="req-789",
            field="value",
        )
        assert error.status_code == 400
        assert error.message == "Bad Request"
        assert error.request_id == "req-789"


class TestServerError:
    """测试 ServerError 异常类"""

    def test_init(self):
        """测试初始化"""
        error = ServerError(
            status_code=503, message="Service Unavailable", request_id="req-000"
        )
        assert error.status_code == 503
        assert error.message == "Service Unavailable"
        assert error.request_id == "req-000"


class TestResourceNotExistError:
    """测试 ResourceNotExistError 异常类"""

    def test_init(self):
        """测试初始化"""
        error = ResourceNotExistError(
            resource_type="AgentRuntime", resource_id="runtime-123"
        )
        assert "AgentRuntime" in str(error)
        assert "runtime-123" in str(error)
        assert "does not exist" in str(error)

    def test_init_without_id(self):
        """测试不带 ID 的初始化"""
        error = ResourceNotExistError(resource_type="AgentRuntime")
        assert "AgentRuntime" in str(error)


class TestResourceAlreadyExistError:
    """测试 ResourceAlreadyExistError 异常类"""

    def test_init(self):
        """测试初始化"""
        error = ResourceAlreadyExistError(
            resource_type="AgentRuntime", resource_id="runtime-456"
        )
        assert "AgentRuntime" in str(error)
        assert "runtime-456" in str(error)
        assert "already exists" in str(error)

    def test_init_without_id(self):
        """测试不带 ID 的初始化"""
        error = ResourceAlreadyExistError(resource_type="AgentRuntime")
        assert "AgentRuntime" in str(error)


class TestDeleteResourceError:
    """测试 DeleteResourceError 异常类"""

    def test_init_without_message(self):
        """测试不带消息的初始化"""
        error = DeleteResourceError()
        assert "Failed to delete resource" in str(error)

    def test_init_with_message(self):
        """测试带消息的初始化"""
        error = DeleteResourceError(message="Resource is locked")
        result = str(error)
        assert "Failed to delete resource" in result
        assert "Resource is locked" in result
