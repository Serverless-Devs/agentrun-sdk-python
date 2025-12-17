"""OpenAI 协议处理器测试

测试 OpenAIProtocolHandler 的各种功能。
"""

import json
from typing import cast

from fastapi.testclient import TestClient
import pytest

from agentrun.server import (
    AgentEvent,
    AgentRequest,
    AgentRunServer,
    EventType,
    OpenAIProtocolHandler,
    ServerConfig,
)


class TestOpenAIProtocolHandler:
    """测试 OpenAIProtocolHandler"""

    def test_get_prefix_default(self):
        """测试默认前缀"""
        handler = OpenAIProtocolHandler()
        assert handler.get_prefix() == "/openai/v1"

    def test_get_prefix_custom(self):
        """测试自定义前缀"""
        from agentrun.server.model import OpenAIProtocolConfig

        config = ServerConfig(openai=OpenAIProtocolConfig(prefix="/custom/api"))
        handler = OpenAIProtocolHandler(config)
        assert handler.get_prefix() == "/custom/api"

    def test_get_model_name_default(self):
        """测试默认模型名称"""
        handler = OpenAIProtocolHandler()
        assert handler.get_model_name() == "agentrun"

    def test_get_model_name_custom(self):
        """测试自定义模型名称"""
        from agentrun.server.model import OpenAIProtocolConfig

        config = ServerConfig(
            openai=OpenAIProtocolConfig(model_name="custom-model")
        )
        handler = OpenAIProtocolHandler(config)
        assert handler.get_model_name() == "custom-model"


class TestOpenAIProtocolEndpoints:
    """测试 OpenAI 协议端点"""

    def get_client(self, invoke_agent):
        server = AgentRunServer(invoke_agent=invoke_agent)
        return TestClient(server.as_fastapi_app())

    @pytest.mark.asyncio
    async def test_list_models(self):
        """测试 /models 端点"""

        def invoke_agent(request: AgentRequest):
            return "Hello"

        client = self.get_client(invoke_agent)
        response = client.get("/openai/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "agentrun"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "agentrun"

    @pytest.mark.asyncio
    async def test_missing_messages_error(self):
        """测试缺少 messages 字段时返回错误"""

        def invoke_agent(request: AgentRequest):
            return "Hello"

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={"model": "test"},  # 缺少 messages
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
        assert "messages" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_message_format(self):
        """测试无效消息格式"""

        def invoke_agent(request: AgentRequest):
            return "Hello"

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={"messages": ["not a dict"]},  # 无效格式
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "Invalid message format" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_missing_role_in_message(self):
        """测试消息缺少 role 字段"""

        def invoke_agent(request: AgentRequest):
            return "Hello"

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={"messages": [{"content": "Hello"}]},  # 缺少 role
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "role" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_role(self):
        """测试无效的消息角色"""

        def invoke_agent(request: AgentRequest):
            return "Hello"

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={"messages": [{"role": "invalid_role", "content": "Hello"}]},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "Invalid message role" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_internal_error(self):
        """测试内部错误处理"""

        def invoke_agent(request: AgentRequest):
            raise RuntimeError("Internal error")

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "internal_error"

    @pytest.mark.asyncio
    async def test_non_stream_with_tool_calls(self):
        """测试非流式响应中的工具调用"""

        def invoke_agent(request: AgentRequest):
            return AgentEvent(
                event=EventType.TOOL_CALL,
                data={
                    "id": "tc-1",
                    "name": "test_tool",
                    "args": '{"param": "value"}',
                },
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        assert "tool_calls" in data["choices"][0]["message"]
        assert data["choices"][0]["message"]["tool_calls"][0]["id"] == "tc-1"

    @pytest.mark.asyncio
    async def test_non_stream_response_collection(self):
        """测试非流式响应收集流式结果"""

        async def invoke_agent(request: AgentRequest):
            yield "Hello"
            yield " World"

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_message_with_tool_calls(self):
        """测试解析带有 tool_calls 的消息"""

        captured_request = {}

        def invoke_agent(request: AgentRequest):
            captured_request["messages"] = request.messages
            return "Done"

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Beijing"}',
                            },
                        }],
                    },
                    {
                        "role": "tool",
                        "content": "Sunny",
                        "tool_call_id": "call_123",
                    },
                ],
            },
        )

        assert response.status_code == 200
        assert len(captured_request["messages"]) == 2
        assert captured_request["messages"][0].tool_calls is not None
        assert captured_request["messages"][0].tool_calls[0].id == "call_123"
        assert captured_request["messages"][1].tool_call_id == "call_123"

    @pytest.mark.asyncio
    async def test_parse_tools(self):
        """测试解析工具列表"""

        captured_request = {}

        def invoke_agent(request: AgentRequest):
            captured_request["tools"] = request.tools
            return "Done"

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }],
            },
        )

        assert response.status_code == 200
        assert captured_request["tools"] is not None
        assert len(captured_request["tools"]) == 1
        assert captured_request["tools"][0].function["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_parse_tools_with_non_dict(self):
        """测试解析工具列表时跳过非字典项"""

        captured_request = {}

        def invoke_agent(request: AgentRequest):
            captured_request["tools"] = request.tools
            return "Done"

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [
                    "invalid_tool",  # 非字典项，应该被跳过
                    {
                        "type": "function",
                        "function": {"name": "valid_tool"},
                    },
                ],
            },
        )

        assert response.status_code == 200
        assert captured_request["tools"] is not None
        assert len(captured_request["tools"]) == 1

    @pytest.mark.asyncio
    async def test_parse_tools_empty_after_filter(self):
        """测试解析工具列表后为空时返回 None"""

        captured_request = {}

        def invoke_agent(request: AgentRequest):
            captured_request["tools"] = request.tools
            return "Done"

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": ["invalid1", "invalid2"],  # 全部非字典项
            },
        )

        assert response.status_code == 200
        assert captured_request["tools"] is None

    @pytest.mark.asyncio
    async def test_addition_merge_overrides(self):
        """测试 addition 默认合并覆盖字段"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TEXT,
                data={"delta": "Hello"},
                addition={"custom": "value"},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

        # 第一个 chunk 应该包含 addition 字段
        first_line = lines[0]
        assert first_line.startswith("data: ")
        data = json.loads(first_line[6:])
        assert "custom" in data["choices"][0]["delta"]

    @pytest.mark.asyncio
    async def test_addition_protocol_only_mode(self):
        """测试 addition PROTOCOL_ONLY 模式"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TEXT,
                data={"delta": "Hello"},
                addition={
                    "content": "overwritten",  # 已存在的字段会被覆盖
                    "new_field": "ignored",  # 新字段会被忽略
                },
                addition_merge_options={"no_new_field": True},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

        first_line = lines[0]
        data = json.loads(first_line[6:])
        delta = data["choices"][0]["delta"]
        # content 被覆盖
        assert delta["content"] == "overwritten"
        # new_field 不存在（被忽略）
        assert "new_field" not in delta

    @pytest.mark.asyncio
    async def test_tool_call_with_addition(self):
        """测试工具调用时的 addition 处理"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "test", "args_delta": "{}"},
                addition={"custom_tool_field": "value"},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

        # 第二个 chunk 是参数增量，应该包含 addition
        args_line = lines[1]
        data = json.loads(args_line[6:])
        delta = data["choices"][0]["delta"]
        assert "custom_tool_field" in delta

    @pytest.mark.asyncio
    async def test_non_stream_multiple_tool_call_chunks(self):
        """测试非流式响应中多个工具调用 chunk 的合并"""

        def invoke_agent(request: AgentRequest):
            return [
                AgentEvent(
                    event=EventType.TOOL_CALL_CHUNK,
                    data={"id": "tc-1", "name": "tool1", "args_delta": '{"a":'},
                ),
                AgentEvent(
                    event=EventType.TOOL_CALL_CHUNK,
                    data={"id": "tc-1", "args_delta": "1}"},
                ),
            ]

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["arguments"] == '{"a":1}'


class TestOpenAIProtocolStreamBranches:
    """测试 OpenAI 协议流式响应的各种分支"""

    def get_client(self, invoke_agent):
        server = AgentRunServer(invoke_agent=invoke_agent)
        return TestClient(server.as_fastapi_app())

    @pytest.mark.asyncio
    async def test_stream_with_only_tool_calls(self):
        """测试只有工具调用没有文本的流式响应"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": "{}"},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

        # 最后一个非 [DONE] 行应该有 finish_reason: tool_calls
        for line in reversed(lines):
            if line.startswith("data: {"):
                data = json.loads(line[6:])
                if data["choices"][0].get("finish_reason"):
                    assert data["choices"][0]["finish_reason"] == "tool_calls"
                    break

    @pytest.mark.asyncio
    async def test_stream_with_empty_content(self):
        """测试空内容不会发送"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TEXT,
                data={"delta": ""},  # 空内容
            )
            yield AgentEvent(
                event=EventType.TEXT,
                data={"delta": "Hello"},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

        # 计算实际内容行数（排除 [DONE] 和 finish_reason 行）
        content_lines = []
        for line in lines:
            if line.startswith("data: {"):
                data = json.loads(line[6:])
                delta = data["choices"][0].get("delta", {})
                if delta.get("content"):
                    content_lines.append(data)

        # 只有一个非空内容
        assert len(content_lines) == 1

    @pytest.mark.asyncio
    async def test_stream_tool_call_without_args(self):
        """测试工具调用没有参数增量"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": ""},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

        # 应该有工具调用开始和结束
        assert len(lines) >= 2

    @pytest.mark.asyncio
    async def test_stream_multiple_tool_calls(self):
        """测试多个工具调用的流式响应"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-2", "name": "tool2", "args_delta": "{}"},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

        # 检查两个工具调用的索引
        tool_indices = set()
        for line in lines:
            if line.startswith("data: {"):
                data = json.loads(line[6:])
                delta = data["choices"][0].get("delta", {})
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        tool_indices.add(tc.get("index"))

        assert 0 in tool_indices
        assert 1 in tool_indices

    @pytest.mark.asyncio
    async def test_stream_raw_event(self):
        """测试 RAW 事件在流式响应中"""

        async def invoke_agent(request: AgentRequest):
            yield "Hello"
            yield AgentEvent(
                event=EventType.RAW,
                data={"raw": "custom raw data"},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        content = response.text
        assert "custom raw data" in content

    @pytest.mark.asyncio
    async def test_stream_tool_result_ignored(self):
        """测试 TOOL_RESULT 事件在流式响应中被忽略"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool1", "args_delta": "{}"},
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "result"},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

        # TOOL_RESULT 不应该出现在响应中
        for line in lines:
            if line.startswith("data: {"):
                data = json.loads(line[6:])
                # 检查没有 tool_result 相关内容
                assert "result" not in str(data)


class TestOpenAIProtocolNonStreamBranches:
    """测试 OpenAI 协议非流式响应的各种分支"""

    def get_client(self, invoke_agent):
        server = AgentRunServer(invoke_agent=invoke_agent)
        return TestClient(server.as_fastapi_app())

    @pytest.mark.asyncio
    async def test_non_stream_with_multiple_tools(self):
        """测试非流式响应中多个工具调用"""

        def invoke_agent(request: AgentRequest):
            return [
                AgentEvent(
                    event=EventType.TOOL_CALL_CHUNK,
                    data={"id": "tc-1", "name": "tool1", "args_delta": "{}"},
                ),
                AgentEvent(
                    event=EventType.TOOL_CALL_CHUNK,
                    data={
                        "id": "tc-2",
                        "name": "tool2",
                        "args_delta": '{"x": 1}',
                    },
                ),
            ]

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 2

    @pytest.mark.asyncio
    async def test_non_stream_with_empty_tool_id(self):
        """测试非流式响应中空工具 ID"""

        def invoke_agent(request: AgentRequest):
            return [
                AgentEvent(
                    event=EventType.TOOL_CALL_CHUNK,
                    data={
                        "id": "",
                        "name": "tool1",
                        "args_delta": "{}",
                    },  # 空 ID
                ),
            ]

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        # 空 ID 的工具调用不会被添加
        assert data["choices"][0]["message"].get("tool_calls") is None

    @pytest.mark.asyncio
    async def test_non_stream_with_empty_args_delta(self):
        """测试非流式响应中空参数增量"""

        def invoke_agent(request: AgentRequest):
            return [
                AgentEvent(
                    event=EventType.TOOL_CALL_CHUNK,
                    data={"id": "tc-1", "name": "tool1", "args_delta": ""},
                ),
            ]

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["arguments"] == ""

    @pytest.mark.asyncio
    async def test_non_stream_with_text_events(self):
        """测试非流式响应中的文本事件"""

        def invoke_agent(request: AgentRequest):
            return [
                AgentEvent(event=EventType.TEXT, data={"delta": "Hello"}),
                AgentEvent(event=EventType.TEXT, data={"delta": " World"}),
            ]

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello World"
        assert data["choices"][0]["finish_reason"] == "stop"


class TestOpenAIProtocolApplyAddition:
    """测试 _apply_addition 方法"""

    def test_apply_addition_default_merge(self):
        """默认合并应覆盖已有字段并保留原字段"""
        handler = OpenAIProtocolHandler()

        delta = {"content": "Hello", "role": "assistant"}
        addition = {"content": "overwritten", "new_field": "added"}

        result = handler._apply_addition(
            delta.copy(),
            addition.copy(),
        )

        assert result["content"] == "overwritten"
        assert result["new_field"] == "added"
        assert result["role"] == "assistant"

    def test_apply_addition_merge_options_none(self):
        """显式传入 merge_options=None 仍按默认合并"""
        handler = OpenAIProtocolHandler()

        delta = {"content": "Hello", "role": "assistant"}
        addition = {"content": "overwritten", "new_field": "added"}

        result = handler._apply_addition(
            delta.copy(),
            addition.copy(),
        )

        assert result["content"] == "overwritten"
        assert result["new_field"] == "added"

    def test_apply_addition_protocol_only_mode(self):
        """测试 PROTOCOL_ONLY 模式（覆盖 527->530 分支）"""
        handler = OpenAIProtocolHandler()

        delta = {"content": "Hello", "role": "assistant"}
        addition = {"content": "overwritten", "new_field": "ignored"}

        result = handler._apply_addition(
            delta.copy(),
            addition.copy(),
            {"no_new_field": True},
        )

        # content 被覆盖
        assert result["content"] == "overwritten"
        # new_field 不存在（被忽略）
        assert "new_field" not in result
        # role 保持不变
        assert result["role"] == "assistant"


class TestOpenAIProtocolRawEvent:
    """测试 RAW 事件处理"""

    def get_client(self, invoke_agent):
        server = AgentRunServer(invoke_agent=invoke_agent)
        return TestClient(server.as_fastapi_app())

    @pytest.mark.asyncio
    async def test_raw_event_with_newline(self):
        """测试 RAW 事件已有换行符"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.RAW,
                data={"raw": "custom data\n\n"},  # 已有换行符
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        content = response.text
        assert "custom data" in content

    @pytest.mark.asyncio
    async def test_raw_event_empty(self):
        """测试空 RAW 事件"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.RAW,
                data={"raw": ""},  # 空内容
            )
            yield "Hello"

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_tool_call_chunk_without_id(self):
        """测试没有 id 的 TOOL_CALL_CHUNK"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "", "name": "tool", "args_delta": "{}"},  # 空 id
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_tool_call_chunk_existing_id(self):
        """测试已存在 id 的 TOOL_CALL_CHUNK"""

        async def invoke_agent(request: AgentRequest):
            # 第一个 chunk
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "name": "tool", "args_delta": '{"a":'},
            )
            # 第二个 chunk（同一个 id）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={"id": "tc-1", "args_delta": "1}"},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

        # 应该有多个工具调用相关的行
        tool_lines = [
            line
            for line in lines
            if line.startswith("data: {") and "tool_calls" in line
        ]
        assert len(tool_lines) >= 2

    @pytest.mark.asyncio
    async def test_stream_no_text_no_tools(self):
        """测试没有文本也没有工具调用的流式响应"""

        async def invoke_agent(request: AgentRequest):
            # 只发送其他类型的事件
            yield AgentEvent(
                event=EventType.CUSTOM,
                data={"name": "test", "value": {}},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

        # 最后应该是 [DONE]
        assert lines[-1] == "data: [DONE]"

    @pytest.mark.asyncio
    async def test_non_stream_tool_call_without_args(self):
        """测试非流式响应中没有参数增量的工具调用"""

        def invoke_agent(request: AgentRequest):
            return [
                AgentEvent(
                    event=EventType.TOOL_CALL_CHUNK,
                    data={
                        "id": "tc-1",
                        "name": "tool1",
                        "args_delta": "",
                    },  # 空参数
                ),
            ]

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["arguments"] == ""
