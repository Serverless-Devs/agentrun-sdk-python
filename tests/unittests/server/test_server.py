import asyncio

from agentrun.server.model import AgentRequest, MessageRole
from agentrun.server.server import AgentRunServer


class TestServer:

    def get_invoke_agent_non_streaming(self):

        def invoke_agent(request: AgentRequest):
            # 检查请求消息，返回预期的响应
            user_message = next(
                (
                    msg.content
                    for msg in request.messages
                    if msg.role == MessageRole.USER
                ),
                "Hello",
            )

            return f"You said: {user_message}"

        return invoke_agent

    def get_invoke_agent_streaming(self):

        async def streaming_invoke_agent(request: AgentRequest):
            yield "Hello, "
            await asyncio.sleep(0.01)  # 短暂延迟
            yield "this is "
            await asyncio.sleep(0.01)
            yield "a test."

        return streaming_invoke_agent

    def get_non_streaming_client(self):
        server = AgentRunServer(
            invoke_agent=self.get_invoke_agent_non_streaming()
        )
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        return TestClient(app)

    def get_streaming_client(self):
        server = AgentRunServer(invoke_agent=self.get_invoke_agent_streaming())
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        return TestClient(app)

    def parse_streaming_line(self, line: str):
        """解析流式响应行，去除前缀 'data: ' 并转换为 JSON"""
        import json

        assert line.startswith("data: ")
        json_str = line[len("data: ") :]
        return json.loads(json_str)

    async def test_server_non_streaming_openai(self):
        """测试服务器基本功能"""

        client = self.get_non_streaming_client()

        # 发送请求
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "AgentRun"}],
                "model": "test-model",
            },
        )

        # 检查响应状态
        assert response.status_code == 200

        # 检查响应内容
        response_data = response.json()

        # 验证响应结构（忽略动态生成的 id 和 created）
        assert response_data["object"] == "chat.completion"
        assert response_data["model"] == "test-model"
        assert "id" in response_data
        assert response_data["id"].startswith("chatcmpl-")
        assert "created" in response_data
        assert isinstance(response_data["created"], int)
        assert response_data["choices"] == [{
            "index": 0,
            "message": {"role": "assistant", "content": "You said: AgentRun"},
            "finish_reason": "stop",
        }]

    async def test_server_streaming_openai(self):
        """测试服务器流式响应功能"""

        client = self.get_streaming_client()

        # 发送流式请求
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "AgentRun"}],
                "model": "test-model",
                "stream": True,
            },
        )

        # 检查响应状态
        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines()]

        # 过滤空行
        lines = [line for line in lines if line]

        # OpenAI 流式格式：第一个 chunk 是 role 声明，后续是内容
        # 格式：data: {...}
        assert (
            len(lines) >= 4
        ), f"Expected at least 4 lines, got {len(lines)}: {lines}"
        assert lines[0].startswith("data: {")

        # 验证所有内容都在响应中（可能在不同的 chunk 中）
        all_content = "".join(lines)
        assert "Hello, " in all_content
        assert "this is " in all_content
        assert "a test." in all_content
        assert lines[-1] == "data: [DONE]"

    async def test_server_streaming_agui(self):
        """测试服务器 AG-UI 流式响应功能"""

        client = self.get_streaming_client()

        # 发送流式请求
        response = client.post(
            "/ag-ui/agent",
            json={
                "messages": [{"role": "user", "content": "AgentRun"}],
                "model": "test-model",
                "stream": True,
            },
        )

        # 检查响应状态
        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines()]

        # 过滤空行
        lines = [line for line in lines if line]

        # AG-UI 流式格式：每个 chunk 是一个 JSON 对象
        assert (
            len(lines) == 7
        ), f"Expected at least 3 lines, got {len(lines)}: {lines}"

        assert lines[0].startswith("data: {")
        line0 = self.parse_streaming_line(lines[0])
        assert line0["type"] == "RUN_STARTED"
        assert line0["runId"]
        assert line0["threadId"]

        thread_id = line0["threadId"]
        run_id = line0["runId"]

        assert lines[1].startswith("data: {")
        line1 = self.parse_streaming_line(lines[1])
        assert line1["type"] == "TEXT_MESSAGE_START"
        assert line1["messageId"]
        assert line1["role"] == "assistant"

        message_id = line1["messageId"]

        assert lines[2].startswith("data: {")
        line2 = self.parse_streaming_line(lines[2])
        assert line2["type"] == "TEXT_MESSAGE_CONTENT"
        assert line2["messageId"] == message_id
        assert line2["delta"] == "Hello, "

        assert lines[3].startswith("data: {")
        line3 = self.parse_streaming_line(lines[3])
        assert line3["type"] == "TEXT_MESSAGE_CONTENT"
        assert line3["messageId"] == message_id
        assert line3["delta"] == "this is "

        assert lines[4].startswith("data: {")
        line4 = self.parse_streaming_line(lines[4])
        assert line4["type"] == "TEXT_MESSAGE_CONTENT"
        assert line4["messageId"] == message_id
        assert line4["delta"] == "a test."

        assert lines[5].startswith("data: {")
        line5 = self.parse_streaming_line(lines[5])
        assert line5["type"] == "TEXT_MESSAGE_END"
        assert line5["messageId"] == message_id

        assert lines[6].startswith("data: {")
        line6 = self.parse_streaming_line(lines[6])
        assert line6["type"] == "RUN_FINISHED"
        assert line6["runId"] == run_id
        assert line6["threadId"] == thread_id

        all_text = ""
        for line in lines:
            assert line.startswith("data: ")
            assert line.endswith("}")
            data = self.parse_streaming_line(line)
            if data["type"] == "TEXT_MESSAGE_CONTENT":
                all_text += data["delta"]

        assert all_text == "Hello, this is a test."

    async def test_server_agui_stream_data_event(self):
        """测试 STREAM_DATA 事件直接返回原始数据（OpenAI 和 AG-UI 协议）"""
        from agentrun.server import (
            AgentRequest,
            AgentResult,
            AgentRunServer,
            EventType,
        )

        async def streaming_invoke_agent(request: AgentRequest):
            # 测试 STREAM_DATA 事件
            yield AgentResult(
                event=EventType.STREAM_DATA,
                data={"raw": '{"custom": "data"}'},
            )

        server = AgentRunServer(invoke_agent=streaming_invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # OpenAI Chat Completions（必须设置 stream=True）
        response_openai = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )

        assert response_openai.status_code == 200
        lines = [line async for line in response_openai.aiter_lines()]
        lines = [line for line in lines if line]

        # OpenAI 流式响应：STREAM_DATA 的原始数据 + [DONE]
        # STREAM_DATA 输出: data: {"custom": "data"}
        # RUN_FINISHED 输出: data: [DONE]
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}: {lines}"
        assert '{"custom": "data"}' in lines[0]
        assert lines[1] == "data: [DONE]"

        # AG-UI 协议
        response_agui = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        assert response_agui.status_code == 200
        lines = [line async for line in response_agui.aiter_lines()]
        lines = [line for line in lines if line]

        # AG-UI 流式响应：RUN_STARTED + STREAM_DATA + RUN_FINISHED
        assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}: {lines}"

        # 验证 RUN_STARTED
        line0 = self.parse_streaming_line(lines[0])
        assert line0["type"] == "RUN_STARTED"

        # 验证 STREAM_DATA 的原始内容被正确输出
        assert '{"custom": "data"}' in lines[1]

        # 验证 RUN_FINISHED
        line2 = self.parse_streaming_line(lines[2])
        assert line2["type"] == "RUN_FINISHED"

    async def test_server_agui_addition_merge(self):
        """测试 addition 字段的合并功能"""
        from agentrun.server import (
            AdditionMode,
            AgentRequest,
            AgentResult,
            AgentRunServer,
            EventType,
        )

        async def streaming_invoke_agent(request: AgentRequest):
            yield AgentResult(
                event=EventType.TEXT_MESSAGE_CONTENT,
                data={"message_id": "msg_1", "delta": "Hello"},
                addition={"custom_field": "custom_value"},
                addition_mode=AdditionMode.MERGE,
            )

        server = AgentRunServer(invoke_agent=streaming_invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)

        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines()]
        lines = [line for line in lines if line]

        # 查找包含 TEXT_MESSAGE_CONTENT 的行
        found_custom_field = False
        for line in lines:
            if "TEXT_MESSAGE_CONTENT" in line:
                data = self.parse_streaming_line(line)
                if data.get("custom_field") == "custom_value":
                    found_custom_field = True
                    break

        assert found_custom_field, "addition 字段应该被合并到事件中"
