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
        """测试非流式的 OpenAI 服务器响应功能"""

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
        """测试流式的 OpenAI 服务器响应功能"""

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
        assert len(lines) == 5

        assert lines[0].startswith("data: {")
        line0 = self.parse_streaming_line(lines[0])
        assert line0["id"].startswith("chatcmpl-")
        assert line0["object"] == "chat.completion.chunk"
        assert line0["model"] == "test-model"
        assert line0["choices"][0]["delta"] == {
            "role": "assistant",
            "content": "Hello, ",
        }

        event_id = line0["id"]

        assert lines[1].startswith("data: {")
        line1 = self.parse_streaming_line(lines[1])
        assert line1["id"] == event_id
        assert line1["object"] == "chat.completion.chunk"
        assert line1["model"] == "test-model"
        assert line1["choices"][0]["delta"] == {"content": "this is "}

        assert lines[2].startswith("data: {")
        line2 = self.parse_streaming_line(lines[2])
        assert line2["id"] == event_id
        assert line2["object"] == "chat.completion.chunk"
        assert line2["model"] == "test-model"
        assert line2["choices"][0]["delta"] == {"content": "a test."}

        assert lines[3].startswith("data: {")
        line3 = self.parse_streaming_line(lines[3])
        assert line3["id"] == event_id
        assert line3["object"] == "chat.completion.chunk"
        assert line3["model"] == "test-model"
        assert line3["choices"][0]["delta"] == {}

        assert lines[4] == "data: [DONE]"

        all_text = ""
        for line in lines:
            if line.startswith("data: {"):
                data = self.parse_streaming_line(line)
                all_text += data["choices"][0]["delta"].get("content", "")

        assert all_text == "Hello, this is a test."

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
        assert len(lines) == 7

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

    async def test_server_raw_event_agui(self):
        """测试 RAW 事件直接返回原始数据（OpenAI 和 AG-UI 协议）

        RAW 事件可以在任何时间触发，输出原始 SSE 内容，不影响其他事件的正常处理。
        支持任意 SSE 格式（data:, :注释, 等）。
        """
        from agentrun.server import (
            AgentEvent,
            AgentRequest,
            AgentRunServer,
            EventType,
        )

        async def streaming_invoke_agent(request: AgentRequest):
            # 测试 RAW 事件与其他事件混合
            yield "你好"
            yield AgentEvent(
                event=EventType.RAW,
                data={"raw": '{"custom": "data"}'},
            )
            yield AgentEvent(event=EventType.TEXT, data={"delta": "再见"})

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

        # OpenAI 流式响应：
        # 1. role: assistant + content: 你好（合并在首个 chunk）
        # 2. RAW: {"custom": "data"}
        # 3. content: 再见
        # 4. finish_reason: stop
        # 5. [DONE]
        assert len(lines) == 5

        # 验证 RAW 事件在中间正确输出
        raw_found = False
        for line in lines:
            if '{"custom": "data"}' in line:
                raw_found = True
                break
        assert raw_found, "RAW 事件内容应该在响应中"

        # 验证最后是 [DONE]
        assert lines[-1] == "data: [DONE]"

        # AG-UI 协议
        response_agui = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        # 检查响应状态
        assert response_agui.status_code == 200
        lines = [line async for line in response_agui.aiter_lines()]

        # 过滤空行
        lines = [line for line in lines if line]

        # AG-UI 流式格式：每个 chunk 是一个 JSON 对象
        # 预期格式：RUN_STARTED + TEXT_MESSAGE_START + TEXT_MESSAGE_CONTENT(你好) + RAW + TEXT_MESSAGE_CONTENT(再见) + TEXT_MESSAGE_END + RUN_FINISHED
        assert len(lines) == 7  # 6 个标准事件 + 1 个 RAW 事件

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
        assert line2["delta"] == "你好"

        # 第 3 行是 RAW 事件，不带 data: 前缀
        assert lines[3] == '{"custom": "data"}'

        assert lines[4].startswith("data: {")
        line4 = self.parse_streaming_line(lines[4])
        assert line4["type"] == "TEXT_MESSAGE_CONTENT"
        assert line4["messageId"] == message_id
        assert line4["delta"] == "再见"

        assert lines[5].startswith("data: {")
        line5 = self.parse_streaming_line(lines[5])
        assert line5["type"] == "TEXT_MESSAGE_END"
        assert line5["messageId"] == message_id

        assert lines[6].startswith("data: {")
        line6 = self.parse_streaming_line(lines[6])
        assert line6["type"] == "RUN_FINISHED"
        assert line6["runId"] == run_id
        assert line6["threadId"] == thread_id

        # 验证所有文本内容
        all_text = ""
        for line in lines:
            if line.startswith("data: "):
                data = self.parse_streaming_line(line)
                if data["type"] == "TEXT_MESSAGE_CONTENT":
                    all_text += data["delta"]

        assert all_text == "你好再见"

    async def test_server_raw_event_openai(self):
        """测试 OpenAI 协议中 RAW 事件的功能

        验证 RAW 事件在 OpenAI 协议中的行为，确保与其他 OpenAI 事件混合时能正确处理。
        """
        from agentrun.server import (
            AgentEvent,
            AgentRequest,
            AgentRunServer,
            EventType,
        )

        async def streaming_invoke_agent(request: AgentRequest):
            # 测试 RAW 事件与其他事件混合
            yield "你好"
            yield AgentEvent(
                event=EventType.RAW,
                data={
                    "raw": '{"custom": "data"}\n\n'
                },  # RAW 事件需要使用 raw 键，并且应该是完整的 SSE 格式
            )
            yield AgentEvent(event=EventType.TEXT, data={"delta": "再见"})

        server = AgentRunServer(invoke_agent=streaming_invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # OpenAI Chat Completions（必须设置 stream=True）
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "model": "test-model",
                "stream": True,
            },
        )

        # 检查响应状态
        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines()]

        # 过滤空行
        lines = [line for line in lines if line]

        # OpenAI 流式格式：第一个 chunk 是 role 声明，后续是内容，然后是完成事件
        # 预期格式：role + 你好 + RAW 事件 + 再见 + finish_reason + [DONE] - 共5行（没有空行）
        assert len(lines) == 5

        # 验证第一个 chunk 包含 role 和初始内容
        assert lines[0].startswith("data: {")
        line0 = self.parse_streaming_line(lines[0])
        assert line0["id"].startswith("chatcmpl-")
        assert line0["object"] == "chat.completion.chunk"
        assert line0["model"] == "test-model"
        assert line0["choices"][0]["delta"] == {
            "role": "assistant",
            "content": "你好",
        }

        event_id = line0["id"]

        # 第二行是 RAW 事件，不带 data: 前缀，直接输出原始数据
        assert lines[1] == '{"custom": "data"}'

        # 验证第三行是 "再见" 内容
        assert lines[2].startswith("data: {")
        line2 = self.parse_streaming_line(lines[2])
        assert line2["id"] == event_id
        assert line2["object"] == "chat.completion.chunk"
        assert line2["model"] == "test-model"
        assert line2["choices"][0]["delta"] == {"content": "再见"}

        # 验证第四行是 finish_reason（在内容行中）
        assert lines[3].startswith("data: {")
        line3 = self.parse_streaming_line(lines[3])
        assert line3["id"] == event_id
        assert line3["object"] == "chat.completion.chunk"
        assert line3["model"] == "test-model"
        assert line3["choices"][0]["delta"] == {}
        assert line3["choices"][0]["finish_reason"] == "stop"

        # 验证最后是 [DONE]
        assert lines[4] == "data: [DONE]"

        # 验证所有文本内容
        all_text = ""
        for line in lines:
            if line.startswith("data: {"):
                data = self.parse_streaming_line(line)
                if "choices" in data and len(data["choices"]) > 0:
                    content = (
                        data["choices"][0].get("delta", {}).get("content", "")
                    )
                    all_text += content

        assert all_text == "你好再见"

    async def test_server_addition_merge(self):
        """测试 addition 字段的合并功能"""
        from agentrun.server import (
            AdditionMode,
            AgentEvent,
            AgentRequest,
            AgentRunServer,
            EventType,
        )

        async def streaming_invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TEXT,
                data={"message_id": "msg_1", "delta": "Hello"},
                addition={
                    "model": "custom_model",
                    "custom_field": "custom_value",
                },
                addition_mode=AdditionMode.MERGE,
            )

        server = AgentRunServer(invoke_agent=streaming_invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # 测试 OpenAI 协议
        response_openai = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "model": "test-model",
                "stream": True,
            },
        )

        assert response_openai.status_code == 200
        lines = [line async for line in response_openai.aiter_lines()]
        lines = [line for line in lines if line]

        # OpenAI 流式格式：只有一个内容行 + 完成行 + [DONE]
        assert (
            len(lines) == 3
        )  # role + content + finish_reason + [DONE] 实际合并为 3 行

        # 验证第一个 chunk 包含原始 model 和 addition 中合并的字段
        assert lines[0].startswith("data: {")
        line0 = self.parse_streaming_line(lines[0])
        assert line0["id"].startswith("chatcmpl-")
        assert line0["object"] == "chat.completion.chunk"
        assert line0["model"] == "test-model"  # 原始模型，不是被覆盖的
        # addition 字段合并到了 delta 中
        assert line0["choices"][0]["delta"] == {
            "role": "assistant",
            "content": "Hello",
            "model": "custom_model",  # addition 中的字段被合并进来
            "custom_field": "custom_value",
        }

        event_id = line0["id"]

        # 验证后续内容行
        assert lines[1].startswith("data: {")
        line1 = self.parse_streaming_line(lines[1])
        assert line1["id"] == event_id
        assert line1["object"] == "chat.completion.chunk"
        assert line1["model"] == "test-model"  # 原始模型
        assert line1["choices"][0]["delta"] == {}
        assert line1["choices"][0]["finish_reason"] == "stop"

        # 验证最后是 [DONE]
        assert lines[2] == "data: [DONE]"

        # 验证 AG-UI 协议
        response_agui = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        assert response_agui.status_code == 200
        lines = [line async for line in response_agui.aiter_lines()]
        lines = [line for line in lines if line]

        # AG-UI 流式格式：RUN_STARTED + TEXT_MESSAGE_START + TEXT_MESSAGE_CONTENT + TEXT_MESSAGE_END + RUN_FINISHED
        assert len(lines) == 5

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
        assert line1["messageId"]  # 确保 message_id 存在（自动生成的 UUID）
        assert line1["role"] == "assistant"

        message_id = line1["messageId"]

        assert lines[2].startswith("data: {")
        line2 = self.parse_streaming_line(lines[2])
        assert line2["type"] == "TEXT_MESSAGE_CONTENT"
        assert line2["messageId"] == message_id
        assert line2["delta"] == "Hello"
        # addition 字段应该被合并到事件中
        # 注意：在 AG-UI 中，addition 合并后会保留所有字段
        assert "model" in line2
        assert line2["model"] == "custom_model"
        assert line2["custom_field"] == "custom_value"

        assert lines[3].startswith("data: {")
        line3 = self.parse_streaming_line(lines[3])
        assert line3["type"] == "TEXT_MESSAGE_END"
        assert line3["messageId"] == message_id

        assert lines[4].startswith("data: {")
        line4 = self.parse_streaming_line(lines[4])
        assert line4["type"] == "RUN_FINISHED"
        assert line4["runId"] == run_id
        assert line4["threadId"] == thread_id

    async def test_server_tool_call_agui(self):
        """测试 AG-UI 协议中的工具调用事件序列"""
        from agentrun.server import (
            AgentEvent,
            AgentRequest,
            AgentRunServer,
            EventType,
        )

        async def streaming_invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TOOL_CALL,
                data={
                    "id": "tc-1",
                    "name": "weather_tool",
                    "args": '{"location": "Beijing"}',
                },
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "Sunny, 25°C"},
            )

        server = AgentRunServer(invoke_agent=streaming_invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # 发送 AG-UI 请求
        response = client.post(
            "/ag-ui/agent",
            json={
                "messages": [{"role": "user", "content": "What's the weather?"}]
            },
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines()]
        lines = [line for line in lines if line]

        # AG-UI 流式格式：RUN_STARTED + TOOL_CALL_START + TOOL_CALL_ARGS + TOOL_CALL_END + TOOL_CALL_RESULT + RUN_FINISHED
        # 注意：由于没有文本内容，所以不会触发 TEXT_MESSAGE_* 事件
        # TOOL_CALL 会先触发 TOOL_CALL_START，然后是 TOOL_CALL_ARGS（使用 args_delta），最后是 TOOL_CALL_END
        # TOOL_RESULT 会被转换为 TOOL_CALL_RESULT
        assert len(lines) == 6

        assert lines[0].startswith("data: {")
        line0 = self.parse_streaming_line(lines[0])
        assert line0["type"] == "RUN_STARTED"
        assert line0["threadId"]
        assert line0["runId"]

        thread_id = line0["threadId"]
        run_id = line0["runId"]

        assert lines[1].startswith("data: {")
        line1 = self.parse_streaming_line(lines[1])
        assert line1["type"] == "TOOL_CALL_START"
        assert line1["toolCallId"] == "tc-1"
        assert line1["toolCallName"] == "weather_tool"

        assert lines[2].startswith("data: {")
        line2 = self.parse_streaming_line(lines[2])
        assert line2["type"] == "TOOL_CALL_ARGS"
        assert line2["toolCallId"] == "tc-1"
        assert line2["delta"] == '{"location": "Beijing"}'

        assert lines[3].startswith("data: {")
        line3 = self.parse_streaming_line(lines[3])
        assert line3["type"] == "TOOL_CALL_END"
        assert line3["toolCallId"] == "tc-1"

        assert lines[4].startswith("data: {")
        line4 = self.parse_streaming_line(lines[4])
        assert line4["type"] == "TOOL_CALL_RESULT"
        assert line4["toolCallId"] == "tc-1"
        assert line4["content"] == "Sunny, 25°C"
        assert line4["role"] == "tool"
        assert line4["messageId"] == "tool-result-tc-1"

        assert lines[5].startswith("data: {")
        line5 = self.parse_streaming_line(lines[5])
        assert line5["type"] == "RUN_FINISHED"
        assert line5["threadId"] == thread_id
        assert line5["runId"] == run_id
