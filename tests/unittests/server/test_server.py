import asyncio
import json
from typing import Any, cast, Union

import pytest

from agentrun.server.model import AgentRequest, MessageRole
from agentrun.server.server import AgentRunServer


class ProtocolValidator:

    def try_parse_streaming_line(self, line: str) -> Union[str, dict, list]:
        """解析流式响应行，去除前缀 'data: ' 并转换为 JSON"""

        if type(line) is not str or line.startswith("data: [DONE]"):
            return line
        if line.startswith("data: {") or line.startswith("data: ["):
            json_str = line[len("data: ") :]
            return json.loads(json_str)
        return line

    def parse_streaming_line(self, line: str) -> Union[str, dict, list]:
        """解析流式响应行，去除前缀 'data: ' 并转换为 JSON"""

        if type(line) is not str or line.startswith("data: [DONE]"):
            return line
        if line.startswith("data: {") or line.startswith("data: ["):
            json_str = line[len("data: ") :]
            return json.loads(json_str)
        return line

    def valid_json(self, got: Any, expect: Any):
        """检查 json 是否匹配，如果 expect 的 value 为 mock-placeholder 则仅检查 key 存在"""

        def valid(path: str, got: Any, expect: Any):
            if expect == "mock-placeholder":
                assert got, f"{path} 存在但值为空"
            else:
                got = self.try_parse_streaming_line(got)
                expect = self.try_parse_streaming_line(expect)

                if isinstance(expect, dict):
                    assert isinstance(
                        got, dict
                    ), f"{path} 类型不匹配，期望 dict，实际 {type(got)}"
                    for k, v in expect.items():
                        valid(f"{path}.{k}", got.get(k), v)

                    for k in got.keys():
                        if k not in expect:
                            assert False, f"{path} 多余的键: {k}"
                elif isinstance(expect, list):
                    assert isinstance(
                        got, list
                    ), f"{path} 类型不匹配，期望 list，实际 {type(got)}"
                    assert len(got) == len(
                        expect
                    ), f"{path} 列表长度不匹配，期望 {len(expect)}，实际 {len(got)}"
                    for i in range(len(expect)):
                        valid(f"{path}[{i}]", got[i], expect[i])
                else:
                    assert (
                        got == expect
                    ), f"{path} 不匹配，期望: {type(expect)} {expect}，实际: {got}"

        print("valid", got, expect)
        valid("", got, expect)

    def all_field_equal(
        self,
        key: str,
        arr: list,
    ):
        """检查列表中所有对象的指定字段值是否相等"""
        print("all_field_equal", arr, key)

        value = ""
        for item in arr:
            data = self.try_parse_streaming_line(item)
            data = cast(dict, data)

            if value == "":
                value = data[key]

            assert value == data[key]
            assert value


class TestServer(ProtocolValidator):

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

    def get_client(self, invoke_agent):
        server = AgentRunServer(invoke_agent=invoke_agent)
        app = server.as_fastapi_app()
        from fastapi.testclient import TestClient

        return TestClient(app)

    async def test_server_non_streaming_protocols(self):
        """测试非流式的 OpenAI 和 AGUI 服务器响应功能"""

        client = self.get_client(self.get_invoke_agent_non_streaming())

        # 测试 OpenAI 协议
        response_openai = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "AgentRun"}],
                "model": "test-model",
            },
        )

        # 检查响应状态
        assert response_openai.status_code == 200

        # 检查响应内容
        response_data_openai = response_openai.json()

        self.valid_json(
            response_data_openai,
            {
                "id": "mock-placeholder",
                "object": "chat.completion",
                "created": "mock-placeholder",
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "You said: AgentRun",
                    },
                    "finish_reason": "stop",
                }],
            },
        )

        # AGUI 协议始终是流式传输，因此没有非流式测试
        # 测试 AGUI 协议（即使非流式请求也会以流式方式处理）
        response_agui = client.post(
            "/ag-ui/agent",
            json={
                "messages": [{"role": "user", "content": "AgentRun"}],
                "model": "test-model",
            },
        )

        # 检查响应状态
        assert response_agui.status_code == 200
        lines_agui = [line async for line in response_agui.aiter_lines()]
        lines_agui = [line for line in lines_agui if line]

        # AG-UI 流式格式：RUN_STARTED + TEXT_MESSAGE_START + TEXT_MESSAGE_CONTENT + TEXT_MESSAGE_END + RUN_FINISHED
        assert len(lines_agui) == 5

        # 验证 AGUI 流式事件序列
        self.valid_json(
            lines_agui,
            [
                (
                    "data:"
                    ' {"type":"RUN_STARTED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_START","messageId":"mock-placeholder","role":"assistant"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_CONTENT","messageId":"mock-placeholder","delta":"You'
                    ' said: AgentRun"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_END","messageId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"RUN_FINISHED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
            ],
        )

    async def test_server_streaming_protocols(self):
        """测试流式的 OpenAI 和 AGUI 服务器响应功能"""

        # 测试 OpenAI 协议流式响应
        client = self.get_client(self.get_invoke_agent_streaming())

        response_openai = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "AgentRun"}],
                "model": "test-model",
                "stream": True,
            },
        )

        # 检查响应状态
        assert response_openai.status_code == 200
        lines_openai = [line async for line in response_openai.aiter_lines()]

        # 过滤空行
        lines_openai = [line for line in lines_openai if line]

        # OpenAI 流式格式：第一个 chunk 是 role 声明，后续是内容
        # 格式：data: {...}
        self.valid_json(
            lines_openai,
            [
                (
                    'data: {"id": "mock-placeholder", "object":'
                    ' "chat.completion.chunk", "created": "mock-placeholder",'
                    ' "model": "test-model", "choices": [{"index": 0, "delta":'
                    ' {"role": "assistant", "content": "Hello, "},'
                    ' "finish_reason": null}]}'
                ),
                (
                    'data: {"id": "mock-placeholder", "object":'
                    ' "chat.completion.chunk", "created": "mock-placeholder",'
                    ' "model": "test-model", "choices": [{"index": 0, "delta":'
                    ' {"content": "this is "}, "finish_reason": null}]}'
                ),
                (
                    'data: {"id": "mock-placeholder", "object":'
                    ' "chat.completion.chunk", "created": "mock-placeholder",'
                    ' "model": "test-model", "choices": [{"index": 0, "delta":'
                    ' {"content": "a test."}, "finish_reason": null}]}'
                ),
                (
                    'data: {"id": "mock-placeholder", "object":'
                    ' "chat.completion.chunk", "created": "mock-placeholder",'
                    ' "model": "test-model", "choices": [{"index": 0, "delta":'
                    ' {}, "finish_reason": "stop"}]}'
                ),
                "data: [DONE]",
            ],
        )
        self.all_field_equal("id", lines_openai[:-1])

        # 测试 AGUI 协议流式响应
        response_agui = client.post(
            "/ag-ui/agent",
            json={
                "messages": [{"role": "user", "content": "AgentRun"}],
                "model": "test-model",
                "stream": True,
            },
        )

        # 检查响应状态
        assert response_agui.status_code == 200
        lines_agui = [line async for line in response_agui.aiter_lines()]

        # 过滤空行
        lines_agui = [line for line in lines_agui if line]

        # AG-UI 流式格式：每个 chunk 是一个 JSON 对象
        self.valid_json(
            lines_agui,
            [
                (
                    "data:"
                    ' {"type":"RUN_STARTED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_START","messageId":"mock-placeholder","role":"assistant"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_CONTENT","messageId":"mock-placeholder","delta":"Hello, "}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_CONTENT","messageId":"mock-placeholder","delta":"this'
                    ' is "}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_CONTENT","messageId":"mock-placeholder","delta":"a'
                    ' test."}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_END","messageId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"RUN_FINISHED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
            ],
        )
        self.all_field_equal("threadId", [lines_agui[0], lines_agui[-1]])
        self.all_field_equal("runId", [lines_agui[0], lines_agui[-1]])
        self.all_field_equal("messageId", lines_agui[1:6])

    async def test_server_raw_event_protocols(self):
        """测试 RAW 事件直接返回原始数据（OpenAI 和 AG-UI 协议）

        RAW 事件可以在任何时间触发，输出原始 SSE 内容，不影响其他事件的正常处理。
        支持任意 SSE 格式（data:, :注释, 等）。
        """
        from agentrun.server import AgentEvent, AgentRequest, EventType

        async def streaming_invoke_agent(request: AgentRequest):
            # 测试 RAW 事件与其他事件混合
            yield "你好"
            yield AgentEvent(
                event=EventType.RAW,
                data={"raw": '{"custom": "data"}'},
            )
            yield AgentEvent(event=EventType.TEXT, data={"delta": "再见"})

        client = self.get_client(streaming_invoke_agent)

        # 测试 OpenAI 协议的 RAW 事件
        response_openai = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "model": "agentrun",
                "stream": True,
            },
        )

        assert response_openai.status_code == 200
        lines_openai = [line async for line in response_openai.aiter_lines()]
        lines_openai = [line for line in lines_openai if line]

        # OpenAI 流式响应：
        # 1. role: assistant + content: 你好（合并在首个 chunk）
        # 2. RAW: {"custom": "data"}
        # 3. content: 再见
        # 4. finish_reason: stop
        # 5. [DONE]
        self.valid_json(
            lines_openai,
            [
                (
                    'data: {"id": "mock-placeholder", "object":'
                    ' "chat.completion.chunk", "created": "mock-placeholder",'
                    ' "model": "agentrun", "choices": [{"index": 0, "delta":'
                    ' {"role": "assistant", "content": "你好"},'
                    ' "finish_reason": null}]}'
                ),
                '{"custom": "data"}',
                (
                    'data: {"id": "mock-placeholder", "object":'
                    ' "chat.completion.chunk", "created": "mock-placeholder",'
                    ' "model": "agentrun", "choices": [{"index": 0, "delta":'
                    ' {"content": "再见"}, "finish_reason": null}]}'
                ),
                (
                    'data: {"id": "mock-placeholder", "object":'
                    ' "chat.completion.chunk", "created": "mock-placeholder",'
                    ' "model": "agentrun", "choices": [{"index": 0, "delta":'
                    ' {}, "finish_reason": "stop"}]}'
                ),
                "data: [DONE]",
            ],
        )
        self.all_field_equal("id", [lines_openai[0], *lines_openai[2:-1]])

        # 测试 AGUI 协议的 RAW 事件
        response_agui = client.post(
            "/ag-ui/agent",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )

        assert response_agui.status_code == 200
        lines_agui = [line async for line in response_agui.aiter_lines()]
        lines_agui = [line for line in lines_agui if line]

        # AGUI 流式响应中应该包含 RAW 事件
        # 1. RUN_STARTED
        # 2. TEXT_MESSAGE_START
        # 3. TEXT_MESSAGE_CONTENT ("你好")
        # 4. RAW 事件 '{"custom": "data"}'
        # 5. TEXT_MESSAGE_CONTENT ("再见")
        # 6. TEXT_MESSAGE_END
        # 7. RUN_FINISHED
        self.valid_json(
            lines_agui,
            [
                (
                    "data:"
                    ' {"type":"RUN_STARTED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_START","messageId":"mock-placeholder","role":"assistant"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_CONTENT","messageId":"mock-placeholder","delta":"你好"}'
                ),
                '{"custom": "data"}',
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_CONTENT","messageId":"mock-placeholder","delta":"再见"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_END","messageId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"RUN_FINISHED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
            ],
        )
        self.all_field_equal("threadId", [lines_agui[0], lines_agui[-1]])
        self.all_field_equal("runId", [lines_agui[0], lines_agui[-1]])
        self.all_field_equal(
            "messageId",
            [lines_agui[1], lines_agui[2], lines_agui[4], lines_agui[5]],
        )

    async def test_server_addition_merge(self):
        """测试 addition 字段的合并功能"""
        from agentrun.server import AgentEvent, AgentRequest, EventType

        async def streaming_invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.TEXT,
                data={"message_id": "msg_1", "delta": "Hello"},
                addition={
                    "model": "custom_model",
                    "custom_field": "custom_value",
                },
            )

        client = self.get_client(streaming_invoke_agent)

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

        self.valid_json(
            lines,
            [
                (
                    'data: {"id": "mock-placeholder", "object":'
                    ' "chat.completion.chunk", "created": "mock-placeholder",'
                    ' "model": "test-model", "choices": [{"index": 0, "delta":'
                    ' {"role": "assistant", "content": "Hello", "model":'
                    ' "custom_model", "custom_field": "custom_value"},'
                    ' "finish_reason": null}]}'
                ),
                (
                    'data: {"id": "mock-placeholder", "object":'
                    ' "chat.completion.chunk", "created": "mock-placeholder",'
                    ' "model": "test-model", "choices": [{"index": 0, "delta":'
                    ' {}, "finish_reason": "stop"}]}'
                ),
                "data: [DONE]",
            ],
        )
        self.all_field_equal("id", lines[:-1])

        response_agui = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
        )

        assert response_agui.status_code == 200
        lines_agui = [line async for line in response_agui.aiter_lines()]
        lines_agui = [line for line in lines_agui if line]

        # AG-UI 流式格式：RUN_STARTED + TEXT_MESSAGE_START + TEXT_MESSAGE_CONTENT + TEXT_MESSAGE_END + RUN_FINISHED
        self.valid_json(
            lines_agui,
            [
                (
                    "data:"
                    ' {"type":"RUN_STARTED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_START","messageId":"mock-placeholder","role":"assistant"}'
                ),
                (
                    'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId":'
                    ' "mock-placeholder", "delta": "Hello", "model":'
                    ' "custom_model", "custom_field": "custom_value"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_END","messageId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"RUN_FINISHED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
            ],
        )
        self.all_field_equal("threadId", [lines_agui[0], lines_agui[-1]])
        self.all_field_equal("runId", [lines_agui[0], lines_agui[-1]])
        self.all_field_equal("messageId", lines_agui[1:4])

    async def test_server_tool_call_protocols(self):
        """测试 OpenAI 和 AG-UI 协议中的工具调用事件序列"""
        from agentrun.server import AgentEvent, AgentRequest, EventType

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

        client = self.get_client(streaming_invoke_agent)

        # 测试 OpenAI 协议的工具调用
        response_openai = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "What's the weather?"}
                ],
                "stream": True,
            },
        )

        assert response_openai.status_code == 200
        lines_openai = [line async for line in response_openai.aiter_lines()]
        lines_openai = [line for line in lines_openai if line]

        # OpenAI 流式格式：包含工具调用的事件序列
        # 1. role + tool_calls（包含 id, type, function.name, function.arguments）
        # 2. tool_calls（包含 id 和 function.arguments delta）
        # 3. finish_reason: tool_calls
        # 4. [DONE]
        assert len(lines_openai) == 4

        # 第一个 chunk 包含工具调用信息（可能不包含role，具体取决于工具调用的类型）
        assert lines_openai[0].startswith("data: {")
        line0 = self.try_parse_streaming_line(lines_openai[0])
        line0 = cast(dict, line0)  # 类型断言
        assert line0["object"] == "chat.completion.chunk"
        assert "tool_calls" in line0["choices"][0]["delta"]
        assert (
            line0["choices"][0]["delta"]["tool_calls"][0]["type"] == "function"
        )
        assert (
            line0["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            == "weather_tool"
        )
        assert line0["choices"][0]["delta"]["tool_calls"][0]["id"] == "tc-1"

        # 第二个 chunk 包含函数参数（不包含ID，只有参数）
        assert lines_openai[1].startswith("data: {")
        line1 = self.try_parse_streaming_line(lines_openai[1])
        line1 = cast(dict, line1)  # 类型断言
        assert line1["object"] == "chat.completion.chunk"
        assert (
            line1["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ]
            == '{"location": "Beijing"}'
        )

        # 第三个 chunk 包含 finish_reason
        assert lines_openai[2].startswith("data: {")
        line2 = self.try_parse_streaming_line(lines_openai[2])
        line2 = cast(dict, line2)  # 类型断言
        assert line2["object"] == "chat.completion.chunk"
        assert line2["choices"][0]["finish_reason"] == "tool_calls"

        # 最后是 [DONE]
        assert lines_openai[3] == "data: [DONE]"

        # 测试 AG-UI 协议的工具调用
        response_agui = client.post(
            "/ag-ui/agent",
            json={
                "messages": [{"role": "user", "content": "What's the weather?"}]
            },
        )

        assert response_agui.status_code == 200
        lines_agui = [line async for line in response_agui.aiter_lines()]
        lines_agui = [line for line in lines_agui if line]

        # AG-UI 流式格式：RUN_STARTED + TOOL_CALL_START + TOOL_CALL_ARGS + TOOL_CALL_END + TOOL_CALL_RESULT + RUN_FINISHED
        # 注意：由于没有文本内容，所以不会触发 TEXT_MESSAGE_* 事件
        # TOOL_CALL 会先触发 TOOL_CALL_START，然后是 TOOL_CALL_ARGS（使用 args_delta），最后是 TOOL_CALL_END
        # TOOL_RESULT 会被转换为 TOOL_CALL_RESULT
        self.valid_json(
            lines_agui,
            [
                (
                    "data:"
                    ' {"type":"RUN_STARTED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"TOOL_CALL_START","toolCallId":"tc-1","toolCallName":"weather_tool"}'
                ),
                (
                    "data:"
                    ' {"type":"TOOL_CALL_ARGS","toolCallId":"tc-1","delta":"{\\"location\\":'
                    ' \\"Beijing\\"}"}'
                ),
                'data: {"type":"TOOL_CALL_END","toolCallId":"tc-1"}',
                (
                    "data:"
                    ' {"type":"TOOL_CALL_RESULT","messageId":"mock-placeholder","toolCallId":"tc-1","content":"Sunny,'
                    ' 25°C","role":"tool"}'
                ),
                (
                    "data:"
                    ' {"type":"RUN_FINISHED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
            ],
        )
        self.all_field_equal("threadId", [lines_agui[0], lines_agui[-1]])
        self.all_field_equal("runId", [lines_agui[0], lines_agui[-1]])
        self.all_field_equal("toolCallId", lines_agui[1:5])

    @pytest.mark.asyncio
    async def test_server_text_then_tool_call_agui(self):
        """测试 AG-UI 协议中先文本后工具调用的事件序列

        AG-UI 协议要求：发送 TOOL_CALL_START 前必须先发送 TEXT_MESSAGE_END
        """
        from agentrun.server import AgentEvent, AgentRequest, EventType

        async def streaming_invoke_agent(request: AgentRequest):
            # 先发送文本
            yield "思考中..."
            # 然后发送工具调用
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "tc-1",
                    "name": "search_tool",
                    "args_delta": '{"query": "test"}',
                },
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "搜索结果"},
            )

        client = self.get_client(streaming_invoke_agent)

        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "搜索一下"}]},
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines()]
        lines = [line for line in lines if line]

        # 预期事件序列：
        # 1. RUN_STARTED
        # 2. TEXT_MESSAGE_START
        # 3. TEXT_MESSAGE_CONTENT
        # 4. TEXT_MESSAGE_END  <-- 必须在 TOOL_CALL_START 之前
        # 5. TOOL_CALL_START
        # 6. TOOL_CALL_ARGS
        # 7. TOOL_CALL_END
        # 8. TOOL_CALL_RESULT
        # 9. RUN_FINISHED
        self.valid_json(
            lines,
            [
                (
                    "data:"
                    ' {"type":"RUN_STARTED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_START","messageId":"mock-placeholder","role":"assistant"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_CONTENT","messageId":"mock-placeholder","delta":"思考中..."}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_END","messageId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"TOOL_CALL_START","toolCallId":"tc-1","toolCallName":"search_tool"}'
                ),
                (
                    "data:"
                    ' {"type":"TOOL_CALL_ARGS","toolCallId":"tc-1","delta":"{\\"query\\":'
                    ' \\"test\\"}"}'
                ),
                'data: {"type":"TOOL_CALL_END","toolCallId":"tc-1"}',
                (
                    "data:"
                    ' {"type":"TOOL_CALL_RESULT","messageId":"mock-placeholder","toolCallId":"tc-1","content":"搜索结果","role":"tool"}'
                ),
                (
                    "data:"
                    ' {"type":"RUN_FINISHED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
            ],
        )
        self.all_field_equal("threadId", [lines[0], lines[-1]])
        self.all_field_equal("runId", [lines[0], lines[-1]])
        self.all_field_equal("messageId", [lines[1], lines[2], lines[3]])
        self.all_field_equal("toolCallId", lines[4:8])

    @pytest.mark.asyncio
    async def test_server_text_tool_text_agui(self):
        """测试 AG-UI 协议中 文本->工具调用->文本 的事件序列

        场景：先输出思考内容，然后调用工具，最后输出结果
        AG-UI 协议要求：
        1. 发送 TOOL_CALL_START 前必须先发送 TEXT_MESSAGE_END
        2. 工具调用后的新文本需要新的 TEXT_MESSAGE_START
        """
        from agentrun.server import AgentEvent, AgentRequest, EventType

        async def streaming_invoke_agent(request: AgentRequest):
            # 第一段文本
            yield "让我搜索一下..."
            # 工具调用
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": "tc-1",
                    "name": "search",
                    "args_delta": '{"q": "天气"}',
                },
            )
            yield AgentEvent(
                event=EventType.TOOL_RESULT,
                data={"id": "tc-1", "result": "晴天"},
            )
            # 第二段文本（工具调用后）
            yield "根据搜索结果，今天是晴天。"

        client = self.get_client(streaming_invoke_agent)

        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "今天天气如何"}]},
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines()]
        lines = [line for line in lines if line]

        # 预期事件序列：
        # 1. RUN_STARTED
        # 2. TEXT_MESSAGE_START (第一个文本消息)
        # 3. TEXT_MESSAGE_CONTENT
        # 4. TEXT_MESSAGE_END   <-- 工具调用前必须结束
        # 5. TOOL_CALL_START
        # 6. TOOL_CALL_ARGS
        # 7. TOOL_CALL_END
        # 8. TOOL_CALL_RESULT
        # 9. TEXT_MESSAGE_START (第二个文本消息，新的 messageId)
        # 10. TEXT_MESSAGE_CONTENT
        # 11. TEXT_MESSAGE_END
        # 12. RUN_FINISHED
        self.valid_json(
            lines,
            [
                (
                    "data:"
                    ' {"type":"RUN_STARTED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_START","messageId":"mock-placeholder","role":"assistant"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_CONTENT","messageId":"mock-placeholder","delta":"让我搜索一下..."}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_END","messageId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"TOOL_CALL_START","toolCallId":"tc-1","toolCallName":"search"}'
                ),
                (
                    "data:"
                    ' {"type":"TOOL_CALL_ARGS","toolCallId":"tc-1","delta":"{\\"q\\":'
                    ' \\"天气\\"}"}'
                ),
                'data: {"type":"TOOL_CALL_END","toolCallId":"tc-1"}',
                (
                    "data:"
                    ' {"type":"TOOL_CALL_RESULT","messageId":"mock-placeholder","toolCallId":"tc-1","content":"晴天","role":"tool"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_START","messageId":"mock-placeholder","role":"assistant"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_CONTENT","messageId":"mock-placeholder","delta":"根据搜索结果，'
                    '今天是晴天。"}'
                ),
                (
                    "data:"
                    ' {"type":"TEXT_MESSAGE_END","messageId":"mock-placeholder"}'
                ),
                (
                    "data:"
                    ' {"type":"RUN_FINISHED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
                ),
            ],
        )
        self.all_field_equal("threadId", [lines[0], lines[-1]])
        self.all_field_equal("runId", [lines[0], lines[-1]])
        self.all_field_equal("messageId", [lines[1], lines[2], lines[3]])
        self.all_field_equal("toolCallId", lines[4:8])

    @pytest.mark.asyncio
    async def test_agent_request_raw_request(self):
        """测试 AgentRequest.raw_request 可以访问原始请求对象

        验证：
        1. raw_request 包含完整的 Starlette Request 对象
        2. 可以访问 headers, query_params, client 等属性
        """
        from agentrun.server import AgentRequest

        captured_request: dict = {}

        async def invoke_agent(request: AgentRequest):
            # 捕获请求信息
            captured_request["protocol"] = request.protocol
            captured_request["has_raw_request"] = (
                request.raw_request is not None
            )
            if request.raw_request:
                captured_request["headers"] = dict(request.raw_request.headers)
                captured_request["path"] = request.raw_request.url.path
                captured_request["method"] = request.raw_request.method
            return "Hello"

        client = self.get_client(invoke_agent)

        # 测试 AG-UI 协议
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "test"}]},
            headers={"X-Custom-Header": "custom-value"},
        )
        assert response.status_code == 200

        # 验证捕获的请求信息
        assert captured_request["protocol"] == "agui"
        assert captured_request["has_raw_request"] is True
        assert captured_request["path"] == "/ag-ui/agent"
        assert captured_request["method"] == "POST"
        assert (
            captured_request["headers"].get("x-custom-header") == "custom-value"
        )

        # 重置
        captured_request.clear()

        # 测试 OpenAI 协议
        response = client.post(
            "/openai/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200

        # 验证捕获的请求信息
        assert captured_request["protocol"] == "openai"
        assert captured_request["has_raw_request"] is True
        assert captured_request["path"] == "/openai/v1/chat/completions"
        assert (
            captured_request["headers"].get("authorization")
            == "Bearer test-token"
        )
