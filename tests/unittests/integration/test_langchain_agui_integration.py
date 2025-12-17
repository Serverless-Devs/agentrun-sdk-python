"""LangChain Integration with AGUI Protocol Integration Test (Optimized with MCP & ProtocolValidator)

测试 LangChain 与 AGUI 协议的集成，包含：
1. 真实的 langchain_mcp_adapters.MultiServerMCPClient
2. Mock MCP SSE 服务器 (Starlette)
3. Mock ChatModel (替代 OpenAI API)
4. ProtocolValidator 严格验证事件序列和内容
"""

import json
import socket
import threading
import time
from typing import Any, cast, Dict, List, Optional, Sequence, Union

import httpx
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent
from mcp.types import Tool as MCPTool
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
import uvicorn

from agentrun.integration.langchain import AgentRunConverter
from agentrun.server import AgentRequest, AgentRunServer

# =============================================================================
# Protocol Validator
# =============================================================================


class ProtocolValidator:

    def try_parse_streaming_line(
        self, line: Union[str, dict, list]
    ) -> Union[str, dict, list]:
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

                    # for k in got.keys():
                    #     if k not in expect:
                    #         assert False, f"{path} 多余的键: {k}"
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

    def all_field_equal(self, key: str, arr: list, strict: bool = False):
        """检查列表中所有对象的指定字段值是否相等"""
        value = ""
        for item in arr:
            data = self.try_parse_streaming_line(item)
            data = cast(dict, data)

            if not strict and key not in data:
                continue

            if value == "":
                value = data.get(key)

            assert value == data.get(
                key
            ), f"Field {key} not equal: {value} != {data.get(key)}"
            assert value, f"Field {key} is empty"


# =============================================================================
# Mock MCP SSE Server
# =============================================================================


def create_mock_mcp_sse_app(tools_config: Dict[str, Any]) -> Starlette:
    """创建 Mock MCP SSE 服务器"""
    mcp_server = Server("mock-mcp-server")

    @mcp_server.list_tools()
    async def list_tools():
        tools = []
        for tool_name, tool_info in tools_config.items():
            tools.append(
                MCPTool(
                    name=tool_name,
                    description=tool_info.get("description", ""),
                    inputSchema=tool_info.get(
                        "input_schema", {"type": "object", "properties": {}}
                    ),
                )
            )
        return tools

    @mcp_server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]):
        tool_info = tools_config.get(name, {})
        result_func = tool_info.get("result_func")
        if result_func:
            result = result_func(**arguments)
            if isinstance(result, dict):
                return [
                    TextContent(
                        type="text", text=json.dumps(result, ensure_ascii=False)
                    )
                ]
            return [TextContent(type="text", text=str(result))]
        return [TextContent(type="text", text="No result")]

    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp_server.run(
                streams[0],
                streams[1],
                mcp_server.create_initialization_options(),
            )
        return Response()

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ]
    )
    return app


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mock_mcp_server():
    """启动 Mock MCP SSE 服务器"""
    tools_config = {
        "get_current_time": {
            "description": "获取当前时间",
            "input_schema": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "时区",
                        "default": "UTC",
                    }
                },
            },
            "result_func": lambda timezone="UTC": {
                "timezone": timezone,
                "datetime": "2025-12-17T11:26:10+08:00",
                "day_of_week": "Wednesday",
                "is_dst": False,
            },
        },
        "maps_weather": {
            "description": "获取天气信息",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"],
            },
            "result_func": lambda city: {
                "city": f"{city}市",
                "forecasts": [
                    {
                        "date": "2025-12-17",
                        "week": "3",
                        "dayweather": "多云",
                        "nightweather": "多云",
                        "daytemp": "14",
                        "nighttemp": "4",
                        "daywind": "北",
                        "nightwind": "北",
                        "daypower": "1-3",
                        "nightpower": "1-3",
                        "daytemp_float": "14.0",
                        "nighttemp_float": "4.0",
                    },
                    {
                        "date": "2025-12-18",
                        "week": "4",
                        "dayweather": "晴",
                        "nightweather": "晴",
                        "daytemp": "15",
                        "nighttemp": "6",
                        "daywind": "东",
                        "nightwind": "东",
                        "daypower": "1-3",
                        "nightpower": "1-3",
                        "daytemp_float": "15.0",
                        "nighttemp_float": "6.0",
                    },
                    {
                        "date": "2025-12-19",
                        "week": "5",
                        "dayweather": "晴",
                        "nightweather": "阴",
                        "daytemp": "21",
                        "nighttemp": "12",
                        "daywind": "东南",
                        "nightwind": "东南",
                        "daypower": "1-3",
                        "nightpower": "1-3",
                        "daytemp_float": "21.0",
                        "nighttemp_float": "12.0",
                    },
                    {
                        "date": "2025-12-20",
                        "week": "6",
                        "dayweather": "阴",
                        "nightweather": "小雨",
                        "daytemp": "20",
                        "nighttemp": "8",
                        "daywind": "东北",
                        "nightwind": "东北",
                        "daypower": "1-3",
                        "nightpower": "1-3",
                        "daytemp_float": "20.0",
                        "nighttemp_float": "8.0",
                    },
                ],
            },
        },
    }

    app = create_mock_mcp_sse_app(tools_config)
    port = _find_free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="critical"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"

    start_time = time.time()
    while time.time() - start_time < 10:
        try:
            with httpx.Client() as client:
                resp = client.get(f"{base_url}/sse", timeout=0.5)
                if resp.status_code in (200, 500):
                    break
        except (httpx.ConnectError, httpx.ReadTimeout):
            time.sleep(0.1)

    yield base_url
    server.should_exit = True
    thread.join(timeout=2)


# =============================================================================
# Local Tools
# =============================================================================


@tool
def get_user_name():
    """获取用户名称"""
    return {"user_name": "张三"}


@tool
def get_user_token(user_name: str):
    """获取用户的密钥，输入为用户名"""
    return "ak_1234asd12341"


# =============================================================================
# Mock Chat Model
# =============================================================================


class MockChatModel(BaseChatModel):
    """模拟 ChatOpenAI 的行为"""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_outputs = [m for m in messages if isinstance(m, ToolMessage)]

        # Round 1: Call MCP time + Local name
        if len(tool_outputs) == 0:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=(
                                "我需要获取当前时间、天气信息和您的密钥信息。"
                                "让我依次处理这些请求。\n\n首先，"
                                "让我获取当前时间：\n\n"
                            ),
                            tool_calls=[
                                {
                                    "name": "get_current_time",
                                    "args": {"timezone": "Asia/Shanghai"},
                                    "id": "call_time",
                                },
                                {
                                    "name": "get_user_name",
                                    "args": {},
                                    "id": "call_name",
                                },
                            ],
                        )
                    )
                ]
            )

        # Round 2: Call MCP weather + Local token
        if len(tool_outputs) == 2:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=(
                                "现在我已获取到当前时间和您的用户名。接下来，"
                                "让我获取天气信息和您的密钥：\n\n"
                            ),
                            tool_calls=[
                                {
                                    "name": "maps_weather",
                                    "args": {"city": "上海"},
                                    "id": "call_weather",
                                },
                                {
                                    "name": "get_user_token",
                                    "args": {"user_name": "张三"},
                                    "id": "call_token",
                                },
                            ],
                        )
                    )
                ]
            )

        # Round 3: Finish
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=(
                            "以下是您请求的信息：\n\n## 当前时间\n-"
                            " 日期：2025年12月17日（星期三）\n-"
                            " 时间：11:26:10（北京时间，UTC+8）\n\n##"
                            " 天气信息（上海市）\n**今日（12月17日）天气：**\n-"
                            " 白天：多云，14°C，北风1-3级\n- 夜间：多云，4°C，"
                            "北风1-3级\n\n**未来几天预报：**\n- 12月18日：晴，"
                            "6-15°C\n- 12月19日：晴转阴，12-21°C  \n-"
                            " 12月20日：阴转小雨，8-20°C\n\n##"
                            " 您的密钥信息\n您的用户密钥为：`ak_1234asd12341`\n\n请注意妥善保管您的密钥信息，"
                            "不要在公共场合泄露。"
                        ),
                    )
                )
            ]
        )

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        return self


# =============================================================================
# Tests
# =============================================================================


class TestLangChainAguiIntegration(ProtocolValidator):

    async def test_multi_tool_query(self, mock_mcp_server):
        """测试多工具查询场景 (MCP + Local + MockLLM)"""

        mcp_client = MultiServerMCPClient(
            {
                "tools": {
                    "url": f"{mock_mcp_server}/sse",
                    "transport": "sse",
                }
            }
        )
        mcp_tools = await mcp_client.get_tools()

        async def invoke_agent(request: AgentRequest):
            llm = MockChatModel()
            tools = [*mcp_tools, get_user_name, get_user_token]

            agent = create_agent(
                model=llm,
                system_prompt="You are a helpful assistant",
                tools=tools,
            )

            input_data = {
                "messages": [{
                    "role": "user",
                    "content": (
                        "查询当前的时间，并获取天气信息，同时输出我的密钥信息"
                    ),
                }]
            }

            converter = AgentRunConverter()
            async for event in agent.astream_events(input_data, version="v2"):
                for item in converter.convert(event):
                    yield item

        app = AgentRunServer(invoke_agent=invoke_agent).app

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/ag-ui/agent",
                json={
                    "messages": [{
                        "role": "user",
                        "content": "查询当前的时间，并获取天气信息，同时输出我的密钥信息",
                    }],
                    "stream": True,
                },
                timeout=60.0,
            )

        assert response.status_code == 200

        events = [line for line in response.text.split("\n") if line]
        expected = [
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
                ' {"type":"TEXT_MESSAGE_CONTENT","messageId":"mock-placeholder","delta":"我需要获取当前时间、'
                "天气信息和您的密钥信息。让我依次处理这些请求。\\n\\n首先，"
                '让我获取当前时间：\\n\\n"}'
            ),
            'data: {"type":"TEXT_MESSAGE_END","messageId":"mock-placeholder"}',
            (
                "data:"
                ' {"type":"TOOL_CALL_START","toolCallId":"call_time","toolCallName":"get_current_time"}'
            ),
            (
                "data:"
                ' {"type":"TOOL_CALL_ARGS","toolCallId":"call_time","delta":"{\\"timezone\\":'
                ' \\"Asia/Shanghai\\"}"}'
            ),
            (
                "data:"
                ' {"type":"TOOL_CALL_START","toolCallId":"call_name","toolCallName":"get_user_name"}'
            ),
            (
                "data:"
                ' {"type":"TOOL_CALL_ARGS","toolCallId":"call_name","delta":""}'
            ),
            'data: {"type":"TOOL_CALL_END","toolCallId":"call_name"}',
            (
                "data:"
                ' {"type":"TOOL_CALL_RESULT","messageId":"tool-result-call_name","toolCallId":"call_name","content":"{\\"user_name\\":'
                ' \\"张三\\"}","role":"tool"}'
            ),
            'data: {"type":"TOOL_CALL_END","toolCallId":"call_time"}',
            (
                "data:"
                ' {"type":"TOOL_CALL_RESULT","messageId":"tool-result-call_time","toolCallId":"call_time","content":"mock-placeholder","role":"tool"}'
            ),
            (
                "data:"
                ' {"type":"TEXT_MESSAGE_START","messageId":"mock-placeholder","role":"assistant"}'
            ),
            (
                "data:"
                ' {"type":"TEXT_MESSAGE_CONTENT","messageId":"mock-placeholder","delta":"现在我已获取到当前时间和您的用户名。'
                '接下来，让我获取天气信息和您的密钥：\\n\\n"}'
            ),
            'data: {"type":"TEXT_MESSAGE_END","messageId":"mock-placeholder"}',
            (
                "data:"
                ' {"type":"TOOL_CALL_START","toolCallId":"call_weather","toolCallName":"maps_weather"}'
            ),
            (
                "data:"
                ' {"type":"TOOL_CALL_ARGS","toolCallId":"call_weather","delta":"{\\"city\\":'
                ' \\"上海\\"}"}'
            ),
            (
                "data:"
                ' {"type":"TOOL_CALL_START","toolCallId":"call_token","toolCallName":"get_user_token"}'
            ),
            (
                "data:"
                ' {"type":"TOOL_CALL_ARGS","toolCallId":"call_token","delta":"{\\"user_name\\":'
                ' \\"张三\\"}"}'
            ),
            'data: {"type":"TOOL_CALL_END","toolCallId":"call_token"}',
            (
                "data:"
                ' {"type":"TOOL_CALL_RESULT","messageId":"tool-result-call_token","toolCallId":"call_token","content":"ak_1234asd12341","role":"tool"}'
            ),
            'data: {"type":"TOOL_CALL_END","toolCallId":"call_weather"}',
            (
                "data:"
                ' {"type":"TOOL_CALL_RESULT","messageId":"tool-result-call_weather","toolCallId":"call_weather","content":"mock-placeholder","role":"tool"}'
            ),
            (
                "data:"
                ' {"type":"TEXT_MESSAGE_START","messageId":"mock-placeholder","role":"assistant"}'
            ),
            (
                "data:"
                ' {"type":"TEXT_MESSAGE_CONTENT","messageId":"mock-placeholder","delta":"以下是您请求的信息：\\n\\n##'
                " 当前时间\\n- 日期：2025年12月17日（星期三）\\n-"
                " 时间：11:26:10（北京时间，UTC+8）\\n\\n##"
                " 天气信息（上海市）\\n**今日（12月17日）天气：**\\n-"
                " 白天：多云，14°C，北风1-3级\\n- 夜间：多云，4°C，"
                "北风1-3级\\n\\n**未来几天预报：**\\n- 12月18日：晴，6-15°C\\n-"
                " 12月19日：晴转阴，12-21°C  \\n- 12月20日：阴转小雨，"
                "8-20°C\\n\\n##"
                " 您的密钥信息\\n您的用户密钥为：`ak_1234asd12341`\\n\\n请注意妥善保管您的密钥信息，"
                '不要在公共场合泄露。"}'
            ),
            'data: {"type":"TEXT_MESSAGE_END","messageId":"mock-placeholder"}',
            (
                "data:"
                ' {"type":"RUN_FINISHED","threadId":"mock-placeholder","runId":"mock-placeholder"}'
            ),
        ]

        self.valid_json(events, expected)

        self.all_field_equal("runId", events)
        self.all_field_equal("threadId", events)
        self.all_field_equal("messageId", events[1:4])
        self.all_field_equal("messageId", events[12:15])
        self.all_field_equal("messageId", events[24:27])
