"""
# cspell:ignore chatcmpl ASGI nonstream
AgentRunServer 集成测试 - 测试不同的 LangChain/LangGraph 调用方式

测试覆盖:
- astream_events: 使用 agent.astream_events(input, version="v2")
- astream: 使用 agent.astream(input, stream_mode="updates")
- stream: 使用 agent.stream(input, stream_mode="updates")
- invoke: 使用 agent.invoke(input)
- ainvoke: 使用 agent.ainvoke(input)

每种方式都测试：
1. 纯文本生成场景
2. 工具调用场景
"""

import json
import socket
import threading
import time
from typing import Any, cast, Dict, List, Sequence, Union
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import pytest
import uvicorn

from agentrun.integration.langchain import model
from agentrun.integration.langgraph import AgentRunConverter
from agentrun.model import ModelService, ModelType, ProviderSettings
from agentrun.server import AgentRequest, AgentRunServer

# =============================================================================
# 配置
# =============================================================================

# =============================================================================
# 工具定义
# =============================================================================


def get_weather(city: str) -> Dict[str, Any]:
    """获取指定城市的天气信息

    Args:
        city: 城市名称

    Returns:
        包含天气信息的字典
    """
    return {"city": city, "weather": "晴天", "temperature": 25}


def get_time() -> str:
    """获取当前时间

    Returns:
        当前时间字符串
    """
    return "2024-01-01 12:00:00"


TOOLS = [get_weather, get_time]


# =============================================================================
# 辅助函数
# =============================================================================


def _find_free_port() -> int:
    """获取可用的本地端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_mock_openai_app() -> FastAPI:
    """构建本地 OpenAI 协议兼容的简单服务"""
    app = FastAPI()

    def _decide_action(messages: List[Dict[str, Any]]):
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                return {"type": "after_tool", "content": msg.get("content", "")}

        user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), {}
        )
        content = user_msg.get("content", "")
        if "天气" in content or "weather" in content:
            return {
                "type": "tool_call",
                "tool": "get_weather",
                "arguments": {"city": "北京"},
            }
        if "时间" in content or "time" in content or "几点" in content:
            return {"type": "tool_call", "tool": "get_time", "arguments": {}}
        return {"type": "chat", "content": content or "你好，我是本地模型"}

    def _chat_response(model: str, content: str) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
        }

    def _tool_response(
        model: str, tool: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": json.dumps(
                                arguments, ensure_ascii=False
                            ),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }

    async def _stream_chat(model: str, content: str):
        yield _sse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }],
        })
        yield _sse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }],
        })
        yield _sse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })
        yield "data: [DONE]\n\n"

    async def _stream_tool(model: str, tool: str, arguments: Dict[str, Any]):
        yield _sse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }],
        })
        yield _sse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": json.dumps(
                                arguments, ensure_ascii=False
                            ),
                        },
                    }]
                },
                "finish_reason": None,
            }],
        })
        yield _sse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
            ],
        })
        yield "data: [DONE]\n\n"

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {"id": "mock-model", "object": "model", "owned_by": "local"}
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        stream = bool(body.get("stream", False))
        model = body.get("model", "mock-model")
        messages = body.get("messages", [])

        action = _decide_action(messages)
        if action["type"] == "chat":
            content = action["content"] or "好的，我在本地为你服务。"
            if stream:
                return StreamingResponse(
                    _stream_chat(model, content), media_type="text/event-stream"
                )
            return JSONResponse(_chat_response(model, content))

        if action["type"] == "tool_call":
            tool = action["tool"]
            arguments = action["arguments"]
            if stream:
                return StreamingResponse(
                    _stream_tool(model, tool, arguments),
                    media_type="text/event-stream",
                )
            return JSONResponse(_tool_response(model, tool, arguments))

        # after tool result
        content = f"工具结果已收到: {action['content']}"
        if stream:
            return StreamingResponse(
                _stream_chat(model, content), media_type="text/event-stream"
            )
        return JSONResponse(_chat_response(model, content))

    return app


def build_agent(model_input: Union[str, Any]):
    """创建测试用的 agent"""
    from langchain.agents import create_agent

    return create_agent(
        model=model(model_input),
        tools=TOOLS,
        system_prompt="你是一个测试助手。",
    )


def parse_sse_events(content: str) -> List[Dict[str, Any]]:
    """解析 SSE 响应内容为事件列表"""
    events = []
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str and data_str != "[DONE]":
                try:
                    events.append(json.loads(data_str))
                except json.JSONDecodeError:
                    pass
    return events


async def request_agui_events(
    server_app,
    messages: List[Dict[str, str]],
    stream: bool = True,
) -> List[Dict[str, Any]]:
    """发送 AG-UI 请求并返回事件列表"""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server_app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/ag-ui/agent",
            json={"messages": messages, "stream": stream},
            timeout=60.0,
        )

    assert response.status_code == 200
    return parse_sse_events(response.text)


def _index(event_types: Sequence[str], target: str) -> int:
    """安全获取事件索引，未找到抛出 AssertionError"""
    assert target in event_types, f"缺少事件: {target}"
    return event_types.index(target)


def _normalize_agui_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """去除可变字段，仅保留语义字段"""
    normalized: Dict[str, Any] = {"type": event.get("type")}
    if "delta" in event:
        normalized["delta"] = event["delta"]
    if "result" in event:
        normalized["result"] = event["result"]
    if "toolCallName" in event:
        normalized["toolCallName"] = event["toolCallName"]
    if "role" in event:
        normalized["role"] = event["role"]
    if "toolCallId" in event:
        normalized["hasToolCallId"] = bool(event["toolCallId"])
    if "messageId" in event:
        normalized["hasMessageId"] = bool(event["messageId"])
    if "threadId" in event:
        normalized["hasThreadId"] = bool(event["threadId"])
    if "runId" in event:
        normalized["hasRunId"] = bool(event["runId"])
    return normalized


AGUI_EXPECTED = {
    "text_basic": [[
        {"type": "RUN_STARTED", "hasThreadId": True, "hasRunId": True},
        {
            "type": "TEXT_MESSAGE_START",
            "role": "assistant",
            "hasMessageId": True,
        },
        {
            "type": "TEXT_MESSAGE_CONTENT",
            "delta": "你好，请简单介绍一下你自己",
            "hasMessageId": True,
        },
        {"type": "TEXT_MESSAGE_END", "hasMessageId": True},
        {"type": "RUN_FINISHED", "hasThreadId": True, "hasRunId": True},
    ]],
    "tool_weather": [
        [
            {"type": "RUN_STARTED", "hasThreadId": True, "hasRunId": True},
            {
                "type": "TOOL_CALL_ARGS",
                "delta": '{"city": "北京"}',
                "hasToolCallId": True,
            },
            {
                "type": "TOOL_CALL_START",
                "toolCallName": "get_weather",
                "hasToolCallId": True,
            },
            {
                "type": "TOOL_CALL_ARGS",
                "delta": '{"city": "北京"}',
                "hasToolCallId": True,
            },
            {
                "type": "TOOL_CALL_RESULT",
                "result": (
                    '{"city": "北京", "weather": "晴天", "temperature": 25}'
                ),
                "hasToolCallId": True,
            },
            {"type": "TOOL_CALL_END", "hasToolCallId": True},
            {
                "type": "TEXT_MESSAGE_START",
                "role": "assistant",
                "hasMessageId": True,
            },
            {
                "type": "TEXT_MESSAGE_CONTENT",
                "delta": (
                    '工具结果已收到: {"city": "北京", "weather": "晴天",'
                    ' "temperature": 25}'
                ),
                "hasMessageId": True,
            },
            {"type": "TEXT_MESSAGE_END", "hasMessageId": True},
            {"type": "RUN_FINISHED", "hasThreadId": True, "hasRunId": True},
        ],
        [
            {"type": "RUN_STARTED", "hasThreadId": True, "hasRunId": True},
            {
                "type": "TOOL_CALL_START",
                "toolCallName": "get_weather",
                "hasToolCallId": True,
            },
            {
                "type": "TOOL_CALL_ARGS",
                "delta": '{"city": "北京"}',
                "hasToolCallId": True,
            },
            {
                "type": "TOOL_CALL_RESULT",
                "result": (
                    '{"city": "北京", "weather": "晴天", "temperature": 25}'
                ),
                "hasToolCallId": True,
            },
            {"type": "TOOL_CALL_END", "hasToolCallId": True},
            {
                "type": "TEXT_MESSAGE_START",
                "role": "assistant",
                "hasMessageId": True,
            },
            {
                "type": "TEXT_MESSAGE_CONTENT",
                "delta": (
                    '工具结果已收到: {"city": "北京", "weather": "晴天",'
                    ' "temperature": 25}'
                ),
                "hasMessageId": True,
            },
            {"type": "TEXT_MESSAGE_END", "hasMessageId": True},
            {"type": "RUN_FINISHED", "hasThreadId": True, "hasRunId": True},
        ],
    ],
    "tool_time": [[
        {"type": "RUN_STARTED", "hasThreadId": True, "hasRunId": True},
        {
            "type": "TOOL_CALL_START",
            "toolCallName": "get_time",
            "hasToolCallId": True,
        },
        {
            "type": "TOOL_CALL_RESULT",
            "result": "2024-01-01 12:00:00",
            "hasToolCallId": True,
        },
        {"type": "TOOL_CALL_END", "hasToolCallId": True},
        {
            "type": "TEXT_MESSAGE_START",
            "role": "assistant",
            "hasMessageId": True,
        },
        {
            "type": "TEXT_MESSAGE_CONTENT",
            "delta": "工具结果已收到: 2024-01-01 12:00:00",
            "hasMessageId": True,
        },
        {"type": "TEXT_MESSAGE_END", "hasMessageId": True},
        {"type": "RUN_FINISHED", "hasThreadId": True, "hasRunId": True},
    ]],
}


def assert_agui_events_exact(
    events: List[Dict[str, Any]], case_key: str
) -> None:
    normalized = [_normalize_agui_event(e) for e in events]
    variants = AGUI_EXPECTED[case_key]
    assert (
        normalized in variants
    ), f"AGUI events mismatch for {case_key}: {normalized}"


def normalize_openai_event_types(chunks: List[Dict[str, Any]]) -> List[str]:
    """将 OpenAI 协议的流式分片转换为统一的事件类型序列"""
    event_types: List[str] = []
    for chunk in chunks:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        finish_reason = choice.get("finish_reason")

        if delta.get("role"):
            event_types.append("TEXT_MESSAGE_START")
        if delta.get("content"):
            event_types.append("TEXT_MESSAGE_CONTENT")

        for tool_call in delta.get("tool_calls") or []:
            function = (tool_call or {}).get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments", "")
            if name:
                event_types.append("TOOL_CALL_START")
            if arguments:
                event_types.append("TOOL_CALL_ARGS")

        if finish_reason == "tool_calls":
            event_types.append("TOOL_CALL_END")
        elif finish_reason == "stop":
            event_types.append("TEXT_MESSAGE_END")

    return event_types


def _normalize_openai_stream(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for chunk in chunks:
        choice = (chunk.get("choices") or [{}])[0] or {}
        delta = choice.get("delta") or {}
        entry: Dict[str, Any] = {
            "object": chunk.get("object"),
            "finish_reason": choice.get("finish_reason"),
        }
        if "role" in delta:
            entry["delta_role"] = delta["role"]
        if "content" in delta:
            entry["delta_content"] = delta["content"]
        if "tool_calls" in delta:
            tools = []
            for tc in delta.get("tool_calls") or []:
                func = (tc or {}).get("function") or {}
                tools.append({
                    "name": func.get("name"),
                    "arguments": func.get("arguments"),
                    "has_id": bool((tc or {}).get("id")),
                })
            entry["tool_calls"] = tools
        normalized.append(entry)
    return normalized


OPENAI_STREAM_EXPECTED = {
    "text_basic": [
        {
            "object": "chat.completion.chunk",
            "delta_role": "assistant",
            "finish_reason": None,
        },
        {
            "object": "chat.completion.chunk",
            "delta_content": "你好，请简单介绍一下你自己",
            "finish_reason": None,
        },
        {
            "object": "chat.completion.chunk",
            "finish_reason": "stop",
        },
    ],
    "tool_weather": [
        {
            "object": "chat.completion.chunk",
            "tool_calls": [
                {"name": None, "arguments": '{"city": "北京"}', "has_id": False}
            ],
            "finish_reason": None,
        },
        {
            "object": "chat.completion.chunk",
            "tool_calls": [{
                "name": "get_weather",
                "arguments": "",
                "has_id": True,
            }],
            "finish_reason": None,
        },
        {
            "object": "chat.completion.chunk",
            "tool_calls": [
                {"name": None, "arguments": '{"city": "北京"}', "has_id": False}
            ],
            "finish_reason": None,
        },
        {
            "object": "chat.completion.chunk",
            "finish_reason": "tool_calls",
        },
        {
            "object": "chat.completion.chunk",
            "finish_reason": None,
            "delta_role": "assistant",
        },
        {
            "object": "chat.completion.chunk",
            "delta_content": (
                '工具结果已收到: {"city": "北京", "weather": "晴天",'
                ' "temperature": 25}'
            ),
            "finish_reason": None,
        },
        {"object": "chat.completion.chunk", "finish_reason": "stop"},
    ],
    "tool_time": [
        {
            "object": "chat.completion.chunk",
            "tool_calls": [{
                "name": "get_time",
                "arguments": "",
                "has_id": True,
            }],
            "finish_reason": None,
        },
        {
            "object": "chat.completion.chunk",
            "finish_reason": "tool_calls",
        },
        {
            "object": "chat.completion.chunk",
            "delta_role": "assistant",
            "finish_reason": None,
        },
        {
            "object": "chat.completion.chunk",
            "delta_content": "工具结果已收到: 2024-01-01 12:00:00",
            "finish_reason": None,
        },
        {"object": "chat.completion.chunk", "finish_reason": "stop"},
    ],
}


def _normalize_openai_nonstream(resp: Dict[str, Any]) -> Dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0] or {}
    msg = choice.get("message") or {}
    tools_norm = None
    if msg.get("tool_calls"):
        tools_norm = []
        for tc in msg.get("tool_calls") or []:
            func = (tc or {}).get("function") or {}
            tools_norm.append({
                "name": func.get("name"),
                "arguments": func.get("arguments"),
                "has_id": bool((tc or {}).get("id")),
            })
    return {
        "object": resp.get("object"),
        "role": msg.get("role"),
        "content": msg.get("content"),
        "tool_calls": tools_norm,
        "finish_reason": choice.get("finish_reason"),
    }


OPENAI_NONSTREAM_EXPECTED = {
    "text_basic": {
        "object": "chat.completion",
        "role": "assistant",
        "content": "你好，请简单介绍一下你自己",
        "tool_calls": None,
        "finish_reason": "stop",
    },
    "tool_weather": {
        "object": "chat.completion",
        "role": "assistant",
        "content": (
            '工具结果已收到: {"city": "北京", "weather": "晴天",'
            ' "temperature": 25}'
        ),
        "tool_calls": [{
            "name": "get_weather",
            "arguments": '{"city": "北京"}',
            "has_id": True,
        }],
        "finish_reason": "tool_calls",
    },
    "tool_time": {
        "object": "chat.completion",
        "role": "assistant",
        "content": "工具结果已收到: 2024-01-01 12:00:00",
        "tool_calls": [{
            "name": "get_time",
            "arguments": "",
            "has_id": True,
        }],
        "finish_reason": "tool_calls",
    },
}


def assert_openai_text_generation_events(chunks: List[Dict[str, Any]]) -> None:
    """校验 OpenAI 协议纯文本流式事件"""
    assert (
        _normalize_openai_stream(chunks) == OPENAI_STREAM_EXPECTED["text_basic"]
    )


def assert_openai_tool_call_events(
    chunks: List[Dict[str, Any]], case_key: str
) -> None:
    """校验 OpenAI 协议工具调用流式事件"""
    assert _normalize_openai_stream(chunks) == OPENAI_STREAM_EXPECTED[case_key]


def assert_openai_text_generation_response(resp: Dict[str, Any]) -> None:
    """校验 OpenAI 协议非流式文本响应"""
    assert (
        _normalize_openai_nonstream(resp)
        == OPENAI_NONSTREAM_EXPECTED["text_basic"]
    )


def assert_openai_tool_call_response(
    resp: Dict[str, Any], case_key: str
) -> None:
    """校验 OpenAI 协议非流式工具调用响应"""
    assert (
        _normalize_openai_nonstream(resp) == OPENAI_NONSTREAM_EXPECTED[case_key]
    )


AGUI_PROMPT_CASES = [
    ("text_basic", "你好，请简单介绍一下你自己"),
    ("tool_weather", "北京的天气怎么样？"),
    ("tool_time", "现在几点？"),
]


async def request_openai_events(
    server_app,
    messages: List[Dict[str, str]],
    stream: bool = True,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """发送 OpenAI 协议请求并返回流式事件列表或响应"""
    payload: Dict[str, Any] = {
        "model": "mock-model",
        "messages": messages,
        "stream": stream,
    }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server_app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/openai/v1/chat/completions",
            json=payload,
            timeout=60.0,
        )

    assert response.status_code == 200

    if stream:
        return parse_sse_events(response.text)

    return response.json()


# =============================================================================
# 模型服务准备
# =============================================================================


@pytest.fixture(scope="session")
def mock_openai_server():
    """启动本地 OpenAI 协议兼容服务"""
    app = _build_mock_openai_app()
    port = _find_free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            httpx.get(f"{base_url}/v1/models", timeout=0.2)
            break
        except Exception:
            time.sleep(0.1)

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="function")
def agent_model(mock_openai_server: str):
    """使用本地 OpenAI 服务构造 ModelService"""
    base_url = f"{mock_openai_server}/v1"
    return ModelService(
        model_service_name="mock-model",
        model_type=ModelType.LLM,
        provider="openai",
        provider_settings=ProviderSettings(
            api_key="sk-local-key",
            base_url=base_url,
            model_names=["mock-model"],
        ),
    )


@pytest.fixture
def server_app_astream_events(agent_model):
    """创建使用 astream_events 的服务器（AG-UI/OpenAI 通用）"""
    agent = build_agent(agent_model)

    async def invoke_agent(request: AgentRequest):
        input_data: Dict[str, Any] = {
            "messages": [
                {
                    "role": (
                        msg.role.value
                        if hasattr(msg.role, "value")
                        else str(msg.role)
                    ),
                    "content": msg.content,
                }
                for msg in request.messages
            ]
        }

        converter = AgentRunConverter()

        async def generator():
            async for event in agent.astream_events(
                cast(Any, input_data), version="v2"
            ):
                for item in converter.convert(event):
                    yield item

        return generator()

    server = AgentRunServer(invoke_agent=invoke_agent)
    return server.app


# =============================================================================
# 测试类: astream_events
# =============================================================================


class TestAstreamEvents:
    """测试 agent.astream_events 调用方式"""

    @pytest.mark.parametrize(
        "case_key,prompt",
        AGUI_PROMPT_CASES,
        ids=[case[0] for case in AGUI_PROMPT_CASES],
    )
    async def test_astream_events(
        self, server_app_astream_events, case_key, prompt
    ):
        """覆盖文本、工具、本地工具的流式事件"""
        events = await request_agui_events(
            server_app_astream_events,
            [{"role": "user", "content": prompt}],
            stream=True,
        )

        assert_agui_events_exact(events, case_key)


# =============================================================================
# 测试类: astream (updates 模式)
# =============================================================================


class TestAstreamUpdates:
    """测试 agent.astream(stream_mode=\"updates\") 调用方式"""

    @pytest.fixture
    def server_app(self, agent_model):
        """创建使用 astream 的服务器"""
        agent = build_agent(agent_model)

        async def invoke_agent(request: AgentRequest):
            input_data: Dict[str, Any] = {
                "messages": [
                    {
                        "role": (
                            msg.role.value
                            if hasattr(msg.role, "value")
                            else str(msg.role)
                        ),
                        "content": msg.content,
                    }
                    for msg in request.messages
                ]
            }

            converter = AgentRunConverter()
            if request.stream:

                async def generator():
                    async for event in agent.astream(
                        cast(Any, input_data), stream_mode="updates"
                    ):
                        for item in converter.convert(event):
                            yield item

                return generator()
            else:
                return await agent.ainvoke(cast(Any, input_data))

        server = AgentRunServer(invoke_agent=invoke_agent)
        return server.app

    @pytest.mark.parametrize(
        "case_key,prompt",
        AGUI_PROMPT_CASES,
        ids=[case[0] for case in AGUI_PROMPT_CASES],
    )
    async def test_astream_updates(self, server_app, case_key, prompt):
        """流式 updates 模式覆盖对话与工具场景"""
        events = await request_agui_events(
            server_app,
            [{"role": "user", "content": prompt}],
            stream=True,
        )

        assert_agui_events_exact(events, case_key)


# =============================================================================
# 测试类: stream (同步 updates 模式)
# =============================================================================


class TestStreamUpdates:
    """测试 agent.stream(stream_mode="updates") 调用方式"""

    @pytest.fixture
    def server_app(self, agent_model):
        """创建使用 stream 的服务器"""
        agent = build_agent(agent_model)

        def invoke_agent(request: AgentRequest):
            input_data: Dict[str, Any] = {
                "messages": [
                    {
                        "role": (
                            msg.role.value
                            if hasattr(msg.role, "value")
                            else str(msg.role)
                        ),
                        "content": msg.content,
                    }
                    for msg in request.messages
                ]
            }

            converter = AgentRunConverter()
            if request.stream:

                def generator():
                    for event in agent.stream(
                        cast(Any, input_data), stream_mode="updates"
                    ):
                        for item in converter.convert(event):
                            yield item

                return generator()
            else:
                return agent.invoke(cast(Any, input_data))

        server = AgentRunServer(invoke_agent=invoke_agent)
        return server.app

    @pytest.mark.parametrize(
        "case_key,prompt",
        AGUI_PROMPT_CASES,
        ids=[case[0] for case in AGUI_PROMPT_CASES],
    )
    async def test_stream_updates(self, server_app, case_key, prompt):
        """同步 stream updates 覆盖对话与工具场景"""
        events = await request_agui_events(
            server_app,
            [{"role": "user", "content": prompt}],
            stream=True,
        )

        assert_agui_events_exact(events, case_key)


# =============================================================================
# 测试类: invoke/ainvoke (非流式)
# =============================================================================


class TestInvoke:
    """测试 agent.invoke/ainvoke 非流式调用方式"""

    @pytest.fixture
    def server_app_sync(self, agent_model):
        """创建使用 invoke 的服务器"""
        agent = build_agent(agent_model)

        async def invoke_agent(request: AgentRequest):
            input_data: Dict[str, Any] = {
                "messages": [
                    {
                        "role": (
                            msg.role.value
                            if hasattr(msg.role, "value")
                            else str(msg.role)
                        ),
                        "content": msg.content,
                    }
                    for msg in request.messages
                ]
            }

            converter = AgentRunConverter()

            async def generator():
                async for event in agent.astream_events(
                    cast(Any, input_data), version="v2"
                ):
                    for item in converter.convert(event):
                        yield item

            return generator()

        server = AgentRunServer(invoke_agent=invoke_agent)
        return server.app

    @pytest.fixture
    def server_app_async(self, agent_model):
        """创建使用 ainvoke 的服务器"""
        agent = build_agent(agent_model)

        async def invoke_agent(request: AgentRequest):
            input_data: Dict[str, Any] = {
                "messages": [
                    {
                        "role": (
                            msg.role.value
                            if hasattr(msg.role, "value")
                            else str(msg.role)
                        ),
                        "content": msg.content,
                    }
                    for msg in request.messages
                ]
            }

            converter = AgentRunConverter()

            async def generator():
                async for event in agent.astream_events(
                    cast(Any, input_data), version="v2"
                ):
                    for item in converter.convert(event):
                        yield item

            return generator()

        server = AgentRunServer(invoke_agent=invoke_agent)
        return server.app

    @pytest.mark.parametrize(
        "case_key,prompt",
        AGUI_PROMPT_CASES,
        ids=[case[0] for case in AGUI_PROMPT_CASES],
    )
    async def test_sync_invoke(self, server_app_sync, case_key, prompt):
        """测试同步 invoke 非流式场景"""
        events = await request_agui_events(
            server_app_sync,
            [{"role": "user", "content": prompt}],
            stream=False,
        )

        assert_agui_events_exact(events, case_key)

    @pytest.mark.parametrize(
        "case_key,prompt",
        AGUI_PROMPT_CASES,
        ids=[case[0] for case in AGUI_PROMPT_CASES],
    )
    async def test_async_invoke(self, server_app_async, case_key, prompt):
        """测试异步 ainvoke 非流式场景"""
        events = await request_agui_events(
            server_app_async,
            [{"role": "user", "content": prompt}],
            stream=False,
        )

        assert_agui_events_exact(events, case_key)


# =============================================================================
# 测试类: OpenAI 协议
# =============================================================================


class TestOpenAIProtocol:
    """测试 OpenAI Chat Completions 协议"""

    async def test_nonstream_text_generation(self, server_app_astream_events):
        """非流式简单对话"""
        resp = await request_openai_events(
            server_app_astream_events,
            [{"role": "user", "content": "你好，请简单介绍一下你自己"}],
            stream=False,
        )

        assert_openai_text_generation_response(cast(Dict[str, Any], resp))

    async def test_stream_text_generation(self, server_app_astream_events):
        """OpenAI 协议纯文本流式输出检查"""
        events = cast(
            List[Dict[str, Any]],
            await request_openai_events(
                server_app_astream_events,
                [{"role": "user", "content": "你好，请简单介绍一下你自己"}],
                stream=True,
            ),
        )

        assert_openai_text_generation_events(events)

    async def test_stream_tool_call(self, server_app_astream_events):
        """OpenAI 协议工具调用流式输出检查"""
        events = cast(
            List[Dict[str, Any]],
            await request_openai_events(
                server_app_astream_events,
                [{"role": "user", "content": "北京的天气怎么样？"}],
                stream=True,
            ),
        )

        assert_openai_tool_call_events(events, "tool_weather")

    async def test_nonstream_tool_call(self, server_app_astream_events):
        """非流式工具调用"""
        resp = await request_openai_events(
            server_app_astream_events,
            [{"role": "user", "content": "北京的天气怎么样？"}],
            stream=False,
        )

        assert_openai_tool_call_response(
            cast(Dict[str, Any], resp), "tool_weather"
        )

    async def test_nonstream_local_tool(self, server_app_astream_events):
        """非流式本地工具调用"""
        resp = await request_openai_events(
            server_app_astream_events,
            [{"role": "user", "content": "现在几点？"}],
            stream=False,
        )

        assert_openai_tool_call_response(
            cast(Dict[str, Any], resp), "tool_time"
        )

    async def test_stream_local_tool(self, server_app_astream_events):
        """流式本地工具调用"""
        events = cast(
            List[Dict[str, Any]],
            await request_openai_events(
                server_app_astream_events,
                [{"role": "user", "content": "现在几点？"}],
                stream=True,
            ),
        )

        assert_openai_tool_call_events(events, "tool_time")
