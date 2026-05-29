"""LangChain AgentRunServer demo e2e case."""

from dataclasses import dataclass
import inspect
import json
import os
import socket
import threading
import time
from typing import Any, Dict, Generator, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
import httpx
from langchain.agents import create_agent
import pydash
import pytest
import uvicorn

from agentrun.integration.langchain import (
    AgentRunConverter,
    model,
    sandbox_toolset,
)
from agentrun.integration.utils.model import CommonModel
import agentrun.memory_collection as memory_collection_module
from agentrun.model import ModelService, ModelType, ProviderSettings
from agentrun.sandbox import TemplateType
from agentrun.server import AgentRequest, AgentRunServer

MODEL_NAME = "demo-model"
MODEL_SERVICE_NAME = "demo-model-service"
MEMORY_COLLECTION_NAME = "cafe-mem"
MODEL_REPLY = "你好，我是网页分析助手。"


@dataclass
class MockOpenAIServer:
    base_url: str
    requests: List[Dict[str, Any]]


@dataclass
class DemoRuntime:
    client: TestClient
    mock_openai: MockOpenAIServer
    memory_collection_names: List[str]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chat_payload(model_name: str, content: str) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }


async def _stream_chat(model_name: str, content: str):
    yield _sse({
        "id": "chatcmpl-demo",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }],
    })
    yield _sse({
        "id": "chatcmpl-demo",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {"content": content},
            "finish_reason": None,
        }],
    })
    yield _sse({
        "id": "chatcmpl-demo",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    yield "data: [DONE]\n\n"


def _build_mock_openai_app(
    request_log: List[Dict[str, Any]],
) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "local"}
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        request_log.append({
            "headers": dict(request.headers),
            "body": body,
        })

        model_name = body.get("model") or MODEL_NAME
        if body.get("stream"):
            return StreamingResponse(
                _stream_chat(model_name, MODEL_REPLY),
                media_type="text/event-stream",
            )

        return JSONResponse(_chat_payload(model_name, MODEL_REPLY))

    return app


def _parse_sse_events(content: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


def _openai_content(events: List[Dict[str, Any]]) -> str:
    chunks = []
    for event in events:
        choice = (event.get("choices") or [{}])[0] or {}
        delta = choice.get("delta") or {}
        chunks.append(delta.get("content", ""))
    return "".join(chunks)


def _agui_content(events: List[Dict[str, Any]]) -> str:
    return "".join(
        event.get("delta", "")
        for event in events
        if event.get("type") == "TEXT_MESSAGE_CONTENT"
    )


def _sandbox_tools() -> List[Any]:
    sandbox_name = os.getenv("SANDBOX_NAME")
    if sandbox_name and not sandbox_name.startswith("请替换"):
        return sandbox_toolset(
            template_name=sandbox_name,
            template_type=TemplateType.CODE_INTERPRETER,
            sandbox_idle_timeout_seconds=300,
        )
    return []


async def _yield_agent_handler_result(
    request: AgentRequest,
    agent_handler: Any,
):
    result = agent_handler(request)
    if inspect.isawaitable(result):
        result = await result

    if hasattr(result, "__aiter__"):
        async for item in result:
            yield item
        return

    if result is not None:
        yield result


def _build_demo_app(
    model_service: ModelService,
    monkeypatch: pytest.MonkeyPatch,
) -> FastAPI:
    monkeypatch.setenv("MODEL_NAME", MODEL_NAME)
    monkeypatch.setenv("MODEL_SERVICE_NAME", MODEL_SERVICE_NAME)
    monkeypatch.setenv("OPENAI_API_KEY", "agentrun")
    monkeypatch.setenv("SANDBOX_NAME", "")

    original_get_model_info = CommonModel.get_model_info

    def get_model_info(self, config=None):
        info = original_get_model_info(self, config)
        if not (info.api_key or "").strip():
            fallback = os.getenv("OPENAI_API_KEY", "agentrun").strip()
            if fallback:
                info.api_key = "agentrun"
        return info

    monkeypatch.setattr(CommonModel, "get_model_info", get_model_info)

    memory_collection_names: List[str] = []

    class MemoryConversationPassthrough:

        def __init__(self, memory_collection_name: str):
            memory_collection_names.append(memory_collection_name)

        async def wrap_invoke_agent(self, request, agent_handler):
            async for item in _yield_agent_handler_result(
                request, agent_handler
            ):
                yield item

    monkeypatch.setattr(
        memory_collection_module,
        "MemoryConversation",
        MemoryConversationPassthrough,
    )

    if not os.getenv("MODEL_SERVICE_NAME"):
        raise ValueError("请将 MODEL_SERVICE_NAME 替换为您已经创建的模型名称")

    agent = create_agent(
        model=model(model_service, model=os.getenv("MODEL_NAME")),
        tools=[*_sandbox_tools()],
        system_prompt="你是网页分析助手。",
    )

    def invoke_agent(request: AgentRequest):
        input_data: Any = {
            "messages": [
                {
                    "content": message.content,
                    "role": (
                        message.role.value
                        if hasattr(message.role, "value")
                        else str(message.role)
                    ),
                }
                for message in request.messages
            ]
        }
        converter = AgentRunConverter()

        if request.stream:

            async def stream_generator():
                result = agent.astream_events(input_data, version="v2")
                async for chunk in result:
                    for item in converter.convert(chunk):
                        yield item

            return stream_generator()

        result = agent.invoke(input_data)
        return pydash.get(result, "messages.-1.content")

    app = AgentRunServer(
        invoke_agent=invoke_agent,
        memory_collection_name=MEMORY_COLLECTION_NAME,
    ).as_fastapi_app()
    app.state.memory_collection_names = memory_collection_names
    return app


@pytest.fixture(scope="module")
def mock_openai_server() -> Generator[MockOpenAIServer, None, None]:
    request_log: List[Dict[str, Any]] = []
    app = _build_mock_openai_app(request_log)
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
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("mock OpenAI server did not start")

    yield MockOpenAIServer(base_url=base_url, requests=request_log)

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def demo_runtime(
    mock_openai_server: MockOpenAIServer,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[DemoRuntime, None, None]:
    mock_openai_server.requests.clear()
    model_service = ModelService(
        model_service_name=MODEL_SERVICE_NAME,
        model_type=ModelType.LLM,
        provider="openai",
        provider_settings=ProviderSettings(
            api_key="",
            base_url=f"{mock_openai_server.base_url}/v1",
            model_names=[MODEL_NAME],
        ),
    )
    app = _build_demo_app(model_service, monkeypatch)

    with TestClient(app) as client:
        yield DemoRuntime(
            client=client,
            mock_openai=mock_openai_server,
            memory_collection_names=app.state.memory_collection_names,
        )


def test_langchain_demo_openai_streaming(demo_runtime: DemoRuntime):
    response = demo_runtime.client.post(
        "/openai/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "你好?"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert MODEL_REPLY == _openai_content(_parse_sse_events(response.text))
    assert MEMORY_COLLECTION_NAME in demo_runtime.memory_collection_names


def test_langchain_demo_openai_plain_response(demo_runtime: DemoRuntime):
    response = demo_runtime.client.post(
        "/openai/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "你好?"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert pydash.get(payload, "choices.0.message.content") == MODEL_REPLY
    assert any(
        item["headers"].get("authorization") == "Bearer agentrun"
        for item in demo_runtime.mock_openai.requests
    )


def test_langchain_demo_agui_events(demo_runtime: DemoRuntime):
    response = demo_runtime.client.post(
        "/ag-ui/agent",
        json={
            "messages": [
                {"role": "user", "content": "写一段代码,查询现在是几点?"}
            ]
        },
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    event_types = [event.get("type") for event in events]
    assert "RUN_STARTED" in event_types
    assert "TEXT_MESSAGE_CONTENT" in event_types
    assert "RUN_FINISHED" in event_types
    assert MODEL_REPLY == _agui_content(events)
