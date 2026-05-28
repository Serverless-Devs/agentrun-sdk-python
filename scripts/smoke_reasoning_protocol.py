"""Smoke test reasoning content through AgentRun protocol handlers."""

import argparse
import asyncio
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import httpx
from dotenv import load_dotenv

from agentrun.model import BackendType, ModelClient
from agentrun.model.api.data import ModelDataAPI
from agentrun.server import (
    AgentEvent,
    AgentRequest,
    AgentRunServer,
    EventType,
)
from agentrun.server.agui_protocol import AGUIProtocolHandler
from agentrun.server.openai_protocol import OpenAIProtocolHandler
from agentrun.utils.reasoning import (
    get_reasoning_content,
    get_thinking_value_from_env,
    is_thinking_enabled_from_env,
    parse_bool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=["openai", "agui", "both"],
        default="both",
    )
    parser.add_argument(
        "--response-mode",
        choices=["stream", "non-stream"],
        default="stream",
    )
    parser.add_argument("--env-file")
    parser.add_argument("--model-resource")
    parser.add_argument("--model")
    parser.add_argument(
        "--prompt",
        default="用一句话回答：AgentRun 是什么？",
    )
    parser.add_argument("--expect-reasoning", action="store_true")
    parser.add_argument("--expect-no-reasoning", action="store_true")
    parser.add_argument("--expect-content", action="store_true")
    return parser.parse_args()


def load_env_file(path: Optional[str]) -> None:
    if path:
        load_dotenv(path, override=False)
    if not os.getenv("AGENTRUN_REGION") and os.getenv("AGENTRUN_REGION_ID"):
        os.environ["AGENTRUN_REGION"] = os.environ["AGENTRUN_REGION_ID"]


def model_parameter_rules() -> Dict[str, Any]:
    raw = os.getenv("MODEL_PARAMETER_RULES")
    if not raw:
        return {}
    try:
        rules = json.loads(raw)
    except ValueError:
        return {}
    return rules if isinstance(rules, dict) else {}


def model_call_kwargs() -> Dict[str, Any]:
    kwargs = model_parameter_rules()
    thinking = get_thinking_value_from_env()
    direct_thinking = parse_bool(kwargs.pop("thinking", None))
    if direct_thinking is not None:
        thinking = direct_thinking

    extra_body = kwargs.pop("extra_body", {})
    if not isinstance(extra_body, dict):
        extra_body = {}
    if thinking is not None:
        extra_body["enable_thinking"] = thinking
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def insecure_ssl_enabled() -> bool:
    return os.getenv("AGENTRUN_SMOKE_INSECURE_SSL", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def read_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def iter_choices(response: Any) -> Iterable[Any]:
    return read_value(response, "choices") or []


def choice_message(choice: Any) -> Any:
    return read_value(choice, "message")


def choice_delta(choice: Any) -> Any:
    return read_value(choice, "delta")


def extract_content(obj: Any) -> Optional[str]:
    value = read_value(obj, "content")
    return value if isinstance(value, str) else None


def collect_non_stream(response: Any) -> tuple[str, str]:
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    for choice in iter_choices(response):
        message = choice_message(choice)
        content = extract_content(message)
        reasoning = get_reasoning_content(message)
        if content:
            content_parts.append(content)
        if reasoning:
            reasoning_parts.append(reasoning)
    return "".join(content_parts), "".join(reasoning_parts)


async def build_agent_result(request: AgentRequest) -> Any:
    model_resource = (
        os.getenv("AGENTRUN_MODEL_SERVICE_NAME")
        or os.getenv("AGENTRUN_MODEL_PROXY_NAME")
        or os.getenv("AGENTRUN_MODEL_NAME")
    )
    model_name = os.getenv("AGENTRUN_MODEL_NAME")
    if not model_resource:
        raise RuntimeError(
            "AGENTRUN_MODEL_SERVICE_NAME or AGENTRUN_MODEL_NAME is required"
        )

    messages = [
        {
            "role": getattr(message.role, "value", message.role),
            "content": message.content or "",
        }
        for message in request.messages
    ]
    kwargs = model_call_kwargs()

    if request.stream:
        chunks = await call_real_model(
            model_resource=model_resource,
            model_name=model_name,
            messages=messages,
            stream=True,
            kwargs=kwargs,
        )

        async def stream():
            for chunk in chunks:
                for event in model_events_from_chunk(chunk):
                    yield event

        return stream()

    response = await call_real_model(
        model_resource=model_resource,
        model_name=model_name,
        messages=messages,
        stream=False,
        kwargs=kwargs,
    )
    content, reasoning = collect_non_stream(response)
    events = []
    if reasoning:
        events.append(
            AgentEvent(event=EventType.REASONING, data={"delta": reasoning})
        )
    if content:
        events.append(AgentEvent(event=EventType.TEXT, data={"delta": content}))
    return events


def model_events_from_chunk(chunk: Any) -> List[AgentEvent]:
    events: List[AgentEvent] = []
    for choice in iter_choices(chunk):
        delta = choice_delta(choice)
        content = extract_content(delta)
        reasoning = (
            get_reasoning_content(delta)
            or get_reasoning_content(choice)
            or get_reasoning_content(chunk)
        )
        if reasoning:
            events.append(
                AgentEvent(
                    event=EventType.REASONING,
                    data={"delta": reasoning},
                )
            )
        if content:
            events.append(
                AgentEvent(event=EventType.TEXT, data={"delta": content})
            )
    return events


async def call_real_model(
    *,
    model_resource: str,
    model_name: Optional[str],
    messages: List[Dict[str, str]],
    stream: bool,
    kwargs: Dict[str, Any],
) -> Any:
    base_url, headers, default_model = resolve_model_endpoint(model_resource)
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name or default_model or model_resource,
        "messages": messages,
        "stream": stream,
        **kwargs,
    }

    async with httpx.AsyncClient(
        timeout=180, verify=not insecure_ssl_enabled()
    ) as client:
        response = await client.post(url, headers=headers, json=payload)

    if response.is_error:
        raise RuntimeError(
            f"real model request failed: {response.status_code} "
            f"{response.text}"
        )

    if not stream:
        return response.json()
    return parse_sse(response.text)


def resolve_model_endpoint(
    model_resource: str,
) -> tuple[str, Dict[str, str], Optional[str]]:
    if os.getenv("AGENTRUN_MODEL_SERVICE_NAME"):
        service = ModelClient().get(
            name=model_resource, backend_type=BackendType.SERVICE
        )
        settings = service.provider_settings
        if not settings or not settings.base_url or not settings.api_key:
            raise RuntimeError(
                f"model service {model_resource} has no provider settings"
            )
        return (
            settings.base_url,
            {
                "authorization": f"Bearer {settings.api_key}",
                "content-type": "application/json",
            },
            (settings.model_names or [None])[0],
        )

    data_api = ModelDataAPI(model_resource)
    info = data_api.model_info()
    return (
        info.base_url or "",
        {
            **(info.headers or {}),
            "content-type": "application/json",
        },
        info.model,
    )


async def call_openai(client: httpx.AsyncClient, args: argparse.Namespace):
    response = await client.post(
        "/openai/v1/chat/completions",
        json={
            "model": args.model or os.getenv("AGENTRUN_MODEL_NAME"),
            "stream": args.response_mode == "stream",
            "messages": [{"role": "user", "content": args.prompt}],
        },
    )
    if response.is_error:
        raise AssertionError(
            f"openai request failed: {response.status_code} {response.text}"
        )

    if args.response_mode == "stream":
        events = parse_sse(response.text)
        content = "".join(
            event.get("choices", [{}])[0].get("delta", {}).get("content", "")
            for event in events
            if isinstance(event, dict)
        )
        reasoning = "".join(
            event.get("choices", [{}])[0]
            .get("delta", {})
            .get("reasoning_content", "")
            for event in events
            if isinstance(event, dict)
        )
        return content, reasoning, events

    payload = response.json()
    message = payload.get("choices", [{}])[0].get("message", {})
    return (
        message.get("content") or "",
        message.get("reasoning_content") or "",
        payload,
    )


async def call_agui(client: httpx.AsyncClient, args: argparse.Namespace):
    response = await client.post(
        "/ag-ui/agent",
        json={
            "threadId": "thread-local",
            "runId": "run-local",
            "messages": [{
                "id": "user-local",
                "role": "user",
                "content": args.prompt,
            }],
            "tools": [],
            "context": [],
            "forwardedProps": {},
        },
    )
    if response.is_error:
        raise AssertionError(
            f"agui request failed: {response.status_code} {response.text}"
        )
    events = parse_sse(response.text)
    content = "".join(
        event.get("delta", "")
        for event in events
        if event.get("type") == "TEXT_MESSAGE_CONTENT"
    )
    reasoning = "".join(
        event.get("delta", "")
        for event in events
        if event.get("type") == "REASONING_MESSAGE_CONTENT"
    )
    return content, reasoning, events


def parse_sse(text: str) -> List[Dict[str, Any]]:
    events = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


def validate_result(name: str, content: str, reasoning: str, args) -> None:
    if args.expect_content and not content:
        raise AssertionError(f"{name}: expected content but got empty content")
    if args.expect_reasoning and not reasoning:
        raise AssertionError(f"{name}: expected reasoning but got none")
    if args.expect_no_reasoning and reasoning:
        raise AssertionError(f"{name}: expected no reasoning but got one")


async def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    if args.model_resource:
        os.environ["AGENTRUN_MODEL_SERVICE_NAME"] = args.model_resource
    if args.model:
        os.environ["AGENTRUN_MODEL_NAME"] = args.model

    app = AgentRunServer(
        invoke_agent=build_agent_result,
        protocols=[OpenAIProtocolHandler(), AGUIProtocolHandler()],
    ).as_fastapi_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://agentrun.local", timeout=180
    ) as client:
        results = {}
        if args.protocol in {"openai", "both"}:
            content, reasoning, raw = await call_openai(client, args)
            validate_result("openai", content, reasoning, args)
            results["openai"] = summarize(content, reasoning, raw)
        if args.protocol in {"agui", "both"}:
            content, reasoning, raw = await call_agui(client, args)
            validate_result("agui", content, reasoning, args)
            results["agui"] = summarize(content, reasoning, raw)

    print(json.dumps({
        "thinkingEnabled": is_thinking_enabled_from_env(),
        "protocol": args.protocol,
        "responseMode": args.response_mode,
        "results": results,
    }, ensure_ascii=False, indent=2))


def summarize(content: str, reasoning: str, raw: Any) -> Dict[str, Any]:
    return {
        "contentPresent": bool(content),
        "reasoningPresent": bool(reasoning),
        "contentSample": content[:120],
        "reasoningSample": reasoning[:120],
        "rawEventCount": len(raw) if isinstance(raw, list) else None,
    }


if __name__ == "__main__":
    asyncio.run(main())
