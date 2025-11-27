"""AgentScope 模型适配器 / AgentScope Model Adapter

将 CommonModel 包装为 AgentScope ChatModelBase。"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, List, Optional

from agentrun.integration.agentscope.message_adapter import (
    AgentScopeMessageAdapter,
)
from agentrun.integration.utils.adapter import ModelAdapter
from agentrun.integration.utils.model import CommonModel


def _ensure_agentscope_installed() -> None:
    try:
        import agentscope  # noqa: F401
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "AgentScope is not installed. Install it with: pip install"
            " agentscope"
        ) from exc


class AgentScopeModelAdapter(ModelAdapter):
    """AgentScope 模型适配器 / AgentScope Model Adapter

    将 CommonModel 包装为 AgentScope ChatModelBase。"""

    def __init__(self):
        """初始化适配器，创建内部的消息适配器 / AgentScope Message Adapter"""
        self._message_adapter = AgentScopeMessageAdapter()

    def wrap_model(self, common_model: CommonModel) -> Any:
        """包装 CommonModel 为 AgentScope ChatModelBase / AgentScope Model Adapter"""
        _ensure_agentscope_installed()

        try:
            from agentscope.model import OpenAIChatModel
        except Exception as e:
            raise ImportError(
                "AgentScope is not installed. Install it with: pip install"
                " agentscope"
            ) from e

        from httpx import AsyncClient

        info = common_model.get_model_info()

        return OpenAIChatModel(
            model_name=info.model or "",
            api_key=info.api_key,
            stream=False,
            client_args={
                "base_url": info.base_url,
                "http_client": AsyncClient(headers=info.headers),
            },
        )

        from agentscope.message import TextBlock, ToolUseBlock
        from agentscope.model import ChatModelBase, ChatResponse
        from agentscope.model._model_usage import ChatUsage

        message_adapter = self._message_adapter

        class AgentRunAgentScopeModel(
            ChatModelBase
        ):  # pragma: no cover - thin wrapper

            def __init__(self, *, stream: bool = False):
                super().__init__(common_model.name, stream=False)
                self._common_model = common_model
                self._message_adapter = message_adapter

            async def __call__(
                self,
                messages: Any,
                tools: Any = None,
                tool_choice: Optional[str] = None,
                **kwargs: Any,
            ) -> ChatResponse:
                openai_messages = self._ensure_openai_messages(messages)
                tool_payload = self._normalize_tools(tools)

                request_kwargs = {
                    "messages": openai_messages,
                    "stream": False,
                }
                if tool_payload:
                    request_kwargs["tools"] = tool_payload
                if tool_choice:
                    request_kwargs["tool_choice"] = tool_choice
                request_kwargs.update(kwargs)

                response = self._common_model.completions(**request_kwargs)
                if inspect.isawaitable(response):
                    response = await response
                elif inspect.isgenerator(response):
                    response = await asyncio.to_thread(list, response)

                return self._build_chat_response(response)

            def _ensure_openai_messages(self, messages: Any) -> List[dict]:
                if not messages:
                    return []

                first = messages[0] if isinstance(messages, list) else messages
                if isinstance(first, dict) and "role" in first:
                    return list(messages)

                if self._message_adapter is None:
                    raise RuntimeError(
                        "AgentScope message adapter is not registered"
                    )

                canonical = self._message_adapter.to_canonical(messages)
                return [msg.to_dict() for msg in canonical]

            @staticmethod
            def _normalize_tools(tools: Any) -> Optional[List[dict]]:
                if tools is None:
                    return None
                if isinstance(tools, list):
                    return tools
                if hasattr(tools, "get_json_schemas"):
                    return tools.get_json_schemas()
                return tools

            def _build_chat_response(self, response: Any) -> ChatResponse:
                payload = self._to_plain_dict(response)
                choices = payload.get("choices") or []

                if not choices:
                    return ChatResponse(
                        content=[
                            TextBlock(type="text", text=str(payload)),
                        ]
                    )

                message = self._to_plain_dict(choices[0].get("message", {}))
                blocks: List[dict] = []

                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and block.get("text")
                        ):
                            blocks.append(
                                TextBlock(
                                    type="text",
                                    text=str(block.get("text")),
                                )
                            )
                elif content:
                    blocks.append(TextBlock(type="text", text=str(content)))

                for call in message.get("tool_calls", []) or []:
                    arguments = call.get("function", {}).get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            pass
                    blocks.append(
                        ToolUseBlock(
                            type="tool_use",
                            id=str(call.get("id", "")),
                            name=call.get("function", {}).get("name", ""),
                            input=arguments
                            if isinstance(arguments, dict)
                            else {},
                        )
                    )

                if not blocks:
                    blocks.append(TextBlock(type="text", text=""))

                usage_payload = payload.get("usage") or {}
                usage = None
                if usage_payload:
                    usage = ChatUsage(
                        input_tokens=int(
                            usage_payload.get("prompt_tokens")
                            or usage_payload.get("input_tokens")
                            or 0
                        ),
                        output_tokens=int(
                            usage_payload.get("completion_tokens")
                            or usage_payload.get("output_tokens")
                            or 0
                        ),
                        time=float(usage_payload.get("time", 0.0)),
                    )

                return ChatResponse(content=blocks, usage=usage)

            @staticmethod
            def _to_plain_dict(value: Any) -> dict:
                if isinstance(value, dict):
                    return value
                for attr in ("model_dump", "dict"):
                    if hasattr(value, attr):
                        try:
                            return getattr(value, attr)()
                        except Exception:  # pragma: no cover - defensive
                            continue
                if hasattr(value, "__dict__"):
                    return {
                        key: val
                        for key, val in value.__dict__.items()
                        if not key.startswith("_")
                    }
                return {"content": str(value)}

        return AgentRunAgentScopeModel()
