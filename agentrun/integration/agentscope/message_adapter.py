"""消息适配器，负责 AgentScope <-> CanonicalMessage 的转换。 / AgentScope Message Adapter"""

from __future__ import annotations

import json
from typing import Any, Iterable, List

from agentrun.integration.utils.adapter import MessageAdapter
from agentrun.integration.utils.canonical import (
    CanonicalMessage,
    CanonicalToolCall,
    MessageRole,
)


def _ensure_agentscope_installed() -> None:
    try:
        import agentscope  # noqa: F401
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "AgentScope is not installed. Install it with: pip install"
            " agentscope"
        ) from exc


class AgentScopeMessageAdapter(MessageAdapter):
    """AgentScope 消息适配器。 / AgentScope Message Adapter"""

    def to_canonical(self, messages: Any) -> List[CanonicalMessage]:
        _ensure_agentscope_installed()

        from agentscope.message import Msg

        if messages is None:
            return []

        if isinstance(messages, Msg):
            iterable: Iterable[Any] = [messages]
        elif isinstance(messages, Iterable):
            iterable = messages
        else:
            iterable = [messages]

        canonical: List[CanonicalMessage] = []
        for item in iterable:
            msg = self._ensure_msg(item)
            text_segments: List[str] = []
            tool_calls: List[CanonicalToolCall] = []

            blocks = (
                msg.content
                if isinstance(msg.content, list)
                else msg.get_content_blocks()
            ) or []

            for block in blocks:
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if text:
                        text_segments.append(str(text))
                elif block_type == "tool_use":
                    tool_calls.append(
                        CanonicalToolCall(
                            id=str(block.get("id", "")),
                            name=str(block.get("name", "")),
                            arguments=block.get("input", {}) or {},
                        )
                    )
                elif block_type == "tool_result":
                    canonical.append(
                        CanonicalMessage(
                            role=MessageRole.TOOL,
                            content=self._format_tool_result(block),
                            tool_call_id=str(block.get("id", "")),
                            name=block.get("name"),
                        )
                    )

            if text_segments or tool_calls or msg.role:
                canonical.append(
                    CanonicalMessage(
                        role=self._resolve_role(msg.role),
                        content="\n".join(text_segments)
                        if text_segments
                        else None,
                        name=msg.name,
                        tool_calls=tool_calls or None,
                    )
                )

        return canonical

    def from_canonical(self, messages: List[CanonicalMessage]) -> Any:
        _ensure_agentscope_installed()

        from agentscope.message import (
            Msg,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
        )

        result: List[Msg] = []
        for msg in messages:
            role = msg.role.value
            blocks: List[dict] = []

            if msg.role == MessageRole.TOOL:
                blocks.append(
                    ToolResultBlock(
                        type="tool_result",
                        id=msg.tool_call_id or "",
                        name=msg.name or "",
                        output=msg.content or "",
                    )
                )
            else:
                if msg.content:
                    blocks.append(
                        TextBlock(
                            type="text",
                            text=msg.content,
                        )
                    )
                for call in msg.tool_calls or []:
                    blocks.append(
                        ToolUseBlock(
                            type="tool_use",
                            id=call.id,
                            name=call.name,
                            input=call.arguments,
                        )
                    )

            content: Any
            if len(blocks) == 1 and blocks[0]["type"] == "text":
                content = blocks[0]["text"]
            else:
                content = blocks or ""

            result.append(
                Msg(
                    name=msg.name or role,
                    role=role,
                    content=content,
                )
            )

        return result

    @staticmethod
    def _ensure_msg(message: Any):
        from agentscope.message import Msg

        if isinstance(message, Msg):
            return message
        if isinstance(message, dict):
            return Msg(
                name=message.get("name", message.get("role", "user")),
                role=message.get("role", "user"),
                content=message.get("content"),
            )
        raise TypeError(f"Unsupported AgentScope message type: {type(message)}")

    @staticmethod
    def _resolve_role(role: str) -> MessageRole:
        role_lower = (role or "user").lower()
        if role_lower.startswith("system"):
            return MessageRole.SYSTEM
        if role_lower.startswith("assistant"):
            return MessageRole.ASSISTANT
        if role_lower.startswith("tool"):
            return MessageRole.TOOL
        return MessageRole.USER

    @staticmethod
    def _format_tool_result(block: dict) -> str:
        output = block.get("output")
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output, ensure_ascii=False)
        except (TypeError, ValueError):  # pragma: no cover - fallback
            return str(output)
