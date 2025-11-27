"""LangChain 消息适配器 / LangChain Message Adapter

将 LangChain BaseMessage 转换为标准格式,供 ModelAdapter 内部使用。
Converts LangChain BaseMessage to canonical format for internal use by ModelAdapter.
"""

import json
from typing import Any, List

from agentrun.integration.utils.adapter import MessageAdapter
from agentrun.integration.utils.canonical import (
    CanonicalMessage,
    CanonicalToolCall,
    MessageRole,
)


class LangChainMessageAdapter(MessageAdapter):
    """LangChain 消息适配器 / LangChain Message Adapter

    实现 LangChain BaseMessage → CanonicalMessage 的转换。
    Implements conversion from LangChain BaseMessage to CanonicalMessage.
    """

    def to_canonical(self, messages: Any) -> List[CanonicalMessage]:
        """将 LangChain BaseMessage 转换为标准格式 / LangChain Message Adapter"""
        try:
            from langchain_core.messages import (
                AIMessage,
                HumanMessage,
                SystemMessage,
                ToolMessage,
            )
        except ImportError as e:
            raise ImportError(
                "LangChain is not installed. "
                "Install it with: pip install langchain-core"
            ) from e

        canonical = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                canonical.append(
                    CanonicalMessage(
                        role=MessageRole.SYSTEM,
                        content=msg.content
                        if hasattr(msg, "content")
                        else None,
                    )
                )
            elif isinstance(msg, HumanMessage):
                canonical.append(
                    CanonicalMessage(
                        role=MessageRole.USER,
                        content=msg.content
                        if hasattr(msg, "content")
                        else None,
                    )
                )
            elif isinstance(msg, AIMessage):
                tool_calls = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls = []
                    for call in msg.tool_calls:
                        # LangChain tool_calls: {"id": ..., "name": ..., "args": ...}
                        call_id = (
                            call.get("id")
                            if isinstance(call, dict)
                            else getattr(call, "id", "")
                        )
                        call_name = (
                            call.get("name")
                            if isinstance(call, dict)
                            else getattr(call, "name", "")
                        )
                        call_args = (
                            call.get("args")
                            if isinstance(call, dict)
                            else getattr(call, "args", {})
                        )

                        # 如果 args 是字符串，尝试解析
                        if isinstance(call_args, str):
                            try:
                                call_args = json.loads(call_args)
                            except json.JSONDecodeError:
                                call_args = {}

                        tool_calls.append(
                            CanonicalToolCall(
                                id=str(call_id),
                                name=str(call_name),
                                arguments=(
                                    call_args
                                    if isinstance(call_args, dict)
                                    else {}
                                ),
                            )
                        )

                canonical.append(
                    CanonicalMessage(
                        role=MessageRole.ASSISTANT,
                        content=msg.content
                        if hasattr(msg, "content")
                        else None,
                        tool_calls=tool_calls,
                    )
                )
            elif isinstance(msg, ToolMessage):
                content = msg.content
                if type(content) is not str:
                    content = str(content)

                canonical.append(
                    CanonicalMessage(
                        role=MessageRole.TOOL,
                        content=content,
                        tool_call_id=(
                            msg.tool_call_id
                            if hasattr(msg, "tool_call_id")
                            else None
                        ),
                    )
                )
            else:
                # 未知消息类型，尝试提取基本信息
                role_str = getattr(msg, "type", "user").lower()
                if "system" in role_str:
                    role = MessageRole.SYSTEM
                elif "assistant" in role_str or "ai" in role_str:
                    role = MessageRole.ASSISTANT
                elif "tool" in role_str:
                    role = MessageRole.TOOL
                else:
                    role = MessageRole.USER

                content = getattr(msg, "content", None)
                canonical.append(CanonicalMessage(role=role, content=content))

        return canonical
