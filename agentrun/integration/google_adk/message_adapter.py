"""Google ADK 消息适配器 / Google ADK Message Adapter

将 Google ADK LlmRequest 转换为标准格式,供 ModelAdapter 内部使用。
Converts Google ADK LlmRequest to canonical format for internal use by ModelAdapter.
"""

import json
from typing import Any, List

from agentrun.integration.utils.adapter import MessageAdapter
from agentrun.integration.utils.canonical import (
    CanonicalMessage,
    CanonicalToolCall,
    MessageRole,
)


class GoogleADKMessageAdapter(MessageAdapter):
    """Google ADK 消息适配器 / Google ADK Message Adapter

    实现 Google ADK LlmRequest → CanonicalMessage 的转换。
    Implements conversion from Google ADK LlmRequest to CanonicalMessage.
    """

    def to_canonical(self, messages: Any) -> List[CanonicalMessage]:
        """将 Google ADK LlmRequest 转换为标准格式 / Google ADK Message Adapter"""
        canonical = []

        # Google ADK 使用 LlmRequest，包含 contents 列表
        if hasattr(messages, "contents"):
            contents = messages.contents
        elif isinstance(messages, list):
            contents = messages
        else:
            contents = [messages]

        # 处理 system_instruction (在 config 中)
        if hasattr(messages, "config") and messages.config:
            if (
                hasattr(messages.config, "system_instruction")
                and messages.config.system_instruction
            ):
                canonical.append(
                    CanonicalMessage(
                        role=MessageRole.SYSTEM,
                        content=messages.config.system_instruction,
                    )
                )

        # 处理 contents
        for content in contents:
            # 确定角色
            role = MessageRole.USER
            if hasattr(content, "role"):
                role_str = str(content.role).lower()
                if "model" in role_str or "assistant" in role_str:
                    role = MessageRole.ASSISTANT
                elif "system" in role_str:
                    role = MessageRole.SYSTEM
                elif "tool" in role_str:
                    role = MessageRole.TOOL
                elif "function" in role_str:
                    role = MessageRole.TOOL
                else:
                    role = MessageRole.USER

            # 处理 parts
            if hasattr(content, "parts"):
                text_parts = []
                tool_calls = []

                for part in content.parts:
                    # 处理文本
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)

                    # 处理 function_call
                    elif hasattr(part, "function_call") and part.function_call:
                        func_call = part.function_call
                        args = {}
                        if hasattr(func_call, "args"):
                            if isinstance(func_call.args, dict):
                                args = func_call.args
                            else:
                                try:
                                    args = json.loads(str(func_call.args))
                                except (json.JSONDecodeError, TypeError):
                                    args = {}

                        tool_calls.append(
                            CanonicalToolCall(
                                id=getattr(
                                    func_call, "id", f"call_{len(tool_calls)}"
                                ),
                                name=getattr(func_call, "name", ""),
                                arguments=args,
                            )
                        )

                    # 处理 function_response
                    elif (
                        hasattr(part, "function_response")
                        and part.function_response
                    ):
                        func_resp = part.function_response
                        response_content = ""
                        if hasattr(func_resp, "response"):
                            if isinstance(func_resp.response, dict):
                                response_content = json.dumps(
                                    func_resp.response
                                )
                            else:
                                response_content = str(func_resp.response)
                        else:
                            response_content = str(func_resp)

                        # function_response 表示工具返回结果
                        canonical.append(
                            CanonicalMessage(
                                role=MessageRole.TOOL,
                                content=response_content,
                                tool_call_id=getattr(func_resp, "id", "call_0"),
                            )
                        )
                        continue

                # 构建消息
                if text_parts or tool_calls:
                    content_text = " ".join(text_parts) if text_parts else None
                    canonical.append(
                        CanonicalMessage(
                            role=role,
                            content=content_text,
                            tool_calls=tool_calls if tool_calls else None,
                        )
                    )
            else:
                # 没有 parts，直接使用字符串内容
                content_text = str(content) if content else None
                canonical.append(
                    CanonicalMessage(role=role, content=content_text)
                )

        return canonical
