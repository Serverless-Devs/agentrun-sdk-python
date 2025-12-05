"""中间格式定义 / Canonical Format Definition

提供统一的中间格式(Canonical Format),作为所有框架转换的桥梁。
Provides a unified canonical format that serves as a bridge for all framework conversions.

这样可以最大化代码复用,减少重复的转换逻辑。
This maximizes code reuse and reduces redundant conversion logic.
"""

from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Protocol
else:
    Protocol = object

from enum import Enum

from agentrun.integration.utils.tool import normalize_tool_name


class MessageRole(str, Enum):
    """统一的消息角色枚举"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class CanonicalToolCall:
    """统一的工具调用格式

    所有框架的工具调用都转换为这个格式，然后再转换为目标框架格式。
    """

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class CanonicalMessage:
    """统一的消息格式

    这是所有框架消息格式的中间表示。
    转换流程：框架A格式 → CanonicalMessage → 框架B格式
    """

    role: MessageRole
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[CanonicalToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于序列化）

        注意：OpenAI API 要求 tool_calls 中的 arguments 必须是 JSON 字符串，
        而不是字典。这里会自动转换。
        """
        result = {
            "role": self.role.value,
        }
        if self.content is not None:
            result["content"] = self.content
        if self.name is not None:
            result["name"] = self.name
        if self.tool_calls is not None:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        # arguments 必须是 JSON 字符串（OpenAI API 要求）
                        "arguments": (
                            json.dumps(call.arguments)
                            if isinstance(call.arguments, dict)
                            else str(call.arguments)
                        ),
                    },
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class CanonicalTool:
    """统一的工具格式

    所有框架的工具都转换为这个格式。
    参数使用 JSON Schema 格式，这是最通用的工具描述格式。
    """

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema 格式
    func: Optional[Callable] = None

    def __post_init__(self):
        # Normalize canonical tool name to avoid exceeding provider limits
        if self.name:
            self.name = normalize_tool_name(self.name)

    def to_openai_function(self) -> Dict[str, Any]:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """转换为 Anthropic Claude Tools 格式"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class CanonicalModelResponse:
    """统一的模型响应格式"""

    content: Optional[str] = None
    tool_calls: Optional[List[CanonicalToolCall]] = None
    usage: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        注意：OpenAI API 要求 tool_calls 中的 arguments 必须是 JSON 字符串。
        """
        result = {}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls is not None:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        # arguments 必须是 JSON 字符串（OpenAI API 要求）
                        "arguments": (
                            json.dumps(call.arguments)
                            if isinstance(call.arguments, dict)
                            else str(call.arguments)
                        ),
                    },
                }
                for call in self.tool_calls
            ]
        if self.usage is not None:
            result["usage"] = self.usage
        return result
