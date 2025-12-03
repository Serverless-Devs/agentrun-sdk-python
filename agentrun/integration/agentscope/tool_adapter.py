"""AgentScope 工具适配器 / AgentScope Tool Adapter

将标准工具定义转换为 AgentScope 工具格式。"""

from __future__ import annotations

import json
from typing import Any, List

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from agentrun.integration.utils.adapter import ToolAdapter
from agentrun.integration.utils.canonical import CanonicalTool


def _ensure_agentscope_installed() -> None:
    try:
        import agentscope  # noqa: F401
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "AgentScope is not installed. Install it with: pip install"
            " agentscope"
        ) from exc


class AgentScopeToolAdapter(ToolAdapter):
    """AgentScope 工具适配器 / AgentScope Tool Adapter

    实现 CanonicalTool → AgentScope 工具格式的转换。"""

    def _modify_tool(self, tool: CanonicalTool, *args, **kwargs) -> Any:
        """修改工具函数以符合 AgentScope 的要求 / AgentScope Tool Adapter"""
        _ensure_agentscope_installed()

        if tool.func is None:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="工具未实现。",
                    ),
                ],
            )

        result = tool.func(*args, **kwargs)
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=result if type(result) is str else json.dumps(result),
                ),
            ],
        )

    def from_canonical(self, tools: List[CanonicalTool]) -> Any:
        """将标准格式转换为 AgentScope 工具 / AgentScope Tool Adapter"""

        return self.function_tools(tools, modify_func=self._modify_tool)
