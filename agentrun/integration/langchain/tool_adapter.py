"""LangChain 工具适配器 / LangChain Tool Adapter

将标准工具定义转换为 LangChain StructuredTool 格式。"""

from typing import Any, List

from agentrun.integration.utils.adapter import ToolAdapter
from agentrun.integration.utils.canonical import CanonicalTool


class LangChainToolAdapter(ToolAdapter):
    """LangChain 工具适配器 / LangChain Tool Adapter

    实现 CanonicalTool → LangChain StructuredTool 的转换。"""

    def from_canonical(self, tools: List[CanonicalTool]) -> Any:
        """将标准格式转换为 LangChain StructuredTool / LangChain Tool Adapter"""
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as e:
            raise ImportError(
                "LangChain is not installed. "
                "Install it with: pip install langchain-core"
            ) from e

        from agentrun.integration.utils.tool import _json_schema_to_pydantic

        result = []
        for tool in tools:
            # 从 JSON Schema 创建 Pydantic 模型
            args_schema = _json_schema_to_pydantic(
                f"{tool.name}_Args", tool.parameters
            )

            if args_schema is None:
                # 如果无法创建 schema，使用空模型
                from pydantic import create_model

                args_schema = create_model(f"{tool.name}_Args")

            result.append(
                StructuredTool.from_function(
                    func=tool.func,
                    name=tool.name,
                    description=tool.description,
                    args_schema=args_schema,
                )
            )

        return result
