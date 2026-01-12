"""知识库工具集 / KnowledgeBase ToolSet

提供知识库检索功能的工具集，支持多知识库联合检索。
Provides toolset for knowledge base retrieval, supporting multi-knowledge-base search.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentrun.integration.utils.tool import CommonToolSet, tool
from agentrun.knowledgebase import KnowledgeBase
from agentrun.utils.config import Config


class KnowledgeBaseToolSet(CommonToolSet):
    """知识库工具集 / KnowledgeBase ToolSet

    提供知识库检索功能，支持对多个知识库进行联合检索。
    Provides knowledge base retrieval capabilities, supporting joint retrieval
    across multiple knowledge bases.

    使用指南 / Usage Guide:
    ============================================================

    ## 基本用法 / Basic Usage

    1. **创建工具集 / Create ToolSet**:
       - 使用 `knowledgebase_toolset` 函数创建工具集实例
       - Use `knowledgebase_toolset` function to create a toolset instance
       - 指定要检索的知识库名称列表
       - Specify the list of knowledge base names to search

    2. **执行检索 / Execute Search**:
       - 调用 `search_document` 工具进行检索
       - Call `search_document` tool to perform retrieval
       - 返回所有指定知识库的检索结果
       - Returns retrieval results from all specified knowledge bases

    ## 示例 / Examples

    ```python
    from agentrun.integration.langchain import knowledgebase_toolset

    # 创建工具集 / Create toolset
    tools = knowledgebase_toolset(
        knowledge_base_names=["kb-product-docs", "kb-faq"],
    )

    # 在 Agent 中使用 / Use in Agent
    agent = create_react_agent(llm, tools)
    ```
    """

    def __init__(
        self,
        knowledge_base_names: List[str],
        config: Optional[Config] = None,
    ) -> None:
        """初始化知识库工具集 / Initialize KnowledgeBase ToolSet

        Args:
            knowledge_base_names: 知识库名称列表 / List of knowledge base names
            config: 配置 / Configuration
        """
        super().__init__()

        self.knowledge_base_names = knowledge_base_names
        self.config = config

    @tool(
        name="search_document",
        description=(
            "Search and retrieve relevant documents from configured knowledge"
            " bases. Use this tool when you need to find information from the"
            " knowledge base to answer user questions. Returns relevant"
            " document chunks with their content and metadata. The search is"
            " performed across all configured knowledge bases and results are"
            " grouped by knowledge base name."
        ),
    )
    def search_document(self, query: str) -> Dict[str, Any]:
        """检索文档 / Search documents

        根据查询文本从配置的知识库中检索相关文档。
        Retrieves relevant documents from configured knowledge bases based on query text.

        Args:
            query: 查询文本 / Query text

        Returns:
            Dict[str, Any]: 检索结果，包含各知识库的检索结果 /
                           Retrieval results containing results from each knowledge base
        """
        return KnowledgeBase.multi_retrieve(
            query=query,
            knowledge_base_names=self.knowledge_base_names,
            config=self.config,
        )


def knowledgebase_toolset(
    knowledge_base_names: List[str],
    *,
    config: Optional[Config] = None,
) -> KnowledgeBaseToolSet:
    """创建知识库工具集 / Create KnowledgeBase ToolSet

    将知识库检索功能封装为通用工具集，可转换为各框架支持的格式。
    Wraps knowledge base retrieval functionality into a common toolset that can be
    converted to formats supported by various frameworks.

    Args:
        knowledge_base_names: 知识库名称列表 / List of knowledge base names
        config: 配置 / Configuration

    Returns:
        KnowledgeBaseToolSet: 知识库工具集实例 / KnowledgeBase toolset instance

    Example:
        >>> from agentrun.integration.builtin import knowledgebase_toolset
        >>>
        >>> # 创建工具集 / Create toolset
        >>> kb_tools = knowledgebase_toolset(
        ...     knowledge_base_names=["kb-docs", "kb-faq"],
        ... )
        >>>
        >>> # 转换为 LangChain 格式 / Convert to LangChain format
        >>> langchain_tools = kb_tools.to_langchain()
        >>>
        >>> # 转换为 Google ADK 格式 / Convert to Google ADK format
        >>> adk_tools = kb_tools.to_google_adk()
    """
    return KnowledgeBaseToolSet(
        knowledge_base_names=knowledge_base_names,
        config=config,
    )
