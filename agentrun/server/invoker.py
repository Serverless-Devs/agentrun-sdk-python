"""Agent 调用器 / Agent Invoker

负责处理 Agent 调用的通用逻辑。
Handles common logic for agent invocations.
"""

import asyncio
import inspect
from typing import cast

from .model import AgentRequest, AgentResult, AgentRunResult
from .protocol import (
    AsyncInvokeAgentHandler,
    InvokeAgentHandler,
    SyncInvokeAgentHandler,
)


class AgentInvoker:
    """Agent 调用器

    职责:
    1. 调用用户的 invoke_agent
    2. 处理同步/异步调用
    3. 自动转换 string/string迭代器为 AgentRunResult
    4. 错误处理

    Example:
        >>> def my_agent(request: AgentRequest) -> str:
        ...     return "Hello"  # 自动转换为 AgentRunResult
        >>>
        >>> invoker = AgentInvoker(my_agent)
        >>> result = await invoker.invoke(AgentRequest(...))
        >>> # result 是 AgentRunResult 对象
    """

    def __init__(self, invoke_agent: InvokeAgentHandler):
        """初始化 Agent 调用器

        Args:
            invoke_agent: Agent 处理函数,可以是同步或异步
        """
        self.invoke_agent = invoke_agent
        self.is_async = inspect.iscoroutinefunction(invoke_agent)

    async def invoke(self, request: AgentRequest) -> AgentResult:
        """调用 Agent 并返回结果

        自动处理各种返回类型:
        - string 或 string 迭代器 -> 转换为 AgentRunResult
        - AgentRunResult -> 直接返回
        - AgentResponse/ModelResponse -> 直接返回

        Args:
            request: AgentRequest 请求对象

        Returns:
            AgentResult: Agent 返回的结果

        Raises:
            Exception: Agent 执行中的任何异常
        """
        if self.is_async:
            # 异步 handler
            async_handler = cast(AsyncInvokeAgentHandler, self.invoke_agent)
            result = await async_handler(request)
        else:
            # 同步 handler: 在线程池中运行,避免阻塞事件循环
            sync_handler = cast(SyncInvokeAgentHandler, self.invoke_agent)
            result = await asyncio.get_event_loop().run_in_executor(
                None, sync_handler, request
            )

        # 自动转换 string 或 string 迭代器为 AgentRunResult
        result = self._normalize_result(result)

        return result

    def _normalize_result(self, result: AgentResult) -> AgentResult:
        """标准化返回结果

        将 string 或 string 迭代器自动转换为 AgentRunResult。

        Args:
            result: 原始返回结果

        Returns:
            AgentResult: 标准化后的结果
        """
        # 如果是字符串,转换为 AgentRunResult
        if isinstance(result, str):
            return AgentRunResult(content=result)

        # 如果是迭代器,检查是否是字符串迭代器
        if self._is_string_iterator(result):
            return AgentRunResult(content=result)  # type: ignore

        # 其他类型直接返回
        return result

    def _is_string_iterator(self, obj) -> bool:
        """检查是否是字符串迭代器

        通过类型注解或启发式方法判断。

        Args:
            obj: 要检查的对象

        Returns:
            bool: 是否是字符串迭代器
        """
        # 排除已知的复杂类型
        from .model import AgentResponse, AgentRunResult

        if isinstance(obj, (AgentResponse, AgentRunResult, str, dict)):
            return False

        # 检查是否是迭代器
        is_iterator = (
            hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict))
        ) or hasattr(obj, "__aiter__")

        if not is_iterator:
            return False

        # 启发式判断: 如果没有 choices 属性,很可能是字符串迭代器
        # (AgentResponse/ModelResponse 都有 choices 属性)
        if hasattr(obj, "choices") or hasattr(obj, "model"):
            return False

        return True
