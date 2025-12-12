"""Agent 调用器 / Agent Invoker

负责处理 Agent 调用的通用逻辑，包括：
- 同步/异步调用处理
- 字符串到 AgentResult 的自动转换
- 流式/非流式结果处理
"""

import asyncio
import inspect
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    cast,
    Iterator,
    List,
    Union,
)
import uuid

from .model import AgentRequest, AgentResult, AgentResultItem, EventType
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
    3. 自动转换 string 为 AgentResult
    4. 处理流式和非流式返回

    Example:
        >>> def my_agent(request: AgentRequest) -> str:
        ...     return "Hello"  # 自动转换为 TEXT_MESSAGE_CONTENT
        >>>
        >>> invoker = AgentInvoker(my_agent)
        >>> async for result in invoker.invoke_stream(AgentRequest(...)):
        ...     print(result)  # AgentResult 对象
    """

    def __init__(self, invoke_agent: InvokeAgentHandler):
        """初始化 Agent 调用器

        Args:
            invoke_agent: Agent 处理函数，可以是同步或异步
        """
        self.invoke_agent = invoke_agent
        # 检测是否是异步函数或异步生成器
        self.is_async = inspect.iscoroutinefunction(
            invoke_agent
        ) or inspect.isasyncgenfunction(invoke_agent)

    async def invoke(
        self, request: AgentRequest
    ) -> Union[List[AgentResult], AsyncGenerator[AgentResult, None]]:
        """调用 Agent 并返回结果

        根据返回值类型决定返回：
        - 非迭代器: 返回 List[AgentResult]
        - 迭代器: 返回 AsyncGenerator[AgentResult, None]

        Args:
            request: AgentRequest 请求对象

        Returns:
            List[AgentResult] 或 AsyncGenerator[AgentResult, None]
        """
        raw_result = await self._call_handler(request)

        if self._is_iterator(raw_result):
            return self._wrap_stream(raw_result)
        else:
            return self._wrap_non_stream(raw_result)

    async def invoke_stream(
        self, request: AgentRequest
    ) -> AsyncGenerator[AgentResult, None]:
        """调用 Agent 并返回流式结果

        始终返回流式结果，即使原始返回值是非流式的。
        自动添加 RUN_STARTED 和 RUN_FINISHED 事件。

        Args:
            request: AgentRequest 请求对象

        Yields:
            AgentResult: 事件结果
        """
        thread_id = self._get_thread_id(request)
        run_id = self._get_run_id(request)
        message_id = str(uuid.uuid4())

        # 状态追踪
        text_started = False
        text_ended = False

        # 发送 RUN_STARTED
        yield AgentResult(
            event=EventType.RUN_STARTED,
            data={"thread_id": thread_id, "run_id": run_id},
        )

        try:
            raw_result = await self._call_handler(request)

            if self._is_iterator(raw_result):
                # 流式结果 - 逐个处理
                async for item in self._iterate_async(raw_result):
                    if item is None:
                        continue

                    if isinstance(item, str):
                        if not item:  # 跳过空字符串
                            continue
                        # 字符串：需要包装为文本消息事件
                        if not text_started:
                            yield AgentResult(
                                event=EventType.TEXT_MESSAGE_START,
                                data={
                                    "message_id": message_id,
                                    "role": "assistant",
                                },
                            )
                            text_started = True
                        yield AgentResult(
                            event=EventType.TEXT_MESSAGE_CONTENT,
                            data={"message_id": message_id, "delta": item},
                        )

                    elif isinstance(item, AgentResult):
                        # 用户返回的事件
                        if item.event == EventType.TEXT_MESSAGE_START:
                            text_started = True
                        elif item.event == EventType.TEXT_MESSAGE_END:
                            text_ended = True
                        yield item
            else:
                # 非流式结果
                results = self._wrap_non_stream(raw_result)
                for result in results:
                    if result.event == EventType.TEXT_MESSAGE_START:
                        text_started = True
                    elif result.event == EventType.TEXT_MESSAGE_END:
                        text_ended = True
                    yield result

            # 发送 TEXT_MESSAGE_END（如果有文本消息且未发送）
            if text_started and not text_ended:
                yield AgentResult(
                    event=EventType.TEXT_MESSAGE_END,
                    data={"message_id": message_id},
                )

            # 发送 RUN_FINISHED
            yield AgentResult(
                event=EventType.RUN_FINISHED,
                data={"thread_id": thread_id, "run_id": run_id},
            )

        except Exception as e:
            # 发送 RUN_ERROR

            from agentrun.utils.log import logger

            logger.error(f"Agent 调用出错: {e}", exc_info=True)
            yield AgentResult(
                event=EventType.RUN_ERROR,
                data={"message": str(e), "code": type(e).__name__},
            )

    async def _call_handler(self, request: AgentRequest) -> Any:
        """调用用户的 handler

        Args:
            request: AgentRequest 请求对象

        Returns:
            原始返回值
        """
        if self.is_async:
            async_handler = cast(AsyncInvokeAgentHandler, self.invoke_agent)
            raw_result = async_handler(request)

            if inspect.isawaitable(raw_result):
                result = await cast(Awaitable[Any], raw_result)
            elif inspect.isasyncgen(raw_result):
                result = raw_result
            else:
                result = raw_result
        else:
            sync_handler = cast(SyncInvokeAgentHandler, self.invoke_agent)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, sync_handler, request)

        return result

    def _wrap_non_stream(self, result: Any) -> List[AgentResult]:
        """包装非流式结果为 AgentResult 列表

        Args:
            result: 原始返回值

        Returns:
            AgentResult 列表
        """
        message_id = str(uuid.uuid4())
        results: List[AgentResult] = []

        if result is None:
            return results

        if isinstance(result, str):
            results.append(
                AgentResult(
                    event=EventType.TEXT_MESSAGE_START,
                    data={"message_id": message_id, "role": "assistant"},
                )
            )
            results.append(
                AgentResult(
                    event=EventType.TEXT_MESSAGE_CONTENT,
                    data={"message_id": message_id, "delta": result},
                )
            )
            results.append(
                AgentResult(
                    event=EventType.TEXT_MESSAGE_END,
                    data={"message_id": message_id},
                )
            )

        elif isinstance(result, AgentResult):
            results.append(result)

        elif isinstance(result, list):
            for item in result:
                if isinstance(item, AgentResult):
                    results.append(item)
                elif isinstance(item, str) and item:
                    results.append(
                        AgentResult(
                            event=EventType.TEXT_MESSAGE_CONTENT,
                            data={"message_id": message_id, "delta": item},
                        )
                    )

        return results

    async def _wrap_stream(
        self, iterator: Any
    ) -> AsyncGenerator[AgentResult, None]:
        """包装迭代器为 AgentResult 异步生成器

        注意：此方法不添加生命周期事件，由 invoke_stream 处理。

        Args:
            iterator: 原始迭代器

        Yields:
            AgentResult: 事件结果
        """
        message_id = str(uuid.uuid4())
        text_started = False

        async for item in self._iterate_async(iterator):
            if item is None:
                continue

            if isinstance(item, str):
                if not item:
                    continue
                if not text_started:
                    yield AgentResult(
                        event=EventType.TEXT_MESSAGE_START,
                        data={"message_id": message_id, "role": "assistant"},
                    )
                    text_started = True
                yield AgentResult(
                    event=EventType.TEXT_MESSAGE_CONTENT,
                    data={"message_id": message_id, "delta": item},
                )

            elif isinstance(item, AgentResult):
                if item.event == EventType.TEXT_MESSAGE_START:
                    text_started = True
                yield item

    async def _iterate_async(
        self, content: Union[Iterator[Any], AsyncIterator[Any]]
    ) -> AsyncGenerator[Any, None]:
        """统一迭代同步和异步迭代器

        对于同步迭代器，每次 next() 调用都在线程池中执行，避免阻塞事件循环。

        Args:
            content: 迭代器

        Yields:
            迭代器中的元素
        """
        if hasattr(content, "__aiter__"):
            async for chunk in content:
                yield chunk
        else:
            loop = asyncio.get_running_loop()
            iterator = iter(content)

            _STOP = object()

            def _safe_next() -> Any:
                try:
                    return next(iterator)
                except StopIteration:
                    return _STOP

            while True:
                chunk = await loop.run_in_executor(None, _safe_next)
                if chunk is _STOP:
                    break
                yield chunk

    def _is_iterator(self, obj: Any) -> bool:
        """检查对象是否是迭代器"""
        if isinstance(obj, (str, bytes, dict, list, AgentResult)):
            return False
        return hasattr(obj, "__iter__") or hasattr(obj, "__aiter__")

    def _get_thread_id(self, request: AgentRequest) -> str:
        """获取 thread ID"""
        return (
            request.body.get("threadId")
            or request.body.get("thread_id")
            or str(uuid.uuid4())
        )

    def _get_run_id(self, request: AgentRequest) -> str:
        """获取 run ID"""
        return (
            request.body.get("runId")
            or request.body.get("run_id")
            or str(uuid.uuid4())
        )
