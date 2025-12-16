"""Agent 调用器 / Agent Invoker

负责处理 Agent 调用的通用逻辑，包括：
- 同步/异步调用处理
- 字符串到 AgentEvent 的自动转换
- 流式/非流式结果处理
- TOOL_CALL 事件的展开

边界事件（如生命周期开始/结束、文本消息开始/结束）由协议层处理。
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

from .model import AgentEvent, AgentRequest, EventType
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
    3. 自动转换 string 为 AgentEvent(TEXT)
    4. 展开 TOOL_CALL 为 TOOL_CALL_CHUNK
    5. 处理流式和非流式返回

    协议层负责:
    - 生成生命周期事件（RUN_STARTED, RUN_FINISHED 等）
    - 生成文本边界事件（TEXT_MESSAGE_START, TEXT_MESSAGE_END 等）
    - 生成工具调用边界事件（TOOL_CALL_START, TOOL_CALL_END 等）

    Example:
        >>> def my_agent(request: AgentRequest) -> str:
        ...     return "Hello"  # 自动转换为 TEXT 事件
        >>>
        >>> invoker = AgentInvoker(my_agent)
        >>> async for event in invoker.invoke_stream(AgentRequest(...)):
        ...     print(event)  # AgentEvent 对象
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
    ) -> Union[List[AgentEvent], AsyncGenerator[AgentEvent, None]]:
        """调用 Agent 并返回结果

        根据返回值类型决定返回：
        - 非迭代器: 返回 List[AgentEvent]
        - 迭代器: 返回 AsyncGenerator[AgentEvent, None]

        Args:
            request: AgentRequest 请求对象

        Returns:
            List[AgentEvent] 或 AsyncGenerator[AgentEvent, None]
        """
        raw_result = await self._call_handler(request)

        if self._is_iterator(raw_result):
            return self._wrap_stream(raw_result)
        else:
            return self._wrap_non_stream(raw_result)

    async def invoke_stream(
        self, request: AgentRequest
    ) -> AsyncGenerator[AgentEvent, None]:
        """调用 Agent 并返回流式结果

        始终返回流式结果，即使原始返回值是非流式的。
        只输出核心事件，边界事件由协议层生成。

        Args:
            request: AgentRequest 请求对象

        Yields:
            AgentEvent: 事件结果
        """
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
                        yield AgentEvent(
                            event=EventType.TEXT,
                            data={"delta": item},
                        )

                    elif isinstance(item, AgentEvent):
                        # 处理用户返回的事件
                        for processed_event in self._process_user_event(item):
                            yield processed_event
            else:
                # 非流式结果
                results = self._wrap_non_stream(raw_result)
                for result in results:
                    yield result

        except Exception as e:
            # 发送错误事件
            from agentrun.utils.log import logger

            logger.error(f"Agent 调用出错: {e}", exc_info=True)
            yield AgentEvent(
                event=EventType.ERROR,
                data={"message": str(e), "code": type(e).__name__},
            )

    def _process_user_event(
        self,
        event: AgentEvent,
    ) -> Iterator[AgentEvent]:
        """处理用户返回的事件

        - TOOL_CALL 事件会被展开为 TOOL_CALL_CHUNK
        - 其他事件直接传递

        Args:
            event: 用户返回的事件

        Yields:
            处理后的事件
        """
        # 展开 TOOL_CALL 为 TOOL_CALL_CHUNK
        if event.event == EventType.TOOL_CALL:
            tool_id = event.data.get("id", str(uuid.uuid4()))
            tool_name = event.data.get("name", "")
            tool_args = event.data.get("args", "")

            # 发送包含名称的 chunk（首个 chunk 包含名称）
            yield AgentEvent(
                event=EventType.TOOL_CALL_CHUNK,
                data={
                    "id": tool_id,
                    "name": tool_name,
                    "args_delta": tool_args,
                },
            )
            return

        # 其他事件直接传递
        yield event

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

    def _wrap_non_stream(self, result: Any) -> List[AgentEvent]:
        """包装非流式结果为 AgentEvent 列表

        Args:
            result: 原始返回值

        Returns:
            AgentEvent 列表
        """
        results: List[AgentEvent] = []

        if result is None:
            return results

        if isinstance(result, str):
            results.append(
                AgentEvent(
                    event=EventType.TEXT,
                    data={"delta": result},
                )
            )

        elif isinstance(result, AgentEvent):
            # 处理可能的 TOOL_CALL 展开
            results.extend(self._process_user_event(result))

        elif isinstance(result, list):
            for item in result:
                if isinstance(item, AgentEvent):
                    results.extend(self._process_user_event(item))
                elif isinstance(item, str) and item:
                    results.append(
                        AgentEvent(
                            event=EventType.TEXT,
                            data={"delta": item},
                        )
                    )

        return results

    async def _wrap_stream(
        self, iterator: Any
    ) -> AsyncGenerator[AgentEvent, None]:
        """包装迭代器为 AgentEvent 异步生成器

        Args:
            iterator: 原始迭代器

        Yields:
            AgentEvent: 事件结果
        """
        async for item in self._iterate_async(iterator):
            if item is None:
                continue

            if isinstance(item, str):
                if not item:
                    continue
                yield AgentEvent(
                    event=EventType.TEXT,
                    data={"delta": item},
                )

            elif isinstance(item, AgentEvent):
                for processed_event in self._process_user_event(item):
                    yield processed_event

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
        if isinstance(obj, (str, bytes, dict, list, AgentEvent)):
            return False
        return hasattr(obj, "__iter__") or hasattr(obj, "__aiter__")
