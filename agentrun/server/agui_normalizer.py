"""AG-UI 事件规范化器

提供事件流规范化功能，确保事件符合 AG-UI 协议的顺序要求：
- TOOL_CALL_START 必须在 TOOL_CALL_ARGS 之前
- TOOL_CALL_END 必须在收到新的文本消息前发送
- 重复的 TOOL_CALL_START 会被忽略

使用示例:

    >>> from agentrun.server.agui_normalizer import AguiEventNormalizer
    >>>
    >>> normalizer = AguiEventNormalizer()
    >>> for event in raw_events:
    ...     for normalized_event in normalizer.normalize(event):
    ...         yield normalized_event
"""

from typing import Any, Dict, Iterator, List, Optional, Set, Union

from .model import AgentResult, EventType


class AguiEventNormalizer:
    """AG-UI 事件规范化器

    自动修正事件顺序，确保符合 AG-UI 协议规范：
    1. 如果收到 TOOL_CALL_ARGS 但之前没有 TOOL_CALL_START，自动补上
    2. 如果收到重复的 TOOL_CALL_START（相同 tool_call_id），忽略
    3. 如果发送 TEXT_MESSAGE_CONTENT 时有未结束的工具调用，自动发送 TOOL_CALL_END

    AG-UI 协议要求的事件顺序：
    TOOL_CALL_START → TOOL_CALL_ARGS (多个) → TOOL_CALL_END → TOOL_CALL_RESULT

    Example:
        >>> normalizer = AguiEventNormalizer()
        >>> for event in agent_events:
        ...     for normalized in normalizer.normalize(event):
        ...         yield normalized
    """

    def __init__(self):
        # 已发送 TOOL_CALL_START 的 tool_call_id 集合
        self._started_tool_calls: Set[str] = set()
        # 已发送 TOOL_CALL_END 的 tool_call_id 集合
        self._ended_tool_calls: Set[str] = set()
        # 活跃的工具调用信息（tool_call_id -> tool_call_name）
        self._active_tool_calls: Dict[str, str] = {}

    def normalize(
        self,
        event: Union[AgentResult, str, Dict[str, Any]],
    ) -> Iterator[AgentResult]:
        """规范化单个事件

        根据 AG-UI 协议要求，可能会产生多个输出事件：
        - 在 TOOL_CALL_ARGS 前补充 TOOL_CALL_START
        - 在 TEXT_MESSAGE_CONTENT 前补充未结束的 TOOL_CALL_END

        Args:
            event: 原始事件（AgentResult、str 或 dict）

        Yields:
            规范化后的事件
        """
        # 将事件标准化为 AgentResult
        normalized_event = self._to_agent_result(event)
        if normalized_event is None:
            return

        # 根据事件类型进行处理
        event_type = normalized_event.event

        if event_type == EventType.TOOL_CALL_START:
            yield from self._handle_tool_call_start(normalized_event)

        elif event_type == EventType.TOOL_CALL_ARGS:
            yield from self._handle_tool_call_args(normalized_event)

        elif event_type == EventType.TOOL_CALL_END:
            yield from self._handle_tool_call_end(normalized_event)

        elif event_type == EventType.TOOL_CALL_RESULT:
            yield from self._handle_tool_call_result(normalized_event)

        elif event_type in (
            EventType.TEXT_MESSAGE_START,
            EventType.TEXT_MESSAGE_CONTENT,
            EventType.TEXT_MESSAGE_END,
            EventType.TEXT_MESSAGE_CHUNK,
        ):
            yield from self._handle_text_message(normalized_event)

        else:
            # 其他事件类型直接传递
            yield normalized_event

    def _to_agent_result(
        self, event: Union[AgentResult, str, Dict[str, Any]]
    ) -> Optional[AgentResult]:
        """将事件转换为 AgentResult"""
        if isinstance(event, AgentResult):
            return event

        if isinstance(event, str):
            # 字符串转为 TEXT_MESSAGE_CONTENT
            return AgentResult(
                event=EventType.TEXT_MESSAGE_CONTENT,
                data={"delta": event},
            )

        if isinstance(event, dict):
            event_type = event.get("event")
            if event_type is None:
                return None

            # 尝试解析 event_type
            if isinstance(event_type, str):
                try:
                    event_type = EventType(event_type)
                except ValueError:
                    try:
                        event_type = EventType[event_type]
                    except KeyError:
                        return None

            return AgentResult(
                event=event_type,
                data=event.get("data", {}),
            )

        return None

    def _handle_tool_call_start(
        self, event: AgentResult
    ) -> Iterator[AgentResult]:
        """处理 TOOL_CALL_START 事件

        如果该 tool_call_id 已经发送过 START，则忽略
        """
        tool_call_id = event.data.get("tool_call_id", "")
        tool_call_name = event.data.get("tool_call_name", "")

        if not tool_call_id:
            yield event
            return

        if tool_call_id in self._started_tool_calls:
            # 重复的 START，忽略
            return

        # 记录并发送
        self._started_tool_calls.add(tool_call_id)
        self._active_tool_calls[tool_call_id] = tool_call_name
        yield event

    def _handle_tool_call_args(
        self, event: AgentResult
    ) -> Iterator[AgentResult]:
        """处理 TOOL_CALL_ARGS 事件

        如果该 tool_call_id 没有发送过 START，自动补上
        """
        tool_call_id = event.data.get("tool_call_id", "")

        if not tool_call_id:
            yield event
            return

        if tool_call_id not in self._started_tool_calls:
            # 需要补充 TOOL_CALL_START
            yield AgentResult(
                event=EventType.TOOL_CALL_START,
                data={
                    "tool_call_id": tool_call_id,
                    "tool_call_name": "",  # 没有名称信息
                },
            )
            self._started_tool_calls.add(tool_call_id)
            self._active_tool_calls[tool_call_id] = ""

        yield event

    def _handle_tool_call_end(
        self, event: AgentResult
    ) -> Iterator[AgentResult]:
        """处理 TOOL_CALL_END 事件

        如果该 tool_call_id 没有发送过 START，先补上 START
        """
        tool_call_id = event.data.get("tool_call_id", "")

        if not tool_call_id:
            yield event
            return

        # 如果没有发送过 START，先补上
        if tool_call_id not in self._started_tool_calls:
            yield AgentResult(
                event=EventType.TOOL_CALL_START,
                data={
                    "tool_call_id": tool_call_id,
                    "tool_call_name": "",
                },
            )
            self._started_tool_calls.add(tool_call_id)

        # 记录已结束并发送
        self._ended_tool_calls.add(tool_call_id)
        self._active_tool_calls.pop(tool_call_id, None)
        yield event

    def _handle_tool_call_result(
        self, event: AgentResult
    ) -> Iterator[AgentResult]:
        """处理 TOOL_CALL_RESULT 事件

        如果该 tool_call_id 没有发送过 END，先补上
        """
        tool_call_id = event.data.get("tool_call_id", "")

        if not tool_call_id:
            yield event
            return

        # 如果没有发送过 START，先补上
        if tool_call_id not in self._started_tool_calls:
            yield AgentResult(
                event=EventType.TOOL_CALL_START,
                data={
                    "tool_call_id": tool_call_id,
                    "tool_call_name": "",
                },
            )
            self._started_tool_calls.add(tool_call_id)

        # 如果没有发送过 END，先补上
        if tool_call_id not in self._ended_tool_calls:
            yield AgentResult(
                event=EventType.TOOL_CALL_END,
                data={"tool_call_id": tool_call_id},
            )
            self._ended_tool_calls.add(tool_call_id)
            self._active_tool_calls.pop(tool_call_id, None)

        yield event

    def _handle_text_message(self, event: AgentResult) -> Iterator[AgentResult]:
        """处理文本消息事件

        在发送文本消息前，确保所有活跃的工具调用都已结束
        """
        # 结束所有未结束的工具调用
        for tool_call_id in list(self._active_tool_calls.keys()):
            if tool_call_id not in self._ended_tool_calls:
                yield AgentResult(
                    event=EventType.TOOL_CALL_END,
                    data={"tool_call_id": tool_call_id},
                )
                self._ended_tool_calls.add(tool_call_id)
        self._active_tool_calls.clear()

        yield event

    def get_active_tool_calls(self) -> List[str]:
        """获取当前活跃（未结束）的工具调用 ID 列表"""
        return list(self._active_tool_calls.keys())

    def reset(self):
        """重置状态

        在处理新的请求时，建议创建新的实例而不是复用。
        """
        self._started_tool_calls.clear()
        self._ended_tool_calls.clear()
        self._active_tool_calls.clear()
