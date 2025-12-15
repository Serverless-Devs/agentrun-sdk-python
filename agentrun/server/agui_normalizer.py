"""AG-UI 事件规范化器

提供事件流规范化功能，确保事件符合 AG-UI 协议的顺序要求。

主要功能：
- 追踪工具调用状态
- 在 TOOL_RESULT 前确保工具调用已开始
- 自动补充缺失的状态

注意：边界事件（如 TEXT_MESSAGE_START/END、TOOL_CALL_START/END）
由协议层（agui_protocol.py）自动生成，不需要用户关心。

使用示例:

    >>> from agentrun.server.agui_normalizer import AguiEventNormalizer
    >>>
    >>> normalizer = AguiEventNormalizer()
    >>> for event in raw_events:
    ...     for normalized_event in normalizer.normalize(event):
    ...         yield normalized_event
"""

from typing import Any, Dict, Iterator, List, Optional, Set, Union

from .model import AgentEvent, EventType


class AguiEventNormalizer:
    """AG-UI 事件规范化器

    追踪工具调用状态，确保事件顺序正确：
    1. 追踪已开始的工具调用
    2. 确保 TOOL_RESULT 前工具调用存在

    协议层会自动处理边界事件（START/END），这个类主要用于
    高级用户需要手动控制事件流时。

    Example:
        >>> normalizer = AguiEventNormalizer()
        >>> for event in agent_events:
        ...     for normalized in normalizer.normalize(event):
        ...         yield normalized
    """

    def __init__(self):
        # 已看到的工具调用 ID 集合
        self._seen_tool_calls: Set[str] = set()
        # 活跃的工具调用信息（tool_call_id -> tool_call_name）
        self._active_tool_calls: Dict[str, str] = {}

    def normalize(
        self,
        event: Union[AgentEvent, str, Dict[str, Any]],
    ) -> Iterator[AgentEvent]:
        """规范化单个事件

        将事件标准化为 AgentEvent，并追踪工具调用状态。

        Args:
            event: 原始事件（AgentEvent、str 或 dict）

        Yields:
            规范化后的事件
        """
        # 将事件标准化为 AgentEvent
        normalized_event = self._to_agent_event(event)
        if normalized_event is None:
            return

        # 根据事件类型进行处理
        event_type = normalized_event.event

        if event_type == EventType.TOOL_CALL_CHUNK:
            yield from self._handle_tool_call_chunk(normalized_event)

        elif event_type == EventType.TOOL_CALL:
            yield from self._handle_tool_call(normalized_event)

        elif event_type == EventType.TOOL_RESULT:
            yield from self._handle_tool_result(normalized_event)

        else:
            # 其他事件类型直接传递
            yield normalized_event

    def _to_agent_event(
        self, event: Union[AgentEvent, str, Dict[str, Any]]
    ) -> Optional[AgentEvent]:
        """将事件转换为 AgentEvent"""
        if isinstance(event, AgentEvent):
            return event

        if isinstance(event, str):
            # 字符串转为 TEXT
            return AgentEvent(
                event=EventType.TEXT,
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

            return AgentEvent(
                event=event_type,
                data=event.get("data", {}),
            )

        return None

    def _handle_tool_call(self, event: AgentEvent) -> Iterator[AgentEvent]:
        """处理 TOOL_CALL 事件

        记录工具调用并直接传递
        """
        tool_call_id = event.data.get("id", "")
        tool_call_name = event.data.get("name", "")

        if tool_call_id:
            self._seen_tool_calls.add(tool_call_id)
            self._active_tool_calls[tool_call_id] = tool_call_name

        yield event

    def _handle_tool_call_chunk(
        self, event: AgentEvent
    ) -> Iterator[AgentEvent]:
        """处理 TOOL_CALL_CHUNK 事件

        记录工具调用并直接传递
        """
        tool_call_id = event.data.get("id", "")
        tool_call_name = event.data.get("name", "")

        if tool_call_id:
            self._seen_tool_calls.add(tool_call_id)
            if tool_call_name:
                self._active_tool_calls[tool_call_id] = tool_call_name

        yield event

    def _handle_tool_result(self, event: AgentEvent) -> Iterator[AgentEvent]:
        """处理 TOOL_RESULT 事件

        标记工具调用完成
        """
        tool_call_id = event.data.get("id", "")

        if tool_call_id:
            # 标记工具调用已完成（从活跃列表移除）
            self._active_tool_calls.pop(tool_call_id, None)

        yield event

    def get_active_tool_calls(self) -> List[str]:
        """获取当前活跃（未结束）的工具调用 ID 列表"""
        return list(self._active_tool_calls.keys())

    def get_seen_tool_calls(self) -> List[str]:
        """获取所有已见过的工具调用 ID 列表"""
        return list(self._seen_tool_calls)

    def reset(self):
        """重置状态

        在处理新的请求时，建议创建新的实例而不是复用。
        """
        self._seen_tool_calls.clear()
        self._active_tool_calls.clear()
