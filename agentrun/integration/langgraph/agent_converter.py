"""LangGraph/LangChain 事件转换模块 / LangGraph/LangChain Event Converter

提供将 LangGraph/LangChain 流式事件转换为 AG-UI 协议事件的方法。

使用示例:

    # 使用 astream_events（支持 token by token）
    >>> async for event in agent.astream_events(input_data, version="v2"):
    ...     for item in to_agui_events(event):
    ...         yield item

    # 使用 stream (updates 模式)
    >>> for event in agent.stream(input_data, stream_mode="updates"):
    ...     for item in to_agui_events(event):
    ...         yield item

    # 使用 astream (updates 模式)
    >>> async for event in agent.astream(input_data, stream_mode="updates"):
    ...     for item in to_agui_events(event):
    ...         yield item
"""

import json
from typing import Any, Dict, Iterator, List, Optional, Union

from agentrun.server.model import AgentResult, EventType

# =============================================================================
# 内部工具函数
# =============================================================================


def _format_tool_output(output: Any) -> str:
    """格式化工具输出为字符串，优先提取常见字段或 content 属性，最后回退到 JSON/str。"""
    if output is None:
        return ""
    # dict-like
    if isinstance(output, dict):
        for key in ("content", "result", "output"):
            if key in output:
                v = output[key]
                if isinstance(v, (dict, list)):
                    return json.dumps(v, ensure_ascii=False)
                return str(v) if v is not None else ""
        try:
            return json.dumps(output, ensure_ascii=False)
        except Exception:
            return str(output)

    # 对象有 content 属性
    if hasattr(output, "content"):
        c = _get_message_content(output)
        if isinstance(c, (dict, list)):
            try:
                return json.dumps(c, ensure_ascii=False)
            except Exception:
                return str(c)
        return c or ""

    try:
        return str(output)
    except Exception:
        return ""


def _safe_json_dumps(obj: Any) -> str:
    """JSON 序列化兜底，无法序列化则回退到 str。"""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return ""


# 需要从工具输入中过滤掉的内部字段（LangGraph/MCP 注入的运行时对象）
_TOOL_INPUT_INTERNAL_KEYS = frozenset({
    "runtime",  # MCP ToolRuntime 对象
    "__pregel_runtime",
    "__pregel_task_id",
    "__pregel_send",
    "__pregel_read",
    "__pregel_checkpointer",
    "__pregel_scratchpad",
    "__pregel_call",
    "config",  # LangGraph config 对象，包含内部状态
    "configurable",
})


def _filter_tool_input(tool_input: Any) -> Any:
    """过滤工具输入中的内部字段，只保留用户传入的实际参数。

    Args:
        tool_input: 工具输入（可能是 dict 或其他类型）

    Returns:
        过滤后的工具输入
    """
    if not isinstance(tool_input, dict):
        return tool_input

    filtered = {}
    for key, value in tool_input.items():
        # 跳过内部字段
        if key in _TOOL_INPUT_INTERNAL_KEYS:
            continue
        # 跳过以 __ 开头的字段（Python 内部属性）
        if key.startswith("__"):
            continue
        filtered[key] = value

    return filtered


def _extract_content(chunk: Any) -> Optional[str]:
    """从 chunk 中提取文本内容"""
    if chunk is None:
        return None

    if hasattr(chunk, "content"):
        content = chunk.content
        if isinstance(content, str):
            return content if content else None
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "".join(text_parts) if text_parts else None

    return None


def _extract_tool_call_chunks(chunk: Any) -> List[Dict]:
    """从 AIMessageChunk 中提取工具调用增量"""
    tool_calls = []

    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
        for tc in chunk.tool_call_chunks:
            if isinstance(tc, dict):
                tool_calls.append(tc)
            else:
                tool_calls.append({
                    "id": getattr(tc, "id", None),
                    "name": getattr(tc, "name", None),
                    "args": getattr(tc, "args", None),
                    "index": getattr(tc, "index", None),
                })

    return tool_calls


def _get_message_type(msg: Any) -> str:
    """获取消息类型"""
    if hasattr(msg, "type"):
        return str(msg.type).lower()

    if isinstance(msg, dict):
        msg_type = msg.get("type", msg.get("role", ""))
        return str(msg_type).lower()

    class_name = type(msg).__name__.lower()
    if "ai" in class_name or "assistant" in class_name:
        return "ai"
    if "tool" in class_name:
        return "tool"
    if "human" in class_name or "user" in class_name:
        return "human"

    return "unknown"


def _get_message_content(msg: Any) -> Optional[str]:
    """获取消息内容"""
    if hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, str):
            return content
        return str(content) if content else None

    if isinstance(msg, dict):
        return msg.get("content")

    return None


def _get_message_tool_calls(msg: Any) -> List[Dict]:
    """获取消息中的工具调用"""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_calls = []
        for tc in msg.tool_calls:
            if isinstance(tc, dict):
                tool_calls.append(tc)
            else:
                tool_calls.append({
                    "id": getattr(tc, "id", None),
                    "name": getattr(tc, "name", None),
                    "args": getattr(tc, "args", None),
                })
        return tool_calls

    if isinstance(msg, dict) and msg.get("tool_calls"):
        return msg["tool_calls"]

    return []


def _get_tool_call_id(msg: Any) -> Optional[str]:
    """获取 ToolMessage 的 tool_call_id"""
    if hasattr(msg, "tool_call_id"):
        return msg.tool_call_id

    if isinstance(msg, dict):
        return msg.get("tool_call_id")

    return None


# =============================================================================
# 事件格式检测
# =============================================================================


def _event_to_dict(event: Any) -> Dict[str, Any]:
    """将 StreamEvent 或 dict 标准化为 dict 以便后续处理"""
    if isinstance(event, dict):
        return event

    result: Dict[str, Any] = {}
    # 常见属性映射，兼容多种 StreamEvent 实现
    if hasattr(event, "event"):
        result["event"] = getattr(event, "event")
    if hasattr(event, "data"):
        result["data"] = getattr(event, "data")
    if hasattr(event, "name"):
        result["name"] = getattr(event, "name")
    if hasattr(event, "run_id"):
        result["run_id"] = getattr(event, "run_id")

    return result


def _is_astream_events_format(event_dict: Dict[str, Any]) -> bool:
    """检测是否是 astream_events 格式的事件

    astream_events 格式特征：有 "event" 字段，值以 "on_" 开头
    """
    event_type = event_dict.get("event", "")
    return isinstance(event_type, str) and event_type.startswith("on_")


def _is_stream_updates_format(event_dict: Dict[str, Any]) -> bool:
    """检测是否是 stream/astream(stream_mode="updates") 格式的事件

    updates 格式特征：{node_name: {messages_key: [...]}} 或 {node_name: state_dict}
    没有 "event" 字段，键是 node 名称（如 "model", "agent", "tools"），值是 state 更新

    与 values 格式的区别：
    - updates: {node_name: {messages: [...]}} - 嵌套结构
    - values: {messages: [...]} - 扁平结构
    """
    if "event" in event_dict:
        return False

    # 如果直接包含 "messages" 键且值是 list，这是 values 格式，不是 updates
    if "messages" in event_dict and isinstance(event_dict["messages"], list):
        return False

    # 检查是否有类似 node 更新的结构
    for key, value in event_dict.items():
        if key == "__end__":
            continue
        # value 应该是一个 dict（state 更新），包含 messages 等字段
        if isinstance(value, dict):
            return True

    return False


def _is_stream_values_format(event_dict: Dict[str, Any]) -> bool:
    """检测是否是 stream/astream(stream_mode="values") 格式的事件

    values 格式特征：直接是完整 state，如 {messages: [...], ...}
    没有 "event" 字段，直接包含 "messages" 或类似的 state 字段

    与 updates 格式的区别：
    - values: {messages: [...]} - 扁平结构，messages 值直接是 list
    - updates: {node_name: {messages: [...]}} - 嵌套结构
    """
    if "event" in event_dict:
        return False

    # 检查是否直接包含 messages 列表（扁平结构）
    if "messages" in event_dict and isinstance(event_dict["messages"], list):
        return True

    return False


# =============================================================================
# 事件转换器
# =============================================================================


def _convert_stream_updates_event(
    event_dict: Dict[str, Any],
    messages_key: str = "messages",
) -> Iterator[Union[AgentResult, str]]:
    """转换 stream/astream(stream_mode="updates") 格式的单个事件

    Args:
        event_dict: 事件字典，格式为 {node_name: state_update}
        messages_key: state 中消息列表的 key

    Yields:
        str (文本内容) 或 AgentResult (事件)

    Note:
        在 updates 模式下，工具调用和结果在不同的事件中：
        - AI 消息包含 tool_calls（仅发送 TOOL_CALL_START + TOOL_CALL_ARGS）
        - Tool 消息包含结果（发送 TOOL_CALL_RESULT + TOOL_CALL_END）
    """
    for node_name, state_update in event_dict.items():
        if node_name == "__end__":
            continue

        if not isinstance(state_update, dict):
            continue

        messages = state_update.get(messages_key, [])
        if not isinstance(messages, list):
            # 尝试其他常见的 key
            for alt_key in ("message", "output", "response"):
                if alt_key in state_update:
                    alt_value = state_update[alt_key]
                    if isinstance(alt_value, list):
                        messages = alt_value
                        break
                    elif hasattr(alt_value, "content"):
                        messages = [alt_value]
                        break

        for msg in messages:
            msg_type = _get_message_type(msg)

            if msg_type == "ai":
                # 文本内容
                content = _get_message_content(msg)
                if content:
                    yield content

                # 工具调用（仅发送 START 和 ARGS，END 在收到结果后发送）
                for tc in _get_message_tool_calls(msg):
                    tc_id = tc.get("id", "")
                    tc_name = tc.get("name", "")
                    tc_args = tc.get("args", {})

                    if tc_id:
                        yield AgentResult(
                            event=EventType.TOOL_CALL_START,
                            data={
                                "tool_call_id": tc_id,
                                "tool_call_name": tc_name,
                            },
                        )
                        if tc_args:
                            args_str = (
                                _safe_json_dumps(tc_args)
                                if isinstance(tc_args, dict)
                                else str(tc_args)
                            )
                            yield AgentResult(
                                event=EventType.TOOL_CALL_ARGS,
                                data={"tool_call_id": tc_id, "delta": args_str},
                            )

            elif msg_type == "tool":
                # 工具结果（发送 RESULT 和 END）
                tool_call_id = _get_tool_call_id(msg)
                if tool_call_id:
                    tool_content = _get_message_content(msg)
                    yield AgentResult(
                        event=EventType.TOOL_CALL_RESULT,
                        data={
                            "tool_call_id": tool_call_id,
                            "result": str(tool_content) if tool_content else "",
                        },
                    )
                    yield AgentResult(
                        event=EventType.TOOL_CALL_END,
                        data={"tool_call_id": tool_call_id},
                    )


def _convert_stream_values_event(
    event_dict: Dict[str, Any],
    messages_key: str = "messages",
) -> Iterator[Union[AgentResult, str]]:
    """转换 stream/astream(stream_mode="values") 格式的单个事件

    Args:
        event_dict: 事件字典，格式为完整的 state dict
        messages_key: state 中消息列表的 key

    Yields:
        str (文本内容) 或 AgentResult (事件)

    Note:
        在 values 模式下，工具调用和结果可能在同一事件中或不同事件中。
        我们只处理最后一条消息。
    """
    messages = event_dict.get(messages_key, [])
    if not isinstance(messages, list):
        return

    # 对于 values 模式，我们只关心最后一条消息（通常是最新的）
    if not messages:
        return

    last_msg = messages[-1]
    msg_type = _get_message_type(last_msg)

    if msg_type == "ai":
        content = _get_message_content(last_msg)
        if content:
            yield content

        # 工具调用（仅发送 START 和 ARGS）
        for tc in _get_message_tool_calls(last_msg):
            tc_id = tc.get("id", "")
            tc_name = tc.get("name", "")
            tc_args = tc.get("args", {})

            if tc_id:
                yield AgentResult(
                    event=EventType.TOOL_CALL_START,
                    data={
                        "tool_call_id": tc_id,
                        "tool_call_name": tc_name,
                    },
                )
                if tc_args:
                    args_str = (
                        _safe_json_dumps(tc_args)
                        if isinstance(tc_args, dict)
                        else str(tc_args)
                    )
                    yield AgentResult(
                        event=EventType.TOOL_CALL_ARGS,
                        data={"tool_call_id": tc_id, "delta": args_str},
                    )

    elif msg_type == "tool":
        tool_call_id = _get_tool_call_id(last_msg)
        if tool_call_id:
            tool_content = _get_message_content(last_msg)
            yield AgentResult(
                event=EventType.TOOL_CALL_RESULT,
                data={
                    "tool_call_id": tool_call_id,
                    "result": str(tool_content) if tool_content else "",
                },
            )
            yield AgentResult(
                event=EventType.TOOL_CALL_END,
                data={"tool_call_id": tool_call_id},
            )


def _convert_astream_events_event(
    event_dict: Dict[str, Any],
) -> Iterator[Union[AgentResult, str]]:
    """转换 astream_events 格式的单个事件

    Args:
        event_dict: 事件字典，格式为 {"event": "on_xxx", "data": {...}}

    Yields:
        str (文本内容) 或 AgentResult (事件)
    """
    event_type = event_dict.get("event", "")
    data = event_dict.get("data", {})

    # 1. LangGraph 格式: on_chat_model_stream
    if event_type == "on_chat_model_stream":
        chunk = data.get("chunk")
        if chunk:
            # 文本内容
            content = _extract_content(chunk)
            if content:
                yield content

            # 流式工具调用参数
            for tc in _extract_tool_call_chunks(chunk):
                tc_id = tc.get("id") or str(tc.get("index", ""))
                tc_args = tc.get("args", "")

                if tc_args and tc_id:
                    if isinstance(tc_args, (dict, list)):
                        tc_args = _safe_json_dumps(tc_args)
                    yield AgentResult(
                        event=EventType.TOOL_CALL_ARGS,
                        data={"tool_call_id": tc_id, "delta": tc_args},
                    )

    # 2. LangChain 格式: on_chain_stream
    elif event_type == "on_chain_stream" and event_dict.get("name") == "model":
        chunk_data = data.get("chunk", {})
        if isinstance(chunk_data, dict):
            messages = chunk_data.get("messages", [])

            for msg in messages:
                content = _get_message_content(msg)
                if content:
                    yield content

                for tc in _get_message_tool_calls(msg):
                    tc_id = tc.get("id", "")
                    tc_args = tc.get("args", {})

                    if tc_id and tc_args:
                        args_str = (
                            _safe_json_dumps(tc_args)
                            if isinstance(tc_args, dict)
                            else str(tc_args)
                        )
                        yield AgentResult(
                            event=EventType.TOOL_CALL_ARGS,
                            data={"tool_call_id": tc_id, "delta": args_str},
                        )

    # 3. 工具开始
    elif event_type == "on_tool_start":
        run_id = event_dict.get("run_id", "")
        tool_name = event_dict.get("name", "")
        tool_input_raw = data.get("input", {})
        # 过滤掉内部字段（如 MCP 注入的 runtime）
        tool_input = _filter_tool_input(tool_input_raw)

        if run_id:
            yield AgentResult(
                event=EventType.TOOL_CALL_START,
                data={"tool_call_id": run_id, "tool_call_name": tool_name},
            )
            if tool_input:
                args_str = (
                    _safe_json_dumps(tool_input)
                    if isinstance(tool_input, dict)
                    else str(tool_input)
                )
                yield AgentResult(
                    event=EventType.TOOL_CALL_ARGS,
                    data={"tool_call_id": run_id, "delta": args_str},
                )

    # 4. 工具结束
    elif event_type == "on_tool_end":
        run_id = event_dict.get("run_id", "")
        output = data.get("output", "")

        if run_id:
            yield AgentResult(
                event=EventType.TOOL_CALL_RESULT,
                data={
                    "tool_call_id": run_id,
                    "result": _format_tool_output(output),
                },
            )
            yield AgentResult(
                event=EventType.TOOL_CALL_END,
                data={"tool_call_id": run_id},
            )

    # 5. LLM 结束
    elif event_type == "on_chat_model_end":
        # 无状态模式下不处理，避免重复
        pass


# =============================================================================
# 主要 API
# =============================================================================


def to_agui_events(
    event: Union[Dict[str, Any], Any],
    messages_key: str = "messages",
) -> Iterator[Union[AgentResult, str]]:
    """将 LangGraph/LangChain 流式事件转换为 AG-UI 协议事件

    支持多种调用方式产生的事件格式：
    - agent.astream_events(input, version="v2")
    - agent.stream(input, stream_mode="updates")
    - agent.astream(input, stream_mode="updates")
    - agent.stream(input, stream_mode="values")
    - agent.astream(input, stream_mode="values")

    Args:
        event: LangGraph/LangChain 流式事件（StreamEvent 对象或 Dict）
        messages_key: state 中消息列表的 key，默认 "messages"

    Yields:
        str (文本内容) 或 AgentResult (AG-UI 事件)

    Example:
        >>> # 使用 astream_events
        >>> async for event in agent.astream_events(input, version="v2"):
        ...     for item in to_agui_events(event):
        ...         yield item

        >>> # 使用 stream (updates 模式)
        >>> for event in agent.stream(input, stream_mode="updates"):
        ...     for item in to_agui_events(event):
        ...         yield item

        >>> # 使用 astream (updates 模式)
        >>> async for event in agent.astream(input, stream_mode="updates"):
        ...     for item in to_agui_events(event):
        ...         yield item
    """
    event_dict = _event_to_dict(event)

    # 根据事件格式选择对应的转换器
    if _is_astream_events_format(event_dict):
        # astream_events 格式：{"event": "on_xxx", "data": {...}}
        yield from _convert_astream_events_event(event_dict)

    elif _is_stream_updates_format(event_dict):
        # stream/astream(stream_mode="updates") 格式：{node_name: state_update}
        yield from _convert_stream_updates_event(event_dict, messages_key)

    elif _is_stream_values_format(event_dict):
        # stream/astream(stream_mode="values") 格式：完整 state dict
        yield from _convert_stream_values_event(event_dict, messages_key)


# 保留 convert 作为别名，兼容旧代码
convert = to_agui_events
