"""LangGraph/LangChain 事件转换模块 / LangGraph/LangChain Event Converter

提供将 LangGraph/LangChain 流式事件转换为 AG-UI 协议事件的方法。

使用示例:

    # 使用 AgentRunConverter 类（推荐）
    >>> converter = AgentRunConverter()
    >>> async for event in agent.astream_events(input_data, version="v2"):
    ...     for item in converter.convert(event):
    ...         yield item

    # 使用静态方法（无状态）
    >>> async for event in agent.astream_events(input_data, version="v2"):
    ...     for item in AgentRunConverter.to_agui_events(event):
    ...         yield item

    # 使用 stream (updates 模式)
    >>> for event in agent.stream(input_data, stream_mode="updates"):
    ...     for item in AgentRunConverter.to_agui_events(event):
    ...         yield item

    # 使用 astream (updates 模式)
    >>> async for event in agent.astream(input_data, stream_mode="updates"):
    ...     for item in AgentRunConverter.to_agui_events(event):
    ...         yield item
"""

import json
from typing import Any, Dict, Iterator, List, Optional, Union

from agentrun.server.model import AgentResult, EventType
from agentrun.utils.log import logger

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


class AgentRunConverter:
    """AgentRun 事件转换器

    将 LangGraph/LangChain 流式事件转换为 AG-UI 协议事件。
    此类维护必要的状态以确保：
    1. 流式工具调用的 tool_call_id 一致性
    2. AG-UI 协议要求的事件顺序（TOOL_CALL_START → TOOL_CALL_ARGS → TOOL_CALL_END）

    在流式工具调用中，第一个 chunk 包含 id 和 name，后续 chunk 只有 index 和 args。
    此类维护 index -> id 的映射，确保所有相关事件使用相同的 tool_call_id。

    同时，此类跟踪已发送 TOOL_CALL_START 的工具调用，确保：
    - 在流式场景中，TOOL_CALL_START 在第一个参数 chunk 前发送
    - 避免在 on_tool_start 中重复发送 TOOL_CALL_START

    Example:
        >>> from agentrun.integration.langgraph import AgentRunConverter
        >>>
        >>> async def invoke_agent(request: AgentRequest):
        ...     converter = AgentRunConverter()
        ...     async for event in agent.astream_events(input, version="v2"):
        ...         for item in converter.convert(event):
        ...             yield item
    """

    def __init__(self) -> None:
        self._tool_call_id_map: Dict[int, str] = {}
        self._tool_call_started_set: set = set()
        # tool_name -> [tool_call_id] 队列映射
        # 用于在 on_tool_start 中查找对应的 tool_call_id（当 runtime.tool_call_id 不可用时）
        self._tool_name_to_call_ids: Dict[str, List[str]] = {}
        # run_id -> tool_call_id 映射
        # 用于在 on_tool_end 中查找对应的 tool_call_id
        self._run_id_to_tool_call_id: Dict[str, str] = {}

    def convert(
        self,
        event: Union[Dict[str, Any], Any],
        messages_key: str = "messages",
    ) -> Iterator[Union[AgentResult, str]]:
        """转换单个事件为 AG-UI 协议事件

        Args:
            event: LangGraph/LangChain 流式事件（StreamEvent 对象或 Dict）
            messages_key: state 中消息列表的 key，默认 "messages"

        Yields:
            str (文本内容) 或 AgentResult (AG-UI 事件)
        """
        # 调试日志：输入事件
        event_dict = self._event_to_dict(event)
        event_type = event_dict.get("event", "")

        # 始终打印事件类型，用于调试
        logger.debug(
            f"[AgentRunConverter] Raw event type: {type(event).__name__}, "
            f"event_type={event_type}, "
            f"is_dict={isinstance(event, dict)}"
        )

        if event_type in (
            "on_chat_model_stream",
            "on_tool_start",
            "on_tool_end",
        ):
            logger.debug(
                f"[AgentRunConverter] Input event: type={event_type}, "
                f"run_id={event_dict.get('run_id', '')}, "
                f"name={event_dict.get('name', '')}, "
                f"tool_call_started_set={self._tool_call_started_set}, "
                f"tool_name_to_call_ids={self._tool_name_to_call_ids}"
            )

        for item in self.to_agui_events(
            event,
            messages_key,
            self._tool_call_id_map,
            self._tool_call_started_set,
            self._tool_name_to_call_ids,
            self._run_id_to_tool_call_id,
        ):
            # 调试日志：输出事件
            if isinstance(item, AgentResult):
                logger.debug(f"[AgentRunConverter] Output event: {item}")
            yield item

    def reset(self) -> None:
        """重置状态，清空 tool_call_id 映射和已发送状态

        在处理新的请求时，建议创建新的 AgentRunConverter 实例，
        而不是复用旧实例并调用 reset。
        """
        self._tool_call_id_map.clear()
        self._tool_call_started_set.clear()
        self._tool_name_to_call_ids.clear()
        self._run_id_to_tool_call_id.clear()

    # =========================================================================
    # 内部工具方法（静态方法）
    # =========================================================================

    @staticmethod
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
            c = AgentRunConverter._get_message_content(output)
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

    @staticmethod
    def _safe_json_dumps(obj: Any) -> str:
        """JSON 序列化兜底，无法序列化则回退到 str。"""
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            try:
                return str(obj)
            except Exception:
                return ""

    @staticmethod
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
            # 跳过所有下划线前缀的内部字段（包含单下划线与双下划线）
            if key.startswith("_"):
                continue
            filtered[key] = value

        return filtered

    @staticmethod
    def _extract_tool_call_id(tool_input: Any) -> Optional[str]:
        """从工具输入中提取原始的 tool_call_id。

        MCP 工具会在 input 中注入 runtime 对象，其中包含 LLM 返回的原始 tool_call_id。
        使用这个 ID 可以保证工具调用事件的 ID 一致性。

        Args:
            tool_input: 工具输入（可能是 dict 或其他类型）

        Returns:
            tool_call_id 或 None
        """
        if not isinstance(tool_input, dict):
            return None

        # 尝试从 runtime 对象中提取 tool_call_id
        runtime = tool_input.get("runtime")
        if runtime is not None and hasattr(runtime, "tool_call_id"):
            tc_id = runtime.tool_call_id
            if isinstance(tc_id, str) and tc_id:
                return tc_id

        return None

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _get_tool_call_id(msg: Any) -> Optional[str]:
        """获取 ToolMessage 的 tool_call_id"""
        if hasattr(msg, "tool_call_id"):
            return msg.tool_call_id

        if isinstance(msg, dict):
            return msg.get("tool_call_id")

        return None

    # =========================================================================
    # 事件格式检测（静态方法）
    # =========================================================================

    @staticmethod
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

    @staticmethod
    def is_astream_events_format(event_dict: Dict[str, Any]) -> bool:
        """检测是否是 astream_events 格式的事件

        astream_events 格式特征：有 "event" 字段，值以 "on_" 开头
        """
        event_type = event_dict.get("event", "")
        return isinstance(event_type, str) and event_type.startswith("on_")

    @staticmethod
    def is_stream_updates_format(event_dict: Dict[str, Any]) -> bool:
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
        if "messages" in event_dict and isinstance(
            event_dict["messages"], list
        ):
            return False

        # 检查是否有类似 node 更新的结构
        for key, value in event_dict.items():
            if key == "__end__":
                continue
            # value 应该是一个 dict（state 更新），包含 messages 等字段
            if isinstance(value, dict):
                return True

        return False

    @staticmethod
    def is_stream_values_format(event_dict: Dict[str, Any]) -> bool:
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
        if "messages" in event_dict and isinstance(
            event_dict["messages"], list
        ):
            return True

        return False

    # =========================================================================
    # 事件转换器（静态方法）
    # =========================================================================

    @staticmethod
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
                msg_type = AgentRunConverter._get_message_type(msg)

                if msg_type == "ai":
                    # 文本内容
                    content = AgentRunConverter._get_message_content(msg)
                    if content:
                        yield content

                    # 工具调用（仅发送 START 和 ARGS，END 在收到结果后发送）
                    for tc in AgentRunConverter._get_message_tool_calls(msg):
                        tc_id = tc.get("id", "")
                        tc_name = tc.get("name", "")
                        tc_args = tc.get("args", {})

                        if tc_id:
                            # 发送带有完整参数的 TOOL_CALL_CHUNK
                            args_str = ""
                            if tc_args:
                                args_str = (
                                    AgentRunConverter._safe_json_dumps(tc_args)
                                    if isinstance(tc_args, dict)
                                    else str(tc_args)
                                )
                            yield AgentResult(
                                event=EventType.TOOL_CALL_CHUNK,
                                data={
                                    "id": tc_id,
                                    "name": tc_name,
                                    "args_delta": args_str,
                                },
                            )

                elif msg_type == "tool":
                    # 工具结果
                    tool_call_id = AgentRunConverter._get_tool_call_id(msg)
                    if tool_call_id:
                        tool_content = AgentRunConverter._get_message_content(
                            msg
                        )
                        yield AgentResult(
                            event=EventType.TOOL_RESULT,
                            data={
                                "id": tool_call_id,
                                "result": (
                                    str(tool_content) if tool_content else ""
                                ),
                            },
                        )

    @staticmethod
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
        msg_type = AgentRunConverter._get_message_type(last_msg)

        if msg_type == "ai":
            content = AgentRunConverter._get_message_content(last_msg)
            if content:
                yield content

            # 工具调用
            for tc in AgentRunConverter._get_message_tool_calls(last_msg):
                tc_id = tc.get("id", "")
                tc_name = tc.get("name", "")
                tc_args = tc.get("args", {})

                if tc_id:
                    # 发送带有完整参数的 TOOL_CALL_CHUNK
                    args_str = ""
                    if tc_args:
                        args_str = (
                            AgentRunConverter._safe_json_dumps(tc_args)
                            if isinstance(tc_args, dict)
                            else str(tc_args)
                        )
                    yield AgentResult(
                        event=EventType.TOOL_CALL_CHUNK,
                        data={
                            "id": tc_id,
                            "name": tc_name,
                            "args_delta": args_str,
                        },
                    )

        elif msg_type == "tool":
            tool_call_id = AgentRunConverter._get_tool_call_id(last_msg)
            if tool_call_id:
                tool_content = AgentRunConverter._get_message_content(last_msg)
                yield AgentResult(
                    event=EventType.TOOL_RESULT,
                    data={
                        "id": tool_call_id,
                        "result": str(tool_content) if tool_content else "",
                    },
                )

    @staticmethod
    def _convert_astream_events_event(
        event_dict: Dict[str, Any],
        tool_call_id_map: Optional[Dict[int, str]] = None,
        tool_call_started_set: Optional[set] = None,
        tool_name_to_call_ids: Optional[Dict[str, List[str]]] = None,
        run_id_to_tool_call_id: Optional[Dict[str, str]] = None,
    ) -> Iterator[Union[AgentResult, str]]:
        """转换 astream_events 格式的单个事件

        Args:
            event_dict: 事件字典，格式为 {"event": "on_xxx", "data": {...}}
            tool_call_id_map: 可选的 index -> tool_call_id 映射字典。
                              在流式工具调用中，第一个 chunk 有 id，后续只有 index。
                              此映射用于确保所有 chunk 使用一致的 tool_call_id。
            tool_call_started_set: 可选的已发送 TOOL_CALL_START 的 tool_call_id 集合。
                                   用于确保每个工具调用只发送一次 TOOL_CALL_START。
            tool_name_to_call_ids: 可选的 tool_name -> [tool_call_id] 队列映射。
                                   用于在 on_tool_start 中查找对应的 tool_call_id。
            run_id_to_tool_call_id: 可选的 run_id -> tool_call_id 映射。
                                    用于在 on_tool_end 中查找对应的 tool_call_id。

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
                content = AgentRunConverter._extract_content(chunk)
                if content:
                    yield content

                # 流式工具调用参数
                for tc in AgentRunConverter._extract_tool_call_chunks(chunk):
                    tc_index = tc.get("index")
                    tc_raw_id = tc.get("id")
                    tc_name = tc.get("name", "")
                    tc_args = tc.get("args", "")

                    # 解析 tool_call_id：
                    # 1. 如果有 id 且非空，使用它并更新映射
                    # 2. 如果 id 为空但有 index，从映射中查找
                    # 3. 最后回退到使用 index 字符串
                    if tc_raw_id:
                        tc_id = tc_raw_id
                        # 更新映射（如果提供了映射字典）
                        # 重要：即使这个 chunk 没有 args，也要更新映射，
                        # 因为后续 chunk 可能只有 index 没有 id
                        if (
                            tool_call_id_map is not None
                            and tc_index is not None
                        ):
                            tool_call_id_map[tc_index] = tc_id
                    elif tc_index is not None:
                        # 从映射中查找，如果没有则使用 index
                        if (
                            tool_call_id_map is not None
                            and tc_index in tool_call_id_map
                        ):
                            tc_id = tool_call_id_map[tc_index]
                        else:
                            tc_id = str(tc_index)
                    else:
                        tc_id = ""

                    if not tc_id:
                        continue

                    # 流式工具调用：第一个 chunk 包含 id 和 name，后续只有 args_delta
                    # 协议层会自动处理 START/END 边界事件
                    is_first_chunk = (
                        tc_raw_id
                        and tc_name
                        and (
                            tool_call_started_set is None
                            or tc_id not in tool_call_started_set
                        )
                    )

                    if is_first_chunk:
                        if tool_call_started_set is not None:
                            tool_call_started_set.add(tc_id)
                        # 记录 tool_name -> tool_call_id 映射，用于 on_tool_start 查找
                        if tool_name_to_call_ids is not None and tc_name:
                            if tc_name not in tool_name_to_call_ids:
                                tool_name_to_call_ids[tc_name] = []
                            tool_name_to_call_ids[tc_name].append(tc_id)
                        # 第一个 chunk 包含 id 和 name
                        args_delta = ""
                        if tc_args:
                            args_delta = (
                                AgentRunConverter._safe_json_dumps(tc_args)
                                if isinstance(tc_args, (dict, list))
                                else str(tc_args)
                            )
                        yield AgentResult(
                            event=EventType.TOOL_CALL_CHUNK,
                            data={
                                "id": tc_id,
                                "name": tc_name,
                                "args_delta": args_delta,
                            },
                        )
                    elif tc_args:
                        # 后续 chunk 只有 args_delta
                        args_delta = (
                            AgentRunConverter._safe_json_dumps(tc_args)
                            if isinstance(tc_args, (dict, list))
                            else str(tc_args)
                        )
                        yield AgentResult(
                            event=EventType.TOOL_CALL_CHUNK,
                            data={
                                "id": tc_id,
                                "args_delta": args_delta,
                            },
                        )

        # 2. LangChain 格式: on_chain_stream
        elif (
            event_type == "on_chain_stream"
            and event_dict.get("name") == "model"
        ):
            chunk_data = data.get("chunk", {})
            if isinstance(chunk_data, dict):
                messages = chunk_data.get("messages", [])

                for msg in messages:
                    content = AgentRunConverter._get_message_content(msg)
                    if content:
                        yield content

                    for tc in AgentRunConverter._get_message_tool_calls(msg):
                        tc_id = tc.get("id", "")
                        tc_name = tc.get("name", "")
                        tc_args = tc.get("args", {})

                        if tc_id:
                            # 检查是否已经发送过这个 tool call
                            already_started = (
                                tool_call_started_set is not None
                                and tc_id in tool_call_started_set
                            )

                            if not already_started:
                                # 标记为已开始，防止 on_tool_start 重复发送
                                if tool_call_started_set is not None:
                                    tool_call_started_set.add(tc_id)

                                # 记录 tool_name -> tool_call_id 映射
                                if (
                                    tool_name_to_call_ids is not None
                                    and tc_name
                                ):
                                    tool_name_to_call_ids.setdefault(
                                        tc_name, []
                                    ).append(tc_id)

                                args_delta = ""
                                if tc_args:
                                    args_delta = (
                                        AgentRunConverter._safe_json_dumps(
                                            tc_args
                                        )
                                        if isinstance(tc_args, dict)
                                        else str(tc_args)
                                    )
                                yield AgentResult(
                                    event=EventType.TOOL_CALL_CHUNK,
                                    data={
                                        "id": tc_id,
                                        "name": tc_name,
                                        "args_delta": args_delta,
                                    },
                                )

        # 3. 工具开始
        elif event_type == "on_tool_start":
            run_id = event_dict.get("run_id", "")
            tool_name = event_dict.get("name", "")
            tool_input_raw = data.get("input", {})
            # 优先使用 runtime 中的原始 tool_call_id，保证 ID 一致性
            tool_call_id = AgentRunConverter._extract_tool_call_id(
                tool_input_raw
            )

            # 如果 runtime.tool_call_id 不可用，尝试从 tool_name_to_call_ids 映射中查找
            # 这用于处理非 MCP 工具的情况，其中 on_chat_model_stream 已经发送了 TOOL_CALL_START
            if (
                not tool_call_id
                and tool_name_to_call_ids is not None
                and tool_name
            ):
                call_ids = tool_name_to_call_ids.get(tool_name, [])
                if call_ids:
                    # 使用队列中的第一个 ID（FIFO），并从队列中移除
                    tool_call_id = call_ids.pop(0)

            # 最后回退到 run_id
            if not tool_call_id:
                tool_call_id = run_id

            # 记录 run_id -> tool_call_id 映射，用于 on_tool_end 查找
            if run_id_to_tool_call_id is not None and run_id and tool_call_id:
                run_id_to_tool_call_id[run_id] = tool_call_id

            # 过滤掉内部字段（如 MCP 注入的 runtime）
            tool_input = AgentRunConverter._filter_tool_input(tool_input_raw)

            if tool_call_id:
                # 检查是否已在 on_chat_model_stream 中发送过
                already_started = (
                    tool_call_started_set is not None
                    and tool_call_id in tool_call_started_set
                )

                if not already_started:
                    # 非流式场景或未收到流式事件，发送完整的 TOOL_CALL_CHUNK
                    if tool_call_started_set is not None:
                        tool_call_started_set.add(tool_call_id)

                    args_delta = ""
                    if tool_input:
                        args_delta = (
                            AgentRunConverter._safe_json_dumps(tool_input)
                            if isinstance(tool_input, dict)
                            else str(tool_input)
                        )
                    yield AgentResult(
                        event=EventType.TOOL_CALL_CHUNK,
                        data={
                            "id": tool_call_id,
                            "name": tool_name,
                            "args_delta": args_delta,
                        },
                    )
                # 协议层会自动处理边界事件，无需手动发送 TOOL_CALL_END

        # 4. 工具结束
        elif event_type == "on_tool_end":
            run_id = event_dict.get("run_id", "")
            output = data.get("output", "")
            tool_input_raw = data.get("input", {})
            # 优先使用 runtime 中的原始 tool_call_id，保证 ID 一致性
            tool_call_id = AgentRunConverter._extract_tool_call_id(
                tool_input_raw
            )

            # 如果 runtime.tool_call_id 不可用，尝试从 run_id_to_tool_call_id 映射中查找
            # 这个映射在 on_tool_start 中建立
            if (
                not tool_call_id
                and run_id_to_tool_call_id is not None
                and run_id
            ):
                tool_call_id = run_id_to_tool_call_id.get(run_id)

            # 最后回退到 run_id
            if not tool_call_id:
                tool_call_id = run_id

            if tool_call_id:
                # 工具执行完成后发送结果
                yield AgentResult(
                    event=EventType.TOOL_RESULT,
                    data={
                        "id": tool_call_id,
                        "result": AgentRunConverter._format_tool_output(output),
                    },
                )

        # 5. LLM 结束
        elif event_type == "on_chat_model_end":
            # 无状态模式下不处理，避免重复
            pass

        # 6. 工具错误
        elif event_type == "on_tool_error":
            run_id = event_dict.get("run_id", "")
            error = data.get("error")
            tool_input_raw = data.get("input", {})
            tool_name = event_dict.get("name", "")
            # 优先使用 runtime 中的原始 tool_call_id
            tool_call_id = AgentRunConverter._extract_tool_call_id(
                tool_input_raw
            )

            # 如果 runtime.tool_call_id 不可用，尝试从 run_id_to_tool_call_id 映射中查找
            if (
                not tool_call_id
                and run_id_to_tool_call_id is not None
                and run_id
            ):
                tool_call_id = run_id_to_tool_call_id.get(run_id)

            # 最后回退到 run_id
            if not tool_call_id:
                tool_call_id = run_id

            # 格式化错误信息
            error_message = ""
            if error is not None:
                if isinstance(error, Exception):
                    error_message = f"{type(error).__name__}: {str(error)}"
                elif isinstance(error, str):
                    error_message = error
                else:
                    error_message = str(error)

            # 发送 ERROR 事件
            yield AgentResult(
                event=EventType.ERROR,
                data={
                    "message": (
                        f"Tool '{tool_name}' error: {error_message}"
                        if tool_name
                        else error_message
                    ),
                    "code": "TOOL_ERROR",
                    "tool_call_id": tool_call_id,
                },
            )

        # 7. LLM 错误
        elif event_type == "on_llm_error":
            error = data.get("error")
            error_message = ""
            if error is not None:
                if isinstance(error, Exception):
                    error_message = f"{type(error).__name__}: {str(error)}"
                elif isinstance(error, str):
                    error_message = error
                else:
                    error_message = str(error)

            yield AgentResult(
                event=EventType.ERROR,
                data={
                    "message": f"LLM error: {error_message}",
                    "code": "LLM_ERROR",
                },
            )

        # 8. Chain 错误
        elif event_type == "on_chain_error":
            error = data.get("error")
            chain_name = event_dict.get("name", "")
            error_message = ""
            if error is not None:
                if isinstance(error, Exception):
                    error_message = f"{type(error).__name__}: {str(error)}"
                elif isinstance(error, str):
                    error_message = error
                else:
                    error_message = str(error)

            yield AgentResult(
                event=EventType.ERROR,
                data={
                    "message": (
                        f"Chain '{chain_name}' error: {error_message}"
                        if chain_name
                        else error_message
                    ),
                    "code": "CHAIN_ERROR",
                },
            )

        # 9. Retriever 错误
        elif event_type == "on_retriever_error":
            error = data.get("error")
            retriever_name = event_dict.get("name", "")
            error_message = ""
            if error is not None:
                if isinstance(error, Exception):
                    error_message = f"{type(error).__name__}: {str(error)}"
                elif isinstance(error, str):
                    error_message = error
                else:
                    error_message = str(error)

            yield AgentResult(
                event=EventType.ERROR,
                data={
                    "message": (
                        f"Retriever '{retriever_name}' error: {error_message}"
                        if retriever_name
                        else error_message
                    ),
                    "code": "RETRIEVER_ERROR",
                },
            )

    # =========================================================================
    # 主要 API（静态方法）
    # =========================================================================

    @staticmethod
    def to_agui_events(
        event: Union[Dict[str, Any], Any],
        messages_key: str = "messages",
        tool_call_id_map: Optional[Dict[int, str]] = None,
        tool_call_started_set: Optional[set] = None,
        tool_name_to_call_ids: Optional[Dict[str, List[str]]] = None,
        run_id_to_tool_call_id: Optional[Dict[str, str]] = None,
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
            tool_call_id_map: 可选的 index -> tool_call_id 映射字典，用于流式工具调用
                              的 ID 一致性。如果提供，函数会自动更新此映射。
            tool_call_started_set: 可选的已发送 TOOL_CALL_START 的 tool_call_id 集合。
                                   用于确保每个工具调用只发送一次 TOOL_CALL_START，
                                   并在正确的时机发送 TOOL_CALL_END。
            tool_name_to_call_ids: 可选的 tool_name -> [tool_call_id] 队列映射。
                                   用于在 on_tool_start 中查找对应的 tool_call_id。
            run_id_to_tool_call_id: 可选的 run_id -> tool_call_id 映射。
                                    用于在 on_tool_end 中查找对应的 tool_call_id。

        Yields:
            str (文本内容) 或 AgentResult (AG-UI 事件)

        Example:
            >>> # 使用 astream_events（推荐使用 AgentRunConverter 类）
            >>> async for event in agent.astream_events(input, version="v2"):
            ...     for item in AgentRunConverter.to_agui_events(event):
            ...         yield item

            >>> # 使用 stream (updates 模式)
            >>> for event in agent.stream(input, stream_mode="updates"):
            ...     for item in AgentRunConverter.to_agui_events(event):
            ...         yield item

            >>> # 使用 astream (updates 模式)
            >>> async for event in agent.astream(input, stream_mode="updates"):
            ...     for item in AgentRunConverter.to_agui_events(event):
            ...         yield item
        """
        event_dict = AgentRunConverter._event_to_dict(event)

        # 根据事件格式选择对应的转换器
        if AgentRunConverter.is_astream_events_format(event_dict):
            # astream_events 格式：{"event": "on_xxx", "data": {...}}
            yield from AgentRunConverter._convert_astream_events_event(
                event_dict,
                tool_call_id_map,
                tool_call_started_set,
                tool_name_to_call_ids,
                run_id_to_tool_call_id,
            )

        elif AgentRunConverter.is_stream_updates_format(event_dict):
            # stream/astream(stream_mode="updates") 格式：{node_name: state_update}
            yield from AgentRunConverter._convert_stream_updates_event(
                event_dict, messages_key
            )

        elif AgentRunConverter.is_stream_values_format(event_dict):
            # stream/astream(stream_mode="values") 格式：完整 state dict
            yield from AgentRunConverter._convert_stream_values_event(
                event_dict, messages_key
            )
