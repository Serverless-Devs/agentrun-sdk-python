"""LangGraph/LangChain Agent 事件转换器

将 LangGraph/LangChain astream_events 的单个事件转换为 AgentRun 事件。

支持两种事件格式：
1. on_chat_model_stream - LangGraph create_react_agent 的流式输出
2. on_chain_stream - LangChain create_agent 的输出

Example:
    >>> async def invoke_agent(request: AgentRequest):
    ...     async for event in agent.astream_events(input_data, version="v2"):
    ...         for item in convert(event, request.hooks):
    ...             yield item
"""

import json
from typing import Any, Dict, Generator, List, Optional, Union

from agentrun.server.model import AgentEvent, AgentLifecycleHooks


def convert(
    event: Dict[str, Any],
    hooks: Optional[AgentLifecycleHooks] = None,
) -> Generator[Union[AgentEvent, str], None, None]:
    """转换单个 astream_events 事件

    Args:
        event: LangGraph/LangChain astream_events 的单个事件
        hooks: AgentLifecycleHooks，用于创建工具调用事件

    Yields:
        str (文本内容) 或 AgentEvent (工具调用事件)
    """
    event_type = event.get("event", "")
    data = event.get("data", {})

    # 1. LangGraph 格式: on_chat_model_stream
    if event_type == "on_chat_model_stream":
        chunk = data.get("chunk")
        if chunk:
            content = _get_content(chunk)
            if content:
                yield content

            # 流式工具调用参数
            if hooks:
                for tc in _get_tool_chunks(chunk):
                    tc_id = tc.get("id") or str(tc.get("index", ""))
                    if tc.get("name") and tc_id:
                        yield hooks.on_tool_call_start(
                            id=tc_id, name=tc["name"]
                        )
                    if tc.get("args") and tc_id:
                        yield hooks.on_tool_call_args_delta(
                            id=tc_id, delta=tc["args"]
                        )

    # 2. LangChain 格式: on_chain_stream (来自 create_agent)
    #    只处理 name="model" 的事件，避免重复（LangGraph 会发送 name="model" 和 name="LangGraph" 两个相同内容的事件）
    elif event_type == "on_chain_stream" and event.get("name") == "model":
        chunk_data = data.get("chunk", {})
        if isinstance(chunk_data, dict):
            # chunk 格式: {"messages": [AIMessage(...)]}
            messages = chunk_data.get("messages", [])

            for msg in messages:
                # 提取文本内容
                content = _get_content(msg)
                if content:
                    yield content

                # 提取工具调用
                if hooks:
                    tool_calls = _get_tool_calls(msg)
                    for tc in tool_calls:
                        tc_id = tc.get("id", "")
                        tc_name = tc.get("name", "")
                        tc_args = tc.get("args", {})
                        if tc_id and tc_name:
                            yield hooks.on_tool_call_start(
                                id=tc_id, name=tc_name
                            )
                            if tc_args:
                                yield hooks.on_tool_call_args(
                                    id=tc_id, args=_to_json(tc_args)
                                )

    # 3. 工具开始 (LangGraph)
    elif event_type == "on_tool_start" and hooks:
        run_id = event.get("run_id", "")
        tool_name = event.get("name", "")
        tool_input = data.get("input", {})

        if run_id:
            yield hooks.on_tool_call_start(id=run_id, name=tool_name)
            if tool_input:
                yield hooks.on_tool_call_args(
                    id=run_id, args=_to_json(tool_input)
                )

    # 4. 工具结束 (LangGraph)
    elif event_type == "on_tool_end" and hooks:
        run_id = event.get("run_id", "")
        output = data.get("output", "")

        if run_id:
            yield hooks.on_tool_call_result(
                id=run_id, result=str(output) if output else ""
            )
            yield hooks.on_tool_call_end(id=run_id)


def _get_content(obj: Any) -> Optional[str]:
    """提取文本内容"""
    if obj is None:
        return None

    # 字符串
    if isinstance(obj, str):
        return obj if obj else None

    # 有 content 属性的对象 (AIMessage, AIMessageChunk, etc.)
    if hasattr(obj, "content"):
        c = obj.content
        if isinstance(c, str) and c:
            return c
        if isinstance(c, list):
            parts = []
            for item in c:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("text", ""))
            return "".join(parts) or None

    return None


def _get_tool_chunks(chunk: Any) -> List[Dict[str, Any]]:
    """提取工具调用增量 (AIMessageChunk.tool_call_chunks)"""
    result: List[Dict[str, Any]] = []
    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
        for tc in chunk.tool_call_chunks:
            if isinstance(tc, dict):
                result.append(tc)
            else:
                result.append({
                    "id": getattr(tc, "id", None),
                    "name": getattr(tc, "name", None),
                    "args": getattr(tc, "args", None),
                    "index": getattr(tc, "index", None),
                })
    return result


def _get_tool_calls(msg: Any) -> List[Dict[str, Any]]:
    """提取完整工具调用 (AIMessage.tool_calls)"""
    result: List[Dict[str, Any]] = []
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            if isinstance(tc, dict):
                result.append(tc)
            else:
                result.append({
                    "id": getattr(tc, "id", None),
                    "name": getattr(tc, "name", None),
                    "args": getattr(tc, "args", None),
                })
    return result


def _to_json(obj: Any) -> str:
    """转 JSON 字符串"""
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=False)
