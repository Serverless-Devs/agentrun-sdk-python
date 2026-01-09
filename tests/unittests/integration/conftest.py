"""Integration 测试的公共 fixtures 和辅助函数

提供所有 integration 测试共享的 fixtures：
- Mock LLM Server
- Mock Model
- Mock ToolSet
- 消息工厂函数
"""

from typing import Any, Dict, List, Union
from unittest.mock import MagicMock

import pytest

from agentrun.integration.builtin.model import CommonModel
from agentrun.integration.langgraph import AgentRunConverter
from agentrun.integration.utils.tool import CommonToolSet, tool
from agentrun.model.model_proxy import ModelProxy
from agentrun.server.model import AgentEvent, EventType

from .mock_llm_server import MockLLMServer
from .scenarios import Scenarios

# =============================================================================
# 共享的 TestToolSet
# =============================================================================


class SharedTestToolSet(CommonToolSet):
    """共享的测试工具集

    提供两个测试工具：
    - weather_lookup: 查询天气
    - get_time_now: 获取当前时间
    """

    def __init__(self, timezone: str = "UTC"):
        self.time_zone = timezone
        self.call_history: List[Any] = []
        super().__init__()

    @tool(description="查询城市天气")
    def weather_lookup(self, city: str) -> str:
        result = f"{city} 天气晴朗"
        self.call_history.append(result)
        return result

    @tool()
    def get_time_now(self) -> dict:
        """返回当前时间"""
        result = {
            "time": "2025-01-02 15:04:05",
            "timezone": self.time_zone,
        }
        self.call_history.append(result)
        return result


# =============================================================================
# Mock 消息工厂函数
# =============================================================================


def create_mock_ai_message(
    content: str = "",
    tool_calls: List[Dict[str, Any]] = None,
) -> MagicMock:
    """创建模拟的 AIMessage 对象

    Args:
        content: 消息内容
        tool_calls: 工具调用列表

    Returns:
        MagicMock: 模拟的 AIMessage 对象
    """
    msg = MagicMock()
    msg.content = content
    msg.type = "ai"
    msg.tool_calls = tool_calls or []
    return msg


def create_mock_ai_message_chunk(
    content: str = "",
    tool_call_chunks: List[Dict] = None,
) -> MagicMock:
    """创建模拟的 AIMessageChunk 对象（流式输出）

    Args:
        content: 内容片段
        tool_call_chunks: 工具调用片段列表

    Returns:
        MagicMock: 模拟的 AIMessageChunk 对象
    """
    chunk = MagicMock()
    chunk.content = content
    chunk.tool_call_chunks = tool_call_chunks or []
    return chunk


def create_mock_tool_message(content: str, tool_call_id: str) -> MagicMock:
    """创建模拟的 ToolMessage 对象

    Args:
        content: 工具执行结果
        tool_call_id: 工具调用 ID

    Returns:
        MagicMock: 模拟的 ToolMessage 对象
    """
    msg = MagicMock()
    msg.content = content
    msg.type = "tool"
    msg.tool_call_id = tool_call_id
    return msg


# =============================================================================
# 事件转换辅助函数
# =============================================================================


def convert_and_collect(events: List[Dict]) -> List[Union[str, AgentEvent]]:
    """转换事件列表并收集所有结果

    Args:
        events: LangChain/LangGraph 事件列表

    Returns:
        List: 转换后的 AgentEvent 列表
    """
    results: List[Union[str, AgentEvent]] = []
    for event in events:
        results.extend(AgentRunConverter.to_agui_events(event))
    return results


def filter_agent_events(
    results: List[Union[str, AgentEvent]], event_type: EventType
) -> List[AgentEvent]:
    """过滤特定类型的 AgentEvent

    Args:
        results: 转换结果列表
        event_type: 要过滤的事件类型

    Returns:
        List[AgentEvent]: 过滤后的事件列表
    """
    return [
        r
        for r in results
        if isinstance(r, AgentEvent) and r.event == event_type
    ]


def get_event_types(results: List[Union[str, AgentEvent]]) -> List[EventType]:
    """获取结果中所有 AgentEvent 的类型

    Args:
        results: 转换结果列表

    Returns:
        List[EventType]: 事件类型列表
    """
    return [r.event for r in results if isinstance(r, AgentEvent)]


# =============================================================================
# astream_events 格式的事件工厂
# =============================================================================


def create_on_chat_model_stream_event(chunk: MagicMock) -> Dict:
    """创建 on_chat_model_stream 事件

    Args:
        chunk: AIMessageChunk 对象

    Returns:
        Dict: astream_events 格式的事件
    """
    return {
        "event": "on_chat_model_stream",
        "data": {"chunk": chunk},
    }


def create_on_tool_start_event(
    tool_name: str,
    tool_input: Dict,
    run_id: str = "run-123",
    tool_call_id: str = None,
) -> Dict:
    """创建 on_tool_start 事件

    Args:
        tool_name: 工具名称
        tool_input: 工具输入参数
        run_id: 运行 ID
        tool_call_id: 工具调用 ID（可选，会放入 metadata）

    Returns:
        Dict: astream_events 格式的事件
    """
    event = {
        "event": "on_tool_start",
        "name": tool_name,
        "run_id": run_id,
        "data": {"input": tool_input},
    }
    if tool_call_id:
        event["metadata"] = {"langgraph_tool_call_id": tool_call_id}
    return event


def create_on_tool_end_event(
    output: Any,
    run_id: str = "run-123",
    tool_call_id: str = None,
) -> Dict:
    """创建 on_tool_end 事件

    Args:
        output: 工具输出
        run_id: 运行 ID
        tool_call_id: 工具调用 ID（可选，会放入 metadata）

    Returns:
        Dict: astream_events 格式的事件
    """
    event = {
        "event": "on_tool_end",
        "run_id": run_id,
        "data": {"output": output},
    }
    if tool_call_id:
        event["metadata"] = {"langgraph_tool_call_id": tool_call_id}
    return event


def create_on_tool_error_event(
    error: str,
    run_id: str = "run-123",
) -> Dict:
    """创建 on_tool_error 事件

    Args:
        error: 错误信息
        run_id: 运行 ID

    Returns:
        Dict: astream_events 格式的事件
    """
    return {
        "event": "on_tool_error",
        "run_id": run_id,
        "data": {"error": error},
    }


# =============================================================================
# stream_mode 格式的事件工厂
# =============================================================================


def create_stream_updates_event(node_name: str, messages: List) -> Dict:
    """创建 stream_mode="updates" 格式的事件

    Args:
        node_name: 节点名称（如 "model", "agent", "tools"）
        messages: 消息列表

    Returns:
        Dict: stream_mode="updates" 格式的事件
    """
    return {node_name: {"messages": messages}}


def create_stream_values_event(messages: List) -> Dict:
    """创建 stream_mode="values" 格式的事件

    Args:
        messages: 消息列表

    Returns:
        Dict: stream_mode="values" 格式的事件
    """
    return {"messages": messages}


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def ai_message_factory():
    """提供 AIMessage 工厂函数"""
    return create_mock_ai_message


@pytest.fixture
def ai_message_chunk_factory():
    """提供 AIMessageChunk 工厂函数"""
    return create_mock_ai_message_chunk


@pytest.fixture
def tool_message_factory():
    """提供 ToolMessage 工厂函数"""
    return create_mock_tool_message


@pytest.fixture
def shared_mock_server(monkeypatch: Any, respx_mock: Any) -> MockLLMServer:
    """提供共享的 Mock LLM Server

    预配置了默认场景。
    """
    server = MockLLMServer(expect_tools=True, validate_tools=False)
    server.install(monkeypatch)
    server.add_default_scenarios()
    return server


@pytest.fixture
def shared_mocked_model(
    shared_mock_server: MockLLMServer, monkeypatch: Any
) -> CommonModel:
    """提供共享的 Mock Model"""
    from agentrun.integration.builtin.model import model

    mock_model_proxy = ModelProxy(model_proxy_name="mock-model-proxy")

    monkeypatch.setattr(
        "agentrun.model.client.ModelClient.get",
        lambda *args, **kwargs: mock_model_proxy,
    )
    return model("mock-model")


@pytest.fixture
def shared_mocked_toolset() -> SharedTestToolSet:
    """提供共享的 Mock ToolSet"""
    return SharedTestToolSet(timezone="UTC")
