"""Mock LLM 场景配置

定义预设的测试场景，用于模拟 LLM 的不同响应行为：
- 简单对话（无工具调用）
- 单次工具调用
- 多工具同时调用
- 多轮工具调用
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid


@dataclass
class MockToolCall:
    """模拟的工具调用"""

    name: str
    arguments: Dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为 OpenAI 格式"""
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


@dataclass
class MockTurn:
    """模拟的对话轮次

    一个轮次可以包含工具调用或文本回复。
    """

    tool_calls: List[MockToolCall] = field(default_factory=list)
    content: Optional[str] = None
    finish_reason: str = "stop"

    def __post_init__(self):
        if self.tool_calls and not self.content:
            self.finish_reason = "tool_calls"

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def to_response(self, model: str = "mock-model") -> Dict[str, Any]:
        """转换为 OpenAI Chat Completion 格式"""
        message: Dict[str, Any] = {"role": "assistant"}

        if self.content:
            message["content"] = self.content
        else:
            message["content"] = None

        if self.tool_calls:
            message["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self.finish_reason,
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }


@dataclass
class MockScenario:
    """模拟场景配置

    一个场景可以包含多个轮次，用于模拟多轮对话。
    场景根据触发条件决定是否匹配当前请求。
    """

    name: str
    """场景名称"""

    trigger: Callable[[List[Dict]], bool]
    """触发条件：根据消息列表判断是否匹配"""

    turns: List[MockTurn]
    """响应轮次列表"""

    _current_turn: int = field(default=0, init=False)
    """当前轮次索引（内部使用）"""

    def match(self, messages: List[Dict]) -> bool:
        """判断消息是否匹配此场景"""
        return self.trigger(messages)

    def get_response(self, messages: List[Dict]) -> MockTurn:
        """获取当前轮次的响应

        根据消息历史判断当前是第几轮：
        - 如果最后一条消息是 tool 类型，说明工具已执行，进入下一轮
        - 否则返回当前轮次
        """
        # 计算当前应该返回哪一轮
        tool_rounds = sum(1 for msg in messages if msg.get("role") == "tool")

        # 根据工具消息数量确定当前轮次
        # 每个工具响应对应一个轮次的推进
        current_idx = min(tool_rounds, len(self.turns) - 1)
        return self.turns[current_idx]

    def reset(self):
        """重置场景状态"""
        self._current_turn = 0


class Scenarios:
    """预定义的测试场景工厂

    提供常用测试场景的快速创建方法。
    """

    @staticmethod
    def simple_chat(trigger: str, response: str) -> MockScenario:
        """创建简单对话场景（无工具调用）

        Args:
            trigger: 触发关键词（出现在用户消息中）
            response: 模型回复内容

        Returns:
            MockScenario: 场景配置
        """

        def trigger_fn(messages: List[Dict]) -> bool:
            # 查找最后一条用户消息
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return trigger in content
                    elif isinstance(content, list):
                        # 处理 content 是列表的情况
                        for item in content:
                            if isinstance(item, dict):
                                text = item.get("text", "")
                                if trigger in text:
                                    return True
            return False

        return MockScenario(
            name=f"simple_chat:{trigger}",
            trigger=trigger_fn,
            turns=[MockTurn(content=response)],
        )

    @staticmethod
    def single_tool_call(
        trigger: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        final_response: str,
        tool_call_id: str = "tool_call_1",
    ) -> MockScenario:
        """创建单次工具调用场景

        Args:
            trigger: 触发关键词
            tool_name: 工具名称
            tool_args: 工具参数
            final_response: 工具执行后的最终回复
            tool_call_id: 工具调用 ID

        Returns:
            MockScenario: 场景配置
        """

        def trigger_fn(messages: List[Dict]) -> bool:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return trigger in content
            return False

        return MockScenario(
            name=f"single_tool:{tool_name}",
            trigger=trigger_fn,
            turns=[
                MockTurn(
                    tool_calls=[
                        MockToolCall(
                            name=tool_name,
                            arguments=tool_args,
                            id=tool_call_id,
                        )
                    ]
                ),
                MockTurn(content=final_response),
            ],
        )

    @staticmethod
    def multi_tool_calls(
        trigger: str,
        tool_calls: List[Tuple[str, Dict[str, Any]]],
        final_response: str,
    ) -> MockScenario:
        """创建多工具同时调用场景

        Args:
            trigger: 触发关键词
            tool_calls: 工具调用列表 [(name, args), ...]
            final_response: 所有工具执行后的最终回复

        Returns:
            MockScenario: 场景配置
        """

        def trigger_fn(messages: List[Dict]) -> bool:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return trigger in content
            return False

        return MockScenario(
            name=f"multi_tools:{trigger}",
            trigger=trigger_fn,
            turns=[
                MockTurn(
                    tool_calls=[
                        MockToolCall(
                            name=name,
                            arguments=args,
                            id=f"tool_call_{i + 1}",
                        )
                        for i, (name, args) in enumerate(tool_calls)
                    ]
                ),
                MockTurn(content=final_response),
            ],
        )

    @staticmethod
    def multi_round_tools(
        trigger: str,
        rounds: List[List[Tuple[str, Dict[str, Any]]]],
        final_response: str,
    ) -> MockScenario:
        """创建多轮工具调用场景

        Args:
            trigger: 触发关键词
            rounds: 多轮工具调用 [[(name, args), ...], ...]
            final_response: 所有轮次完成后的最终回复

        Returns:
            MockScenario: 场景配置
        """

        def trigger_fn(messages: List[Dict]) -> bool:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return trigger in content
            return False

        turns = []
        tool_call_counter = 1

        for round_tools in rounds:
            mock_tool_calls = []
            for name, args in round_tools:
                mock_tool_calls.append(
                    MockToolCall(
                        name=name,
                        arguments=args,
                        id=f"tool_call_{tool_call_counter}",
                    )
                )
                tool_call_counter += 1
            turns.append(MockTurn(tool_calls=mock_tool_calls))

        turns.append(MockTurn(content=final_response))

        return MockScenario(
            name=f"multi_round:{trigger}",
            trigger=trigger_fn,
            turns=turns,
        )

    @staticmethod
    def default_weather_scenario() -> MockScenario:
        """创建默认的天气查询场景

        这是一个预设的测试场景，用于测试工具调用。
        触发词: "天气" 或 "weather"
        工具: weather_lookup(city)
        """
        return Scenarios.single_tool_call(
            trigger="天气",
            tool_name="weather_lookup",
            tool_args={"city": "上海"},
            final_response="上海今天天气晴朗，温度 25°C。",
        )

    @staticmethod
    def default_multi_tool_scenario() -> MockScenario:
        """创建默认的多工具调用场景

        触发词: "查询" 或 "上海"
        工具: weather_lookup, get_time_now
        """
        return Scenarios.multi_tool_calls(
            trigger="上海",
            tool_calls=[
                ("weather_lookup", {"city": "上海"}),
                ("get_time_now", {}),
            ],
            final_response="final result",
        )
