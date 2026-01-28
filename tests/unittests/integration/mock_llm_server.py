"""统一的 Mock LLM Server 模块

为所有 integration 测试提供统一的 Mock LLM 能力，支持：
- HTTP 请求拦截和响应模拟
- litellm 函数 mock
- 请求参数捕获和验证
- 流式/非流式响应模拟
- 场景配置（简单对话、工具调用、多轮对话）
"""

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Dict, List, Optional

from litellm.files.main import ModelResponse
import pydash
import respx

from agentrun.model.api.data import BaseInfo
from agentrun.utils.log import logger

from .scenarios import MockScenario, Scenarios


@dataclass
class CapturedRequest:
    """捕获的请求信息"""

    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]]
    stream: Optional[bool]
    stream_options: Optional[Dict[str, Any]]
    model: Optional[str]
    all_kwargs: Dict[str, Any]


@dataclass
class MockLLMServer:
    """统一的 Mock LLM Server

    提供 HTTP 和 litellm 的 mock 能力，支持捕获请求参数和场景配置。

    使用方式:
        # 基本用法
        server = MockLLMServer()
        server.install(monkeypatch, respx_mock)  # 需要传入 respx_mock

        # 添加自定义场景
        server.add_scenario(Scenarios.simple_chat("你好", "你好！"))
        server.add_scenario(Scenarios.single_tool_call(
            "天气", "weather_lookup", {"city": "上海"}, "晴天"
        ))

        # 测试代码...

        # 验证捕获的请求
        assert server.captured_requests[0].stream is True
    """

    base_url: str = "https://mock-llm.local/v1"
    expect_tools: bool = True
    captured_requests: List[CapturedRequest] = field(default_factory=list)
    scenarios: List[MockScenario] = field(default_factory=list)
    response_builder: Optional[Callable[[List[Dict], Optional[List]], Dict]] = (
        None
    )
    validate_tools: bool = True
    """是否验证工具格式（默认 True）"""

    _respx_router: Any = field(default=None, init=False, repr=False)
    """内部使用的 respx router 实例"""

    def install(
        self, monkeypatch: Any, respx_mock: Any = None
    ) -> "MockLLMServer":
        """安装所有 mock

        Args:
            monkeypatch: pytest monkeypatch fixture
            respx_mock: pytest respx_mock fixture（必须传入以确保 mock 生效）

        Returns:
            self: 返回自身以便链式调用
        """
        self._respx_router = respx_mock
        self._patch_model_info(monkeypatch)
        self._patch_litellm(monkeypatch)
        self._setup_respx()
        return self

    def add_scenario(self, scenario: MockScenario) -> "MockLLMServer":
        """添加测试场景

        Args:
            scenario: 场景配置

        Returns:
            self: 返回自身以便链式调用
        """
        self.scenarios.append(scenario)
        return self

    def add_default_scenarios(self) -> "MockLLMServer":
        """添加默认测试场景

        包括：
        - 多工具调用场景（触发词：上海）
        - 简单对话场景（触发词：你好）

        Returns:
            self: 返回自身以便链式调用
        """
        self.add_scenario(Scenarios.default_multi_tool_scenario())
        self.add_scenario(Scenarios.simple_chat("你好", "你好！我是AI助手。"))
        return self

    def clear_scenarios(self) -> "MockLLMServer":
        """清除所有场景

        Returns:
            self: 返回自身以便链式调用
        """
        self.scenarios.clear()
        return self

    def reset_scenarios(self) -> "MockLLMServer":
        """重置所有场景状态（用于多次测试）

        Returns:
            self: 返回自身以便链式调用
        """
        for scenario in self.scenarios:
            scenario.reset()
        return self

    def clear_captured_requests(self) -> "MockLLMServer":
        """清除捕获的请求

        Returns:
            self: 返回自身以便链式调用
        """
        self.captured_requests.clear()
        return self

    def get_last_request(self) -> Optional[CapturedRequest]:
        """获取最后一个捕获的请求"""
        return self.captured_requests[-1] if self.captured_requests else None

    def assert_no_stream_options_when_not_streaming(self):
        """断言非流式请求不包含 stream_options

        这是检测 stream_options 使用错误的核心断言。
        """
        for req in self.captured_requests:
            if req.stream is False or req.stream is None:
                # 非流式模式不应该传递 stream_options
                if req.stream_options is not None:
                    include_usage = pydash.get(
                        req.stream_options, "include_usage"
                    )
                    if include_usage is True:
                        raise AssertionError(
                            "非流式请求不应包含"
                            " stream_options.include_usage=True，"
                            f"但收到: stream={req.stream}, "
                            f"stream_options={req.stream_options}"
                        )

    def assert_stream_options_when_streaming(self):
        """断言流式请求包含正确的 stream_options"""
        for req in self.captured_requests:
            if req.stream is True:
                include_usage = pydash.get(
                    req.stream_options, "include_usage", False
                )
                assert include_usage is True, (
                    "流式请求应包含 stream_options.include_usage=True，"
                    f"但收到: stream={req.stream}, "
                    f"stream_options={req.stream_options}"
                )

    def _capture_request(self, **kwargs: Any) -> CapturedRequest:
        """捕获请求参数"""
        captured = CapturedRequest(
            messages=kwargs.get("messages", []),
            tools=kwargs.get("tools"),
            stream=kwargs.get("stream"),
            stream_options=kwargs.get("stream_options"),
            model=kwargs.get("model"),
            all_kwargs=kwargs,
        )
        self.captured_requests.append(captured)
        logger.debug(
            "Captured request: stream=%s, stream_options=%s, messages=%d",
            captured.stream,
            captured.stream_options,
            len(captured.messages),
        )
        return captured

    def _patch_model_info(self, monkeypatch: Any):
        """Mock ModelDataAPI.model_info 方法"""

        def fake_model_info(inner_self: Any, config: Any = None) -> BaseInfo:
            return BaseInfo(
                api_key="mock-api-key",
                base_url=self.base_url,
                model=inner_self.model_name or "mock-model",
                headers={
                    "Authorization": "Bearer mock-token",
                    "Agentrun-Access-Token": "mock-token",
                },
            )

        monkeypatch.setattr(
            "agentrun.model.api.data.ModelDataAPI.model_info",
            fake_model_info,
        )

    def _patch_litellm(self, monkeypatch: Any):
        """Mock litellm.completion 和 litellm.acompletion"""

        def fake_completion(*args: Any, **kwargs: Any) -> ModelResponse:
            self._capture_request(**kwargs)
            messages = kwargs.get("messages") or []
            tools_payload = kwargs.get("tools")
            return self._build_model_response(messages, tools_payload)

        async def fake_acompletion(*args: Any, **kwargs: Any) -> ModelResponse:
            self._capture_request(**kwargs)
            messages = kwargs.get("messages") or []
            tools_payload = kwargs.get("tools")
            return self._build_model_response(messages, tools_payload)

        monkeypatch.setattr("litellm.completion", fake_completion)
        monkeypatch.setattr("litellm.acompletion", fake_acompletion)

        # Patch Google ADK 的 litellm 导入
        try:
            import google.adk.models.lite_llm as lite_llm_module

            monkeypatch.setattr(
                lite_llm_module, "acompletion", fake_acompletion
            )
            monkeypatch.setattr(lite_llm_module, "completion", fake_completion)
        except ImportError:
            pass  # google.adk not installed

    def _setup_respx(self):
        """设置 respx HTTP mock

        关键修复：使用 pytest-respx fixture 提供的 router 而不是全局 respx

        问题背景：
        - 之前直接使用全局 respx.route() 在 CI 环境中不生效
        - 全局 respx router 在某些环境中可能没有正确初始化
        - 导致 HTTP 请求没有被拦截，Google ADK 发送真实请求

        解决方案：
        - 使用 pytest-respx 提供的 respx_mock fixture
        - 通过 install() 方法传入 respx_mock
        - 确保 mock 在所有环境中一致生效
        """

        def extract_payload(request: Any) -> Dict[str, Any]:
            try:
                if request.content:
                    body = request.content
                    if isinstance(body, (bytes, bytearray)):
                        body = body.decode()
                    if isinstance(body, str) and body.strip():
                        return json.loads(body)
            except (json.JSONDecodeError, AttributeError):
                pass
            return {}

        def build_response(request: Any, route: Any) -> respx.MockResponse:
            payload = extract_payload(request)

            # 捕获 HTTP 请求
            self._capture_request(**payload)

            is_stream = payload.get("stream", False)
            response_json = self._build_response(
                payload.get("messages") or [],
                payload.get("tools"),
            )

            if is_stream:
                return respx.MockResponse(
                    status_code=200,
                    content=self._build_sse_stream(response_json),
                    headers={"content-type": "text/event-stream"},
                )
            return respx.MockResponse(status_code=200, json=response_json)

        # 关键修复：使用传入的 respx_router 而不是全局 respx
        # 如果没有传入 respx_router，回退到全局 respx（向后兼容）
        router = self._respx_router if self._respx_router is not None else respx
        router.route(url__startswith=self.base_url).mock(
            side_effect=build_response
        )

    def _find_matching_scenario(
        self, messages: List[Dict]
    ) -> Optional[MockScenario]:
        """查找匹配的场景"""
        for scenario in self.scenarios:
            if scenario.match(messages):
                return scenario
        return None

    def _build_response(
        self, messages: List[Dict], tools_payload: Optional[List]
    ) -> Dict[str, Any]:
        """构建响应 JSON

        优先使用场景配置，如果没有匹配的场景则使用默认逻辑。
        """
        # 使用自定义 response_builder
        if self.response_builder:
            return self.response_builder(messages, tools_payload)

        logger.debug(
            "Building response for %d messages, tools=%s",
            len(messages),
            tools_payload is not None,
        )

        # 添加详细的消息日志，帮助调试框架的消息格式
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content_preview = str(msg.get("content", ""))[:100]
            logger.debug(
                "Message[%d] role=%s, content_preview=%s",
                i,
                role,
                content_preview,
            )
            if "tool_calls" in msg:
                logger.debug(
                    "Message[%d] has tool_calls: %s", i, msg.get("tool_calls")
                )
            if "tool_call_id" in msg:
                logger.debug(
                    "Message[%d] has tool_call_id: %s",
                    i,
                    msg.get("tool_call_id"),
                )

        # 验证工具格式
        if self.validate_tools and self.expect_tools and tools_payload:
            self._assert_tools(tools_payload)
        elif tools_payload is not None:
            assert isinstance(tools_payload, list)

        if not messages:
            raise AssertionError("messages payload cannot be empty")

        # 查找匹配的场景
        scenario = self._find_matching_scenario(messages)
        if scenario:
            turn = scenario.get_response(messages)
            return turn.to_response()

        # 默认逻辑：未匹配场景时使用
        return self._build_default_response(messages, tools_payload)

    def _build_default_response(
        self, messages: List[Dict], tools_payload: Optional[List]
    ) -> Dict[str, Any]:
        """构建默认响应（无场景匹配时使用）"""
        # 检查消息历史中是否已经有 tool 结果
        # 这是关键修复：不只检查最后一条消息，而是检查整个历史
        has_tool_results = any(msg.get("role") == "tool" for msg in messages)

        if has_tool_results:
            # 已经有 tool 结果，应该返回最终答案而不是再次调用工具
            return {
                "id": "chatcmpl-mock-final",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "mock-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "final result",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }

        # 如果有工具且未调用过，返回工具调用
        if tools_payload:
            return {
                "id": "chatcmpl-mock-tools",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "mock-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tool_call_1",
                                "type": "function",
                                "function": {
                                    "name": "weather_lookup",
                                    "arguments": '{"city": "上海"}',
                                },
                            },
                            {
                                "id": "tool_call_2",
                                "type": "function",
                                "function": {
                                    "name": "get_time_now",
                                    "arguments": "{}",
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            }

        # 无工具时返回简单文本响应
        return {
            "id": "chatcmpl-mock-simple",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm a helpful assistant.",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 5,
                "total_tokens": 8,
            },
        }

    def _build_model_response(
        self, messages: List[Dict], tools_payload: Optional[List]
    ) -> ModelResponse:
        """构建 ModelResponse 对象"""
        response_dict = self._build_response(messages, tools_payload)
        return ModelResponse(**response_dict)

    def _build_sse_stream(self, response_json: Dict[str, Any]) -> bytes:
        """构建 SSE 流式响应"""
        chunks = []
        choice = response_json.get("choices", [{}])[0]
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls")

        # First chunk with role
        first_chunk = {
            "id": response_json.get("id", "chatcmpl-mock"),
            "object": "chat.completion.chunk",
            "created": response_json.get("created", 1234567890),
            "model": response_json.get("model", "mock-model"),
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        }
        chunks.append(f"data: {json.dumps(first_chunk)}\n\n")

        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                tc_chunk = {
                    "id": response_json.get("id", "chatcmpl-mock"),
                    "object": "chat.completion.chunk",
                    "created": response_json.get("created", 1234567890),
                    "model": response_json.get("model", "mock-model"),
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": i,
                                "id": tool_call.get("id"),
                                "type": "function",
                                "function": tool_call.get("function"),
                            }],
                        },
                        "finish_reason": None,
                    }],
                }
                chunks.append(f"data: {json.dumps(tc_chunk)}\n\n")
        else:
            content = message.get("content", "")
            if content:
                content_chunk = {
                    "id": response_json.get("id", "chatcmpl-mock"),
                    "object": "chat.completion.chunk",
                    "created": response_json.get("created", 1234567890),
                    "model": response_json.get("model", "mock-model"),
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }],
                }
                chunks.append(f"data: {json.dumps(content_chunk)}\n\n")

        # Final chunk with finish_reason
        finish_reason = "tool_calls" if tool_calls else "stop"
        final_chunk = {
            "id": response_json.get("id", "chatcmpl-mock"),
            "object": "chat.completion.chunk",
            "created": response_json.get("created", 1234567890),
            "model": response_json.get("model", "mock-model"),
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }],
        }
        chunks.append(f"data: {json.dumps(final_chunk)}\n\n")
        chunks.append("data: [DONE]\n\n")

        return "".join(chunks).encode("utf-8")

    def _assert_tools(self, tools_payload: List[Dict]):
        """验证工具参数格式"""
        assert isinstance(tools_payload, list)
        assert (
            pydash.get(tools_payload, "[0].function.name") == "weather_lookup"
        )
        assert (
            pydash.get(tools_payload, "[0].function.description")
            == "查询城市天气"
        )
        assert (
            pydash.get(
                tools_payload,
                "[0].function.parameters.properties.city.type",
            )
            == "string"
        )
        assert (
            pydash.get(tools_payload, "[0].function.parameters.type")
            == "object"
        )
        assert "city" in (
            pydash.get(tools_payload, "[0].function.parameters.required", [])
            or []
        )

        assert pydash.get(tools_payload, "[1].function.name") == "get_time_now"
        assert (
            pydash.get(tools_payload, "[1].function.description")
            == "返回当前时间"
        )
        assert pydash.get(
            tools_payload, "[1].function.parameters.properties"
        ) in (
            {},
            None,
        )
        assert (
            pydash.get(tools_payload, "[1].function.parameters.type")
            == "object"
        )
