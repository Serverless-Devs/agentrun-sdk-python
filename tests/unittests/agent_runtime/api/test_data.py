"""Agent Runtime Data API 单元测试"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.agent_runtime.api.data import AgentRuntimeDataAPI, InvokeArgs
from agentrun.utils.config import Config


class TestAgentRuntimeDataAPIInit:
    """AgentRuntimeDataAPI 初始化测试"""

    def test_init(self):
        """测试初始化"""
        with patch.dict(
            os.environ,
            {
                "AGENTRUN_ACCESS_KEY_ID": "test-key",
                "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
                "AGENTRUN_ACCOUNT_ID": "test-account",
            },
        ):
            api = AgentRuntimeDataAPI(
                agent_runtime_name="test-runtime",
                agent_runtime_endpoint_name="Default",
            )

            assert api.resource_name == "test-runtime"
            assert (
                "agent-runtimes/test-runtime/endpoints/Default/invocations"
                in api.namespace
            )

    def test_init_with_custom_endpoint(self):
        """测试使用自定义端点初始化"""
        with patch.dict(
            os.environ,
            {
                "AGENTRUN_ACCESS_KEY_ID": "test-key",
                "AGENTRUN_ACCESS_KEY_SECRET": "test-secret",
                "AGENTRUN_ACCOUNT_ID": "test-account",
            },
        ):
            api = AgentRuntimeDataAPI(
                agent_runtime_name="my-agent",
                agent_runtime_endpoint_name="custom-endpoint",
            )

            assert (
                "agent-runtimes/my-agent/endpoints/custom-endpoint/invocations"
                in api.namespace
            )

    def test_init_with_config(self):
        """测试使用 config 初始化"""
        config = Config(
            access_key_id="test-key",
            access_key_secret="test-secret",
            account_id="test-account",
        )
        api = AgentRuntimeDataAPI(
            agent_runtime_name="test-runtime",
            config=config,
        )

        # Config 可能被合并，检查关键属性而不是对象相等
        assert api.config._access_key_id == "test-key"
        assert api.config._access_key_secret == "test-secret"
        assert api.config._account_id == "test-account"


class TestInvokeArgs:
    """InvokeArgs TypedDict 测试"""

    def test_invoke_args_structure(self):
        """测试 InvokeArgs 结构"""
        args: InvokeArgs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "config": None,
        }

        assert "messages" in args
        assert "stream" in args
        assert "config" in args


class TestAgentRuntimeDataAPIInvokeOpenai:
    """AgentRuntimeDataAPI invoke_openai 方法测试"""

    def test_invoke_openai(self):
        """测试 invoke_openai"""
        config = Config(
            access_key_id="test-key",
            access_key_secret="test-secret",
            account_id="test-account",
        )
        api = AgentRuntimeDataAPI(
            agent_runtime_name="test-runtime",
            agent_runtime_endpoint_name="Default",
            config=config,
        )

        # Mock OpenAI 客户端 - 因为是 lazy import，所以 mock openai 模块
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_completions = MagicMock()
            mock_completions.create.return_value = {
                "choices": [{"message": {"content": "Hello!"}}]
            }
            mock_client.chat.completions = mock_completions
            mock_openai.return_value = mock_client

            with patch("httpx.Client"):
                result = api.invoke_openai(
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

                assert result is not None
                mock_completions.create.assert_called_once()

    def test_invoke_openai_with_stream(self):
        """测试 invoke_openai 流式模式"""
        config = Config(
            access_key_id="test-key",
            access_key_secret="test-secret",
            account_id="test-account",
        )
        api = AgentRuntimeDataAPI(
            agent_runtime_name="test-runtime",
            agent_runtime_endpoint_name="Default",
            config=config,
        )

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_completions = MagicMock()
            # 流式返回生成器
            mock_completions.create.return_value = iter([
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo!"}}]},
            ])
            mock_client.chat.completions = mock_completions
            mock_openai.return_value = mock_client

            with patch("httpx.Client"):
                result = api.invoke_openai(
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True,
                )

                assert result is not None

    def test_invoke_openai_with_config_override(self):
        """测试 invoke_openai 使用 config 覆盖"""
        config = Config(
            access_key_id="test-key",
            access_key_secret="test-secret",
            account_id="test-account",
        )
        api = AgentRuntimeDataAPI(
            agent_runtime_name="test-runtime",
            agent_runtime_endpoint_name="Default",
            config=config,
        )

        override_config = Config(
            access_key_id="custom-key",
            access_key_secret="custom-secret",
            account_id="custom-account",
        )

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_completions = MagicMock()
            mock_completions.create.return_value = {"choices": []}
            mock_client.chat.completions = mock_completions
            mock_openai.return_value = mock_client

            with patch("httpx.Client"):
                result = api.invoke_openai(
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                    config=override_config,
                )

                assert result is not None


class TestAgentRuntimeDataAPIInvokeOpenaiAsync:
    """AgentRuntimeDataAPI invoke_openai_async 方法测试"""

    def test_invoke_openai_async(self):
        """测试 invoke_openai_async"""
        config = Config(
            access_key_id="test-key",
            access_key_secret="test-secret",
            account_id="test-account",
        )
        api = AgentRuntimeDataAPI(
            agent_runtime_name="test-runtime",
            agent_runtime_endpoint_name="Default",
            config=config,
        )

        with patch("openai.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_completions = MagicMock()

            # 返回一个同步调用结果（因为 create 返回的是 coroutine）
            async def mock_create(*args, **kwargs):
                return {"choices": [{"message": {"content": "Hello!"}}]}

            mock_completions.create = mock_create
            mock_client.chat.completions = mock_completions
            mock_async_openai.return_value = mock_client

            with patch("httpx.AsyncClient"):
                # invoke_openai_async 返回的是 coroutine，需要 await
                result = asyncio.run(
                    api.invoke_openai_async(
                        messages=[{"role": "user", "content": "Hello"}],
                        stream=False,
                    )
                )

                # 验证返回结果
                assert result is not None

    def test_invoke_openai_async_with_stream(self):
        """测试 invoke_openai_async 流式模式"""
        config = Config(
            access_key_id="test-key",
            access_key_secret="test-secret",
            account_id="test-account",
        )
        api = AgentRuntimeDataAPI(
            agent_runtime_name="test-runtime",
            agent_runtime_endpoint_name="Default",
            config=config,
        )

        with patch("openai.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_completions = MagicMock()

            async def mock_create(*args, **kwargs):
                async def async_gen():
                    yield {"choices": [{"delta": {"content": "Hello"}}]}

                return async_gen()

            mock_completions.create = mock_create
            mock_client.chat.completions = mock_completions
            mock_async_openai.return_value = mock_client

            with patch("httpx.AsyncClient"):
                result = asyncio.run(
                    api.invoke_openai_async(
                        messages=[{"role": "user", "content": "Hello"}],
                        stream=True,
                    )
                )

                assert result is not None


class TestAgentRuntimeDataAPIWithPath:
    """AgentRuntimeDataAPI with_path 方法测试"""

    def test_with_path(self):
        """测试 with_path 方法"""
        config = Config(
            access_key_id="test-key",
            access_key_secret="test-secret",
            account_id="test-account",
        )
        api = AgentRuntimeDataAPI(
            agent_runtime_name="test-runtime",
            agent_runtime_endpoint_name="Default",
            config=config,
        )

        # 测试 with_path 返回正确的 URL
        result = api.with_path("openai/v1")
        assert "openai/v1" in result


class TestAgentRuntimeDataAPIAuth:
    """AgentRuntimeDataAPI auth 方法测试"""

    def test_auth(self):
        """测试 auth 方法"""
        config = Config(
            access_key_id="test-key",
            access_key_secret="test-secret",
            account_id="test-account",
        )
        api = AgentRuntimeDataAPI(
            agent_runtime_name="test-runtime",
            agent_runtime_endpoint_name="Default",
            config=config,
        )

        # 测试 auth 返回三元组
        result = api.auth(headers={})
        assert len(result) == 3  # (body, headers, params)
