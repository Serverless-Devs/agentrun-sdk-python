"""协议基类测试

测试 ProtocolHandler 和 BaseProtocolHandler 的基类方法。
"""

import pytest

from agentrun.server.protocol import BaseProtocolHandler, ProtocolHandler


class TestProtocolHandler:
    """测试 ProtocolHandler 基类"""

    def test_get_prefix_default(self):
        """测试 get_prefix 默认返回空字符串"""

        class TestHandler(ProtocolHandler):
            name = "test"

            def as_fastapi_router(self, agent_invoker):
                pass

        handler = TestHandler()
        # 调用父类的 get_prefix 方法
        assert ProtocolHandler.get_prefix(handler) == ""

    def test_get_prefix_override(self):
        """测试子类可以覆盖 get_prefix"""

        class TestHandler(ProtocolHandler):
            name = "test"

            def as_fastapi_router(self, agent_invoker):
                pass

            def get_prefix(self):
                return "/custom"

        handler = TestHandler()
        assert handler.get_prefix() == "/custom"


class TestBaseProtocolHandler:
    """测试 BaseProtocolHandler 基类"""

    def test_parse_request_not_implemented(self):
        """测试 parse_request 未实现时抛出异常"""

        class TestHandler(BaseProtocolHandler):
            name = "test"

            def as_fastapi_router(self, agent_invoker):
                pass

        handler = TestHandler()

        with pytest.raises(
            NotImplementedError, match="Subclass must implement"
        ):
            import asyncio

            asyncio.run(handler.parse_request(None, {}))

    def test_is_iterator_with_dict(self):
        """测试 _is_iterator 对字典返回 False"""

        class TestHandler(BaseProtocolHandler):
            name = "test"

            def as_fastapi_router(self, agent_invoker):
                pass

        handler = TestHandler()
        assert handler._is_iterator({}) is False
        assert handler._is_iterator({"a": 1}) is False

    def test_is_iterator_with_list(self):
        """测试 _is_iterator 对列表返回 False"""

        class TestHandler(BaseProtocolHandler):
            name = "test"

            def as_fastapi_router(self, agent_invoker):
                pass

        handler = TestHandler()
        assert handler._is_iterator([]) is False
        assert handler._is_iterator([1, 2, 3]) is False

    def test_is_iterator_with_string(self):
        """测试 _is_iterator 对字符串返回 False"""

        class TestHandler(BaseProtocolHandler):
            name = "test"

            def as_fastapi_router(self, agent_invoker):
                pass

        handler = TestHandler()
        assert handler._is_iterator("") is False
        assert handler._is_iterator("hello") is False

    def test_is_iterator_with_bytes(self):
        """测试 _is_iterator 对字节返回 False"""

        class TestHandler(BaseProtocolHandler):
            name = "test"

            def as_fastapi_router(self, agent_invoker):
                pass

        handler = TestHandler()
        assert handler._is_iterator(b"") is False
        assert handler._is_iterator(b"hello") is False

    def test_is_iterator_with_generator(self):
        """测试 _is_iterator 对生成器返回 True"""

        class TestHandler(BaseProtocolHandler):
            name = "test"

            def as_fastapi_router(self, agent_invoker):
                pass

        handler = TestHandler()

        def gen():
            yield 1

        assert handler._is_iterator(gen()) is True

    def test_is_iterator_with_async_generator(self):
        """测试 _is_iterator 对异步生成器返回 True"""

        class TestHandler(BaseProtocolHandler):
            name = "test"

            def as_fastapi_router(self, agent_invoker):
                pass

        handler = TestHandler()

        async def async_gen():
            yield 1

        assert handler._is_iterator(async_gen()) is True
