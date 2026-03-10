"""BrowserToolSet 错误处理单元测试 / BrowserToolSet Error Handling Unit Tests

测试 BrowserToolSet 的错误处理机制，确保工具级错误不会触发沙箱重建。
Tests BrowserToolSet error handling to ensure tool-level errors don't trigger sandbox recreation.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from agentrun.integration.builtin.sandbox import BrowserToolSet


class TestBrowserToolSetIsInfrastructureError:
    """测试 _is_infrastructure_error 方法"""

    @pytest.fixture
    def toolset(self):
        """创建 BrowserToolSet 实例（不初始化沙箱）"""
        with patch.object(BrowserToolSet, "__init__", lambda self: None):
            ts = BrowserToolSet()
            ts._playwright_sync = None
            ts.sandbox = None
            ts.sandbox_id = ""
            return ts

    def test_connection_closed_is_infrastructure_error(self, toolset):
        """测试连接关闭是基础设施错误"""
        assert toolset._is_infrastructure_error("Target closed") is True
        assert toolset._is_infrastructure_error("Connection closed") is True
        assert toolset._is_infrastructure_error("Browser closed") is True

    def test_protocol_error_is_infrastructure_error(self, toolset):
        """测试协议错误是基础设施错误"""
        assert (
            toolset._is_infrastructure_error("Protocol error: session closed")
            is True
        )
        assert (
            toolset._is_infrastructure_error("WebSocket disconnected") is True
        )

    def test_network_error_is_infrastructure_error(self, toolset):
        """测试网络错误是基础设施错误"""
        assert toolset._is_infrastructure_error("ECONNREFUSED") is True
        assert toolset._is_infrastructure_error("ECONNRESET") is True
        assert toolset._is_infrastructure_error("EPIPE") is True

    def test_js_error_is_not_infrastructure_error(self, toolset):
        """测试 JS 执行错误不是基础设施错误"""
        assert (
            toolset._is_infrastructure_error(
                "Evaluation failed: TypeError: Cannot read property"
                " 'textContent' of null"
            )
            is False
        )

    def test_element_not_found_is_not_infrastructure_error(self, toolset):
        """测试元素找不到错误不是基础设施错误"""
        assert (
            toolset._is_infrastructure_error(
                "Error: Timeout 30000ms exceeded while waiting for selector"
                " '.nonexistent'"
            )
            is False
        )

    def test_timeout_error_is_not_infrastructure_error(self, toolset):
        """测试超时错误不是基础设施错误"""
        assert (
            toolset._is_infrastructure_error(
                "Error: page.click: Timeout 5000ms exceeded."
            )
            is False
        )


class TestBrowserToolSetRunInSandbox:
    """测试 _run_in_sandbox 方法的错误处理"""

    @pytest.fixture
    def mock_sandbox(self):
        """创建模拟的沙箱"""
        return MagicMock()

    @pytest.fixture
    def toolset(self, mock_sandbox):
        """创建带有模拟沙箱的 BrowserToolSet 实例"""
        with patch.object(BrowserToolSet, "__init__", lambda self: None):
            ts = BrowserToolSet()
            ts._playwright_sync = None
            ts._playwright_thread = None
            ts.sandbox = mock_sandbox
            ts.sandbox_id = "test-sandbox-id"
            ts.lock = MagicMock()
            ts._reset_playwright = MagicMock()
            ts._ensure_sandbox = MagicMock(return_value=mock_sandbox)
            return ts

    def test_successful_callback_returns_result(self, toolset):
        """测试成功的回调返回结果"""

        def callback(sb):
            return {"success": True, "data": "test"}

        result = toolset._run_in_sandbox(callback)

        assert result == {"success": True, "data": "test"}
        assert toolset.sandbox is not None

    def test_tool_level_error_returns_error_without_rebuild(self, toolset):
        """测试工具级错误返回错误字典，不重建沙箱"""
        try:
            from playwright.sync_api import Error as PlaywrightError
        except ImportError:
            pytest.skip("Playwright not installed")

        original_sandbox = toolset.sandbox

        def callback(sb):
            raise PlaywrightError(
                "Evaluation failed: TypeError: Cannot read property"
            )

        result = toolset._run_in_sandbox(callback)

        assert "error" in result
        assert "Evaluation failed" in result["error"]
        assert toolset.sandbox is original_sandbox
        toolset._reset_playwright.assert_not_called()

    def test_infrastructure_error_triggers_rebuild(self, toolset, mock_sandbox):
        """测试基础设施错误触发沙箱重建"""
        try:
            from playwright.sync_api import Error as PlaywrightError
        except ImportError:
            pytest.skip("Playwright not installed")

        call_count = 0

        def callback(sb):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PlaywrightError("Target closed")
            return {"success": True}

        result = toolset._run_in_sandbox(callback)

        assert result == {"success": True}
        assert call_count == 2
        toolset._reset_playwright.assert_called_once()

    def test_connection_error_triggers_rebuild(self, toolset, mock_sandbox):
        """测试连接错误触发沙箱重建"""
        call_count = 0

        def callback(sb):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection refused")
            return {"success": True}

        result = toolset._run_in_sandbox(callback)

        assert result == {"success": True}
        assert call_count == 2
        toolset._reset_playwright.assert_called_once()

    def test_os_error_triggers_rebuild(self, toolset, mock_sandbox):
        """测试 OS 错误触发沙箱重建"""
        call_count = 0

        def callback(sb):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Broken pipe")
            return {"success": True}

        result = toolset._run_in_sandbox(callback)

        assert result == {"success": True}
        assert call_count == 2
        toolset._reset_playwright.assert_called_once()

    def test_unexpected_error_returns_error_without_rebuild(self, toolset):
        """测试未知异常返回错误，不重建沙箱"""
        original_sandbox = toolset.sandbox

        def callback(sb):
            raise ValueError("Some unexpected error")

        result = toolset._run_in_sandbox(callback)

        assert "error" in result
        assert "Some unexpected error" in result["error"]
        assert toolset.sandbox is original_sandbox
        toolset._reset_playwright.assert_not_called()


class TestBrowserToolSetPlaywrightCaching:
    """测试 Playwright 连接缓存机制"""

    @pytest.fixture
    def mock_sandbox(self):
        """创建模拟的沙箱"""
        sb = MagicMock()
        mock_playwright = MagicMock()
        sb.sync_playwright.return_value = mock_playwright
        return sb

    @pytest.fixture
    def toolset(self, mock_sandbox):
        """创建带有模拟沙箱的 BrowserToolSet 实例"""
        with patch.object(BrowserToolSet, "__init__", lambda self: None):
            ts = BrowserToolSet()
            ts._playwright_sync = None
            ts._playwright_thread = None
            ts.sandbox = mock_sandbox
            ts.sandbox_id = "test-sandbox-id"
            ts.lock = threading.Lock()
            return ts

    def test_get_playwright_creates_connection_once(
        self, toolset, mock_sandbox
    ):
        """测试 _get_playwright 只创建一次连接"""
        p1 = toolset._get_playwright(mock_sandbox)
        p2 = toolset._get_playwright(mock_sandbox)

        assert p1 is p2
        mock_sandbox.sync_playwright.assert_called_once()
        p1.open.assert_called_once()

    def test_reset_playwright_clears_connection(self, toolset, mock_sandbox):
        """测试 _reset_playwright 清理连接"""
        p = toolset._get_playwright(mock_sandbox)

        toolset._reset_playwright()

        assert toolset._playwright_sync is None
        p.close.assert_called_once()

    def test_reset_playwright_handles_close_error(self, toolset, mock_sandbox):
        """测试 _reset_playwright 处理关闭错误"""
        p = toolset._get_playwright(mock_sandbox)
        p.close.side_effect = Exception("Close failed")

        toolset._reset_playwright()

        assert toolset._playwright_sync is None

    def test_concurrent_get_playwright_creates_only_one_connection(
        self, toolset, mock_sandbox
    ):
        """测试并发调用 _get_playwright 只创建一个连接，不会泄漏

        所有工作线程在同一 executor 内并发运行（即创建线程仍存活），
        应复用同一连接，不会触发重建。
        """
        start_barrier = threading.Barrier(5)
        # Keep all threads alive until every thread has obtained playwright,
        # simulating concurrent workers within the same executor context.
        hold_barrier = threading.Barrier(5)
        results: list = []

        def worker():
            start_barrier.wait()
            p = toolset._get_playwright(mock_sandbox)
            results.append(p)
            hold_barrier.wait()  # stay alive so is_alive() == True for peers

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(p is results[0] for p in results)
        mock_sandbox.sync_playwright.assert_called_once()


class TestBrowserToolSetClose:
    """测试 close 方法"""

    @pytest.fixture
    def mock_sandbox(self):
        """创建模拟的沙箱"""
        return MagicMock()

    @pytest.fixture
    def toolset(self, mock_sandbox):
        """创建带有模拟沙箱的 BrowserToolSet 实例"""
        with patch.object(BrowserToolSet, "__init__", lambda self: None):
            ts = BrowserToolSet()
            ts._playwright_sync = MagicMock()
            ts._playwright_thread = threading.current_thread()
            ts.sandbox = mock_sandbox
            ts.sandbox_id = "test-sandbox-id"
            ts.lock = threading.Lock()
            return ts

    def test_close_cleans_up_playwright_and_sandbox(
        self, toolset, mock_sandbox
    ):
        """测试 close 清理 Playwright 和沙箱"""
        playwright_mock = toolset._playwright_sync

        toolset.close()

        playwright_mock.close.assert_called_once()
        assert toolset._playwright_sync is None
        mock_sandbox.stop.assert_called_once()
        assert toolset.sandbox is None
        assert toolset.sandbox_id == ""


class TestBrowserToolSetThreadAwareness:
    """测试 _get_playwright 的线程感知行为 / Tests for thread-aware Playwright caching"""

    @pytest.fixture
    def mock_sandbox(self):
        """创建模拟的沙箱"""
        sb = MagicMock()
        sb.sync_playwright.return_value = MagicMock()
        return sb

    @pytest.fixture
    def toolset(self, mock_sandbox):
        """创建带有模拟沙箱的 BrowserToolSet 实例"""
        with patch.object(BrowserToolSet, "__init__", lambda self: None):
            ts = BrowserToolSet()
            ts._playwright_sync = None
            ts._playwright_thread = None
            ts.sandbox = mock_sandbox
            ts.sandbox_id = "test-sandbox-id"
            ts.lock = threading.Lock()
            return ts

    def test_get_playwright_records_creating_thread(self, toolset, mock_sandbox):
        """测试 _get_playwright 记录创建连接的线程"""
        toolset._get_playwright(mock_sandbox)

        assert toolset._playwright_thread is threading.current_thread()

    def test_get_playwright_same_thread_reuses_connection(
        self, toolset, mock_sandbox
    ):
        """测试同一线程多次调用复用连接"""
        p1 = toolset._get_playwright(mock_sandbox)
        p2 = toolset._get_playwright(mock_sandbox)

        assert p1 is p2
        mock_sandbox.sync_playwright.assert_called_once()

    def test_get_playwright_dead_thread_recreates_connection(
        self, toolset, mock_sandbox
    ):
        """测试创建线程退出后重建 Playwright 连接（Bug 1 修复）

        模拟 LangGraph ToolNode 的行为：每次工具调用在不同的线程上执行。
        当创建连接的工作线程退出后，缓存的 Playwright 实例必须重建，
        因为 Playwright 内部 greenlet 绑定到创建它的线程。
        """
        first_instance: list = []
        second_instance: list = []

        def first_call():
            p = toolset._get_playwright(mock_sandbox)
            first_instance.append(p)

        t1 = threading.Thread(target=first_call)
        t1.start()
        t1.join()
        # t1 has now exited — its greenlet binding is dead

        def second_call():
            p = toolset._get_playwright(mock_sandbox)
            second_instance.append(p)

        t2 = threading.Thread(target=second_call)
        t2.start()
        t2.join()

        assert len(first_instance) == 1
        assert len(second_instance) == 1
        # A new connection must have been created for the second call
        assert mock_sandbox.sync_playwright.call_count == 2

    def test_get_playwright_live_thread_not_recreated(
        self, toolset, mock_sandbox
    ):
        """测试创建线程仍存活时不重建连接（并发安全）

        即使在不同线程中调用，只要创建线程仍然存活，就应复用同一连接。
        这模拟同一 executor 内并发工具调用的场景。
        """
        results: list = []

        # Create connection in main thread first
        toolset._get_playwright(mock_sandbox)
        # The creating thread (main test thread) is still alive

        # Another thread should reuse the same connection
        def worker():
            p = toolset._get_playwright(mock_sandbox)
            results.append(p)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert len(results) == 1
        assert results[0] is toolset._playwright_sync
        mock_sandbox.sync_playwright.assert_called_once()

    def test_reset_playwright_clears_thread(self, toolset, mock_sandbox):
        """测试 _reset_playwright 清理线程引用"""
        toolset._get_playwright(mock_sandbox)
        assert toolset._playwright_thread is not None

        toolset._reset_playwright()

        assert toolset._playwright_thread is None
        assert toolset._playwright_sync is None


class TestBrowserToolSetGreenletErrorHandling:
    """测试 _run_in_sandbox 对 greenlet 死亡错误的处理（Bug 3 修复）"""

    @pytest.fixture
    def mock_sandbox(self):
        """创建模拟的沙箱"""
        return MagicMock()

    @pytest.fixture
    def toolset(self, mock_sandbox):
        """创建带有模拟沙箱的 BrowserToolSet 实例"""
        with patch.object(BrowserToolSet, "__init__", lambda self: None):
            ts = BrowserToolSet()
            ts._playwright_sync = None
            ts._playwright_thread = None
            ts.sandbox = mock_sandbox
            ts.sandbox_id = "test-sandbox-id"
            ts.lock = MagicMock()
            ts._reset_playwright = MagicMock()
            ts._ensure_sandbox = MagicMock(return_value=mock_sandbox)
            return ts

    def test_greenlet_thread_error_resets_playwright_and_sandbox(
        self, toolset, mock_sandbox
    ):
        """测试 greenlet 线程绑定错误触发 Playwright 和沙箱重置

        当 'cannot switch to a different thread' 错误发生时，
        必须重置缓存的 Playwright 实例，避免后续调用持续失败。
        """

        def callback(sb):
            raise Exception(
                "cannot switch to a different thread (which happens to have"
                " exited)"
            )

        result = toolset._run_in_sandbox(callback)

        assert "error" in result
        assert "cannot switch to" in result["error"]
        toolset._reset_playwright.assert_called_once()
        assert toolset.sandbox is None

    def test_non_greenlet_unexpected_error_does_not_reset(
        self, toolset, mock_sandbox
    ):
        """测试普通未知错误不触发 Playwright 重置"""
        original_sandbox = toolset.sandbox

        def callback(sb):
            raise ValueError("Some other unexpected error")

        result = toolset._run_in_sandbox(callback)

        assert "error" in result
        toolset._reset_playwright.assert_not_called()
        assert toolset.sandbox is original_sandbox
