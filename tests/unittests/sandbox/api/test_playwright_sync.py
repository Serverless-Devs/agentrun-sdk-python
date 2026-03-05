"""Tests for agentrun.sandbox.api.playwright_sync module."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_playwright():
    with patch(
        "agentrun.sandbox.api.playwright_sync.sync_playwright"
    ) as mock_sp:
        mock_pw_instance = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_page.context = mock_context
        mock_page.goto.return_value = None
        mock_page.content.return_value = "<html></html>"
        mock_page.screenshot.return_value = b"image"
        mock_page.title.return_value = "Title"
        mock_page.evaluate.return_value = "result"

        mock_context.pages = [mock_page]
        mock_context.new_page.return_value = mock_page

        mock_browser.contexts = [mock_context]
        mock_browser.new_context.return_value = mock_context

        mock_pw_instance.chromium = MagicMock()
        mock_pw_instance.chromium.connect_over_cdp.return_value = mock_browser

        mock_cm = MagicMock()
        mock_cm.start.return_value = mock_pw_instance
        mock_sp.return_value = mock_cm

        yield {
            "sp": mock_sp,
            "pw_instance": mock_pw_instance,
            "browser": mock_browser,
            "context": mock_context,
            "page": mock_page,
        }


@pytest.fixture
def pw(mock_playwright):
    from agentrun.sandbox.api.playwright_sync import BrowserPlaywrightSync

    return BrowserPlaywrightSync(
        url="ws://example.com/ws/automation",
        browser_type="chrome",
        headers={"Authorization": "Bearer tok"},
    )


class TestInit:

    def test_constructor(self, mock_playwright):
        from agentrun.sandbox.api.playwright_sync import BrowserPlaywrightSync

        obj = BrowserPlaywrightSync(
            "ws://test", browser_type="firefox", headers={"x": "1"}
        )
        assert obj.url == "ws://test"
        assert obj.browser_type == "firefox"
        assert obj.auto_close_browser is False
        assert obj.auto_close_page is False
        assert obj._browser is None
        assert obj._page is None


class TestOpenClose:

    def test_open(self, pw, mock_playwright):
        result = pw.open()
        assert result is pw
        assert pw._browser is mock_playwright["browser"]

    def test_open_already_connected(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        result = pw.open()
        assert result is pw
        mock_playwright[
            "pw_instance"
        ].chromium.connect_over_cdp.assert_not_called()

    def test_close_with_auto_close(self, pw, mock_playwright):
        pw.auto_close_page = True
        pw.auto_close_browser = True
        pw._page = mock_playwright["page"]
        pw._browser = mock_playwright["browser"]
        pw._playwright_instance = mock_playwright["pw_instance"]

        pw.close()
        mock_playwright["page"].close.assert_called_once()
        mock_playwright["browser"].close.assert_called_once()
        mock_playwright["pw_instance"].stop.assert_called_once()

    def test_close_without_auto_close(self, pw, mock_playwright):
        pw._playwright_instance = mock_playwright["pw_instance"]
        pw.close()
        mock_playwright["pw_instance"].stop.assert_called_once()

    def test_close_no_playwright_instance(self, pw):
        pw.close()


class TestContextManager:

    def test_enter(self, pw, mock_playwright):
        result = pw.__enter__()
        assert result is pw

    def test_exit(self, pw, mock_playwright):
        pw._playwright_instance = mock_playwright["pw_instance"]
        pw.__exit__(None, None, None)
        mock_playwright["pw_instance"].stop.assert_called_once()


class TestEnsure:

    def test_ensure_browser_cached(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        result = pw.ensure_browser()
        assert result is mock_playwright["browser"]

    def test_ensure_browser_opens(self, pw, mock_playwright):
        result = pw.ensure_browser()
        assert result is mock_playwright["browser"]

    def test_ensure_context_cached(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        result = pw.ensure_context()
        assert result is mock_playwright["context"]

    def test_ensure_context_from_browser(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        result = pw.ensure_context()
        assert result is mock_playwright["context"]

    def test_ensure_context_no_contexts(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        mock_playwright["browser"].contexts = []
        result = pw.ensure_context()
        mock_playwright["browser"].new_context.assert_called_once()

    def test_ensure_page_cached(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        result = pw.ensure_page()
        assert result is mock_playwright["page"]
        mock_playwright["page"].bring_to_front.assert_called()

    def test_ensure_page_from_context(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        result = pw.ensure_page()
        assert result is mock_playwright["page"]

    def test_ensure_page_no_pages(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        mock_playwright["context"].pages = []
        result = pw.ensure_page()
        mock_playwright["context"].new_page.assert_called_once()


class TestPageOps:

    def test_use_page(self, pw, mock_playwright):
        page = mock_playwright["page"]
        result = pw._use_page(page)
        assert result is page
        assert pw._page is page

    def test_list_pages(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pages = pw.list_pages()
        assert len(pages) == 1

    def test_new_page(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        result = pw.new_page()
        assert result is mock_playwright["page"]

    def test_select_tab_valid(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        result = pw.select_tab(0)
        assert result is mock_playwright["page"]

    def test_select_tab_invalid(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        with pytest.raises(IndexError, match="Tab index out of range"):
            pw.select_tab(99)


class TestNavigationAndActions:

    def _setup(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]

    def test_goto(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        pw.goto("https://example.com")
        mock_playwright["page"].goto.assert_called_once()

    def test_click(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        pw.click("#btn")
        mock_playwright["page"].click.assert_called_once()

    def test_drag_and_drop(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        pw.drag_and_drop("#src", "#dst")
        mock_playwright["page"].drag_and_drop.assert_called_once()

    def test_dblclick(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        pw.dblclick("#btn")
        mock_playwright["page"].dblclick.assert_called_once()

    def test_fill(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        pw.fill("#input", "text")
        mock_playwright["page"].fill.assert_called_once()

    def test_hover(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        pw.hover("#elem")
        mock_playwright["page"].hover.assert_called_once()

    def test_type(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        pw.type("#input", "text")
        mock_playwright["page"].type.assert_called_once()

    def test_go_forward(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        pw.go_forward()
        mock_playwright["page"].go_forward.assert_called_once()

    def test_go_back(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        pw.go_back()
        mock_playwright["page"].go_back.assert_called_once()

    def test_evaluate(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        result = pw.evaluate("1+1")
        assert result == "result"

    def test_wait(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        pw.wait(1000)
        mock_playwright["page"].wait_for_timeout.assert_called_once()

    def test_html_content(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        result = pw.html_content()
        assert result == "<html></html>"

    def test_screenshot(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        result = pw.screenshot()
        assert result == b"image"

    def test_title(self, pw, mock_playwright):
        self._setup(pw, mock_playwright)
        result = pw.title()
        assert result == "Title"
