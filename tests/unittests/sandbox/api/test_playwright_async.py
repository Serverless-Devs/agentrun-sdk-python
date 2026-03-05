"""Tests for agentrun.sandbox.api.playwright_async module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_playwright():
    with patch(
        "agentrun.sandbox.api.playwright_async.async_playwright"
    ) as mock_ap:
        mock_pw_instance = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_page.bring_to_front = AsyncMock()
        mock_page.context = mock_context
        mock_page.goto = AsyncMock(return_value=None)
        mock_page.click = AsyncMock()
        mock_page.dblclick = AsyncMock()
        mock_page.drag_and_drop = AsyncMock()
        mock_page.fill = AsyncMock()
        mock_page.hover = AsyncMock()
        mock_page.type = AsyncMock()
        mock_page.go_forward = AsyncMock(return_value=None)
        mock_page.go_back = AsyncMock(return_value=None)
        mock_page.evaluate = AsyncMock(return_value="result")
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.content = AsyncMock(return_value="<html></html>")
        mock_page.screenshot = AsyncMock(return_value=b"image")
        mock_page.title = AsyncMock(return_value="Title")
        mock_page.close = AsyncMock()

        mock_context.pages = [mock_page]
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_browser.contexts = [mock_context]
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()

        mock_pw_instance.chromium = MagicMock()
        mock_pw_instance.chromium.connect_over_cdp = AsyncMock(
            return_value=mock_browser
        )
        mock_pw_instance.stop = AsyncMock()

        mock_cm = MagicMock()
        mock_cm.start = AsyncMock(return_value=mock_pw_instance)
        mock_ap.return_value = mock_cm

        yield {
            "ap": mock_ap,
            "pw_instance": mock_pw_instance,
            "browser": mock_browser,
            "context": mock_context,
            "page": mock_page,
        }


@pytest.fixture
def pw(mock_playwright):
    from agentrun.sandbox.api.playwright_async import BrowserPlaywrightAsync

    return BrowserPlaywrightAsync(
        url="ws://example.com/ws/automation",
        browser_type="chrome",
        headers={"Authorization": "Bearer tok"},
    )


class TestInit:

    def test_constructor(self, mock_playwright):
        from agentrun.sandbox.api.playwright_async import BrowserPlaywrightAsync

        obj = BrowserPlaywrightAsync(
            "ws://test", browser_type="firefox", headers={"x": "1"}
        )
        assert obj.url == "ws://test"
        assert obj.browser_type == "firefox"
        assert obj.auto_close_browser is False
        assert obj.auto_close_page is False
        assert obj._browser is None
        assert obj._page is None


class TestOpenClose:

    @pytest.mark.asyncio
    async def test_open(self, pw, mock_playwright):
        result = await pw.open()
        assert result is pw
        assert pw._browser is mock_playwright["browser"]

    @pytest.mark.asyncio
    async def test_open_already_connected(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        result = await pw.open()
        assert result is pw
        mock_playwright[
            "pw_instance"
        ].chromium.connect_over_cdp.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_with_auto_close(self, pw, mock_playwright):
        pw.auto_close_page = True
        pw.auto_close_browser = True
        pw._page = mock_playwright["page"]
        pw._browser = mock_playwright["browser"]
        pw._playwright_instance = mock_playwright["pw_instance"]

        await pw.close()
        mock_playwright["page"].close.assert_awaited_once()
        mock_playwright["browser"].close.assert_awaited_once()
        mock_playwright["pw_instance"].stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_without_auto_close(self, pw, mock_playwright):
        pw._playwright_instance = mock_playwright["pw_instance"]
        await pw.close()
        mock_playwright["pw_instance"].stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_no_playwright_instance(self, pw):
        await pw.close()


class TestContextManager:

    @pytest.mark.asyncio
    async def test_aenter(self, pw, mock_playwright):
        result = await pw.__aenter__()
        assert result is pw

    @pytest.mark.asyncio
    async def test_aexit(self, pw, mock_playwright):
        pw._playwright_instance = mock_playwright["pw_instance"]
        await pw.__aexit__(None, None, None)
        mock_playwright["pw_instance"].stop.assert_awaited_once()


class TestEnsure:

    @pytest.mark.asyncio
    async def test_ensure_browser_cached(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        result = await pw.ensure_browser()
        assert result is mock_playwright["browser"]

    @pytest.mark.asyncio
    async def test_ensure_browser_opens(self, pw, mock_playwright):
        result = await pw.ensure_browser()
        assert result is mock_playwright["browser"]

    @pytest.mark.asyncio
    async def test_ensure_context_cached(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        result = await pw.ensure_context()
        assert result is mock_playwright["context"]

    @pytest.mark.asyncio
    async def test_ensure_context_from_browser(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        result = await pw.ensure_context()
        assert result is mock_playwright["context"]

    @pytest.mark.asyncio
    async def test_ensure_context_no_contexts(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        mock_playwright["browser"].contexts = []
        result = await pw.ensure_context()
        mock_playwright["browser"].new_context.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ensure_page_cached(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        result = await pw.ensure_page()
        assert result is mock_playwright["page"]
        mock_playwright["page"].bring_to_front.assert_awaited()

    @pytest.mark.asyncio
    async def test_ensure_page_from_context(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        result = await pw.ensure_page()
        assert result is mock_playwright["page"]

    @pytest.mark.asyncio
    async def test_ensure_page_no_pages(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        mock_playwright["context"].pages = []
        result = await pw.ensure_page()
        mock_playwright["context"].new_page.assert_awaited_once()


class TestPageOps:

    @pytest.mark.asyncio
    async def test_use_page(self, pw, mock_playwright):
        page = mock_playwright["page"]
        result = await pw._use_page(page)
        assert result is page
        assert pw._page is page

    @pytest.mark.asyncio
    async def test_list_pages(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pages = await pw.list_pages()
        assert len(pages) == 1

    @pytest.mark.asyncio
    async def test_new_page(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        result = await pw.new_page()
        assert result is mock_playwright["page"]

    @pytest.mark.asyncio
    async def test_select_tab_valid(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        result = await pw.select_tab(0)
        assert result is mock_playwright["page"]

    @pytest.mark.asyncio
    async def test_select_tab_invalid(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        with pytest.raises(IndexError, match="Tab index out of range"):
            await pw.select_tab(99)


class TestNavigationAndActions:

    @pytest.mark.asyncio
    async def test_goto(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        await pw.goto("https://example.com")
        mock_playwright["page"].goto.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_click(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        await pw.click("#btn")
        mock_playwright["page"].click.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_drag_and_drop(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        await pw.drag_and_drop("#src", "#dst")
        mock_playwright["page"].drag_and_drop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dblclick(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        await pw.dblclick("#btn")
        mock_playwright["page"].dblclick.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fill(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        await pw.fill("#input", "text")
        mock_playwright["page"].fill.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hover(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        await pw.hover("#elem")
        mock_playwright["page"].hover.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_type(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        await pw.type("#input", "text")
        mock_playwright["page"].type.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_go_forward(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        await pw.go_forward()
        mock_playwright["page"].go_forward.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_go_back(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        await pw.go_back()
        mock_playwright["page"].go_back.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_evaluate(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        result = await pw.evaluate("1+1")
        assert result == "result"

    @pytest.mark.asyncio
    async def test_wait(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        await pw.wait(1000)
        mock_playwright["page"].wait_for_timeout.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_html_content(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        result = await pw.html_content()
        assert result == "<html></html>"

    @pytest.mark.asyncio
    async def test_screenshot(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        result = await pw.screenshot()
        assert result == b"image"

    @pytest.mark.asyncio
    async def test_title(self, pw, mock_playwright):
        pw._browser = mock_playwright["browser"]
        pw._context = mock_playwright["context"]
        pw._page = mock_playwright["page"]
        result = await pw.title()
        assert result == "Title"
