"""Playwright异步API封装 / Playwright Async API Wrapper

提供Playwright的异步API封装,用于浏览器沙箱操作。
Provides async API wrapper for Playwright, used for browser sandbox operations.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    FloatRect,
    Locator,
    Page,
    Playwright,
    Position,
    Response,
)


class BrowserPlaywrightAsync:
    """A small helper wrapper around Playwright's Async API.

    This class connects to an existing Chromium instance over CDP and exposes a set
    of common page operations. Internally it lazily ensures a `Browser`,
    `BrowserContext`, and `Page` and keeps references to reuse them across calls.

    Notes
    -----
    - Connection is established via CDP using the given `url`.
    - If `auto_close_browser`/`auto_close_page` are enabled, `close()` will attempt
      to close the browser/page respectively.
    - Methods that act on the page automatically bring the page to the front.
    """

    def __init__(
        self,
        url: str,
        browser_type: str = "chrome",
        auto_close_browser: bool = False,
        auto_close_page: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.url = url
        self.browser_type = browser_type
        self.auto_close_browser = auto_close_browser
        self.auto_close_page = auto_close_page
        self.headers = headers

        self._playwright = async_playwright()
        self._playwright_instance: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    async def open(self) -> "BrowserPlaywrightAsync":
        """Establish a connection to the remote browser if not already connected.

        Returns
        -------
        BrowserPlaywrightAsync
            The current instance for fluent chaining.
        """
        if self._browser:
            return self

        self._playwright_instance = await self._playwright.start()
        self._browser = (
            await self._playwright_instance.chromium.connect_over_cdp(
                self.url, headers=self.headers
            )
        )

        return self

    async def close(self) -> None:
        """Close held resources according to the auto-close flags.

        Closes page and/or browser if the corresponding auto-close flags are set
        and stops the Playwright driver when present.
        """
        if self.auto_close_page and self._page:
            await self._page.close()

        if self.auto_close_browser and self._browser:
            await self._browser.close()

        if self._playwright_instance:
            await self._playwright_instance.stop()

    async def __aenter__(self) -> "BrowserPlaywrightAsync":
        """Enter context by ensuring the connection is open.

        Returns
        -------
        BrowserPlaywrightAsync
            The current instance.
        """
        return await self.open()

    async def __aexit__(self, *args: Any) -> None:
        """Exit context by closing resources based on auto-close flags."""
        await self.close()

    async def ensure_browser(self) -> Browser:
        """Ensure a `Browser` instance is available.

        Returns
        -------
        Browser
            A connected Playwright `Browser` instance.
        """
        if self._browser:
            return self._browser

        await self.open()

        assert self._browser is not None

        return self._browser

    async def ensure_context(self) -> BrowserContext:
        """Ensure a `BrowserContext` is available, creating one if necessary.

        Returns
        -------
        BrowserContext
            The ensured `BrowserContext`.
        """
        browser = await self.ensure_browser()

        if self._context:
            return self._context

        if len(browser.contexts) > 0:
            self._context = browser.contexts[0]
        else:
            self._context = await browser.new_context()

        return self._context

    async def ensure_page(self) -> Page:
        """Ensure a `Page` is available in the current context.

        Returns
        -------
        Page
            The ensured `Page` which is brought to the front.
        """
        ctx = await self.ensure_context()

        if self._page:
            await self._page.bring_to_front()
            return self._page

        if len(ctx.pages) > 0:
            self._page = ctx.pages[0]
        else:
            self._page = await ctx.new_page()

        await self._page.bring_to_front()
        return self._page

    async def _use_page(self, page: Page) -> Page:
        """Set the active page and context.

        Parameters
        ----------
        page : Page
            The page to make active.

        Returns
        -------
        Page
            The provided page.
        """
        await page.bring_to_front()
        self._page = page
        self._context = page.context
        return self._page

    async def list_pages(self) -> List[Page]:
        """List all pages across all contexts in the connected browser."""
        pages: List[Page] = []

        browser = await self.ensure_browser()
        for context in browser.contexts:
            for page in context.pages:
                pages.append(page)

        return pages

    async def new_page(self) -> Page:
        """Create and switch to a new page in the ensured context."""
        context = await self.ensure_context()
        page = await context.new_page()
        return await self._use_page(page)

    async def select_tab(self, index: int) -> Page:
        """Select a page by index across all open tabs.

        Parameters
        ----------
        index : int
            Zero-based page index.

        Returns
        -------
        Page
            The selected page.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        pages = await self.list_pages()
        if 0 <= index < len(pages):
            return await self._use_page(pages[index])
        else:
            raise IndexError("Tab index out of range")

    async def goto(
        self,
        url: str,
        timeout: Optional[float] = None,
        wait_until: Optional[
            Literal["commit", "domcontentloaded", "load", "networkidle"]
        ] = None,
        referer: Optional[str] = None,
    ) -> Optional[Response]:
        """Navigate to a URL on the active page.

        Returns
        -------
        Optional[Response]
            The main resource response if available; otherwise `None`.
        """
        page = await self.ensure_page()
        return await page.goto(
            url, timeout=timeout, wait_until=wait_until, referer=referer
        )

    async def click(
        self,
        selector: str,
        modifiers: Optional[
            Sequence[
                Literal["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]
            ]
        ] = None,
        position: Optional[Position] = None,
        delay: Optional[float] = None,
        button: Optional[Literal["left", "middle", "right"]] = None,
        click_count: Optional[int] = None,
        timeout: Optional[float] = None,
        force: Optional[bool] = None,
        no_wait_after: Optional[bool] = None,
        trial: Optional[bool] = None,
        strict: Optional[bool] = None,
    ) -> None:
        """Click an element matching the selector on the active page."""
        page = await self.ensure_page()
        return await page.click(
            selector,
            modifiers=modifiers,
            position=position,
            delay=delay,
            button=button,
            click_count=click_count,
            timeout=timeout,
            force=force,
            no_wait_after=no_wait_after,
            trial=trial,
            strict=strict,
        )

    async def drag_and_drop(
        self,
        source: str,
        target: str,
        source_position: Optional[Position] = None,
        target_position: Optional[Position] = None,
        force: Optional[bool] = None,
        no_wait_after: Optional[bool] = None,
        timeout: Optional[float] = None,
        strict: Optional[bool] = None,
        trial: Optional[bool] = None,
    ) -> None:
        """Alias for `drag` using Playwright's `drag_and_drop` under the hood."""
        page = await self.ensure_page()
        return await page.drag_and_drop(
            source=source,
            target=target,
            source_position=source_position,
            target_position=target_position,
            force=force,
            no_wait_after=no_wait_after,
            timeout=timeout,
            strict=strict,
            trial=trial,
        )

    async def dblclick(
        self,
        selector: str,
        modifiers: Optional[
            Sequence[
                Literal["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]
            ]
        ] = None,
        position: Optional[Position] = None,
        delay: Optional[float] = None,
        button: Optional[Literal["left", "middle", "right"]] = None,
        timeout: Optional[float] = None,
        force: Optional[bool] = None,
        no_wait_after: Optional[bool] = None,
        strict: Optional[bool] = None,
        trial: Optional[bool] = None,
    ) -> None:
        """Double-click an element matching the selector on the active page."""
        page = await self.ensure_page()
        return await page.dblclick(
            selector=selector,
            modifiers=modifiers,
            position=position,
            delay=delay,
            button=button,
            timeout=timeout,
            force=force,
            no_wait_after=no_wait_after,
            strict=strict,
            trial=trial,
        )

    async def fill(
        self,
        selector: str,
        value: str,
        timeout: Optional[float] = None,
        no_wait_after: Optional[bool] = None,
        strict: Optional[bool] = None,
        force: Optional[bool] = None,
    ) -> None:
        """Fill an input/textarea matched by selector with the provided value."""
        page = await self.ensure_page()
        return await page.fill(
            selector=selector,
            value=value,
            timeout=timeout,
            no_wait_after=no_wait_after,
            strict=strict,
            force=force,
        )

    async def hover(
        self,
        selector: str,
        modifiers: Optional[
            Sequence[
                Literal["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]
            ]
        ] = None,
        position: Optional[Position] = None,
        timeout: Optional[float] = None,
        no_wait_after: Optional[bool] = None,
        force: Optional[bool] = None,
        strict: Optional[bool] = None,
        trial: Optional[bool] = None,
    ) -> None:
        """Hover over the element matched by the selector."""
        page = await self.ensure_page()
        return await page.hover(
            selector=selector,
            modifiers=modifiers,
            position=position,
            timeout=timeout,
            no_wait_after=no_wait_after,
            force=force,
            strict=strict,
            trial=trial,
        )

    async def type(
        self,
        selector: str,
        text: str,
        delay: Optional[float] = None,
        timeout: Optional[float] = None,
        no_wait_after: Optional[bool] = None,
        strict: Optional[bool] = None,
    ) -> None:
        """Type text into an element matched by the selector."""
        page = await self.ensure_page()
        return await page.type(
            selector=selector,
            text=text,
            delay=delay,
            timeout=timeout,
            no_wait_after=no_wait_after,
            strict=strict,
        )

    async def go_forward(
        self,
        timeout: Optional[float] = None,
        wait_until: Optional[
            Literal["commit", "domcontentloaded", "load", "networkidle"]
        ] = None,
    ) -> Optional[Response]:
        """Go forward in the page history if possible."""
        page = await self.ensure_page()
        return await page.go_forward(timeout=timeout, wait_until=wait_until)

    async def go_back(
        self,
        timeout: Optional[float] = None,
        wait_until: Optional[
            Literal["commit", "domcontentloaded", "load", "networkidle"]
        ] = None,
    ) -> Optional[Response]:
        """Go back in the page history if possible."""
        page = await self.ensure_page()
        return await page.go_back(timeout=timeout, wait_until=wait_until)

    async def evaluate(
        self,
        expression: str,
        arg: Optional[Any] = None,
    ) -> Any:
        """Evaluate a JavaScript expression in the page context."""

        page = await self.ensure_page()
        return await page.evaluate(expression=expression, arg=arg)

    async def wait(self, timeout: float) -> None:
        """Wait for the given timeout in milliseconds."""
        page = await self.ensure_page()
        return await page.wait_for_timeout(timeout=timeout)

    async def html_content(
        self,
    ) -> str:
        """Get the current page's HTML content as a string."""
        page = await self.ensure_page()
        return await page.content()

    async def screenshot(
        self,
        timeout: Optional[float] = None,
        type: Optional[Literal["jpeg", "png"]] = None,
        path: Union[Path, str, None] = None,
        quality: Optional[int] = None,
        omit_background: Optional[bool] = None,
        full_page: Optional[bool] = None,
        clip: Optional[FloatRect] = None,
        animations: Optional[Literal["allow", "disabled"]] = None,
        caret: Optional[Literal["hide", "initial"]] = None,
        scale: Optional[Literal["css", "device"]] = None,
        mask: Optional[Sequence[Locator]] = None,
        mask_color: Optional[str] = None,
        style: Optional[str] = None,
    ) -> bytes:
        """Capture a screenshot of the page.

        Returns
        -------
        bytes
            The image bytes of the screenshot.
        """
        page = await self.ensure_page()
        return await page.screenshot(
            timeout=timeout,
            type=type,
            path=path,
            quality=quality,
            omit_background=omit_background,
            full_page=full_page,
            clip=clip,
            animations=animations,
            caret=caret,
            scale=scale,
            mask=mask,
            mask_color=mask_color,
            style=style,
        )

    async def title(self) -> str:
        page = await self.ensure_page()
        return await page.title()
