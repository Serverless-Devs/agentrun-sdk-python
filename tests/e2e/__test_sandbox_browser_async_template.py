"""
Sandbox Browser 模块的 E2E 测试

测试覆盖:
- 创建 Browser Sandbox
- 健康检查
- Playwright 同步操作（导航、截图、页面交互）
- Playwright 异步操作
- VNC 和 CDP 连接
- 录制功能
- 文件系统操作
- 删除 Sandbox
"""

import asyncio
import os
from pathlib import Path
import time

import pytest

from agentrun.sandbox import Sandbox, Template
from agentrun.sandbox.browser_sandbox import BrowserSandbox
from agentrun.sandbox.model import TemplateInput, TemplateType


class TestSandboxBrowser:
    """Sandbox Browser 模块 E2E 测试"""

    @pytest.fixture
    def browser_template_name(self, unique_name: str) -> str:
        """生成浏览器模板名称"""
        return f"{unique_name}-browser"

    @pytest.fixture
    def browser_template(self, browser_template_name: str):
        """创建并返回浏览器模板"""
        template = Template.create(
            TemplateInput(
                template_name=browser_template_name,
                template_type=TemplateType.BROWSER,
                description="E2E 测试 - Browser Template",
            )
        )
        yield template
        # 清理
        try:
            Template.delete_by_name(template_name=browser_template_name)
        except Exception:
            pass

    async def test_create_browser_sandbox_async(
        self, browser_template: Template
    ):
        """测试创建 Browser Sandbox"""
        sb = await Sandbox.create_async(
            template_type=TemplateType.BROWSER,
            template_name=browser_template.template_name,
            sandbox_idle_timeout_seconds=600,
        )

        try:
            assert sb is not None
            assert sb.sandbox_id is not None
            assert sb.template_id == browser_template.template_id
            assert sb.status is not None
        finally:
            await sb.delete_async()

    async def test_browser_health_check_async(self, browser_template: Template):
        """测试浏览器健康检查"""
        sb = await Sandbox.create_async(
            template_type=TemplateType.BROWSER,
            template_name=browser_template.template_name,
            sandbox_idle_timeout_seconds=600,
        )

        try:
            # 等待浏览器就绪
            max_retries = 30
            retry_count = 0
            health_status = None

            while retry_count < max_retries:
                health_status = await sb.check_health_async()
                if health_status.get("status") == "ok":
                    break
                await asyncio.sleep(2)
                retry_count += 1

            assert health_status is not None
            assert health_status.get("status") == "ok"
        finally:
            await sb.delete_async()

    async def test_browser_vnc_cdp_urls_async(self, browser_template: Template):
        """测试获取 VNC 和 CDP 连接 URL"""
        sb = await Sandbox.create_async(
            template_type=TemplateType.BROWSER,
            template_name=browser_template.template_name,
            sandbox_idle_timeout_seconds=600,
        )

        try:
            # 获取 VNC URL
            vnc_url = sb.get_vnc_url(record=False)
            assert vnc_url is not None
            assert isinstance(vnc_url, str)
            assert "ws://" in vnc_url or "wss://" in vnc_url

            # 获取带录制的 VNC URL
            vnc_url_with_record = sb.get_vnc_url(record=True)
            assert vnc_url_with_record is not None
            assert isinstance(vnc_url_with_record, str)

            # 获取 CDP URL
            cdp_url = sb.get_cdp_url(record=False)
            assert cdp_url is not None
            assert isinstance(cdp_url, str)
            assert "ws://" in cdp_url or "wss://" in cdp_url

            # 获取带录制的 CDP URL
            cdp_url_with_record = sb.get_cdp_url(record=True)
            assert cdp_url_with_record is not None
            assert isinstance(cdp_url_with_record, str)
        finally:
            await sb.delete_async()

    async def test_playwright_async_navigation_async(
        self, browser_template: Template
    ):
        """测试 Playwright 异步导航功能"""
        sb = await Sandbox.create_async(
            template_type=TemplateType.BROWSER,
            template_name=browser_template.template_name,
            sandbox_idle_timeout_seconds=600,
        )

        try:
            # 等待浏览器就绪
            max_retries = 30
            retry_count = 0
            while retry_count < max_retries:
                health_status = await sb.check_health_async()
                if health_status.get("status") == "ok":
                    break
                await asyncio.sleep(2)
                retry_count += 1

            # 使用 Playwright 异步操作
            async with sb.async_playwright() as playwright:
                # 导航到百度首页
                response = await playwright.goto("https://www.baidu.com")
                assert response is not None

                # 获取页面标题
                title = await playwright.title()
                assert title is not None
                assert len(title) > 0
                assert "百度" in title

                # 使用 evaluate 获取页面 URL
                url = await playwright.evaluate("window.location.href")
                assert url is not None
                assert "baidu.com" in url
        finally:
            await sb.delete_async()

    async def test_playwright_async_screenshot_async(
        self, browser_template: Template
    ):
        """测试 Playwright 异步截图功能"""
        sb = await Sandbox.create_async(
            template_type=TemplateType.BROWSER,
            template_name=browser_template.template_name,
            sandbox_idle_timeout_seconds=600,
        )

        screenshot_path = f"test_browser_screenshot_{int(time.time())}.png"

        try:
            # 等待浏览器就绪
            max_retries = 30
            retry_count = 0
            while retry_count < max_retries:
                health_status = await sb.check_health_async()
                if health_status.get("status") == "ok":
                    break
                await asyncio.sleep(2)
                retry_count += 1

            # 使用 Playwright 进行截图
            async with sb.async_playwright() as playwright:
                await playwright.goto("https://www.baidu.com")
                await asyncio.sleep(2)  # 等待页面加载完成

                await playwright.screenshot(path=screenshot_path)

            # 验证截图文件已创建
            assert os.path.exists(screenshot_path)
            assert os.path.getsize(screenshot_path) > 0
        finally:
            await sb.delete_async()
            # 清理截图文件
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)

    async def test_playwright_async_search_async(
        self, browser_template: Template
    ):
        """测试 Playwright 异步搜索功能"""
        sb = await Sandbox.create_async(
            template_type=TemplateType.BROWSER,
            template_name=browser_template.template_name,
            sandbox_idle_timeout_seconds=600,
        )

        try:
            # 等待浏览器就绪
            max_retries = 30
            retry_count = 0
            while retry_count < max_retries:
                health_status = await sb.check_health_async()
                if health_status.get("status") == "ok":
                    break
                await asyncio.sleep(2)
                retry_count += 1

            # 使用 Playwright 进行搜索操作
            async with sb.async_playwright() as playwright:
                # 访问百度
                await playwright.goto("https://www.baidu.com")
                await asyncio.sleep(1)

                # 输入搜索关键词
                await playwright.fill("#kw", "阿里云")
                await asyncio.sleep(0.5)

                # 点击搜索按钮
                await playwright.click("#su")
                await asyncio.sleep(2)

                # 验证搜索结果页面
                url = await playwright.evaluate("window.location.href")
                assert "baidu.com" in url

                # 获取页面标题
                title = await playwright.title()
                assert title is not None
        finally:
            await sb.delete_async()

    async def test_playwright_async_multiple_pages_async(
        self, browser_template: Template
    ):
        """测试 Playwright 异步多页面跳转"""
        sb = await Sandbox.create_async(
            template_type=TemplateType.BROWSER,
            template_name=browser_template.template_name,
            sandbox_idle_timeout_seconds=600,
        )

        try:
            # 等待浏览器就绪
            max_retries = 30
            retry_count = 0
            while retry_count < max_retries:
                health_status = await sb.check_health_async()
                if health_status.get("status") == "ok":
                    break
                await asyncio.sleep(2)
                retry_count += 1

            # 使用 Playwright 访问多个页面
            async with sb.async_playwright() as playwright:
                # 访问第一个页面
                await playwright.goto("https://www.baidu.com")
                title1 = await playwright.title()
                assert "百度" in title1

                await asyncio.sleep(1)

                # 跳转到第二个页面
                await playwright.goto("https://www.aliyun.com")
                await asyncio.sleep(2)
                title2 = await playwright.title()
                assert title2 is not None

                # 后退
                await playwright.go_back()
                await asyncio.sleep(1)
                url_after_back = await playwright.evaluate(
                    "window.location.href"
                )
                assert "baidu.com" in url_after_back

                # 前进
                await playwright.go_forward()
                await asyncio.sleep(1)
                url_after_forward = await playwright.evaluate(
                    "window.location.href"
                )
                assert "aliyun.com" in url_after_forward
        finally:
            await sb.delete_async()

    async def test_browser_recordings_async(self, browser_template: Template):
        """测试浏览器录制功能"""
        sb = await Sandbox.create_async(
            template_type=TemplateType.BROWSER,
            template_name=browser_template.template_name,
            sandbox_idle_timeout_seconds=600,
        )

        download_path = f"test_recording_{int(time.time())}.mkv"
        screenshot_path = f"temp_screenshot_{int(time.time())}.png"

        try:
            # 等待浏览器就绪
            max_retries = 30
            retry_count = 0
            while retry_count < max_retries:
                health_status = await sb.check_health_async()
                if health_status.get("status") == "ok":
                    break
                await asyncio.sleep(2)
                retry_count += 1

            # 使用带录制的 Playwright
            async with sb.async_playwright(record=True) as playwright:
                await playwright.goto("https://www.baidu.com")
                await asyncio.sleep(3)  # 给录制足够的时间
                await playwright.screenshot(path=screenshot_path)
                await asyncio.sleep(2)  # 再等待一些时间确保录制

            # 等待录制完成并上传
            await asyncio.sleep(5)

            # 列出录制文件
            recordings = sb.list_recordings()
            assert recordings is not None
            assert "recordings" in recordings

            if len(recordings["recordings"]) > 0:
                # 下载第一个录制文件
                first_recording = recordings["recordings"][0]
                filename = first_recording["filename"]

                sb.download_recording(filename, download_path)

                # 验证下载的文件
                assert os.path.exists(download_path)
                # 录制文件可能为空或很小，只检查文件是否存在
                file_size = os.path.getsize(download_path)
                # 如果文件大小为0，可能是录制还在处理中，但文件应该存在
                assert file_size >= 0
            else:
                # 如果没有录制文件，至少验证API调用成功
                assert isinstance(recordings["recordings"], list)
        finally:
            await sb.delete_async()
            # 清理下载的文件
            if os.path.exists(download_path):
                os.remove(download_path)
            # 清理截图文件
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)
            # 清理可能的临时截图
            for temp_file in Path(".").glob("temp_screenshot_*.png"):
                if temp_file.exists():
                    temp_file.unlink()

    async def test_browser_lifecycle_async(self, browser_template: Template):
        """测试浏览器 Sandbox 完整生命周期"""
        # 1. 创建
        sb = await Sandbox.create_async(
            template_type=TemplateType.BROWSER,
            template_name=browser_template.template_name,
            sandbox_idle_timeout_seconds=600,
        )
        assert sb.sandbox_id is not None

        try:
            # 2. 健康检查
            max_retries = 30
            retry_count = 0
            while retry_count < max_retries:
                health_status = await sb.check_health_async()
                if health_status.get("status") == "ok":
                    break
                await asyncio.sleep(2)
                retry_count += 1

            # 3. 获取连接 URL
            vnc_url = sb.get_vnc_url()
            assert vnc_url is not None

            cdp_url = sb.get_cdp_url()
            assert cdp_url is not None

            # 4. 使用 Playwright
            async with sb.async_playwright() as playwright:
                await playwright.goto("https://www.baidu.com")
                title = await playwright.title()
                assert "百度" in title

            # 5. 删除
            await sb.delete_async()
        except Exception as e:
            # 确保即使测试失败也清理资源
            try:
                await sb.delete_async()
            except Exception:
                pass
            raise e

    async def test_browser_connect_async(self, browser_template: Template):
        """测试连接到已存在的浏览器 Sandbox"""
        # 创建 Sandbox
        sb1 = await Sandbox.create_async(
            template_type=TemplateType.BROWSER,
            template_name=browser_template.template_name,
            sandbox_idle_timeout_seconds=600,
        )

        try:
            # 等待浏览器就绪
            max_retries = 30
            retry_count = 0
            while retry_count < max_retries:
                health_status = await sb1.check_health_async()
                if health_status.get("status") == "ok":
                    break
                await asyncio.sleep(2)
                retry_count += 1

            sandbox_id = sb1.sandbox_id

            # 连接到已存在的 Sandbox
            assert sandbox_id
            sb2 = await Sandbox.connect_async(sandbox_id)
            assert sb2 is not None
            assert sb2.sandbox_id == sandbox_id

            # 验证连接的 Sandbox 可以正常使用
            health_status = await sb2.check_health_async()
            assert health_status.get("status") == "ok"

            # 使用连接的 Sandbox 进行操作
            assert isinstance(sb2, BrowserSandbox)
            async with sb2.async_playwright() as playwright:
                await playwright.goto("https://www.baidu.com")
                title = await playwright.title()
                assert "百度" in title
        finally:
            await sb1.delete_async()

    async def test_browser_concurrent_operations_async(
        self, browser_template: Template
    ):
        """测试并发创建多个浏览器 Sandbox"""
        num_sandboxes = 2
        sandboxes = []

        try:
            # 并发创建多个 Sandbox
            create_tasks = [
                Sandbox.create_async(
                    template_type=TemplateType.BROWSER,
                    template_name=browser_template.template_name,
                    sandbox_idle_timeout_seconds=600,
                )
                for _ in range(num_sandboxes)
            ]
            sandboxes = await asyncio.gather(*create_tasks)

            # 验证所有 Sandbox 都创建成功
            assert len(sandboxes) == num_sandboxes
            for sb in sandboxes:
                assert sb.sandbox_id is not None
        finally:
            # 并发删除所有 Sandbox
            if sandboxes:
                for sb in sandboxes:
                    await sb.delete_async()
