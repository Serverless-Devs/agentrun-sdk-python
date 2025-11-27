from __future__ import annotations

import atexit
import threading
import time
from typing import Any, Callable, Dict, Optional

from agentrun.integration.utils.tool import CommonToolSet, tool
from agentrun.sandbox import Sandbox, TemplateType
from agentrun.sandbox.browser_sandbox import BrowserSandbox
from agentrun.sandbox.client import SandboxClient
from agentrun.sandbox.code_interpreter_sandbox import CodeInterpreterSandbox
from agentrun.utils.config import Config
from agentrun.utils.log import logger


class SandboxToolSet(CommonToolSet):

    def __init__(
        self,
        template_name: str,
        template_type: TemplateType,
        *,
        sandbox_idle_timeout_seconds: int,
        config: Optional[Config],
    ):
        super().__init__()

        self.config = config
        self.client = SandboxClient(config)
        self.lock = threading.Lock()

        self.template_name = template_name
        self.template_type = template_type
        self.sandbox_idle_timeout_seconds = sandbox_idle_timeout_seconds

        self.sandbox: Optional[Sandbox] = None
        self.sandbox_id = ""

    def close(self):
        if self.sandbox:
            try:
                self.sandbox.stop()
            except Exception as e:
                logger.debug("delete sandbox failed, due to %s", e)

    def _ensure_sandbox(self):
        # 先确定当前是否有 sandbox，如果有的话提供这个
        if self.sandbox is not None:
            return self.sandbox

        # 如果没有，则创建一个新的 Sandbox
        with self.lock:
            if self.sandbox is None:
                self.sandbox = Sandbox.create(
                    template_type=self.template_type,
                    template_name=self.template_name,
                    sandbox_idle_timeout_seconds=self.sandbox_idle_timeout_seconds,
                    config=self.config,
                )
                self.sandbox_id = self.sandbox.sandbox_id
                self.sandbox.__enter__()

        return self.sandbox

    def _run_in_sandbox(self, callback: Callable[[Sandbox], Any]):
        sb = self._ensure_sandbox()
        try:
            return callback(sb)
        except Exception as e:
            try:
                logger.debug(
                    "run in sandbox failed, due to %s, try to re-create"
                    " sandbox",
                    e,
                )

                self.sandbox = None
                sb = self._ensure_sandbox()
                return callback(sb)
            except Exception as e2:
                logger.debug("re-created sandbox run failed, due to %s", e2)
                return {"error": f"{e!s}"}


class CodeInterpreterToolSet(SandboxToolSet):
    """LangChain 代码沙箱工具适配器。"""

    def __init__(
        self,
        template_name: str,
        config: Optional[Config],
        sandbox_idle_timeout_seconds: int,
    ) -> None:
        super().__init__(
            template_name=template_name,
            template_type=TemplateType.CODE_INTERPRETER,
            sandbox_idle_timeout_seconds=sandbox_idle_timeout_seconds,
            config=config,
        )

    @tool()
    def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 60,
    ) -> Dict[str, Any]:
        "在指定的 Code Interpreter 沙箱中执行代码"

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            with sb.context.create() as ctx:
                try:
                    result = ctx.execute(code=code, timeout=timeout)
                finally:
                    try:
                        ctx.delete()
                    except Exception:
                        pass
                return {
                    "stdout": result.get("stdout"),
                    "stderr": result.get("stderr"),
                    "raw": result,
                }

        return self._run_in_sandbox(inner)

    @tool()
    def list_directory(self, path: str = "/") -> Dict[str, Any]:
        """列出沙箱中的文件"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            return {
                "path": path,
                "entries": sb.file_system.list(path=path),
            }

        return self._run_in_sandbox(inner)

    @tool()
    def read_file(self, path: str) -> Dict[str, Any]:
        """读取沙箱文件内容"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            return {
                "path": path,
                "content": sb.file.read(path=path),
            }

        return self._run_in_sandbox(inner)

    @tool(
        name="sandbox_write_file",
        description="向沙箱写入文本文件",
    )
    def write_file(self, path: str, content: str) -> Dict[str, Any]:

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            return {
                "path": path,
                "result": sb.file.write(path=path, content=content),
            }

        return self._run_in_sandbox(inner)


class BrowserToolSet(SandboxToolSet):
    """LangChain 浏览器工具适配器。"""

    def __init__(
        self,
        template_name: str,
        config: Optional[Config],
        sandbox_idle_timeout_seconds: int,
    ) -> None:

        super().__init__(
            template_name=template_name,
            template_type=TemplateType.BROWSER,
            sandbox_idle_timeout_seconds=sandbox_idle_timeout_seconds,
            config=config,
        )

    @tool()
    def goto(self, url: str):
        """导航到 URL"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            with sb.sync_playwright() as p:
                return p.goto(url)

        return self._run_in_sandbox(inner)

    @tool()
    def html_content(
        self,
    ):
        """获取页面 html 内容"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            with sb.sync_playwright() as p:
                return p.html_content()

        return self._run_in_sandbox(inner)

    @tool()
    def fill(self, selector: str, value: str):
        """在页面中填充输入框"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            with sb.sync_playwright() as p:
                return p.fill(selector, value)

        return self._run_in_sandbox(inner)

    @tool()
    def click(self, selector: str):
        """
        在网页上执行点击操作

        Args:
            selector: 要点击的元素选择器
        """

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            with sb.sync_playwright() as p:
                return p.click(selector)

        return self._run_in_sandbox(inner)

    @tool()
    def evaluate(self, expression: str):
        """
        在网页上执行 js 脚本

        Args:
            expression: 要执行的脚本
        """

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            with sb.sync_playwright() as p:
                return p.evaluate(expression)

        return self._run_in_sandbox(inner)


def sandbox_toolset(
    template_name: str,
    *,
    template_type: TemplateType = TemplateType.CODE_INTERPRETER,
    config: Optional[Config] = None,
    sandbox_idle_timeout_seconds: int = 5 * 60,
) -> CommonToolSet:
    """将沙箱模板封装为 LangChain ``StructuredTool`` 列表。"""

    if template_type != TemplateType.CODE_INTERPRETER:
        return BrowserToolSet(
            template_name=template_name,
            config=config,
            sandbox_idle_timeout_seconds=sandbox_idle_timeout_seconds,
        )
    else:
        return CodeInterpreterToolSet(
            template_name=template_name,
            config=config,
            sandbox_idle_timeout_seconds=sandbox_idle_timeout_seconds,
        )
