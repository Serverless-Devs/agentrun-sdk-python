from __future__ import annotations

import base64
import threading
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from agentrun.integration.utils.tool import CommonToolSet, tool
from agentrun.sandbox import Sandbox, TemplateType
from agentrun.sandbox.browser_sandbox import BrowserSandbox
from agentrun.sandbox.client import SandboxClient
from agentrun.sandbox.code_interpreter_sandbox import CodeInterpreterSandbox
from agentrun.utils.config import Config
from agentrun.utils.log import logger

if TYPE_CHECKING:
    from agentrun.sandbox.api.playwright_sync import BrowserPlaywrightSync
    from agentrun.sandbox.model import NASConfig, OSSMountConfig, PolarFsConfig

try:
    from playwright.sync_api import Error as PlaywrightError
except ImportError:

    class PlaywrightError(Exception):  # type: ignore[no-redef]
        """Fallback Playwright error used when Playwright is not installed."""

        pass


class SandboxToolSet(CommonToolSet):
    """沙箱工具集基类

    提供沙箱生命周期管理和工具执行的基础设施。
    """

    def __init__(
        self,
        template_name: str,
        template_type: TemplateType,
        *,
        sandbox_idle_timeout_seconds: int,
        config: Optional[Config],
        oss_mount_config: Optional["OSSMountConfig"] = None,
        nas_config: Optional["NASConfig"] = None,
        polar_fs_config: Optional["PolarFsConfig"] = None,
    ):
        super().__init__()

        self.config = config
        self.client = SandboxClient(config)
        self.lock = threading.Lock()

        self.template_name = template_name
        self.template_type = template_type
        self.sandbox_idle_timeout_seconds = sandbox_idle_timeout_seconds

        self.oss_mount_config = oss_mount_config
        self.nas_config = nas_config
        self.polar_fs_config = polar_fs_config

        self.sandbox: Optional[Sandbox] = None
        self.sandbox_id = ""

    def close(self):
        """关闭并释放沙箱资源"""
        if self.sandbox:
            try:
                self.sandbox.stop()
            except Exception as e:
                logger.debug("delete sandbox failed, due to %s", e)
            self.sandbox = None
            self.sandbox_id = ""

    def _ensure_sandbox(self):
        """确保沙箱实例存在，如果不存在则创建"""
        if self.sandbox is not None:
            return self.sandbox

        with self.lock:
            if self.sandbox is None:
                self.sandbox = Sandbox.create(
                    template_type=self.template_type,
                    template_name=self.template_name,
                    sandbox_idle_timeout_seconds=self.sandbox_idle_timeout_seconds,
                    oss_mount_config=self.oss_mount_config,
                    nas_config=self.nas_config,
                    polar_fs_config=self.polar_fs_config,
                    config=self.config,
                )
                self.sandbox_id = self.sandbox.sandbox_id
                self.sandbox.__enter__()

        return self.sandbox

    def _run_in_sandbox(self, callback: Callable[[Sandbox], Any]):
        """在沙箱中执行操作，失败时自动重试"""
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
    """代码解释器沙箱工具集 / Code Interpreter Sandbox ToolSet

    提供代码执行、文件操作、进程管理等功能，兼容官方 MCP 工具接口。
    Provides code execution, file operations, and process management capabilities,
    compatible with official MCP tool interfaces.

    功能分类 / Feature Categories:
    - 健康检查 / Health Check: health
    - 代码执行 / Code Execution: run_code
    - 上下文管理 / Context Management: list_contexts, create_context,
      get_context, delete_context
    - 文件操作 / File Operations: read_file, write_file
    - 文件系统 / File System: file_system_list, file_system_stat,
      file_system_mkdir, file_system_move, file_system_remove
    - 进程管理 / Process Management: process_exec_cmd, process_list,
      process_stat, process_kill

    使用指南 / Usage Guide:
    ============================================================

    ## 代码执行最佳实践 / Code Execution Best Practices

    1. **简单代码执行 / Simple Code Execution**:
       - 使用 `run_code` 执行一次性代码
       - Use `run_code` for one-time code execution
       - 代码执行后上下文会自动清理
       - Context is automatically cleaned up after execution

    2. **有状态代码执行 / Stateful Code Execution**:
       - 先使用 `create_context` 创建上下文
       - First create a context using `create_context`
       - 使用 `run_code` 并传入 `context_id` 在同一上下文中执行多段代码
       - Use `run_code` with `context_id` to execute multiple code segments
       - 变量和导入会在同一上下文中保持
       - Variables and imports persist within the same context
       - 完成后使用 `delete_context` 清理
       - Clean up with `delete_context` when done

    3. **错误处理 / Error Handling**:
       - 检查返回结果中的 `stderr` 和 `exit_code`
       - Check `stderr` and `exit_code` in the response
       - `exit_code` 为 0 表示执行成功
       - `exit_code` of 0 indicates successful execution

    ## 文件操作指南 / File Operations Guide

    1. **读写文件 / Read/Write Files**:
       - 使用 `read_file` 读取文件内容
       - Use `read_file` to read file contents
       - 使用 `write_file` 写入文件，支持指定编码和权限
       - Use `write_file` to write files with encoding and permissions

    2. **目录操作 / Directory Operations**:
       - 使用 `file_system_list` 列出目录内容
       - Use `file_system_list` to list directory contents
       - 使用 `file_system_mkdir` 创建目录
       - Use `file_system_mkdir` to create directories
       - 使用 `file_system_move` 移动或重命名
       - Use `file_system_move` to move or rename

    ## 进程管理指南 / Process Management Guide

    1. **执行命令 / Execute Commands**:
       - 使用 `process_exec_cmd` 执行 Shell 命令
       - Use `process_exec_cmd` to execute shell commands
       - 适合运行系统工具、安装包等操作
       - Suitable for running system tools, installing packages, etc.

    2. **进程监控 / Process Monitoring**:
       - 使用 `process_list` 查看所有运行中的进程
       - Use `process_list` to view all running processes
       - 使用 `process_kill` 终止不需要的进程
       - Use `process_kill` to terminate unwanted processes
    """

    def __init__(
        self,
        template_name: str,
        config: Optional[Config],
        sandbox_idle_timeout_seconds: int,
        oss_mount_config: Optional["OSSMountConfig"] = None,
        nas_config: Optional["NASConfig"] = None,
        polar_fs_config: Optional["PolarFsConfig"] = None,
    ) -> None:
        super().__init__(
            template_name=template_name,
            template_type=TemplateType.CODE_INTERPRETER,
            sandbox_idle_timeout_seconds=sandbox_idle_timeout_seconds,
            config=config,
            oss_mount_config=oss_mount_config,
            nas_config=nas_config,
            polar_fs_config=polar_fs_config,
        )

    # ==================== 健康检查 / Health Check ====================

    @tool(
        name="health",
        description=(
            "Check the health status of the code interpreter sandbox. Returns"
            " status='ok' if the sandbox is running normally. Recommended to"
            " call before other operations to confirm sandbox readiness."
        ),
    )
    def check_health(self) -> Dict[str, Any]:
        """检查沙箱健康状态 / Check sandbox health status"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            return sb.check_health()

        return self._run_in_sandbox(inner)

    # ==================== 代码执行 / Code Execution ====================

    @tool(
        name="run_code",
        description=(
            "Execute code in a secure isolated sandbox environment. Supports"
            " Python and JavaScript languages. Can specify context_id to"
            " execute in an existing context, preserving variable state."
            " Returns stdout, stderr, and exit_code."
        ),
    )
    def run_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 60,
        context_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行代码 / Execute code"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            if context_id:
                result = sb.context.execute(
                    code=code, context_id=context_id, timeout=timeout
                )
            else:
                with sb.context.create() as ctx:
                    try:
                        result = ctx.execute(code=code, timeout=timeout)
                    finally:
                        try:
                            ctx.delete()
                        except Exception:
                            pass
                    return {
                        "stdout": result.get("stdout", ""),
                        "stderr": result.get("stderr", ""),
                        "exit_code": result.get("exitCode", 0),
                        "result": result,
                    }

        return self._run_in_sandbox(inner)

    # 保留旧的 execute_code 作为别名
    @tool(
        name="execute_code",
        description=(
            "Execute code in sandbox (alias for run_code, for backward"
            " compatibility)."
        ),
    )
    def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """执行代码（run_code 的别名）/ Execute code (alias for run_code)"""
        return self.run_code(code=code, language=language, timeout=timeout)

    # ==================== 上下文管理 / Context Management ====================

    @tool(
        name="list_contexts",
        description=(
            "List all created execution contexts. Contexts preserve code"
            " execution state like variables and imported modules."
        ),
    )
    def list_contexts(self) -> Dict[str, Any]:
        """列出所有执行上下文 / List all execution contexts"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            contexts = sb.context.list()
            return {"contexts": contexts}

        return self._run_in_sandbox(inner)

    @tool(
        name="create_context",
        description=(
            "Create a new execution context for stateful code execution. Code"
            " executed in the same context can share variables and imports."
            " Returns context_id for subsequent run_code calls. Call"
            " delete_context to release resources when done."
        ),
    )
    def create_context(
        self,
        language: str = "python",
        cwd: str = "/home/user",
    ) -> Dict[str, Any]:
        """创建新的执行上下文 / Create a new execution context"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            ctx = sb.context.create(language=language, cwd=cwd)
            return {
                "context_id": ctx.context_id,
                "language": ctx._language,
                "cwd": ctx._cwd,
            }

        return self._run_in_sandbox(inner)

    @tool(
        name="get_context",
        description=(
            "Get details of a specific execution context, including language"
            " and working directory."
        ),
    )
    def get_context(self, context_id: str) -> Dict[str, Any]:
        """获取上下文详情 / Get context details"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            ctx = sb.context.get(context_id=context_id)
            return {
                "context_id": ctx.context_id,
                "language": ctx._language,
                "cwd": ctx._cwd,
            }

        return self._run_in_sandbox(inner)

    @tool(
        name="delete_context",
        description=(
            "Delete a specific execution context and release related resources."
            " All variables and state in the context will be lost after"
            " deletion."
        ),
    )
    def delete_context(self, context_id: str) -> Dict[str, Any]:
        """删除执行上下文 / Delete execution context"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            result = sb.context.delete(context_id=context_id)
            return {"success": True, "result": result}

        return self._run_in_sandbox(inner)

    # ==================== 文件操作 / File Operations ====================

    @tool(
        name="read_file",
        description=(
            "Read the content of a file at the specified path in the sandbox."
            " Returns the text content. Suitable for reading code files,"
            " configs, logs, etc."
        ),
    )
    def read_file(self, path: str) -> Dict[str, Any]:
        """读取文件内容 / Read file content"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            content = sb.file.read(path=path)
            return {"path": path, "content": content}

        return self._run_in_sandbox(inner)

    @tool(
        name="write_file",
        description=(
            "Write content to a file at the specified path in the sandbox."
            " Creates the file automatically if it doesn't exist, including"
            " parent directories. Can specify file permission mode and"
            " encoding."
        ),
    )
    def write_file(
        self,
        path: str,
        content: str,
        mode: str = "644",
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        """写入文件内容 / Write file content"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            result = sb.file.write(
                path=path, content=content, mode=mode, encoding=encoding
            )
            return {"path": path, "success": True, "result": result}

        return self._run_in_sandbox(inner)

    # ==================== 文件系统操作 / File System Operations ====================

    @tool(
        name="file_system_list",
        description=(
            "List the contents of a directory in the sandbox, including files"
            " and subdirectories. Can specify traversal depth to get nested"
            " directory structure. Returns name, type, size, etc. for each"
            " entry."
        ),
    )
    def file_system_list(
        self, path: str = "/", depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """列出目录内容 / List directory contents"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            entries = sb.file_system.list(path=path, depth=depth)
            return {"path": path, "entries": entries}

        return self._run_in_sandbox(inner)

    # 保留旧的 list_directory 作为别名
    @tool(
        name="list_directory",
        description="List directory contents (alias for file_system_list).",
    )
    def list_directory(self, path: str = "/") -> Dict[str, Any]:
        """列出目录内容 / List directory contents"""
        return self.file_system_list(path=path)

    @tool(
        name="file_system_stat",
        description=(
            "Get detailed status information of a file or directory. Includes"
            " size, permissions, modification time, access time, and other"
            " metadata. Can be used to check file existence, get file size,"
            " etc."
        ),
    )
    def file_system_stat(self, path: str) -> Dict[str, Any]:
        """获取文件/目录状态 / Get file/directory status"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            stat_info = sb.file_system.stat(path=path)
            return {"path": path, "stat": stat_info}

        return self._run_in_sandbox(inner)

    @tool(
        name="file_system_mkdir",
        description=(
            "Create a directory in the sandbox. By default, automatically"
            " creates all necessary parent directories (like mkdir -p). Can"
            " specify directory permission mode."
        ),
    )
    def file_system_mkdir(
        self,
        path: str,
        parents: bool = True,
        mode: str = "0755",
    ) -> Dict[str, Any]:
        """创建目录 / Create directory"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            result = sb.file_system.mkdir(path=path, parents=parents, mode=mode)
            return {"path": path, "success": True, "result": result}

        return self._run_in_sandbox(inner)

    @tool(
        name="file_system_move",
        description=(
            "Move or rename a file/directory. Can rename within the same"
            " directory or move to a different directory."
        ),
    )
    def file_system_move(self, source: str, destination: str) -> Dict[str, Any]:
        """移动/重命名文件或目录 / Move/rename file or directory"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            result = sb.file_system.move(source=source, destination=destination)
            return {
                "source": source,
                "destination": destination,
                "success": True,
                "result": result,
            }

        return self._run_in_sandbox(inner)

    @tool(
        name="file_system_remove",
        description=(
            "Delete a file or directory. Recursively deletes all contents when"
            " removing a directory. Use with caution."
        ),
    )
    def file_system_remove(self, path: str) -> Dict[str, Any]:
        """删除文件或目录 / Delete file or directory"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            result = sb.file_system.remove(path=path)
            return {"path": path, "success": True, "result": result}

        return self._run_in_sandbox(inner)

    # ==================== 进程管理 / Process Management ====================

    @tool(
        name="process_exec_cmd",
        description=(
            "Execute a shell command in the sandbox. Suitable for running"
            " system tools, installing packages, executing scripts, etc."
            " Returns stdout, stderr, and exit code. Can specify working"
            " directory and timeout."
        ),
    )
    def process_exec_cmd(
        self,
        command: str,
        cwd: str = "/home/user",
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """执行命令 / Execute command"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            result = sb.process.cmd(command=command, cwd=cwd, timeout=timeout)
            return {
                "command": command,
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "exit_code": result.get("exitCode", 0),
                "result": result,
            }

        return self._run_in_sandbox(inner)

    @tool(
        name="process_list",
        description=(
            "List all running processes in the sandbox. Returns PID, name,"
            " status, etc. for each process. Can be used to monitor resource"
            " usage or find processes to terminate."
        ),
    )
    def process_list(self) -> Dict[str, Any]:
        """列出所有进程 / List all processes"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            processes = sb.process.list()
            return {"processes": processes}

        return self._run_in_sandbox(inner)

    @tool(
        name="process_stat",
        description=(
            "Get detailed status information of a specific process. "
            "Includes CPU usage, memory consumption, runtime, etc."
        ),
    )
    def process_stat(self, pid: str) -> Dict[str, Any]:
        """获取进程状态 / Get process status"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            process_info = sb.process.get(pid=pid)
            return {"pid": pid, "process": process_info}

        return self._run_in_sandbox(inner)

    @tool(
        name="process_kill",
        description=(
            "Terminate a specific process. "
            "Used to stop long-running or unresponsive processes. "
            "Make sure not to terminate critical system processes."
        ),
    )
    def process_kill(self, pid: str) -> Dict[str, Any]:
        """终止进程 / Terminate process"""

        def inner(sb: Sandbox):
            assert isinstance(sb, CodeInterpreterSandbox)
            result = sb.process.kill(pid=pid)
            return {"pid": pid, "success": True, "result": result}

        return self._run_in_sandbox(inner)


class BrowserToolSet(SandboxToolSet):
    """浏览器沙箱工具集 / Browser Sandbox ToolSet

    提供浏览器自动化操作功能，兼容官方 MCP 工具接口。
    Provides browser automation capabilities, compatible with official MCP tool interfaces.

    功能分类 / Feature Categories:
    - 健康检查 / Health Check: health
    - 导航 / Navigation: browser_navigate, browser_navigate_back, browser_go_forward
    - 页面交互 / Page Interaction: browser_click, browser_type, browser_fill,
      browser_hover, browser_drag, browser_dblclick
    - 页面信息 / Page Info: browser_snapshot, browser_take_screenshot, browser_get_title
    - 标签页管理 / Tab Management: browser_tabs_list, browser_tabs_new, browser_tabs_select
    - 高级功能 / Advanced: browser_evaluate, browser_wait_for

    使用指南 / Usage Guide:
    ============================================================

    ## 基本操作流程 / Basic Operation Flow

    1. **导航到目标页面 / Navigate to Target Page**:
       - 使用 `browser_navigate` 访问目标 URL
       - Use `browser_navigate` to visit the target URL
       - 可以指定等待条件：load, domcontentloaded, networkidle
       - Can specify wait condition: load, domcontentloaded, networkidle

    2. **获取页面信息 / Get Page Information**:
       - 使用 `browser_snapshot` 获取页面 HTML 结构
       - Use `browser_snapshot` to get page HTML structure
       - 使用 `browser_take_screenshot` 截取页面截图
       - Use `browser_take_screenshot` to capture page screenshot

    3. **与页面交互 / Interact with Page**:
       - 使用 `browser_click` 点击元素
       - Use `browser_click` to click elements
       - 使用 `browser_fill` 填充表单
       - Use `browser_fill` to fill forms
       - 使用 `browser_type` 逐字符输入（适用于需要触发输入事件的场景）
       - Use `browser_type` for character-by-character input (for triggering input events)

    ## 元素选择器指南 / Selector Guide

    支持多种选择器类型 / Supports multiple selector types:
    - CSS 选择器 / CSS Selectors: `#id`, `.class`, `tag`, `[attr=value]`
    - 文本选择器 / Text Selectors: `text=Submit`, `text="Exact Match"`
    - XPath: `xpath=//button[@type="submit"]`
    - 组合选择器 / Combined: `div.container >> button.submit`

    ## 常见操作示例 / Common Operation Examples

    1. **搜索操作 / Search Operation**:
       ```
       browser_navigate(url="https://www.bing.com")
       browser_fill(selector='input[name="q"]', value="搜索关键词")
       browser_click(selector='input[type="submit"]')
       ```

    2. **表单填写 / Form Filling**:
       ```
       browser_fill(selector='#username', value="user@example.com")
       browser_fill(selector='#password', value="password123")
       browser_click(selector='button[type="submit"]')
       ```

    3. **页面分析 / Page Analysis**:
       ```
       browser_snapshot()  # 获取 HTML 结构
       browser_take_screenshot(full_page=True)  # 截取完整页面
       ```

    ## 错误处理建议 / Error Handling Tips

    - 操作可能因元素未加载而失败，使用 `browser_wait_for` 等待
    - Operations may fail if elements not loaded, use `browser_wait_for` to wait
    - 如果选择器找不到元素，尝试使用更具体或不同的选择器
    - If selector doesn't find element, try more specific or different selectors
    - 使用 `browser_snapshot` 检查页面当前状态
    - Use `browser_snapshot` to check current page state

    ## 多标签页操作 / Multi-Tab Operations

    - 使用 `browser_tabs_new` 创建新标签页
    - Use `browser_tabs_new` to create new tabs
    - 使用 `browser_tabs_list` 查看所有标签页
    - Use `browser_tabs_list` to view all tabs
    - 使用 `browser_tabs_select` 切换标签页
    - Use `browser_tabs_select` to switch tabs
    """

    def __init__(
        self,
        template_name: str,
        config: Optional[Config],
        sandbox_idle_timeout_seconds: int,
        oss_mount_config: Optional["OSSMountConfig"] = None,
        nas_config: Optional["NASConfig"] = None,
        polar_fs_config: Optional["PolarFsConfig"] = None,
    ) -> None:

        super().__init__(
            template_name=template_name,
            template_type=TemplateType.BROWSER,
            sandbox_idle_timeout_seconds=sandbox_idle_timeout_seconds,
            config=config,
            oss_mount_config=oss_mount_config,
            nas_config=nas_config,
            polar_fs_config=polar_fs_config,
        )
        self._playwright_sync: Optional["BrowserPlaywrightSync"] = None
        self._playwright_thread: Optional[threading.Thread] = None

    def _get_playwright(self, sb: BrowserSandbox) -> "BrowserPlaywrightSync":
        """获取或创建 Playwright 连接 / Get or create Playwright connection

        复用已有连接以减少连接建立开销和瞬态错误。
        使用双重检查锁定避免并发调用时创建多个连接导致资源泄漏。
        当创建连接的线程已退出时，自动重建连接（Playwright greenlet 绑定到创建它的线程）。

        Reuses existing connection to reduce connection overhead and transient errors.
        Uses double-checked locking to avoid leaking connections under concurrent calls.
        Automatically recreates the connection when the thread that created it has exited,
        because Playwright's internal greenlet is bound to the thread that created it.
        """
        if self._playwright_sync is not None and self._playwright_thread is not None:
            current_thread = threading.current_thread()
            creator_thread = self._playwright_thread
            if not creator_thread.is_alive() or current_thread is not creator_thread:
                if not creator_thread.is_alive():
                    logger.debug(
                        "Playwright creating thread (id=%s) has exited, recreating"
                        " connection",
                        creator_thread.ident,
                    )
                else:
                    logger.debug(
                        "Playwright creating thread (id=%s) differs from current"
                        " thread (id=%s), recreating connection",
                        creator_thread.ident,
                        current_thread.ident,
                    )
                self._reset_playwright()

        if self._playwright_sync is None:
            with self.lock:
                if self._playwright_sync is None:
                    playwright_sync = sb.sync_playwright()
                    playwright_sync.open()
                    self._playwright_sync = playwright_sync
                    self._playwright_thread = threading.current_thread()
        return self._playwright_sync

    def _reset_playwright(self) -> None:
        """重置 Playwright 连接 / Reset Playwright connection

        在沙箱重建时调用，清理旧的连接。
        Called when sandbox is recreated, cleans up old connection.
        """
        with self.lock:
            if self._playwright_sync is not None:
                try:
                    self._playwright_sync.close()
                except Exception as e:
                    logger.debug(
                        "Error while closing Playwright connection: %s",
                        e,
                        exc_info=True,
                    )
                self._playwright_sync = None
            self._playwright_thread = None

    def _run_in_sandbox(self, callback: Callable[[Sandbox], Any]) -> Any:
        """在沙箱中执行操作，智能区分错误类型 / Execute in sandbox with smart error handling

        与基类不同，此方法区分两类错误：
        - 基础设施错误（连接断开、沙箱崩溃）：重建沙箱并重试
        - 工具级错误（JS 执行失败、元素找不到）：直接返回错误，不重建

        Unlike the base class, this method distinguishes two types of errors:
        - Infrastructure errors (connection lost, sandbox crashed): recreate sandbox and retry
        - Tool-level errors (JS execution failed, element not found): return error without recreating
        """
        sb = self._ensure_sandbox()
        try:
            return callback(sb)
        except (ConnectionError, OSError, BrokenPipeError) as e:
            logger.debug(
                "Browser sandbox infrastructure error: %s, recreating sandbox",
                e,
            )
            self._reset_playwright()
            self.sandbox = None
            try:
                sb = self._ensure_sandbox()
                return callback(sb)
            except Exception as e2:
                logger.debug("Recreated sandbox run failed: %s", e2)
                return {"error": f"{e!s}"}
        except PlaywrightError as e:
            error_msg = str(e)
            if self._is_infrastructure_error(error_msg):
                logger.debug(
                    "Browser sandbox CDP connection error: %s, recreating"
                    " sandbox",
                    e,
                )
                self._reset_playwright()
                self.sandbox = None
                try:
                    sb = self._ensure_sandbox()
                    return callback(sb)
                except Exception as e2:
                    logger.debug("Recreated sandbox run failed: %s", e2)
                    return {"error": f"{e!s}"}
            else:
                logger.debug(
                    "Browser tool-level error (no sandbox rebuild): %s", e
                )
                return {"error": f"{e!s}"}
        except Exception as e:
            error_msg = str(e)
            if "cannot switch to" in error_msg:
                logger.debug(
                    "Greenlet thread-binding error, resetting Playwright: %s",
                    e,
                )
                self._reset_playwright()
                self.sandbox = None
            else:
                logger.debug("Unexpected error in browser sandbox: %s", e)
            return {"error": f"{e!s}"}

    def _is_infrastructure_error(self, error_msg: str) -> bool:
        """判断是否为基础设施错误 / Check if error is infrastructure-level

        基础设施错误表示沙箱或 CDP 连接出现问题，需要重建。
        工具级错误（如 JS 执行失败、元素找不到）不需要重建。

        Infrastructure errors indicate sandbox or CDP connection issues, requiring rebuild.
        Tool-level errors (e.g., JS execution failed, element not found) don't need rebuild.
        """
        infrastructure_patterns = [
            "Target closed",
            "Connection closed",
            "Browser closed",
            "Protocol error",
            "WebSocket",
            "ECONNREFUSED",
            "ECONNRESET",
            "EPIPE",
            "Target page, context or browser has been closed",
        ]
        error_lower = error_msg.lower()
        return any(
            pattern.lower() in error_lower
            for pattern in infrastructure_patterns
        )

    def close(self) -> None:
        """关闭并释放沙箱和 Playwright 资源 / Close and release sandbox and Playwright resources"""
        self._reset_playwright()
        super().close()

    # ==================== 健康检查 / Health Check ====================

    @tool(
        name="health",
        description=(
            "Check the health status of the browser sandbox. Returns"
            " status='ok' if the browser is running normally. Recommended to"
            " call before other operations to confirm browser readiness."
        ),
    )
    def check_health(self) -> Dict[str, Any]:
        """检查浏览器沙箱健康状态 / Check browser sandbox health status"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            return sb.check_health()

        return self._run_in_sandbox(inner)

    # ==================== 导航 / Navigation ====================

    @tool(
        name="browser_navigate",
        description=(
            "Navigate to the specified URL. This is the first step in browser"
            " automation. Can specify wait condition to ensure page is loaded"
            " before continuing. wait_until options: 'load' (fully loaded),"
            " 'domcontentloaded' (DOM ready), 'networkidle' (network idle),"
            " 'commit' (response received)."
        ),
    )
    def browser_navigate(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """导航到 URL / Navigate to URL"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            response = p.goto(url, wait_until=wait_until, timeout=timeout)
            return {
                "url": url,
                "success": True,
                "status": response.status if response else None,
            }

        return self._run_in_sandbox(inner)

    # 保留旧的 goto 作为别名
    @tool(
        name="goto",
        description=(
            "Navigate to URL (alias for browser_navigate, for backward"
            " compatibility)."
        ),
    )
    def goto(self, url: str) -> Dict[str, Any]:
        """导航到 URL / Navigate to URL"""
        return self.browser_navigate(url=url)

    @tool(
        name="browser_navigate_back",
        description=(
            "Go back to the previous page, equivalent to clicking the browser's"
            " back button. If there's no history to go back to, the operation"
            " has no effect."
        ),
    )
    def browser_navigate_back(
        self,
        wait_until: str = "load",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """返回上一页 / Go back to previous page"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            response = p.go_back(wait_until=wait_until, timeout=timeout)
            return {
                "success": True,
                "status": response.status if response else None,
            }

        return self._run_in_sandbox(inner)

    @tool(
        name="browser_go_forward",
        description=(
            "Go forward to the next page, equivalent to clicking the browser's"
            " forward button. Only effective if a back navigation was"
            " previously performed."
        ),
    )
    def browser_go_forward(
        self,
        wait_until: str = "load",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """前进到下一页 / Go forward to next page"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            response = p.go_forward(wait_until=wait_until, timeout=timeout)
            return {
                "success": True,
                "status": response.status if response else None,
            }

        return self._run_in_sandbox(inner)

    # ==================== 页面交互 - 点击/拖拽 / Click/Drag ====================

    @tool(
        name="browser_click",
        description=(
            "Click an element matching the selector on the page. "
            "Supports CSS selectors, text selectors (text=xxx), XPath, etc. "
            "Can specify mouse button (left/right/middle) and click count. "
            "Automatically scrolls to make the element visible if needed."
        ),
    )
    def browser_click(
        self,
        selector: str,
        button: str = "left",
        click_count: int = 1,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """点击元素 / Click element"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            p.click(
                selector,
                button=button,
                click_count=click_count,
                timeout=timeout,
            )
            return {"selector": selector, "success": True}

        return self._run_in_sandbox(inner)

    # 保留旧的 click 作为别名
    @tool(
        name="click",
        description=(
            "Click element (alias for browser_click, for backward"
            " compatibility)."
        ),
    )
    def click(self, selector: str) -> Dict[str, Any]:
        """点击元素 / Click element"""
        return self.browser_click(selector=selector)

    @tool(
        name="browser_dblclick",
        description=(
            "Double-click an element matching the selector on the page."
            " Commonly used for opening files, editing text, and other"
            " double-click actions."
        ),
    )
    def browser_dblclick(
        self,
        selector: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """双击元素 / Double-click element"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            p.dblclick(selector, timeout=timeout)
            return {"selector": selector, "success": True}

        return self._run_in_sandbox(inner)

    @tool(
        name="browser_drag",
        description=(
            "Drag the source element to the target element position. "
            "Commonly used for drag-and-drop sorting, file upload areas, etc."
        ),
    )
    def browser_drag(
        self,
        source_selector: str,
        target_selector: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """拖拽元素 / Drag element"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            p.drag_and_drop(source_selector, target_selector, timeout=timeout)
            return {
                "source": source_selector,
                "target": target_selector,
                "success": True,
            }

        return self._run_in_sandbox(inner)

    @tool(
        name="browser_hover",
        description=(
            "Hover the mouse over an element matching the selector. Commonly"
            " used to trigger hover menus, tooltips, and other hover effects."
        ),
    )
    def browser_hover(
        self,
        selector: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """鼠标悬停 / Mouse hover"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            p.hover(selector, timeout=timeout)
            return {"selector": selector, "success": True}

        return self._run_in_sandbox(inner)

    # ==================== 页面交互 - 输入 / Input ====================

    @tool(
        name="browser_type",
        description=(
            "Type text character by character in an element matching the"
            " selector. Triggers keydown, keypress, keyup events for each"
            " character. Suitable for scenarios that need input events (like"
            " autocomplete, live search). Use browser_fill if you just need to"
            " fill a value without triggering events."
        ),
    )
    def browser_type(
        self,
        selector: str,
        text: str,
        delay: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """输入文本 / Type text"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            p.type(selector, text, delay=delay, timeout=timeout)
            return {"selector": selector, "text": text, "success": True}

        return self._run_in_sandbox(inner)

    @tool(
        name="browser_fill",
        description=(
            "Fill a form input with a value all at once. Clears existing"
            " content first, then fills the new value. Faster than browser_type"
            " but doesn't trigger character-by-character input events. Suitable"
            " for regular form filling scenarios."
        ),
    )
    def browser_fill(
        self,
        selector: str,
        value: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """填充输入框 / Fill input"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            p.fill(selector, value, timeout=timeout)
            return {"selector": selector, "value": value, "success": True}

        return self._run_in_sandbox(inner)

    # 保留旧的 fill 作为别名
    @tool(
        name="fill",
        description=(
            "Fill input (alias for browser_fill, for backward compatibility)."
        ),
    )
    def fill(self, selector: str, value: str) -> Dict[str, Any]:
        """填充输入框 / Fill input"""
        return self.browser_fill(selector=selector, value=value)

    # ==================== 页面信息 / Page Information ====================

    @tool(
        name="browser_snapshot",
        description=(
            "Get the HTML snapshot and title of the current page. This is the"
            " preferred method for analyzing page structure, useful for"
            " understanding layout and finding elements. Returns complete HTML"
            " content for extracting information or determining next steps."
        ),
    )
    def browser_snapshot(self) -> Dict[str, Any]:
        """获取页面 HTML 快照 / Get page HTML snapshot"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            html = p.html_content()
            title = p.title()
            return {"html": html, "title": title}

        return self._run_in_sandbox(inner)

    # 保留旧的 html_content 作为别名
    @tool(
        name="html_content",
        description=(
            "Get page HTML content (alias for browser_snapshot, for backward"
            " compatibility)."
        ),
    )
    def html_content(self) -> Dict[str, Any]:
        """获取页面 HTML 内容 / Get page HTML content"""
        return self.browser_snapshot()

    @tool(
        name="browser_take_screenshot",
        description=(
            "Capture a screenshot of the current page, returns base64 encoded"
            " image data. Can capture viewport only or full page"
            " (full_page=True). Supports PNG and JPEG formats. Suitable for"
            " visual verification or saving page state."
        ),
    )
    def browser_take_screenshot(
        self,
        full_page: bool = False,
        type: str = "png",
    ) -> Dict[str, Any]:
        """截取页面截图 / Take page screenshot"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            screenshot_bytes = p.screenshot(full_page=full_page, type=type)
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode(
                "utf-8"
            )
            return {
                "screenshot": screenshot_base64,
                "format": type,
                "full_page": full_page,
            }

        return self._run_in_sandbox(inner)

    @tool(
        name="browser_get_title",
        description=(
            "Get the title of the current page (content of title tag). "
            "Can be used to verify navigation to the correct page."
        ),
    )
    def browser_get_title(self) -> Dict[str, Any]:
        """获取页面标题 / Get page title"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            title = p.title()
            return {"title": title}

        return self._run_in_sandbox(inner)

    # ==================== 标签页管理 / Tab Management ====================

    @tool(
        name="browser_tabs_list",
        description=(
            "List all open browser tabs. Returns index, URL, and title for each"
            " tab. Useful for knowing which pages are open and determining"
            " which tab to switch to."
        ),
    )
    def browser_tabs_list(self) -> Dict[str, Any]:
        """列出所有标签页 / List all tabs"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            pages = p.list_pages()
            tabs = []
            for i, page in enumerate(pages):
                tabs.append({
                    "index": i,
                    "url": page.url,
                    "title": page.title(),
                })
            return {"tabs": tabs, "count": len(tabs)}

        return self._run_in_sandbox(inner)

    @tool(
        name="browser_tabs_new",
        description=(
            "Create a new browser tab. Can optionally specify an initial URL,"
            " otherwise opens a blank page. The new tab automatically becomes"
            " the active tab after creation."
        ),
    )
    def browser_tabs_new(self, url: Optional[str] = None) -> Dict[str, Any]:
        """创建新标签页 / Create new tab"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            page = p.new_page()
            if url:
                page.goto(url)
            return {
                "success": True,
                "url": page.url,
                "title": page.title(),
            }

        return self._run_in_sandbox(inner)

    @tool(
        name="browser_tabs_select",
        description=(
            "Switch to the tab at the specified index. Index starts from 0, use"
            " browser_tabs_list to get all tab indices. After switching, all"
            " subsequent operations will be performed on the selected tab."
        ),
    )
    def browser_tabs_select(self, index: int) -> Dict[str, Any]:
        """切换标签页 / Switch tab"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            page = p.select_tab(index)
            return {
                "success": True,
                "index": index,
                "url": page.url,
                "title": page.title(),
            }

        return self._run_in_sandbox(inner)

    # ==================== 高级功能 / Advanced Features ====================

    @tool(
        name="browser_evaluate",
        description=(
            "Execute JavaScript code in the page context and return the result."
            " Can access page DOM, window object, etc. Suitable for getting"
            " page data, performing complex operations, or calling page"
            " functions. Return value is automatically serialized to"
            " JSON-compatible format."
        ),
    )
    def browser_evaluate(
        self,
        expression: str,
        arg: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """执行 JavaScript / Execute JavaScript"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            result = p.evaluate(expression, arg=arg)
            return {"result": result}

        return self._run_in_sandbox(inner)

    # 保留旧的 evaluate 作为别名
    @tool(
        name="evaluate",
        description=(
            "执行 JavaScript（browser_evaluate 的别名，保持向后兼容）。 /"
            " Execute JavaScript (alias for browser_evaluate, for backward"
            " compatibility)."
        ),
    )
    def evaluate(self, expression: str) -> Dict[str, Any]:
        """执行 JavaScript / Execute JavaScript"""
        return self.browser_evaluate(expression=expression)

    @tool(
        name="browser_wait_for",
        description=(
            "Wait for the specified time (in milliseconds). Used to wait for"
            " page animations, AJAX requests, and other async operations. Note:"
            " Prefer more precise wait conditions (like waiting for element),"
            " fixed time wait should be a last resort."
        ),
    )
    def browser_wait_for(self, timeout: float) -> Dict[str, Any]:
        """等待指定时间 / Wait for specified time"""

        def inner(sb: Sandbox):
            assert isinstance(sb, BrowserSandbox)
            p = self._get_playwright(sb)
            p.wait(timeout)
            return {"success": True, "waited_ms": timeout}

        return self._run_in_sandbox(inner)


def sandbox_toolset(
    template_name: str,
    *,
    template_type: TemplateType = TemplateType.CODE_INTERPRETER,
    config: Optional[Config] = None,
    sandbox_idle_timeout_seconds: int = 5 * 60,
    oss_mount_config: Optional["OSSMountConfig"] = None,
    nas_config: Optional["NASConfig"] = None,
    polar_fs_config: Optional["PolarFsConfig"] = None,
) -> CommonToolSet:
    """将沙箱模板封装为 LangChain ``StructuredTool`` 列表。"""

    if template_type != TemplateType.CODE_INTERPRETER:
        return BrowserToolSet(
            template_name=template_name,
            config=config,
            sandbox_idle_timeout_seconds=sandbox_idle_timeout_seconds,
            oss_mount_config=oss_mount_config,
            nas_config=nas_config,
            polar_fs_config=polar_fs_config,
        )
    else:
        return CodeInterpreterToolSet(
            template_name=template_name,
            config=config,
            sandbox_idle_timeout_seconds=sandbox_idle_timeout_seconds,
            oss_mount_config=oss_mount_config,
            nas_config=nas_config,
            polar_fs_config=polar_fs_config,
        )
