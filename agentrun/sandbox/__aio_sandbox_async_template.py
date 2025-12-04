"""All-in-One 沙箱高层API模板 / All-in-One Sandbox High-Level API Template

此模板用于生成 All-in-One 沙箱资源的高级API代码，结合了浏览器和代码解释器的功能。
This template is used to generate high-level API code for All-in-One sandbox resources,
combining browser and code interpreter capabilities.
"""

import asyncio
import time  # noqa: F401
from typing import Optional

from agentrun.sandbox.api.aio_data import AioDataAPI
from agentrun.sandbox.model import CodeLanguage, TemplateType
from agentrun.utils.exception import ServerError
from agentrun.utils.log import logger

from .sandbox import Sandbox

# ========================================
# Code Interpreter Helper Classes
# ========================================


class FileOperations:
    """File upload/download operations."""

    def __init__(self, sandbox: "AioSandbox"):
        self._sandbox = sandbox

    async def read_async(self, path: str):
        """Read a file from the sandbox (async).
        Args:
            path: Remote file path in the sandbox
        Returns:
            File content
        """
        return await self._sandbox.data_api.read_file_async(path=path)

    async def write_async(
        self,
        path: str,
        content: str,
        mode: str = "644",
        encoding: str = "utf-8",
        create_dir=True,
    ):
        """Write a file to the sandbox (async).
        Args:
            path: Remote file path in the sandbox
            content: File content
        """
        return await self._sandbox.data_api.write_file_async(
            path=path,
            content=content,
            mode=mode,
            encoding=encoding,
            create_dir=create_dir,
        )


class FileSystemOperations:
    """File system operations (list, move, remove, stat, mkdir)."""

    def __init__(self, sandbox: "AioSandbox"):
        self._sandbox = sandbox

    async def list_async(
        self, path: Optional[str] = None, depth: Optional[int] = None
    ):
        """List directory contents (async).

        Args:
            path: Directory path (optional)
            depth: Traversal depth (optional)

        Returns:
            Directory contents
        """
        return await self._sandbox.data_api.list_directory_async(
            path=path, depth=depth
        )

    async def move_async(self, source: str, destination: str):
        """Move a file or directory (async).

        Args:
            source: Source file or directory path
            destination: Target file or directory path

        Returns:
            Move operation result
        """
        return await self._sandbox.data_api.move_file_async(
            source=source, destination=destination
        )

    async def remove_async(self, path: str):
        """Remove a file or directory (async).

        Args:
            path: File or directory path to remove

        Returns:
            Remove operation result
        """
        return await self._sandbox.data_api.remove_file_async(path=path)

    async def stat_async(self, path: str):
        """Get file or directory statistics (async).

        Args:
            path: File or directory path

        Returns:
            File/directory statistics
        """
        return await self._sandbox.data_api.stat_async(path=path)

    async def mkdir_async(
        self,
        path: str,
        parents: Optional[bool] = True,
        mode: Optional[str] = "0755",
    ):
        """Create a directory (async).

        Args:
            path: Directory path to create
            parents: Whether to create parent directories (default: True)
            mode: Directory permissions mode (default: "0755")

        Returns:
            Mkdir operation result
        """
        return await self._sandbox.data_api.mkdir_async(
            path=path, parents=parents, mode=mode
        )

    async def upload_async(
        self,
        local_file_path: str,
        target_file_path: str,
    ):
        """Upload a file to the sandbox (async).

        Args:
            local_file_path: Local file path to upload
            target_file_path: Target file path in sandbox

        Returns:
            Upload result
        """
        return await self._sandbox.data_api.upload_file_async(
            local_file_path=local_file_path, target_file_path=target_file_path
        )

    async def download_async(self, path: str, save_path: str):
        """Download a file from the sandbox (async).

        Args:
            path: Remote file path in the sandbox
            save_path: Local file path to save the downloaded file

        Returns:
            Download result with 'saved_path' and 'size'
        """
        return await self._sandbox.data_api.download_file_async(
            path=path, save_path=save_path
        )


class ProcessOperations:
    """Process management operations."""

    def __init__(self, sandbox: "AioSandbox"):
        self._sandbox = sandbox

    async def cmd_async(
        self, command: str, cwd: str, timeout: Optional[int] = 30
    ):
        """Execute a command in the sandbox (async).

        Args:
            command: Command to execute
            cwd: Working directory
            timeout: Execution timeout in seconds (default: 30)

        Returns:
            Command execution result
        """
        return await self._sandbox.data_api.cmd_async(
            command=command, cwd=cwd, timeout=timeout
        )

    async def list_async(self):
        """List all processes (async).

        Returns:
            List of processes
        """
        return await self._sandbox.data_api.list_processes_async()

    async def get_async(self, pid: str):
        """Get a specific process by PID (async).

        Args:
            pid: Process ID

        Returns:
            Process information
        """
        return await self._sandbox.data_api.get_process_async(pid=pid)

    async def kill_async(self, pid: str):
        """Kill a specific process by PID (async).

        Args:
            pid: Process ID

        Returns:
            Kill operation result
        """
        return await self._sandbox.data_api.kill_process_async(pid=pid)


class ContextOperations:
    """Context management operations."""

    def __init__(self, sandbox: "AioSandbox"):
        self._sandbox = sandbox
        self._context_id: Optional[str] = None
        self._language: Optional[str] = None
        self._cwd: Optional[str] = None

    @property
    def context_id(self) -> Optional[str]:
        """Get the current context ID."""
        return self._context_id

    async def list_async(self):
        """List all contexts (async)."""
        return await self._sandbox.data_api.list_contexts_async()

    async def create_async(
        self,
        language: Optional[CodeLanguage] = CodeLanguage.PYTHON,
        cwd: str = "/home/user",
    ) -> "ContextOperations":
        """Create a new context and save its ID (async).

        Args:
            language: Programming language (default: "python")
            cwd: Working directory (default: "/home/user")

        Returns:
            ContextOperations: Returns self for chaining and context manager support
        """
        result = await self._sandbox.data_api.create_context_async(
            language=language, cwd=cwd
        )
        if all(result.get(key) for key in ("id", "cwd", "language")):
            self._context_id = result["id"]
            self._language = result["language"]
            self._cwd = result["cwd"]
            return self
        raise ServerError(500, "Failed to create context")

    async def get_async(
        self, context_id: Optional[str] = None
    ) -> "ContextOperations":
        """Get a specific context by ID (async).
        Args:
            context_id: Context ID
        Returns:
            ContextOperations: Returns self for chaining and context manager support
        """
        if context_id is None:
            context_id = self._context_id
        if context_id is None:
            logger.error(f"context id is not set")
            raise ValueError("context id is not set,")
        result = await self._sandbox.data_api.get_context_async(
            context_id=context_id
        )
        if all(result.get(key) for key in ("id", "cwd", "language")):
            self._context_id = result["id"]
            self._language = result["language"]
            self._cwd = result["cwd"]
            return self
        raise ServerError(500, "Failed to create context")

    async def execute_async(
        self,
        code: str,
        language: Optional[CodeLanguage] = None,
        context_id: Optional[str] = None,
        timeout: Optional[int] = 30,
    ):
        """Execute code in a context (async).

        Args:
            code: Code to execute
            language: Programming language (optional)
            context_id: Context ID (optional, uses saved context_id if not provided)
            timeout: Execution timeout in seconds (default: 30)

        Returns:
            Execution result

        Raises:
            ValueError: If no context_id is provided and none is saved
        """
        if context_id is None:
            context_id = self._context_id
        if context_id is None and language is None:
            logger.debug("context id is not set, use default language: python")
            language = CodeLanguage.PYTHON
        return await self._sandbox.data_api.execute_code_async(
            context_id=context_id, language=language, code=code, timeout=timeout
        )

    async def delete_async(self, context_id: Optional[str] = None):
        """Delete a context (async).

        Args:
            context_id: Context ID (optional, uses saved context_id if not provided)

        Returns:
            Delete result

        Raises:
            ValueError: If no context_id is provided and none is saved
        """
        if context_id is None:
            context_id = self._context_id
        if context_id is None:
            raise ValueError(
                "context_id is required. Either pass it as parameter or create"
                " a context first."
            )
        result = await self._sandbox.data_api.delete_context_async(
            context_id=context_id
        )
        # Clear the saved context_id after deletion
        self._context_id = None
        return result

    async def __aenter__(self):
        """Asynchronous context manager entry."""
        if self._context_id is None:
            raise ValueError(
                "No context has been created. Call create() first or use: "
                "async with await sandbox.context.create_async(...) as ctx:"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Asynchronous context manager exit - deletes the context."""
        if self._context_id is not None:
            try:
                await self._sandbox.data_api.delete_context_async(
                    self._context_id
                )
            except Exception as e:
                logger.error(
                    f"Warning: Failed to delete context {self._context_id}: {e}"
                )
        return False


class AioSandbox(Sandbox):
    """All-in-One Sandbox combining Browser and Code Interpreter capabilities.

    This class combines the functionality of BrowserSandbox and CodeInterpreterSandbox,
    providing a unified interface for all-in-one sandbox operations.
    """

    _template_type = TemplateType.AIO

    _data_api: Optional["AioDataAPI"] = None
    _file: Optional[FileOperations] = None
    _file_system: Optional[FileSystemOperations] = None
    _context: Optional[ContextOperations] = None
    _process: Optional[ProcessOperations] = None

    async def __aenter__(self):
        """Asynchronous context manager entry."""
        # Poll health check asynchronously
        max_retries = 60  # Maximum 60 seconds
        retry_count = 0

        logger.debug("Waiting for All-in-One sandbox to be ready...")

        while retry_count < max_retries:
            retry_count += 1

            try:
                health = await self.check_health_async()

                if health["status"] == "ok":
                    logger.debug(
                        "✓ All-in-One sandbox is ready! (took"
                        f" {retry_count} seconds)"
                    )
                    return self

                logger.debug(
                    "[%d/%d] Health status: %d %s",
                    retry_count,
                    max_retries,
                    health.get("code", 0),
                    health.get("message", "not ready"),
                )

            except Exception as e:
                logger.error(
                    f"[{retry_count}/{max_retries}] Health check failed: {e}"
                )

            if retry_count < max_retries:
                await asyncio.sleep(1)

        raise RuntimeError(
            f"Health check timeout after {max_retries} seconds. "
            "All-in-One sandbox did not become ready in time."
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Asynchronous context manager exit."""
        if self.sandbox_id is None:
            raise ValueError("Sandbox ID is not set")
        logger.debug(f"Deleting All-in-One sandbox {self.sandbox_id}...")
        await self.delete_async()

    @property
    def data_api(self) -> "AioDataAPI":
        """Get data client."""
        if self._data_api is None:
            if self.sandbox_id is None:
                raise ValueError("Sandbox ID is not set")

            self._data_api = AioDataAPI(
                sandbox_id=self.sandbox_id, config=self._config
            )

        return self._data_api

    async def check_health_async(self):
        """Check sandbox health status (async)."""
        return await self.data_api.check_health_async()

    # ========================================
    # Browser API Methods
    # ========================================

    def get_cdp_url(self, record: Optional[bool] = False):
        """Get CDP WebSocket URL for browser automation."""
        return self.data_api.get_cdp_url(record=record)

    def get_vnc_url(self, record: Optional[bool] = False):
        """Get VNC WebSocket URL for live view."""
        return self.data_api.get_vnc_url(record=record)

    def sync_playwright(self, record: Optional[bool] = False):
        """Get synchronous Playwright browser instance."""
        return self.data_api.sync_playwright(record=record)

    def async_playwright(self, record: Optional[bool] = False):
        """Get asynchronous Playwright browser instance."""
        return self.data_api.async_playwright(record=record)

    async def list_recordings_async(self):
        """List all recordings (async)."""
        return await self.data_api.list_recordings_async()

    async def download_recording_async(self, filename: str, save_path: str):
        """
        Asynchronously download a recording video file and save it to local path.

        Args:
            filename: The name of the recording file to download
            save_path: Local file path to save the downloaded video file (.mkv)

        Returns:
            Dictionary with 'saved_path' and 'size' keys
        """
        return await self.data_api.download_recording_async(filename, save_path)

    async def delete_recording_async(self, filename: str):
        """Delete a recording file (async)."""
        return await self.data_api.delete_recording_async(filename)

    # ========================================
    # Code Interpreter API Properties
    # ========================================

    @property
    def file(self) -> FileOperations:
        """Access file upload/download operations."""
        if self._file is None:
            self._file = FileOperations(self)
        return self._file

    @property
    def file_system(self) -> FileSystemOperations:
        """Access file system operations."""
        if self._file_system is None:
            self._file_system = FileSystemOperations(self)
        return self._file_system

    @property
    def context(self) -> ContextOperations:
        """Access context management operations."""
        if self._context is None:
            self._context = ContextOperations(self)
        return self._context

    @property
    def process(self) -> ProcessOperations:
        """Access process management operations."""
        if self._process is None:
            self._process = ProcessOperations(self)
        return self._process
