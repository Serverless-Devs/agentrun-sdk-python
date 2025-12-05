"""All-in-One 沙箱数据API模板 / All-in-One Sandbox Data API Template

此模板用于生成 All-in-One 沙箱数据API代码，结合了浏览器和代码解释器的功能。
This template is used to generate All-in-One sandbox data API code,
combining browser and code interpreter capabilities.
"""

from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

from agentrun.sandbox.model import CodeLanguage
from agentrun.utils.config import Config

from .sandbox_data import SandboxDataAPI


class AioDataAPI(SandboxDataAPI):
    """All-in-One Sandbox Data API

    This class combines the functionality of BrowserDataAPI and CodeInterpreterDataAPI,
    providing a unified interface for all-in-one sandbox operations.
    """

    def __init__(
        self,
        sandbox_id: str,
        config: Optional[Config] = None,
    ):
        self.sandbox_id = sandbox_id
        super().__init__(
            sandbox_id=sandbox_id,
            config=config,
        )

    # ========================================
    # Browser API Methods
    # ========================================

    def get_cdp_url(self, record: Optional[bool] = False):
        """
        Generate the WebSocket URL for Chrome DevTools Protocol (CDP) connection.

        This method constructs a WebSocket URL by:
        1. Converting the HTTP endpoint to WebSocket protocol (ws://)
        2. Parsing the existing URL and query parameters
        3. Adding the session ID to the query parameters
        4. Reconstructing the complete WebSocket URL

        Returns:
            str: The complete WebSocket URL for CDP automation connection,
                 including the session ID in the query parameters.

        Example:
            >>> api = AioDataAPI("sandbox123")
            >>> api.get_cdp_url()
            'ws://example.com/ws/automation?sessionId=session456'
        """
        cdp_url = self.with_path("/ws/automation").replace("http", "ws")
        u = urlparse(cdp_url)
        query_dict = parse_qs(u.query)
        query_dict["tenantId"] = [self.config.get_account_id()]
        if record:
            query_dict["recording"] = ["true"]
        new_query = urlencode(query_dict, doseq=True)
        new_u = u._replace(query=new_query)
        return new_u.geturl()

    def get_vnc_url(self, record: Optional[bool] = False):
        """
        Generate the WebSocket URL for VNC (Virtual Network Computing) live view connection.

        This method constructs a WebSocket URL for real-time browser viewing by:
        1. Converting the HTTP endpoint to WebSocket protocol (ws://)
        2. Parsing the existing URL and query parameters
        3. Adding the session ID to the query parameters
        4. Reconstructing the complete WebSocket URL

        Returns:
            str: The complete WebSocket URL for VNC live view connection,
                 including the session ID in the query parameters.

        Example:
            >>> api = AioDataAPI("sandbox123")
            >>> api.get_vnc_url()
            'ws://example.com/ws/liveview?sessionId=session456'
        """
        vnc_url = self.with_path("/ws/liveview").replace("http", "ws")
        u = urlparse(vnc_url)
        query_dict = parse_qs(u.query)
        query_dict["tenantId"] = [self.config.get_account_id()]
        if record:
            query_dict["recording"] = ["true"]
        new_query = urlencode(query_dict, doseq=True)
        new_u = u._replace(query=new_query)
        return new_u.geturl()

    def sync_playwright(
        self,
        browser_type: str = "chrome",
        record: Optional[bool] = False,
        config: Optional[Config] = None,
    ):
        from .playwright_sync import BrowserPlaywrightSync

        cfg = Config.with_configs(self.config, config)
        _, headers, _ = self.auth(headers=cfg.get_headers(), config=cfg)
        return BrowserPlaywrightSync(
            self.get_cdp_url(record=record),
            browser_type=browser_type,
            headers=headers,
        )

    def async_playwright(
        self,
        browser_type: str = "chrome",
        record: Optional[bool] = False,
        config: Optional[Config] = None,
    ):
        from .playwright_async import BrowserPlaywrightAsync

        cfg = Config.with_configs(self.config, config)
        _, headers, _ = self.auth(headers=cfg.get_headers(), config=cfg)
        return BrowserPlaywrightAsync(
            self.get_cdp_url(record=record),
            browser_type=browser_type,
            headers=headers,
        )

    async def list_recordings_async(self):
        return await self.get_async("/recordings")

    async def delete_recording_async(self, filename: str):
        return await self.delete_async(f"/recordings/{filename}")

    async def download_recording_async(self, filename: str, save_path: str):
        """
        Asynchronously download a recording video file and save it to local path.

        Args:
            filename: The name of the recording file to download
            save_path: Local file path to save the downloaded video file (.mkv)

        Returns:
            Dictionary with 'saved_path' and 'size' keys

        Examples:
            >>> await api.download_recording_async("recording.mp4", "/local/video.mkv")
        """
        return await self.get_video_async(
            f"/recordings/{filename}", save_path=save_path
        )

    # ========================================
    # Code Interpreter API Methods
    # ========================================

    async def list_directory_async(
        self,
        path: Optional[str] = None,
        depth: Optional[int] = None,
    ):
        query = {}
        if path is not None:
            query["path"] = path
        if depth is not None:
            query["depth"] = depth

        return await self.get_async("/filesystem", query=query)

    async def stat_async(
        self,
        path: str,
    ):
        query = {
            "path": path,
        }
        return await self.get_async("/filesystem/stat", query=query)

    async def mkdir_async(
        self,
        path: str,
        parents: Optional[bool] = True,
        mode: Optional[str] = "0755",
    ):
        data = {
            "path": path,
            "parents": parents,
            "mode": mode,
        }
        return await self.post_async("/filesystem/mkdir", data=data)

    async def move_file_async(
        self,
        source: str,
        destination: str,
    ):
        data = {
            "source": source,
            "destination": destination,
        }
        return await self.post_async("/filesystem/move", data=data)

    async def remove_file_async(
        self,
        path: str,
    ):
        data = {
            "path": path,
        }
        return await self.post_async("/filesystem/remove", data=data)

    async def list_contexts_async(self):
        return await self.get_async("/contexts")

    async def create_context_async(
        self,
        language: Optional[CodeLanguage] = CodeLanguage.PYTHON,
        cwd: str = "/home/user",
    ):
        # Validate language parameter
        if language not in ("python", "javascript"):
            raise ValueError(
                f"language must be 'python' or 'javascript', got: {language}"
            )

        data: Dict[str, Any] = {
            "cwd": cwd,
            "language": language,
        }
        return await self.post_async("/contexts", data=data)

    async def get_context_async(
        self,
        context_id: str,
    ):
        return await self.get_async(f"/contexts/{context_id}")

    async def execute_code_async(
        self,
        code: str,
        context_id: Optional[str],
        language: Optional[CodeLanguage] = None,
        timeout: Optional[int] = 30,
    ):
        if language and language not in ("python", "javascript"):
            raise ValueError(
                f"language must be 'python' or 'javascript', got: {language}"
            )

        data: Dict[str, Any] = {
            "code": code,
        }
        if timeout is not None:
            data["timeout"] = timeout
        if language is not None:
            data["language"] = language
        if context_id is not None:
            data["contextId"] = context_id
        return await self.post_async(f"/contexts/execute", data=data)

    async def delete_context_async(
        self,
        context_id: str,
    ):
        return await self.delete_async(f"/contexts/{context_id}")

    async def read_file_async(
        self,
        path: str,
    ):
        query = {
            "path": path,
        }
        return await self.get_async("/files", query=query)

    async def write_file_async(
        self,
        path: str,
        content: str,
        mode: Optional[str] = "644",
        encoding: Optional[str] = "utf-8",
        create_dir: Optional[bool] = True,
    ):
        data = {
            "path": path,
            "content": content,
            "mode": mode,
            "encoding": encoding,
            "createDir": create_dir,
        }
        return await self.post_async("/files", data=data)

    async def upload_file_async(
        self,
        local_file_path: str,
        target_file_path: str,
    ):
        return await self.post_file_async(
            path="/filesystem/upload",
            local_file_path=local_file_path,
            target_file_path=target_file_path,
        )

    async def download_file_async(
        self,
        path: str,
        save_path: str,
    ):
        query = {"path": path}
        return await self.get_file_async(
            path="/filesystem/download", save_path=save_path, query=query
        )

    async def cmd_async(
        self,
        command: str,
        cwd: str,
        timeout: Optional[int] = 30,
    ):
        data: Dict[str, Any] = {
            "command": command,
            "cwd": cwd,
        }
        if timeout is not None:
            data["timeout"] = timeout
        return await self.post_async("/processes/cmd", data=data)

    async def list_processes_async(self):
        return await self.get_async("/processes")

    async def get_process_async(self, pid: str):
        return await self.get_async(f"/processes/{pid}")

    async def kill_process_async(self, pid: str):
        return await self.delete_async(f"/processes/{pid}")
