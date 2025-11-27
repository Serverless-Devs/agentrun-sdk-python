"""代码解释器沙箱数据API模板 / Code Interpreter Sandbox Data API Template

此模板用于生成代码解释器沙箱数据API代码。
This template is used to generate code interpreter sandbox data API code.
"""

from typing import Any, Dict, Optional

from agentrun.sandbox.model import CodeLanguage
from agentrun.utils.config import Config

from .sandbox_data import SandboxDataAPI


class CodeInterpreterDataAPI(SandboxDataAPI):

    def __init__(
        self,
        sandbox_id: str,
        config: Optional[Config] = None,
    ):

        super().__init__(
            sandbox_id=sandbox_id,
            config=config,
        )

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
