"""Sandbox数据API模板 / Sandbox Data API Template

此模板用于生成沙箱数据API代码。
This template is used to generate sandbox data API code.
"""

from typing import Optional

from agentrun.utils.config import Config
from agentrun.utils.data_api import DataAPI, ResourceType


class SandboxDataAPI(DataAPI):

    def __init__(
        self,
        *,
        sandbox_id: Optional[str] = None,
        template_name: Optional[str] = None,
        config: Optional[Config] = None,
    ):

        super().__init__(
            resource_name="",
            resource_type=ResourceType.Template,
            namespace="sandboxes",
            config=config,
        )
        self.access_token_map = {}

        if sandbox_id or template_name:
            self.__refresh_access_token(
                sandbox_id=sandbox_id,
                template_name=template_name,
                config=config,
            )

    def __refresh_access_token(
        self,
        *,
        sandbox_id: Optional[str] = None,
        template_name: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        cfg = Config.with_configs(config, self.config)
        token = self.access_token_map.get(sandbox_id or template_name)
        if sandbox_id:
            self.resource_name = sandbox_id
            self.resource_type = ResourceType.Sandbox
            self.namespace = f"sandboxes/{sandbox_id}"
        else:
            self.resource_name = template_name
            self.resource_type = ResourceType.Template
            self.namespace = "sandboxes"

        if token:
            self.access_token = token
            return

        # 没有缓存过的 token

        self.access_token = None
        self.auth(config=cfg)
        self.access_token_map[sandbox_id or template_name] = self.access_token

    async def check_health_async(self):
        return await self.get_async("/health")

    async def create_sandbox_async(
        self,
        template_name: str,
        sandbox_idle_timeout_seconds: Optional[int] = 600,
        config: Optional[Config] = None,
    ):
        self.__refresh_access_token(template_name=template_name, config=config)
        return await self.post_async(
            "/",
            data={
                "templateName": template_name,
                "sandboxIdleTimeoutSeconds": sandbox_idle_timeout_seconds,
            },
        )

    async def delete_sandbox_async(
        self, sandbox_id: str, config: Optional[Config] = None
    ):
        self.__refresh_access_token(sandbox_id=sandbox_id, config=config)
        return await self.delete_async("/")

    async def stop_sandbox_async(
        self, sandbox_id: str, config: Optional[Config] = None
    ):
        self.__refresh_access_token(sandbox_id=sandbox_id, config=config)
        return await self.post_async("/stop")

    async def get_sandbox_async(
        self, sandbox_id: str, config: Optional[Config] = None
    ):
        self.__refresh_access_token(sandbox_id=sandbox_id, config=config)
        return await self.get_async("/")
