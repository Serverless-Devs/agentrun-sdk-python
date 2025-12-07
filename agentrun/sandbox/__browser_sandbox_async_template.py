"""浏览器沙箱高层API模板 / Browser Sandbox High-Level API Template

此模板用于生成浏览器沙箱资源的高级API代码。
This template is used to generate high-level API code for browser sandbox resources.
"""

import asyncio
import time  # noqa: F401
from typing import Optional

from agentrun.sandbox.api import BrowserDataAPI
from agentrun.sandbox.model import TemplateType
from agentrun.utils.log import logger

from .sandbox import Sandbox


class BrowserSandbox(Sandbox):
    _template_type = TemplateType.BROWSER

    _data_api: Optional["BrowserDataAPI"] = None

    async def __aenter__(self):
        # Poll health check asynchronously
        max_retries = 60  # Maximum 60 seconds
        retry_count = 0

        logger.debug("Waiting for browser to be ready...")

        while retry_count < max_retries:
            retry_count += 1

            try:
                health = await self.check_health_async()

                if health["status"] == "ok":
                    logger.debug(
                        f"✓ Browser is ready! (took {retry_count} seconds)"
                    )
                    return self

                logger.debug(
                    f"[{retry_count}/{max_retries}] Health status:"
                    f" {health.get('code')} {health.get('message')}",
                )

            except Exception as e:
                logger.error(
                    f"[{retry_count}/{max_retries}] Health check failed: {e}"
                )

            if retry_count < max_retries:
                await asyncio.sleep(1)

        raise RuntimeError(
            f"Health check timeout after {max_retries} seconds. "
            "Browser did not become ready in time."
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.sandbox_id is None:
            raise ValueError("Sandbox ID is not set")
        logger.debug(f"Deleting browser sandbox {self.sandbox_id}...")
        await self.delete_async()

    @property
    def data_api(self):
        if self._data_api is None:
            if self.sandbox_id is None:
                raise ValueError("Sandbox ID is not set")

            self._data_api = BrowserDataAPI(
                sandbox_id=self.sandbox_id, config=self._config
            )

        return self._data_api

    async def check_health_async(self):
        return await self.data_api.check_health_async()

    def get_cdp_url(self, record: Optional[bool] = False):
        return self.data_api.get_cdp_url(record=record)

    def get_vnc_url(self, record: Optional[bool] = False):
        return self.data_api.get_vnc_url(record=record)

    def sync_playwright(self, record: Optional[bool] = False):
        return self.data_api.sync_playwright(record=record)

    def async_playwright(self, record: Optional[bool] = False):
        return self.data_api.async_playwright(record=record)

    async def list_recordings_async(self):
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
        return await self.data_api.delete_recording_async(filename)
