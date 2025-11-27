"""浏览器沙箱数据API模板 / Browser Sandbox Data API Template

此模板用于生成浏览器沙箱数据API代码。
This template is used to generate browser sandbox data API code.
"""

from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

from agentrun.utils.config import Config

from .sandbox_data import SandboxDataAPI


class BrowserDataAPI(SandboxDataAPI):

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
            >>> api = BrowserDataAPI("browser123", "session456")
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
            >>> api = BrowserDataAPI("browser123", "session456")
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
