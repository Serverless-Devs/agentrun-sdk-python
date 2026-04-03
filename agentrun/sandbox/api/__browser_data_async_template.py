"""浏览器沙箱数据API模板 / Browser Sandbox Data API Template

此模板用于生成浏览器沙箱数据API代码。
This template is used to generate browser sandbox data API code.
"""

from typing import Dict, Literal, Optional, overload, Tuple, Union
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

    def _assemble_ws_url(
        self, base: str, ws_path: str, record: Optional[bool] = False
    ) -> str:
        path = ws_path.lstrip("/")
        raw = "/".join(
            part.strip("/") for part in [base, self.namespace, path] if part
        )
        ws_url = raw.replace("http", "ws")
        u = urlparse(ws_url)
        query_dict = parse_qs(u.query)
        query_dict["tenantId"] = [self.config.get_account_id()]
        if record:
            query_dict["recording"] = ["true"]
        new_query = urlencode(query_dict, doseq=True)
        return u._replace(query=new_query).geturl()

    def _build_ws_url(
        self,
        ws_path: str,
        record: Optional[bool] = False,
        config: Optional[Config] = None,
    ) -> str:
        cfg = Config.with_configs(self.config, config)
        return self._assemble_ws_url(cfg.get_data_endpoint(), ws_path, record)

    def _build_ws_url_with_headers(
        self,
        ws_path: str,
        record: Optional[bool] = False,
        config: Optional[Config] = None,
    ) -> Tuple[str, Dict[str, str]]:
        cfg = Config.with_configs(self.config, config)
        url = self._assemble_ws_url(self.get_base_url(cfg), ws_path, record)
        url, headers, _ = self.auth(
            url=url, headers=cfg.get_headers(), config=cfg
        )
        return url, headers

    @overload
    def get_cdp_url(
        self,
        record: Optional[bool] = False,
        *,
        with_headers: Literal[True],
        config: Optional[Config] = None,
    ) -> Tuple[str, Dict[str, str]]:
        ...

    @overload
    def get_cdp_url(
        self,
        record: Optional[bool] = False,
        *,
        with_headers: Literal[False] = False,
        config: Optional[Config] = None,
    ) -> str:
        ...

    def get_cdp_url(
        self,
        record: Optional[bool] = False,
        *,
        with_headers: bool = False,
        config: Optional[Config] = None,
    ) -> Union[str, Tuple[str, Dict[str, str]]]:
        """
        Generate the WebSocket URL for Chrome DevTools Protocol (CDP) connection.
        生成 Chrome DevTools Protocol (CDP) 连接的 WebSocket URL。

        Args:
            record: Whether to enable recording / 是否启用录制
            with_headers: If True, return (url, headers) tuple with authentication headers.
                If False (default), return only the URL string for backward compatibility.
                当为 True 时，返回 (url, headers) 元组，包含鉴权头信息。
                当为 False（默认）时，仅返回 URL 字符串以保持向后兼容。
            config: Optional config override / 可选的配置覆盖

        Returns:
            str or Tuple[str, Dict[str, str]]: CDP WebSocket URL, or (url, headers) tuple
                when with_headers=True.

        Example:
            >>> api = BrowserDataAPI("browser123")
            >>> api.get_cdp_url()
            'wss://example.com/sandboxes/browser123/ws/automation?tenantId=123'
            >>> url, headers = api.get_cdp_url(with_headers=True)
        """
        if with_headers:
            return self._build_ws_url_with_headers(
                "/ws/automation", record=record, config=config
            )
        return self._build_ws_url("/ws/automation", record=record)

    @overload
    def get_vnc_url(
        self,
        record: Optional[bool] = False,
        *,
        with_headers: Literal[True],
        config: Optional[Config] = None,
    ) -> Tuple[str, Dict[str, str]]:
        ...

    @overload
    def get_vnc_url(
        self,
        record: Optional[bool] = False,
        *,
        with_headers: Literal[False] = False,
        config: Optional[Config] = None,
    ) -> str:
        ...

    def get_vnc_url(
        self,
        record: Optional[bool] = False,
        *,
        with_headers: bool = False,
        config: Optional[Config] = None,
    ) -> Union[str, Tuple[str, Dict[str, str]]]:
        """
        Generate the WebSocket URL for VNC (Virtual Network Computing) live view connection.
        生成 VNC 实时预览连接的 WebSocket URL。

        Args:
            record: Whether to enable recording / 是否启用录制
            with_headers: If True, return (url, headers) tuple with authentication headers.
                If False (default), return only the URL string for backward compatibility.
                当为 True 时，返回 (url, headers) 元组，包含鉴权头信息。
                当为 False（默认）时，仅返回 URL 字符串以保持向后兼容。
            config: Optional config override / 可选的配置覆盖

        Returns:
            str or Tuple[str, Dict[str, str]]: VNC WebSocket URL, or (url, headers) tuple
                when with_headers=True.

        Example:
            >>> api = BrowserDataAPI("browser123")
            >>> api.get_vnc_url()
            'wss://example.com/sandboxes/browser123/ws/liveview?tenantId=123'
            >>> url, headers = api.get_vnc_url(with_headers=True)
        """
        if with_headers:
            return self._build_ws_url_with_headers(
                "/ws/liveview", record=record, config=config
            )
        return self._build_ws_url("/ws/liveview", record=record)

    def sync_playwright(
        self,
        browser_type: str = "chrome",
        record: Optional[bool] = False,
        config: Optional[Config] = None,
    ):
        from .playwright_sync import BrowserPlaywrightSync

        url, headers = self._build_ws_url_with_headers(
            "/ws/automation", record=record, config=config
        )
        return BrowserPlaywrightSync(
            url,
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

        url, headers = self._build_ws_url_with_headers(
            "/ws/automation", record=record, config=config
        )
        return BrowserPlaywrightAsync(
            url,
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
