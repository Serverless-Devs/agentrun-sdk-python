"""Tool 管控链路 API / Tool Control API

通过底层 agentrun20250910 SDK 与平台交互，获取 Tool 资源。
Interacts with the platform via the agentrun20250910 SDK to get Tool resources.
"""

from typing import Dict, Optional

from alibabacloud_agentrun20250910.models import Tool as InnerTool
from alibabacloud_tea_openapi.exceptions._client import ClientException
from alibabacloud_tea_openapi.exceptions._server import ServerException
from darabonba.runtime import RuntimeOptions
import pydash

from agentrun.utils.config import Config
from agentrun.utils.control_api import ControlAPI
from agentrun.utils.exception import ClientError, ServerError
from agentrun.utils.log import logger


class ToolControlAPI(ControlAPI):
    """Tool 管控链路 API / Tool Control API"""

    def __init__(self, config: Optional[Config] = None):
        """初始化 API 客户端 / Initialize API client

        Args:
            config: 全局配置对象 / Global configuration object
        """
        super().__init__(config)

    def get_tool(
        self,
        name: str,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[Config] = None,
    ) -> InnerTool:
        """获取工具 / Get tool

        Args:
            name: Tool 名称 / Tool name
            headers: 请求头 / Request headers
            config: 配置 / Configuration

        Returns:
            InnerTool: 底层 SDK 的 Tool 对象 / Inner SDK Tool object

        Raises:
            ClientError: 客户端错误 / Client error
            ServerError: 服务器错误 / Server error
        """
        try:
            client = self._get_client(config)
            response = client.get_tool_with_options(
                name,
                headers=headers or {},
                runtime=RuntimeOptions(),
            )

            logger.debug(
                "request api get_tool, request Request ID:"
                f" {response.headers['x-acs-request-id'] if response.headers else ''}\n"
                f"  request: {[name]}\n  response: {response.body.data}"
            )

            return response.body.data
        except ClientException as e:
            raise ClientError(
                e.status_code,
                pydash.get(e, "data.message", pydash.get(e, "message", "")),
                request_id=e.request_id,
                request=[name],
            ) from e
        except ServerException as e:
            raise ServerError(
                e.status_code,
                pydash.get(e, "data.message", pydash.get(e, "message", "")),
                request_id=e.request_id,
            ) from e

    async def get_tool_async(
        self,
        name: str,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[Config] = None,
    ) -> InnerTool:
        """异步获取工具 / Get tool asynchronously

        Args:
            name: Tool 名称 / Tool name
            headers: 请求头 / Request headers
            config: 配置 / Configuration

        Returns:
            InnerTool: 底层 SDK 的 Tool 对象 / Inner SDK Tool object

        Raises:
            ClientError: 客户端错误 / Client error
            ServerError: 服务器错误 / Server error
        """
        try:
            client = self._get_client(config)
            response = await client.get_tool_with_options_async(
                name,
                headers=headers or {},
                runtime=RuntimeOptions(),
            )

            logger.debug(
                "request api get_tool, request Request ID:"
                f" {response.headers['x-acs-request-id'] if response.headers else ''}\n"
                f"  request: {[name]}\n  response: {response.body.data}"
            )

            return response.body.data
        except ClientException as e:
            raise ClientError(
                e.status_code,
                pydash.get(e, "data.message", pydash.get(e, "message", "")),
                request_id=e.request_id,
                request=[name],
            ) from e
        except ServerException as e:
            raise ServerError(
                e.status_code,
                pydash.get(e, "data.message", pydash.get(e, "message", "")),
                request_id=e.request_id,
            ) from e
