"""AgentRun Data Client SDK / AgentRun 数据客户端 SDK

This module provides an async HTTP client for interacting with the AgentRun Data API.
此模块提供用于与 AgentRun Data API 交互的异步 HTTP 客户端。

It supports standard HTTP methods (GET, POST, PUT, PATCH, DELETE) with proper
error handling, type hints, and JSON serialization.
"""

from enum import Enum
from typing import Any, Dict, Optional, Union
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import httpx

from agentrun.utils.config import Config
from agentrun.utils.exception import ClientError
from agentrun.utils.helper import mask_password
from agentrun.utils.log import logger


class ResourceType(Enum):
    Runtime = "runtime"
    LiteLLM = "litellm"
    Tool = "tool"
    Template = "template"
    Sandbox = "sandbox"


class DataAPI:
    """
    Async HTTP client for AgentRun Data API.

    This client provides async methods for making HTTP requests to the AgentRun Data API
    with automatic URL construction, JSON handling, and error management.

    The client automatically manages HTTP sessions - no need for manual session management
    or context managers in simple use cases.
    """

    def __init__(
        self,
        resource_name: str,
        resource_type: ResourceType,
        config: Optional[Config] = None,
        namespace: str = "agents",
    ):
        """
        Initialize the AgentRun Data Client.

        Args:
            region: Aliyun region (default: "cn-hangzhou")
            protocol: Protocol to use (default: "https")
            account_id: Aliyun account ID
            version: API version (default: "2025-09-10")
            namespace: API namespace (default: "agents")
            timeout: Request timeout in seconds (default: 600)
            headers: Default headers to include in all requests
            auto_manage_session: Whether to automatically manage sessions (default: True)

        Raises:
            ValueError: If account_id is not provided or protocol is invalid
        """

        self.resource_name = resource_name
        self.resource_type = resource_type
        self.access_token = None

        self.config = Config.with_configs(config)
        self.namespace = namespace

        if self.config.get_token():
            logger.debug(
                "using provided access token from config, %s",
                mask_password(self.config.get_token() or ""),
            )
            self.access_token = self.config.get_token()

    def get_base_url(self) -> str:
        """
        Get the base URL for API requests.

        Returns:
            The base URL string
        """
        return self.config.get_data_endpoint()

    def with_path(
        self, path: str, query: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Construct full URL with the given path and query parameters.

        Args:
            path: API path (may include query string)
            query: Query parameters to add/merge

        Returns:
            Complete URL string with query parameters

        Examples:
            >>> client.with_path("resources")
            "http://account.agentrun-data.cn-hangzhou.aliyuncs.com/2025-09-10/agents/resources"

            >>> client.with_path("resources", {"limit": 10})
            "http://account.agentrun-data.cn-hangzhou.aliyuncs.com/2025-09-10/agents/resources?limit=10"

            >>> client.with_path("resources?page=1", {"limit": 10})
            "http://account.agentrun-data.cn-hangzhou.aliyuncs.com/2025-09-10/agents/resources?page=1&limit=10"
        """
        # Remove leading slash if present
        path = path.lstrip("/")
        base_url = "/".join([
            part.strip("/")
            for part in [
                self.get_base_url(),
                self.namespace,
                path,
            ]
            if part
        ])

        # If no query parameters, return the base URL
        if not query:
            return base_url

        # Parse the URL to handle existing query parameters
        parsed = urlparse(base_url)

        # Parse existing query parameters
        existing_params = parse_qs(parsed.query, keep_blank_values=True)

        # Merge with new query parameters
        # Convert new query dict to the same format as parse_qs (values as lists)
        for key, value in query.items():
            if isinstance(value, list):
                existing_params[key] = value
            else:
                existing_params[key] = [str(value)]

        # Flatten the parameters (convert lists to single values where appropriate)
        flattened_params = {}
        for key, value_list in existing_params.items():
            if len(value_list) == 1:
                flattened_params[key] = value_list[0]
            else:
                flattened_params[key] = value_list

        # Encode query string
        new_query = urlencode(flattened_params, doseq=True)

        # Reconstruct URL with new query string
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment,
        ))

    def auth(
        self,
        url: str = "",
        headers: Optional[Dict[str, str]] = None,
        query: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
    ) -> tuple[str, Dict[str, str], Optional[Dict[str, Any]]]:
        """
        Authentication hook for modifying requests before sending.

        This method can be overridden in subclasses to implement custom
        authentication logic (e.g., signing requests, adding auth tokens).

        Args:
            url: The request URL
            headers: The request headers
            query: The query parameters

        Returns:
            Tuple of (modified_url, modified_headers, modified_query)

        Examples:
            Override this method to add custom authentication:

            >>> class AuthedClient(AgentRunDataClient):
            ...     def auth(self, url, headers, query):
            ...         # Add auth token to headers
            ...         headers["Authorization"] = "Bearer token123"
            ...         # Or add signature to query
            ...         query = query or {}
            ...         query["signature"] = self._sign_request(url)
            ...         return url, headers, query
        """
        cfg = Config.with_configs(self.config, config)

        if (
            self.access_token is None
            and self.resource_name
            and self.resource_type
            and not cfg.get_token()
        ):
            try:
                from alibabacloud_agentrun20250910.models import (
                    GetAccessTokenRequest,
                )

                from .control_api import ControlAPI

                cli = ControlAPI(self.config)._get_client()
                input = (
                    GetAccessTokenRequest(
                        resource_id=self.resource_name,
                        resource_type=self.resource_type.value,
                    )
                    if self.resource_type == ResourceType.Sandbox
                    else GetAccessTokenRequest(
                        resource_name=self.resource_name,
                        resource_type=self.resource_type.value,
                    )
                )

                resp = cli.get_access_token(input)
                self.access_token = resp.body.data.access_token

            except Exception as e:
                logger.warning(
                    "Failed to get access token for"
                    f" {self.resource_type}({self.resource_name}): {e}"
                )

            logger.debug(
                "fetching access token for resource %s of type %s, %s",
                self.resource_name,
                self.resource_type,
                mask_password(self.access_token or ""),
            )
        headers = {
            "Agentrun-Access-Token": cfg.get_token() or self.access_token or "",
            **cfg.get_headers(),
            **(headers or {}),
        }

        return url, headers, query

    def _prepare_request(
        self,
        method: str,
        url: str,
        data: Optional[Union[Dict[str, Any], str]] = None,
        headers: Optional[Dict[str, str]] = None,
        query: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
    ):
        req_headers = {
            "Content-Type": "application/json",
            "User-Agent": "AgentRunDataClient/1.0",
        }

        # Merge with instance default headers
        cfg = Config.with_configs(self.config, config)

        req_headers.update(cfg.get_headers())

        # Merge request-specific headers
        if headers:
            req_headers.update(headers)

        # Apply authentication (may modify URL, headers, and query)
        url, req_headers, query = self.auth(url, req_headers, query, config=cfg)

        # Add query parameters to URL if provided
        if query:
            parsed = urlparse(url)
            existing_params = parse_qs(parsed.query, keep_blank_values=True)

            # Merge query parameters
            for key, value in query.items():
                if isinstance(value, list):
                    existing_params[key] = value
                else:
                    existing_params[key] = [str(value)]

            # Flatten and encode
            flattened_params = {}
            for key, value_list in existing_params.items():
                if len(value_list) == 1:
                    flattened_params[key] = value_list[0]
                else:
                    flattened_params[key] = value_list

            new_query = urlencode(flattened_params, doseq=True)
            url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment,
            ))

        # Prepare request body
        req_json = None
        req_content = None

        if data is not None:
            if isinstance(data, dict):
                req_json = data
            elif isinstance(data, str):
                req_content = data
            else:
                req_content = str(data)

        logger.debug(
            "%s %s headers=%s, json=%s, content=%s",
            method,
            url,
            req_headers,
            req_json,
            req_content,
        )
        return method, url, req_headers, req_json, req_content

    async def _make_request_async(
        self,
        method: str,
        url: str,
        data: Optional[Union[Dict[str, Any], str]] = None,
        headers: Optional[Dict[str, str]] = None,
        query: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Make a sync HTTP request using httpx.

        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE)
            url: Full URL to request
            data: Request body (dict or string)
            headers: Additional headers to include
            query: Query parameters to add to URL

        Returns:
            Response body as dictionary

        Raises:
            AgentRunClientError: If the request fails
        """
        method, url, req_headers, req_json, req_content = self._prepare_request(
            method, url, data, headers, query, config=config
        )

        try:
            async with httpx.AsyncClient(
                timeout=self.config.get_timeout()
            ) as client:
                response = await client.request(
                    method,
                    url,
                    headers=req_headers,
                    json=req_json,
                    content=req_content,
                )

                # # Raise for HTTP error status codes
                # response.raise_for_status()

                response_text = response.text
                logger.debug(f"Response: {response_text}")

                # Parse JSON response
                if response_text:
                    try:
                        return response.json()
                    except ValueError as e:
                        error_msg = f"Failed to parse JSON response: {e}"
                        bad_gateway_error_message = "502 Bad Gateway"
                        if response.status_code == 502 and (
                            bad_gateway_error_message in response_text
                        ):
                            error_msg = bad_gateway_error_message
                        logger.error(error_msg)
                        raise ClientError(
                            status_code=response.status_code, message=error_msg
                        ) from e

                return {}

        except httpx.HTTPStatusError as e:
            # HTTP error response
            error_text = ""
            try:
                error_text = e.response.text
            except Exception:
                error_text = str(e)

            error_msg = (
                f"HTTP {e.response.status_code} error:"
                f" {error_text or e.response.reason_phrase}"
            )
            raise ClientError(
                status_code=e.response.status_code, message=error_msg
            ) from e

        except httpx.RequestError as e:
            error_msg = f"Request error: {e!s}"
            raise ClientError(status_code=0, message=error_msg) from e

    async def get_async(
        self,
        path: str,
        query: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Make an async GET request.

        Args:
            path: API path (may include query string)
            query: Query parameters to add/merge
            headers: Additional headers

        Returns:
            Response body as dictionary

        Raises:
            AgentRunClientError: If the request fails

        Examples:
            >>> await client.get("resources")
            >>> await client.get("resources", query={"limit": 10, "page": 1})
            >>> await client.get("resources?status=active", query={"limit": 10})
        """
        return await self._make_request_async(
            "GET",
            self.with_path(path, query=query),
            headers=headers,
            config=config,
        )

    async def post_async(
        self,
        path: str,
        data: Optional[Union[Dict[str, Any], str]] = None,
        query: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Make an async POST request.

        Args:
            path: API path (may include query string)
            data: Request body
            query: Query parameters to add/merge
            headers: Additional headers

        Returns:
            Response body as dictionary

        Raises:
            AgentRunClientError: If the request fails

        Examples:
            >>> await client.post("resources", data={"name": "test"})
            >>> await client.post("resources", data={"name": "test"}, query={"async": "true"})
        """

        return await self._make_request_async(
            "POST",
            self.with_path(path, query=query),
            data=data,
            headers=headers,
            config=config,
        )

    async def put_async(
        self,
        path: str,
        data: Optional[Union[Dict[str, Any], str]] = None,
        query: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Make an async PUT request.

        Args:
            path: API path (may include query string)
            data: Request body
            query: Query parameters to add/merge
            headers: Additional headers

        Returns:
            Response body as dictionary

        Raises:
            AgentRunClientError: If the request fails
        """
        return await self._make_request_async(
            "PUT",
            self.with_path(path, query=query),
            data=data,
            headers=headers,
            config=config,
        )

    async def patch_async(
        self,
        path: str,
        data: Optional[Union[Dict[str, Any], str]] = None,
        query: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Make an async PATCH request.

        Args:
            path: API path (may include query string)
            data: Request body
            query: Query parameters to add/merge
            headers: Additional headers

        Returns:
            Response body as dictionary

        Raises:
            AgentRunClientError: If the request fails
        """
        return await self._make_request_async(
            "PATCH",
            self.with_path(path, query=query),
            data=data,
            headers=headers,
            config=config,
        )

    async def delete_async(
        self,
        path: str,
        query: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Make an async DELETE request.

        Args:
            path: API path (may include query string)
            query: Query parameters to add/merge
            headers: Additional headers

        Returns:
            Response body as dictionary

        Raises:
            AgentRunClientError: If the request fails
        """
        return await self._make_request_async(
            "DELETE",
            self.with_path(path, query=query),
            headers=headers,
            config=config,
        )

    async def post_file_async(
        self,
        path: str,
        local_file_path: str,
        target_file_path: str,
        form_data: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously upload a file using multipart/form-data (POST request).

        Args:
            path: API path (may include query string)
            local_file_path: Local file path to upload
            target_file_path: Target file path on the server
            form_data: Additional form data fields
            query: Query parameters to add/merge
            headers: Additional headers

        Returns:
            Response body as dictionary

        Raises:
            AgentRunClientError: If the request fails

        Examples:
            >>> await client.post_file_async("/files", local_file_path="/local/data.csv", target_file_path="/remote/data.csv")
            >>> await client.post_file_async("/files", local_file_path="/local/data.csv", target_file_path="/remote/input.csv")
        """
        import os

        filename = os.path.basename(local_file_path)

        url = self.with_path(path, query=query)
        req_headers = self.config.get_headers()
        req_headers.update(headers or {})

        try:
            with open(local_file_path, "rb") as f:
                file_content = f.read()
                files = {"file": (filename, file_content)}
                data = form_data or {}
                data["path"] = target_file_path

                async with httpx.AsyncClient(
                    timeout=self.config.get_timeout()
                ) as client:
                    response = await client.post(
                        url, files=files, data=data, headers=req_headers
                    )
                    response.raise_for_status()
                    return response.json()
        except httpx.HTTPStatusError as e:
            raise ClientError(
                status_code=e.response.status_code, message=e.response.text
            ) from e

    async def get_file_async(
        self,
        path: str,
        save_path: str,
        query: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously download a file and save it to local path (GET request).

        Args:
            path: API path (may include query string)
            save_path: Local file path to save the downloaded file
            query: Query parameters to add/merge
            headers: Additional headers

        Returns:
            Dictionary with 'saved_path' and 'size' keys

        Raises:
            AgentRunClientError: If the request fails

        Examples:
            >>> await client.get_file_async("/files", save_path="/local/data.csv", query={"path": "/remote/file.csv"})
        """
        url = self.with_path(path, query=query)
        req_headers = self.config.get_headers()
        req_headers.update(headers or {})

        try:
            async with httpx.AsyncClient(
                timeout=self.config.get_timeout()
            ) as client:
                response = await client.get(url, headers=req_headers)
                response.raise_for_status()

                with open(save_path, "wb") as f:
                    f.write(response.content)

                return {"saved_path": save_path, "size": len(response.content)}
        except httpx.HTTPStatusError as e:
            raise ClientError(
                status_code=e.response.status_code, message=e.response.text
            ) from e

    async def get_video_async(
        self,
        path: str,
        save_path: str,
        query: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously download a video file and save it to local path (GET request).

        Args:
            path: API path (may include query string)
            save_path: Local file path to save the downloaded video file (.mkv)
            query: Query parameters to add/merge
            headers: Additional headers

        Returns:
            Dictionary with 'saved_path' and 'size' keys

        Raises:
            AgentRunClientError: If the request fails

        Examples:
            >>> await client.get_video_async("/videos", save_path="/local/video.mkv", query={"path": "/remote/video.mp4"})
        """
        url = self.with_path(path, query=query)
        req_headers = self.config.get_headers()
        req_headers.update(headers or {})
        # Apply authentication (may modify URL, headers, and query)
        cfg = Config.with_configs(self.config, config)
        url, req_headers, query = self.auth(url, req_headers, query, config=cfg)

        try:
            async with httpx.AsyncClient(
                timeout=self.config.get_timeout()
            ) as client:
                response = await client.get(url, headers=req_headers)
                response.raise_for_status()

                with open(save_path, "wb") as f:
                    f.write(response.content)

                return {"saved_path": save_path, "size": len(response.content)}
        except httpx.HTTPStatusError as e:
            raise ClientError(
                status_code=e.response.status_code, message=e.response.text
            ) from e
