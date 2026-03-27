"""Tool OpenAPI 数据链路 / Tool OpenAPI Data API

解析 FunctionCall 类型 Tool 的 protocol_spec（OpenAPI JSON），
提取 operations 转换为 ToolInfo 列表，并通过 Server URL 发起 HTTP 调用。
Parses protocol_spec (OpenAPI JSON) for FunctionCall type Tools,
extracts operations as ToolInfo list, and makes HTTP calls via Server URL.
"""

import json
from typing import Any, Dict, List, Optional

import httpx

from agentrun.tool.model import ToolInfo, ToolSchema
from agentrun.utils.log import logger


class ToolOpenAPIClient:
    """FunctionCall 类型 Tool 的 OpenAPI 客户端 / OpenAPI Client for FunctionCall Tools

    解析 protocol_spec 中的 OpenAPI Schema，提供 list_tools 和 call_tool 能力。
    Parses OpenAPI Schema from protocol_spec, provides list_tools and call_tool capabilities.
    """

    def __init__(
        self,
        protocol_spec: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        fallback_server_url: Optional[str] = None,
    ):
        """初始化 OpenAPI 客户端 / Initialize OpenAPI client

        Args:
            protocol_spec: OpenAPI JSON 字符串 / OpenAPI JSON string
            headers: 请求头 / Request headers
            fallback_server_url: 当 OpenAPI spec 中没有 servers 时的备用 URL /
                Fallback URL when servers is not present in OpenAPI spec
        """
        self.headers = headers or {}
        self._fallback_server_url = fallback_server_url
        self._spec: Optional[Dict[str, Any]] = None
        self._operations: Optional[List[Dict[str, Any]]] = None

        if protocol_spec:
            try:
                self._spec = json.loads(protocol_spec)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse protocol_spec as JSON")

    @property
    def server_url(self) -> Optional[str]:
        """获取 OpenAPI Schema 中的 Server URL / Get Server URL from OpenAPI Schema

        优先从 spec.servers 获取，如果不存在则使用 fallback_server_url。
        Prefers spec.servers, falls back to fallback_server_url if not present.
        """
        if self._spec:
            servers = self._spec.get("servers", [])
            if servers and isinstance(servers, list):
                url = servers[0].get("url")
                if url:
                    return url
        return self._fallback_server_url

    def _resolve_ref(self, ref: str) -> Dict[str, Any]:
        """解析 $ref 引用 / Resolve $ref reference

        支持 JSON Pointer 格式的本地引用，如 #/components/schemas/WeatherRequest。
        Supports local JSON Pointer references like #/components/schemas/WeatherRequest.

        Args:
            ref: $ref 字符串 / $ref string

        Returns:
            解析后的 schema 字典 / Resolved schema dict
        """
        if not self._spec or not ref.startswith("#/"):
            return {}

        parts = ref[2:].split("/")
        current: Any = self._spec
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, {})
            else:
                return {}
        return current if isinstance(current, dict) else {}

    def _resolve_schema(
        self, schema: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """递归解析 schema 中的所有 $ref 引用 / Recursively resolve all $ref in schema

        Args:
            schema: 可能包含 $ref 的 schema / Schema that may contain $ref

        Returns:
            解析后的完整 schema / Fully resolved schema
        """
        if not schema or not isinstance(schema, dict):
            return schema

        if "$ref" in schema:
            resolved = self._resolve_ref(schema["$ref"])
            return self._resolve_schema(resolved)

        result = {}
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                result[key] = {
                    prop_name: self._resolve_schema(prop_schema) or prop_schema
                    for prop_name, prop_schema in value.items()
                }
            elif key in ("items", "additionalProperties") and isinstance(
                value, dict
            ):
                result[key] = self._resolve_schema(value) or value
            elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
                result[key] = [
                    self._resolve_schema(item) or item for item in value
                ]
            else:
                result[key] = value

        return result

    def _parse_operations(self) -> List[Dict[str, Any]]:
        """解析 OpenAPI Schema 中的所有 operations / Parse all operations from OpenAPI Schema"""
        if self._operations is not None:
            return self._operations

        self._operations = []
        if not self._spec:
            return self._operations

        paths = self._spec.get("paths", {})
        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            for method in ("get", "post", "put", "delete", "patch"):
                operation = path_item.get(method)
                if not operation or not isinstance(operation, dict):
                    continue

                operation_id = operation.get("operationId", f"{method}_{path}")
                summary = operation.get("summary", "")
                description = operation.get("description", "")

                request_body_schema = None
                request_body = operation.get("requestBody", {})
                if isinstance(request_body, dict):
                    content = request_body.get("content", {})
                    json_content = content.get("application/json", {})
                    raw_schema = json_content.get("schema")
                    request_body_schema = self._resolve_schema(raw_schema)

                parameters_schema = None
                parameters = operation.get("parameters", [])
                if parameters and isinstance(parameters, list):
                    props = {}
                    required_params = []
                    for param in parameters:
                        if not isinstance(param, dict):
                            continue
                        param_name = param.get("name", "")
                        param_schema = param.get("schema", {"type": "string"})
                        param_schema["description"] = param.get(
                            "description", ""
                        )
                        props[param_name] = param_schema
                        if param.get("required"):
                            required_params.append(param_name)
                    if props:
                        parameters_schema = {
                            "type": "object",
                            "properties": props,
                        }
                        if required_params:
                            parameters_schema["required"] = required_params

                input_schema = request_body_schema or parameters_schema

                self._operations.append({
                    "operation_id": operation_id,
                    "summary": summary,
                    "description": description,
                    "method": method.upper(),
                    "path": path,
                    "input_schema": input_schema,
                })

        return self._operations

    def list_tools(self) -> List[ToolInfo]:
        """获取工具列表 / Get tool list

        Returns:
            List[ToolInfo]: 工具信息列表 / List of tool information
        """
        operations = self._parse_operations()
        tools = []
        for operation in operations:
            parameters = None
            if operation.get("input_schema"):
                parameters = ToolSchema.from_any_openapi_schema(
                    operation["input_schema"]
                )

            tool_description = operation["summary"] or operation["description"]
            tools.append(
                ToolInfo(
                    name=operation["operation_id"],
                    description=tool_description,
                    parameters=parameters
                    or ToolSchema(type="object", properties={}),
                )
            )
        return tools

    async def list_tools_async(self) -> List[ToolInfo]:
        """异步获取工具列表 / Get tool list asynchronously"""
        return self.list_tools()

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """调用工具 / Call tool

        Args:
            name: operationId / Operation ID
            arguments: 调用参数 / Call arguments

        Returns:
            Any: 调用结果 / Call result

        Raises:
            ValueError: operation 不存在 / Operation not found
        """
        operations = self._parse_operations()
        target_operation = None
        for operation in operations:
            if operation["operation_id"] == name:
                target_operation = operation
                break

        if not target_operation:
            raise ValueError(
                f"Operation '{name}' not found in OpenAPI spec. Available"
                f" operations: {[op['operation_id'] for op in operations]}"
            )

        base_url = self.server_url
        if not base_url:
            raise ValueError("No server URL found in OpenAPI spec")

        url = f"{base_url.rstrip('/')}{target_operation['path']}"
        method = target_operation["method"]

        logger.debug(
            f"Calling FunctionCall tool: {method} {url} with args={arguments}"
        )

        with httpx.Client(headers=self.headers, timeout=30.0) as client:
            if method in ("POST", "PUT", "PATCH"):
                response = client.request(method, url, json=arguments or {})
            else:
                response = client.request(method, url, params=arguments or {})

            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return response.text

    async def call_tool_async(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """异步调用工具 / Call tool asynchronously

        Args:
            name: operationId / Operation ID
            arguments: 调用参数 / Call arguments

        Returns:
            Any: 调用结果 / Call result

        Raises:
            ValueError: operation 不存在 / Operation not found
        """
        operations = self._parse_operations()
        target_operation = None
        for operation in operations:
            if operation["operation_id"] == name:
                target_operation = operation
                break

        if not target_operation:
            raise ValueError(
                f"Operation '{name}' not found in OpenAPI spec. Available"
                f" operations: {[op['operation_id'] for op in operations]}"
            )

        base_url = self.server_url
        if not base_url:
            raise ValueError("No server URL found in OpenAPI spec")

        url = f"{base_url.rstrip('/')}{target_operation['path']}"
        method = target_operation["method"]

        logger.debug(
            f"Calling FunctionCall tool async: {method} {url} with"
            f" args={arguments}"
        )

        async with httpx.AsyncClient(
            headers=self.headers, timeout=30.0
        ) as client:
            if method in ("POST", "PUT", "PATCH"):
                response = await client.request(
                    method, url, json=arguments or {}
                )
            else:
                response = await client.request(
                    method, url, params=arguments or {}
                )

            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return response.text
