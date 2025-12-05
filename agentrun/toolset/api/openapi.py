"""OpenAPI协议处理 / OpenAPI Protocol Handler

处理OpenAPI规范的工具调用。
Handles tool invocations for OpenAPI specification.
"""

from copy import deepcopy
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from pydash import get as pg

from agentrun.integration.utils.tool import normalize_tool_name
from agentrun.utils.config import Config
from agentrun.utils.log import logger
from agentrun.utils.model import BaseModel

from ..model import ToolInfo, ToolSchema


class ApiSet:
    """统一的工具集接口，支持 OpenAPI 和 MCP 工具"""

    def __init__(
        self,
        tools: List[ToolInfo],
        invoker: Any,
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
    ):
        self._tools = {tool.name: tool for tool in tools if tool.name}
        self._invoker = invoker
        self._base_url = base_url
        self._default_headers = headers.copy() if headers else {}
        self._default_query_params = query_params.copy() if query_params else {}
        self._base_config = config

    def invoke(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """调用指定的工具"""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found.")

        # 先将 arguments 中的不序列化类型转换为原生 python 类型
        arguments = self._convert_arguments(arguments)

        # 合并配置：优先使用传入的 config，否则使用 base_config
        effective_config = Config.with_configs(self._base_config, config)

        # 调用实际的 invoker
        if hasattr(self._invoker, "invoke_tool"):
            return self._invoker.invoke_tool(name, arguments, effective_config)
        elif hasattr(self._invoker, "call_tool"):
            return self._invoker.call_tool(name, arguments, effective_config)
        elif callable(self._invoker):
            return self._invoker(name, arguments)
        else:
            raise ValueError("Invalid invoker provided.")

    async def invoke_async(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        """异步调用指定的工具"""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found.")

        # 先将 arguments 中的不序列化类型转换为原生 python 类型
        arguments = self._convert_arguments(arguments)

        # 合并配置：优先使用传入的 config，否则使用 base_config
        effective_config = Config.with_configs(self._base_config, config)

        # 调用实际的 invoker
        if hasattr(self._invoker, "invoke_tool_async"):
            return await self._invoker.invoke_tool_async(
                name, arguments, effective_config
            )
        elif hasattr(self._invoker, "call_tool_async"):
            return await self._invoker.call_tool_async(
                name, arguments, effective_config
            )
        else:
            raise ValueError("Async invoker not available.")

    def tools(self) -> List[ToolInfo]:
        """返回所有工具列表"""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[ToolInfo]:
        """获取指定名称的工具"""
        return self._tools.get(name)

    def to_function_tool(self, name: str):
        """将工具转换为 Python 函数

        Args:
            name: 工具名称

        Returns:
            一个 Python 函数，其 __name__ 是工具名称，__doc__ 是描述，
            参数与工具规范定义相同
        """
        from functools import wraps
        import inspect
        import re

        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found.")

        tool = self._tools[name]
        parameters_schema = tool.parameters

        # 构建函数签名和参数映射
        sig_params = []
        doc_parts = [tool.description or ""] if tool.description else []
        param_mapping = {}  # 记录参数来源（path/query/body）

        if parameters_schema and parameters_schema.properties:
            doc_parts.append("\n参数:")

            # 首先处理嵌套的 path、query、body 参数
            for container_name in ["path", "query", "body"]:
                if container_name in parameters_schema.properties:
                    container_schema = parameters_schema.properties[
                        container_name
                    ]
                    if container_schema.properties:
                        for (
                            param_name,
                            param_schema,
                        ) in container_schema.properties.items():
                            # 如果参数已经在顶层存在，跳过（避免重名）
                            if param_name in param_mapping:
                                continue

                            param_type = param_schema.type or "Any"
                            param_desc = param_schema.description or ""

                            # 确定 Python 类型
                            python_type = self._schema_type_to_python_type(
                                param_type
                            )

                            # 检查是否必需
                            is_required = (
                                container_schema.required is not None
                                and param_name in container_schema.required
                            )

                            # 创建参数
                            if is_required:
                                param = inspect.Parameter(
                                    param_name,
                                    inspect.Parameter.KEYWORD_ONLY,
                                    annotation=python_type,
                                )
                            else:
                                from typing import Optional

                                param = inspect.Parameter(
                                    param_name,
                                    inspect.Parameter.KEYWORD_ONLY,
                                    default=None,
                                    annotation=Optional[python_type],
                                )
                            sig_params.append(param)
                            param_mapping[param_name] = container_name

                            # 添加到文档
                            doc_parts.append(
                                f"  {param_name} ({param_type},"
                                f" {container_name}): {param_desc}"
                            )

            # 然后处理顶层参数（不包括 path、query、body）
            for (
                param_name,
                param_schema,
            ) in parameters_schema.properties.items():
                if param_name in ["path", "query", "body"]:
                    continue
                if param_name in param_mapping:
                    continue

                param_type = param_schema.type or "Any"
                param_desc = param_schema.description or ""

                # 确定 Python 类型
                python_type = self._schema_type_to_python_type(param_type)

                # 检查是否必需
                is_required = (
                    parameters_schema.required is not None
                    and param_name in parameters_schema.required
                )

                # 创建参数
                if is_required:
                    param = inspect.Parameter(
                        param_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        annotation=python_type,
                    )
                else:
                    from typing import Optional

                    param = inspect.Parameter(
                        param_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=Optional[python_type],
                    )
                sig_params.append(param)
                param_mapping[param_name] = "top"

                # 添加到文档
                doc_parts.append(f"  {param_name} ({param_type}): {param_desc}")

            # 添加 path、query、body 作为可选参数（用于显式传递）
            for container_name in ["path", "query", "body"]:
                if container_name in parameters_schema.properties:
                    from typing import Dict, Optional

                    param = inspect.Parameter(
                        container_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=Optional[Dict[str, Any]],
                    )
                    sig_params.append(param)

        # 创建函数实现
        def impl(**kwargs):
            # 处理嵌套的 path、query、body 参数
            normalized_kwargs = {}

            # 收集 path 参数
            path_params = {}
            if "path" in kwargs and isinstance(kwargs["path"], dict):
                path_params.update(kwargs.pop("path"))
            # 从顶层参数中提取 path 参数
            for param_name, source in param_mapping.items():
                if source == "path" and param_name in kwargs:
                    path_params[param_name] = kwargs.pop(param_name)
            if path_params:
                normalized_kwargs["path"] = path_params

            # 收集 query 参数
            query_params = {}
            if "query" in kwargs and isinstance(kwargs["query"], dict):
                query_params.update(kwargs.pop("query"))
            # 从顶层参数中提取 query 参数
            for param_name, source in param_mapping.items():
                if source == "query" and param_name in kwargs:
                    query_params[param_name] = kwargs.pop(param_name)
            if query_params:
                normalized_kwargs["query"] = query_params

            # 处理 body 参数 - 检查是否有来自 requestBody 的参数
            if "body" in kwargs:
                normalized_kwargs["body"] = kwargs.pop("body")
            else:
                # 收集所有非 path/query 参数作为 body
                # 首先检查是否有显式的 body 类型参数
                body_params = {}
                for param_name, source in param_mapping.items():
                    if source == "body" and param_name in kwargs:
                        body_params[param_name] = kwargs.pop(param_name)

                # 如果没有显式的 body 参数，检查是否有来自 requestBody 的顶层参数
                if not body_params:
                    # 从 self._operations 中获取操作信息（对于 OpenAPI 工具）
                    operation = {}
                    if (
                        hasattr(self._invoker, "_operations")
                        and name in self._invoker._operations
                    ):
                        operation = self._invoker._operations[name]
                    request_body = operation.get("request_body", {})
                    if request_body:
                        content = request_body.get("content", {})
                        for content_type in [
                            "application/json",
                            "application/x-www-form-urlencoded",
                        ]:
                            if content_type in content:
                                body_schema = content[content_type].get(
                                    "schema", {}
                                )
                                if body_schema.get("type") == "object":
                                    body_properties = body_schema.get(
                                        "properties", {}
                                    )
                                    for prop_name in body_properties:
                                        if prop_name in kwargs:
                                            body_params[prop_name] = kwargs.pop(
                                                prop_name
                                            )
                                break

                if body_params:
                    normalized_kwargs["body"] = body_params

            # 合并其他参数（headers、timeout 等）
            normalized_kwargs.update(kwargs)

            return self.invoke(name, normalized_kwargs)

        # 包装函数
        @wraps(impl)
        def wrapper(**kwargs):
            return impl(**kwargs)

        # 设置函数属性
        clean_name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
        # Normalize for provider limits / external frameworks
        clean_name = normalize_tool_name(clean_name)
        wrapper.__name__ = clean_name
        wrapper.__qualname__ = clean_name
        wrapper.__doc__ = "\n".join(doc_parts)

        # 设置函数签名
        if sig_params:
            wrapper.__signature__ = inspect.Signature(sig_params)  # type: ignore

        return wrapper

    def _schema_type_to_python_type(self, schema_type: str):
        """将 schema 类型转换为 Python 类型"""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        return type_mapping.get(schema_type, Any)

    def _convert_to_native(self, value: Any) -> Any:
        """将常见框架类型或 Pydantic/GoogleADK 等对象转换为 Python 原生类型

        目的是确保我们发送到 OpenAPI 的 JSON body 是可以被 json 序列化的
        """
        # 基本类型
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        # 列表与字典
        if isinstance(value, list):
            return [self._convert_to_native(item) for item in value]
        if isinstance(value, dict):
            return {k: self._convert_to_native(v) for k, v in value.items()}

        # Pydantic v2
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump(mode="python", exclude_unset=True)
                return self._convert_to_native(dumped)
            except Exception:
                pass

        # Pydantic v1 or other obj
        if hasattr(value, "dict"):
            try:
                dumped = value.dict(exclude_none=True)
                return self._convert_to_native(dumped)
            except Exception:
                pass

        # Google ADK 的 Nestedobject 等
        if hasattr(value, "to_dict"):
            try:
                dumped = value.to_dict()
                return self._convert_to_native(dumped)
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            try:
                dumped = getattr(value, "__dict__")
                return self._convert_to_native(dumped)
            except Exception:
                pass

        # 无法转换，返回原值，httpx/JSON 序列化时可能会失败
        return value

    def _convert_arguments(
        self, args: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if args is None:
            return None
        if not isinstance(args, dict):
            return args
        return {k: self._convert_to_native(v) for k, v in args.items()}

    @classmethod
    def from_openapi_schema(
        cls,
        schema: Union[str, dict],
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
        timeout: Optional[int] = None,
    ) -> "ApiSet":
        """从 OpenAPI schema 创建 ApiSet

        Args:
            input: OpenAPI schema (字符串或字典)
            base_url: 基础 URL
            headers: 默认请求头
            query_params: 默认查询参数
            config: 配置对象
            timeout: 超时时间
        """
        # 创建 OpenAPI 客户端（会解析 $ref）
        openapi_client = OpenAPI(
            schema=schema,
            base_url=base_url,
            headers=headers,
            query_params=query_params,
            config=config,
            timeout=timeout,
        )

        tools = []
        # 使用已解析 $ref 的 schema
        resolved_schema = openapi_client._schema

        if isinstance(resolved_schema, dict):
            paths = resolved_schema.get("paths", {})
            if isinstance(paths, dict):
                for path, path_item in paths.items():
                    if not isinstance(path_item, dict):
                        continue

                    # 获取路径级别的参数
                    path_parameters = path_item.get("parameters", [])

                    for method, operation in path_item.items():
                        if method.upper() not in OpenAPI._SUPPORTED_METHODS:
                            continue
                        if not isinstance(operation, dict):
                            continue

                        operation_id = (
                            operation.get("operationId")
                            or f"{method.upper()} {path}"
                        )
                        description = operation.get(
                            "description"
                        ) or operation.get("summary", "")

                        # 构建参数 schema，区分 path、query、body
                        properties = {}
                        required = []

                        # 合并路径级别和操作级别的参数
                        all_parameters = []
                        if isinstance(path_parameters, list):
                            all_parameters.extend(path_parameters)
                        operation_parameters = operation.get("parameters", [])
                        if isinstance(operation_parameters, list):
                            all_parameters.extend(operation_parameters)

                        # 处理 path 和 query 参数
                        path_params = {}
                        query_params = {}
                        for param in all_parameters:
                            if not isinstance(param, dict):
                                continue
                            param_name = param.get("name")
                            param_in = param.get("in", "query")
                            param_schema = param.get("schema", {})
                            param_description = param.get("description", "")
                            param_required = param.get("required", False)

                            if not param_name:
                                continue

                            # 为了避免重名，使用命名空间
                            if param_in == "path":
                                path_params[param_name] = {
                                    "schema": (
                                        ToolSchema.from_any_openapi_schema(
                                            param_schema
                                        )
                                    ),
                                    "description": param_description,
                                    "required": param_required,
                                }
                            elif param_in == "query":
                                query_params[param_name] = {
                                    "schema": (
                                        ToolSchema.from_any_openapi_schema(
                                            param_schema
                                        )
                                    ),
                                    "description": param_description,
                                    "required": param_required,
                                }

                        # 将 path 和 query 参数添加到 properties
                        if path_params:
                            path_properties = {
                                name: info["schema"]
                                for name, info in path_params.items()
                            }
                            properties["path"] = ToolSchema(
                                type="object",
                                properties=path_properties,
                                required=[
                                    name
                                    for name, info in path_params.items()
                                    if info["required"]
                                ],
                                description="Path parameters",
                            )
                            # 同时将 path 参数直接添加到顶层（方便使用）
                            for name, info in path_params.items():
                                if (
                                    name not in properties
                                ):  # 避免与 query 参数重名
                                    properties[name] = info["schema"]
                                    if info["required"]:
                                        required.append(name)

                        if query_params:
                            query_properties = {
                                name: info["schema"]
                                for name, info in query_params.items()
                            }
                            properties["query"] = ToolSchema(
                                type="object",
                                properties=query_properties,
                                required=[
                                    name
                                    for name, info in query_params.items()
                                    if info["required"]
                                ],
                                description="Query parameters",
                            )
                            # 同时将 query 参数直接添加到顶层（方便使用）
                            for name, info in query_params.items():
                                if (
                                    name not in properties
                                ):  # 避免与 path 参数重名
                                    properties[name] = info["schema"]
                                    if info["required"]:
                                        required.append(name)

                        # 处理 requestBody - 将属性直接展开到顶层
                        request_body = operation.get("requestBody", {})
                        if isinstance(request_body, dict):
                            content = request_body.get("content", {})
                            for content_type in [
                                "application/json",
                                "application/x-www-form-urlencoded",
                            ]:
                                if content_type in content:
                                    body_schema = content[content_type].get(
                                        "schema", {}
                                    )
                                    if (
                                        body_schema
                                        and body_schema.get("type") == "object"
                                    ):
                                        # 将 requestBody schema 的属性直接展开到顶层
                                        body_properties = body_schema.get(
                                            "properties", {}
                                        )
                                        for (
                                            prop_name,
                                            prop_schema,
                                        ) in body_properties.items():
                                            if (
                                                prop_name not in properties
                                            ):  # 避免与已有参数冲突
                                                properties[prop_name] = (
                                                    ToolSchema.from_any_openapi_schema(
                                                        prop_schema
                                                    )
                                                )

                                        # 添加必需的 requestBody 参数
                                        if request_body.get("required"):
                                            body_required = body_schema.get(
                                                "required", []
                                            )
                                            for req_param in body_required:
                                                if req_param not in required:
                                                    required.append(req_param)
                                    break

                        parameters = ToolSchema(
                            type="object",
                            properties=properties if properties else None,
                            required=required if required else None,
                        )

                        tools.append(
                            ToolInfo(
                                name=normalize_tool_name(operation_id),
                                description=description,
                                parameters=parameters,
                            )
                        )

        return cls(
            tools=tools,
            invoker=openapi_client,
            base_url=base_url,
            headers=headers,
            query_params=query_params,
            config=config,
        )

    @classmethod
    def from_mcp_tools(
        cls,
        tools: Any,
        mcp_client: Any,
        config: Optional[Config] = None,
    ) -> "ApiSet":
        """从 MCP tools 创建 ApiSet

        Args:
            tools: MCP tools 列表或单个工具
            mcp_client: MCP 客户端（MCPToolSet 实例）
            config: 配置对象
        """
        # 获取工具列表
        tool_infos = []
        if tools:

            # 如果 tools 是单个工具，转换为列表
            if not isinstance(tools, (list, tuple)):
                tools = [tools]

            for tool in tools:
                # 处理不同的 MCP tool 格式
                if hasattr(tool, "name"):
                    # MCP Tool 对象
                    tool_name = tool.name
                    tool_description = getattr(tool, "description", None)
                    input_schema = getattr(
                        tool, "inputSchema", None
                    ) or getattr(tool, "input_schema", None)
                elif isinstance(tool, dict):
                    # 字典格式
                    tool_name = tool.get("name")
                    tool_description = tool.get("description")
                    input_schema = tool.get("inputSchema") or tool.get(
                        "input_schema"
                    )
                else:
                    continue

                if not tool_name:
                    continue

                # 构建 parameters schema
                parameters = None
                if input_schema:
                    if isinstance(input_schema, dict):
                        parameters = ToolSchema.from_any_openapi_schema(
                            input_schema
                        )
                    elif hasattr(input_schema, "model_dump"):
                        parameters = ToolSchema.from_any_openapi_schema(
                            input_schema.model_dump()
                        )

                tool_infos.append(
                    ToolInfo(
                        name=normalize_tool_name(tool_name),
                        description=tool_description,
                        parameters=parameters
                        or ToolSchema(type="object", properties={}),
                    )
                )

        # 创建调用器包装
        class MCPInvoker:

            def __init__(self, mcp_client, config):
                self.mcp_client = mcp_client
                self.config = config

            def invoke_tool(
                self,
                name: str,
                arguments: Optional[Dict[str, Any]] = None,
                config: Optional[Config] = None,
            ):
                cfg = Config.with_configs(self.config, config)
                return self.mcp_client.call_tool(name, arguments, cfg)

            async def invoke_tool_async(
                self,
                name: str,
                arguments: Optional[Dict[str, Any]] = None,
                config: Optional[Config] = None,
            ):
                cfg = Config.with_configs(self.config, config)
                return await self.mcp_client.call_tool_async(
                    name, arguments, cfg
                )

        invoker = MCPInvoker(mcp_client, config)

        return cls(
            tools=tool_infos,
            invoker=invoker,
            config=config,
        )


class OpenAPI:
    """OpenAPI schema based tool client."""

    _SUPPORTED_METHODS = {
        "DELETE",
        "GET",
        "HEAD",
        "OPTIONS",
        "PATCH",
        "POST",
        "PUT",
    }

    def __init__(
        self,
        schema: Any,
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
        timeout: Optional[int] = None,
    ):
        self._raw_schema = schema or ""
        self._schema = self._load_schema(self._raw_schema)
        # Resolve local $ref (e.g. "#/components/schemas/...") to simplify later usage.
        # Only resolves internal JSON Pointers; external refs (urls) are left as-is.
        try:
            self._resolve_refs(self._schema)
        except Exception:
            # If resolving fails for any reason, fall back to original schema.
            logger.debug(
                "OpenAPI $ref resolution failed; continuing without expansion"
            )
        self._operations = self._build_operations(self._schema)
        self._base_url = base_url or self._extract_base_url(self._schema)
        self._default_headers = headers.copy() if headers else {}
        self._default_query_params = query_params.copy() if query_params else {}
        self._base_config = config
        self._default_timeout = (
            timeout or (config.get_timeout() if config else None) or 60
        )

    def list_tools(self, name: Optional[str] = None):
        """List tools defined in the OpenAPI schema.

        Args:
            name: OperationId of the tool. When provided, return the single
                tool definition; otherwise return all tools.

        Returns:
            A list of tool metadata dictionaries.
        """

        if name:
            if name not in self._operations:
                raise ValueError(f"Tool '{name}' not found in OpenAPI schema.")
            return [deepcopy(self._operations[name])]

        return [deepcopy(item) for item in self._operations.values()]

        # return [
        #     ToolInfo(
        #         name=item["operationId"],
        #         description=item.get("description") or item.get("summary"),
        #         parameters=ToolSchema(
        #             type="object",
        #             properties={
        #                 "path": ToolSchema(
        #                     type="object",
        #                     properties={
        #                         param["name"]: ToolSchema(
        #                             type=pg(param, "schema.type", "string"),
        #                             description=pg(param, "schema.description", pg("description", "")),
        #                             properties:
        #                         )
        #                         for param in item["parameters"]
        #                     },
        #                 ),
        #                 "parameters": ToolSchema(
        #                     type="object",
        #                 ),
        #                 "body": ToolSchema(type="object", description="Request body"),
        #             },
        #         ),
        #     )
        #     for item in self._operations.values()
        # ]

    def has_tool(self, name: str) -> bool:
        return name in self._operations

    def invoke_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        arguments = self._convert_arguments(arguments)
        request_kwargs, timeout, raise_for_status = self._prepare_request(
            name, arguments, config
        )
        with httpx.Client(timeout=timeout) as client:
            response = client.request(**request_kwargs)
            if raise_for_status:
                response.raise_for_status()
            return self._format_response(response)

    async def invoke_tool_async(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        config: Optional[Config] = None,
    ) -> Dict[str, Any]:
        arguments = self._convert_arguments(arguments)
        request_kwargs, timeout, raise_for_status = self._prepare_request(
            name, arguments, config
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(**request_kwargs)
            if raise_for_status:
                response.raise_for_status()
            return self._format_response(response)

    def _load_schema(self, schema: Any) -> Dict[str, Any]:
        if isinstance(schema, dict):
            return schema

        if isinstance(schema, (bytes, bytearray)):
            schema = schema.decode("utf-8")

        if not schema:
            raise ValueError("OpenAPI schema detail is required.")

        try:
            return json.loads(schema)
        except json.JSONDecodeError:
            pass

        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to parse OpenAPI YAML schema."
            ) from exc

        try:
            return yaml.safe_load(schema) or {}
        except yaml.YAMLError as exc:  # pragma: no cover
            raise ValueError("Invalid OpenAPI schema content.") from exc

    def _resolve_refs(self, root: Any) -> None:
        """Resolve local JSON-Pointer $ref entries in-place.

        This implementation only handles refs that start with "#/" and
        resolves them by replacing the node with a deep copy of the target.
        It also merges other sibling keys (e.g. description) onto the
        resolved target, matching common OpenAPI semantics.
        """
        if not isinstance(root, (dict, list)):
            return

        def resolve_pointer(doc: Dict[str, Any], parts: List[str]):
            cur = doc
            for p in parts:
                # unescape per JSON Pointer spec
                p = p.replace("~1", "/").replace("~0", "~")
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
            return cur

        def _walk(node: Any):
            if isinstance(node, dict):
                # if this node is a $ref, resolve it and return replacement
                if "$ref" in node and isinstance(node["$ref"], str):
                    ref = node["$ref"]
                    if ref.startswith("#/"):
                        parts = ref[2:].split("/")
                        target = resolve_pointer(root, parts)
                        if target is None:
                            return node
                        replacement = deepcopy(target)
                        # merge other keys (except $ref) into replacement
                        for k, v in node.items():
                            if k == "$ref":
                                continue
                            replacement[k] = v
                        # continue resolving inside replacement
                        return _walk(replacement)
                    # external refs: leave as-is
                    return node

                # otherwise recurse into children
                for k, v in list(node.items()):
                    node[k] = _walk(v)
                return node
            elif isinstance(node, list):
                return [_walk(item) for item in node]
            else:
                return node

        # perform in-place walk and resolution
        resolved = _walk(root)
        # If root was replaced by a resolved value, try to update it in-place
        if (
            resolved is not root
            and isinstance(root, dict)
            and isinstance(resolved, dict)
        ):
            root.clear()
            root.update(resolved)

    def _extract_base_url(self, schema: Dict[str, Any]) -> Optional[str]:
        servers = schema.get("servers") or []
        return self._pick_server_url(servers)

    def _convert_to_native(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [self._convert_to_native(item) for item in value]
        if isinstance(value, dict):
            return {k: self._convert_to_native(v) for k, v in value.items()}
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump(mode="python", exclude_unset=True)
                return self._convert_to_native(dumped)
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                dumped = value.dict(exclude_none=True)
                return self._convert_to_native(dumped)
            except Exception:
                pass
        if hasattr(value, "to_dict"):
            try:
                dumped = value.to_dict()
                return self._convert_to_native(dumped)
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                dumped = getattr(value, "__dict__")
                return self._convert_to_native(dumped)
            except Exception:
                pass
        return value

    def _convert_arguments(
        self, args: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if args is None:
            return None
        if not isinstance(args, dict):
            return args
        return {k: self._convert_to_native(v) for k, v in args.items()}

    def _pick_server_url(self, servers: Any) -> Optional[str]:
        if not servers:
            return None

        if isinstance(servers, dict):
            servers = [servers]

        if not isinstance(servers, list):
            return None

        for server in servers:
            if isinstance(server, str):
                return server
            if not isinstance(server, dict):
                continue
            url = server.get("url")
            if not url:
                continue
            variables = server.get("variables", {})
            if isinstance(variables, dict):
                for key, meta in variables.items():
                    default = (
                        meta.get("default") if isinstance(meta, dict) else None
                    )
                    if default is not None:
                        url = url.replace(f"{{{key}}}", str(default))
            return url

        return None

    def _build_operations(
        self, schema: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        operations: Dict[str, Dict[str, Any]] = {}
        paths = schema.get("paths") or {}
        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            path_parameters = path_item.get("parameters", [])
            path_servers = path_item.get("servers")
            for method, operation in path_item.items():
                if method.upper() not in self._SUPPORTED_METHODS:
                    continue
                if not isinstance(operation, dict):
                    continue
                operation_id = operation.get("operationId") or (
                    f"{method.upper()} {path}"
                )
                parameters = []
                if isinstance(path_parameters, list):
                    parameters.extend(path_parameters)
                operation_parameters = operation.get("parameters", [])
                if isinstance(operation_parameters, list):
                    parameters.extend(operation_parameters)
                server_url = self._pick_server_url(
                    operation.get("servers") or path_servers
                )
                operations[operation_id] = {
                    "operationId": operation_id,
                    "method": method.upper(),
                    "path": path,
                    "summary": operation.get("summary"),
                    "description": operation.get("description"),
                    "parameters": parameters,
                    "path_parameters": [
                        param.get("name")
                        for param in parameters
                        if isinstance(param, dict)
                        and param.get("in") == "path"
                        and param.get("name")
                    ],
                    "request_body": operation.get("requestBody"),
                    "responses": operation.get("responses"),
                    "tags": operation.get("tags", []),
                    "server_url": server_url,
                }
        return operations

    def _prepare_request(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]],
        config: Optional[Config],
    ) -> Tuple[Dict[str, Any], float, bool]:
        if name not in self._operations:
            raise ValueError(f"Tool '{name}' not found in OpenAPI schema.")

        operation = self._operations[name]
        args = deepcopy(arguments) if arguments else {}

        combined_config = config or self._base_config
        headers: Dict[str, str] = {}
        timeout = self._default_timeout
        if combined_config:
            headers.update(combined_config.get_headers())
            _timeout = combined_config.get_timeout()
            if _timeout:
                timeout = _timeout

        headers.update(self._default_headers)

        user_headers = args.pop("headers", {})
        if isinstance(user_headers, dict):
            headers.update(user_headers)

        timeout_override = args.pop("timeout", None)
        if isinstance(timeout_override, (int, float)):
            timeout = timeout_override

        raise_for_status = bool(args.pop("raise_for_status", True))

        path_params = self._extract_dict(args, ["path", "path_params"])
        query_params = self._merge_dicts(
            self._default_query_params,
            self._extract_dict(args, ["query", "query_params"]),
        )
        body = self._extract_body(args)
        files = args.pop("files", None)
        data = args.pop("data", None)

        # Fill parameters defined in the schema.
        for param in operation.get("parameters", []):
            if not isinstance(param, dict):
                continue
            name = param.get("name", "")
            location = param.get("in")
            if not name or name not in args:
                continue
            value = args.pop(name)
            if location == "path":
                path_params[name] = value
            elif location == "query":
                query_params[name] = value
            elif location == "header":
                headers[name] = value

        method = operation["method"]
        path = self._render_path(
            operation["path"], operation["path_parameters"], path_params
        )

        base_url = (
            args.pop("base_url", None)
            or operation.get("server_url")
            or self._base_url
        )

        if not base_url:
            raise ValueError(
                "Base URL is required to invoke an OpenAPI tool. Provide it "
                "via OpenAPI(..., base_url=...) or in arguments['base_url']."
            )

        url = self._join_url(base_url, path)

        if method in {"GET", "HEAD"} and args:
            if not isinstance(query_params, dict):
                query_params = {}
            for key, value in args.items():
                query_params[key] = value
            args.clear()
        elif args and body is None and data is None:
            body = args
            args = {}

        request_kwargs: Dict[str, Any] = {
            "method": method,
            "url": url,
            "headers": headers,
        }

        if query_params:
            request_kwargs["params"] = query_params
        if files is not None:
            request_kwargs["files"] = files
        if data is not None:
            request_kwargs["data"] = data
        if body is not None and method not in {"GET", "HEAD"}:
            request_kwargs["json"] = body

        if args:
            logger.debug(
                "Unused arguments when invoking OpenAPI tool '%s': %s",
                name,
                args,
            )

        return request_kwargs, timeout, raise_for_status

    def _format_response(self, response: httpx.Response) -> Dict[str, Any]:
        try:
            body = response.json()
        except ValueError:
            body = response.text

        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": body,
            "raw_body": response.text,
            "url": str(response.request.url),
            "method": response.request.method,
        }

    def _extract_dict(
        self, source: Dict[str, Any], keys: List[str]
    ) -> Dict[str, Any]:
        for key in keys:
            value = source.pop(key, None)
            if value is None:
                continue
            if isinstance(value, dict):
                return value
            logger.warning(
                "Expected dictionary for argument '%s', got %s. Ignoring.",
                key,
                type(value).__name__,
            )
        return {}

    def _extract_body(self, source: Dict[str, Any]) -> Optional[Any]:
        if "json" in source:
            return source.pop("json")
        if "body" in source:
            return source.pop("body")
        if "payload" in source:
            return source.pop("payload")
        return None

    def _merge_dicts(
        self, base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if isinstance(base, dict):
            merged.update(base)
        if isinstance(override, dict):
            merged.update(override)
        return merged

    def _render_path(
        self, path: str, expected_params: List[str], path_params: Dict[str, Any]
    ) -> str:
        rendered = path
        missing = []
        for name in expected_params:
            if name not in path_params:
                missing.append(name)
                continue
            rendered = rendered.replace(f"{{{name}}}", str(path_params[name]))

        if missing:
            raise ValueError(
                f"Missing path parameters for {path}: {', '.join(missing)}"
            )
        return rendered

    def _join_url(self, base_url: str, path: str) -> str:
        if not base_url:
            raise ValueError("Base URL cannot be empty.")
        if not path:
            return base_url
        return f"{base_url.rstrip('/')}/{path.lstrip('/')}"
