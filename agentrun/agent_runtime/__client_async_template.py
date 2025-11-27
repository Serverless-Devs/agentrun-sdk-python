"""Agent Runtime 客户端 / Agent Runtime Client

此模块提供 Agent Runtime 的客户端API,用于管理智能体运行时。
This module provides the client API for Agent Runtime to manage agent runtimes.
"""

from typing import List, Optional

from alibabacloud_agentrun20250910.models import (
    CreateAgentRuntimeEndpointInput,
    CreateAgentRuntimeInput,
    GetAgentRuntimeRequest,
    ListAgentRuntimeEndpointsRequest,
    ListAgentRuntimesRequest,
    ListAgentRuntimeVersionsRequest,
    UpdateAgentRuntimeEndpointInput,
    UpdateAgentRuntimeInput,
)
from typing_extensions import Unpack

from agentrun.agent_runtime.api.data import InvokeArgs
from agentrun.agent_runtime.model import (
    AgentRuntimeArtifact,
    AgentRuntimeCreateInput,
    AgentRuntimeEndpointCreateInput,
    AgentRuntimeEndpointListInput,
    AgentRuntimeEndpointUpdateInput,
    AgentRuntimeListInput,
    AgentRuntimeUpdateInput,
    AgentRuntimeVersion,
    AgentRuntimeVersionListInput,
    NetworkConfig,
)
from agentrun.agent_runtime.runtime import AgentRuntime
from agentrun.utils.config import Config
from agentrun.utils.exception import HTTPError

from .api import AgentRuntimeControlAPI, AgentRuntimeDataAPI
from .endpoint import AgentRuntimeEndpoint


class AgentRuntimeClient:
    """Agent Runtime 客户端 / Agent Runtime Client

    提供 Agent Runtime 的创建、删除、更新、查询和端点管理功能。
    Provides create, delete, update, query and endpoint management functions for Agent Runtime.
    """

    def __init__(self, config: Optional[Config] = None):
        """初始化客户端 / Initialize client

        Args:
            config: 配置对象,可选 / Configuration object, optional
        """
        self.config = config
        self.__control_api = AgentRuntimeControlAPI(config)

    async def create_async(
        self, input: AgentRuntimeCreateInput, config: Optional[Config] = None
    ) -> AgentRuntime:
        """异步创建 Agent Runtime / Create Agent Runtime asynchronously

        Args:
            input: Agent Runtime 创建配置 / Agent Runtime creation configuration
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            AgentRuntime: 创建的 Agent Runtime 对象 / Created Agent Runtime object

        Raises:
            ValueError: 当既未提供代码配置也未提供容器配置时 / When neither code nor container configuration is provided
            ResourceAlreadyExistError: 资源已存在 / Resource already exists
            ResourceNotExistError: 资源不存在 / Resource does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        if input.network_configuration is None:
            input.network_configuration = NetworkConfig()

        if input.artifact_type is None:
            if input.code_configuration is not None:
                input.artifact_type = AgentRuntimeArtifact.CODE
            elif input.container_configuration is not None:
                input.artifact_type = AgentRuntimeArtifact.CONTAINER
            else:
                raise ValueError(
                    "Either code_configuration or image_configuration must be"
                    " provided."
                )

        try:
            result = await self.__control_api.create_agent_runtime_async(
                CreateAgentRuntimeInput().from_map(input.model_dump()),
                config=config,
            )
        except HTTPError as e:
            raise e.to_resource_error(
                "AgentRuntime", input.agent_runtime_name
            ) from e
        return AgentRuntime.from_inner_object(result)

    async def delete_async(
        self, id: str, config: Optional[Config] = None
    ) -> AgentRuntime:
        """异步删除 Agent Runtime / Delete Agent Runtime asynchronously

        Args:
            id: Agent Runtime ID
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            AgentRuntime: 删除的 Agent Runtime 对象 / Deleted Agent Runtime object

        Raises:
            ResourceNotExistError: 资源不存在 / Resource does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        try:
            result = await self.__control_api.delete_agent_runtime_async(
                id, config=config
            )
            return AgentRuntime.from_inner_object(result)
        except HTTPError as e:
            raise e.to_resource_error("AgentRuntime", id) from e

    async def update_async(
        self,
        id: str,
        input: AgentRuntimeUpdateInput,
        config: Optional[Config] = None,
    ) -> AgentRuntime:
        """异步更新 Agent Runtime / Update Agent Runtime asynchronously

        Args:
            id: Agent Runtime ID
            input: Agent Runtime 更新配置 / Agent Runtime update configuration
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            AgentRuntime: 更新后的 Agent Runtime 对象 / Updated Agent Runtime object

        Raises:
            ResourceNotExistError: 资源不存在 / Resource does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        try:
            result = await self.__control_api.update_agent_runtime_async(
                id,
                UpdateAgentRuntimeInput().from_map(input.model_dump()),
                config=config,
            )
            return AgentRuntime.from_inner_object(result)
        except HTTPError as e:
            raise e.to_resource_error("AgentRuntime", id) from e

    async def get_async(
        self,
        id: str,
        config: Optional[Config] = None,
    ) -> AgentRuntime:
        """异步获取 Agent Runtime / Get Agent Runtime asynchronously

        Args:
            id: Agent Runtime ID
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            AgentRuntime: Agent Runtime 对象 / Agent Runtime object

        Raises:
            ResourceNotExistError: 资源不存在 / Resource does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        try:
            result = await self.__control_api.get_agent_runtime_async(
                id, GetAgentRuntimeRequest(), config=config
            )
            return AgentRuntime.from_inner_object(result)
        except HTTPError as e:
            raise e.to_resource_error("AgentRuntime", id) from e

    async def list_async(
        self,
        input: Optional[AgentRuntimeListInput] = None,
        config: Optional[Config] = None,
    ) -> List[AgentRuntime]:
        """异步列出 Agent Runtimes / List Agent Runtimes asynchronously

        Args:
            input: 列表查询配置,可选 / List query configuration, optional
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            List[AgentRuntime]: Agent Runtime 对象列表 / List of Agent Runtime objects

        Raises:
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        try:
            if input is None:
                input = AgentRuntimeListInput()

            results = await self.__control_api.list_agent_runtimes_async(
                ListAgentRuntimesRequest().from_map(input.model_dump()),
                config=config,
            )
            return [
                AgentRuntime.from_inner_object(item) for item in results.items
            ]
        except HTTPError as e:
            raise e.to_resource_error("AgentRuntime") from e

    async def create_endpoint_async(
        self,
        agent_runtime_id: str,
        endpoint: AgentRuntimeEndpointCreateInput,
        config: Optional[Config] = None,
    ) -> AgentRuntimeEndpoint:
        """异步创建 Agent Runtime 端点 / Create Agent Runtime Endpoint asynchronously

        Args:
            agent_runtime_id: Agent Runtime ID
            endpoint: 端点创建配置 / Endpoint creation configuration
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            AgentRuntimeEndpoint: 创建的端点对象 / Created endpoint object

        Raises:
            ResourceAlreadyExistError: 资源已存在 / Resource already exists
            ResourceNotExistError: Agent Runtime 不存在 / Agent Runtime does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        try:
            result = (
                await self.__control_api.create_agent_runtime_endpoint_async(
                    agent_runtime_id,
                    CreateAgentRuntimeEndpointInput().from_map(
                        endpoint.model_dump()
                    ),
                    config=config,
                )
            )
        except HTTPError as e:
            raise e.to_resource_error(
                "AgentRuntimeEndpoint",
                "/".join([
                    agent_runtime_id,
                    endpoint.agent_runtime_endpoint_name or "",
                ]),
            ) from e

        return AgentRuntimeEndpoint.from_inner_object(result)

    async def delete_endpoint_async(
        self,
        agent_runtime_id: str,
        endpoint_id: str,
        config: Optional[Config] = None,
    ) -> AgentRuntimeEndpoint:
        """异步删除 Agent Runtime 端点 / Delete Agent Runtime Endpoint asynchronously

        Args:
            agent_runtime_id: Agent Runtime ID
            endpoint_id: 端点 ID / Endpoint ID
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            AgentRuntimeEndpoint: 删除的端点对象 / Deleted endpoint object

        Raises:
            ResourceNotExistError: 资源不存在 / Resource does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        try:
            result = (
                await self.__control_api.delete_agent_runtime_endpoint_async(
                    agent_runtime_id,
                    endpoint_id,
                    config=config,
                )
            )
        except HTTPError as e:
            raise e.to_resource_error(
                "AgentRuntimeEndpoint",
                "/".join([
                    agent_runtime_id,
                    endpoint_id,
                ]),
            ) from e

        return AgentRuntimeEndpoint.from_inner_object(result)

    async def update_endpoint_async(
        self,
        agent_runtime_id: str,
        endpoint_id: str,
        endpoint: AgentRuntimeEndpointUpdateInput,
        config: Optional[Config] = None,
    ) -> AgentRuntimeEndpoint:
        """异步更新 Agent Runtime 端点 / Update Agent Runtime Endpoint asynchronously

        Args:
            agent_runtime_id: Agent Runtime ID
            endpoint_id: 端点 ID / Endpoint ID
            endpoint: 端点更新配置 / Endpoint update configuration
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            AgentRuntimeEndpoint: 更新后的端点对象 / Updated endpoint object

        Raises:
            ResourceNotExistError: 资源不存在 / Resource does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        try:
            result = (
                await self.__control_api.update_agent_runtime_endpoint_async(
                    agent_runtime_id,
                    endpoint_id,
                    UpdateAgentRuntimeEndpointInput().from_map(
                        endpoint.model_dump()
                    ),
                    config=config,
                )
            )
        except HTTPError as e:
            raise e.to_resource_error(
                "AgentRuntimeEndpoint",
                "/".join([
                    agent_runtime_id,
                    endpoint_id,
                ]),
            ) from e

        return AgentRuntimeEndpoint.from_inner_object(result)

    async def get_endpoint_async(
        self,
        agent_runtime_id: str,
        endpoint_id: str,
        config: Optional[Config] = None,
    ) -> AgentRuntimeEndpoint:
        """异步获取 Agent Runtime 端点 / Get Agent Runtime Endpoint asynchronously

        Args:
            agent_runtime_id: Agent Runtime ID
            endpoint_id: 端点 ID / Endpoint ID
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            AgentRuntimeEndpoint: 端点对象 / Endpoint object

        Raises:
            ResourceNotExistError: 资源不存在 / Resource does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        try:
            result = await self.__control_api.get_agent_runtime_endpoint_async(
                agent_runtime_id,
                endpoint_id,
                config=config,
            )
        except HTTPError as e:
            raise e.to_resource_error(
                "AgentRuntimeEndpoint",
                "/".join([
                    agent_runtime_id,
                    endpoint_id,
                ]),
            ) from e

        return AgentRuntimeEndpoint.from_inner_object(result)

    async def list_endpoints_async(
        self,
        agent_runtime_id: str,
        input: Optional[AgentRuntimeEndpointListInput] = None,
        config: Optional[Config] = None,
    ) -> List[AgentRuntimeEndpoint]:
        """异步列出 Agent Runtime 端点 / List Agent Runtime Endpoints asynchronously

        Args:
            agent_runtime_id: Agent Runtime ID
            input: 列表查询配置,可选 / List query configuration, optional
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            List[AgentRuntimeEndpoint]: 端点对象列表 / List of endpoint objects

        Raises:
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        try:
            if input is None:
                input = AgentRuntimeEndpointListInput()

            results = (
                await self.__control_api.list_agent_runtime_endpoints_async(
                    agent_runtime_id,
                    ListAgentRuntimeEndpointsRequest().from_map(
                        input.model_dump()
                    ),
                    config=config,
                )
            )
            return [
                AgentRuntimeEndpoint.from_inner_object(item)
                for item in results.items
            ]
        except HTTPError as e:
            raise e.to_resource_error("AgentRuntime", agent_runtime_id) from e

    async def list_versions_async(
        self,
        agent_runtime_id: str,
        input: Optional[AgentRuntimeVersionListInput] = None,
        config: Optional[Config] = None,
    ) -> List[AgentRuntimeVersion]:
        """异步列出 Agent Runtime 版本 / List Agent Runtime Versions asynchronously

        Args:
            agent_runtime_id: Agent Runtime ID
            input: 列表查询配置,可选 / List query configuration, optional
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            List[AgentRuntimeVersion]: 版本对象列表 / List of version objects

        Raises:
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        try:
            if input is None:
                input = AgentRuntimeVersionListInput()

            results = (
                await self.__control_api.list_agent_runtime_versions_async(
                    agent_runtime_id,
                    ListAgentRuntimeVersionsRequest().from_map(
                        input.model_dump()
                    ),
                    config=config,
                )
            )
            return [
                AgentRuntimeVersion.from_inner_object(item)
                for item in results.items
            ]
        except HTTPError as e:
            raise e.to_resource_error("AgentRuntime", agent_runtime_id) from e

    async def invoke_openai_async(
        self,
        agent_runtime_name: str,
        agent_runtime_endpoint_name: str = "Default",
        **kwargs: Unpack[InvokeArgs],
    ):
        cfg = Config.with_configs(self.config, kwargs.get("config"))
        kwargs["config"] = cfg

        data_api = AgentRuntimeDataAPI(
            agent_runtime_name=agent_runtime_name,
            agent_runtime_endpoint_name=agent_runtime_endpoint_name,
            config=cfg,
        )
        return await data_api.invoke_openai_async(**kwargs)
