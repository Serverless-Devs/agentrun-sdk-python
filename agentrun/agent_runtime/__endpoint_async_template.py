"""Agent Runtime 端点资源 / Agent Runtime Endpoint Resource

此模块定义 Agent Runtime 端点的高级API。
This module defines the high-level API for Agent Runtime Endpoint.
"""

from typing import List, Optional

from typing_extensions import Self, Unpack

from agentrun.agent_runtime.api.data import AgentRuntimeDataAPI, InvokeArgs
from agentrun.agent_runtime.model import (
    AgentRuntimeEndpointCreateInput,
    AgentRuntimeEndpointImmutableProps,
    AgentRuntimeEndpointListInput,
    AgentRuntimeEndpointMutableProps,
    AgentRuntimeEndpointSystemProps,
    AgentRuntimeEndpointUpdateInput,
)
from agentrun.utils.config import Config
from agentrun.utils.resource import ResourceBase


class AgentRuntimeEndpoint(
    AgentRuntimeEndpointMutableProps,
    AgentRuntimeEndpointImmutableProps,
    AgentRuntimeEndpointSystemProps,
    ResourceBase,
):
    """智能体运行时端点信息 / Agent Runtime Endpoint Information

    提供端点的创建、删除、更新、查询功能。
    Provides create, delete, update, and query functions for endpoints.
    """

    @classmethod
    def __get_client(cls, config: Optional[Config] = None):
        """获取客户端实例 / Get client instance

        Returns:
            AgentRuntimeClient: 客户端实例 / Client instance
        """
        from agentrun.agent_runtime.client import AgentRuntimeClient

        return AgentRuntimeClient(config=config)

    @classmethod
    async def create_by_id_async(
        cls,
        agent_runtime_id: str,
        input: AgentRuntimeEndpointCreateInput,
        config: Optional[Config] = None,
    ):
        """根据 ID 异步创建端点 / Create endpoint by ID asynchronously

        Args:
            agent_runtime_id: Agent Runtime ID
            input: 端点创建配置 / Endpoint creation configuration
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            AgentRuntimeEndpoint: 创建的端点对象 / Created endpoint object

        Raises:
            ResourceAlreadyExistError: 资源已存在 / Resource already exists
            ResourceNotExistError: Agent Runtime 不存在 / Agent Runtime does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        cli = cls.__get_client()
        return await cli.create_endpoint_async(
            agent_runtime_id,
            input,
            config=config,
        )

    @classmethod
    async def delete_by_id_async(
        cls,
        agent_runtime_id: str,
        endpoint_id: str,
        config: Optional[Config] = None,
    ):
        """根据 ID 异步删除端点 / Delete endpoint by ID asynchronously

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
        cli = cls.__get_client()
        return await cli.delete_endpoint_async(
            agent_runtime_id,
            endpoint_id,
            config=config,
        )

    @classmethod
    async def update_by_id_async(
        cls,
        agent_runtime_id: str,
        endpoint_id: str,
        input: AgentRuntimeEndpointUpdateInput,
        config: Optional[Config] = None,
    ):
        """根据 ID 异步更新端点 / Update endpoint by ID asynchronously

        Args:
            agent_runtime_id: Agent Runtime ID
            endpoint_id: 端点 ID / Endpoint ID
            input: 端点更新配置 / Endpoint update configuration
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            AgentRuntimeEndpoint: 更新后的端点对象 / Updated endpoint object

        Raises:
            ResourceNotExistError: 资源不存在 / Resource does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        cli = cls.__get_client()
        return await cli.update_endpoint_async(
            agent_runtime_id,
            endpoint_id,
            input,
            config=config,
        )

    @classmethod
    async def get_by_id_async(
        cls,
        agent_runtime_id: str,
        endpoint_id: str,
        config: Optional[Config] = None,
    ):
        """根据 ID 异步获取端点 / Get endpoint by ID asynchronously

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
        cli = cls.__get_client()
        return await cli.get_endpoint_async(
            agent_runtime_id,
            endpoint_id,
            config=config,
        )

    @classmethod
    async def list_by_id_async(
        cls, agent_runtime_id: str, config: Optional[Config] = None
    ):
        """根据 Agent Runtime ID 异步列出所有端点 / List all endpoints by Agent Runtime ID asynchronously

        此方法会自动分页获取所有端点并去重。
        This method automatically paginates to get all endpoints and deduplicates them.

        Args:
            agent_runtime_id: Agent Runtime ID
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            List[AgentRuntimeEndpoint]: 端点对象列表 / List of endpoint objects

        Raises:
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        cli = cls.__get_client()

        endpoints: List[AgentRuntimeEndpoint] = []
        page = 1
        page_size = 50
        while True:
            result = await cli.list_endpoints_async(
                agent_runtime_id,
                AgentRuntimeEndpointListInput(
                    page_number=page,
                    page_size=page_size,
                ),
                config=config,
            )
            page += 1
            endpoints.extend(result)  # type: ignore
            if len(result) < page_size:
                break

        endpoint_id_set = set()
        results: List[AgentRuntimeEndpoint] = []
        for endpoint in endpoints:
            if endpoint.agent_runtime_endpoint_id not in endpoint_id_set:
                endpoint_id_set.add(endpoint.agent_runtime_endpoint_id)
                results.append(endpoint)

        return results

    async def delete_async(
        self, config: Optional[Config] = None
    ) -> "AgentRuntimeEndpoint":
        """异步删除当前端点 / Delete current endpoint asynchronously

        Args:
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            AgentRuntimeEndpoint: 删除后的端点对象(self) / Deleted endpoint object (self)

        Raises:
            ValueError: 当 agent_runtime_id 或 agent_runtime_endpoint_id 为空时 / When agent_runtime_id or agent_runtime_endpoint_id is None
            ResourceNotExistError: 资源不存在 / Resource does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        if (
            self.agent_runtime_id is None
            or self.agent_runtime_endpoint_id is None
        ):
            raise ValueError(
                "agent_runtime_id and agent_runtime_endpoint_id are required to"
                " delete an Agent Runtime Endpoint"
            )

        result = await self.delete_by_id_async(
            self.agent_runtime_id, self.agent_runtime_endpoint_id, config=config
        )
        self.update_self(result)
        return self

    async def update_async(
        self,
        input: AgentRuntimeEndpointUpdateInput,
        config: Optional[Config] = None,
    ) -> Self:
        """异步更新当前端点 / Update current endpoint asynchronously

        Args:
            input: 端点更新配置 / Endpoint update configuration
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            Self: 更新后的端点对象(self) / Updated endpoint object (self)

        Raises:
            ValueError: 当 agent_runtime_id 或 agent_runtime_endpoint_id 为空时 / When agent_runtime_id or agent_runtime_endpoint_id is None
            ResourceNotExistError: 资源不存在 / Resource does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        if (
            self.agent_runtime_id is None
            or self.agent_runtime_endpoint_id is None
        ):
            raise ValueError(
                "agent_runtime_id and agent_runtime_endpoint_id are required to"
                " update an Agent Runtime Endpoint"
            )

        result = await self.update_by_id_async(
            self.agent_runtime_id,
            self.agent_runtime_endpoint_id,
            input=input,
            config=config,
        )
        self.update_self(result)
        return self

    async def get_async(self, config: Optional[Config] = None):
        """异步获取当前端点信息 / Get current endpoint information asynchronously

        Args:
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            AgentRuntimeEndpoint: 获取后的端点对象(self) / Retrieved endpoint object (self)

        Raises:
            ValueError: 当 agent_runtime_id 或 agent_runtime_endpoint_id 为空时 / When agent_runtime_id or agent_runtime_endpoint_id is None
            ResourceNotExistError: 资源不存在 / Resource does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        if (
            self.agent_runtime_id is None
            or self.agent_runtime_endpoint_id is None
        ):
            raise ValueError(
                "agent_runtime_id and agent_runtime_endpoint_id are required to"
                " get an Agent Runtime Endpoint"
            )

        result = await self.get_by_id_async(
            self.agent_runtime_id, self.agent_runtime_endpoint_id
        )
        self.update_self(result)
        return self

    async def refresh_async(self, config: Optional[Config] = None):
        """刷新当前端点信息 / Refresh current endpoint information

        这是 get_async 的别名方法。
        This is an alias method for get_async.

        Args:
            config: 配置对象,可选 / Configuration object, optional

        Returns:
            AgentRuntimeEndpoint: 刷新后的端点对象(self) / Refreshed endpoint object (self)

        Raises:
            ValueError: 当 agent_runtime_id 或 agent_runtime_endpoint_id 为空时 / When agent_runtime_id or agent_runtime_endpoint_id is None
            ResourceNotExistError: 资源不存在 / Resource does not exist
            HTTPError: HTTP 请求错误 / HTTP request error
        """
        return await self.get_async(config=config)

    async def invoke_openai_async(self, **kwargs: Unpack[InvokeArgs]):
        cfg = Config.with_configs(self._config, kwargs.get("config"))
        kwargs["config"] = cfg

        if self.__data_api is None:
            if not self.__agent_runtime_name:
                client = self.__get_client(cfg)
                ar = await client.get_async(
                    id=self.agent_runtime_id or "", config=cfg
                )
                self.__agent_runtime_name = ar.agent_runtime_name

            self.__data_api = AgentRuntimeDataAPI(
                agent_runtime_name=self.__agent_runtime_name or "",
                agent_runtime_endpoint_name=self.agent_runtime_endpoint_name
                or "",
                config=cfg,
            )

        return await self.__data_api.invoke_openai_async(**kwargs)
