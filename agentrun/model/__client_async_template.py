"""Model Service 客户端 / Model Service Client

此模块提供模型服务和模型代理的客户端API。
This module provides the client API for model services and model proxies.
"""

from typing import List, Literal, Optional, overload, Union

from alibabacloud_agentrun20250910.models import (
    CreateModelProxyInput,
    CreateModelServiceInput,
    ListModelProxiesRequest,
    ListModelServicesRequest,
    UpdateModelProxyInput,
    UpdateModelServiceInput,
)
import pydash

from agentrun.utils.config import Config
from agentrun.utils.exception import HTTPError

from .api.control import ModelControlAPI
from .model import (
    BackendType,
    ModelProxyCreateInput,
    ModelProxyListInput,
    ModelProxyUpdateInput,
    ModelServiceCreateInput,
    ModelServiceListInput,
    ModelServiceUpdateInput,
    ProxyMode,
)
from .model_proxy import ModelProxy
from .model_service import ModelService


class ModelClient:
    """Model Service 客户端 / Model Service Client

    提供模型服务和模型代理的创建、删除、更新和查询功能。
    Provides create, delete, update and query functions for model services and model proxies.
    """

    def __init__(self, config: Optional[Config] = None):
        """初始化客户端 / Initialize client

        Args:
            config: 配置对象,可选 / Configuration object, optional
        """
        self.__control_api = ModelControlAPI(config)

    @overload
    async def create_async(
        self, input: ModelServiceCreateInput, config: Optional[Config] = None
    ) -> ModelService:
        ...

    @overload
    async def create_async(
        self, input: ModelProxyCreateInput, config: Optional[Config] = None
    ) -> ModelProxy:
        ...

    async def create_async(
        self,
        input: Union[ModelServiceCreateInput, ModelProxyCreateInput],
        config: Optional[Config] = None,
    ):
        """创建模型服务（异步）

        Args:
            input: 模型服务输入参数
            config: 配置

        Returns:
            model: 创建的对象
        """
        try:
            if isinstance(input, ModelProxyCreateInput):
                if not input.proxy_mode:
                    input.proxy_mode = (
                        ProxyMode.MULTI
                        if len(pydash.get(input, "proxy_config.endpoints", []))
                        > 1
                        else ProxyMode.SINGLE
                    )

                result = await self.__control_api.create_model_proxy_async(
                    CreateModelProxyInput().from_map(input.model_dump()),
                    config=config,
                )
                return ModelProxy.from_inner_object(result)
            else:
                result = await self.__control_api.create_model_service_async(
                    CreateModelServiceInput().from_map(input.model_dump()),
                    config=config,
                )
                return ModelService.from_inner_object(result)
        except HTTPError as e:
            raise e.to_resource_error(
                "Model",
                (
                    input.model_proxy_name
                    if isinstance(input, ModelProxyCreateInput)
                    else input.model_service_name
                ),
            ) from e

    @overload
    async def delete_async(
        self,
        name: str,
        backend_type: Literal[BackendType.PROXY],
        config: Optional[Config] = None,
    ) -> ModelProxy:
        ...

    @overload
    async def delete_async(
        self,
        name: str,
        backend_type: Literal[BackendType.SERVICE],
        config: Optional[Config] = None,
    ) -> ModelService:
        ...

    @overload
    async def delete_async(
        self,
        name: str,
        backend_type: None = None,
        config: Optional[Config] = None,
    ) -> Union[ModelProxy, ModelService]:
        ...

    async def delete_async(
        self,
        name: str,
        backend_type: Optional[BackendType] = None,
        config: Optional[Config] = None,
    ):
        """删除模型服务（异步）

        Args:
            model_service_name: 模型服务名称
            config: 配置

        Raises:
            ResourceNotExistError: 模型服务不存在
        """

        error: Optional[HTTPError] = None
        if backend_type == BackendType.PROXY or backend_type is None:
            try:
                result = await self.__control_api.delete_model_proxy_async(
                    model_proxy_name=name, config=config
                )
                return ModelProxy.from_inner_object(result)
            except HTTPError as e:
                error = e

        if backend_type == BackendType.PROXY and error is not None:
            raise error.to_resource_error("Model", name) from error

        try:
            result = await self.__control_api.delete_model_service_async(
                model_service_name=name, config=config
            )
            return ModelService.from_inner_object(result)
        except HTTPError as e:
            raise e.to_resource_error("Model", name) from error

    @overload
    async def update_async(
        self,
        name: str,
        input: ModelServiceUpdateInput,
        config: Optional[Config] = None,
    ) -> ModelService:
        ...

    @overload
    async def update_async(
        self,
        name: str,
        input: ModelProxyUpdateInput,
        config: Optional[Config] = None,
    ) -> ModelProxy:
        ...

    async def update_async(
        self,
        name: str,
        input: Union[ModelServiceUpdateInput, ModelProxyUpdateInput],
        config: Optional[Config] = None,
    ):
        """更新模型服务（异步）

        Args:
            model_service_name: 模型服务名称
            input: 模型服务更新输入参数
            config: 配置

        Returns:
            ModelService: 更新后的模型服务对象

        Raises:
            ResourceNotExistError: 模型服务不存在
        """

        if isinstance(input, ModelProxyUpdateInput):
            try:
                if not input.proxy_mode:
                    input.proxy_mode = (
                        ProxyMode.MULTI
                        if len(pydash.get(input, "proxy_config.endpoints", []))
                        > 1
                        else ProxyMode.SINGLE
                    )
                result = await self.__control_api.update_model_proxy_async(
                    model_proxy_name=name,
                    input=UpdateModelProxyInput().from_map(input.model_dump()),
                    config=config,
                )
                return ModelProxy.from_inner_object(result)
            except HTTPError as e:
                raise e.to_resource_error("Model", name) from e

        else:
            try:
                result = await self.__control_api.update_model_service_async(
                    model_service_name=name,
                    input=UpdateModelServiceInput().from_map(
                        input.model_dump()
                    ),
                    config=config,
                )
                return ModelService.from_inner_object(result)
            except HTTPError as e:
                raise e.to_resource_error("Model", name) from e

    @overload
    async def get_async(
        self,
        name: str,
        backend_type: Literal[BackendType.SERVICE],
        config: Optional[Config] = None,
    ) -> ModelService:
        ...

    @overload
    async def get_async(
        self,
        name: str,
        backend_type: Literal[BackendType.PROXY],
        config: Optional[Config] = None,
    ) -> ModelProxy:
        ...

    @overload
    async def get_async(
        self,
        name: str,
        backend_type: None = None,
        config: Optional[Config] = None,
    ) -> Union[ModelService, ModelProxy]:
        ...

    async def get_async(
        self,
        name: str,
        backend_type: Optional[BackendType] = None,
        config: Optional[Config] = None,
    ):
        """获取模型服务（异步）

        Args:
            model_service_name: 模型服务名称
            config: 配置

        Returns:
            ModelService: 模型服务对象

        Raises:
            ResourceNotExistError: 模型服务不存在
        """

        error: Optional[HTTPError] = None
        if backend_type == BackendType.PROXY or backend_type is None:
            try:
                result = await self.__control_api.get_model_proxy_async(
                    model_proxy_name=name, config=config
                )
                return ModelProxy.from_inner_object(result)
            except HTTPError as e:
                error = e

        if backend_type == BackendType.PROXY and error is not None:
            raise error.to_resource_error("Model", name) from error

        try:
            result = await self.__control_api.get_model_service_async(
                model_service_name=name, config=config
            )
            return ModelService.from_inner_object(result)
        except HTTPError as e:
            raise e.to_resource_error("Model", name) from e

    @overload
    async def list_async(
        self,
        input: ModelProxyListInput,
        config: Optional[Config] = None,
    ) -> List[ModelProxy]:
        ...

    @overload
    async def list_async(
        self,
        input: ModelServiceListInput,
        config: Optional[Config] = None,
    ) -> List[ModelService]:
        ...

    async def list_async(
        self,
        input: Union[ModelServiceListInput, ModelProxyListInput],
        config: Optional[Config] = None,
    ):
        """列出模型服务（异步）

        Args:
            input: 分页查询参数
            config: 配置

        Returns:
            List[ModelService]: 模型服务列表
        """

        if isinstance(input, ModelProxyListInput):
            result = await self.__control_api.list_model_proxies_async(
                ListModelProxiesRequest().from_map(input.model_dump()),
                config=config,
            )
            return [
                ModelProxy.from_inner_object(item)
                for item in result.items or []
            ]
        else:
            result = await self.__control_api.list_model_services_async(
                ListModelServicesRequest().from_map(input.model_dump()),
                config=config,
            )
            return [
                ModelService.from_inner_object(item)
                for item in result.items or []
            ]
