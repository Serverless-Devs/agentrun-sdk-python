"""Model Proxy 高层 API / Model Proxy High-Level API

此模块定义模型代理资源的高级API。
This module defines the high-level API for model proxy resources.
"""

from typing import List, Optional

import pydash

from agentrun.model.api.data import BaseInfo, ModelDataAPI
from agentrun.utils.config import Config
from agentrun.utils.model import Status
from agentrun.utils.resource import ResourceBase

from .model import (
    BackendType,
    ModelProxyCreateInput,
    ModelProxyImmutableProps,
    ModelProxyListInput,
    ModelProxyMutableProps,
    ModelProxySystemProps,
    ModelProxyUpdateInput,
    PageableInput,
    ProxyMode,
)


class ModelProxy(
    ModelProxyImmutableProps,
    ModelProxyMutableProps,
    ModelProxySystemProps,
    ResourceBase,
):
    """模型服务"""

    _data_client: Optional[ModelDataAPI] = None

    @classmethod
    def __get_client(cls):
        from .client import ModelClient

        return ModelClient()

    @classmethod
    async def create_async(
        cls, input: ModelProxyCreateInput, config: Optional[Config] = None
    ):
        """创建模型服务（异步）

        Args:
            input: 模型服务输入参数
            config: 配置

        Returns:
            ModelProxy: 创建的模型服务对象
        """
        return await cls.__get_client().create_async(input, config=config)

    @classmethod
    async def delete_by_name_async(
        cls, model_Proxy_name: str, config: Optional[Config] = None
    ):
        """根据名称删除模型服务（异步）

        Args:
            model_Proxy_name: 模型服务名称
            config: 配置
        """
        return await cls.__get_client().delete_async(
            model_Proxy_name, backend_type=BackendType.PROXY, config=config
        )

    @classmethod
    async def update_by_name_async(
        cls,
        model_proxy_name: str,
        input: ModelProxyUpdateInput,
        config: Optional[Config] = None,
    ):
        """根据名称更新模型服务（异步）

        Args:
            model_Proxy_name: 模型服务名称
            input: 模型服务更新输入参数
            config: 配置

        Returns:
            ModelProxy: 更新后的模型服务对象
        """
        return await cls.__get_client().update_async(
            model_proxy_name, input, config=config
        )

    @classmethod
    async def get_by_name_async(
        cls, model_proxy_name: str, config: Optional[Config] = None
    ):
        """根据名称获取模型服务（异步）

        Args:
            model_Proxy_name: 模型服务名称
            config: 配置

        Returns:
            ModelProxy: 模型服务对象
        """
        return await cls.__get_client().get_async(
            model_proxy_name, backend_type=BackendType.PROXY, config=config
        )

    @classmethod
    async def _list_page_async(
        cls, page_input: PageableInput, config: Config | None = None, **kwargs
    ):
        return await cls.__get_client().list_async(
            input=ModelProxyListInput(
                **kwargs,
                **page_input.model_dump(),
            ),
            config=config,
        )

    @classmethod
    async def list_all_async(
        cls,
        *,
        proxy_mode: Optional[str] = None,
        status: Optional[Status] = None,
        config: Optional[Config] = None,
    ) -> List["ModelProxy"]:
        return await cls._list_all_async(
            lambda m: m.model_proxy_id or "",
            config=config,
            proxy_mode=proxy_mode,
            status=status,
        )

    async def update_async(
        self, input: ModelProxyUpdateInput, config: Optional[Config] = None
    ):
        """更新模型服务（异步）

        Args:
            input: 模型服务更新输入参数
            config: 配置

        Returns:
            ModelProxy: 更新后的模型服务对象
        """
        if self.model_proxy_name is None:
            raise ValueError(
                "model_Proxy_name is required to update a ModelProxy"
            )

        result = await self.update_by_name_async(
            self.model_proxy_name, input, config=config
        )
        self.update_self(result)

        return self

    async def delete_async(self, config: Optional[Config] = None):
        """删除模型服务（异步）

        Args:
            config: 配置
        """
        if self.model_proxy_name is None:
            raise ValueError(
                "model_Proxy_name is required to delete a ModelProxy"
            )

        return await self.delete_by_name_async(
            self.model_proxy_name, config=config
        )

    async def get_async(self, config: Optional[Config] = None):
        """刷新模型服务信息（异步）

        Args:
            config: 配置

        Returns:
            ModelProxy: 刷新后的模型服务对象
        """
        if self.model_proxy_name is None:
            raise ValueError(
                "model_Proxy_name is required to refresh a ModelProxy"
            )

        result = await self.get_by_name_async(
            self.model_proxy_name, config=config
        )
        self.update_self(result)

        return self

    async def refresh_async(self, config: Optional[Config] = None):
        """刷新模型服务信息（异步）

        Args:
            config: 配置

        Returns:
            ModelProxy: 刷新后的模型服务对象
        """
        return await self.get_async(config=config)

    def model_info(self, config: Optional[Config] = None) -> BaseInfo:
        cfg = Config.with_configs(self._config, config)

        if self._data_client is None:
            self._data_client = ModelDataAPI(
                self.model_proxy_name or "",
                credential_name=self.credential_name,
                config=cfg,
            )

        self._data_client.update_model_name(
            model_proxy_name=self.model_proxy_name,
            model_name=(
                pydash.get(self, "proxy_config.endpoints[0].model_names[0]")
                if self.proxy_mode == ProxyMode.SINGLE
                else self.model_proxy_name
            )
            or "",
            credential_name=self.credential_name,
            config=cfg,
        )

        return self._data_client.model_info()

    def completions(
        self,
        messages: list,
        model: Optional[str] = None,
        stream: bool = False,
        config: Optional[Config] = None,
        **kwargs,
    ):
        self.model_info(config)
        assert self._data_client

        return self._data_client.completions(
            **kwargs,
            messages=messages,
            model=model,
            stream=stream,
            config=config,
        )

    def responses(
        self,
        messages: list,
        model: Optional[str] = None,
        stream: bool = False,
        config: Optional[Config] = None,
        **kwargs,
    ):
        self.model_info(config)
        assert self._data_client

        return self._data_client.responses(
            **kwargs,
            messages=messages,
            model=model,
            stream=stream,
            config=config,
        )
