"""Model Service 高层 API / Model Service High-Level API

此模块定义模型服务资源的高级API。
This module defines the high-level API for model service resources.
"""

from typing import List, Optional

from agentrun.model.api.data import BaseInfo, ModelCompletionAPI
from agentrun.utils.config import Config
from agentrun.utils.model import PageableInput
from agentrun.utils.resource import ResourceBase

from .model import (
    BackendType,
    ModelServiceCreateInput,
    ModelServiceImmutableProps,
    ModelServiceListInput,
    ModelServiceMutableProps,
    ModelServicesSystemProps,
    ModelServiceUpdateInput,
    ModelType,
)


class ModelService(
    ModelServiceImmutableProps,
    ModelServiceMutableProps,
    ModelServicesSystemProps,
    ResourceBase,
):
    """模型服务"""

    @classmethod
    def __get_client(cls):
        from .client import ModelClient

        return ModelClient()

    @classmethod
    async def create_async(
        cls, input: ModelServiceCreateInput, config: Optional[Config] = None
    ):
        """创建模型服务（异步）

        Args:
            input: 模型服务输入参数
            config: 配置

        Returns:
            ModelService: 创建的模型服务对象
        """
        return await cls.__get_client().create_async(input, config=config)

    @classmethod
    async def delete_by_name_async(
        cls, model_service_name: str, config: Optional[Config] = None
    ):
        """根据名称删除模型服务（异步）

        Args:
            model_service_name: 模型服务名称
            config: 配置
        """
        return await cls.__get_client().delete_async(
            model_service_name, backend_type=BackendType.SERVICE, config=config
        )

    @classmethod
    async def update_by_name_async(
        cls,
        model_service_name: str,
        input: ModelServiceUpdateInput,
        config: Optional[Config] = None,
    ):
        """根据名称更新模型服务（异步）

        Args:
            model_service_name: 模型服务名称
            input: 模型服务更新输入参数
            config: 配置

        Returns:
            ModelService: 更新后的模型服务对象
        """
        return await cls.__get_client().update_async(
            model_service_name, input, config=config
        )

    @classmethod
    async def get_by_name_async(
        cls, model_service_name: str, config: Optional[Config] = None
    ):
        """根据名称获取模型服务（异步）

        Args:
            model_service_name: 模型服务名称
            config: 配置

        Returns:
            ModelService: 模型服务对象
        """
        return await cls.__get_client().get_async(
            model_service_name, backend_type=BackendType.SERVICE, config=config
        )

    @classmethod
    async def _list_page_async(
        cls, page_input: PageableInput, config: Config | None = None, **kwargs
    ):
        return await cls.__get_client().list_async(
            input=ModelServiceListInput(
                **kwargs,
                **page_input.model_dump(),
            ),
            config=config,
        )

    @classmethod
    async def list_all_async(
        cls,
        *,
        model_type: Optional[ModelType] = None,
        provider: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> List["ModelService"]:
        return await cls._list_all_async(
            lambda m: m.model_service_id or "",
            config=config,
            model_type=model_type,
            provider=provider,
        )

    async def update_async(
        self, input: ModelServiceUpdateInput, config: Optional[Config] = None
    ):
        """更新模型服务（异步）

        Args:
            input: 模型服务更新输入参数
            config: 配置

        Returns:
            ModelService: 更新后的模型服务对象
        """
        if self.model_service_name is None:
            raise ValueError(
                "model_service_name is required to update a ModelService"
            )

        result = await self.update_by_name_async(
            self.model_service_name, input, config=config
        )
        self.update_self(result)

        return self

    async def delete_async(self, config: Optional[Config] = None):
        """删除模型服务（异步）

        Args:
            config: 配置
        """
        if self.model_service_name is None:
            raise ValueError(
                "model_service_name is required to delete a ModelService"
            )

        return await self.delete_by_name_async(
            self.model_service_name, config=config
        )

    async def get_async(self, config: Optional[Config] = None):
        """刷新模型服务信息（异步）

        Args:
            config: 配置

        Returns:
            ModelService: 刷新后的模型服务对象
        """
        if self.model_service_name is None:
            raise ValueError(
                "model_service_name is required to refresh a ModelService"
            )

        result = await self.get_by_name_async(
            self.model_service_name, config=config
        )
        self.update_self(result)

        return self

    async def refresh_async(self, config: Optional[Config] = None):
        """刷新模型服务信息（异步）

        Args:
            config: 配置

        Returns:
            ModelService: 刷新后的模型服务对象
        """
        return await self.get_async(config=config)

    def model_info(self, config: Optional[Config] = None) -> BaseInfo:
        cfg = Config.with_configs(self._config, config)

        assert self.provider_settings is not None
        assert self.provider_settings.base_url is not None

        api_key = self.provider_settings.api_key or ""
        if not api_key and self.credential_name:
            from agentrun.credential import Credential

            credential = Credential.get_by_name(
                self.credential_name, config=cfg
            )
            api_key = credential.credential_secret or ""

        default_model = (
            self.provider_settings.model_names[0]
            if self.provider_settings.model_names is not None
            and len(self.provider_settings.model_names) > 0
            else None
        )

        return BaseInfo(
            api_key=api_key,
            base_url=self.provider_settings.base_url,
            model=default_model,
            headers=cfg.get_headers(),
        )

    def completions(
        self,
        messages: list,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ):
        info = self.model_info(config=kwargs.get("config"))

        m = ModelCompletionAPI(
            api_key=info.api_key or "",
            base_url=info.base_url or "",
            model=model or info.model or self.model_service_name or "",
        )

        return m.completions(**kwargs, messages=messages, stream=stream)

    def responses(
        self,
        messages: list,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ):
        info = self.model_info(config=kwargs.get("config"))

        m = ModelCompletionAPI(
            api_key=info.api_key or "",
            base_url=info.base_url or "",
            model=model or info.model or self.model_service_name or "",
            provider=(self.provider or "openai").lower(),
        )

        return m.responses(**kwargs, messages=messages, stream=stream)
