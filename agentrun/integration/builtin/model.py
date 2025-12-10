"""内置模型集成函数 / Built-in Model Integration Functions

提供快速创建通用模型对象的便捷函数。
Provides convenient functions for quickly creating common model objects.
"""

from typing import Optional, overload, TypedDict, Union

from typing_extensions import NotRequired, Unpack

from agentrun.integration.utils.model import CommonModel
from agentrun.model import ModelService
from agentrun.model.model import BackendType
from agentrun.model.model_proxy import ModelProxy
from agentrun.utils.config import Config


class ModelArgs(TypedDict):
    model: NotRequired[Optional[str]]
    backend_type: NotRequired[Optional[BackendType]]
    config: NotRequired[Optional[Config]]


@overload
def model(input: ModelService, **kwargs: Unpack[ModelArgs]) -> CommonModel:
    ...


@overload
def model(input: ModelProxy, **kwargs: Unpack[ModelArgs]) -> CommonModel:
    ...


@overload
def model(input: str, **kwargs: Unpack[ModelArgs]) -> CommonModel:
    ...


def model(
    input: Union[str, ModelProxy, ModelService], **kwargs: Unpack[ModelArgs]
) -> CommonModel:
    """获取 AgentRun 模型并封装为通用 Model 对象 / Get AgentRun model and wrap as CommonModel

    等价于 ModelClient.get(),但返回通用 Model 对象。
    Equivalent to ModelClient.get(), but returns a CommonModel object.

    Args:
        input: AgentRun 模型名称、ModelProxy 或 ModelService 实例 / Model name, ModelProxy or ModelService instance
        model: 要请求的模型名称（默认请求数组的第一个模型或模型治理的自动负载均衡模型） / Model name to request (defaults to the first model in the array or the auto-load balancing model of model governance)
        backend_type: 后端类型(PROXY 或 SERVICE) / Backend type (PROXY or SERVICE)
        config: 配置对象 / Configuration object

    Returns:
        CommonModel: 通用模型实例 / CommonModel instance

    Examples:
        >>> # 从模型名称创建 / Create from model name
        >>> m = model("qwen-max")
        >>>
        >>> # 从 ModelProxy 创建 / Create from ModelProxy
        >>> proxy = ModelProxy.get_by_name("my-proxy")
        >>> m = model(proxy)
        >>>
        >>> # 从 ModelService 创建 / Create from ModelService
        >>> service = ModelService.get_by_name("my-service")
        >>> m = model(service)
    """
    config = kwargs.get("config")
    backend_type = kwargs.get("backend_type")
    model = kwargs.get("model")

    if isinstance(input, str):
        from agentrun.model.client import ModelClient

        client = ModelClient(config=config)
        input = client.get(name=input, backend_type=backend_type, config=config)

    if isinstance(input, ModelProxy):
        return CommonModel(
            model_obj=input,
            backend_type=BackendType.PROXY,
            specific_model=model,
            config=config,
        )
    elif isinstance(input, ModelService):
        return CommonModel(
            model_obj=input,
            backend_type=BackendType.SERVICE,
            specific_model=model,
            config=config,
        )
    else:
        raise TypeError("input must be str, ModelProxy or ModelService")
