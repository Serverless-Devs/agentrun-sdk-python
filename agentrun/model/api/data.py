from typing import Dict, Optional, TYPE_CHECKING, Union

from agentrun.utils.config import Config

if TYPE_CHECKING:
    from litellm import ResponseInputParam

from agentrun.utils.data_api import DataAPI, ResourceType
from agentrun.utils.helper import mask_password
from agentrun.utils.log import logger
from agentrun.utils.model import BaseModel


class BaseInfo(BaseModel):
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    provider: Optional[str] = None


class ModelCompletionAPI:

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        headers: Optional[dict] = None,
        provider: str = "openai",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.provider = provider
        self.headers = headers or {}

    def completions(
        self,
        messages: list = [],
        model: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        logger.debug(
            "ModelCompletionAPI completions called %s, api_key: %s,"
            " messages: %s",
            self.base_url,
            mask_password(self.api_key),
            messages,
        )
        kwargs["headers"] = {
            **self.headers,
            **kwargs.get("headers", {}),
        }
        if kwargs.get("stream_options") is None:
            kwargs["stream_options"] = {}
        kwargs["stream_options"]["include_usage"] = True

        from litellm import completion

        return completion(
            **kwargs,
            api_key=self.api_key,
            base_url=self.base_url,
            model=model or self.model,
            custom_llm_provider=custom_llm_provider or self.provider,
            messages=messages,
        )

    def responses(
        self,
        input: Union[str, "ResponseInputParam"],
        model: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        logger.debug(
            "ModelCompletionAPI responses called %s, api_key: %s, input: %s",
            self.base_url,
            mask_password(self.api_key),
            input,
        )
        kwargs["headers"] = {
            **self.headers,
            **kwargs.get("headers", {}),
        }
        if kwargs.get("stream_options") is None:
            kwargs["stream_options"] = {}
        kwargs["stream_options"]["include_usage"] = True
        from litellm import responses

        return responses(
            **kwargs,
            api_key=self.api_key,
            base_url=self.base_url,
            model=model or self.model,
            custom_llm_provider=custom_llm_provider or self.provider,
            input=input,
        )


class ModelDataAPI(DataAPI):

    def __init__(
        self,
        model_proxy_name: str,
        model_name: Optional[str] = None,
        credential_name: Optional[str] = None,
        provider: Optional[str] = "openai",
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(
            resource_name=model_proxy_name,
            resource_type=ResourceType.LiteLLM,
            config=config,
        )
        self.update_model_name(
            model_proxy_name=model_proxy_name,
            model_name=model_name,
            credential_name=credential_name,
            provider=provider,
            config=config,
        )

    def update_model_name(
        self,
        model_proxy_name,
        model_name: Optional[str],
        credential_name: Optional[str] = None,
        provider: Optional[str] = "openai",
        config: Optional[Config] = None,
    ):
        self.model_proxy_name = model_proxy_name
        self.namespace = f"models/{self.model_proxy_name}"
        self.model_name = model_name
        self.provider = provider
        self.access_token = None
        if not credential_name:
            self.access_token = ""

        self.config.update(config)

    def model_info(self, config: Optional[Config] = None) -> BaseInfo:
        cfg = Config.with_configs(self.config, config)
        _, headers, _ = self.auth(headers=self.config.get_headers(), config=cfg)

        return BaseInfo(
            api_key="",
            base_url=self.with_path("/v1").rstrip("/"),
            model=self.model_name or "",
            headers=headers,
            provider=self.provider or "openai",
        )

    def completions(
        self,
        messages: list = [],
        model: Optional[str] = None,
        config: Optional[Config] = None,
        **kwargs,
    ):
        info = self.model_info(config=config)

        return ModelCompletionAPI(
            base_url=info.base_url or "",
            api_key=info.api_key or "",
            model=model or info.model or "",
            headers=info.headers,
        ).completions(
            **kwargs,
            messages=messages,
            custom_llm_provider="openai",
        )

    def responses(
        self,
        input: Union[str, "ResponseInputParam"],
        model: Optional[str] = None,
        config: Optional[Config] = None,
        **kwargs,
    ):
        info = self.model_info(config=config)

        return ModelCompletionAPI(
            base_url=info.base_url or "",
            api_key=info.api_key or "",
            model=model or info.model or "",
            headers=info.headers,
        ).responses(
            **kwargs,
            custom_llm_provider="openai",
            model=model,
            input=input,
        )
