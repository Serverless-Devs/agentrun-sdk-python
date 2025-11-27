from typing import Iterable, Optional, TypedDict

from openai.types.chat import ChatCompletionMessageParam
from typing_extensions import Required, Unpack

from agentrun.utils.config import Config
from agentrun.utils.data_api import DataAPI, ResourceType


class InvokeArgs(TypedDict):
    messages: Required[Iterable[ChatCompletionMessageParam]]
    stream: Required[bool]
    config: Optional[Config]


class AgentRuntimeDataAPI(DataAPI):

    def __init__(
        self,
        agent_runtime_name: str,
        agent_runtime_endpoint_name: str = "Default",
        config: Optional[Config] = None,
    ):
        super().__init__(
            resource_name=agent_runtime_name,
            resource_type=ResourceType.Runtime,
            namespace=f"agent-runtimes/{agent_runtime_name}/endpoints/{agent_runtime_endpoint_name}/invocations",
            config=config,
        )

    async def invoke_openai_async(
        self,
        **kwargs: Unpack[InvokeArgs],
    ):
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        config = kwargs.get("config", None)

        cfg = Config.with_configs(self.config, config)
        api_base = self.with_path("openai/v1")
        _, headers, _ = self.auth(headers=cfg.get_headers())

        from httpx import AsyncClient
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key="",
            base_url=api_base,
            http_client=AsyncClient(headers=headers),
        )
        timeout = cfg.get_timeout()

        return client.chat.completions.create(
            model=self.resource_name,
            messages=messages,
            timeout=timeout,
            stream=stream,
        )
