import os
import re
import time

from agentrun import model
from agentrun.model import (
    BackendType,
    ModelClient,
    ModelProxy,
    ModelProxyCreateInput,
    ModelProxyListInput,
    ModelProxyUpdateInput,
    ModelService,
    ModelServiceCreateInput,
    ModelServiceListInput,
    ModelServiceUpdateInput,
)
from agentrun.utils.exception import (
    ResourceAlreadyExistError,
    ResourceNotExistError,
)
from agentrun.utils.log import logger
from agentrun.utils.model import Status

base_url = os.getenv(
    "BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
api_key = os.getenv("API_KEY", "sk-xxxxx")
model_names = re.split(r"\s|,", os.getenv("MODEL_NAMES", "qwen-max").strip())


client = ModelClient()
model_service_name = "sdk-test-model-service"
model_proxy_name = "sdk-test-model-proxy"


def create_or_get_model_service():
    """
    为您演示如何进行创建 / 获取
    """
    logger.info("创建或获取已有的资源")

    try:
        ms = client.create(
            ModelServiceCreateInput(
                model_service_name=model_service_name,
                description="测试模型服务",
                model_type=model.ModelType.LLM,
                provider="openai",
                provider_settings=model.ProviderSettings(
                    api_key=api_key,
                    base_url=base_url,
                    model_names=model_names,
                ),
            )
        )
    except ResourceAlreadyExistError:
        logger.info("已存在，获取已有资源")
        ms = client.get(
            name=model_service_name, backend_type=BackendType.SERVICE
        )

    ms.wait_until_ready_or_failed()
    if ms.status != Status.READY:
        raise Exception(f"状态异常：{ms.status}")

    logger.info("已就绪状态，当前信息: %s", ms)

    return ms


def update_model_service(ms: ModelService):
    """
    为您演示如何进行更新
    """
    logger.info("更新描述为当前时间")

    # 也可以使用 client.update
    ms.update(
        ModelServiceUpdateInput(description=f"当前时间戳：{time.time()}"),
    )
    ms.wait_until_ready_or_failed()
    if ms.status != Status.READY:
        raise Exception(f"状态异常：{ms.status}")

    logger.info("更新成功，当前信息: %s", ms)


def list_model_services():
    """
    为您演示如何进行枚举
    """
    logger.info("枚举资源列表")
    ms_arr = client.list(ModelServiceListInput(model_type=model.ModelType.LLM))
    logger.info(
        "共有 %d 个资源，分别为 %s",
        len(ms_arr),
        [c.model_service_name for c in ms_arr],
    )


def delete_model_service(ms: ModelService):
    """
    为您演示如何进行删除
    """
    logger.info("开始清理资源")
    # 也可以使用 client.delete / cred.delete + 轮询状态
    ms.delete_and_wait_until_finished()

    logger.info("再次尝试获取")
    try:
        ms.refresh()
    except ResourceNotExistError as e:
        logger.info("得到资源不存在报错，删除成功，%s", e)


def invoke_model_service(ms: ModelService):
    logger.info("调用模型服务进行推理")

    result = ms.completions(
        messages=[{"role": "user", "content": "写一首赞美 AI 大模型的诗歌"}],
        stream=True,
    )
    for chunk in result:
        from litellm.types.utils import ModelResponseStream

        assert isinstance(chunk, ModelResponseStream)
        print(chunk.choices[0].delta.content, end="", flush=True)
    logger.info("")


def create_or_get_model_proxy():
    """
    为您演示如何进行创建 / 获取
    """
    logger.info("创建或获取已有的资源")

    try:
        cred = client.create(
            ModelProxyCreateInput(
                model_proxy_name=model_proxy_name,
                description="测试模型治理",
                model_type=model.ModelType.LLM,
                proxy_config=model.ProxyConfig(
                    endpoints=[
                        model.ProxyConfigEndpoint(
                            model_names=[model_name],
                            model_service_name="test-model-service",
                        )
                        for model_name in model_names
                    ],
                    policies={},
                ),
            )
        )
    except ResourceAlreadyExistError:
        logger.info("已存在，获取已有资源")
        cred = client.get(name=model_proxy_name, backend_type=BackendType.PROXY)

    cred.wait_until_ready_or_failed()
    if cred.status != Status.READY:
        raise Exception(f"状态异常：{cred.status}")

    logger.info("已就绪状态，当前信息: %s", cred)

    return cred


def update_model_proxy(mp: ModelProxy):
    """
    为您演示如何进行更新
    """
    logger.info("更新描述为当前时间")

    # 也可以使用 client.update
    mp.update(
        ModelProxyUpdateInput(description=f"当前时间戳：{time.time()}"),
    )
    mp.wait_until_ready_or_failed()
    if mp.status != Status.READY:
        raise Exception(f"状态异常：{mp.status}")

    logger.info("更新成功，当前信息: %s", mp)


def list_model_proxies():
    """
    为您演示如何进行枚举
    """
    logger.info("枚举资源列表")
    mp_arr = client.list(ModelProxyListInput())
    logger.info(
        "共有 %d 个资源，分别为 %s",
        len(mp_arr),
        [c.model_proxy_name for c in mp_arr],
    )


def delete_model_proxy(mp: ModelProxy):
    """
    为您演示如何进行删除
    """
    logger.info("开始清理资源")
    # 也可以使用 client.delete / cred.delete + 轮询状态
    mp.delete_and_wait_until_finished()

    logger.info("再次尝试获取")
    try:
        mp.refresh()
    except ResourceNotExistError as e:
        logger.info("得到资源不存在报错，删除成功，%s", e)


def invoke_model_proxy(mp: ModelProxy):
    logger.info("调用模型服务进行推理")

    result = mp.completions(
        messages=[{"role": "user", "content": "写一首赞美 AI 大模型的诗歌"}],
        stream=True,
    )
    for chunk in result:
        from litellm.types.utils import ModelResponseStream

        assert isinstance(chunk, ModelResponseStream)
        print(chunk.choices[0].delta.content, end="", flush=True)
    logger.info("")


def model_example():
    """
    为您演示模型模块的基本功能
    """
    logger.info("==== 模型模块基本功能示例 ====")
    logger.info("    base_url=%s", base_url)
    logger.info("    api_key=%s", len(api_key) * "*")
    logger.info("    model_names=%s", model_names)

    list_model_services()
    ms = create_or_get_model_service()
    update_model_service(ms)

    invoke_model_service(ms)

    list_model_proxies()
    mp = create_or_get_model_proxy()
    update_model_proxy(mp)
    # invoke_model_proxy(mp)
    delete_model_proxy(mp)
    list_model_proxies()

    delete_model_service(ms)
    list_model_services()


if __name__ == "__main__":
    model_example()
