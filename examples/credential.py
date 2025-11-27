import time

from agentrun.credential import (
    Credential,
    CredentialClient,
    CredentialConfig,
    CredentialCreateInput,
    CredentialUpdateInput,
)
from agentrun.utils.exception import (
    ResourceAlreadyExistError,
    ResourceNotExistError,
)
from agentrun.utils.log import logger

client = CredentialClient()
credential_name = "sdk-test-credential"


def create_or_get_credential():
    """
    为您演示如何进行创建 / 获取
    """
    logger.info("创建或获取已有的资源")

    try:
        cred = client.create(
            CredentialCreateInput(
                credential_name=credential_name,
                description="这是通过 SDK 创建的测试凭证",
                credential_config=CredentialConfig.inbound_api_key(
                    "sk-test-123456"
                ),
            )
        )
    except ResourceAlreadyExistError:
        logger.info("已存在，获取已有资源")
        cred = client.get(credential_name=credential_name)

    #

    logger.info("已就绪状态，当前信息: %s", cred)

    return cred


def update_credential(cred: Credential):
    """
    为您演示如何进行更新
    """
    logger.info("更新描述为当前时间")

    # 也可以使用 client.update
    cred.update(
        CredentialUpdateInput(description=f"当前时间戳：{time.time()}"),
    )

    logger.info("更新成功，当前信息: %s", cred)


def list_credentials():
    """
    为您演示如何进行枚举
    """
    logger.info("枚举资源列表")
    cred_arr = client.list()
    logger.info(
        "共有 %d 个资源，分别为 %s",
        len(cred_arr),
        [c.credential_name for c in cred_arr],
    )


def delete_credential(cred: Credential):
    """
    为您演示如何进行删除
    """
    logger.info("开始清理资源")
    # 也可以使用 client.delete / cred.delete + 轮询状态
    cred.delete()

    logger.info("再次尝试获取")
    try:
        cred.refresh()
    except ResourceNotExistError as e:
        logger.info("得到资源不存在报错，删除成功，%s", e)


def credential_example():
    """
    为您演示凭证模块的基本功能
    """
    logger.info("==== 凭证模块基本功能示例 ====")
    list_credentials()
    cred = create_or_get_credential()
    list_credentials()
    update_credential(cred)
    delete_credential(cred)
    list_credentials()


if __name__ == "__main__":
    credential_example()
