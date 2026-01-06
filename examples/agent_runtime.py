import os
from pathlib import Path
import time

from agentrun.agent_runtime import (
    AgentRuntime,
    AgentRuntimeClient,
    AgentRuntimeCode,
    AgentRuntimeCreateInput,
    AgentRuntimeLanguage,
    AgentRuntimeUpdateInput,
    Status,
)
from agentrun.agent_runtime.model import AgentRuntimeListInput
from agentrun.utils.exception import (
    ResourceAlreadyExistError,
    ResourceNotExistError,
)
from agentrun.utils.log import logger


def prepare_code():
    """
    完成简单的 HTTP 服务器代码，实际工作中，请替换为您的 Agent Run 代码
    """

    code = """
    const http = require('http');

    const server = http.createServer((req, res) => {
    res.statusCode = 200;
    res.setHeader('Content-Type', 'text/plain');
    res.end('Hello Agent Run');
    });

    server.listen(3000, () => {
    console.log('Server running at http://localhost:9000/');
    });
    """
    root_path = Path().cwd()
    dir_path = root_path / "examples" / "http_server_code"
    code_path = dir_path / "index.js"

    if os.path.exists(dir_path):
        import shutil

        shutil.rmtree(dir_path)

    os.mkdir(dir_path)
    with open(code_path, "w") as f:
        f.write(code)

    with open(dir_path / ".gitignore", "w") as f:
        f.write("*")

    return dir_path


client = AgentRuntimeClient()
agent_runtime_name = "sdk-test-agentruntime"


def create_or_get_agentruntime():
    """
    为您演示如何进行创建 / 获取
    """
    logger.info("创建或获取已有的资源")

    try:
        ar = client.create(
            AgentRuntimeCreateInput(
                agent_runtime_name=agent_runtime_name,
                code_configuration=AgentRuntimeCode().from_file(
                    language=AgentRuntimeLanguage.NODEJS18,
                    command=["node", "index.js"],
                    file_path=str(prepare_code()),
                ),
            )
        )
    except ResourceAlreadyExistError as e:
        logger.info("已存在，获取已有资源", e)

        ar = list(
            filter(
                lambda a: a.agent_runtime_name == agent_runtime_name,
                client.list(
                    AgentRuntimeListInput(agent_runtime_name=agent_runtime_name)
                ),
            )
        )[0]

    ar.wait_until_ready_or_failed()
    if ar.status != Status.READY:
        raise Exception(f"状态异常：{ar.status}")

    logger.info("已就绪状态，当前信息: %s", ar)

    return ar


def update_agentruntime(ar: AgentRuntime):
    """
    为您演示如何进行更新
    """
    logger.info("更新描述为当前时间")

    # 也可以使用 client.update
    ar.update(
        AgentRuntimeUpdateInput(description=f"当前时间戳：{time.time()}"),
    )
    ar.wait_until_ready_or_failed()
    if ar.status != Status.READY:
        raise Exception(f"状态异常：{ar.status}")

    logger.info("更新成功，当前信息: %s", ar)


def list_agentruntimes():
    """
    为您演示如何进行枚举
    """
    logger.info("枚举资源列表")
    agent_runtime_arr = client.list()
    logger.info(
        "共有 %d 个资源，分别为 %s",
        len(agent_runtime_arr),
        [c.agent_runtime_name for c in agent_runtime_arr],
    )


def delete_agentruntime(ar: AgentRuntime):
    """
    为您演示如何进行删除
    """
    logger.info("开始清理资源")
    # 也可以使用 client.delete / cred.delete + 轮询状态
    ar.delete_and_wait_until_finished()

    logger.info("再次尝试获取")
    try:
        ar.refresh()
    except ResourceNotExistError as e:
        logger.info("得到资源不存在报错，删除成功，%s", e)


def agentruntime_example():
    """
    为您演示凭证模块的基本功能
    """
    logger.info("==== Agent 运行时模块基本功能示例 ====")
    prepare_code()
    list_agentruntimes()
    cred = create_or_get_agentruntime()
    update_agentruntime(cred)
    delete_agentruntime(cred)
    list_agentruntimes()


if __name__ == "__main__":
    agentruntime_example()
