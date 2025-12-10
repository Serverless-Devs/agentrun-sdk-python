"""AgentRun Server 快速开始示例

curl http://127.0.0.1:9000/openai/v1/chat/completions -X POST \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "写一段代码,查询现在是几点?"}], "stream": true}'
"""

from langchain.agents import create_agent
import pydash

from agentrun.integration.langchain import convert, model, sandbox_toolset
from agentrun.sandbox import TemplateType
from agentrun.server import AgentRequest, AgentRunServer
from agentrun.utils.log import logger

# 请替换为您已经创建的 模型 和 沙箱 名称
MODEL_NAME = "<your-model-name>"
SANDBOX_NAME = "<your-sandbox-name>"

if MODEL_NAME.startswith("<"):
    raise ValueError("请将 MODEL_NAME 替换为您已经创建的模型名称")

code_interpreter_tools = []
if SANDBOX_NAME and not SANDBOX_NAME.startswith("<"):
    code_interpreter_tools = sandbox_toolset(
        template_name=SANDBOX_NAME,
        template_type=TemplateType.CODE_INTERPRETER,
        sandbox_idle_timeout_seconds=300,
    )
else:
    logger.warning("SANDBOX_NAME 未设置或未替换，跳过加载沙箱工具。")


def get_weather_tool():
    """
    获取天气工具"""
    import time

    logger.debug("调用获取天气工具")
    time.sleep(5)
    return {"weather": "晴天，25度"}


agent = create_agent(
    model=model(MODEL_NAME),
    tools=[
        *code_interpreter_tools,
        get_weather_tool,
    ],
    system_prompt="你是一个 AgentRun 的 AI 专家，可以通过沙箱运行代码来回答用户的问题。",
)


async def invoke_agent(request: AgentRequest):
    content = request.messages[0].content
    input = {"messages": [{"role": "user", "content": content}]}

    try:
        if request.stream:

            async def stream_generator():
                result = agent.astream_events(input, stream_mode="messages")
                async for event in result:
                    for item in convert(event, request.hooks):
                        yield item

            return stream_generator()
        else:
            result = agent.invoke(input)
            return pydash.get(result, "messages.-1.content")
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error("调用出错: %s", e)
        raise e


AgentRunServer(invoke_agent=invoke_agent).start()
