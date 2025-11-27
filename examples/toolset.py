import json

from agentrun.toolset.client import ToolSetClient
from agentrun.utils.log import logger


def toolset_example():
    client = ToolSetClient()
    toolset = client.get(
        name="<your-baidu-search-tool-name>",  # 替换为您的 百度搜索 工具
    )
    logger.info("%s", toolset)
    logger.info("%s", toolset.type())
    # 使用统一的 list_tools() 接口,返回 ToolInfo 列表
    tools = toolset.list_tools()
    logger.info("%s", json.dumps([t.model_dump() for t in tools]))
    logger.info(
        "%s",
        toolset.call_tool(
            name="baidu_search",
            arguments={"search_input": "比特币价格"},
        ),
    )

    toolset = client.get(
        name="your-mcp-time-tool-name"
    )  # 替换为您的 获取当前时间 工具
    logger.info("%s", toolset)
    logger.info("%s", toolset.type())
    # 使用统一的 list_tools() 接口,返回 ToolInfo 列表
    tools = toolset.list_tools()
    logger.info("%s", json.dumps([item.model_dump() for item in tools]))
    logger.info(
        "%s",
        toolset.call_tool(
            name="get_current_time",
            arguments={"timezone": "Asia/Shanghai"},
        ),
    )

    # tools = mcp.tools()
    # for tool in tools:
    #     print(f"- {tool.name}: {tool.description}")
    #     print("  ", tool.inputSchema)

    # print(
    #     mcp.call_tool(
    #         name="get_current_time",
    #         arguments={"timezone": "Asia/Shanghai"},
    #     )
    # )


if __name__ == "__main__":
    toolset_example()
