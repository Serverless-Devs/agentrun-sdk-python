"""通用集成工具模块 / 通用集成工具 Module

提供跨框架的通用工具定义和转换功能。 / 提供跨framework的通用工具定义和转换功能。

支持多种工具定义方式:
1. 使用 @tool 装饰器 - 最简单,推荐
2. 使用 Pydantic BaseModel - 类型安全
3. 使用 ToolParameter 列表 - 灵活但繁琐
4. 从 AgentRun ToolSet 转换 - 直接使用远程工具集

支持模型集成:
1. 使用 model() 函数 - 获取模型并转换为各种框架格式 / 1. 使用 model() 函数 - 获取模型并转换为各种framework格式
2. 从已有模型对象创建 - from_agentrun_model()

Examples:
>>> # 工具集成示例
>>> # 方式 1: 使用装饰器从函数创建 Tool
>>> from agentrun.integration.common import tool
>>> from typing import Literal
>>>
>>> @tool()
... def calculator(
...     operation: Literal["add", "subtract"],
...     a: float,
...     b: float,
... ) -> float:
...     '''执行数学运算'''
...     if operation == "add":
...         return a + b
...     return a - b
>>>
>>> # 方式 2: 使用 Pydantic 模型
>>> from agentrun.integration.common import Tool
>>> from pydantic import BaseModel, Field
>>>
>>> class SearchArgs(BaseModel):
...     query: str = Field(description="搜索关键词")
...     limit: int = Field(description="结果数量", ge=1, le=100, default=10)
>>>
>>> search_tool = Tool(
...     name="search",
...     description="搜索网络信息",
...     args_schema=SearchArgs,
...     func=lambda query, limit: f"搜索: {query}"
... )
>>>
>>> # 方式 3: 使用 ToolParameter
>>> from agentrun.integration.common import ToolParameter
>>>
>>> weather_tool = Tool(
...     name="weather",
...     description="获取天气信息",
...     parameters=[
...         ToolParameter(
...             name="city",
...             param_type="string",
...             description="城市名称",
...             required=True
...         )
...     ]
... )
>>>
>>> # 方式 4: 从 AgentRun ToolSet 转换
>>> from agentrun.toolset.client import ToolSetClient
>>> from agentrun.integration.common import from_agentrun_toolset
>>>
>>> client = ToolSetClient()
>>> remote_toolset = client.get(name="my-toolset")
>>> common_toolset = from_agentrun_toolset(remote_toolset)
>>>
>>> # 导出为各种框架格式 / >>> # 导出为各种framework格式
>>> openai_tools = common_toolset.to_openai_function()
>>> langchain_tools = common_toolset.to_langchain()
>>>
>>> # 模型集成示例
>>> # 方式 1: 使用 model() 函数
>>> from agentrun.integration.common import model
>>>
>>> # 获取模型并转换为 LangChain
>>> llm = model("my-model").to_langchain()
>>>
>>> # 转换为 Google ADK
>>> google_model = model("my-model").to_google_adk()
>>>
>>> # 方式 2: 从已有模型对象创建
>>> from agentrun.model.client import ModelClient
>>> from agentrun.integration.common import from_agentrun_model
>>>
>>> client = ModelClient()
>>> remote_model = client.get("my-model")
>>> common_model = from_agentrun_model(remote_model)
>>> llm = common_model.to_langchain()"""

from agentrun.integration.utils.model import CommonModel
from agentrun.integration.utils.tool import (
    CommonToolSet,
    from_pydantic,
    Tool,
    tool,
    ToolParameter,
)

__all__ = [
    # Tool related
    "Tool",
    "ToolParameter",
    "CommonToolSet",
    "tool",
    "from_pydantic",
    # Model related
    "CommonModel",
]
