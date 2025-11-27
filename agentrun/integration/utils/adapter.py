"""适配器接口定义 / Adapter Interface Definition

定义统一的适配器接口,所有框架适配器都实现这些接口。
Defines unified adapter interfaces that all framework adapters implement.

这样可以确保一致的转换行为,并最大化代码复用。
This ensures consistent conversion behavior and maximizes code reuse.
"""

from abc import ABC, abstractmethod
import inspect
from typing import Any, Callable, Dict, List, Optional

from agentrun.integration.utils.canonical import CanonicalMessage, CanonicalTool
from agentrun.integration.utils.model import CommonModel


class MessageAdapter(ABC):
    """消息格式适配器接口

    用于在 ModelAdapter 内部进行消息格式转换。
    只需要将框架消息转换为标准 OpenAI 格式。

    转换流程：
    - 框架消息 → to_canonical() → CanonicalMessage（OpenAI 格式）
    """

    @abstractmethod
    def to_canonical(self, messages: Any) -> List[CanonicalMessage]:
        """将框架消息转换为标准格式（供 ModelAdapter 内部使用）

        Args:
            messages: 框架特定的消息格式

        Returns:
            标准格式消息列表
        """
        pass


class ToolAdapter(ABC):
    """工具格式适配器接口 / Utils Adapters

    用于将标准工具定义转换为框架特定格式。
        单向转换：CanonicalTool → 框架工具"""

    def __init__(self) -> None:
        super().__init__()

        # 记录工具定义，便于模型适配器回溯参数 schema
        self._registered_tools: Dict[str, CanonicalTool] = {}

    @abstractmethod
    def from_canonical(self, tools: List[CanonicalTool]) -> Any:
        """将标准工具转换为框架特定格式 / 将标准工具Converts为框架特定格式

        Args:
                    tools: 标准格式工具列表

                Returns:
                    框架特定的工具格式"""
        pass

    def function_tools(
        self,
        tools: List[CanonicalTool],
        modify_func: Optional[Callable[..., Any]] = None,
    ):
        """将标准格式转换为 Google ADK 工具 / 将标准格式Converts为 Google ADK 工具

        Google ADK 通过函数的类型注解推断参数，需要动态创建带注解的函数。"""
        result = []

        for tool in tools:
            # 记录工具定义
            self._registered_tools[tool.name] = tool

            # 从 parameters schema 构建函数签名
            parameters_schema = tool.parameters or {
                "type": "object",
                "properties": {},
            }
            properties = parameters_schema.get("properties", {})
            required = set(parameters_schema.get("required", []))

            # 构建函数参数
            params = []
            annotations = {}

            for param_name, param_schema in properties.items():
                # 映射 JSON Schema 类型到 Python 类型
                param_type_str = param_schema.get("type", "string")
                type_mapping = {
                    "string": str,
                    "integer": int,
                    "number": float,
                    "boolean": bool,
                    "array": list,
                    "object": dict,
                }
                param_type = type_mapping.get(param_type_str, str)

                # 设置默认值
                default = (
                    inspect.Parameter.empty if param_name in required else None
                )

                params.append(
                    inspect.Parameter(
                        param_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=default,
                        annotation=param_type,
                    )
                )
                annotations[param_name] = param_type

            # 创建带正确签名的函数
            def make_tool_function(
                canonical_tool: CanonicalTool,
                sig: inspect.Signature,
                annots: dict,
            ):
                def tool_func(**kwargs):
                    if canonical_tool.func is None:
                        raise NotImplementedError(
                            f"Tool function for '{canonical_tool.name}' "
                            "is not implemented."
                        )

                    if modify_func:
                        return modify_func(canonical_tool, **kwargs)

                    return canonical_tool.func(**kwargs)

                # 设置函数元数据
                tool_func.__name__ = canonical_tool.name
                tool_func.__doc__ = canonical_tool.description
                tool_func.__annotations__ = annots
                object.__setattr__(tool_func, "__signature__", sig)

                return tool_func

            # 创建签名
            signature = inspect.Signature(params)
            wrapped_func = make_tool_function(tool, signature, annotations)

            result.append(wrapped_func)

        return result


class ModelAdapter(ABC):
    """模型适配器接口 / Utils Model Adapter

    用于包装框架模型，使其能够与 CommonModel 协同工作。"""

    @abstractmethod
    def wrap_model(self, common_model: CommonModel) -> Any:
        """包装 CommonModel 为框架特定的模型格式 / 包装 CommonModel 为framework特定的模型格式

        Args:
                    common_model: CommonModel 实例

                Returns:
                    框架特定的模型对象"""
        pass
