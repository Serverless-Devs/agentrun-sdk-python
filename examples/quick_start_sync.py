"""AgentRun Server 快速开始示例 - 同步版本

本示例展示了如何使用 AgentRunServer 创建一个支持 OpenAI 和 AG-UI 协议的 Agent 服务。

主要特性:
- 支持 OpenAI Chat Completions 协议 (POST /openai/v1/chat/completions)
- 支持 AG-UI 协议 (POST /agui/v1/run)
- 使用生命周期钩子发送工具调用事件
- 真正的流式输出（每行立即发送到客户端）
- 内置获取时间工具

使用方法:
1. 运行: python examples/quick_start_sync.py
2. 测试 OpenAI 协议:
   curl 127.0.0.1:9000/openai/v1/chat/completions -XPOST \\
       -H "content-type: application/json" \\
       -d '{"messages": [{"role": "user", "content": "现在几点了?"}], "stream": true}' -N

3. 测试 AG-UI 协议（可以看到工具调用事件）:
   curl 127.0.0.1:9000/agui/v1/run -XPOST \\
       -H "content-type: application/json" \\
       -d '{"messages": [{"role": "user", "content": "现在几点了?"}]}' -N
"""

import time
from typing import Any, Callable, Dict, Iterator, List, Optional
import uuid

from agentrun.server import AgentLifecycleHooks, AgentRequest, AgentRunServer
from agentrun.utils.log import logger

# =============================================================================
# 工具定义
# =============================================================================


def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间

    Args:
        timezone: 时区，默认为 Asia/Shanghai

    Returns:
        当前时间的字符串表示
    """
    from datetime import datetime
    import time

    time.sleep(5)
    now = datetime.now()

    return now.strftime("%Y-%m-%d %H:%M:%S")


# 工具注册表
TOOLS: Dict[str, Callable] = {
    "get_current_time": get_current_time,
}


# =============================================================================
# 简单的 Agent 实现（带工具调用）
# =============================================================================


class SimpleAgent:
    """简单的 Agent 实现，支持工具调用和生命周期钩子"""

    def __init__(self, tools: Dict[str, Callable]):
        self.tools = tools

    def run(
        self,
        user_message: str,
        hooks: AgentLifecycleHooks,
    ) -> Iterator:
        """运行 Agent

        Args:
            user_message: 用户消息
            hooks: 生命周期钩子

        Yields:
            响应内容或事件
        """
        # 简单的意图识别：检查是否需要调用工具
        needs_time = any(
            keyword in user_message
            for keyword in ["时间", "几点", "日期", "time", "date", "clock"]
        )

        if needs_time:
            # 需要调用获取时间工具
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_name = "get_current_time"
            tool_args = {"timezone": "Asia/Shanghai"}

            # 1. 发送工具调用开始事件
            yield hooks.on_tool_call_start(id=tool_call_id, name=tool_name)

            # 2. 发送工具调用参数事件
            yield hooks.on_tool_call_args(id=tool_call_id, args=tool_args)

            # 3. 执行工具（模拟一点延迟）
            time.sleep(0.1)
            try:
                tool_func = self.tools.get(tool_name)
                if tool_func:
                    result = tool_func(**tool_args)
                else:
                    result = f"工具 {tool_name} 不存在"
            except Exception as e:
                result = f"工具执行错误: {str(e)}"

            # 4. 发送工具调用结果事件
            yield hooks.on_tool_call_result(id=tool_call_id, result=result)

            # 5. 发送工具调用结束事件
            yield hooks.on_tool_call_end(id=tool_call_id)

            # 6. 生成最终回复
            response = f"现在的时间是: {result}"
        else:
            # 简单回复
            response = f"你好！你说的是: {user_message}"

        # 流式输出响应（逐字输出，每个字之间有小延迟确保流式效果）
        for char in response:
            time.sleep(0.02)  # 小延迟确保流式效果可见
            yield char


# 创建 Agent 实例
agent = SimpleAgent(tools=TOOLS)


# =============================================================================
# Agent 调用处理函数
# =============================================================================


def invoke_agent(request: AgentRequest) -> Iterator:
    """Agent 调用处理函数（同步版本）

    Args:
        request: AgentRequest 对象

    Yields:
        流式输出的内容字符串或事件
    """
    hooks = request.hooks

    # 获取用户消息
    user_message = ""
    for msg in request.messages:
        if msg.role.value == "user":
            user_message = msg.content or ""

    try:
        # 发送步骤开始事件
        yield hooks.on_step_start("agent_processing")

        # 运行 Agent
        for chunk in agent.run(user_message, hooks):
            yield chunk

        # 发送步骤结束事件
        yield hooks.on_step_finish("agent_processing")

    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error("调用出错: %s", e)

        # 发送错误事件
        yield hooks.on_run_error(str(e), "AGENT_ERROR")

        raise e


# =============================================================================
# 启动服务器
# =============================================================================

if __name__ == "__main__":
    print("启动 AgentRun Server (同步版本)...")
    print("支持的端点:")
    print("  - POST /openai/v1/chat/completions (OpenAI 协议)")
    print("  - POST /agui/v1/run (AG-UI 协议，可看到工具调用事件)")
    print()
    server = AgentRunServer(invoke_agent=invoke_agent)
    server.start(port=9000)


# =============================================================================
# 测试命令
# =============================================================================
"""
# 测试 OpenAI 协议（流式）- 触发工具调用
# 注意：OpenAI 协议不会显示工具调用事件，只显示最终文本
curl 127.0.0.1:9000/openai/v1/chat/completions -XPOST \
    -H "content-type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "现在几点了?"}], 
        "stream": true
    }' -N

# 测试 AG-UI 协议 - 触发工具调用
# AG-UI 协议会显示完整的工具调用事件流:
# - STEP_STARTED
# - TOOL_CALL_START
# - TOOL_CALL_ARGS
# - TOOL_CALL_RESULT
# - TOOL_CALL_END
# - TEXT_MESSAGE_*
# - STEP_FINISHED
curl 127.0.0.1:9000/agui/v1/run -XPOST \
    -H "content-type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "现在几点了?"}]
    }' -N

# 测试简单对话（不触发工具）
curl 127.0.0.1:9000/agui/v1/run -XPOST \
    -H "content-type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "你好"}]
    }' -N

# 测试健康检查
curl 127.0.0.1:9000/agui/v1/health
curl 127.0.0.1:9000/openai/v1/models
"""
