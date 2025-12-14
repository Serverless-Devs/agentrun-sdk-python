import asyncio

from agentrun.server.model import AgentRequest, MessageRole
from agentrun.server.server import AgentRunServer


async def test_server():
    """测试服务器基本功能"""

    def invoke_agent(request: AgentRequest):
        # 检查请求消息，返回预期的响应
        user_message = next(
            (
                msg.content
                for msg in request.messages
                if msg.role == MessageRole.USER
            ),
            "Hello",
        )

        return f"You said: {user_message}"

    # 创建服务器实例
    server = AgentRunServer(invoke_agent=invoke_agent)

    # 创建一个用于测试的 FastAPI 应用
    app = server.as_fastapi_app()

    # 使用 TestClient 进行测试（模拟请求而不实际启动服务器）
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # 发送请求
    response = client.post(
        "/openai/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "AgentRun"}],
            "model": "test-model",
        },
    )

    # 检查响应状态
    assert response.status_code == 200

    # 检查响应内容
    response_data = response.json()

    # 替换可变的部分
    assert response_data == {
        "id": "chatcmpl-124525ca742f",
        "object": "chat.completion",
        "created": 1765525651,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "You said: AgentRun"},
            "finish_reason": "stop",
        }],
    }


async def test_server_streaming():
    """测试服务器流式响应功能"""

    async def streaming_invoke_agent(request: AgentRequest):
        yield "Hello, "
        await asyncio.sleep(0.01)  # 短暂延迟
        yield "this is "
        await asyncio.sleep(0.01)
        yield "a test."

    # 创建服务器实例
    server = AgentRunServer(invoke_agent=streaming_invoke_agent)

    # 创建一个用于测试的 FastAPI 应用
    app = server.as_fastapi_app()

    # 使用 TestClient 进行测试
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # 发送流式请求
    response = client.post(
        "/openai/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "AgentRun"}],
            "model": "test-model",
            "stream": True,
        },
    )

    # 检查响应状态
    assert response.status_code == 200
    lines = [line async for line in response.aiter_lines()]
    assert lines[0].startswith("data: {")
    assert "Hello, " in lines[0]
    assert "this is " in lines[1]
    assert "a test." in lines[2]
    assert lines[3] == "data: [DONE]"
