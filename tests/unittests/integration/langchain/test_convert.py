from langchain.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)

from agentrun.integration.langgraph.agent_converter import AgentRunConverter


class TestAstreamEvents:
    """测试 agent.astream_events 调用方式"""

    async def test_convert_python_3_10(self):

        events = [
            {
                "event": "on_chain_start",
                "data": {
                    "input": {
                        "messages": [
                            {"role": "user", "content": "你好,你是谁?"}
                        ]
                    }
                },
                "name": "LangGraph",
                "tags": [],
                "run_id": "a3399113-dc02-4529-8653-b23b118cb15d",
                "metadata": {},
                "parent_ids": [],
            },
            {
                "event": "on_chain_start",
                "data": {
                    "input": {
                        "messages": [
                            HumanMessage(
                                content="你好,你是谁?",
                                additional_kwargs={},
                                response_metadata={},
                                id="4d7b93af-4fac-48c0-808c-548a30025bde",
                            )
                        ]
                    }
                },
                "name": "model",
                "tags": ["graph:step:1"],
                "run_id": "882152b1-54e6-4ba1-b36d-fc9a8e76932c",
                "metadata": {
                    "langgraph_step": 1,
                    "langgraph_node": "model",
                    "langgraph_triggers": ("branch:to:model",),
                    "langgraph_path": ("__pregel_pull", "model"),
                    "langgraph_checkpoint_ns": (
                        "model:38e1bfb3-41b1-f0f5-24ff-096b74ca48a9"
                    ),
                },
                "parent_ids": ["a3399113-dc02-4529-8653-b23b118cb15d"],
            },
            {
                "event": "on_chain_stream",
                "run_id": "882152b1-54e6-4ba1-b36d-fc9a8e76932c",
                "name": "model",
                "tags": ["graph:step:1"],
                "metadata": {
                    "langgraph_step": 1,
                    "langgraph_node": "model",
                    "langgraph_triggers": ("branch:to:model",),
                    "langgraph_path": ("__pregel_pull", "model"),
                    "langgraph_checkpoint_ns": (
                        "model:38e1bfb3-41b1-f0f5-24ff-096b74ca48a9"
                    ),
                },
                "data": {
                    "chunk": {
                        "messages": [
                            AIMessage(
                                content=(
                                    "你好！我是 AgentRun 的 AI 专家，"
                                    "可以通过沙箱运行代码和查询知识库文档来回答你的问题。"
                                    "有什么我可以帮你的吗？"
                                ),
                                additional_kwargs={},
                                response_metadata={
                                    "finish_reason": "stop",
                                    "model_name": "qwen3-max",
                                    "model_provider": "openai",
                                },
                                id="lc_run--e1d31286-1ca4-4232-bc13-f9da6d878db3",
                                usage_metadata={
                                    "input_tokens": 265,
                                    "output_tokens": 31,
                                    "total_tokens": 296,
                                    "input_token_details": {"cache_read": 0},
                                    "output_token_details": {},
                                },
                            )
                        ]
                    }
                },
                "parent_ids": ["a3399113-dc02-4529-8653-b23b118cb15d"],
            },
            {
                "event": "on_chain_end",
                "data": {
                    "output": {
                        "messages": [
                            AIMessage(
                                content=(
                                    "你好！我是 AgentRun 的 AI 专家，"
                                    "可以通过沙箱运行代码和查询知识库文档来回答你的问题。"
                                    "有什么我可以帮你的吗？"
                                ),
                                additional_kwargs={},
                                response_metadata={
                                    "finish_reason": "stop",
                                    "model_name": "qwen3-max",
                                    "model_provider": "openai",
                                },
                                id="lc_run--e1d31286-1ca4-4232-bc13-f9da6d878db3",
                                usage_metadata={
                                    "input_tokens": 265,
                                    "output_tokens": 31,
                                    "total_tokens": 296,
                                    "input_token_details": {"cache_read": 0},
                                    "output_token_details": {},
                                },
                            )
                        ]
                    },
                    "input": {
                        "messages": [
                            HumanMessage(
                                content="你好,你是谁?",
                                additional_kwargs={},
                                response_metadata={},
                                id="4d7b93af-4fac-48c0-808c-548a30025bde",
                            )
                        ]
                    },
                },
                "run_id": "882152b1-54e6-4ba1-b36d-fc9a8e76932c",
                "name": "model",
                "tags": ["graph:step:1"],
                "metadata": {
                    "langgraph_step": 1,
                    "langgraph_node": "model",
                    "langgraph_triggers": ("branch:to:model",),
                    "langgraph_path": ("__pregel_pull", "model"),
                    "langgraph_checkpoint_ns": (
                        "model:38e1bfb3-41b1-f0f5-24ff-096b74ca48a9"
                    ),
                },
                "parent_ids": ["a3399113-dc02-4529-8653-b23b118cb15d"],
            },
            {
                "event": "on_chain_stream",
                "run_id": "a3399113-dc02-4529-8653-b23b118cb15d",
                "name": "LangGraph",
                "tags": [],
                "metadata": {},
                "data": {
                    "chunk": {
                        "model": {
                            "messages": [
                                AIMessage(
                                    content=(
                                        "你好！我是 AgentRun 的 AI 专家，"
                                        "可以通过沙箱运行代码和查询知识库文档来回答你的问题。"
                                        "有什么我可以帮你的吗？"
                                    ),
                                    additional_kwargs={},
                                    response_metadata={
                                        "finish_reason": "stop",
                                        "model_name": "qwen3-max",
                                        "model_provider": "openai",
                                    },
                                    id="lc_run--e1d31286-1ca4-4232-bc13-f9da6d878db3",
                                    usage_metadata={
                                        "input_tokens": 265,
                                        "output_tokens": 31,
                                        "total_tokens": 296,
                                        "input_token_details": {
                                            "cache_read": 0
                                        },
                                        "output_token_details": {},
                                    },
                                )
                            ]
                        }
                    }
                },
                "parent_ids": [],
            },
            {
                "event": "on_chain_end",
                "data": {
                    "output": {
                        "messages": [
                            HumanMessage(
                                content="你好,你是谁?",
                                additional_kwargs={},
                                response_metadata={},
                                id="4d7b93af-4fac-48c0-808c-548a30025bde",
                            ),
                            AIMessage(
                                content=(
                                    "你好！我是 AgentRun 的 AI 专家，"
                                    "可以通过沙箱运行代码和查询知识库文档来回答你的问题。"
                                    "有什么我可以帮你的吗？"
                                ),
                                additional_kwargs={},
                                response_metadata={
                                    "finish_reason": "stop",
                                    "model_name": "qwen3-max",
                                    "model_provider": "openai",
                                },
                                id="lc_run--e1d31286-1ca4-4232-bc13-f9da6d878db3",
                                usage_metadata={
                                    "input_tokens": 265,
                                    "output_tokens": 31,
                                    "total_tokens": 296,
                                    "input_token_details": {"cache_read": 0},
                                    "output_token_details": {},
                                },
                            ),
                        ]
                    }
                },
                "run_id": "a3399113-dc02-4529-8653-b23b118cb15d",
                "name": "LangGraph",
                "tags": [],
                "metadata": {},
                "parent_ids": [],
            },
        ]

        converter = AgentRunConverter()
        full_text = ""
        for event in events:
            for item in converter.convert(event):
                assert type(item) is str
                full_text += item

        assert (
            full_text
            == "你好！我是 AgentRun 的 AI 专家，"
            "可以通过沙箱运行代码和查询知识库文档来回答你的问题。"
            "有什么我可以帮你的吗？"
        )

    async def test_convert_python_3_12(self):

        events = [
            {
                "event": "on_chain_start",
                "data": {
                    "input": {
                        "messages": [
                            HumanMessage(
                                content="你好,你是谁?",
                                additional_kwargs={},
                                response_metadata={},
                                id="2b223a08-a036-4a80-aa68-cc2c7671f272",
                            )
                        ]
                    }
                },
                "name": "model",
                "tags": ["graph:step:1"],
                "run_id": "9f179629-d14c-4d09-b91a-b4f95863e635",
                "metadata": {},
                "parent_ids": ["04ea2074-a058-42ff-94cd-c6b73630a037"],
            },
            {
                "event": "on_chat_model_start",
                "data": {
                    "input": {
                        "messages": [[
                            SystemMessage(
                                content=(
                                    "你是一个 AgentRun 的 AI 专家，"
                                    "可以通过沙箱运行代码和查询知识库文档来回答用户的问题。"
                                ),
                                additional_kwargs={},
                                response_metadata={},
                            ),
                            HumanMessage(
                                content="你好,你是谁?",
                                additional_kwargs={},
                                response_metadata={},
                                id="2b223a08-a036-4a80-aa68-cc2c7671f272",
                            ),
                        ]]
                    }
                },
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": AIMessageChunk(
                        content="",
                        additional_kwargs={},
                        response_metadata={"model_provider": "openai"},
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                    )
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": AIMessageChunk(
                        content="你好",
                        additional_kwargs={},
                        response_metadata={"model_provider": "openai"},
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                    )
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": AIMessageChunk(
                        content="！",
                        additional_kwargs={},
                        response_metadata={"model_provider": "openai"},
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                    )
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": AIMessageChunk(
                        content="我是 Agent",
                        additional_kwargs={},
                        response_metadata={"model_provider": "openai"},
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                    )
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": AIMessageChunk(
                        content="Run",
                        additional_kwargs={},
                        response_metadata={"model_provider": "openai"},
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                    )
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": AIMessageChunk(
                        content=" 的 AI 专家",
                        additional_kwargs={},
                        response_metadata={"model_provider": "openai"},
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                    )
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": AIMessageChunk(
                        content="，可以通过沙箱运行代码",
                        additional_kwargs={},
                        response_metadata={"model_provider": "openai"},
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                    )
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": AIMessageChunk(
                        content="和查询知识库文档",
                        additional_kwargs={},
                        response_metadata={"model_provider": "openai"},
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                    )
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": AIMessageChunk(
                        content="来回答你的问题。有什么",
                        additional_kwargs={},
                        response_metadata={"model_provider": "openai"},
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                    )
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": AIMessageChunk(
                        content="我可以帮你的吗？",
                        additional_kwargs={},
                        response_metadata={"model_provider": "openai"},
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                    )
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": AIMessageChunk(
                        content="",
                        additional_kwargs={},
                        response_metadata={
                            "finish_reason": "stop",
                            "model_name": "qwen3-max",
                            "model_provider": "openai",
                        },
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                        chunk_position="last",
                    )
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": AIMessageChunk(
                        content="",
                        additional_kwargs={},
                        response_metadata={},
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                        usage_metadata={
                            "input_tokens": 265,
                            "output_tokens": 31,
                            "total_tokens": 296,
                            "input_token_details": {"cache_read": 0},
                            "output_token_details": {},
                        },
                    )
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": AIMessageChunk(
                        content="",
                        additional_kwargs={},
                        response_metadata={},
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                        chunk_position="last",
                    )
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chat_model_end",
                "data": {
                    "output": AIMessage(
                        content=(
                            "你好！我是 AgentRun 的 AI 专家，"
                            "可以通过沙箱运行代码和查询知识库文档来回答你的问题。"
                            "有什么我可以帮你的吗？"
                        ),
                        additional_kwargs={},
                        response_metadata={
                            "finish_reason": "stop",
                            "model_name": "qwen3-max",
                            "model_provider": "openai",
                        },
                        id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                        usage_metadata={
                            "input_tokens": 265,
                            "output_tokens": 31,
                            "total_tokens": 296,
                            "input_token_details": {"cache_read": 0},
                            "output_token_details": {},
                        },
                    ),
                    "input": {
                        "messages": [[
                            SystemMessage(
                                content=(
                                    "你是一个 AgentRun 的 AI 专家，"
                                    "可以通过沙箱运行代码和查询知识库文档来回答用户的问题。"
                                ),
                                additional_kwargs={},
                                response_metadata={},
                            ),
                            HumanMessage(
                                content="你好,你是谁?",
                                additional_kwargs={},
                                response_metadata={},
                                id="2b223a08-a036-4a80-aa68-cc2c7671f272",
                            ),
                        ]]
                    },
                },
                "run_id": "c5a51e0d-12c9-448e-9565-644be6b6bcef",
                "name": "qwen3-max",
                "tags": ["seq:step:1"],
                "metadata": {},
                "parent_ids": [
                    "04ea2074-a058-42ff-94cd-c6b73630a037",
                    "9f179629-d14c-4d09-b91a-b4f95863e635",
                ],
            },
            {
                "event": "on_chain_stream",
                "run_id": "9f179629-d14c-4d09-b91a-b4f95863e635",
                "name": "model",
                "tags": ["graph:step:1"],
                "metadata": {},
                "data": {
                    "chunk": {
                        "messages": [
                            AIMessage(
                                content=(
                                    "你好！我是 AgentRun 的 AI 专家，"
                                    "可以通过沙箱运行代码和查询知识库文档来回答你的问题。"
                                    "有什么我可以帮你的吗？"
                                ),
                                additional_kwargs={},
                                response_metadata={
                                    "finish_reason": "stop",
                                    "model_name": "qwen3-max",
                                    "model_provider": "openai",
                                },
                                id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                                usage_metadata={
                                    "input_tokens": 265,
                                    "output_tokens": 31,
                                    "total_tokens": 296,
                                    "input_token_details": {"cache_read": 0},
                                    "output_token_details": {},
                                },
                            )
                        ]
                    }
                },
                "parent_ids": ["04ea2074-a058-42ff-94cd-c6b73630a037"],
            },
            {
                "event": "on_chain_end",
                "data": {
                    "output": {
                        "messages": [
                            AIMessage(
                                content=(
                                    "你好！我是 AgentRun 的 AI 专家，"
                                    "可以通过沙箱运行代码和查询知识库文档来回答你的问题。"
                                    "有什么我可以帮你的吗？"
                                ),
                                additional_kwargs={},
                                response_metadata={
                                    "finish_reason": "stop",
                                    "model_name": "qwen3-max",
                                    "model_provider": "openai",
                                },
                                id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                                usage_metadata={
                                    "input_tokens": 265,
                                    "output_tokens": 31,
                                    "total_tokens": 296,
                                    "input_token_details": {"cache_read": 0},
                                    "output_token_details": {},
                                },
                            )
                        ]
                    },
                    "input": {
                        "messages": [
                            HumanMessage(
                                content="你好,你是谁?",
                                additional_kwargs={},
                                response_metadata={},
                                id="2b223a08-a036-4a80-aa68-cc2c7671f272",
                            )
                        ]
                    },
                },
                "run_id": "9f179629-d14c-4d09-b91a-b4f95863e635",
                "name": "model",
                "tags": ["graph:step:1"],
                "metadata": {},
                "parent_ids": ["04ea2074-a058-42ff-94cd-c6b73630a037"],
            },
            {
                "event": "on_chain_stream",
                "run_id": "04ea2074-a058-42ff-94cd-c6b73630a037",
                "name": "LangGraph",
                "tags": [],
                "metadata": {},
                "data": {
                    "chunk": {
                        "model": {
                            "messages": [
                                AIMessage(
                                    content=(
                                        "你好！我是 AgentRun 的 AI 专家，"
                                        "可以通过沙箱运行代码和查询知识库文档来回答你的问题。"
                                        "有什么我可以帮你的吗？"
                                    ),
                                    additional_kwargs={},
                                    response_metadata={
                                        "finish_reason": "stop",
                                        "model_name": "qwen3-max",
                                        "model_provider": "openai",
                                    },
                                    id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                                    usage_metadata={
                                        "input_tokens": 265,
                                        "output_tokens": 31,
                                        "total_tokens": 296,
                                        "input_token_details": {
                                            "cache_read": 0
                                        },
                                        "output_token_details": {},
                                    },
                                )
                            ]
                        }
                    }
                },
                "parent_ids": [],
            },
            {
                "event": "on_chain_end",
                "data": {
                    "output": {
                        "messages": [
                            HumanMessage(
                                content="你好,你是谁?",
                                additional_kwargs={},
                                response_metadata={},
                                id="2b223a08-a036-4a80-aa68-cc2c7671f272",
                            ),
                            AIMessage(
                                content=(
                                    "你好！我是 AgentRun 的 AI 专家，"
                                    "可以通过沙箱运行代码和查询知识库文档来回答你的问题。"
                                    "有什么我可以帮你的吗？"
                                ),
                                additional_kwargs={},
                                response_metadata={
                                    "finish_reason": "stop",
                                    "model_name": "qwen3-max",
                                    "model_provider": "openai",
                                },
                                id="lc_run--c5a51e0d-12c9-448e-9565-644be6b6bcef",
                                usage_metadata={
                                    "input_tokens": 265,
                                    "output_tokens": 31,
                                    "total_tokens": 296,
                                    "input_token_details": {"cache_read": 0},
                                    "output_token_details": {},
                                },
                            ),
                        ]
                    }
                },
                "run_id": "04ea2074-a058-42ff-94cd-c6b73630a037",
                "name": "LangGraph",
                "tags": [],
                "metadata": {},
                "parent_ids": [],
            },
        ]

        converter = AgentRunConverter()
        full_text = ""
        for event in events:
            for item in converter.convert(event):
                assert type(item) is str
                full_text += item

        assert (
            full_text
            == "你好！我是 AgentRun 的 AI 专家，"
            "可以通过沙箱运行代码和查询知识库文档来回答你的问题。"
            "有什么我可以帮你的吗？"
        )
