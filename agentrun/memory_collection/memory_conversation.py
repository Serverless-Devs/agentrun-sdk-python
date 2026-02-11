"""AgentRun Memory Conversation / AgentRun 记忆对话

提供与 TableStore Memory 的集成能力，自动存储用户和 Agent 的对话历史。

"""

import json
import os
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
)
import uuid

import tablestore

from agentrun.utils.config import Config
from agentrun.utils.log import logger

if TYPE_CHECKING:
    from agentrun.server.model import (
        AgentEvent,
        AgentRequest,
        EventType,
        MessageRole,
    )


class MemoryConversation:
    """Memory Conversation / 记忆对话

    自动将用户和 Agent 的对话存储到 TableStore Memory 中。

    Attributes:
        memory_collection_name: MemoryCollection 名称
        config: AgentRun 配置
        user_id_extractor: 从请求中提取 user_id 的函数
        session_id_extractor: 从请求中提取 session_id 的函数
        agent_id_extractor: 从请求中提取 agent_id 的函数
    """

    def __init__(
        self,
        memory_collection_name: str,
        config: Optional[Config] = None,
        user_id_extractor: Optional[Callable[[Any], str]] = None,
        session_id_extractor: Optional[Callable[[Any], str]] = None,
        agent_id_extractor: Optional[Callable[[Any], str]] = None,
    ):
        """初始化 Memory Conversation

        Args:
            memory_collection_name: MemoryCollection 名称
            config: AgentRun 配置（可选，默认从环境变量读取）
            user_id_extractor: 从请求中提取 user_id 的函数（可选）
            session_id_extractor: 从请求中提取 session_id 的函数（可选）
            agent_id_extractor: 从请求中提取 agent_id 的函数（可选）
        """
        self.memory_collection_name = memory_collection_name
        self.config = config or Config()
        self.user_id_extractor = (
            user_id_extractor or self._default_user_id_extractor
        )
        self.session_id_extractor = (
            session_id_extractor or self._default_session_id_extractor
        )
        self.agent_id_extractor = (
            agent_id_extractor or self._default_agent_id_extractor
        )

        # 延迟初始化
        self._memory_store = None
        self._ots_client = None

    @staticmethod
    def _default_user_id_extractor(req: Any) -> str:
        """默认的 user_id 提取器

        优先级：
        1. X-User-ID 请求头（支持多种格式）
        2. user_id 查询参数
        3. 默认值 "default_user"
        """
        if req.raw_request:
            # 从请求头获取
            user_id = (
                req.raw_request.headers.get("X-AgentRun-User-ID")
                or req.raw_request.headers.get("x-agentrun-user-id")
                or req.raw_request.headers.get("X-Agentrun-User-Id")
            )
            if user_id:
                return user_id

            # 从查询参数获取
            user_id = req.raw_request.query_params.get("user_id")
            if user_id:
                return user_id

        return "default_user"

    @staticmethod
    def _default_session_id_extractor(req: Any) -> str:
        """默认的 session_id 提取器

        优先级：
        1. X-Conversation-ID 请求头（支持多种格式）
        2. sessionId 查询参数
        3. 生成新的 UUID
        """
        if req.raw_request:
            # 从请求头获取（兼容多种格式）
            # 支持：X-AgentRun-Conversation-ID, x-agentrun-conversation-id, X-Agentrun-Conversation-Id
            session_id = (
                req.raw_request.headers.get("X-AgentRun-Conversation-ID")
                or req.raw_request.headers.get("x-agentrun-conversation-id")
                or req.raw_request.headers.get("X-Agentrun-Conversation-Id")
            )
            if session_id:
                return session_id

            # 从查询参数获取
            session_id = req.raw_request.query_params.get("sessionId")
            if session_id:
                return session_id

        # 生成新的 session_id
        return f"session_{uuid.uuid4().hex[:16]}"

    @staticmethod
    def _default_agent_id_extractor(req: Any) -> str:
        """默认的 agent_id 提取器

        优先级：
        1. X-Agent-ID 请求头（支持多种格式）
        2. 默认值 "default_agent"
        """
        if req.raw_request:
            # 从请求头获取（兼容多种格式）
            # 支持：X-AgentRun-Agent-ID, x-agentrun-agent-id, X-Agentrun-Agent-Id
            agent_id = (
                req.raw_request.headers.get("X-AgentRun-Agent-ID")
                or req.raw_request.headers.get("x-agentrun-agent-id")
                or req.raw_request.headers.get("X-Agentrun-Agent-Id")
            )
            if agent_id:
                return agent_id

        return "default_agent"

    async def _get_memory_store(self):
        """获取或创建 AsyncMemoryStore 实例"""
        if self._memory_store is not None:
            return self._memory_store

        try:
            # 导入依赖
            from tablestore_for_agent_memory.base.base_memory_store import (
                Message,
                Session,
            )
            from tablestore_for_agent_memory.base.common import (
                microseconds_timestamp,
            )
            from tablestore_for_agent_memory.memory.async_memory_store import (
                AsyncMemoryStore,
            )
        except ImportError as e:
            raise ImportError(
                "tablestore-for-agent-memory package is required. "
                "Install it with: pip install tablestore-for-agent-memory"
            ) from e

        # 从 MemoryCollection 获取配置
        ots_config = await self._get_ots_config_from_memory_collection()

        # 创建 AsyncOTSClient
        # 支持使用 STS 临时凭证访问 TableStore
        client_kwargs = {
            "end_point": ots_config["endpoint"],
            "access_key_id": ots_config["access_key_id"],
            "access_key_secret": ots_config["access_key_secret"],
            "instance_name": ots_config["instance_name"],
        }

        # 如果提供了 security_token，则添加到参数中（支持 STS 临时凭证）
        if ots_config.get("security_token"):
            client_kwargs["sts_token"] = ots_config["security_token"]

        self._ots_client = tablestore.AsyncOTSClient(**client_kwargs)

        # 配置会话表的二级索引元数据字段
        # agent_id 字段用于标识会话所属的 Agent
        from tablestore_for_agent_memory.base.common import MetaType

        session_secondary_index_meta = {
            "agent_id": (
                MetaType.STRING
            ),  # Agent 标识符，用于区分不同的 AI Agent
        }

        # 配置会话表的搜索索引结构
        # agent_id: Agent 标识符，支持精确匹配查询
        session_search_index_schema = [
            tablestore.FieldSchema("agent_id", tablestore.FieldType.KEYWORD),
        ]

        # 配置消息表的搜索索引结构（消息表不需要额外索引）
        message_search_index_schema = []

        # 创建 AsyncMemoryStore
        self._memory_store = AsyncMemoryStore(
            tablestore_client=self._ots_client,
            session_secondary_index_meta=session_secondary_index_meta,
            session_search_index_schema=session_search_index_schema,
            message_search_index_schema=message_search_index_schema,
        )

        # 初始化表和索引（如果表已存在会忽略错误）
        try:
            logger.info(
                "Initializing tables and indexes for collection:"
                f" {self.memory_collection_name}"
            )
            await self._memory_store.init_table()
            await self._memory_store.init_search_index()
            logger.info(f"Tables and indexes initialized successfully")
        except Exception as e:
            # 如果表已存在，会抛出异常，这是正常的
            logger.info(
                "Tables and indexes already exist or initialization"
                f" skipped: {e}"
            )

        logger.info(
            "Memory Store initialized for collection:"
            f" {self.memory_collection_name}"
        )

        return self._memory_store

    async def _get_ots_config_from_memory_collection(self) -> Dict[str, Any]:
        """从 MemoryCollection 获取 OTS 配置信息

        Returns:
            Dict[str, Any]: OTS 配置字典，包含：
                - endpoint: OTS endpoint
                - access_key_id: 访问密钥 ID
                - access_key_secret: 访问密钥 Secret
                - security_token: STS 安全令牌（可选，用于临时凭证）
                - instance_name: OTS 实例名称
        """
        from agentrun.memory_collection import MemoryCollection

        # 获取 MemoryCollection
        memory_collection = await MemoryCollection.get_by_name_async(
            self.memory_collection_name, config=self.config
        )

        if not memory_collection.vector_store_config:
            raise ValueError(
                f"MemoryCollection {self.memory_collection_name} does not have "
                "vector_store_config"
            )

        vector_store_config = memory_collection.vector_store_config
        provider = vector_store_config.provider or ""

        # 只支持 aliyun_tablestore provider
        if provider != "aliyun_tablestore":
            raise ValueError(
                f"Only aliyun_tablestore provider is supported, got: {provider}"
            )

        if not vector_store_config.config:
            raise ValueError(
                f"MemoryCollection {self.memory_collection_name} does not have "
                "vector_store_config.config"
            )

        vs_config = vector_store_config.config

        # 获取 endpoint 并根据运行环境决定是否转换为公网地址
        endpoint = vs_config.endpoint or ""
        is_running_on_fc = os.getenv("FC_REGION") is not None
        if not is_running_on_fc and ".vpc.tablestore.aliyuncs.com" in endpoint:
            original_endpoint = endpoint
            endpoint = endpoint.replace(
                ".vpc.tablestore.aliyuncs.com", ".tablestore.aliyuncs.com"
            )
            logger.info(
                "Running on local, converted VPC endpoint to public endpoint:"
                f" {original_endpoint} -> {endpoint}"
            )

        # 构建 OTS 配置
        ots_config = {
            "endpoint": endpoint,
            "instance_name": vs_config.instance_name or "",
            "access_key_id": self.config.get_access_key_id(),
            "access_key_secret": self.config.get_access_key_secret(),
            "security_token": (
                self.config.get_security_token()
            ),  # 支持 STS 临时凭证
        }

        return ots_config

    async def wrap_invoke_agent(
        self,
        request: Any,
        agent_handler: Callable[[Any], AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:
        """包装 invoke_agent 函数，自动存储对话历史

        Args:
            request: AgentRequest 对象
            agent_handler: 原始的 agent 处理函数

        Yields:
            Any: Agent 返回的事件或字符串

        Example:
            >>> async def my_agent(req: AgentRequest):
            ...     yield "Hello, world!"
            >>>
            >>> async def invoke_agent(req: AgentRequest):
            ...     async for event in memory.wrap_invoke_agent(req, my_agent):
            ...         yield event
        """
        try:
            # 导入依赖
            from tablestore_for_agent_memory.base.base_memory_store import (
                Message,
                Session,
            )
            from tablestore_for_agent_memory.base.common import (
                microseconds_timestamp,
            )

            from agentrun.server.model import AgentEvent, EventType, MessageRole
        except ImportError as e:
            logger.warning(
                "tablestore-for-agent-memory not installed, skipping memory"
                " storage"
            )
            # 如果没有安装依赖，直接透传
            async for event in agent_handler(request):
                yield event
            return

        # 提取 user_id、session_id 和 agent_id
        user_id = self.user_id_extractor(request)
        session_id = self.session_id_extractor(request)
        agent_id = self.agent_id_extractor(request)

        logger.debug(
            f"Memory: user_id={user_id}, session_id={session_id},"
            f" agent_id={agent_id}"
        )

        # 获取 MemoryStore
        try:
            memory_store = await self._get_memory_store()
        except Exception as e:
            logger.error(
                f"Failed to initialize memory store: {e}", exc_info=True
            )
            # 初始化失败，直接透传
            async for event in agent_handler(request):
                yield event
            return

        # 创建或更新 Session
        current_time = microseconds_timestamp()
        session = Session(
            user_id=user_id,
            session_id=session_id,
            update_time=current_time,
            metadata={"agent_id": agent_id},
        )

        try:
            await memory_store.put_session(session)
        except Exception as e:
            logger.error(f"Failed to save session: {e}", exc_info=True)

        # 构建输入消息列表（包含所有历史消息）
        input_messages = []
        for msg in request.messages:
            input_messages.append({
                "role": (
                    msg.role.value
                    if hasattr(msg.role, "value")
                    else str(msg.role)
                ),
                "content": self._extract_message_content(msg.content),
            })

        # 收集 Agent 响应（包括文本和工具调用）
        agent_response_content = ""
        tool_calls: Dict[str, Dict[str, Any]] = (
            {}
        )  # tool_call_id -> tool_call_info
        tool_results: List[Dict[str, Any]] = []  # 工具执行结果列表

        try:
            # 流式处理 Agent 响应
            async for event in agent_handler(request):
                # 收集文本内容
                if isinstance(event, str):
                    agent_response_content += event
                elif isinstance(event, AgentEvent):
                    if event.event == EventType.TEXT and "delta" in event.data:
                        agent_response_content += event.data["delta"]

                    # 收集工具调用信息
                    elif event.event == EventType.TOOL_CALL:
                        # 完整的工具调用
                        tool_id = event.data.get("id", "")
                        if tool_id:
                            tool_calls[tool_id] = {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": event.data.get("name", ""),
                                    "arguments": event.data.get("args", ""),
                                },
                            }

                    elif event.event == EventType.TOOL_CALL_CHUNK:
                        # 工具调用片段（流式场景）
                        tool_id = event.data.get("id", "")
                        if tool_id:
                            if tool_id not in tool_calls:
                                tool_calls[tool_id] = {
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": event.data.get("name", ""),
                                        "arguments": "",
                                    },
                                }
                            # 累积参数片段
                            if "args_delta" in event.data:
                                tool_calls[tool_id]["function"][
                                    "arguments"
                                ] += event.data["args_delta"]

                    # 收集工具执行结果
                    elif event.event == EventType.TOOL_RESULT:
                        tool_id = event.data.get("id", "")
                        if tool_id:
                            tool_results.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": str(event.data.get("result", "")),
                            })

                # 透传事件
                yield event

            # 保存完整的对话轮次（输入 + 输出）
            # 只有当有文本内容或工具调用时才保存
            if agent_response_content or tool_calls or tool_results:
                try:
                    # 构建助手响应消息
                    assistant_message: Dict[str, Any] = {
                        "role": "assistant",
                    }

                    # 添加文本内容（如果有）
                    if agent_response_content:
                        assistant_message["content"] = agent_response_content
                    else:
                        # OpenAI 格式要求：如果有 tool_calls，content 可以为 null
                        assistant_message["content"] = None

                    # 添加工具调用（如果有）
                    if tool_calls:
                        assistant_message["tool_calls"] = list(
                            tool_calls.values()
                        )

                    # 构建完整的消息列表
                    output_messages = input_messages + [assistant_message]

                    # 添加工具执行结果（如果有）
                    if tool_results:
                        output_messages.extend(tool_results)

                    # 将完整的对话历史存储为一条消息
                    # content 字段存储 JSON 格式的消息列表
                    conversation_message = Message(
                        session_id=session_id,
                        message_id=f"msg_{uuid.uuid4().hex[:16]}",
                        content=json.dumps(output_messages, ensure_ascii=False),
                    )
                    await memory_store.put_message(conversation_message)

                    # 更新 Session 时间
                    session.update_time = microseconds_timestamp()
                    await memory_store.update_session(session)

                    logger.debug(
                        f"Saved conversation: {len(output_messages)} messages,"
                        f" text length: {len(agent_response_content)} chars,"
                        f" tool_calls: {len(tool_calls)}, tool_results:"
                        f" {len(tool_results)}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to save conversation: {e}", exc_info=True
                    )

        except Exception as e:
            logger.error(f"Error in agent handler: {e}", exc_info=True)
            raise

    @staticmethod
    def _extract_message_content(content: Any) -> str:
        """提取消息内容为字符串

        Args:
            content: 消息内容（可能是字符串或多模态内容列表）

        Returns:
            str: 提取的文本内容
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # 多模态内容，提取文本部分
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return " ".join(text_parts)
        else:
            return str(content) if content else ""

    async def close(self):
        """关闭 OTS 客户端连接"""
        if self._ots_client:
            await self._ots_client.close()
            logger.info("Memory Store connection closed")
