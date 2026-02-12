# Conversation Service

为不同 Agent 开发框架提供**统一的会话状态持久化**能力，底层存储选用阿里云 TableStore（OTS，宽表模型）。

## 架构概览

采用 **统一存储 + 中心 Service + 薄 Adapter** 的三层设计：

```
ADK Agent ──→ OTSSessionService ──┐
                                  │    ┌─────────────┐    ┌─────────┐
LangChain ──→ OTSChatMessageHistory ──→│ SessionStore │───→│  OTS    │
                                  │    │ (业务逻辑层)  │───→│ Tables  │
LangGraph ──→ (LG Adapter)  ─────┘    └─────────────┘    └─────────┘
                                              │
                                        OTSBackend
                                       (存储操作层)
```

- **SessionStore**：核心业务层，理解 OTS 表结构，提供 Session / Event / State 的 CRUD、级联删除、三级状态合并等统一接口。
- **OTSBackend**：存储操作层，封装 TableStore SDK 的底层调用。
- **Adapter**：薄适配层，仅负责框架数据模型转换。

## 快速开始

### 前置条件

- 阿里云账号，配置 AK/SK 环境变量
- AgentRun 平台上已创建 MemoryCollection（包含 OTS 实例配置）

### 安装

```bash
pip install agentrun
```

### 初始化

**方式一（推荐）：通过 MemoryCollection 自动获取 OTS 连接信息**

```python
from agentrun.conversation_service import SessionStore

# 环境变量：AGENTRUN_ACCESS_KEY_ID / AGENTRUN_ACCESS_KEY_SECRET
store = SessionStore.from_memory_collection("your-memory-collection-name")

# 首次使用时创建表
store.init_tables()
```

`from_memory_collection()` 内部自动完成：
1. 调用 AgentRun API 获取 MemoryCollection 配置
2. 从中提取 OTS 的 endpoint 和 instance_name
3. 从 `Config` 读取 AK/SK 凭证
4. 构建 OTSClient 和 OTSBackend

**方式二：手动传入 OTSClient**

```python
import tablestore
from agentrun.conversation_service import SessionStore, OTSBackend

ots_client = tablestore.OTSClient(
    endpoint, access_key_id, access_key_secret, instance_name,
    retry_policy=tablestore.WriteRetryPolicy(),
)
backend = OTSBackend(ots_client)
store = SessionStore(backend)
store.init_tables()
```

### 表初始化策略

表和索引按用途分组创建，避免创建不必要的表：

| 方法 | 创建的资源 | 适用场景 |
|------|-----------|---------|
| `init_core_tables()` | Conversation + Event + 二级索引 | 所有框架 |
| `init_state_tables()` | State + App_state + User_state | ADK 三级 State |
| `init_search_index()` | 多元索引（conversation_search_index） | 需要搜索/过滤 |
| `init_tables()` | 以上全部 | 快速开发 |

> 多元索引创建耗时较长（数秒级），建议与核心表创建分离，不阻塞核心流程。

## 使用示例

### Google ADK 集成

```python
import asyncio
from agentrun.conversation_service import SessionStore
from agentrun.conversation_service.adapters import OTSSessionService
from google.adk.agents import Agent
from google.adk.runners import Runner

# 初始化
store = SessionStore.from_memory_collection("my-collection")
store.init_tables()
session_service = OTSSessionService(session_store=store)

# 创建 Agent + Runner
agent = Agent(name="assistant", model=my_model, instruction="...")
runner = Runner(agent=agent, app_name="my_app", session_service=session_service)

# 对话自动持久化到 OTS
async def chat():
    session = await session_service.create_session(
        app_name="my_app", user_id="user_1"
    )
    async for event in runner.run_async(
        user_id="user_1", session_id=session.id, new_message=content
    ):
        ...

asyncio.run(chat())
```

### LangChain 集成

```python
from agentrun.conversation_service import SessionStore
from agentrun.conversation_service.adapters import OTSChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# 初始化
store = SessionStore.from_memory_collection("my-collection")
store.init_core_tables()

# 创建消息历史（自动关联 Session）
history = OTSChatMessageHistory(
    session_store=store,
    agent_id="my_agent",
    user_id="user_1",
    session_id="session_1",
)

# 添加消息（自动持久化到 OTS）
history.add_message(HumanMessage(content="你好"))
history.add_message(AIMessage(content="你好！有什么可以帮你的？"))

# 读取历史消息
for msg in history.messages:
    print(f"{msg.type}: {msg.content}")
```

### 直接使用 SessionStore

```python
from agentrun.conversation_service import SessionStore

store = SessionStore.from_memory_collection("my-collection")
store.init_tables()

# Session CRUD
session = store.create_session("agent_1", "user_1", "sess_1", summary="测试会话")
sessions = store.list_sessions("agent_1", "user_1")

# Event CRUD
event = store.append_event("agent_1", "user_1", "sess_1", "message", {"text": "hello"})
events = store.get_events("agent_1", "user_1", "sess_1")
recent = store.get_recent_events("agent_1", "user_1", "sess_1", n=10)

# 三级 State 管理（ADK 概念）
store.update_app_state("agent_1", {"model": "qwen-max"})
store.update_user_state("agent_1", "user_1", {"language": "zh-CN"})
store.update_session_state("agent_1", "user_1", "sess_1", {"topic": "weather"})
merged = store.get_merged_state("agent_1", "user_1", "sess_1")
# merged = app_state <- user_state <- session_state（浅合并）

# 多元索引搜索
results, total = store.search_sessions(
    "agent_1",
    summary_keyword="天气",
    updated_after=1700000000000000,
    limit=20,
)

# 级联删除（Event → State → Session 行）
store.delete_session("agent_1", "user_1", "sess_1")
```

## API 参考

### SessionStore

核心业务层，所有方法同时提供同步和异步（`_async` 后缀）版本。

**工厂方法**

| 方法 | 说明 |
|------|------|
| `from_memory_collection(name, *, config, table_prefix)` | 通过 MemoryCollection 名称创建实例 |

**初始化**

| 方法 | 说明 |
|------|------|
| `init_tables()` | 创建所有表和索引 |
| `init_core_tables()` | 创建核心表 + 二级索引 |
| `init_state_tables()` | 创建三张 State 表 |
| `init_search_index()` | 创建多元索引 |

**Session 管理**

| 方法 | 说明 |
|------|------|
| `create_session(agent_id, user_id, session_id, ...)` | 创建新会话 |
| `get_session(agent_id, user_id, session_id)` | 获取单个会话 |
| `list_sessions(agent_id, user_id, limit)` | 列出用户会话（按 updated_at 倒序） |
| `list_all_sessions(agent_id, limit)` | 列出 agent 下所有会话 |
| `search_sessions(agent_id, *, user_id, summary_keyword, ...)` | 多元索引搜索会话 |
| `update_session(agent_id, user_id, session_id, *, version, ...)` | 更新会话属性（乐观锁） |
| `delete_session(agent_id, user_id, session_id)` | 级联删除会话 |

**Event 管理**

| 方法 | 说明 |
|------|------|
| `append_event(agent_id, user_id, session_id, event_type, content)` | 追加事件 |
| `get_events(agent_id, user_id, session_id)` | 获取全部事件（正序） |
| `get_recent_events(agent_id, user_id, session_id, n)` | 获取最近 N 条事件 |
| `delete_events(agent_id, user_id, session_id)` | 删除会话下所有事件 |

**State 管理**

| 方法 | 说明 |
|------|------|
| `get_session_state / update_session_state` | 会话级状态读写 |
| `get_app_state / update_app_state` | 应用级状态读写 |
| `get_user_state / update_user_state` | 用户级状态读写 |
| `get_merged_state(agent_id, user_id, session_id)` | 三级状态浅合并 |

### 框架适配器

| 适配器 | 框架 | 基类 |
|--------|------|------|
| `OTSSessionService` | Google ADK | `BaseSessionService` |
| `OTSChatMessageHistory` | LangChain | `BaseChatMessageHistory` |

### 领域模型

| 模型 | 说明 |
|------|------|
| `ConversationSession` | 会话对象（含 agent_id, user_id, session_id, summary, labels 等） |
| `ConversationEvent` | 事件对象（含 seq_id 自增序号、type、content、raw_event） |
| `StateData` | 状态数据对象（含 state 字典、version 乐观锁） |
| `StateScope` | 状态作用域枚举：APP / USER / SESSION |

## OTS 表结构

共五张表 + 一个二级索引 + 一个多元索引：

| 表名 | 主键 | 用途 |
|------|------|------|
| `conversation` | agent_id, user_id, session_id | 会话元信息 |
| `event` | agent_id, user_id, session_id, seq_id (自增) | 事件/消息流 |
| `state` | agent_id, user_id, session_id | 会话级状态 |
| `app_state` | agent_id | 应用级状态 |
| `user_state` | agent_id, user_id | 用户级状态 |
| `conversation_secondary_index` | agent_id, user_id, updated_at, session_id | 二级索引（list 热路径） |
| `conversation_search_index` | 多元索引 | 全文搜索 / 标签过滤 / 组合查询 |

> 表名支持通过 `table_prefix` 参数添加前缀，实现多租户隔离。

## 示例代码

| 文件 | 说明 |
|------|------|
| [`conversation_service_adk_agent.py`](../../examples/conversation_service_adk_agent.py) | ADK Agent 完整对话示例，自动持久化到 OTS |
| [`conversation_service_adk_example.py`](../../examples/conversation_service_adk_example.py) | ADK 数据读写验证（Session / Event / State） |
| [`conversation_service_adk_data.py`](../../examples/conversation_service_adk_data.py) | ADK 模拟数据填充 + 多元索引搜索验证 |
| [`conversation_service_langchain_example.py`](../../examples/conversation_service_langchain_example.py) | LangChain 消息历史读写验证 |
| [`conversation_service_langchain_data.py`](../../examples/conversation_service_langchain_data.py) | LangChain 模拟数据填充 |
| [`conversation_service_verify.py`](../../examples/conversation_service_verify.py) | 端到端 CRUD 验证脚本 |

## 环境变量

| 变量 | 说明 | 必填 |
|------|------|------|
| `AGENTRUN_ACCESS_KEY_ID` | 阿里云 Access Key ID | 是（使用 `from_memory_collection` 时） |
| `AGENTRUN_ACCESS_KEY_SECRET` | 阿里云 Access Key Secret | 是（使用 `from_memory_collection` 时） |
| `ALIBABA_CLOUD_ACCESS_KEY_ID` | 备选 AK 环境变量 | 否（AK 候选） |
| `ALIBABA_CLOUD_ACCESS_KEY_SECRET` | 备选 SK 环境变量 | 否（SK 候选） |
| `MEMORY_COLLECTION_NAME` | MemoryCollection 名称（示例脚本使用） | 否 |

## 设计文档

详细的表设计、访问模式分析和分层架构说明见 [conversation_design.md](./conversation_design.md)。
