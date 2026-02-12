

## 功能描述
为不同 Agent 开发框架提供会话状态持久化能力，持久化数据库选用阿里云 TableStore （OTS，宽表模型）。通过一套表结构兼容多种框架。

方案：统一存储 + 中心 Service + 薄 Adapter

                                   ┌─────────────────┐
  ADK Agent ──→ ADK Adapter ──→    │                 │    ┌──────────┐
                                   │  SessionStore   │───→│          │
  LangChain ──→ LC Adapter ───→   │  (Central Svc)  │───→│   OTS    │
                                   │                 │───→│  Tables  │
  LangGraph ──→ LG Adapter ──→    │                 │    │          │
                                   └─────────────────┘    └──────────┘

  Central Service 职责:
    ① 理解 OTS 表结构
    ② 实现 OTS 读写操作
    ③ 实现业务逻辑（级联删除、状态合并…）
    ④ 暴露框架无关的统一接口

  Adapter 职责:
    仅做 ④ 框架数据模型转换


## 访问模式分析

访问模式                              频率    操作类型
─────────────────────────────────────────────────────
1. 创建 session                       中     PutRow
2. 获取 session (app, user, sid)      高     GetRow（点读）
3. 列出用户所有 session               中     GetRange（二级索引扫描）
4. 删除 session + 所有消息             低     BatchWrite + GetRange
5. 追加消息/事件                       高     PutRow（自增排序）
6. 获取 session 全部消息               高     GetRange
7. 获取最近 N 条消息                   高     GetRange（反向 + limit）
8. 按时间过滤消息                      中     GetRange + Filter / 多元索引
9. 读写 app/user 级状态               中     GetRow / UpdateRow
10. 全文搜索 summary / 标签过滤        中     多元索引 SearchQuery
11. 跨 user_id 查询（管理后台场景）     低     多元索引 BoolQuery

## 表设计
### 会话表
Conversation 表
PK:
    agent_id   (String, 分区键)
    user_id    (String)
    session_id (String)

Defined Columns（二级索引 / 多元索引引用的非 PK 列，建表时需声明）:
    updated_at  : Integer
    summary     : String
    labels      : String
    framework   : String
    extensions  : String

Attributes:
    created_at  : Integer   -- 纳秒时间戳
    updated_at  : Integer   -- 纳秒时间戳
    is_pinned   : Boolean   -- 是否置顶
    summary     : String    -- 会话摘要
    labels      : String    -- 会话标签（JSON 字符串）
    framework   : String    -- 'adk' / 'langchain' / …
    extensions  : String    -- JSON 框架扩展数据
    version     : Integer   -- 版本号, 用于乐观锁


二级索引（GLOBAL_INDEX）：
conversation_secondary_index:
PK:
    agent_id   (String, 分区键)
    user_id    (String)
    updated_at (Integer)   -- 纳秒时间戳，支持按更新时间排序
    session_id (String)

Attributes:
    summary    (String)  -- 会话摘要
    labels     (String)  -- 会话标签
    framework  (String)  -- 'adk' / 'langchain' / …
    extensions (String)  -- JSON 框架扩展数据

用途：list_sessions(agent_id, user_id) 热路径，低延迟（毫秒级）。


多元索引（Search Index）：
conversation_search_index:

字段          OTS FieldType  index  enable_sort_and_agg  说明
──────────────────────────────────────────────────────────────────
agent_id      KEYWORD        True   True                 精确匹配 + routing_field
user_id       KEYWORD        True   True                 精确匹配
session_id    KEYWORD        True   True                 精确匹配
updated_at    LONG           True   True                 范围查询 + 排序
created_at    LONG           True   True                 范围查询 + 排序
is_pinned     KEYWORD        True   True                 过滤置顶（"true"/"false"）
framework     KEYWORD        True   True                 按框架过滤
summary       TEXT           True   False                全文检索（SINGLEWORD 分词）
labels        KEYWORD        True   True                 精确匹配标签 JSON

高级配置:
  routing_fields = ["agent_id"]          -- 同一 agent 数据路由到同一分区
  index_sort = updated_at DESC           -- 预排序，最近更新优先

用途：全文搜索 summary、标签过滤、跨 user_id 查询、组合条件搜索。
延迟稍高（10-50ms），适合搜索/过滤场景，与二级索引并行保留。


### Event 表

Event 表
PK:
    agent_id   (String, 分区键)
    user_id    (String)
    session_id (String)
    seq_id     (Integer, AUTO_INCREMENT)  -- 事件序号，OTS 自增列

Attributes:
    type       : String    -- 事件类型
    content    : String    -- JSON 序列化的事件数据
    created_at : Integer   -- 纳秒时间戳
    updated_at : Integer   -- 纳秒时间戳
    version    : Integer   -- 版本号, 用于乐观锁
    raw_event  : String    -- 框架原生 Event 的完整 JSON 序列化（可选）
                              用于精确还原框架特定的 Event 对象（如 ADK Event）

说明：统一用 Event 抽象，Message 是 Event 的子集

### State 表
State 表
PK:
    agent_id   (String, 分区键)
    user_id    (String)
    session_id (String)

Attributes:
    state       : String    -- JSON 序列化的状态数据（未分片时使用）
    chunk_count : Integer   -- 分片数量，0 表示未分片
    state_0..N  : String    -- 分片列（当 JSON 超过 1.5M 字符时自动分片）
    created_at  : Integer   -- 纳秒时间戳
    updated_at  : Integer   -- 纳秒时间戳
    version     : Integer   -- 版本号, 用于乐观锁

说明：State 以 JSON 字符串存储。当 JSON 超过 1.5M 字符时，
自动拆分为 state_0, state_1, ... 多列存储（列分片），
读取时按 chunk_count 拼接还原。

### App_state 表
App_state 表
PK:
    agent_id (String, 分区键)

Attributes:
    state       : String    -- JSON 序列化的状态数据（未分片时使用）
    chunk_count : Integer   -- 分片数量，0 表示未分片
    state_0..N  : String    -- 分片列（当 JSON 超过 1.5M 字符时自动分片）
    created_at  : Integer   -- 纳秒时间戳
    updated_at  : Integer   -- 纳秒时间戳
    version     : Integer   -- 版本号, 用于乐观锁

说明：三级 State 是 ADK 的概念，其他框架按需使用

### User_state 表
User_state 表
PK:
    agent_id (String, 分区键)
    user_id  (String)

Attributes:
    state       : String    -- JSON 序列化的状态数据（未分片时使用）
    chunk_count : Integer   -- 分片数量，0 表示未分片
    state_0..N  : String    -- 分片列（当 JSON 超过 1.5M 字符时自动分片）
    created_at  : Integer   -- 纳秒时间戳
    updated_at  : Integer   -- 纳秒时间戳
    version     : Integer   -- 版本号, 用于乐观锁

说明：三级 State 是 ADK 的概念，其他框架按需使用

## 初始化策略

表和索引按用途分组创建，避免为未使用的框架创建不必要的表：

方法                     创建的资源                                 适用场景
─────────────────────────────────────────────────────────────────
init_core_tables()       Conversation + Event + 二级索引              所有框架
init_state_tables()      State + App_state + User_state             ADK 三级 State
init_search_index()      conversation_search_index (多元索引)         需要搜索/过滤
init_tables()            以上全部（向后兼容）                          快速开发
多元索引创建耗时较长（数秒级），建议与核心表创建分离，不阻塞核心流程。

## 分层架构

┌─────────────────────────────────────────────────────────┐
│ Layer 1: Framework Adapters（薄，只做模型转换）           │
│                                                         │
│  ┌─────────────┐ ┌───────────────┐ ┌──────────────┐    │
│  │ ADK Adapter  │ │ LC Adapter    │ │ LG Adapter   │    │
│  │ implements   │ │ implements    │ │ implements   │    │
│  │ BaseSession  │ │ BaseChatMsg   │ │ BaseCheck    │    │
│  │ Service      │ │ History       │ │ pointSaver   │    │
│  └──────┬───────┘ └──────┬────────┘ └──────┬───────┘    │
│         │                │                 │            │
│         ▼                ▼                 ▼            │
├─────────────────────────────────────────────────────────┤
│ Layer 2: SessionStore（厚，核心业务逻辑）                   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ class SessionStore:                             │    │
│  │   # 统一领域对象                                  │    │
│  │   @dataclass ConversationSession                │    │
│  │   @dataclass ConversationEvent                  │    │
│  │   @dataclass StateData                          │    │
│  │                                                 │    │
│  │   # 工厂方法                                      │    │
│  │   from_memory_collection(name)  # 推荐入口        │    │
│  │     → 从 MemoryCollection 获取 OTS 连接信息       │    │
│  │     → 自动构建 OTSClient + OTSBackend             │    │
│  │                                                 │    │
│  │   # 初始化                                       │    │
│  │   init_tables()            # 全量建表（向后兼容）  │    │
│  │   init_core_tables()       # 核心表 + 二级索引    │    │
│  │   init_state_tables()      # 三级 State 表       │    │
│  │   init_search_index()      # 多元索引（按需）     │    │
│  │                                                 │    │
│  │   # Session CRUD                                │    │
│  │   create_session(...)  → ConversationSession     │    │
│  │   get_session(...)     → ConversationSession?    │    │
│  │   list_sessions(...)   → [ConversationSession]   │    │
│  │   list_all_sessions(...)→ [ConversationSession]  │    │
│  │   search_sessions(...)  → ([Session], total)     │    │
│  │   update_session(...)  # 乐观锁                  │    │
│  │   delete_session(...)  # 级联删除 Event→State→Row │    │
│  │                                                 │    │
│  │   # Event CRUD                                  │    │
│  │   append_event(...)    → ConversationEvent       │    │
│  │   get_events(...)      → [ConversationEvent]     │    │
│  │   get_recent_events(...)→ [ConversationEvent]    │    │
│  │   delete_events(...)   → int                    │    │
│  │                                                 │    │
│  │   # 三级 State 管理                               │    │
│  │   get_session_state / update_session_state       │    │
│  │   get_app_state     / update_app_state           │    │
│  │   get_user_state    / update_user_state          │    │
│  │   get_merged_state(...)→ dict  # 三级浅合并       │    │
│  └──────────────┬──────────────────────────────────┘    │
│                 │                                       │
├─────────────────┼───────────────────────────────────────┤
│ Layer 3: Storage Backend                                │
│                 ▼                                       │
│  ┌──────────────────────┐                               │
│  │ OTSBackend           │  ← 当前实现                    │
│  │ (OTS SDK 调用)       │                               │
│  └──────────────────────┘                               │
│                                                         │
│  OTS 连接信息来源：                                       │
│    SessionStore.from_memory_collection(name)             │
│      → MemoryCollection.get_by_name(name)               │
│      → vector_store_config.endpoint / instance_name     │
│      → Config (AK/SK)                                   │
│      → OTSClient → OTSBackend                           │
│                                                         │
│  也可手动传入 OTSClient 构建 OTSBackend（向后兼容）        │
└─────────────────────────────────────────────────────────┘