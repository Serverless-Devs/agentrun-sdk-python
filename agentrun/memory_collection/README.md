# MemoryCollection 模块开发参考

## 目录

- [模块概述](#模块概述)
- [目录结构](#目录结构)
- [架构分层](#架构分层)
- [数据模型](#数据模型)
- [API 使用指南](#api-使用指南)
- [mem0 集成](#mem0-集成)
- [代码生成机制](#代码生成机制)
- [开发注意事项](#开发注意事项)

---

## 模块概述

`memory_collection` 模块提供 **记忆集合 (MemoryCollection)** 资源的完整生命周期管理，包括创建、获取、更新、删除和列表查询。同时支持将 MemoryCollection 转换为 `agentrun-mem0ai` 客户端，以便直接操作记忆数据。

所有 API 均提供 **同步** 和 **异步** 两种调用方式。

---

## 目录结构

```
memory_collection/
├── __init__.py                              # 模块入口，导出公开 API
├── model.py                                 # 数据模型定义（手动维护）
├── client.py                                # 客户端封装（自动生成，勿手动修改）
├── memory_collection.py                     # 高层资源 API（自动生成，勿手动修改）
├── __client_async_template.py               # client.py 的异步模板（手动维护）
├── __memory_collection_async_template.py     # memory_collection.py 的异步模板（手动维护）
└── api/
    ├── __init__.py
    └── control.py                           # 底层 SDK 交互层（自动生成，勿手动修改）
```

### 文件职责速查

| 文件 | 可编辑 | 职责 |
|------|--------|------|
| `model.py` | 是 | 定义所有数据模型（输入/输出/属性） |
| `__client_async_template.py` | 是 | Client 的异步模板，用于生成 `client.py` |
| `__memory_collection_async_template.py` | 是 | MemoryCollection 的异步模板，用于生成 `memory_collection.py` |
| `client.py` | **否** | 自动生成的客户端代码（包含同步+异步方法） |
| `memory_collection.py` | **否** | 自动生成的高层 API（包含同步+异步方法） |
| `api/control.py` | **否** | 自动生成的底层 SDK 调用封装 |

---

## 架构分层

模块采用三层架构设计：

```
┌────────────────────────────────────────────────────────────────┐
│                   用户代码                                      │
└────────────────────┬───────────────────────────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │    MemoryCollection (高层 API)       │  memory_collection.py
    │    - 类方法: create / get_by_name   │  继承自 ResourceBase
    │    - 实例方法: update / delete      │  + MutableProps
    │    - 列表: list_all                 │  + ImmutableProps
    │    - 转换: to_mem0_memory           │  + SystemProps
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │    MemoryCollectionClient (客户端)    │  client.py
    │    - create / delete               │  封装输入输出转换
    │    - update / get / list           │  错误处理转换
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │  MemoryCollectionControlAPI (底层)    │  api/control.py
    │  - 直接调用阿里云底层 SDK             │  继承自 ControlAPI
    │  - 处理 HTTP 异常映射               │
    └────────────────────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │  alibabacloud_agentrun20250910      │  底层 SDK
    └────────────────────────────────────┘
```

### 各层职责

1. **api/control.py（底层 API 层）**
   - 直接调用 `alibabacloud_agentrun20250910` 底层 SDK
   - 将 `ClientException` / `ServerException` 转换为 `ClientError` / `ServerError`
   - 提供日志输出（debug 级别）

2. **client.py（客户端层）**
   - 将 AgentRun SDK 的 Model（如 `MemoryCollectionCreateInput`）转换为底层 SDK 的 Input
   - 将底层 SDK 返回的对象转换为 `MemoryCollection` 高层对象
   - 将 `HTTPError` 转换为语义化的资源错误（如 `ResourceAlreadyExistError`、`ResourceNotExistError`）

3. **memory_collection.py（高层 API 层）**
   - 提供类方法（静态操作）：`create`、`get_by_name`、`delete_by_name`、`update_by_name`、`list_all`
   - 提供实例方法（基于当前对象）：`update`、`delete`、`get`、`refresh`
   - 提供 mem0 集成能力：`to_mem0_memory`、`to_mem0_memory_async`

---

## 数据模型

### 模型继承关系

```
BaseModel
├── MemoryCollectionMutableProps        # 可变属性（可通过 update 修改）
│   ├── description
│   ├── embedder_config
│   ├── execution_role_arn
│   ├── llm_config
│   ├── network_configuration
│   └── vector_store_config
│
├── MemoryCollectionImmutableProps       # 不可变属性（创建时指定，不可修改）
│   ├── memory_collection_name
│   └── type
│
├── MemoryCollectionSystemProps          # 系统属性（只读，由服务端生成）
│   ├── memory_collection_id
│   ├── created_at
│   └── last_updated_at
│
├── MemoryCollectionCreateInput         # 创建输入 = Immutable + Mutable
├── MemoryCollectionUpdateInput         # 更新输入 = Mutable
├── MemoryCollectionListInput           # 列表查询输入（含分页）
└── MemoryCollectionListOutput          # 列表查询输出（摘要信息）
```

### 配置子模型

| 模型 | 说明 | 关键字段 |
|------|------|----------|
| `EmbedderConfig` | 嵌入模型配置 | `config.model`, `model_service_name` |
| `LLMConfig` | LLM 配置 | `config.model`, `model_service_name` |
| `VectorStoreConfig` | 向量存储配置 | `provider`, `config` (TableStore), `mysql_config` (MySQL) |
| `VectorStoreConfigConfig` | TableStore 向量存储内部配置 | `endpoint`, `instance_name`, `collection_name`, `vector_dimension` |
| `VectorStoreConfigMysqlConfig` | MySQL 向量存储配置 | `host`, `port`, `db_name`, `user`, `credential_name`, `collection_name`, `vector_dimension` |
| `NetworkConfiguration` | 网络配置 | `vpc_id`, `vswitch_ids`, `security_group_id`, `network_mode` |

### 完整的 MemoryCollection 属性

`MemoryCollection` 类同时继承了三组属性和 `ResourceBase`：

```python
class MemoryCollection(
    MemoryCollectionMutableProps,       # 可变属性
    MemoryCollectionImmutableProps,     # 不可变属性
    MemoryCollectionSystemProps,        # 系统属性
    ResourceBase,                       # 资源基类（提供 from_inner_object 等）
):
```

---

## API 使用指南

### 1. 创建记忆集合

```python
from agentrun.memory_collection import (
    MemoryCollection,
    MemoryCollectionCreateInput,
    EmbedderConfig,
    EmbedderConfigConfig,
    LLMConfig,
    LLMConfigConfig,
    VectorStoreConfig,
    VectorStoreConfigConfig,
)

# 方式一：通过高层 API（推荐）
mc = MemoryCollection.create(
    MemoryCollectionCreateInput(
        memory_collection_name="my-collection",
        type="mem0",
        description="示例记忆集合",
        embedder_config=EmbedderConfig(
            model_service_name="my-embedder-service",
            config=EmbedderConfigConfig(model="text-embedding-v3"),
        ),
        llm_config=LLMConfig(
            model_service_name="my-llm-service",
            config=LLMConfigConfig(model="qwen-plus"),
        ),
        vector_store_config=VectorStoreConfig(
            provider="aliyun_tablestore",
            config=VectorStoreConfigConfig(
                endpoint="https://xxx.cn-hangzhou.ots.aliyuncs.com",
                instance_name="my-instance",
                collection_name="my-collection",
                vector_dimension=1024,
            ),
        ),
    )
)

# 方式二：通过 Client
from agentrun.memory_collection import MemoryCollectionClient

client = MemoryCollectionClient()
mc = client.create(input=MemoryCollectionCreateInput(...))
```

### 2. 获取记忆集合

```python
# 类方法（通过名称）
mc = MemoryCollection.get_by_name("my-collection")

# 实例方法（刷新当前对象）
mc.refresh()  # 等价于 mc.get()
```

### 3. 更新记忆集合

```python
from agentrun.memory_collection import MemoryCollectionUpdateInput

# 类方法
mc = MemoryCollection.update_by_name(
    "my-collection",
    MemoryCollectionUpdateInput(description="更新后的描述"),
)

# 实例方法（就地更新）
mc.update(MemoryCollectionUpdateInput(description="更新后的描述"))
# mc 对象的属性会被自动更新
```

### 4. 删除记忆集合

```python
# 类方法
MemoryCollection.delete_by_name("my-collection")

# 实例方法
mc.delete()
```

### 5. 列出记忆集合

```python
# 列出所有（自动分页）
collections = MemoryCollection.list_all()

# 带过滤条件
collections = MemoryCollection.list_all(
    memory_collection_name="my-collection",
    status="READY",
    type="mem0",
)

# 列表项转完整对象
for item in collections:
    full_mc = item.to_memory_collection()
```

### 6. 异步调用

所有方法都有对应的 `_async` 版本：

```python
import asyncio

async def main():
    mc = await MemoryCollection.create_async(input=...)
    mc = await MemoryCollection.get_by_name_async("my-collection")
    await mc.update_async(MemoryCollectionUpdateInput(description="新描述"))
    await mc.delete_async()
    collections = await MemoryCollection.list_all_async()

asyncio.run(main())
```

---

## mem0 集成

MemoryCollection 提供了与 `agentrun-mem0ai` 包的集成能力，可以将平台上的 MemoryCollection 配置直接转换为可操作的 mem0 Memory 客户端。

### 前置条件

```bash
pip install agentrun-mem0ai
```

### 使用方式

```python
# 同步
memory = MemoryCollection.to_mem0_memory("my-collection")
memory.add("用户喜欢吃苹果", user_id="user123")
results = memory.search("用户喜欢什么水果", user_id="user123")

# 异步
memory = await MemoryCollection.to_mem0_memory_async("my-collection")
await memory.add("用户喜欢吃苹果", user_id="user123")
```

### 内部工作流程

`to_mem0_memory` 方法内部会：

1. 通过 `get_by_name` 获取 MemoryCollection 的完整配置
2. 调用 `_build_mem0_config` 构建 mem0 兼容的配置字典，包括：
   - **vector_store 配置**：支持 `aliyun_tablestore` 和 `alibabacloud_mysql` 两种 provider
   - **llm 配置**：通过 `ModelService` 解析 base_url 和 api_key
   - **embedder 配置**：通过 `ModelService` 解析 base_url 和 api_key，同步 vector_dimension
3. 使用配置字典创建 `Memory` / `AsyncMemory` 实例

### 向量存储 Provider 支持

| Provider | 配置来源 | 地址转换 | 认证方式 |
|----------|---------|---------|---------|
| `aliyun_tablestore` | `VectorStoreConfigConfig` | VPC 内网自动转公网 | AK/SK + 可选 STS Token |
| `alibabacloud_mysql` | `VectorStoreConfigMysqlConfig` | 环境变量 `AGENTRUN_MYSQL_PUBLIC_HOST` | Credential 获取密码 |

### 相关环境变量

| 环境变量 | 说明 |
|----------|------|
| `AGENTRUN_MYSQL_PUBLIC_HOST` | MySQL 公网地址覆盖（当内网地址不可达时使用） |

### 跨模块依赖

`to_mem0_memory` 会依赖以下其他模块：

- `agentrun.model.ModelService` - 解析 LLM/Embedder 的 base_url
- `agentrun.credential.Credential` - 获取 API Key 或 MySQL 密码

---

## 代码生成机制

本模块使用 **模板 + 代码生成** 的模式来同时维护同步和异步代码。

### 工作流

```
模板文件（手动编写异步代码）
        │
        │  make codegen
        ▼
生成文件（自动生成同步+异步代码）
```

### 模板与生成文件的对应关系

| 模板文件 (手动维护) | 生成文件 (自动生成) |
|-------|---------|
| `__client_async_template.py` | `client.py` |
| `__memory_collection_async_template.py` | `memory_collection.py` |
| `codegen/configs/memory_collection_control_api.yaml` | `api/control.py` |

### 开发流程

1. 修改对应的 `_async_template.py` 模板文件（只需要编写 async 方法）
2. 运行 `make codegen` 自动生成包含同步和异步方法的完整文件
3. **切勿直接修改** `client.py`、`memory_collection.py`、`api/control.py`

---

## 开发注意事项

### 新增/修改字段

1. 先检查底层 SDK (`alibabacloud_agentrun20250910`) 的输入输出参数定义
2. 在 `model.py` 中添加/修改对应的数据模型字段
3. 如需修改 Client 或 MemoryCollection 的行为，编辑对应的 `_async_template.py` 模板
4. 运行 `make codegen` 重新生成代码
5. 运行单元测试验证

### 错误处理链路

```
底层 SDK 异常
  → ClientException / ServerException
    → ClientError / ServerError (api/control.py)
      → HTTPError (client.py)
        → ResourceAlreadyExistError / ResourceNotExistError (client.py)
```

### 类型系统

- 所有 Model 继承自 `agentrun.utils.model.BaseModel`
- `BaseModel` 提供 `model_dump()`（序列化）和 `from_inner_object()`（从底层 SDK 对象反序列化）
- `ResourceBase` 提供 `update_self()` 方法，用于就地更新实例属性
- `PageableInput` 提供分页参数（`page_number`、`page_size` 等）

### 常用命令

```bash
# 代码生成
make codegen

# 类型检查
uv run mypy --config-file mypy.ini .

# 运行测试
uv run pytest tests/
```
