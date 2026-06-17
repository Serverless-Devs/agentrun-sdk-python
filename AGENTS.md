# AgentRun 核心概念

本文档详细介绍 AgentRun 的核心概念、架构设计和最佳实践。

## 📚 目录

- [什么是 AgentRun](#什么是-agentrun)
- [核心概念](#核心概念)
- [架构设计](#架构设计)
- [部署方式](#部署方式)
- [网络配置](#网络配置)
- [最佳实践](#最佳实践)

## 什么是 AgentRun

AgentRun 是阿里云提供的 AI Agent 运行时服务，为 AI Agent 应用提供托管的运行环境。开发者无需关心底层基础设施，即可快速部署和运行各类 AI Agent 应用。

### 核心优势

- **🚀 快速部署** - 支持代码包和容器镜像两种部署方式，几分钟内完成部署
- **📈 弹性伸缩** - 自动根据负载调整资源，按需付费
- **🔒 安全可靠** - 企业级安全防护，多可用区容灾
- **🔌 易于集成** - 提供丰富的 SDK 和 API，轻松集成到现有系统
- **📊 监控运维** - 完善的日志、监控和告警体系

## 核心概念

### Agent Runtime（智能体运行时）

Agent Runtime 是 AgentRun 中的核心资源，代表一个运行中的 AI Agent 实例。每个 Agent Runtime 包含以下关键属性：

- **名称（agent_runtime_name）** - 唯一标识符，用于区分不同的 Agent
- **制品类型（artifact_type）** - 部署方式，支持 `CODE`（代码包）或 `CONTAINER`（容器镜像）
- **配置信息** - 包括代码配置、容器配置、网络配置等
- **状态（status）** - 运行时的当前状态
- **版本（version）** - 支持多版本管理

#### Agent Runtime 状态

| 状态 | 说明 |
|-----|------|
| `CREATING` | 创建中 |
| `READY` | 就绪，可正常提供服务 |
| `UPDATING` | 更新中 |
| `DELETING` | 删除中 |
| `FAILED` | 失败 |
| `DELETE_FAILED` | 删除失败 |

### Agent Runtime Endpoint（访问端点）

Endpoint 是 Agent Runtime 的对外访问入口，每个 Agent Runtime 可以创建多个 Endpoint 以支持不同的访问场景。

#### Endpoint 特性

- **公网访问** - 自动分配公网域名，支持 HTTPS
- **内网访问** - VPC 内网访问，低延迟高安全
- **路由配置** - 支持基于权重的流量分发
- **健康检查** - 自动检测 Agent 健康状态
- **协议支持** - HTTP/HTTPS/gRPC

#### Endpoint 状态

| 状态 | 说明 |
|-----|------|
| `CREATING` | 创建中 |
| `READY` | 就绪，可正常访问 |
| `UPDATING` | 更新中 |
| `DELETING` | 删除中 |
| `FAILED` | 失败 |

### Agent Runtime Version（版本）

版本管理允许您维护 Agent Runtime 的多个历史版本，支持版本回滚和灰度发布。

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                        用户应用                          │
└───────────────────┬─────────────────────────────────────┘
                    │
                    │ SDK/API 调用
                    │
┌───────────────────▼─────────────────────────────────────┐
│                   AgentRun 控制面                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Runtime    │  │   Endpoint   │  │   Version    │  │
│  │  Management  │  │  Management  │  │  Management  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└───────────────────┬─────────────────────────────────────┘
                    │
                    │ 编排调度
                    │
┌───────────────────▼─────────────────────────────────────┐
│                   AgentRun 数据面                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │          Agent Runtime 实例池                    │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │  │
│  │  │ Agent A  │  │ Agent B  │  │ Agent C  │  ...  │  │
│  │  └──────────┘  └──────────┘  └──────────┘       │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │          负载均衡 & 路由                         │  │
│  └──────────────────────────────────────────────────┘  │
└───────────────────┬─────────────────────────────────────┘
                    │
                    │ 用户请求
                    │
┌───────────────────▼─────────────────────────────────────┐
│                    外部访问                              │
│              (公网/VPC 内网)                             │
└─────────────────────────────────────────────────────────┘
```

### 工作流程

1. **部署阶段**
   - 开发者通过 SDK 提交代码或镜像
   - AgentRun 创建 Agent Runtime 实例
   - 系统自动完成环境配置和依赖安装

2. **运行阶段**
   - Agent Runtime 进入 READY 状态
   - 创建 Endpoint 对外提供服务
   - 负载均衡器分发请求到 Agent 实例

3. **更新阶段**
   - 提交新版本代码或配置
   - 系统创建新版本实例
   - 平滑切换流量，无缝升级

4. **销毁阶段**
   - 删除 Endpoint，停止接收新请求
   - 优雅停止 Agent 实例
   - 释放相关资源

## 部署方式

AgentRun 支持两种部署方式，满足不同场景需求。

### 方式一：代码包部署（CODE）

适合快速开发和部署简单应用。

**特点：**
- 直接上传代码文件
- 支持多种编程语言（Node.js、Python、Java 等）
- 自动安装依赖
- 快速迭代

**示例：**

```python
from agentrun import agent_runtime

agent = client.create(
    agent_runtime.AgentRuntimeInput(
        agent_runtime_name="my-agent",
        code_configuration=agent_runtime.AgentRuntimeCode().from_file(
            language=agent_runtime.AgentRuntimeLanguage.NODEJS18,
            command=["node", "index.js"],
            file_path="/path/to/code",
        ),
    )
)
```

**支持的语言：**
- Node.js 14/16/18/20
- Python 3.10+
- Java 8/11/17
- Go 1.x
- .NET Core 3.1/6.0

### 方式二：容器镜像部署（CONTAINER）

适合复杂应用和生产环境。

**特点：**
- 完全自定义运行环境
- 支持任何容器化应用
- 版本管理更清晰
- 与 CI/CD 流程集成

**示例：**

```python
from agentrun import agent_runtime

agent = client.create(
    agent_runtime.AgentRuntimeInput(
        agent_runtime_name="my-agent",
        container_configuration=agent_runtime.AgentRuntimeContainer(
            image="registry.cn-hangzhou.aliyuncs.com/your-namespace/agent:latest",
            command=["python", "app.py"],
            port=8080,
        ),
    )
)
```

## 网络配置

### 网络模式

AgentRun 支持灵活的网络配置，满足不同安全和性能需求。

#### 公网模式（INTERNET）

- 自动分配公网域名
- 支持 HTTPS 加密
- 适合对外提供服务

#### VPC 模式

- 私有网络隔离
- 低延迟高带宽
- 适合内部服务调用

**配置示例：**

```python
from agentrun import agent_runtime

network_config = agent_runtime.AgentRuntimeNetworkConfig(
    network_mode=agent_runtime.AgentRuntimeNetworkMode.INTERNET,
    vpc_config=None  # 公网模式不需要 VPC 配置
)
```

### 健康检查

配置健康检查确保 Agent 正常运行：

```python
health_check = agent_runtime.AgentRuntimeHealthCheckConfig(
    enabled=True,
    path="/health",  # 健康检查路径
    initial_delay_seconds=10,  # 初始延迟
    period_seconds=30,  # 检查间隔
    timeout_seconds=5,  # 超时时间
    failure_threshold=3,  # 失败阈值
)
```

### 协议配置

支持多种协议类型：

```python
protocol_config = agent_runtime.AgentRuntimeProtocolConfig(
    protocol_type=agent_runtime.AgentRuntimeProtocolType.HTTP,
    port=8080,
)
```

## 最佳实践

### 1. 环境变量管理

使用环境变量管理敏感信息和配置：

```bash
# 生产环境
export AGENTRUN_ACCESS_KEY_ID="prod-key"
export AGENTRUN_ACCESS_KEY_SECRET="prod-secret"
export AGENTRUN_REGION="cn-shanghai"

# 开发环境
export AGENTRUN_ACCESS_KEY_ID="dev-key"
export AGENTRUN_ACCESS_KEY_SECRET="dev-secret"
export AGENTRUN_REGION="cn-hangzhou"
```

### 2. 状态管理

正确处理 Agent Runtime 状态：

```python
# 等待 Agent 就绪
agent.wait_until_ready(
    timeout_seconds=300,
    before_check_callback=lambda a: print(f"当前状态: {a.status}")
)

# 检查状态并处理
if agent.status == agent_runtime.AgentRuntimeStatus.FAILED:
    print(f"部署失败: {agent.status_reason}")
    # 进行错误处理
```

### 3. 资源清理

及时清理不再使用的资源：

```python
# 删除 Endpoint
for endpoint in agent.list_endpoints():
    endpoint.delete()

# 删除 Agent Runtime
agent.delete()
```

### 4. 版本管理

维护多个版本支持灰度发布：

```python
# 创建新版本
new_agent = client.create(updated_config)

# 配置流量分配
endpoint.update(
    agent_runtime.AgentRuntimeEndpointInput(
        routing_config=agent_runtime.AgentRuntimeEndpointRoutingConfig(
            weights=[
                agent_runtime.AgentRuntimeEndpointRoutingWeight(
                    agent_runtime_id=old_agent.agent_runtime_id,
                    weight=80  # 80% 流量给旧版本
                ),
                agent_runtime.AgentRuntimeEndpointRoutingWeight(
                    agent_runtime_id=new_agent.agent_runtime_id,
                    weight=20  # 20% 流量给新版本
                )
            ]
        )
    )
)
```

### 5. 错误处理

实现健壮的错误处理机制：

```python
from agentrun.utils.exception import ResourceNotExistError, ClientError

try:
    agent = client.get("agent-id")
except ResourceNotExistError:
    print("Agent 不存在")
except ClientError as e:
    print(f"API 调用失败: {e.message}")
    print(f"错误码: {e.error_code}")
```

### 6. 日志配置

配置日志收集便于问题排查：

```python
log_config = agent_runtime.AgentRuntimeLogConfig(
    enabled=True,
    log_store="your-log-store",  # SLS 日志库
    project="your-project",  # SLS 项目
)
```

### 7. 异步编程

对于高并发场景，使用异步 API：

```python
import asyncio
from agentrun import agent_runtime

async def deploy_multiple_agents():
    client = agent_runtime.AgentRuntimeClient()
    
    # 并发创建多个 Agent
    tasks = [
        client.create_async(config1),
        client.create_async(config2),
        client.create_async(config3),
    ]
    
    agents = await asyncio.gather(*tasks)
    return agents

# 运行
agents = asyncio.run(deploy_multiple_agents())

### 8. 变更后的类型检查要求

所有由 AI（或自动化 agent）提交或修改的代码变更，必须在提交/合并前后执行静态类型检查，并在变更记录中包含检查结果摘要：

- **运行命令**：使用项目根目录的 mypy 配置运行：

    ```bash
    mypy --config-file mypy.ini .
    ```

- **必需项**：AI 在每次修改代码并准备提交时，必须：
    - 运行上述类型检查命令并等待完成；
    - 若检查通过，在提交消息或 PR 描述中写入简短摘要（例如："类型检查通过，检查文件数 N"）；
    - 若检查失败，AI 应在 PR 描述中列出前 30 条错误（或最关键的若干条），并给出优先修复建议或自动修复方案。

- **CI 行为**：项目 CI 可根据仓库策略决定是否将类型检查失败作为阻断条件；AI 应遵从仓库当前 CI 策略并在 PR 中说明检查结果。

此要求旨在保证类型安全随代码变更持续得到验证，减少回归并提高编辑器与 Copilot 的诊断可靠性。
```

### 9. 运行命令约定

请使用 `uv run ...` 执行所有 Python 相关命令，避免直接调用系统的 `python`。例如：

- `uv run mypy --config-file mypy.ini .`
- `uv run python examples/quick_start.py`

## 常见问题

### Q: Agent Runtime 启动失败怎么办？

A: 检查以下几点：
1. 代码或镜像是否正确
2. 启动命令是否正确
3. 端口配置是否匹配
4. 查看 `status_reason` 字段获取详细错误信息

### Q: 如何实现零停机更新？

A: 使用版本管理和流量路由：
1. 创建新版本 Agent Runtime
2. 等待新版本就绪
3. 配置 Endpoint 路由权重，逐步切换流量
4. 确认新版本稳定后删除旧版本

### Q: 如何优化 Agent 启动速度？

A: 建议：
1. 使用容器镜像部署，提前构建好环境
2. 优化应用启动逻辑，减少初始化时间
3. 合理配置健康检查参数
4. 使用预留实例（如果支持）

### Q: 如何控制成本？

A: 建议：
1. 及时删除不用的 Agent Runtime
2. 根据实际负载配置合适的实例规格
3. 使用按量付费，避免资源闲置
4. 合理设置自动伸缩策略

## 更多资源

- [Python SDK 文档](./python/README.md)
- [示例代码](./python/examples/)
- [阿里云 AgentRun 官方文档](https://help.aliyun.com/zh/agentrun/)

## 工作流程

### 为特定模块增加参数

名词解释
- AgentRun SDK：当前项目
- 底层 SDK：我们会依赖底层 SDK 与平台进行交互，如果是 python sdk，则为 `agentrun-20250910`；如果是 nodejs sdk，则为 `@alicloud/agentrun20250910`
- 要修改模块：sandbox、model 等

此时，应该遵循如下流程

该模块的代码位于 ${component} 目录，包含如下部分
- api/ 							和底层 SDK 交互的逻辑。你可以根据该文件的调用方式，跳转到底层 sdk 的逻辑，进行检查
- client 						为用户封装的客户端，你需要检查一下是否需要修改
- model							相关类型描述，你需要严格检查当前文件是否需要修改，这里可能存在大量修改
- 其他文件						你也需要一并检查，如果是 nodejs sdk，需要检查 class 的属性
- _xxx_async_template.py 		python sdk 中生成 xxx.py 文件的模板，只需要声明 async 即可（使用 `make codegen` 进行生成）

1. 请严格检查底层 SDK 的输入、输出参数，并对照现有类型定义，整理出新增的参数列表
2. 根据新增的参数列表，在 agentrun sdk 中进行修改，补充对应的中英文注释
3. 执行相关模块的 ut 测试，确保可以正确执行
4. 进行修改内容的总结汇报
5. 根据汇报内容进行检查，重新检查底层 SDK 和 AgentRun SDK 的定义

## 凭证注入与 STS 静默刷新约定（强制）

STS 临时凭证（ak/sk/security_token）会过期。部署在函数计算（FC）时，最新轮转
后的 STS 通过**每次请求的 HTTP 头**下发，而非进程级环境变量。为让所有 client 在
凭证过期后静默刷新，本仓库采用统一机制：

1. **请求级 overlay**：`agentrun/server/sts_middleware.py` 解析 FC 头
   （默认 `x-fc-access-key-id` / `x-fc-access-key-secret` / `x-fc-security-token`，
   可经构造参数或 `AGENTRUN_STS_HEADER_*` 环境变量覆盖），写入
   `agentrun/utils/credential_context.py` 的 `contextvars` overlay。

2. **Config 懒解析**：`Config` 的三个凭证 getter 按
   **显式传入 > 请求级 overlay（仅当三者均未显式传入）> 环境变量** 解析。
   切勿在 `Config.__init__` 里把凭证快照成固定字符串。

3. **client 一律用 credential provider，禁止传静态 ak/sk/sts**：

   - **alibabacloud OpenAPI**（控制面 / Bailian / GPDB / Devs 等）：
     构造 `open_api_util_models.Config` 时传
     `credential=build_openapi_credential(cfg)`
     （见 `agentrun/utils/credential_providers.py`），**不要**再传
     `access_key_id` / `access_key_secret` / `security_token`。
     注意 tea_openapi 的优先级是「静态 ak/sk 优先于 credential」，传了静态值
     会让 provider 失效。

   - **TableStore `OTSClient` / `AsyncOTSClient`**：
     构造时传 `credentials_provider=build_ots_credentials_provider(cfg)`
     （见 `agentrun/conversation_service/utils.py`），**不要**再传
     `access_key_id` / `access_key_secret` / `sts_token`。

   原因：直接传静态凭证会在 client 构造时把凭证冻结，长生命周期 client（如
   server 启动时仅创建一次的 OTSClient）在 STS 过期后所有请求都会失败。
   provider 会在**每次请求**被底层 SDK 调用，从而拿到最新 STS。

4. **数据面手写签名**（`agentrun/utils/data_api.py` 的 RAM 签名）无需 provider：
   它本就每次请求调用 `cfg.get_*()`，已随 Config 懒解析自动刷新。

5. **自定义 httpx 签名器**（如 `_AgentrunRamAuth`，用于 MCP SSE / OpenAPI 工具）
   必须**持有 `Config`**、在 `auth_flow` 内调用 `cfg.get_*()` 取证，
   **不要**在 `__init__` 把 ak/sk/sts 快照成字段。否则长连接（SSE 一次建连、
   多请求复用）会冻结建连时的 STS。

新增任何与阿里云 / TableStore 交互的 client 时，必须遵循第 3 条；新增单测应覆盖
「overlay 生效」与「显式凭证不被 overlay 覆盖」两种情况
（参考 `tests/unittests/test_sts_refresh.py`）。
