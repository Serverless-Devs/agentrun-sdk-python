# AgentRun Python SDK

AgentRun Python SDK 是阿里云 AgentRun 服务的 Python 客户端库，提供简洁易用的 API 来管理 AI Agent 运行时环境。

## ✨ 特性

- 🎯 **简洁 API** - 面向对象的设计，直观易用
- ⚡ **异步支持** - 同时提供同步和异步接口
- 🔧 **类型提示** - 完整的类型注解，IDE 友好
- 🔐 **多种认证** - 支持 Access Key、STS Token 等
- 🌐 **多区域** - 支持阿里云所有可用区域
- 📝 **详细文档** - 完善的代码注释和示例

## 📦 安装

### 使用 pip 安装

```bash
pip install agentrun-sdk
```

可选依赖项
- `server`: 使用 AgentRunServer 集成 HTTP 服务
- `playwright：使用` Browser Sandbox 并集成 playwright
- `mcp：使用` MCP ToolSet
- `agentscope：集成` AgentScope
- `langchain：集成` LangChain
- `google`-adk：集成 Google ADK
- `crewai：集成` CrewAI
- `pydantic`-ai：集成 PydanticAI

假设您需要使用 agentscope，并且需要用到 Browser Sandbox，AgentRun 上的 MCP 服务，那么您应该通过如下方式安装
```bash
pip install agentrun-sdk[playwright,mcp,agentscope]
```


### 依赖要求

- Python 3.10 或更高版本

## 🚀 快速开始

你可以使用任意您喜欢的框架进行 Agent 开发，这里以 langchain 为例

### 1. 安装 Serverless Devs

运行脚手架，您需要使用 Serverless Devs 工具，请参考对应 [安装教程](https://serverless-devs.com/docs/user-guide/install)
> 如果您拥有 NodeJS 开发环境，可以使用 `npm i -g @serverless-devs/s` 快速安装 Serverless Devs
> 您也可以直接下载 [Serverless Devs 二进制程序](https://github.com/Serverless-Devs/Serverless-Devs/releases) 使用 Serverless Devs


### 2. 创建模板

使用快速创建脚手架创建您的 Agent

**注意！** 您需要确保您的 python 环境在 3.10 以上

```bash
# 初始化模板
s init agentrun-quick-start-langchain

# 按照实际情况进入代码目录
cd agentrun-quick-start-langchain/code

# 初始化虚拟环境并安装依赖
uv venv && uv pip install -r requirements.txt
```

### 3. 配置认证信息

设置环境变量（建议通过 `.env` 配置您的环境变量）

```bash
export AGENTRUN_ACCESS_KEY_ID="your-access-key-id"
export AGENTRUN_ACCESS_KEY_SECRET="your-access-key-secret"
export AGENTRUN_ACCOUNT_ID="your-account-id"
export AGENTRUN_REGION="cn-hangzhou"
```

### 4. 了解 Agent 如何与 LangChain 集成


使用 `from agentrun.integration.langchain import model, sandbox_toolset` 导入 langchain 的集成能力，这里默认提供了 `model`、`sandbox_toolset`、`toolset`，可以快速创建 langchain 可识别的大模型、工具
同时，通过 AgentRunServer 可以快速开放 HTTP Server 供其他业务集成

```python
from agentrun.integration.langchain import model, sandbox_toolset
from agentrun.sandbox import TemplateType
from agentrun.server import AgentRequest, AgentRunServer
from agentrun.utils.log import logger

# 请替换为您已经创建的 模型 和 沙箱 名称
MODEL_NAME = "<your-model-name>"
SANDBOX_NAME = "<your-sandbox-name>"

if MODEL_NAME.startswith("<"):
    raise ValueError("请将 MODEL_NAME 替换为您已经创建的模型名称")

code_interpreter_tools = []
if SANDBOX_NAME and not SANDBOX_NAME.startswith("<"):
    code_interpreter_tools = sandbox_toolset(
        template_name=SANDBOX_NAME,
        template_type=TemplateType.CODE_INTERPRETER,
        sandbox_idle_timeout_seconds=300,
    )
else:
    logger.warning("SANDBOX_NAME 未设置或未替换，跳过加载沙箱工具。")

# ...

# 自动启动 http server，提供 OpenAI 协议
AgentRunServer(invoke_agent=invoke_agent).start()
```

### 5. 调用 Agent

```bash
curl 127.0.0.1:9000/v1/chat/completions \
  -XPOST \
  -H "content-type: application/json" \
  -d '{"messages": [{"role": "user", "content": "通过代码查询现在是几点?"}], "stream":true}'
```

### 6. 部署项目

项目中已经存在 `s.yaml` 文件，这是 Serverless Devs 的部署配置文件，通过这个文件，您可以配置当前 Agent 在 Agent Run 上的名称、CPU/内存规格、日志投递信息

在示例情况下，您只需要简单修改该文件即可。修改 `role` 字段为拥有 AgentRun Full Access 的角色（如果您拥有精细化权限控制的需求，可以根据实际使用的 API 收敛权限）

```yaml
role: acs:ram::{您的阿里云主账号 ID}:role/{您的阿里云角色名称}
```

> 如果在未来的使用中遇到了任何 Serverless Devs 相关问题，都可以参考 [Serverless Devs 相关文档](https://serverless-devs.com/docs/overview)

在部署前，您需要配置您的部署密钥，使用 `s config add` 进入交互式密钥管理，并按照引导录入您在阿里云的 Access Key ID 与 Access Key Secret。在录入过程中，您需要短期记忆一下您输入的密钥对名称（假设为 `agentrun-deploy`）

配置完成后，即可执行部署

```bash
s deploy -a agentrun-deploy
# agentrun-deploy 是您使用的密钥对名称，也可以将该名称写入到 s.yaml 开头的 access: 字段中
```

### 7. 在线上进行调用

部署完成后，您可以看到如下格式的输出
```
endpoints: 
      - 
        id:          ...
        arn:         ...
        name:        ...
        url:         https://12345.agentrun-data.cn-hangzhou.aliyuncs.com/agent-runtimes/abcd/endpoints/prod/invocations
```

此处的 url 为您的 Agent 调用地址，将实际的请求 path 拼接到该 base url 后，即可调用云上的 Agent 资源

```bash
curl https://12345.agentrun-data.cn-hangzhou.aliyuncs.com/agent-runtimes/abcd/endpoints/prod/invocations/v1/chat/completions \
  -XPOST \
  -H "content-type: application/json" \
  -d '{"messages": [{"role": "user", "content": "通过代码查询现在是几点?"}], "stream":true}'
```

## ⚙️ 配置说明

### Config 类

用于配置认证信息和客户端参数。

```python
from agentrun.utils.config import Config

config = Config(
    access_key_id="your-key-id",            # Access Key ID
    access_key_secret="your-secret",        # Access Key Secret
    security_token="your-sts-token",        # 可选：STS Token
    token="token",                          # 数据链路 token（可以在无 AK 情况下调用数据链路）
    headers={},                             # 附加的请求头
    account_id="your-account-id",           # 账号 ID
    region_id="cn-hangzhou",                # 区域
    timeout=30,                             # 可选：请求超时（秒）
    control_endpoint="",                    # 可选：自定义控制端点
    data_endpoint="",                       # 可选：自定义数据端点
)

# 使用配置创建客户端
client = agent_runtime.AgentRuntimeClient()
agent = client.create(input_config, config=config)
```

### 环境变量

SDK 会自动读取以下环境变量：

| 环境变量 | 说明 | 备用变量 |
|---------|------|---------|
| `AGENTRUN_ACCESS_KEY_ID` | Access Key ID | `ALIBABA_CLOUD_ACCESS_KEY_ID` |
| `AGENTRUN_ACCESS_KEY_SECRET` | Access Key Secret | `ALIBABA_CLOUD_ACCESS_KEY_SECRET` |
| `AGENTRUN_SECURITY_TOKEN` | STS Token | `ALIBABA_CLOUD_SECURITY_TOKEN` |
| `AGENTRUN_ACCOUNT_ID` | 账号 ID | `FC_ACCOUNT_ID` |
| `AGENTRUN_REGION` | 区域 | `FC_REGION` |
| `AGENTRUN_CONTROL_ENDPOINT` | 控制端点 | - |
| `AGENTRUN_DATA_ENDPOINT` | 数据端点 | - |
| `AGENTRUN_SDK_DEBUG` | 开启 DEBUG 日志 | - |

