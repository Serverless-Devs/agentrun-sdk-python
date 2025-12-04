# AgentRun Sandbox SDK

AgentRun Sandbox SDK 提供了一种强大而灵活的方式来创建隔离环境，用于代码执行和浏览器自动化。该 SDK 支持三种沙箱类型：

- **Code Interpreter** - 用于在隔离环境中执行 Python 代码
- **Browser** - 用于自动化 Web 交互，支持 Playwright、VNC 和 CDP
- **All-in-One (AIO)** - 在单个沙箱中结合 Code Interpreter 和 Browser 的全部功能

## 目录

- [概述](#概述)
- [安装](#安装)
- [快速开始](#快速开始)
  - [Code Interpreter 快速开始](#code-interpreter-快速开始)
  - [Browser 快速开始](#browser-快速开始)
  - [All-in-One 快速开始](#all-in-one-快速开始)
- [核心概念](#核心概念)
  - [Templates](#templates)
  - [Sandboxes](#sandboxes)
- [Template 操作](#template-操作)
- [Sandbox 操作](#sandbox-操作)
- [Code Interpreter](#code-interpreter)
  - [Context 管理](#context-管理)
  - [文件操作](#文件操作)
  - [文件系统操作](#文件系统操作)
  - [进程管理](#进程管理)
- [Browser Sandbox](#browser-sandbox)
  - [Playwright 集成](#playwright-集成)
  - [VNC 和 CDP 访问](#vnc-和-cdp-访问)
  - [录制管理](#录制管理)
- [All-in-One Sandbox](#all-in-one-sandbox)
  - [快速开始](#all-in-one-快速开始-1)
  - [功能特性](#功能特性)
  - [使用示例](#使用示例)
- [异步支持](#异步支持)
- [最佳实践](#最佳实践)
- [示例](#示例)

## 概述

Sandbox SDK 使您能够：

- **安全执行代码**：在隔离环境中运行 Python 代码
- **自动化浏览器**：控制 Web 浏览器进行测试和自动化
- **管理文件**：在沙箱中上传、下载和操作文件
- **监控进程**：跟踪和管理运行中的进程
- **录制会话**：捕获浏览器会话以进行调试和分析

### 支持的功能

- ✅ **Code Interpreter**：支持 Python 执行和 context 管理
- ✅ **Browser Automation**：Playwright 集成，支持 CDP 和 VNC
- ✅ **All-in-One Sandbox**：在单个环境中结合代码解释器和浏览器功能
- ✅ **File Management**：上传、下载、读取、写入、移动文件
- ✅ **Process Control**：执行命令和管理进程
- ✅ **Session Recording**：录制浏览器会话
- ✅ **Health Monitoring**：内置健康检查
- ✅ **Async/Await**：完整的异步 API 支持
- ✅ **Context Managers**：自动资源清理

## 安装

```bash
pip install agentrun
```

## 配置

### 使用环境变量

```bash
export AGENTRUN_ACCESS_KEY_ID=your_access_key_id
export AGENTRUN_ACCESS_KEY_SECRET=your_access_key_secret
export AGENTRUN_ACCOUNT_ID=your_account_id
```

## 快速开始

### 沙箱类型概览

| 类型 | 类名 | 描述 |
|------|------|------|
| `TemplateType.CODE_INTERPRETER` | `CodeInterpreterSandbox` | Python 代码执行、文件管理、进程控制 |
| `TemplateType.BROWSER` | `BrowserSandbox` | 浏览器自动化，支持 Playwright、VNC、CDP |
| `TemplateType.AIO` | `AioSandbox` | **All-in-One**：结合 Code Interpreter 和 Browser 的全部功能 |

### Code Interpreter 快速开始

```python
import time
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType
from agentrun.sandbox.model import CodeLanguage

# 创建模板
template_name = f"my-code-template-{time.strftime('%Y%m%d%H%M%S')}"
template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.CODE_INTERPRETER,
    )
)

print(f"Template created: {template.template_name}")

# 使用上下文管理器创建和使用沙箱
with Sandbox.create(
    template_type=TemplateType.CODE_INTERPRETER,
    template_name=template_name,
    sandbox_idle_timeout_seconds=600,
) as sandbox:
    print(f"Sandbox created: {sandbox.sandbox_id}")
    
    # 执行代码
    result = sandbox.context.execute(code="print('Hello, AgentRun!')")
    print(f"Execution result: {result}")
    
    # 创建文件
    sandbox.file.write(path="/tmp/test.txt", content="Hello World")
    
    # 读取文件
    content = sandbox.file.read(path="/tmp/test.txt")
    print(f"File content: {content}")

# 退出上下文后沙箱自动清理

# 清理模板
Sandbox.delete_template(template_name)
```

### Browser 快速开始

```python
import time
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType

# 创建浏览器模板
template_name = f"my-browser-template-{time.strftime('%Y%m%d%H%M%S')}"
template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.BROWSER,
    )
)

print(f"Browser template created: {template.template_name}")

# 创建浏览器沙箱
sandbox = Sandbox.create(
    template_type=TemplateType.BROWSER,
    template_name=template_name,
)

print(f"Browser sandbox created: {sandbox.sandbox_id}")
print(f"VNC URL: {sandbox.get_vnc_url()}")
print(f"CDP URL: {sandbox.get_cdp_url()}")

# 等待浏览器就绪
while sandbox.check_health()["status"] != "ok":
    time.sleep(1)

# 使用 Playwright 控制浏览器
with sandbox.sync_playwright(record=True) as playwright:
    playwright.new_page().goto("https://www.example.com")
    
    title = playwright.title()
    print(f"Page title: {title}")
    
    # 截图
    playwright.screenshot(path="screenshot.png")
    print("Screenshot saved")

# 列出录制
recordings = sandbox.list_recordings()
print(f"Recordings: {recordings}")

# 下载录制
if recordings.get("recordings"):
    filename = recordings["recordings"][0]["filename"]
    sandbox.download_recording(filename, f"./{filename}")

# 清理
sandbox.delete()
Sandbox.delete_template(template_name)
```

### All-in-One 快速开始

```python
import time
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType

# 创建 All-in-One 模板
template_name = f"my-aio-template-{time.strftime('%Y%m%d%H%M%S')}"
template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.AIO,  # All-in-One 类型
    )
)

print(f"All-in-One template created: {template.template_name}")

# 使用上下文管理器创建 All-in-One 沙箱
with Sandbox.create(
    template_type=TemplateType.AIO,
    template_name=template_name,
) as sandbox:
    print(f"All-in-One sandbox created: {sandbox.sandbox_id}")
    
    # Code Interpreter 功能
    result = sandbox.context.execute(code="print('Hello from AIO!')")
    print(f"Code execution result: {result}")
    
    sandbox.file.write(path="/tmp/test.txt", content="Hello World")
    content = sandbox.file.read(path="/tmp/test.txt")
    print(f"File content: {content}")
    
    # Browser 功能
    print(f"VNC URL: {sandbox.get_vnc_url()}")
    
    with sandbox.sync_playwright(record=True) as playwright:
        playwright.new_page().goto("https://www.example.com")
        title = playwright.title()
        print(f"Page title: {title}")

# 退出上下文后沙箱自动清理

# 清理模板
Sandbox.delete_template(template_name)
```

## 核心概念

### Templates

Template 定义了沙箱的配置，它们指定了：
- **类型**：Code Interpreter、Browser 或 All-in-One (AIO)
- **资源**：CPU、内存、磁盘大小
- **网络**：网络模式和 VPC 配置
- **环境**：环境变量和凭证
- **超时时间**：空闲超时和 TTL

Template 是可重用的，可以用来创建多个具有相同配置的沙箱。

#### 模板类型

| 类型 | 描述 | 使用场景 |
|------|------|----------|
| `TemplateType.CODE_INTERPRETER` | Python 代码执行环境 | 数据处理、脚本执行、计算任务 |
| `TemplateType.BROWSER` | 浏览器自动化环境 | Web 抓取、测试、自动化 |
| `TemplateType.AIO` | All-in-One 组合环境 | 需要同时使用代码和浏览器的复杂工作流 |

### Sandboxes

Sandbox 是从模板创建的隔离运行时环境。每个沙箱：
- 独立运行，拥有自己的资源
- 具有唯一的 ID 用于识别
- 可以被创建、连接、停止和删除
- 在空闲一段时间后自动超时
- 支持健康监控

## Template 操作

### 创建 Template

创建一个新的沙箱模板：

```python
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType, TemplateNetworkConfiguration, TemplateNetworkMode

# 基础模板
template = Sandbox.create_template(
    input=TemplateInput(
        template_name="my-template",
        template_type=TemplateType.CODE_INTERPRETER,
    )
)

# 具有自定义配置的高级模板
template = Sandbox.create_template(
    input=TemplateInput(
        template_name="my-advanced-template",
        template_type=TemplateType.CODE_INTERPRETER,
        cpu=4.0,  # 4 核 CPU
        memory=8192,  # 8GB 内存
        disk_size=10240,  # 10GB 磁盘
        sandbox_idle_timeout_in_seconds=7200,  # 2 小时
        environment_variables={"MY_VAR": "value"},
        network_configuration=TemplateNetworkConfiguration(
            network_mode=TemplateNetworkMode.PUBLIC
        ),
    )
)

# All-in-One 模板（结合 Code Interpreter + Browser）
# 默认值：4 核 CPU，8GB 内存，10GB 磁盘
aio_template = Sandbox.create_template(
    input=TemplateInput(
        template_name="my-aio-template",
        template_type=TemplateType.AIO,
        # cpu, memory, disk_size 使用 AIO 的更高默认值
    )
)
```

### 获取 Template

检索现有模板：

```python
template = Sandbox.get_template("my-template")
print(f"Template ID: {template.template_id}")
print(f"Template Type: {template.template_type}")
print(f"Status: {template.status}")
```

### 更新 Template

更新现有模板：

```python
from agentrun.sandbox import TemplateInput

updated_template = Sandbox.update_template(
    template_name="my-template",
    input=TemplateInput(
        template_name="my-template",
        template_type=TemplateType.CODE_INTERPRETER,
        cpu=8.0,  # 增加 CPU
        memory=16384,  # 增加内存
    )
)
```

### 删除 Template

删除模板：

```python
Sandbox.delete_template("my-template")
```

### 列出 Templates

列出所有模板：

```python
from agentrun.sandbox.model import PageableInput

templates = Sandbox.list_templates(
    input=PageableInput(
        page_number=1,
        page_size=10,
        template_type=TemplateType.CODE_INTERPRETER
    )
)

for template in templates:
    print(f"Template: {template.template_name} ({template.template_type})")
```

## Sandbox 操作

### 创建 Sandbox

从模板创建新沙箱：

```python
# 使用上下文管理器（推荐）
with Sandbox.create(
    template_type=TemplateType.CODE_INTERPRETER,
    template_name="my-template",
    sandbox_idle_timeout_seconds=600,
) as sandbox:
    # 使用沙箱
    print(f"Sandbox ID: {sandbox.sandbox_id}")
    # 退出时自动清理

# 手动创建
sandbox = Sandbox.create(
    template_type=TemplateType.CODE_INTERPRETER,
    template_name="my-template",
)
```

### 连接到现有 Sandbox

连接到运行中的沙箱：

```python
# 通过 ID 连接（自动检测类型）
sandbox = Sandbox.connect(sandbox_id="your-sandbox-id")

# 使用显式类型连接以获得更好的类型提示
sandbox = Sandbox.connect(
    sandbox_id="your-sandbox-id",
    template_type=TemplateType.CODE_INTERPRETER
)

# 连接到 Browser 沙箱
browser_sandbox = Sandbox.connect(
    sandbox_id="your-browser-sandbox-id",
    template_type=TemplateType.BROWSER
)

# 连接到 All-in-One 沙箱
aio_sandbox = Sandbox.connect(
    sandbox_id="your-aio-sandbox-id",
    template_type=TemplateType.AIO
)
```

### 停止 Sandbox

停止运行中的沙箱：

```python
Sandbox.stop_by_id("sandbox-id")
sandbox.stop() # 实例方法，等同于 Sandbox.stop_by_id("sandbox-id")
```

### 删除 Sandbox

删除沙箱：

```python
Sandbox.delete_by_id("sandbox-id")
sandbox.delete() # 实例方法，等同于 Sandbox.delete_by_id("sandbox-id")
```

### 列出 Sandboxes

列出所有沙箱：

```python
from agentrun.sandbox.model import ListSandboxesInput

result = Sandbox.list(
    input=ListSandboxesInput(
        max_results=20,
        template_type=TemplateType.CODE_INTERPRETER,
        status="Running"
    )
)

for sandbox in result.sandboxes:
    print(f"Sandbox: {sandbox.sandbox_id} - {sandbox.status}")
```

### 健康检查

检查沙箱是否健康：

```python
health = sandbox.check_health()
if health["status"] == "ok":
    print("Sandbox is ready!")
```

## Code Interpreter

Code Interpreter 沙箱允许您在安全、隔离的环境中执行代码。

### Context 管理

Context 在多次代码执行之间维护执行状态（变量、导入等）。

#### 创建 Context

```python
from agentrun.sandbox.model import CodeLanguage

# 创建 context
with sandbox.context.create(language=CodeLanguage.PYTHON) as ctx:
    # 在 context 中执行代码
    result = ctx.execute(code="x = 10")
    print(result)
    
    result = ctx.execute(code="print(x)")  # x 仍然可用
    print(result)
    # 退出时自动清理 context
```

#### 执行代码

使用或不使用 context 执行代码：

```python
# 不使用 context 执行（无状态）
result = sandbox.context.execute(code="print('Hello World')")

# 在特定 context 中执行（有状态）
result = sandbox.context.execute(
    code="print(x + 5)",
    context_id="your-context-id",
    timeout=30
)
```

#### 列出 Contexts

```python
contexts = sandbox.context.list()
for ctx in contexts:
    print(f"Context: {ctx['id']} - {ctx['language']}")
```

#### 获取 Context

```python
ctx = sandbox.context.get(context_id="your-context-id")
print(f"Context language: {ctx._language}")
```

#### 删除 Context

```python
sandbox.context.delete(context_id="your-context-id")
```

### 文件操作

简单的文件读写操作。

#### 写入文件

```python
sandbox.file.write(
    path="/tmp/data.txt",
    content="Hello, World!",
    mode="644",
    encoding="utf-8",
    create_dir=True  # 如果需要，创建父目录
)
```

#### 读取文件

```python
content = sandbox.file.read(path="/tmp/data.txt")
print(content)
```

### 文件系统操作

用于管理文件和目录的高级文件系统操作。

#### 列出目录

```python
# 列出根目录
files = sandbox.file_system.list(path="/")
print(files)

# 控制深度的列出
files = sandbox.file_system.list(path="/tmp", depth=2)
```

#### 创建目录

```python
sandbox.file_system.mkdir(
    path="/tmp/my-dir",
    parents=True,  # 创建父目录
    mode="0755"
)
```

#### 移动文件/目录

```python
sandbox.file_system.move(
    source="/tmp/old-file.txt",
    destination="/tmp/new-file.txt"
)
```

#### 删除文件/目录

```python
sandbox.file_system.remove(path="/tmp/my-dir")
```

#### 获取文件统计信息

```python
stats = sandbox.file_system.stat(path="/tmp/data.txt")
print(f"Size: {stats['size']}")
print(f"Modified: {stats['mtime']}")
```

#### 上传文件

```python
result = sandbox.file_system.upload(
    local_file_path="./local-file.txt",
    target_file_path="/tmp/remote-file.txt"
)
print(f"Uploaded: {result}")
```

#### 下载文件

```python
result = sandbox.file_system.download(
    path="/tmp/remote-file.txt",
    save_path="./downloaded-file.txt"
)
print(f"Downloaded: {result['saved_path']}, Size: {result['size']}")
```

### 进程管理

管理和监控沙箱中运行的进程。

#### 执行命令

```python
result = sandbox.process.cmd(
    command="ls -la",
    cwd="/tmp",
    timeout=30
)
print(f"Exit code: {result['exitCode']}")
print(f"Output: {result['stdout']}")
```

#### 列出进程

```python
processes = sandbox.process.list()
for proc in processes:
    print(f"PID: {proc['pid']} - {proc['name']}")
```

#### 获取进程

```python
proc = sandbox.process.get(pid="1234")
print(f"Process: {proc['name']} - Status: {proc['status']}")
```

#### 终止进程

```python
sandbox.process.kill(pid="1234")
```

## Browser Sandbox

Browser 沙箱提供了与 Playwright 集成的自动化浏览器控制。

### Playwright 集成

#### 同步 Playwright

```python
with sandbox.sync_playwright(record=True) as playwright:
    # 创建新页面
    playwright.new_page().goto("https://www.example.com")
    
    # 获取页面标题
    title = playwright.title()
    print(f"Title: {title}")
    
    # 截图
    playwright.screenshot(path="page.png")
    
    # 点击元素
    playwright.click("button#submit")
    
    # 填写表单
    playwright.fill("input[name='username']", "myuser")
    
    # 获取内容
    content = playwright.content()
```

#### 异步 Playwright

```python
async with sandbox.async_playwright(record=True) as playwright:
    await playwright.goto("https://www.example.com")
    
    title = await playwright.title()
    print(f"Title: {title}")
    
    await playwright.screenshot(path="page.png")
```

### VNC 和 CDP 访问

通过 VNC 或 Chrome DevTools Protocol 直接访问浏览器。

#### 获取 VNC URL

```python
# 获取用于查看浏览器 UI 的 VNC URL
vnc_url = sandbox.get_vnc_url(record=True)
print(f"Connect with noVNC: https://novnc.com/noVNC/vnc.html")
print(f"VNC WebSocket: {vnc_url}")
```

#### 获取 CDP URL

```python
# 获取用于编程控制的 CDP URL
cdp_url = sandbox.get_cdp_url(record=True)
print(f"CDP WebSocket: {cdp_url}")

# 与 pyppeteer 或其他 CDP 客户端一起使用
import asyncio
from pyppeteer import connect

async def use_cdp():
    browser = await connect(browserWSEndpoint=cdp_url)
    page = await browser.newPage()
    await page.goto('https://www.example.com')
    await page.screenshot({'path': 'example.png'})
    await browser.disconnect()

asyncio.run(use_cdp())
```

### 录制管理

录制和下载浏览器会话。

#### 列出录制

```python
recordings = sandbox.list_recordings()
print(f"Total recordings: {len(recordings['recordings'])}")

for rec in recordings["recordings"]:
    print(f"File: {rec['filename']}, Size: {rec['size']}")
```

#### 下载录制

```python
filename = "vnc_global_20251126_111648_seg001.mkv"
result = sandbox.download_recording(
    filename=filename,
    save_path=f"./{filename}"
)
print(f"Downloaded: {result['saved_path']}, Size: {result['size']} bytes")
```

#### 删除录制

```python
sandbox.delete_recording(filename="recording.mkv")
```

## All-in-One Sandbox

**All-in-One (AIO) Sandbox** 将 **Code Interpreter** 和 **Browser** 沙箱的功能结合到一个统一的环境中。这使您可以在同一个沙箱实例中执行代码、管理文件和自动化浏览器。

### 为什么选择 All-in-One？

- **统一环境**：在同一个沙箱中运行 Python 代码和浏览器自动化
- **无缝集成**：在代码执行和浏览器任务之间共享文件和数据
- **资源效率**：单个沙箱实例即可完成复杂工作流
- **简化管理**：一个模板、一个沙箱、全部功能

### All-in-One 快速开始

```python
import time
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType

# 创建 All-in-One 模板
template_name = f"my-aio-template-{time.strftime('%Y%m%d%H%M%S')}"
template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.AIO,  # All-in-One 类型
        # 默认值：4 核 CPU，8GB 内存，10GB 磁盘
    )
)

print(f"All-in-One template created: {template.template_name}")

# 创建并使用 All-in-One 沙箱
with Sandbox.create(
    template_type=TemplateType.AIO,
    template_name=template_name,
    sandbox_idle_timeout_seconds=600,
) as sandbox:
    print(f"Sandbox created: {sandbox.sandbox_id}")
    
    # ===== Code Interpreter 功能 =====
    # 执行 Python 代码
    result = sandbox.context.execute(code="print('Hello from AIO!')")
    print(f"Code result: {result}")
    
    # 文件操作
    sandbox.file.write(path="/tmp/data.txt", content="Hello World")
    content = sandbox.file.read(path="/tmp/data.txt")
    print(f"File content: {content}")
    
    # ===== Browser 功能 =====
    # 获取浏览器访问 URL
    print(f"VNC URL: {sandbox.get_vnc_url()}")
    print(f"CDP URL: {sandbox.get_cdp_url()}")
    
    # 使用 Playwright 进行浏览器自动化
    with sandbox.sync_playwright(record=True) as playwright:
        playwright.new_page().goto("https://www.example.com")
        title = playwright.title()
        print(f"Page title: {title}")
        playwright.screenshot(path="screenshot.png")

# 沙箱自动清理
Sandbox.delete_template(template_name)
```

### 功能特性

AIO 沙箱提供了 Code Interpreter 和 Browser 沙箱的所有功能：

| 功能 | 描述 | 访问方式 |
|------|------|----------|
| **代码执行** | 支持 context 管理的 Python 代码执行 | `sandbox.context.execute()` |
| **文件操作** | 读写文件 | `sandbox.file.read()`, `sandbox.file.write()` |
| **文件系统** | 列出、移动、删除、上传、下载文件 | `sandbox.file_system.*` |
| **进程管理** | 执行命令、管理进程 | `sandbox.process.*` |
| **Playwright** | Playwright 浏览器自动化 | `sandbox.sync_playwright()`, `sandbox.async_playwright()` |
| **VNC 访问** | 实时浏览器视图 | `sandbox.get_vnc_url()` |
| **CDP 访问** | Chrome DevTools Protocol | `sandbox.get_cdp_url()` |
| **录制** | 录制浏览器会话 | `sandbox.list_recordings()`, `sandbox.download_recording()` |

### 默认资源

All-in-One 沙箱使用更高的默认资源配置以支持浏览器和代码执行：

| 资源 | 默认值 |
|------|--------|
| CPU | 4 核 |
| 内存 | 8192 MB (8GB) |
| 磁盘大小 | 10240 MB (10GB) |

### 使用示例

#### 示例 1：Web 抓取与数据处理

```python
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType
import time

template_name = f"scrape-and-process-{time.strftime('%Y%m%d%H%M%S')}"

# 创建 AIO 模板
template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.AIO,
    )
)

with Sandbox.create(
    template_type=TemplateType.AIO,
    template_name=template_name,
) as sandbox:
    # 步骤 1：使用浏览器抓取数据
    with sandbox.sync_playwright(record=True) as playwright:
        playwright.new_page().goto("https://news.ycombinator.com")
        playwright.wait_for_selector(".titleline")
        
        # 使用 JavaScript 提取标题
        titles = playwright.evaluate("""
            Array.from(document.querySelectorAll('.titleline a'))
                .slice(0, 10)
                .map(a => a.textContent)
        """)
        
        # 将抓取的数据保存到文件
        sandbox.file.write(
            path="/tmp/scraped_titles.txt",
            content="\\n".join(titles)
        )
    
    # 步骤 2：使用 Python 处理数据
    code = '''
import json

# 读取抓取的数据
with open("/tmp/scraped_titles.txt", "r") as f:
    titles = f.read().strip().split("\\n")

# 处理：添加编号并计算统计信息
processed = {
    "total_count": len(titles),
    "titles": [{"index": i+1, "title": t} for i, t in enumerate(titles)],
    "avg_title_length": sum(len(t) for t in titles) / len(titles)
}

# 保存处理结果
with open("/tmp/processed_data.json", "w") as f:
    json.dump(processed, f, indent=2)

print(f"Processed {len(titles)} titles")
print(f"Average title length: {processed['avg_title_length']:.1f} chars")
'''
    
    result = sandbox.context.execute(code=code)
    print(result)
    
    # 步骤 3：下载处理后的数据
    sandbox.file_system.download(
        path="/tmp/processed_data.json",
        save_path="./processed_data.json"
    )

Sandbox.delete_template(template_name)
```

#### 示例 2：自动化测试与结果分析

```python
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType
import time

template_name = f"test-and-analyze-{time.strftime('%Y%m%d%H%M%S')}"

template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.AIO,
    )
)

with Sandbox.create(
    template_type=TemplateType.AIO,
    template_name=template_name,
) as sandbox:
    # 运行浏览器测试
    test_results = []
    
    with sandbox.sync_playwright(record=True) as playwright:
        # 测试 1：首页加载
        playwright.new_page().goto("https://example.com")
        title = playwright.title()
        test_results.append({
            "test": "homepage_loads",
            "passed": "Example Domain" in title,
            "details": f"Title: {title}"
        })
        
        # 测试 2：截图
        playwright.screenshot(path="/tmp/test_screenshot.png")
        test_results.append({
            "test": "screenshot_captured",
            "passed": True,
            "details": "Screenshot saved"
        })
    
    # 使用 Python 分析结果
    code = f'''
import json

results = {test_results}

passed = sum(1 for r in results if r["passed"])
failed = len(results) - passed

report = {{
    "total_tests": len(results),
    "passed": passed,
    "failed": failed,
    "pass_rate": f"{{passed/len(results)*100:.1f}}%",
    "details": results
}}

with open("/tmp/test_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"Test Results: {{passed}}/{{len(results)}} passed ({{report['pass_rate']}})")
'''
    
    result = sandbox.context.execute(code=code)
    print(result)

Sandbox.delete_template(template_name)
```

#### 示例 3：截图与图像分析

```python
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType
import time

template_name = f"screenshot-analysis-{time.strftime('%Y%m%d%H%M%S')}"

template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.AIO,
    )
)

with Sandbox.create(
    template_type=TemplateType.AIO,
    template_name=template_name,
) as sandbox:
    # 捕获截图
    with sandbox.sync_playwright() as playwright:
        playwright.new_page().goto("https://www.google.com")
        playwright.screenshot(path="/tmp/google.png", full_page=True)
    
    # 使用 Python 分析截图
    code = '''
from PIL import Image
import os

# 加载并分析截图
img = Image.open("/tmp/google.png")

analysis = {
    "filename": "google.png",
    "format": img.format,
    "mode": img.mode,
    "size": f"{img.width}x{img.height}",
    "file_size_kb": os.path.getsize("/tmp/google.png") / 1024
}

print(f"Image Analysis:")
print(f"  Size: {analysis['size']}")
print(f"  Format: {analysis['format']}")
print(f"  Mode: {analysis['mode']}")
print(f"  File size: {analysis['file_size_kb']:.1f} KB")
'''
    
    result = sandbox.context.execute(code=code)
    print(result)
    
    # 下载截图
    sandbox.file_system.download(
        path="/tmp/google.png",
        save_path="./google_screenshot.png"
    )

Sandbox.delete_template(template_name)
```

#### 异步示例

```python
import asyncio
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType

async def aio_workflow():
    template_name = "async-aio-example"
    
    # 创建模板
    template = await Sandbox.create_template_async(
        input=TemplateInput(
            template_name=template_name,
            template_type=TemplateType.AIO,
        )
    )
    
    # 创建并使用 AIO 沙箱
    async with await Sandbox.create_async(
        template_type=TemplateType.AIO,
        template_name=template_name,
    ) as sandbox:
        # 异步执行代码
        result = await sandbox.context.execute_async(
            code="print('Async AIO sandbox!')"
        )
        print(result)
        
        # 异步文件操作
        await sandbox.file.write_async(
            path="/tmp/async_test.txt",
            content="Async content"
        )
        
        content = await sandbox.file.read_async(path="/tmp/async_test.txt")
        print(f"File content: {content}")
        
        # 异步浏览器自动化
        async with sandbox.async_playwright(record=True) as playwright:
            await playwright.goto("https://example.com")
            title = await playwright.title()
            print(f"Page title: {title}")
    
    # 清理
    await Sandbox.delete_template_async(template_name)

asyncio.run(aio_workflow())
```

## 异步支持

所有操作都支持使用 `async`/`await` 进行异步执行：

```python
import asyncio

async def main():
    # 创建模板
    template = await Sandbox.create_template_async(
        input=TemplateInput(
            template_name="async-template",
            template_type=TemplateType.CODE_INTERPRETER,
        )
    )
    
    # 创建沙箱
    async with await Sandbox.create_async(
        template_type=TemplateType.CODE_INTERPRETER,
        template_name="async-template",
    ) as sandbox:
        # 执行代码
        result = await sandbox.context.execute_async(
            code="print('Async execution!')"
        )
        print(result)
        
        # 文件操作
        await sandbox.file.write_async(
            path="/tmp/test.txt",
            content="Async content"
        )
        
        content = await sandbox.file.read_async(path="/tmp/test.txt")
        print(content)
    
    # 清理
    await Sandbox.delete_template_async("async-template")

asyncio.run(main())
```

## 最佳实践

### 1. 使用 Context Managers

始终使用上下文管理器进行自动资源清理：

```python
# 推荐
with Sandbox.create(...) as sandbox:
    # 使用 sandbox
    pass
# 自动清理

# 避免
sandbox = Sandbox.create(...)
# ... 使用 sandbox ...
Sandbox.delete(sandbox.sandbox_id)  # 手动清理
```

### 2. 处理健康检查

在使用沙箱之前等待其就绪：

```python
sandbox = Sandbox.create(...)

# 等待就绪状态
import time
while sandbox.check_health()["status"] != "ok":
    time.sleep(1)

# 现在使用沙箱
```

### 3. 设置适当的超时时间

根据您的使用场景配置超时时间：

```python
template = Sandbox.create_template(
    input=TemplateInput(
        template_name="my-template",
        template_type=TemplateType.CODE_INTERPRETER,
        sandbox_idle_timeout_in_seconds=1800,  # 30 分钟
        sandbox_ttlin_seconds=3600,  # 最大生存期 1 小时
    )
)
```

### 4. 使用 Contexts 进行有状态执行

当您需要在多次代码执行之间维护状态时：

```python
with sandbox.context.create(language=CodeLanguage.PYTHON) as ctx:
    ctx.execute(code="import numpy as np")
    ctx.execute(code="arr = np.array([1, 2, 3])")
    result = ctx.execute(code="print(arr.sum())")  # 使用之前导入的 numpy
```

### 5. 清理资源

完成后始终清理模板和沙箱：

```python
try:
    # 创建和使用资源
    template = Sandbox.create_template(...)
    sandbox = Sandbox.create(...)
    # ... 使用 sandbox ...
finally:
    # 清理
    Sandbox.delete(sandbox.sandbox_id)
    Sandbox.delete_template(template.template_name)
```

### 6. 启用录制以进行调试

为浏览器会话启用录制以调试问题：

```python
with sandbox.sync_playwright(record=True) as playwright:
    # 您的浏览器自动化
    pass

# 下载录制进行分析
recordings = sandbox.list_recordings()
if recordings["recordings"]:
    sandbox.download_recording(
        recordings["recordings"][0]["filename"],
        "./debug-session.mkv"
    )
```

## 示例

### 示例 1：数据处理流水线

```python
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType
import time

template_name = f"data-pipeline-{time.strftime('%Y%m%d%H%M%S')}"

# 创建模板
template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.CODE_INTERPRETER,
    )
)

# 在沙箱中处理数据
with Sandbox.create(
    template_type=TemplateType.CODE_INTERPRETER,
    template_name=template_name,
) as sandbox:
    # 上传数据
    sandbox.file_system.upload(
        local_file_path="./data.csv",
        target_file_path="/tmp/data.csv"
    )
    
    # 处理数据
    code = """
import pandas as pd

df = pd.read_csv('/tmp/data.csv')
result = df.describe()
result.to_csv('/tmp/result.csv')
print('Processing complete')
"""
    
    result = sandbox.context.execute(code=code)
    print(result)
    
    # 下载结果
    sandbox.file_system.download(
        path="/tmp/result.csv",
        save_path="./result.csv"
    )

# 清理
Sandbox.delete_template(template_name)
```

### 示例 2：Web 抓取

```python
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType
import time

template_name = f"scraper-{time.strftime('%Y%m%d%H%M%S')}"

# 创建浏览器模板
template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.BROWSER,
    )
)

# 抓取网站
sandbox = Sandbox.create(
    template_type=TemplateType.BROWSER,
    template_name=template_name,
)

# 等待就绪
while sandbox.check_health()["status"] != "ok":
    time.sleep(1)

# 执行抓取
with sandbox.sync_playwright(record=True) as playwright:
    playwright.new_page().goto("https://news.ycombinator.com")
    
    # 提取数据
    playwright.wait_for_selector(".titleline")
    
    # 截图
    playwright.screenshot(path="hackernews.png", full_page=True)
    
    print("Scraping complete!")

# 清理
sandbox.delete()
Sandbox.delete_template(template_name)
```

### 示例 3：自动化测试

```python
import asyncio
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType

async def run_tests():
    template_name = "test-runner"
    
    # 创建模板
    template = await Sandbox.create_template_async(
        input=TemplateInput(
            template_name=template_name,
            template_type=TemplateType.BROWSER,
        )
    )
    
    # 运行测试
    async with await Sandbox.create_async(
        template_type=TemplateType.BROWSER,
        template_name=template_name,
    ) as sandbox:
        async with sandbox.async_playwright(record=True) as playwright:
            # 测试 1：首页加载
            await playwright.goto("https://example.com")
            assert "Example Domain" in await playwright.title()
            
            # 测试 2：导航工作正常
            await playwright.click("a")
            await playwright.wait_for_load_state()
            
            # 测试 3：截取证据截图
            await playwright.screenshot(path="test-evidence.png")
            
            print("All tests passed!")
        
        # 下载测试录制
        recordings = await sandbox.list_recordings_async()
        if recordings["recordings"]:
            await sandbox.download_recording_async(
                recordings["recordings"][0]["filename"],
                "./test-recording.mkv"
            )
    
    # 清理
    await Sandbox.delete_template_async(template_name)

asyncio.run(run_tests())
```


