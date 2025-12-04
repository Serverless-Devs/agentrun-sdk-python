# AgentRun Sandbox SDK

The AgentRun Sandbox SDK provides a powerful and flexible way to create isolated environments for code execution and browser automation. This SDK supports three sandbox types:

- **Code Interpreter** - For executing Python code in isolated environments
- **Browser** - For automated web interactions with Playwright, VNC, and CDP support
- **All-in-One (AIO)** - Combines both Code Interpreter and Browser capabilities in a single sandbox

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Code Interpreter Quick Start](#code-interpreter-quick-start)
  - [Browser Quick Start](#browser-quick-start)
  - [All-in-One Quick Start](#all-in-one-quick-start-1)
- [Core Concepts](#core-concepts)
  - [Templates](#templates)
  - [Sandboxes](#sandboxes)
- [Template Operations](#template-operations)
- [Sandbox Operations](#sandbox-operations)
- [Code Interpreter](#code-interpreter)
  - [Context Management](#context-management)
  - [File Operations](#file-operations)
  - [File System Operations](#file-system-operations)
  - [Process Management](#process-management)
- [Browser Sandbox](#browser-sandbox)
  - [Playwright Integration](#playwright-integration)
  - [VNC and CDP Access](#vnc-and-cdp-access)
  - [Recording Management](#recording-management)
- [All-in-One Sandbox](#all-in-one-sandbox)
  - [Quick Start](#all-in-one-quick-start)
  - [Features](#all-in-one-features)
  - [Usage Examples](#all-in-one-usage-examples)
- [Async Support](#async-support)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Overview

The Sandbox SDK enables you to:

- **Execute Code Safely**: Run Python code in isolated environments
- **Automate Browsers**: Control web browsers for testing and automation
- **Manage Files**: Upload, download, and manipulate files within sandboxes
- **Monitor Processes**: Track and manage running processes
- **Record Sessions**: Capture browser sessions for debugging and analysis

### Supported Features

- ✅ **Code Interpreter**: Python execution with context management
- ✅ **Browser Automation**: Playwright integration with CDP and VNC support
- ✅ **All-in-One Sandbox**: Combined code interpreter + browser in single environment
- ✅ **File Management**: Upload, download, read, write, move files
- ✅ **Process Control**: Execute commands and manage processes
- ✅ **Session Recording**: Record browser sessions
- ✅ **Health Monitoring**: Built-in health checks
- ✅ **Async/Await**: Full asynchronous API support
- ✅ **Context Managers**: Automatic resource cleanup

## Installation

```bash
pip install agentrun-sdk
```

## Configurations

### Use Environment Variables

```bash
export AGENTRUN_ACCESS_KEY_ID=your_access_key_id
export AGENTRUN_ACCESS_KEY_SECRET=your_access_key_secret
export AGENTRUN_ACCOUNT_ID=your_account_id
```

## Quick Start

### Sandbox Types Overview

| Type | Class | Description |
|------|-------|-------------|
| `TemplateType.CODE_INTERPRETER` | `CodeInterpreterSandbox` | Python code execution, file management, process control |
| `TemplateType.BROWSER` | `BrowserSandbox` | Browser automation with Playwright, VNC, CDP |
| `TemplateType.AIO` | `AioSandbox` | **All-in-One**: Combines both Code Interpreter and Browser capabilities |

### Code Interpreter Quick Start

```python
import time
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType
from agentrun.sandbox.model import CodeLanguage

# Create a template
template_name = f"my-code-template-{time.strftime('%Y%m%d%H%M%S')}"
template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.CODE_INTERPRETER,
    )
)

print(f"Template created: {template.template_name}")

# Create and use a sandbox with context manager
with Sandbox.create(
    template_type=TemplateType.CODE_INTERPRETER,
    template_name=template_name,
    sandbox_idle_timeout_seconds=600,
) as sandbox:
    print(f"Sandbox created: {sandbox.sandbox_id}")
    
    # Execute code
    result = sandbox.context.execute(code="print('Hello, AgentRun!')")
    print(f"Execution result: {result}")
    
    # Create a file
    sandbox.file.write(path="/tmp/test.txt", content="Hello World")
    
    # Read the file
    content = sandbox.file.read(path="/tmp/test.txt")
    print(f"File content: {content}")

# Sandbox is automatically cleaned up after exiting the context

# Clean up template
Sandbox.delete_template(template_name)
```

### Browser Quick Start

```python
import time
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType

# Create a browser template
template_name = f"my-browser-template-{time.strftime('%Y%m%d%H%M%S')}"
template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.BROWSER,
    )
)

print(f"Browser template created: {template.template_name}")

# Create a browser sandbox
sandbox = Sandbox.create(
    template_type=TemplateType.BROWSER,
    template_name=template_name,
)

print(f"Browser sandbox created: {sandbox.sandbox_id}")
print(f"VNC URL: {sandbox.get_vnc_url()}")
print(f"CDP URL: {sandbox.get_cdp_url()}")

# Wait for browser to be ready
while sandbox.check_health()["status"] != "ok":
    time.sleep(1)

# Use Playwright to control the browser
with sandbox.sync_playwright(record=True) as playwright:
    playwright.new_page().goto("https://www.example.com")
    
    title = playwright.title()
    print(f"Page title: {title}")
    
    # Take a screenshot
    playwright.screenshot(path="screenshot.png")
    print("Screenshot saved")

# List recordings
recordings = sandbox.list_recordings()
print(f"Recordings: {recordings}")

# Download recording
if recordings.get("recordings"):
    filename = recordings["recordings"][0]["filename"]
    sandbox.download_recording(filename, f"./{filename}")

# Clean up
sandbox.delete()
Sandbox.delete_template(template_name)
```

### All-in-One Quick Start

```python
import time
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType

# Create an All-in-One template
template_name = f"my-aio-template-{time.strftime('%Y%m%d%H%M%S')}"
template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.AIO,  # All-in-One type
    )
)

print(f"All-in-One template created: {template.template_name}")

# Create an All-in-One sandbox with context manager
with Sandbox.create(
    template_type=TemplateType.AIO,
    template_name=template_name,
) as sandbox:
    print(f"All-in-One sandbox created: {sandbox.sandbox_id}")
    
    # Code Interpreter features
    result = sandbox.context.execute(code="print('Hello from AIO!')")
    print(f"Code execution result: {result}")
    
    sandbox.file.write(path="/tmp/test.txt", content="Hello World")
    content = sandbox.file.read(path="/tmp/test.txt")
    print(f"File content: {content}")
    
    # Browser features
    print(f"VNC URL: {sandbox.get_vnc_url()}")
    
    with sandbox.sync_playwright(record=True) as playwright:
        playwright.new_page().goto("https://www.example.com")
        title = playwright.title()
        print(f"Page title: {title}")

# Sandbox is automatically cleaned up after exiting the context

# Clean up template
Sandbox.delete_template(template_name)
```

## Core Concepts

### Templates

Templates define the configuration for sandboxes. They specify:
- **Type**: Code Interpreter, Browser, or All-in-One (AIO)
- **Resources**: CPU, memory, disk size
- **Network**: Network mode and VPC configuration
- **Environment**: Environment variables and credentials
- **Timeouts**: Idle timeout and TTL

Templates are reusable and can be used to create multiple sandboxes with the same configuration.

#### Template Types

| Type | Description | Use Case |
|------|-------------|----------|
| `TemplateType.CODE_INTERPRETER` | Python code execution environment | Data processing, scripting, computation |
| `TemplateType.BROWSER` | Browser automation environment | Web scraping, testing, automation |
| `TemplateType.AIO` | All-in-One combined environment | Complex workflows requiring both code and browser |

### Sandboxes

Sandboxes are isolated runtime environments created from templates. Each sandbox:
- Runs independently with its own resources
- Has a unique ID for identification
- Can be created, connected to, stopped, and deleted
- Automatically times out after idle period
- Supports health monitoring

## Template Operations

### Create Template

Create a new sandbox template:

```python
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType
from agentrun.sandbox import TemplateNetworkConfiguration, TemplateOSSPermission
from agentrun.sandbox import TemplateNetworkMode, TemplateLogConfiguration, TemplateOssConfiguration

# Basic template
template = Sandbox.create_template(
    input=TemplateInput(
        template_name="my-template",
        template_type=TemplateType.CODE_INTERPRETER,
    )
)

# Advanced template with custom configuration
template = Sandbox.create_template(
    input=TemplateInput(
        template_name="my-advanced-template",
        template_type=TemplateType.CODE_INTERPRETER,
        cpu=4.0,  # 4 CPU cores
        memory=8192,  # 8GB RAM
        disk_size=10240,  # 10GB disk
        sandbox_idle_timeout_in_seconds=7200,  # 2 hours
        environment_variables={"MY_VAR": "value"},
        network_configuration=TemplateNetworkConfiguration(
            network_mode=TemplateNetworkMode.PUBLIC
        ),
        log_configuration=TemplateLogConfiguration(
            project="my-project",
            logstore="my-logstore",
        ),
        oss_configuration=[TemplateOssConfiguration(
            bucket_name="my-bucket",
            mount_point="/mnt/oss",
            permission=TemplateOSSPermission.READ_WRITE,
            region="cn-hangzhou",
            prefix="my-object",
        )],
    )
)

# All-in-One template (combines Code Interpreter + Browser)
# Default: 4 CPU cores, 8GB RAM, 10GB disk
aio_template = Sandbox.create_template(
    input=TemplateInput(
        template_name="my-aio-template",
        template_type=TemplateType.AIO,
        # cpu, memory, disk_size use higher defaults for AIO
    )
)
```

### Get Template

Retrieve an existing template:

```python
template = Sandbox.get_template("my-template")
print(f"Template ID: {template.template_id}")
print(f"Template Type: {template.template_type}")
print(f"Status: {template.status}")
```

### Update Template

Update an existing template:

```python
from agentrun.sandbox import TemplateInput

updated_template = Sandbox.update_template(
    template_name="my-template",
    input=TemplateInput(
        template_name="my-template",
        template_type=TemplateType.CODE_INTERPRETER,
        cpu=8.0,  # Increase CPU
        memory=16384,  # Increase memory
    )
)
```

### Delete Template

Delete a template:

```python
Sandbox.delete_template("my-template")
```

### List Templates

List all templates:

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

## Sandbox Operations

### Create Sandbox

Create a new sandbox from a template:

```python
# Using context manager (recommended)
with Sandbox.create(
    template_type=TemplateType.CODE_INTERPRETER,
    template_name="my-template",
    sandbox_idle_timeout_seconds=600,
) as sandbox:
    # Use the sandbox
    print(f"Sandbox ID: {sandbox.sandbox_id}")
    # Automatically cleaned up on exit

# Manual creation
sandbox = Sandbox.create(
    template_type=TemplateType.CODE_INTERPRETER,
    template_name="my-template",
)
```

### Connect to Existing Sandbox

Connect to a running sandbox:

```python
# Connect by ID (type auto-detected)
sandbox = Sandbox.connect(sandbox_id="your-sandbox-id")

# Connect with explicit type for better type hints
sandbox = Sandbox.connect(
    sandbox_id="your-sandbox-id",
    template_type=TemplateType.CODE_INTERPRETER
)

# Connect to a Browser sandbox
browser_sandbox = Sandbox.connect(
    sandbox_id="your-browser-sandbox-id",
    template_type=TemplateType.BROWSER
)

# Connect to an All-in-One sandbox
aio_sandbox = Sandbox.connect(
    sandbox_id="your-aio-sandbox-id",
    template_type=TemplateType.AIO
)
```

### Stop Sandbox

Stop a running sandbox:

```python
Sandbox.stop_by_id("sandbox-id")
sandbox.stop() # Instance method, equal to Sandbox.stop_by_id("sandbox-id")
```

### Delete Sandbox

Delete a sandbox:

```python
Sandbox.delete_by_id("sandbox-id")
sandbox.delete() # Instance method, equal to Sandbox.delete_by_id("sandbox-id")
```

### List Sandboxes

List all sandboxes:

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

### Health Check

Check if a sandbox is healthy:

```python
health = sandbox.check_health()
if health["status"] == "ok":
    print("Sandbox is ready!")
```

## Code Interpreter

The Code Interpreter sandbox allows you to execute code in a secure, isolated environment.

### Context Management

Contexts maintain execution state (variables, imports, etc.) across multiple code executions.

#### Create Context

```python
from agentrun.sandbox.model import CodeLanguage

# Create a context
with sandbox.context.create(language=CodeLanguage.PYTHON) as ctx:
    # Execute code in the context
    result = ctx.execute(code="x = 10")
    print(result)
    
    result = ctx.execute(code="print(x)")  # x is still available
    print(result)
    # Context is automatically cleaned up on exit
```

#### Execute Code

Execute code with or without a context:

```python
# Execute without context (stateless)
result = sandbox.context.execute(code="print('Hello World')")

# Execute in a specific context (stateful)
result = sandbox.context.execute(
    code="print(x + 5)",
    context_id="your-context-id",
    timeout=30
)
```

#### List Contexts

```python
contexts = sandbox.context.list()
for ctx in contexts:
    print(f"Context: {ctx['id']} - {ctx['language']}")
```

#### Get Context

```python
ctx = sandbox.context.get(context_id="your-context-id")
print(f"Context language: {ctx._language}")
```

#### Delete Context

```python
sandbox.context.delete(context_id="your-context-id")
```

### File Operations

Simple file read/write operations.

#### Write File

```python
sandbox.file.write(
    path="/tmp/data.txt",
    content="Hello, World!",
    mode="644",
    encoding="utf-8",
    create_dir=True  # Create parent directories if needed
)
```

#### Read File

```python
content = sandbox.file.read(path="/tmp/data.txt")
print(content)
```

### File System Operations

Advanced file system operations for managing files and directories.

#### List Directory

```python
# List root directory
files = sandbox.file_system.list(path="/")
print(files)

# List with depth control
files = sandbox.file_system.list(path="/tmp", depth=2)
```

#### Create Directory

```python
sandbox.file_system.mkdir(
    path="/tmp/my-dir",
    parents=True,  # Create parent directories
    mode="0755"
)
```

#### Move File/Directory

```python
sandbox.file_system.move(
    source="/tmp/old-file.txt",
    destination="/tmp/new-file.txt"
)
```

#### Remove File/Directory

```python
sandbox.file_system.remove(path="/tmp/my-dir")
```

#### Get File Stats

```python
stats = sandbox.file_system.stat(path="/tmp/data.txt")
print(f"Size: {stats['size']}")
print(f"Modified: {stats['mtime']}")
```

#### Upload File

```python
result = sandbox.file_system.upload(
    local_file_path="./local-file.txt",
    target_file_path="/tmp/remote-file.txt"
)
print(f"Uploaded: {result}")
```

#### Download File

```python
result = sandbox.file_system.download(
    path="/tmp/remote-file.txt",
    save_path="./downloaded-file.txt"
)
print(f"Downloaded: {result['saved_path']}, Size: {result['size']}")
```

### Process Management

Manage and monitor processes running in the sandbox.

#### Execute Command

```python
result = sandbox.process.cmd(
    command="ls -la",
    cwd="/tmp",
    timeout=30
)
print(f"Exit code: {result['exitCode']}")
print(f"Output: {result['stdout']}")
```

#### List Processes

```python
processes = sandbox.process.list()
for proc in processes:
    print(f"PID: {proc['pid']} - {proc['name']}")
```

#### Get Process

```python
proc = sandbox.process.get(pid="1234")
print(f"Process: {proc['name']} - Status: {proc['status']}")
```

#### Kill Process

```python
sandbox.process.kill(pid="1234")
```

## Browser Sandbox

The Browser sandbox provides automated browser control with Playwright integration.

### Playwright Integration

#### Synchronous Playwright

```python
with sandbox.sync_playwright(record=True) as playwright:
    # Create a new page
    playwright.new_page().goto("https://www.example.com")
    
    # Get page title
    title = playwright.title()
    print(f"Title: {title}")
    
    # Take screenshot
    playwright.screenshot(path="page.png")
    
    # Click elements
    playwright.click("button#submit")
    
    # Fill forms
    playwright.fill("input[name='username']", "myuser")
    
    # Get content
    content = playwright.content()
```

#### Asynchronous Playwright

```python
async with sandbox.async_playwright(record=True) as playwright:
    await playwright.goto("https://www.example.com")
    
    title = await playwright.title()
    print(f"Title: {title}")
    
    await playwright.screenshot(path="page.png")
```

### VNC and CDP Access

Access the browser directly via VNC or Chrome DevTools Protocol.

#### Get VNC URL

```python
# Get VNC URL for viewing the browser UI
vnc_url = sandbox.get_vnc_url(record=True)
print(f"Connect with noVNC: https://novnc.com/noVNC/vnc.html")
print(f"VNC WebSocket: {vnc_url}")
```

#### Get CDP URL

```python
# Get CDP URL for programmatic control
cdp_url = sandbox.get_cdp_url(record=True)
print(f"CDP WebSocket: {cdp_url}")

# Use with pyppeteer or other CDP clients
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

### Recording Management

Record and download browser sessions.

#### List Recordings

```python
recordings = sandbox.list_recordings()
print(f"Total recordings: {len(recordings['recordings'])}")

for rec in recordings["recordings"]:
    print(f"File: {rec['filename']}, Size: {rec['size']}")
```

#### Download Recording

```python
filename = "vnc_global_20251126_111648_seg001.mkv"
result = sandbox.download_recording(
    filename=filename,
    save_path=f"./{filename}"
)
print(f"Downloaded: {result['saved_path']}, Size: {result['size']} bytes")
```

#### Delete Recording

```python
sandbox.delete_recording(filename="recording.mkv")
```

## All-in-One Sandbox

The **All-in-One (AIO) Sandbox** combines the capabilities of both **Code Interpreter** and **Browser** sandboxes into a single unified environment. This allows you to execute code, manage files, and automate browsers all within the same sandbox instance.

### Why All-in-One?

- **Unified Environment**: Run Python code and browser automation in the same sandbox
- **Seamless Integration**: Share files and data between code execution and browser tasks
- **Resource Efficiency**: Single sandbox instance for complex workflows
- **Simplified Management**: One template, one sandbox, all capabilities

### All-in-One Quick Start

```python
import time
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType

# Create an All-in-One template
template_name = f"my-aio-template-{time.strftime('%Y%m%d%H%M%S')}"
template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.AIO,  # All-in-One type
        # Default: 4 CPU cores, 8GB RAM, 10GB disk
    )
)

print(f"All-in-One template created: {template.template_name}")

# Create and use an All-in-One sandbox
with Sandbox.create(
    template_type=TemplateType.AIO,
    template_name=template_name,
    sandbox_idle_timeout_seconds=600,
) as sandbox:
    print(f"Sandbox created: {sandbox.sandbox_id}")
    
    # ===== Code Interpreter Features =====
    # Execute Python code
    result = sandbox.context.execute(code="print('Hello from AIO!')")
    print(f"Code result: {result}")
    
    # File operations
    sandbox.file.write(path="/tmp/data.txt", content="Hello World")
    content = sandbox.file.read(path="/tmp/data.txt")
    print(f"File content: {content}")
    
    # ===== Browser Features =====
    # Get browser access URLs
    print(f"VNC URL: {sandbox.get_vnc_url()}")
    print(f"CDP URL: {sandbox.get_cdp_url()}")
    
    # Use Playwright for browser automation
    with sandbox.sync_playwright(record=True) as playwright:
        playwright.new_page().goto("https://www.example.com")
        title = playwright.title()
        print(f"Page title: {title}")
        playwright.screenshot(path="screenshot.png")

# Sandbox is automatically cleaned up
Sandbox.delete_template(template_name)
```

### All-in-One Features

The AIO sandbox provides access to all features from both Code Interpreter and Browser sandboxes:

| Feature | Description | Access |
|---------|-------------|--------|
| **Code Execution** | Execute Python code with context management | `sandbox.context.execute()` |
| **File Operations** | Read/write files | `sandbox.file.read()`, `sandbox.file.write()` |
| **File System** | List, move, remove, upload, download files | `sandbox.file_system.*` |
| **Process Management** | Execute commands, manage processes | `sandbox.process.*` |
| **Playwright** | Browser automation with Playwright | `sandbox.sync_playwright()`, `sandbox.async_playwright()` |
| **VNC Access** | Live browser view | `sandbox.get_vnc_url()` |
| **CDP Access** | Chrome DevTools Protocol | `sandbox.get_cdp_url()` |
| **Recording** | Record browser sessions | `sandbox.list_recordings()`, `sandbox.download_recording()` |

### Default Resources

The All-in-One sandbox uses higher default resources to support both browser and code execution:

| Resource | Default Value |
|----------|---------------|
| CPU | 4 cores |
| Memory | 8192 MB (8GB) |
| Disk Size | 10240 MB (10GB) |

### All-in-One Usage Examples

#### Example 1: Web Scraping with Data Processing

```python
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType
import time

template_name = f"scrape-and-process-{time.strftime('%Y%m%d%H%M%S')}"

# Create AIO template
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
    # Step 1: Scrape data using browser
    with sandbox.sync_playwright(record=True) as playwright:
        playwright.new_page().goto("https://news.ycombinator.com")
        playwright.wait_for_selector(".titleline")
        
        # Extract titles using JavaScript
        titles = playwright.evaluate("""
            Array.from(document.querySelectorAll('.titleline a'))
                .slice(0, 10)
                .map(a => a.textContent)
        """)
        
        # Save scraped data to file
        sandbox.file.write(
            path="/tmp/scraped_titles.txt",
            content="\\n".join(titles)
        )
    
    # Step 2: Process data using Python
    code = '''
import json

# Read scraped data
with open("/tmp/scraped_titles.txt", "r") as f:
    titles = f.read().strip().split("\\n")

# Process: add numbering and calculate stats
processed = {
    "total_count": len(titles),
    "titles": [{"index": i+1, "title": t} for i, t in enumerate(titles)],
    "avg_title_length": sum(len(t) for t in titles) / len(titles)
}

# Save processed result
with open("/tmp/processed_data.json", "w") as f:
    json.dump(processed, f, indent=2)

print(f"Processed {len(titles)} titles")
print(f"Average title length: {processed['avg_title_length']:.1f} chars")
'''
    
    result = sandbox.context.execute(code=code)
    print(result)
    
    # Step 3: Download processed data
    sandbox.file_system.download(
        path="/tmp/processed_data.json",
        save_path="./processed_data.json"
    )

Sandbox.delete_template(template_name)
```

#### Example 2: Automated Testing with Result Analysis

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
    # Run browser tests
    test_results = []
    
    with sandbox.sync_playwright(record=True) as playwright:
        # Test 1: Homepage loads
        playwright.new_page().goto("https://example.com")
        title = playwright.title()
        test_results.append({
            "test": "homepage_loads",
            "passed": "Example Domain" in title,
            "details": f"Title: {title}"
        })
        
        # Test 2: Take screenshot
        playwright.screenshot(path="/tmp/test_screenshot.png")
        test_results.append({
            "test": "screenshot_captured",
            "passed": True,
            "details": "Screenshot saved"
        })
    
    # Analyze results with Python
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

#### Example 3: Screenshot and Image Analysis

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
    # Capture screenshot
    with sandbox.sync_playwright() as playwright:
        playwright.new_page().goto("https://www.google.com")
        playwright.screenshot(path="/tmp/google.png", full_page=True)
    
    # Analyze screenshot with Python
    code = '''
from PIL import Image
import os

# Load and analyze the screenshot
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
    
    # Download the screenshot
    sandbox.file_system.download(
        path="/tmp/google.png",
        save_path="./google_screenshot.png"
    )

Sandbox.delete_template(template_name)
```

#### Async Example

```python
import asyncio
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType

async def aio_workflow():
    template_name = "async-aio-example"
    
    # Create template
    template = await Sandbox.create_template_async(
        input=TemplateInput(
            template_name=template_name,
            template_type=TemplateType.AIO,
        )
    )
    
    # Create and use AIO sandbox
    async with await Sandbox.create_async(
        template_type=TemplateType.AIO,
        template_name=template_name,
    ) as sandbox:
        # Execute code asynchronously
        result = await sandbox.context.execute_async(
            code="print('Async AIO sandbox!')"
        )
        print(result)
        
        # Async file operations
        await sandbox.file.write_async(
            path="/tmp/async_test.txt",
            content="Async content"
        )
        
        content = await sandbox.file.read_async(path="/tmp/async_test.txt")
        print(f"File content: {content}")
        
        # Async browser automation
        async with sandbox.async_playwright(record=True) as playwright:
            await playwright.goto("https://example.com")
            title = await playwright.title()
            print(f"Page title: {title}")
    
    # Clean up
    await Sandbox.delete_template_async(template_name)

asyncio.run(aio_workflow())
```

## Async Support

All operations support asynchronous execution with `async`/`await`:

```python
import asyncio

async def main():
    # Create template
    template = await Sandbox.create_template_async(
        input=TemplateInput(
            template_name="async-template",
            template_type=TemplateType.CODE_INTERPRETER,
        )
    )
    
    # Create sandbox
    async with await Sandbox.create_async(
        template_type=TemplateType.CODE_INTERPRETER,
        template_name="async-template",
    ) as sandbox:
        # Execute code
        result = await sandbox.context.execute_async(
            code="print('Async execution!')"
        )
        print(result)
        
        # File operations
        await sandbox.file.write_async(
            path="/tmp/test.txt",
            content="Async content"
        )
        
        content = await sandbox.file.read_async(path="/tmp/test.txt")
        print(content)
    
    # Clean up
    await Sandbox.delete_template_async("async-template")

asyncio.run(main())
```

## Best Practices

### 1. Use Context Managers

Always use context managers for automatic resource cleanup:

```python
# Good
with Sandbox.create(...) as sandbox:
    # Use sandbox
    pass
# Automatically cleaned up

# Avoid
sandbox = Sandbox.create(...)
# ... use sandbox ...
Sandbox.delete(sandbox.sandbox_id)  # Manual cleanup
```

### 2. Handle Health Checks

Wait for sandboxes to be ready before use:

```python
sandbox = Sandbox.create(...)

# Wait for ready state
import time
while sandbox.check_health()["status"] != "ok":
    time.sleep(1)

# Now use the sandbox
```

### 3. Set Appropriate Timeouts

Configure timeouts based on your use case:

```python
template = Sandbox.create_template(
    input=TemplateInput(
        template_name="my-template",
        template_type=TemplateType.CODE_INTERPRETER,
        sandbox_idle_timeout_in_seconds=1800,  # 30 minutes
        sandbox_ttlin_seconds=3600,  # 1 hour max lifetime
    )
)
```

### 4. Use Contexts for Stateful Execution

When you need to maintain state across multiple code executions:

```python
with sandbox.context.create(language=CodeLanguage.PYTHON) as ctx:
    ctx.execute(code="import numpy as np")
    ctx.execute(code="arr = np.array([1, 2, 3])")
    result = ctx.execute(code="print(arr.sum())")  # Uses previously imported numpy
```

### 5. Clean Up Resources

Always clean up templates and sandboxes when done:

```python
try:
    # Create and use resources
    template = Sandbox.create_template(...)
    sandbox = Sandbox.create(...)
    # ... use sandbox ...
finally:
    # Clean up
    Sandbox.delete(sandbox.sandbox_id)
    Sandbox.delete_template(template.template_name)
```

### 6. Enable Recording for Debugging

Enable recording for browser sessions to debug issues:

```python
with sandbox.sync_playwright(record=True) as playwright:
    # Your browser automation
    pass

# Download recording for analysis
recordings = sandbox.list_recordings()
if recordings["recordings"]:
    sandbox.download_recording(
        recordings["recordings"][0]["filename"],
        "./debug-session.mkv"
    )
```

## Examples

### Example 1: Data Processing Pipeline

```python
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType
import time

template_name = f"data-pipeline-{time.strftime('%Y%m%d%H%M%S')}"

# Create template
template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.CODE_INTERPRETER,
    )
)

# Process data in sandbox
with Sandbox.create(
    template_type=TemplateType.CODE_INTERPRETER,
    template_name=template_name,
) as sandbox:
    # Upload data
    sandbox.file_system.upload(
        local_file_path="./data.csv",
        target_file_path="/tmp/data.csv"
    )
    
    # Process data
    code = """
import pandas as pd

df = pd.read_csv('/tmp/data.csv')
result = df.describe()
result.to_csv('/tmp/result.csv')
print('Processing complete')
"""
    
    result = sandbox.context.execute(code=code)
    print(result)
    
    # Download result
    sandbox.file_system.download(
        path="/tmp/result.csv",
        save_path="./result.csv"
    )

# Clean up
Sandbox.delete_template(template_name)
```

### Example 2: Web Scraping

```python
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType
import time

template_name = f"scraper-{time.strftime('%Y%m%d%H%M%S')}"

# Create browser template
template = Sandbox.create_template(
    input=TemplateInput(
        template_name=template_name,
        template_type=TemplateType.BROWSER,
    )
)

# Scrape website
sandbox = Sandbox.create(
    template_type=TemplateType.BROWSER,
    template_name=template_name,
)

# Wait for ready
while sandbox.check_health()["status"] != "ok":
    time.sleep(1)

# Perform scraping
with sandbox.sync_playwright(record=True) as playwright:
    playwright.new_page().goto("https://news.ycombinator.com")
    
    # Extract data
    playwright.wait_for_selector(".titleline")
    
    # Take screenshot
    playwright.screenshot(path="hackernews.png", full_page=True)
    
    print("Scraping complete!")

# Clean up
sandbox.delete()
Sandbox.delete_template(template_name)
```

### Example 3: Automated Testing

```python
import asyncio
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType

async def run_tests():
    template_name = "test-runner"
    
    # Create template
    template = await Sandbox.create_template_async(
        input=TemplateInput(
            template_name=template_name,
            template_type=TemplateType.BROWSER,
        )
    )
    
    # Run tests
    async with await Sandbox.create_async(
        template_type=TemplateType.BROWSER,
        template_name=template_name,
    ) as sandbox:
        async with sandbox.async_playwright(record=True) as playwright:
            # Test 1: Homepage loads
            await playwright.goto("https://example.com")
            assert "Example Domain" in await playwright.title()
            
            # Test 2: Navigation works
            await playwright.click("a")
            await playwright.wait_for_load_state()
            
            # Test 3: Take evidence screenshot
            await playwright.screenshot(path="test-evidence.png")
            
            print("All tests passed!")
        
        # Download test recording
        recordings = await sandbox.list_recordings_async()
        if recordings["recordings"]:
            await sandbox.download_recording_async(
                recordings["recordings"][0]["filename"],
                "./test-recording.mkv"
            )
    
    # Clean up
    await Sandbox.delete_template_async(template_name)

asyncio.run(run_tests())
```




