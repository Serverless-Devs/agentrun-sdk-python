# AgentRun Sandbox SDK

The AgentRun Sandbox SDK provides a powerful and flexible way to create isolated environments for code execution and browser automation. This SDK supports two main sandbox types: **Code Interpreter** for executing code in various languages, and **Browser** for automated web interactions.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Code Interpreter Quick Start](#code-interpreter-quick-start)
  - [Browser Quick Start](#browser-quick-start)
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
- ✅ **File Management**: Upload, download, read, write, move files
- ✅ **Process Control**: Execute commands and manage processes
- ✅ **Session Recording**: Record browser sessions
- ✅ **Health Monitoring**: Built-in health checks
- ✅ **Async/Await**: Full asynchronous API support
- ✅ **Context Managers**: Automatic resource cleanup

## Installation

```bash
pip install agentrun
```

## Configurations

### Use Environment Variables

```bash
export AGENTRUN_ACCESS_KEY_ID=your_access_key_id
export AGENTRUN_ACCESS_KEY_SECRET=your_access_key_secret
export AGENTRUN_ACCOUNT_ID=your_account_id
```

## Quick Start

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

## Core Concepts

### Templates

Templates define the configuration for sandboxes. They specify:
- **Type**: Code Interpreter or Browser
- **Resources**: CPU, memory, disk size
- **Network**: Network mode and VPC configuration
- **Environment**: Environment variables and credentials
- **Timeouts**: Idle timeout and TTL

Templates are reusable and can be used to create multiple sandboxes with the same configuration.

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
from agentrun.sandbox import Sandbox, TemplateInput, TemplateType, TemplateNetworkConfiguration, TemplateNetworkMode

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




