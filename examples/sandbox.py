import os
import time

from dotenv import load_dotenv

from agentrun.sandbox import (
    AioSandbox,
    CodeInterpreterSandbox,
    CodeLanguage,
    Sandbox,
    TemplateInput,
    TemplateLogConfiguration,
    TemplateType,
)
from agentrun.utils.log import logger

# 加载同目录下的 .env 文件
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(env_path)


def code_interpreter_sync():
    """测试 Code Interpreter 同步方法"""
    logger.info("=" * 60)
    logger.info("开始测试 Code Interpreter - 同步方法")
    logger.info("=" * 60)

    template_name = f"sdk-create-template-{time.strftime('%Y%m%d%H%M%S')}"

    # 创建模板
    template = Sandbox.create_template(
        input=TemplateInput(
            template_name=template_name,
            template_type=TemplateType.CODE_INTERPRETER,
        )
    )

    logger.info(f"✓ 创建模板成功: {template.template_name}")
    logger.info(f"  模板 ID: {template.template_id}")
    logger.info(f"  模板状态: {template.status}")

    # 测试上下文管理器方式创建 Sandbox
    logger.info("\n--- 测试上下文管理器方式 ---")
    with Sandbox.create(
        template_type=TemplateType.CODE_INTERPRETER,
        template_name=template_name,
        sandbox_idle_timeout_seconds=600,
    ) as sb:
        logger.info(f"✓ 创建 Sandbox 成功: {sb.sandbox_id}")

        # 测试代码执行上下文
        with sb.context.create(language=CodeLanguage.PYTHON) as ctx:
            result = ctx.execute(code="print('hello world')")
            logger.info(f"✓ 执行代码结果: {result}")

            result = ctx.list()
            logger.info(f"✓ 上下文列表: {result}")

            result = ctx.get()
            logger.info(f"✓ 获取上下文详情: {result._context_id}")

    # 测试连接已存在的 Sandbox
    logger.info("\n--- 测试连接已存在的 Sandbox ---")
    sandbox_for_connect = Sandbox.create(
        template_type=TemplateType.CODE_INTERPRETER, template_name=template_name
    )
    logger.info(f"✓ 创建 Sandbox 成功: {sandbox_for_connect.sandbox_id}")

    while True:
        health_status = sandbox_for_connect.check_health()
        if health_status["status"] == "ok":
            break
        time.sleep(1)
        logger.info(f"✓ 健康检查结果: {health_status}")

    assert sandbox_for_connect.sandbox_id

    sbc = Sandbox.connect(sandbox_for_connect.sandbox_id)
    logger.info(f"✓ 连接 Sandbox 成功: {sbc.sandbox_id}")

    assert isinstance(sbc, CodeInterpreterSandbox)

    # 测试代码执行
    logger.info("\n--- 测试代码执行 ---")
    result = sbc.context.execute(code="print('hello world')")
    logger.info(f"✓ 代码执行结果: {result}")

    # 测试文件系统操作
    logger.info("\n--- 测试文件系统操作 ---")
    list_result = sbc.file_system.list(path="/")
    logger.info(f"✓ 根目录文件列表: {list_result}")

    create_result = sbc.file_system.mkdir(path="/test")
    logger.info(f"✓ 创建文件夹 /test: {create_result}")

    create_result = sbc.file_system.mkdir(path="/test-move")
    logger.info(f"✓ 创建文件夹 /test-move: {create_result}")

    # 测试上传下载
    logger.info("\n--- 测试上传下载 ---")
    # 创建临时测试文件
    test_file_path = "./temp_test_file.txt"
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write("这是一个测试文件，用于验证 Sandbox 文件上传下载功能。\n")
        f.write(
            "This is a test file for validating Sandbox file upload/download.\n"
        )
        f.write(f"创建时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    logger.info(f"✓ 创建临时测试文件: {test_file_path}")

    upload_result = sbc.file_system.upload(
        local_file_path=test_file_path,
        target_file_path="/test-move/test_file.txt",
    )
    logger.info(f"✓ 上传文件结果: {upload_result}")

    list_result = sbc.file_system.stat(path="/test-move/test_file.txt")
    logger.info(f"✓ 上传文件列表: {list_result}")

    download_path = "./downloaded_test_file.txt"
    download_result = sbc.file_system.download(
        path="/test-move/test_file.txt", save_path=download_path
    )
    logger.info(f"✓ 下载文件结果: {download_result}")

    # 验证下载的文件内容
    with open(download_path, "r", encoding="utf-8") as f:
        downloaded_content = f.read()
    logger.info(f"✓ 验证下载文件内容: {downloaded_content[:50]}...")

    # 测试文件读写
    logger.info("\n--- 测试文件读写 ---")
    write_result = sbc.file.write(path="/test/test.txt", content="hello world")
    logger.info(f"✓ 写入文件结果: {write_result}")

    read_result = sbc.file.read(path="/test/test.txt")
    logger.info(f"✓ 读取文件结果: {read_result}")

    # 测试文件移动
    logger.info("\n--- 测试文件移动 ---")
    move_result = sbc.file_system.move(
        source="/test/test.txt", destination="/test-move/test2.txt"
    )
    logger.info(f"✓ 移动文件结果: {move_result}")

    read_result = sbc.file.read(path="/test-move/test2.txt")
    logger.info(f"✓ 读取移动后的文件: {read_result}")

    # 测试文件详情
    logger.info("\n--- 测试文件详情 ---")
    stat_result = sbc.file_system.stat(path="/test-move")
    logger.info(f"✓ 文件详情: {stat_result}")

    # 测试删除文件
    logger.info("\n--- 测试删除文件 ---")
    delete_result = sbc.file_system.remove(path="/test-move")
    logger.info(f"✓ 删除文件夹结果: {delete_result}")

    # 测试进程操作
    logger.info("\n--- 测试进程操作 ---")
    process_result = sbc.process.list()
    logger.info(f"✓ 进程列表: {process_result}")

    process_result = sbc.process.cmd(command="ls", cwd="/")
    logger.info(f"✓ 进程执行结果: {process_result}")

    process_result = sbc.process.get(pid="1")
    logger.info(f"✓ 进程详情: {process_result}")

    # 清理资源
    # logger.info("\n--- 清理资源 ---")
    # sandbox_for_connect.stop()
    # logger.info(f"✓ 停止 Sandbox 成功")

    # 清理资源
    logger.info("\n--- 清理资源 ---")
    d = sandbox_for_connect.delete()
    logger.info(f"✓ 删除 Sandbox 成功, {d}")

    Sandbox.delete_template(template_name)
    logger.info(f"✓ 删除模板成功: {template_name}")

    # 清理临时测试文件
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        logger.info(f"✓ 删除临时测试文件: {test_file_path}")
    if os.path.exists(download_path):
        os.remove(download_path)
        logger.info(f"✓ 删除下载的测试文件: {download_path}")

    logger.info("\n✓ Code Interpreter 同步测试完成\n")


async def code_interpreter_async():
    """测试 Code Interpreter 异步方法"""
    logger.info("=" * 60)
    logger.info("开始测试 Code Interpreter - 异步方法")
    logger.info("=" * 60)

    template_name = f"sdk-create-template-async-{time.strftime('%Y%m%d%H%M%S')}"

    # 创建模板
    template = await Sandbox.create_template_async(
        input=TemplateInput(
            template_name=template_name,
            template_type=TemplateType.CODE_INTERPRETER,
        )
    )

    logger.info(f"✓ 创建模板成功 (async): {template.template_name}")
    logger.info(f"  模板 ID: {template.template_id}")
    logger.info(f"  模板状态: {template.status}")

    # 创建 Sandbox
    logger.info("\n--- 测试异步创建 Sandbox ---")

    with await Sandbox.create_async(
        template_type=TemplateType.CODE_INTERPRETER,
        template_name=template_name,
        sandbox_idle_timeout_seconds=600,
    ) as sb:
        logger.info(f"✓ 创建 Sandbox 成功 (async): {sb.sandbox_id}")
        result = await sb.context.execute_async(
            code="print('hello async world')"
        )
    logger.info(f"✓ 代码执行结果 (async): {result}")
    sb = await Sandbox.create_async(
        template_type=TemplateType.CODE_INTERPRETER,
        template_name=template_name,
        sandbox_idle_timeout_seconds=600,
    )
    logger.info(f"✓ 创建 Sandbox 成功 (async): {sb.sandbox_id}")
    # 测试文件系统操作
    logger.info("\n--- 测试异步文件系统操作 ---")
    list_result = await sb.file_system.list_async(path="/")
    logger.info(f"✓ 根目录文件列表 (async): {list_result}")

    create_result = await sb.file_system.mkdir_async(path="/test-async")
    logger.info(f"✓ 创建文件夹 (async): {create_result}")

    # 测试文件读写
    logger.info("\n--- 测试异步文件读写 ---")
    write_result = await sb.file.write_async(
        path="/test-async/test.txt", content="hello async world"
    )
    logger.info(f"✓ 写入文件 (async): {write_result}")

    read_result = await sb.file.read_async(path="/test-async/test.txt")
    logger.info(f"✓ 读取文件 (async): {read_result}")

    # 测试进程操作
    logger.info("\n--- 测试异步进程操作 ---")
    process_result = await sb.process.list_async()
    logger.info(f"✓ 进程列表 (async): {process_result}")

    process_result = await sb.process.cmd_async(command="ls", cwd="/")
    logger.info(f"✓ 进程执行 (async): {process_result}")

    # 清理资源
    logger.info("\n--- 清理资源 ---")
    await sb.delete_async()
    logger.info(f"✓ 删除 Sandbox 成功 (async)")

    await Sandbox.delete_template_async(template_name)
    logger.info(f"✓ 删除模板成功 (async): {template_name}")

    logger.info("\n✓ Code Interpreter 异步测试完成\n")


def browser_sync():
    """测试 Browser 同步方法"""
    logger.info("=" * 60)
    logger.info("开始测试 Browser - 同步方法")
    logger.info("=" * 60)

    browser_template_name = (
        f"sdk-create-browser-template-{time.strftime('%Y%m%d%H%M%S')}"
    )

    # 创建浏览器模板
    browser_template = Sandbox.create_template(
        input=TemplateInput(
            template_name=browser_template_name,
            template_type=TemplateType.BROWSER,
        )
    )
    logger.info(f"✓ 创建浏览器模板成功: {browser_template.template_name}")
    logger.info(f"  浏览器模板 ID: {browser_template.template_id}")
    logger.info(f"  浏览器模板状态: {browser_template.status}")

    # 创建浏览器 Sandbox
    logger.info("\n--- 创建浏览器 Sandbox ---")
    sb = Sandbox.create(
        template_type=TemplateType.BROWSER, template_name=browser_template_name
    )
    logger.info(f"✓ 创建浏览器 Sandbox 成功: {sb.sandbox_id}")
    logger.info(
        f"  You can use noVNC (https://novnc.com/noVNC/vnc.html) to view the"
        f" browser UI"
    )
    logger.info(f"  VNC websocket: {sb.get_vnc_url(record=True)}")
    logger.info(
        f"  You can use playwright, pyppeteer to control the browser via Chrome"
        f" Debug Protocol"
    )
    logger.info(f"  CDP websocket: {sb.get_cdp_url(record=True)}")

    # 测试健康检查
    logger.info("\n--- 测试健康检查 ---")
    while True:
        health_status = sb.check_health()
        if health_status["status"] == "ok":
            break
        time.sleep(1)

        logger.info(f"✓ 健康检查结果: {health_status}")

    # 测试 Playwright
    logger.info("\n--- 测试 Playwright 同步操作 ---")
    with sb.sync_playwright(record=True) as playwright:
        playwright.new_page().goto("https://www.aliyun.com")
        logger.info(f"✓ 导航到阿里云首页")

        title = playwright.title()
        logger.info(f"✓ 页面标题: {title}")

        screenshot_path = "baidu_sync.png"
        playwright.screenshot(path=screenshot_path)
        logger.info(f"✓ 截图已保存: {screenshot_path}")

    rec_list = sb.list_recordings()
    logger.info(f"✓ 录制列表: {rec_list}")
    first_filename = rec_list["recordings"][0]["filename"]
    logger.info(f"✓ 录制文件名: {first_filename}")
    sb.download_recording(first_filename, f"{first_filename}")

    # 清理资源
    logger.info("\n--- 清理资源 ---")
    sb.delete()
    logger.info(f"✓ 删除浏览器 Sandbox 成功")

    Sandbox.delete_template(browser_template_name)
    logger.info(f"✓ 删除浏览器模板成功: {browser_template_name}")

    logger.info("\n✓ Browser 同步测试完成\n")


async def browser_async():
    """测试 Browser 异步方法"""
    logger.info("=" * 60)
    logger.info("开始测试 Browser - 异步方法")
    logger.info("=" * 60)

    browser_template_name = (
        f"sdk-create-browser-template-async-{time.strftime('%Y%m%d%H%M%S')}"
    )

    # 创建浏览器模板
    browser_template = await Sandbox.create_template_async(
        input=TemplateInput(
            template_name=browser_template_name,
            template_type=TemplateType.BROWSER,
        )
    )
    logger.info(
        f"✓ 创建浏览器模板成功 (async): {browser_template.template_name}"
    )
    logger.info(f"  浏览器模板 ID: {browser_template.template_id}")
    logger.info(f"  浏览器模板状态: {browser_template.status}")

    # 创建浏览器 Sandbox
    logger.info("\n--- 创建浏览器 Sandbox (async) ---")
    sb = await Sandbox.create_async(
        template_type=TemplateType.BROWSER, template_name=browser_template_name
    )
    logger.info(f"✓ 创建浏览器 Sandbox 成功 (async): {sb.sandbox_id}")
    logger.info(f"  VNC websocket: {sb.get_vnc_url()}")
    logger.info(f"  CDP websocket: {sb.get_cdp_url()}")

    # 测试健康检查
    logger.info("\n--- 测试健康检查 (async) ---")
    while True:
        health_status = await sb.check_health_async()
        if health_status["status"] == "ok":
            break

        logger.info(f"✓ 健康检查结果: {health_status}")
    logger.info(f"✓ 健康检查结果 (async): {health_status}")

    # 测试 Playwright 异步
    logger.info("\n--- 测试 Playwright 异步操作 ---")
    async with sb.async_playwright() as playwright:
        await playwright.goto("https://www.baidu.com")
        logger.info(f"✓ 导航到百度首页 (async)")

        title = await playwright.title()
        logger.info(f"✓ 页面标题 (async): {title}")

        screenshot_path = "baidu_async.png"
        await playwright.screenshot(path=screenshot_path)
        logger.info(f"✓ 截图已保存 (async): {screenshot_path}")

    # 清理资源
    logger.info("\n--- 清理资源 ---")
    await sb.delete_async()
    logger.info(f"✓ 删除浏览器 Sandbox 成功 (async)")

    await Sandbox.delete_template_async(browser_template_name)
    logger.info(f"✓ 删除浏览器模板成功 (async): {browser_template_name}")

    logger.info("\n✓ Browser 异步测试完成\n")


def all_in_one_sync():
    """测试 All-in-One Sandbox 同步方法"""
    logger.info("=" * 60)
    logger.info("开始测试 All-in-One Sandbox - 同步方法")
    logger.info("=" * 60)

    template_name = f"sdk-create-aio-template-{time.strftime('%Y%m%d%H%M%S')}"

    # 创建 AIO 模板
    template = Sandbox.create_template(
        input=TemplateInput(
            template_name=template_name,
            template_type=TemplateType.AIO,
        )
    )
    logger.info(f"✓ 创建 AIO 模板成功: {template.template_name}")
    logger.info(f"  模板 ID: {template.template_id}")
    logger.info(f"  模板状态: {template.status}")

    # 创建 AIO Sandbox
    logger.info("\n--- 创建 All-in-One Sandbox ---")
    with Sandbox.create(
        template_type=TemplateType.AIO,
        template_name=template_name,
        sandbox_idle_timeout_seconds=600,
    ) as sb:
        assert isinstance(sb, AioSandbox)
        logger.info(f"✓ 创建 AIO Sandbox 成功: {sb.sandbox_id}")

        # ========== Browser 功能测试 ==========
        logger.info("\n--- Browser 功能测试 ---")
        logger.info(f"  VNC websocket: {sb.get_vnc_url()}")
        logger.info(f"  CDP websocket: {sb.get_cdp_url()}")

        # 测试 Playwright
        logger.info("\n--- 测试 Playwright 同步操作 ---")
        with sb.sync_playwright() as playwright:
            playwright.goto("https://www.aliyun.com")
            logger.info(f"✓ 导航到阿里巴巴首页")

            title = playwright.title()
            logger.info(f"✓ 页面标题: {title}")

            logger.info(f"✓ 页面列表: {playwright.list_pages()}")

            html = playwright.html_content()
            logger.info(f"✓ 页面 HTML: {html[:500]}...")

            screenshot_path = "aio_sync_screenshot.png"
            playwright.screenshot(path=screenshot_path)
            logger.info(f"✓ 截图已保存: {screenshot_path}")

        # 测试录制列表
        rec_list = sb.list_recordings()
        logger.info(f"✓ 录制列表: {rec_list}")

        # ========== Code Interpreter 功能测试 ==========
        logger.info("\n--- Code Interpreter 功能测试 ---")

        # 测试代码执行
        logger.info("\n--- 测试代码执行 ---")
        with sb.context.create(language=CodeLanguage.PYTHON) as ctx:
            result = ctx.execute(code="print('Hello from All-in-One Sandbox!')")
            logger.info(f"✓ 代码执行结果: {result}")

            result = ctx.execute(code="import sys; print(sys.version)")
            logger.info(f"✓ Python 版本: {result}")

        # 测试文件系统操作
        logger.info("\n--- 测试文件系统操作 ---")

        create_result = sb.file_system.mkdir(path="/home/user/aio-test")
        logger.info(f"✓ 创建文件夹 /aio-test: {create_result}")

        logger.info("\n--- 测试文件系统操作 ---")
        list_result = sb.file_system.list(path="/home/user")
        logger.info(f"✓ 根目录文件列表: {list_result}")

        create_result = sb.file_system.mkdir(path="/home/user/test")
        logger.info(f"✓ 创建文件夹 /test: {create_result}")

        create_result = sb.file_system.mkdir(path="/home/user/test-move")
        logger.info(f"✓ 创建文件夹 /test-move: {create_result}")

        # 测试上传下载
        logger.info("\n--- 测试上传下载 ---")
        # 创建临时测试文件
        test_file_path = "./temp_test_file.txt"
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write("这是一个测试文件，用于验证 Sandbox 文件上传下载功能。\n")
            f.write(
                "This is a test file for validating Sandbox file"
                " upload/download.\n"
            )
            f.write(f"创建时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logger.info(f"✓ 创建临时测试文件: {test_file_path}")

        upload_result = sb.file_system.upload(
            local_file_path=test_file_path,
            target_file_path="/home/user/test-move/test_file.txt",
        )
        logger.info(f"✓ 上传文件结果: {upload_result}")

        list_result = sb.file_system.stat(
            path="/home/user/test-move/test_file.txt"
        )
        logger.info(f"✓ 上传文件列表: {list_result}")

        download_path = "./downloaded_test_file.txt"
        download_result = sb.file_system.download(
            path="/home/user/test-move/test_file.txt", save_path=download_path
        )
        logger.info(f"✓ 下载文件结果: {download_result}")

        # 验证下载的文件内容
        with open(download_path, "r", encoding="utf-8") as f:
            downloaded_content = f.read()
        logger.info(f"✓ 验证下载文件内容: {downloaded_content[:500]}...")

        # 测试文件读写
        logger.info("\n--- 测试文件读写 ---")
        write_result = sb.file.write(
            path="/home/user/test/test.txt", content="hello world"
        )
        logger.info(f"✓ 写入文件结果: {write_result}")

        read_result = sb.file.read(path="/home/user/test/test.txt")
        logger.info(f"✓ 读取文件结果: {read_result}")

        # 测试文件移动
        logger.info("\n--- 测试文件移动 ---")
        move_result = sb.file_system.move(
            source="/home/user/test/test.txt",
            destination="/home/user/test-move/test2.txt",
        )
        logger.info(f"✓ 移动文件结果: {move_result}")

        read_result = sb.file.read(path="/home/user/test-move/test2.txt")
        logger.info(f"✓ 读取移动后的文件: {read_result}")

        # 测试文件详情
        logger.info("\n--- 测试文件详情 ---")
        stat_result = sb.file_system.stat(path="/home/user/test-move")
        logger.info(f"✓ 文件详情: {stat_result}")

        # 测试删除文件
        logger.info("\n--- 测试删除文件 ---")
        delete_result = sb.file_system.remove(path="/home/user/test-move")
        logger.info(f"✓ 删除文件夹结果: {delete_result}")

        # 测试进程操作
        logger.info("\n--- 测试进程操作 ---")
        process_result = sb.process.list()
        logger.info(f"✓ 进程列表: {process_result}")

        process_result = sb.process.cmd(command="ls", cwd="/")
        logger.info(f"✓ 进程执行结果: {process_result}")

        process_result = sb.process.get(pid="1")
        logger.info(f"✓ 进程详情: {process_result}")

        # 测试文件读写
        logger.info("\n--- 测试文件读写 ---")
        write_result = sb.file.write(
            path="/home/user/aio-test/hello.txt",
            content="Hello from All-in-One Sandbox!",
        )
        logger.info(f"✓ 写入文件结果: {write_result}")

        read_result = sb.file.read(path="/home/user/aio-test/hello.txt")
        logger.info(f"✓ 读取文件结果: {read_result}")

        # 测试进程操作
        logger.info("\n--- 测试进程操作 ---")
        process_result = sb.process.list()
        logger.info(f"✓ 进程列表: {process_result}")

        process_result = sb.process.cmd(
            command="ls -la /home/user/aio-test", cwd="/"
        )
        logger.info(f"✓ 命令执行结果: {process_result}")

        # 清理测试目录
        sb.file_system.remove(path="/home/user/aio-test")
        logger.info(f"✓ 删除测试目录 /home/user/aio-test")

    # 清理资源
    logger.info("\n--- 清理资源 ---")
    Sandbox.delete_template(template_name)
    logger.info(f"✓ 删除模板成功: {template_name}")

    # 清理截图文件
    if os.path.exists("aio_sync_screenshot.png"):
        os.remove("aio_sync_screenshot.png")
        logger.info(f"✓ 删除截图文件")

    logger.info("\n✓ All-in-One Sandbox 同步测试完成\n")


async def all_in_one_async():
    """测试 All-in-One Sandbox 异步方法"""
    import asyncio

    logger.info("=" * 60)
    logger.info("开始测试 All-in-One Sandbox - 异步方法")
    logger.info("=" * 60)

    template_name = (
        f"sdk-create-aio-template-async-{time.strftime('%Y%m%d%H%M%S')}"
    )

    # 创建 AIO 模板
    template = await Sandbox.create_template_async(
        input=TemplateInput(
            template_name=template_name,
            template_type=TemplateType.AIO,
        )
    )
    logger.info(f"✓ 创建 AIO 模板成功 (async): {template.template_name}")
    logger.info(f"  模板 ID: {template.template_id}")
    logger.info(f"  模板状态: {template.status}")

    # 创建 AIO Sandbox
    logger.info("\n--- 创建 All-in-One Sandbox (async) ---")
    async with await Sandbox.create_async(
        template_type=TemplateType.AIO,
        template_name=template_name,
        sandbox_idle_timeout_seconds=600,
    ) as sb:
        assert isinstance(sb, AioSandbox)
        logger.info(f"✓ 创建 AIO Sandbox 成功 (async): {sb.sandbox_id}")

        # ========== Browser 功能测试 ==========
        logger.info("\n--- Browser 功能测试 (async) ---")
        logger.info(f"  VNC websocket: {sb.get_vnc_url()}")
        logger.info(f"  CDP websocket: {sb.get_cdp_url()}")

        # 测试 Playwright 异步
        logger.info("\n--- 测试 Playwright 异步操作 ---")
        async with sb.async_playwright(record=True) as playwright:
            await playwright.goto("https://www.baidu.com")
            logger.info(f"✓ 导航到百度首页 (async)")

            title = await playwright.title()
            logger.info(f"✓ 页面标题 (async): {title}")

            screenshot_path = "aio_async_screenshot.png"
            await playwright.screenshot(path=screenshot_path)
            logger.info(f"✓ 截图已保存 (async): {screenshot_path}")

        # 测试录制列表
        rec_list = await sb.list_recordings_async()
        logger.info(f"✓ 录制列表 (async): {rec_list}")

        # ========== Code Interpreter 功能测试 ==========
        logger.info("\n--- Code Interpreter 功能测试 (async) ---")

        # 测试代码执行
        logger.info("\n--- 测试代码执行 (async) ---")
        async with await sb.context.create_async(
            language=CodeLanguage.PYTHON
        ) as ctx:
            result = await ctx.execute_async(
                code="print('Hello from All-in-One Sandbox async!')"
            )
            logger.info(f"✓ 代码执行结果 (async): {result}")

            result = await ctx.execute_async(
                code="import platform; print(platform.platform())"
            )
            logger.info(f"✓ 平台信息 (async): {result}")

        # 测试文件系统操作
        logger.info("\n--- 测试文件系统操作 (async) ---")
        list_result = await sb.file_system.list_async(path="/home/user")
        logger.info(f"✓ 根目录文件列表 (async): {list_result}")

        create_result = await sb.file_system.mkdir_async(
            path="/home/user/aio-test-async"
        )
        logger.info(f"✓ 创建文件夹 (async): {create_result}")

        # 测试文件读写
        logger.info("\n--- 测试文件读写 (async) ---")
        write_result = await sb.file.write_async(
            path="/home/user/aio-test-async/hello.txt",
            content="Hello from All-in-One Sandbox async!",
        )
        logger.info(f"✓ 写入文件结果 (async): {write_result}")

        read_result = await sb.file.read_async(
            path="/home/user/aio-test-async/hello.txt"
        )
        logger.info(f"✓ 读取文件结果 (async): {read_result}")

        # 测试进程操作
        logger.info("\n--- 测试进程操作 (async) ---")
        process_result = await sb.process.list_async()
        logger.info(f"✓ 进程列表 (async): {process_result}")

        process_result = await sb.process.cmd_async(
            command="ls -la /home/user/aio-test-async", cwd="/"
        )
        logger.info(f"✓ 命令执行结果 (async): {process_result}")

        # 清理测试目录
        await sb.file_system.remove_async(path="/home/user/aio-test-async")
        logger.info(f"✓ 删除测试目录 (async)")

    # 清理资源
    logger.info("\n--- 清理资源 ---")
    await Sandbox.delete_template_async(template_name)
    logger.info(f"✓ 删除模板成功 (async): {template_name}")

    # 清理截图文件
    if os.path.exists("aio_async_screenshot.png"):
        os.remove("aio_async_screenshot.png")
        logger.info(f"✓ 删除截图文件")

    logger.info("\n✓ All-in-One Sandbox 异步测试完成\n")


if __name__ == "__main__":
    import asyncio

    # code_interpreter_sync()
    # browser_sync()
    all_in_one_sync()
    # asyncio.run(browser_async())
    # asyncio.run(code_interpreter_async())
    # asyncio.run(all_in_one_async())
