"""
Sandbox Code Interpreter 模块的 E2E 测试

测试覆盖:
- 代码执行 (Context)
- 文件系统操作 (FileSystem)
- 文件读写操作 (File)
- 进程操作 (Process)
- 上传下载
"""

import asyncio
import os
import tempfile
import time

import pytest

from agentrun.sandbox import Sandbox, Template
from agentrun.sandbox.model import (
    CodeLanguage,
    TemplateInput,
    TemplateNetworkConfiguration,
    TemplateNetworkMode,
    TemplateType,
)
from agentrun.utils.exception import ResourceNotExistError


class TestSandboxCodeInterpreter:
    """Sandbox Code Interpreter 模块 E2E 测试"""

    @pytest.fixture
    async def template(self, unique_name: str):
        """创建测试模板"""
        template_name = f"{unique_name}-ci-template"
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 - Code Interpreter",
                cpu=2.0,
                memory=4096,
                disk_size=512,
                sandbox_idle_timeout_in_seconds=600,
                sandbox_ttlin_seconds=600,
                network_configuration=TemplateNetworkConfiguration(
                    network_mode=TemplateNetworkMode.PUBLIC
                ),
            )
        )
        yield template
        # 清理资源
        try:
            await Template.delete_by_name_async(template_name=template_name)
        except ResourceNotExistError:
            pass

    @pytest.fixture
    async def sandbox(self, template):
        """创建测试 Sandbox"""
        sb = await Sandbox.create_async(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name=template.template_name,
            sandbox_idle_timeout_seconds=600,
        )

        # 等待 Sandbox 就绪
        max_retries = 60
        for _ in range(max_retries):
            health = await sb.check_health_async()
            if health.get("status") == "ok":
                break
            await asyncio.sleep(1)

        yield sb

        # 清理资源
        try:
            await sb.delete_async()
        except Exception:
            pass

    # ========== 代码执行测试 (Context) ==========

    async def test_context_execute_python_async(self, sandbox):
        """测试执行 Python 代码"""
        result = await sandbox.context.execute_async(
            code="print('hello world')",
            language=CodeLanguage.PYTHON,
        )

        assert result is not None
        # 验证输出包含预期内容
        if isinstance(result, dict):
            assert "hello world" in str(result)

    async def test_context_execute_shell_async(self, sandbox):
        """测试执行 Shell 命令（通过 process.cmd）"""
        result = await sandbox.process.cmd_async(
            command="echo 'test'",
            cwd="/",
        )

        assert result is not None
        if isinstance(result, dict):
            output_str = str(result)
            assert (
                "test" in output_str
                or result.get("exit_code") == 0
                or result.get("success") is True
            )

    async def test_context_create_and_execute_async(self, sandbox):
        """测试创建上下文并执行代码"""
        # 创建 Python 上下文
        async with await sandbox.context.create_async(
            language=CodeLanguage.PYTHON
        ) as ctx:
            # 执行代码
            result = await ctx.execute_async(code="x = 10\nprint(x)")
            assert result is not None

            # 在同一上下文中继续执行
            result = await ctx.execute_async(code="print(x + 5)")
            assert result is not None

            # 获取上下文信息
            ctx_info = await ctx.get_async()
            assert ctx_info is not None
            assert ctx_info._context_id is not None

    async def test_context_list_async(self, sandbox):
        """测试列举上下文"""
        # 创建一个上下文
        async with await sandbox.context.create_async(
            language=CodeLanguage.PYTHON
        ) as ctx:
            # 列举上下文
            contexts = await ctx.list_async()
            assert contexts is not None
            assert isinstance(contexts, list) or isinstance(contexts, dict)

    # ========== 文件系统测试 (FileSystem) ==========

    async def test_filesystem_list_async(self, sandbox):
        """测试列举目录"""
        result = await sandbox.file_system.list_async(path="/")
        assert result is not None
        assert isinstance(result, (list, dict))

    async def test_filesystem_mkdir_async(self, sandbox):
        """测试创建目录"""
        test_dir = f"/test-dir-{int(time.time())}"
        result = await sandbox.file_system.mkdir_async(path=test_dir)
        assert result is not None

        # 验证目录已创建
        stat_result = await sandbox.file_system.stat_async(path=test_dir)
        assert stat_result is not None

    async def test_filesystem_stat_async(self, sandbox):
        """测试获取文件/目录状态"""
        # 创建测试目录
        test_dir = f"/test-stat-{int(time.time())}"
        await sandbox.file_system.mkdir_async(path=test_dir)

        # 获取状态
        result = await sandbox.file_system.stat_async(path=test_dir)
        assert result is not None
        assert isinstance(result, dict)

    async def test_filesystem_move_async(self, sandbox):
        """测试移动文件"""
        # 创建源文件
        source_path = f"/test-source-{int(time.time())}.txt"
        await sandbox.file.write_async(path=source_path, content="test content")

        # 移动文件
        dest_path = f"/test-dest-{int(time.time())}.txt"
        result = await sandbox.file_system.move_async(
            source=source_path,
            destination=dest_path,
        )
        assert result is not None

        # 验证新文件存在
        read_result = await sandbox.file.read_async(path=dest_path)
        content = (
            read_result.get("content", read_result)
            if isinstance(read_result, dict)
            else read_result
        )
        assert "test content" in str(content)

    async def test_filesystem_remove_async(self, sandbox):
        """测试删除文件/目录"""
        # 创建测试目录
        test_dir = f"/test-remove-{int(time.time())}"
        await sandbox.file_system.mkdir_async(path=test_dir)

        # 删除目录
        result = await sandbox.file_system.remove_async(path=test_dir)
        assert result is not None

    async def test_filesystem_upload_download_async(self, sandbox):
        """测试上传和下载文件"""
        # 创建临时测试文件
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as f:
            test_content = f"Test file created at {time.time()}\n"
            test_content += (
                "This is a test for upload/download functionality.\n"
            )
            f.write(test_content)
            local_file_path = f.name

        try:
            # 上传文件
            remote_path = f"/test-upload-{int(time.time())}.txt"
            upload_result = await sandbox.file_system.upload_async(
                local_file_path=local_file_path,
                target_file_path=remote_path,
            )
            assert upload_result is not None

            # 验证文件已上传
            stat_result = await sandbox.file_system.stat_async(path=remote_path)
            assert stat_result is not None

            # 下载文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                download_path = f.name

            download_result = await sandbox.file_system.download_async(
                path=remote_path,
                save_path=download_path,
            )
            assert download_result is not None

            # 验证下载的内容
            with open(download_path, "r", encoding="utf-8") as f:
                downloaded_content = f.read()
            assert test_content in downloaded_content

            # 清理下载的文件
            os.remove(download_path)

        finally:
            # 清理临时文件
            if os.path.exists(local_file_path):
                os.remove(local_file_path)

    # ========== 文件读写测试 (File) ==========

    async def test_file_write_read_async(self, sandbox):
        """测试文件写入和读取"""
        test_path = f"/test-file-{int(time.time())}.txt"
        test_content = "Hello, World!\nThis is a test file."

        # 写入文件
        write_result = await sandbox.file.write_async(
            path=test_path,
            content=test_content,
        )
        assert write_result is not None

        # 读取文件
        read_result = await sandbox.file.read_async(path=test_path)
        assert read_result is not None
        content = (
            read_result.get("content", read_result)
            if isinstance(read_result, dict)
            else read_result
        )
        assert test_content in str(content)

    async def test_file_write_unicode_async(self, sandbox):
        """测试写入和读取 Unicode 内容"""
        test_path = f"/test-unicode-{int(time.time())}.txt"
        test_content = "你好，世界！\nこんにちは世界\n안녕하세요 세계"

        # 写入 Unicode 内容
        await sandbox.file.write_async(path=test_path, content=test_content)

        # 读取并验证
        read_result = await sandbox.file.read_async(path=test_path)
        content = (
            read_result.get("content", read_result)
            if isinstance(read_result, dict)
            else read_result
        )
        assert test_content in str(content)

    async def test_file_overwrite_async(self, sandbox):
        """测试覆盖写入文件"""
        test_path = f"/test-overwrite-{int(time.time())}.txt"

        # 第一次写入
        await sandbox.file.write_async(
            path=test_path, content="original content"
        )

        # 覆盖写入
        new_content = "new content"
        await sandbox.file.write_async(path=test_path, content=new_content)

        # 验证内容已更新
        read_result = await sandbox.file.read_async(path=test_path)
        content = (
            read_result.get("content", read_result)
            if isinstance(read_result, dict)
            else read_result
        )
        assert new_content in str(content)
        assert "original content" not in str(content)

    async def test_file_nested_directory_async(self, sandbox):
        """测试在嵌套目录中创建文件"""
        # 创建嵌套目录
        nested_dir = f"/test-nested-{int(time.time())}/subdir"
        await sandbox.file_system.mkdir_async(path=nested_dir)

        # 在嵌套目录中写入文件
        test_path = f"{nested_dir}/test.txt"
        test_content = "file in nested directory"
        await sandbox.file.write_async(path=test_path, content=test_content)

        # 读取并验证
        read_result = await sandbox.file.read_async(path=test_path)
        content = (
            read_result.get("content", read_result)
            if isinstance(read_result, dict)
            else read_result
        )
        assert test_content in str(content)

    # ========== 进程测试 (Process) ==========

    async def test_process_list_async(self, sandbox):
        """测试列举进程"""
        result = await sandbox.process.list_async()
        assert result is not None
        assert isinstance(result, (list, dict))

    async def test_process_cmd_async(self, sandbox):
        """测试执行命令"""
        result = await sandbox.process.cmd_async(command="ls", cwd="/")
        assert result is not None
        assert isinstance(result, dict)
        # 验证命令执行成功
        inner_result = result.get("result", {})
        exit_code = inner_result.get("exitCode")
        if exit_code is not None:
            assert exit_code == 0
        # 或者检查 status 字段
        assert result.get("status") in ["completed", "success", None]

    async def test_process_cmd_with_output_async(self, sandbox):
        """测试执行命令并获取输出"""
        result = await sandbox.process.cmd_async(
            command="echo 'test output'",
            cwd="/",
        )
        assert result is not None
        # 验证输出包含预期内容
        if isinstance(result, dict):
            # 结果在 result.result.stdout 或 result.result.exitCode
            inner_result = result.get("result", {})
            output = inner_result.get("stdout", "") or inner_result.get(
                "output", ""
            )
            exit_code = inner_result.get("exitCode")
            assert "test output" in str(output) or exit_code == 0

    async def test_process_get_async(self, sandbox):
        """测试获取进程信息"""
        # 列举进程
        processes = await sandbox.process.list_async()

        # 如果有进程，尝试获取第一个进程的信息
        if isinstance(processes, list) and len(processes) > 0:
            first_pid = processes[0].get("pid", "1")
            result = await sandbox.process.get_async(pid=str(first_pid))
            assert result is not None
        else:
            # 尝试获取 PID 1 的信息
            result = await sandbox.process.get_async(pid="1")
            assert result is not None

    async def test_process_cmd_multiple_commands_async(self, sandbox):
        """测试执行多个命令"""
        # 执行多个命令
        commands = [
            "echo 'command 1'",
            "ls /",
            "pwd",
        ]

        for cmd in commands:
            result = await sandbox.process.cmd_async(command=cmd, cwd="/")
            assert result is not None

    async def test_process_cmd_different_cwd_async(self, sandbox):
        """测试在不同工作目录执行命令"""
        # 创建测试目录
        test_dir = f"/test-cwd-{int(time.time())}"
        await sandbox.file_system.mkdir_async(path=test_dir)

        # 在测试目录中执行命令
        result = await sandbox.process.cmd_async(command="pwd", cwd=test_dir)
        assert result is not None

    # ========== 综合测试 ==========

    async def test_code_interpreter_workflow_async(self, sandbox):
        """测试完整的 Code Interpreter 工作流"""
        workflow_dir = f"/workflow-{int(time.time())}"

        # 1. 创建工作目录
        await sandbox.file_system.mkdir_async(path=workflow_dir)

        # 2. 写入 Python 脚本
        script_path = f"{workflow_dir}/script.py"
        script_content = """
import os

# 创建数据文件
with open('/workflow-{}/data.txt', 'w') as f:
    f.write('Generated by Python script\\n')
    f.write('Number: 42\\n')

print('Script executed successfully')
""".format(workflow_dir.replace("/workflow-", ""))

        await sandbox.file.write_async(path=script_path, content=script_content)

        # 3. 执行 Python 脚本
        result = await sandbox.context.execute_async(
            code=f"exec(open('{script_path}').read())",
            language=CodeLanguage.PYTHON,
        )
        assert result is not None

        # 4. 验证生成的数据文件
        data_path = f"{workflow_dir}/data.txt"
        read_result = await sandbox.file.read_async(path=data_path)
        content = (
            read_result.get("content", read_result)
            if isinstance(read_result, dict)
            else read_result
        )
        assert "Generated by Python script" in str(content)
        assert "Number: 42" in str(content)

        # 5. 使用命令行工具处理数据
        await sandbox.process.cmd_async(
            command=f"cat {data_path}",
            cwd=workflow_dir,
        )

        # 6. 清理工作目录
        await sandbox.file_system.remove_async(path=workflow_dir)

    async def test_concurrent_operations_async(self, sandbox):
        """测试并发操作"""
        base_dir = f"/concurrent-{int(time.time())}"
        await sandbox.file_system.mkdir_async(path=base_dir)

        # 并发执行多个操作
        tasks = []

        # 创建多个文件
        for i in range(5):
            file_path = f"{base_dir}/file-{i}.txt"
            task = sandbox.file.write_async(
                path=file_path,
                content=f"Content of file {i}",
            )
            tasks.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
        assert len(results) == 5

        # 并发读取所有文件
        read_tasks = [
            sandbox.file.read_async(path=f"{base_dir}/file-{i}.txt")
            for i in range(5)
        ]
        read_results = await asyncio.gather(*read_tasks)

        # 验证内容
        for i, read_result in enumerate(read_results):
            content = (
                read_result.get("content", read_result)
                if isinstance(read_result, dict)
                else read_result
            )
            assert f"Content of file {i}" in str(content)

        # 清理
        await sandbox.file_system.remove_async(path=base_dir)

    async def test_large_file_operations_async(self, sandbox):
        """测试大文件操作"""
        # 创建较大的内容（约 1MB）
        large_content = "x" * (1024 * 1024)  # 1MB of 'x'
        test_path = f"/test-large-{int(time.time())}.txt"

        # 写入大文件
        await sandbox.file.write_async(path=test_path, content=large_content)

        # 读取大文件
        read_result = await sandbox.file.read_async(path=test_path)
        content = (
            read_result.get("content", read_result)
            if isinstance(read_result, dict)
            else read_result
        )
        assert len(str(content)) >= len(large_content) * 0.9  # 允许一些编码差异

        # 清理
        await sandbox.file_system.remove_async(path=test_path)

    async def test_error_handling_nonexistent_file_async(self, sandbox):
        """测试读取不存在的文件的错误处理"""
        nonexistent_path = "/nonexistent-file-xyz.txt"

        # 尝试读取不存在的文件应该引发异常或返回错误
        try:
            result = await sandbox.file.read_async(path=nonexistent_path)
            # 如果没有抛出异常，检查返回值是否表明错误
            if isinstance(result, dict):
                assert "error" in result or result.get("success") is False
        except Exception:
            # 预期会抛出异常
            pass

    async def test_sandbox_context_manager_async(self, template):
        """测试使用上下文管理器创建 Sandbox"""
        async with await Sandbox.create_async(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name=template.template_name,
            sandbox_idle_timeout_seconds=600,
        ) as sb:
            # 等待就绪
            max_retries = 60
            for _ in range(max_retries):
                health = await sb.check_health_async()
                if health.get("status") == "ok":
                    break
                await asyncio.sleep(1)

            # 执行简单操作
            result = await sb.context.execute_async(
                code="print('context manager test')",
                language=CodeLanguage.PYTHON,
            )
            assert result is not None

        # Sandbox 应该在退出上下文后自动清理

    async def test_sandbox_health_check_async(self, sandbox):
        """测试健康检查"""
        result = await sandbox.check_health_async()
        assert result is not None
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "ok"
