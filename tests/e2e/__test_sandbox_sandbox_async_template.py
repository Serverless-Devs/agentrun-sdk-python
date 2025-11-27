"""
Sandbox 模块的 E2E 测试

测试覆盖:
- 创建 Sandbox (CodeInterpreter 和 Browser 类型)
- 连接 Sandbox
- 获取 Sandbox
- 列举 Sandboxes
- 删除 Sandbox
- Sandbox 生命周期测试
"""

import asyncio
import time  # noqa: F401

import pytest

from agentrun.sandbox import Sandbox, Template
from agentrun.sandbox.model import (
    ListSandboxesInput,
    TemplateInput,
    TemplateNetworkConfiguration,
    TemplateNetworkMode,
    TemplateType,
)
from agentrun.utils.exception import ClientError, ResourceNotExistError


class TestSandbox:
    """Sandbox 模块 E2E 测试"""

    @pytest.fixture
    def template_name_code_interpreter(self, unique_name: str) -> str:
        """生成 Code Interpreter 模板名称"""
        return f"{unique_name}-ci-template"

    @pytest.fixture
    def template_name_browser(self, unique_name: str) -> str:
        """生成 Browser 模板名称"""
        return f"{unique_name}-browser-template"

    async def test_create_code_interpreter_sandbox_async(
        self, template_name_code_interpreter: str
    ):
        """测试创建 Code Interpreter 类型的 Sandbox"""
        # 先创建模板
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name_code_interpreter,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 - Code Interpreter Template",
                cpu=2.0,
                memory=4096,
                disk_size=512,
                sandbox_idle_timeout_in_seconds=600,
                network_configuration=TemplateNetworkConfiguration(
                    network_mode=TemplateNetworkMode.PUBLIC
                ),
            )
        )

        # 等待模板就绪
        await asyncio.sleep(5)

        try:
            # 创建 Sandbox
            sandbox = await Sandbox.create_async(
                template_type=TemplateType.CODE_INTERPRETER,
                template_name=template_name_code_interpreter,
                sandbox_idle_timeout_seconds=600,
            )

            assert sandbox is not None
            assert sandbox.sandbox_id is not None
            assert sandbox.template_name == template_name_code_interpreter
            assert sandbox.status is not None
            assert sandbox.created_at is not None

            # 清理 Sandbox
            await Sandbox.delete_by_id_async(sandbox_id=sandbox.sandbox_id)

        finally:
            # 清理 Template
            await Template.delete_by_name_async(
                template_name=template_name_code_interpreter
            )

    async def test_create_browser_sandbox_async(
        self, template_name_browser: str
    ):
        """测试创建 Browser 类型的 Sandbox"""
        # 先创建模板
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name_browser,
                template_type=TemplateType.BROWSER,
                description="E2E 测试 - Browser Template",
                cpu=2.0,
                memory=4096,
                disk_size=10240,  # Browser 类型必须是 10240
                sandbox_idle_timeout_in_seconds=600,
                network_configuration=TemplateNetworkConfiguration(
                    network_mode=TemplateNetworkMode.PUBLIC
                ),
            )
        )

        # 等待模板就绪
        await asyncio.sleep(5)

        try:
            # 创建 Sandbox
            sandbox = await Sandbox.create_async(
                template_type=TemplateType.BROWSER,
                template_name=template_name_browser,
                sandbox_idle_timeout_seconds=600,
            )

            assert sandbox is not None
            assert sandbox.sandbox_id is not None
            assert sandbox.template_name == template_name_browser
            assert sandbox.status is not None

            # 清理 Sandbox
            await Sandbox.delete_by_id_async(sandbox_id=sandbox.sandbox_id)

        finally:
            # 清理 Template
            await Template.delete_by_name_async(
                template_name=template_name_browser
            )

    async def test_connect_sandbox_async(
        self, template_name_code_interpreter: str
    ):
        """测试连接 Sandbox"""
        # 先创建模板
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name_code_interpreter,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 Template",
                cpu=2.0,
                memory=2048,
            )
        )

        # 等待模板就绪
        await asyncio.sleep(5)

        try:
            # 创建 Sandbox
            sandbox = await Sandbox.create_async(
                template_type=TemplateType.CODE_INTERPRETER,
                template_name=template_name_code_interpreter,
                sandbox_idle_timeout_seconds=600,
            )

            # 等待 Sandbox 就绪
            await asyncio.sleep(10)

            # 连接 Sandbox
            assert sandbox.sandbox_id
            connected_sandbox = await Sandbox.connect_async(
                sandbox_id=sandbox.sandbox_id,
                template_type=TemplateType.CODE_INTERPRETER,
            )

            assert connected_sandbox is not None
            assert connected_sandbox.sandbox_id == sandbox.sandbox_id
            assert (
                connected_sandbox.template_name
                == template_name_code_interpreter
            )

            # 清理 Sandbox
            await Sandbox.delete_by_id_async(sandbox_id=sandbox.sandbox_id)

        finally:
            # 清理 Template
            await Template.delete_by_name_async(
                template_name=template_name_code_interpreter
            )

    async def test_connect_nonexistent_sandbox_async(self):
        """测试连接不存在的 Sandbox 会抛出异常"""
        with pytest.raises(ClientError):
            await Sandbox.connect_async(
                sandbox_id="nonexistent-sandbox-xyz-12345",
                template_type=TemplateType.CODE_INTERPRETER,
            )

    async def test_get_sandbox_async(self, template_name_code_interpreter: str):
        """测试获取 Sandbox"""
        # 先创建模板
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name_code_interpreter,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 Template",
                cpu=2.0,
                memory=2048,
            )
        )

        # 等待模板就绪
        await asyncio.sleep(5)

        try:
            # 创建 Sandbox
            created_sandbox = await Sandbox.create_async(
                template_type=TemplateType.CODE_INTERPRETER,
                template_name=template_name_code_interpreter,
                sandbox_idle_timeout_seconds=600,
            )

            # 等待 Sandbox 就绪
            await asyncio.sleep(10)

            # 通过实例方法获取 Sandbox
            fetched_sandbox = await created_sandbox.get_async()

            assert fetched_sandbox is not None
            assert fetched_sandbox.sandbox_id == created_sandbox.sandbox_id
            assert (
                fetched_sandbox.template_name == template_name_code_interpreter
            )

            # 清理 Sandbox
            assert created_sandbox.sandbox_id
            await Sandbox.delete_by_id_async(
                sandbox_id=created_sandbox.sandbox_id
            )

        finally:
            # 清理 Template
            await Template.delete_by_name_async(
                template_name=template_name_code_interpreter
            )

    async def test_list_sandboxes_async(
        self, template_name_code_interpreter: str
    ):
        """测试列举 Sandboxes"""
        # 先创建模板
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name_code_interpreter,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 Template",
                cpu=2.0,
                memory=2048,
            )
        )

        # 等待模板就绪
        await asyncio.sleep(5)

        try:
            # 创建 Sandbox
            sandbox = await Sandbox.create_async(
                template_type=TemplateType.CODE_INTERPRETER,
                template_name=template_name_code_interpreter,
                sandbox_idle_timeout_seconds=600,
            )

            # 等待 Sandbox 就绪
            await asyncio.sleep(5)

            # 列举 Sandboxes
            result = await Sandbox.list_async()

            assert result is not None
            assert isinstance(result.sandboxes, list)
            assert len(result.sandboxes) > 0

            # 验证我们创建的 Sandbox 在列表中
            sandbox_ids = [s.sandbox_id for s in result.sandboxes]
            assert sandbox.sandbox_id in sandbox_ids

            # 清理 Sandbox
            assert sandbox.sandbox_id
            await Sandbox.delete_by_id_async(sandbox_id=sandbox.sandbox_id)

        finally:
            # 清理 Template
            await Template.delete_by_name_async(
                template_name=template_name_code_interpreter
            )

    async def test_list_sandboxes_with_filters_async(
        self, template_name_code_interpreter: str
    ):
        """测试带过滤条件列举 Sandboxes"""
        # 先创建模板
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name_code_interpreter,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 Template",
                cpu=2.0,
                memory=2048,
            )
        )

        # 等待模板就绪
        await asyncio.sleep(5)

        try:
            # 创建 Sandbox
            sandbox = await Sandbox.create_async(
                template_type=TemplateType.CODE_INTERPRETER,
                template_name=template_name_code_interpreter,
                sandbox_idle_timeout_seconds=600,
            )

            # 等待 Sandbox 就绪
            await asyncio.sleep(5)

            # 使用过滤条件列举
            result = await Sandbox.list_async(
                input=ListSandboxesInput(
                    max_results=5,
                    template_name=template_name_code_interpreter,
                    template_type=TemplateType.CODE_INTERPRETER,
                )
            )

            assert result is not None
            assert isinstance(result.sandboxes, list)
            # 验证返回的 Sandboxes 都符合过滤条件
            for s in result.sandboxes:
                assert s.template_name == template_name_code_interpreter

            # 清理 Sandbox
            assert sandbox.sandbox_id
            await Sandbox.delete_by_id_async(sandbox_id=sandbox.sandbox_id)

        finally:
            # 清理 Template
            await Template.delete_by_name_async(
                template_name=template_name_code_interpreter
            )

    async def test_delete_sandbox_async(
        self, template_name_code_interpreter: str
    ):
        """测试删除 Sandbox"""
        # 先创建模板
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name_code_interpreter,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 Template",
                cpu=2.0,
                memory=2048,
            )
        )

        # 等待模板就绪
        await asyncio.sleep(5)

        try:
            # 创建 Sandbox
            sandbox = await Sandbox.create_async(
                template_type=TemplateType.CODE_INTERPRETER,
                template_name=template_name_code_interpreter,
                sandbox_idle_timeout_seconds=600,
            )

            # 确认创建成功
            assert sandbox.sandbox_id is not None

            # 删除 Sandbox
            deleted_sandbox = await Sandbox.delete_by_id_async(
                sandbox_id=sandbox.sandbox_id
            )
            assert deleted_sandbox is not None

            # 等待删除操作完成
            await asyncio.sleep(5)

            # 验证删除成功 - 尝试连接应该抛出异常
            with pytest.raises(ClientError):
                await Sandbox.connect_async(
                    sandbox_id=sandbox.sandbox_id,
                    template_type=TemplateType.CODE_INTERPRETER,
                )

        finally:
            # 清理 Template
            await Template.delete_by_name_async(
                template_name=template_name_code_interpreter
            )

    async def test_delete_sandbox_via_instance_method_async(
        self, template_name_code_interpreter: str
    ):
        """测试通过实例方法删除 Sandbox"""
        # 先创建模板
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name_code_interpreter,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 Template",
                cpu=2.0,
                memory=2048,
            )
        )

        # 等待模板就绪
        await asyncio.sleep(5)

        try:
            # 创建 Sandbox
            sandbox = await Sandbox.create_async(
                template_type=TemplateType.CODE_INTERPRETER,
                template_name=template_name_code_interpreter,
                sandbox_idle_timeout_seconds=600,
            )

            # 通过实例方法删除
            await sandbox.delete_async()

            # 等待删除操作完成
            await asyncio.sleep(5)

            # 验证删除成功
            with pytest.raises(ClientError):
                assert sandbox.sandbox_id
                await Sandbox.connect_async(
                    sandbox_id=sandbox.sandbox_id,
                    template_type=TemplateType.CODE_INTERPRETER,
                )

        finally:
            # 清理 Template
            await Template.delete_by_name_async(
                template_name=template_name_code_interpreter
            )

    async def test_delete_nonexistent_sandbox_async(self):
        """测试删除不存在的 Sandbox 会抛出异常"""
        with pytest.raises(ClientError):
            await Sandbox.delete_by_id_async(
                sandbox_id="nonexistent-sandbox-xyz-12345"
            )

    async def test_sandbox_lifecycle_async(
        self, template_name_code_interpreter: str
    ):
        """测试 Sandbox 的完整生命周期"""
        # 1. 创建模板
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name_code_interpreter,
                template_type=TemplateType.CODE_INTERPRETER,
                description="生命周期测试 Template",
                cpu=2.0,
                memory=2048,
                environment_variables={"LIFECYCLE": "test"},
            )
        )
        assert template.template_name == template_name_code_interpreter

        # 等待模板就绪
        await asyncio.sleep(5)

        try:
            # 2. 创建 Sandbox
            sandbox = await Sandbox.create_async(
                template_type=TemplateType.CODE_INTERPRETER,
                template_name=template_name_code_interpreter,
                sandbox_idle_timeout_seconds=600,
            )
            assert sandbox.sandbox_id is not None
            assert sandbox.template_name == template_name_code_interpreter

            # 等待 Sandbox 就绪
            await asyncio.sleep(10)

            # 3. 连接 Sandbox
            connected_sandbox = await Sandbox.connect_async(
                sandbox_id=sandbox.sandbox_id,
                template_type=TemplateType.CODE_INTERPRETER,
            )
            assert connected_sandbox.sandbox_id == sandbox.sandbox_id

            # 4. 获取 Sandbox
            fetched_sandbox = await connected_sandbox.get_async()
            assert fetched_sandbox.sandbox_id == sandbox.sandbox_id

            # 5. 列举 Sandboxes
            result = await Sandbox.list_async()
            sandbox_ids = [s.sandbox_id for s in result.sandboxes]
            assert sandbox.sandbox_id in sandbox_ids

            # 6. 删除 Sandbox
            await Sandbox.delete_by_id_async(sandbox_id=sandbox.sandbox_id)

            # 等待删除完成
            await asyncio.sleep(5)

            # 7. 验证删除
            with pytest.raises(ClientError):
                await Sandbox.connect_async(
                    sandbox_id=sandbox.sandbox_id,
                    template_type=TemplateType.CODE_INTERPRETER,
                )

        finally:
            # 清理 Template
            await Template.delete_by_name_async(
                template_name=template_name_code_interpreter
            )

    async def test_sandbox_with_different_template_types_async(
        self, template_name_code_interpreter: str, template_name_browser: str
    ):
        """测试不同类型的 Template 创建 Sandbox"""
        # 创建 Code Interpreter 模板
        ci_template = await Template.create_async(
            TemplateInput(
                template_name=template_name_code_interpreter,
                template_type=TemplateType.CODE_INTERPRETER,
                cpu=2.0,
                memory=2048,
            )
        )

        # 创建 Browser 模板
        browser_template = await Template.create_async(
            TemplateInput(
                template_name=template_name_browser,
                template_type=TemplateType.BROWSER,
                cpu=2.0,
                memory=4096,
                disk_size=10240,
            )
        )

        # 等待模板就绪
        await asyncio.sleep(5)

        try:
            # 创建 Code Interpreter Sandbox
            ci_sandbox = await Sandbox.create_async(
                template_type=TemplateType.CODE_INTERPRETER,
                template_name=template_name_code_interpreter,
                sandbox_idle_timeout_seconds=600,
            )
            assert ci_sandbox.sandbox_id is not None

            # 创建 Browser Sandbox
            browser_sandbox = await Sandbox.create_async(
                template_type=TemplateType.BROWSER,
                template_name=template_name_browser,
                sandbox_idle_timeout_seconds=600,
            )
            assert browser_sandbox.sandbox_id is not None

            # 等待就绪
            await asyncio.sleep(10)

            # 验证可以分别连接
            connected_ci = await Sandbox.connect_async(
                sandbox_id=ci_sandbox.sandbox_id,
                template_type=TemplateType.CODE_INTERPRETER,
            )
            assert connected_ci.sandbox_id == ci_sandbox.sandbox_id

            connected_browser = await Sandbox.connect_async(
                sandbox_id=browser_sandbox.sandbox_id,
                template_type=TemplateType.BROWSER,
            )
            assert connected_browser.sandbox_id == browser_sandbox.sandbox_id

            # 清理 Sandboxes
            await Sandbox.delete_by_id_async(sandbox_id=ci_sandbox.sandbox_id)
            await Sandbox.delete_by_id_async(
                sandbox_id=browser_sandbox.sandbox_id
            )

        finally:
            # 清理 Templates
            await Template.delete_by_name_async(
                template_name=template_name_code_interpreter
            )
            await Template.delete_by_name_async(
                template_name=template_name_browser
            )

    async def test_connect_with_wrong_template_type_async(
        self, template_name_code_interpreter: str
    ):
        """测试用错误的 template_type 连接 Sandbox 会抛出异常"""
        # 创建 Code Interpreter 模板
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name_code_interpreter,
                template_type=TemplateType.CODE_INTERPRETER,
                cpu=2.0,
                memory=2048,
            )
        )

        # 等待模板就绪
        await asyncio.sleep(5)

        try:
            # 创建 Code Interpreter Sandbox
            sandbox = await Sandbox.create_async(
                template_type=TemplateType.CODE_INTERPRETER,
                template_name=template_name_code_interpreter,
                sandbox_idle_timeout_seconds=600,
            )

            # 等待就绪
            await asyncio.sleep(10)

            # 尝试用错误的 template_type 连接
            with pytest.raises(ValueError):
                assert sandbox.sandbox_id
                await Sandbox.connect_async(
                    sandbox_id=sandbox.sandbox_id,
                    template_type=TemplateType.BROWSER,  # 错误的类型
                )

            # 清理 Sandbox
            assert sandbox.sandbox_id
            await Sandbox.delete_by_id_async(sandbox_id=sandbox.sandbox_id)

        finally:
            # 清理 Template
            await Template.delete_by_name_async(
                template_name=template_name_code_interpreter
            )

    async def test_create_multiple_sandboxes_async(
        self, template_name_code_interpreter: str
    ):
        """测试基于同一模板创建多个 Sandboxes"""
        # 创建模板
        await Template.create_async(
            TemplateInput(
                template_name=template_name_code_interpreter,
                template_type=TemplateType.CODE_INTERPRETER,
                cpu=2.0,
                memory=2048,
            )
        )

        # 等待模板就绪
        await asyncio.sleep(5)

        sandboxes = []

        try:
            # 创建多个 Sandboxes
            for i in range(3):
                sandbox = await Sandbox.create_async(
                    template_type=TemplateType.CODE_INTERPRETER,
                    template_name=template_name_code_interpreter,
                    sandbox_idle_timeout_seconds=600,
                )
                sandboxes.append(sandbox)
                assert sandbox.sandbox_id is not None

            # 验证所有 Sandboxes 都创建成功
            assert len(sandboxes) == 3

            # 验证它们有不同的 ID
            sandbox_ids = [s.sandbox_id for s in sandboxes]
            assert len(set(sandbox_ids)) == 3

        finally:
            # 清理所有 Sandboxes
            for sandbox in sandboxes:
                try:
                    await Sandbox.delete_by_id_async(
                        sandbox_id=sandbox.sandbox_id
                    )
                except Exception:
                    pass  # 忽略清理错误

            # 清理 Template
            await Template.delete_by_name_async(
                template_name=template_name_code_interpreter
            )

    async def test_create_sandbox_without_template_async(self):
        """测试没有提供 template_name 时创建 Sandbox 会抛出异常"""
        with pytest.raises(ValueError, match="template_name is required"):
            await Sandbox.create_async(
                template_type=TemplateType.CODE_INTERPRETER,
                template_name=None,
            )

    async def test_connect_without_template_type_async(
        self, template_name_code_interpreter: str
    ):
        """测试不提供 template_type 参数也能连接 Sandbox"""
        # 创建模板
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name_code_interpreter,
                template_type=TemplateType.CODE_INTERPRETER,
                cpu=2.0,
                memory=2048,
            )
        )

        # 等待模板就绪
        await asyncio.sleep(5)

        try:
            # 创建 Sandbox
            sandbox = await Sandbox.create_async(
                template_type=TemplateType.CODE_INTERPRETER,
                template_name=template_name_code_interpreter,
                sandbox_idle_timeout_seconds=600,
            )

            # 等待就绪
            await asyncio.sleep(10)

            # 不提供 template_type 参数连接
            assert sandbox.sandbox_id
            connected_sandbox = await Sandbox.connect_async(
                sandbox_id=sandbox.sandbox_id
            )
            assert connected_sandbox.sandbox_id == sandbox.sandbox_id
            assert (
                connected_sandbox.template_name
                == template_name_code_interpreter
            )

            # 清理 Sandbox
            await Sandbox.delete_by_id_async(sandbox_id=sandbox.sandbox_id)

        finally:
            # 清理 Template
            await Template.delete_by_name_async(
                template_name=template_name_code_interpreter
            )
