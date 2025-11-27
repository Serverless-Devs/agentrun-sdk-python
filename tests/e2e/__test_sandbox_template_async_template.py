"""
Sandbox Template 模块的 E2E 测试

测试覆盖:
- 创建 Template
- 获取 Template
- 列举 Templates
- 更新 Template
- 删除 Template
"""

import asyncio
import datetime
import time

import pytest

from agentrun.sandbox import Sandbox, Template
from agentrun.sandbox.model import (
    PageableInput,
    TemplateInput,
    TemplateNetworkConfiguration,
    TemplateNetworkMode,
    TemplateType,
)
from agentrun.utils.exception import ClientError, ResourceNotExistError


class TestSandboxTemplate:
    """Sandbox Template 模块 E2E 测试"""

    @pytest.fixture
    def template_name(self, unique_name: str) -> str:
        """生成模板名称"""
        return f"{unique_name}-template"

    async def test_create_template_async(self, template_name: str):
        """测试创建 Template - Code Interpreter 类型"""
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 - Code Interpreter Template",
                cpu=2.0,
                memory=4096,
                disk_size=512,
                sandbox_idle_timeout_in_seconds=600,
                sandbox_ttlin_seconds=600,
                environment_variables={"TEST_ENV": "e2e"},
                network_configuration=TemplateNetworkConfiguration(
                    network_mode=TemplateNetworkMode.PUBLIC
                ),
            )
        )

        assert template is not None
        assert template.template_name == template_name
        assert template.template_type == TemplateType.CODE_INTERPRETER
        # 注意：服务端 API 不返回 description 字段
        # assert template.description == "E2E 测试 - Code Interpreter Template"
        assert template.cpu == 2.0
        assert template.memory == 4096
        assert template.disk_size == 512
        assert template.status is not None
        assert template.template_id is not None
        assert template.created_at is not None

        # 清理资源
        await Template.delete_by_name_async(template_name=template_name)

    async def test_create_browser_template_async(self, template_name: str):
        """测试创建 Template - Browser 类型"""
        browser_template_name = f"{template_name}-browser"

        template = await Template.create_async(
            TemplateInput(
                template_name=browser_template_name,
                template_type=TemplateType.BROWSER,
                description="E2E 测试 - Browser Template",
                cpu=2.0,
                memory=4096,
                disk_size=10240,  # Browser 类型必须是 10240
                sandbox_idle_timeout_in_seconds=600,
                sandbox_ttlin_seconds=600,
                network_configuration=TemplateNetworkConfiguration(
                    network_mode=TemplateNetworkMode.PUBLIC
                ),
            )
        )

        assert template is not None
        assert template.template_name == browser_template_name
        assert template.template_type == TemplateType.BROWSER
        assert template.disk_size == 10240
        assert template.status is not None

        # 清理资源
        await Template.delete_by_name_async(template_name=browser_template_name)

    async def test_create_duplicate_template_async(self, template_name: str):
        """测试创建重复 Template 会抛出异常"""
        # 第一次创建
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 Template",
            )
        )

        try:
            # 第二次创建应该抛出异常
            with pytest.raises(ClientError):
                await Template.create_async(
                    TemplateInput(
                        template_name=template_name,
                        template_type=TemplateType.CODE_INTERPRETER,
                        description="重复的 Template",
                    )
                )
        finally:
            # 清理资源
            await Template.delete_by_name_async(template_name=template_name)

    async def test_get_template_async(self, template_name: str):
        """测试获取 Template"""
        # 先创建 Template
        created_template = await Template.create_async(
            TemplateInput(
                template_name=template_name,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 Template",
                cpu=2.0,
                memory=2048,
            )
        )

        try:
            # 获取 Template
            template = await Template.get_by_name_async(
                template_name=template_name
            )

            assert template is not None
            assert template.template_name == template_name
            assert template.template_id == created_template.template_id
            assert template.template_type == TemplateType.CODE_INTERPRETER
            assert template.cpu == 2.0
            assert template.memory == 2048
            # 注意：服务端 API 不返回 description 字段
        finally:
            # 清理资源
            await Template.delete_by_name_async(template_name=template_name)

    async def test_get_nonexistent_template_async(self):
        """测试获取不存在的 Template 会抛出异常"""
        with pytest.raises(ResourceNotExistError):
            await Template.get_by_name_async(
                template_name="nonexistent-template-xyz-12345"
            )

    async def test_list_templates_async(self, template_name: str):
        """测试列举 Templates"""
        # 创建测试 Template
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 Template",
            )
        )

        try:
            # 列举 Templates
            templates = await Template.list_templates_async()

            assert isinstance(templates, list)
            assert len(templates) > 0

            # 验证我们创建的 Template 在列表中
            template_names = [t.template_name for t in templates]
            assert template_name in template_names
        finally:
            # 清理资源
            await Template.delete_by_name_async(template_name=template_name)

    async def test_list_templates_with_pagination_async(
        self, template_name: str
    ):
        """测试带分页参数列举 Templates"""
        # 创建测试 Template
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 Template",
            )
        )

        try:
            # 使用分页参数列举 Templates
            templates = await Template.list_templates_async(
                input=PageableInput(page_number=1, page_size=5)
            )

            assert isinstance(templates, list)
            # 最多返回 5 条
            assert len(templates) <= 5
        finally:
            # 清理资源
            await Template.delete_by_name_async(template_name=template_name)

    async def test_list_templates_by_type_async(self, template_name: str):
        """测试按类型过滤列举 Templates"""
        # 创建 Code Interpreter Template
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 Template",
            )
        )

        try:
            # 按类型过滤列举
            templates = await Template.list_templates_async(
                input=PageableInput(
                    page_number=1,
                    page_size=10,
                    template_type=TemplateType.CODE_INTERPRETER,
                )
            )

            assert isinstance(templates, list)
            # 验证所有返回的 Template 都是指定类型
            for t in templates:
                assert t.template_type == TemplateType.CODE_INTERPRETER
        finally:
            # 清理资源
            await Template.delete_by_name_async(template_name=template_name)

    async def test_update_template_async(self, template_name: str):
        """测试更新 Template"""
        try:
            time1 = datetime.datetime.now(datetime.timezone.utc)

            # 创建 Template
            template = await Template.create_async(
                TemplateInput(
                    template_name=template_name,
                    template_type=TemplateType.CODE_INTERPRETER,
                    description="原始描述",
                    cpu=2.0,
                    memory=2048,
                    sandbox_idle_timeout_in_seconds=600,
                    environment_variables={"OLD_KEY": "old_value"},
                )
            )

            # 验证初始状态
            assert template.template_name == template_name
            # 注意：服务端 API 不返回 description 字段，但会返回 environment_variables
            # assert template.description == "原始描述"
            assert template.cpu == 2.0
            assert template.memory == 2048
            assert template.environment_variables == {"OLD_KEY": "old_value"}
            assert template.created_at is not None
            created_at = datetime.datetime.strptime(
                template.created_at, "%Y-%m-%dT%H:%M:%S.%f%z"
            )
            assert created_at > time1

            # 更新 Template
            new_description = f"更新后的描述 - {time.time()}"
            updated_template = await Template.update_by_name_async(
                template_name=template_name,
                input=TemplateInput(
                    template_type=TemplateType.CODE_INTERPRETER,
                    description=new_description,
                    cpu=2.0,
                    memory=4096,
                    sandbox_idle_timeout_in_seconds=1200,
                    environment_variables={
                        "NEW_KEY": "new_value",
                        "UPDATED": "true",
                    },
                ),
            )

            # 验证更新后的状态
            # 注意：服务端 API 不返回 description 字段，但会返回 environment_variables
            # assert updated_template.description == new_description
            # 更新操作是异步的，需要等待更新完成再验证
            import asyncio

            await asyncio.sleep(5)
            # 重新获取确保更新完成
            fetched_template = await Template.get_by_name_async(
                template_name=template_name
            )
            assert fetched_template.cpu == 2.0
            assert fetched_template.memory == 4096
            assert fetched_template.sandbox_idle_timeout_in_seconds == 1200
            # 验证 environment_variables 已更新
            assert fetched_template.environment_variables is not None
            assert (
                fetched_template.environment_variables.get("NEW_KEY")
                == "new_value"
            )
            assert (
                fetched_template.environment_variables.get("UPDATED") == "true"
            )

            # 验证时间戳
            assert fetched_template.created_at is not None
            updated_created_at = datetime.datetime.strptime(
                fetched_template.created_at, "%Y-%m-%dT%H:%M:%S.%f%z"
            )
            assert updated_created_at == created_at  # created_at 应该不变

            if fetched_template.last_updated_at:
                last_updated_at = datetime.datetime.strptime(
                    fetched_template.last_updated_at, "%Y-%m-%dT%H:%M:%S.%f%z"
                )
                assert last_updated_at >= created_at  # last_updated_at 应该更新

        finally:
            # 清理资源
            await Template.delete_by_name_async(template_name=template_name)

    async def test_delete_template_async(self, template_name: str):
        """测试删除 Template"""
        # 创建 Template
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name,
                template_type=TemplateType.CODE_INTERPRETER,
                description="E2E 测试 Template",
            )
        )

        # 确认创建成功
        assert template.template_name == template_name

        # 删除 Template
        deleted_template = await Template.delete_by_name_async(
            template_name=template_name
        )
        assert deleted_template is not None

        # 等待删除操作完成（删除是异步的，需要更长时间）
        import asyncio

        await asyncio.sleep(5)

        # 验证删除成功 - 尝试获取应该抛出异常或状态为 DELETING
        try:
            template = await Template.get_by_name_async(
                template_name=template_name
            )
            # 如果还能获取到，状态应该是 DELETING
            assert template.status in [
                "DELETING"
            ], f"Expected DELETING, got {template.status}"
        except ResourceNotExistError:
            # 这是期望的情况 - Template 已经被删除
            pass

    async def test_delete_nonexistent_template_async(self):
        """测试删除不存在的 Template 会抛出异常"""
        with pytest.raises(ResourceNotExistError):
            await Template.delete_by_name_async(
                template_name="nonexistent-template-xyz-12345"
            )

    async def test_template_lifecycle_async(self, template_name: str):
        """测试 Template 的完整生命周期"""
        # 1. 创建
        template = await Template.create_async(
            TemplateInput(
                template_name=template_name,
                template_type=TemplateType.CODE_INTERPRETER,
                description="生命周期测试",
                cpu=2.0,  # CPU 必须至少为 2
                memory=2048,
                environment_variables={"LIFECYCLE": "test"},
            )
        )
        assert template.template_name == template_name
        assert template.template_type == TemplateType.CODE_INTERPRETER

        # 2. 获取
        fetched_template = await Template.get_by_name_async(
            template_name=template_name
        )
        assert fetched_template.template_id == template.template_id
        # 注意：服务端 API 不返回 description 字段
        # assert fetched_template.description == "生命周期测试"

        # 3. 更新
        updated_template = await Template.update_by_name_async(
            template_name=template_name,
            input=TemplateInput(
                template_type=TemplateType.CODE_INTERPRETER,
                description="更新后的描述",
                cpu=2.0,
                memory=4096,
            ),
        )
        # 注意：服务端 API 不返回 description 字段
        # assert updated_template.description == "更新后的描述"
        # 更新操作是异步的，需要等待更新完成再验证
        import asyncio

        await asyncio.sleep(2)
        updated_template = await Template.get_by_name_async(
            template_name=template_name
        )
        assert updated_template.cpu == 2.0
        assert updated_template.memory == 4096

        # 4. 列举
        templates = await Template.list_templates_async()
        template_names = [t.template_name for t in templates]
        assert template_name in template_names

        # 5. 删除
        await Template.delete_by_name_async(template_name=template_name)

        # 等待删除操作完成
        import asyncio

        await asyncio.sleep(5)

        # 6. 验证删除
        try:
            template = await Template.get_by_name_async(
                template_name=template_name
            )
            # 如果还能获取到，状态应该是 DELETING
            assert template.status in [
                "DELETING"
            ], f"Expected DELETING, got {template.status}"
        except ResourceNotExistError:
            # 这是期望的情况 - Template 已经被删除
            pass

    async def test_template_via_sandbox_class_async(self, template_name: str):
        """测试通过 Sandbox 类操作 Template"""
        # 通过 Sandbox 类创建 Template
        template = await Sandbox.create_template_async(
            input=TemplateInput(
                template_name=template_name,
                template_type=TemplateType.CODE_INTERPRETER,
                description="通过 Sandbox 类创建",
                cpu=2.0,
                memory=2048,
            )
        )

        try:
            assert template is not None
            assert template.template_name == template_name

            # 通过 Sandbox 类获取 Template
            fetched_template = await Sandbox.get_template_async(
                template_name=template_name
            )
            assert fetched_template.template_id == template.template_id

            # 通过 Sandbox 类更新 Template
            updated_template = await Sandbox.update_template_async(
                template_name=template_name,
                input=TemplateInput(
                    template_type=TemplateType.CODE_INTERPRETER,
                    description="通过 Sandbox 类更新",
                    cpu=2.0,  # 更新一个可验证的字段
                    memory=4096,
                ),
            )
            # 注意：服务端 API 不返回 description 字段
            # assert updated_template.description == "通过 Sandbox 类更新"
            # 更新操作可能是异步的，需要重新获取验证
            import asyncio

            await asyncio.sleep(2)
            fetched_again = await Sandbox.get_template_async(
                template_name=template_name
            )
            assert fetched_again.cpu == 2.0
            assert fetched_again.memory == 4096

            # 通过 Sandbox 类列举 Templates
            templates = await Sandbox.list_templates_async()
            assert isinstance(templates, list)
            template_names = [t.template_name for t in templates]
            assert template_name in template_names

        finally:
            # 通过 Sandbox 类删除 Template
            await Sandbox.delete_template_async(template_name=template_name)

    async def test_template_validation_browser_disk_size_async(
        self, template_name: str
    ):
        """测试 Browser 类型 Template 的磁盘大小验证"""
        # Browser 类型的 disk_size 必须是 10240
        with pytest.raises(ValueError, match="disk_size should be 10240"):
            await Template.create_async(
                TemplateInput(
                    template_name=template_name,
                    template_type=TemplateType.BROWSER,
                    description="错误的磁盘大小",
                    disk_size=512,  # 错误: Browser 类型必须是 10240
                )
            )

    async def test_template_validation_code_interpreter_network_async(
        self, template_name: str
    ):
        """测试 Code Interpreter 类型 Template 的网络模式验证"""
        # Code Interpreter 类型不能使用 PRIVATE 网络模式
        with pytest.raises(ValueError, match="network_mode cannot be PRIVATE"):
            await Template.create_async(
                TemplateInput(
                    template_name=template_name,
                    template_type=TemplateType.CODE_INTERPRETER,
                    description="错误的网络模式",
                    network_configuration=TemplateNetworkConfiguration(
                        network_mode=TemplateNetworkMode.PRIVATE
                    ),
                )
            )

    async def test_concurrent_template_operations_async(
        self, template_name: str
    ):
        """测试并发创建多个 Templates"""
        template_names = [f"{template_name}-{i}" for i in range(3)]

        try:
            # 并发创建多个 Templates
            create_tasks = [
                Template.create_async(
                    TemplateInput(
                        template_name=name,
                        template_type=TemplateType.CODE_INTERPRETER,
                        description=f"并发测试 Template {i}",
                        cpu=2.0,
                        memory=2048,
                    )
                )
                for i, name in enumerate(template_names)
            ]

            templates = await asyncio.gather(*create_tasks)

            # 验证所有 Templates 都创建成功
            assert len(templates) == 3
            for i, template in enumerate(templates):
                assert template.template_name == template_names[i]
                assert template.template_type == TemplateType.CODE_INTERPRETER

        finally:
            # 并发删除所有 Templates
            for name in template_names:
                await Template.delete_by_name_async(template_name=name)
