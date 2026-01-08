"""测试 agentrun.utils.resource 模块 / Test agentrun.utils.resource module"""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentrun.utils.config import Config
from agentrun.utils.exception import DeleteResourceError, ResourceNotExistError
from agentrun.utils.model import PageableInput, Status
from agentrun.utils.resource import ResourceBase


class MockResource(ResourceBase):
    """用于测试的 Mock 资源类"""

    resource_id: Optional[str] = None
    resource_name: Optional[str] = None

    @classmethod
    async def _list_page_async(
        cls,
        page_input: PageableInput,
        config: Optional[Config] = None,
        **kwargs,
    ) -> list:
        # 模拟分页返回
        if page_input.page_number == 1:
            return [
                MockResource(resource_id="1", status=Status.READY),
                MockResource(resource_id="2", status=Status.READY),
            ]
        return []

    @classmethod
    def _list_page(
        cls,
        page_input: PageableInput,
        config: Optional[Config] = None,
        **kwargs,
    ) -> list:
        # 模拟分页返回
        if page_input.page_number == 1:
            return [
                MockResource(resource_id="1", status=Status.READY),
                MockResource(resource_id="2", status=Status.READY),
            ]
        return []

    async def refresh_async(self, config: Optional[Config] = None):
        return self

    def refresh(self, config: Optional[Config] = None):
        return self

    async def delete_async(self, config: Optional[Config] = None):
        return self

    def delete(self, config: Optional[Config] = None):
        return self


class TestResourceBaseListAll:
    """测试 ResourceBase._list_all 方法"""

    def test_list_all_sync(self):
        """测试同步列出所有资源"""
        results = MockResource._list_all(
            uniq_id_callback=lambda r: r.resource_id or ""
        )
        assert len(results) == 2
        assert results[0].resource_id == "1"
        assert results[1].resource_id == "2"

    @pytest.mark.asyncio
    async def test_list_all_async(self):
        """测试异步列出所有资源"""
        results = await MockResource._list_all_async(
            uniq_id_callback=lambda r: r.resource_id or ""
        )
        assert len(results) == 2
        assert results[0].resource_id == "1"
        assert results[1].resource_id == "2"

    def test_list_all_deduplicates(self):
        """测试去重功能"""

        class DuplicateResource(MockResource):

            @classmethod
            def _list_page(
                cls,
                page_input: PageableInput,
                config: Optional[Config] = None,
                **kwargs,
            ) -> list:
                if page_input.page_number == 1:
                    return [
                        DuplicateResource(resource_id="1", status=Status.READY),
                        DuplicateResource(resource_id="1", status=Status.READY),
                        DuplicateResource(resource_id="2", status=Status.READY),
                    ]
                return []

        results = DuplicateResource._list_all(
            uniq_id_callback=lambda r: r.resource_id or ""
        )
        assert len(results) == 2

    def test_list_all_with_config(self):
        """测试带配置的列表"""
        config = Config(access_key_id="test")
        results = MockResource._list_all(
            uniq_id_callback=lambda r: r.resource_id or "",
            config=config,
        )
        assert len(results) == 2

    def test_list_all_with_exact_page_size(self):
        """测试分页结果恰好等于 page_size 时继续分页"""

        class ExactPageSizeResource(MockResource):

            @classmethod
            def _list_page(
                cls,
                page_input: PageableInput,
                config: Optional[Config] = None,
                **kwargs,
            ) -> list:
                # 第一页返回恰好 50 条记录（等于 page_size）
                if page_input.page_number == 1:
                    return [
                        ExactPageSizeResource(
                            resource_id=str(i), status=Status.READY
                        )
                        for i in range(50)
                    ]
                # 第二页返回空，表示没有更多数据
                return []

        results = ExactPageSizeResource._list_all(
            uniq_id_callback=lambda r: r.resource_id or ""
        )
        assert len(results) == 50

    @pytest.mark.asyncio
    async def test_list_all_async_with_exact_page_size(self):
        """测试异步分页结果恰好等于 page_size 时继续分页"""

        class ExactPageSizeResourceAsync(MockResource):

            @classmethod
            async def _list_page_async(
                cls,
                page_input: PageableInput,
                config: Optional[Config] = None,
                **kwargs,
            ) -> list:
                # 第一页返回恰好 50 条记录（等于 page_size）
                if page_input.page_number == 1:
                    return [
                        ExactPageSizeResourceAsync(
                            resource_id=str(i), status=Status.READY
                        )
                        for i in range(50)
                    ]
                # 第二页返回空，表示没有更多数据
                return []

        results = await ExactPageSizeResourceAsync._list_all_async(
            uniq_id_callback=lambda r: r.resource_id or ""
        )
        assert len(results) == 50


class TestResourceBaseWaitUntilReadyOrFailed:
    """测试 ResourceBase.wait_until_ready_or_failed 方法"""

    def test_wait_until_ready_immediately(self):
        """测试资源已就绪时立即返回"""
        resource = MockResource(status=Status.READY)
        callback_called = []

        resource.wait_until_ready_or_failed(
            callback=lambda r: callback_called.append(r),
            interval_seconds=1,
            timeout_seconds=5,
        )

        assert len(callback_called) == 1

    def test_wait_until_ready_with_transition(self):
        """测试资源状态转换"""
        call_count = [0]

        class TransitionResource(MockResource):

            def refresh(self, config: Optional[Config] = None):
                call_count[0] += 1
                if call_count[0] >= 2:
                    self.status = Status.READY
                return self

        resource = TransitionResource(status=Status.CREATING)
        resource.wait_until_ready_or_failed(
            interval_seconds=0.1,
            timeout_seconds=5,
        )
        assert resource.status == Status.READY

    def test_wait_until_ready_timeout(self):
        """测试等待超时"""

        class NeverReadyResource(MockResource):

            def refresh(self, config: Optional[Config] = None):
                self.status = Status.CREATING
                return self

        resource = NeverReadyResource(status=Status.CREATING)
        with pytest.raises(TimeoutError):
            resource.wait_until_ready_or_failed(
                interval_seconds=0.1,
                timeout_seconds=0.3,
            )

    @pytest.mark.asyncio
    async def test_wait_until_ready_async_immediately(self):
        """测试异步资源已就绪时立即返回"""
        resource = MockResource(status=Status.READY)
        callback_called = []

        await resource.wait_until_ready_or_failed_async(
            callback=lambda r: callback_called.append(r),
            interval_seconds=1,
            timeout_seconds=5,
        )

        assert len(callback_called) == 1

    @pytest.mark.asyncio
    async def test_wait_until_ready_async_timeout(self):
        """测试异步等待超时"""

        class NeverReadyResourceAsync(MockResource):

            async def refresh_async(self, config: Optional[Config] = None):
                self.status = Status.CREATING
                return self

        resource = NeverReadyResourceAsync(status=Status.CREATING)
        with pytest.raises(TimeoutError):
            await resource.wait_until_ready_or_failed_async(
                interval_seconds=0.1,
                timeout_seconds=0.3,
            )


class TestResourceBaseDeleteAndWait:
    """测试 ResourceBase.delete_and_wait_until_finished 方法"""

    def test_delete_already_not_exist(self):
        """测试删除已不存在的资源"""

        class NotExistResource(MockResource):

            def delete(self, config: Optional[Config] = None):
                raise ResourceNotExistError("MockResource", "1")

        resource = NotExistResource(resource_id="1")
        # 不应该抛出异常
        resource.delete_and_wait_until_finished(
            interval_seconds=0.1, timeout_seconds=1
        )

    def test_delete_and_wait_success(self):
        """测试删除并等待成功"""
        refresh_count = [0]

        class DeletingResource(MockResource):

            def refresh(self, config: Optional[Config] = None):
                refresh_count[0] += 1
                if refresh_count[0] >= 2:
                    raise ResourceNotExistError("MockResource", "1")
                self.status = Status.DELETING
                return self

        resource = DeletingResource(resource_id="1", status=Status.READY)
        resource.delete_and_wait_until_finished(
            interval_seconds=0.1,
            timeout_seconds=5,
        )
        assert refresh_count[0] >= 2

    def test_delete_and_wait_with_callback(self):
        """测试删除并等待带回调"""
        callbacks = []
        refresh_count = [0]

        class DeletingResource(MockResource):

            def refresh(self, config: Optional[Config] = None):
                refresh_count[0] += 1
                if refresh_count[0] >= 2:
                    raise ResourceNotExistError("MockResource", "1")
                self.status = Status.DELETING
                return self

        resource = DeletingResource(resource_id="1", status=Status.READY)
        resource.delete_and_wait_until_finished(
            callback=lambda r: callbacks.append(r),
            interval_seconds=0.1,
            timeout_seconds=5,
        )
        assert len(callbacks) >= 1

    def test_delete_and_wait_error_status(self):
        """测试删除后状态异常"""

        class FailedDeleteResource(MockResource):

            def refresh(self, config: Optional[Config] = None):
                self.status = Status.DELETE_FAILED
                return self

        resource = FailedDeleteResource(resource_id="1", status=Status.READY)
        with pytest.raises(DeleteResourceError):
            resource.delete_and_wait_until_finished(
                interval_seconds=0.1,
                timeout_seconds=5,
            )

    @pytest.mark.asyncio
    async def test_delete_async_already_not_exist(self):
        """测试异步删除已不存在的资源"""

        class NotExistResourceAsync(MockResource):

            async def delete_async(self, config: Optional[Config] = None):
                raise ResourceNotExistError("MockResource", "1")

        resource = NotExistResourceAsync(resource_id="1")
        # 不应该抛出异常
        await resource.delete_and_wait_until_finished_async(
            interval_seconds=0.1, timeout_seconds=1
        )

    @pytest.mark.asyncio
    async def test_delete_async_and_wait_success(self):
        """测试异步删除并等待成功"""
        refresh_count = [0]

        class DeletingResourceAsync(MockResource):

            async def refresh_async(self, config: Optional[Config] = None):
                refresh_count[0] += 1
                if refresh_count[0] >= 2:
                    raise ResourceNotExistError("MockResource", "1")
                self.status = Status.DELETING
                return self

        resource = DeletingResourceAsync(resource_id="1", status=Status.READY)
        await resource.delete_and_wait_until_finished_async(
            interval_seconds=0.1,
            timeout_seconds=5,
        )
        assert refresh_count[0] >= 2

    @pytest.mark.asyncio
    async def test_delete_async_and_wait_error_status(self):
        """测试异步删除后状态异常"""

        class FailedDeleteResourceAsync(MockResource):

            async def refresh_async(self, config: Optional[Config] = None):
                self.status = Status.DELETE_FAILED
                return self

        resource = FailedDeleteResourceAsync(
            resource_id="1", status=Status.READY
        )
        with pytest.raises(DeleteResourceError):
            await resource.delete_and_wait_until_finished_async(
                interval_seconds=0.1,
                timeout_seconds=5,
            )


class TestResourceBaseSetConfig:
    """测试 ResourceBase.set_config 方法"""

    def test_set_config(self):
        """测试设置配置"""
        resource = MockResource()
        config = Config(access_key_id="test")
        result = resource.set_config(config)
        assert result is resource
        assert resource._config is config
