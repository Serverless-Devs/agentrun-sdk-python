"""资源基类模板 / Resource Base Template

此模板用于生成资源对象的基类代码。
This template is used to generate base class code for resource objects.
"""

from abc import abstractmethod
import asyncio
import time
from typing import Awaitable, Callable, List, Optional

from typing_extensions import Self

from agentrun.utils.config import Config
from agentrun.utils.exception import DeleteResourceError, ResourceNotExistError
from agentrun.utils.log import logger

from .model import BaseModel, PageableInput, Status


class ResourceBase(BaseModel):
    status: Optional[Status] = None
    _config: Optional[Config] = None

    @classmethod
    @abstractmethod
    async def _list_page_async(
        cls,
        page_input: PageableInput,
        config: Optional[Config] = None,
        **kwargs,
    ) -> list:
        ...

    @classmethod
    async def _list_all_async(
        cls,
        uniq_id_callback: Callable[[Self], str],
        config: Optional[Config] = None,
        **kwargs,
    ) -> list:
        all_results: List[Self] = []
        page = 1
        page_size = 50
        while True:
            page_results = await cls._list_page_async(
                PageableInput(
                    page_number=page,
                    page_size=page_size,
                ),
                config=config,
                **kwargs,
            )
            page += 1
            all_results.extend(page_results)  # type: ignore
            if len(page_results) < page_size:
                break

        result_set = set()
        results: list = []
        for item in all_results:
            uniq_id = uniq_id_callback(item)
            if uniq_id not in result_set:
                result_set.add(uniq_id)
                results.append(item)

        return results

    @abstractmethod
    async def refresh_async(self, config: Optional[Config] = None) -> Self:
        ...

    @abstractmethod
    async def delete_async(self, config: Optional[Config] = None) -> Self:
        ...

    async def __wait_until_async(
        self,
        check_finished_callback_async: Callable[[Self], Awaitable[bool]],
        interval_seconds: int = 5,
        timeout_seconds: int = 300,
    ) -> Self:
        """等待智能体运行时进入就绪状态"""

        start_time = time.time()
        while True:
            if await check_finished_callback_async(self):
                return self

            if time.time() - start_time > timeout_seconds:
                raise TimeoutError("等待就绪超时")

            await asyncio.sleep(interval_seconds)

    async def wait_until_ready_or_failed_async(
        self,
        callback: Optional[Callable[[Self], None]] = None,
        interval_seconds: int = 5,
        timeout_seconds: int = 300,
    ):
        """等待智能体运行时进入就绪状态"""

        async def check_ready_callback(resource: Self) -> bool:
            await resource.refresh_async()
            if callback:
                callback(resource)
            logger.debug("当前状态：%s", resource.status)

            return Status.is_final_status(resource.status)

        await self.__wait_until_async(
            check_ready_callback,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
        )

    async def delete_and_wait_until_finished_async(
        self,
        callback: Optional[Callable[[Self], None]] = None,
        interval_seconds: int = 5,
        timeout_seconds: int = 300,
    ):
        """等待智能体运行时被删除"""
        try:
            await self.delete_async()
        except ResourceNotExistError:
            return

        async def check_deleted_callback(resource: Self) -> bool:
            try:
                await resource.refresh_async()
                if callback:
                    callback(resource)
            except ResourceNotExistError:
                return True

            if resource.status == Status.DELETING:
                return False

            raise DeleteResourceError(f"Resource status is {resource.status}")

        await self.__wait_until_async(
            check_deleted_callback,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
        )

    def set_config(self, config: Config) -> Self:
        """设置配置

        Args:
            config: 配置

        Returns:
            Self: 当前对象
        """
        self._config = config
        return self
