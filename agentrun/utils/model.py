"""数据模型基类模块 / Data Model Base Module

此模块定义所有数据模型的基类和通用配置。
This module defines base classes and common configurations for all data models.
"""

from enum import Enum
from typing import List, Optional, Union

from darabonba.model import DaraModel
from pydantic import AliasGenerator
from pydantic import BaseModel as PydanticModel
from pydantic import ConfigDict, Field, ValidationError
from Tea.model import TeaModel
from typing_extensions import Self

from agentrun.utils.log import logger


def to_camel_case(field_name: str) -> str:
    """将下划线命名转换为驼峰命名 / Convert snake_case to camelCase

    Args:
        field_name: 下划线命名的字段名 / Field name in snake_case

    Returns:
        str: 驼峰命名的字段名 / Field name in camelCase

    Examples:
        >>> to_camel_case("hello_world")
        'helloWorld'
        >>> to_camel_case("access_key_id")
        'accessKeyId'
    """
    if "_" not in field_name:
        return field_name
    parts = field_name.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class BaseModel(PydanticModel):
    model_config = ConfigDict(
        use_attribute_docstrings=True,
        validate_by_name=True,
        validate_by_alias=False,
        serialize_by_alias=True,
        use_enum_values=True,
        extra="allow",
        arbitrary_types_allowed=True,
        alias_generator=AliasGenerator(
            validation_alias=to_camel_case,
            serialization_alias=to_camel_case,
        ),
    )

    @classmethod
    def from_inner_object(
        cls, obj: Union[DaraModel, TeaModel], extra: Optional[dict] = None
    ) -> Self:
        """从 Darabonba 模型对象创建 Pydantic 模型对象，可选地合并额外的字段"""
        logger.debug(
            "before parse object obj=%s, extra=%s", obj.to_map(), extra
        )

        d = {**extra, **obj.to_map()} if extra else obj.to_map()  # type: ignore
        try:
            result = cls.model_validate(d, by_alias=True)
        except ValidationError as e:
            logger.warning("validate type failed, %s", e)
            result = cls.model_construct(None, **d)  # type: ignore
        return result

    def update_self(self, other: Optional["BaseModel"]) -> Self:
        """更新自身属性 / Update self attributes

        用另一个模型对象的属性更新当前对象。
        Update current object with attributes from another model object.

        Args:
            other: 另一个模型对象,可选 / Another model object, optional

        Returns:
            Self: 更新后的自身 / Updated self
        """
        if other is not None:
            self.__dict__.update(other.__dict__)

        return self


class NetworkMode(str, Enum):
    """网络访问模式 / Network Access Mode

    定义 Agent Runtime 的网络访问模式。
    Defines network access modes for Agent Runtime.
    """

    PUBLIC = "PUBLIC"
    """公网模式 / Public network mode"""
    PRIVATE = "PRIVATE"
    """私网模式 / Private network mode"""
    PUBLIC_AND_PRIVATE = "PUBLIC_AND_PRIVATE"
    """公私网混合模式 / Public and private network mode"""


class NetworkConfig(BaseModel):
    """网络配置 / Network Configuration

    定义 Agent Runtime 的网络配置。
    Defines network configuration for Agent Runtime.
    """

    network_mode: NetworkMode = Field(
        alias="networkMode", default=NetworkMode.PUBLIC
    )
    """网络访问模式 / Network access mode"""
    security_group_id: Optional[str] = Field(
        alias="securityGroupId", default=None
    )
    """安全组 ID"""
    vpc_id: Optional[str] = Field(alias="vpcId", default=None)
    """私有网络 ID"""
    vswitch_ids: Optional[List[str]] = Field(alias="vswitchIds", default=None)
    """私有网络交换机 ID 列表"""


class PageableInput(BaseModel):
    page_number: Optional[int] = None
    """页码"""
    page_size: Optional[int] = None
    """页大小"""


class Status(str, Enum):
    """Agent Runtime 状态"""

    CREATING = "CREATING"
    CREATE_FAILED = "CREATE_FAILED"
    UPDATING = "UPDATING"
    UPDATE_FAILED = "UPDATE_FAILED"
    READY = "READY"
    DELETING = "DELETING"
    DELETE_FAILED = "DELETE_FAILED"

    @staticmethod
    def is_final_status(status: Optional["Status"] = None) -> bool:
        """判断状态是否为最终状态"""
        return status in {
            None,
            Status.READY,
            Status.CREATE_FAILED,
            Status.UPDATE_FAILED,
            Status.DELETE_FAILED,
        }

    def is_final(self) -> bool:
        """判断状态是否为最终状态"""
        return Status.is_final_status(self)


__all__ = [
    "BaseModel",
    "Field",
    "NetworkMode",
    "NetworkConfig",
    "PageableInput",
    "Status",
]
