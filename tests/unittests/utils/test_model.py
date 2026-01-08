"""测试 agentrun.utils.model 模块 / Test agentrun.utils.model module"""

from pydantic import ValidationError
import pytest

from agentrun.utils.model import (
    BaseModel,
    NetworkConfig,
    NetworkMode,
    PageableInput,
    Status,
    to_camel_case,
)


class TestToCamelCase:
    """测试 to_camel_case 函数"""

    def test_simple_conversion(self):
        """测试简单的转换"""
        assert to_camel_case("hello_world") == "helloWorld"

    def test_multiple_underscores(self):
        """测试多个下划线"""
        assert to_camel_case("access_key_id") == "accessKeyId"

    def test_no_underscore(self):
        """测试没有下划线的情况"""
        assert to_camel_case("hello") == "hello"

    def test_single_char(self):
        """测试单字符"""
        assert to_camel_case("a") == "a"

    def test_empty_string(self):
        """测试空字符串"""
        assert to_camel_case("") == ""


class TestBaseModel:
    """测试 BaseModel 类"""

    def test_from_inner_object(self):
        """测试从 Darabonba 模型对象创建"""

        class MockDaraModel:

            def to_map(self):
                return {"pageNumber": 1, "pageSize": 10}

        obj = MockDaraModel()
        result = PageableInput.from_inner_object(obj)
        assert result.page_number == 1
        assert result.page_size == 10

    def test_from_inner_object_with_extra(self):
        """测试从 Darabonba 模型对象创建并合并额外字段"""

        class MockDaraModel:

            def to_map(self):
                return {"pageNumber": 1}

        obj = MockDaraModel()
        result = PageableInput.from_inner_object(obj, extra={"pageSize": 20})
        assert result.page_number == 1
        assert result.page_size == 20

    def test_from_inner_object_with_validation_error(self):
        """测试验证失败时使用 model_construct"""

        class MockDaraModel:

            def to_map(self):
                # 返回无法验证的数据
                return {"pageNumber": "invalid", "extra_field": "value"}

        obj = MockDaraModel()
        # 不应该抛出异常，应该使用 model_construct
        result = PageableInput.from_inner_object(obj)
        assert result is not None

    def test_update_self(self):
        """测试 update_self 方法"""
        model1 = PageableInput(page_number=1, page_size=10)
        model2 = PageableInput(page_number=2, page_size=20)

        result = model1.update_self(model2)
        assert result.page_number == 2
        assert result.page_size == 20
        assert result is model1

    def test_update_self_with_none(self):
        """测试 update_self 传入 None"""
        model = PageableInput(page_number=1, page_size=10)
        result = model.update_self(None)
        assert result.page_number == 1
        assert result.page_size == 10


class TestNetworkMode:
    """测试 NetworkMode 枚举"""

    def test_public_mode(self):
        """测试公网模式"""
        assert NetworkMode.PUBLIC.value == "PUBLIC"

    def test_private_mode(self):
        """测试私网模式"""
        assert NetworkMode.PRIVATE.value == "PRIVATE"

    def test_mixed_mode(self):
        """测试混合模式"""
        assert NetworkMode.PUBLIC_AND_PRIVATE.value == "PUBLIC_AND_PRIVATE"


class TestNetworkConfig:
    """测试 NetworkConfig 类"""

    def test_default_values(self):
        """测试默认值"""
        config = NetworkConfig()
        assert config.network_mode == NetworkMode.PUBLIC
        assert config.security_group_id is None
        assert config.vpc_id is None
        assert config.vswitch_ids is None

    def test_with_all_fields(self):
        """测试所有字段"""
        config = NetworkConfig(
            network_mode=NetworkMode.PRIVATE,
            security_group_id="sg-123",
            vpc_id="vpc-456",
            vswitch_ids=["vsw-1", "vsw-2"],
        )
        assert config.network_mode == NetworkMode.PRIVATE
        assert config.security_group_id == "sg-123"
        assert config.vpc_id == "vpc-456"
        assert config.vswitch_ids == ["vsw-1", "vsw-2"]

    def test_alias_serialization(self):
        """测试别名序列化"""
        config = NetworkConfig(network_mode=NetworkMode.PUBLIC)
        data = config.model_dump(by_alias=True)
        assert "networkMode" in data


class TestPageableInput:
    """测试 PageableInput 类"""

    def test_default_values(self):
        """测试默认值"""
        input_obj = PageableInput()
        assert input_obj.page_number is None
        assert input_obj.page_size is None

    def test_with_values(self):
        """测试带值"""
        input_obj = PageableInput(page_number=1, page_size=20)
        assert input_obj.page_number == 1
        assert input_obj.page_size == 20


class TestStatus:
    """测试 Status 枚举"""

    def test_all_status_values(self):
        """测试所有状态值"""
        assert Status.CREATING.value == "CREATING"
        assert Status.CREATE_FAILED.value == "CREATE_FAILED"
        assert Status.UPDATING.value == "UPDATING"
        assert Status.UPDATE_FAILED.value == "UPDATE_FAILED"
        assert Status.READY.value == "READY"
        assert Status.DELETING.value == "DELETING"
        assert Status.DELETE_FAILED.value == "DELETE_FAILED"

    def test_is_final_status_ready(self):
        """测试 READY 是最终状态"""
        assert Status.is_final_status(Status.READY) is True

    def test_is_final_status_failed(self):
        """测试失败状态是最终状态"""
        assert Status.is_final_status(Status.CREATE_FAILED) is True
        assert Status.is_final_status(Status.UPDATE_FAILED) is True
        assert Status.is_final_status(Status.DELETE_FAILED) is True

    def test_is_final_status_none(self):
        """测试 None 是最终状态"""
        assert Status.is_final_status(None) is True

    def test_is_final_status_creating(self):
        """测试 CREATING 不是最终状态"""
        assert Status.is_final_status(Status.CREATING) is False

    def test_is_final_status_updating(self):
        """测试 UPDATING 不是最终状态"""
        assert Status.is_final_status(Status.UPDATING) is False

    def test_is_final_status_deleting(self):
        """测试 DELETING 不是最终状态"""
        assert Status.is_final_status(Status.DELETING) is False

    def test_is_final_instance_method(self):
        """测试实例方法 is_final"""
        assert Status.READY.is_final() is True
        assert Status.CREATING.is_final() is False
