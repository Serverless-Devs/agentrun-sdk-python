"""测试 agentrun.sandbox.model 模块 / Test agentrun.sandbox.model module"""

import pytest

from agentrun.sandbox.model import (
    CodeLanguage,
    ListSandboxesInput,
    ListSandboxesOutput,
    NASConfig,
    NASMountConfig,
    OSSMountConfig,
    OSSMountPoint,
    PageableInput,
    PolarFsConfig,
    PolarFsMountConfig,
    SandboxInput,
    TemplateArmsConfiguration,
    TemplateContainerConfiguration,
    TemplateCredentialConfiguration,
    TemplateInput,
    TemplateLogConfiguration,
    TemplateMcpOptions,
    TemplateMcpState,
    TemplateNetworkConfiguration,
    TemplateNetworkMode,
    TemplateOssConfiguration,
    TemplateOSSPermission,
    TemplateType,
)

# ==================== 枚举测试 ====================


class TestTemplateOSSPermission:

    def test_read_write_value(self):
        assert TemplateOSSPermission.READ_WRITE.value == "READ_WRITE"

    def test_read_only_value(self):
        assert TemplateOSSPermission.READ_ONLY.value == "READ_ONLY"

    def test_is_string_enum(self):
        assert isinstance(TemplateOSSPermission.READ_WRITE, str)
        assert TemplateOSSPermission.READ_WRITE == "READ_WRITE"


class TestTemplateType:

    def test_code_interpreter_value(self):
        assert TemplateType.CODE_INTERPRETER.value == "CodeInterpreter"

    def test_browser_value(self):
        assert TemplateType.BROWSER.value == "Browser"

    def test_aio_value(self):
        assert TemplateType.AIO.value == "AllInOne"

    def test_custom_value(self):
        assert TemplateType.CUSTOM.value == "CustomImage"

    def test_is_string_enum(self):
        assert isinstance(TemplateType.CODE_INTERPRETER, str)


class TestTemplateNetworkMode:

    def test_public_value(self):
        assert TemplateNetworkMode.PUBLIC.value == "PUBLIC"

    def test_private_value(self):
        assert TemplateNetworkMode.PRIVATE.value == "PRIVATE"

    def test_public_and_private_value(self):
        assert (
            TemplateNetworkMode.PUBLIC_AND_PRIVATE.value == "PUBLIC_AND_PRIVATE"
        )


class TestCodeLanguage:

    def test_python_value(self):
        assert CodeLanguage.PYTHON.value == "python"

    def test_is_string_enum(self):
        assert isinstance(CodeLanguage.PYTHON, str)
        assert CodeLanguage.PYTHON == "python"


# ==================== NAS 配置测试 ====================


class TestNASMountConfig:

    def test_create_full(self):
        config = NASMountConfig(
            enable_tls=True,
            mount_dir="/mnt/nas",
            server_addr="nas-server.example.com",
        )
        assert config.enable_tls is True
        assert config.mount_dir == "/mnt/nas"
        assert config.server_addr == "nas-server.example.com"

    def test_optional_fields(self):
        config = NASMountConfig()
        assert config.enable_tls is None
        assert config.mount_dir is None
        assert config.server_addr is None

    def test_partial_fields(self):
        config = NASMountConfig(mount_dir="/mnt/data")
        assert config.mount_dir == "/mnt/data"
        assert config.enable_tls is None

    def test_model_dump(self):
        config = NASMountConfig(
            enable_tls=True, mount_dir="/mnt/nas", server_addr="addr"
        )
        data = config.model_dump(by_alias=True)
        assert "enableTls" in data
        assert "mountDir" in data
        assert "serverAddr" in data


class TestNASConfig:

    def test_create_full(self):
        mount = NASMountConfig(mount_dir="/mnt/nas", server_addr="addr")
        config = NASConfig(group_id=1000, mount_points=[mount], user_id=1000)
        assert config.group_id == 1000
        assert config.user_id == 1000
        assert len(config.mount_points) == 1

    def test_optional_fields(self):
        config = NASConfig()
        assert config.group_id is None
        assert config.mount_points is None
        assert config.user_id is None


# ==================== OSS 配置测试 ====================


class TestOSSMountPoint:

    def test_create_full(self):
        point = OSSMountPoint(
            bucket_name="my-bucket",
            bucket_path="/data",
            endpoint="oss-cn-hangzhou.aliyuncs.com",
            mount_dir="/mnt/oss",
            read_only=True,
        )
        assert point.bucket_name == "my-bucket"
        assert point.bucket_path == "/data"
        assert point.endpoint == "oss-cn-hangzhou.aliyuncs.com"
        assert point.mount_dir == "/mnt/oss"
        assert point.read_only is True

    def test_optional_fields(self):
        point = OSSMountPoint()
        assert point.bucket_name is None
        assert point.bucket_path is None
        assert point.endpoint is None
        assert point.mount_dir is None
        assert point.read_only is None

    def test_model_dump(self):
        point = OSSMountPoint(bucket_name="b", mount_dir="/mnt")
        data = point.model_dump(by_alias=True)
        assert "bucketName" in data
        assert "mountDir" in data


class TestOSSMountConfig:

    def test_create_with_mount_points(self):
        point = OSSMountPoint(bucket_name="my-bucket", mount_dir="/mnt/oss")
        config = OSSMountConfig(mount_points=[point])
        assert len(config.mount_points) == 1
        assert config.mount_points[0].bucket_name == "my-bucket"

    def test_optional_fields(self):
        config = OSSMountConfig()
        assert config.mount_points is None

    def test_multiple_mount_points(self):
        points = [
            OSSMountPoint(bucket_name="bucket-1"),
            OSSMountPoint(bucket_name="bucket-2"),
        ]
        config = OSSMountConfig(mount_points=points)
        assert len(config.mount_points) == 2


# ==================== PolarFS 配置测试 ====================


class TestPolarFsMountConfig:

    def test_create_full(self):
        config = PolarFsMountConfig(
            instance_id="inst-123",
            mount_dir="/mnt/polar",
            remote_dir="/remote/data",
        )
        assert config.instance_id == "inst-123"
        assert config.mount_dir == "/mnt/polar"
        assert config.remote_dir == "/remote/data"

    def test_optional_fields(self):
        config = PolarFsMountConfig()
        assert config.instance_id is None
        assert config.mount_dir is None
        assert config.remote_dir is None


class TestPolarFsConfig:

    def test_create_full(self):
        mount = PolarFsMountConfig(instance_id="inst-123", mount_dir="/mnt")
        config = PolarFsConfig(
            group_id=1000, mount_points=[mount], user_id=1000
        )
        assert config.group_id == 1000
        assert config.user_id == 1000
        assert len(config.mount_points) == 1

    def test_optional_fields(self):
        config = PolarFsConfig()
        assert config.group_id is None
        assert config.mount_points is None
        assert config.user_id is None


# ==================== 模板配置测试 ====================


class TestTemplateNetworkConfiguration:

    def test_default_values(self):
        config = TemplateNetworkConfiguration()
        assert config.network_mode == TemplateNetworkMode.PUBLIC

    def test_create_full(self):
        config = TemplateNetworkConfiguration(
            network_mode=TemplateNetworkMode.PRIVATE,
            security_group_id="sg-123",
            vpc_id="vpc-456",
            vswitch_ids=["vsw-789"],
        )
        assert config.network_mode == TemplateNetworkMode.PRIVATE
        assert config.security_group_id == "sg-123"
        assert config.vpc_id == "vpc-456"
        assert config.vswitch_ids == ["vsw-789"]

    def test_optional_fields(self):
        config = TemplateNetworkConfiguration()
        assert config.security_group_id is None
        assert config.vpc_id is None
        assert config.vswitch_ids is None


class TestTemplateOssConfiguration:

    def test_create(self):
        config = TemplateOssConfiguration(
            bucket_name="my-bucket",
            mount_point="/mnt/oss",
            prefix="data/",
            region="cn-hangzhou",
        )
        assert config.bucket_name == "my-bucket"
        assert config.mount_point == "/mnt/oss"
        assert config.prefix == "data/"
        assert config.region == "cn-hangzhou"
        assert config.permission == TemplateOSSPermission.READ_WRITE

    def test_create_with_read_only(self):
        config = TemplateOssConfiguration(
            bucket_name="b",
            mount_point="/mnt",
            prefix="/",
            region="cn-shanghai",
            permission=TemplateOSSPermission.READ_ONLY,
        )
        assert config.permission == TemplateOSSPermission.READ_ONLY


class TestTemplateLogConfiguration:

    def test_create_full(self):
        config = TemplateLogConfiguration(
            project="my-project", logstore="my-logstore"
        )
        assert config.project == "my-project"
        assert config.logstore == "my-logstore"

    def test_optional_fields(self):
        config = TemplateLogConfiguration()
        assert config.project is None
        assert config.logstore is None


class TestTemplateCredentialConfiguration:

    def test_create(self):
        config = TemplateCredentialConfiguration(
            credential_name="my-credential"
        )
        assert config.credential_name == "my-credential"

    def test_optional_fields(self):
        config = TemplateCredentialConfiguration()
        assert config.credential_name is None


class TestTemplateArmsConfiguration:

    def test_create_full(self):
        config = TemplateArmsConfiguration(
            arms_license_key="key-123", enable_arms=True
        )
        assert config.arms_license_key == "key-123"
        assert config.enable_arms is True

    def test_enable_arms_required(self):
        with pytest.raises(Exception):
            TemplateArmsConfiguration()  # type: ignore

    def test_disabled_arms(self):
        config = TemplateArmsConfiguration(enable_arms=False)
        assert config.enable_arms is False
        assert config.arms_license_key is None


class TestTemplateContainerConfiguration:

    def test_create_full(self):
        config = TemplateContainerConfiguration(
            image="registry.example.com/my-image:latest",
            command=["python", "app.py"],
            acr_instance_id="acr-123",
            image_registry_type="enterprise",
            port=8080,
        )
        assert config.image == "registry.example.com/my-image:latest"
        assert config.command == ["python", "app.py"]
        assert config.acr_instance_id == "acr-123"
        assert config.image_registry_type == "enterprise"
        assert config.port == 8080

    def test_optional_fields(self):
        config = TemplateContainerConfiguration()
        assert config.image is None
        assert config.command is None
        assert config.port is None


class TestTemplateMcpOptions:

    def test_create_full(self):
        config = TemplateMcpOptions(
            enabled_tools=["tool1", "tool2"], transport="sse"
        )
        assert config.enabled_tools == ["tool1", "tool2"]
        assert config.transport == "sse"

    def test_optional_fields(self):
        config = TemplateMcpOptions()
        assert config.enabled_tools is None
        assert config.transport is None


class TestTemplateMcpState:

    def test_create_full(self):
        state = TemplateMcpState(
            access_endpoint="https://mcp.example.com",
            status="READY",
            status_reason="OK",
        )
        assert state.access_endpoint == "https://mcp.example.com"
        assert state.status == "READY"
        assert state.status_reason == "OK"

    def test_optional_fields(self):
        state = TemplateMcpState()
        assert state.access_endpoint is None
        assert state.status is None
        assert state.status_reason is None


# ==================== TemplateInput 测试 ====================


class TestTemplateInput:

    def test_create_code_interpreter(self):
        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="my-ci-template",
        )
        assert input_obj.template_type == TemplateType.CODE_INTERPRETER
        assert input_obj.template_name == "my-ci-template"
        assert input_obj.cpu == 2.0
        assert input_obj.memory == 4096
        assert input_obj.disk_size == 512

    def test_create_browser_default_disk_size(self):
        input_obj = TemplateInput(
            template_type=TemplateType.BROWSER,
            template_name="my-browser-template",
        )
        assert input_obj.disk_size == 10240

    def test_create_aio_default_values(self):
        input_obj = TemplateInput(
            template_type=TemplateType.AIO,
            template_name="my-aio-template",
        )
        assert input_obj.disk_size == 10240
        assert input_obj.cpu == 4.0
        assert input_obj.memory == 8192

    def test_create_custom_default_disk_size(self):
        input_obj = TemplateInput(
            template_type=TemplateType.CUSTOM,
            template_name="my-custom-template",
        )
        assert input_obj.disk_size == 512

    def test_explicit_disk_size_not_overridden(self):
        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test",
            disk_size=1024,
        )
        assert input_obj.disk_size == 1024

    def test_browser_wrong_disk_size_raises(self):
        with pytest.raises(ValueError, match="disk_size should be 10240"):
            TemplateInput(
                template_type=TemplateType.BROWSER,
                template_name="test",
                disk_size=512,
            )

    def test_aio_wrong_disk_size_raises(self):
        with pytest.raises(ValueError, match="disk_size should be 10240"):
            TemplateInput(
                template_type=TemplateType.AIO,
                template_name="test",
                disk_size=512,
            )

    def test_default_network_configuration(self):
        input_obj = TemplateInput(template_type=TemplateType.CODE_INTERPRETER)
        assert input_obj.network_configuration is not None
        assert (
            input_obj.network_configuration.network_mode
            == TemplateNetworkMode.PUBLIC
        )

    def test_default_idle_timeout(self):
        input_obj = TemplateInput(template_type=TemplateType.CODE_INTERPRETER)
        assert input_obj.sandbox_idle_timeout_in_seconds == 1800

    def test_default_ttl(self):
        input_obj = TemplateInput(template_type=TemplateType.CODE_INTERPRETER)
        assert input_obj.sandbox_ttlin_seconds == 21600

    def test_default_concurrency(self):
        input_obj = TemplateInput(template_type=TemplateType.CODE_INTERPRETER)
        assert input_obj.share_concurrency_limit_per_sandbox == 200

    def test_all_optional_fields(self):
        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="full-test",
            cpu=4.0,
            memory=8192,
            execution_role_arn="acs:ram::123:role/my-role",
            sandbox_idle_timeout_in_seconds=3600,
            description="Test template",
            environment_variables={"KEY": "VALUE"},
            oss_configuration=[
                TemplateOssConfiguration(
                    bucket_name="b",
                    mount_point="/mnt",
                    prefix="/",
                    region="cn-hangzhou",
                )
            ],
            log_configuration=TemplateLogConfiguration(
                project="p", logstore="l"
            ),
            credential_configuration=TemplateCredentialConfiguration(
                credential_name="cred"
            ),
            arms_configuration=TemplateArmsConfiguration(enable_arms=False),
            container_configuration=TemplateContainerConfiguration(image="img"),
            allow_anonymous_manage=True,
        )
        assert input_obj.description == "Test template"
        assert input_obj.environment_variables == {"KEY": "VALUE"}
        assert input_obj.oss_configuration is not None
        assert input_obj.log_configuration is not None
        assert input_obj.credential_configuration is not None
        assert input_obj.arms_configuration is not None
        assert input_obj.container_configuration is not None
        assert input_obj.allow_anonymous_manage is True

    def test_model_dump_alias(self):
        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            template_name="test",
        )
        data = input_obj.model_dump(by_alias=True)
        assert "templateType" in data
        assert "templateName" in data

    def test_auto_generated_template_name(self):
        input_obj = TemplateInput(template_type=TemplateType.CODE_INTERPRETER)
        assert input_obj.template_name is not None
        assert input_obj.template_name.startswith("sandbox_template_")

    def test_set_disk_size_default_with_explicit_disk_size(self):
        input_obj = TemplateInput(
            template_type=TemplateType.CODE_INTERPRETER,
            disk_size=2048,
        )
        assert input_obj.disk_size == 2048

    def test_set_disk_size_default_browser_enum(self):
        input_obj = TemplateInput(
            template_type=TemplateType.BROWSER,
        )
        assert input_obj.disk_size == 10240

    def test_set_disk_size_default_aio_enum(self):
        input_obj = TemplateInput(
            template_type=TemplateType.AIO,
        )
        assert input_obj.disk_size == 10240
        assert input_obj.cpu == 4.0
        assert input_obj.memory == 8192

    def test_model_validator_with_camel_case_dict(self):
        input_obj = TemplateInput.model_validate({
            "template_type": "CodeInterpreter",
            "disk_size": 2048,
        })
        assert input_obj.disk_size == 2048

    def test_model_validator_browser_from_dict(self):
        input_obj = TemplateInput.model_validate({"template_type": "Browser"})
        assert input_obj.disk_size == 10240

    def test_model_validator_aio_from_dict(self):
        input_obj = TemplateInput.model_validate({"template_type": "AllInOne"})
        assert input_obj.disk_size == 10240
        assert input_obj.cpu == 4.0
        assert input_obj.memory == 8192


# ==================== SandboxInput 测试 ====================


class TestSandboxInput:

    def test_create_minimal(self):
        input_obj = SandboxInput(template_name="my-template")
        assert input_obj.template_name == "my-template"
        assert input_obj.sandbox_idle_timeout_seconds == 600

    def test_create_full(self):
        input_obj = SandboxInput(
            template_name="my-template",
            sandbox_idle_timeout_seconds=1200,
            sandbox_id="sandbox-123",
            nas_config=NASConfig(group_id=1000),
            oss_mount_config=OSSMountConfig(
                mount_points=[OSSMountPoint(bucket_name="b")]
            ),
            polar_fs_config=PolarFsConfig(user_id=1000),
        )
        assert input_obj.sandbox_id == "sandbox-123"
        assert input_obj.sandbox_idle_timeout_seconds == 1200
        assert input_obj.nas_config is not None
        assert input_obj.oss_mount_config is not None
        assert input_obj.polar_fs_config is not None

    def test_optional_fields(self):
        input_obj = SandboxInput(template_name="t")
        assert input_obj.sandbox_id is None
        assert input_obj.nas_config is None
        assert input_obj.oss_mount_config is None
        assert input_obj.polar_fs_config is None


# ==================== ListSandboxesInput 测试 ====================


class TestListSandboxesInput:

    def test_default_values(self):
        input_obj = ListSandboxesInput()
        assert input_obj.max_results == 10
        assert input_obj.next_token is None
        assert input_obj.status is None
        assert input_obj.template_name is None
        assert input_obj.template_type is None

    def test_create_full(self):
        input_obj = ListSandboxesInput(
            max_results=20,
            next_token="token-123",
            status="RUNNING",
            template_name="my-template",
            template_type=TemplateType.CODE_INTERPRETER,
        )
        assert input_obj.max_results == 20
        assert input_obj.next_token == "token-123"
        assert input_obj.status == "RUNNING"
        assert input_obj.template_name == "my-template"
        assert input_obj.template_type == TemplateType.CODE_INTERPRETER


# ==================== ListSandboxesOutput 测试 ====================


class TestListSandboxesOutput:

    def test_create_empty(self):
        output = ListSandboxesOutput(sandboxes=[])
        assert len(output.sandboxes) == 0
        assert output.next_token is None

    def test_create_with_next_token(self):
        output = ListSandboxesOutput(sandboxes=[], next_token="next-page-token")
        assert output.next_token == "next-page-token"


# ==================== PageableInput 测试 ====================


class TestPageableInput:

    def test_default_values(self):
        input_obj = PageableInput()
        assert input_obj.page_number == 1
        assert input_obj.page_size == 10

    def test_create_custom(self):
        input_obj = PageableInput(
            page_number=3,
            page_size=20,
            template_type=TemplateType.BROWSER,
        )
        assert input_obj.page_number == 3
        assert input_obj.page_size == 20
        assert input_obj.template_type == TemplateType.BROWSER

    def test_template_type_optional(self):
        input_obj = PageableInput()
        assert input_obj.template_type is None
