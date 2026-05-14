"""Agent Runtime 数据模型单元测试"""

import base64
import os
import tempfile
from unittest.mock import MagicMock, patch
import zipfile

import pytest

from agentrun.agent_runtime.model import (
    AgentRuntimeArtifact,
    AgentRuntimeCode,
    AgentRuntimeContainer,
    AgentRuntimeCreateInput,
    AgentRuntimeEndpointCreateInput,
    AgentRuntimeEndpointImmutableProps,
    AgentRuntimeEndpointListInput,
    AgentRuntimeEndpointMutableProps,
    AgentRuntimeEndpointRoutingConfig,
    AgentRuntimeEndpointRoutingWeight,
    AgentRuntimeEndpointSystemProps,
    AgentRuntimeEndpointUpdateInput,
    AgentRuntimeHealthCheckConfig,
    AgentRuntimeImmutableProps,
    AgentRuntimeLanguage,
    AgentRuntimeListInput,
    AgentRuntimeLogConfig,
    AgentRuntimeMutableProps,
    AgentRuntimeProtocolConfig,
    AgentRuntimeProtocolType,
    AgentRuntimeSystemProps,
    AgentRuntimeUpdateInput,
    AgentRuntimeVersion,
    AgentRuntimeVersionListInput,
    NASConfig,
    NASMountConfig,
    OSSMountConfig,
    OSSMountPoint,
    ProtocolSettings,
    RegistryAuthConfig,
    RegistryCertConfig,
    RegistryConfig,
    RegistryNetworkConfig,
    ScalingConfig,
    ScheduledPolicy,
)
from agentrun.utils.model import Status


class TestAgentRuntimeArtifact:
    """AgentRuntimeArtifact 枚举测试"""

    def test_code_value(self):
        assert AgentRuntimeArtifact.CODE == "Code"
        assert AgentRuntimeArtifact.CODE.value == "Code"

    def test_container_value(self):
        assert AgentRuntimeArtifact.CONTAINER == "Container"
        assert AgentRuntimeArtifact.CONTAINER.value == "Container"


class TestAgentRuntimeLanguage:
    """AgentRuntimeLanguage 枚举测试"""

    def test_python310(self):
        assert AgentRuntimeLanguage.PYTHON310 == "python3.10"

    def test_python312(self):
        assert AgentRuntimeLanguage.PYTHON312 == "python3.12"

    def test_nodejs18(self):
        assert AgentRuntimeLanguage.NODEJS18 == "nodejs18"

    def test_nodejs20(self):
        assert AgentRuntimeLanguage.NODEJS20 == "nodejs20"

    def test_java8(self):
        assert AgentRuntimeLanguage.JAVA8 == "java8"

    def test_java11(self):
        assert AgentRuntimeLanguage.JAVA11 == "java11"


class TestAgentRuntimeProtocolType:
    """AgentRuntimeProtocolType 枚举测试"""

    def test_http(self):
        assert AgentRuntimeProtocolType.HTTP == "HTTP"

    def test_mcp(self):
        assert AgentRuntimeProtocolType.MCP == "MCP"


class TestAgentRuntimeCode:
    """AgentRuntimeCode 测试"""

    def test_init_empty(self):
        code = AgentRuntimeCode()
        assert code.checksum is None
        assert code.command is None
        assert code.language is None
        assert code.oss_bucket_name is None
        assert code.oss_object_name is None
        assert code.zip_file is None

    def test_init_with_values(self):
        code = AgentRuntimeCode(
            checksum="123456",
            command=["python", "main.py"],
            language=AgentRuntimeLanguage.PYTHON312,
            oss_bucket_name="my-bucket",
            oss_object_name="my-object",
            zip_file="base64data",
        )
        assert code.checksum == "123456"
        assert code.command == ["python", "main.py"]
        assert code.language == AgentRuntimeLanguage.PYTHON312
        assert code.oss_bucket_name == "my-bucket"
        assert code.oss_object_name == "my-object"
        assert code.zip_file == "base64data"

    def test_from_oss(self):
        code = AgentRuntimeCode.from_oss(
            language=AgentRuntimeLanguage.PYTHON312,
            command=["python", "app.py"],
            bucket="test-bucket",
            object="code.zip",
        )
        assert code.language == AgentRuntimeLanguage.PYTHON312
        assert code.command == ["python", "app.py"]
        assert code.oss_bucket_name == "test-bucket"
        assert code.oss_object_name == "code.zip"
        assert code.zip_file is None

    def test_from_zip_file(self):
        """测试从 zip 文件创建 AgentRuntimeCode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建一个测试 zip 文件
            zip_path = os.path.join(tmpdir, "test.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.writestr("main.py", "print('hello')")

            code = AgentRuntimeCode.from_zip_file(
                language=AgentRuntimeLanguage.PYTHON312,
                command=["python", "main.py"],
                zip_file_path=zip_path,
            )

            assert code.language == AgentRuntimeLanguage.PYTHON312
            assert code.command == ["python", "main.py"]
            assert code.zip_file is not None
            # 验证 base64 编码的内容可以解码
            decoded = base64.b64decode(code.zip_file)
            assert len(decoded) > 0
            # 验证 checksum 存在
            assert code.checksum is not None

    def test_from_file_directory(self):
        """测试从目录创建 AgentRuntimeCode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试目录结构
            code_dir = os.path.join(tmpdir, "code")
            os.makedirs(code_dir)
            with open(os.path.join(code_dir, "main.py"), "w") as f:
                f.write("print('hello')")
            with open(os.path.join(code_dir, "utils.py"), "w") as f:
                f.write("def helper(): pass")

            code = AgentRuntimeCode.from_file(
                language=AgentRuntimeLanguage.PYTHON312,
                command=["python", "main.py"],
                file_path=code_dir,
            )

            assert code.language == AgentRuntimeLanguage.PYTHON312
            assert code.command == ["python", "main.py"]
            assert code.zip_file is not None
            assert code.checksum is not None

    def test_from_file_single_file(self):
        """测试从单个文件创建 AgentRuntimeCode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建单个测试文件
            file_path = os.path.join(tmpdir, "main.py")
            with open(file_path, "w") as f:
                f.write("print('hello')")

            code = AgentRuntimeCode.from_file(
                language=AgentRuntimeLanguage.PYTHON312,
                command=["python", "main.py"],
                file_path=file_path,
            )

            assert code.language == AgentRuntimeLanguage.PYTHON312
            assert code.command == ["python", "main.py"]
            assert code.zip_file is not None
            assert code.checksum is not None


class TestAgentRuntimeContainer:
    """AgentRuntimeContainer 测试"""

    def test_init_empty(self):
        container = AgentRuntimeContainer()
        assert container.command is None
        assert container.image is None

    def test_init_with_values(self):
        container = AgentRuntimeContainer(
            command=["python", "app.py"],
            image="registry.cn-hangzhou.aliyuncs.com/test/agent:v1",
        )
        assert container.command == ["python", "app.py"]
        assert (
            container.image == "registry.cn-hangzhou.aliyuncs.com/test/agent:v1"
        )


class TestAgentRuntimeHealthCheckConfig:
    """AgentRuntimeHealthCheckConfig 测试"""

    def test_init_empty(self):
        config = AgentRuntimeHealthCheckConfig()
        assert config.failure_threshold is None
        assert config.http_get_url is None
        assert config.initial_delay_seconds is None
        assert config.period_seconds is None
        assert config.success_threshold is None
        assert config.timeout_seconds is None

    def test_init_with_values(self):
        config = AgentRuntimeHealthCheckConfig(
            failure_threshold=3,
            http_get_url="/health",
            initial_delay_seconds=10,
            period_seconds=30,
            success_threshold=1,
            timeout_seconds=5,
        )
        assert config.failure_threshold == 3
        assert config.http_get_url == "/health"
        assert config.initial_delay_seconds == 10
        assert config.period_seconds == 30
        assert config.success_threshold == 1
        assert config.timeout_seconds == 5


class TestAgentRuntimeLogConfig:
    """AgentRuntimeLogConfig 测试"""

    def test_init_with_values(self):
        config = AgentRuntimeLogConfig(
            project="my-project",
            logstore="my-logstore",
        )
        assert config.project == "my-project"
        assert config.logstore == "my-logstore"


class TestAgentRuntimeProtocolConfig:
    """AgentRuntimeProtocolConfig 测试"""

    def test_init_default(self):
        config = AgentRuntimeProtocolConfig()
        assert config.type == AgentRuntimeProtocolType.HTTP

    def test_init_with_mcp(self):
        config = AgentRuntimeProtocolConfig(type=AgentRuntimeProtocolType.MCP)
        assert config.type == AgentRuntimeProtocolType.MCP


class TestAgentRuntimeEndpointRoutingWeight:
    """AgentRuntimeEndpointRoutingWeight 测试"""

    def test_init_empty(self):
        weight = AgentRuntimeEndpointRoutingWeight()
        assert weight.version is None
        assert weight.weight is None

    def test_init_with_values(self):
        weight = AgentRuntimeEndpointRoutingWeight(version="v1", weight=80)
        assert weight.version == "v1"
        assert weight.weight == 80


class TestAgentRuntimeEndpointRoutingConfig:
    """AgentRuntimeEndpointRoutingConfig 测试"""

    def test_init_empty(self):
        config = AgentRuntimeEndpointRoutingConfig()
        assert config.version_weights is None

    def test_init_with_weights(self):
        weights = [
            AgentRuntimeEndpointRoutingWeight(version="v1", weight=80),
            AgentRuntimeEndpointRoutingWeight(version="v2", weight=20),
        ]
        config = AgentRuntimeEndpointRoutingConfig(version_weights=weights)
        assert config.version_weights is not None
        assert len(config.version_weights) == 2
        assert config.version_weights[0].version == "v1"
        assert config.version_weights[1].weight == 20


class TestAgentRuntimeMutableProps:
    """AgentRuntimeMutableProps 测试"""

    def test_init_empty(self):
        props = AgentRuntimeMutableProps()
        assert props.agent_runtime_name is None
        assert props.artifact_type is None
        assert props.cpu == 2
        assert props.memory == 4096
        assert props.port == 9000

    def test_init_with_values(self):
        props = AgentRuntimeMutableProps(
            agent_runtime_name="test-runtime",
            artifact_type=AgentRuntimeArtifact.CODE,
            cpu=4,
            memory=8192,
            port=8080,
            description="Test description",
        )
        assert props.agent_runtime_name == "test-runtime"
        assert props.artifact_type == AgentRuntimeArtifact.CODE
        assert props.cpu == 4
        assert props.memory == 8192
        assert props.port == 8080
        assert props.description == "Test description"


class TestAgentRuntimeImmutableProps:
    """AgentRuntimeImmutableProps 测试"""

    def test_init_empty(self):
        props = AgentRuntimeImmutableProps()
        # 这是一个空类，只是为了继承结构
        assert props is not None


class TestAgentRuntimeSystemProps:
    """AgentRuntimeSystemProps 测试"""

    def test_init_empty(self):
        props = AgentRuntimeSystemProps()
        assert props.agent_runtime_arn is None
        assert props.agent_runtime_id is None
        assert props.created_at is None
        assert props.last_updated_at is None
        assert props.resource_name is None
        assert props.status is None
        assert props.status_reason is None
        assert props.agent_runtime_version is None

    def test_init_with_values(self):
        props = AgentRuntimeSystemProps(
            agent_runtime_arn="arn:acs:agentrun:cn-hangzhou:123456:agent/test",
            agent_runtime_id="ar-123456",
            created_at="2024-01-01T00:00:00Z",
            last_updated_at="2024-01-02T00:00:00Z",
            resource_name="test-runtime",
            status=Status.READY,
            status_reason="",
            agent_runtime_version="1",
        )
        assert (
            props.agent_runtime_arn
            == "arn:acs:agentrun:cn-hangzhou:123456:agent/test"
        )
        assert props.agent_runtime_id == "ar-123456"
        # status 可能被序列化为字符串
        assert props.status == Status.READY or props.status == "READY"


class TestAgentRuntimeEndpointMutableProps:
    """AgentRuntimeEndpointMutableProps 测试"""

    def test_init_empty(self):
        props = AgentRuntimeEndpointMutableProps()
        assert props.agent_runtime_endpoint_name is None
        assert props.description is None
        assert props.disable_public_network_access is None
        assert props.routing_configuration is None
        assert props.scaling_config is None
        assert props.target_version == "LATEST"


class TestAgentRuntimeEndpointImmutableProps:
    """AgentRuntimeEndpointImmutableProps 测试"""

    def test_init_empty(self):
        props = AgentRuntimeEndpointImmutableProps()
        assert props is not None


class TestAgentRuntimeEndpointSystemProps:
    """AgentRuntimeEndpointSystemProps 测试"""

    def test_init_empty(self):
        props = AgentRuntimeEndpointSystemProps()
        assert props.agent_runtime_endpoint_arn is None
        assert props.agent_runtime_endpoint_id is None
        assert props.agent_runtime_id is None
        assert props.endpoint_public_url is None
        assert props.resource_name is None
        assert props.status is None
        assert props.status_reason is None


class TestAgentRuntimeCreateInput:
    """AgentRuntimeCreateInput 测试"""

    def test_inherits_from_mutable_and_immutable(self):
        input_obj = AgentRuntimeCreateInput(
            agent_runtime_name="test-runtime",
            artifact_type=AgentRuntimeArtifact.CODE,
        )
        assert input_obj.agent_runtime_name == "test-runtime"
        assert input_obj.artifact_type == AgentRuntimeArtifact.CODE


class TestAgentRuntimeUpdateInput:
    """AgentRuntimeUpdateInput 测试"""

    def test_inherits_from_mutable(self):
        input_obj = AgentRuntimeUpdateInput(description="Updated description")
        assert input_obj.description == "Updated description"


class TestAgentRuntimeListInput:
    """AgentRuntimeListInput 测试"""

    def test_init_empty(self):
        input_obj = AgentRuntimeListInput()
        assert input_obj.agent_runtime_name is None
        assert input_obj.system_tags is None
        assert input_obj.search_mode is None
        assert input_obj.status is None
        assert input_obj.workspace_id is None
        assert input_obj.workspace_ids is None

    def test_init_with_values(self):
        input_obj = AgentRuntimeListInput(
            agent_runtime_name="test",
            system_tags="env:prod,team:ai",
            search_mode="prefix",
            status="READY",
            workspace_ids="ws-1,ws-2",
        )
        assert input_obj.agent_runtime_name == "test"
        assert input_obj.system_tags == "env:prod,team:ai"
        assert input_obj.search_mode == "prefix"
        assert input_obj.status == "READY"
        assert input_obj.workspace_ids == "ws-1,ws-2"


class TestAgentRuntimeEndpointCreateInput:
    """AgentRuntimeEndpointCreateInput 测试"""

    def test_inherits_correctly(self):
        input_obj = AgentRuntimeEndpointCreateInput(
            agent_runtime_endpoint_name="test-endpoint",
            target_version="v1",
        )
        assert input_obj.agent_runtime_endpoint_name == "test-endpoint"
        assert input_obj.target_version == "v1"


class TestAgentRuntimeEndpointUpdateInput:
    """AgentRuntimeEndpointUpdateInput 测试"""

    def test_inherits_from_mutable(self):
        input_obj = AgentRuntimeEndpointUpdateInput(
            description="Updated endpoint",
            target_version="v2",
        )
        assert input_obj.description == "Updated endpoint"
        assert input_obj.target_version == "v2"


class TestAgentRuntimeEndpointListInput:
    """AgentRuntimeEndpointListInput 测试"""

    def test_init_empty(self):
        input_obj = AgentRuntimeEndpointListInput()
        assert input_obj.endpoint_name is None
        assert input_obj.search_mode is None


class TestAgentRuntimeVersion:
    """AgentRuntimeVersion 测试"""

    def test_init_empty(self):
        version = AgentRuntimeVersion()
        assert version.agent_runtime_arn is None
        assert version.agent_runtime_id is None
        assert version.agent_runtime_name is None
        assert version.agent_runtime_version is None
        assert version.description is None
        assert version.last_updated_at is None

    def test_init_with_values(self):
        version = AgentRuntimeVersion(
            agent_runtime_arn="arn:test",
            agent_runtime_id="ar-123",
            agent_runtime_name="test-runtime",
            agent_runtime_version="1",
            description="Version 1",
            last_updated_at="2024-01-01T00:00:00Z",
        )
        assert version.agent_runtime_arn == "arn:test"
        assert version.agent_runtime_version == "1"


class TestAgentRuntimeVersionListInput:
    """AgentRuntimeVersionListInput 测试"""

    def test_init_empty(self):
        input_obj = AgentRuntimeVersionListInput()
        # 继承自 PageableInput
        assert input_obj is not None


class TestModelDump:
    """model_dump 方法测试"""

    def test_agent_runtime_code_model_dump(self):
        code = AgentRuntimeCode(
            language=AgentRuntimeLanguage.PYTHON312,
            command=["python", "main.py"],
        )
        dumped = code.model_dump()
        # pydantic 使用 camelCase 别名
        assert "language" in dumped or "Language" in dumped
        assert "command" in dumped or "Command" in dumped

    def test_create_input_model_dump(self):
        input_obj = AgentRuntimeCreateInput(
            agent_runtime_name="test-runtime",
            cpu=4,
            memory=8192,
        )
        dumped = input_obj.model_dump()
        assert dumped is not None
        # 验证值存在
        assert "agentRuntimeName" in dumped or "agent_runtime_name" in dumped


class TestRegistryConfig:
    """RegistryConfig 及其子模型测试"""

    def test_auth_config_round_trip(self):
        cfg = RegistryAuthConfig(user_name="alice", password="pwd")
        dumped = cfg.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {"userName": "alice", "password": "pwd"}

    def test_cert_config_round_trip(self):
        cfg = RegistryCertConfig(insecure=True, root_ca_cert_base_64="abc")
        dumped = cfg.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {
            "insecure": True,
            "rootCaCertBase64": "abc",
        }

    def test_network_config_round_trip(self):
        cfg = RegistryNetworkConfig(
            security_group_id="sg-1",
            v_switch_id="vsw-1",
            vpc_id="vpc-1",
        )
        dumped = cfg.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {
            "securityGroupId": "sg-1",
            "vSwitchId": "vsw-1",
            "vpcId": "vpc-1",
        }

    def test_registry_config_nested(self):
        cfg = RegistryConfig(
            auth_config=RegistryAuthConfig(user_name="u", password="p"),
            cert_config=RegistryCertConfig(insecure=False),
            network_config=RegistryNetworkConfig(vpc_id="vpc-1"),
        )
        dumped = cfg.model_dump(by_alias=True, exclude_none=True)
        assert dumped["authConfig"]["userName"] == "u"
        assert dumped["certConfig"]["insecure"] is False
        assert dumped["networkConfig"]["vpcId"] == "vpc-1"


class TestContainerNewFields:
    """AgentRuntimeContainer 新增字段测试"""

    def test_acr_and_registry_fields(self):
        c = AgentRuntimeContainer(
            image="img:1",
            command=["python"],
            acr_instance_id="cri-xxx",
            image_registry_type="CUSTOM",
            port=9001,
            registry_config=RegistryConfig(
                auth_config=RegistryAuthConfig(user_name="u", password="p")
            ),
        )
        dumped = c.model_dump(by_alias=True, exclude_none=True)
        assert dumped["acrInstanceId"] == "cri-xxx"
        assert dumped["imageRegistryType"] == "CUSTOM"
        assert dumped["port"] == 9001
        assert dumped["registryConfig"]["authConfig"]["userName"] == "u"


class TestProtocolSettings:
    """ProtocolSettings & AgentRuntimeProtocolConfig.protocol_settings 测试"""

    def test_protocol_settings_aliases(self):
        ps = ProtocolSettings(
            a_2aagent_card="legacy",
            a_2a_agent_card="new",
            a_2a_agent_card_url="https://example.com/card",
            config='{"k":"v"}',
            method="POST",
            path="/invoke",
            path_prefix="/api",
            request_content_type="application/json",
            response_content_type="application/json",
            type="http",
        )
        dumped = ps.model_dump(by_alias=True, exclude_none=True)
        assert dumped["A2AAgentCard"] == "legacy"
        assert dumped["a2aAgentCard"] == "new"
        assert dumped["a2aAgentCardUrl"] == "https://example.com/card"
        assert dumped["pathPrefix"] == "/api"
        assert dumped["requestContentType"] == "application/json"
        assert dumped["responseContentType"] == "application/json"

    def test_protocol_config_with_settings(self):
        cfg = AgentRuntimeProtocolConfig(
            type=AgentRuntimeProtocolType.HTTP,
            protocol_settings=[
                ProtocolSettings(name="s1", type="http"),
                ProtocolSettings(name="s2", type="grpc"),
            ],
        )
        dumped = cfg.model_dump(by_alias=True, exclude_none=True)
        assert len(dumped["protocolSettings"]) == 2
        assert dumped["protocolSettings"][0]["name"] == "s1"


class TestRoutingWeightFloat:
    """AgentRuntimeEndpointRoutingWeight.weight 改为 float"""

    def test_weight_accepts_float(self):
        w = AgentRuntimeEndpointRoutingWeight(version="v1", weight=0.3)
        assert w.weight == pytest.approx(0.3)

    def test_routing_config_serialization(self):
        cfg = AgentRuntimeEndpointRoutingConfig(
            version_weights=[
                AgentRuntimeEndpointRoutingWeight(version="v1", weight=0.7),
                AgentRuntimeEndpointRoutingWeight(version="v2", weight=0.3),
            ]
        )
        dumped = cfg.model_dump(by_alias=True, exclude_none=True)
        assert dumped["versionWeights"][0]["weight"] == pytest.approx(0.7)
        assert dumped["versionWeights"][1]["weight"] == pytest.approx(0.3)


class TestNASAndOSSMountConfigs:
    """NASConfig / OSSMountConfig 及其子模型测试"""

    def test_nas_mount_config_round_trip(self):
        m = NASMountConfig(
            enable_tls=True,
            mount_dir="/mnt/nas",
            server_addr="addr.cn-hangzhou.nas.aliyuncs.com",
        )
        dumped = m.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {
            "enableTLS": True,
            "mountDir": "/mnt/nas",
            "serverAddr": "addr.cn-hangzhou.nas.aliyuncs.com",
        }

    def test_nas_config_with_mount_points(self):
        cfg = NASConfig(
            group_id=100,
            user_id=200,
            mount_points=[
                NASMountConfig(
                    mount_dir="/mnt/a", server_addr="addr.aliyuncs.com"
                ),
            ],
        )
        dumped = cfg.model_dump(by_alias=True, exclude_none=True)
        assert dumped["groupId"] == 100
        assert dumped["userId"] == 200
        assert dumped["mountPoints"][0]["mountDir"] == "/mnt/a"

    def test_oss_mount_point_round_trip(self):
        p = OSSMountPoint(
            bucket_name="bkt",
            bucket_path="/path",
            endpoint="oss-cn-hangzhou.aliyuncs.com",
            mount_dir="/mnt/oss",
            read_only=True,
        )
        dumped = p.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {
            "bucketName": "bkt",
            "bucketPath": "/path",
            "endpoint": "oss-cn-hangzhou.aliyuncs.com",
            "mountDir": "/mnt/oss",
            "readOnly": True,
        }

    def test_oss_mount_config_with_points(self):
        cfg = OSSMountConfig(
            mount_points=[
                OSSMountPoint(bucket_name="bkt", mount_dir="/mnt"),
            ]
        )
        dumped = cfg.model_dump(by_alias=True, exclude_none=True)
        assert dumped["mountPoints"][0]["bucketName"] == "bkt"


class TestScalingConfig:
    """ScalingConfig & ScheduledPolicy 测试"""

    def test_scheduled_policy_round_trip(self):
        sp = ScheduledPolicy(
            name="daily",
            schedule_expression="0 0 9 * * ?",
            start_time="2026-05-01T00:00:00Z",
            end_time="2026-06-01T00:00:00Z",
            target=3,
            time_zone="Asia/Shanghai",
        )
        dumped = sp.model_dump(by_alias=True, exclude_none=True)
        assert dumped["scheduleExpression"] == "0 0 9 * * ?"
        assert dumped["startTime"] == "2026-05-01T00:00:00Z"
        assert dumped["endTime"] == "2026-06-01T00:00:00Z"
        assert dumped["timeZone"] == "Asia/Shanghai"
        assert dumped["target"] == 3

    def test_scaling_config_with_policies(self):
        cfg = ScalingConfig(
            min_instances=2,
            scheduled_policies=[
                ScheduledPolicy(name="p1", target=5),
            ],
        )
        dumped = cfg.model_dump(by_alias=True, exclude_none=True)
        assert dumped["minInstances"] == 2
        assert dumped["scheduledPolicies"][0]["name"] == "p1"


class TestEndpointNewFields:
    """AgentRuntimeEndpoint Create/Update 新增字段测试"""

    def test_create_input_with_disable_public_and_scaling(self):
        i = AgentRuntimeEndpointCreateInput(
            agent_runtime_endpoint_name="ep",
            disable_public_network_access=True,
            scaling_config=ScalingConfig(min_instances=1),
        )
        dumped = i.model_dump(by_alias=True, exclude_none=True)
        assert dumped["disablePublicNetworkAccess"] is True
        assert dumped["scalingConfig"]["minInstances"] == 1

    def test_update_input_delete_scaling_config(self):
        i = AgentRuntimeEndpointUpdateInput(
            agent_runtime_endpoint_name="ep",
            delete_scaling_config=True,
        )
        dumped = i.model_dump(by_alias=True, exclude_none=True)
        assert dumped["deleteScalingConfig"] is True


class TestRuntimeNewFields:
    """AgentRuntimeMutableProps 新增字段测试"""

    def test_disk_size_and_session_isolation(self):
        m = AgentRuntimeMutableProps(
            disk_size=30,
            enable_session_isolation=True,
        )
        dumped = m.model_dump(by_alias=True, exclude_none=True)
        assert dumped["diskSize"] == 30
        assert dumped["enableSessionIsolation"] is True

    def test_nas_and_oss_mount_in_create_input(self):
        i = AgentRuntimeCreateInput(
            agent_runtime_name="r",
            nas_config=NASConfig(
                user_id=1, mount_points=[NASMountConfig(mount_dir="/mnt/nas")]
            ),
            oss_mount_config=OSSMountConfig(
                mount_points=[OSSMountPoint(bucket_name="b", mount_dir="/mnt")]
            ),
        )
        dumped = i.model_dump(by_alias=True, exclude_none=True)
        assert dumped["nasConfig"]["userId"] == 1
        assert dumped["nasConfig"]["mountPoints"][0]["mountDir"] == "/mnt/nas"
        assert dumped["ossMountConfig"]["mountPoints"][0]["bucketName"] == "b"
