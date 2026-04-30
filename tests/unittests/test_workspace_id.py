"""跨模块 workspace_id 字段单元测试 / Cross-module workspace_id field unit tests

验证各资源模块的 Create / List / Output 输入类都正确暴露 ``workspace_id`` 字段，
并保证序列化时落到底层 SDK 期望的 ``workspaceId`` (camelCase) 键。
Verifies every resource module's Create / List / Output input class exposes
``workspace_id`` correctly and serializes it to the ``workspaceId`` (camelCase)
key expected by the underlying SDK.
"""

from typing import List, Type

import pytest

from agentrun.agent_runtime.model import (
    AgentRuntimeCreateInput,
    AgentRuntimeListInput,
)
from agentrun.credential.model import (
    CredentialCreateInput,
    CredentialListInput,
    CredentialListOutput,
)
from agentrun.knowledgebase.model import (
    KnowledgeBaseListInput,
    KnowledgeBaseListOutput,
    KnowledgeBaseProvider,
)
from agentrun.memory_collection.model import (
    MemoryCollectionCreateInput,
    MemoryCollectionListInput,
    MemoryCollectionListOutput,
)
from agentrun.model.model import (
    ModelProxyCreateInput,
    ModelProxyListInput,
    ModelServiceCreateInput,
    ModelServiceListInput,
)
from agentrun.sandbox.model import (
    PageableInput as SandboxPageableInput,
)
from agentrun.sandbox.model import (
    TemplateInput,
    TemplateType,
)
from agentrun.utils.model import BaseModel

WORKSPACE_ID = "ws-test-12345"


# ---------------------------------------------------------------------------
# 1. 创建输入：每个 Create Input 都要支持 workspace_id 入参与序列化
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls",
    [
        AgentRuntimeCreateInput,
        MemoryCollectionCreateInput,
        ModelServiceCreateInput,
        ModelProxyCreateInput,
    ],
)
def test_create_input_accepts_and_serializes_workspace_id(
    model_cls: Type[BaseModel],
):
    """所有可独立构造的 CreateInput 都接受 workspace_id 并序列化为 workspaceId"""
    instance = model_cls(workspace_id=WORKSPACE_ID)
    assert instance.workspace_id == WORKSPACE_ID  # type: ignore[attr-defined]

    dumped = instance.model_dump(by_alias=True, exclude_none=True)
    assert dumped.get("workspaceId") == WORKSPACE_ID

    # 反序列化：模拟 from_inner_object 的行为（显式 by_alias=True）
    parsed = model_cls.model_validate(
        {"workspaceId": WORKSPACE_ID}, by_alias=True
    )
    assert parsed.workspace_id == WORKSPACE_ID  # type: ignore[attr-defined]


def test_credential_create_input_accepts_workspace_id():
    """CredentialCreateInput 因有必填字段，单独构造测试"""
    from agentrun.credential.model import CredentialConfig

    instance = CredentialCreateInput(
        credential_name="ws-cred",
        credential_config=CredentialConfig.inbound_api_key("sk-test"),
        workspace_id=WORKSPACE_ID,
    )
    assert instance.workspace_id == WORKSPACE_ID

    dumped = instance.model_dump(by_alias=True, exclude_none=True)
    assert dumped["workspaceId"] == WORKSPACE_ID


def test_template_input_accepts_workspace_id():
    """TemplateInput 因有 model_validator 派生默认值，单独构造测试"""
    instance = TemplateInput(
        template_type=TemplateType.CODE_INTERPRETER,
        workspace_id=WORKSPACE_ID,
    )
    assert instance.workspace_id == WORKSPACE_ID

    dumped = instance.model_dump(by_alias=True, exclude_none=True)
    assert dumped["workspaceId"] == WORKSPACE_ID


# ---------------------------------------------------------------------------
# 2. 默认行为：不传 workspace_id 时不应注入键到序列化结果
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls",
    [
        AgentRuntimeCreateInput,
        MemoryCollectionCreateInput,
        ModelServiceCreateInput,
        ModelProxyCreateInput,
    ],
)
def test_create_input_workspace_id_default_none(
    model_cls: Type[BaseModel],
):
    instance = model_cls()
    assert instance.workspace_id is None  # type: ignore[attr-defined]

    dumped = instance.model_dump(by_alias=True, exclude_none=True)
    assert "workspaceId" not in dumped
    # 老调用方（不传 workspace_id）行为不变
    dumped_with_none = instance.model_dump(by_alias=True)
    assert dumped_with_none.get("workspaceId") is None


# ---------------------------------------------------------------------------
# 3. List 输入：每个 ListInput 都要支持 workspace_id 过滤参数
# ---------------------------------------------------------------------------

LIST_INPUT_CLASSES: List[Type[BaseModel]] = [
    AgentRuntimeListInput,
    CredentialListInput,
    KnowledgeBaseListInput,
    MemoryCollectionListInput,
    ModelServiceListInput,
    ModelProxyListInput,
    SandboxPageableInput,
]


@pytest.mark.parametrize("list_input_cls", LIST_INPUT_CLASSES)
def test_list_input_supports_workspace_id_filter(
    list_input_cls: Type[BaseModel],
):
    instance = list_input_cls(workspace_id=WORKSPACE_ID)
    assert instance.workspace_id == WORKSPACE_ID  # type: ignore[attr-defined]

    dumped = instance.model_dump(by_alias=True, exclude_none=True)
    assert dumped.get("workspaceId") == WORKSPACE_ID


@pytest.mark.parametrize("list_input_cls", LIST_INPUT_CLASSES)
def test_list_input_workspace_id_default_none(
    list_input_cls: Type[BaseModel],
):
    instance = list_input_cls()
    assert instance.workspace_id is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 4. List 输出：每个 ListOutput 都要能从底层 SDK 的 workspaceId 反序列化
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "list_output_cls",
    [
        CredentialListOutput,
        KnowledgeBaseListOutput,
        MemoryCollectionListOutput,
    ],
)
def test_list_output_parses_workspace_id_from_camel_case(
    list_output_cls: Type[BaseModel],
):
    """ListOutput 模拟 from_inner_object 行为反序列化 camelCase workspaceId"""
    instance = list_output_cls.model_validate(
        {"workspaceId": WORKSPACE_ID}, by_alias=True
    )
    assert instance.workspace_id == WORKSPACE_ID  # type: ignore[attr-defined]


def test_knowledgebase_workspace_id_distinct_from_bailian_workspace():
    """KnowledgeBase 的 workspace_id 与 BailianProviderSettings.workspace_id 在不同层级，
    互不影响。"""
    from agentrun.knowledgebase.model import (
        BailianProviderSettings,
        KnowledgeBaseCreateInput,
    )

    bailian_ws = "bailian-ws-9999"
    agentrun_ws = WORKSPACE_ID

    kb_input = KnowledgeBaseCreateInput(
        knowledge_base_name="ws-test-kb",
        provider=KnowledgeBaseProvider.BAILIAN,
        provider_settings=BailianProviderSettings(
            workspace_id=bailian_ws, index_ids=["idx-1"]
        ),
        workspace_id=agentrun_ws,
    )
    assert kb_input.workspace_id == agentrun_ws
    assert isinstance(kb_input.provider_settings, BailianProviderSettings)
    assert kb_input.provider_settings.workspace_id == bailian_ws

    dumped = kb_input.model_dump(by_alias=True, exclude_none=True)
    # 顶层是 AgentRun 的 workspaceId
    assert dumped["workspaceId"] == agentrun_ws
    # provider_settings 内部是百炼的 workspaceId（嵌套在 providerSettings 下）
    assert dumped["providerSettings"]["workspaceId"] == bailian_ws
