"""Regression tests for credentials bound to imported FunctionCall tools."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from agentrun.credential import Credential
from agentrun.tool.tool import Tool
from agentrun.utils.config import Config
from agentrun.utils.credential_context import use_sts_credentials


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def async_call(request):
    return request.param


@pytest.fixture
def tool():
    # Use the same camelCase fields returned by GetTool.
    return Tool.model_validate(
        {
            "toolName": "test-tool",
            "toolType": "FUNCTIONCALL",
            "createMethod": "OPENAPI_IMPORT",
            "credentialName": "test-credential",
            "dataEndpoint": "https://unused.example.com",
            "protocolSpec": json.dumps({
                "openapi": "3.0.0",
                "servers": [{"url": "https://api.example.com"}],
                "paths": {"/check": {"get": {"operationId": "check"}}},
            }),
        },
        by_alias=True,
    )


@pytest.fixture
def credential_lookup(async_call):
    method = "get_by_name_async" if async_call else "get_by_name"
    kwargs = {"new_callable": AsyncMock} if async_call else {}
    with patch.object(Credential, method, **kwargs) as lookup:
        lookup.return_value = Credential(
            credential_auth_type="api_key",
            credential_source_type="external_tool",
            credential_public_config={"headerKey": "X-API-Key", "prefix": ""},
            credential_secret="test-secret",
        )
        yield lookup


async def invoke(tool, async_call, config):
    if async_call:
        return await tool.call_tool_async("check", {"q": "test"}, config=config)
    return tool.call_tool("check", {"q": "test"}, config=config)


@pytest.mark.parametrize(
    "public_config,header,value",
    [
        ({"headerKey": "X-API-Key", "prefix": ""}, "X-API-Key", "test-secret"),
        (
            {"headerKey": "Authorization", "prefix": "Bearer "},
            "Authorization",
            "Bearer test-secret",
        ),
        (
            {"headerKey": "X-Token", "prefix": "Token:"},
            "X-Token",
            "Token:test-secret",
        ),
        ({}, "X-API-Key", "test-secret"),
        (None, "X-API-Key", "test-secret"),
    ],
)
async def test_injects_bound_api_key(
    tool, async_call, credential_lookup, public_config, header, value
):
    credential_lookup.return_value.credential_public_config = public_config
    config = Config(
        account_id="test-account",
        region_id="cn-beijing",
        headers={"X-Custom": "preserved", header.lower(): "stale"},
    )
    original_headers = config.get_headers().copy()
    with respx.mock as router:
        route = router.get("https://api.example.com/check").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        assert await invoke(tool, async_call, config) == {"ok": True}
    request = route.calls.last.request
    assert request.headers.get_list(header) == [value]
    assert request.headers["X-Custom"] == "preserved"
    assert dict(request.url.params) == {"q": "test"}
    assert config.get_headers() == original_headers
    credential_lookup.assert_called_once()
    assert credential_lookup.call_args.args == ("test-credential",)
    lookup_config = credential_lookup.call_args.kwargs["config"]
    assert lookup_config.get_account_id() == "test-account"
    assert lookup_config.get_region_id() == "cn-beijing"
    assert lookup_config.get_headers() == original_headers
    if async_call:
        credential_lookup.assert_awaited_once()
    assert tool.credential_name == "test-credential"
    assert "test-secret" not in tool.protocol_spec


async def test_reads_rotated_secret_on_next_call(
    tool, async_call, credential_lookup
):
    with respx.mock as router:
        route = router.get("https://api.example.com/check").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        await invoke(tool, async_call, Config())
        credential_lookup.return_value.credential_secret = "rotated-secret"
        await invoke(tool, async_call, Config())
    assert [call.request.headers["X-API-Key"] for call in route.calls] == [
        "test-secret",
        "rotated-secret",
    ]
    assert credential_lookup.call_count == 2


@pytest.mark.parametrize("explicit", [False, True])
async def test_credential_lookup_uses_request_sts(
    tool, async_call, credential_lookup, explicit
):
    credentials = ("explicit-ak", "explicit-sk", "explicit-sts")
    config = (
        Config(
            access_key_id=credentials[0],
            access_key_secret=credentials[1],
            security_token=credentials[2],
        )
        if explicit
        else Config()
    )
    observed = []

    def lookup(name, config):
        observed.append((
            config.get_access_key_id(),
            config.get_access_key_secret(),
            config.get_security_token(),
        ))
        return credential_lookup.return_value

    credential_lookup.side_effect = lookup
    with respx.mock as router:
        router.get("https://api.example.com/check").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        for token in ("first", "second"):
            overlay = (f"{token}-ak", f"{token}-sk", f"{token}-sts")
            with use_sts_credentials(*overlay):
                await invoke(tool, async_call, config)
            assert observed[-1] == (credentials if explicit else overlay)


@pytest.mark.parametrize(
    "create_method,credential_name",
    [
        ("OPENAPI_IMPORT", None),
        ("OPENAPI_IMPORT", ""),
        ("CODE_PACKAGE", "test-credential"),
    ],
)
async def test_other_modes_keep_explicit_headers(
    tool, async_call, credential_lookup, create_method, credential_name
):
    tool.create_method = create_method
    tool.credential_name = credential_name
    with respx.mock as router:
        route = router.get("https://api.example.com/check").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        await invoke(
            tool, async_call, Config(headers={"X-API-Key": "explicit"})
        )
    assert route.calls.last.request.headers["X-API-Key"] == "explicit"
    credential_lookup.assert_not_called()


async def test_preserves_spec_fixed_header_precedence(
    tool, async_call, credential_lookup
):
    spec = json.loads(tool.protocol_spec)
    spec["paths"]["/check"]["get"]["parameters"] = [{
        "name": "x-api-key",
        "in": "header",
        "schema": {"type": "string", "enum": ["spec-value"]},
    }]
    tool.protocol_spec = json.dumps(spec)
    with respx.mock as router:
        route = router.get("https://api.example.com/check").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        await invoke(tool, async_call, Config())
    assert route.calls.last.request.headers.get_list("X-API-Key") == [
        "spec-value"
    ]


@pytest.mark.parametrize(
    "failure", ["lookup", "empty-secret", "unsupported-auth"]
)
async def test_credential_failures_stop_outbound_request(
    tool, async_call, credential_lookup, failure
):
    if failure == "lookup":
        credential_lookup.side_effect = RuntimeError("credential lookup failed")
        expected_error, message = RuntimeError, "credential lookup failed"
    elif failure == "empty-secret":
        credential_lookup.return_value.credential_secret = ""
        expected_error, message = ValueError, "has no secret"
    else:
        credential_lookup.return_value.credential_auth_type = "ak_sk"
        expected_error, message = ValueError, "api_key"
    with respx.mock as router:
        with pytest.raises(expected_error, match=message):
            await invoke(tool, async_call, Config())
        assert not router.calls
