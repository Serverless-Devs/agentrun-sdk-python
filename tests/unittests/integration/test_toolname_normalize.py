import hashlib
import json

from agentrun.integration.utils.canonical import CanonicalTool
from agentrun.integration.utils.tool import normalize_tool_name, Tool
from agentrun.toolset.api.openapi import ApiSet


def test_normalize_tool_name_short():
    name = "short_name"
    assert normalize_tool_name(name) == name


def test_normalize_tool_name_long():
    long_name = "a" * 80
    normalized = normalize_tool_name(long_name)
    assert len(normalized) == 64
    assert normalized[:32] == long_name[:32]
    expected_md5 = hashlib.md5(long_name.encode("utf-8")).hexdigest()
    assert normalized[32:] == expected_md5


def test_tool_normalizes_in_tool_class():
    long_name = "tool_" + "x" * 80
    t = Tool(name=long_name, func=lambda: None)
    assert len(t.name) <= 64
    assert t.name == normalize_tool_name(long_name)


def test_canonical_tool_normalizes():
    long_name = "canonical_" + "y" * 80
    c = CanonicalTool(name=long_name, description="desc", parameters={})
    assert c.name == normalize_tool_name(long_name)


def test_openapi_from_schema_tool_name_truncation():
    long_name = "op_" + "z" * 100
    schema = {
        "openapi": "3.0.0",
        "paths": {
            "/test": {
                "get": {
                    "operationId": long_name,
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"type": "string"}
                                }
                            }
                        }
                    },
                }
            }
        },
    }

    apiset = ApiSet.from_openapi_schema(schema=json.dumps(schema))
    names = [t.name for t in apiset.tools()]
    assert normalize_tool_name(long_name) in names


def test_google_adk_declaration_name_normalized():
    try:
        from agentrun.integration.google_adk.tool_adapter import (
            GoogleADKToolAdapter,
        )
    except Exception:
        # google.genai not installed — skip
        return

    long_name = "gtool_" + "x" * 100
    ct = CanonicalTool(name=long_name, description="d", parameters={})
    adapter = GoogleADKToolAdapter()
    tools = adapter.from_canonical([ct])
    decl = tools[0]._get_declaration()
    assert decl.name == normalize_tool_name(long_name)
