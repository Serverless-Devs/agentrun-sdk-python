import pytest

from scripts.check_test_evidence import (
    ChangedFile,
    find_missing_test_evidence,
    get_changed_files,
    main,
)


def changed(path, status="M"):
    return ChangedFile(status=status, path=path)


def test_integration_source_change_requires_matching_test_evidence():
    missing = find_missing_test_evidence([
        changed("agentrun/integration/langchain/model_adapter.py"),
    ])

    assert {item.evidence_type for item in missing} == {"unit", "e2e"}
    by_type = {item.evidence_type: item for item in missing}
    assert (
        by_type["unit"].source_path
        == "agentrun/integration/langchain/model_adapter.py"
    )
    assert "tests/unittests/integration/" in by_type["unit"].expected_patterns
    assert "tests/e2e/test_integration.py" in by_type["e2e"].expected_patterns


def test_integration_source_change_rejects_unittest_only_evidence():
    missing = find_missing_test_evidence([
        changed("agentrun/integration/langchain/model_adapter.py"),
        changed("tests/unittests/integration/test_integration.py"),
    ])

    assert [item.evidence_type for item in missing] == ["e2e"]


def test_integration_source_change_accepts_unit_and_e2e_evidence():
    missing = find_missing_test_evidence([
        changed("agentrun/integration/langchain/model_adapter.py"),
        changed("tests/unittests/integration/test_integration.py"),
        changed("tests/e2e/test_integration.py"),
    ])

    assert missing == []


def test_server_source_change_rejects_e2e_only_evidence():
    missing = find_missing_test_evidence([
        changed("agentrun/server/server.py"),
        changed("tests/e2e/test_server.py"),
    ])

    assert len(missing) == 1
    assert missing[0].source_path == "agentrun/server/server.py"
    assert missing[0].evidence_type == "unit"


def test_server_source_change_rejects_unittest_only_evidence():
    missing = find_missing_test_evidence([
        changed("agentrun/server/server.py"),
        changed("tests/unittests/server/test_server.py"),
    ])

    assert len(missing) == 1
    assert missing[0].source_path == "agentrun/server/server.py"
    assert missing[0].evidence_type == "e2e"


def test_server_source_change_accepts_unit_and_e2e_evidence():
    missing = find_missing_test_evidence([
        changed("agentrun/server/server.py"),
        changed("tests/unittests/server/test_server.py"),
        changed("tests/e2e/test_server.py"),
    ])

    assert missing == []


def test_non_sdk_changes_do_not_require_test_evidence():
    missing = find_missing_test_evidence([
        changed("examples/integration_examples.py"),
        changed("docs/docs/quick-start.md"),
    ])

    assert missing == []


def test_deleted_source_is_ignored():
    missing = find_missing_test_evidence([
        changed("agentrun/integration/langchain/model_adapter.py", "D"),
    ])

    assert missing == []


def test_missing_base_ref_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "scripts.check_test_evidence.git_ref_exists", lambda ref: False
    )

    with pytest.raises(RuntimeError, match="Base ref 'origin/main'"):
        get_changed_files("origin/main")


def test_main_returns_failure_for_missing_evidence(capsys):
    exit_code = main([
        "--changed-file",
        "agentrun/integration/langchain/model_adapter.py",
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "SDK test evidence gate failed" in captured.err
    assert "tests/unittests/integration/" in captured.err
    assert "tests/e2e/test_integration.py" in captured.err


@pytest.mark.parametrize(
    ("source_path", "test_path"),
    [
        (
            "agentrun/toolset/api/openapi.py",
            "tests/unittests/toolset/test_openapi.py",
        ),
        (
            "agentrun/utils/helper.py",
            "tests/unittests/utils/test_helper.py",
        ),
    ],
)
def test_matching_leaf_test_evidence_passes(source_path, test_path):
    missing = find_missing_test_evidence([
        changed(source_path),
        changed(test_path),
        changed(test_path.replace("tests/unittests", "tests/e2e")),
    ])

    assert missing == []
