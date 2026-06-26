#!/usr/bin/env python3
"""Require test evidence for changed SDK source files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import PurePosixPath
import subprocess
import sys
from typing import Iterable, Sequence

SDK_ROOT = PurePosixPath("agentrun")
UNITTEST_ROOT = PurePosixPath("tests/unittests")
E2E_ROOT = PurePosixPath("tests/e2e")
DEFAULT_BASE_REF = "origin/main"


@dataclass(frozen=True)
class ChangedFile:
    status: str
    path: str


@dataclass(frozen=True)
class MissingTestEvidence:
    source_path: str
    evidence_type: str
    expected_patterns: tuple[str, ...]


def is_sdk_source(path: str) -> bool:
    posix_path = PurePosixPath(path)
    return (
        path.endswith(".py")
        and posix_path.is_relative_to(SDK_ROOT)
        and "__pycache__" not in posix_path.parts
    )


def is_unit_test_evidence(path: str) -> bool:
    return is_test_evidence(path, UNITTEST_ROOT)


def is_e2e_test_evidence(path: str) -> bool:
    return is_test_evidence(path, E2E_ROOT)


def is_test_evidence(path: str, test_root: PurePosixPath) -> bool:
    posix_path = PurePosixPath(path)
    return path.endswith(".py") and posix_path.is_relative_to(test_root)


def _module_parts(source_path: str) -> tuple[str, ...]:
    relative = PurePosixPath(source_path).relative_to(SDK_ROOT)
    if relative.name == "__init__.py":
        return relative.parent.parts
    return relative.with_suffix("").parts


def expected_test_patterns(
    source_path: str, test_root: PurePosixPath
) -> tuple[str, ...]:
    module_parts = _module_parts(source_path)
    if not module_parts:
        return (f"{test_root}/test_agentrun.py",)

    top_level = module_parts[0]
    leaf = module_parts[-1]
    nearest_package = module_parts[-2] if len(module_parts) > 1 else top_level
    patterns = [
        f"{test_root}/{top_level}/",
        f"{test_root}/test_{top_level}.py",
        f"{test_root}/**/test_{top_level}.py",
        f"{test_root}/**/test_{nearest_package}.py",
        f"{test_root}/**/test_{leaf}.py",
    ]
    return tuple(dict.fromkeys(patterns))


def has_matching_test_evidence(
    source_path: str,
    changed_test_paths: Iterable[str],
    test_root: PurePosixPath,
) -> bool:
    module_parts = _module_parts(source_path)
    if not module_parts:
        return any(
            is_test_evidence(path, test_root) for path in changed_test_paths
        )

    top_level = module_parts[0]
    leaf = module_parts[-1]
    nearest_package = module_parts[-2] if len(module_parts) > 1 else top_level
    acceptable_names = {
        f"test_{top_level}.py",
        f"test_{nearest_package}.py",
        f"test_{leaf}.py",
    }

    for path in changed_test_paths:
        posix_path = PurePosixPath(path)
        if not is_test_evidence(path, test_root):
            continue
        if posix_path.name in acceptable_names:
            return True
        if posix_path.is_relative_to(test_root / top_level):
            return True
    return False


def find_missing_test_evidence(
    changed_files: Sequence[ChangedFile],
) -> list[MissingTestEvidence]:
    changed_sources = [
        changed.path
        for changed in changed_files
        if changed.status != "D" and is_sdk_source(changed.path)
    ]
    changed_tests = [
        changed.path
        for changed in changed_files
        if changed.status != "D"
        and (
            is_unit_test_evidence(changed.path)
            or is_e2e_test_evidence(changed.path)
        )
    ]

    missing: list[MissingTestEvidence] = []
    required_evidence = (
        ("unit", UNITTEST_ROOT),
        ("e2e", E2E_ROOT),
    )
    for source_path in changed_sources:
        for evidence_type, test_root in required_evidence:
            if has_matching_test_evidence(
                source_path, changed_tests, test_root
            ):
                continue
            missing.append(
                MissingTestEvidence(
                    source_path=source_path,
                    evidence_type=evidence_type,
                    expected_patterns=expected_test_patterns(
                        source_path, test_root
                    ),
                )
            )
    return missing


def parse_changed_file_specs(specs: Sequence[str]) -> list[ChangedFile]:
    changed_files: list[ChangedFile] = []
    for spec in specs:
        if not spec:
            continue
        if ":" in spec and spec.split(":", 1)[0] in {
            "A",
            "C",
            "D",
            "M",
            "R",
            "T",
        }:
            status, path = spec.split(":", 1)
        else:
            status, path = "M", spec
        changed_files.append(ChangedFile(status=status[0], path=path.strip()))
    return changed_files


def _run_git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout


def _parse_git_name_status(output: str) -> list[ChangedFile]:
    changed_files: list[ChangedFile] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        status = fields[0][0]
        path = fields[-1]
        changed_files.append(ChangedFile(status=status, path=path))
    return changed_files


def git_ref_exists(ref: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def get_changed_files(base_ref: str) -> list[ChangedFile]:
    changed: list[ChangedFile] = []
    if not git_ref_exists(base_ref):
        raise RuntimeError(
            f"Base ref {base_ref!r} was not found. Fetch the PR base ref or "
            "pass explicit --changed-file entries."
        )
    base = _run_git(["merge-base", base_ref, "HEAD"]).strip()
    changed.extend(
        _parse_git_name_status(
            _run_git([
                "diff",
                "--name-status",
                "--diff-filter=ACMRT",
                base,
                "HEAD",
            ])
        )
    )

    changed.extend(
        _parse_git_name_status(
            _run_git(
                ["diff", "--name-status", "--diff-filter=ACMRT", "HEAD", "--"]
            )
        )
    )

    deduped: dict[str, ChangedFile] = {}
    for changed_file in changed:
        deduped[changed_file.path] = changed_file
    return list(deduped.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail when changed agentrun source files do not have "
            "matching unit and e2e test evidence in the same change."
        )
    )
    parser.add_argument(
        "--base-ref",
        default=DEFAULT_BASE_REF,
        help=(
            "Git base ref for committed PR changes. Default:"
            f" {DEFAULT_BASE_REF}"
        ),
    )
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        metavar="[STATUS:]PATH",
        help=(
            "Explicit changed file for tests or custom CI integrations. "
            "Status defaults to M."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        changed_files = (
            parse_changed_file_specs(args.changed_file)
            if args.changed_file
            else get_changed_files(args.base_ref)
        )
    except RuntimeError as exc:
        print(
            f"SDK test evidence gate could not inspect git diff: {exc}",
            file=sys.stderr,
        )
        return 2

    missing = find_missing_test_evidence(changed_files)
    if not missing:
        print("SDK test evidence gate passed.")
        return 0

    print("SDK test evidence gate failed.", file=sys.stderr)
    print(
        "Every changed agentrun/**/*.py file must include matching "
        "tests/unittests and tests/e2e evidence.",
        file=sys.stderr,
    )
    for item in missing:
        print(
            f"\nMissing {item.evidence_type} test evidence for:"
            f" {item.source_path}",
            file=sys.stderr,
        )
        print("Expected one of:", file=sys.stderr)
        for pattern in item.expected_patterns:
            print(f"  - {pattern}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
