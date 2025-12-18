#!/usr/bin/env python3
"""覆盖率检查脚本

功能：
1. 运行测试并收集覆盖率数据
2. 计算全量代码和增量代码的覆盖率
3. 根据配置文件检查覆盖率是否达标
4. 输出详细的覆盖率报告

使用方式：
    # 运行覆盖率检查（默认运行全量并尝试计算增量）
    python scripts/check_coverage.py

    # 指定基准分支（用于增量计算）
    python scripts/check_coverage.py --base-branch main

    # 只检查特定目录
    python scripts/check_coverage.py --source agentrun/server

    # 运行测试但不检查阈值
    python scripts/check_coverage.py --no-check
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class CoverageResult:
    """覆盖率结果"""

    total_statements: int
    covered_statements: int
    total_branches: int
    covered_branches: int

    @property
    def line_coverage(self) -> float:
        """行覆盖率百分比"""
        if self.total_statements == 0:
            return 100.0
        return (self.covered_statements / self.total_statements) * 100

    @property
    def branch_coverage(self) -> float:
        """分支覆盖率百分比"""
        if self.total_branches == 0:
            return 100.0
        return (self.covered_branches / self.total_branches) * 100


@dataclass
class CoverageThreshold:
    """覆盖率阈值"""

    branch_coverage: float = 95.0
    line_coverage: float = 95.0


@dataclass
class CoverageConfig:
    """覆盖率配置"""

    full: CoverageThreshold = None
    incremental: CoverageThreshold = None
    directory_overrides: dict[str, dict[str, CoverageThreshold]] = None
    exclude_directories: list[str] = None
    exclude_patterns: list[str] = None

    def __post_init__(self):
        if self.full is None:
            self.full = CoverageThreshold()
        if self.incremental is None:
            self.incremental = CoverageThreshold()
        if self.directory_overrides is None:
            self.directory_overrides = {}
        if self.exclude_directories is None:
            self.exclude_directories = []
        if self.exclude_patterns is None:
            self.exclude_patterns = []

    @classmethod
    def _parse_threshold(
        cls, data: dict[str, Any], default: CoverageThreshold = None
    ) -> CoverageThreshold:
        """解析覆盖率阈值配置"""
        if default is None:
            default = CoverageThreshold()
        if not data:
            return default
        return CoverageThreshold(
            branch_coverage=data.get(
                "branch_coverage", default.branch_coverage
            ),
            line_coverage=data.get("line_coverage", default.line_coverage),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "CoverageConfig":
        """从 YAML 文件加载配置"""
        if not path.exists():
            print(f"⚠️  配置文件 {path} 不存在，使用默认配置")
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 解析全量和增量配置
        full = cls._parse_threshold(data.get("full") or {})
        incremental = cls._parse_threshold(
            data.get("incremental") or {}, default=full
        )

        # 解析目录覆盖配置
        directory_overrides = {}
        for dir_path, dir_config in (
            data.get("directory_overrides") or {}
        ).items():
            if dir_config:
                directory_overrides[dir_path] = {
                    "full": cls._parse_threshold(
                        dir_config.get("full") or {}, default=full
                    ),
                    "incremental": cls._parse_threshold(
                        dir_config.get("incremental") or {}, default=incremental
                    ),
                }

        return cls(
            full=full,
            incremental=incremental,
            directory_overrides=directory_overrides,
            exclude_directories=data.get("exclude_directories") or [],
            exclude_patterns=data.get("exclude_patterns") or [],
        )

    def get_threshold_for_directory(
        self, directory: str, is_incremental: bool = False
    ) -> CoverageThreshold:
        """获取特定目录的覆盖率阈值

        Args:
            directory: 目录路径
            is_incremental: 是否为增量覆盖率

        Returns:
            CoverageThreshold: 覆盖率阈值
        """
        threshold_key = "incremental" if is_incremental else "full"
        default_threshold = self.incremental if is_incremental else self.full

        if directory in self.directory_overrides:
            return self.directory_overrides[directory].get(
                threshold_key, default_threshold
            )
        return default_threshold


def run_command(
    cmd: list[str], capture_output: bool = True
) -> subprocess.CompletedProcess:
    """运行命令"""
    print(f"🔧 运行命令: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    return result


def get_changed_files(base_branch: str = "main") -> list[str]:
    """获取相对于基准分支的变更文件列表"""
    # 获取 merge-base
    result = run_command(["git", "merge-base", base_branch, "HEAD"])
    if result.returncode != 0:
        print(f"⚠️  无法获取 merge-base: {result.stderr}")
        return []

    merge_base = result.stdout.strip()

    # 获取变更文件
    result = run_command(
        ["git", "diff", "--name-only", merge_base, "HEAD", "--", "*.py"]
    )
    if result.returncode != 0:
        print(f"⚠️  无法获取变更文件: {result.stderr}")
        return []

    files = [f for f in result.stdout.strip().split("\n") if f]
    return files


def get_changed_lines(
    base_branch: str = "main",
) -> dict[str, set[int]]:
    """获取变更的行号

    Returns:
        dict: {文件路径: {行号集合}}
    """
    result = run_command(["git", "merge-base", base_branch, "HEAD"])
    if result.returncode != 0:
        return {}

    merge_base = result.stdout.strip()

    # 获取 unified diff
    result = run_command(
        ["git", "diff", "-U0", merge_base, "HEAD", "--", "*.py"]
    )
    if result.returncode != 0:
        return {}

    changed_lines: dict[str, set[int]] = {}
    current_file = None

    for line in result.stdout.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:]
            if current_file not in changed_lines:
                changed_lines[current_file] = set()
        elif line.startswith("@@") and current_file:
            # 解析 @@ -start,count +start,count @@
            parts = line.split(" ")
            if len(parts) >= 3:
                new_range = parts[2]  # +start,count 或 +start
                if new_range.startswith("+"):
                    new_range = new_range[1:]
                    if "," in new_range:
                        start, count = map(int, new_range.split(","))
                    else:
                        start = int(new_range)
                        count = 1
                    for i in range(start, start + count):
                        changed_lines[current_file].add(i)

    return changed_lines


def run_coverage(
    source: Optional[str] = None,
    test_path: str = "tests/",
    extra_args: Optional[list[str]] = None,
) -> bool:
    """运行覆盖率测试

    Args:
        source: 源代码目录
        test_path: 测试目录
        extra_args: 额外的 pytest 参数

    Returns:
        bool: 测试是否成功
    """
    cmd = [
        "uv",
        "run",
        "pytest",
        test_path,
        "--cov-branch",
        "--cov-report=json:coverage.json",
        "--cov-report=term-missing",
    ]

    if source:
        cmd.append(f"--cov={source}")
    else:
        cmd.append("--cov=agentrun")

    if extra_args:
        cmd.extend(extra_args)

    result = run_command(cmd, capture_output=False)
    return result.returncode == 0


def parse_coverage_json(
    json_path: Path = Path("coverage.json"),
) -> dict[str, Any]:
    """解析覆盖率 JSON 报告"""
    if not json_path.exists():
        print(f"❌ 覆盖率报告 {json_path} 不存在")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_coverage(
    coverage_data: dict[str, Any],
    files_filter: Optional[list[str]] = None,
    lines_filter: Optional[dict[str, set[int]]] = None,
) -> CoverageResult:
    """计算覆盖率

    Args:
        coverage_data: 覆盖率 JSON 数据
        files_filter: 只计算这些文件的覆盖率
        lines_filter: 只计算这些行的覆盖率 {文件: {行号集合}}

    Returns:
        CoverageResult: 覆盖率结果
    """
    total_statements = 0
    covered_statements = 0
    total_branches = 0
    covered_branches = 0

    files = coverage_data.get("files", {})

    for file_path, file_data in files.items():
        # 应用文件过滤
        if files_filter is not None:
            if file_path not in files_filter:
                continue

        summary = file_data.get("summary", {})

        if lines_filter is not None and file_path in lines_filter:
            # 增量覆盖率：只计算变更行
            changed_lines = lines_filter[file_path]
            executed_lines = set(file_data.get("executed_lines", []))
            missing_lines = set(file_data.get("missing_lines", []))

            # 只统计变更行中的语句
            changed_executed = changed_lines & executed_lines
            changed_missing = changed_lines & missing_lines
            file_statements = len(changed_executed) + len(changed_missing)
            file_covered = len(changed_executed)

            total_statements += file_statements
            covered_statements += file_covered

            # 分支覆盖率（简化处理：按比例计算）
            if summary.get("num_branches", 0) > 0:
                branch_ratio = file_statements / max(
                    summary.get("num_statements", 1), 1
                )
                total_branches += int(
                    summary.get("num_branches", 0) * branch_ratio
                )
                covered_branches += int(
                    summary.get("covered_branches", 0) * branch_ratio
                )
        else:
            # 全量覆盖率
            total_statements += summary.get("num_statements", 0)
            covered_statements += summary.get("covered_lines", 0)
            total_branches += summary.get("num_branches", 0)
            covered_branches += summary.get("covered_branches", 0)

    return CoverageResult(
        total_statements=total_statements,
        covered_statements=covered_statements,
        total_branches=total_branches,
        covered_branches=covered_branches,
    )


def calculate_directory_coverage(
    coverage_data: dict[str, Any], directory: str
) -> CoverageResult:
    """计算特定目录的覆盖率"""
    files = coverage_data.get("files", {})
    matching_files = [f for f in files.keys() if f.startswith(directory)]
    return calculate_coverage(coverage_data, files_filter=matching_files)


def print_coverage_report(
    full_coverage: CoverageResult,
    incremental_coverage: Optional[CoverageResult] = None,
    directory_coverages: Optional[dict[str, CoverageResult]] = None,
):
    """打印覆盖率报告"""
    print("\n" + "=" * 60)
    print("📊 覆盖率报告")
    print("=" * 60)

    print("\n📈 全量代码覆盖率:")
    print(f"   行覆盖率:   {full_coverage.line_coverage:.2f}%")
    print(
        f"             ({full_coverage.covered_statements}/{full_coverage.total_statements} 行)"
    )
    print(f"   分支覆盖率: {full_coverage.branch_coverage:.2f}%")
    print(
        f"             ({full_coverage.covered_branches}/{full_coverage.total_branches} 分支)"
    )

    print("\n📈 增量代码覆盖率 (相对于 基准分支):")
    if incremental_coverage and incremental_coverage.total_statements > 0:
        print(f"   行覆盖率:   {incremental_coverage.line_coverage:.2f}%")
        print(
            f"             ({incremental_coverage.covered_statements}/{incremental_coverage.total_statements} 行)"
        )
        print(f"   分支覆盖率: {incremental_coverage.branch_coverage:.2f}%")
        print(
            f"             ({incremental_coverage.covered_branches}/{incremental_coverage.total_branches} 分支)"
        )
    else:
        print("   ⚠️  无增量覆盖数据（未检测到变更或基准分支差异），增量检查已跳过。")

    if directory_coverages:
        print("\n📁 目录覆盖率:")
        for directory, coverage in directory_coverages.items():
            print(f"\n   {directory}:")
            print(f"      行覆盖率:   {coverage.line_coverage:.2f}%")
            print(f"      分支覆盖率: {coverage.branch_coverage:.2f}%")

    print("\n" + "=" * 60)


def check_coverage_thresholds(
    config: CoverageConfig,
    full_coverage: CoverageResult,
    incremental_coverage: Optional[CoverageResult] = None,
    directory_coverages: Optional[dict[str, CoverageResult]] = None,
) -> tuple[bool, list[str]]:
    """检查覆盖率是否达标

    Returns:
        bool: 是否通过检查
    """
    passed = True
    failures: list[str] = []
    print("\n🔍 覆盖率检查:")

    # 检查全量覆盖率
    full_threshold = config.full
    if full_coverage.branch_coverage < full_threshold.branch_coverage:
        msg = (
            f"全量分支覆盖率 {full_coverage.branch_coverage:.2f}% < {full_threshold.branch_coverage}%"
        )
        print(f"   ❌ {msg}")
        failures.append(msg)
        passed = False
    else:
        print(
            f"   ✅ 全量分支覆盖率 {full_coverage.branch_coverage:.2f}% "
            f">= {full_threshold.branch_coverage}%"
        )

    if full_coverage.line_coverage < full_threshold.line_coverage:
        msg = (
            f"全量行覆盖率 {full_coverage.line_coverage:.2f}% < {full_threshold.line_coverage}%"
        )
        print(f"   ❌ {msg}")
        failures.append(msg)
        passed = False
    else:
        print(
            f"   ✅ 全量行覆盖率 {full_coverage.line_coverage:.2f}% "
            f">= {full_threshold.line_coverage}%"
        )

    # 检查增量覆盖率（如果有）
    if incremental_coverage and incremental_coverage.total_statements > 0:
        incr_threshold = config.incremental
        if incremental_coverage.branch_coverage < incr_threshold.branch_coverage:
            msg = (
                f"增量分支覆盖率 {incremental_coverage.branch_coverage:.2f}% < {incr_threshold.branch_coverage}%"
            )
            print(f"   ❌ {msg}")
            failures.append(msg)
            passed = False
        else:
            print(
                f"   ✅ 增量分支覆盖率 {incremental_coverage.branch_coverage:.2f}% "
                f">= {incr_threshold.branch_coverage}%"
            )

        if incremental_coverage.line_coverage < incr_threshold.line_coverage:
            msg = (
                f"增量行覆盖率 {incremental_coverage.line_coverage:.2f}% < {incr_threshold.line_coverage}%"
            )
            print(f"   ❌ {msg}")
            failures.append(msg)
            passed = False
        else:
            print(
                f"   ✅ 增量行覆盖率 {incremental_coverage.line_coverage:.2f}% "
                f">= {incr_threshold.line_coverage}%"
            )

    # 检查目录覆盖率
    if directory_coverages:
        for directory, coverage in directory_coverages.items():
            # 全量覆盖率检查
            dir_full_threshold = config.get_threshold_for_directory(
                directory, is_incremental=False
            )

            if coverage.branch_coverage < dir_full_threshold.branch_coverage:
                msg = (
                    f"{directory} 分支覆盖率 {coverage.branch_coverage:.2f}% < {dir_full_threshold.branch_coverage}%"
                )
                print(f"   ❌ {msg}")
                failures.append(msg)
                passed = False
            else:
                print(
                    f"   ✅ {directory} 分支覆盖率 {coverage.branch_coverage:.2f}% "
                    f">= {dir_full_threshold.branch_coverage}%"
                )

            if coverage.line_coverage < dir_full_threshold.line_coverage:
                msg = (
                    f"{directory} 行覆盖率 {coverage.line_coverage:.2f}% < {dir_full_threshold.line_coverage}%"
                )
                print(f"   ❌ {msg}")
                failures.append(msg)
                passed = False
            else:
                print(
                    f"   ✅ {directory} 行覆盖率 {coverage.line_coverage:.2f}% "
                    f">= {dir_full_threshold.line_coverage}%"
                )

    return passed, failures


def main():
    parser = argparse.ArgumentParser(
        description="覆盖率检查脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-branch",
        default="main",
        help="增量覆盖率的基准分支（默认: main）",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="源代码目录（默认: agentrun）",
    )
    parser.add_argument(
        "--test-path",
        default="tests/unittests/",
        help="测试目录（默认: tests/unittests/）",
    )
    parser.add_argument(
        "--config",
        default="coverage.yaml",
        help="覆盖率配置文件（默认: coverage.yaml）",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="只运行测试，不检查覆盖率阈值",
    )
    parser.add_argument(
        "--check-directories",
        nargs="*",
        help="检查特定目录的覆盖率",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="只输出 JSON 格式的结果",
    )

    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    config = CoverageConfig.from_yaml(config_path)

    # 运行覆盖率测试
    print("🚀 运行覆盖率测试...")
    test_path = args.test_path
    if test_path == "tests/":  # 如果用户未指定 test-path 参数，则使用 unittests
        test_path = "tests/unittests/"

    if not run_coverage(
        source=args.source,
        test_path=test_path,
    ):
        print("❌ 测试失败")
        sys.exit(1)

    # 解析覆盖率数据
    coverage_data = parse_coverage_json()
    if not coverage_data:
        sys.exit(1)

    # 计算全量覆盖率（包含项目中所有文件）
    full_coverage = calculate_coverage(coverage_data)

    # 尝试计算增量覆盖率（与全量测试合并执行一次后再计算）
    print("🔎 计算增量覆盖率（与基准分支相比）...")
    incremental_coverage = None
    changed_lines = get_changed_lines(args.base_branch)
    if not changed_lines:
        print("⚠️  未检测到相对于基准分支的变更行；增量覆盖率将被跳过。")
    else:
        incremental_coverage = calculate_coverage(
            coverage_data,
            files_filter=list(changed_lines.keys()),
            lines_filter=changed_lines,
        )

    # 计算目录覆盖率：优先使用 config 中的 directory_overrides，然后合并 --check-directories 参数
    directory_coverages: dict[str, CoverageResult] = {}
    overrides = list(config.directory_overrides.keys()) if config.directory_overrides else []
    for directory in overrides:
        directory_coverages[directory] = calculate_directory_coverage(coverage_data, directory)

    if args.check_directories:
        for directory in args.check_directories:
            directory_coverages[directory] = calculate_directory_coverage(coverage_data, directory)

    # 包含覆盖率数据中存在但未在 YAML 中声明的目录，使用默认阈值进行计算
    files = coverage_data.get("files", {})
    discovered_dirs: set[str] = set()
    for f in files.keys():
        if f.startswith("agentrun/"):
            parts = f.split("/")
            if len(parts) >= 2:
                discovered_dirs.add("/".join(parts[:2]))

    for d in sorted(discovered_dirs):
        if d not in directory_coverages:
            directory_coverages[d] = calculate_directory_coverage(coverage_data, d)

    # 输出报告
    if args.json_only:
        result = {
            "full_coverage": {
                "line_coverage": full_coverage.line_coverage,
                "branch_coverage": full_coverage.branch_coverage,
                "total_statements": full_coverage.total_statements,
                "covered_statements": full_coverage.covered_statements,
                "total_branches": full_coverage.total_branches,
                "covered_branches": full_coverage.covered_branches,
            }
        }
        if incremental_coverage:
            result["incremental_coverage"] = {
                "line_coverage": incremental_coverage.line_coverage,
                "branch_coverage": incremental_coverage.branch_coverage,
                "total_statements": incremental_coverage.total_statements,
                "covered_statements": incremental_coverage.covered_statements,
                "total_branches": incremental_coverage.total_branches,
                "covered_branches": incremental_coverage.covered_branches,
            }
        if directory_coverages:
            result["directory_coverages"] = {
                d: {
                    "line_coverage": c.line_coverage,
                    "branch_coverage": c.branch_coverage,
                }
                for d, c in directory_coverages.items()
            }
        print(json.dumps(result, indent=2))
    else:
        print_coverage_report(
            full_coverage, incremental_coverage, directory_coverages
        )

    # 检查覆盖率阈值
    if not args.no_check:
        passed, failures = check_coverage_thresholds(
            config, full_coverage, incremental_coverage, directory_coverages
        )
        if not passed:
            print("\n❌ 覆盖率检查未通过")
            if failures:
                print("\n未通过项:")
                for f in failures:
                    print(f" - {f}")
            sys.exit(1)
        else:
            print("\n✅ 覆盖率检查通过")

    sys.exit(0)


if __name__ == "__main__":
    main()
