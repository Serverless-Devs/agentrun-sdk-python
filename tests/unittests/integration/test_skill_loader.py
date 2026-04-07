"""SkillLoader 单元测试 / SkillLoader Unit Tests

测试 Skill 加载器的核心功能：
- _parse_frontmatter() 函数
- SkillLoader 类（scan_skills / load_skill / to_common_toolset）
- skill_tools() 入口函数
- builtin/skill.py 导出
- 各框架 builtin 中的 skill_tools() 函数
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

import agentrun.integration.builtin.skill as _builtin_skill_mod
from agentrun.integration.utils.skill_loader import (
    _parse_frontmatter,
    skill_tools,
    SkillDetail,
    SkillInfo,
    SkillLoader,
)
from agentrun.integration.utils.tool import CommonToolSet

# =============================================================================
# Helper: 创建临时 skill 目录结构
# =============================================================================


def _create_skill_dir(
    base_dir: str,
    skill_name: str,
    *,
    skill_md_content: Optional[str] = None,
    package_json: Optional[Dict[str, Any]] = None,
    extra_files: Optional[Dict[str, str]] = None,
) -> str:
    """在 base_dir 下创建一个 skill 子目录，写入可选的 SKILL.md / package.json / 其他文件。"""
    skill_path = os.path.join(base_dir, skill_name)
    os.makedirs(skill_path, exist_ok=True)

    if skill_md_content is not None:
        with open(
            os.path.join(skill_path, "SKILL.md"), "w", encoding="utf-8"
        ) as fh:
            fh.write(skill_md_content)

    if package_json is not None:
        with open(
            os.path.join(skill_path, "package.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(package_json, fh)

    if extra_files:
        for filename, content in extra_files.items():
            file_path = os.path.join(skill_path, filename)
            sub_dir = os.path.dirname(file_path)
            if sub_dir and not os.path.isdir(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(content)

    return skill_path


# =============================================================================
# 1. _parse_frontmatter 测试
# =============================================================================


class TestParseFrontmatter:
    """测试 YAML frontmatter 解析函数"""

    def test_valid_frontmatter(self) -> None:
        content = (
            "---\nname: my-skill\ndescription: A test skill\nversion:"
            " 1.0.0\n---\n# Body"
        )
        result = _parse_frontmatter(content)
        assert result["name"] == "my-skill"
        assert result["description"] == "A test skill"
        assert result["version"] == "1.0.0"

    def test_no_frontmatter(self) -> None:
        content = "# Just a markdown file\nNo frontmatter here."
        result = _parse_frontmatter(content)
        assert result == {}

    def test_empty_string(self) -> None:
        result = _parse_frontmatter("")
        assert result == {}

    def test_quoted_values(self) -> None:
        content = (
            "---\nname: \"quoted-name\"\ndescription: 'single-quoted'\n---\n"
        )
        result = _parse_frontmatter(content)
        assert result["name"] == "quoted-name"
        assert result["description"] == "single-quoted"

    def test_empty_value(self) -> None:
        content = "---\nname: my-skill\ndescription:\n---\n"
        result = _parse_frontmatter(content)
        assert result["name"] == "my-skill"
        assert result["description"] == ""

    def test_colon_in_value(self) -> None:
        content = (
            "---\nname: my-skill\ndescription: A skill: does things\n---\n"
        )
        result = _parse_frontmatter(content)
        assert result["description"] == "A skill: does things"

    def test_blank_lines_in_frontmatter(self) -> None:
        content = "---\nname: my-skill\n\ndescription: test\n---\n"
        result = _parse_frontmatter(content)
        assert result["name"] == "my-skill"
        assert result["description"] == "test"

    def test_no_closing_delimiter(self) -> None:
        content = "---\nname: my-skill\ndescription: test\n"
        result = _parse_frontmatter(content)
        assert result == {}


# =============================================================================
# 2. SkillLoader.scan_skills 测试
# =============================================================================


class TestScanSkills:
    """测试 SkillLoader.scan_skills()"""

    def test_empty_directory(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.scan_skills()
        assert result == []

    def test_nonexistent_directory(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "nonexistent")
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.scan_skills()
        assert result == []

    def test_skill_with_frontmatter(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "my-skill",
            skill_md_content=(
                "---\nname: custom-name\ndescription: A great skill\nversion:"
                " 2.0\n---\n# Skill"
            ),
        )
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.scan_skills()
        assert len(result) == 1
        assert result[0].name == "custom-name"
        assert result[0].description == "A great skill"
        assert result[0].version == "2.0"

    def test_skill_with_package_json_only(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "pkg-skill",
            package_json={
                "name": "pkg-name",
                "description": "From package.json",
                "version": "3.0",
            },
        )
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.scan_skills()
        assert len(result) == 1
        assert result[0].name == "pkg-name"
        assert result[0].description == "From package.json"
        assert result[0].version == "3.0"

    def test_skill_with_no_metadata(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(skills_dir, "bare-skill")
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.scan_skills()
        assert len(result) == 1
        assert result[0].name == "bare-skill"
        assert result[0].description == ""

    def test_frontmatter_takes_priority_over_package_json(
        self, tmp_path: Any
    ) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "priority-skill",
            skill_md_content=(
                "---\nname: from-frontmatter\ndescription: FM desc\n---\n"
            ),
            package_json={"name": "from-pkg", "description": "PKG desc"},
        )
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.scan_skills()
        assert len(result) == 1
        assert result[0].name == "from-frontmatter"
        assert result[0].description == "FM desc"

    def test_multiple_skills_sorted(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(skills_dir, "beta-skill")
        _create_skill_dir(skills_dir, "alpha-skill")
        _create_skill_dir(skills_dir, "gamma-skill")
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.scan_skills()
        assert len(result) == 3
        assert [s.name for s in result] == [
            "alpha-skill",
            "beta-skill",
            "gamma-skill",
        ]

    def test_hidden_directories_skipped(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(skills_dir, ".hidden-skill")
        _create_skill_dir(skills_dir, "visible-skill")
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.scan_skills()
        assert len(result) == 1
        assert result[0].name == "visible-skill"

    def test_files_in_root_are_skipped(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        with open(os.path.join(skills_dir, "not-a-skill.txt"), "w") as fh:
            fh.write("just a file")
        _create_skill_dir(skills_dir, "real-skill")
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.scan_skills()
        assert len(result) == 1
        assert result[0].name == "real-skill"

    def test_cache_is_used(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(skills_dir, "cached-skill")
        loader = SkillLoader(skills_dir=skills_dir)
        first_result = loader.scan_skills()
        # Add another skill after first scan
        _create_skill_dir(skills_dir, "new-skill")
        second_result = loader.scan_skills()
        # Should return cached result
        assert first_result is second_result
        assert len(second_result) == 1

    def test_malformed_package_json(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        skill_path = os.path.join(skills_dir, "bad-pkg")
        os.makedirs(skill_path)
        with open(os.path.join(skill_path, "package.json"), "w") as fh:
            fh.write("{invalid json")
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.scan_skills()
        assert len(result) == 1
        assert result[0].name == "bad-pkg"

    def test_skill_md_without_frontmatter_falls_to_package_json(
        self, tmp_path: Any
    ) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "fallback-skill",
            skill_md_content="# No frontmatter here\nJust content.",
            package_json={"name": "pkg-fallback", "description": "From pkg"},
        )
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.scan_skills()
        assert len(result) == 1
        assert result[0].name == "pkg-fallback"
        assert result[0].description == "From pkg"


# =============================================================================
# 3. SkillLoader.load_skill 测试
# =============================================================================


class TestLoadSkill:
    """测试 SkillLoader.load_skill()"""

    def test_load_existing_skill(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        md_content = (
            "---\nname: test-skill\ndescription: Test\n---\n# Instructions\nDo"
            " stuff."
        )
        _create_skill_dir(
            skills_dir,
            "test-skill",
            skill_md_content=md_content,
            extra_files={"helper.py": "print('hello')"},
        )
        loader = SkillLoader(skills_dir=skills_dir)
        detail = loader.load_skill("test-skill")
        assert detail is not None
        assert detail.name == "test-skill"
        assert detail.description == "Test"
        assert detail.instruction == md_content
        assert "SKILL.md" in detail.files
        assert "helper.py" in detail.files

    def test_load_nonexistent_skill(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        detail = loader.load_skill("nonexistent")
        assert detail is None

    def test_load_skill_with_subdirectory(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        skill_path = _create_skill_dir(
            skills_dir,
            "dir-skill",
            skill_md_content="---\nname: dir-skill\n---\n",
        )
        sub_dir = os.path.join(skill_path, "scripts")
        os.makedirs(sub_dir)
        with open(os.path.join(sub_dir, "run.sh"), "w") as fh:
            fh.write("#!/bin/bash")
        loader = SkillLoader(skills_dir=skills_dir)
        detail = loader.load_skill("dir-skill")
        assert detail is not None
        assert "scripts/" in detail.files

    def test_load_skill_hidden_files_excluded(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "hidden-files-skill",
            skill_md_content="---\nname: hidden-files-skill\n---\n",
            extra_files={".hidden": "secret", "visible.txt": "public"},
        )
        loader = SkillLoader(skills_dir=skills_dir)
        detail = loader.load_skill("hidden-files-skill")
        assert detail is not None
        assert ".hidden" not in detail.files
        assert "visible.txt" in detail.files

    def test_load_skill_without_skill_md(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "no-md-skill",
            extra_files={"readme.txt": "hello"},
        )
        loader = SkillLoader(skills_dir=skills_dir)
        detail = loader.load_skill("no-md-skill")
        assert detail is not None
        assert detail.instruction == ""
        assert "readme.txt" in detail.files


# =============================================================================
# 4. SkillLoader._build_tool_description 测试
# =============================================================================


class TestBuildToolDescription:
    """测试 load_skills 工具描述的构建"""

    def test_no_skills(self) -> None:
        loader = SkillLoader(skills_dir="/nonexistent")
        desc = loader._build_tool_description([])
        assert "No skills available" in desc

    def test_with_skills(self) -> None:
        loader = SkillLoader(skills_dir="/nonexistent")
        skills = [
            SkillInfo(name="alpha", description="Alpha skill"),
            SkillInfo(name="beta", description=""),
        ]
        desc = loader._build_tool_description(skills)
        assert "alpha: Alpha skill" in desc
        assert "- beta" in desc
        assert "Available skills:" in desc


# =============================================================================
# 5. SkillLoader._load_skills_func 测试
# =============================================================================


class TestLoadSkillsFunc:
    """测试 load_skills 工具的执行函数"""

    def test_list_all_skills(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "skill-a",
            skill_md_content="---\nname: skill-a\ndescription: Skill A\n---\n",
        )
        _create_skill_dir(
            skills_dir,
            "skill-b",
            skill_md_content="---\nname: skill-b\ndescription: Skill B\n---\n",
        )
        loader = SkillLoader(skills_dir=skills_dir)
        result_json = loader._load_skills_func(name=None)
        result = json.loads(result_json)
        assert "skills" in result
        assert len(result["skills"]) == 2
        names = [s["name"] for s in result["skills"]]
        assert "skill-a" in names
        assert "skill-b" in names

    def test_list_with_empty_string(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(skills_dir, "only-skill")
        loader = SkillLoader(skills_dir=skills_dir)
        result_json = loader._load_skills_func(name="")
        result = json.loads(result_json)
        assert "skills" in result

    def test_load_specific_skill(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        md_content = (
            "---\nname: target\ndescription: Target skill\n---\n# Instructions"
        )
        _create_skill_dir(
            skills_dir,
            "target",
            skill_md_content=md_content,
        )
        loader = SkillLoader(skills_dir=skills_dir)
        result_json = loader._load_skills_func(name="target")
        result = json.loads(result_json)
        assert result["name"] == "target"
        assert result["description"] == "Target skill"
        assert "instruction" in result
        assert "files" in result

    def test_load_nonexistent_skill_returns_error(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(skills_dir, "existing")
        loader = SkillLoader(skills_dir=skills_dir)
        result_json = loader._load_skills_func(name="missing")
        result = json.loads(result_json)
        assert "error" in result
        assert "missing" in result["error"]
        assert "existing" in result["error"]


# =============================================================================
# 6. SkillLoader.to_common_toolset 测试
# =============================================================================


class TestToCommonToolset:
    """测试 to_common_toolset() 返回的 CommonToolSet"""

    def test_returns_common_toolset(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "test-skill",
            skill_md_content="---\nname: test-skill\ndescription: Test\n---\n",
        )
        loader = SkillLoader(skills_dir=skills_dir)
        toolset = loader.to_common_toolset()
        assert isinstance(toolset, CommonToolSet)

    def test_toolset_has_load_skills_tool(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(skills_dir, "test-skill")
        loader = SkillLoader(skills_dir=skills_dir)
        toolset = loader.to_common_toolset()
        tools_list = toolset.tools()
        assert len(tools_list) == 3
        tool_names = [t.name for t in tools_list]
        assert "load_skills" in tool_names
        assert "read_skill_file" in tool_names
        assert "execute_command" in tool_names

    def test_tool_description_contains_skill_names(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "alpha",
            skill_md_content="---\nname: alpha\ndescription: Alpha desc\n---\n",
        )
        loader = SkillLoader(skills_dir=skills_dir)
        toolset = loader.to_common_toolset()
        tool = toolset.tools()[0]
        assert "alpha" in tool.description
        assert "Alpha desc" in tool.description

    def test_tool_has_name_parameter(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(skills_dir, "test-skill")
        loader = SkillLoader(skills_dir=skills_dir)
        toolset = loader.to_common_toolset()
        tool = toolset.tools()[0]
        # CanonicalTool.parameters is a JSON schema dict
        assert "properties" in tool.parameters
        assert "name" in tool.parameters["properties"]
        name_prop = tool.parameters["properties"]["name"]
        assert name_prop["type"] == "string"

    def test_tool_func_is_callable(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "callable-skill",
            skill_md_content="---\nname: callable-skill\n---\n",
        )
        loader = SkillLoader(skills_dir=skills_dir)
        toolset = loader.to_common_toolset()
        tool = toolset.tools()[0]
        result_json = tool.func()
        result = json.loads(result_json)
        assert "skills" in result

    def test_empty_skills_dir_still_returns_toolset(
        self, tmp_path: Any
    ) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        toolset = loader.to_common_toolset()
        assert isinstance(toolset, CommonToolSet)
        tools_list = toolset.tools()
        assert len(tools_list) == 3
        load_skills_tool = [t for t in tools_list if t.name == "load_skills"][0]
        assert "No skills available" in load_skills_tool.description


# =============================================================================
# 7. skill_tools() 入口函数测试
# =============================================================================


class TestSkillToolsFunction:
    """测试 skill_tools() 入口函数"""

    def test_local_only(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "local-skill",
            skill_md_content=(
                "---\nname: local-skill\ndescription: Local\n---\n"
            ),
        )
        toolset = skill_tools(skills_dir=skills_dir)
        assert isinstance(toolset, CommonToolSet)
        assert len(toolset.tools()) == 3

    def test_with_string_name_triggers_remote_download(
        self, tmp_path: Any
    ) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)

        mock_tool_client = MagicMock()
        mock_tool_resource = MagicMock()
        mock_tool_client.return_value.get.return_value = mock_tool_resource

        with patch(
            "agentrun.integration.utils.skill_loader.SkillLoader._ensure_skills_available"
        ) as mock_ensure:
            toolset = skill_tools(name="remote-skill", skills_dir=skills_dir)
            assert isinstance(toolset, CommonToolSet)

    def test_with_list_of_names(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)

        with patch(
            "agentrun.integration.utils.skill_loader.SkillLoader._ensure_skills_available"
        ):
            toolset = skill_tools(
                name=["skill-a", "skill-b"], skills_dir=skills_dir
            )
            assert isinstance(toolset, CommonToolSet)

    def test_with_tool_resource_instance(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)

        mock_resource = MagicMock()
        mock_resource.name = "resource-skill"

        toolset = skill_tools(name=mock_resource, skills_dir=skills_dir)
        assert isinstance(toolset, CommonToolSet)
        mock_resource.download_skill.assert_called_once()

    def test_with_tool_resource_already_downloaded(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        # Pre-create the skill directory so download is skipped
        _create_skill_dir(skills_dir, "existing-resource")

        mock_resource = MagicMock()
        mock_resource.name = "existing-resource"

        toolset = skill_tools(name=mock_resource, skills_dir=skills_dir)
        assert isinstance(toolset, CommonToolSet)
        mock_resource.download_skill.assert_not_called()

    def test_none_name_loads_local_only(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(skills_dir, "local-only")
        toolset = skill_tools(name=None, skills_dir=skills_dir)
        assert isinstance(toolset, CommonToolSet)


# =============================================================================
# 8. _ensure_skills_available 测试
# =============================================================================


class TestEnsureSkillsAvailable:
    """测试远程 skill 下载逻辑"""

    def test_no_remote_names_does_nothing(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir, remote_skill_names=[])
        # Should not raise
        loader._ensure_skills_available()

    def test_existing_skill_skips_download(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(skills_dir, "already-here")

        loader = SkillLoader(
            skills_dir=skills_dir, remote_skill_names=["already-here"]
        )
        with patch("agentrun.tool.client.ToolClient") as mock_client:
            loader._ensure_skills_available()
            mock_client.assert_not_called()

    def test_missing_skill_triggers_download(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)

        mock_tool_resource = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.get.return_value = mock_tool_resource

        loader = SkillLoader(
            skills_dir=skills_dir, remote_skill_names=["new-skill"]
        )
        with patch(
            "agentrun.tool.client.ToolClient",
            return_value=mock_client_instance,
        ):
            loader._ensure_skills_available()
            mock_client_instance.get.assert_called_once_with(
                name="new-skill", config=None
            )
            mock_tool_resource.download_skill.assert_called_once_with(
                target_dir=skills_dir, config=None
            )


# =============================================================================
# 9. builtin/skill.py 导出测试
# =============================================================================


class TestBuiltinSkillExport:
    """测试 builtin/skill.py 的导出"""

    def test_skill_tools_is_exported(self) -> None:
        assert hasattr(_builtin_skill_mod, "skill_tools")
        assert callable(_builtin_skill_mod.skill_tools)

    def test_skill_tools_in_all(self) -> None:
        assert "skill_tools" in _builtin_skill_mod.__all__

    def test_import_from_builtin_init(self) -> None:
        from agentrun.integration.builtin import skill_tools as imported_fn

        assert callable(imported_fn)


# =============================================================================
# 10. 各框架 builtin skill_tools 测试
# =============================================================================


class TestFrameworkBuiltinSkillTools:
    """测试各框架 builtin 中的 skill_tools() 函数"""

    def _run_framework_test(self, framework_module_path: str) -> None:
        """通用框架测试：mock builtin skill_tools 返回 CommonToolSet，
        验证框架 skill_tools 调用了正确的转换方法。"""
        mock_toolset = MagicMock(spec=CommonToolSet)
        mock_toolset.to_langchain.return_value = [MagicMock()]
        mock_toolset.to_google_adk.return_value = [MagicMock()]
        mock_toolset.to_crewai.return_value = [MagicMock()]
        mock_toolset.to_langgraph.return_value = [MagicMock()]
        mock_toolset.to_pydantic_ai.return_value = [MagicMock()]
        mock_toolset.to_agentscope.return_value = [MagicMock()]

        with patch(
            f"{framework_module_path}._skill_tools",
            return_value=mock_toolset,
        ):
            module = sys.modules.get(framework_module_path)
            if module is None:
                import importlib

                module = importlib.import_module(framework_module_path)
            result = module.skill_tools(skills_dir=".test-skills")
            assert isinstance(result, list)
            assert len(result) == 1

    def test_langchain_skill_tools(self) -> None:
        self._run_framework_test("agentrun.integration.langchain.builtin")

    def test_google_adk_skill_tools(self) -> None:
        self._run_framework_test("agentrun.integration.google_adk.builtin")

    def test_crewai_skill_tools(self) -> None:
        self._run_framework_test("agentrun.integration.crewai.builtin")

    def test_langgraph_skill_tools(self) -> None:
        self._run_framework_test("agentrun.integration.langgraph.builtin")

    def test_pydantic_ai_skill_tools(self) -> None:
        self._run_framework_test("agentrun.integration.pydantic_ai.builtin")

    def test_agentscope_skill_tools(self) -> None:
        self._run_framework_test("agentrun.integration.agentscope.builtin")

    def test_framework_import_from_init(self) -> None:
        """验证各框架 __init__.py 正确导出 skill_tools"""
        from agentrun.integration.agentscope import skill_tools as as_fn
        from agentrun.integration.crewai import skill_tools as crew_fn
        from agentrun.integration.google_adk import skill_tools as adk_fn
        from agentrun.integration.langchain import skill_tools as lc_fn
        from agentrun.integration.langgraph import skill_tools as lg_fn
        from agentrun.integration.pydantic_ai import skill_tools as pai_fn

        assert callable(lc_fn)
        assert callable(adk_fn)
        assert callable(crew_fn)
        assert callable(lg_fn)
        assert callable(pai_fn)
        assert callable(as_fn)


# =============================================================================
# 11. SkillInfo / SkillDetail 数据类测试
# =============================================================================


class TestDataClasses:
    """测试 SkillInfo 和 SkillDetail 数据类"""

    def test_skill_info_defaults(self) -> None:
        info = SkillInfo(name="test")
        assert info.name == "test"
        assert info.description == ""
        assert info.version == ""
        assert info.path == ""

    def test_skill_info_with_all_fields(self) -> None:
        info = SkillInfo(
            name="full", description="desc", version="1.0", path="/path"
        )
        assert info.name == "full"
        assert info.description == "desc"
        assert info.version == "1.0"
        assert info.path == "/path"

    def test_skill_detail_defaults(self) -> None:
        detail = SkillDetail(name="test")
        assert detail.name == "test"
        assert detail.instruction == ""
        assert detail.files == []

    def test_skill_detail_inherits_skill_info(self) -> None:
        detail = SkillDetail(
            name="full",
            description="desc",
            version="1.0",
            path="/path",
            instruction="# Do stuff",
            files=["a.py", "b.py"],
        )
        assert isinstance(detail, SkillInfo)
        assert detail.instruction == "# Do stuff"
        assert detail.files == ["a.py", "b.py"]


# =============================================================================
# 12. 端到端集成测试
# =============================================================================


class TestEndToEnd:
    """端到端测试：从创建 skill 目录到调用 load_skills 工具"""

    def test_full_workflow(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)

        md_content = (
            "---\n"
            "name: e2e-skill\n"
            "description: End-to-end test skill\n"
            "version: 1.0.0\n"
            "---\n"
            "\n"
            "# E2E Skill\n"
            "\n"
            "Follow these instructions to use the skill.\n"
        )
        _create_skill_dir(
            skills_dir,
            "e2e-skill",
            skill_md_content=md_content,
            extra_files={"scripts/run.sh": "#!/bin/bash\necho hello"},
        )

        toolset = skill_tools(skills_dir=skills_dir)
        assert isinstance(toolset, CommonToolSet)
        tools_list = toolset.tools()
        assert len(tools_list) == 3

        tool_map = {t.name: t for t in tools_list}
        tool = tool_map["load_skills"]
        assert "e2e-skill" in tool.description

        # List all skills
        list_result = json.loads(tool.func())
        assert len(list_result["skills"]) == 1
        assert list_result["skills"][0]["name"] == "e2e-skill"

        # Load specific skill
        detail_result = json.loads(tool.func(name="e2e-skill"))
        assert detail_result["name"] == "e2e-skill"
        assert detail_result["description"] == "End-to-end test skill"
        assert "Follow these instructions" in detail_result["instruction"]
        assert "SKILL.md" in detail_result["files"]
        assert "scripts/" in detail_result["files"]

        # Load nonexistent skill
        error_result = json.loads(tool.func(name="nonexistent"))
        assert "error" in error_result
        assert "e2e-skill" in error_result["error"]

    def test_multiple_skills_workflow(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)

        _create_skill_dir(
            skills_dir,
            "skill-alpha",
            skill_md_content=(
                "---\nname: skill-alpha\ndescription: Alpha\n---\n# Alpha"
            ),
        )
        _create_skill_dir(
            skills_dir,
            "skill-beta",
            package_json={"name": "skill-beta", "description": "Beta"},
        )

        toolset = skill_tools(skills_dir=skills_dir)
        tool = toolset.tools()[0]

        list_result = json.loads(tool.func())
        assert len(list_result["skills"]) == 2

        alpha = json.loads(tool.func(name="skill-alpha"))
        assert alpha["name"] == "skill-alpha"
        assert "# Alpha" in alpha["instruction"]

        beta = json.loads(tool.func(name="skill-beta"))
        assert beta["name"] == "skill-beta"
        assert beta["instruction"] == ""


# =============================================================================
# 13. read_skill_file 工具测试
# =============================================================================


class TestReadSkillFile:
    """测试 _read_skill_file_func 方法"""

    def _get_read_skill_file_tool(self, skills_dir: str, **kwargs: Any) -> Any:
        loader = SkillLoader(skills_dir=skills_dir, **kwargs)
        toolset = loader.to_common_toolset()
        tool_map = {t.name: t for t in toolset.tools()}
        return tool_map["read_skill_file"]

    def test_read_existing_file(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "my-skill",
            skill_md_content="---\nname: my-skill\n---\n# Hello",
            extra_files={"config.json": '{"key": "value"}'},
        )
        loader = SkillLoader(skills_dir=skills_dir)
        result = json.loads(
            loader._read_skill_file_func("my-skill", "config.json")
        )
        assert "content" in result
        assert '"key": "value"' in result["content"]

    def test_file_not_found(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "my-skill",
            skill_md_content="---\nname: my-skill\n---\n",
        )
        loader = SkillLoader(skills_dir=skills_dir)
        result = json.loads(
            loader._read_skill_file_func("my-skill", "nonexistent.txt")
        )
        assert "error" in result
        assert "not found" in result["error"]

    def test_skill_not_found(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "existing-skill",
            skill_md_content="---\nname: existing-skill\n---\n",
        )
        loader = SkillLoader(skills_dir=skills_dir)
        result = json.loads(
            loader._read_skill_file_func("no-such-skill", "file.txt")
        )
        assert "error" in result
        assert "not found" in result["error"]
        assert "existing-skill" in result["error"]

    def test_path_traversal_with_dotdot(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "my-skill",
            skill_md_content="---\nname: my-skill\n---\n",
        )
        # Create a file outside the skill dir
        with open(tmp_path / "secret.txt", "w") as fh:
            fh.write("secret data")

        loader = SkillLoader(skills_dir=skills_dir)
        result = json.loads(
            loader._read_skill_file_func("my-skill", "../../secret.txt")
        )
        assert "error" in result
        assert (
            "outside" in result["error"].lower()
            or "denied" in result["error"].lower()
        )

    def test_path_traversal_with_absolute_path(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "my-skill",
            skill_md_content="---\nname: my-skill\n---\n",
        )
        loader = SkillLoader(skills_dir=skills_dir)
        result = json.loads(
            loader._read_skill_file_func("my-skill", "/etc/passwd")
        )
        assert "error" in result
        assert (
            "outside" in result["error"].lower()
            or "denied" in result["error"].lower()
        )

    def test_binary_file(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        skill_path = _create_skill_dir(
            skills_dir,
            "my-skill",
            skill_md_content="---\nname: my-skill\n---\n",
        )
        # Write a binary file
        with open(os.path.join(skill_path, "data.bin"), "wb") as fh:
            fh.write(bytes(range(256)))

        loader = SkillLoader(skills_dir=skills_dir)
        result = json.loads(
            loader._read_skill_file_func("my-skill", "data.bin")
        )
        assert "error" in result
        assert "binary" in result["error"].lower()

    def test_directory_listing(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "my-skill",
            skill_md_content="---\nname: my-skill\n---\n",
            extra_files={
                "scripts/run.sh": "#!/bin/bash\necho hi",
                "scripts/setup.py": "print('setup')",
            },
        )
        loader = SkillLoader(skills_dir=skills_dir)
        result = json.loads(loader._read_skill_file_func("my-skill", "scripts"))
        assert "files" in result
        assert "run.sh" in result["files"]
        assert "setup.py" in result["files"]

    def test_read_skill_file_via_tool(self, tmp_path: Any) -> None:
        """Test read_skill_file via the Tool object's func"""
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "tool-skill",
            skill_md_content="---\nname: tool-skill\n---\n",
            extra_files={"readme.txt": "Hello from tool"},
        )
        tool = self._get_read_skill_file_tool(skills_dir)
        result = json.loads(
            tool.func(name="tool-skill", relative_path="readme.txt")
        )
        assert "content" in result
        assert "Hello from tool" in result["content"]


# =============================================================================
# 14. execute_command 工具测试
# =============================================================================


class TestExecuteCommand:
    """测试 _execute_command_func 方法"""

    def test_normal_execution(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        result = json.loads(loader._execute_command_func("echo hello"))
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]
        assert result["timed_out"] is False

    def test_nonzero_exit_code(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        result = json.loads(loader._execute_command_func("exit 42"))
        assert result["exit_code"] == 42
        assert result["timed_out"] is False

    def test_stderr_output(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        result = json.loads(loader._execute_command_func("echo error_msg >&2"))
        assert "error_msg" in result["stderr"]

    def test_custom_cwd(self, tmp_path: Any) -> None:
        custom_dir = str(tmp_path / "custom")
        os.makedirs(custom_dir)
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        result = json.loads(loader._execute_command_func("pwd", cwd=custom_dir))
        assert result["exit_code"] == 0
        assert custom_dir in result["stdout"]

    def test_nonexistent_cwd(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        result = json.loads(
            loader._execute_command_func(
                "echo hi", cwd="/nonexistent/path/12345"
            )
        )
        assert "error" in result
        assert "does not exist" in result["error"]

    def test_timeout(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir, command_timeout=1)
        result = json.loads(loader._execute_command_func("sleep 10", timeout=1))
        assert result["timed_out"] is True
        assert result["exit_code"] == -1

    def test_output_truncation(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        # Generate output larger than 100KB
        large_output_cmd = "python3 -c \"print('A' * 200000)\""
        result = json.loads(loader._execute_command_func(large_output_cmd))
        assert result["exit_code"] == 0
        assert "truncated" in result["stdout"]

    def test_command_approval_approved(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        approval_calls: List[tuple[str, str]] = []

        def approve(command: str, cwd: str) -> bool:
            approval_calls.append((command, cwd))
            return True

        loader = SkillLoader(skills_dir=skills_dir, command_approval=approve)
        result = json.loads(loader._execute_command_func("echo approved"))
        assert result["exit_code"] == 0
        assert "approved" in result["stdout"]
        assert len(approval_calls) == 1
        assert approval_calls[0][0] == "echo approved"

    def test_command_approval_rejected(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)

        def reject(command: str, cwd: str) -> bool:
            return False

        loader = SkillLoader(skills_dir=skills_dir, command_approval=reject)
        result = json.loads(loader._execute_command_func("echo should_not_run"))
        assert "error" in result
        assert "rejected" in result["error"].lower()

    def test_no_command_approval(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir, command_approval=None)
        result = json.loads(loader._execute_command_func("echo no_approval"))
        assert result["exit_code"] == 0
        assert "no_approval" in result["stdout"]

    def test_command_approval_raises_exception(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)

        def broken_approval(command: str, cwd: str) -> bool:
            raise RuntimeError("approval callback broken")

        loader = SkillLoader(
            skills_dir=skills_dir, command_approval=broken_approval
        )
        result = json.loads(loader._execute_command_func("echo should_not_run"))
        assert "error" in result
        assert (
            "approval callback" in result["error"].lower()
            or "broken" in result["error"].lower()
        )

    def test_default_cwd_is_skills_dir(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        result = json.loads(loader._execute_command_func("pwd"))
        assert result["exit_code"] == 0
        # The resolved real path should match
        assert os.path.realpath(skills_dir) in os.path.realpath(
            result["stdout"].strip()
        )

    def test_execute_command_via_tool(self, tmp_path: Any) -> None:
        """Test execute_command via the Tool object's func"""
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        toolset = loader.to_common_toolset()
        tool_map = {t.name: t for t in toolset.tools()}
        tool = tool_map["execute_command"]
        result = json.loads(tool.func(command="echo via_tool"))
        assert result["exit_code"] == 0
        assert "via_tool" in result["stdout"]


# =============================================================================
# 15. skill_tools() 新参数测试
# =============================================================================


class TestSkillToolsNewParams:
    """测试 skill_tools() 的 command_approval 和 command_timeout 参数"""

    def test_command_approval_passed_through(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        approval_called = False

        def approval(command: str, cwd: str) -> bool:
            nonlocal approval_called
            approval_called = True
            return True

        toolset = skill_tools(skills_dir=skills_dir, command_approval=approval)
        tool_map = {t.name: t for t in toolset.tools()}
        exec_tool = tool_map["execute_command"]
        result = json.loads(exec_tool.func(command="echo test"))
        assert approval_called
        assert result["exit_code"] == 0

    def test_command_timeout_passed_through(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        toolset = skill_tools(skills_dir=skills_dir, command_timeout=1)
        tool_map = {t.name: t for t in toolset.tools()}
        exec_tool = tool_map["execute_command"]
        result = json.loads(exec_tool.func(command="sleep 10"))
        assert result["timed_out"] is True

    def test_default_values(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        toolset = skill_tools(skills_dir=skills_dir)
        tool_map = {t.name: t for t in toolset.tools()}
        exec_tool = tool_map["execute_command"]
        # Default timeout is 30, command should succeed quickly
        result = json.loads(exec_tool.func(command="echo default"))
        assert result["exit_code"] == 0
        assert "default" in result["stdout"]


# =============================================================================
# 16. to_common_toolset() 返回 3 个工具测试
# =============================================================================


class TestToCommonToolsetThreeTools:
    """测试 to_common_toolset() 返回包含 3 个工具的 CommonToolSet"""

    def test_returns_three_tools(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        _create_skill_dir(
            skills_dir,
            "test-skill",
            skill_md_content="---\nname: test-skill\n---\n",
        )
        loader = SkillLoader(skills_dir=skills_dir)
        toolset = loader.to_common_toolset()
        tools_list = toolset.tools()
        assert len(tools_list) == 3
        tool_names = {t.name for t in tools_list}
        assert tool_names == {
            "load_skills",
            "read_skill_file",
            "execute_command",
        }

    def test_load_skills_tool_has_correct_params(self, tmp_path: Any) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        toolset = loader.to_common_toolset()
        tool_map = {t.name: t for t in toolset.tools()}
        load_tool = tool_map["load_skills"]
        assert "name" in load_tool.parameters["properties"]

    def test_read_skill_file_tool_has_correct_params(
        self, tmp_path: Any
    ) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        toolset = loader.to_common_toolset()
        tool_map = {t.name: t for t in toolset.tools()}
        read_tool = tool_map["read_skill_file"]
        assert "name" in read_tool.parameters["properties"]
        assert "relative_path" in read_tool.parameters["properties"]
        required = read_tool.parameters.get("required", [])
        assert "name" in required
        assert "relative_path" in required

    def test_execute_command_tool_has_correct_params(
        self, tmp_path: Any
    ) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        toolset = loader.to_common_toolset()
        tool_map = {t.name: t for t in toolset.tools()}
        exec_tool = tool_map["execute_command"]
        assert "command" in exec_tool.parameters["properties"]
        assert "cwd" in exec_tool.parameters["properties"]
        assert "timeout" in exec_tool.parameters["properties"]
        required = exec_tool.parameters.get("required", [])
        assert "command" in required

    def test_execute_command_description_has_safety_warning(
        self, tmp_path: Any
    ) -> None:
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        loader = SkillLoader(skills_dir=skills_dir)
        toolset = loader.to_common_toolset()
        tool_map = {t.name: t for t in toolset.tools()}
        exec_tool = tool_map["execute_command"]
        assert "IMPORTANT" in exec_tool.description
        assert "approval" in exec_tool.description.lower()


# =============================================================================
# 17. ALLOW_EXECUTE_COMMAND 环境变量控制测试
# =============================================================================


class TestAllowExecuteCommandEnvVar:
    """测试 ALLOW_EXECUTE_COMMAND 环境变量对 execute_command 工具加载的控制"""

    def test_default_includes_execute_command(self, tmp_path: Any) -> None:
        """未设置环境变量时，默认包含 execute_command 工具"""
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALLOW_EXECUTE_COMMAND", None)
            loader = SkillLoader(skills_dir=skills_dir)
            toolset = loader.to_common_toolset()
            tool_names = [t.name for t in toolset.tools()]
            assert "execute_command" in tool_names
            assert len(toolset.tools()) == 3

    def test_env_true_includes_execute_command(self, tmp_path: Any) -> None:
        """ALLOW_EXECUTE_COMMAND=true 时，包含 execute_command 工具"""
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        with patch.dict(os.environ, {"ALLOW_EXECUTE_COMMAND": "true"}):
            loader = SkillLoader(skills_dir=skills_dir)
            toolset = loader.to_common_toolset()
            tool_names = [t.name for t in toolset.tools()]
            assert "execute_command" in tool_names
            assert len(toolset.tools()) == 3

    def test_env_false_excludes_execute_command(self, tmp_path: Any) -> None:
        """ALLOW_EXECUTE_COMMAND=false 时，不包含 execute_command 工具"""
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        with patch.dict(os.environ, {"ALLOW_EXECUTE_COMMAND": "false"}):
            loader = SkillLoader(skills_dir=skills_dir)
            toolset = loader.to_common_toolset()
            tool_names = [t.name for t in toolset.tools()]
            assert "execute_command" not in tool_names
            assert len(toolset.tools()) == 2
            assert "load_skills" in tool_names
            assert "read_skill_file" in tool_names

    def test_env_false_case_insensitive(self, tmp_path: Any) -> None:
        """ALLOW_EXECUTE_COMMAND=False / FALSE 等大小写变体均生效"""
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        for value in ("False", "FALSE", "fAlSe"):
            with patch.dict(os.environ, {"ALLOW_EXECUTE_COMMAND": value}):
                loader = SkillLoader(skills_dir=skills_dir)
                toolset = loader.to_common_toolset()
                tool_names = [t.name for t in toolset.tools()]
                assert (
                    "execute_command" not in tool_names
                ), f"execute_command should be excluded for value={value!r}"

    def test_env_non_false_includes_execute_command(
        self, tmp_path: Any
    ) -> None:
        """ALLOW_EXECUTE_COMMAND 设置为非 false 的值时，包含 execute_command"""
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        for value in ("True", "TRUE", "1", "yes", "anything"):
            with patch.dict(os.environ, {"ALLOW_EXECUTE_COMMAND": value}):
                loader = SkillLoader(skills_dir=skills_dir)
                toolset = loader.to_common_toolset()
                tool_names = [t.name for t in toolset.tools()]
                assert (
                    "execute_command" in tool_names
                ), f"execute_command should be included for value={value!r}"

    def test_is_execute_command_allowed_static_method(self) -> None:
        """直接测试 _is_execute_command_allowed 静态方法"""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALLOW_EXECUTE_COMMAND", None)
            assert SkillLoader._is_execute_command_allowed() is True

        with patch.dict(os.environ, {"ALLOW_EXECUTE_COMMAND": "true"}):
            assert SkillLoader._is_execute_command_allowed() is True

        with patch.dict(os.environ, {"ALLOW_EXECUTE_COMMAND": "false"}):
            assert SkillLoader._is_execute_command_allowed() is False

        with patch.dict(os.environ, {"ALLOW_EXECUTE_COMMAND": "False"}):
            assert SkillLoader._is_execute_command_allowed() is False

    def test_skill_tools_func_respects_env_var(self, tmp_path: Any) -> None:
        """测试顶层 skill_tools() 函数也受环境变量控制"""
        skills_dir = str(tmp_path / "skills")
        os.makedirs(skills_dir)
        with patch.dict(os.environ, {"ALLOW_EXECUTE_COMMAND": "false"}):
            toolset = skill_tools(skills_dir=skills_dir)
            tool_names = [t.name for t in toolset.tools()]
            assert "execute_command" not in tool_names
            assert "load_skills" in tool_names
            assert "read_skill_file" in tool_names
