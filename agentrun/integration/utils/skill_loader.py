"""Skill 加载器模块 / Skill Loader Module

提供从本地 .skills 目录加载 Skill 包的能力，并构造 load_skills 工具供 Agent 运行时调用。
Provides the ability to load Skill packages from a local .skills directory
and construct a load_skills tool for Agent runtime invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from agentrun.integration.utils.tool import CommonToolSet, Tool, ToolParameter
from agentrun.utils.log import logger

if TYPE_CHECKING:
    from agentrun.tool.tool import Tool as ToolResource
    from agentrun.utils.config import Config


@dataclass
class SkillInfo:
    """Skill 摘要信息 / Skill summary information

    Attributes:
        name: skill 名称 / skill name
        description: skill 描述 / skill description
        version: skill 版本 / skill version
        path: 本地目录路径 / local directory path
    """

    name: str
    description: str = ""
    version: str = ""
    path: str = ""


@dataclass
class SkillDetail(SkillInfo):
    """Skill 详细信息 / Skill detail information

    Attributes:
        instruction: SKILL.md 全文内容 / full content of SKILL.md
        files: 目录下的文件/文件夹列表 / list of files/folders in the directory
    """

    instruction: str = ""
    files: List[str] = field(default_factory=list)


def _parse_frontmatter(content: str) -> Dict[str, str]:
    """解析 SKILL.md 的 YAML frontmatter / Parse YAML frontmatter from SKILL.md

    使用简单的正则解析，避免引入 PyYAML 依赖。
    Uses simple regex parsing to avoid introducing PyYAML dependency.

    Args:
        content: SKILL.md 文件内容 / SKILL.md file content

    Returns:
        解析出的 key-value 字典 / parsed key-value dictionary
    """
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {}
    result: Dict[str, str] = {}
    for line in match.group(1).split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            result[key] = value
    return result


class SkillLoader:
    """Skill 加载器 / Skill Loader

    负责扫描本地 .skills 目录、解析 skill 元信息、读取 skill 指令内容，
    并构造 load_skills 工具供 Agent 运行时调用。

    Responsible for scanning the local .skills directory, parsing skill metadata,
    reading skill instruction content, and constructing the load_skills tool
    for Agent runtime invocation.

    Args:
        skills_dir: 本地 skill 目录路径 / local skill directory path
        remote_skill_names: 需要从远程下载的 skill 名称列表 / list of remote skill names to download
        config: 配置对象 / configuration object
    """

    def __init__(
        self,
        skills_dir: str = ".skills",
        remote_skill_names: Optional[List[str]] = None,
        config: Optional["Config"] = None,
    ):
        self._skills_dir = skills_dir
        self._remote_skill_names = remote_skill_names or []
        self._config = config
        self._skills_cache: Optional[List[SkillInfo]] = None

    def _ensure_skills_available(self) -> None:
        """确保远程 skill 已下载到本地 / Ensure remote skills are downloaded locally

        对每个 remote_skill_name，检查本地是否已存在对应目录，
        不存在则通过 ToolClient 下载。

        For each remote_skill_name, check if the local directory exists,
        download via ToolClient if not.
        """
        if not self._remote_skill_names:
            return

        from agentrun.tool.client import ToolClient

        for skill_name in self._remote_skill_names:
            skill_path = os.path.join(self._skills_dir, skill_name)
            if os.path.isdir(skill_path):
                logger.debug(
                    f"Skill '{skill_name}' already exists at {skill_path}, "
                    "skipping download"
                )
                continue
            logger.info(
                f"Downloading remote skill '{skill_name}' to {self._skills_dir}"
            )
            tool_resource = ToolClient().get(
                name=skill_name, config=self._config
            )
            tool_resource.download_skill(
                target_dir=self._skills_dir, config=self._config
            )

    def _parse_skill_metadata(self, skill_dir: str) -> SkillInfo:
        """解析 skill 元信息 / Parse skill metadata

        按以下优先级获取 skill 的 name 和 description：
        1. SKILL.md 的 YAML frontmatter
        2. package.json
        3. 目录名作为 name，description 为空字符串

        Priority for getting skill name and description:
        1. SKILL.md YAML frontmatter
        2. package.json
        3. Directory name as name, empty string as description

        Args:
            skill_dir: skill 目录的完整路径 / full path to skill directory

        Returns:
            SkillInfo 实例 / SkillInfo instance
        """
        dir_name = os.path.basename(skill_dir)
        name = dir_name
        description = ""
        version = ""

        skill_md_path = os.path.join(skill_dir, "SKILL.md")
        if os.path.isfile(skill_md_path):
            try:
                with open(skill_md_path, "r", encoding="utf-8") as file_handle:
                    content = file_handle.read()
                frontmatter = _parse_frontmatter(content)
                if frontmatter.get("name"):
                    name = frontmatter["name"]
                if frontmatter.get("description"):
                    description = frontmatter["description"]
                if frontmatter.get("version"):
                    version = frontmatter["version"]
                if name != dir_name or description or version:
                    return SkillInfo(
                        name=name,
                        description=description,
                        version=version,
                        path=skill_dir,
                    )
            except (OSError, UnicodeDecodeError) as error:
                logger.warning(
                    f"Failed to read SKILL.md in {skill_dir}: {error}"
                )

        package_json_path = os.path.join(skill_dir, "package.json")
        if os.path.isfile(package_json_path):
            try:
                with open(
                    package_json_path, "r", encoding="utf-8"
                ) as file_handle:
                    package_data = json.load(file_handle)
                if package_data.get("name"):
                    name = package_data["name"]
                if package_data.get("description"):
                    description = package_data["description"]
                if package_data.get("version"):
                    version = package_data["version"]
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
                logger.warning(
                    f"Failed to read package.json in {skill_dir}: {error}"
                )

        return SkillInfo(
            name=name, description=description, version=version, path=skill_dir
        )

    def scan_skills(self) -> List[SkillInfo]:
        """扫描 .skills/ 目录，返回所有 skill 的摘要信息 / Scan .skills/ directory and return all skill summaries

        Returns:
            SkillInfo 列表 / list of SkillInfo
        """
        if self._skills_cache is not None:
            return self._skills_cache

        self._ensure_skills_available()

        if not os.path.isdir(self._skills_dir):
            self._skills_cache = []
            return self._skills_cache

        skills: List[SkillInfo] = []
        try:
            entries = sorted(os.listdir(self._skills_dir))
        except OSError as error:
            logger.warning(
                f"Failed to list skills directory {self._skills_dir}: {error}"
            )
            self._skills_cache = []
            return self._skills_cache

        for entry in entries:
            entry_path = os.path.join(self._skills_dir, entry)
            if os.path.isdir(entry_path) and not entry.startswith("."):
                skill_info = self._parse_skill_metadata(entry_path)
                skills.append(skill_info)

        self._skills_cache = skills
        return self._skills_cache

    def load_skill(self, name: str) -> Optional[SkillDetail]:
        """加载指定 skill 的详细信息 / Load detailed information for a specific skill

        Args:
            name: skill 名称 / skill name

        Returns:
            SkillDetail 实例，如果 skill 不存在则返回 None /
            SkillDetail instance, or None if skill does not exist
        """
        skills = self.scan_skills()
        target_skill: Optional[SkillInfo] = None
        for skill in skills:
            if skill.name == name:
                target_skill = skill
                break

        if target_skill is None:
            return None

        instruction = ""
        skill_md_path = os.path.join(target_skill.path, "SKILL.md")
        if os.path.isfile(skill_md_path):
            try:
                with open(skill_md_path, "r", encoding="utf-8") as file_handle:
                    instruction = file_handle.read()
            except (OSError, UnicodeDecodeError) as error:
                logger.warning(
                    f"Failed to read SKILL.md for skill '{name}': {error}"
                )

        files: List[str] = []
        try:
            for entry in sorted(os.listdir(target_skill.path)):
                if not entry.startswith("."):
                    entry_path = os.path.join(target_skill.path, entry)
                    if os.path.isdir(entry_path):
                        files.append(entry + "/")
                    else:
                        files.append(entry)
        except OSError as error:
            logger.warning(f"Failed to list files for skill '{name}': {error}")

        return SkillDetail(
            name=target_skill.name,
            description=target_skill.description,
            version=target_skill.version,
            path=target_skill.path,
            instruction=instruction,
            files=files,
        )

    def _build_tool_description(self, skills: List[SkillInfo]) -> str:
        """构建 load_skills 工具的 description / Build the description for the load_skills tool

        将所有可用 skill 的名称和描述写入工具描述中。
        Writes all available skill names and descriptions into the tool description.

        Args:
            skills: skill 摘要列表 / list of skill summaries

        Returns:
            工具描述字符串 / tool description string
        """
        if not skills:
            return (
                "Load skill instructions for the agent. "
                "No skills available in the configured directory."
            )

        skill_lines = []
        for skill in skills:
            desc_part = f": {skill.description}" if skill.description else ""
            skill_lines.append(f"- {skill.name}{desc_part}")

        skills_list = "\n".join(skill_lines)
        return (
            "Load skill instructions for the agent. "
            "Call without arguments to list all skills, "
            "or with a skill name to get detailed instructions.\n\n"
            f"Available skills:\n{skills_list}"
        )

    def _load_skills_func(self, name: Optional[str] = None) -> str:
        """load_skills 工具的执行函数 / Execution function for the load_skills tool

        Args:
            name: skill 名称（可选）/ skill name (optional)

        Returns:
            JSON 字符串 / JSON string
        """
        if name is None or name == "":
            skills = self.scan_skills()
            result: Dict[str, Any] = {
                "skills": [
                    {"name": skill.name, "description": skill.description}
                    for skill in skills
                ]
            }
            return json.dumps(result, ensure_ascii=False)

        detail = self.load_skill(name)
        if detail is None:
            available = [skill.name for skill in self.scan_skills()]
            available_str = ", ".join(available) if available else "none"
            error_result: Dict[str, str] = {
                "error": (
                    f"Skill '{name}' not found. "
                    f"Available skills: {available_str}"
                )
            }
            return json.dumps(error_result, ensure_ascii=False)

        detail_result: Dict[str, Any] = {
            "name": detail.name,
            "description": detail.description,
            "instruction": detail.instruction,
            "files": detail.files,
        }
        return json.dumps(detail_result, ensure_ascii=False)

    def to_common_toolset(self) -> CommonToolSet:
        """构造包含 load_skills 工具的 CommonToolSet / Construct CommonToolSet with load_skills tool

        Returns:
            CommonToolSet 实例 / CommonToolSet instance
        """
        skills = self.scan_skills()
        description = self._build_tool_description(skills)

        load_skills_tool = Tool(
            name="load_skills",
            description=description,
            parameters=[
                ToolParameter(
                    name="name",
                    param_type="string",
                    description=(
                        "The name of the skill to load. "
                        "If omitted, returns a list of all available skills."
                    ),
                    required=False,
                ),
            ],
            func=self._load_skills_func,
        )

        return CommonToolSet(tools_list=[load_skills_tool])


def skill_tools(
    name: Optional[Union[str, List[str], "ToolResource"]] = None,
    *,
    skills_dir: str = ".skills",
    config: Optional["Config"] = None,
) -> CommonToolSet:
    """将 Skill 封装为通用工具集 / Wrap Skills as CommonToolSet

    支持从工具名称、名称列表或 ToolResource 实例创建通用工具集。
    Supports creating CommonToolSet from tool name, name list, or ToolResource instance.

    Args:
        name: 远程 skill 名称、名称列表或 ToolResource 实例（可选）/
              Remote skill name, name list, or ToolResource instance (optional).
              如果提供，会先下载到 skills_dir 再加载 /
              If provided, downloads to skills_dir before loading.
              如果不提供，仅从 skills_dir 加载本地已有的 skill /
              If not provided, only loads local skills from skills_dir.
        skills_dir: 本地 skill 目录，默认 ".skills" / Local skill directory, default ".skills"
        config: 配置对象 / Configuration object

    Returns:
        CommonToolSet: 包含 load_skills 工具的通用工具集 /
                       CommonToolSet containing the load_skills tool

    Examples:
        >>> # 仅加载本地 skill / Load local skills only
        >>> ts = skill_tools(skills_dir=".skills")
        >>>
        >>> # 下载远程 skill 后加载 / Download remote skill then load
        >>> ts = skill_tools("my-remote-skill")
        >>>
        >>> # 下载多个远程 skill / Download multiple remote skills
        >>> ts = skill_tools(["skill-a", "skill-b"])
        >>>
        >>> # 转换为 LangChain 工具 / Convert to LangChain tools
        >>> lc_tools = ts.to_langchain()
    """
    remote_names: List[str] = []

    if name is not None:
        if isinstance(name, str):
            remote_names = [name]
        elif isinstance(name, list):
            remote_names = name
        else:
            # ToolResource instance — extract its name and download
            tool_resource_instance = name
            resource_name = getattr(
                tool_resource_instance, "name", None
            ) or getattr(tool_resource_instance, "tool_name", None)
            if resource_name:
                skill_path = os.path.join(skills_dir, resource_name)
                if not os.path.isdir(skill_path):
                    tool_resource_instance.download_skill(
                        target_dir=skills_dir, config=config
                    )

    loader = SkillLoader(
        skills_dir=skills_dir,
        remote_skill_names=remote_names,
        config=config,
    )
    return loader.to_common_toolset()
