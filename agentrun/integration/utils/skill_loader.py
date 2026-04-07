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
import subprocess
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

from agentrun.integration.utils.tool import CommonToolSet, Tool, ToolParameter
from agentrun.utils.log import logger

if TYPE_CHECKING:
    from agentrun.tool.tool import Tool as ToolResource
    from agentrun.utils.config import Config

# Maximum output size for execute_command (bytes)
# execute_command 输出大小限制（字节）
MAX_OUTPUT_SIZE = 102400  # 100KB


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
        command_approval: Optional[Callable[[str, str], bool]] = None,
        command_timeout: int = 300,
    ):
        self._skills_dir = skills_dir
        self._remote_skill_names = remote_skill_names or []
        self._config = config
        self._skills_cache: Optional[List[SkillInfo]] = None
        self._command_approval = command_approval
        self._command_timeout = command_timeout

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

    def _read_skill_file_func(self, name: str, relative_path: str) -> str:
        """read_skill_file 工具的执行函数 / Execution function for the read_skill_file tool

        读取指定 skill 目录内的文件内容，带路径穿越保护。
        Reads file content within a specific skill directory with path traversal protection.

        Args:
            name: skill 名称 / skill name
            relative_path: skill 目录内的相对路径 / relative path within skill directory

        Returns:
            JSON 字符串 / JSON string
        """
        skills = self.scan_skills()
        target_skill: Optional[SkillInfo] = None
        for skill in skills:
            if skill.name == name:
                target_skill = skill
                break

        if target_skill is None:
            available = [s.name for s in skills]
            available_str = ", ".join(available) if available else "none"
            return json.dumps(
                {
                    "error": (
                        f"Skill '{name}' not found. "
                        f"Available skills: {available_str}"
                    )
                },
                ensure_ascii=False,
            )

        # Path traversal protection / 路径穿越保护
        skill_real_dir = os.path.realpath(target_skill.path)
        target_path = os.path.realpath(
            os.path.join(target_skill.path, relative_path)
        )
        if (
            not target_path.startswith(skill_real_dir + os.sep)
            and target_path != skill_real_dir
        ):
            return json.dumps(
                {
                    "error": (
                        f"Path '{relative_path}' is outside the skill"
                        " directory. Access denied."
                    )
                },
                ensure_ascii=False,
            )

        if not os.path.exists(target_path):
            return json.dumps(
                {
                    "error": (
                        f"File '{relative_path}' not found in skill '{name}'."
                    )
                },
                ensure_ascii=False,
            )

        # Directory listing / 目录列表
        if os.path.isdir(target_path):
            try:
                entries: List[str] = []
                for entry in sorted(os.listdir(target_path)):
                    if not entry.startswith("."):
                        entry_full = os.path.join(target_path, entry)
                        if os.path.isdir(entry_full):
                            entries.append(entry + "/")
                        else:
                            entries.append(entry)
                return json.dumps({"files": entries}, ensure_ascii=False)
            except OSError as error:
                return json.dumps(
                    {"error": f"Failed to list directory: {error}"},
                    ensure_ascii=False,
                )

        # File reading / 文件读取
        try:
            with open(target_path, "r", encoding="utf-8") as file_handle:
                content = file_handle.read()
            return json.dumps({"content": content}, ensure_ascii=False)
        except UnicodeDecodeError:
            return json.dumps(
                {
                    "error": (
                        f"File '{relative_path}' cannot be read as text. "
                        "It may be a binary file."
                    )
                },
                ensure_ascii=False,
            )
        except OSError as error:
            return json.dumps(
                {"error": f"Failed to read file: {error}"},
                ensure_ascii=False,
            )

    def _truncate_output(self, output: str) -> str:
        """截断过大的输出 / Truncate oversized output

        Args:
            output: 原始输出 / original output

        Returns:
            截断后的输出 / truncated output
        """
        if len(output.encode("utf-8", errors="replace")) <= MAX_OUTPUT_SIZE:
            return output
        # Truncate by bytes then decode safely
        truncated = output.encode("utf-8", errors="replace")[:MAX_OUTPUT_SIZE]
        return truncated.decode("utf-8", errors="replace") + (
            f"\n... [output truncated, exceeded {MAX_OUTPUT_SIZE} bytes]"
        )

    def _execute_command_func(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """execute_command 工具的执行函数 / Execution function for the execute_command tool

        在本地机器上执行 shell 命令。
        Executes a shell command on the local machine.

        Args:
            command: 要执行的命令 / command to execute
            cwd: 工作目录（可选，默认 skills_dir）/ working directory (optional, defaults to skills_dir)
            timeout: 超时秒数（可选，默认使用 command_timeout）/ timeout in seconds (optional)

        Returns:
            JSON 字符串 / JSON string
        """
        resolved_cwd = cwd if cwd else self._skills_dir
        resolved_timeout = (
            timeout if timeout is not None else self._command_timeout
        )

        # Validate cwd exists / 验证工作目录存在
        if not os.path.isdir(resolved_cwd):
            return json.dumps(
                {
                    "error": (
                        f"Working directory '{resolved_cwd}' does not exist."
                    )
                },
                ensure_ascii=False,
            )

        # Command approval callback / 命令确认回调
        if self._command_approval is not None:
            try:
                approved = self._command_approval(command, resolved_cwd)
            except Exception as approval_error:
                logger.warning(
                    "Command approval callback raised an error:"
                    f" {approval_error}"
                )
                return json.dumps(
                    {
                        "error": (
                            "Command approval callback failed: "
                            f"{approval_error}"
                        )
                    },
                    ensure_ascii=False,
                )
            if not approved:
                return json.dumps(
                    {"error": "Command execution rejected by user."},
                    ensure_ascii=False,
                )

        logger.info(
            f"Executing command: {command!r} in cwd={resolved_cwd!r} "
            f"timeout={resolved_timeout}s"
        )

        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=resolved_cwd,
                timeout=resolved_timeout,
            )
            stdout = self._truncate_output(completed.stdout)
            stderr = self._truncate_output(completed.stderr)

            logger.info(f"Command finished: exit_code={completed.returncode}")

            return json.dumps(
                {
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": completed.returncode,
                    "timed_out": False,
                },
                ensure_ascii=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Command timed out after {resolved_timeout}s: {command!r}"
            )
            return json.dumps(
                {
                    "stdout": "",
                    "stderr": (
                        f"Command timed out after {resolved_timeout} seconds."
                    ),
                    "exit_code": -1,
                    "timed_out": True,
                },
                ensure_ascii=False,
            )
        except OSError as error:
            logger.error(f"Failed to execute command: {error}")
            return json.dumps(
                {"error": f"Failed to execute command: {error}"},
                ensure_ascii=False,
            )

    @staticmethod
    def _is_execute_command_allowed() -> bool:
        """检查环境变量是否允许加载 execute_command 工具

        Check whether the ALLOW_EXECUTE_COMMAND environment variable permits
        loading the execute_command tool.

        The variable is read from ``os.environ``.  When it is absent or set to
        any value other than a case-insensitive ``"false"``, the tool is
        allowed (default **True**).

        Returns:
            True 表示允许 / True means allowed
        """
        value = os.environ.get("ALLOW_EXECUTE_COMMAND", "true")
        return value.lower() != "false"

    def to_common_toolset(self) -> CommonToolSet:
        """构造包含 load_skills、read_skill_file 以及可选的 execute_command 工具的 CommonToolSet

        Construct CommonToolSet with load_skills, read_skill_file, and
        optionally execute_command tools.

        The execute_command tool is included only when the environment variable
        ``ALLOW_EXECUTE_COMMAND`` is not set to ``"false"`` (case-insensitive).
        When the variable is absent, it defaults to ``"true"`` (included).

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

        read_skill_file_tool = Tool(
            name="read_skill_file",
            description=(
                "Read a file from a skill's directory. "
                "Returns the file content as text, or lists directory contents "
                "if the path points to a directory. "
                "Only files within the skill directory can be accessed."
            ),
            parameters=[
                ToolParameter(
                    name="name",
                    param_type="string",
                    description="The name of the skill containing the file.",
                    required=True,
                ),
                ToolParameter(
                    name="relative_path",
                    param_type="string",
                    description=(
                        "Relative path to the file within the skill directory "
                        "(e.g., 'scripts/run.sh', 'requirements.txt')."
                    ),
                    required=True,
                ),
            ],
            func=self._read_skill_file_func,
        )

        tools_list: List[Tool] = [load_skills_tool, read_skill_file_tool]

        if self._is_execute_command_allowed():
            execute_command_tool = Tool(
                name="execute_command",
                description=(
                    "Execute a shell command on the local machine. Use this to"
                    " run scripts, install dependencies, or perform file"
                    " operations as instructed by skill documentation. Returns"
                    " stdout, stderr, exit_code, and timeout status.\n\n⚠️"
                    " IMPORTANT: Before calling this tool, you MUST first"
                    " display the exact command to the user and ask for"
                    " explicit confirmation. Only proceed if the user approves."
                    " Never execute commands without user approval."
                ),
                parameters=[
                    ToolParameter(
                        name="command",
                        param_type="string",
                        description="The shell command to execute.",
                        required=True,
                    ),
                    ToolParameter(
                        name="cwd",
                        param_type="string",
                        description=(
                            "Working directory for the command. "
                            "Defaults to the skills directory if not specified."
                        ),
                        required=False,
                    ),
                    ToolParameter(
                        name="timeout",
                        param_type="integer",
                        description=(
                            "Maximum execution time in seconds. "
                            f"Defaults to {self._command_timeout}."
                        ),
                        required=False,
                    ),
                ],
                func=self._execute_command_func,
            )
            tools_list.append(execute_command_tool)

        return CommonToolSet(tools_list=tools_list)


def skill_tools(
    name: Optional[Union[str, List[str], "ToolResource"]] = None,
    *,
    skills_dir: str = ".skills",
    config: Optional["Config"] = None,
    command_approval: Optional[Callable[[str, str], bool]] = None,
    command_timeout: int = 300,
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
        command_approval: 命令执行前的确认回调函数（可选）/
                          Optional approval callback invoked before executing commands.
                          接收 (command, cwd) 参数，返回 True 允许执行，False 拒绝 /
                          Receives (command, cwd), returns True to allow, False to reject.
        command_timeout: execute_command 的默认超时秒数，默认 30 /
                         Default timeout in seconds for execute_command, default 30.

    Returns:
        CommonToolSet: 包含 load_skills、read_skill_file、execute_command 工具的通用工具集 /
                       CommonToolSet containing load_skills, read_skill_file, and execute_command tools

    Examples:
        >>> # 仅加载本地 skill / Load local skills only
        >>> ts = skill_tools(skills_dir=".skills")
        >>>
        >>> # 下载远程 skill 后加载 / Download remote skill then load
        >>> ts = skill_tools("my-remote-skill")
        >>>
        >>> # 带命令确认回调 / With command approval callback
        >>> ts = skill_tools(
        ...     skills_dir=".skills",
        ...     command_approval=lambda cmd, cwd: input(f"Execute '{cmd}'? [y/N]: ").lower() == "y",
        ... )
        >>>
        >>> # 自定义超时 / Custom timeout
        >>> ts = skill_tools(skills_dir=".skills", command_timeout=120)
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
        command_approval=command_approval,
        command_timeout=command_timeout,
    )
    return loader.to_common_toolset()
