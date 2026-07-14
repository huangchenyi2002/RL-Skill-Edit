# src/skill_library.py
"""
Skill Library 数据结构
包含：
  - 受保护区域（APPENDIX + SLOW_UPDATE）
  - 精细化编辑操作（append/replace/delete）
  - 去重机制
  - Z_named 提取接口（供 label_space.py 使用）
"""

import os
import re
import copy
import json
import yaml
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────
# 受保护区域标记
# ─────────────────────────────────────────────────

SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END   = "<!-- SLOW_UPDATE_END -->"
APPENDIX_START    = "<!-- APPENDIX_START -->"
APPENDIX_END      = "<!-- APPENDIX_END -->"
APPENDIX_HEADING  = "## Execution Notes Appendix"

_PROTECTED_REGIONS = (
    (SLOW_UPDATE_START, SLOW_UPDATE_END),
    (APPENDIX_START, APPENDIX_END),
)


# ─────────────────────────────────────────────────
# Appendix 操作
# ─────────────────────────────────────────────────

def _canonicalize(text: str) -> str:
    normalized = re.sub(
        r"\s+", " ", str(text or "").strip())
    normalized = normalized.rstrip(" .;:,_-")
    return normalized.casefold()


def _dedupe_preserve_order(
    notes: list[str],
) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for note in notes:
        text = re.sub(
            r"\s+", " ", str(note).strip())
        if not text:
            continue
        key = _canonicalize(text)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def has_appendix_field(skill: str) -> bool:
    return (APPENDIX_START in skill
            and APPENDIX_END in skill)


def extract_appendix_notes(skill: str) -> list[str]:
    start = skill.find(APPENDIX_START)
    end   = skill.find(APPENDIX_END)
    if start == -1 or end == -1:
        return []
    inner = skill[
        start + len(APPENDIX_START):end].strip()
    notes = []
    for line in inner.splitlines():
        line = line.strip()
        if not line or line == APPENDIX_HEADING:
            continue
        if (line.lstrip("#").strip()
                == APPENDIX_HEADING.lstrip("#").strip()):
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        elif line.startswith(("*", "-")):
            line = line[1:].strip()
        if line:
            notes.append(line)
    return notes


def _render_appendix_block(
    notes: list[str],
) -> str:
    lines = [APPENDIX_START, APPENDIX_HEADING]
    for note in notes:
        lines.append(f"- {note}")
    lines.append(APPENDIX_END)
    return "\n".join(lines)


def _strip_appendix_fields(skill: str) -> str:
    while True:
        start = skill.find(APPENDIX_START)
        if start == -1:
            break
        end = skill.find(APPENDIX_END, start)
        if end == -1:
            skill = skill[:start]
            break
        skill = (skill[:start]
                 + skill[end + len(APPENDIX_END):])
    skill = skill.replace(APPENDIX_END, "")
    while "\n\n\n" in skill:
        skill = skill.replace("\n\n\n", "\n\n")
    return skill.rstrip()


def inject_empty_appendix_field(
    skill: str,
) -> str:
    if has_appendix_field(skill):
        return skill
    block = (f"\n\n{APPENDIX_START}\n"
             f"{APPENDIX_HEADING}\n"
             f"{APPENDIX_END}\n")
    return skill.rstrip() + block


def append_to_appendix_field(
    skill: str, new_notes: list[str],
) -> str:
    incoming = _dedupe_preserve_order(
        new_notes or [])
    existing = extract_appendix_notes(skill)
    merged = _dedupe_preserve_order(
        existing + incoming)
    base = _strip_appendix_fields(skill)
    block = _render_appendix_block(merged)
    return f"{base}\n\n{block}\n"


# ─────────────────────────────────────────────────
# Slow Update 操作
# ─────────────────────────────────────────────────

def has_slow_update_field(skill: str) -> bool:
    return (SLOW_UPDATE_START in skill
            and SLOW_UPDATE_END in skill)


def extract_slow_update_field(
    skill: str,
) -> str:
    start = skill.find(SLOW_UPDATE_START)
    end   = skill.find(SLOW_UPDATE_END)
    if start == -1 or end == -1:
        return ""
    return skill[
        start + len(SLOW_UPDATE_START):end].strip()


def _strip_slow_update_fields(
    skill: str,
) -> str:
    while True:
        start = skill.find(SLOW_UPDATE_START)
        if start == -1:
            break
        end = skill.find(
            SLOW_UPDATE_END, start)
        if end == -1:
            skill = skill[:start]
            break
        skill = (skill[:start]
                 + skill[end + len(SLOW_UPDATE_END):])
    skill = skill.replace(SLOW_UPDATE_END, "")
    while "\n\n\n" in skill:
        skill = skill.replace("\n\n\n", "\n\n")
    return skill.rstrip()


def replace_slow_update_field(
    skill: str, new_content: str,
) -> str:
    skill = _strip_slow_update_fields(skill)
    block = (f"\n\n{SLOW_UPDATE_START}\n"
             f"{new_content.strip()}\n"
             f"{SLOW_UPDATE_END}\n")
    return skill + block


def inject_empty_slow_update_field(
    skill: str,
) -> str:
    if has_slow_update_field(skill):
        return skill
    block = (f"\n\n{SLOW_UPDATE_START}\n"
             f"{SLOW_UPDATE_END}\n")
    return skill.rstrip() + block


# ─────────────────────────────────────────────────
# 精细化编辑操作
# ─────────────────────────────────────────────────

def resolve_parent_scope(
    skill: str, parent_location: str,
) -> tuple[int, int, str] | None:
    """Resolve a frozen parent(z) to one Markdown section, fail closed."""
    parent = (parent_location or "").strip()
    if not parent:
        return None
    headings = list(re.finditer(
        r"^(#{1,6})\s+(.+?)\s*$", skill, re.MULTILINE))
    wanted = [part.strip().casefold() for part in
              re.split(r"\s*>\s*|\s*/\s*", parent) if part.strip()]
    # Prefer the most specific (right-most) component in "A > B > C".
    for wanted_part in reversed(wanted):
        matches = []
        for index, heading in enumerate(headings):
            title = heading.group(2).strip().casefold()
            if title != wanted_part and wanted_part not in title:
                continue
            level = len(heading.group(1))
            end = len(skill)
            for following in headings[index + 1:]:
                if len(following.group(1)) <= level:
                    end = following.start()
                    break
            matches.append((heading.start(), end,
                            skill[heading.start():end]))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
    return None


def apply_scoped_parent_edit(
    skill: str, parent_location: str,
    old_text: str, new_text: str,
) -> tuple[str, dict]:
    """Apply one exact replacement wholly inside parent(z)."""
    scope = resolve_parent_scope(skill, parent_location)
    report = {"parent_location": parent_location, "status": "unknown"}
    if scope is None:
        report["status"] = "skipped_parent_unresolved"
        return skill, report
    start, end, scoped_text = scope
    if _is_in_protected_region(skill, old_text) or any(
            marker in new_text for region in _PROTECTED_REGIONS
            for marker in region):
        report["status"] = "skipped_protected_region"
        return skill, report
    if not old_text or old_text == new_text:
        report["status"] = "skipped_empty_or_noop"
        return skill, report
    if scoped_text.count(old_text) != 1 or skill.count(old_text) != 1:
        report["status"] = "skipped_target_not_unique_in_parent"
        return skill, report
    local_at = scoped_text.index(old_text)
    absolute_at = start + local_at
    if absolute_at + len(old_text) > end:
        report["status"] = "skipped_target_outside_parent"
        return skill, report
    updated = (skill[:absolute_at] + new_text
               + skill[absolute_at + len(old_text):])
    report["status"] = "applied_scoped_replace"
    return updated, report

def _earliest_protected_start(
    skill: str,
) -> int:
    positions = [
        idx for idx in (
            skill.find(start)
            for start, _ in _PROTECTED_REGIONS
        ) if idx != -1
    ]
    return min(positions) if positions else -1


def _is_in_protected_region(
    skill: str, target: str,
) -> bool:
    if not target:
        return False
    target_idx = skill.find(target)
    if target_idx == -1:
        return False
    for start_marker, end_marker in _PROTECTED_REGIONS:
        s = skill.find(start_marker)
        e = skill.find(end_marker)
        if s == -1 or e == -1:
            continue
        if s <= target_idx < e + len(end_marker):
            return True
    return False


def apply_text_edit(
    skill: str, edit: dict,
) -> tuple[str, dict]:
    op      = edit.get("op", "append")
    content = edit.get("content", "").strip()
    target  = edit.get("target", "")

    for s, e in _PROTECTED_REGIONS:
        content = content.replace(s, "").replace(e, "")

    report = {
        "op": op,
        "target": target[:100],
        "content_preview": content[:100],
        "status": "unknown",
    }

    if target and _is_in_protected_region(
        skill, target
    ):
        report["status"] = "skipped_protected_region"
        return skill, report

    if op == "append":
        prot = _earliest_protected_start(skill)
        if prot != -1:
            before = skill[:prot].rstrip()
            after  = skill[prot:]
            report["status"] = \
                "applied_append_before_protected"
            return (before + "\n\n" + content
                    + "\n\n" + after), report
        report["status"] = "applied_append"
        return (skill.rstrip() + "\n\n"
                + content + "\n"), report

    if op == "insert_after":
        if not target or target not in skill:
            prot = _earliest_protected_start(skill)
            if prot != -1:
                before = skill[:prot].rstrip()
                after  = skill[prot:]
                report["status"] = \
                    "applied_insert_fallback"
                return (before + "\n\n" + content
                        + "\n\n" + after), report
            report["status"] = \
                "applied_insert_fallback_append"
            return (skill.rstrip() + "\n\n"
                    + content + "\n"), report
        idx = skill.index(target) + len(target)
        nl  = skill.find("\n", idx)
        at  = nl + 1 if nl != -1 else len(skill)
        report["status"] = "applied_insert_after"
        return (skill[:at] + "\n" + content
                + "\n" + skill[at:]), report

    if op == "replace":
        if not target:
            report["status"] = \
                "skipped_replace_no_target"
            return skill, report
        if target not in skill:
            report["status"] = \
                "skipped_replace_not_found"
            return skill, report
        report["status"] = "applied_replace"
        return skill.replace(
            target, content, 1), report

    if op == "delete":
        if not target:
            report["status"] = \
                "skipped_delete_no_target"
            return skill, report
        if target not in skill:
            report["status"] = \
                "skipped_delete_not_found"
            return skill, report
        report["status"] = "applied_delete"
        return skill.replace(
            target, "", 1), report

    report["status"] = "skipped_unknown_op"
    return skill, report

# ─────────────────────────────────────────────────
# Skill Core
# ─────────────────────────────────────────────────

@dataclass
class SkillCore:
    skill_id:          str
    name:              str  = ""
    description:       str  = ""
    name_history:      list = field(
        default_factory=list)
    slow_update_field: str  = ""
    license:           Optional[str]  = None
    compatibility:     Optional[dict] = None
    allowed_tools:     Optional[list] = None
    allow_implicit:    bool           = True
    execution_body:    str  = ""
    references:        dict = field(
        default_factory=dict)
    scripts:           dict = field(
        default_factory=dict)
    raw_skill_md:      str  = ""

    def rename(self, new_name: str):
        if self.name and self.name != new_name:
            if self.name not in self.name_history:
                self.name_history.append(self.name)
        self.name = new_name

    def all_names(self) -> list[str]:
        names = [self.name]
        for n in self.name_history:
            if n not in names:
                names.append(n)
        return names

    # ── Appendix 操作 ─────────────────────────

    def get_appendix_notes(self) -> list[str]:
        return extract_appendix_notes(
            self.execution_body)

    def add_appendix_notes(self, notes: list[str]):
        self.execution_body = \
            append_to_appendix_field(
                self.execution_body, notes)

    def ensure_appendix(self):
        self.execution_body = \
            inject_empty_appendix_field(
                self.execution_body)

    # ── Slow Update 操作 ─────────────────────

    def get_slow_update_content(self) -> str:
        return extract_slow_update_field(
            self.execution_body)

    def set_slow_update_content(
        self, content: str,
    ):
        self.execution_body = \
            replace_slow_update_field(
                self.execution_body, content)

    def ensure_slow_update(self):
        self.execution_body = \
            inject_empty_slow_update_field(
                self.execution_body)

    # ── Z_named 提取（供 label_space.py）─────

    def extract_rule_names(self) -> list[str]:
        """
        从 execution_body 提取规则名称
        作为 Z_named 的候选

        识别格式：
          ### Rule 1: Find columns by header
          1. Load workbook
          - Always check None
        """
        rules = []
        body = self.execution_body or ""

        for m in re.finditer(
            r"###?\s*Rule\s*\d+[:\s]*(.+)",
            body, re.MULTILINE
        ):
            rules.append(m.group(1).strip())

        for m in re.finditer(
            r"^\s*\d+\.\s+(.+)",
            body, re.MULTILINE
        ):
            text = m.group(1).strip()
            if len(text) > 10:
                rules.append(text)

        for m in re.finditer(
            r"^\s*[-*]\s+(?:Always|Never|Must"
            r"|Do not|Check|Use|Handle|Verify"
            r"|Ensure)\s+(.+)",
            body, re.MULTILINE | re.IGNORECASE
        ):
            rules.append(m.group(1).strip())

        return rules

    def extract_code_patterns(self) -> list[str]:
        """
        从代码块中提取关键模式
        作为 labeler 的匹配依据
        """
        patterns = []
        body = self.execution_body or ""

        for m in re.finditer(
            r"```python\s*\n(.*?)```",
            body, re.DOTALL
        ):
            code = m.group(1)
            for fn in re.finditer(
                r"def\s+(\w+)\(", code
            ):
                patterns.append(fn.group(1))
            for var in re.finditer(
                r"(\w+)\s*=\s*\{", code
            ):
                patterns.append(var.group(1))

        return patterns

    def get_module_info(self) -> dict:
        """
        module 级别元信息
        供 label_space.py 使用
        """
        return {
            "skill_id":      self.skill_id,
            "name":          self.name,
            "description":   self.description[:200],
            "rules":         self.extract_rule_names(),
            "code_patterns": self.extract_code_patterns(),
            "has_appendix":  bool(
                self.get_appendix_notes()),
            "has_slow_update": bool(
                self.get_slow_update_content()),
        }

    # ── 序列化 ────────────────────────────────

    def to_skill_md(self) -> str:
        fm = {
            "name":        self.name,
            "description": self.description,
        }
        if self.license:
            fm["license"] = self.license
        if self.compatibility:
            fm["compatibility"] = self.compatibility
        if self.allowed_tools:
            fm["allowed-tools"] = self.allowed_tools
        if not self.allow_implicit:
            fm["allow-implicit-invocation"] = False
        yaml_str = yaml.dump(
            fm, default_flow_style=False,
            allow_unicode=True)
        return (f"---\n{yaml_str}---\n\n"
                f"{self.execution_body}")

    def to_dict(self) -> dict:
        return {
            "skill_id":          self.skill_id,
            "name":              self.name,
            "description":       self.description,
            "name_history":      self.name_history,
            "slow_update_field": self.slow_update_field,
            "execution_body":    self.execution_body,
        }

    @classmethod
    def from_directory(
        cls, skill_dir: str, skill_id: str,
    ) -> "SkillCore":
        skill_md_path = os.path.join(
            skill_dir, "SKILL.md")
        if not os.path.exists(skill_md_path):
            raise FileNotFoundError(
                f"SKILL.md not found: {skill_dir}")
        with open(
            skill_md_path, "r", encoding="utf-8"
        ) as f:
            raw = f.read()

        fm   = {}
        body = raw.strip()
        if raw.strip().startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm   = yaml.safe_load(
                        parts[1]) or {}
                    body = parts[2].strip()
                except yaml.YAMLError:
                    fm   = {}
                    body = raw.strip()

        dir_name    = os.path.basename(skill_dir)
        name        = fm.get("name", "") or dir_name
        description = fm.get("description", "")
        if not description and body:
            for line in body.split("\n"):
                line = line.strip().lstrip(
                    "#").strip()
                if line and not line.startswith("```"):
                    description = line[:300]
                    break

        name_history      = []
        slow_update_field = ""
        meta_path = os.path.join(
            skill_dir, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(
                    meta_path, "r", encoding="utf-8"
                ) as f:
                    meta = json.load(f)
                name_history = meta.get(
                    "name_history", [])
                slow_update_field = meta.get(
                    "slow_update_field", "")
            except Exception:
                pass

        references = {}
        refs_dir = os.path.join(
            skill_dir, "references")
        if os.path.isdir(refs_dir):
            for fname in sorted(
                os.listdir(refs_dir)
            ):
                fpath = os.path.join(
                    refs_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        with open(
                            fpath, "r",
                            encoding="utf-8"
                        ) as f:
                            references[fname] = \
                                f.read()
                    except Exception:
                        pass

        scripts = {}
        scripts_dir = os.path.join(
            skill_dir, "scripts")
        if os.path.isdir(scripts_dir):
            for fname in sorted(
                os.listdir(scripts_dir)
            ):
                fpath = os.path.join(
                    scripts_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        with open(
                            fpath, "r",
                            encoding="utf-8"
                        ) as f:
                            scripts[fname] = \
                                f.read()
                    except Exception:
                        pass

        return cls(
            skill_id          = skill_id,
            name              = name,
            description       = description,
            name_history      = name_history,
            slow_update_field = slow_update_field,
            license           = fm.get("license"),
            compatibility     = fm.get(
                "compatibility"),
            allowed_tools     = fm.get(
                "allowed-tools"),
            allow_implicit    = fm.get(
                "allow-implicit-invocation", True),
            execution_body    = body,
            references        = references,
            scripts           = scripts,
            raw_skill_md      = raw,
        )

    def clone(self) -> "SkillCore":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        hist = (f", history={self.name_history}"
                if self.name_history else "")
        return (f"SkillCore(id={self.skill_id!r}, "
                f"name={self.name!r}{hist})")


# ─────────────────────────────────────────────────
# Skill Library
# ─────────────────────────────────────────────────

class SkillLibrary:

    def __init__(self, skills: list = None):
        self._skills: dict = {}
        for sk in (skills or []):
            self._skills[sk.skill_id] = sk

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self):
        return iter(self._skills.values())

    def __repr__(self) -> str:
        names = [sk.name
                 for sk in self._skills.values()]
        return (f"SkillLibrary("
                f"{len(self)} skills: {names})")

    def get(self, skill_id: str):
        return self._skills.get(skill_id)

    def add(self, skill: SkillCore):
        self._skills[skill.skill_id] = skill

    def remove(self, skill_id: str):
        self._skills.pop(skill_id, None)

    def skill_ids(self) -> list:
        return list(self._skills.keys())

    def clone(self) -> "SkillLibrary":
        return SkillLibrary([
            sk.clone()
            for sk in self._skills.values()])

    def startup_catalog(self) -> str:
        if not self._skills:
            return "（暂无可用 Skill）"
        lines = ["# 可用 Skills\n"]
        for sk in self._skills.values():
            lines.append(f"## {sk.name}")
            if sk.description:
                lines.append(f"{sk.description}\n")
        return "\n".join(lines)

    # ── 批量 appendix/slow_update ─────────────

    def ensure_all_appendix(self):
        for sk in self._skills.values():
            sk.ensure_appendix()

    def ensure_all_slow_update(self):
        for sk in self._skills.values():
            sk.ensure_slow_update()

    # ── Z_named 提取接口 ─────────────────────

    def get_all_module_info(self) -> list[dict]:
        return [
            sk.get_module_info()
            for sk in self._skills.values()
        ]

    def get_all_rule_names(
        self,
    ) -> dict[str, list[str]]:
        """
        {skill_id: [rule_names]}
        Z_named 的原始来源
        """
        return {
            sk.skill_id: sk.extract_rule_names()
            for sk in self._skills.values()
        }

    def get_skill_by_module_id(
        self, module_id: str,
    ):
        """module_id = skill_id"""
        return self.get(module_id)

    # ── 序列化 ────────────────────────────────

    def save(self, library_dir: str):
        os.makedirs(library_dir, exist_ok=True)
        for skill_id, sk in self._skills.items():
            skill_dir = os.path.join(
                library_dir, skill_id)
            os.makedirs(skill_dir, exist_ok=True)

            with open(
                os.path.join(
                    skill_dir, "SKILL.md"),
                "w", encoding="utf-8"
            ) as f:
                f.write(sk.to_skill_md())

            if (sk.name_history
                    or sk.slow_update_field):
                with open(
                    os.path.join(
                        skill_dir, "meta.json"),
                    "w", encoding="utf-8"
                ) as f:
                    json.dump({
                        "skill_id": skill_id,
                        "name": sk.name,
                        "name_history":
                            sk.name_history,
                        "slow_update_field":
                            sk.slow_update_field,
                    }, f, indent=2)

            if sk.references:
                refs_dir = os.path.join(
                    skill_dir, "references")
                os.makedirs(
                    refs_dir, exist_ok=True)
                for fname, content in \
                        sk.references.items():
                    with open(
                        os.path.join(
                            refs_dir, fname),
                        "w", encoding="utf-8"
                    ) as f:
                        f.write(content)

        with open(
            os.path.join(
                library_dir, "manifest.json"),
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                self.skill_ids(), f, indent=2)

    @classmethod
    def load(
        cls, library_dir: str,
    ) -> "SkillLibrary":
        manifest_path = os.path.join(
            library_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"manifest.json 不存在："
                f"{manifest_path}")
        with open(
            manifest_path, "r", encoding="utf-8"
        ) as f:
            skill_ids = json.load(f)

        skills = []
        for sid in skill_ids:
            skill_dir = os.path.join(
                library_dir, sid)
            if not os.path.isdir(skill_dir):
                print(
                    f"  ⚠ 跳过：{sid}")
                continue
            try:
                skills.append(
                    SkillCore.from_directory(
                        skill_dir, sid))
            except Exception as e:
                print(
                    f"  ⚠ 加载失败 {sid}：{e}")
                continue

        print(f"  加载 Library："
              f"{len(skills)} 个 Skills")
        return cls(skills)

    @classmethod
    def load_from_task(
        cls, task_dir: str, task_name: str,
    ) -> "SkillLibrary":
        skills_dir = os.path.join(
            task_dir, "environment", "skills")
        if not os.path.isdir(skills_dir):
            return cls([])
        skills = []
        for skill_name in sorted(
            os.listdir(skills_dir)
        ):
            skill_path = os.path.join(
                skills_dir, skill_name)
            if not os.path.isdir(skill_path):
                continue
            skill_id = (
                f"{task_name}__{skill_name}")
            try:
                skills.append(
                    SkillCore.from_directory(
                        skill_path, skill_id))
            except Exception as e:
                print(
                    f"  ⚠ 跳过 {skill_id}：{e}")
        return cls(skills)

    @classmethod
    def load_single_file(
        cls, skill_path: str,
        skill_id: str = "main",
    ) -> "SkillLibrary":
        """
        从单个 SKILL.md 文件创建只含一个 skill 的库。
        用于 data/skill/SKILL.md 单文件模式。
        """
        if not os.path.exists(skill_path):
            raise FileNotFoundError(
                f"Skill 文件不存在：{skill_path}")

        with open(
            skill_path, "r", encoding="utf-8"
        ) as f:
            raw = f.read()

        fm   = {}
        body = raw.strip()
        if raw.strip().startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm   = yaml.safe_load(
                        parts[1]) or {}
                    body = parts[2].strip()
                except yaml.YAMLError:
                    fm   = {}
                    body = raw.strip()

        name = fm.get("name", "") or skill_id
        description = fm.get("description", "")
        if not description and body:
            for line in body.split("\n"):
                line = line.strip().lstrip(
                    "#").strip()
                if line and not line.startswith(
                    "```"):
                    description = line[:300]
                    break

        sk = SkillCore(
            skill_id=skill_id,
            name=name,
            description=description,
            execution_body=body,
            raw_skill_md=raw,
        )
        print(f"  加载单 Skill：{skill_id}"
              f" ({len(body)} chars)")
        return cls([sk])

    def save_single_file(
        self, skill_path: str,
        skill_id: str = "main",
    ):
        """
        将库中的单个 skill 保存为 .md 文件。
        """
        sk = self.get(skill_id)
        if sk is None and len(self._skills) > 0:
            sk = self._skills[0]
        if sk is None:
            return

        os.makedirs(
            os.path.dirname(skill_path),
            exist_ok=True)

        content = f"---\nname: {sk.name}\n"
        if sk.description:
            content += (
                f"description: {sk.description}\n")
        content += f"---\n\n{sk.execution_body}\n"

        with open(
            skill_path, "w", encoding="utf-8"
        ) as f:
            f.write(content)
        print(f"  💾 Skill 已保存：{skill_path}")
