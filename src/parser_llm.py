# src/parser_llm.py
"""
Parser LLM：定义测量结构（§3.1）

独立于 Student / Teacher / Editor 的 LLM。
职责：拆分 skill → modules/groups + 提取 labels Z + 定义 φ 规则

论文要求（§3.1）：
  - Z: 有限行为标签空间
  - φ: 从可见证据映射到 met/unmet/cannot_assess
  - parent(z): 标签对应的 skill 可编辑位置
  - pooling groups g: λ_g 的估计单位

Parser 只在正式估计窗口前调用一次，输出后冻结。
"""

import json
import re
from dataclasses import dataclass, field
from src.client import OpenRouterClient
from src.label_space import Module, OTHER_LABEL


@dataclass
class ParsedLabel:
    """一个行为标签的完整定义"""
    name: str                # 标签名（如 "formula_check"）
    description: str = ""    # 可读描述
    evidence_rule: str = ""  # φ 判定规则
    parent_location: str = ""  # 对应 skill 中的位置
    code_patterns: list = field(
        default_factory=list)  # 用于 φ 的代码正则


@dataclass
class ParsedMeasurement:
    """Parser LLM 输出的完整测量结构"""
    labels: list[ParsedLabel] = field(
        default_factory=list)
    groups: list[dict] = field(
        default_factory=list)
    raw_response: str = ""


def validate_measurement_structure(
    parsed: ParsedMeasurement,
    min_labels_per_group: int = 3,
) -> None:
    """Fail-closed structural validation before the estimation window."""
    labels = {lb.name: lb for lb in parsed.labels}
    if not labels:
        raise ValueError("parser returned no behavior labels")
    assigned = set()
    for group in parsed.groups:
        gid = group.get("group_id")
        histories = group.get("histories", [])
        if histories:
            if not gid:
                raise ValueError("history group lacks group_id")
            seen_history_ids = set()
            for history in histories:
                hid = history.get("history_id")
                names = history.get("label_names", [])
                patterns = history.get(
                    "applicability_patterns",
                    history.get("reach_patterns", []))
                if not hid or hid in seen_history_ids:
                    raise ValueError(
                        f"invalid or duplicate history_id {hid!r} in {gid}")
                seen_history_ids.add(hid)
                if len(names) < min_labels_per_group:
                    raise ValueError(
                        f"history {hid!r} requires at least "
                        f"{min_labels_per_group} labels")
                if not history.get("skill_step_text") or not patterns:
                    raise ValueError(
                        f"history {hid!r} lacks skill_step_text/applicability")
                for pattern in patterns:
                    re.compile(pattern)
                unknown = set(names) - set(labels)
                overlap = set(names) & assigned
                if unknown or overlap or len(names) != len(set(names)):
                    raise ValueError(
                        f"invalid history {hid}: unknown={sorted(unknown)}, "
                        f"reused={sorted(overlap)}")
                for name in names:
                    lb = labels[name]
                    if not lb.parent_location or not lb.evidence_rule \
                            or not lb.code_patterns:
                        raise ValueError(
                            f"label {name} lacks observable/editable "
                            "measurement metadata")
                    for pattern in lb.code_patterns:
                        try:
                            re.compile(pattern)
                        except re.error as exc:
                            raise ValueError(
                                f"invalid regex for label {name}: "
                                f"{exc}") from exc
                assigned.update(names)
            continue
        names = group.get("label_names", [])
        if not gid or len(names) < min_labels_per_group:
            raise ValueError(
                f"invalid pooling group {gid!r}: requires at least "
                f"{min_labels_per_group} labels")
        reach_patterns = group.get("reach_patterns", [])
        if not reach_patterns:
            raise ValueError(
                f"pooling group {gid!r} lacks reach_patterns")
        for pattern in reach_patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"invalid reach regex for group {gid}: {exc}") from exc
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate labels inside group {gid}")
        unknown = set(names) - set(labels)
        overlap = set(names) & assigned
        if unknown or overlap:
            raise ValueError(
                f"invalid group {gid}: unknown={sorted(unknown)}, "
                f"reused={sorted(overlap)}")
        for name in names:
            lb = labels[name]
            if not lb.parent_location or not lb.evidence_rule \
                    or not lb.code_patterns:
                raise ValueError(
                    f"label {name} lacks observable/editable measurement metadata")
            for pattern in lb.code_patterns:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(
                        f"invalid regex for label {name}: {exc}") from exc
        assigned.update(names)
    if assigned != set(labels):
        raise ValueError(
            f"unassigned parser labels: {sorted(set(labels) - assigned)}")


PARSER_SYSTEM = """You are a measurement structure parser for a skill distillation system.

Your job is to analyze a skill document and define the behavior labels that can be:
1. NAMED as concrete behaviors (e.g., "check_referenced_range", "use_header_lookup")
2. OBSERVED from code output, logs, or benchmark feedback
3. EDITED by changing the skill document

═════════════════════════════════════════════════════════
CORE PRINCIPLE — labels within a Skill history form a PARTITION
═════════════════════════════════════════════════════════

First split the skill into workflow GROUPS, like paragraphs. Then split each
group into concrete SKILL HISTORIES, like sentences/steps. Similar and
consecutive Skill histories belong to one group and share one lambda_g.

A rollout action/result step is evidence and may apply to MULTIPLE Skill
histories, including histories from different groups. For every applicable
`(rollout step, Skill history)` pair, exactly one label is assigned:

    z_h(τ) ∈ Z_h = {label_1, label_2, ..., label_k, OTHER}

where **exactly one** label z_g(h) is assigned to each reached history, and

    Σ_{z ∈ Z_h}  P(z | h)  =  1

This is a strict probabilistic requirement. If labels overlap so that a single
rollout step could reasonably be tagged with two labels for the same history,
the sum-to-one property breaks.

Therefore:

    (a) Labels inside one Skill history MUST be MUTUALLY EXCLUSIVE — it is a
      partition of a shared design decision (e.g., "how does the student
      access columns?" has options {header_map, hardcoded_letters,
      iter_rows_no_column}).

  (b) Labels inside one Skill history MUST be COLLECTIVELY DISCRIMINATING —
      different rollout evidence should tend to land on different labels.
      If 90%+ of trajectories match the same label, split or refine.

  (c) An `OTHER` bucket is automatically appended by the framework to
      catch trajectories that match no named label. Do NOT include OTHER
      yourself; the framework adds it.

  (d) Every Skill history must have AT LEAST 3 named labels (excluding OTHER),
      so it is a real decision point with alternatives. If you cannot
      find 3+ mutually-exclusive alternatives for a design decision, do
      NOT create that group.

═════════════════════════════════════════════════════════
GROUPS ARE DESIGN DECISIONS, not section headings
═════════════════════════════════════════════════════════

Do not create groups by copy-pasting the skill's Markdown section titles.
Create groups by identifying **design decision points** in the skill:

  ✗ Bad: {group="Environment", labels=["uses_openpyxl", ...]}
    (Not a decision — everyone uses openpyxl.)

  ✓ Good: {group="column_access_strategy",
           labels=["header_map_lookup",
                   "hardcoded_column_letter",
                   "iter_rows_with_offset"]}
    (A real decision: how does the student find columns?)

  ✓ Good: {group="none_value_policy",
           labels=["explicit_none_guard",
                   "or_default_shortcut",
                   "no_none_handling"]}
    (A real decision: how does the student handle None values?)

For each label, provide:
- name: short snake_case identifier
- description: one-line explanation
- evidence_rule: how to judge met/unmet from visible evidence
- parent_location: which section of the skill affects this behavior
- code_patterns: regex patterns that detect this behavior in Python code

For each Skill history, provide `applicability_patterns`: 1-3 regexes deciding
whether rollout evidence applies to that skill step. Different Skill histories
MAY apply to the same rollout step. Labels are mutually exclusive only WITHIN
one Skill history.

Output ONLY valid JSON in this format:
{
  "labels": [
    {
      "name": "header_map_lookup",
      "description": "Build a header→column dict from row 1, then use it",
      "evidence_rule": "met if `header_map` dict is constructed from ws[1] iteration",
      "parent_location": "Code Template > column resolution",
      "code_patterns": ["header_map\\\\s*=\\\\s*\\\\{", "for\\\\s+cell\\\\s+in\\\\s+ws\\\\[1\\\\]"]
    }
  ],
  "groups": [
    {
            "group_id": "column_access_workflow",
            "histories": [
                {
                    "history_id": "inspect_headers",
                    "skill_step_text": "Inspect headers before resolving columns.",
                    "applicability_patterns": ["ws\\[1\\]", "iter_rows\\("],
                    "label_names": ["header_map_lookup", "hardcoded_column_letter", "iter_rows_with_offset"]
                }
            ]
    }
  ]
}

Rules:
- Each label must be observable from code or output (no hidden reasoning)
- Each label must map to an editable part of the skill
- Include 3-6 GROUPS, each with 1-5 consecutive/similar Skill histories;
    each Skill history has 3-5 mutually-exclusive labels
  → Total 10-20 labels (KEEP JSON < 4000 chars)
- Always include patterns that work with openpyxl code
- Keep `description` and `evidence_rule` to ONE short sentence each
- Keep `code_patterns` list to 1-3 patterns per label (short regexes)
- Keep `applicability_patterns` to 1-3 patterns per Skill history
- DO NOT add trailing commentary or explanations

**MUTUAL EXCLUSIVITY VERIFICATION** (before you output, check yourself):
For each group g, imagine 4 different plausible student solutions to the same task.
Each solution must be labelled by exactly ONE label in group g (or fall into OTHER).
If two labels would BOTH match the same solution, your group is broken — fix it by:
  - Making patterns more specific
  - Merging the two labels
  - Removing the less discriminating one

Do not force different histories' applicability patterns to be exclusive: one
rollout code block often implements several skill steps at once.

**COMMON PITFALLS TO AVOID**:
- Avoid overly generic labels like "uses_openpyxl", "proper_save", "loads_workbook"
  that would match every solution → they carry no distinguishing information
- Avoid labels whose patterns are near-duplicates
  (e.g., "save_workbook" and "proper_workbook_save" — pick one)
- Prefer patterns that catch a SPECIFIC decision, not a common utility"""


PARSER_USER_TEMPLATE = """Analyze this skill document and define the measurement structure.

=== SKILL DOCUMENT ===
{skill_text}

=== TASK REQUIREMENTS ===
Tasks are SpreadsheetBench Excel manipulation tasks. The student writes Python code using openpyxl to modify Excel workbooks.

=== YOUR JOB ===
Identify 3-6 **design decision points** in this skill where students can choose
between mutually-exclusive strategies. For each decision, list the 3-5 alternative
strategies as labels. These labels form a POOLING GROUP.

Examples of good decision points for openpyxl tasks:
  - How to find columns:    header_map / hardcoded_letters / iter_rows_offset
  - How to handle None:     explicit_guard / or_default / no_handling
  - How to iterate rows:    max_row_range / iter_rows / hardcoded_range
  - How to read formulas:   data_only_dual_load / single_load / no_check
  - How to normalize text:  strip_lower / casefold / no_normalization

Verify: for each group, imagine 4 real student solutions — each must land on
EXACTLY ONE label (or OTHER). If two labels can both match the same solution,
your group is broken. Fix it before outputting.

CRITICAL: Output ONLY a JSON object starting with `{{` and ending with `}}`. Do NOT wrap in markdown code blocks. Do NOT add explanation text before or after. Just the raw JSON."""


HISTORY_MAPPING_SYSTEM = """You map visible rollout steps onto an already frozen
set of Skill histories. You must not create, rename, merge, or reinterpret any
history. A rollout step may apply to zero, one, or multiple histories. Judge only
from the supplied action and external result. Use status=assigned with one or more
frozen history IDs; status=unassigned when none apply; status=uncertain when the
visible evidence is insufficient. For unassigned/uncertain, history_ids must be
empty. Return every item exactly once. Output only strict JSON."""


class ParserLLM:
    """
    Parser LLM：定义 Z, φ, parent(z), groups

    每轮调用一次，输出冻结后不可修改。
    """

    def __init__(self, config: dict, client: OpenRouterClient):
        self.client = client
        pcfg = config.get("parser", {})
        self.model = pcfg.get(
            "model", "anthropic/claude-opus-4.5")
        self.max_tokens = pcfg.get("max_tokens", 2048)
        self.temperature = pcfg.get("temperature", 0.0)

    def parse_skill(
        self,
        skill_text: str,
        task_description: str = "",
    ) -> ParsedMeasurement:
        """
        调用 Parser LLM 解析 skill，返回测量结构。

        如果 LLM 调用失败，fallback 到正则提取。
        """
        user_msg = PARSER_USER_TEMPLATE.format(
            skill_text=skill_text,
        )

        response, usage = self.client.chat(
            model=self.model,
            messages=[{"role": "user",
                       "content": user_msg}],
            system=PARSER_SYSTEM,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            call_type="parser_llm",
        )

        print(f"    [Parser LLM] "
              f"{usage.get('total_tokens', 0)} tokens, "
              f"response={len(response)} chars")

        # 解析 JSON
        parsed = self._parse_response(response)
        if not parsed.labels:
            print("    ⚠️ Parser LLM 输出无效，"
                  "fallback 到正则提取")
            print(f"    [Parser LLM raw] "
                  f"{response[:500]}")
            return parsed

        print(f"    [Parser LLM] 解析出 "
              f"{len(parsed.labels)} labels, "
              f"{len(parsed.groups)} groups")
        return parsed

    def map_rollout_steps(
        self,
        trajectories: list,
        frozen_modules: list[Module],
        chunk_steps: int = 30,
        evidence_chars: int = 6000,
    ) -> dict:
        """Batch-map actual rollout steps to frozen Skill histories.

        The operation is transactional: any malformed/incomplete chunk resets
        every input step to ``missing`` and returns ``complete=False``. There is
        deliberately no regex fallback.
        """
        histories = []
        valid_ids = set()
        for module in frozen_modules:
            history_id = module.history_id or module.module_id
            if not history_id or history_id in valid_ids:
                return {"complete": False,
                        "error": "duplicate_or_empty_frozen_history"}
            valid_ids.add(history_id)
            histories.append({
                "history_id": history_id,
                "skill_id": module.skill_id,
                "group_id": module.group_id,
                "skill_step_text": module.skill_step_text,
            })

        items = []
        for trajectory_index, trajectory in enumerate(trajectories):
            if not getattr(trajectory, "evaluation_valid", True):
                continue
            for step_index, step in enumerate(trajectory.steps):
                step.applicable_history_ids = None
                step.history_mapping_status = "missing"
                step.history_mapping_reason = ""
                item_id = f"r{trajectory_index}:s{step_index}"
                evidence = "\n".join([
                    getattr(step, "action", "") or "",
                    getattr(step, "external_result", "") or "",
                ])
                items.append((item_id, step, {
                    "item_id": item_id,
                    "task_id": str(trajectory.task_id),
                    "step_index": int(getattr(step, "step", step_index)),
                    "evidence": evidence[:evidence_chars],
                }))
        if not items:
            return {"complete": False, "error": "no_valid_rollout_steps"}

        all_assignments = []
        try:
            for start in range(0, len(items), max(int(chunk_steps), 1)):
                chunk = items[start:start + max(int(chunk_steps), 1)]
                payload = {
                    "frozen_histories": histories,
                    "rollout_steps": [item[2] for item in chunk],
                    "output_schema": {
                        "assignments": [{
                            "item_id": "r0:s0",
                            "status": "assigned|unassigned|uncertain",
                            "history_ids": ["frozen history ID"],
                            "reason": "brief visible-evidence reason",
                        }],
                    },
                }
                response, usage = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": json.dumps(
                        payload, ensure_ascii=False)}],
                    system=HISTORY_MAPPING_SYSTEM,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    call_type="parser_history_mapping",
                )
                data = json.loads(response.strip())
                assignments = data.get("assignments")
                if not isinstance(assignments, list):
                    raise ValueError("mapping response lacks assignments")
                expected = {item[0]: item[1] for item in chunk}
                returned = [a.get("item_id") for a in assignments
                            if isinstance(a, dict)]
                if len(returned) != len(expected) or set(returned) != set(expected):
                    raise ValueError("mapping response is incomplete or duplicated")
                for assignment in assignments:
                    item_id = assignment["item_id"]
                    status = assignment.get("status")
                    history_ids = assignment.get("history_ids")
                    if status not in {"assigned", "unassigned", "uncertain"}:
                        raise ValueError(f"invalid mapping status for {item_id}")
                    if not isinstance(history_ids, list) or any(
                            not isinstance(value, str) for value in history_ids):
                        raise ValueError(f"invalid history_ids for {item_id}")
                    if len(history_ids) != len(set(history_ids)) or not set(
                            history_ids).issubset(valid_ids):
                        raise ValueError(f"unknown/duplicate history ID for {item_id}")
                    if ((status == "assigned") != bool(history_ids)):
                        raise ValueError(f"status/history mismatch for {item_id}")
                    step = expected[item_id]
                    step.applicable_history_ids = list(history_ids)
                    step.history_mapping_status = status
                    step.history_mapping_reason = str(
                        assignment.get("reason", ""))[:500]
                    all_assignments.append({
                        "item_id": item_id, "status": status,
                        "history_ids": list(history_ids),
                        "reason": step.history_mapping_reason,
                    })
        except Exception as exc:
            for _, step, _ in items:
                step.applicable_history_ids = None
                step.history_mapping_status = "missing"
                step.history_mapping_reason = str(exc)[:500]
            return {"complete": False, "error": str(exc),
                    "n_steps": len(items), "assignments": []}
        return {"complete": True, "error": "", "n_steps": len(items),
                "assignments": all_assignments}

    def _parse_response(
        self, response: str,
    ) -> ParsedMeasurement:
        """解析 LLM 的 JSON 输出（多策略）"""
        if not response or not response.strip():
            return ParsedMeasurement(
                raw_response=response)

        # 策略1：优先提取 ```json ... ``` 块
        code_block = re.search(
            r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',
            response)
        candidates = []
        if code_block:
            candidates.append(code_block.group(1))

        # 策略2：贪婪匹配最大 {...}
        greedy = re.search(
            r'\{[\s\S]*\}', response)
        if greedy:
            candidates.append(greedy.group())

        # 策略3：截断修复 —— 从 `{` 开始
        first_brace = response.find("{")
        if first_brace >= 0:
            truncated = response[first_brace:]
            candidates.append(truncated)
            # 尝试补齐结尾
            candidates.append(
                self._try_repair_json(truncated))

        for text in candidates:
            if not text:
                continue
            try:
                data = json.loads(text)
                if isinstance(data, dict) and \
                        "labels" in data:
                    return self._build_result(
                        data, response)
            except json.JSONDecodeError:
                continue

        # 策略4：正则宽松抽取 labels
        # 从截断的 JSON 里提出所有 label
        loose_labels = self._loose_extract_labels(
            response)
        if loose_labels:
            print(f"    [Parser LLM] 宽松抽取"
                  f"到 {len(loose_labels)} labels")
            return ParsedMeasurement(
                labels=loose_labels,
                groups=[],
                raw_response=response,
            )

        return ParsedMeasurement(
            raw_response=response)

    @staticmethod
    def _try_repair_json(text: str) -> str:
        """
        尝试补齐被截断的 JSON。
        统计未闭合的 {[ 并补上对应的 }]
        """
        if not text:
            return ""
        # 去掉尾部不完整行
        text = text.rstrip()
        # 去掉尾部逗号+可能的半截字符串
        text = re.sub(
            r',\s*"[^"]*$', "", text)
        text = re.sub(
            r',\s*\{[^}]*$', "", text)
        text = text.rstrip(", \n")

        # 统计括号
        depth_brace = 0
        depth_bracket = 0
        in_str = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace -= 1
            elif ch == "[":
                depth_bracket += 1
            elif ch == "]":
                depth_bracket -= 1

        # 若在字符串里，先关掉字符串
        if in_str:
            text += '"'
        # 补齐 ]、}
        text += "]" * max(depth_bracket, 0)
        text += "}" * max(depth_brace, 0)
        return text

    @staticmethod
    def _loose_extract_labels(
        response: str,
    ) -> list:
        """
        从半结构化文本里正则抽取 label。
        匹配 "name": "xxx" 块。
        """
        labels = []
        seen = set()
        # 匹配每个 label 对象里的核心字段
        # 至少要有 name
        pattern = re.compile(
            r'"name"\s*:\s*"([^"\n]+)"'
            r'(?:.*?"description"\s*:\s*"([^"\n]*)")?'
            r'(?:.*?"evidence_rule"\s*:\s*'
            r'"([^"\n]*)")?',
            re.DOTALL,
        )
        for m in pattern.finditer(response):
            name = m.group(1).strip()
            # 过滤明显非 label 的 name
            if (not name or len(name) > 60
                    or name in seen):
                continue
            # skip 一些明显是 group 名的
            if name.lower() in {
                "labels", "groups", "modules",
            }:
                continue
            seen.add(name)
            labels.append(ParsedLabel(
                name=name,
                description=(m.group(2) or "").strip(),
                evidence_rule=(
                    m.group(3) or "").strip(),
                parent_location="",
                code_patterns=[],
            ))
        return labels

    def _build_result(
        self, data: dict, response: str,
    ) -> ParsedMeasurement:

        labels = []
        for lb in data.get("labels", []):
            if not isinstance(lb, dict):
                continue
            name = lb.get("name", "")
            if not name:
                continue
            labels.append(ParsedLabel(
                name=name,
                description=lb.get(
                    "description", ""),
                evidence_rule=lb.get(
                    "evidence_rule", ""),
                parent_location=lb.get(
                    "parent_location", ""),
                code_patterns=lb.get(
                    "code_patterns", []),
            ))

        groups = data.get("groups", [])

        return ParsedMeasurement(
            labels=labels,
            groups=groups,
            raw_response=response,
        )

    def to_modules(
        self,
        parsed: ParsedMeasurement,
        skill_id: str = "main",
        force_single_group: bool = False,
        min_labels_per_group: int = 5,
    ) -> list[Module]:
        """
        将 Parser LLM 输出转为 Module 列表
        （供 label_space.py 使用）

        参数
        ----
        force_single_group : True 时忽略 Parser 输出的 group 划分，
            把全部 label 合并成一个 module。避免 group 内 label
            太少导致 WLS 分母崩坏、λ 估计退化。
        min_labels_per_group : Parser 输出的 group 少于此阈值时，
            合并到 "misc" module；若所有 group 都小于此阈值，
            退化为单 group。
        """
        validate_measurement_structure(
            parsed, min_labels_per_group=(
                1 if force_single_group
                else max(3, min_labels_per_group)))

        named = [lb.name for lb in parsed.labels]
        labels_by_name = {lb.name: lb for lb in parsed.labels}

        # 正式 nested schema：每个 Module 实际是一个 Skill history；
        # group_id 只用于在 λ 求解阶段 pooling。
        if parsed.groups and any(
                group.get("histories") for group in parsed.groups):
            histories = []
            for group in parsed.groups:
                gid = group["group_id"]
                for history in group.get("histories", []):
                    hid = history["history_id"]
                    history_labels = list(history.get("label_names", []))
                    histories.append(Module(
                        module_id=f"{skill_id}::{hid}",
                        history_id=f"{skill_id}::{hid}",
                        group_id=f"{skill_id}::{gid}",
                        skill_id=skill_id,
                        name=hid,
                        skill_step_text=history["skill_step_text"],
                        named_behaviors=history_labels,
                        reach_patterns=list(history.get(
                            "applicability_patterns",
                            history.get("reach_patterns", []))),
                        label_patterns={
                            name: list(labels_by_name[name].code_patterns)
                            for name in history_labels},
                        label_parents={
                            name: labels_by_name[name].parent_location
                            for name in history_labels},
                    ))
            if not histories:
                raise ValueError("parser produced no Skill histories")
            if force_single_group:
                common_group = f"{skill_id}::{skill_id}"
                for history in histories:
                    history.group_id = common_group
            return histories

        # ─── 强制单 group（ablation / 稳健模式）───
        if force_single_group:
            return [Module(
                module_id=f"{skill_id}::{skill_id}",
                skill_id=skill_id,
                name=skill_id,
                named_behaviors=named,
                reach_patterns=[r"[\s\S]"],
                label_patterns={
                    lb.name: list(lb.code_patterns)
                    for lb in parsed.labels},
                label_parents={
                    lb.name: lb.parent_location
                    for lb in parsed.labels},
            )]

        # ─── 正常多 group 模式 ───
        # ⚠ 老师 2026-07-11：labels within a group form a
        # partition. 保证：
        #   (a) label 全局不重复（一个 label 只属于一个
        #       group，否则 P_s 双重计数）
        #   (b) 每 group 至少 min_labels_per_group（=3）
        #       个 named labels，才是合法 pooling group
        #   (c) 落单的 labels（含 Parser 未分组的）合并到 misc
        if parsed.groups:
            big_groups = []
            small_labels = []
            seen_labels = set()  # 全局 label 去重
            for g in parsed.groups:
                gid = g.get("group_id", skill_id)
                # 只保留：在 named 中 且 未出现过（去重）
                g_labels = []
                for n in g.get("label_names", []):
                    if n in named and n not in seen_labels:
                        g_labels.append(n)
                        seen_labels.add(n)
                if len(g_labels) >= min_labels_per_group:
                    big_groups.append((gid, g_labels))
                else:
                    # label 数不足 → 收集起来合并到 misc
                    small_labels.extend(g_labels)

            # 全部太小 → fail closed；不改变 Parser 冻结的 ruler。
            if not big_groups:
                raise ValueError(
                    "parser produced no valid pooling group")

            modules = [
                Module(
                    module_id=f"{skill_id}::{gid}",
                    skill_id=skill_id,
                    name=gid,
                    named_behaviors=g_labels,
                    reach_patterns=list(next(
                        g.get("reach_patterns", [])
                        for g in parsed.groups
                        if g.get("group_id", skill_id) == gid)),
                    label_patterns={
                        lb.name: list(lb.code_patterns)
                        for lb in parsed.labels
                        if lb.name in g_labels},
                    label_parents={
                        lb.name: lb.parent_location
                        for lb in parsed.labels
                        if lb.name in g_labels},
                )
                for gid, g_labels in big_groups
            ]

            # 小 group 的 label 合并到 "misc"
            # 也把 named 里但 Parser 没分组的 label 加进来
            leftover = [
                n for n in named
                if n not in seen_labels]
            all_misc = small_labels + leftover
            if all_misc:
                # 冻结后不把不同 design decisions 偷合并成 misc。
                seen2 = set()
                misc = [x for x in all_misc
                        if not (x in seen2
                                or seen2.add(x))]
                if len(misc) >= min_labels_per_group:
                    raise ValueError(
                        "parser labels from different design decisions "
                        "cannot be merged into a synthetic misc group")
                else:
                    raise ValueError(
                        f"labels cannot form a valid frozen group: {misc}")
            return modules

        # ─── Parser 输出就是单 group ───
        return [Module(
            module_id=f"{skill_id}::{skill_id}",
            skill_id=skill_id,
            name=skill_id,
            named_behaviors=named,
            reach_patterns=list(
                parsed.groups[0].get("reach_patterns", [])),
            label_patterns={
                lb.name: list(lb.code_patterns)
                for lb in parsed.labels},
            label_parents={
                lb.name: lb.parent_location
                for lb in parsed.labels},
        )]

    def to_code_patterns(
        self,
        parsed: ParsedMeasurement,
    ) -> dict[str, list[str]]:
        """
        将 Parser LLM 的 code_patterns 转为
        _CODE_BEHAVIOR_PATTERNS 格式
        （供 φ labeler 使用）
        """
        patterns = {}
        for lb in parsed.labels:
            if lb.code_patterns:
                patterns[lb.name] = lb.code_patterns
        return patterns
