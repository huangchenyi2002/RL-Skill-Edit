# src/label_space.py
"""
Label Space：行为标签空间

核心数学流水线（对应 lambda_estimation_selfcontained_cn）：
  1. Parser LLM 拆 skill → modules/groups → Z (labels)
  2. φ labeler：从 rollout 检测行为标签
  3. P 估计：Dirichlet smoothed P_s / P_T / P_0
  4. η(h,z) = E[Y|h,z]：局部成功率
  5. P_+(z|h)：成功加权目标
  6. a_+(h,z) = log(P_+/P_0)：有用度倾斜
  7. r_T(h,z) = log(P_T/P_0)：teacher-reference contrast
  8. λ_g：将 a_+ 投影到 r_T 方向的 WLS 斜率 + 收缩
  9. Q_λ(z|h) ∝ P_0 exp{λ_g r_T}：target 分布
  10. g_n = log(Q/P_s)：edit gradient → donor pair

λ 不是信心分数，而是 reward-calibrated projection slope：
  "teacher 指的方向和任务真正有用的方向对得有多齐"
"""

import re
import math
import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────

@dataclass
class GroupDefinition:
    """相似 Skill histories 的 pooling group；只负责共享 λ_g。"""
    group_id: str
    skill_id: str
    name: str = ""
    history_ids: list = field(default_factory=list)


@dataclass
class Module:
    """一个 Skill history 定义；Module 名仅保留旧接口兼容。

    正式语义：skill 中的一条 step = 一个 history。多个相似 histories
    通过 group_id 归组并共享 λ_g，但各自保留 Z_h 与 P(·|h)。
    """
    module_id: str
    skill_id:  str
    name:      str = ""
    named_behaviors: list = field(
        default_factory=list)
    # 冻结测量规则：history 先由 reach_patterns 判定属于本 group，
    # 再由本 group 独立的 label_patterns 映射到一个 categorical z。
    reach_patterns: list = field(default_factory=list)
    label_patterns: dict = field(default_factory=dict)
    # Frozen parent(z): each named behavior maps to its editable skill location.
    label_parents: dict = field(default_factory=dict)
    group_id: str = ""
    history_id: str = ""
    skill_step_text: str = ""

    def __post_init__(self):
        if not self.history_id:
            self.history_id = self.module_id
        if not self.group_id:
            self.group_id = self.module_id


SkillHistoryDefinition = Module


@dataclass(frozen=True)
class HistoryRecord:
    """rollout evidence 对某个 Skill history 的 categorical observation。"""
    task_id: str
    trajectory_id: str
    step_index: int
    source: str
    module_id: str
    skill_id: str
    label: str
    reward: float
    group_id: str = ""
    history_id: str = ""


@dataclass
class LabelDistribution:
    """一个 module 内的行为标签分布"""
    module_id:  str
    counts:     dict = field(default_factory=dict)
    total:      int  = 0
    probs:      dict = field(default_factory=dict)
    alpha:      float = 1.0  # Dirichlet smoothing


@dataclass
class ModuleSignal:
    """一个 Skill history 的完整信号；同 group histories 共享 λ。"""
    module_id:   str
    skill_id:    str
    d_s:         float = 0.0   # 访问频率
    m:           float = 1.0   # history 权重（已中性化=1）
    w:           float = 0.0   # d_s × m
    P_s:         dict  = field(default_factory=dict)
    P_T:         dict  = field(default_factory=dict)
    P_0:         dict  = field(default_factory=dict)
    R_hat:       dict  = field(default_factory=dict)
    # reward-calibrated λ 的中间量（供事后审计）：
    # η(h,z) = E[Y|h,z]  局部成功率
    # a_+(h,z) = log(P_+/P_0)  有用度倾斜
    eta:         dict  = field(default_factory=dict)
    a_plus:      dict  = field(default_factory=dict)
    lambda_:     dict  = field(default_factory=dict)
    Q_n:         dict  = field(default_factory=dict)
    g_n:         dict  = field(default_factory=dict)
    donor_plus:  str   = ""
    donor_minus: str   = ""
    # 观测次数：n_h in group（供 w_h 权重复现）
    obs_counts:  dict  = field(default_factory=dict)
    # 有限样本审计量（task-id clusters）
    lambda_raw:  float = 0.0
    N_g:         int = 0
    s_sq_g:      float = 0.0
    shrinkage_B: float = 0.0
    tau_sq_hat:  float = 0.0
    estimable: bool = True
    fallback_reason: str = ""
    reach_patterns: list = field(default_factory=list)
    label_patterns: dict = field(default_factory=dict)
    label_parents: dict = field(default_factory=dict)
    group_id: str = ""
    history_id: str = ""
    skill_step_text: str = ""


@dataclass
class EpochTarget:
    """一轮的完整冻结目标（步骤10冻结原则）"""
    modules:     list  = field(default_factory=list)
    beta:        float = 1.0
    frozen:      bool  = False
    freeze_digest: str = ""

    def freeze(self):
        """深拷贝并记录摘要；后续可用 assert_frozen() 检测任何变更。"""
        self.modules = copy.deepcopy(self.modules)
        self.frozen = True
        self.freeze_digest = self._digest()

    def _digest(self) -> str:
        payload = self.to_dict(include_digest=False)
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode("utf-8")).hexdigest()

    def assert_frozen(self):
        if not self.frozen:
            raise RuntimeError("epoch target has not been frozen")
        if self.freeze_digest != self._digest():
            raise RuntimeError("frozen epoch target was mutated")

    def to_dict(self, include_digest: bool = True) -> dict:
        out = {
            "beta": self.beta,
            "frozen": self.frozen,
            "modules": [
                {
                    "module_id": ms.module_id,
                    "skill_id": ms.skill_id,
                    "d_s": ms.d_s,
                    "m": ms.m,
                    "w": ms.w,
                    "P_s": ms.P_s,
                    "P_T": ms.P_T,
                    "P_0": ms.P_0,
                    "R_hat": ms.R_hat,
                    "eta": ms.eta,
                    "a_plus": ms.a_plus,
                    "lambda": ms.lambda_,
                    "Q_n": ms.Q_n,
                    "g_n": ms.g_n,
                    "donor_plus": ms.donor_plus,
                    "donor_minus": ms.donor_minus,
                    "obs_counts": ms.obs_counts,
                    "lambda_raw": ms.lambda_raw,
                    "N_g": ms.N_g,
                    "s_sq_g": ms.s_sq_g,
                    "shrinkage_B": ms.shrinkage_B,
                    "tau_sq_hat": ms.tau_sq_hat,
                    "estimable": ms.estimable,
                    "fallback_reason": ms.fallback_reason,
                    "reach_patterns": ms.reach_patterns,
                    "label_patterns": ms.label_patterns,
                    "label_parents": ms.label_parents,
                    "Z_h": list(ms.Q_n.keys() or ms.P_s.keys()),
                    "group_id": ms.group_id,
                    "history_id": ms.history_id,
                    "skill_step_text": ms.skill_step_text,
                }
                for ms in self.modules
            ],
        }
        if include_digest:
            out["freeze_digest"] = self.freeze_digest
        return out


# ─────────────────────────────────────────────────
# Step 1：从 SKILL.md 提取 modules 和 Z_named
# ─────────────────────────────────────────────────

# 行为标签的匹配模式
_RULE_PATTERNS = [
    # "### Rule 1: Find columns by header"
    r"###?\s*Rule\s*\d+[:\s]*(.+)",
    # "1. Load workbook"
    r"^\s*\d+\.\s+(.+)",
    # "- Always check None"
    r"^\s*[-*]\s+(?:Always|Never|Must|Do not|Check|Use|Handle|Verify|Ensure)\s+(.+)",
]

# 代码模式检测
# 每个 "语义 label" 有若干常见别名（Parser LLM 会给出多种命名），
# 每个别名下都放**宽泛且高召回**的 pattern，
# 避免因命名不一致或 pattern 过严导致 obs_counts 全为 0。
#
# 设计原则：宁可 recall 高一点也别让 label 客观存在却匹配不到。
_CODE_BEHAVIOR_PATTERNS = {
    # ── 列查找 ──
    "header_based_lookup": [
        r"header_map", r"headers\s*=\s*\{",
        r"col_map", r"column.?map",
        r"for\s+cell\s+in\s+ws\[1\]",
        r"for\s+cell\s+in\s+ws\.rows\.__next__",
        r"\.column\s+for\s+\w+\s+in",
        r"c\.value.*c\.column",
        r"header.*index",
        r"cell\.value\s*==?\s*['\"]",
        r"if\s+.*\.value\s*==?\s*['\"]",
    ],
    "column_lookup_by_header": [  # alias
        r"header_map", r"col_map",
        r"for\s+cell\s+in\s+ws\[1\]",
        r"cell\.value\s*==?\s*['\"]",
    ],
    "hardcoded_column_access": [
        r"ws\[['\"]?[A-Z]{1,3}\d*['\"]?\]",
        r"row\[\d+\]",
        r"column=\d+",
        r"cell\(\s*row=\w+,\s*column=\d+",
    ],

    # ── 公式相关 ──
    "formula_check": [
        r"data_only\s*=\s*True",
        r"load_workbook.*data_only",
        r"wb_val\s*=", r"wb_data\s*=",
        r"\.value.*formula",
        r"check.*formula", r"verify.*formula",
    ],
    "formula_value_read": [  # alias
        r"data_only\s*=\s*True",
        r"load_workbook.*data_only",
        r"wb_val\s*=", r"wb_data\s*=",
    ],
    "data_only_dual_load": [  # alias
        r"data_only\s*=\s*True",
        r"load_workbook.*data_only",
    ],

    # ── 数字处理 ──
    "safe_number_parsing": [
        r"safe_num", r"try.*float.*except",
        r"replace\(['\"].*,.*['\"].*\)",
        r"try:.*int\(", r"try:.*float\(",
        r"isinstance.*\(.*(?:int|float)",
        r"except\s+(?:ValueError|TypeError)",
        r"float\(str\(",
        r"str\(.*\)\.replace",
    ],

    # ── None 保护 ──
    "none_check": [
        r"if\s+.*\s+is\s+None",
        r"if\s+.*\s+is\s+not\s+None",
        r"if\s+not\s+\w+\s*:",
        r"if\s+\w+\s*(?:==|!=)\s*None",
        r"\bor\s+['\"]{2}\b", r"\bor\s+0\b",
        r"\.value\s+or\s+",
    ],
    "none_value_handling": [  # alias
        r"if\s+.*\s+is\s+None",
        r"if\s+.*\s+is\s+not\s+None",
        r"\.value\s+or\s+",
        r"or\s+['\"]{2}\b", r"or\s+0\b",
    ],
    "none_guard_check": [  # alias
        r"if\s+.*\s+is\s+None",
        r"if\s+.*\s+is\s+not\s+None",
        r"if\s+not\s+\w+\s*:",
    ],

    # ── save ──
    "save_workbook": [
        r"wb\.save\(", r"workbook\.save\(",
        r"\.save\(.*\.xlsx",
        r"wb\.save\(wb_path\)",
    ],
    "proper_workbook_save": [  # alias
        r"wb\.save\(", r"workbook\.save\(",
        r"\.save\(wb_path",
    ],

    # ── 文本处理 ──
    "text_normalization": [
        r"\.strip\(\)", r"\.lower\(\)",
        r"\.casefold\(\)", r"norm\(",
        r"\.upper\(\)", r"\.replace\(",
        r"str\(.*\)\.strip",
    ],

    # ── 行迭代 ──
    "range_iteration": [
        r"for\s+row\s+in\s+range\(",
        r"ws\.max_row", r"ws\.iter_rows",
        r"for\s+row\s+in\s+ws\.",
        r"for\s+\w+\s+in\s+range\(2",
        r"ws\.min_row", r"ws\.max_column",
    ],
    "dynamic_row_iteration": [  # alias
        r"for\s+row\s+in\s+range\(\s*\d+,\s*ws\.max_row",
        r"ws\.max_row", r"ws\.iter_rows\(",
    ],
    "max_row_iteration": [  # alias
        r"ws\.max_row",
        r"for\s+row\s+in\s+range\(",
    ],

    # ── 跨表 ──
    "cross_sheet_lookup": [
        r"wb\[['\"]?\w+['\"]?\]",
        r"wb\.sheetnames",
        r"for\s+sheet.*in\s+wb\.",
    ],
    "cross_table_lookup": [  # alias
        r"wb\[['\"]?\w+['\"]?\]",
        r"lookup\s*=\s*\{",
        r"lookup_dict",
    ],
    "cross_sheet_lookup_dict": [  # alias
        r"lookup\s*=\s*\{",
        r"lookup_dict",
        r"wb\[['\"]?\w+['\"]?\]",
    ],

    # ── 聚合 ──
    "defaultdict_aggregation": [
        r"defaultdict\(",
        r"from\s+collections\s+import\s+defaultdict",
        r"Counter\(",
    ],
    "manual_dict_aggregation": [
        r"totals\s*=\s*\{\}", r"counts\s*=\s*\{\}",
        r"if\s+\w+\s+not\s+in\s+\w+:",
        r"\.get\(\s*\w+,\s*0\s*\)",
    ],

    # ── 写值 ──
    "value_write": [
        r"ws\.cell.*\.value\s*=",
        r"ws\[.*\]\s*=",
        r"cell\.value\s*=",
    ],
    "writes_literal_values": [  # alias
        r"ws\.cell.*\.value\s*=",
        r"ws\[['\"]?[A-Z]",
        r"cell\.value\s*=",
    ],
    "clear_before_write": [
        r"\.value\s*=\s*None",
        r"clear.*range",
        r"delete_rows", r"delete_cols",
    ],
    "visible_value_answer": [
        r"ws\.cell.*\.value(?!\s*=)",
        r"=\s*ws\[",
        r"cell\.value(?!\s*=)",
        r"print\(.*\.value",
    ],
    "execute_python_solution": [
        r"openpyxl", r"import\s+os",
        r"load_workbook\(",
        r"wb\s*=\s*", r"ws\s*=\s*",
    ],
}

OTHER_LABEL = "other"
UNASSIGNED_GROUP_ID = "__unassigned_history__"
UNASSIGNED_LABEL = "unassigned"


class HistoryRoutingError(ValueError):
    """一个真实 history 无法被冻结 ruler 唯一路由时 fail closed。"""


class HistoryMappingUnavailableError(HistoryRoutingError):
    """Parser LLM mapping was not completed for every valid rollout step."""


def extract_modules_and_behaviors(
    library,
) -> list[Module]:
    """
    Step 1：从 SkillLibrary 提取 module 结构和 Z_named

    每个 skill = 一个 module
    每个 skill 的规则/步骤 = named behaviors
    """
    modules = []

    for sk in library:
        # 从 execution_body 提取规则名称
        named = set()
        body = sk.execution_body or ""

        # 方法1：从规则标题提取
        for pattern in _RULE_PATTERNS:
            for match in re.finditer(
                pattern, body, re.MULTILINE
            ):
                rule_text = match.group(1).strip()
                # 简化为标签名
                label = _text_to_label(rule_text)
                if label and label != OTHER_LABEL:
                    named.add(label)

        # 方法2：从代码模式推断
        for label, patterns in \
                _CODE_BEHAVIOR_PATTERNS.items():
            for p in patterns:
                if re.search(p, body, re.IGNORECASE):
                    named.add(label)
                    break

        # 确保至少有一些基本标签
        if not named:
            named = {"generic_execution"}

        modules.append(Module(
            module_id=sk.skill_id,
            skill_id=sk.skill_id,
            name=sk.name,
            named_behaviors=sorted(named),
        ))

    return modules


def _text_to_label(text: str) -> str:
    """把规则文本 / teacher_preferred_action 转成标签名"""
    text = text.lower().strip()
    # 关键词映射（按优先级排列，先匹配到的先返回）
    mappings = [
        # formula_check
        (["formula", "data_only", "referenced",
          "verify.*cell", "check.*range",
          "cross.?check"], "formula_check"),
        # header_based_lookup
        (["header", "column.*name", "find.*col",
          "index.?match", "look.?up"], "header_based_lookup"),
        # safe_number_parsing
        (["number", "numeric", "parse.*int",
          "float", "convert.*num"], "safe_number_parsing"),
        # none_check
        (["none", "null", "empty", "blank",
          "missing"], "none_check"),
        # save_workbook
        (["save", "wb.save"], "save_workbook"),
        # text_normalization
        (["normalize", "strip", "lower", "upper",
          "clean", "trim"], "text_normalization"),
        # range_iteration
        (["iterate", "row", "loop", "range",
          "max_row", "each cell"], "range_iteration"),
        # clear_before_write
        (["clear", "delete", "remove"],
         "clear_before_write"),
        # value_write
        (["write", "assign", "set.*value",
          "update.*cell"], "value_write"),
        # visible_value_answer
        (["visible", "display", "read.*value",
          "direct.*answer"], "visible_value_answer"),
        # execute_python_solution
        (["python", "openpyxl", "script",
          "code", "execute"], "execute_python_solution"),
    ]
    for keywords, label in mappings:
        for kw in keywords:
            if re.search(kw, text):
                return label
    # 生成通用标签
    words = re.findall(r"[a-z]+", text)[:3]
    if words:
        return "_".join(words)
    return ""


def get_Z_for_module(module: Module) -> list[str]:
    """返回 history 所属 group 内的 categorical label support。"""
    return list(module.named_behaviors) + [OTHER_LABEL]


# ─────────────────────────────────────────────────
# Step 2：φ labeler（从 rollout 检测行为）
# ─────────────────────────────────────────────────

# Parser LLM 输出的额外 patterns（每轮冻结）
_PARSER_CODE_PATTERNS: dict[str, list[str]] = {}


def set_parser_patterns(
    patterns: dict[str, list[str]],
):
    """设置 Parser LLM 输出的 code patterns（冻结）"""
    global _PARSER_CODE_PATTERNS
    _PARSER_CODE_PATTERNS = patterns
    n = sum(len(v) for v in patterns.values())
    print(f"    [Parser] 设置 {len(patterns)} 标签 "
          f"{n} 个 patterns")


def label_trajectory_step(
    step_code: str,
    step_action: str,
    module: Module,
    multi_label: bool = True,
) -> list[str]:
    """
    φ(τ,h) → {z}

    检测 Student 在某一步的代码匹配的 named behaviors。
    返回**列表**：一段代码里同时出现的所有 named behavior。

    参数
    ----
    multi_label : True (默认) 返回所有命中的 label 列表；
        False 保持旧行为，只返回最佳单 label（供 ablation）。

    关键设计：
      一段 Student 代码通常同时体现多个 label，比如：
      ```
      wb = load_workbook(wb_path)         ← 没触发 named
      for cell in ws[1]:                  ← header_based_lookup
          if cell.value == "Name":        ← header_based_lookup
      for row in range(2, ws.max_row+1):  ← range_iteration
          v = ws.cell(row, col).value
          if v is None: continue          ← none_check
      wb.save(wb_path)                    ← save_workbook
      ```
      若强行只选 1 个 label，其他 3 个的 obs_counts 永远是 0，
      λ 投影就退化为单点回归。
    """
    text = (step_code + " " + step_action).lower()
    return _label_visible_text(text, module, multi_label)


def _label_visible_text(
    text: str,
    module: Module,
    multi_label: bool = True,
):
    """使用 module-scoped patterns 标注一段可见 action/result evidence。"""

    # 每个 label 的命中率 & 命中数
    scores = {}
    for label in module.named_behaviors:
        # 合并内置 + Parser 的 patterns
        patterns = list(module.label_patterns.get(label, []))
        if not patterns:
            # legacy module only；正式 Parser module 总是自带 scoped rules。
            patterns = list(
                _CODE_BEHAVIOR_PATTERNS.get(label, []))
            patterns.extend(
                _PARSER_CODE_PATTERNS.get(label, []))
        if not patterns:
            continue

        hits = 0
        for p in patterns:
            try:
                if re.search(p, text, re.IGNORECASE):
                    hits += 1
                    if not multi_label:
                        # 单 label 模式：只算命中率，
                        # 早停不影响
                        continue
            except re.error:
                continue
        if hits == 0:
            continue
        hit_rate = hits / max(len(patterns), 1)
        scores[label] = (hit_rate, hits)

    if not scores:
        return [OTHER_LABEL]

    if not multi_label:
        # 单 label 模式（旧行为，仅供 ablation）
        best = max(
            scores.keys(),
            key=lambda lb: (
                scores[lb][0], scores[lb][1]))
        return [best]

    # 多 label 模式：返回所有命中的 label
    # 按命中率排序（好观察）
    return sorted(
        scores.keys(),
        key=lambda lb: (
            -scores[lb][0], -scores[lb][1]))


def assign_label_for_history(
    step,
    module: Module,
) -> str:
    """冻结 φ 的 categorical 映射：每个 step-level history 恰好一个 z。

    命中一个 named label 时返回该 label；未命中或规则重叠时返回
    OTHER。重叠不按 label 顺序猜测，从而避免不可审计的顺序破平。
    """
    evidence = " ".join([
        getattr(step, "action", "") or "",
        getattr(step, "external_result", "") or "",
    ])
    labels = label_trajectory_step(
        "", evidence, module, multi_label=True)
    named = list(dict.fromkeys(
        z for z in labels if z in module.named_behaviors))
    if len(named) == 1:
        return named[0]
    return OTHER_LABEL


def _assign_label_for_history_strict(step, module: Module) -> str:
    """正式路径 categorical φ：无命中为 OTHER，多命中则 ruler 非互斥。"""
    evidence = " ".join([
        getattr(step, "action", "") or "",
        getattr(step, "external_result", "") or "",
    ])
    labels = label_trajectory_step(
        "", evidence, module, multi_label=True)
    named = list(dict.fromkeys(
        z for z in labels if z in module.named_behaviors))
    if len(named) > 1:
        raise HistoryRoutingError(
            f"history matches multiple labels in {module.module_id}: "
            f"{named}")
    return named[0] if named else OTHER_LABEL


def _history_reached(step, module: Module) -> bool:
    """仅使用可见 action/result 判断 history 是否属于该 frozen group。"""
    text = " ".join([
        getattr(step, "action", "") or "",
        getattr(step, "external_result", "") or "",
    ])
    if not module.reach_patterns:
        return False
    for pattern in module.reach_patterns:
        try:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def build_history_records(
    trajectories: list,
    modules: list[Module],
    source: str,
) -> list[HistoryRecord]:
    """用 Parser LLM 的冻结映射生成零或多个 Skill-history records。

    一个 rollout step 可为多个 Skill histories 提供 evidence；但对每个
    `(rollout step, skill history)` 只能产生一个 categorical label。
    不适用于或不确定的 step 生成 UNASSIGNED audit record。映射未执行、
    缺失或含非法 frozen history ID 时 fail closed，且绝不回退 regex。
    """
    modules_by_history = {}
    for module in modules:
        history_id = module.history_id or module.module_id
        if history_id in modules_by_history:
            raise HistoryRoutingError(
                f"duplicate Skill history definition: {history_id}")
        modules_by_history[history_id] = module
    records = []
    for rollout_idx, traj in enumerate(trajectories):
        if not getattr(traj, "evaluation_valid", True):
            continue
        trajectory_id = str(getattr(
            traj, "trajectory_id",
            f"{traj.task_id}:{source}:{rollout_idx}"))
        for step_idx, step in enumerate(traj.steps):
            mapping_status = getattr(
                step, "history_mapping_status", "missing")
            mapped_ids = getattr(
                step, "applicable_history_ids", None)
            if mapped_ids is None or mapping_status == "missing":
                raise HistoryMappingUnavailableError(
                    f"missing Parser history mapping for {trajectory_id} "
                    f"step {getattr(step, 'step', step_idx)}")
            if mapping_status not in {
                    "assigned", "unassigned", "uncertain"}:
                raise HistoryMappingUnavailableError(
                    f"invalid Parser mapping status {mapping_status!r}")
            if not isinstance(mapped_ids, list):
                raise HistoryMappingUnavailableError(
                    "applicable_history_ids must be a list")
            if len(mapped_ids) != len(set(mapped_ids)):
                raise HistoryRoutingError(
                    f"duplicate mapped history ids: {mapped_ids}")
            unknown = set(mapped_ids) - set(modules_by_history)
            if unknown:
                raise HistoryMappingUnavailableError(
                    f"Parser returned unknown frozen histories: {unknown}")
            if mapping_status != "assigned" and mapped_ids:
                raise HistoryMappingUnavailableError(
                    "non-assigned mapping cannot contain history ids")
            if mapping_status == "assigned" and not mapped_ids:
                raise HistoryMappingUnavailableError(
                    "assigned mapping must contain history ids")
            matched_histories = [
                modules_by_history[history_id]
                for history_id in mapped_ids]
            if not matched_histories:
                records.append(HistoryRecord(
                    task_id=str(traj.task_id),
                    trajectory_id=trajectory_id,
                    step_index=int(getattr(step, "step", step_idx)),
                    source=source,
                    module_id=UNASSIGNED_GROUP_ID,
                    skill_id="",
                    label=UNASSIGNED_LABEL,
                    reward=float(traj.final_reward),
                    group_id=UNASSIGNED_GROUP_ID,
                    history_id=UNASSIGNED_GROUP_ID,
                ))
                continue
            seen_histories = set()
            for module in matched_histories:
                history_id = module.history_id or module.module_id
                if history_id in seen_histories:
                    raise HistoryRoutingError(
                        f"duplicate Skill history definition: {history_id}")
                seen_histories.add(history_id)
                label = _assign_label_for_history_strict(step, module)
                records.append(HistoryRecord(
                    task_id=str(traj.task_id),
                    trajectory_id=trajectory_id,
                    step_index=int(getattr(step, "step", step_idx)),
                    source=source,
                    module_id=history_id,
                    skill_id=module.skill_id,
                    label=label,
                    reward=float(traj.final_reward),
                    group_id=module.group_id,
                    history_id=history_id,
                ))
    return records


def count_record_labels(
    records: list[HistoryRecord],
    module: Module,
) -> dict[str, int]:
    counts = {z: 0 for z in get_Z_for_module(module)}
    for record in records:
        if record.module_id == module.module_id:
            counts[record.label if record.label in counts
                   else OTHER_LABEL] += 1
    return counts


def estimate_eta_from_records(
    records: list[HistoryRecord],
    module: Module,
) -> tuple[dict[str, float], dict[str, int], list[str]]:
    """不插补 zero-support；返回 unsupported labels 供整组 fail closed。"""
    Z = get_Z_for_module(module)
    rewards = {z: [] for z in Z}
    for record in records:
        if record.module_id == module.module_id:
            rewards[record.label].append(record.reward)
    counts = {z: len(values) for z, values in rewards.items()}
    unsupported = [z for z, n in counts.items() if n == 0]
    eta = {
        z: (sum(values) / len(values))
        for z, values in rewards.items() if values
    }
    return eta, counts, unsupported


def label_trajectory(
    trajectory,
    module: Module,
    multi_label: bool = True,
) -> list[str]:
    """
    对整条轨迹的每一步做 labeling。

    multi_label=True (默认)：
      返回**扁平**的 label 列表。每一步的多个 label
      都会被独立记录：
        step 1 → [header_based_lookup, none_check]
        step 2 → [range_iteration, save_workbook]
      → labels = [header_based_lookup, none_check,
                  range_iteration, save_workbook]

    ⚠ 注意：此函数**丢失了 step 索引 h**。
    若要严格按老师 §3.4 估 η(h,z)，请用
    `label_trajectory_by_step` 保留 step 结构。
    """
    labels = []
    for step in trajectory.steps:
        code = ""
        cm = re.search(
            r"```python\s*\n(.*?)```",
            step.action, re.DOTALL)
        if cm:
            code = cm.group(1)
        step_labels = label_trajectory_step(
            code, step.action, module,
            multi_label=multi_label)
        if isinstance(step_labels, list):
            labels.extend(step_labels)
        elif step_labels:
            labels.append(step_labels)
    return labels


def label_trajectory_by_step(
    trajectory,
    module: Module,
    multi_label: bool = True,
) -> list[list[str]]:
    """
    ⚠ 已弃用：此函数保留了 step 边界，但没有执行唯一的
    history→group 路由。正式路径请使用 `build_history_records()`：
    一个 action step 是一个 history，每个 history 唯一属于一个
    pooling group，并在该 group 内取得一个 categorical label。

    保留此函数只为向后兼容（比如旧 bootstrap 诊断路径）。
    """
    per_step = []
    for step in trajectory.steps:
        code = ""
        cm = re.search(
            r"```python\s*\n(.*?)```",
            step.action, re.DOTALL)
        if cm:
            code = cm.group(1)
        step_labels = label_trajectory_step(
            code, step.action, module,
            multi_label=multi_label)
        if isinstance(step_labels, list):
            per_step.append(list(dict.fromkeys(
                step_labels)))
        elif step_labels:
            per_step.append([step_labels])
        else:
            per_step.append([OTHER_LABEL])
    return per_step


def assign_label_for_group(
    trajectory,
    module: Module,
) -> str:
    """
    ⚠ Legacy compatibility only：将整条 trajectory 压缩成一个
    group label 不符合正式 history 语义。正式路径中 h 是真实 step，
    group 只是共享 λ_g 的 history pooling class。

      z_g(τ) = argmax_{z ∈ module.named_behaviors} count(z in τ)
      若 named 都未命中 → OTHER_LABEL

    这样保证归一性：Σ_{z ∈ Z_g} P(z | h=g) = 1
    （其中 Z_g = module.named_behaviors ∪ {OTHER}）

    与旧 label_trajectory 的区别：
    - 旧函数：把所有 step 的 multi-label extend 成扁平列表
      → 一条 traj 可能同时算入 6-8 个 label 的观测 →
      Σ P(z|h) 远大于 1（违反概率归一）
    - 新函数：**一条 traj = 一个 label**（primary or OTHER）→
      Σ P(z|h) = 1（老师要求）

    这里的 module 就是 Parser 定义的一个 pooling group。
    """
    # 兼容旧调用：仅当整条轨迹的所有 step-level histories 给出同一
    # named label 时返回它；否则返回 OTHER，不再使用 winner-take-all。
    labels = [assign_label_for_history(step, module)
              for step in trajectory.steps]
    named = set(z for z in labels if z != OTHER_LABEL)
    return next(iter(named)) if len(named) == 1 else OTHER_LABEL


# ─────────────────────────────────────────────────
# Step 3：P 估计（Dirichlet smoothed）
# ─────────────────────────────────────────────────

def estimate_label_distribution(
    label_counts: dict[str, int],
    Z: list[str],
    alpha: float = 1.0,
) -> dict[str, float]:
    """
    Dirichlet smoothed 估计：
    P̂(z) = (count(z) + α) / (N + α|Z|)

    α > 0 确保每个标签都有正概率
    """
    N = sum(label_counts.values())
    Z_size = len(Z)
    probs = {}
    for z in Z:
        c = label_counts.get(z, 0)
        probs[z] = (c + alpha) / (N + alpha * Z_size)
    return probs


def count_labels_for_module(
    trajectories: list,
    module: Module,
) -> dict[str, int]:
    """
    统计 module (= pooling group) 内各 label 的**独立 traj 数**。

    ⚠ 老师 2026-07-11 澄清（核心修复 #1）：
    一条 trajectory 对该 group 只贡献 **1 个互斥 label**
    （由 `assign_label_for_group` 选 primary 或 OTHER）。
    Σ_z counts[z] = N_g（独立 traj 数），保证下游
    P(z|h=g) = counts[z]/N_g 归一。

    此前用 `label_trajectory` (flat multi-label) 让一条
    traj 计入多个 label → Σ counts >> N_g → 违反概率归一。
    """
    Z = get_Z_for_module(module)
    counts = {z: 0 for z in Z}

    for traj in trajectories:
        if module.skill_id not in (
                traj.activated_skills or []):
            continue

        z = assign_label_for_group(traj, module)
        counts[z if z in counts else OTHER_LABEL] += 1

    return counts


def compute_all_distributions(
    trajectories: list,
    modules: list[Module],
    alpha: float = 1.0,
) -> dict[str, LabelDistribution]:
    """
    对所有 module 计算 P_s（Student 行为分布）
    """
    results = {}
    for mod in modules:
        Z = get_Z_for_module(mod)
        counts = count_labels_for_module(
            trajectories, mod)
        probs = estimate_label_distribution(
            counts, Z, alpha)
        results[mod.module_id] = LabelDistribution(
            module_id=mod.module_id,
            counts=counts,
            total=sum(counts.values()),
            probs=probs,
            alpha=alpha,
        )
    return results


def compute_distributions_from_records(
    records: list[HistoryRecord],
    modules: list[Module],
    alpha: float = 1.0,
) -> dict[str, LabelDistribution]:
    """三端共享的严格分布估计器；唯一输入单位为 HistoryRecord。"""
    results = {}
    for mod in modules:
        Z = get_Z_for_module(mod)
        counts = count_record_labels(records, mod)
        if sum(counts.values()) == 0:
            raise ValueError(
                f"zero reached valid records for module {mod.module_id}")
        results[mod.module_id] = LabelDistribution(
            module_id=mod.module_id,
            counts=counts,
            total=sum(counts.values()),
            probs=estimate_label_distribution(counts, Z, alpha),
            alpha=alpha,
        )
    return results


# ─────────────────────────────────────────────────
# Step 3b：Teacher 和 Reference 的分布
# ─────────────────────────────────────────────────

def estimate_teacher_distribution(
    teacher_grades: dict,
    modules: list[Module],
    alpha: float = 1.0,
    success_trajectories: list = None,
    strict_no_fallback: bool = True,
) -> dict[str, dict[str, float]]:
    """
    估计 P_T（Teacher 偏好的行为分布）

    ⚠ 方法学正确性说明：
    P_T 必须来自 Teacher 端点（Teacher rollout 或
    Teacher grading）。用"成功 Student 轨迹"当 P_T
    会让 R̂(z) 与 η(h,z) 变成同一个信号，λ 投影退化。

    strict_no_fallback=True (默认 & 推荐)：
      - 只用 b4 grading 的 teacher_preferred_action
      - b4 数据不足 → 返回均匀分布 + 打警告
      - 绝不用 success student trajectories 冒充

    strict_no_fallback=False（旧行为，保留兼容）：
      - b4 不够时用 success trajectories 补充
      - 只在 ablation 时使用
    """
    results = {}

    for mod in modules:
        Z = get_Z_for_module(mod)
        counts = {z: 0 for z in Z}
        n_b4 = 0

        # 来源 1：b4 grading（唯一合法来源）
        if teacher_grades:
            for tid, grade in teacher_grades.items():
                if not hasattr(grade, "step_grades"):
                    continue
                for sg in (grade.step_grades or []):
                    if isinstance(sg, dict):
                        impl = sg.get(
                            "implicated_skill_id", "")
                        pref = sg.get(
                            "teacher_preferred_action", "")
                    else:
                        impl = getattr(
                            sg, "implicated_skill_id", "")
                        pref = getattr(
                            sg, "teacher_preferred_action", "")
                    if impl and impl != mod.skill_id:
                        continue
                    label = _text_to_label(pref) \
                        if pref else ""
                    if label and label in counts:
                        counts[label] += 1
                        n_b4 += 1

        # 来源 2：成功轨迹补充 — 已默认禁用
        if (not strict_no_fallback
                and success_trajectories
                and n_b4 < 5):
            print(f"    ⚠ [P_T for {mod.module_id[:30]}] "
                  f"b4={n_b4}<5, USING success "
                  f"trajectories (ablation only!)")
            for traj in success_trajectories:
                if mod.skill_id not in (
                    traj.activated_skills or []):
                    continue
                for step in traj.steps:
                    z = assign_label_for_history(step, mod)
                    counts[z if z in counts else OTHER_LABEL] += 1

        # 若 counts 全 0 且 strict → 均匀分布
        if strict_no_fallback and sum(
                counts.values()) == 0:
            print(f"    ⚠ [P_T for {mod.module_id[:30]}] "
                  f"NO teacher signal available "
                  f"(b4={n_b4})，退回均匀分布")

        probs = estimate_label_distribution(
            counts, Z, alpha)
        results[mod.module_id] = probs

    return results


def estimate_reference_distribution(
    ref_trajectories: list,
    modules: list[Module],
    alpha: float = 1.0,
    ref_baseline: object = None,
    failed_trajectories: list = None,
    strict_no_fallback: bool = True,
) -> dict[str, dict[str, float]]:
    """
    估计 P_0（Reference 行为分布）

    ⚠ 方法学正确性说明：
    P_0 必须来自 no-skill Reference 端点。用"失败
    Student 轨迹"（带 skill）当 P_0 会让 R̂ 与 η 变成
    同一个信号（都基于 success vs failure 差异），
    λ 投影退化为用 η 拟合自己 → slope 恒为 1 或不稳。

    strict_no_fallback=True (默认 & 推荐)：
      1. baseline 的已保存 label 分布
      2. reference (no-skill) trajectories
      3. 均匀分布 + 警告日志
      不用 failed trajectories 冒充。

    strict_no_fallback=False（旧行为）：
      3. failed trajectories → 均匀分布
      仅供 ablation 使用。
    """
    results = {}

    for mod in modules:
        Z = get_Z_for_module(mod)

        # 来源 1：baseline 的已保存 label 分布
        if ref_baseline is not None:
            saved = None
            if hasattr(ref_baseline,
                       'get_label_distribution'):
                saved = \
                    ref_baseline.get_label_distribution(
                        mod.module_id)
            if saved and len(saved) > 0:
                results[mod.module_id] = saved
                continue

        # 来源 2：reference trajectories
        if ref_trajectories:
            counts = count_labels_for_module(
                ref_trajectories, mod)
            if sum(counts.values()) > 0:
                probs = estimate_label_distribution(
                    counts, Z, alpha)
                results[mod.module_id] = probs
                continue

        # 来源 3：失败轨迹补充 — 已默认禁用
        if (not strict_no_fallback
                and failed_trajectories):
            print(f"    ⚠ [P_0 for {mod.module_id[:30]}] "
                  f"NO ref rollout, USING failed "
                  f"trajectories (ablation only!)")
            counts = {z: 0 for z in Z}
            for traj in failed_trajectories:
                if mod.skill_id not in (
                    traj.activated_skills or []):
                    continue
                for step in traj.steps:
                    z = assign_label_for_history(step, mod)
                    counts[z if z in counts else OTHER_LABEL] += 1
            if sum(counts.values()) > 0:
                probs = estimate_label_distribution(
                    counts, Z, alpha)
                results[mod.module_id] = probs
                continue

        # 来源 4：均匀分布（strict 模式的最终 fallback）
        if strict_no_fallback:
            print(f"    ⚠ [P_0 for {mod.module_id[:30]}] "
                  f"NO ref rollout data，退回均匀分布 "
                  f"(λ 会被压回 0)")
        p = 1.0 / len(Z)
        results[mod.module_id] = {z: p for z in Z}

    return results


# ─────────────────────────────────────────────────
# 辅助：从任意轨迹列表估计 label 分布
# ─────────────────────────────────────────────────

def _estimate_from_trajectories(
    trajectories: list,
    modules: list[Module],
    alpha: float = 1.0,
) -> dict[str, dict[str, float]]:
    """
    从轨迹列表估计每个 module 的 label 分布。
    用于 Teacher rollout → P_T 和 Reference rollout → P_0。

    不检查 activated_skills：因为 teacher/reference
    rollout 不一定有 skill_id 匹配。
    """
    results = {}
    for mod in modules:
        Z = get_Z_for_module(mod)
        counts = {z: 0 for z in Z}

        for traj in trajectories:
            if not getattr(traj, "evaluation_valid", True):
                continue
            z = assign_label_for_group(traj, mod)
            counts[z if z in counts else OTHER_LABEL] += 1

        probs = estimate_label_distribution(
            counts, Z, alpha)
        results[mod.module_id] = probs

    return results


# ─────────────────────────────────────────────────
# Step 3c：R̂ 计算（label-level contrast）
# ─────────────────────────────────────────────────

def compute_label_contrast(
    P_T: dict[str, float],
    P_0: dict[str, float],
) -> dict[str, float]:
    """
    R̂(z) = log P̂_T(z) - log P̂_0(z)

    R̂ > 0：Teacher 更偏好这个行为
    R̂ < 0：Teacher 更避免这个行为
    """
    R_hat = {}
    for z in P_T:
        p_t = max(P_T.get(z, 1e-6), 1e-6)
        p_0 = max(P_0.get(z, 1e-6), 1e-6)
        R_hat[z] = math.log(p_t / p_0)
    return R_hat


# ─────────────────────────────────────────────────
# Step 4：估 m（module-level history weight）
# ─────────────────────────────────────────────────

def estimate_m(
    module: Module,
    trajectories: list,
    diagnoses_map: dict,
    homogeneity_override: float = None,
) -> float:
    """
    修复：homogeneity 可从外部传入（P_s 估计）
    """
    n_total = max(len(trajectories), 1)

    n_relevant = sum(
        1 for t in trajectories
        if module.skill_id in (
            t.activated_skills or []))
    coverage = min(n_relevant / n_total, 1.0)

    defect_count = 0
    lapse_count = 0
    for t in trajectories:
        if module.skill_id not in (
            t.activated_skills or []):
            continue
        diag = diagnoses_map.get(t.task_id, {})
        ft = diag.get("failure_type", "")
        if ft == "skill_defect":
            defect_count += 1
        elif ft == "execution_lapse":
            lapse_count += 1
    total_diag = max(
        defect_count + lapse_count, 1)
    stability = max(
        defect_count, lapse_count) / total_diag

    n_fail = sum(
        1 for t in trajectories
        if module.skill_id in (
            t.activated_skills or [])
        and not t.success)
    relevance = n_fail / max(n_relevant, 1)

    # 修复：从外部传入或用默认值
    homogeneity = (homogeneity_override
                   if homogeneity_override is not None
                   else 0.5)

    score = (coverage * stability * relevance
             * max(homogeneity, 0.1))
    m = max(0.5, min(2.0, 0.5 + 1.5 * score))

    return m


# ─────────────────────────────────────────────────
# Step 5：估 λ（behavior-level trust gate）
# ─────────────────────────────────────────────────

def estimate_lambda(
    R_hat: dict[str, float],
    module: Module,
    P_s: dict[str, float],
    P_T: dict[str, float],
    n_samples: int,
) -> dict[str, float]:
    """
    λ(z) = clip(1 + 0.5 × T(z), 0.5, 1.6)

    T(z) = trust score，综合以下因子：
      - contrast 大小（|R̂| 越大越可信）
      - 样本量（n 越大越可信）
      - Teacher/Student 差异的一致性
    """
    lambda_ = {}

    for z in R_hat:
        if z == OTHER_LABEL:
            # other 标签保持默认
            lambda_[z] = 1.0
            continue

        if z not in module.named_behaviors:
            lambda_[z] = 1.0
            continue

        # contrast 大小
        abs_r = abs(R_hat.get(z, 0))
        contrast_trust = min(abs_r / 2.0, 1.0)

        # 样本量
        sample_trust = min(n_samples / 50.0, 1.0)

        # P_T 和 P_s 差异
        diff = abs(
            P_T.get(z, 0) - P_s.get(z, 0))
        diff_trust = min(diff / 0.3, 1.0)

        T = contrast_trust * 0.4 \
            + sample_trust * 0.3 \
            + diff_trust * 0.3

        lambda_[z] = max(0.5, min(1.6,
            1.0 + 0.5 * T))

    return lambda_


# ─────────────────────────────────────────────────
# NEW: Reward-calibrated λ estimation
# 对应 lambda_estimation_selfcontained_cn
# ─────────────────────────────────────────────────

def estimate_eta(
    trajectories: list,
    module: "Module",
    Z: list[str],
    mode: str = "occurrence_weighted",
) -> dict[str, float]:
    """
    ⚠ 已弃用（2026-07-11 老师澄清）：
    此函数用 flat multi-label（一条 traj 计入多个 label
    的观测），违反 Σ_z P(z|h=g) = 1 归一性。
    **主路径请用 `estimate_eta_step_aware`**（互斥 label）。
    保留此函数仅供 legacy 对照（contrastive / presence /
    primary_only 三种旧模式）。

    η(h,z) = E[Y | h, z]
    局部成功率：group 内每个 label 对应的轨迹平均 reward。

    与"每条轨迹只贡献主 label"的旧实现相比，本函数默认使用
    **occurrence_weighted**：一条轨迹里每次 label z 出现都
    贡献一次 (Y)，然后除以总出现次数。这样每个 label 都能
    由数据估计出来，而不是退回 global_mean。

    Modes
    -----
    occurrence_weighted  (默认，推荐)
        η(z) = Σ_i n_i(z) · Y_i / Σ_i n_i(z)
        其中 n_i(z) 是 label z 在轨迹 i 中的出现次数。
        直觉：z 的每一次出现都是一次 "z 参与了这次执行"
        的观测，理应得到该轨迹 reward 的归因份额。

    presence_weighted
        η(z) = mean{ Y_i : z ∈ labels(τ_i) }
        直觉：轨迹里只要出现过 z 就算一次观测，不重复计入。

    primary_only  (原实现，仅保留供 ablation)
        η(z) = mean{ Y_i : primary_label(τ_i) = z }
        每条轨迹只算给主 label。

    Fallback
    --------
    某个 label 没有任何轨迹支撑时，退回 group 内的
    global_mean，避免 P_+ 计算出现除零。
    """
    if mode not in ("occurrence_weighted",
                    "presence_weighted",
                    "primary_only"):
        mode = "occurrence_weighted"

    label_rewards: dict[str, list[float]] = {
        z: [] for z in Z}
    # 用来算 group 内的 global_mean（仅作 fallback）
    all_traj_rewards: list[float] = []

    for traj in trajectories:
        if module.skill_id not in (
            traj.activated_skills or []):
            continue

        reward = float(traj.final_reward)
        labels = label_trajectory(traj, module)
        all_traj_rewards.append(reward)

        if not labels:
            # 没匹配到任何 named behavior → 归到 other
            label_rewards[OTHER_LABEL].append(reward)
            continue

        # 统计每个 label 出现次数
        counts: dict[str, int] = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1

        if mode == "primary_only":
            # 每条轨迹只算给主 label
            primary = max(counts, key=counts.get)
            key = (primary
                   if primary in label_rewards
                   else OTHER_LABEL)
            label_rewards[key].append(reward)

        elif mode == "presence_weighted":
            # 出现即一次观测（不重复计入次数）
            for lab in counts:
                key = (lab if lab in label_rewards
                       else OTHER_LABEL)
                label_rewards[key].append(reward)

        else:  # occurrence_weighted
            # 每次出现都是一次观测，重复计入
            for lab, n in counts.items():
                key = (lab if lab in label_rewards
                       else OTHER_LABEL)
                label_rewards[key].extend(
                    [reward] * n)

    # group 内的 global_mean，只当某 label 无数据时用
    global_mean = (
        sum(all_traj_rewards) / len(all_traj_rewards)
        if all_traj_rewards else 0.5)

    eta = {}
    for z in Z:
        rewards = label_rewards.get(z, [])
        if rewards:
            eta[z] = sum(rewards) / len(rewards)
        else:
            eta[z] = global_mean

    return eta


def estimate_eta_contrastive(
    trajectories: list,
    module: "Module",
    Z: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    正例-负例对比估计：同时算 η(z) 和 η(¬z)。

    问题背景（2026-07-10）：
      occurrence_weighted 模式下，如果 multi-label φ 让
      几乎每条轨迹都命中 6-8 个 label，那么每个 label 的
      η(z) 都≈全局 success rate，η 之间几乎没差别 →
      a_+ 各项极小 → λ 分子接近 0 → shrinkage 归零。

    解决方案：**对比正例（出现 z）与负例（未出现 z）的
    平均 reward**。稀少但每次都成功的 label（比如
    formula_value_read: obs=4, all success）会得到极大
    a_+；普遍出现但对成功无差别的 label（口癖）a_+≈0。

    返回
    ----
    (eta_pos, eta_neg) 两个 dict:
      eta_pos[z] = mean(Y_i | z 在 τ_i 中出现)
      eta_neg[z] = mean(Y_i | z 未在 τ_i 中出现)
    """
    pos_rewards: dict[str, list[float]] = {
        z: [] for z in Z}
    neg_rewards: dict[str, list[float]] = {
        z: [] for z in Z}
    all_traj_rewards: list[float] = []

    for traj in trajectories:
        if module.skill_id not in (
            traj.activated_skills or []):
            continue

        reward = float(traj.final_reward)
        labels = label_trajectory(traj, module)
        all_traj_rewards.append(reward)

        # 该轨迹里"出现过"的 label 集合
        present = set(labels) if labels else set()
        # 若没匹配任何 named → 视为只有 OTHER 出现
        if not present:
            present = {OTHER_LABEL}

        for z in Z:
            if z in present:
                pos_rewards[z].append(reward)
            else:
                neg_rewards[z].append(reward)

    global_mean = (
        sum(all_traj_rewards) / len(all_traj_rewards)
        if all_traj_rewards else 0.5)

    eta_pos = {}
    eta_neg = {}
    for z in Z:
        pos = pos_rewards.get(z, [])
        neg = neg_rewards.get(z, [])
        eta_pos[z] = (
            sum(pos) / len(pos)
            if pos else global_mean)
        eta_neg[z] = (
            sum(neg) / len(neg)
            if neg else global_mean)

    return eta_pos, eta_neg


def compute_a_plus_contrastive(
    eta_pos: dict[str, float],
    eta_neg: dict[str, float],
    eps: float = 1e-3,
) -> dict[str, float]:
    """
    对比式 a_+：a_+(z) = log(η_pos(z) / η_neg(z))

    直觉：
      - 稀少但每次都成功的 label（formula_value_read）：
        η_pos 高，η_neg 接近全局平均 → a_+ 很大正
      - 到处都是且成功率跟全局差不多的 label（口癖）：
        η_pos ≈ η_neg → a_+ ≈ 0（正确地识别为口癖）
      - 出现就失败的 label：η_pos 低，η_neg 高 → a_+ 大负

    eps 是数值下限，防止 log(0)。
    """
    a = {}
    for z in eta_pos:
        pp = max(eta_pos.get(z, eps), eps)
        pn = max(eta_neg.get(z, eps), eps)
        a[z] = math.log(pp / pn)
    return a


# ─────────────────────────────────────────────────
# η(h,z) 严格 step-aware 估计（2026-07-10 audit #1）
# ─────────────────────────────────────────────────

def estimate_eta_step_aware(
    trajectories: list,
    module: "Module",
    Z: list[str],
) -> tuple[dict[str, float], dict[str, int]]:
    """
    ⚠ 老师 2026-07-11 澄清（核心修复 #1/#2）：
    h = "skill 中一个 pooling group" (= module)，不是 action step。
    一条 trajectory 对该 group 产生**恰好一个互斥 label**：
      z_g(τ) = assign_label_for_group(τ, module)

    η(h=g, z) = Σ_i 1{z_g(τ_i)=z} · Y_i / Σ_i 1{z_g(τ_i)=z}
              = 那些"group=g 时被打上 label=z"的 traj 的平均 reward

    归一性保障（老师要求）：
      P(z | h=g) = |{i: z_g(τ_i)=z}| / N_g
      Σ_{z ∈ Z_g} P(z | h=g) = 1

    这与旧 flat-multi-label 估计的本质区别：
    - 旧：一条 traj 同时算入 6-8 个 label 的观测 → Σ P > 1
    - 新：一条 traj 只算 1 个 label 的观测 → Σ P = 1 ✅

    返回
    ----
    (eta_dict, obs_counts):
      eta_dict[z]   = group=g 且 label=z 的 trajs 的平均 reward
                      （无观测 → global_mean fallback）
      obs_counts[z] = 该 label 在本组的**独立 traj 数** N_{g,z}
                      Σ_z obs_counts[z] = N_g（独立轨迹数）
    """
    label_rewards: dict[str, list[float]] = {
        z: [] for z in Z}
    all_traj_rewards: list[float] = []

    for traj in trajectories:
        if not getattr(traj, "evaluation_valid", True):
            continue
        reward = float(traj.final_reward)
        all_traj_rewards.append(reward)
        z = assign_label_for_group(traj, module)
        label_rewards[z if z in Z else OTHER_LABEL].append(reward)

    global_mean = (
        sum(all_traj_rewards) / len(all_traj_rewards)
        if all_traj_rewards else 0.5)

    eta: dict[str, float] = {}
    obs_counts: dict[str, int] = {}
    for z in Z:
        rewards = label_rewards.get(z, [])
        if rewards:
            eta[z] = sum(rewards) / len(rewards)
        else:
            # 无 traj 落到该 label → 该 label 无本地成功率证据
            # 用 group 全局均值作 fallback（不引入偏见）
            eta[z] = global_mean
        obs_counts[z] = len(rewards)

    return eta, obs_counts


def compute_P_plus(
    P_0: dict[str, float],
    eta: dict[str, float],
) -> dict[str, float]:
    """
    成功加权目标：
    P_+(z|h) = P_0(z|h) × η(h,z) / Σ P_0(z') × η(h,z')

    一个行为在 P_+ 里的份额 = 它在 P_0 的份额 × 它的成功率
    """
    unnorm = {}
    for z in P_0:
        p0 = max(P_0.get(z, 1e-8), 1e-8)
        e  = max(eta.get(z, 0.5), 1e-8)
        unnorm[z] = p0 * e

    total = sum(unnorm.values())
    if total <= 0:
        n = len(P_0)
        return {z: 1.0 / n for z in P_0}

    return {z: v / total for z, v in unnorm.items()}


def compute_a_plus(
    P_plus: dict[str, float],
    P_0: dict[str, float],
) -> dict[str, float]:
    """
    有用度倾斜：
    a_+(h,z) = log(P_+(z|h) / P_0(z|h))
             = log η(h,z) − log E_{P_0}[η]

    a_+ > 0：成功率高于 P_0 下平均，应往这个方向推
    a_+ < 0：成功率低于平均
    a_+ ≈ 0：无信号（包括 teacher 口癖）
    """
    a = {}
    for z in P_plus:
        pp = max(P_plus.get(z, 1e-8), 1e-8)
        p0 = max(P_0.get(z, 1e-8), 1e-8)
        a[z] = math.log(pp / p0)
    return a


def estimate_lambda_exact(
    P_plus: dict[str, float],
    P_0: dict[str, float],
    r_T: dict[str, float],
    named_behaviors: list[str],
    lambda_neutral: float = 0.0,
    n_trajectories: int = None,
    tau_sq_default: float = 0.25,
    obs_counts: dict = None,
    apply_shrinkage: bool = True,
    apply_negative_check: bool = True,
    s_sq_override: float = None,
) -> float:
    """
    精确求解：λ_g* = argmin_ℓ KL(P_+ ‖ q_ℓ)
      q_ℓ(z) ∝ P_0(z) · exp{ℓ · r_T(z)}

    ⚠ 老师 audit #4（2026-07-10）：
    与主方法（estimate_lambda_projection）**唯一区别**是把
    WLS 小倾斜近似换成 1D 精确 KL 优化。其他一切保持相同：
    - 使用**完整 P_0 支持集**（含 OTHER）作分布归一
    - 应用相同的 shrinkage（`apply_shrinkage=True`）
    - 应用相同的负 λ 显著性检查（`apply_negative_check=True`）
    - 硬 clip 到 [-1.5, 1.5]（与主方法一致）

    这样 exact vs WLS 的 ablation 可以严格归因于"近似 vs 精确"，
    而非"标签集合改变"或"有限样本处理改变"。

    参数
    ----
    P_plus, P_0, r_T : **完整字典**（含 OTHER），归一化和期望
        都在 full support 上做。
    named_behaviors : 只用于 fallback 判定（少于 2 个 named
        label 时退回 neutral）。**不用于限制 KL 优化的支持集**。
    n_trajectories : shrinkage 用的独立轨迹数。传 None 会跳过
        shrinkage（仅供 ablation `apply_shrinkage=False` 使用）。
    apply_shrinkage : 是否应用有限样本收缩（默认 True）。
    apply_negative_check : λ < 0 时是否检查显著性（默认 True）。

    1D 凸优化：
      d/dℓ KL(P_+ ‖ q_ℓ) = E_{q_ℓ}[r_T] − E_{P_+}[r_T] = 0
      → ℓ* 使得 q_ℓ 在 r_T 方向上的均值等于 P_+ 在同方向均值。
    Newton-Raphson 求解，二阶导 = Var_{q_ℓ}[r_T]。
    """
    # 使用**完整字典**（含 OTHER）作 support，与主方法一致
    all_z = [z for z in P_0 if z in r_T]
    # named 数只用于判定是否有足够信号
    named_in_z = [
        z for z in named_behaviors if z in P_0]
    if len(named_in_z) < 2 or len(all_z) < 2:
        return lambda_neutral

    # E_{P_+}[r_T]（**在完整 z 空间**上）
    P_plus_sum = sum(
        P_plus.get(z, 0) for z in all_z)
    if P_plus_sum < 1e-8:
        return lambda_neutral
    E_pplus = sum(
        P_plus.get(z, 0) * r_T.get(z, 0)
        for z in all_z) / P_plus_sum

    def _gradient(ell_value: float) -> tuple[float, float]:
        """返回 KL 一阶导和 Var_q；log-sum-exp 保证数值稳定。"""
        log_unnorm = {
            z: math.log(max(P_0.get(z, 1e-8), 1e-8))
            + ell_value * r_T.get(z, 0)
            for z in all_z}
        log_max = max(log_unnorm.values())
        log_Z = log_max + math.log(sum(
            math.exp(v - log_max)
            for v in log_unnorm.values()))
        q = {z: math.exp(v - log_Z)
             for z, v in log_unnorm.items()}

        # E_{q_ℓ}[r_T]（**在完整 z 空间**上）
        E_q = sum(q[z] * r_T.get(z, 0)
                  for z in all_z)
        deriv = E_q - E_pplus
        Var_q = sum(
            q[z] * (r_T.get(z, 0) - E_q) ** 2
            for z in all_z)
        return deriv, Var_q

    # 凸目标的一阶导单调递增。自适应扩张区间后用二分求唯一根，
    # 不施加与 information projection 定义无关的 lambda clip。
    lo, hi = -1.0, 1.0
    grad_lo, _ = _gradient(lo)
    grad_hi, _ = _gradient(hi)
    for _ in range(60):
        if grad_lo <= 0 <= grad_hi:
            break
        if grad_lo > 0:
            hi, grad_hi = lo, grad_lo
            lo *= 2.0
            grad_lo, _ = _gradient(lo)
        else:
            lo, grad_lo = hi, grad_hi
            hi *= 2.0
            grad_hi, _ = _gradient(hi)
    else:
        raise RuntimeError("failed to bracket exact lambda projection")

    ell = 0.0
    Var_q = 0.0
    for _ in range(100):
        ell = (lo + hi) / 2.0
        deriv, Var_q = _gradient(ell)
        if abs(deriv) <= 1e-10 or (hi - lo) <= 1e-10:
            break
        if deriv < 0:
            lo = ell
        else:
            hi = ell
    final_deriv, Var_q = _gradient(ell)
    if abs(final_deriv) > 1e-7:
        raise RuntimeError(
            f"exact lambda projection did not converge: {final_deriv}")

    lambda_raw = float(ell)

    # ── 应用与主方法一致的 shrinkage ──
    if apply_shrinkage and n_trajectories is not None \
            and n_trajectories > 0:
        # ⚠ 老师 audit 补充 #3：优先用 task-level bootstrap
        # 得到的真 s²。若上游传了 s_sq_override 就用它，
        # 否则 fallback 到 exact KL 的 Fisher information
        # 近似（asymptotic，不真正随 n_traj 变化）。
        if s_sq_override is not None \
                and s_sq_override > 0:
            s_sq = float(s_sq_override)
        else:
            # SE 的近似：exact KL 的 Fisher information 在
            # 最优点 = Var_q[r_T]（因 Var_q 是二阶导）。
            # 有效样本数 = n_trajectories。
            n_eff = int(n_trajectories)
            s_sq = 1.0 / (n_eff * max(Var_q, 1e-4))
        s_sq = max(s_sq, 1e-4)

        tau_sq = tau_sq_default
        B = tau_sq / (tau_sq + s_sq)
        lambda_shrunk = (
            B * lambda_raw
            + (1.0 - B) * lambda_neutral)

        # 负 λ 显著性检查（与主方法一致）
        if apply_negative_check and lambda_shrunk < 0:
            se = math.sqrt(s_sq) if s_sq > 0 else 1.0
            if abs(lambda_raw) < 2.0 * se:
                lambda_shrunk = 0.0

        return lambda_shrunk

    # 未应用 shrinkage（ablation-only 模式）
    return lambda_raw


def estimate_group_lambda_from_histories(
    histories: list[dict],
    lambda_neutral: float = 0.0,
    use_exact: bool = True,
) -> float:
    """在 history 条件对象上联合估计一个共享 λ_g。

    每项包含独立的 P_0(·|h)、r_T(h,·)、a_+(h,·)、obs_counts。
    不合并不同 Z_h 的 label counts；只汇总各 history 对同一 scalar
    λ_g 的目标/score contribution。
    """
    usable = [h for h in histories if h.get("weight", 0.0) > 0]
    if not usable:
        return lambda_neutral
    total_weight = sum(h["weight"] for h in usable)

    if not use_exact:
        numerator = 0.0
        denominator = 0.0
        for history in usable:
            support = list(history["R_hat"])
            counts = history["obs_counts"]
            n = sum(max(counts.get(z, 0), 0) for z in support)
            if n <= 0:
                continue
            weights = {z: counts.get(z, 0) / n for z in support}
            r_mean = sum(weights[z] * history["R_hat"][z]
                         for z in support)
            a_mean = sum(weights[z] * history["a_plus"].get(z, 0.0)
                         for z in support)
            rho = history["weight"] / total_weight
            for z in support:
                r = history["R_hat"][z] - r_mean
                a = history["a_plus"].get(z, 0.0) - a_mean
                numerator += rho * weights[z] * r * a
                denominator += rho * weights[z] * r * r
        return (numerator / denominator
                if denominator > 1e-12 else lambda_neutral)

    def group_gradient(ell: float) -> float:
        gradient = 0.0
        for history in usable:
            support = list(history["P_0"])
            p0 = history["P_0"]
            r_t = history["R_hat"]
            a_plus = history["a_plus"]
            log_pplus = {
                z: math.log(max(p0[z], 1e-12)) + a_plus.get(z, 0.0)
                for z in support}
            max_pp = max(log_pplus.values())
            pp_norm = sum(math.exp(v - max_pp) for v in log_pplus.values())
            p_plus = {z: math.exp(v - max_pp) / pp_norm
                      for z, v in log_pplus.items()}
            log_q = {z: math.log(max(p0[z], 1e-12)) + ell * r_t[z]
                     for z in support}
            max_q = max(log_q.values())
            q_norm = sum(math.exp(v - max_q) for v in log_q.values())
            q = {z: math.exp(v - max_q) / q_norm
                 for z, v in log_q.items()}
            grad_h = sum((q[z] - p_plus[z]) * r_t[z] for z in support)
            gradient += history["weight"] / total_weight * grad_h
        return gradient

    lo, hi = -1.0, 1.0
    for _ in range(60):
        if group_gradient(lo) <= 0 <= group_gradient(hi):
            break
        lo *= 2.0
        hi *= 2.0
    else:
        raise RuntimeError("failed to bracket pooled group lambda")
    for _ in range(100):
        mid = (lo + hi) / 2.0
        grad = group_gradient(mid)
        if abs(grad) <= 1e-10:
            return mid
        if grad < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ─────────────────────────────────────────────────
# Task-level bootstrap：真实 s² 估计
# ─────────────────────────────────────────────────

def estimate_slope_variance_bootstrap(
    trajectories: list,
    module: "Module",
    Z: list[str],
    R_hat: dict[str, float],
    named_behaviors: list[str],
    n_bootstrap: int = 500,
    seed: int = 20260710,
    eta_mode: str = "occurrence_weighted",
) -> dict:
    """
    ⚠ 老师 audit（2026-07-10 补充 #3）：
    Task-level (rollout-level) bootstrap 估计 λ_g 的斜率方差 s²。

    为什么需要
    ----------
    Analytic closed-form  s² = σ_resid² / (n_eff · denom)
    是渐近形式，实际上：
      - σ_resid² 只是 label 空间上的加权残差方差
      - denom 是 label 空间上的 Σw·r̃²
      - n_eff 只作除数，但 σ_resid 本身不真正随 n_traj 变化
    → 60 个 tasks vs 20 个 tasks 得到相似的 s²，违反验收标准。

    正确做法：**按 task 重采样（with replacement）**，每次
    完整重估 η(h,z) → P_+ → a_+ → 得到 λ̂*_b。用 B 次
    bootstrap 得到 λ̂*_1, ..., λ̂*_B 的经验方差 = 真 s²。

    此 s² **自然**随 n_traj 增加而下降（bootstrap 有限样本方差
    的经典性质 O(1/n)），且不依赖于渐近正态假设。

    实现细节
    --------
    - 只重采样 module 相关轨迹（`skill_id ∈ activated_skills`）
    - 每 bootstrap 副本 → 重跑 estimate_eta → compute_P_plus
      → compute_a_plus → analytic WLS λ_raw（**不做 shrinkage**，
      因为 shrinkage 依赖 s²，会循环）
    - 报告 λ̂*_b 的样本方差 = s²
    - **P_0 / R_hat 保持冻结**（因为它们来自 Reference 和
      Teacher rollout，与 Student rollout 独立；bootstrap
      只是想估 Student rollout 的采样方差）

    返回
    ----
    dict:
      "s_sq"       : λ̂ 的 bootstrap 方差（用于 shrinkage）
      "lambda_boot": B 个 bootstrap λ̂* 列表（供诊断）
      "n_boot"     : 实际 bootstrap 次数
      "n_traj"     : 输入的独立轨迹数
    """
    # 过滤 module 相关轨迹
    relevant = [
        t for t in trajectories
        if module.skill_id in (
            t.activated_skills or [])]
    n_traj = len(relevant)
    if n_traj < 3:
        # 样本太少 → 高 s²（保守）
        return {
            "s_sq": None,
            "lambda_boot": [],
            "n_boot": 0,
            "n_traj": n_traj,
        }

    named_in_z = [
        z for z in named_behaviors if z in R_hat]
    if len(named_in_z) < 3:
        return {
            "s_sq": None,
            "lambda_boot": [],
            "n_boot": 0,
            "n_traj": n_traj,
        }

    import numpy as _np
    rng = _np.random.default_rng(seed)

    lambda_boot = []
    for _ in range(n_bootstrap):
        # with-replacement sample of trajectory indices
        idx = rng.integers(0, n_traj, size=n_traj)
        boot_trajs = [relevant[i] for i in idx]

        # 重估 η（step-aware） → P_+ → a_+
        if eta_mode == "occurrence_weighted":
            eta_b, obs_b = \
                estimate_eta_step_aware(
                    boot_trajs, module, Z)
        else:
            eta_b = estimate_eta(
                boot_trajs, module, Z,
                mode=eta_mode)
            obs_b = count_labels_for_module(
                boot_trajs, module)

        # 需要 P_0 → 用一个中性 uniform 或传进来的
        # 但为了不破坏封装，这里用 obs_b 归一化作 P_0 proxy
        # ⚠ 关键：正确 bootstrap 应使用**主流程冻结的 P_0**
        # 我们把 P_0 冻结传入（见下面 wrapper 用法）
        # 这里我们用一个占位：
        p0_total = sum(
            max(obs_b.get(z, 1), 1)
            for z in named_in_z)
        P_0_b = {z: max(obs_b.get(z, 1), 1)
                 / p0_total
                 for z in named_in_z}

        # P_+ = P_0 · η / Σ
        unnorm = {
            z: P_0_b.get(z, 1e-8)
            * eta_b.get(z, 0.5)
            for z in named_in_z}
        s = sum(unnorm.values())
        if s < 1e-8:
            continue
        P_plus_b = {z: v/s for z, v in
                    unnorm.items()}

        # a_+ = log(P_+ / P_0)
        a_plus_b = {}
        for z in named_in_z:
            pp = max(P_plus_b.get(z, 1e-8), 1e-8)
            p0v = max(P_0_b.get(z, 1e-8), 1e-8)
            a_plus_b[z] = math.log(pp / p0v)

        # WLS raw slope（不做 shrinkage）
        total_n = sum(
            max(obs_b.get(z, 0), 0)
            for z in named_in_z)
        if total_n <= 0:
            continue
        w_b = {z: obs_b.get(z, 0) / total_n
               for z in named_in_z}
        r_mean = sum(
            w_b[z] * R_hat.get(z, 0)
            for z in named_in_z)
        a_mean = sum(
            w_b[z] * a_plus_b.get(z, 0)
            for z in named_in_z)

        numer = 0.0
        denom = 0.0
        for z in named_in_z:
            r_t = R_hat.get(z, 0) - r_mean
            a_t = a_plus_b.get(z, 0) - a_mean
            numer += w_b[z] * r_t * a_t
            denom += w_b[z] * r_t * r_t

        if denom < 1e-8:
            continue

        lam_b = numer / (denom + 1e-4)
        # 硬 clip 与主方法一致
        lam_b = max(-2.5, min(2.5, lam_b))
        lambda_boot.append(lam_b)

    if len(lambda_boot) < 10:
        return {
            "s_sq": None,
            "lambda_boot": lambda_boot,
            "n_boot": len(lambda_boot),
            "n_traj": n_traj,
        }

    # s² = 样本方差
    import numpy as _np
    arr = _np.array(lambda_boot, dtype=float)
    s_sq = float(_np.var(arr, ddof=1))

    return {
        "s_sq": s_sq,
        "lambda_boot": lambda_boot,
        "n_boot": len(lambda_boot),
        "n_traj": n_traj,
        "boot_mean": float(_np.mean(arr)),
        "boot_std": float(_np.std(arr, ddof=1)),
    }


def estimate_lambda_projection(
    R_hat: dict[str, float],
    a_plus: dict[str, float],
    P_0: dict[str, float],
    named_behaviors: list[str],
    n_samples: int,
    lambda_neutral: float = 0.0,
    tau_sq_default: float = 0.25,
    obs_counts: dict[str, int] = None,
    n_trajectories: int = None,
    s_sq_override: float = None,
    apply_shrinkage: bool = True,
) -> dict[str, float]:
    """
    Reward-calibrated λ：将 a_+ 投影到 r_T 方向

    小倾斜近似（WLS 斜率）：
    λ_g ≈ Σ w_h · r̃_T · ã_+ / Σ w_h · r̃_T²

    其中：
        w_h = n_h / Σ_{h' in group} n_{h'}
    n_h 是 label h 在 group 内的**观测次数**
    （老师公式：每个 history 的权重 = 它的样本占比）

    然后收缩：
    λ_shrunk = B · λ + (1-B) · λ_neutral
    B = τ² / (τ² + s²)

    参数
    ----
    obs_counts : {label: count} 观测次数字典
        用于 w_h 加权；缺失时用 P_0 fallback
    n_samples : label 观测总数（Σ n_h），供向后兼容
    n_trajectories : ⚠ 关键（audit issue #3）
        **独立轨迹数**（比如 60 个 dev tasks 就传 60）。
        用于计算 s_g² 时的有效样本量。**不是 label 出现
        总数**——若一条轨迹命中 5 个 label，label 总数会
        是 5 但独立样本仍是 1。用 label 总数会大幅高估
        n → s² 偏小 → shrinkage 过弱。
        默认 None 时退回 n_samples（旧行为，会高估 n）。

    返回 label-level 的 λ 字典（group 内共用同一个 λ_g）
    """
    # ── 1. 中心化 + 观测频率权重（在**完整 Z 上**） ──
    # ⚠ 老师 2026-07-11 澄清（核心修复 #2）：
    #   之前 WLS 排除 OTHER，只在 named_behaviors 上做
    #   中心化和加权，会导致 λ 方向与 exact projection
    #   不一致，甚至方向相反（合成核对：exact=+0.25,
    #   排除 OTHER 的 WLS=-0.42）。
    #
    #   OTHER 是 Z 的正式类别，占据概率质量，会参与
    #   归一和中心化。删除 OTHER 会改变其他 label 的
    #   相对关系，不再是"完整 KL projection 的局部近似"。
    #
    #   修复：WLS 与 exact 使用**完全相同的支持集** = 所有
    #   `R_hat.keys()`（含 OTHER）。唯一区别应是"WLS
    #   一阶近似 vs 精确 KL argmin"。
    #
    # 权重 w_h 用观测频率 n_h / Σ n（老师公式）
    all_z = list(R_hat.keys())

    # 判定：需要至少 3 个有观测的 label 才做 WLS
    # （dof = k - 2）
    if len(all_z) < 3:
        lambda_shrunk = lambda_neutral
    else:
        # ── 计算 w_h = n_h / Σ n_{h'}（完整 Z 上）──
        w_h = {}
        if obs_counts:
            total_n = sum(
                max(obs_counts.get(z, 0), 0)
                for z in all_z)
            if total_n > 0:
                for z in all_z:
                    w_h[z] = (
                        obs_counts.get(z, 0)
                        / total_n)
            else:
                # obs_counts 全 0 → 用 P_0 fallback
                total_p0 = sum(
                    max(P_0.get(z, 1e-8), 1e-8)
                    for z in all_z)
                if total_p0 < 1e-8:
                    total_p0 = 1e-8
                for z in all_z:
                    w_h[z] = (
                        max(P_0.get(z, 1e-8), 1e-8)
                        / total_p0)
        else:
            # 没传 obs_counts → 用 P_0 fallback
            total_p0 = sum(
                max(P_0.get(z, 1e-8), 1e-8)
                for z in all_z)
            if total_p0 < 1e-8:
                total_p0 = 1e-8
            for z in all_z:
                w_h[z] = (
                    max(P_0.get(z, 1e-8), 1e-8)
                    / total_p0)

        # 完整 Z 上的 w_h 加权均值
        r_mean = sum(
            w_h[z] * R_hat.get(z, 0)
            for z in all_z)
        a_mean = sum(
            w_h[z] * a_plus.get(z, 0)
            for z in all_z)

        # ── 2. 检测：r_T 是否是常数？ ─────
        # 如果 Z 内所有 R_hat 都相同，Teacher 无可分辨
        # 偏好方向 → λ 必为 0
        r_values = [R_hat.get(z, 0) for z in all_z]
        r_range = max(r_values) - min(r_values)
        insufficient_dof = len(all_z) < 3
        R_CONSTANT_EPS = 1e-2

        if r_range < R_CONSTANT_EPS or insufficient_dof:
            lambda_shrunk = lambda_neutral
        else:
            # ── 3. WLS 斜率（在完整 Z 上） ──────
            numer = 0.0
            denom = 0.0
            n_labels_active = 0

            for z in all_z:
                r_tilde = R_hat.get(z, 0) - r_mean
                a_tilde = (a_plus.get(z, 0)
                           - a_mean)
                w = w_h[z]

                if obs_counts is not None:
                    if obs_counts.get(z, 0) > 0:
                        n_labels_active += 1
                elif w > 1e-3:
                    n_labels_active += 1

                numer += w * r_tilde * a_tilde
                denom += w * r_tilde * r_tilde

            if denom <= 1e-12:
                lambda_raw = lambda_neutral
            else:
                lambda_raw = numer / denom

            # 记 clip 前的 lambda 用于算 residual
            lambda_pre_clip = lambda_raw

            # ── 5. 收缩（老师 audit issue #3）─
            #
            # 分析（2026-07-10 修订）：
            #   之前公式 s² = σ² / (n_obs · denom) 用
            #   n_obs = Σ n_h（label 观测总数），但一条
            #   trajectory 命中多个 label 时，这些 label
            #   观测**不独立**（都来自同一条轨迹）→ 用
            #   总观测数高估 effective sample size →
            #   s² 太小 → shrinkage 过弱。
            #
            # 修正：用 **独立轨迹数** n_eff（若上层传入
            #   n_trajectories 就用它，否则 fallback 到
            #   n_samples 的启发式估计）。
            #
            # WLS slope 方差公式（1D，权重归一化 Σw=1）：
            #
            #   Var(λ̂) = σ_resid² / (n_eff · denom)
            #
            # σ_resid² 用 clip 前的 lambda_pre_clip 算，
            # 避免 clip 后 residual 被人为压小。
            # 自由度 k-2（intercept from centering + slope）。

            # 有效独立样本数
            if n_trajectories is not None \
                    and n_trajectories > 0:
                n_eff = int(n_trajectories)
            else:
                # Fallback：用 obs_counts 求和作独立轨迹数
                # （互斥 label 下 Σ obs = N_g）
                if obs_counts and len(all_z):
                    total_obs = sum(
                        obs_counts.get(z, 0)
                        for z in all_z)
                    n_eff = max(int(total_obs), 1)
                else:
                    n_eff = max(int(n_samples), 1)

            # 加权残差平方和（用 pre_clip lambda，在完整 Z 上）
            resid_sq_sum = 0.0
            for z in all_z:
                r_tilde = R_hat.get(z, 0) - r_mean
                a_tilde = (a_plus.get(z, 0)
                           - a_mean)
                pred = lambda_pre_clip * r_tilde
                resid_sq_sum += (
                    w_h[z]
                    * (a_tilde - pred) ** 2)
            # 自由度：k - 2（centering + slope）
            df_adj = max(
                len(all_z) - 2, 1)
            sigma_resid_sq = (
                resid_sq_sum
                * len(all_z) / df_adj)
            sigma_resid_sq = max(
                sigma_resid_sq, 1e-4)

            denom_strength = max(denom, 1e-4)
            # ⚠ 老师 audit（2026-07-10 深夜 补充 #3）：
            # 优先用 task-level bootstrap 得到的**真 s²**
            # （若上游传了 s_sq_override）。这是老师验收
            # 标准要求的做法：s² 由 task/rollout 层面独立
            # 观测估计，且必须随 n_traj 增加而单调下降。
            #
            # 如果没有 bootstrap（老路径），仍用 analytic
            # closed-form：s² = σ_resid² / (n_eff · denom)。
            # 这是渐近形式，不真正随 n_traj 变化（因为
            # σ_resid² 是 label 空间残差方差，n_eff 只是
            # 除数）——仅作 fallback。
            if s_sq_override is not None \
                    and s_sq_override > 0:
                s_sq = float(s_sq_override)
                s_sq_source = "bootstrap"
            else:
                s_sq = sigma_resid_sq / (
                    n_eff * denom_strength)
                s_sq_source = "analytic"
            s_sq = max(s_sq, 1e-4)

            # tau_g² = 先验方差超参
            # τ²=0.25 → sd=0.5，事前 95% 区间约 [−1, 1]
            # ⚠ apply_shrinkage=False：跳过 shrinkage，
            # 直接返回 raw λ̂，供 build_epoch_target 做
            # 跨 group empirical Bayes（老师公式 §3.5）。
            if not apply_shrinkage:
                lambda_shrunk = lambda_raw
            else:
                tau_sq = tau_sq_default
                B = tau_sq / (tau_sq + s_sq)
                lambda_shrunk = (
                    B * lambda_raw
                    + (1.0 - B) * lambda_neutral)

                # ── 6. 负 λ 安全阀 ─────────
                if lambda_shrunk < 0:
                    se = (math.sqrt(s_sq)
                          if s_sq > 0 else 1.0)
                    if abs(lambda_raw) < 2.0 * se:
                        lambda_shrunk = 0.0

    # ── 5. 赋值给所有 label（group 内共用同一 λ_g）──
    # ⚠ 老师 audit（2026-07-10 issue #4）：
    #   同一个 group 必须用同一个标量 λ_g，
    #   包括 OTHER。否则 target 路径
    #   q_λ(z) ∝ P_0(z)·exp(λ·r_T(z)) 不再由一个
    #   scalar 参数化，端点性质丢失：λ=1 时 named
    #   labels 移到 P_T 但 OTHER 停留 neutral →
    #   q_1 ≠ P_T。
    #
    #   之前的实现给 OTHER 单独用 lambda_neutral，
    #   破坏了 λ 作为 teacher-reference 路径坐标的
    #   解释。现修复为：所有 z ∈ Z 共用 lambda_shrunk。
    lambda_ = {}
    for z in all_z:
        lambda_[z] = lambda_shrunk

    return lambda_


def estimate_lambda_full(
    trajectories: list,
    module: "Module",
    P_0: dict[str, float],
    P_T: dict[str, float],
    R_hat: dict[str, float],
    P_s: dict[str, float],
    n_samples: int,
    lambda_neutral: float = 0.0,
    eta_mode: str = "occurrence_weighted",
    return_intermediates: bool = False,
    use_exact_lambda: bool = False,
    apply_shrinkage: bool = True,
    s_sq_override: float = None,
    eta_trajectories: list = None,
    eta_records: list[HistoryRecord] = None,
):
    """
    完整的 reward-calibrated λ 估计流水线：
    η → P_+ → a_+ → projection → shrinkage

    如果数据不足，fallback 到旧的 trust-score 方法

    参数
    ----
    return_intermediates : True 时返回 dict 而不是 lambda_ dict:
        {
          "lambda_":    {label: λ},        # 主输出
          "eta":        {label: η(h,z)},   # 局部成功率
          "a_plus":     {label: a_+},      # 有用度倾斜
          "obs_counts": {label: n_h},      # 观测次数
          "fallback":   bool,              # 是否走了旧 fallback
        }
    False (默认) 保持向后兼容，只返回 lambda_ dict。

    eta_mode
    --------
    "contrastive"        (推荐, 2026-07-10 新增)
        用 η(z) 与 η(¬z) 对比：a_+(z) = log(η_pos/η_neg)。
        解决 multi-label φ 让 η 磨平的问题。稀少但有效的
        label 能得到大的 a_+，普遍口癖 a_+≈0。
    "occurrence_weighted" (旧默认)
        每次 label 出现算一次观测，a_+ = log(P_+ / P_0)。
        multi-label 场景下 η 之间差异被 magnitude of
        occurrences 抵消。
    "presence_weighted"  : 出现即一次观测（不重复计入次数）
    "primary_only"       : 每条轨迹只算主 label（仅 ablation）
    """
    Z = get_Z_for_module(module)
    if eta_records is not None and eta_mode != "occurrence_weighted":
        raise ValueError(
            "formal HistoryRecord path only supports occurrence_weighted; "
            "legacy eta modes use incompatible observation semantics")
    eta_source = (eta_trajectories
                  if eta_trajectories is not None
                  else trajectories)
    task_clusters = ({
        r.task_id for r in eta_records
        if r.module_id == module.module_id
    } if eta_records is not None else {
        str(t.task_id) for t in eta_source
        if getattr(t, "task_id", None) is not None
        and getattr(t, "steps", None)
    })
    n_traj = len(task_clusters)
    s_sq_default = (1.0 / n_traj if n_traj > 0
                    else float("inf"))

    # ─── contrastive mode（推荐）───
    if eta_mode == "contrastive":
        eta_pos, eta_neg = estimate_eta_contrastive(
            trajectories, module, Z)
        # 检查是否有足够的 reward 变化
        pos_vals = list(eta_pos.values())
        eta_range = (
            max(pos_vals) - min(pos_vals)
            if pos_vals else 0)

        if eta_range < 0.01 or n_samples < 5:
            # 证据不足 → 退到 lambda_neutral（用户定义中性值）
            # 0 = "证据不足，退回 P_0"（保守）
            # 1 = "证据不足，照抄 teacher"（激进 baseline）
            # 严禁退到旧 trust-score，那是不同的估计器
            all_z = list(R_hat.keys())
            lambda_fb = {z: lambda_neutral
                         for z in all_z}
            if return_intermediates:
                return {
                    "lambda_":    lambda_fb,
                    "eta":        eta_pos,
                    "eta_neg":    eta_neg,
                    "a_plus":     {},
                    "obs_counts": {},
                    "fallback":   True,
                    "n_traj": n_traj,
                    "s_sq_g": s_sq_default,
                    "s_sq_source": "task_clusters",
                    "fallback_reason": (
                        f"eta_pos_range="
                        f"{eta_range:.3f}<0.01"
                        if eta_range < 0.01
                        else f"n_samples="
                             f"{n_samples}<5"),
                }
            return lambda_fb

        # a_+ = log(η_pos / η_neg) 对比式
        a_plus = compute_a_plus_contrastive(
            eta_pos, eta_neg)

        obs_counts = count_labels_for_module(
            trajectories, module)

        # ⚠ n_trajectories = 独立轨迹数
        # （audit issue #3：不是 label 观测总数）
        n_traj = sum(
            1 for t in trajectories
            if module.skill_id in (
                t.activated_skills or []))

        lambda_ = estimate_lambda_projection(
            R_hat, a_plus, P_0,
            module.named_behaviors,
            n_samples, lambda_neutral,
            obs_counts=obs_counts,
            n_trajectories=n_traj)

        if return_intermediates:
            return {
                "lambda_":    lambda_,
                "eta":        eta_pos,
                "eta_neg":    eta_neg,
                "a_plus":     a_plus,
                "obs_counts": obs_counts,
                "fallback":   False,
            }
        return lambda_

    # ─── 旧的 P_+ mode（occurrence/presence/primary）───
    # Step 1: η(h,z) 局部成功率
    # ⚠ 老师 audit #1（2026-07-10）：
    #   step_aware 模式（默认 True）严格保留 skill step h：
    #   η̂(h,z) = Σ_i A_ih · 1{Z_ih=z} · Y_i /
    #            Σ_i A_ih · 1{Z_ih=z}
    #   group-level η(z) = Σ_h ω_h · η̂(h,z)
    #   同时返回严格的 obs_counts (含 step 边界)。
    #
    #   老 estimate_eta 抹去 step 索引，把长 trajectory
    #   的 label 重复计入，估的不是 η(h,z)。仅保留供
    #   ablation 对照。
    use_step_aware = (
        eta_mode == "occurrence_weighted")
    if eta_records is not None:
        eta_observed, obs_counts, unsupported = \
            estimate_eta_from_records(eta_records, module)
        if unsupported:
            all_z = list(R_hat.keys())
            lambda_fb = {z: lambda_neutral for z in all_z}
            result = {
                "lambda_": lambda_fb,
                "eta": eta_observed,
                "a_plus": {},
                "obs_counts": obs_counts,
                "fallback": True,
                "n_traj": n_traj,
                "s_sq_g": s_sq_default,
                "s_sq_source": "task_clusters",
                "fallback_reason": (
                    "zero_support_labels=" + ",".join(unsupported)),
            }
            return result if return_intermediates else lambda_fb
        eta = eta_observed
    elif use_step_aware:
        eta, obs_counts = estimate_eta_step_aware(
            eta_source, module, Z)
    else:
        eta = estimate_eta(
            trajectories, module, Z,
            mode=eta_mode)
        obs_counts = count_labels_for_module(
            trajectories, module)

    # 检查是否有足够的 reward 变化
    eta_vals = list(eta.values())
    eta_range = (max(eta_vals) - min(eta_vals)
                 if eta_vals else 0)

    if eta_range < 0.01 or n_samples < 5:
        # 证据不足 → 退到 lambda_neutral（用户定义中性值）
        # 严禁退到旧 trust-score（那是不同的估计器）
        all_z = list(R_hat.keys())
        lambda_fb = {z: lambda_neutral
                     for z in all_z}
        if return_intermediates:
            return {
                "lambda_":    lambda_fb,
                "eta":        eta,
                "a_plus":     {},
                "obs_counts": {},
                "fallback":   True,
                "n_traj": n_traj,
                "s_sq_g": s_sq_default,
                "s_sq_source": "task_clusters",
                "fallback_reason": (
                    f"eta_range={eta_range:.3f}<0.01"
                    if eta_range < 0.01
                    else f"n_samples={n_samples}<5"),
            }
        return lambda_fb

    active_eta_labels = sum(
        1 for count in obs_counts.values() if count > 0)
    if active_eta_labels < 2:
        all_z = list(R_hat.keys())
        lambda_fb = {z: lambda_neutral for z in all_z}
        if return_intermediates:
            return {
                "lambda_": lambda_fb,
                "eta": eta,
                "a_plus": {},
                "obs_counts": obs_counts,
                "fallback": True,
                "n_traj": n_traj,
                "s_sq_g": s_sq_default,
                "s_sq_source": "task_clusters",
                "fallback_reason": "fewer_than_two_observed_labels",
            }
        return lambda_fb

    # Step 2: P_+
    P_plus = compute_P_plus(P_0, eta)

    # Step 3: a_+
    a_plus = compute_a_plus(P_plus, P_0)

    # ⚠ n_trajectories = 独立轨迹数
    # （audit issue #3：不是 label 观测总数）
    #
    # ⚠ 老师公式 §3.5（2026-07-11）：N_g 应为**能对该 group
    # 提供证据的独立 task 数**。多 group 时不同 group 会看到
    # 不同 traj 子集：如果一条 traj 里 module g 的所有 named
    # label 都是 0 观测 → 这条 traj 对 g 无证据 → 不算入 N_g。
    #
    # 单 group（force_single_group=true）时，所有 named 都在
    # 一个 module 里 → N_g 退化为激活 skill 的 traj 数。
    #
    # 多 group 时：N_g = |{traj: 至少有一个 g 的 named label
    # 在该 traj 里被观测到}|
    # ⚠ 老师公式（2026-07-11）：s²_g = 1 / N_g
    # 其中 N_g = group 内独立 task 数 = n_traj（上面已算）。
    # τ² 由跨 group empirical Bayes 估（在 build_epoch_target
    # 层面做），此处只报 s²_g 和 raw λ̂_g，让上层聚合。
    #
    # 老 bootstrap s² 保留供 diagnostic（return_intermediates
    # 时返回 s_sq_bootstrap），但不用于 shrinkage。
    #
    # 优先级：
    # 1. s_sq_override（上层传入，比如 EB 已算好）
    # 2. 1 / N_g（老师公式，默认）
    if s_sq_override is not None \
            and s_sq_override > 0:
        s_sq_g = float(s_sq_override)
        s_sq_source = "override"
    elif n_traj > 0:
        s_sq_g = 1.0 / n_traj
        s_sq_source = "teacher_1_over_N"
    else:
        s_sq_g = float("inf")
        s_sq_source = "no_task_clusters"

    # 可选：老 bootstrap s² 作诊断量（不用于 shrinkage）
    boot_info = None
    if return_intermediates and n_traj >= 5:
        boot_info = estimate_slope_variance_bootstrap(
            trajectories, module, Z, R_hat,
            module.named_behaviors,
            n_bootstrap=200,
            eta_mode=eta_mode,
        )

    # Step 5: λ 估计（两种模式）
    # - use_exact_lambda=True: 精确 argmin_ℓ KL(P_+ ‖ q_ℓ)
    #   一维 Newton-Raphson，无小倾斜近似。
    # - use_exact_lambda=False（默认）: WLS 小倾斜近似
    #
    # apply_shrinkage=True：单 group 局部收缩（用固定 τ²）
    # apply_shrinkage=False：只返回 raw λ̂_g，供 build_epoch_target
    #   跨 group 做 empirical Bayes shrinkage（老师公式）
    if use_exact_lambda:
        lam_scalar = estimate_lambda_exact(
            P_plus, P_0, R_hat,
            module.named_behaviors,
            lambda_neutral,
            n_trajectories=n_traj,
            obs_counts=obs_counts,
            apply_shrinkage=apply_shrinkage,
            apply_negative_check=apply_shrinkage,
            s_sq_override=s_sq_g)
        # 广播到所有 z（含 other 用同一 λ_g）
        lambda_ = {z: lam_scalar
                   for z in R_hat.keys()}
    else:
        # projection 内部会根据 s_sq_override 用 s²_g
        # 若 apply_shrinkage=False，我们后面把 lambda 强制
        # 覆盖成 raw slope（不做 shrinkage）
        lambda_ = estimate_lambda_projection(
            R_hat, a_plus, P_0,
            module.named_behaviors,
            n_samples, lambda_neutral,
            obs_counts=obs_counts,
            n_trajectories=n_traj,
            s_sq_override=s_sq_g,
            apply_shrinkage=apply_shrinkage)

    if return_intermediates:
        out = {
            "lambda_":    lambda_,
            "eta":        eta,
            "a_plus":     a_plus,
            "obs_counts": obs_counts,
            "fallback":   False,
            "s_sq_g":     s_sq_g,
            "s_sq_source": s_sq_source,
            "n_traj":     n_traj,
        }
        if boot_info is not None:
            out["s_sq_bootstrap"] = boot_info.get(
                "s_sq")
            out["n_boot"] = boot_info.get("n_boot")
        return out
    return lambda_


# ─────────────────────────────────────────────────
# Step 6：选 β
# ─────────────────────────────────────────────────

def select_beta(
    R_hat_all: dict[str, dict],
    P_0_all: dict[str, dict],
    lambda_all: dict[str, dict],
    w_all: dict[str, float],
    budget: float = 1.0,
) -> float:
    """
    β 已按老师要求中性化（改为常数 1.0）。

    理由：λ_g 本身已经是 reward-calibrated projection slope，
    已经承担了 epoch-level trust 的作用。原公式 exp(β·λ·r) 中
    的 β 会额外放大/压缩信号，与老师论文 §3.10 的
    Q_λ(z) ∝ P_0(z) · exp(λ_g · r_T(z)) 不符。
    保留函数签名以维持向后兼容。
    """
    return 1.0


# ─────────────────────────────────────────────────
# 老师公式（2026-07-11）：Empirical Bayes 聚合
# s_g² = 1/N_g,  τ̂² = max{0, (1/G)Σ[(λ̂_g − λ_neu)² − s_g²]}
# B_g = τ̂² / (τ̂² + s_g²)
# λ_shrunk_g = B_g·λ̂_g + (1-B_g)·λ_neu
# ─────────────────────────────────────────────────

def empirical_bayes_shrink(
    lambda_hat_by_group: dict[str, float],
    N_by_group: dict[str, int],
    lambda_neutral: float = 0.0,
    apply_negative_check: bool = True,
    clip_range: tuple = None,
) -> dict:
    """
    老师公式（2026-07-11）：跨 group empirical Bayes shrinkage。

    输入
    ----
    lambda_hat_by_group : {group_id: raw λ̂_g}  (未 shrunk)
    N_by_group          : {group_id: N_g}      (独立 task 数)
    lambda_neutral      : λ_neutral（默认 0）
    apply_negative_check: 显著性检查，|λ_raw| < 2·SE_g 且负 → 0
    clip_range          : 最终 clip

    公式
    ----
    s_g² = 1 / N_g                           (老师公式)
    τ̂² = max{0, (1/G) Σ_g [(λ̂_g − λ_neu)² − s_g²]}
                                              (跨 group MoM)
    B_g = τ̂² / (τ̂² + s_g²)
    λ_shrunk_g = B_g·λ̂_g + (1 − B_g)·λ_neu

    返回
    ----
    dict:
      "lambda_shrunk_by_group": {group_id: λ_shrunk}
      "s_sq_by_group"         : {group_id: s²_g}
      "B_by_group"            : {group_id: B_g}
      "tau_sq_hat"            : 估计出的 τ̂²
      "G"                     : group 数
      "signal_var"            : (1/G)Σ(λ̂ − λ_neu)²
      "avg_noise_var"         : (1/G)Σ s²_g
    """
    groups = list(lambda_hat_by_group.keys())
    G = len(groups)
    if G == 0:
        return {
            "lambda_shrunk_by_group": {},
            "s_sq_by_group": {},
            "B_by_group": {},
            "tau_sq_hat": 0.0,
            "G": 0,
            "signal_var": 0.0,
            "avg_noise_var": 0.0,
        }

    # s_g² = 1 / N_g（老师公式）
    s_sq_by_group = {}
    for g in groups:
        N_g = int(N_by_group.get(g, 0))
        s_sq_by_group[g] = (
            1.0 / N_g if N_g > 0 else float("inf"))

    # τ̂² = max{0, (1/G) Σ_g [(λ̂_g − λ_neu)² − s_g²]}
    signal_terms = []
    noise_terms = []
    for g in groups:
        lam_hat = lambda_hat_by_group[g]
        s_sq = s_sq_by_group[g]
        signal_terms.append(
            (lam_hat - lambda_neutral) ** 2)
        noise_terms.append(s_sq)

    signal_var = sum(signal_terms) / G
    finite_noise = [v for v in noise_terms if math.isfinite(v)]
    avg_noise_var = (sum(finite_noise) / len(finite_noise)
                     if finite_noise else float("inf"))
    tau_sq_hat = (max(0.0, signal_var - avg_noise_var)
                  if math.isfinite(avg_noise_var) else 0.0)

    # 每 group 独立 shrinkage
    lambda_shrunk_by_group = {}
    B_by_group = {}
    for g in groups:
        s_sq = s_sq_by_group[g]
        # B_g = τ̂² / (τ̂² + s_g²)
        denom = tau_sq_hat + s_sq
        if not math.isfinite(s_sq) or denom < 1e-10:
            # 两者都为 0 → 无信号，退回 neutral
            B_g = 0.0
        else:
            B_g = tau_sq_hat / denom
        B_by_group[g] = B_g

        lam_hat = lambda_hat_by_group[g]
        lam_shrunk = (
            B_g * lam_hat
            + (1.0 - B_g) * lambda_neutral)

        # 负 λ 显著性检查
        if apply_negative_check and lam_shrunk < 0:
            se_g = math.sqrt(s_sq) if s_sq > 0 else 1.0
            if abs(lam_hat - lambda_neutral) < 2.0 * se_g:
                lam_shrunk = lambda_neutral

        if clip_range is not None:
            lo, hi = clip_range
            lam_shrunk = max(lo, min(hi, lam_shrunk))
        lambda_shrunk_by_group[g] = lam_shrunk

    return {
        "lambda_shrunk_by_group": lambda_shrunk_by_group,
        "s_sq_by_group": s_sq_by_group,
        "B_by_group": B_by_group,
        "tau_sq_hat": tau_sq_hat,
        "G": G,
        "signal_var": signal_var,
        "avg_noise_var": avg_noise_var,
    }


# ─────────────────────────────────────────────────
# Step 7：Q_n 构造（target distribution）
# ─────────────────────────────────────────────────

def compute_Q(
    P_0: dict[str, float],
    R_hat: dict[str, float],
    lambda_: dict[str, float],
    beta: float,
    lambda_neutral: float = 0.0,
) -> dict[str, float]:
    """
    Q_n(z) ∝ P̂_0(z) × exp[β × λ(z) × R̂(z)]

    lambda_neutral: 若 lambda_ dict 缺失 z 时使用的默认值。
    应该等于 study1 config 的 lambda_neutral（默认 0.0 =
    保守退回 P_0）。旧默认 1.0 会让 OTHER 或 未估计 label
    默认满信任 teacher，不合理。
    """
    return _compute_Q(
        P_0, R_hat, lambda_, beta, lambda_neutral)


def _compute_Q(
    P_0: dict, R_hat: dict,
    lambda_: dict, beta: float,
    lambda_neutral: float = 0.0,
) -> dict:
    # 数值稳定：先算指数项，再做 log-sum-exp 归一
    log_unnorm = {}
    for z in P_0:
        p0 = max(P_0.get(z, 1e-6), 1e-6)
        r  = R_hat.get(z, 0.0)
        lam = lambda_.get(z, lambda_neutral)
        log_unnorm[z] = (
            math.log(p0) + beta * lam * r)

    # log-sum-exp 归一
    if not log_unnorm:
        return {}
    log_max = max(log_unnorm.values())
    log_sum = log_max + math.log(sum(
        math.exp(v - log_max)
        for v in log_unnorm.values()))
    result = {
        z: math.exp(v - log_sum)
        for z, v in log_unnorm.items()
    }
    # 数值兜底
    total = sum(result.values())
    if total <= 0 or not math.isfinite(total):
        n = len(P_0)
        return {z: 1.0/n for z in P_0}
    # 二次归一（防浮点误差）
    return {z: v/total for z, v in result.items()}


def _kl_divergence(
    P: dict, Q: dict,
) -> float:
    """KL(P ‖ Q)"""
    kl = 0.0
    for z in P:
        p = max(P.get(z, 1e-10), 1e-10)
        q = max(Q.get(z, 1e-10), 1e-10)
        kl += p * math.log(p / q)
    return max(kl, 0.0)


# ─────────────────────────────────────────────────
# Step 8：g_n 计算（edit gradient）
# ─────────────────────────────────────────────────

def compute_g_n(
    Q_n: dict[str, float],
    P_s: dict[str, float],
) -> dict[str, float]:
    """
    g_n(z) = log(Q_n(z) / P_s(z))

    g_n > 0：需要增加这个行为
    g_n < 0：需要减少这个行为
    """
    g = {}
    for z in Q_n:
        q = max(Q_n.get(z, 1e-6), 1e-6)
        p = max(P_s.get(z, 1e-6), 1e-6)
        g[z] = math.log(q / p)
    return g


def select_donor_pair(
    g_n: dict[str, float],
) -> tuple[str, str]:
    """
    z⁺ = argmax g_n（增加的行为）
    z⁻ = argmin g_n（减少的行为）
    """
    if not g_n:
        return "", ""
    z_plus  = max(g_n, key=g_n.get)
    z_minus = min(g_n, key=g_n.get)
    return z_plus, z_minus


# ─────────────────────────────────────────────────
# Step 9：生成给 Teacher 的自然语言指令
# ─────────────────────────────────────────────────

def format_edit_direction(
    module_signal: ModuleSignal,
) -> str:
    """
    把 g_n 翻译成自然语言
    """
    lam_vals = [v for z, v in
                module_signal.lambda_.items()
                if z != OTHER_LABEL]
    lam_mean = (sum(lam_vals) / len(lam_vals)
                if lam_vals else 0.0)
    lines = [
        f"Module: {module_signal.skill_id} "
        f"(λ_g={lam_mean:+.2f}, "
        f"w={module_signal.w:.3f})",
        "",
        "Current vs Target behavior distribution:",
    ]

    g = module_signal.g_n
    P_s = module_signal.P_s
    Q_n = module_signal.Q_n

    # 按 |g_n| 降序排列
    sorted_z = sorted(
        g.keys(),
        key=lambda z: abs(g.get(z, 0)),
        reverse=True)

    for z in sorted_z:
        gv = g.get(z, 0)
        ps = P_s.get(z, 0)
        qn = Q_n.get(z, 0)
        direction = "INCREASE" if gv > 0 else "DECREASE"
        lines.append(
            f"  {z}: {ps:.1%} → {qn:.1%} "
            f"(g={gv:+.2f}, {direction}, "
            f"parent={module_signal.label_parents.get(z, '')!r})")

    # 明确的 donor pair
    z_plus  = module_signal.donor_plus
    z_minus = module_signal.donor_minus
    if z_plus and z_minus:
        lines.append("")
        lines.append(
            f"Primary edit direction: "
            f"REDUCE {z_minus}, "
            f"INCREASE {z_plus}")

    return "\n".join(lines)


def format_all_edit_directions(
    epoch_target: EpochTarget,
) -> str:
    """格式化所有 module 的修改方向"""
    parts = [
        f"β = {epoch_target.beta:.2f}",
        "=" * 50,
    ]
    # 按 w 降序（最重要的先说）
    sorted_modules = sorted(
        epoch_target.modules,
        key=lambda ms: ms.w,
        reverse=True)

    for ms in sorted_modules:
        if ms.w < 0.01:
            continue
        parts.append(format_edit_direction(ms))
        parts.append("")

    return "\n".join(parts)


# ─────────────────────────────────────────────────
# 完整流水线：Steps 1-9 一次性执行
# ─────────────────────────────────────────────────

def build_epoch_target(
    library,
    trajectories: list,
    diagnoses_map: dict,
    teacher_grades: dict = None,
    ref_trajectories: list = None,
    teacher_trajectories: list = None,
    beta_budget: float = 1.0,
    alpha: float = 1.0,
    ref_baseline: object = None,
    lambda_neutral: float = 0.0,
    use_reward_calibrated_lambda: bool = True,
    parsed_modules: list = None,
    eta_mode: str = "occurrence_weighted",
    use_exact_lambda: bool = False,
    student_records: list[HistoryRecord] = None,
    reference_records: list[HistoryRecord] = None,
    teacher_records: list[HistoryRecord] = None,
) -> EpochTarget:
    """
    完整流水线（对应论文 §3.10 步骤 9-14）：

    1. modules + Z_named（从 Parser LLM 或正则提取）
    2. P_s（Student 行为分布）
    3. P_T（Teacher rollout）, P_0（Reference rollout）, R̂
    4. η → P_+ → a_+（成功加权目标 + 有用度倾斜）
    5. λ_g = project(a_+, r_T) + shrinkage
    6. β（二分搜索 KL budget）
    7. Q_λ, g_n, donor pair

    参数：
      parsed_modules: Parser LLM 输出的 Module 列表
                      为 None 时 fallback 到正则提取
      teacher_trajectories: Teacher rollout 轨迹（§3.2 步骤5）
      ref_trajectories: Reference rollout 轨迹（§3.2 步骤6）
    """
    # ── Step 1: modules + Z_named ────────────
    if parsed_modules:
        modules = parsed_modules
        print(f"    [Epoch] 使用 Parser LLM 的 "
              f"{len(modules)} 个 modules")
    else:
        modules = extract_modules_and_behaviors(
            library)

    if student_records is None:
        student_records = build_history_records(
            trajectories, modules, "student")
    if reference_records is None and ref_trajectories:
        reference_records = build_history_records(
            ref_trajectories, modules, "reference")
    if teacher_records is None and teacher_trajectories:
        teacher_records = build_history_records(
            teacher_trajectories, modules, "teacher")
    reference_records = reference_records or []
    teacher_records = teacher_records or []
    if use_reward_calibrated_lambda and not reference_records:
        raise ValueError(
            "reward-calibrated lambda requires valid reached reference "
            "history records")

    # ── Step 2: P_s ──────────────────────────
    P_s_all = compute_distributions_from_records(
        student_records, modules, alpha)

    # ── Step 3: P_T, P_0, R̂ ─────────────────
    # ⚠ 方法学重点：P_T 和 P_0 都必须来自各自的合法端点。
    # 用 "成功 student" 冒充 Teacher 或用 "失败 student"
    # 冒充 Reference，会让 R̂ 与 η 变成同一个信号，
    # λ 投影退化 → 恒为 0。所有这类 fallback 都已禁用。

    # ── P_T：Teacher rollout 或 b4 grading（严格端点）──
    if teacher_records:
        # 优先：Teacher rollout（哪怕成功很少也用它，
        # 因为它是唯一"来自 teacher 模型"的行为数据）
        P_T_all = {
            mid: dist.probs for mid, dist in
            compute_distributions_from_records(
                teacher_records, modules, alpha).items()
        }
        n_succ = sum(
            1 for t in teacher_trajectories
            if t.success)
        print(f"    [P_T] 从 {len(teacher_trajectories)}"
              f" 条 Teacher rollout 估计 "
              f"(成功 {n_succ} 条)")
        if n_succ == 0:
            print(f"    ⚠ Teacher 全败：P_T 只反映"
                  f"其失败行为，λ 可能偏向 0")
    elif teacher_grades:
        # 备选：b4 grading 的 teacher_preferred_action
        # （仍是来自 teacher 端点的信号）
        P_T_all = estimate_teacher_distribution(
            teacher_grades, modules, alpha,
            success_trajectories=None,
            strict_no_fallback=True)
        print(f"    [P_T] 从 b4 grading 估计 "
              f"(no rollout available)")
    else:
        raise ValueError(
            "teacher endpoint data are required; uniform P_T is not "
            "a valid framework estimate")

    # ── P_0：Reference rollout（严格端点）──
    if reference_records:
        P_0_all = {
            mid: dist.probs for mid, dist in
            compute_distributions_from_records(
                reference_records, modules, alpha).items()
        }
        print(f"    [P_0] 从 {len(ref_trajectories)}"
              f" 条 Reference rollout 估计")
    elif (ref_baseline is not None and
          hasattr(ref_baseline,
                  'get_label_distribution')):
        # 备选：已保存的 baseline label 分布
        P_0_all = estimate_reference_distribution(
            [], modules, alpha,
            ref_baseline=ref_baseline,
            failed_trajectories=None,
            strict_no_fallback=True)
        print(f"    [P_0] 从 baseline 已保存分布估计")
    else:
        raise ValueError(
            "reference endpoint data are required; uniform P_0 is not "
            "a valid framework estimate")

    R_hat_all = {}
    for mod in modules:
        mid = mod.module_id
        P_T = P_T_all.get(mid, {})
        P_0 = P_0_all.get(mid, {})
        R_hat_all[mid] = compute_label_contrast(
            P_T, P_0)

    # ── Step 4-5: λ（reward-calibrated）──────
    # 第一遍估 raw λ_g 与 N_g；第二遍严格按跨组 MoM 公式估 τ̂²
    # 并收缩。eta 仅从 reference-policy trajectories 估计。
    #
    # m 已移除；w = d_s（纯占用率加权）
    n_total = max(len(student_records), 1)
    d_s_all = {}
    w_all = {}
    lambda_all = {}
    lambda_raw_by_group = {}  # 每 group 的 raw λ̂（诊断用）
    N_by_group = {}           # 每 group 独立 task 数
    # 中间量落地
    eta_all = {}
    a_plus_all = {}
    obs_counts_all = {}
    estimable_all = {}
    fallback_reason_all = {}
    history_fit_data = {}
    group_by_history = {
        mod.module_id: (mod.group_id or mod.module_id)
        for mod in modules}

    for mod in modules:
        mid = mod.module_id
        n_rel = sum(
            1 for r in student_records
            if r.module_id == mid)
        d_s = n_rel / n_total
        d_s_all[mid] = d_s
        w_all[mid] = d_s

        P_s = P_s_all.get(mid)
        P_s_probs = P_s.probs if P_s else {}
        n_samples = P_s.total if P_s else 0

        if use_reward_calibrated_lambda:
            _full = estimate_lambda_full(
                trajectories, mod,
                P_0_all.get(mid, {}),
                P_T_all.get(mid, {}),
                R_hat_all.get(mid, {}),
                P_s_probs, n_samples,
                lambda_neutral,
                eta_mode=eta_mode,
                return_intermediates=True,
                use_exact_lambda=(
                    use_exact_lambda),
                apply_shrinkage=False,
                eta_trajectories=ref_trajectories,
                eta_records=reference_records,
            )
            # 第一遍得到 raw group scalar
            lam_dict = _full["lambda_"]
            lambda_all[mid] = lam_dict
            if lam_dict:
                lam_scalar = next(
                    iter(lam_dict.values()))
            else:
                lam_scalar = lambda_neutral
            estimable = not bool(_full.get("fallback"))
            estimable_all[mid] = estimable
            fallback_reason_all[mid] = _full.get(
                "fallback_reason", "")
            eta_all[mid] = _full.get("eta", {})
            a_plus_all[mid] = _full.get("a_plus", {})
            obs_counts_all[mid] = _full.get(
                "obs_counts", {})
            # N_g：reference-policy 中唯一 task-id clusters 数
            if estimable:
                history_fit_data[mid] = {
                    "P_0": P_0_all.get(mid, {}),
                    "R_hat": R_hat_all.get(mid, {}),
                    "a_plus": a_plus_all[mid],
                    "obs_counts": obs_counts_all[mid],
                    "weight": float(max(n_samples, 0)),
                }
            if _full.get("fallback"):
                print(f"      ⚠️ λ fallback: "
                      f"{_full.get('fallback_reason','')}")
        else:
            estimable_all[mid] = True
            fallback_reason_all[mid] = ""
            lambda_all[mid] = estimate_lambda(
                R_hat_all.get(mid, {}),
                mod, P_s_probs,
                P_T_all.get(mid, {}),
                n_samples)
            eta_all[mid] = {}
            a_plus_all[mid] = {}
            obs_counts_all[mid] = {}

    # λ 只在此处按 group pooling；P/η/r/a 始终保留 history 维度。
    if use_reward_calibrated_lambda:
        histories_by_group = {}
        for mid, fit in history_fit_data.items():
            histories_by_group.setdefault(
                group_by_history[mid], []).append(fit)
        for gid, fits in histories_by_group.items():
            lambda_raw_by_group[gid] = estimate_group_lambda_from_histories(
                fits, lambda_neutral=lambda_neutral,
                use_exact=use_exact_lambda)
            history_ids = {
                mid for mid, group_id in group_by_history.items()
                if group_id == gid and estimable_all.get(mid, False)}
            N_by_group[gid] = len({
                record.task_id for record in reference_records
                if record.module_id in history_ids})

    eb_result = empirical_bayes_shrink(
        lambda_raw_by_group, N_by_group,
        lambda_neutral=lambda_neutral,
        apply_negative_check=True,
        clip_range=None)
    if use_reward_calibrated_lambda:
        for mod in modules:
            mid = mod.module_id
            if not estimable_all.get(mid, False):
                support = R_hat_all.get(mid, {}).keys()
                lambda_all[mid] = {
                    z: lambda_neutral for z in support}
                w_all[mid] = 0.0
                continue
            gid = group_by_history[mid]
            lam = eb_result["lambda_shrunk_by_group"].get(
                gid, lambda_neutral)
            support = R_hat_all.get(mid, {}).keys()
            lambda_all[mid] = {z: lam for z in support}
            print(f"    [group {mid[:30]}] "
                  f"group={gid[:24]}, N={N_by_group.get(gid, 0)}, "
                  f"λ_raw={lambda_raw_by_group.get(gid, lambda_neutral):+.4f}, "
                  f"λ={lam:+.4f}, "
                  f"s²={eb_result['s_sq_by_group'].get(gid, float('inf')):.4f}, "
                  f"B={eb_result['B_by_group'].get(gid, 0.0):.4f}")

    # ── Step 6: β ────────────────────────────
    beta = select_beta(
        R_hat_all, P_0_all,
        lambda_all, w_all,
        beta_budget)

    # ── Step 7-8: Q_λ, g_n, donor pair ──────
    module_signals = []
    for mod in modules:
        mid = mod.module_id
        P_0 = P_0_all.get(mid, {})
        R_hat = R_hat_all.get(mid, {})
        lam = lambda_all.get(mid, {})
        P_s = P_s_all.get(mid)
        P_s_probs = P_s.probs if P_s else {}

        estimable = estimable_all.get(mid, True)
        gid = group_by_history[mid]
        if estimable:
            Q_n = compute_Q(
                P_0, R_hat, lam, beta,
                lambda_neutral=lambda_neutral)
            g_n = compute_g_n(Q_n, P_s_probs)
            z_plus, z_minus = select_donor_pair(g_n)
        else:
            Q_n = {}
            g_n = {}
            z_plus = z_minus = ""

        ms = ModuleSignal(
            module_id=mid,
            skill_id=mod.skill_id,
            d_s=d_s_all.get(mid, 0),
            m=1.0,  # m 已移除，保持接口兼容
            w=w_all.get(mid, 0),
            P_s=P_s_probs,
            P_T=P_T_all.get(mid, {}),
            P_0=P_0,
            R_hat=R_hat,
            eta=eta_all.get(mid, {}),
            a_plus=a_plus_all.get(mid, {}),
            lambda_=lam,
            Q_n=Q_n,
            g_n=g_n,
            donor_plus=z_plus,
            donor_minus=z_minus,
            obs_counts=obs_counts_all.get(mid, {}),
            lambda_raw=lambda_raw_by_group.get(
                gid, lambda_neutral),
            N_g=N_by_group.get(gid, 0),
            s_sq_g=eb_result.get(
                "s_sq_by_group", {}).get(gid, 0.0),
            shrinkage_B=eb_result.get(
                "B_by_group", {}).get(gid, 0.0),
            tau_sq_hat=eb_result.get("tau_sq_hat", 0.0),
            estimable=estimable,
            fallback_reason=fallback_reason_all.get(mid, ""),
            reach_patterns=copy.deepcopy(mod.reach_patterns),
            label_patterns=copy.deepcopy(mod.label_patterns),
            label_parents=copy.deepcopy(mod.label_parents),
            group_id=gid,
            history_id=mod.history_id or mid,
            skill_step_text=mod.skill_step_text,
        )
        module_signals.append(ms)

    epoch = EpochTarget(
        modules=module_signals,
        beta=beta,
    )

    # ── Z 质量检查 ───────────────────────────
    for ms in module_signals:
        other_ratio = ms.P_s.get(OTHER_LABEL, 0)
        if other_ratio > 0.5:
            print(f"    ⚠️ Z quality: "
                  f"{ms.skill_id[:30]} "
                  f"other={other_ratio:.0%} "
                  f"(too high)")
        r_range = 0
        if ms.R_hat:
            r_vals = [v for v in ms.R_hat.values()
                      if v != 0]
            if r_vals:
                r_range = max(r_vals) - min(r_vals)
        if r_range < 0.1:
            print(f"    ⚠️ Z quality: "
                  f"{ms.skill_id[:30]} "
                  f"R̂ range={r_range:.2f} "
                  f"(teacher/ref nearly same)")

        # 打印 λ 估计结果
        lam_vals = [v for z, v in ms.lambda_.items()
                    if z != OTHER_LABEL]
        if lam_vals:
            lam_mean = sum(lam_vals) / len(lam_vals)
            print(f"    λ: {ms.skill_id[:30]} "
                  f"λ_g={lam_mean:+.3f}")

    return epoch


# ─────────────────────────────────────────────────
# 辅助：label-KL gate（用于 dual gate 的第一道门）
# ─────────────────────────────────────────────────

def compute_label_kl_decrease(
    P_s_old: dict[str, float],
    P_s_new: dict[str, float],
    Q_n: dict[str, float],
) -> float:
    """
    ΔKL = KL(P_s_old ‖ Q_n) - KL(P_s_new ‖ Q_n)

    > 0 表示新 skill 让行为更接近目标
    """
    kl_old = _kl_divergence(P_s_old, Q_n)
    kl_new = _kl_divergence(P_s_new, Q_n)
    return kl_old - kl_new
