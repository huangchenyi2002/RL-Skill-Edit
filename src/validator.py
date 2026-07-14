# src/validator.py
"""
Validator Gate
对应提案 §8.2（Dual-layer Security Gate）
修复：放宽安全门，避免误拦截正常 dispatch edits
"""

import re
from src.skill_library import SkillLibrary, SkillCore


class Validator:

    def __init__(self, config: dict):
        self.deny_patterns = config["security"]["deny_patterns"]
        self.allowed_caps  = set(
            config["security"]["allowed_capability_flags"])
        self.B_V           = config["security"][
            "residual_risk_budget"]

    # ─────────────────────────────────────────────
    # 统一入口
    # ─────────────────────────────────────────────

    def validate(
        self,
        candidate_lib: SkillLibrary,
        current_lib:   SkillLibrary,
    ) -> dict:
        """
        修复版：只检查发生变化的 skill
        Study 1 的 dispatch edit 只改 name/description
        不应该触发 capability 和 script 检查
        """
        all_passed    = True
        skill_results = {}

        # 找出真正发生变化的 skill
        changed_skills = self._find_changed_skills(
            candidate_lib, current_lib)

        for sk in candidate_lib:
            # 只检查变化的 skill
            if sk.skill_id not in changed_skills:
                skill_results[sk.skill_id] = {
                    "layer_a": {"passed": True, "violations": []},
                    "layer_b": {"passed": True, "risk_score": 0.0},
                    "passed":  True,
                    "changed": False,
                }
                continue

            la = self._check_layer_a(
                sk, changed_skills[sk.skill_id])
            lb = self._check_layer_b(sk)

            passed = la["passed"] and lb["passed"]
            if not passed:
                all_passed = False

            skill_results[sk.skill_id] = {
                "layer_a": la,
                "layer_b": lb,
                "passed":  passed,
                "changed": True,
            }

        return {
            "all_passed":    all_passed,
            "skill_results": skill_results,
        }

    def _find_changed_skills(
        self,
        candidate_lib: SkillLibrary,
        current_lib:   SkillLibrary,
    ) -> dict:
        """返回每个 skill 的精确 changed fields。"""
        changed = {}
        for sk in candidate_lib:
            current_sk = current_lib.get(sk.skill_id)
            if current_sk is None:
                changed[sk.skill_id] = {
                    "name", "description", "execution_body", "references"}
                continue
            fields = set()
            if sk.name != current_sk.name:
                fields.add("name")
            if sk.description != current_sk.description:
                fields.add("description")
            if sk.execution_body != current_sk.execution_body:
                fields.add("execution_body")
            if sk.references != current_sk.references:
                fields.add("references")
            if fields:
                changed[sk.skill_id] = fields
        return changed

    # ─────────────────────────────────────────────
    # Layer A：Hard Static Gate（修复版）
    # ─────────────────────────────────────────────

    def _check_layer_a(
        self,
        sk:              SkillCore,
        changed_fields:  set,
    ) -> dict:
        """
        修复版 Layer A：
        - Dispatch edit 只改 name/description
        - 只对 name/description 做 deny-list 检查
        - execution_body 和 scripts 不重复检查
          （已在原始 skill 入库时检查过）
        """
        violations = []
        is_dispatch_only = changed_fields <= {"name", "description"}

        if is_dispatch_only:
            # Dispatch edit：只检查新的 name 和 description
            check_text = f"{sk.name} {sk.description}"
        else:
            check_text = (
                sk.name + " " + sk.description + " "
                + sk.execution_body + " "
                + " ".join(sk.references.values())
            )

        # A.1 Deny-list（只检查明确的恶意模式）
        for pattern in self.deny_patterns:
            if pattern.lower() in check_text.lower():
                violations.append(
                    f"A.1 Deny-list：检测到 '{pattern}'")

        # A.2 Capability（只对非 dispatch-only 的 skill 检查）
        if not is_dispatch_only:
            detected_caps = self._detect_capabilities(
                check_text)
            over_caps = detected_caps - self.allowed_caps
            if over_caps:
                violations.append(
                    f"A.2 Capability：未授权能力 {over_caps}")

        # A.3 Provenance
        if sk.compatibility:
            source  = sk.compatibility.get("source", "local")
            trusted = {"local", "verified", "org-vetted", ""}
            if source and source not in trusted:
                violations.append(
                    f"A.3 Provenance：未知来源 '{source}'")

        # A.4 Script Hygiene（只对非 dispatch-only 检查）
        if not is_dispatch_only and sk.references:
            for fname, content in sk.references.items():
                if fname.endswith(".py"):
                    issues = self._check_hygiene(
                        content, fname)
                    violations.extend(issues)

        return {
            "passed":     len(violations) == 0,
            "violations": violations,
        }

    def _detect_capabilities(self, text: str) -> set:
        caps = set()
        if re.search(
                r"open\s*\(|\.write\(|shutil|os\.remove",
                text):
            caps.add("file_write")
        if re.search(
                r"requests\.|urllib|http\.client|socket\.",
                text):
            caps.add("network")
        if re.search(r"subprocess|os\.system|popen", text):
            caps.add("subprocess")
        return caps

    def _check_hygiene(
        self, script: str, fname: str
    ) -> list:
        issues = []
        if "input(" in script:
            issues.append(
                f"A.4 ({fname})：含 input()")
        if re.search(
                r"pip install\s+\w+\s*$", script, re.M):
            issues.append(
                f"A.4 ({fname})：依赖未固定")
        if re.search(r"\beval\s*\(|\bexec\s*\(", script):
            issues.append(
                f"A.4 ({fname})：含 eval()/exec()")
        return issues

    # ─────────────────────────────────────────────
    # Layer B：Residual Risk Budget（修复版）
    # ─────────────────────────────────────────────

    def _check_layer_b(self, sk: SkillCore) -> dict:
        """
        修复版：dispatch edit 只改 name/description
        风险分数应该基于 execution_body
        不应该因为 description 含有技术词汇而误判
        """
        # 只对 execution_body 和 scripts 计算风险
        check_text = sk.execution_body
        for content in sk.references.values():
            check_text += " " + content

        caps  = self._detect_capabilities(check_text)
        risk_weights = {
            "file_write": 0.3,
            "network":    0.5,
            "subprocess": 0.8,
        }
        score = sum(risk_weights.get(c, 0.1) for c in caps)

        return {
            "passed":     score <= self.B_V,
            "risk_score": score,
            "B_V":        self.B_V,
        }
