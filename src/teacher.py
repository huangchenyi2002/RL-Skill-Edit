# src/teacher.py
"""
Teacher Service
实现：
  - SKILL_DEFECT vs EXECUTION_LAPSE 诊断
  - g_n 方向感知的提案生成
  - 成功轨迹分析
  - minibatch 分组诊断
  - appendix_notes 提取
  - meta_skill optimizer 记忆
  - b4 Teacher-on-student grading
"""

import json
import re
import jsonschema
from src.client import OpenRouterClient
from src.skill_library import SkillLibrary
from src.skill_library import resolve_parent_scope
from src.agent import Trajectory
from src.exskill_signal import (
    TrajectorySignal,
    format_signal_for_prompt,
    get_high_priority_tasks,
)
from src.opd_teacher_signal import (
    TrajectoryTeacherGrade,
    summarize_teacher_grade_for_prompt,
)

ERROR_TYPES = [
    "dispatch", "execution",
    "missing-skill", "tool/env", "base-model",
]

DISPATCH_EDIT_SCHEMA = {
    "type": "object",
    "required": [
        "edit_type", "skill_id",
        "new_name", "new_description",
        "rationale"],
    "properties": {
        "edit_type": {"type": "string",
            "enum": ["dispatch_edit"]},
        "skill_id": {"type": "string",
            "minLength": 1},
        "new_name": {"type": "string",
            "minLength": 3, "maxLength": 50},
        "new_description": {"type": "string",
            "minLength": 20, "maxLength": 500},
        "rationale": {"type": "string",
            "minLength": 10}},
    "additionalProperties": False,
}

EXECUTION_EDIT_SCHEMA = {
    "type": "object",
    "required": [
        "edit_type", "skill_id",
        "parent_location", "old_text", "new_text", "rationale"],
    "properties": {
        "edit_type": {"type": "string",
            "enum": ["execution_edit"]},
        "skill_id": {"type": "string",
            "minLength": 1},
        "parent_location": {"type": "string", "minLength": 1},
        "old_text": {"type": "string", "minLength": 1},
        "new_text": {"type": "string", "minLength": 1},
        "rationale": {"type": "string",
            "minLength": 10},
        "new_description": {"type": "string"}},
    "additionalProperties": False,
}

APPENDIX_EDIT_SCHEMA = {
    "type": "object",
    "required": [
        "edit_type", "skill_id",
        "appendix_notes", "rationale"],
    "properties": {
        "edit_type": {"type": "string",
            "enum": ["appendix_edit"]},
        "skill_id": {"type": "string",
            "minLength": 1},
        "appendix_notes": {"type": "array",
            "items": {"type": "string"}},
        "rationale": {"type": "string",
            "minLength": 10}},
    "additionalProperties": False,
}

DEFECT_LAPSE_SUFFIX = """

## Skill-Aware Reflection

Classify EACH failure as:
- **SKILL_DEFECT**: rule wrong/missing/underspecified → edit body
- **EXECUTION_LAPSE**: correct rule not followed → appendix reminder only

Test: "Does a rule exist that prevents this?" Yes → LAPSE. No → DEFECT.
Default to EXECUTION_LAPSE when unsure (protect the body).

Add "failure_type" and "appendix_notes" to your JSON.
"""


class TeacherService:

    def __init__(self, config: dict,
                 client: OpenRouterClient):
        self.config  = config
        self.client  = client
        self.model   = config["teacher"]["model"]
        self.temp    = config["teacher"]["temperature"]
        self.max_tok = config["teacher"]["max_tokens"]
        self.roles = {
            role: {
                "model": config.get(role, config["teacher"])["model"],
                "temperature": config.get(
                    role, config["teacher"]).get("temperature", 0.0),
                "max_tokens": config.get(
                    role, config["teacher"]).get("max_tokens", self.max_tok),
            }
            for role in ("teacher", "editor", "expert")
        }
        self._last_prompt       = ""
        self._last_raw_response = ""
        self.meta_skill_content = ""

    # ──────────────────────────────────────────────
    # 核心调用
    # ──────────────────────────────────────────────

    def _call(self, prompt: str,
              call_type: str = "teacher_diagnosis",
              temp: float = None,
              max_tok: int = None,
              role: str = "teacher",
              ) -> tuple[str, dict]:
        self._last_prompt = prompt
        profile = self.roles[role]
        response, usage = self.client.chat(
            model=profile["model"],
            messages=[{"role": "user",
                       "content": prompt}],
            temperature=(profile["temperature"]
                         if temp is None else temp),
            max_tokens=(profile["max_tokens"]
                        if max_tok is None else max_tok),
            call_type=call_type)
        self._last_raw_response = response
        return response, usage

    def _meta_skill_context(self) -> str:
        if not self.meta_skill_content:
            return ""
        return (
            "## Optimizer Meta Skill\n"
            "Cross-epoch lessons:\n\n"
            f"{self.meta_skill_content}\n\n")

    def update_meta_skill(self, improvement,
                          regression, **kwargs):
        imp = "\n".join(
            f"  + {x.get('desc','')[:80]}"
            for x in improvement[:5]) or "  (none)"
        reg = "\n".join(
            f"  - {x.get('desc','')[:80]}"
            for x in regression[:5]) or "  (none)"
        prev = self.meta_skill_content or "(first)"
        prompt = (
            "Distill cross-epoch lessons.\n"
            f"Improved:\n{imp}\n"
            f"Regressed:\n{reg}\n"
            f"Previous:\n{prev}\n\n"
            "Write 3-5 bullet points. JSON:\n"
            '{"meta_skill_content":"- lesson1\\n..."}'
        )
        try:
            resp, _ = self._call(
                prompt, temp=0.3, max_tok=512)
            s = resp.find("{")
            e = resp.rfind("}") + 1
            if s != -1 and e > 0:
                data = json.loads(resp[s:e])
                new = str(data.get(
                    "meta_skill_content", "")).strip()
                if new:
                    self.meta_skill_content = new
        except Exception:
            pass
        return self.meta_skill_content

    # ──────────────────────────────────────────────
    # b4：Teacher-on-student grading
    # ──────────────────────────────────────────────

    def generate_student_trajectory_grade(
        self, trajectory: Trajectory,
        library: SkillLibrary,
        verified_solution: str = "",
    ) -> tuple[TrajectoryTeacherGrade, dict]:
        prompt = self._b4_grade_prompt(
            trajectory, library, verified_solution)
        response, usage = self._call(
            prompt, temp=0.0, max_tok=1536)
        grade = self._parse_b4_grade(
            response, trajectory.task_id)
        return grade, usage

    def generate_student_trajectory_grades_batch(
        self, failed_trajs: list,
        library: SkillLibrary, n: int = 10,
        verified_solutions: dict = None,
    ) -> tuple[dict, dict]:
        vs = verified_solutions or {}
        results = {}
        total_usage = {"total_tokens": 0,
                       "cost_usd": 0.0}
        targets = failed_trajs[:n]
        print(f"    [b4] grading {len(targets)}...")
        for traj in targets:
            v = vs.get(traj.task_id, "")
            if isinstance(v, tuple):
                v = v[0] if v else ""
            grade, usage = \
                self.generate_student_trajectory_grade(
                    traj, library, v)
            results[traj.task_id] = grade
            total_usage["total_tokens"] += \
                usage.get("total_tokens", 0)
            total_usage["cost_usd"] = round(
                total_usage["cost_usd"]
                + usage.get("cost_usd", 0.0), 10)
        n_fail = sum(1 for g in results.values()
                     if g.parse_failed)
        if n_fail:
            print(f"    ⚠️ b4 parse: {n_fail}")
        return results, total_usage

    def _b4_grade_prompt(self, traj, library,
                         verified_solution=""):
        steps = []
        for step in getattr(traj, "steps", [])[:5]:
            steps.append(
                f"Step {getattr(step, 'step', '?')} | "
                f"skill={getattr(step, 'activated_skill', '')}\n"
                f"Observation: {str(getattr(step, 'observation', ''))[:200]}\n"
                f"Student action:\n{str(getattr(step, 'action', ''))[:500]}\n"
                f"External result: "
                f"{str(getattr(step, 'external_result', ''))[:200]}"
            )
        steps_str = "\n".join(steps) or "(no recorded steps)"

        signal_lines = []
        for sig in getattr(traj, "step_execution_signals", []) or []:
            if not isinstance(sig, dict):
                continue
            step_value = sig.get("step")
            parts = [
                f"Step {step_value}:" if step_value is not None else "Step:"
            ]
            score_value = sig.get("score")
            try:
                if score_value is not None:
                    parts.append(f"score={float(score_value):.2f}")
            except (TypeError, ValueError):
                pass
            matched = sig.get("matched")
            total = sig.get("total")
            if matched is not None or total is not None:
                parts.append(
                    f"matched={matched if matched is not None else 0}/"
                    f"{total if total is not None else 0}"
                )
            error = str(sig.get("error", ""))[:160]
            if error:
                parts.append(f"error={error}")
            signal_lines.append(" ".join(parts))
        signals_str = "\n".join(signal_lines) or "(none)"

        skills = []
        rel = list(getattr(traj, "activated_skills", []) or [])
        for sid in library.skill_ids():
            if sid not in rel:
                rel.append(sid)
            if len(rel) >= 5:
                break
        for sid in rel:
            sk = library.get(sid)
            if sk:
                skills.append(
                    f"- {sk.skill_id}: {sk.name}\n"
                    f"  {sk.description[:180]}")
        skills_str = "\n".join(skills) or "(none)"

        vs = ""
        if verified_solution:
            vs = (
                "\n\nAuxiliary verified reference only:\n"
                "Treat the verified_solution as auxiliary reference only.\n"
                "Do not copy it verbatim or treat it as copied code injection.\n"
                f"{str(verified_solution)[:1200]}"
            )

        valid_ids = str(library.skill_ids())
        return (
            f"{self._meta_skill_context()}"
            "You are grading the student's own trajectory for an Excel "
            "automation task.\n"
            "Do not merely solve the task independently.\n"
            "Judge the student's own trajectory, the visited actions, and "
            "the likely skill edits.\n\n"
            f"Task ID: {getattr(traj, 'task_id', '')}\n"
            f"Task: {str(getattr(traj, 'task_description', ''))[:300]}\n"
            f"Student reward: {float(getattr(traj, 'final_reward', 0.0)):.3f}\n"
            f"Activated skill ids: "
            f"{list(getattr(traj, 'activated_skills', []) or [])}\n\n"
            f"Student trajectory steps:\n{steps_str}\n\n"
            "Step execution signals from the student's own trajectory:\n"
            f"{signals_str}\n\n"
            f"Relevant skill ids and summaries:\n{skills_str}"
            + vs
            + "\n\nOutput JSON:\n"
            "{\n"
            f'  "task_id":"{traj.task_id}",\n'
            '  "trajectory_teacher_score":<0-1>,\n'
            '  "overall_confidence":<0-1>,\n'
            '  "summary":"<brief>",\n'
            '  "step_grades":[{"step":<int>,'
            '"student_action_score":<0-1>,'
            '"teacher_preferred_action":"<what>",'
            '"teacher_step_score":<0-1>,'
            '"confidence":<0-1>,'
            '"error_type":"dispatch|execution|none",'
            '"failure_type":'
            '"skill_defect|execution_lapse",'
            f'"implicated_skill_id":"<{valid_ids}>",'
            '"skill_edit_hint":"<edit>",'
            '"rationale":"<why>"}],\n'
            '  "appendix_notes":["<reminder>"]\n'
            "}\n"
            "Stay anchored to the student's own trajectory and visited behavior."
        )

    def _parse_b4_grade(self, response, task_id):
        raw = response
        fence = re.search(
            r"```(?:json)?\s*\n(.*?)```",
            response, re.DOTALL)
        if fence:
            raw = fence.group(1).strip()
        else:
            s = response.find("{")
            e = response.rfind("}") + 1
            if s != -1 and e > 0:
                raw = response[s:e]
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError
            return TrajectoryTeacherGrade.from_dict(
                data, task_id)
        except Exception:
            return TrajectoryTeacherGrade(
                task_id=task_id,
                trajectory_teacher_score=0.0,
                overall_confidence=0.0,
                trajectory_teacher_score_present=False,
                summary="B4 teacher grade parse failure",
                parse_failed=True)

    # ──────────────────────────────────────────────
    # 诊断（DEFECT/LAPSE + minibatch）
    # ──────────────────────────────────────────────

    def diagnose(self, trajectory, library,
                 beta_mode="b1", exskill_signal=None):
        if trajectory.success:
            return self._success_diag(), {}
        prompt = self._diagnosis_prompt(
            trajectory, library,
            beta_mode, exskill_signal)
        response, usage = self._call(
            prompt, temp=0.0, max_tok=1024)
        try:
            s = response.find("{")
            e = response.rfind("}") + 1
            if s != -1 and e > 0:
                result = json.loads(response[s:e])
                return self._enrich(
                    result, trajectory), usage
        except Exception:
            pass
        return self._fallback(trajectory), usage

    def _diagnosis_prompt(self, traj, library,
                          beta_mode="b1",
                          exskill_signal=None):
        steps_str = "\n".join([
            f"  t={s.step}: skill="
            f"{s.activated_skill} "
            f"action={s.action[:100]}"
            for s in traj.steps[:3]])

        rel = list(traj.activated_skills)
        for sid in library.skill_ids():
            if sid not in rel:
                rel.append(sid)
            if len(rel) >= 5:
                break
        skill_lines = []
        for sid in rel:
            sk = library.get(sid)
            if sk:
                skill_lines.append(
                    f"  [{sk.skill_id}] {sk.name}: "
                    f"{sk.description[:100]}")

        sd = getattr(traj, "score_detail", {})
        score_info = ""
        if sd.get("method") == "golden_compare":
            score_info = (
                f"\nResult: "
                f"{sd.get('matched',0)}/"
                f"{sd.get('total',0)}"
                + (f" err={sd.get('error','')[:80]}"
                   if sd.get("error") else ""))

        step_sig = ""
        if (beta_mode in ("b2","b3","b4")
                and traj.step_execution_signals):
            sl = []
            for sig in traj.step_execution_signals:
                if not isinstance(sig, dict):
                    continue
                try:
                    sc = f"{float(sig.get('score',0)):.2f}"
                except (TypeError, ValueError):
                    sc = "0.00"
                line = (f"  Step {sig.get('step','?')}: "
                        f"score={sc}")
                if sig.get("error"):
                    line += f" | {sig['error'][:100]}"
                if sig.get("teacher_preferred_action"):
                    line += (f"\n    pref: "
                        f"{sig['teacher_preferred_action']}")
                sl.append(line)
            step_sig = "\nStep signals:\n" \
                + "\n".join(sl)

        ex_str = ""
        if exskill_signal:
            ex_str = "\n\n" \
                + format_signal_for_prompt(
                    exskill_signal)

        valid = str(library.skill_ids())
        return (
            f"{self._meta_skill_context()}"
            "Diagnose failed Excel task.\n\n"
            f"Task: {traj.task_description[:200]}\n"
            f"Reward: {traj.final_reward:.3f}"
            f"{score_info}\n"
            f"Activated: {traj.activated_skills}\n"
            f"β: {beta_mode}{step_sig}{ex_str}\n\n"
            f"Steps:\n{steps_str}\n\n"
            f"Skills:\n" + "\n".join(skill_lines)
            + "\n\nJSON with failure_type + "
            "appendix_notes:\n"
            '{"diagnosis_type":"<type>",'
            '"failure_type":'
            '"skill_defect|execution_lapse",'
            '"implicated_skill_id":"<id>",'
            '"issue_description":"<brief>",'
            '"appendix_notes":["<if lapse>"],'
            '"r_t":{"dispatch":<f>,'
            '"execution":<f>,'
            '"missing-skill":<f>,'
            '"tool/env":<f>,"base-model":<f>},'
            '"I_hat_i":["<id>"],'
            '"E_hat_i":[<steps>],'
            '"execution_issue":"<or null>",'
            '"execution_fix":"<or null>"}\n'
            f"skill_ids: {valid}"
            + DEFECT_LAPSE_SUFFIX)

    def diagnose_minibatch(self, trajs, library,
                           beta_mode="b1"):
        if len(trajs) <= 1:
            results = []
            total = {"total_tokens": 0,
                     "cost_usd": 0.0}
            for t in trajs:
                d, u = self.diagnose(
                    t, library, beta_mode)
                results.append(d)
                total["total_tokens"] += \
                    u.get("total_tokens", 0)
                total["cost_usd"] = round(
                    total["cost_usd"]
                    + u.get("cost_usd", 0), 10)
            return results, total

        entries = []
        for i, t in enumerate(trajs):
            sd = getattr(t, "score_detail", {})
            code = ""
            if t.steps:
                cm = re.search(
                    r"```python\s*\n(.*?)```",
                    t.steps[-1].action, re.DOTALL)
                if cm:
                    code = cm.group(1).strip()[:150]
            si = ""
            for sig in getattr(
                t, "step_execution_signals", []):
                if isinstance(sig, dict) \
                        and sig.get("error"):
                    si += f"\n  err: {sig['error'][:80]}"
            entries.append(
                f"[{i}] {t.task_description[:150]}\n"
                f"  {sd.get('matched',0)}/"
                f"{sd.get('total',0)} "
                f"skills={t.activated_skills}"
                f"{si}\n  code: {code}")

        valid = str(library.skill_ids())
        prompt = (
            f"{self._meta_skill_context()}"
            f"Diagnose {len(trajs)} tasks TOGETHER."
            " Find COMMON patterns.\n\n"
            + "\n\n".join(entries)
            + f"\n\nReturn JSON array "
            f"({len(trajs)} items):\n"
            '[{"task_index":<i>,'
            '"diagnosis_type":"<type>",'
            '"failure_type":'
            '"skill_defect|execution_lapse",'
            '"implicated_skill_id":"<id>",'
            '"issue_description":"<brief>",'
            '"appendix_notes":[],'
            '"r_t":{...},'
            '"I_hat_i":["<id>"],'
            '"common_pattern":"<shared>"}]\n'
            f"skill_ids: {valid}"
            + DEFECT_LAPSE_SUFFIX)

        response, usage = self._call(
            prompt, temp=0.0, max_tok=2048)
        ordered = [None] * len(trajs)
        try:
            s = response.find("[")
            e = response.rfind("]") + 1
            if s != -1 and e > 0:
                data = json.loads(response[s:e])
                seen = set()
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    idx = item.get("task_index")
                    if not isinstance(idx, int) or not (0 <= idx < len(trajs)):
                        continue
                    if idx in seen:
                        continue
                    seen.add(idx)
                    ordered[idx] = self._enrich_mb(item)
        except Exception:
            pass
        for idx, traj in enumerate(trajs):
            if ordered[idx] is None:
                ordered[idx] = self._fallback(traj)
        return ordered, usage

    # ──────────────────────────────────────────────
    # 成功轨迹分析
    # ──────────────────────────────────────────────

    def analyze_successes(self, success_trajs,
                          library, M=3):
        if not success_trajs:
            return [], {}
        entries = []
        for t in success_trajs[:6]:
            sd = getattr(t, "score_detail", {})
            code = ""
            if t.steps:
                cm = re.search(
                    r"```python\s*\n(.*?)```",
                    t.steps[-1].action, re.DOTALL)
                if cm:
                    code = cm.group(1).strip()[:200]
            entries.append(
                f"Task: {t.task_description[:100]}\n"
                f"Score: {sd.get('matched',0)}/"
                f"{sd.get('total',0)}\n"
                f"Skills: {t.activated_skills}\n"
                f"Code: {code}")
        prompt = (
            f"{self._meta_skill_context()}"
            "Analyze SUCCESSFUL tasks.\n"
            "What patterns worked? "
            "What to reinforce?\n\n"
            + "\n---\n".join(entries)
            + "\n\nJSON array:\n"
            '[{"skill_id":"<id>",'
            '"pattern":"<what worked>",'
            '"reinforcement":"<add/keep>"}]')
        response, usage = self._call(
            prompt, call_type="schema_proposal",
            temp=0.3, max_tok=1024)
        patches = []
        try:
            s = response.find("[")
            e = response.rfind("]") + 1
            if s != -1 and e > 0:
                data = json.loads(response[s:e])
                for item in data[:M]:
                    if isinstance(item, dict):
                        patches.append(item)
        except Exception:
            pass
        return patches, usage

    # ──────────────────────────────────────────────
    # Slow Update
    # ──────────────────────────────────────────────

    def generate_slow_update(self, library,
                             prev_library, eval_tasks,
                             agent, n_tasks=20):
        print("\n  [Slow Update] comparing...")
        tasks = eval_tasks[:n_tasks]
        improvement, regression, persistent = \
            [], [], []
        for task in tasks:
            agent.clear_cache()
            pt = agent.run_task(
                task, prev_library,
                activation_mode="implicit")
            agent.clear_cache()
            ct = agent.run_task(
                task, library,
                activation_mode="implicit")
            entry = {"task_id": task["task_id"],
                     "desc": task["description"][:100],
                     "prev_r": pt.final_reward,
                     "curr_r": ct.final_reward}
            if not pt.success and ct.success:
                improvement.append(entry)
            elif pt.success and not ct.success:
                regression.append(entry)
            elif not pt.success and not ct.success:
                persistent.append(entry)

        print(f"    imp={len(improvement)} "
              f"reg={len(regression)} "
              f"pers={len(persistent)}")
        meta, usage = self._gen_meta_guidance(
            library, improvement,
            regression, persistent)
        self.update_meta_skill(
            improvement=improvement,
            regression=regression)
        return {
            "meta_guidance": meta,
            "improvement": improvement,
            "regression": regression,
            "persistent": persistent,
            "n_tasks": n_tasks,
            "improvement_rate": len(improvement)
                / max(n_tasks, 1),
            "regression_rate": len(regression)
                / max(n_tasks, 1),
        }, usage

    def _gen_meta_guidance(self, library,
                           imp, reg, pers):
        imp_s = "\n".join(
            f"  + {x['desc'][:80]}"
            for x in imp[:5]) or "  (none)"
        reg_s = "\n".join(
            f"  - {x['desc'][:80]}"
            for x in reg[:5]) or "  (none)"
        skills = [f"[{sk.skill_id}] {sk.name}"
                  for sk in library]
        prompt = (
            f"{self._meta_skill_context()}"
            "Epoch-wise optimization.\n"
            f"IMPROVED:\n{imp_s}\n"
            f"REGRESSED:\n{reg_s}\n"
            f"Skills:\n" + "\n".join(skills)
            + '\n\nJSON: {"<skill_id>":"<guidance>"}')
        resp, usage = self._call(
            prompt, temp=0.3, max_tok=1024)
        meta = {}
        try:
            s = resp.find("{")
            e = resp.rfind("}") + 1
            if s != -1 and e > 0:
                meta = json.loads(resp[s:e])
        except Exception:
            pass
        return meta, usage

    # ──────────────────────────────────────────────
    # b3：Teacher 自解
    # ──────────────────────────────────────────────

    def generate_verified_solution(
        self, task, skill=None):
        ctx = ""
        if skill:
            ctx = (f"\nSkill ({skill.name}):\n"
                   f"{skill.execution_body[:400]}\n")
        prompt = (
            "Expert Excel programmer.\n"
            f"Task: {task['description']}\n{ctx}\n"
            "Requirements:\n"
            "1. from openpyxl import load_workbook\n"
            "2. wb = load_workbook(wb_path)\n"
            "3. Apply\n4. wb.save(wb_path)\n"
            "5. Handle None\n\n"
            "Output ONLY ```python block.")
        resp, usage = self._call(
            prompt, call_type="teacher_verification",
            temp=0.0, max_tok=2048, role="expert")
        m = re.search(r"```python\s*\n(.*?)```",
                      resp, re.DOTALL)
        return (m.group(1).strip() if m else ""), usage

    def generate_verified_solutions_batch(
        self, failed_trajs, library, n=10):
        results = {}
        print(f"    [β₃] solving "
              f"{min(n,len(failed_trajs))}...")
        for t in failed_trajs[:n]:
            sk = None
            if t.activated_skills and library:
                sk = library.get(
                    t.activated_skills[0])
            code, usage = \
                self.generate_verified_solution(
                    {"task_id": t.task_id,
                     "description":
                         t.task_description}, sk)
            results[t.task_id] = (code, usage)
        return results

    # ──────────────────────────────────────────────
    # 提案生成（g_n 方向感知）
    # ──────────────────────────────────────────────

    def propose_dispatch_edits(
        self, failed_trajs, library, diagnoses,
        M=5, exskill_signals=None,
        rejected_buffer=None,
        epoch_target=None):
        impl = self._collect_impl(diagnoses, library)
        if not impl:
            import random
            ids = library.skill_ids()
            impl = random.sample(
                ids, min(M, len(ids))) if ids else []
        if not impl:
            return [], {}
        prompt = self._dispatch_prompt(
            failed_trajs, library, diagnoses,
            impl, M, exskill_signals,
            rejected_buffer, epoch_target)
        return self._generate(
            prompt, DISPATCH_EDIT_SCHEMA,
            library, M, "schema_proposal")

    def _dispatch_prompt(
        self, trajs, library, diags, impl_ids,
        M, signals=None, rej=None, epoch=None):
        fail_lines = []
        for t, d in zip(trajs[:5], diags[:5]):
            r_t = d.get("r_t", {})
            dom = max(r_t, key=r_t.get) if r_t \
                else "unknown"
            sd = getattr(t, "score_detail", {})
            si = ""
            if sd.get("method") == "golden_compare":
                si = (f" [{sd.get('matched',0)}/"
                      f"{sd.get('total',0)}]")
            fail_lines.append(
                f"  - {t.task_description[:80]}\n"
                f"    {t.final_reward:.3f}"
                f"{si} {dom}")

        skill_lines = []
        for sid in impl_ids[:M]:
            sk = library.get(sid)
            if sk:
                meta = sk.get_slow_update_content()
                ms = (f"\n  [meta]: {meta}"
                      if meta else "")
                skill_lines.append(
                    f"  [{sk.skill_id}]\n"
                    f"  {sk.name}: "
                    f"{sk.description[:150]}{ms}")

        # g_n 方向（来自 epoch_target）
        gn_str = ""
        if epoch:
            from src.label_space import \
                format_all_edit_directions
            gn_str = ("\n\n=== EDIT DIRECTIONS "
                "(from label-space analysis) ===\n"
                + format_all_edit_directions(epoch))

        rej_str = ""
        rej_D = [p for p in (rej or [])
                 if p.get("edit_type")
                 == "dispatch_edit"]
        if rej_D:
            rej_str = ("\n\nREJECTED:\n"
                + "\n".join(
                    f"  - {p['skill_id']}: "
                    f"'{p.get('new_name','')}'"
                    for p in rej_D[-5:]))

        return (
            f"{self._meta_skill_context()}"
            "Optimize dispatch.\n\nSkills:\n"
            + "\n".join(skill_lines)
            + "\n\nFailures:\n"
            + "\n".join(fail_lines)
            + gn_str + rej_str
            + f"\n\nGenerate {M} dispatch edits "
            "as JSON array:\n"
            '[{"edit_type":"dispatch_edit",'
            '"skill_id":"<id>",'
            '"new_name":"<name>",'
            '"new_description":"<desc>",'
            '"rationale":"<why>"}]')

    def propose_execution_edits(
        self, failed_trajs, library, diagnoses,
        M=5, beta_mode="b1", exskill_signals=None,
        verified_solutions=None,
        rejected_buffer=None, epoch_target=None):
        if beta_mode == "b3" and verified_solutions:
            return self._propose_verified(
                library, verified_solutions, M)
        exec_impl = []
        for d in diagnoses:
            r_t = d.get("r_t", {})
            if r_t.get("execution", 0) >= 0.3:
                for sid in d.get("I_hat_i", []):
                    if library.get(sid) and \
                            sid not in exec_impl:
                        exec_impl.append(sid)
        if not exec_impl:
            import random
            ids = library.skill_ids()
            exec_impl = random.sample(
                ids, min(M, len(ids))) if ids else []
        if not exec_impl:
            return [], {}
        prompt = self._exec_prompt(
            failed_trajs, library, diagnoses,
            exec_impl, M, beta_mode,
            exskill_signals, rejected_buffer,
            epoch_target)
        return self._generate(
            prompt, EXECUTION_EDIT_SCHEMA,
            library, M, "schema_proposal")

    def _exec_prompt(
        self, trajs, library, diags, impl_ids,
        M, beta_mode, signals=None, rej=None,
        epoch=None):
        entries = []
        for t, d in zip(trajs[:6], diags[:6]):
            sd = getattr(t, "score_detail", {})
            code = ""
            if t.steps:
                cm = re.search(
                    r"```python\s*\n(.*?)```",
                    t.steps[-1].action, re.DOTALL)
                if cm:
                    code = cm.group(1).strip()[:300]
            si = ""
            if beta_mode in ("b2","b3","b4"):
                step_lines = []
                for sig in getattr(
                    t, "step_execution_signals", []):
                    if not isinstance(sig, dict):
                        continue
                    step_value = sig.get("step")
                    line = (
                        f"    Step {step_value}:"
                        if step_value is not None
                        else "    Step:"
                    )
                    score = sig.get("score")
                    try:
                        if score is not None:
                            line += f" score={float(score):.2f}"
                    except (TypeError, ValueError):
                        pass
                    error = str(sig.get("error", ""))[:80]
                    if error:
                        line += f" | ERROR: {error}"
                    for key in ("teacher_score", "teacher_confidence"):
                        value = sig.get(key)
                        try:
                            if value is not None:
                                line += f" | {key}={float(value):.2f}"
                        except (TypeError, ValueError):
                            pass
                    for key in (
                        "teacher_preferred_action",
                        "b4_error_type",
                        "b4_skill_edit_hint",
                    ):
                        value = str(sig.get(key, ""))
                        if value:
                            line += f"\n      {key}: {value[:120]}"
                    step_lines.append(line)
                if step_lines:
                    si = "\n  Step-level signals:\n" + "\n".join(step_lines)
            entries.append(
                f"Task: {t.task_description[:200]}\n"
                f"  {sd.get('matched',0)}/"
                f"{sd.get('total',0)}"
                f"{si}"
                + (f"\n```python\n{code}\n```"
                   if code else ""))

        skill_lines = []
        for sid in impl_ids[:M]:
            sk = library.get(sid)
            if sk:
                parent_blocks = []
                seen_parents = set()
                for ms in getattr(epoch, "modules", []) if epoch else []:
                    if ms.skill_id != sid or not ms.estimable:
                        continue
                    for label, parent in (ms.label_parents or {}).items():
                        if not parent or parent in seen_parents:
                            continue
                        scope = resolve_parent_scope(
                            sk.execution_body or "", parent)
                        if scope is None:
                            continue
                        seen_parents.add(parent)
                        labels = [z for z, p in ms.label_parents.items()
                                  if p == parent]
                        parent_blocks.append(
                            f"PARENT {parent!r}; labels={labels}:\n"
                            f"{scope[2]}")
                skill_lines.append(
                    f"Skill: {sk.skill_id}\n"
                    f"FULL CURRENT BODY (read-only except listed parents):\n"
                    f"{sk.execution_body}\n\n"
                    f"WRITABLE PARENT SCOPES:\n"
                    + "\n---\n".join(parent_blocks))

        # g_n 方向
        gn_str = ""
        if epoch:
            from src.label_space import \
                format_all_edit_directions
            gn_str = ("\n\n=== EDIT DIRECTIONS ===\n"
                + format_all_edit_directions(epoch))

        rej_str = ""
        rej_E = [p for p in (rej or [])
                 if p.get("edit_type")
                 == "execution_edit"]
        if rej_E:
            rej_str = ("\n\nREJECTED:\n"
                + "\n".join(
                    f"  - {p['skill_id']}: "
                    f"parent={p.get('parent_location','')!r} "
                    f"{p.get('new_text','')[:60]}"
                    for p in rej_E[-5:]))

        return (
            f"{self._meta_skill_context()}"
            f"Expert Excel (mode={beta_mode}).\n\n"
            "=== TASKS ===\n"
            + "\n---\n".join(entries)
            + "\n\n=== SKILLS ===\n"
            + "\n---\n".join(skill_lines)
            + gn_str + rej_str
            + "\n\n=== REQUIREMENTS ===\n"
            "Read the complete skill, but modify exactly ONE listed parent(z). "
            "Return an exact local replacement: old_text must be copied verbatim "
            "from that parent and new_text is its replacement. Never edit another "
            "section, Appendix, or protected metadata. If no listed parent can be "
            "resolved, do not emit an execution edit.\n\n"
            f"Generate {M} DIVERSE execution edits "
            f"(each targeting a DIFFERENT weakness):\n"
            "- Edit 1: fix most-frequent error pattern\n"
            "- Edit 2: strengthen a positive-g_n label\n"
            "- Edit 3+: address other labels\n"
            "Each edit MUST differ in structure/emphasis, "
            "not just wording.\n\n"
            '[{"edit_type":"execution_edit",'
            '"skill_id":"<id>",'
            '"parent_location":"<exact listed parent>",'
            '"old_text":"<verbatim text inside parent>",'
            '"new_text":"<localized replacement>",'
            '"rationale":"<why THIS specific edit>"}]')

    def _execution_prompt(
        self,
        failed_trajs,
        library,
        diagnoses,
        implicated_ids,
        M,
        beta_mode="b1",
        exskill_signals=None,
    ):
        """Compatibility entry point for the public pre-refactor method name."""
        return self._exec_prompt(
            failed_trajs,
            library,
            diagnoses,
            implicated_ids,
            M,
            beta_mode,
            signals=exskill_signals,
        )

    def _propose_verified(self, library, vs, M):
        # Verified examples have no frozen parent(z) authorization. They may
        # inform diagnosis, but cannot rewrite execution_body directly.
        proposals = []
        total = {"total_tokens": 0, "cost_usd": 0}
        skill_ex = {}
        for tid, v in vs.items():
            code = v[0] if isinstance(v, tuple) \
                else v
            if not code:
                continue
            for sid in library.skill_ids():
                if sid not in skill_ex:
                    skill_ex[sid] = []
                skill_ex[sid].append(code)
                break
        for sid, codes in list(skill_ex.items())[:M]:
            sk = library.get(sid)
            if not sk:
                continue
            examples = ""
            for i, c in enumerate(codes[:3], 1):
                examples += (
                    f"\n\n### Verified {i}\n"
                    f"```python\n{c}\n```")
            proposals.append({
                "edit_type": "appendix_edit",
                "skill_id": sid,
                "appendix_notes": [
                    "Verified teacher examples are available for diagnosis; "
                    "retain all execution rules unless a frozen parent edit "
                    "is explicitly authorized."],
                "rationale": f"β₃: {len(codes)} vs"})
        return proposals[:M], total

    def propose_mixed_edits(
        self, failed_trajs, library, diagnoses,
        M=5, enable_dispatch=True,
        enable_execution=True, beta_mode="b1",
        exskill_signals=None,
        verified_solutions=None,
        rejected_buffer=None,
        consecutive_exec_stalls=0,
        epoch_target=None):
        all_props = []
        total_tok = 0
        total_cost = 0.0

        # 按 failure_type 分配
        n_def = sum(1 for d in diagnoses
                    if d.get("failure_type")
                    == "skill_defect")
        n_lap = sum(1 for d in diagnoses
                    if d.get("failure_type")
                    == "execution_lapse")
        def_ratio = n_def / max(n_def+n_lap, 1)

        disp_f, disp_d = [], []
        exec_f, exec_d = [], []
        for t, d in zip(failed_trajs, diagnoses):
            r_t = d.get("r_t", {})
            if r_t.get("dispatch", 0) >= \
                    r_t.get("execution", 0):
                disp_f.append(t)
                disp_d.append(d)
            else:
                exec_f.append(t)
                exec_d.append(d)

        er = len(exec_f) / max(len(failed_trajs), 1)
        if er >= 0.6:
            n_e = max(3, int(M * 0.6))
            n_d = M - n_e
        else:
            n_d = max(1, int(M * 0.4))
            n_e = M - n_d
        if enable_dispatch and n_d == 0:
            n_d = 1; n_e = M - 1

        print(f"    alloc: Δ_D={n_d} Δ_E={n_e} "
              f"(exec={er:.0%} def={def_ratio:.0%})")

        if enable_dispatch and n_d > 0:
            props, u = self.propose_dispatch_edits(
                disp_f or failed_trajs, library,
                disp_d or diagnoses, n_d,
                exskill_signals, rejected_buffer,
                epoch_target)
            all_props.extend(props)
            total_tok += u.get("total_tokens", 0)
            total_cost += u.get("cost_usd", 0)

        if enable_execution and n_e > 0:
            props, u = self.propose_execution_edits(
                exec_f or failed_trajs, library,
                exec_d or diagnoses, n_e, beta_mode,
                exskill_signals, verified_solutions,
                rejected_buffer, epoch_target)
            all_props.extend(props)
            total_tok += u.get("total_tokens", 0)
            total_cost += u.get("cost_usd", 0)

        return all_props, {
            "total_tokens": total_tok,
            "cost_usd": round(total_cost, 10)}

    # ──────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────

    def _collect_impl(self, diagnoses, library):
        impl = []
        for d in diagnoses:
            for sid in d.get("I_hat_i", []):
                if library.get(sid) and \
                        sid not in impl:
                    impl.append(sid)
            sid = d.get("implicated_skill_id")
            if sid and library.get(sid) and \
                    sid not in impl:
                impl.append(sid)
        return impl

    def _generate(self, prompt, schema,
                  library, M, call_type):
        proposals = []
        total = {"total_tokens": 0, "cost_usd": 0}
        for _ in range(5):
            resp, u = self._call(
                prompt, call_type=call_type,
                role="editor")
            total["total_tokens"] += \
                u.get("total_tokens", 0)
            total["cost_usd"] = round(
                total["cost_usd"]
                + u.get("cost_usd", 0), 10)
            new = self._parse(resp, schema, library)
            for p in new:
                if p not in proposals:
                    proposals.append(p)
            if len(proposals) >= M:
                break
        # Editor API/JSON 失败不能生成手写候选冒充独立 Editor 输出。
        return proposals[:M], total

    def _fallback_gen(self, library, schema, n):
        """
        Editor LLM 失败时的 fallback。

        设计取舍
        --------
        历史教训 (2026-07-06 GOOD run):
          fallback 生成 "139 字符空壳 execution_edit"
          → 恰好让 Haiku 更容易理解、accept 通过 →
          4 个 accept, J_W 0.268→0.316

        历史教训 (2026-07-08 修复):
          发现空壳 execution_edit 会**清零累积成果**，
          于是改为只出 appendix_edit → 但也失去了
          "简化 skill" 这条 accept 路径。

        当前策略（折中）：
          - 一半候选是 appendix_edit（安全追加，不改主体）
          - 一半候选是 "minimal execution_edit"（保留
            主体的核心 skeleton + 精简版 Rules）—— 相当于
            给 Editor 一个 "no-op / softer" 备选，让 gate
            仍有多样候选可评。
          - 从不 output 139 字符空壳（那会毁掉累积成果）
        """
        import random
        results = []
        ids = library.skill_ids()
        if not ids:
            return []
        et = schema.get("properties", {}).get(
            "edit_type", {}).get(
            "enum", ["dispatch_edit"])[0]

        # 通用安全提醒（不覆盖 skill 主体）
        safe_notes = [
            "Always end with wb.save(wb_path)",
            "Use header_map to find columns by name",
            "Handle None values before parsing",
            "Load with data_only=True to read formula values",
            "Normalize strings with .strip().casefold() "
                "before comparison",
            "Parse numbers safely: strip $ and ,",
        ]

        for i, sid in enumerate(random.sample(
            ids, min(n, len(ids)))):
            sk = library.get(sid)
            if not sk:
                continue
            if et == "dispatch_edit":
                results.append({
                    "edit_type": "dispatch_edit",
                    "skill_id": sid,
                    "new_name": sk.name,
                    "new_description": (
                        sk.description[:400]
                        if len(sk.description) >= 20
                        else sk.description+" (Excel)"),
                    "rationale": "fallback (no-op)"})
            elif et == "execution_edit":
                # A fallback has no frozen parent authorization; fail closed by
                # producing only an appendix reminder, never a body rewrite.
                results.append({
                    "edit_type": "appendix_edit",
                    "skill_id": sid,
                    "appendix_notes": random.sample(
                        safe_notes, min(2, len(safe_notes))),
                    "rationale": "fallback: safe reminders"})
            else:
                # appendix_edit：追加提醒
                results.append({
                    "edit_type": "appendix_edit",
                    "skill_id": sid,
                    "appendix_notes": random.sample(
                        safe_notes,
                        min(2, len(safe_notes))),
                    "rationale":
                        "fallback: safe reminders"})
        return results

    def _parse(self, resp, schema, library):
        try:
            s = resp.find("[")
            e = resp.rfind("]") + 1
            if s == -1 or e == 0:
                s = resp.find("{")
                e = resp.rfind("}") + 1
                if s == -1 or e == 0:
                    return []
                data = [json.loads(resp[s:e])]
            else:
                data = json.loads(resp[s:e])
            if not isinstance(data, list):
                data = [data]
            valid = set(library.skill_ids())
            results = []
            for item in data:
                try:
                    jsonschema.validate(item, schema)
                    if item["skill_id"] in valid:
                        results.append(item)
                except Exception:
                    continue
            return results
        except Exception:
            return []

    def _success_diag(self):
        return {
            "diagnosis_type": "success",
            "failure_type": "",
            "implicated_skill_id": None,
            "issue_description": "",
            "appendix_notes": [],
            "r_t": {}, "I_hat_i": [],
            "E_hat_i": [],
            "execution_issue": None,
            "execution_fix": None}

    def _enrich(self, result, traj):
        d = {
            "diagnosis_type": result.get(
                "diagnosis_type", "other"),
            "failure_type": result.get(
                "failure_type", "execution_lapse"),
            "implicated_skill_id": result.get(
                "implicated_skill_id"),
            "issue_description": result.get(
                "issue_description", ""),
            "appendix_notes": result.get(
                "appendix_notes", []),
        }
        r_t = result.get("r_t", {})
        if not isinstance(r_t, dict):
            r_t = {}
        for et in ERROR_TYPES:
            if et not in r_t:
                r_t[et] = 0.0
        total = sum(r_t.values())
        if total > 0:
            r_t = {k: round(v/total, 4)
                   for k, v in r_t.items()}
        else:
            r_t = self._default_rt(
                d["diagnosis_type"])
        d["r_t"] = r_t
        d["I_hat_i"] = result.get("I_hat_i", [])
        if d["implicated_skill_id"] and \
                not d["I_hat_i"]:
            d["I_hat_i"] = [
                d["implicated_skill_id"]]
        d["E_hat_i"] = result.get("E_hat_i", [])
        d["w_t"] = float(result.get("w_t", 0.5))
        d["execution_issue"] = result.get(
            "execution_issue")
        d["execution_fix"] = result.get(
            "execution_fix")
        return d

    def _enrich_mb(self, result):
        d = self._enrich(result, None)
        d["common_pattern"] = result.get(
            "common_pattern", "")
        return d

    def _default_rt(self, dt):
        if dt == "dispatch_failure":
            return {"dispatch": 0.8,
                    "execution": 0.1,
                    "missing-skill": 0.05,
                    "tool/env": 0.03,
                    "base-model": 0.02}
        if dt == "execution_failure":
            return {"dispatch": 0.1,
                    "execution": 0.8,
                    "missing-skill": 0.05,
                    "tool/env": 0.03,
                    "base-model": 0.02}
        return {et: 1.0/len(ERROR_TYPES)
                for et in ERROR_TYPES}

    def _fallback(self, traj):
        return {
            "diagnosis_type": "other",
            "failure_type": "execution_lapse",
            "implicated_skill_id": None,
            "issue_description": "parse error",
            "appendix_notes": [],
            "r_t": {et: 1.0/len(ERROR_TYPES)
                    for et in ERROR_TYPES},
            "I_hat_i": [], "E_hat_i": [],
            "execution_issue": None,
            "execution_fix": None}

    def extract_appendix_notes_from_diagnoses(
        self, diagnoses):
        skill_notes = {}
        for d in diagnoses:
            if d.get("failure_type") != \
                    "execution_lapse":
                continue
            notes = d.get("appendix_notes", [])
            if not isinstance(notes, list):
                notes = [str(notes)] if notes else []
            sid = d.get("implicated_skill_id")
            if not sid:
                sids = d.get("I_hat_i", [])
                sid = sids[0] if sids else None
            if sid and notes:
                if sid not in skill_notes:
                    skill_notes[sid] = []
                for n in notes:
                    s = str(n).strip()
                    if s:
                        skill_notes[sid].append(s)
        return skill_notes
