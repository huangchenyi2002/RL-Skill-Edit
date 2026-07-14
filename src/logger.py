# src/logger.py
"""
OSD Structured Event Logger
============================

设计原则
--------
1. **一次记录，多处可用**：JSONL 事件流是"事实"，`.log` 是"给人看的摘要"，
   可选的 `prompts/` 目录是"完整交互 dump"（默认关闭以省磁盘）
2. **与主循环共生**：所有事件都能对应 `study1_main.py` 里的一个可观测节点
3. **面向后分析**：JSONL schema 稳定，study2/study3 可以直接 `pd.read_json(lines=True)`
4. **不装饰**:不打印装饰框——太多信号会掩盖重要事件

事件类型
--------
- session_start / session_end
- round_start / round_end
- parser_output      Parser LLM 抽出的 modules / labels
- rollout_summary    单轮 Student rollout 汇总（sr / R̄ / 用时）
- diagnose_summary   Teacher 诊断汇总（defect / lapse 分布、r_t）
- grade_summary      Teacher b4 分步评分汇总（sr / parse_failure）
- epoch_target       λ / β / donor pair / g_n top-k
- proposal_batch     Editor LLM 生成的候选集合
- candidate_gate     单个候选的 dual gate 结果
- accept / stall
- slow_update        含 imp/reg/是否 skipped
- witness_eval       Witness rollout J_W 变化
- error              任何异常/降级
- llm_interaction    可选：完整 LLM prompt/response

可选 dump（dump_prompts=True 时）
---------------------------------
- logs/<run>/prompts/rNN_<role>_<tid>.txt
  完整 prompt + response，便于事后 debug LLM 行为
"""

import os
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import Any


# ─────────────────────────────────────────────────
# JSON safe serializer
# ─────────────────────────────────────────────────

def _json_default(obj):
    """把 numpy / dataclass / 未知对象降级为字符串"""
    try:
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "__dataclass_fields__"):
            from dataclasses import asdict
            return asdict(obj)
    except Exception:
        pass
    return str(obj)


def _clip(s: Any, n: int = 300) -> str:
    """安全截断字符串"""
    if s is None:
        return ""
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"…<{len(s) - n} more>"


def _preview_dict(d: dict, k: int = 6) -> dict:
    """dict 只保留前 k 个键，浮点保留 4 位"""
    if not d:
        return {}
    out = {}
    for i, (kk, vv) in enumerate(d.items()):
        if i >= k:
            out["…"] = f"+{len(d) - k} more"
            break
        if isinstance(vv, float):
            out[str(kk)] = round(vv, 4)
        else:
            out[str(kk)] = vv
    return out


# ─────────────────────────────────────────────────
# OSDLogger
# ─────────────────────────────────────────────────

class OSDLogger:
    """
    结构化事件日志器。

    Files
    -----
    logs/<experiment>_<ts>.jsonl   事件流（机读，一行一 event）
    logs/<experiment>_<ts>.log     人读摘要（每轮 3-10 行）
    logs/<experiment>_<ts>/prompts/  可选：完整 prompt/response dump
    """

    def __init__(
        self,
        log_dir:       str = "logs",
        experiment:    str = "study1",
        dump_prompts:  bool = False,
        console_level: str = "WARNING",
    ):
        self.log_dir      = log_dir
        self.experiment   = experiment
        self.round_n      = 0
        self.timestamp    = datetime.now().strftime(
            "%Y%m%d_%H%M%S")
        self.run_id = experiment
        self.framework_version = "ooosd-parent-kl-point-v1"
        self.artifact_schema_version = 2
        self.dump_prompts = dump_prompts
        self._t_start     = time.time()
        self._t_round     = time.time()

        os.makedirs(log_dir, exist_ok=True)

        # 事件流（机读）
        self.jsonl_path = os.path.join(
            log_dir,
            f"{experiment}.jsonl",
        )
        # 人读摘要
        self.log_path = os.path.join(
            log_dir,
            f"{experiment}.log",
        )
        # 可选：prompts dump
        if dump_prompts:
            self.dump_dir = os.path.join(
                log_dir,
                f"{experiment}",
                "prompts",
            )
            os.makedirs(self.dump_dir, exist_ok=True)
        else:
            self.dump_dir = None

        # logging.Logger（.log 文件详细，终端默认 WARNING）
        self._log = logging.getLogger(
            f"OSD.{experiment}.{self.timestamp}")
        self._log.setLevel(logging.DEBUG)
        self._log.propagate = False

        if not self._log.handlers:
            fh = logging.FileHandler(
                self.log_path, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-5s %(message)s",
                datefmt="%H:%M:%S"))
            self._log.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setLevel(getattr(
                logging, console_level.upper(),
                logging.WARNING))
            ch.setFormatter(logging.Formatter(
                "  [log] %(message)s"))
            self._log.addHandler(ch)

        self._emit(
            "session_start",
            experiment=experiment,
            run_id=self.run_id,
            framework_version=self.framework_version,
            artifact_schema_version=self.artifact_schema_version,
            timestamp=self.timestamp,
            dump_prompts=dump_prompts,
        )
        self._log.info(
            f"=== SESSION START  {experiment} "
            f"{self.timestamp} ===")
        self._log.info(
            f"JSONL: {self.jsonl_path}")
        if dump_prompts:
            self._log.info(
                f"DUMP:  {self.dump_dir}")

    # ─────────────────────────────────────────────
    # 底层：所有事件都走 _emit
    # ─────────────────────────────────────────────

    def _emit(self, event: str, **fields):
        """
        写一条 JSONL。始终包含：
          event / round / t_elapsed / t_round / ts
        """
        rec = {
            "event":     event,
            "round":     self.round_n,
            "t_elapsed": round(
                time.time() - self._t_start, 2),
            "t_round":   round(
                time.time() - self._t_round, 2),
            "ts":        datetime.now().strftime(
                "%H:%M:%S"),
            "run_id": self.run_id,
            "framework_version": self.framework_version,
            "artifact_schema_version": self.artifact_schema_version,
        }
        rec.update(fields)
        try:
            with open(self.jsonl_path, "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(
                    rec, ensure_ascii=False,
                    default=_json_default) + "\n")
        except Exception as e:
            print(f"  [logger] write failed: {e}")

    def _dump_prompt(
        self, tag: str, task_id: str,
        prompt: str, response: str = "",
    ):
        """可选：把完整 prompt+response 落盘"""
        if not self.dump_dir:
            return
        safe_tid = "".join(
            c if c.isalnum() or c in "-_." else "_"
            for c in (task_id or "unknown"))[:40]
        path = os.path.join(
            self.dump_dir,
            f"r{self.round_n:02d}_{tag}_{safe_tid}.txt",
        )
        try:
            with open(path, "w",
                      encoding="utf-8") as f:
                f.write(f"# {tag} | round={self.round_n}"
                        f" | task={task_id}\n")
                f.write("=" * 60 + "\n")
                f.write("PROMPT:\n")
                f.write((prompt or "") + "\n")
                if response:
                    f.write("=" * 60 + "\n")
                    f.write("RESPONSE:\n")
                    f.write(response + "\n")
        except Exception:
            pass

    def dump_text(self, name: str, content: str):
        """
        对外通用 dump：把任意文本落盘到
        logs/<run>/prompts/rNN_<name>.txt
        用于保存 skill 快照 / diagnose 详情 / etc.
        """
        if not self.dump_dir:
            return
        safe = "".join(
            c if c.isalnum() or c in "-_." else "_"
            for c in str(name))[:80]
        path = os.path.join(
            self.dump_dir,
            f"r{self.round_n:02d}_{safe}.txt",
        )
        try:
            with open(path, "w",
                      encoding="utf-8") as f:
                f.write(content or "")
            return path
        except Exception:
            return None

    # ─────────────────────────────────────────────
    # Round 生命周期
    # ─────────────────────────────────────────────

    def set_round(self, n: int):
        self.round_n = n
        self._t_round = time.time()

    def round_start(
        self, n: int, *,
        J_W: float, stall: int, cur_M: int,
        library_size: int, skill_len: int = 0,
        beta_mode: str = "",
    ):
        """每轮开头调用一次"""
        self.set_round(n)
        self._emit(
            "round_start",
            J_W=round(float(J_W), 4),
            stall=stall,
            cur_M=cur_M,
            library_size=library_size,
            skill_len=skill_len,
            beta_mode=beta_mode,
        )
        self._log.info(
            f"[R{n:02d}] START J_W={J_W:.4f} "
            f"stall={stall} M={cur_M} "
            f"|K|={library_size} "
            f"|skill|={skill_len}c")

    def round_end(
        self, *,
        action: str,
        J_W_after: float,
        delta_hat: float = 0.0,
        n_proposals: int = 0,
        n_admissible: int = 0,
        cost_this_round: float = 0.0,
        **extra,
    ):
        """每轮结尾调用一次"""
        payload = dict(
            action=action,
            J_W_after=round(float(J_W_after), 4),
            delta_hat=round(float(delta_hat), 4),
            n_proposals=n_proposals,
            n_admissible=n_admissible,
            cost=round(float(cost_this_round), 4),
        )
        payload.update(extra)
        self._emit("round_end", **payload)
        icon = {"accept": "✓",
                "stall":  "·",
                "break":  "×"}.get(action, "?")
        self._log.info(
            f"[R{self.round_n:02d}] END   {icon} "
            f"{action:<7s} J_W={J_W_after:.4f} "
            f"Δ={delta_hat:+.4f} "
            f"prop={n_proposals} "
            f"adm={n_admissible} "
            f"${cost_this_round:.3f}")

    # ─────────────────────────────────────────────
    # Parser / Rollout / Diagnose / Grade
    # ─────────────────────────────────────────────

    def parser_output(
        self, *,
        n_labels: int,
        n_modules: int,
        labels: list = None,
        patterns: dict = None,
    ):
        self._emit(
            "parser_output",
            n_labels=n_labels,
            n_modules=n_modules,
            labels=(labels or [])[:20],
            measurement=(patterns or {}),
        )
        self._log.debug(
            f"[R{self.round_n:02d}] parser: "
            f"labels={n_labels} modules={n_modules}")

    def history_mapping(
        self, *, phase: str, complete: bool,
        n_steps: int, assignments: list = None,
        error: str = "", frozen_ruler_digest: str = "",
    ):
        assignments = assignments or []
        statuses = {"assigned": 0, "unassigned": 0, "uncertain": 0}
        multi_history = 0
        for assignment in assignments:
            status = assignment.get("status", "")
            if status in statuses:
                statuses[status] += 1
            if len(assignment.get("history_ids", [])) > 1:
                multi_history += 1
        self._emit(
            "history_mapping", phase=phase, complete=bool(complete),
            n_steps=int(n_steps), status_counts=statuses,
            multi_history_steps=multi_history,
            error=str(error), frozen_ruler_digest=frozen_ruler_digest,
        )

    def rollout_summary(
        self, *,
        n: int,
        success_rate: float,
        mean_reward: float,
        activation_mode: str = "",
        durations_s: float = 0.0,
        per_task: list = None,
    ):
        self._emit(
            "rollout_summary",
            n=n,
            success_rate=round(float(success_rate), 4),
            mean_reward=round(float(mean_reward), 4),
            activation_mode=activation_mode,
            durations_s=round(float(durations_s), 2),
            per_task_preview=(per_task or [])[:5],
        )
        self._log.info(
            f"[R{self.round_n:02d}] rollout: "
            f"n={n} sr={success_rate:.2%} "
            f"R̄={mean_reward:.3f} "
            f"({durations_s:.1f}s)")

    def diagnose_summary(
        self, *,
        n: int,
        n_defect: int,
        n_lapse: int,
        error_type_dist: dict = None,
        r_t_avg: dict = None,
    ):
        self._emit(
            "diagnose_summary",
            n=n,
            n_defect=n_defect,
            n_lapse=n_lapse,
            error_type_dist=error_type_dist or {},
            r_t_avg=_preview_dict(r_t_avg or {}),
        )
        self._log.info(
            f"[R{self.round_n:02d}] diagnose: "
            f"defect={n_defect} lapse={n_lapse}")

    def grade_summary(
        self, *,
        n: int,
        success_rate: float,
        parse_failures: int,
        mean_score: float = 0.0,
    ):
        self._emit(
            "grade_summary",
            n=n,
            success_rate=round(float(success_rate), 4),
            parse_failures=parse_failures,
            mean_score=round(float(mean_score), 4),
        )
        self._log.info(
            f"[R{self.round_n:02d}] b4: "
            f"n={n} sr={success_rate:.2%} "
            f"parse_fail={parse_failures}")

    # ─────────────────────────────────────────────
    # Epoch Target (λ / β / g_n)
    # ─────────────────────────────────────────────

    def epoch_target(
        self, *,
        beta: float,
        modules: list = None,
        target_summary: str = "",
    ):
        """
        每轮 build_epoch_target 后调用，记录完整的
        per-module × per-label 量。

        modules: list[dict]，每个 module 期望字段:
          - module_id / skill_id / d_s / w
          - named_behaviors      (list[str])
          - lambda_              ({label: λ})
          - P_s / P_T / P_0      ({label: prob})
          - R_hat                ({label: log P_T - log P_0})
          - Q_n                  ({label: target prob})
          - g_n                  ({label: log Q/P_s})
          - donor_plus / donor_minus / g_n_top
          - eta                  ({label: local success rate}, 可选)
          - a_plus               ({label: log P_+ / P_0}, 可选)
        """
        modules = modules or []
        mods_full = []
        for m in modules:
            lam = m.get("lambda_", m.get("lambda", {}))
            if isinstance(lam, dict):
                lam_vals = [
                    v for k, v in lam.items()
                    if k != "other"]
                lam_g = (lam_vals[0] if lam_vals
                         else 0.0)
                lam_mean = (
                    sum(lam_vals) / len(lam_vals)
                    if lam_vals else 0.0)
            else:
                lam_g = float(lam or 0.0)
                lam_mean = lam_g

            mods_full.append({
                "module_id":       str(m.get(
                    "module_id",
                    m.get("skill_id", "")))[:60],
                "skill_id":        str(
                    m.get("skill_id", ""))[:40],
                "d_s":             round(
                    float(m.get("d_s", 0.0)), 3),
                "w":               round(
                    float(m.get("w", 0.0)), 3),
                "named_behaviors": list(
                    m.get("named_behaviors", []))
                    or list(lam.keys()
                            if isinstance(lam, dict)
                            else [])[:20],
                # 组内共用的标量 λ_g
                "lambda_g":        round(lam_g, 4),
                # per-label（虽然组内共用，但存下来
                # 方便后续 ablation 校验）
                "lambda_":         _preview_dict(
                    lam if isinstance(lam, dict)
                    else {}, 20),
                # 分布家族：全部落盘
                "P_s":             _preview_dict(
                    m.get("P_s", {}), 20),
                "P_T":             _preview_dict(
                    m.get("P_T", {}), 20),
                "P_0":             _preview_dict(
                    m.get("P_0", {}), 20),
                "R_hat":           _preview_dict(
                    m.get("R_hat", {}), 20),
                "Q_n":             _preview_dict(
                    m.get("Q_n", {}), 20),
                "g_n":             _preview_dict(
                    m.get("g_n", {}), 20),
                # 有用度倾斜（反映 eta→P_+→a_+）
                "eta":             _preview_dict(
                    m.get("eta", {}), 20),
                "a_plus":          _preview_dict(
                    m.get("a_plus", {}), 20),
                # 观测次数（λ 投影的 w_h 复现依据）
                "obs_counts":      _preview_dict(
                    m.get("obs_counts", {}), 20),
                # donor pair + g_n top-k
                "donor_plus":      str(
                    m.get("donor_plus", "")),
                "donor_minus":     str(
                    m.get("donor_minus", "")),
                "g_n_top":         _preview_dict(
                    m.get("g_n_top", {}), 5),
                "group_id": str(m.get("group_id", "")),
                "history_id": str(m.get("history_id", "")),
                "skill_step_text": str(m.get("skill_step_text", "")),
                "label_patterns": m.get("label_patterns", {}),
                "label_parents": m.get("label_parents", {}),
                "Z_h": list(m.get("Z_h", [])),
            })

        self._emit(
            "epoch_target",
            beta=round(float(beta), 3),
            n_modules=len(modules),
            modules=mods_full,
            summary=_clip(target_summary, 400),
        )
        # 人读摘要：每个 module 一行
        lines = [
            f"[R{self.round_n:02d}] target: "
            f"β={beta:.2f} |mods|={len(modules)}"
        ]
        for m in mods_full[:6]:
            lines.append(
                f"  · [{m['module_id'][:24]:24s}] "
                f"λ_g={m['lambda_g']:+.3f} "
                f"w={m['w']:.2f} "
                f"|labels|={len(m['named_behaviors'])} "
                f"donor +{m['donor_plus'][:12]} "
                f"-{m['donor_minus'][:12]}")
        self._log.info("\n".join(lines))

    # ─────────────────────────────────────────────
    # Proposals
    # ─────────────────────────────────────────────

    def proposal_batch(
        self, *,
        n_raw: int,
        n_valid: int,
        proposals: list = None,
        raw_response: str = "",
        prompt: str = "",
    ):
        """
        proposals: list[dict]，含 skill_id / edit_type /
        rationale / parent_location / old_text / new_text / appendix_note /
        appendix_notes

        execution_edit 的精确 scoped diff 会完整记录到 JSONL。
        """
        props = proposals or []
        full = []
        for p in props:
            new_body = str(p.get(
                "new_execution_body", "") or "")
            old_text = str(p.get("old_text", "") or "")
            new_text = str(p.get("new_text", "") or "")
            # appendix_note (单条) 或 appendix_notes (list)
            app_note = p.get("appendix_note", "")
            app_notes = p.get("appendix_notes", [])
            if isinstance(app_notes, list):
                app_all = "\n".join(str(x)
                                    for x in app_notes)
            else:
                app_all = str(app_notes or "")
            if app_note:
                app_all = (
                    str(app_note) + "\n" + app_all
                    if app_all else str(app_note))

            full.append({
                "skill_id":  str(
                    p.get("skill_id", ""))[:60],
                "edit_type": str(
                    p.get("edit_type", "")),
                "rationale": _clip(
                    p.get("rationale", ""), 400),
                "new_execution_body": new_body,
                "new_execution_body_len": len(new_body),
                "parent_location": str(
                    p.get("parent_location", "")),
                "old_text": old_text,
                "new_text": new_text,
                "scoped_diff_len": len(old_text) + len(new_text),
                "appendix_content": app_all,
                "appendix_len": len(app_all),
                "new_name": str(
                    p.get("new_name", "")),
                "new_description": _clip(
                    p.get("new_description", ""),
                    300),
            })
        self._emit(
            "proposal_batch",
            n_raw=n_raw,
            n_valid=n_valid,
            proposals=full,
            response_preview=_clip(raw_response, 500),
        )
        self._dump_prompt(
            "proposal", "batch",
            prompt=prompt, response=raw_response)
        self._log.info(
            f"[R{self.round_n:02d}] proposals: "
            f"raw={n_raw} valid={n_valid} "
            + " ".join([
                f"[{p['edit_type'][:4]}"
                f":{p['skill_id'][:14]}"
                f"|{p['new_execution_body_len']}c]"
                for p in full[:3]
            ]))

    # ─────────────────────────────────────────────
    # Candidate Gate（dual gate 三条件）
    # ─────────────────────────────────────────────

    def candidate_gate(
        self, *,
        cand_idx: int,
        skill_id: str,
        edit_type: str,
        label_kl_decrease: float | None,
        delta_hat: float,
        reward_epsilon: float,
        J_W_baseline: float,
        J_W_candidate: float,
        J_W_floor: float,
        is_admissible: bool,
        gate_details: dict = None,
    ):
        self._emit(
            "candidate_gate",
            cand_idx=cand_idx,
            skill_id=str(skill_id)[:60],
            edit_type=str(edit_type),
            gate_mode="paired_point_estimate",
            kl_metric="frozen_q_true_kl_decrease",
            label_kl_decrease=(None if label_kl_decrease is None else
                round(float(label_kl_decrease), 4)),
            reward_delta_point=round(float(delta_hat), 4),
            reward_epsilon=round(float(reward_epsilon), 4),
            reward_floor_policy="strict_running_best",
            J_W_baseline=round(
                float(J_W_baseline), 4),
            J_W_candidate=round(
                float(J_W_candidate), 4),
            J_W_floor=round(float(J_W_floor), 4),
            is_admissible=bool(is_admissible),
            gate_details=gate_details or {},
        )
        icon = "✓" if is_admissible else "×"
        self._log.info(
            f"[R{self.round_n:02d}] gate {icon} "
            f"[{cand_idx}][{edit_type[:4]}] "
            f"KL={label_kl_decrease if label_kl_decrease is not None else 'MISSING'} "
            f"Δ={delta_hat:+.4f} "
            f"ε_R={reward_epsilon:.4f} "
            f"J={J_W_baseline:.3f}→"
            f"{J_W_candidate:.3f}")

    # ─────────────────────────────────────────────
    # Accept / Stall
    # ─────────────────────────────────────────────

    def accept(
        self, *,
        skill_id: str,
        edit_type: str,
        J_W_before: float,
        J_W_after: float,
        delta_hat: float,
        proposal: dict = None,
        skill_before: str = "",
        skill_after: str = "",
    ):
        """
        接受一个 candidate 编辑。记录：
          - J_W / Δ 数值
          - 完整 proposal（含 execution_body / appendix）
          - skill 全文的前后对比（可选）
          - 若 dump_prompts=True，把前后 skill 落盘为 .txt

        skill_before / skill_after：整个 skill 的
        execution_body（可从 K_n / cand_lib 取）
        """
        p = proposal or {}
        new_body = str(p.get(
            "new_execution_body", "") or "")
        app_note = p.get("appendix_note", "")
        app_notes = p.get("appendix_notes", [])
        if isinstance(app_notes, list):
            app_all = "\n".join(str(x)
                                for x in app_notes)
        else:
            app_all = str(app_notes or "")
        if app_note:
            app_all = (
                str(app_note) + "\n" + app_all
                if app_all else str(app_note))

        self._emit(
            "accept",
            skill_id=str(skill_id)[:60],
            edit_type=str(edit_type),
            J_W_before=round(float(J_W_before), 4),
            J_W_after=round(float(J_W_after), 4),
            delta_J_W=round(
                float(J_W_after - J_W_before), 4),
            delta_hat=round(float(delta_hat), 4),
            proposal={
                "skill_id":  str(
                    p.get("skill_id", ""))[:60],
                "edit_type": str(
                    p.get("edit_type", "")),
                "rationale": _clip(
                    p.get("rationale", ""), 500),
                "new_execution_body": new_body,
                "new_execution_body_len":
                    len(new_body),
                "appendix_content": app_all,
                "appendix_len": len(app_all),
                "parent_location": str(
                    p.get("parent_location", "")),
                "old_text": str(p.get("old_text", "") or ""),
                "new_text": str(p.get("new_text", "") or ""),
            },
            skill_before_sha256=hashlib.sha256(
                (skill_before or "").encode("utf-8")).hexdigest(),
            skill_after_sha256=hashlib.sha256(
                (skill_after or "").encode("utf-8")).hexdigest(),
            skill_len_before=len(skill_before or ""),
            skill_len_after=len(skill_after or ""),
            skill_delta_chars=(
                len(skill_after or "")
                - len(skill_before or "")),
        )
        # 落盘完整前后对比
        if self.dump_dir:
            safe_id = "".join(
                c if c.isalnum()
                else "_" for c in str(skill_id))[:30]
            path = os.path.join(
                self.dump_dir,
                f"r{self.round_n:02d}_ACCEPT_"
                f"{safe_id}_{edit_type[:8]}.txt",
            )
            try:
                with open(path, "w",
                          encoding="utf-8") as f:
                    f.write(
                        f"# ACCEPT r={self.round_n} "
                        f"skill={skill_id} "
                        f"type={edit_type}\n")
                    f.write(
                        f"# J_W {J_W_before:.4f} → "
                        f"{J_W_after:.4f} "
                        f"(Δ={J_W_after-J_W_before:+.4f})\n")
                    f.write("=" * 60 + "\n")
                    f.write("RATIONALE:\n")
                    f.write(str(p.get(
                        "rationale", "")) + "\n\n")
                    if new_body:
                        f.write("=" * 60 + "\n")
                        f.write("NEW EXECUTION BODY:\n")
                        f.write(new_body + "\n\n")
                    if app_all:
                        f.write("=" * 60 + "\n")
                        f.write("APPENDIX ADDED:\n")
                        f.write(app_all + "\n\n")
                    if skill_before:
                        f.write("=" * 60 + "\n")
                        f.write("SKILL BEFORE:\n")
                        f.write(skill_before + "\n\n")
                    if skill_after:
                        f.write("=" * 60 + "\n")
                        f.write("SKILL AFTER:\n")
                        f.write(skill_after + "\n")
            except Exception:
                pass

        self._log.info(
            f"[R{self.round_n:02d}] "
            f"✓ ACCEPT [{edit_type}] "
            f"J_W {J_W_before:.4f}→{J_W_after:.4f} "
            f"(Δ={J_W_after-J_W_before:+.4f}) "
            f"|new_body|={len(new_body)}c "
            f"|appendix|={len(app_all)}c")

    def stall(self, reason: str = ""):
        self._emit("stall", reason=reason)
        self._log.info(
            f"[R{self.round_n:02d}] · STALL "
            f"{reason}")

    # ─────────────────────────────────────────────
    # Slow Update / Witness / Error
    # ─────────────────────────────────────────────

    def slow_update(
        self, *,
        triggered: bool,
        improvement_rate: float = 0.0,
        regression_rate: float = 0.0,
        skipped: bool = False,
        reason: str = "",
        n_updated: int = 0,
    ):
        self._emit(
            "slow_update",
            triggered=triggered,
            improvement_rate=round(
                float(improvement_rate), 4),
            regression_rate=round(
                float(regression_rate), 4),
            skipped=skipped,
            reason=reason,
            n_updated=n_updated,
        )
        if not triggered:
            return
        icon = "→skip" if skipped else "→apply"
        self._log.info(
            f"[R{self.round_n:02d}] slow_update "
            f"{icon} imp={improvement_rate:.0%} "
            f"reg={regression_rate:.0%} "
            f"updated={n_updated}")

    def witness_eval(
        self, *,
        tag: str,
        J_W: float,
        J_W_hard: float = None,
        J_W_soft: float = None,
        n_tasks: int = 0,
        durations_s: float = 0.0,
    ):
        self._emit(
            "witness_eval",
            tag=tag,
            J_W=round(float(J_W), 4),
            J_W_hard=(round(float(J_W_hard), 4)
                      if J_W_hard is not None else None),
            J_W_soft=(round(float(J_W_soft), 4)
                      if J_W_soft is not None else None),
            n_tasks=n_tasks,
            durations_s=round(float(durations_s), 2),
        )
        self._log.info(
            f"[R{self.round_n:02d}] witness[{tag}] "
            f"J_W={J_W:.4f} "
            f"({n_tasks} tasks, {durations_s:.1f}s)")

    def error(
        self, *,
        where: str,
        message: str,
        exc_type: str = "",
    ):
        self._emit(
            "error",
            where=where,
            exc_type=exc_type,
            message=_clip(message, 500),
        )
        self._log.warning(
            f"[R{self.round_n:02d}] ERROR@{where}: "
            f"{_clip(message, 200)}")

    # ─────────────────────────────────────────────
    # Optional: 单条 LLM 交互详细 dump
    # ─────────────────────────────────────────────

    def llm_interaction(
        self, *,
        role:     str,   # "student"/"teacher"/"editor"/"parser"/"expert"
        task_id:  str = "",
        prompt:   str = "",
        response: str = "",
        tokens:   int = 0,
        cost:     float = 0.0,
        extra:    dict = None,
    ):
        """
        默认只写摘要到 JSONL；若 dump_prompts=True 则完整落盘到
        logs/<run>/prompts/rNN_<role>_<tid>.txt
        """
        self._emit(
            "llm_interaction",
            role=role,
            task_id=str(task_id)[:60],
            prompt_len=len(prompt or ""),
            response_len=len(response or ""),
            prompt_preview=_clip(prompt, 300),
            response_preview=_clip(response, 300),
            tokens=tokens,
            cost=round(float(cost), 5),
            extra=extra or {},
        )
        self._dump_prompt(
            role, task_id,
            prompt=prompt, response=response)

    # ─────────────────────────────────────────────
    # Session end
    # ─────────────────────────────────────────────

    def session_end(
        self, *,
        total_rounds: int,
        total_accepted: int,
        J_W_init: float,
        J_W_final: float,
        J_W_best: float = None,
        total_cost: float = 0.0,
        cost_summary: dict = None,
    ):
        self._emit(
            "session_end",
            total_rounds=total_rounds,
            total_accepted=total_accepted,
            J_W_init=round(float(J_W_init), 4),
            J_W_final=round(float(J_W_final), 4),
            J_W_best=(round(float(J_W_best), 4)
                      if J_W_best is not None
                      else None),
            delta_J_W=round(
                float(J_W_final - J_W_init), 4),
            total_cost=round(float(total_cost), 4),
            cost_summary=cost_summary or {},
        )
        self._log.info(
            f"=== SESSION END  "
            f"rounds={total_rounds} "
            f"accepted={total_accepted} "
            f"J_W {J_W_init:.4f}→{J_W_final:.4f} "
            + (f"(best={J_W_best:.4f}) "
               if J_W_best is not None else "")
            + f"${total_cost:.3f} ===")

    # ─────────────────────────────────────────────
    # 兼容旧 API（保留最少必需，退化为 no-op 或转发）
    # ─────────────────────────────────────────────

    def log_round_summary(
        self, round_n, J_W, action,
        kappa_grammar=0.0, kappa_yield=0.0,
        cost_this_round=0.0, total_cost=0.0,
    ):
        """旧接口 → 转发到 round_end"""
        self.round_end(
            action=action,
            J_W_after=J_W,
            cost_this_round=cost_this_round,
            kappa_grammar=kappa_grammar,
            kappa_yield=kappa_yield,
            total_cost=total_cost,
        )

    def log_session_end(
        self, total_rounds, total_accepted,
        J_W_init, J_W_final, total_cost,
    ):
        self.session_end(
            total_rounds=total_rounds,
            total_accepted=total_accepted,
            J_W_init=J_W_init,
            J_W_final=J_W_final,
            total_cost=total_cost,
        )

    def log_stall(self, reason: str):
        self.stall(reason)

    def log_update_accepted(
        self, proposal, J_W_before,
        J_W_after, lcb,
    ):
        self.accept(
            skill_id=proposal.get("skill_id", ""),
            edit_type=proposal.get("edit_type", ""),
            J_W_before=J_W_before,
            J_W_after=J_W_after,
            delta_hat=J_W_after - J_W_before,
            proposal=proposal,
        )
