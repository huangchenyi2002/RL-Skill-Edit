# study1_main.py
"""
Study 1：Reward-Calibrated Skill Distillation 主循环

对应 lambda_estimation_selfcontained_cn + executable_details 的完整流程：
  1. Parser LLM 冻结 Z, φ, parent(z), pooling groups
  2. Student/Teacher/Reference rollout
  3. 估计 label laws + 局部成功率 η
  4. 构造 P_+, a_+（成功加权目标 + 有用度倾斜）
  5. 投影 a_+ 到 teacher 方向 → λ_g（+ 收缩）
  6. 构造 q_λ（target 分布）
  7. 计算 g_n → donor pair → 编辑方向
  8. Editor LLM 生成候选 skill edit
  9. Student 重新 rollout candidate skill
    10. Dual gate（frozen-Q true KL point + paired reward point）接受/拒绝
  11. 冻结原则：一轮内 λ, β, Q_n, g_n 不变

关键变化（vs 旧版）：
  - λ 是 reward-calibrated projection slope，不是信心分数
  - m 权重已移除，w = d_s（纯占用率）
  - dispatch edit 已移除，只保留 execution + appendix
  - 单 skill 文件（data/skill/SKILL.md）
"""

import json
import os
import random
import math
import hashlib
from collections import defaultdict
from concurrent.futures import (
    ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout,
)
from datetime import datetime

import numpy as np
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from src.client import OpenRouterClient
from src.agent import StudentAgent
from src.teacher import TeacherService
from src.evaluator import Evaluator, select_gate_score
from src.validator import Validator
from src.skill_library import (
    SkillLibrary, append_to_appendix_field,
    apply_text_edit, apply_scoped_parent_edit,
)
from src.logger import OSDLogger
from src.library_registry import register_best_library
from src.reference_baseline import ReferenceBaseline
from src.exskill_signal import (
    ResidualEvidence, get_high_priority_tasks,
    merge_signals_by_skill,
)
from src.opd_teacher_signal import (
    attach_teacher_grades_to_trajectories,
    teacher_scores_from_grades,
)
from src.label_space import (
    build_epoch_target, format_all_edit_directions,
    Module, build_history_records,
)
from src.parser_llm import ParserLLM

load_dotenv()


# ─────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────

def load_config(path="config/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(
        os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2,
                  ensure_ascii=False, default=str)


MAX_SKILL_CHARS = 8000  # 限制 skill 总长度


def apply_edit(library, proposal, epoch_target=None):
    new_lib = library.clone()
    sk = new_lib.get(proposal["skill_id"])
    if sk is None:
        return new_lib
    et = proposal.get("edit_type", "")
    if et == "execution_edit":
        parent = proposal.get("parent_location", "")
        allowed = set()
        if epoch_target is not None:
            for ms in epoch_target.modules:
                if ms.skill_id == proposal["skill_id"] and ms.estimable:
                    allowed.update(p for p in (ms.label_parents or {}).values()
                                   if p)
        if not parent or parent not in allowed:
            return library.clone()
        new_body, report = apply_scoped_parent_edit(
            sk.execution_body or "", parent,
            proposal.get("old_text", ""), proposal.get("new_text", ""))
        if report["status"] != "applied_scoped_replace" \
                or len(new_body) > MAX_SKILL_CHARS:
            return library.clone()
        sk.execution_body = new_body
    elif et == "appendix_edit":
        notes = proposal.get("appendix_notes", [])
        if notes:
            sk.add_appendix_notes(notes)
    return new_lib


def parallel_map_interruptible(
    fn, items, workers, desc="",
    per_task_timeout=None,   # 已弃用，保留兼容
    global_timeout=None,     # 已弃用，保留兼容
):
    """
    并行执行 fn(item)，返回结果列表（顺序保留）。

    行为：
    - 没有超时兜底，一直等所有 task 完成
      （API 层已有 45s httpx read timeout；
       subprocess 层有 8s timeout；
       都够快失败，不需要外层再兜底）
    - Ctrl+C 时立即返回（不等 executor）
    - 单个 task 抛异常返回 None，其他继续
    """
    import time as _t
    n = len(items)
    if n == 0:
        return []

    results = [None] * n
    # 关键 bug 修复（2026-07-10）：
    # 之前用 `if results[idx] is None` 判断未处理，
    # 但 fn 返回 None 或抛异常时 results[idx] 依然是 None，
    # 于是同一个 future 会被反复计数，done_count 提前到达
    # n 而外循环退出，剩下的 future 直接被 cancel。
    # 用独立的 processed[idx] bool 数组，绝不依赖 results 的
    # 值来判断"是否已处理"。
    processed = [False] * n
    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=f"pmap-{desc[:15]}")

    futs = {
        executor.submit(fn, item): idx
        for idx, item in enumerate(items)
    }
    fut_list = list(futs.keys())
    bar = tqdm(total=n, desc=desc, leave=False)
    done_count = 0

    try:
        # 每 0.5s 轮询一次已完成的 future
        # 无全局超时，一直等到所有 task 完成
        while done_count < n:
            newly_done = []
            for fut in fut_list:
                if fut.done():
                    idx = futs[fut]
                    if not processed[idx]:
                        try:
                            results[idx] = (
                                fut.result(timeout=0))
                        except Exception:
                            # 静默：task 异常留 None
                            results[idx] = None
                        processed[idx] = True
                        done_count += 1
                        newly_done.append(fut)
                        bar.update(1)

            if not newly_done:
                # 没有新完成的，睡 0.5s 后再看
                # 期间可响应 Ctrl+C
                _t.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  🛑 KeyboardInterrupt: 立即取消...")
    finally:
        bar.close()
        # 强制取消所有未完成的 future
        for fut in fut_list:
            if not fut.done():
                fut.cancel()
        # 不等 executor 关闭（可能有网络 IO 卡住）
        executor.shutdown(
            wait=False, cancel_futures=True)

    return results


def compute_learning_rate(
    step, total, max_lr=5, min_lr=2,
    mode="cosine",
):
    if total <= 1:
        return max_lr
    t = min(step, total) / total
    if mode == "constant":
        return max_lr
    elif mode == "linear":
        lr = max_lr + (min_lr - max_lr) * t
    elif mode == "cosine":
        lr = (min_lr + 0.5 * (max_lr - min_lr)
              * (1 + math.cos(math.pi * t)))
    else:
        return max_lr
    return max(min_lr, round(lr))


def try_slow_update(
    K_n, K_prev, teacher, agent,
    dev_pool, n_tasks,
    min_margin: float = 0.10,
    evaluator=None,
    witness=None,
    w_cache=None,
    J_W_floor=None,
    require_gate: bool = True,
    epoch_target=None,
    act_mode="implicit",
    forced_id=None,
    dirichlet_a=1.0,
    witness_workers=4,
):
    """
    Slow update 走**完整 dual gate**（老师 audit issue #6）：

    流程：
      1. Teacher 生成 meta_guidance
      2. 两道预筛护栏（imp/reg + 显著性 margin）
      3. 构造 candidate library（写入 meta_guidance）
      4. Candidate 在 witness 上跑 rollout
    5. 用 epoch_target 冻结 Z 算真实 KL decrease point estimate
    6. Dual gate: ΔKL > 0 AND Δreward ≥ -ε_R AND reward floor
      7. 通过 → apply + 返回 candidate_witness 让主循环
         刷新 w_cache / K_best / J_W_best

    返回
    ----
    (result_dict, ok_bool)：
      result_dict 关键字段：
        - accepted: bool（gate 是否通过）
        - candidate_witness: dict（若 accepted，供刷新 cache）
        - candidate_lib: SkillLibrary（若 accepted）
    """
    try:
        result, _ = teacher.generate_slow_update(
            library=K_n, prev_library=K_prev,
            eval_tasks=dev_pool, agent=agent,
            n_tasks=n_tasks)
        imp_rate = result["improvement_rate"]
        reg_rate = result["regression_rate"]
        margin = imp_rate - reg_rate

        # ── Guardrail 1: 必须净改进 ──
        if imp_rate <= reg_rate:
            print(f"    imp={imp_rate:.0%}"
                  f" reg={reg_rate:.0%}"
                  f" → skipped (imp ≤ reg)")
            return {
                "triggered": True,
                "improvement_rate": imp_rate,
                "regression_rate": reg_rate,
                "meta_skills": [],
                "skipped": True,
                "reason": "imp<=reg",
            }, True

        # ── Guardrail 2: 差距必须显著 ──
        if margin < min_margin:
            print(f"    imp={imp_rate:.0%}"
                  f" reg={reg_rate:.0%}"
                  f" margin={margin:.0%}"
                  f" → skipped "
                  f"(margin < {min_margin:.0%})")
            return {
                "triggered": True,
                "improvement_rate": imp_rate,
                "regression_rate": reg_rate,
                "meta_skills": [],
                "skipped": True,
                "reason":
                    f"margin<{min_margin:.0%}",
            }, True

        meta = result.get("meta_guidance", {})
        if not meta:
            return {
                "triggered": True,
                "improvement_rate": imp_rate,
                "regression_rate": reg_rate,
                "meta_skills": [],
                "skipped": True,
                "reason": "empty_meta",
            }, True

        # ── 3. 构造 candidate library（先不 apply）──
        candidate_lib = K_n.clone()
        pending = []
        for sid, guidance in meta.items():
            sk = candidate_lib.get(sid)
            if sk and guidance:
                sk.set_slow_update_content(guidance)
                pending.append(sid)

        # ── 4. 若需要 gate，走完整 dual gate ──
        if require_gate and evaluator and witness \
                and w_cache is not None:
            print(f"    imp={imp_rate:.0%}"
                  f" reg={reg_rate:.0%}"
                  f" margin={margin:.0%}"
                  f" → 触发 dual gate...")

            # 4a. Candidate 在 witness 上跑 rollout
            # （给 frozen-target true KL gate 用）
            label_kl_point = None
            if epoch_target:
                try:
                    def _run_cand_slow(task):
                        return agent.run_task(
                            task, candidate_lib,
                            activation_mode=act_mode,
                            forced_skill_id=forced_id)
                    cand_trajs = (
                        parallel_map_interruptible(
                            _run_cand_slow, witness,
                            workers=witness_workers,
                            desc="      SlowCand"))
                    cand_trajs = [
                        t for t in cand_trajs
                        if t is not None]
                    # baseline (K_n) 在 witness 上的
                    # trajectories 存不存在？主循环把
                    # incumbent trajectories 存在
                    # w_cache 里吗？没有 —— 直接跑一次
                    def _run_old_slow(task):
                        return agent.run_task(
                            task, K_n,
                            activation_mode=act_mode,
                            forced_skill_id=forced_id)
                    old_trajs = (
                        parallel_map_interruptible(
                            _run_old_slow, witness,
                            workers=witness_workers,
                            desc="      SlowBase"))
                    old_trajs = [
                        t for t in old_trajs
                        if t is not None]
                    label_kl_res = (
                        compute_candidate_label_kl(
                            candidate_lib,
                            cand_trajs,
                            epoch_target, K_n,
                            dirichlet_a,
                            trajectories_old_aligned=(
                                old_trajs or None),
                            return_lcb=False))
                    label_kl_point = label_kl_res
                except Exception as _e:
                    print(f"    ⚠ slow_update "
                          f"true KL 算失败: {_e}")

            # 4b. Dual gate
            try:
                gate_result = \
                    evaluator.evaluate_candidate(
                        K_n, candidate_lib,
                        witness, w_cache,
                        label_kl_decrease=(
                            label_kl_point),
                        J_W_floor=J_W_floor)
            except Exception as _e:
                print(f"    ⚠ slow_update gate "
                      f"eval 失败: {_e}")
                return {
                    "triggered": True,
                    "improvement_rate": imp_rate,
                    "regression_rate": reg_rate,
                    "meta_skills": [],
                    "accepted": False,
                    "skipped": True,
                    "reason": "gate_eval_error",
                }, True

            if not gate_result.get(
                    "is_admissible", False):
                delta = gate_result.get(
                    "delta_hat", 0)
                print(f"    → gate 拒绝 "
                      f"(Δ={delta:+.4f}, "
                        f"ΔKL={label_kl_point or 0:+.4f}), "
                      f"skip")
                return {
                    "triggered": True,
                    "improvement_rate": imp_rate,
                    "regression_rate": reg_rate,
                    "meta_skills": [],
                    "accepted": False,
                    "skipped": True,
                    "reason": "gate_rejected",
                    "gate_delta_hat": delta,
                    "label_kl_decrease": label_kl_point,
                }, True

            # gate 过了，apply 到 K_n
            for sid in pending:
                sk_src = candidate_lib.get(sid)
                sk_tgt = K_n.get(sid)
                if sk_src and sk_tgt:
                    sk_tgt.set_slow_update_content(
                        sk_src.get_slow_update_content())
            print(f"    imp={imp_rate:.0%}"
                  f" reg={reg_rate:.0%}"
                  f" margin={margin:.0%}"
                  f" → ✓ DUAL GATE PASS "
                  f"(Δ={gate_result.get('delta_hat', 0):+.4f}, "
                  f"ΔKL={label_kl_point or 0:+.4f}) "
                  f"updated={len(pending)}")
            return {
                "triggered": True,
                "improvement_rate": imp_rate,
                "regression_rate": reg_rate,
                "meta_skills": pending,
                "accepted": True,
                "gate_passed": True,
                "gate_delta_hat":
                    gate_result.get(
                        "delta_hat", 0),
                "label_kl_decrease": label_kl_point,
                "candidate_witness": gate_result,
                "candidate_lib": candidate_lib,
            }, True

        # ── require_gate=False 的旧行为（仅 debug）──
        for sid in pending:
            sk = K_n.get(sid)
            if sk:
                sk.set_slow_update_content(
                    meta.get(sid, ""))
        print(f"    imp={imp_rate:.0%}"
              f" reg={reg_rate:.0%}"
              f" margin={margin:.0%}"
              f" updated={len(pending)} "
              f"(⚠ no gate — 不推荐)")
        return {
            "triggered": True,
            "improvement_rate": imp_rate,
            "regression_rate": reg_rate,
            "accepted": True,
            "meta_skills": pending,
        }, True
    except Exception as e:
        print(f"    ⚠️ slow update: {e}")
        return {"triggered": False,
                "error": str(e)}, False


def stable_score(item):
    w = item["witness"]
    delta = w["delta_hat"]
    if delta <= 0:
        return delta
    rewards = w.get("candidate_rewards", [])
    if not rewards:
        return delta
    means = [float(np.mean(r)) for r in rewards]
    std = float(np.std(means))
    stability = 1.0 - std / (abs(delta)+std+1e-6)
    return delta * stability


def stratified_sample_vn(
    dev_pool, round_n, stall_count=0,
    last_action="", n_per_domain=1,
    vn_stall_min=20, stall_growth=1.5,
    post_accept_min=20, seed=None,
    full_pool_every_round=False,
):
    """
    分层采样 V_n。

    full_pool_every_round=True （2026-07-10 新增）：
      每一轮都返回整个 dev_pool。
      老师方法学要求每轮重新估 P_T/P_0/P_s + WLS λ_g，
      需要 30-60 个独立 rollout 才有可辨认信号。
      正常轮 V_n=10 会让 shrinkage 强制 λ→0（不是 bug，
      是采样预算不够）。开启此模式后 λ 估计最稳。
      Token 成本 ~6× 正常轮，但保证方法学合规。
    """
    if seed is not None:
        random.seed(seed)
    # ⚠ 全量模式：每轮跑满整个 dev_pool
    if full_pool_every_round:
        print(f"\n    [采样] R{round_n} "
              f"(full_pool) = {len(dev_pool)}")
        return dev_pool
    if round_n == 0:
        return dev_pool
    domain_groups = defaultdict(list)
    for t in dev_pool:
        domain_groups[
            t.get("domain", "unknown")
        ].append(t)
    if stall_count > 0:
        target = int(vn_stall_min
            * (stall_growth ** (stall_count-1)))
        mode = f"stall={stall_count}"
    elif last_action == "accept":
        target = post_accept_min
        mode = "post-accept"
    else:
        target = n_per_domain * len(domain_groups)
        mode = "normal"
    target = min(max(target, 2), len(dev_pool))
    total = len(dev_pool)
    selected = []
    for domain, tasks in domain_groups.items():
        ratio = len(tasks) / max(total, 1)
        n = max(1, int(target * ratio))
        n = min(n, len(tasks))
        selected.extend(random.sample(tasks, n))
    remaining = [t for t in dev_pool
                 if t not in selected]
    if len(selected) < target and remaining:
        selected.extend(random.sample(
            remaining,
            min(target-len(selected),
                len(remaining))))
    print(f"\n    [采样] R{round_n} "
          f"({mode}) = {len(selected)}")
    return selected


def compute_attribution(
    trajectories, diagnoses, library,
    alpha_attr, beta_unc,
    cum_attr, cum_n,
):
    attr_sum = {sid: 0.0
                for sid in library.skill_ids()}
    n = len(trajectories)
    uncovered = []
    for t, d in zip(trajectories, diagnoses):
        for sid in t.activated_skills:
            if sid in attr_sum:
                attr_sum[sid] += 0.5
        for sid in d.get("I_hat_i", []):
            if sid in attr_sum:
                attr_sum[sid] += 0.5
        impl = d.get("implicated_skill_id")
        if impl and impl in attr_sum:
            attr_sum[impl] += 0.5
        if not t.success:
            mx = max((attr_sum.get(s, 0)
                      for s in t.activated_skills),
                     default=0)
            if (1-mx) >= beta_unc:
                uncovered.append(t)
    attr_norm = {sid: v/max(n,1)
                 for sid, v in attr_sum.items()}
    new_n = cum_n + n
    for sid, v in attr_norm.items():
        cum_attr[sid] += v * n
    cum_norm = {sid: v/max(new_n,1)
                for sid, v in cum_attr.items()}
    stable = [sid for sid, v in cum_norm.items()
              if v >= alpha_attr]
    single = [sid for sid, v in attr_norm.items()
              if v >= alpha_attr]
    return {
        "neighborhood": list(set(single+stable)),
        "stable_neighborhood": stable,
        "uncovered_rate": len(uncovered)/max(n,1),
    }, cum_attr, new_n


def proposal_diagnostics(n_raw, n_valid, M):
    m = max(M, 1)
    return {
        "kappa_grammar": min(n_raw/m, 1.0),
        "kappa_yield": min(n_valid/m, 1.0),
    }


def select_final_library(K_n, K_best, ablation):
    """Choose the single library used by every final output and evaluation."""
    if ablation == "no_gate":
        return K_n.clone(), "last_static_valid"
    if K_best is None:
        raise RuntimeError("witness-selected final library is missing")
    return K_best.clone(), "best_witness_gated"


# ─────────────────────────────────────────────────
# 计算候选 skill 在 frozen target 上的真实 KL loss decrease
# ─────────────────────────────────────────────────

def compute_candidate_label_kl(
    candidate_lib, trajectories_new,
    epoch_target, library_old,
    alpha=1.0,
    trajectories_old_aligned=None,
    return_lcb=False,
):
    """
        Paired Witness 上的真实 KL loss decrease point estimate：

            Σ_h d_inc(h)[KL(P_inc,W(.|h)||Q_n(.|h))
                                     - KL(P_cand,W(.|h)||Q_n(.|h))]

        Z、history routing、φ 与 Q_n 全部来自冻结 epoch_target。
        old/new 必须有完全相同的有效 task ids；缺失或不可估计时拒绝。
        return_lcb 仅为旧调用兼容，返回的 lcb 始终为 None。
    """
    if not epoch_target or not epoch_target.modules:
        return None if not return_lcb else {
            "point": None, "lcb": None,
            "n_boot": 0}
    epoch_target.assert_frozen()

    def _true_kl_decrease(new_trajs, old_trajs):
        old_valid = {
            t.task_id for t in (old_trajs or [])
            if getattr(t, "evaluation_valid", False)}
        new_valid = {
            t.task_id for t in new_trajs
            if getattr(t, "evaluation_valid", False)}
        paired_tasks = old_valid & new_valid
        if not old_valid or old_valid != new_valid:
            return None
        frozen_modules = []
        signals_by_module = {}
        for ms in epoch_target.modules:
            if not ms.estimable or not ms.Q_n:
                continue
            frozen_modules.append(Module(
                module_id=ms.module_id,
                history_id=ms.history_id or ms.module_id,
                group_id=ms.group_id or ms.module_id,
                skill_id=ms.skill_id,
                named_behaviors=[
                    z for z in ms.g_n
                    if z != "other"],
                reach_patterns=list(ms.reach_patterns),
                label_patterns=dict(ms.label_patterns),
                label_parents=dict(ms.label_parents),
                skill_step_text=ms.skill_step_text,
            ))
            signals_by_module[ms.module_id] = ms
        if not frozen_modules:
            return None
        old_records_all = build_history_records(
            [t for t in (old_trajs or [])
             if t.task_id in paired_tasks],
            frozen_modules, "incumbent")
        new_records_all = build_history_records(
            [t for t in new_trajs
             if t.task_id in paired_tasks],
            frozen_modules, "candidate")
        frozen_ids = {m.module_id for m in frozen_modules}
        old_records_all = [r for r in old_records_all
                           if r.module_id in frozen_ids]
        new_records_all = [r for r in new_records_all
                           if r.module_id in frozen_ids]
        total_old_observations = len(old_records_all)
        if total_old_observations <= 0:
            return None
        weighted_delta = 0.0
        used_weight = 0.0
        from src.label_space import compute_distributions_from_records
        old_dists = compute_distributions_from_records(
            old_records_all, frozen_modules, alpha)
        new_dists = compute_distributions_from_records(
            new_records_all, frozen_modules, alpha)
        from src.label_space import compute_label_kl_decrease
        for frozen_mod in frozen_modules:
            ms = signals_by_module[frozen_mod.module_id]
            old_records = [
                r for r in old_records_all
                if r.module_id == frozen_mod.module_id]
            new_records = [
                r for r in new_records_all
                if r.module_id == frozen_mod.module_id]
            if not old_records or not new_records:
                return None
            old_tasks = {r.task_id for r in old_records}
            new_tasks = {r.task_id for r in new_records}
            if old_tasks != new_tasks:
                return None
            old_dist = old_dists.get(frozen_mod.module_id)
            new_dist = new_dists.get(frozen_mod.module_id)
            if old_dist is None or new_dist is None:
                return None
            occupancy = len(old_records) / total_old_observations
            weighted_delta += occupancy * compute_label_kl_decrease(
                old_dist.probs, new_dist.probs, ms.Q_n)
            used_weight += occupancy
        return weighted_delta / used_weight if used_weight > 0 else None

    point_delta = _true_kl_decrease(
        trajectories_new,
        trajectories_old_aligned)
    if point_delta is None:
        return None if not return_lcb else {
            "point": None, "lcb": None,
            "n_boot": 0}
    if not return_lcb:
        return point_delta
    return {"point": point_delta,
            "lcb": None,
            "n_boot": 0,
            "n_paired": len({t.task_id for t in trajectories_new})}

# ─────────────────────────────────────────────────
# Study 1 主流程
# ─────────────────────────────────────────────────

def run_study1(
    config_path="config/config.yaml",
    beta_mode_override=None,
    ablation_override=None,
):
    print("=" * 60)
    print("  STUDY 1: Label-Space ExSkill-OPD")
    print("=" * 60)

    cfg   = load_config(config_path)
    s1    = cfg["study1"]
    paths = cfg["paths"]

    B_W          = s1["B_W"]
    M            = s1["M"]
    N_max        = s1["N_max"]
    S_max        = s1["S_max"]
    reward_epsilon = s1.get(
        "reward_epsilon", s1.get("tau_min", 0.001))
    alpha_attr   = s1["alpha_attr"]
    beta_unc     = s1["beta_uncovered"]
    vn_stall_min = s1.get("V_n_stall_min", 20)
    stall_growth = s1.get("stall_sample_growth", 1.5)
    post_accept  = s1.get("post_accept_min", 20)
    # ⚠ 2026-07-10：全量模式。每轮 V_n = 整个 dev_pool，
    # 保证 P_T/P_0/λ 估计有足够独立轨迹。token 消耗大，
    # 但方法学最稳。
    full_pool_vn = s1.get(
        "full_pool_every_round", False)
    en_dispatch  = False  # dispatch edit 已移除
    en_execution = s1.get("enable_execution_edits", True)
    compute_fn   = s1.get("compute_fn_correlation", True)
    run_ref      = s1.get("run_reference_baseline", True)
    ref_path     = s1.get("reference_baseline_path",
        "results/study1/reference_baseline.json")
    teacher_n    = s1.get("teacher_exec_tasks_n", 10)
    # ⚠ 2026-07-10 audit：新增 enable_slow_update 总开关。
    # 设为 false 完全跳过 slow_update，即使 slow_freq 满足。
    enable_slow = s1.get("enable_slow_update", False)
    slow_freq    = s1.get("slow_update_freq", 1)
    slow_n       = s1.get("slow_update_n_tasks", 20)
    slow_min_margin = s1.get(
        "slow_update_min_margin", 0.10)
    slow_require_gate = s1.get(
        "slow_update_require_gate", True)
    lr_mode      = s1.get("lr_schedule", "cosine")
    lr_max       = s1.get("lr_max", M)
    lr_min       = s1.get("lr_min", 2)
    max_cost     = s1.get("max_cost_usd", 0)
    mb_size      = s1.get("minibatch_size", 3)
    beta_budget  = s1.get("beta_budget", 1.0)
    dirichlet_a  = s1.get("dirichlet_alpha", 1.0)
    use_rc_lambda = s1.get(
        "use_reward_calibrated_lambda", True)
    lambda_neutral = s1.get("lambda_neutral", 0.0)
    eta_mode = s1.get(
        "eta_mode", "occurrence_weighted")
    # ── λ 估计模式（Q2 后续补充）────────────
    # false = WLS 小倾斜近似 + shrinkage
    # true  = 1D 精确 argmin KL(P_+‖q_ℓ)
    use_exact_lambda = s1.get(
        "use_exact_lambda", False)

    beta_mode = beta_mode_override or s1.get(
        "beta_mode", "b4")
    beta_value = s1.get("beta_value", 1.0)
    ablation = ablation_override or s1.get(
        "ablation_mode", "full")
    if ablation not in {"full", "execution_only", "no_gate"}:
        raise ValueError(f"unknown ablation mode: {ablation}")
    if ablation == "execution_only":
        en_dispatch = False

    # ── 智能路径探测 ──────────────────
    # 优先使用 config 里的路径；如果不存在，试常见备选
    def _resolve_data_path(configured, name):
        if os.path.exists(configured):
            return configured
        # 常见备选路径
        alternates = [
            configured.replace(
                "data/spreadsheet/",
                "data/spreadsheetbench/"),
            configured.replace(
                "data/spreadsheetbench/",
                "data/spreadsheet/"),
            f"data/spreadsheet/splits/{name}",
            f"data/spreadsheetbench/splits/{name}",
        ]
        for alt in alternates:
            if os.path.exists(alt):
                print(f"  ⚠ '{configured}' 不存在，"
                      f"自动切换到 '{alt}'")
                return alt
        return None

    dev_pool_path = _resolve_data_path(
        paths["dev_pool"], "dev_pool.json")
    witness_path = _resolve_data_path(
        paths["witness"], "witness.json")

    if not dev_pool_path:
        print(f"\n❌ dev_pool missing: "
              f"tried {paths['dev_pool']}")
        print(f"  cwd: {os.getcwd()}")
        # 列出 data 目录看看有什么
        if os.path.exists("data"):
            print("  data/ contents:")
            for root, dirs, files in os.walk(
                "data", topdown=True
            ):
                for d in dirs:
                    print(f"    {os.path.join(root, d)}")
                # 只走 2 层
                if root.count(os.sep) >= 2:
                    dirs.clear()
        return None, None

    dev_pool = load_json(dev_pool_path)
    # ⚠ 老师 audit：dev_pool 也要 cap 到 dev_pool_size
    # （之前只 cap witness，dev_pool 用完整文件 199 条，
    #  破坏 60/20/20 协议）
    _dev_cap = cfg.get("data", {}).get(
        "dev_pool_size", 60)
    if len(dev_pool) > _dev_cap:
        random.seed(42)
        dev_pool = random.sample(dev_pool, _dev_cap)
        print(f"  Dev pool capped to "
              f"{_dev_cap} tasks")

    witness = load_json(
        witness_path or paths["witness"])
    # cap witness 到 witness_size（20）
    _witness_cap = cfg.get("data", {}).get(
        "witness_size", 20)
    if len(witness) > _witness_cap:
        random.seed(42)
        witness = random.sample(
            witness, _witness_cap)
        print(f"  Witness capped to "
              f"{_witness_cap} tasks")
    trigger  = load_json(paths["literacy_probe"])

    # ── Final Holdout（关键：不参与 gate/best selection）──
    # 论文合规要求：witness 用于 gate 决策，会被反复访问，
    # 不能再充当"最终无偏评估"。必须有独立的 final_holdout
    # split，只在整个优化结束后**评估一次**。
    #
    # 2026-07-10：数据目录里有 test.json 但主流程之前
    # 从未加载，J_W_final/J_W_best 都用的是 witness →
    # 测试集泄漏。此处修复。
    # ⚠ audit issue #7 严格化（2026-07-10）：
    #   - 缺失 → 硬失败（不再是 warning）
    #   - 检查 dev/witness/final 两两不重叠
    #   - final_holdout 会自动移除与 dev/witness 重叠的
    #     task（若有），确保严格隔离
    #   - 主实验缺 final_holdout 直接 raise
    require_final_holdout = s1.get(
        "require_final_holdout", True)
    final_holdout_path = paths.get(
        "final_holdout", paths.get("test", ""))
    if not (final_holdout_path
            and os.path.exists(final_holdout_path)):
        msg = (f"final_holdout 路径不存在: "
               f"{final_holdout_path}. "
               f"主实验要求正式 test set。")
        if require_final_holdout:
            raise FileNotFoundError(msg)
        else:
            print(f"  ⚠ {msg}（require=False，继续）")
            final_holdout = []
    else:
        final_holdout = load_json(final_holdout_path)
        _fh_cap = cfg.get("data", {}).get(
            "final_holdout_size",
            cfg.get("data", {}).get("test_size", 50))
        if len(final_holdout) > _fh_cap:
            random.seed(20260710)
            final_holdout = random.sample(
                final_holdout, _fh_cap)
        # ⚠ 三分：dev / witness / final 两两不重叠
        dev_ids = {t.get("task_id", "")
                   for t in dev_pool}
        w_ids = {t.get("task_id", "")
                 for t in witness}
        fh_ids = {t.get("task_id", "")
                  for t in final_holdout}
        overlap_wf = w_ids & fh_ids
        overlap_df = dev_ids & fh_ids
        if overlap_wf or overlap_df:
            raise ValueError(
                f"❌ final_holdout 与优化数据重叠: "
                f"witness∩final={len(overlap_wf)}, "
                f"dev∩final={len(overlap_df)}。"
                f"正式 test manifest 不允许自动过滤；"
                f"请修复数据划分。")
        # ⚠ 老师 audit #3（2026-07-10）：
        # dev ∩ witness 必须为空。dev 用于生成 candidate
        # 与估 P_T/P_0，witness 用于 gate。若重叠 →
        # candidate 在 dev 阶段已针对该 task 调整过，
        # witness 通过率偏乐观 → gate 失效。
        # 硬失败：不允许静默污染实验。
        dev_w_overlap = dev_ids & w_ids
        if dev_w_overlap:
            _example = list(dev_w_overlap)[:5]
            raise ValueError(
                f"❌ dev ∩ witness "
                f"= {len(dev_w_overlap)} tasks 重叠 "
                f"({_example}...)。"
                f"witness 不能与 dev 有交集，否则 "
                f"gate 结果偏乐观。请重新准备数据 "
                f"split。"
            )

        # 同时也检查 dev ∩ final_holdout / witness ∩
        # final_holdout：过滤后仍然可能因 seed 采样重复
        # 出现。硬失败保护。
        overlap_wf_post = w_ids & {
            t.get("task_id", "")
            for t in final_holdout}
        overlap_df_post = dev_ids & {
            t.get("task_id", "")
            for t in final_holdout}
        if overlap_wf_post or overlap_df_post:
            raise ValueError(
                f"❌ final_holdout 过滤后仍有重叠: "
                f"witness∩final={len(overlap_wf_post)}, "
                f"dev∩final={len(overlap_df_post)}。"
                f"请检查数据 split。"
            )

        if require_final_holdout \
                and len(final_holdout) < 5:
            raise ValueError(
                f"final_holdout 有效 tasks 少于 5 "
                f"(={len(final_holdout)})，"
                f"过滤重叠后剩余太少。")
        print(f"  Final Holdout: "
              f"{len(final_holdout)} tasks "
              f"(dev={len(dev_ids)}, "
              f"witness={len(w_ids)}, "
              f"不参与 gate 或 best selection)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # dump_prompts=True 会把每个 LLM 交互完整落盘到
    # logs/<run>/prompts/，便于事后 debug。
    # 金钱/磁盘敏感时置为 False。
    _dump_prompts = bool(s1.get(
        "log_dump_prompts", False))
    logger = OSDLogger(
        log_dir="logs",
        experiment=f"study1_{ts}",
        dump_prompts=_dump_prompts,
        console_level="WARNING",
    )

    print(f"\n  Student: {cfg['student']['model']}")
    print(f"  Teacher: {cfg['teacher']['model']}")
    print(f"  β={beta_mode} abl={ablation}")
    print(f"  N={N_max} S={S_max} M={M} B={B_W}")

    client = OpenRouterClient(cfg)
    agent  = StudentAgent(cfg, client)
    teacher = TeacherService(cfg, client)
    evaluator = Evaluator(cfg, agent)
    validator = Validator(cfg)
    parser_llm = ParserLLM(cfg, client)

    # Teacher rollout 参数
    teacher_rollout_n = s1.get(
        "teacher_rollout_n", 15)
    ref_rollout_n = s1.get(
        "ref_rollout_n", 15)
    # 并行 worker 数（Teacher/Ref/Candidate rollout 共用）
    witness_workers = s1.get("witness_workers", 4)

    # ── 加载 skill ─────────────────────────
    # 优先：单文件模式 data/skill/SKILL.md
    # 其次：start_library_dir（指定目录）
    # 最后：library_dir（旧多 skill 库）
    skill_file = paths.get("skill_file", "")
    start_dir = s1.get("start_library_dir", "")
    if skill_file and os.path.exists(skill_file):
        K_n = SkillLibrary.load_single_file(
            skill_file)
    elif start_dir and os.path.exists(start_dir):
        K_n = SkillLibrary.load(start_dir)
    else:
        K_n = SkillLibrary.load(paths["library_dir"])
    print(f"  Skills: {len(K_n)}")

    K_n.ensure_all_appendix()
    K_n.ensure_all_slow_update()

    # π₀ baseline
    print("\n  [π₀] baseline...")
    ref = ReferenceBaseline(cfg, agent)
    if run_ref:
        if not ref.load(ref_path):
            ref.build(witness, save_path=ref_path)
    print(f"  R̄_0 = {ref.mean():.4f}")

    evaluator.update_study_config("study1")
    w_cache = evaluator.witness_estimate(
        K_n, witness, desc="Initial")
    J_W_init = w_cache["J_W"]
    print(f"  J_W = {J_W_init:.4f}")

    # ─ Logger: initial witness ─
    logger.witness_eval(
        tag="initial",
        J_W=J_W_init,
        J_W_hard=w_cache.get("J_W_hard"),
        J_W_soft=w_cache.get("J_W_soft"),
        n_tasks=len(witness),
    )
    # ─ Logger: dump initial skill snapshot ─
    try:
        _snap0 = "\n\n".join([
            f"# skill_id: {sk.skill_id}\n"
            f"# name: {sk.name}\n"
            f"# description:\n{sk.description}\n"
            f"\n## execution_body:\n"
            f"{sk.execution_body or ''}"
            for sk in K_n
        ])
        logger.dump_text(
            f"SNAPSHOT_initial", _snap0)
    except Exception:
        pass

    fn_corr_init = None
    if compute_fn:
        fn_r = evaluator.compute_fn_correlation(
            K_n, witness[:15])
        fn_corr_init = fn_r.get("rho_FN")

    os.makedirs(paths["results_study1"], exist_ok=True)

    K_best    = K_n.clone()
    J_W_best  = J_W_init
    best_step = -1

    history = {
        "timestamp": ts,
        "student_model": cfg["student"]["model"],
        "teacher_model": cfg["teacher"]["model"],
        "hyperparams": s1,
        "beta_mode": beta_mode,
        "beta_value": beta_value,
        "ablation_mode": ablation,
        "J_W_init": J_W_init,
        "fn_corr_init": fn_corr_init,
        "R0_mean": ref.mean(),
        "rounds": [],
        "total_accepted": 0,
        "total_stalls": 0,
        "accepted_by_type": {
            "execution_edit": 0,
            "appendix_edit": 0},
    }

    residual_ev  = ResidualEvidence(cfg)
    stall_count  = 0
    n_accepted   = 0
    last_action  = ""
    cum_attr     = defaultdict(float)
    cum_n        = 0
    rej_buffer   = []
    K_prev       = None
    epoch_rounds = 0

    # ════════════════════════════════════════
    # 主循环
    # ════════════════════════════════════════
    for n in range(N_max):
        cost_start = client.total_cost_usd
        epoch_rounds += 1

        if max_cost > 0 and \
                client.total_cost_usd >= max_cost:
            print(f"\n  💰 budget exhausted")
            break

        cur_M = compute_learning_rate(
            n, N_max, lr_max, lr_min, lr_mode)

        print(f"\n{'═'*60}")
        print(f"  R{n} | stall={stall_count}/{S_max}"
              f" | acc={n_accepted}"
              f" | J_W={w_cache['J_W']:.4f}"
              f" | M={cur_M} β={beta_mode}")
        print(f"{'═'*60}")

        logger.round_start(
            n, J_W=w_cache["J_W"], stall=stall_count,
            cur_M=cur_M, library_size=len(K_n),
            skill_len=sum(len(sk.execution_body or "") for sk in K_n),
            beta_mode=beta_mode)

        log = {"round": n,
               "stall_before": stall_count,
               "beta_mode": beta_mode,
               "current_M": cur_M}

        # ══════════════════════════════════════
        # Step 2 (§3.1): Parser LLM 冻结 Z, φ
        # 每轮调用一次，输出后冻结
        # ══════════════════════════════════════
        parsed_modules = []
        parsed_labels = []
        for sk in K_n:
            skill_text = "\n".join([
                f"# Skill: {sk.name}",
                f"Description: {sk.description}",
                sk.execution_body or "",
                *[f"## Reference {name}\n{body}"
                  for name, body in sk.references.items()],
            ])
            if not skill_text.strip():
                raise RuntimeError(
                    f"cannot parse empty skill {sk.skill_id}")
            parsed_one = parser_llm.parse_skill(skill_text)
            modules_one = parser_llm.to_modules(
                parsed_one, skill_id=sk.skill_id,
                force_single_group=s1.get(
                    "parser_force_single_group", False),
                min_labels_per_group=s1.get(
                    "parser_min_labels_per_group", 3))
            parsed_modules.extend(modules_one)
            parsed_labels.extend(parsed_one.labels)
        if not parsed_modules:
            raise RuntimeError(
                "parser produced no valid frozen measurement groups")
        log["parser_n_labels"] = len(parsed_labels)

        # ─ Logger: parser output ─
        try:
            logger.parser_output(
                n_labels=log["parser_n_labels"],
                n_modules=len(parsed_modules)
                    if parsed_modules else 0,
                labels=[
                    getattr(l, "name", str(l))
                    for l in parsed_labels
                ][:20],
                patterns={
                    m.module_id: {
                        "history_id": m.history_id,
                        "group_id": m.group_id,
                        "skill_id": m.skill_id,
                        "skill_step_text": m.skill_step_text,
                        "Z_h": list(m.named_behaviors) + ["other"],
                        "label_patterns": m.label_patterns,
                        "label_parents": m.label_parents,
                    }
                    for m in parsed_modules},
            )
        except Exception as _e:
            logger.error(
                where="parser_output",
                message=str(_e),
                exc_type=type(_e).__name__)

        # ── Step 3 (§3.2): Sample + Rollout ───
        agent.clear_cache()
        V_n = stratified_sample_vn(
            dev_pool, n, stall_count, last_action,
            max(1, s1.get("B_roll",10)//10),
            vn_stall_min, stall_growth,
            post_accept, n*42,
            full_pool_every_round=full_pool_vn)
        log["V_n_size"] = len(V_n)

        # 单 skill → harness 强制激活
        # 多 skill → implicit 自然激活
        is_single = (len(K_n) == 1)
        act_mode = "harness" if is_single else "implicit"
        forced_id = (K_n.skill_ids()[0]
                     if is_single else None)

        def _run_student(task):
            return agent.run_task(
                task, K_n,
                activation_mode=act_mode,
                forced_skill_id=forced_id)

        trajectories = parallel_map_interruptible(
            _run_student, V_n,
            workers=witness_workers,
            desc=f"    Rollouts (x{witness_workers})",
        )
        # 过滤 None（异常失败的）
        trajectories = [t for t in trajectories
                        if t is not None]

        sr = float(np.mean(
            [t.success for t in trajectories])
            if trajectories else 0.0)
        log["rollout_success_rate"] = sr
        failed = [t for t in trajectories
                  if not t.success]
        success = [t for t in trajectories
                   if t.success]
        print(f"\n    {len(trajectories)} traj, "
              f"sr={sr:.3f}")

        # ─ Logger: rollout summary ─
        _mean_r = float(np.mean([
            t.final_reward for t in trajectories
        ])) if trajectories else 0.0
        logger.rollout_summary(
            n=len(trajectories),
            success_rate=sr,
            mean_reward=_mean_r,
            activation_mode=act_mode,
            per_task=[
                {"task_id": t.task_id[:30],
                 "reward": round(
                     float(t.final_reward), 3),
                 "success": bool(t.success)}
                for t in trajectories[:5]
            ],
        )

        # ── Step 2: Diagnose ──────────────────
        # 限制诊断数量：最多诊断 60 条失败
        max_diagnose = min(len(failed), 60)
        failed_sample = (
            random.sample(failed, max_diagnose)
            if len(failed) > max_diagnose
            else failed)
        print(f"    Diagnosing {len(failed_sample)}"
              f"/{len(failed)} failed...")

        diagnoses = []
        diag_batches = list(range(
            0, len(failed_sample), mb_size))
        for i in tqdm(diag_batches,
                      desc="    Diagnose",
                      leave=False):
            batch = failed_sample[i:i+mb_size]
            if len(batch) > 1:
                bd, _ = teacher.diagnose_minibatch(
                    batch, K_n, beta_mode)
                diagnoses.extend(bd)
            else:
                for t in batch:
                    d, _ = teacher.diagnose(
                        t, K_n, beta_mode)
                    diagnoses.append(d)

        error_types = defaultdict(int)
        n_defect = n_lapse = 0
        for d in diagnoses:
            r_t = d.get("r_t", {})
            if r_t:
                error_types[
                    max(r_t, key=r_t.get)] += 1
            ft = d.get("failure_type", "")
            if ft == "skill_defect":
                n_defect += 1
            elif ft == "execution_lapse":
                n_lapse += 1

        # appendix notes
        app_notes = \
            teacher.extract_appendix_notes_from_diagnoses(
                diagnoses)
        log["error_type_dist"] = dict(error_types)
        log["n_skill_defect"] = n_defect
        log["n_execution_lapse"] = n_lapse

        # ─ Logger: diagnose summary ─
        _r_t_avg = defaultdict(float)
        _r_t_cnt = 0
        for d in diagnoses:
            r_t = d.get("r_t", {}) or {}
            for k_, v_ in r_t.items():
                try:
                    _r_t_avg[str(k_)] += float(v_)
                except Exception:
                    pass
            if r_t:
                _r_t_cnt += 1
        if _r_t_cnt:
            for k_ in list(_r_t_avg.keys()):
                _r_t_avg[k_] /= _r_t_cnt
        logger.diagnose_summary(
            n=len(diagnoses),
            n_defect=n_defect,
            n_lapse=n_lapse,
            error_type_dist=dict(error_types),
            r_t_avg=dict(_r_t_avg),
        )

        # success analysis
        success_patches = []
        if success:
            sp, _ = teacher.analyze_successes(
                success, K_n, M=2)
            success_patches = sp

        diagnoses_map = {}
        for t, d in zip(failed_sample, diagnoses):
            diagnoses_map[t.task_id] = d
        for t in trajectories:
            if t.task_id not in diagnoses_map:
                diagnoses_map[t.task_id] = {}

        # ── Step 3: b4 grading ────────────────
        print("\n  [Step 3] ExSkill-OPD 信号...")
        teacher_scores = {}
        verified_solutions = {}
        b4_grades = {}

        if (beta_mode == "b3"
                and s1.get("teacher_self_execute")):
            vs = teacher.generate_verified_solutions_batch(
                failed, K_n, teacher_n)
            verified_solutions = vs
            for tid, (code, _) in vs.items():
                teacher_scores[tid] = \
                    1.0 if code else 0.5

        if beta_mode == "b4":
            b4_grades, _ = \
                teacher.generate_student_trajectory_grades_batch(
                    failed, K_n, teacher_n)
            attach_teacher_grades_to_trajectories(
                trajectories, b4_grades)
            teacher_scores = \
                teacher_scores_from_grades(
                    trajectories, b4_grades)
            n_b4_fail = sum(
                1 for g in b4_grades.values()
                if g.parse_failed)
            log["b4_parse_failure_count"] = n_b4_fail
            # ─── 网络健康检测 ──────────────
            # 全部 b4 grading 都解析失败 → API 大概率挂了
            # 早停，避免空跑浪费时间
            if b4_grades and \
                    n_b4_fail == len(b4_grades) \
                    and sr == 0:
                print(f"\n  ❌ API 疑似全部失败："
                      f"sr=0, b4_parse_fail="
                      f"{n_b4_fail}/{len(b4_grades)}")
                print("  🛑 提前终止：请检查网络/代理/额度")
                break

        verified_solutions = (
            verified_solutions
            if beta_mode == "b3"
            else None)

        # ══════════════════════════════════════
        # Step 5-6 (§3.2): Teacher + Reference rollout
        # 关键（2026-07-10 修复）：
        # Teacher / Reference / Student 必须跑**同一任务集**，
        # 否则 P_T / P_0 / P_s 的对比会混入任务采样偏差，
        # r_T = log(P_T/P_0) 的信号被稀释。
        #
        # 做法：从 V_n 里选出 shared_sample，Teacher 和
        # Reference 都跑这个子集；Student 已经跑过完整 V_n
        # （包含 shared_sample），所以 P_s 也能在同一子集上算。
        #
        # ⚠ 老师 audit（2026-07-10 issue #2）：
        #   三端必须**逐一跑同一批** task_id，且用
        #   完全相同的 max_steps / 环境 / 评分器。
        #   之前 shared_n = max(...) 让三端分别取
        #   前 N 个（不同 N）→ 实际 50/30/50 而非
        #   同一批；且 Teacher max_steps=2, Ref/Student=3
        #   → 协议不同。
        #
        # 新做法：
        # - shared_n = teacher_rollout_n = ref_rollout_n
        #   （取二者共同值；若不同以 min 为准以保证
        #    三端完全对齐）
        # - Teacher / Ref / Student 都跑这 shared_n 个
        #   完全相同的 task_id
        # - Teacher / Ref 都用 max_steps=None（走 config
        #   里的 student.max_steps 默认，与 Student 一致）
        # ══════════════════════════════════════
        # ⚠ 2026-07-10 严格对齐：
        # shared_sample = V_n 整个（无截断）。
        # Teacher / Ref / Student 都跑完整 V_n。
        # teacher_rollout_n / ref_rollout_n 只作 cap
        # 上限，不再限制三端不同 N。
        # 若 teacher_rollout_n = 0 → 跳过 Teacher
        # 若 ref_rollout_n = 0 → 跳过 Reference
        shared_sample = V_n
        # 用 task_id 集合方便对齐 P_s
        shared_task_ids = {
            t.get("task_id", "")
            for t in shared_sample}
        print(f"    [align] shared_sample "
              f"{len(shared_sample)} tasks "
              f"= 完整 V_n，Teacher/Ref/Student "
              f"严格同任务同协议")

        if teacher_rollout_n <= 0:
            print(f"    ⏭ Teacher rollout skipped "
                  f"(teacher_rollout_n=0)")
            teacher_trajs = []
            log["teacher_rollout_n"] = 0
            log["teacher_sr"] = 0.0
        else:
            # Teacher 跑完整 shared_sample（不再截断）
            teacher_sample = shared_sample
            print(f"    Teacher rollout: "
                  f"{len(teacher_sample)} tasks "
                  f"(no-skill, 并行 x{witness_workers}, "
                  f"预计 ~{len(teacher_sample)*15//witness_workers}s)...")
            import time as _time
            _t_start = _time.time()

            def _run_teacher(task):
                # 与 Student/Ref 同 max_steps（None=用
                # student.max_steps 默认值），协议对齐。
                # 之前 max_steps_override=2 是"给 Sonnet
                # 一次自纠错机会"的作弊，破坏协议对称。
                return agent.run_task_with_model(
                    task, K_n,
                    model_override=cfg[
                        "expert"]["model"],
                    activation_mode="no_skill",
                    forced_skill_id=None,
                    max_steps_override=None)

            teacher_trajs = (
                parallel_map_interruptible(
                    _run_teacher, teacher_sample,
                    workers=witness_workers,
                    desc="    Teacher",
                )
            )
            teacher_trajs = [t for t in teacher_trajs
                             if t is not None]
            _t_elapsed = _time.time() - _t_start
            n_done = len(teacher_trajs)
            print(f"    Teacher done in "
                  f"{_t_elapsed:.0f}s "
                  f"({n_done}/{len(teacher_sample)} "
                  f"完成)")
            t_sr = float(np.mean(
                [t.success for t in teacher_trajs])
                if teacher_trajs else 0.0)
            print(f"    Teacher sr={t_sr:.3f}")
            log["teacher_rollout_n"] = n_done
            log["teacher_sr"] = t_sr
            # ─ Logger: teacher rollout ─
            _t_mean_r = float(np.mean([
                t.final_reward for t in teacher_trajs
            ])) if teacher_trajs else 0.0
            logger.rollout_summary(
                n=n_done,
                success_rate=t_sr,
                mean_reward=_t_mean_r,
                activation_mode="teacher_no_skill",
                durations_s=_t_elapsed,
                per_task=[
                    {"task_id": t.task_id[:30],
                     "reward": round(
                         float(t.final_reward), 3),
                     "success": bool(t.success)}
                    for t in teacher_trajs[:5]
                ],
            )

            # 诊断：Teacher 0% 时打印首条失败详情
            if t_sr == 0 and teacher_trajs:
                failed_t = teacher_trajs[0]
                err = (failed_t.score_detail
                       or {}).get("error", "unknown")
                last_resp = (
                    failed_t.steps[-1].action
                    if failed_t.steps else "")
                print(f"    ⚠ Teacher first-fail: "
                      f"err={err[:80]}")
                print(f"    ⚠ Teacher resp[0:200]="
                      f"{last_resp[:200]!r}")

        # ══════════════════════════════════════
        # Step 6 (§3.2): Reference rollout
        # 严格与 Teacher/Student 相同的 shared_sample
        # （老师 audit issue #2：三端同任务同协议）
        # ══════════════════════════════════════
        ref_sample = shared_sample
        print(f"    Reference rollout: "
              f"{len(ref_sample)} tasks "
              f"(严格同 Teacher/Student 任务, "
              f"并行 x{witness_workers})...")

        ref_trajs = parallel_map_interruptible(
            agent.run_task_no_skill, ref_sample,
            workers=witness_workers,
            desc="    Reference",
        )
        ref_trajs = [t for t in ref_trajs
                     if t is not None]
        r_sr = float(np.mean(
            [t.success for t in ref_trajs])
            if ref_trajs else 0.0)
        print(f"    Reference sr={r_sr:.3f}")
        log["ref_rollout_n"] = len(ref_trajs)
        log["ref_sr"] = r_sr
        # ─ Logger: reference rollout ─
        _r_mean_r = float(np.mean([
            t.final_reward for t in ref_trajs
        ])) if ref_trajs else 0.0
        logger.rollout_summary(
            n=len(ref_trajs),
            success_rate=r_sr,
            mean_reward=_r_mean_r,
            activation_mode="reference_no_skill",
            per_task=[
                {"task_id": t.task_id[:30],
                 "reward": round(
                     float(t.final_reward), 3),
                 "success": bool(t.success)}
                for t in ref_trajs[:5]
            ],
        )

        # 诊断：Ref 0% 时打印首条失败详情
        if r_sr == 0 and ref_trajs:
            failed_r = ref_trajs[0]
            err = (failed_r.score_detail or {}).get(
                "error", "unknown")
            last_resp = (
                failed_r.steps[-1].action
                if failed_r.steps else "")
            print(f"    ⚠ Ref first-fail: "
                  f"err={err[:80]}")
            print(f"    ⚠ Ref resp[0:200]="
                  f"{last_resp[:200]!r}")

        # ══════════════════════════════════════
        # Steps 9-14: Build Epoch Target（冻结）
        # η → P_+ → a_+ → λ_g → q_λ → g_n
        #
        # 关键（老师 audit issue #2）：把三端严格对齐
        # 到 shared_task_ids。任何一端缺失该 task 时，
        # 三端同时删除该 task，绝不做单端 fallback，
        # 否则 P_T/P_0/P_s/η 会混入任务采样偏差。
        # ══════════════════════════════════════
        # Student trajectories 过滤到 shared_task_ids
        student_by_tid = {
            t.task_id: t for t in trajectories
            if t.task_id in shared_task_ids
            and getattr(t, "evaluation_valid", False)
        }
        teacher_by_tid = {
            t.task_id: t
            for t in (teacher_trajs or [])
            if t.task_id in shared_task_ids
            and getattr(t, "evaluation_valid", False)
        }
        ref_by_tid = {
            t.task_id: t for t in (ref_trajs or [])
            if t.task_id in shared_task_ids
            and getattr(t, "evaluation_valid", False)
        }
        # 只保留三端都跑成的 task_id
        common_tids = (
            set(student_by_tid.keys())
            & (set(teacher_by_tid.keys())
               if teacher_trajs
               else set(student_by_tid.keys()))
            & (set(ref_by_tid.keys())
               if ref_trajs
               else set(student_by_tid.keys()))
        )
        ordered_common_tids = [
            task.get("task_id", "") for task in shared_sample
            if task.get("task_id", "") in common_tids]
        aligned_trajs = [student_by_tid[tid]
                         for tid in ordered_common_tids]
        aligned_teacher = [
            teacher_by_tid[tid]
            for tid in ordered_common_tids
            if tid in teacher_by_tid]
        aligned_ref = [
            ref_by_tid[tid]
            for tid in ordered_common_tids
            if tid in ref_by_tid]

        aligned_ids = set(common_tids)
        if ({t.task_id for t in aligned_trajs} != aligned_ids
                or {t.task_id for t in aligned_teacher} != aligned_ids
                or {t.task_id for t in aligned_ref} != aligned_ids):
            raise RuntimeError(
                "Student/Teacher/Reference valid task conditioning mismatch")

        if not aligned_trajs:
            # 三端交集为空 → 中止本轮 epoch_target 估计
            raise RuntimeError(
                "no common valid Student/Teacher/Reference tasks")
        else:
            n_align = len(aligned_trajs)
            n_shared = len(shared_task_ids)
            print(f"    [align] 三端交集 "
                  f"{n_align}/{n_shared} tasks "
                  f"(Student={len(student_by_tid)}, "
                  f"Teacher={len(teacher_by_tid)}, "
                  f"Ref={len(ref_by_tid)})")
            # 若对齐率过低，警告但仍用交集
            if n_align < n_shared * 0.5:
                print(f"    ⚠ 对齐率过低 "
                      f"({n_align}/{n_shared}"
                      f"={n_align/n_shared:.0%})，"
                      f"P_T/P_0 估计方差会较大")

        # Parser LLM maps all three aligned arms in one logical batch onto the
        # same frozen Skill-history universe. No applicability-regex fallback.
        epoch_mapping = parser_llm.map_rollout_steps(
            aligned_trajs + aligned_teacher + aligned_ref,
            parsed_modules,
            chunk_steps=s1.get("parser_mapping_chunk_steps", 30),
            evidence_chars=s1.get("parser_mapping_evidence_chars", 6000),
        )
        if not epoch_mapping.get("complete"):
            raise RuntimeError(
                "Parser rollout-history mapping failed: "
                f"{epoch_mapping.get('error', 'unknown')}")
        frozen_ruler_digest = hashlib.sha256(json.dumps(
            [m.__dict__ for m in parsed_modules], sort_keys=True,
            ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
        logger.history_mapping(
            phase="epoch_student_teacher_reference",
            complete=epoch_mapping.get("complete", False),
            n_steps=epoch_mapping.get("n_steps", 0),
            assignments=epoch_mapping.get("assignments", []),
            error=epoch_mapping.get("error", ""),
            frozen_ruler_digest=frozen_ruler_digest,
        )
        student_history_records = build_history_records(
            aligned_trajs, parsed_modules, "student")
        teacher_history_records = build_history_records(
            aligned_teacher, parsed_modules, "teacher")
        reference_history_records = build_history_records(
            aligned_ref, parsed_modules, "reference")

        epoch_target = build_epoch_target(
            library=K_n,
            trajectories=aligned_trajs,
            diagnoses_map=diagnoses_map,
            teacher_grades=b4_grades or None,
            ref_trajectories=aligned_ref or None,
            teacher_trajectories=(
                aligned_teacher or None),
            beta_budget=beta_budget,
            alpha=dirichlet_a,
            ref_baseline=ref,
            lambda_neutral=lambda_neutral,
            use_reward_calibrated_lambda=use_rc_lambda,
            parsed_modules=parsed_modules,
            eta_mode=eta_mode,
            use_exact_lambda=use_exact_lambda,
            student_records=student_history_records,
            teacher_records=teacher_history_records,
            reference_records=reference_history_records,
        )
        epoch_target.freeze()  # 步骤10：冻结
        epoch_target.assert_frozen()

        # 打印 epoch target 摘要
        # 每个 module = 一个 pooling group，独立 λ_g
        _n_groups = len(epoch_target.modules)
        print(f"    [epoch_target] "
              f"{_n_groups} pooling group(s), "
              f"β={epoch_target.beta:.2f} "
              f"(β 已中性化为常数 1.0)")
        for ms in epoch_target.modules:
            if ms.w > 0.01 and ms.g_n:
                top_g = sorted(
                    ms.g_n.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True)[:3]
                g_str = " ".join(
                    f"{z}:{v:+.2f}"
                    for z, v in top_g)
                # named 内的 λ 都是同一个标量，
                # 打印任一个即可（用第一个 non-other）
                lam_g_val = 0.0
                for z, v in ms.lambda_.items():
                    if z != "other":
                        lam_g_val = v
                        break
                # 组内 named label 数量
                n_named = len([
                    z for z in ms.lambda_.keys()
                    if z != "other"])
                gid_disp = (
                    ms.module_id
                    if hasattr(ms, "module_id")
                    and ms.module_id
                    else ms.skill_id)
                print(f"    [{gid_disp[:30]}] "
                      f"λ_g={lam_g_val:+.2f} "
                      f"(shared by {n_named} labels) "
                      f"w={ms.w:.3f}"
                      f" | top_g: {g_str}")
        log["epoch_beta"] = epoch_target.beta
        log["epoch_target_summary"] = \
            epoch_target.to_dict()
        # 每轮立即落地完整 frozen estimator table，避免异常退出丢失。
        _round_artifact_dir = os.path.join(
            paths["results_study1"], f"round_{n:03d}_{ts}")
        os.makedirs(_round_artifact_dir, exist_ok=True)
        save_json(epoch_target.to_dict(), os.path.join(
            _round_artifact_dir, "estimator_target.json"))
        save_json({
            "parser_labels": [
                getattr(label, "__dict__", str(label))
                for label in parsed_labels],
            "skill_histories": [m.__dict__ for m in parsed_modules],
        }, os.path.join(
            _round_artifact_dir, "parser_measurement.json"))
        save_json({
            "framework_version": "ooosd-parent-kl-point-v1",
            "artifact_schema_version": 2,
            "ordered_task_ids": ordered_common_tids,
            "frozen_ruler_digest": frozen_ruler_digest,
            "mapping": epoch_mapping,
        }, os.path.join(
            _round_artifact_dir, "epoch_history_mapping.json"))

        # ─ Logger: epoch target ─
        # 把 ModuleSignal 的完整字段落盘，事后可分析
        # 每个 label 的 P_s / P_T / P_0 / R_hat / η / a_+ /
        # Q_n / g_n / λ 的具体数值。
        _mods_for_log = []
        for ms in epoch_target.modules:
            _top_g = dict(sorted(
                (ms.g_n or {}).items(),
                key=lambda x: abs(x[1]),
                reverse=True)[:5])
            # named_behaviors 可能不在 ModuleSignal 里，
            # 从 lambda_ / P_s 的键推断
            _named = [
                k for k in (ms.lambda_ or {}).keys()
                if k != "other"]
            _mods_for_log.append({
                "module_id":       ms.module_id,
                "skill_id":        ms.skill_id,
                "d_s":             float(ms.d_s),
                "w":               float(ms.w),
                "named_behaviors": _named,
                "lambda_":         ms.lambda_ or {},
                "P_s":             ms.P_s or {},
                "P_T":             ms.P_T or {},
                "P_0":             ms.P_0 or {},
                "R_hat":           ms.R_hat or {},
                # reward-calibrated 中间量
                "eta":             getattr(
                    ms, "eta", {}) or {},
                "a_plus":          getattr(
                    ms, "a_plus", {}) or {},
                "obs_counts":      getattr(
                    ms, "obs_counts", {}) or {},
                "Q_n":             ms.Q_n or {},
                "g_n":             ms.g_n or {},
                "donor_plus":      ms.donor_plus or "",
                "donor_minus":     ms.donor_minus or "",
                "g_n_top":         _top_g,
                "group_id":        ms.group_id,
                "history_id":      ms.history_id,
                "skill_step_text": ms.skill_step_text,
                "label_patterns":  ms.label_patterns,
                "label_parents":   ms.label_parents,
                "Z_h":             list(ms.Q_n.keys()),
            })
        logger.epoch_target(
            beta=float(epoch_target.beta),
            modules=_mods_for_log,
            target_summary=str(
                log.get("epoch_target_summary", ""))[:400],
        )

        # ── Step 5: ExSkill-OPD signals ───────
        exskill_signals = residual_ev.compute_batch(
            trajectories, ref, teacher_scores,
            epoch_target.beta, beta_mode,
            diagnoses_map, en_dispatch,
            en_execution, epoch_target)
        ex_summary = residual_ev.summary(
            exskill_signals)
        log["exskill_summary"] = ex_summary

        skill_support = merge_signals_by_skill(
            exskill_signals, diagnoses_map)
        log["skill_support"] = {
            sid: v["support_count"]
            for sid, v in skill_support.items()}

        attr, cum_attr, cum_n = compute_attribution(
            trajectories, diagnoses, K_n,
            alpha_attr, beta_unc,
            cum_attr, cum_n)
        log["neighborhood"] = attr["neighborhood"]
        log["uncovered_rate"] = attr["uncovered_rate"]

        # ── Step 6: Propose ───────────────────
        sig_map = {s.task_id: s
                   for s in exskill_signals}
        f_sigs = [sig_map.get(t.task_id)
                  for t in failed
                  if sig_map.get(t.task_id)]

        raw_props, _ = teacher.propose_mixed_edits(
            failed, K_n, diagnoses, cur_M,
            en_dispatch, en_execution, beta_mode,
            f_sigs,
            verified_solutions,
            rej_buffer, epoch_target=epoch_target)

        # appendix edits from EXECUTION_LAPSE
        for sid, notes in app_notes.items():
            if ablation != "execution_only" and notes and K_n.get(sid):
                raw_props.append({
                    "edit_type": "appendix_edit",
                    "skill_id": sid,
                    "appendix_notes": notes,
                    "rationale": "lapse reminders"})

        log["n_proposals"] = len(raw_props)

        # ─ Logger: proposal batch (初次载入，后面 filter 后再补 valid) ─
        # (先预存 raw_props，valid_cands 数量在下面知道)

        # ── Step 7: Filter ────────────────────
        valid_cands = []
        for prop in raw_props:
            et = prop.get("edit_type", "")
            if et == "execution_edit":
                if not all(k in prop for k in
                    ["skill_id",
                     "parent_location", "old_text", "new_text"]):
                    continue
            elif et == "appendix_edit":
                if not all(k in prop for k in
                    ["skill_id","appendix_notes"]):
                    continue
            else:
                continue
            if K_n.get(prop["skill_id"]) is None:
                continue
            cand_lib = apply_edit(K_n, prop, epoch_target)
            if (et == "execution_edit" and
                    cand_lib.get(prop["skill_id"]).execution_body ==
                    K_n.get(prop["skill_id"]).execution_body):
                continue
            sec = validator.validate(cand_lib, K_n)
            if sec["all_passed"]:
                valid_cands.append({
                    "proposal": prop,
                    "cand_lib": cand_lib,
                    "edit_type": et})

        print(f"    valid: "
              f"{len(valid_cands)}/{len(raw_props)}")

        # ─ Logger: proposal batch ─
        logger.proposal_batch(
            n_raw=len(raw_props),
            n_valid=len(valid_cands),
            proposals=raw_props,
        )

        # ── No valid candidates ───────────────
        if not valid_cands:
            stall_count += 1
            last_action = "stall"
            log["action"] = "stall"
            log["J_W_after"] = w_cache["J_W"]
            for p in raw_props:
                if p not in rej_buffer:
                    rej_buffer.append(p)
            if K_prev is None:
                K_prev = K_n.clone()
            if enable_slow and K_prev \
                    and epoch_rounds >= slow_freq \
                    and len(dev_pool) >= slow_n:
                sl, ok = try_slow_update(
                    K_n, K_prev, teacher, agent,
                    dev_pool, slow_n,
                    min_margin=slow_min_margin,
                    evaluator=evaluator,
                    witness=witness,
                    w_cache=w_cache,
                    J_W_floor=J_W_best,
                    require_gate=slow_require_gate,
                    epoch_target=epoch_target,
                    act_mode=act_mode,
                    forced_id=forced_id,
                    dirichlet_a=dirichlet_a,
                    witness_workers=witness_workers)
                if ok: epoch_rounds = 0
                log["slow_update"] = sl
                # ── 若 slow_update 接受，刷新
                # w_cache / K_best / J_W_best（issue #6）
                if sl.get("accepted") and \
                        sl.get("candidate_lib"):
                    cw = sl.get(
                        "candidate_witness", {})
                    if cw.get("J_W_candidate"
                              ) is not None:
                        K_n = sl["candidate_lib"]
                        w_cache = {
                            "J_W": cw[
                                "J_W_candidate"],
                            "J_W_hard":
                                cw.get(
                                    "J_W_hard_candidate",
                                    cw["J_W_candidate"]),
                            "J_W_soft":
                                cw.get(
                                    "J_W_soft_candidate",
                                    0),
                            "per_task_rewards":
                                cw.get(
                                    "candidate_rewards",
                                    w_cache["per_task_rewards"]),
                            "per_task_soft_rewards":
                                cw.get(
                                    "candidate_soft_rewards",
                                    w_cache.get(
                                        "per_task_soft_rewards")),
                        }
                        if cw["J_W_candidate"] \
                                > J_W_best:
                            K_best = K_n.clone()
                            J_W_best = cw[
                                "J_W_candidate"]
                            best_step = n
                            print(f"    🏆 "
                                  f"slow_update BEST "
                                  f"J_W={J_W_best:.4f}")
            else:
                log["slow_update"] = {
                    "triggered": False}
            log["cost"] = round(
                client.total_cost_usd-cost_start, 4)
            history["rounds"].append(log)
            print(f"\n  ⏸️ STALL "
                  f"buf={len(rej_buffer)}")
            logger.stall(
                reason=f"no valid candidates "
                       f"(raw={len(raw_props)})")
            if log.get("slow_update", {}).get(
                "triggered"):
                _sl = log["slow_update"]
                logger.slow_update(
                    triggered=True,
                    improvement_rate=_sl.get(
                        "improvement_rate", 0.0),
                    regression_rate=_sl.get(
                        "regression_rate", 0.0),
                    skipped=bool(_sl.get(
                        "skipped", False)),
                    reason=str(_sl.get("reason", "")),
                    n_updated=len(_sl.get(
                        "meta_skills", []) or []),
                )
            logger.round_end(
                action="stall",
                J_W_after=log.get(
                    "J_W_after", w_cache["J_W"]),
                delta_hat=0.0,
                n_proposals=log.get("n_proposals", 0),
                n_admissible=0,
                cost_this_round=log.get("cost", 0.0),
                rollout_sr=log.get(
                    "rollout_success_rate", 0.0),
                teacher_sr=log.get("teacher_sr", 0.0),
                ref_sr=log.get("ref_sr", 0.0),
                reason="no_valid_candidates",
            )
            if stall_count >= S_max:
                print(f"  🛑 S_max={S_max}")
                break
            continue

        # ── Step 8: Witness + Dual Gate ───────
        admissible = []

        # ── frozen-Q true KL gate 用 witness ──────────────
        # KL 与 reward 都在同一批 paired witness 上计算。
        # ⚠ 老师 2026-07-11（核心修复 #4）：
        # 每次 evaluate candidate 时，incumbent 也在 witness
        # 上重跑 B_W 次（完整 witness_estimate + return_trajectories）。
        # KL gate 用 inc/cand 各 B_W 条 trajs 算 P_inc/P_cand；
        # reward gate 用同一批 rewards 算 paired point estimate。
        # 这样两道门共用**同一批对称 paired bundles**。
        #
        # 跑法：每次 evaluate 前重跑 incumbent（不再用旧 w_cache）
        # 见下面 for 循环内 inc_witness = ...
        old_aligned_trajs = []

        if ablation == "no_gate":
            # 真 no-gate：选择第一个通过静态验证的候选；不访问 witness、
            # 不计算任何 gate measurement。
            item = valid_cands[0]
            result = {
                "is_admissible": True,
                "ablation_no_gate": True,
                "reason": "first_static_valid_candidate",
                "J_W_baseline": w_cache["J_W"],
                "J_W_candidate": w_cache["J_W"],
                "J_W_hard_candidate": w_cache.get(
                    "J_W_hard", w_cache["J_W"]),
                "J_W_soft_candidate": w_cache.get("J_W_soft", 0.0),
                "candidate_rewards": w_cache.get("per_task_rewards", []),
                "candidate_soft_rewards": w_cache.get(
                    "per_task_soft_rewards", []),
                "delta_hat": 0.0,
                "label_kl_decrease": 0.0,
            }
            item["witness"] = result
            admissible.append(item)
            print("    [no_gate] first static-valid candidate selected")

        for idx, item in enumerate(
            ([] if ablation == "no_gate" else
             valid_cands[:min(3, len(valid_cands))])
        ):
            prop = item["proposal"]
            et = item["edit_type"]
            print(f"    [Cand {idx+1}] "
                  f"[{et[:8]}] "
                  f"{prop['skill_id'][:35]}")

            # ⚠ 老师 2026-07-11（核心修复 #4）：
            # 同一批对称 paired bundles。
            # incumbent 和 candidate 都跑 witness B_W 次，
            # 每 task 的 (inc_trajs, cand_trajs) 构成 bundle。
            # 两道门共用这批 bundles。
            print(f"      running paired witness "
                  f"(inc + cand, B_W×2 rollouts/task)...")
            inc_witness = evaluator.witness_estimate(
                K_n, witness,
                desc=f"Inc(cand{idx+1})",
                activation_mode=act_mode,
                forced_skill_id=forced_id,
                return_trajectories=True,
            )
            cand_witness = evaluator.witness_estimate(
                item["cand_lib"], witness,
                desc=f"Cand{idx+1}",
                activation_mode=act_mode,
                forced_skill_id=forced_id,
                return_trajectories=True,
            )

            # 展开 per_task_trajectories → 扁平列表
            def _flatten(w):
                out = []
                for tl in w.get(
                    "per_task_trajectories", []):
                    for t in tl:
                        if t is not None:
                            out.append(t)
                return out
            inc_trajs = _flatten(inc_witness)
            cand_trajs = _flatten(cand_witness)

            frozen_witness_modules = [Module(
                module_id=ms.module_id,
                history_id=ms.history_id or ms.module_id,
                group_id=ms.group_id or ms.module_id,
                skill_id=ms.skill_id,
                named_behaviors=[z for z in ms.Q_n if z != "other"],
                reach_patterns=list(ms.reach_patterns),
                label_patterns=dict(ms.label_patterns),
                label_parents=dict(ms.label_parents),
                skill_step_text=ms.skill_step_text,
            ) for ms in epoch_target.modules if ms.estimable and ms.Q_n]
            witness_mapping = parser_llm.map_rollout_steps(
                inc_trajs + cand_trajs, frozen_witness_modules,
                chunk_steps=s1.get("parser_mapping_chunk_steps", 30),
                evidence_chars=s1.get(
                    "parser_mapping_evidence_chars", 6000),
            )
            if witness_mapping.get("complete"):
                label_kl_res = compute_candidate_label_kl(
                    item["cand_lib"], cand_trajs,
                    epoch_target, K_n, dirichlet_a,
                    trajectories_old_aligned=inc_trajs,
                    return_lcb=False,
                )
            else:
                print("      ⚠ Parser witness mapping failed: "
                      f"{witness_mapping.get('error', 'unknown')}")
                label_kl_res = None
            logger.history_mapping(
                phase=f"witness_candidate_{idx + 1}",
                complete=witness_mapping.get("complete", False),
                n_steps=witness_mapping.get("n_steps", 0),
                assignments=witness_mapping.get("assignments", []),
                error=witness_mapping.get("error", ""),
                frozen_ruler_digest=epoch_target.freeze_digest,
            )
            label_kl = label_kl_res
            _kl_display = (f"{label_kl:+.4f}"
                       if label_kl is not None else "MISSING")
            print(f"      true_KL_decrease "
                f"point={_kl_display}, "
                  f"applying dual gate...")

            # Dual gate: true KL point + paired reward point + strict floor
            # 关键：incumbent 和 candidate 都传本次的 rollout
            # 结果（同批对称）
            result = evaluator.evaluate_candidate(
                K_n, item["cand_lib"], witness,
                baseline_cache=inc_witness,
                label_kl_decrease=label_kl,
                J_W_floor=J_W_best,
                candidate_cache=cand_witness,
            )
            item["witness"] = result

            save_json({
                "framework_version": "ooosd-parent-kl-point-v1",
                "artifact_schema_version": 2,
                "gate_mode": "paired_point_estimate",
                "kl_metric": "frozen_q_true_kl_decrease",
                "reward_floor_policy": "strict_running_best",
                "reward_epsilon": reward_epsilon,
                "label_kl": label_kl_res,
                "history_mapping": witness_mapping,
                "incumbent_rewards": inc_witness.get(
                    "per_task_rewards", []),
                "candidate_rewards": cand_witness.get(
                    "per_task_rewards", []),
                "incumbent_soft_rewards": inc_witness.get(
                    "per_task_soft_rewards", []),
                "candidate_soft_rewards": cand_witness.get(
                    "per_task_soft_rewards", []),
                "gate_result": result,
            }, os.path.join(
                _round_artifact_dir,
                f"candidate_{idx + 1:02d}_gate.json"))

            if result["is_admissible"]:
                admissible.append(item)
                print(f"      ✅ ADMISSIBLE")
            else:
                print(f"      ❌ REJECTED")

            # ─ Logger: witness eval for this candidate ─
            logger.witness_eval(
                tag=f"cand{idx+1}",
                J_W=result.get(
                    "J_W_candidate", 0.0),
                J_W_hard=result.get(
                    "J_W_hard_candidate"),
                J_W_soft=result.get(
                    "J_W_soft_candidate"),
                n_tasks=len(witness),
            )

            # ─ Logger: candidate gate ─
            logger.candidate_gate(
                cand_idx=idx + 1,
                skill_id=prop.get("skill_id", ""),
                edit_type=et,
                label_kl_decrease=(
                    label_kl),
                delta_hat=result.get(
                    "delta_hat", 0.0),
                reward_epsilon=result.get(
                    "reward_epsilon", reward_epsilon),
                J_W_baseline=result.get(
                    "J_W_baseline",
                    w_cache["J_W"]),
                J_W_candidate=result.get(
                    "J_W_candidate", 0.0),
                J_W_floor=J_W_best,
                is_admissible=bool(
                    result["is_admissible"]),
                gate_details={
                    k: v for k, v in result.items()
                    if k in (
                        "reason",
                        "reward_passed",
                        "label_kl_passed",
                        "floor_passed",
                        "ablation_no_gate",
                    )
                },
            )

        log["n_admissible"] = len(admissible)

        # ── Step 9: Accept or Stall ───────────
        if admissible:
            # 候选排序使用同一个 frozen target 上的真实 KL decrease。
            best = max(
                admissible,
                key=lambda item: item["witness"].get(
                    "label_kl_decrease", float("-inf")))
            prop = best["proposal"]
            result = best["witness"]
            et = best["edit_type"]

            # ─ 捕获 accept 前后 skill 内容（供 logger）─
            _sid_for_log = prop.get("skill_id", "")
            _skill_before_body = ""
            _skill_after_body = ""
            try:
                _sk_before = K_n.get(_sid_for_log)
                if _sk_before:
                    _skill_before_body = (
                        _sk_before.execution_body or "")
                _sk_after = best["cand_lib"].get(
                    _sid_for_log)
                if _sk_after:
                    _skill_after_body = (
                        _sk_after.execution_body or "")
            except Exception:
                pass

            K_prev = K_n.clone()
            epoch_rounds = 0
            K_n = best["cand_lib"]
            w_cache = {
                "J_W": result["J_W_candidate"],
                "J_W_hard": result.get(
                    "J_W_hard_candidate",
                    result["J_W_candidate"]),
                "J_W_soft": result.get(
                    "J_W_soft_candidate", 0),
                "per_task_rewards": result.get(
                    "candidate_rewards",
                    w_cache["per_task_rewards"]),
                "per_task_soft_rewards": result.get(
                    "candidate_soft_rewards",
                    w_cache.get(
                        "per_task_soft_rewards")),
            }
            stall_count = 0
            last_action = "accept"
            n_accepted += 1
            if et in history["accepted_by_type"]:
                history["accepted_by_type"][et] += 1

            if ablation == "no_gate":
                # no-gate 不使用 witness 做模型选择；最终产物就是最后一个
                # static-valid K_n，Final Holdout 仅在收尾评估一次。
                K_best = K_n.clone()
                best_step = n
            elif result["J_W_candidate"] > J_W_best:
                K_best = K_n.clone()
                J_W_best = result["J_W_candidate"]
                best_step = n
                print(f"    🏆 BEST "
                      f"J_W={J_W_best:.4f}")

            rej_buffer.clear()
            log["action"] = "accept"
            log["accepted_proposal"] = prop
            log["accepted_edit_type"] = et
            log["J_W_after"] = result[
                "J_W_candidate"]
            log["delta_hat"] = result["delta_hat"]
            print(f"\n  ✅ ACCEPT [{et}] "
                  f"Δ={result['delta_hat']:+.4f}")

            # ─ Logger: accept ─
            # 传入 skill 前后完整 execution_body，
            # 落盘完整 diff（若 dump_prompts=True）
            logger.accept(
                skill_id=prop.get("skill_id", ""),
                edit_type=et,
                J_W_before=result.get(
                    "J_W_baseline",
                    w_cache.get("J_W", 0.0)),
                J_W_after=result["J_W_candidate"],
                delta_hat=result["delta_hat"],
                proposal=prop,
                skill_before=_skill_before_body,
                skill_after=_skill_after_body,
            )
            # 快照：把当前完整 skill 落盘一份，
            # 方便事后逐轮回放
            try:
                _snap = "\n\n".join([
                    f"# skill_id: {sk.skill_id}\n"
                    f"# name: {sk.name}\n"
                    f"# description:\n{sk.description}\n"
                    f"\n## execution_body:\n"
                    f"{sk.execution_body or ''}"
                    for sk in K_n
                ])
                logger.dump_text(
                    f"SNAPSHOT_after_accept",
                    _snap)
            except Exception:
                pass
        else:
            stall_count += 1
            last_action = "stall"
            log["action"] = "stall"
            log["J_W_after"] = w_cache["J_W"]
            for item in valid_cands:
                if item.get("witness"):
                    rej_buffer.append(
                        item["proposal"])
            print(f"\n  ⏸️ STALL "
                  f"buf={len(rej_buffer)}")
            logger.stall(
                reason=f"no admissible "
                       f"(cands={len(valid_cands)})")

        if K_prev is None:
            K_prev = K_n.clone()

        # Slow update
        if enable_slow and ablation != "no_gate" and K_prev \
                and epoch_rounds >= slow_freq \
                and len(dev_pool) >= slow_n:
            sl, ok = try_slow_update(
                K_n, K_prev, teacher, agent,
                dev_pool, slow_n,
                min_margin=slow_min_margin,
                evaluator=evaluator,
                witness=witness,
                w_cache=w_cache,
                J_W_floor=J_W_best,
                require_gate=slow_require_gate,
                epoch_target=epoch_target,
                act_mode=act_mode,
                forced_id=forced_id,
                dirichlet_a=dirichlet_a,
                witness_workers=witness_workers)
            if ok: epoch_rounds = 0
            log["slow_update"] = sl
            # ── slow_update accept → 刷新 best state
            if sl.get("accepted") and \
                    sl.get("candidate_lib"):
                cw = sl.get(
                    "candidate_witness", {})
                if cw.get("J_W_candidate"
                          ) is not None:
                    K_n = sl["candidate_lib"]
                    w_cache = {
                        "J_W": cw["J_W_candidate"],
                        "J_W_hard":
                            cw.get(
                                "J_W_hard_candidate",
                                cw["J_W_candidate"]),
                        "J_W_soft":
                            cw.get(
                                "J_W_soft_candidate",
                                0),
                        "per_task_rewards":
                            cw.get(
                                "candidate_rewards",
                                w_cache["per_task_rewards"]),
                        "per_task_soft_rewards":
                            cw.get(
                                "candidate_soft_rewards",
                                w_cache.get(
                                    "per_task_soft_rewards")),
                    }
                    if cw["J_W_candidate"] \
                            > J_W_best:
                        K_best = K_n.clone()
                        J_W_best = cw[
                            "J_W_candidate"]
                        best_step = n
                        print(f"    🏆 "
                              f"slow_update BEST "
                              f"J_W={J_W_best:.4f}")
            # ─ Logger: slow update ─
            logger.slow_update(
                triggered=bool(sl.get(
                    "triggered", False)),
                improvement_rate=sl.get(
                    "improvement_rate", 0.0),
                regression_rate=sl.get(
                    "regression_rate", 0.0),
                skipped=bool(sl.get(
                    "skipped", False)),
                reason=str(sl.get("reason", "")),
                n_updated=len(sl.get(
                    "meta_skills", []) or []),
            )
        else:
            log["slow_update"] = {
                "triggered": False}

        log["cost"] = round(
            client.total_cost_usd-cost_start, 4)
        history["rounds"].append(log)

        # ─ Logger: round end ─
        logger.round_end(
            action=log.get("action", "stall"),
            J_W_after=log.get(
                "J_W_after", w_cache["J_W"]),
            delta_hat=log.get("delta_hat", 0.0),
            n_proposals=log.get("n_proposals", 0),
            n_admissible=log.get(
                "n_admissible", 0),
            cost_this_round=log.get("cost", 0.0),
            rollout_sr=log.get(
                "rollout_success_rate", 0.0),
            teacher_sr=log.get("teacher_sr", 0.0),
            ref_sr=log.get("ref_sr", 0.0),
        )

        if stall_count >= S_max:
            print(f"  🛑 S_max={S_max}")
            break

    # ════════════════════════════════════════
    # 收尾
    # ════════════════════════════════════════
    history["total_accepted"] = n_accepted
    history["total_stalls"] = stall_count
    history["J_W_last_accepted"] = (
        None if ablation == "no_gate" else w_cache["J_W"])
    history["J_W_best"] = (
        None if ablation == "no_gate" else J_W_best)
    history["witness_selection_evaluated"] = (
        ablation != "no_gate")
    history["best_step"] = best_step

    K_final, final_selection_policy = select_final_library(
        K_n, K_best, ablation)
    final_witness_score = (
        None if ablation == "no_gate" else J_W_best)
    history["J_W_final"] = final_witness_score
    history["final_selection_policy"] = final_selection_policy
    history["final_selected_step"] = (
        len(history["rounds"]) if ablation == "no_gate" else best_step)

    if compute_fn:
        fn_f = evaluator.compute_fn_correlation(
            K_final, witness[:15])
        history["fn_corr_final"] = fn_f.get("rho_FN")

    print("\n" + "=" * 60)
    print("  STUDY 1 DONE")
    print("=" * 60)
    if ablation == "no_gate":
        print("  J_W: not evaluated after edits "
              "(no_gate forbids witness selection)")
    else:
        print(f"  J_W: {J_W_init:.4f} → "
              f"{history['J_W_final']:.4f} "
              f"(best={J_W_best:.4f} @{best_step})")
    print(f"  Accept: {n_accepted} "
          f"(E={history['accepted_by_type']['execution_edit']}"
          f" A={history['accepted_by_type']['appendix_edit']})")

    final_dir = os.path.join(
        paths["results_study1"], f"K_final_{ts}")
    K_final.save(final_dir)
    # One physical artifact; legacy keys are aliases to the same directory.
    history["final_library_dir"] = final_dir
    history["best_library_dir"] = final_dir

    # ── Final Holdout 评估（严格：一次性 & 只 K_best）──
    # ⚠ audit issue #7：
    #   - 只评估 witness 已选定并冻结的 K_best
    #   - 不评估 K_n_last（避免"事后二选一"的隐性泄漏）
    #   - 失败 → 整个 run 标记失败（不静默留 None）
    J_final_holdout_best = None
    final_holdout_ok = True
    if final_holdout:
        print(f"\n  [Final Holdout] 无偏评估 "
              f"({len(final_holdout)} tasks, "
              f"只跑一次)...")
        try:
            fh_result_best = \
                evaluator.witness_estimate(
                      K_final, final_holdout,
                      desc=("FinalHoldout-LastStaticValid"
                          if ablation == "no_gate"
                          else "FinalHoldout-Best"),
                    require_complete=True,
                    verifier_feedback=False,
                    expose_answer_metadata=False)
            J_final_holdout_best = \
                fh_result_best["J_W"]
            print(f"    {final_selection_policy}: "
                  f"J_final_holdout = "
                  f"{J_final_holdout_best:.4f}")
        except Exception as _e:
            final_holdout_ok = False
            print(f"    ❌ Final holdout eval "
                  f"失败: {_e}")
            if require_final_holdout:
                # 正式实验失败：写入 history 后 raise
                history["final_holdout_error"] = \
                    str(_e)
                history["final_holdout_ok"] = False
                # 保存 history 便于事后调试
                save_json(history, os.path.join(
                    paths["results_study1"],
                    f"study1_history_"
                    f"{ts}_FAILED.json"))
                raise RuntimeError(
                    f"Final holdout evaluation "
                    f"failed: {_e}")

    history["final_holdout_n"] = len(final_holdout)
    history["J_final_holdout_best"] = \
        J_final_holdout_best
    history["final_holdout_ok"] = final_holdout_ok

    register_best_library(
        study="study1", path=final_dir,
        timestamp=ts,
        metrics={
            "J_W_init": J_W_init,
            "J_W_final": history["J_W_final"],
            "J_W_best": history["J_W_best"],
            "J_final_holdout_best":
                J_final_holdout_best,
            "total_accepted": n_accepted,
            "total_cost": round(
                client.total_cost_usd, 4),
            "beta_mode": beta_mode})

    client.print_cost_summary()
    history["cost_summary"] = client.cost_summary()
    save_json(history, os.path.join(
        paths["results_study1"],
        f"study1_history_{ts}.json"))

    # ─ Logger: final holdout + session end ─
    try:
        if J_final_holdout_best is not None:
            logger.witness_eval(
                tag="final_holdout_best",
                J_W=J_final_holdout_best,
                n_tasks=len(final_holdout),
            )
        # ⚠ 严格只评 K_best（audit #7）
        # 不再评 K_n_last，避免事后二选一
        logger.session_end(
            total_rounds=len(history["rounds"]),
            total_accepted=n_accepted,
            J_W_init=J_W_init,
            J_W_final=history["J_W_final"],
            J_W_best=J_W_best,
            total_cost=round(
                client.total_cost_usd, 4),
            cost_summary=history["cost_summary"],
        )
    except Exception as _e:
        print(f"  [logger] session_end fail: {_e}")

    return history, K_final


# ─────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import signal
    import sys as _sys

    # 让 Ctrl+C 立即退出（不等 executor）
    # 计数：连按 2 次直接 kill 自身进程
    _sigint_count = [0]

    def _handle_sigint(signum, frame):
        _sigint_count[0] += 1
        if _sigint_count[0] >= 2:
            print("\n💀 第 2 次 Ctrl+C，SIGKILL "
                  "自身...")
            import subprocess
            subprocess.Popen(
                ["kill", "-9", str(os.getpid())])
            os._exit(137)
        print(f"\n🛑 收到 Ctrl+C ({_sigint_count[0]}"
              f"/2)，强制退出... "
              f"(再按一次立即杀进程)")
        # 直接 os._exit —— 不等任何东西
        # 用 os._exit 而非 sys.exit：绕过 finally 块
        os._exit(130)
    signal.signal(signal.SIGINT, _handle_sigint)
    # 服务器上 tmux 里 SIGTERM 也支持
    try:
        signal.signal(signal.SIGTERM, _handle_sigint)
    except Exception:
        pass  # Windows 不支持 SIGTERM

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beta_mode",
        choices=["b1", "b2", "b3", "b4"],
        default=None,
        help="b1/b2/b3, or b4=teacher-on-student")
    parser.add_argument(
        "--ablation",
        choices=["full",
                 "execution_only","no_gate"],
        default=None)
    parser.add_argument(
        "--config",
        default="config/config.yaml")
    args = parser.parse_args()
    run_study1(
        config_path=args.config,
        beta_mode_override=args.beta_mode,
        ablation_override=args.ablation)
