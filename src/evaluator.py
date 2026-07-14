# src/evaluator.py
"""
评估器
实现：
  1. soft/mixed gate（检测微小 execution 改善）
  2. dual gate（label-KL + reward，对应老师 §3.4）
  3. R_floor（running-best floor）
    4. 当前阶段仅使用 paired point estimates
    5. 向后兼容原有的 hard gate
"""

import math
import numpy as np
from concurrent.futures import (
    ThreadPoolExecutor, as_completed)
from tqdm import tqdm
from src.agent import StudentAgent
from src.skill_library import SkillLibrary


# ─────────────────────────────────────────────────
# Gate 指标选择
# ─────────────────────────────────────────────────

def select_gate_score(
    hard: float,
    soft: float,
    metric: str = "hard",
    mixed_weight: float = 0.5,
) -> float:
    if metric == "hard":
        return float(hard)
    if metric == "soft":
        return float(soft)
    if metric == "mixed":
        w = max(0.0, min(1.0, float(mixed_weight)))
        return (1.0 - w) * float(hard) \
            + w * float(soft)
    raise ValueError(
        f"Unknown metric {metric!r}")


class Evaluator:

    def __init__(
        self, config: dict, agent: StudentAgent,
    ):
        self.config = config
        self.agent  = agent
        s1          = config.get("study1", {})
        self.B_W    = s1.get("B_W", 5)
        self.alpha  = s1.get("alpha", 0.10)
        self.n_boot = s1.get(
            "bootstrap_samples", 500)
        self.reward_epsilon = s1.get(
            "reward_epsilon", s1.get("tau_min", 0.001))
        self.tau_min = self.reward_epsilon  # legacy result alias
        self.parallel = s1.get(
            "parallel_witness", True)
        self.workers = s1.get(
            "witness_workers", 4)
        self.gate_metric = s1.get(
            "gate_metric", "mixed")
        self.gate_mixed_weight = s1.get(
            "gate_mixed_weight", 0.5)
        # ── 新增：dual gate 参数 ─────────────
        self.enable_label_kl_gate = s1.get(
            "enable_label_kl_gate", True)

    def update_study_config(self, study: str):
        cfg = self.config.get(study, {})
        if cfg:
            self.B_W = cfg.get(
                "B_W", self.B_W)
            self.alpha = cfg.get(
                "alpha", self.alpha)
            self.n_boot = cfg.get(
                "bootstrap_samples", self.n_boot)
            self.reward_epsilon = cfg.get(
                "reward_epsilon", cfg.get(
                    "tau_min", self.reward_epsilon))
            self.tau_min = self.reward_epsilon
            self.parallel = cfg.get(
                "parallel_witness", self.parallel)
            self.workers = cfg.get(
                "witness_workers", self.workers)
            self.gate_metric = cfg.get(
                "gate_metric", self.gate_metric)
            self.gate_mixed_weight = cfg.get(
                "gate_mixed_weight",
                self.gate_mixed_weight)
            self.enable_label_kl_gate = cfg.get(
                "enable_label_kl_gate",
                self.enable_label_kl_gate)

    # ──────────────────────────────────────────────
    # Dispatch F1
    # ──────────────────────────────────────────────

    def dispatch_f1(
        self,
        library: SkillLibrary,
        trigger_queries: list[dict],
    ) -> float:
        tp = fp = fn = 0
        for q in tqdm(
            trigger_queries,
            desc="    Dispatch F1",
            leave=False,
        ):
            traj = self.agent.run_task(
                q, library,
                activation_mode="implicit")
            all_exp = set()
            exp = q.get("expected_skill_id", "")
            if exp:
                all_exp.add(exp)
            all_exp.update(q.get("skills", []))
            pred = set(traj.activated_skills)
            if pred & all_exp:
                tp += 1
            elif pred:
                fp += 1
                fn += 1
            else:
                fn += 1

        precision = (tp / (tp + fp)
                     if (tp + fp) > 0 else 0.0)
        recall = (tp / (tp + fn)
                  if (tp + fn) > 0 else 0.0)
        f1 = (2 * precision * recall
              / (precision + recall)
              if (precision + recall) > 0
              else 0.0)
        return f1

    # ──────────────────────────────────────────────
    # Witness 评估（返回 hard + soft）
    # ──────────────────────────────────────────────

    def witness_estimate(
        self,
        library: SkillLibrary,
        witness_tasks: list[dict],
        B_W: int = None,
        desc: str = "Witness 评估",
        activation_mode: str = None,
        forced_skill_id: str = None,
        return_trajectories: bool = False,
        require_complete: bool = True,
        verifier_feedback: bool = True,
        expose_answer_metadata: bool = True,
        seed: int = None,
    ) -> dict:
        """
        ⚠ 老师 audit #2（2026-07-10）：
        return_trajectories=True 时返回 per_task_trajectories
        （list[list[Trajectory]]，长度 = n_tasks × B_W），供
        KL gate 用同一批 rollout 算 frozen-Q true KL decrease。避免
        candidate 跑两次导致 KL/reward 使用不同样本。
        """
        B_W = B_W or self.B_W
        # 单 skill → 自动强制激活
        if activation_mode is None:
            if len(library) == 1:
                activation_mode = "harness"
                forced_skill_id = (
                    forced_skill_id
                    or library.skill_ids()[0])
            else:
                activation_mode = "implicit"

        def eval_one(task, task_index):
            hard_rewards = []
            soft_rewards = []
            trajs_this = []
            for repetition in range(B_W):
                rollout_seed = (
                    None
                    if seed is None
                    else int(seed) + task_index * B_W + repetition)
                traj = self.agent.run_task(
                    task, library,
                    activation_mode=activation_mode,
                    forced_skill_id=forced_skill_id,
                    verifier_feedback=verifier_feedback,
                    expose_answer_metadata=expose_answer_metadata,
                    seed=rollout_seed)
                if not getattr(traj, "evaluation_valid", True):
                    raise RuntimeError(
                        f"invalid rollout {traj.task_id}: "
                        f"{getattr(traj, 'invalid_reason', '')}")
                hard_rewards.append(traj.hard_reward)
                sd = getattr(
                    traj, "score_detail", {})
                soft = traj.soft_reward
                soft_rewards.append(soft)
                if return_trajectories:
                    trajs_this.append(traj)
            return (float(np.mean(hard_rewards)),
                    float(np.mean(soft_rewards)),
                    hard_rewards, soft_rewards,
                    trajs_this)

        # 关键 bug 修复（2026-07-10）：
        # 之前用 `as_completed` + `.append(...)`，per_task_*
        # 里的顺序是**完成顺序**，跟 witness_tasks 的下标不对应。
        # 后续 `compute_lcb` 用列表下标做 paired difference
        # （baseline[i] - candidate[i]），错配 → paired 变 unpaired
        # → 方差爆炸，delta_hat 偏差不可控。
        # 修复：预分配等长列表，按 witness_tasks 的下标写入。
        n = len(witness_tasks)
        per_task_hard = [0.0] * n
        per_task_soft = [0.0] * n
        per_task_hard_rewards = [[0.0]] * n
        per_task_soft_rewards = [[0.0]] * n
        per_task_trajectories = (
            [[] for _ in range(n)]
            if return_trajectories else None)

        if self.parallel and self.workers > 1:
            bar = tqdm(
                total=n,
                desc=f"  [{desc}] 并行",
                leave=True, ncols=80)
            executor = ThreadPoolExecutor(
                max_workers=self.workers)
            try:
                # 提交时记录下标，as_completed 后按下标写回
                futs = {
                    executor.submit(eval_one, t, i): i
                    for i, t in enumerate(
                        witness_tasks)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    try:
                        # 每 task 最多 300s（B_W 次 rollout）
                        h, s, hr, sr, tr = fut.result(
                            timeout=300)
                    except Exception as e:
                        print(f"  ⚠ witness task[{i}] "
                              f"失败/超时: "
                              f"{str(e)[:80]}")
                        if require_complete:
                            executor.shutdown(
                                wait=False, cancel_futures=True)
                            raise RuntimeError(
                                f"invalid witness bundle at task {i}: {e}") from e
                        h = s = 0.0
                        hr = [0.0]
                        sr = [0.0]
                        tr = []
                    per_task_hard[i] = h
                    per_task_soft[i] = s
                    per_task_hard_rewards[i] = hr
                    per_task_soft_rewards[i] = sr
                    if per_task_trajectories \
                            is not None:
                        per_task_trajectories[i] = tr
                    bar.update(1)
                    bar.set_postfix({
                        "h": f"{np.mean(per_task_hard):.3f}",
                        "s": f"{np.mean(per_task_soft):.3f}",
                    })
            except KeyboardInterrupt:
                print("\n  🛑 中断 witness eval")
                executor.shutdown(
                    wait=False, cancel_futures=True)
                raise
            finally:
                executor.shutdown(wait=False)
                bar.close()
        else:
            bar = tqdm(
                enumerate(witness_tasks),
                total=n,
                desc=f"  [{desc}]",
                leave=True, ncols=80)
            for i, task in bar:
                h, s, hr, sr, tr = eval_one(task, i)
                per_task_hard[i] = h
                per_task_soft[i] = s
                per_task_hard_rewards[i] = hr
                per_task_soft_rewards[i] = sr
                if per_task_trajectories is not None:
                    per_task_trajectories[i] = tr
                bar.set_postfix({
                    "h": f"{np.mean(per_task_hard):.3f}",
                    "s": f"{np.mean(per_task_soft):.3f}",
                })

        J_W_hard = float(np.mean(per_task_hard))
        J_W_soft = float(np.mean(per_task_soft))
        J_W = select_gate_score(
            J_W_hard, J_W_soft,
            self.gate_metric,
            self.gate_mixed_weight)

        tqdm.write(
            f"  ✅ {desc}  "
            f"hard={J_W_hard:.4f} "
            f"soft={J_W_soft:.4f} "
            f"J_W({self.gate_metric})="
            f"{J_W:.4f}")

        result = {
            "J_W": J_W,
            "J_W_hard": J_W_hard,
            "J_W_soft": J_W_soft,
            "per_task_means": per_task_hard,
            "per_task_rewards": per_task_hard_rewards,
            "per_task_soft_means": per_task_soft,
            "per_task_soft_rewards": per_task_soft_rewards,
            "gate_metric": self.gate_metric,
        }
        if per_task_trajectories is not None:
            result["per_task_trajectories"] = \
                per_task_trajectories
        return result

    # ──────────────────────────────────────────────
    # Forced-Execution 评估
    # ──────────────────────────────────────────────

    def forced_exec_estimate(
        self,
        library: SkillLibrary,
        eval_tasks: list[dict],
        B_W: int = None,
        desc: str = "强制执行",
    ) -> dict:
        B_W = B_W or self.B_W
        per_task_means = []
        per_task_rewards = []
        bar = tqdm(
            eval_tasks,
            desc=f"  [{desc}]",
            leave=True, ncols=80)
        for task in bar:
            target = (
                task.get("expected_skill_id", "")
                or (task.get("skills", [None])[0]))
            if not target or not library.get(target):
                continue
            rewards = []
            for _ in range(B_W):
                traj = self.agent.run_task(
                    task, library,
                    activation_mode="harness",
                    forced_skill_id=target)
                rewards.append(traj.final_reward)
            mean_r = float(np.mean(rewards))
            per_task_means.append(mean_r)
            per_task_rewards.append(rewards)
            bar.set_postfix({
                "m": f"{mean_r:.3f}",
                "c": f"{np.mean(per_task_means):.3f}",
            })
        M_exec = (float(np.mean(per_task_means))
                  if per_task_means else 0.0)
        tqdm.write(f"  ✅ {desc}  "
                   f"M̂_exec={M_exec:.4f}")
        return {
            "M_exec": M_exec,
            "per_task_means": per_task_means,
            "per_task_rewards": per_task_rewards,
        }

    # ──────────────────────────────────────────────
    # FN 相关性
    # ──────────────────────────────────────────────

    def compute_fn_correlation(
        self,
        library: SkillLibrary,
        eval_tasks: list[dict],
        B_W: int = None,
    ) -> dict:
        from scipy import stats
        B_W = B_W or self.B_W
        forced_rewards = []
        natural_rewards = []
        for task in tqdm(
            eval_tasks[:20],
            desc="    FN 相关性",
            leave=False,
        ):
            target = (
                task.get("expected_skill_id", "")
                or (task.get("skills", [None])[0]))
            if not target or not library.get(target):
                continue
            self.agent.clear_cache()
            ft = self.agent.run_task(
                task, library,
                activation_mode="harness",
                forced_skill_id=target)
            self.agent.clear_cache()
            nt = self.agent.run_task(
                task, library,
                activation_mode="implicit")
            forced_rewards.append(ft.final_reward)
            natural_rewards.append(nt.final_reward)

        if len(forced_rewards) < 3:
            return {"rho_FN": None,
                    "p_value": None,
                    "n_tasks": len(forced_rewards),
                    "note": "insufficient_data"}
        if (np.std(forced_rewards) < 1e-9
                or np.std(natural_rewards) < 1e-9):
            return {"rho_FN": None,
                    "p_value": None,
                    "n_tasks": len(forced_rewards),
                    "note": "zero_variance",
                    "forced_mean": float(
                        np.mean(forced_rewards)),
                    "natural_mean": float(
                        np.mean(natural_rewards))}
        corr, p_val = stats.pearsonr(
            forced_rewards, natural_rewards)
        tqdm.write(
            f"  ρ̂_FN={corr:.4f} "
            f"(p={p_val:.4f})")
        return {
            "rho_FN": float(corr),
            "p_value": float(p_val),
            "n_tasks": len(forced_rewards),
            "forced_mean": float(
                np.mean(forced_rewards)),
            "natural_mean": float(
                np.mean(natural_rewards)),
        }

    # ──────────────────────────────────────────────
    # Paired reward point estimate
    # ──────────────────────────────────────────────

    def compute_lcb(
        self,
        baseline_rewards: list[list[float]],
        candidate_rewards: list[list[float]],
        baseline_soft: list[list[float]] = None,
        candidate_soft: list[list[float]] = None,
        alpha_override: float = None,
    ) -> dict:
        """
        Paired task-level Δ_i (= mean(cand rewards) - mean(inc rewards)).
        当前阶段不运行 bootstrap/LCB；数据缺失或无法对齐时拒绝。
        """
        assert len(baseline_rewards) == \
            len(candidate_rewards)
        n_tasks = len(baseline_rewards)

        # 有效 paired tasks 检查
        if n_tasks == 0:
            return {
                "delta_hat": 0.0,
                "lcb": -1.0,
                "is_admissible": False,
                "reward_epsilon": self.reward_epsilon,
                "alpha": self.alpha,
                "mode": "missing_paired_tasks",
                "gate_metric": self.gate_metric,
                "n_paired": n_tasks,
            }

        use_soft = (
            self.gate_metric in ("soft", "mixed")
            and baseline_soft is not None
            and candidate_soft is not None)

        if use_soft:
            diffs = []
            for i in range(n_tasks):
                bh = float(np.mean(
                    baseline_rewards[i]))
                bs = float(np.mean(
                    baseline_soft[i]))
                ch = float(np.mean(
                    candidate_rewards[i]))
                cs = float(np.mean(
                    candidate_soft[i]))
                b_score = select_gate_score(
                    bh, bs, self.gate_metric,
                    self.gate_mixed_weight)
                c_score = select_gate_score(
                    ch, cs, self.gate_metric,
                    self.gate_mixed_weight)
                diffs.append(c_score - b_score)
        else:
            diffs = [
                float(np.mean(
                    candidate_rewards[i]))
                - float(np.mean(
                    baseline_rewards[i]))
                for i in range(n_tasks)]

        delta_hat = float(np.mean(diffs))

        return {
            "delta_hat": delta_hat,
            "lcb": None,
            "is_admissible": (
                delta_hat + 1e-12 >= -self.reward_epsilon),
            "reward_epsilon": self.reward_epsilon,
            "alpha": None,
            "mode": "paired_point_estimate",
            "gate_metric": self.gate_metric,
        }

    # ──────────────────────────────────────────────
    # Dual Gate（label-KL + reward）
    # 对应老师 §3.4
    # ──────────────────────────────────────────────

    def dual_gate(
        self,
        reward_result: dict,
        label_kl_decrease: float = None,
        J_W_floor: float = None,
    ) -> dict:
        """
        两道门同时通过才接受（老师 §3.4，2026-07-10 audit）：

        门1（frozen-Q true label-KL gate）：
          若启用（enable_label_kl_gate=True），必须提供
          label_kl_decrease point estimate > 0。
          **label_kl_decrease 缺失 → 直接拒绝**（不是默认通过）

        门2（reward gate）：
          Δreward >= -epsilon_R
          AND J_W_candidate >= J_W_floor

        参数
        ----
        reward_result : `compute_lcb` 的输出
        label_kl_decrease : ΔKL 点估计（老 KL - 新 KL）
        J_W_floor : 历史最好 J_W，防止累积回撤
        """
        # 门2：reward gate（已有）
        reward_passed = reward_result.get(
            "is_admissible", False)
        delta_hat = reward_result.get(
            "delta_hat", 0)
        J_W_cand = reward_result.get(
            "J_W_candidate", 0)

        # R_floor 检查
        floor_passed = True
        if J_W_floor is not None:
            floor_threshold = (
                J_W_floor
                )
            floor_passed = (
                J_W_cand >= floor_threshold)

        # 门1：frozen-Q true label-KL gate
        # ⚠ audit #5：缺失 → 拒绝（不再默认通过）
        if self.enable_label_kl_gate:
            if label_kl_decrease is None:
                # 无 KL 证据 → 拒绝
                label_kl_passed = False
                label_kl_reason = "missing"
            else:
                label_kl_passed = label_kl_decrease > 0
                label_kl_reason = (
                    "point_pos" if label_kl_passed else "point_nonpos")
        else:
            # 门 1 关闭（仅 debug 用）
            label_kl_passed = True
            label_kl_reason = "disabled"

        # 综合判定
        is_admissible = (
            reward_passed
            and floor_passed
            and label_kl_passed)

        gate_detail = {
            "is_admissible": is_admissible,
            "reward_passed": reward_passed,
            "floor_passed": floor_passed,
            "label_kl_passed": label_kl_passed,
            "label_kl_reason": label_kl_reason,
            "delta_hat": delta_hat,
            "J_W_candidate": J_W_cand,
        }
        if label_kl_decrease is not None:
            gate_detail["label_kl_decrease"] = \
                label_kl_decrease
        if J_W_floor is not None:
            gate_detail["J_W_floor"] = J_W_floor
            gate_detail["floor_threshold"] = \
                J_W_floor

        return gate_detail

    # ──────────────────────────────────────────────
    # evaluate_candidate（整合 dual gate）
    # ──────────────────────────────────────────────

    def evaluate_candidate(
        self,
        baseline_lib: SkillLibrary,
        candidate_lib: SkillLibrary,
        witness_tasks: list[dict],
        baseline_cache: dict = None,
        label_kl_decrease: float = None,
        J_W_floor: float = None,
        candidate_cache: dict = None,
    ) -> dict:
        """
        ⚠ 老师 audit #2（2026-07-10）：candidate 只能跑一次。

        candidate_cache: 若非 None，从这里读 candidate 的
        witness 结果（per_task_rewards / per_task_soft_rewards
        / J_W），KL gate 和 reward gate 使用同一批
        rollout。避免 candidate 跑两次导致 KL/reward
        用不同随机结果 lucky pass。
        """
        # Baseline
        if baseline_cache is not None:
            baseline_rewards = baseline_cache[
                "per_task_rewards"]
            baseline_soft = baseline_cache.get(
                "per_task_soft_rewards")
            J_W_baseline = baseline_cache["J_W"]
            tqdm.write("    复用 Baseline 缓存")
        else:
            base = self.witness_estimate(
                baseline_lib, witness_tasks,
                desc="Baseline")
            baseline_rewards = base[
                "per_task_rewards"]
            baseline_soft = base.get(
                "per_task_soft_rewards")
            J_W_baseline = base["J_W"]

        # Candidate（关键：优先复用 cache，避免跑两次）
        if candidate_cache is not None:
            candidate_rewards = candidate_cache[
                "per_task_rewards"]
            candidate_soft = candidate_cache.get(
                "per_task_soft_rewards")
            J_W_candidate = candidate_cache["J_W"]
            cand = candidate_cache
            tqdm.write("    复用 Candidate cache "
                       "(KL/reward 同一批)")
        else:
            cand = self.witness_estimate(
                candidate_lib, witness_tasks,
                desc="Candidate")
            candidate_rewards = cand["per_task_rewards"]
            candidate_soft = cand.get(
                "per_task_soft_rewards")
            J_W_candidate = cand["J_W"]

        # Reward gate（paired point estimate）
        lcb_result = self.compute_lcb(
            baseline_rewards,
            candidate_rewards,
            baseline_soft,
            candidate_soft)

        # Dual gate
        reward_result_for_gate = {
            **lcb_result,
            "J_W_candidate": J_W_candidate,
        }
        gate = self.dual_gate(
            reward_result_for_gate,
            label_kl_decrease=label_kl_decrease,
            J_W_floor=J_W_floor)

        # 日志
        mode = lcb_result.get("mode", "")
        gm   = lcb_result.get("gate_metric", "")
        ok   = gate["is_admissible"]
        parts = [f"[{mode}/{gm}]",
                 "✅" if ok else "❌",
                 f"δ̂={lcb_result['delta_hat']:+.4f}",
                 f"ε_R={self.reward_epsilon}"]
        if label_kl_decrease is not None:
            parts.append(
                f"ΔKL={'✅' if gate['label_kl_passed'] else '❌'}"
                f"({label_kl_decrease:+.4f})")
        if J_W_floor is not None:
            parts.append(
                f"floor={'✅' if gate['floor_passed'] else '❌'}"
                f"({J_W_floor:.4f})")
        tqdm.write("    " + " ".join(parts))

        return {
            "J_W_baseline": J_W_baseline,
            "J_W_candidate": J_W_candidate,
            "J_W_hard_baseline": baseline_cache.get(
                "J_W_hard", J_W_baseline)
                if baseline_cache else J_W_baseline,
            "J_W_hard_candidate": cand.get(
                "J_W_hard", J_W_candidate),
            "J_W_soft_baseline": baseline_cache.get(
                "J_W_soft", 0.0)
                if baseline_cache else 0.0,
            "J_W_soft_candidate": cand.get(
                "J_W_soft", 0.0),
            "candidate_rewards": candidate_rewards,
            "candidate_soft_rewards": candidate_soft,
            **lcb_result,
            **gate,
        }
