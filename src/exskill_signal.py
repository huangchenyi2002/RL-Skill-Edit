# src/exskill_signal.py
"""
ExSkill-OPD 信号计算
重构：
  - ρ 从单一标量变成 (m, λ) 二元组
  - m(h)：module-level history 权重
  - λ(h,z)：label-level action 门控
  - 保留 A^{β,ρ} 作为向后兼容的任务级摘要
  - 新增 label-space 感知的信号
"""

import math
from dataclasses import dataclass, field
from typing import Optional

try:
    import numpy as np
except ModuleNotFoundError:
    class _FakeNp:
        @staticmethod
        def mean(x):
            return sum(x)/len(x) if x else 0.0
        @staticmethod
        def std(x):
            if not x: return 0.0
            m = sum(x)/len(x)
            return (sum((v-m)**2 for v in x)
                    / len(x)) ** 0.5
        @staticmethod
        def clip(v, lo, hi):
            return max(lo, min(hi, v))
        @staticmethod
        def max(x):
            return max(x) if x else 0.0
        @staticmethod
        def percentile(x, q):
            s = sorted(x)
            i = int(len(s)*q/100)
            return s[min(i, len(s)-1)]
    np = _FakeNp()


# ─────────────────────────────────────────────────
# 安全工具
# ─────────────────────────────────────────────────

def _safe_clamp01(
    value, default: float = 0.0,
) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return max(0.0, min(1.0, v))


# ─────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────

@dataclass
class StepDensity:
    step:          int
    coverage:      float = 0.0
    confidence:    float = 0.0
    granularity:   float = 0.0
    actionability: float = 0.0
    rho:           float = 0.0
    rho_D:         float = 0.0
    rho_E:         float = 0.0
    error_type:    str   = ""
    localization:  str   = ""


@dataclass
class TrajectorySignal:
    """单条轨迹的完整信号"""
    task_id:          str
    R_s:              float = 0.0
    R_0:              float = 0.0
    R_T:              float = 0.0
    r0_T:             float = 0.0
    # ── (m, λ) 信号 ──────────────────────────
    module_id:        str   = ""
    m_value:          float = 1.0
    lambda_mean:      float = 1.0
    w_value:          float = 0.0   # d_s × m
    # ── 向后兼容的 ρ 摘要 ─────────────────────
    rho_final_mean:         float = 0.0
    rho_coverage_mean:      float = 0.0
    rho_confidence_mean:    float = 0.0
    rho_granularity_mean:   float = 0.0
    rho_actionability_mean: float = 0.0
    rho_dispatch_mean:      float = 0.0
    rho_execution_mean:     float = 0.0
    high_density_step_ratio:           float = 0.0
    high_density_dispatch_step_ratio:  float = 0.0
    high_density_execution_step_ratio: float = 0.0
    # ── A^{β,ρ} ──────────────────────────────
    A_beta:           float = 0.0
    A_beta_rho:       float = 0.0
    beta:             float = 1.0
    beta_mode:        str   = "b1"
    priority:         str   = "low"
    # ── failure 分类 ──────────────────────────
    failure_type:     str   = ""
    support_count:    int   = 1
    # ── label-space 信号 ──────────────────────
    g_n_summary:      dict  = field(
        default_factory=dict)
    donor_plus:       str   = ""
    donor_minus:      str   = ""
    step_densities:   list  = field(
        default_factory=list)

    @property
    def rho_mean(self) -> float:
        return self.rho_final_mean


@dataclass
class ContrastiveEvidence:
    """同一任务多次 rollout 的对比信号"""
    task_id:    str
    pass_rate:  float = 0.0
    spread:     float = 0.0
    best_score: float = 0.0
    worst_score: float = 0.0
    n_attempts: int   = 0
    success_skills: list = field(
        default_factory=list)
    failure_skills: list = field(
        default_factory=list)


# ─────────────────────────────────────────────────
# SKILL_DEFECT vs EXECUTION_LAPSE 分类
# ─────────────────────────────────────────────────

def classify_failure_type(
    diagnosis: dict,
) -> str:
    if not diagnosis:
        return "execution_lapse"

    ft = diagnosis.get("failure_type", "")
    if ft in ("skill_defect", "execution_lapse"):
        return ft

    dt = diagnosis.get("diagnosis_type", "")
    if dt == "missing_skill":
        return "skill_defect"
    if dt == "dispatch_failure":
        return "skill_defect"
    if dt == "execution_failure":
        ef = diagnosis.get("execution_fix", "")
        ei = diagnosis.get("execution_issue", "")
        if ef and len(ef) > 20:
            return "skill_defect"
        if ei and "not follow" in ei.lower():
            return "execution_lapse"
        if ei and "ignore" in ei.lower():
            return "execution_lapse"
        if ei and "missing" in ei.lower():
            return "skill_defect"
        return "execution_lapse"

    return "execution_lapse"


# ─────────────────────────────────────────────────
# Contrastive Evidence
# ─────────────────────────────────────────────────

def compute_contrastive_evidence(
    task_id: str,
    rewards: list[float],
    skills: list[list[str]] = None,
) -> ContrastiveEvidence:
    if not rewards:
        return ContrastiveEvidence(task_id=task_id)

    pass_rate = sum(
        1 for r in rewards if r > 0.5
    ) / len(rewards)
    best  = max(rewards)
    worst = min(rewards)
    spread = best - worst

    success_sk = []
    failure_sk = []
    if skills:
        for r, sk in zip(rewards, skills):
            if r > 0.5:
                success_sk.extend(sk)
            else:
                failure_sk.extend(sk)

    return ContrastiveEvidence(
        task_id=task_id,
        pass_rate=pass_rate,
        spread=spread,
        best_score=best,
        worst_score=worst,
        n_attempts=len(rewards),
        success_skills=list(set(success_sk)),
        failure_skills=list(set(failure_sk)),
    )


# ─────────────────────────────────────────────────
# m 估计器（module-level history weight）
# 对应老师 §3.2 + PDF 步骤4
# ─────────────────────────────────────────────────

class ModuleWeightEstimator:
    """
    估计 m(M_k)：每个 module 的 history 权重

    score_k = coverage × stability × relevance × homogeneity
    m_k = clip(0.5 + 1.5 × score_k, 0.5, 2.0)
    """

    def __init__(self, config: dict):
        s1 = config.get("study1", {})
        self.m_min = s1.get("m_min", 0.5)
        self.m_max = s1.get("m_max", 2.0)

    def estimate(
        self,
        module_id: str,
        trajectories: list,
        diagnoses_map: dict,
        homogeneity_override: float = None,
    ) -> dict:
        """
        修复：homogeneity 可外部传入
        """
        n_total = max(len(trajectories), 1)

        n_relevant = sum(
            1 for t in trajectories
            if module_id in (
                t.activated_skills or []))
        coverage = min(
            n_relevant / n_total, 1.0)

        defect_count = 0
        lapse_count = 0
        for t in trajectories:
            if module_id not in (
                t.activated_skills or []):
                continue
            diag = diagnoses_map.get(
                t.task_id, {})
            ft = diag.get("failure_type", "")
            if ft == "skill_defect":
                defect_count += 1
            elif ft == "execution_lapse":
                lapse_count += 1
        total_diag = max(
            defect_count + lapse_count, 1)
        stability = max(
            defect_count, lapse_count
        ) / total_diag

        n_fail = sum(
            1 for t in trajectories
            if module_id in (
                t.activated_skills or [])
            and not t.success)
        relevance = n_fail / max(n_relevant, 1)

        homogeneity = (homogeneity_override
                       if homogeneity_override
                       is not None
                       else 0.5)

        score = (coverage * stability
                 * relevance
                 * max(homogeneity, 0.1))
        m = max(self.m_min, min(self.m_max,
            0.5 + 1.5 * score))

        return {
            "m": m,
            "coverage": coverage,
            "stability": stability,
            "relevance": relevance,
            "homogeneity": homogeneity,
            "score": score,
            "n_relevant": n_relevant,
        }


# ─────────────────────────────────────────────────
# λ 估计器（label-level action gate）
# 新版：reward-calibrated projection slope
# 旧版 trust-score 保留为 fallback
# ─────────────────────────────────────────────────

class RewardCalibratedLambdaEstimator:
    """
    Reward-calibrated λ：将 a_+ 投影到 r_T 方向

    λ 不是信心分数，而是校准斜率：
    "teacher 指的方向和任务有用方向对得有多齐"

    使用 label_space.estimate_lambda_full()
    """

    def __init__(self, config: dict):
        s1 = config.get("study1", {})
        self.lambda_neutral = s1.get(
            "lambda_neutral", 0.0)
        self.eta_mode = s1.get(
            "eta_mode", "occurrence_weighted")
        self.use_exact_lambda = s1.get(
            "use_exact_lambda", False)

    def estimate(
        self,
        trajectories: list,
        module_id: str,
        named_behaviors: list,
        P_0: dict, P_T: dict,
        R_hat: dict, P_s: dict,
        n_samples: int,
        reference_trajectories: list = None,
    ) -> dict[str, float]:
        from src.label_space import (
            estimate_lambda_full,
            Module,
        )
        mod = Module(
            module_id=module_id,
            skill_id=module_id,
            named_behaviors=named_behaviors,
        )
        if not reference_trajectories:
            raise ValueError(
                "reference_trajectories are required to estimate eta")
        return estimate_lambda_full(
            trajectories, mod,
            P_0, P_T, R_hat, P_s,
            n_samples, self.lambda_neutral,
            eta_mode=self.eta_mode,
            use_exact_lambda=(
                self.use_exact_lambda),
            eta_trajectories=reference_trajectories,
        )


class LambdaEstimator:
    """
    估计 λ(k, z)：每个 named behavior 的 trust gate

    λ(z) = clip(1 + 0.5 × T(z), 0.5, 1.6)
    T(z) = 综合 contrast 大小 + 样本量 + 差异一致性
    """

    def __init__(self, config: dict):
        s1 = config.get("study1", {})
        self.lambda_min = s1.get("lambda_min", 0.5)
        self.lambda_max = s1.get("lambda_max", 1.6)

    def estimate(
        self,
        R_hat: dict[str, float],
        named_behaviors: list[str],
        P_s: dict[str, float],
        P_T: dict[str, float],
        n_samples: int,
        min_count_for_active: int = 3,
    ) -> dict[str, float]:
        """
        修复：S_λ 显式筛选

        只有满足以下条件的 label 才进入 S_λ：
          1. 是 named behavior（不是 other）
          2. |R̂| > 0.1（contrast 足够大）
          3. P_s 中至少出现过 min_count 次
             （用 P_s × n_samples 近似）

        不在 S_λ 中的 label → λ = 1.0
        """
        lambda_ = {}
        OTHER = "other"

        for z in R_hat:
            # 条件1：not named → default
            if z == OTHER or \
                    z not in named_behaviors:
                lambda_[z] = 1.0
                continue

            # 条件2：contrast 太小 → default
            abs_r = abs(R_hat.get(z, 0))
            if abs_r < 0.1:
                lambda_[z] = 1.0
                continue

            # 条件3：样本太少 → default
            p_s_z = P_s.get(z, 0)
            approx_count = p_s_z * n_samples
            if approx_count < min_count_for_active:
                lambda_[z] = 1.0
                continue

            # 通过 S_λ 筛选，计算实际 λ
            contrast_trust = min(abs_r / 2.0, 1.0)
            sample_trust = min(
                n_samples / 50.0, 1.0)
            diff = abs(
                P_T.get(z, 0) - P_s.get(z, 0))
            diff_trust = min(diff / 0.3, 1.0)

            T = (contrast_trust * 0.4
                 + sample_trust * 0.3
                 + diff_trust * 0.3)

            lambda_[z] = max(
                self.lambda_min,
                min(self.lambda_max,
                    1.0 + 0.5 * T))

        return lambda_


# ─────────────────────────────────────────────────
# 向后兼容的步级 ρ 计算
# ─────────────────────────────────────────────────

class StepDensityEstimator:
    """保留原有的步级密度计算（向后兼容）"""

    def __init__(self, config: dict):
        s1 = config.get("study1", {})
        self.rho_default  = s1.get(
            "rho_default", 0.1)
        self.rho_step_err = s1.get(
            "rho_step_error", 0.5)
        self.rho_verified = s1.get(
            "rho_verified", 1.0)

    def compute_step_rho(
        self, step_info: dict,
    ) -> float:
        if step_info.get(
            "has_teacher_grade", False
        ):
            raw = step_info.get(
                "teacher_confidence", 0.5)
            c = _safe_clamp01(raw, default=0.5)
            return max(c, self.rho_step_err)
        if step_info.get("has_verified", False):
            return self.rho_verified
        if (step_info.get("has_error", False)
                or step_info.get("error", "")):
            return self.rho_step_err
        return self.rho_default

    def compute_trajectory_rho(
        self,
        step_signals: list,
        beta_mode: str = "b1",
    ) -> float:
        if not step_signals:
            return self.rho_default
        rhos = [
            self.compute_step_rho(sig)
            for sig in step_signals
            if isinstance(sig, dict)
        ]
        return (sum(rhos) / len(rhos)
                if rhos else self.rho_default)

    def compute_step_density(
        self, step_idx: int,
        total_steps: int,
        diagnosis: dict,
        score_detail: dict,
        error_steps: list,
    ) -> StepDensity:
        C = self._coverage(
            diagnosis, step_idx, error_steps)
        K = self._confidence(
            score_detail, diagnosis)
        G = self._granularity(
            score_detail, diagnosis,
            step_idx, total_steps)
        U = self._actionability(diagnosis)

        rho = float(max(0.0, min(1.0,
            0.25*C + 0.25*K
            + 0.25*G + 0.25*U)))

        dt = (diagnosis.get("diagnosis_type", "")
              if diagnosis else "")
        rho_D = rho if dt == "dispatch_failure" \
            else rho * 0.3
        rho_E = rho if dt == "execution_failure" \
            else rho * 0.3

        return StepDensity(
            step=step_idx, coverage=C,
            confidence=K, granularity=G,
            actionability=U, rho=rho,
            rho_D=rho_D, rho_E=rho_E,
            error_type=dt)

    def _coverage(self, diag, step_idx, errors):
        if not diag:
            return 0.1
        if step_idx in diag.get("E_hat_i", []):
            return 1.0
        if diag.get("implicated_skill_id"):
            return 0.7
        dt = diag.get("diagnosis_type", "")
        if dt in ("execution_failure",
                  "dispatch_failure"):
            return 0.4
        return 0.1

    def _confidence(self, sd, diag):
        if not sd and not diag:
            return 0.2
        if (sd and sd.get("method") ==
                "golden_compare"):
            if sd.get("total", 0) > 0:
                return 0.8
        w = float(
            diag.get("w_t", 0.5)
            if diag else 0.5)
        if w >= 0.8:
            return 0.8
        elif w >= 0.5:
            return 0.5
        return 0.2

    def _granularity(self, sd, diag,
                     step_idx, total):
        if not diag:
            return 0.2
        if (sd and sd.get("method") ==
                "golden_compare"
                and sd.get("total", 0) > 0
                and sd.get("error", "")):
            return 1.0
        dt = diag.get("diagnosis_type", "")
        if dt == "execution_failure":
            ei = diag.get("execution_issue", "")
            if ei and len(ei) > 10:
                return 0.8
            return 0.7
        if diag.get("implicated_skill_id"):
            return 0.5
        return 0.2

    def _actionability(self, diag):
        if not diag:
            return 0.1
        dt = diag.get("diagnosis_type", "")
        if dt == "execution_failure":
            ef = diag.get("execution_fix", "")
            if ef and len(ef) > 20:
                return 1.0
            ei = diag.get("execution_issue", "")
            if ei and len(ei) > 10:
                return 0.8
            return 0.5
        if dt == "dispatch_failure":
            pi = diag.get("pi_bar_t", "")
            if pi and len(pi) > 10:
                return 0.8
            return 0.5
        return 0.1


# Historical public name. Keep it as an alias so the compatibility API and
# the current residual pipeline execute the same implementation.
DensityEstimator = StepDensityEstimator

# ─────────────────────────────────────────────────
# Teacher-Reference Contrast
# ─────────────────────────────────────────────────

class TeacherReferenceContrast:
    @staticmethod
    def compute(R_T: float, R_0: float) -> float:
        return float(R_T - R_0)

    @staticmethod
    def is_reliable(
        r0_T: float, threshold: float = 0.0,
    ) -> bool:
        return r0_T > threshold


# ─────────────────────────────────────────────────
# 残差证据 A^{β,ρ}
# 保留任务级计算，新增 (m, λ) 感知
# ─────────────────────────────────────────────────

class ResidualEvidence:

    def __init__(self, config: dict):
        self.step_density = StepDensityEstimator(
            config)
        self.m_estimator = ModuleWeightEstimator(
            config)
        self.lambda_estimator = LambdaEstimator(
            config)
        self.rc_lambda_estimator = \
            RewardCalibratedLambdaEstimator(config)
        self.contrast = TeacherReferenceContrast()

    def _estimate_R_T(
        self, traj, R_0: float,
        beta_mode: str,
        teacher_scores: dict,
    ) -> float:
        tid = traj.task_id
        R_s = traj.final_reward

        if (beta_mode in ("b3", "b4")
                and tid in teacher_scores):
            return teacher_scores[tid]

        step_sigs = getattr(
            traj, "step_execution_signals", [])
        for sig in step_sigs:
            if not isinstance(sig, dict):
                continue
            if sig.get("has_teacher_grade", False):
                ts = sig.get("teacher_score")
                if ts is not None:
                    return _safe_clamp01(
                        ts, default=R_s)

        has_error = any(
            isinstance(s, dict)
            and s.get("has_error", False)
            for s in step_sigs)

        if R_s == 0.0 and has_error:
            return min(R_0 + 0.4, 1.0)
        elif R_s == 0.0:
            return min(R_0 + 0.2, 1.0)
        elif has_error:
            return min(R_s + 0.3, 1.0)
        else:
            return min(R_s + 0.1, 1.0)

    def _get_module_id(self, traj) -> str:
        """从轨迹获取 primary module_id"""
        skills = getattr(traj, "activated_skills", []) or []
        return skills[0] if skills else ""

    def compute_single(
        self,
        traj, R_0: float, R_T: float,
        m_value: float,
        lambda_mean: float,
        rho_compat: float,
        beta: float, beta_mode: str,
        diagnosis: dict = None,
    ) -> dict:
        R_s = traj.final_reward
        r0_T = self.contrast.compute(R_T, R_0)
        student_excess = R_s - R_0

        # A^{β,ρ} 用 m × lambda_mean 替代旧的 ρ
        effective_rho = min(
            m_value * lambda_mean, 2.0)
        A_beta = beta * r0_T - student_excess
        A_beta_rho = effective_rho * A_beta

        if A_beta_rho > 0.3:
            priority = "high"
        elif A_beta_rho > 0.05:
            priority = "medium"
        else:
            priority = "low"

        failure_type = ""
        if not traj.success and diagnosis:
            failure_type = classify_failure_type(
                diagnosis)

        return {
            "R_s": R_s, "R_0": R_0, "R_T": R_T,
            "r0_T": r0_T,
            "student_excess": student_excess,
            "A_beta": A_beta,
            "m_value": m_value,
            "lambda_mean": lambda_mean,
            "effective_rho": effective_rho,
            "rho_compat": rho_compat,
            "A_beta_rho": A_beta_rho,
            "beta": beta,
            "priority": priority,
            "failure_type": failure_type,
        }

    def compute_batch(
        self,
        trajectories: list,
        reference_baseline: object,
        teacher_scores: dict,
        beta: float,
        beta_mode: str,
        diagnoses_map: dict = None,
        enable_dispatch: bool = True,
        enable_exec: bool = True,
        epoch_target: object = None,
    ) -> list[TrajectorySignal]:
        results = []
        if diagnoses_map is None:
            diagnoses_map = {}

        # 预计算 per-module 的 m
        m_cache = {}
        if epoch_target:
            for ms in epoch_target.modules:
                m_cache[ms.module_id] = ms.m
                # lambda_mean from epoch_target
        else:
            # 退化：从 trajectories 估计
            module_ids = set()
            for t in trajectories:
                for sid in (
                    getattr(t, "activated_skills", []) or []
                ):
                    module_ids.add(sid)
            for mid in module_ids:
                # 计算 homogeneity from P_s
                homo = 0.5
                if hasattr(self, '_p_s_cache') \
                        and mid in self._p_s_cache:
                    vals = [v for z, v in
                        self._p_s_cache[mid].items()
                        if z != "other"]
                    if vals:
                        homo = max(vals)
                m_info = self.m_estimator.estimate(
                    mid, trajectories,
                    diagnoses_map,
                    homogeneity_override=homo)
                m_cache[mid] = m_info["m"]

        for traj in trajectories:
            tid = traj.task_id
            R_s = traj.final_reward
            R_0 = reference_baseline.get(tid)
            diagnosis = diagnoses_map.get(tid, {})
            sd_detail = getattr(
                traj, "score_detail", {})
            module_id = self._get_module_id(traj)

            R_T = self._estimate_R_T(
                traj, R_0, beta_mode,
                teacher_scores)

            # m from cache or default
            m_value = m_cache.get(module_id, 1.0)

            # λ mean from epoch_target or default
            lambda_mean = 1.0
            if epoch_target:
                for ms in epoch_target.modules:
                    if ms.module_id == module_id:
                        vals = [
                            v for v in
                            ms.lambda_.values()
                            if v != 1.0]
                        if vals:
                            lambda_mean = sum(vals) \
                                / len(vals)
                        break

            # w = d_s × m（近似）
            n_total = max(len(trajectories), 1)
            n_rel = sum(
                1 for t in trajectories
                if module_id in (
                    getattr(t, "activated_skills", []) or []))
            d_s = n_rel / n_total
            w = d_s * m_value

            # 向后兼容的步级 ρ
            step_sigs = getattr(
                traj, "step_execution_signals", [])
            rho_compat = \
                self.step_density\
                    .compute_trajectory_rho(
                        step_sigs, beta_mode)

            # 步级密度（向后兼容）
            error_steps = [
                s.get("step", 0)
                for s in step_sigs
                if isinstance(s, dict)
                and s.get("has_error", False)]
            step_dens = []
            for sig in step_sigs:
                if not isinstance(sig, dict):
                    continue
                if sig.get(
                    "has_teacher_grade", False
                ):
                    rho_val = \
                        self.step_density\
                            .compute_step_rho(sig)
                    sd_obj = StepDensity(
                        step=sig.get("step", 0),
                        rho=rho_val,
                        rho_D=rho_val * 0.3,
                        rho_E=rho_val * 0.3)
                else:
                    sd_obj = \
                        self.step_density\
                            .compute_step_density(
                                sig.get("step", 0),
                                len(step_sigs),
                                diagnosis,
                                sd_detail,
                                error_steps)
                step_dens.append(sd_obj)

            # 计算 A^{β,ρ}
            ev = self.compute_single(
                traj=traj, R_0=R_0, R_T=R_T,
                m_value=m_value,
                lambda_mean=lambda_mean,
                rho_compat=rho_compat,
                beta=beta,
                beta_mode=beta_mode,
                diagnosis=diagnosis)

            # label-space 摘要
            g_n_summary = {}
            donor_plus  = ""
            donor_minus = ""
            if epoch_target:
                for ms in epoch_target.modules:
                    if ms.module_id == module_id:
                        g_n_summary = ms.g_n
                        donor_plus  = ms.donor_plus
                        donor_minus = ms.donor_minus
                        break

            # 步级 ρ 统计
            if step_dens:
                cov_m = float(np.mean(
                    [s.coverage for s in step_dens]))
                kon_m = float(np.mean(
                    [s.confidence for s in step_dens]))
                gran_m = float(np.mean(
                    [s.granularity for s in step_dens]))
                act_m = float(np.mean(
                    [s.actionability
                     for s in step_dens]))
                rho_m = float(np.mean(
                    [s.rho for s in step_dens]))
                rho_D_m = float(np.mean(
                    [s.rho_D for s in step_dens]))
                rho_E_m = float(np.mean(
                    [s.rho_E for s in step_dens]))
            else:
                cov_m = kon_m = gran_m = act_m = 0.0
                rho_m = rho_compat
                rho_D_m = rho_E_m = rho_compat * 0.3

            n_steps = max(len(step_dens), 1)
            high_total = sum(
                1 for s in step_dens
                if s.rho >= 0.7)
            high_D = sum(
                1 for s in step_dens
                if s.rho_D >= 0.7)
            high_E = sum(
                1 for s in step_dens
                if s.rho_E >= 0.7)

            sig = TrajectorySignal(
                task_id=tid,
                R_s=R_s, R_0=R_0, R_T=R_T,
                r0_T=ev["r0_T"],
                module_id=module_id,
                m_value=m_value,
                lambda_mean=lambda_mean,
                w_value=w,
                rho_final_mean=rho_m,
                rho_coverage_mean=cov_m,
                rho_confidence_mean=kon_m,
                rho_granularity_mean=gran_m,
                rho_actionability_mean=act_m,
                rho_dispatch_mean=rho_D_m,
                rho_execution_mean=rho_E_m,
                high_density_step_ratio=(
                    high_total / n_steps),
                high_density_dispatch_step_ratio=(
                    high_D / n_steps),
                high_density_execution_step_ratio=(
                    high_E / n_steps),
                A_beta=ev["A_beta"],
                A_beta_rho=ev["A_beta_rho"],
                beta=beta,
                beta_mode=beta_mode,
                priority=ev["priority"],
                failure_type=ev["failure_type"],
                support_count=1,
                g_n_summary=g_n_summary,
                donor_plus=donor_plus,
                donor_minus=donor_minus,
                step_densities=step_dens,
            )
            results.append(sig)

        return results

    def summary(self, signals: list) -> dict:
        if not signals:
            return {}

        A_vals = [s.A_beta_rho for s in signals]
        n_defect = sum(
            1 for s in signals
            if s.failure_type == "skill_defect")
        n_lapse = sum(
            1 for s in signals
            if s.failure_type == "execution_lapse")

        # m 和 λ 统计
        m_vals = [s.m_value for s in signals
                  if s.m_value != 1.0]
        lam_vals = [s.lambda_mean for s in signals
                    if s.lambda_mean != 1.0]

        return {
            "rho_final_mean": float(np.mean(
                [s.rho_final_mean
                 for s in signals])),
            "rho_coverage_mean": float(np.mean(
                [s.rho_coverage_mean
                 for s in signals])),
            "rho_confidence_mean": float(np.mean(
                [s.rho_confidence_mean
                 for s in signals])),
            "rho_dispatch_mean": float(np.mean(
                [s.rho_dispatch_mean
                 for s in signals])),
            "rho_execution_mean": float(np.mean(
                [s.rho_execution_mean
                 for s in signals])),
            "A_beta_rho_mean": float(
                np.mean(A_vals)),
            "A_beta_rho_std": float(
                np.std(A_vals)),
            "A_beta_rho_max": float(
                np.max(A_vals)) if A_vals else 0,
            "n_trajectories": len(signals),
            "priority_high": sum(
                1 for s in signals
                if s.priority == "high"),
            "priority_medium": sum(
                1 for s in signals
                if s.priority == "medium"),
            "priority_low": sum(
                1 for s in signals
                if s.priority == "low"),
            "R_s_mean": float(np.mean(
                [s.R_s for s in signals])),
            "R_T_mean": float(np.mean(
                [s.R_T for s in signals])),
            "R_0_mean": float(np.mean(
                [s.R_0 for s in signals])),
            "r0_T_mean": float(np.mean(
                [s.r0_T for s in signals])),
            "n_skill_defect": n_defect,
            "n_execution_lapse": n_lapse,
            "defect_ratio": (n_defect / max(
                n_defect + n_lapse, 1)),
            # 新增：(m, λ) 统计
            "m_mean": float(np.mean(m_vals))
                if m_vals else 1.0,
            "m_std": float(np.std(m_vals))
                if m_vals else 0.0,
            "lambda_mean": float(
                np.mean(lam_vals))
                if lam_vals else 1.0,
        }


# ─────────────────────────────────────────────────
# support_count 聚合
# ─────────────────────────────────────────────────

def aggregate_support_counts(
    signals: list[TrajectorySignal],
    key: str = "failure_type",
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sig in signals:
        ft = getattr(sig, key, "") or "unknown"
        counts[ft] = counts.get(ft, 0) + 1
    return counts


def merge_signals_by_skill(
    signals: list[TrajectorySignal],
    diagnoses_map: dict,
) -> dict[str, dict]:
    skill_agg: dict[str, dict] = {}
    for sig in signals:
        diag = diagnoses_map.get(
            sig.task_id, {})
        sids = diag.get("I_hat_i", [])
        impl = diag.get("implicated_skill_id")
        if impl and impl not in sids:
            sids.append(impl)
        for sid in sids:
            if sid not in skill_agg:
                skill_agg[sid] = {
                    "support_count": 0,
                    "A_beta_rho_sum": 0.0,
                    "n_defect": 0,
                    "n_lapse": 0,
                    "tasks": [],
                }
            entry = skill_agg[sid]
            entry["support_count"] += 1
            entry["A_beta_rho_sum"] += \
                sig.A_beta_rho
            if sig.failure_type == "skill_defect":
                entry["n_defect"] += 1
            elif sig.failure_type == "execution_lapse":
                entry["n_lapse"] += 1
            entry["tasks"].append(sig.task_id)
    return skill_agg


# ─────────────────────────────────────────────────
# 便捷函数
# ─────────────────────────────────────────────────

def format_signal_for_prompt(
    sig: TrajectorySignal,
) -> str:
    priority_desc = {
        "high": "HIGH", "medium": "MEDIUM",
        "low": "LOW"}
    lines = [
        f"Signal (β={sig.beta:.1f}, "
        f"mode={sig.beta_mode}):",
        f"  R_s={sig.R_s:.3f} R_0={sig.R_0:.3f} "
        f"R_T={sig.R_T:.3f}",
        f"  r⁰_T={sig.r0_T:+.3f}",
        f"  m={sig.m_value:.2f} "
        f"λ_mean={sig.lambda_mean:.2f} "
        f"w={sig.w_value:.3f}",
        f"  A^{{β,ρ}}={sig.A_beta_rho:+.4f} "
        f"[{priority_desc.get(sig.priority, sig.priority)}]",
    ]
    if sig.failure_type:
        lines.append(
            f"  failure: {sig.failure_type}")
    if sig.donor_plus:
        lines.append(
            f"  direction: ↑{sig.donor_plus} "
            f"↓{sig.donor_minus}")
    return "\n".join(lines)


def get_high_priority_tasks(
    signals: list, top_n: int = 5,
) -> list:
    return sorted(
        signals,
        key=lambda s: s.A_beta_rho,
        reverse=True)[:top_n]


def get_defect_signals(signals: list) -> list:
    return [s for s in signals
            if s.failure_type == "skill_defect"]


def get_lapse_signals(signals: list) -> list:
    return [s for s in signals
            if s.failure_type == "execution_lapse"]
