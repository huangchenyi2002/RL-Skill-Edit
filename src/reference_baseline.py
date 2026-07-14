# src/reference_baseline.py
"""
π₀ Reference Baseline
建立无 skill 状态下的 Student 基线表现
对应 Density-Aware ExSkill-Opt 框架的 π₀ 参考分布

用法：
  baseline = ReferenceBaseline(config, agent)
  baseline.build(witness_tasks)
  R_0 = baseline.get(task_id)
"""

import json
import os
import numpy as np
from tqdm import tqdm
from src.agent import StudentAgent


class ReferenceBaseline:

    def __init__(
        self,
        config: dict,
        agent:  StudentAgent,
    ):
        self.config  = config
        self.agent   = agent
        self.B_W     = config["study1"].get("B_W", 5)
        self.R_0: dict[str, float] = {}
        self._built  = False

    # ──────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────

    def build(
        self,
        witness_tasks: list[dict],
        save_path:     str = "",
    ) -> dict[str, float]:
        """
        对每个 witness task，用 no-skill 模式跑 B_W 次
        记录平均得分作为 R_0(x) ≈ π_0 的近似
        """
        print(f"\n[Reference Baseline] 建立 π₀ 基线")
        print(f"  任务数 = {len(witness_tasks)}")
        print(f"  B_W   = {self.B_W}")
        print(f"  模式  = no-skill implicit")

        self.R_0 = {}

        for task in tqdm(
            witness_tasks,
            desc="  π₀ baseline rollouts",
            leave=True,
            ncols=80,
        ):
            tid     = task["task_id"]
            rewards = []

            for _ in range(self.B_W):
                # no-skill：library=None，agent 不加载任何 skill
                traj = self.agent.run_task_no_skill(task)
                rewards.append(traj.final_reward)

            self.R_0[tid] = float(np.mean(rewards))

        mean_R0 = float(np.mean(list(self.R_0.values())))
        print(f"\n  ✅ π₀ 基线建立完成")
        print(f"  R̄_0 = {mean_R0:.4f}")

        self._built = True

        if save_path:
            self.save(save_path)

        return self.R_0

    def get(self, task_id: str) -> float:
        """获取某任务的 R_0 基线值，不存在则返回 0.0"""
        return self.R_0.get(task_id, 0.0)

    def mean(self) -> float:
        """所有任务的平均 R_0"""
        if not self.R_0:
            return 0.0
        return float(np.mean(list(self.R_0.values())))

    def is_built(self) -> bool:
        return self._built

    # ──────────────────────────────────────────────
    # 持久化
    # ──────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "R_0":     self.R_0,
                "mean_R0": self.mean(),
                "n_tasks": len(self.R_0),
                "B_W":     self.B_W,
            }, f, indent=2)
        print(f"  💾 π₀ 基线已保存：{path}")

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.R_0    = data.get("R_0", {})
        self._built = bool(self.R_0)
        print(f"  📂 π₀ 基线已加载：{path}")
        print(f"     R̄_0 = {self.mean():.4f}"
              f"  ({len(self.R_0)} 任务)")
        return True

    # ──────────────────────────────────────────────
    # P1 修复：label 分布统计
    # ──────────────────────────────────────────────

    def build_with_labels(
        self,
        witness_tasks: list,
        library,
        save_path: str = None,
    ):
        """
        P1 修复：建立 baseline 时同时统计 label 分布
        用于 P_0 的精确估计（替代均匀分布 fallback）
        """
        from src.label_space import (
            extract_modules_and_behaviors,
            count_labels_for_module,
            estimate_label_distribution,
            get_Z_for_module,
        )

        # 原有的 reward baseline
        self.build(witness_tasks, save_path)

        # 新增：label 统计
        modules = extract_modules_and_behaviors(
            library)
        # 跑 no-skill rollout（和 build 时共用）
        ref_trajs = []
        for task in witness_tasks[:30]:
            traj = self.agent.run_task(
                task, library,
                activation_mode="no_skill")
            ref_trajs.append(traj)

        label_dists = {}
        for mod in modules:
            Z = get_Z_for_module(mod)
            counts = count_labels_for_module(
                ref_trajs, mod)
            probs = estimate_label_distribution(
                counts, Z, alpha=1.0)
            label_dists[mod.module_id] = probs

        self._label_dists = label_dists

        # 保存
        if save_path:
            label_path = save_path.replace(
                ".json", "_labels.json")
            with open(label_path, "w") as f:
                json.dump(label_dists, f, indent=2)
            print(f"  P_0 labels saved: {label_path}")

    def get_label_distribution(
        self, module_id: str,
    ) -> dict:
        """返回某个 module 的 P_0 label 分布"""
        if hasattr(self, '_label_dists'):
            return self._label_dists.get(
                module_id, {})
        return {}

    def load_labels(self, path: str) -> bool:
        """加载已保存的 P_0 label 分布"""
        label_path = path.replace(
            ".json", "_labels.json")
        if os.path.exists(label_path):
            try:
                with open(label_path) as f:
                    self._label_dists = json.load(f)
                print(f"  P_0 labels loaded: "
                      f"{len(self._label_dists)} modules")
                return True
            except Exception:
                pass
        return False

    @classmethod
    def load_or_build(
        cls,
        config:        dict,
        agent:         StudentAgent,
        witness_tasks: list[dict],
        save_path:     str = "",
    ) -> "ReferenceBaseline":
        """
        如果已有缓存则直接加载，否则重新建立
        """
        baseline = cls(config, agent)

        if save_path and baseline.load(save_path):
            # 尝试加载已保存的 label 分布
            baseline.load_labels(save_path)
            return baseline

        baseline.build(witness_tasks, save_path)
        return baseline
