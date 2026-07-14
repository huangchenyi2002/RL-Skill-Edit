# src/library_registry.py
"""
最优库路径注册表
自动追踪每个 Study 产生的最优库路径
避免手动在 config.yaml 里填写路径

存储位置：results/best_library.json
格式：
{
  "latest": "results/study1/K_star_20260514_221755",
  "studies": {
    "study0_seed":  "skills",
    "study1":       "results/study1/K_star_20260514_221755",
    "study2":       "results/study2/K_star_20260515_XXXXXX"
  },
  "history": [
    {
      "study":     "study1",
      "timestamp": "20260514_221755",
      "path":      "results/study1/K_star_20260514_221755",
      "J_W_final": 0.5148,
      "saved_at":  "2026-05-14 22:19:55"
    }
  ]
}
"""

import os
import json
from datetime import datetime


REGISTRY_PATH = "results/best_library.json"


def load_registry() -> dict:
    """加载注册表，不存在则返回空注册表"""
    if not os.path.exists(REGISTRY_PATH):
        return {
            "latest":  "skills",   # 默认用初始库
            "studies": {
                "study0_seed": "skills",
            },
            "history": [],
        }
    with open(REGISTRY_PATH, "r",
              encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict):
    """保存注册表"""
    os.makedirs(
        os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w",
              encoding="utf-8") as f:
        json.dump(registry, f, indent=2,
                  ensure_ascii=False)


def register_best_library(
    study:     str,
    path:      str,
    timestamp: str,
    metrics:   dict = None,
):
    """
    注册某个 Study 产生的最优库

    参数：
      study     : 实验名称（如 "study1"）
      path      : 最优库的目录路径
      timestamp : 时间戳（如 "20260514_221755"）
      metrics   : 评估指标（如 J_W_final、M_exec_final）

    使用示例：
      register_best_library(
          study     = "study1",
          path      = "results/study1/K_star_20260514",
          timestamp = "20260514_221755",
          metrics   = {"J_W_final": 0.5148},
      )
    """
    registry = load_registry()

    # 更新 latest 和 studies
    registry["latest"]          = path
    registry["studies"][study]  = path

    # 追加 history
    entry = {
        "study":     study,
        "timestamp": timestamp,
        "path":      path,
        "saved_at":  datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"),
    }
    if metrics:
        entry.update(metrics)
    registry["history"].append(entry)

    save_registry(registry)

    print(f"\n  📌 最优库已注册：")
    print(f"     Study    : {study}")
    print(f"     路径     : {path}")
    if metrics:
        for k, v in metrics.items():
            print(f"     {k:12s}: {v}")
    print(f"     注册表   : {REGISTRY_PATH}")


def get_best_library(
    study:    str = None,
    fallback: str = "skills",
) -> str:
    """
    获取最优库路径

    参数：
      study    : 指定获取哪个 Study 的最优库
                 None → 获取最新的（latest）
      fallback : 找不到时的默认路径

    使用示例：
      # 获取 Study 1 的最优库（Study 2 的起点）
      path = get_best_library("study1")

      # 获取最新的最优库
      path = get_best_library()
    """
    registry = load_registry()

    if study:
        path = registry["studies"].get(study, "")
    else:
        path = registry.get("latest", "")

    if not path or not os.path.exists(path):
        print(f"  ⚠️  找不到 {study or 'latest'} "
              f"的最优库，使用默认：{fallback}")
        return fallback

    return path


def print_registry():
    """打印当前注册表状态"""
    registry = load_registry()
    print("\n" + "─" * 50)
    print("  📚 最优库注册表")
    print("─" * 50)
    print(f"  最新库：{registry.get('latest', '无')}")
    print(f"\n  各 Study 最优库：")
    for study, path in registry["studies"].items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"    {exists} {study:15s}: {path}")
    print(f"\n  历史记录（最近5条）：")
    for entry in registry["history"][-5:]:
        j_w = entry.get("J_W_final", "N/A")
        print(f"    [{entry['study']:10s}] "
              f"{entry['timestamp']}  "
              f"J_W={j_w}  "
              f"{entry['path']}")
    print("─" * 50)
