# -*- coding: utf-8 -*-
"""
fla_fixhead 三组实验统一分析脚本：MQAR / Composition / Forgetting

用法:
  # 首次使用（从 TASKS 里硬编码的 sweep_id 引导注册表 + 拉 wandb + 出图）
  conda run -n py310_ljl python zoology/experiments/mqar_kda/analyze_results.py

  # 跑完一个新架构实验后，增量注册该 sweep 并自动合并 + 重画所有图
  conda run -n py310_ljl python zoology/experiments/mqar_kda/analyze_results.py \\
      --register MQAR <new_sweep_id>

  # 同时注册到多个任务（例如同一架构同时跑了 MQAR 和 Composition）
  conda run -n py310_ljl python zoology/experiments/mqar_kda/analyze_results.py \\
      --register MQAR <sweep_A> --register Composition <sweep_B>

  # 仅用本地 CSV 重新出图（不访问 wandb）
  conda run -n py310_ljl python zoology/experiments/mqar_kda/analyze_results.py --offline

  # 强制刷新某个 sweep 的缓存（删除本地 cache 再重拉）
  conda run -n py310_ljl python zoology/experiments/mqar_kda/analyze_results.py \\
      --refetch <sweep_id>

数据与注册表:
  results/sweep_registry.json   所有已注册的 sweep_id（每个 task 一个列表）
  results/data/cache/*.csv      每个 sweep 的原始拉取缓存（只拉一次）
  results/data/*_runs.csv       每个 task 的合并 CSV（所有已注册 sweep 的并集，自动去重）
  results/data/eval_matrix.csv  用于出图和 summary.json 的长表
  results/figures/*.png         图表
  results/summary.json          结构化摘要（architecture → d_model → task → metric）
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"

from zoology.analysis.utils import fetch_wandb_runs

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = RESULTS_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
FIG_DIR = RESULTS_DIR / "figures"
REGISTRY_PATH = RESULTS_DIR / "sweep_registry.json"

# wandb project 配置
# 仅作为「首次引导」使用：注册表不存在时会用这里的 sweep_ids 初始化 registry，
# 之后所有增删都走 --register / --refetch CLI，不需要手动改这个字典。
TASKS = {
    "MQAR": {
        "project": "zoology_kda_mqar",
        "csv": "mqar_runs.csv",
        "sweep_ids": [
            "mqar_configs_random_false0c2620",
        ],
    },
    "Composition": {
        "project": "zoology_kda_mqar",
        "csv": "composition_runs.csv",
        "sweep_ids": [
            "kda-composition-random-false-2d5758",
        ],
    },
    "Forgetting": {
        "project": "zoology_kda_mqar",
        "csv": "forgetting_runs.csv",
        "sweep_ids": [
            "kda-forgetting-18cf0e",
        ],
    },
}

# 自动配色色板（按模型出现顺序分配）
_PALETTE_POOL = [
    "#4E79A7", "#E15759", "#9C755F", "#F28E2B", "#59A14F",
    "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7", "#BAB0AC",
]

# ---------------------------------------------------------------------------
# Sweep registry: 单一真值源，记录每个 task 下所有已注册的 sweep_id
# ---------------------------------------------------------------------------

def load_registry() -> dict[str, list[str]]:
    """加载 sweep_registry.json；不存在时用 TASKS 默认值引导并落盘。"""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            reg = json.load(f)
        # 保证所有 TASKS 里的 key 都存在（新增 task 不丢）
        for task in TASKS:
            reg.setdefault(task, [])
        return reg
    # 首次引导
    reg = {task: list(info.get("sweep_ids", [])) for task, info in TASKS.items()}
    save_registry(reg)
    print(f"[registry] 首次引导 → {REGISTRY_PATH}")
    return reg


def save_registry(reg: dict[str, list[str]]):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)


def register_sweeps(reg: dict[str, list[str]], task: str, sweep_ids: list[str]):
    """将若干 sweep_id 追加到指定 task，自动去重且保留插入顺序。"""
    if task not in TASKS:
        raise ValueError(
            f"未知 task `{task}`，允许值：{list(TASKS.keys())}"
        )
    existing = set(reg.setdefault(task, []))
    added = []
    for sid in sweep_ids:
        if sid not in existing:
            reg[task].append(sid)
            existing.add(sid)
            added.append(sid)
    if added:
        print(f"[registry] {task}: 新增 {added}")
    else:
        print(f"[registry] {task}: 已存在，无新增")
    save_registry(reg)


def invalidate_cache(sweep_id: str):
    """删除某个 sweep 的缓存 CSV，强制下次重新拉 wandb。"""
    cache = CACHE_DIR / f"{sweep_id}.csv"
    if cache.exists():
        cache.unlink()
        print(f"[cache]    已删除 {cache}")
    else:
        print(f"[cache]    {cache} 不存在，跳过")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _build_palette(models: list[str]) -> dict[str, str]:
    """根据实际模型列表自动分配颜色。"""
    return {m: _PALETTE_POOL[i % len(_PALETTE_POOL)] for i, m in enumerate(models)}


def extract_kv_columns(df: pd.DataFrame) -> dict[int, str]:
    """返回 {num_kv_pairs: column_name} 映射。"""
    pattern = re.compile(r"valid/num_kv_pairs/accuracy-(\d+)")
    return {
        int(m.group(1)): col
        for col in df.columns
        if (m := pattern.search(col))
    }


def best_per_model(df: pd.DataFrame, metric: str = "valid/accuracy") -> pd.DataFrame:
    """每个 (model, d_model) 取 lr sweep 中 metric 最大的 run。"""
    idx = df.groupby(["model.name", "model.d_model"])[metric].idxmax(skipna=True)
    return df.loc[idx.dropna()].copy()


def melt_kv_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """将 valid/num_kv_pairs/accuracy-{N} 列 melt 成长表。"""
    kv_cols = extract_kv_columns(df)
    if not kv_cols:
        return pd.DataFrame()

    id_vars = ["model.name", "model.d_model", "Model"]
    records = []
    for _, row in df.iterrows():
        for kv, col in sorted(kv_cols.items()):
            if col in df.columns and pd.notna(row.get(col)):
                rec = {v: row[v] for v in id_vars if v in row.index}
                rec["num_kv_pairs"] = kv
                rec["accuracy"] = row[col]
                records.append(rec)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _fetch_single_sweep(project: str, sweep_id: str, offline: bool) -> pd.DataFrame:
    """拉取单个 sweep：优先走 per-sweep 缓存，缓存缺失且非 offline 时才访问 wandb。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{sweep_id}.csv"
    if cache.exists():
        print(f"[cache]    命中 {sweep_id} ← {cache.name}")
        return pd.read_csv(cache)
    if offline:
        print(f"[offline] 无缓存 {sweep_id}，跳过")
        return pd.DataFrame()
    print(f"[wandb]    拉取 {project} / {sweep_id} ...")
    df = fetch_wandb_runs(project_name=project, sweep_id=sweep_id)
    if "state" in df.columns:
        df = df[df["state"] == "finished"]
    df.to_csv(cache, index=False)
    print(f"[cache]    已落盘 {cache.name}  ({len(df)} runs)")
    return df


def fetch_all(registry: dict[str, list[str]], offline: bool = False,
              tasks: dict = None) -> dict[str, pd.DataFrame]:
    """
    按 registry 拉取/加载每个 sweep 的数据，拼成 per-task DataFrame。

    流程：
      对每个 task → 依次加载其所有 sweep_id 的缓存（缺失且非 offline 时从 wandb 拉）
                  → concat + 以 run_id 去重
                  → 写回 results/data/<task>_runs.csv（所有已注册 sweep 的合并视图）
    """
    if tasks is None:
        tasks = TASKS
    result = {}
    for task_name, info in tasks.items():
        sweep_ids = registry.get(task_name, [])
        if not sweep_ids:
            print(f"  ⚠ {task_name}: 注册表为空，跳过")
            continue

        per_sweep = []
        for sid in sweep_ids:
            sub = _fetch_single_sweep(info["project"], sid, offline)
            if not sub.empty:
                sub = sub.copy()
                sub["_source_sweep"] = sid
                per_sweep.append(sub)

        if not per_sweep:
            print(f"  ⚠ {task_name}: 所有 sweep 均无数据，跳过")
            continue

        merged = pd.concat(per_sweep, ignore_index=True)
        # 按 run_id 去重；若同一 run_id 在多个 sweep 里出现，保留第一个（更早注册的）
        if "run_id" in merged.columns:
            before = len(merged)
            merged = merged.drop_duplicates(subset=["run_id"], keep="first")
            dropped = before - len(merged)
            if dropped:
                print(f"  - {task_name}: 去重丢弃 {dropped} 条")

        # 写合并后的 per-task CSV（方便下游/离线用）
        out_csv = DATA_DIR / info["csv"]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_csv, index=False)
        print(f"  ✓ {task_name}: 合并 {len(merged)} runs → {out_csv.name}")

        result[task_name] = merged
    return result


# ---------------------------------------------------------------------------
# 图 A: 难度曲线  (accuracy vs num_kv_pairs, facet=task, hue=model)
# ---------------------------------------------------------------------------

def plot_difficulty_curves(all_data: dict[str, pd.DataFrame]):
    panels = []
    for task, df in all_data.items():
        best = best_per_model(df)
        best["Model"] = best["model.name"]
        melted = melt_kv_accuracy(best)
        if melted.empty:
            continue
        melted["Task"] = task
        panels.append(melted)

    if not panels:
        print("[图A] 无分片数据，跳过")
        return

    plot_df = pd.concat(panels, ignore_index=True)
    # 每个 (Model, Task, num_kv_pairs) 取所有 d_model 中的最高 accuracy
    plot_df = plot_df.groupby(["Model", "Task", "num_kv_pairs"], as_index=False)["accuracy"].max()
    models = sorted(plot_df["Model"].unique())
    palette = _build_palette(models)

    g = sns.relplot(
        data=plot_df,
        x="num_kv_pairs", y="accuracy",
        hue="Model", col="Task",
        kind="line", marker="o",
        hue_order=models,
        palette=palette,
        height=4, aspect=1.1,
        facet_kws={"sharey": True},
    )
    g.set(ylim=(0, 1.05), ylabel="Accuracy", xlabel="num_kv_pairs")
    for ax in g.axes.flat:
        ax.axhline(0.99, ls="--", color="grey", lw=0.8, alpha=0.6)

    out = FIG_DIR / "difficulty_curves.png"
    g.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[图A] → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 图 B: 热力图  (model × num_kv_pairs, facet=d_model×task)
# ---------------------------------------------------------------------------

def plot_heatmaps(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """绘制热力图，并返回完整评估矩阵 (Model, d_model, Task, num_kv_pairs, accuracy)。"""
    tasks_with_data = []
    for task, df in all_data.items():
        best = best_per_model(df)
        best["Model"] = best["model.name"]
        melted = melt_kv_accuracy(best)
        if melted.empty:
            continue
        melted["Task"] = task
        tasks_with_data.append(melted)

    if not tasks_with_data:
        print("[图B] 无分片数据，跳过")
        return None

    plot_df = pd.concat(tasks_with_data, ignore_index=True)

    # 保存完整评估矩阵
    eval_csv = DATA_DIR / "eval_matrix.csv"
    plot_df[["Model", "model.d_model", "Task", "num_kv_pairs", "accuracy"]].to_csv(
        eval_csv, index=False
    )
    print(f"[数据] 完整评估矩阵 → {eval_csv}")
    d_models = sorted(plot_df["model.d_model"].unique())
    tasks = [t for t in TASKS if t in plot_df["Task"].unique()]
    model_order = sorted(plot_df["Model"].unique())

    fig, axes = plt.subplots(
        len(d_models), len(tasks),
        figsize=(4.5 * len(tasks), 3 * len(d_models)),
        squeeze=False,
    )

    for i, dm in enumerate(d_models):
        for j, task in enumerate(tasks):
            ax = axes[i][j]
            sub = plot_df[(plot_df["model.d_model"] == dm) & (plot_df["Task"] == task)]
            if sub.empty:
                ax.set_visible(False)
                continue
            pivot = sub.pivot_table(
                index="Model", columns="num_kv_pairs",
                values="accuracy", aggfunc="max",
            )
            order = [m for m in model_order if m in pivot.index]
            pivot = pivot.reindex(order)

            sns.heatmap(
                pivot, annot=True, fmt=".2f",
                vmin=0, vmax=1, cmap="RdYlGn",
                cbar=(j == len(tasks) - 1),
                ax=ax,
            )
            ax.set_title(f"{task}  d={int(dm)}", fontsize=11)
            if j > 0:
                ax.set_ylabel("")
            if i < len(d_models) - 1:
                ax.set_xlabel("")

    fig.tight_layout()
    out = FIG_DIR / "heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[图B] → {out}")
    plt.close()
    return plot_df


# ---------------------------------------------------------------------------
# 图 C: 雷达图  (每个模型在 3 种任务最高难度下的 accuracy)
# ---------------------------------------------------------------------------

def plot_radar(all_data: dict[str, pd.DataFrame]):
    # 取每个任务中最大 kv_pairs 的分片精度
    task_scores = {}
    for task, df in all_data.items():
        best = best_per_model(df)
        best["Model"] = best["model.name"]
        kv_cols = extract_kv_columns(best)
        if not kv_cols:
            continue
        max_kv = max(kv_cols.keys())
        col = kv_cols[max_kv]
        scores = best.groupby("Model")[col].max()
        task_scores[task] = scores

    if len(task_scores) < 2:
        print("[图C] 不足 2 个任务有分片数据，跳过")
        return

    score_df = pd.DataFrame(task_scores).fillna(0)
    models = sorted(score_df.index.tolist())
    score_df = score_df.loc[models]
    palette = _build_palette(models)
    categories = list(score_df.columns)
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for model in models:
        values = score_df.loc[model].tolist() + [score_df.loc[model].iloc[0]]
        ax.plot(angles, values, "o-", label=model, color=palette[model], linewidth=1.5)
        ax.fill(angles, values, alpha=0.08, color=palette[model])

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="grey")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("Accuracy @ Hardest Difficulty", fontsize=13, pad=20)

    out = FIG_DIR / "radar.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[图C] → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# 图 D: Fact Capacity  (最大可解 kv_pairs vs d_model, facet=task)
# ---------------------------------------------------------------------------

def plot_fact_capacity(all_data: dict[str, pd.DataFrame], threshold: float = 0.99):
    panels = []
    for task, df in all_data.items():
        kv_cols = extract_kv_columns(df)
        if not kv_cols:
            continue

        # 对每个 run 计算能达到 threshold 的最大 kv
        records = []
        for _, row in df.iterrows():
            max_kv = 0
            for kv in sorted(kv_cols.keys()):
                col = kv_cols[kv]
                if col in df.columns and pd.notna(row.get(col)) and row[col] >= threshold:
                    max_kv = kv
            records.append({
                "model.name": row.get("model.name"),
                "model.d_model": row.get("model.d_model"),
                "max_kv": max_kv,
            })
        cap = pd.DataFrame(records)
        cap["Model"] = cap["model.name"]
        # 每个 (Model, d_model) 取最高 capacity
        cap = cap.groupby(["Model", "model.d_model"])["max_kv"].max().reset_index()
        cap["Task"] = task
        panels.append(cap)

    if not panels:
        print("[图D] 无分片数据，跳过")
        return

    plot_df = pd.concat(panels, ignore_index=True)
    models = sorted(plot_df["Model"].unique())
    palette = _build_palette(models)

    g = sns.catplot(
        data=plot_df,
        x="model.d_model", y="max_kv",
        hue="Model", col="Task",
        kind="bar",
        hue_order=models,
        palette=palette,
        height=4, aspect=1.1,
    )
    g.set(ylabel=f"Max KV Pairs (acc ≥ {threshold:.0%})", xlabel="d_model")

    out = FIG_DIR / "fact_capacity.png"
    g.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[图D] → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Summary JSON 导出
# ---------------------------------------------------------------------------

def build_summary(eval_df: pd.DataFrame, threshold: float = 0.99) -> dict:
    """
    从完整评估矩阵 (eval_matrix.csv 同源) 构建结构化摘要:
        architecture → d_model → task → {max_kv_solved_99, accuracy_by_kv}
    """
    summary = {
        "meta": {
            "tasks": sorted(eval_df["Task"].unique().tolist()),
            "threshold": threshold,
        },
        "models": {},
    }

    for (arch, d_model, task), grp in eval_df.groupby(["Model", "model.d_model", "Task"]):
        d_key = str(int(d_model))

        if arch not in summary["models"]:
            summary["models"][arch] = {}
        if d_key not in summary["models"][arch]:
            summary["models"][arch][d_key] = {}

        accuracy_by_kv = {}
        max_kv_solved = 0
        for _, row in grp.sort_values("num_kv_pairs").iterrows():
            kv = int(row["num_kv_pairs"])
            acc = round(float(row["accuracy"]), 4)
            accuracy_by_kv[str(kv)] = acc
            if acc >= threshold:
                max_kv_solved = kv

        summary["models"][arch][d_key][task] = {
            "max_kv_solved_99": max_kv_solved,
            "accuracy_by_kv": accuracy_by_kv,
        }

    return summary


def save_summary(eval_df: pd.DataFrame, threshold: float = 0.99):
    """从评估矩阵生成并保存 summary.json。"""
    summary = build_summary(eval_df, threshold)
    out = RESULTS_DIR / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Summary] → {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zoology 三组 MQAR 实验结果分析（支持增量注册 sweep）"
    )
    parser.add_argument("--offline", action="store_true",
                        help="仅使用本地缓存，不访问 wandb")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                        help=f"指定要分析的任务子集（默认全部）。可选：{list(TASKS.keys())}")
    parser.add_argument(
        "--register", nargs="+", action="append", metavar=("TASK", "SWEEP_ID"),
        default=[],
        help=("注册新 sweep 到指定 task，格式 `--register TASK SWEEP_ID [SWEEP_ID ...]`。"
              "可多次传入以同时注册多个 task。"),
    )
    parser.add_argument("--refetch", nargs="+", default=[],
                        help="强制重拉指定 sweep_id 的数据（删除本地缓存）")
    parser.add_argument("--list", action="store_true",
                        help="仅打印当前注册表，然后退出")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 加载 / 引导注册表
    registry = load_registry()

    # 2) 注册新 sweep（可多次）
    for group in args.register:
        if len(group) < 2:
            parser.error("--register 至少需要 `TASK SWEEP_ID`")
        task, *sids = group
        register_sweeps(registry, task, sids)

    # 3) 失效指定缓存
    for sid in args.refetch:
        invalidate_cache(sid)

    # 4) --list: 打印并退出
    if args.list:
        print("\n[当前注册表]")
        for task, sids in registry.items():
            print(f"  {task} ({len(sids)} sweeps):")
            for s in sids:
                cached = (CACHE_DIR / f"{s}.csv").exists()
                tag = "cached" if cached else "  not cached"
                print(f"    - [{tag}] {s}")
        return

    # 5) 选定 task 子集
    tasks = {k: v for k, v in TASKS.items() if k in args.tasks}

    all_data = fetch_all(registry, offline=args.offline, tasks=tasks)
    if not all_data:
        print("没有可用数据，退出。")
        return

    # 打印摘要
    print("\n" + "=" * 50)
    for task, df in all_data.items():
        models = df["model.name"].unique() if "model.name" in df.columns else []
        print(f"  {task}: {len(df)} runs, models={sorted(set(models))}")
    print("=" * 50 + "\n")

    plot_difficulty_curves(all_data)
    eval_df = plot_heatmaps(all_data)
    plot_radar(all_data)
    plot_fact_capacity(all_data)
    if eval_df is not None:
        save_summary(eval_df)

    print(f"\n完成！图表 → {FIG_DIR}/，摘要 → {RESULTS_DIR}/summary.json")


if __name__ == "__main__":
    main()
