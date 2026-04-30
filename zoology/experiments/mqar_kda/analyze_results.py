# -*- coding: utf-8 -*-
"""
fla_fixhead 三组实验统一分析脚本：MQAR / Composition / Forgetting

用法:
  # 从 wandb 拉取 + 出图（首次）
  conda run -n py310_ljl python zoology/experiments/fla_fixhead/analyze_results.py

  # 仅用本地 CSV 重新出图（离线）
  conda run -n py310_ljl python zoology/experiments/fla_fixhead/analyze_results.py --offline

输出:
  results/data/*.csv        原始数据备份
  results/figures/*.png     图表
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
FIG_DIR = RESULTS_DIR / "figures"

# wandb project 配置
# ⚠️ 实验跑完后，在下面填入对应的 sweep_id（即 launch 时打印的 sweep_name）
# 支持填入多个 sweep_id（列表），会合并拉取
TASKS = {
    "MQAR": {
        "project": "zoology_kda_mqar",
        "csv": "mqar_runs.csv",
        "sweep_ids": [
            "mqar_configs_random_false0c2620"
            # 例: "fixhd-a1b2c3"
        ],
    },
    "Composition": {
        "project": "zoology_kda_mqar",
        "csv": "composition_runs.csv",
        "sweep_ids": [
            "kda-composition-random-false-2d5758"
            # 例: "fixhd-comp-d4
            # e5f6"
        ],
    },
    "Forgetting": {
        "project": "zoology_kda_mqar",
        "csv": "forgetting_runs.csv",
        "sweep_ids": [
            "kda-forgetting-18cf0e"
            # 例: "fixhd-forget-g7h8i9"
        ],
    },
}

# 自动配色色板（按模型出现顺序分配）
_PALETTE_POOL = [
    "#4E79A7", "#E15759", "#9C755F", "#F28E2B", "#59A14F",
    "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7", "#BAB0AC",
]

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

def fetch_all(offline: bool = False, tasks: dict = None) -> dict[str, pd.DataFrame]:
    """拉取或读取三组实验数据，返回 {task_name: DataFrame}。"""
    if tasks is None:
        tasks = TASKS
    result = {}
    for task_name, info in tasks.items():
        csv_path = DATA_DIR / info["csv"]

        if offline and csv_path.exists():
            print(f"[offline] 读取 {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            sweep_ids = info.get("sweep_ids", [])
            if not sweep_ids:
                print(f"  ⚠ {task_name}: 未配置 sweep_ids，跳过 wandb 拉取")
                if csv_path.exists():
                    print(f"         尝试读取本地缓存 {csv_path}")
                    df = pd.read_csv(csv_path)
                else:
                    continue
            else:
                print(f"[wandb]  拉取 project={info['project']}, sweep_ids={sweep_ids} ...")
                df = fetch_wandb_runs(
                    project_name=info["project"],
                    sweep_id=sweep_ids,
                )
                # 只保留 finished 的 runs
                if "state" in df.columns:
                    df = df[df["state"] == "finished"]
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)
                print(f"         已保存 → {csv_path}  ({len(df)} runs)")

        if df.empty:
            print(f"  ⚠ {task_name}: 无数据，跳过")
            continue

        result[task_name] = df
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
    parser = argparse.ArgumentParser(description="fla_fixhead 实验结果分析")
    parser.add_argument("--offline", action="store_true", help="仅使用本地 CSV，不访问 wandb")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                        help="指定要分析的任务子集，如 --tasks MQAR Composition")
    parser.add_argument("--sweep-id", nargs="+", default=None,
                        help="覆盖所有任务的 sweep_ids（CLI 优先于文件内配置）")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 仅加载指定任务
    tasks = {k: v for k, v in TASKS.items() if k in args.tasks}

    # CLI --sweep-id 覆盖
    if args.sweep_id:
        for info in tasks.values():
            info["sweep_ids"] = args.sweep_id

    all_data = fetch_all(offline=args.offline, tasks=tasks)
    if not all_data:
        print("没有可用数据，退出。")
        return

    # 打印摘要
    print("\n" + "=" * 50)
    for task, df in all_data.items():
        models = df["model.name"].unique() if "model.name" in df.columns else []
        print(f"  {task}: {len(df)} runs, models={list(models)}")
    print("=" * 50 + "\n")

    plot_difficulty_curves(all_data)
    eval_df = plot_heatmaps(all_data)
    plot_radar(all_data)
    plot_fact_capacity(all_data)
    if eval_df is not None:
        save_summary(eval_df)

    print(f"\n完成！所有图表已保存至 {FIG_DIR}/，摘要已保存至 {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
