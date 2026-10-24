"""训练和评估结果的非交互式 PNG 绘图。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

# Windows 受限环境可能无法写用户默认缓存；显式使用临时目录避免导入告警。
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dynamic-topology-mas-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluator import ScenarioSummary


def plot_training_metrics(metrics_path: str | Path, output_path: str | Path) -> Path:
    """从训练 JSONL 绘制奖励、Actor 和 Value 损失曲线。"""

    source = Path(metrics_path)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("训练指标文件为空")
    updates = [row["update"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(updates, [row["mean_team_reward"] for row in rows], marker="o")
    axes[0].set_title("Mean team reward")
    axes[1].plot(updates, [row["actor_loss"] for row in rows], marker="o")
    axes[1].set_title("Actor loss")
    axes[2].plot(updates, [row["value_loss"] for row in rows], marker="o")
    axes[2].set_title("Value loss")
    for axis in axes:
        axis.set_xlabel("Update")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def plot_ablation_summary(rows: Sequence[dict], output_path: str | Path) -> Path:
    """按场景分面比较规模泛化；误差棒只使用可估计的训练种子样本标准差。

    rows 为跨种子汇总行，返回 PNG 路径。缺失事件显示为空缺，单种子及去重后
    的规则基线不画误差棒，避免把“无法估计”误画成确定的零波动。
    """

    if not rows:
        raise ValueError("消融汇总不能为空")
    scenarios = sorted({row["scenario"] for row in rows})
    curves = sorted({(row["aggregation"], row["policy"]) for row in rows})
    metrics = (("success_rate", "Success rate"),
               ("mean_cumulative_reward", "Cumulative reward"),
               ("mean_final_detection_ratio", "Final detection ratio"),
               ("mean_recovery_time", "Recovery time (s)"))
    colors = {"none": "tab:blue", "mean": "tab:orange", "attention": "tab:green",
              "rule": "tab:gray", "replacement": "tab:purple"}
    figure, axes = plt.subplots(len(scenarios), 4, squeeze=False,
                                figsize=(16, max(3.8, len(scenarios) * 3.1)))
    for row_index, scenario in enumerate(scenarios):
        for axis, (metric, title) in zip(axes[row_index], metrics):
            has_observation = False
            for aggregation, policy in curves:
                points = sorted((row for row in rows if row["scenario"] == scenario
                                 and row["aggregation"] == aggregation and row["policy"] == policy),
                                key=lambda row: row["agent_count"])
                if not points:
                    continue
                label = (f"{aggregation} (seeds={points[0]['training_seed_count']})"
                         if policy == "learned" else f"{policy} (fixed)")
                color = colors[aggregation if policy == "learned" else policy]
                x = [point["agent_count"] for point in points]
                values = [point["metrics"][metric]["mean"] for point in points]
                axis.plot(x, [float("nan") if value is None else value for value in values],
                          marker="o", label=label, color=color,
                          linestyle="-" if policy == "learned" else "--")
                observed = [point for point in points if point["metrics"][metric]["mean"] is not None]
                has_observation |= bool(observed)
                uncertain = [point for point in observed if point["metrics"][metric]["std"] is not None]
                if uncertain:
                    axis.errorbar([point["agent_count"] for point in uncertain],
                                  [point["metrics"][metric]["mean"] for point in uncertain],
                                  yerr=[point["metrics"][metric]["std"] for point in uncertain],
                                  fmt="none", capsize=3, color=color)
            axis.set_title(f"{scenario}\n{title}", fontsize=10)
            axis.set_xlabel("Agents")
            axis.set_xticks(sorted({row["agent_count"] for row in rows if row["scenario"] == scenario}))
            axis.grid(alpha=0.25)
            if not has_observation:
                axis.text(0.5, 0.5, "No observed event", transform=axis.transAxes, ha="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=len(curves))
    figure.suptitle("Ablation comparison across independent training seeds")
    figure.text(0.5, 0.01, "Error bars: sample SD; omitted for <2 valid seeds. Rule baselines deduplicated.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.11 if len(scenarios) == 1 else 0.07, 1, 0.95))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def plot_evaluation_summary(
    summaries: Sequence[ScenarioSummary], output_path: str | Path
) -> Path:
    """绘制规模/故障场景下的成功率、奖励、一致性和恢复时间。"""

    if not summaries:
        raise ValueError("summaries 不能为空")
    labels = [
        "\n".join(
            (item.policy, item.scenario.replace("_", "\n"), f"N={item.agent_count}")
        )
        for item in summaries
    ]
    x_values = list(range(len(summaries)))
    recovery = [
        item.mean_recovery_time if item.mean_recovery_time is not None else 0.0
        for item in summaries
    ]
    figure, axes = plt.subplots(2, 2, figsize=(max(10, len(summaries) * 1.6), 7))
    series = (
        ("Success rate", [item.success_rate for item in summaries]),
        ("Cumulative reward", [item.mean_cumulative_reward for item in summaries]),
        ("Consistency error", [item.mean_consistency_error for item in summaries]),
        ("Recovery time (s)", recovery),
    )
    for axis_index, (axis, (title, values)) in enumerate(zip(axes.flat, series)):
        bars = axis.bar(x_values, values)
        axis.set_title(title)
        # 三行标签已经按策略/场景/规模分组；保持水平可避免多场景时相邻文字交叉。
        axis.set_xticks(x_values, labels, rotation=0, ha="center", fontsize=8)
        axis.grid(axis="y", alpha=0.3)
        # 显式标注零值和缺失值，避免空白坐标轴被误解成没有执行实验。
        for item_index, (bar, value) in enumerate(zip(bars, values)):
            missing_recovery = (
                axis_index == 3
                and summaries[item_index].mean_recovery_time is None
            )
            label = "N/A" if missing_recovery else f"{value:.2f}"
            axis.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def plot_replacement_summary(
    summaries: Sequence[ScenarioSummary], output_path: str | Path
) -> Path:
    """绘制补位响应、真实覆盖恢复、未覆盖比例和任务切换专项图。"""

    if not summaries:
        raise ValueError("summaries 不能为空")
    labels = [
        "\n".join(
            (item.policy, item.scenario.replace("_", "\n"), f"N={item.agent_count}")
        )
        for item in summaries
    ]
    x_values = list(range(len(summaries)))
    series = (
        (
            "Replacement response (s)",
            [item.mean_replacement_response_time for item in summaries],
        ),
        (
            "Coverage recovery (s)",
            [item.mean_coverage_recovery_time for item in summaries],
        ),
        (
            "Mean uncovered lane ratio",
            [item.mean_uncovered_lane_ratio for item in summaries],
        ),
        (
            "Replacement switches",
            [item.mean_replacement_switches for item in summaries],
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(max(10, len(summaries) * 1.6), 7))
    for axis, (title, optional_values) in zip(axes.flat, series):
        values = [0.0 if value is None else float(value) for value in optional_values]
        bars = axis.bar(x_values, values)
        axis.set_title(title)
        # 专项图通常包含更多长策略名，水平三行标签比倾斜标签更易逐组比较。
        axis.set_xticks(x_values, labels, rotation=0, ha="center", fontsize=8)
        axis.grid(axis="y", alpha=0.3)
        for bar, raw_value in zip(bars, optional_values):
            # None 表示事件从未发生，必须显式标注，不能让零高度柱伪装成零秒恢复。
            label = "N/A" if raw_value is None else f"{float(raw_value):.2f}"
            axis.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target
