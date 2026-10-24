"""核对单种子试训的前后独立测试，保存可追溯的配对比较，不修改原始结果。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import SharedGraphActorCritic
from evaluation.ablation import (
    _OPTIONAL_EPISODE_METRICS, _integer, _read_json, _training_signature,
)
from evaluation.evaluator import (
    ExperimentScenario, LinkFaultSpec, NodeFaultSpec, summarize_scenario,
)
from evaluation.metrics import EpisodeMetrics
from interaction import CommunicationConfig


def _evaluation(directory: Path) -> tuple[dict, dict]:
    """读取场景及逐回合指标；返回配置表和按策略分组的有序回合。

    参数为现有评估目录。严格检查回合完整性和配对种子，防止漏掉失败回合后
    仍输出偏高均值；未发生的恢复事件继续保留为空值。
    """

    scenarios = {}
    rows = _read_json(directory / "scenario_config.json")
    if not isinstance(rows, list) or not rows:
        raise ValueError("评估场景不能为空")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {item.name for item in fields(ExperimentScenario)}:
            raise ValueError("评估场景字段不完整")
        for name in ("agent_count", "episodes", "max_steps"):
            _integer(row[name], name, minimum=1)
        _integer(row["base_seed"], "测试种子")
        # 使用真实场景类型检查故障引用，不只检查两份 JSON 是否碰巧相等。
        ExperimentScenario(**{
            **row, "communication_config": CommunicationConfig(**row["communication_config"]),
            "link_faults": tuple(LinkFaultSpec(**value) for value in row["link_faults"]),
            "node_faults": tuple(NodeFaultSpec(**value) for value in row["node_faults"]),
        })
        key = row["name"], row["agent_count"]
        if key in scenarios:
            raise ValueError("重复的场景配置")
        scenarios[key] = row

    grouped = {}
    for row in _read_json(directory / "episode_metrics.jsonl", lines=True):
        if not isinstance(row, dict) or set(row) != {item.name for item in fields(EpisodeMetrics)}:
            raise ValueError("逐回合指标字段不完整")
        if row["policy"] not in {"rule", "replacement", "learned"}:
            raise ValueError("未知策略")
        key = row["scenario"], row["agent_count"]
        if key not in scenarios:
            raise ValueError("回合引用未知场景")
        scenario = scenarios[key]
        episode = _integer(row["episode"], "回合编号")
        _integer(row["steps"], "回合步数", minimum=1)
        if (episode >= scenario["episodes"] or row["seed"] != scenario["base_seed"] + episode
                or row["steps"] > scenario["max_steps"]):
            raise ValueError("回合种子、编号或步数不符合场景协议")
        if (type(row["success"]) is not bool or row["convergence_steps"]
                != (row["steps"] if row["success"] else None)):
            raise ValueError("成功状态与收敛步数不一致")
        if not row["success"] and row["steps"] != scenario["max_steps"]:
            # 原始指标未保存提前失败的终止原因，不能把缺失轨迹误当合法终止。
            # 本入口保守限制为失败至时间上限；全离线等提前终止需另行诊断。
            raise ValueError("失败回合未达到步数上限；本分析不支持无法核验原因的提前失败终止")
        for name, value in row.items():
            if name in {"scenario", "policy", "success"}:
                continue
            if value is None:
                if name not in _OPTIONAL_EPISODE_METRICS:
                    raise ValueError(f"必填指标 {name} 缺失")
            elif type(value) not in (int, float):
                raise ValueError(f"指标 {name} 必须为数值")
        group = grouped.setdefault((row["policy"], *key), {})
        if episode in group:
            raise ValueError("重复的评估回合")
        group[episode] = EpisodeMetrics(**row)
    for (name, count), scenario in scenarios.items():
        for policy in ("rule", "replacement", "learned"):
            key = policy, name, count
            if set(grouped.get(key, {})) != set(range(scenario["episodes"])):
                raise ValueError("评估回合不完整")
            grouped[key] = [grouped[key][index] for index in range(scenario["episodes"])]
    return scenarios, grouped


def analyze_pilot(
    input_dir: str | Path, *, initial_evaluation: Path | None = None,
    final_evaluation: Path | None = None,
) -> dict:
    """分析指定试训目录，返回训练诊断、去重汇总及逐场景配对差值。

    目录包含 initial/、training/ 和前后评估目录；可显式传入新的评估子目录。
    每份评估必须有执行时生成的来源清单。失败回合仅接受达到时间上限的情况，
    全部离线等提前失败因旧指标缺少终止原因而保守拒绝，不将其伪装成完整数据。
    这里只评估一个模型训练种子，不估计跨训练种子方差或声称算法显著性。
    """

    root = Path(input_dir).resolve()
    config = _read_json(root / "training/run_config.json")
    args = config["arguments"]
    _training_signature(root, dict(
        seed=args["seed"], aggregation=args["aggregation"],
        train=[sys.executable, str(PROJECT_ROOT / "scripts/train_mappo.py")],
    ))
    metrics = _read_json(root / "training/training_metrics.jsonl", lines=True)
    for row in metrics:
        finished = _integer(row.get("episodes_finished"), "完成回合数")
        successful = _integer(row.get("successful_episodes"), "成功回合数")
        if successful > finished:
            raise ValueError("成功回合不能多于完成回合")
    finished = sum(row["episodes_finished"] for row in metrics)
    initial = torch.load(root / "initial/mappo_checkpoint.pt", map_location="cpu", weights_only=True)
    final = torch.load(root / "training/mappo_checkpoint.pt", map_location="cpu", weights_only=True)
    if (initial.get("environment_steps") != 0 or initial.get("initialization_seed") != args["seed"]
            or final.get("environment_steps") != metrics[-1]["environment_steps"]
            or final.get("config") != config["training"]
            or any(item.get("model_config") != config["model"] for item in (initial, final))):
        raise ValueError("训练前后检查点、种子或采样预算与配置不一致")
    for checkpoint in (initial, final):
        model = SharedGraphActorCritic(**config["model"])
        model.load_state_dict(checkpoint["model"])
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError("检查点参数存在非有限数值")
    difference = sum(float((final["model"][name].double() - value.double()).square().sum())
                     for name, value in initial["model"].items()) ** 0.5

    evaluation_dirs = {
        "initial": Path(initial_evaluation or root / "evaluation_initial").resolve(),
        "final": Path(final_evaluation or root / "evaluation_final").resolve(),
    }
    source_files = [root / name for name in (
        "training/run_config.json", "training/training_metrics.jsonl",
        "initial/mappo_checkpoint.pt", "training/mappo_checkpoint.pt",
    )]
    for stage, directory in evaluation_dirs.items():
        # 独立评估可放在新的子目录；不允许输入根外的文件混入同一试训来源清单。
        if not directory.is_relative_to(root):
            raise ValueError("评估目录必须位于本次试训根目录内")
        metadata = _read_json(directory / "evaluation_provenance.json")
        checkpoint_path = root / ("initial" if stage == "initial" else "training") / "mappo_checkpoint.pt"
        if (metadata.get("schema_version") != 1 or metadata.get("deterministic") is not True
                or metadata.get("model_config") != config["model"]
                or metadata.get("checkpoint_sha256") != hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()):
            raise ValueError("评估来源的检查点指纹或推理配置不一致")
        for name in ("scenario_config.json", "episode_metrics.jsonl"):
            path = directory / name
            if metadata.get("result_sha256", {}).get(name) != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("评估来源清单中的结果指纹不一致")
            source_files.append(path)
        source_files.append(directory / "evaluation_provenance.json")
    initial_scenarios, before = _evaluation(evaluation_dirs["initial"])
    final_scenarios, after = _evaluation(evaluation_dirs["final"])
    if initial_scenarios != final_scenarios:
        raise ValueError("训练前后评估场景协议不一致")
    # Trainer 按完成回合递增 reset 种子，最后一个未完成回合也占用一个种子。
    training_seeds = set(range(args["seed"], args["seed"] + finished + 1))
    test_seeds = set()
    for scenario in final_scenarios.values():
        test_seeds.update(range(scenario["base_seed"], scenario["base_seed"] + scenario["episodes"]))
        for name in ("width", "height", "n_targets", "max_steps",
                     "neighbor_observation_mode", "neighbor_state_ttl"):
            if scenario[name] != config["environment"][name]:
                raise ValueError(f"训练与评估任务定义不一致: {name}")
        if scenario["agent_count"] != args["agents"]:
            raise ValueError("本入口只比较同规模试训，规模泛化请使用消融入口")
    if training_seeds & test_seeds:
        raise ValueError("独立测试种子与训练回合种子发生重叠")

    summaries, comparisons = [], []
    for (scenario, count), protocol in sorted(final_scenarios.items()):
        trained = after["learned", scenario, count]
        references = {"initial": before["learned", scenario, count]}
        for policy in ("rule", "replacement"):
            key = policy, scenario, count
            if before[key] != after[key]:
                raise ValueError(f"配对测试中的规则基线不一致: {key}")
            references[policy] = after[key]
        for label, episodes in {**references, "trained": trained}.items():
            summaries.append({**summarize_scenario(episodes).as_dict(), "label": label})
        for label, reference in references.items():
            # 先逐测试种子作差再取均值；规则对照只保存一份，不冒充独立训练重复。
            comparisons.append(dict(
                scenario=scenario, agent_count=count, reference=label,
                paired_episodes=protocol["episodes"],
                reward_delta=mean(a.cumulative_team_reward - b.cumulative_team_reward
                                  for a, b in zip(trained, reference)),
                success_rate_delta=mean(int(a.success) - int(b.success) for a, b in zip(trained, reference)),
                detection_ratio_delta=mean(a.final_detection_ratio - b.final_detection_ratio
                                           for a, b in zip(trained, reference)),
            ))
    window = min(10, len(metrics))
    return dict(
        training_seed_count=1, training_seed=args["seed"],
        training=dict(environment_steps=metrics[-1]["environment_steps"], updates=len(metrics),
                      episodes_finished=finished,
                      successful_episodes=sum(row["successful_episodes"] for row in metrics),
                      parameters_changed=difference > 0, parameter_l2_change=difference,
                      reward_window_updates=window,
                      first_window_reward=mean(row["mean_team_reward"] for row in metrics[:window]),
                      last_window_reward=mean(row["mean_team_reward"] for row in metrics[-window:])),
        evaluation_seeds=sorted(test_seeds), summaries=summaries, comparisons=comparisons,
        source_sha256={path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in source_files},
        warnings=["只有一个训练种子，不估计跨训练种子方差，不据此声称收敛或泛化优势。",
                  "训练日志的平均每步奖励与评估的回合累计奖励口径不同，不能直接比较。",
                  "规则基线可能使用成员索引等元数据；这是系统级对照，不是同信息集的聚合器消融。"],
    )


def save_report(report: dict, output_dir: Path) -> None:
    """将分析保存到全新目录；返回前绘图，拒绝覆盖已有分析和原始结果。"""

    from evaluation.plotting import plt

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "pilot_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output_dir / "paired_comparisons.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["comparisons"][0]))
        writer.writeheader()
        writer.writerows(report["comparisons"])
    scenarios = sorted({row["scenario"] for row in report["summaries"]})
    labels = ("initial", "trained", "rule", "replacement")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, (metric, title) in zip(axes, (
        ("success_rate", "Task success rate"),
        ("mean_final_detection_ratio", "Final detection ratio"),
        ("mean_cumulative_reward", "Cumulative reward"),
    )):
        for index, label in enumerate(labels):
            values = [next(row[metric] for row in report["summaries"]
                           if row["scenario"] == scenario and row["label"] == label) for scenario in scenarios]
            axis.bar([position + (index - 1.5) * 0.2 for position in range(len(scenarios))],
                     values, width=0.2, label=label)
        axis.set_xticks(range(len(scenarios)), [value.replace("_", "\n") for value in scenarios])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=4)
    figure.suptitle("Single-seed pilot: paired held-out episodes (no cross-training-seed error bars)")
    figure.tight_layout(rect=(0, 0.09, 1, 0.94))
    figure.savefig(output_dir / "pilot_comparison.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """解析现有试训目录并保存结果；不训练模型、不执行输入文件中的命令。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-evaluation", type=Path, help="新的训练前评估子目录；不指定时使用默认布局")
    parser.add_argument("--final-evaluation", type=Path, help="新的训练后评估子目录；不指定时使用默认布局")
    args = parser.parse_args()
    try:
        report = analyze_pilot(args.input_dir, initial_evaluation=args.initial_evaluation,
                               final_evaluation=args.final_evaluation)
        save_report(report, args.output_dir)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(1, f"分析未完成: {exc}\n")
    print(f"配对分析完成: {args.output_dir}")
    for warning in report["warnings"]:
        print(warning)


if __name__ == "__main__":
    main()
