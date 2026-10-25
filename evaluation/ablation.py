"""读取真实实验矩阵，检查可比性，并按独立训练种子汇总消融结果。"""

from __future__ import annotations

import csv
from dataclasses import fields
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .evaluator import ScenarioSummary, compare_with_nominal, summarize_scenario
from .metrics import EpisodeMetrics


_POLICIES = ("rule", "replacement", "learned")
_OPTIONAL_EPISODE_METRICS = {
    "convergence_steps", "fault_recovery_time", "topology_performance_drop",
    "replacement_response_time", "coverage_recovery_time", "replacement_success_rate",
}
_METRICS = tuple(
    item.name for item in fields(ScenarioSummary)
    if item.name not in {"scenario", "policy", "agent_count", "episodes"}
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"结果必须为有限数值，不能包含 {value}")


def _finite_float(text: str) -> float:
    """JSON 的指数溢出不经过 parse_constant，因此还需检查普通浮点解析。"""

    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"配置和指标必须为有限数值: {text}")
    return value


def _read_json(path: Path, *, lines: bool = False) -> Any:
    """读取 JSON/JSONL；缺文件、损坏文件和非有限数值都不能被静默跳过。"""

    try:
        text = path.read_text(encoding="utf-8-sig")
        if lines:
            return [json.loads(line, parse_constant=_reject_constant, parse_float=_finite_float)
                    for line in text.splitlines() if line.strip()]
        return json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取实验文件 {path}: {exc}") from exc


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} 必须为不小于 {minimum} 的整数")
    return value


def _options(command: Any) -> dict[str, Any]:
    """只读取计划中的命令参数，不执行计划提供的程序或命令。"""

    if not isinstance(command, list) or len(command) < 2 or not all(
        isinstance(token, str) for token in command
    ):
        raise ValueError("实验计划的 train/evaluate 必须为命令参数列表")
    options: dict[str, Any] = {}
    index = 2
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            raise ValueError(f"实验计划中存在无法识别的参数: {token}")
        key, separator, value = token[2:].partition("=")
        key = key.replace("-", "_")
        if key in options:
            raise ValueError(f"实验计划中存在重复参数: {key}")
        if not separator:
            value = True
            if index + 1 < len(command) and not command[index + 1].startswith("--"):
                index += 1
                value = command[index]
        options[key] = value
        index += 1
    return options


def _plan(root: Path) -> list[dict[str, Any]]:
    plan = _read_json(root / "ablation_plan.json")
    if not isinstance(plan, list) or not plan:
        raise ValueError("实验计划必须是非空列表")
    keys = set()
    for entry in plan:
        if not isinstance(entry, dict) or entry.get("aggregation") not in {"none", "mean", "attention"}:
            raise ValueError("实验计划包含不支持的聚合器")
        seed = _integer(entry.get("seed"), "训练种子")
        key = entry["aggregation"], seed
        if key in keys:
            raise ValueError(f"重复的聚合器/训练种子: {key}")
        keys.add(key)
        _options(entry.get("train"))
        _options(entry.get("evaluate"))
    expected = {(aggregation, seed) for aggregation, _ in keys for _, seed in keys}
    if missing := expected - keys:
        raise ValueError(f"实验矩阵不完整，缺少训练组: {sorted(missing)}")
    return sorted(plan, key=lambda entry: (entry["aggregation"], entry["seed"]))


def _training_signature(run: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """核对实际训练配置、计划与采样预算，返回去除种子/聚合器差异的配置。"""

    config = _read_json(run / "training/run_config.json")
    for section in ("arguments", "model", "environment", "communication", "training"):
        if not isinstance(config, dict) or not isinstance(config.get(section), dict):
            raise ValueError(f"训练配置缺少 {section}: {run}")
    arguments = config["arguments"]
    if arguments.get("seed") != entry["seed"] or any(
        section.get("aggregation") != entry["aggregation"]
        for section in (arguments, config["model"])
    ):
        raise ValueError(f"训练种子或聚合器与配置不一致: {run}")
    for key, planned in _options(entry["train"]).items():
        if key == "output_dir":
            continue  # 允许把完整实验目录迁移到另一台机器后重新汇总。
        if key.startswith("no_"):
            key, planned = key[3:], False
        actual = arguments.get(key)
        try:
            expected = type(actual)(planned) if type(actual) in (int, float) else planned
        except (TypeError, ValueError) as exc:
            raise ValueError(f"实验计划的训练参数无效: {key}") from exc
        if actual != expected:
            raise ValueError(f"训练配置与实验计划不一致: {key}, {run}")
    updates = _integer(arguments.get("updates"), "训练更新次数", minimum=1)
    rollout_steps = _integer(arguments.get("rollout_steps"), "采样预算", minimum=1)
    if any(config["training"].get(key) != arguments.get(key)
           for key in ("rollout_steps", "update_epochs")):
        raise ValueError(f"优化器配置与训练参数不一致: {run}")
    # 缺失字段按历史 MSE 解释；显式的新损失必须与实际优化配置一致。
    loss_type = config["training"].get("value_loss_type", "mse")
    if loss_type not in {"mse", "huber"} or loss_type != arguments.get("value_loss_type", "mse"):
        raise ValueError(f"价值损失配置与训练参数不一致: {run}")
    memory = config["environment"].get("coverage_memory", False)
    profile = config["model"].get("observation_profile", "legacy")
    if (type(memory) is not bool or memory != arguments.get("coverage_memory", False)
            or profile != ("local_coverage_v1" if memory else "legacy")
            or config["model"].get("task_dim") != (20 if memory else 2)):
        raise ValueError(f"覆盖记忆的参数、环境或模型输入配置不一致: {run}")
    metrics = _read_json(run / "training/training_metrics.jsonl", lines=True)
    if len(metrics) != updates:
        raise ValueError(f"训练日志更新次数不完整: {run}")
    for index, record in enumerate(metrics, 1):
        if (not isinstance(record, dict) or record.get("update") != index
                or record.get("environment_steps") != index * rollout_steps):
            raise ValueError(f"实际训练采样预算不完整: {run}")
        for key in ("mean_team_reward", "actor_loss", "value_loss", "entropy", "total_loss"):
            value = record.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"训练指标 {key} 必须为有限数值: {run}")
    # 种子、聚合器、输出路径是组间允许的差异，其余实际配置必须完全一致。
    return {
        **config,
        # 签名也规范化历史缺省值，才能让旧完成组与新版 MSE 重跑组共同汇总。
        "arguments": {**{key: value for key, value in arguments.items()
                         if key not in {"seed", "aggregation", "output_dir"}},
                      "value_loss_type": loss_type, "coverage_memory": memory},
        "environment": {**config["environment"], "coverage_memory": memory},
        "training": {**config["training"], "value_loss_type": loss_type},
        "model": {key: value for key, value in config["model"].items() if key != "aggregation"},
    }


def _scenarios(run: Path, entry: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """加载场景及配对测试种子，保证不同训练组面对相同的测试任务。"""

    rows = _read_json(run / "evaluation/scenario_config.json")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"评估场景配置为空: {run}")
    options = _options(entry["evaluate"])
    # 评估入口会按检查点推断观测类型；离线汇总也必须遵守训练时的实际语义，
    # 不能让评估命令中的相反开关把不兼容的旧产物误判为可复用结果。
    training_config = _read_json(run / "training/run_config.json")
    training_memory = training_config["environment"].get("coverage_memory", False)
    expected_memory = options.get("coverage_memory", training_memory)
    if "no_coverage_memory" in options:
        expected_memory = False
    if type(training_memory) is not bool or expected_memory != training_memory:
        raise ValueError(f"评估覆盖记忆配置与实际训练不一致: {run}")
    scenarios = {}
    for row in rows:
        if isinstance(row, dict):
            row = {"coverage_memory": False, **row}
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"]:
            raise ValueError(f"评估场景配置无效: {run}")
        for name in ("agent_count", "episodes", "max_steps"):
            _integer(row.get(name), name, minimum=1)
        _integer(row.get("base_seed"), "测试种子")
        key = row["name"], row["agent_count"]
        if key in scenarios:
            raise ValueError(f"重复的评估场景配置: {key}, {run}")
        for flag, name in (("seed", "base_seed"), ("episodes", "episodes"), ("max_steps", "max_steps")):
            if flag in options and str(row[name]) != str(options[flag]):
                raise ValueError(f"测试种子或场景配置与计划不一致: {name}, {run}")
        for flag, default in (("observation_mode", "communicated"), ("neighbor_state_ttl", 1.0)):
            name = "neighbor_observation_mode" if flag == "observation_mode" else flag
            expected = options.get(flag, default)
            if flag == "neighbor_state_ttl":
                expected = float(expected)
            if row.get(name) != expected:
                raise ValueError(f"通信观测配置与计划不一致: {name}, {run}")
        if type(row["coverage_memory"]) is not bool or row["coverage_memory"] != expected_memory:
            raise ValueError(f"评估覆盖记忆配置与计划不一致: {run}")
        if row["name"] == "communication_fault":
            for name in ("packet_loss_rate", "base_delay", "delay_jitter", "noise_std"):
                if row["communication_config"].get(name) != float(options.get(name, 0)):
                    raise ValueError(f"通信扰动配置与计划不一致: {name}, {run}")
        scenarios[key] = row
    if "agent_counts" in options:
        counts = {int(value) for value in str(options["agent_counts"]).split(",")}
        if counts != {key[1] for key in scenarios}:
            raise ValueError(f"评估智能体规模不完整: {run}")
    return scenarios


def _episodes(run: Path, scenarios: dict) -> dict[tuple[str, str, int], list[EpisodeMetrics]]:
    """检查每个策略/场景/规模的回合是否齐全，拒绝重复、错种子和非有限指标。"""

    records = _read_json(run / "evaluation/episode_metrics.jsonl", lines=True)
    expected_fields = {item.name for item in fields(EpisodeMetrics)}
    grouped: dict[tuple[str, str, int], list[EpisodeMetrics]] = {}
    seen = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError(f"逐回合指标字段不完整或不兼容: {run}")
        if record["policy"] not in _POLICIES:
            raise ValueError(f"未知评估策略: {record['policy']}")
        scenario_key = record["scenario"], record["agent_count"]
        if scenario_key not in scenarios:
            raise ValueError(f"逐回合指标引用未知场景配置: {scenario_key}")
        scenario = scenarios[scenario_key]
        episode = _integer(record["episode"], "回合编号")
        _integer(record["seed"], "测试种子")
        _integer(record["steps"], "实际步数", minimum=1)
        if record["steps"] > scenario["max_steps"]:
            raise ValueError(f"回合步数超过场景配置: {run}")
        if episode >= scenario["episodes"] or record["seed"] != scenario["base_seed"] + episode:
            raise ValueError(f"评估回合编号或测试种子与配置不一致: {run}")
        key = record["policy"], *scenario_key
        identity = *key, episode
        if identity in seen:
            raise ValueError(f"重复的评估回合: {identity}, {run}")
        seen.add(identity)
        if type(record["success"]) is not bool:
            raise ValueError(f"success 必须为布尔值: {run}")
        if record["convergence_steps"] != (record["steps"] if record["success"] else None):
            raise ValueError(f"收敛步数与任务成功状态不一致: {run}")
        for name, value in record.items():
            if name in {"scenario", "policy", "success"}:
                continue
            if value is None and name not in _OPTIONAL_EPISODE_METRICS:
                raise ValueError(f"必填回合指标 {name} 不能缺失: {run}")
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f"回合指标 {name} 必须为有限数值或空值: {run}")
        grouped.setdefault(key, []).append(EpisodeMetrics(**record))
    for (scenario, count), config in scenarios.items():
        for policy in _POLICIES:
            key = policy, scenario, count
            if len(grouped.get(key, [])) != config["episodes"]:
                raise ValueError(f"评估回合数量不完整，缺失 {key}: {run}")
            grouped[key].sort(key=lambda item: item.episode)
    return grouped


def _statistics(values: list[float | None]) -> dict[str, float | int | None]:
    """空值不参与均值；不足两个有效训练种子时标准差保持空值。"""

    valid = [float(value) for value in values if value is not None]
    return {"mean": mean(valid) if valid else None,
            "std": stdev(valid) if len(valid) >= 2 else None, "count": len(valid)}


def summarize_ablation(input_dir: str | Path) -> dict[str, Any]:
    """检查完整实验矩阵并返回汇总，不修改源文件。

    参数为包含 ablation_plan.json 的目录。先复用现有回合汇总算法计算每个训练
    种子的指标，再等权汇总训练种子；返回来源、配置、警告和逐指标有效样本数。
    规则基线与模型初始化种子无关，重复结果核对一致后只保留一份。
    """

    root = Path(input_dir).resolve()
    plan = _plan(root)
    reference_training = reference_scenarios = None
    groups: dict[tuple[str, str, str, int], list[tuple[int, ScenarioSummary]]] = {}
    baselines: dict[tuple[str, str, int], list[EpisodeMetrics]] = {}
    for entry in plan:
        run = root / entry["aggregation"] / f"seed_{entry['seed']}"
        training = _training_signature(run, entry)
        scenarios = _scenarios(run, entry)
        if reference_training is None:
            reference_training, reference_scenarios = training, scenarios
        elif training != reference_training or scenarios != reference_scenarios:
            raise ValueError(f"不同训练组的训练预算、模型或测试配置不一致: {run}")
        episodes = _episodes(run, scenarios)
        summaries = compare_with_nominal([summarize_scenario(items) for items in episodes.values()])
        for summary in summaries:
            key = summary.policy, summary.scenario, summary.agent_count
            aggregation = entry["aggregation"] if summary.policy == "learned" else "baseline"
            if summary.policy != "learned":
                if key in baselines:
                    if episodes[key] != baselines[key]:
                        raise ValueError(f"同一配对测试任务的规则基线结果不一致: {key}, {run}")
                    continue
                baselines[key] = episodes[key]
            groups.setdefault((aggregation, *key), []).append((entry["seed"], summary))
    rows = []
    for (aggregation, policy, scenario, count), replicates in sorted(groups.items()):
        rows.append(dict(
            aggregation=aggregation, policy=policy, scenario=scenario, agent_count=count,
            training_seed_count=len(replicates) if policy == "learned" else 0,
            training_seeds=[seed for seed, _ in replicates] if policy == "learned" else [],
            replicate_count=len(replicates),
            statistical_unit="training_seed" if policy == "learned" else "deterministic_baseline",
            evaluation_episodes_per_run=replicates[0][1].episodes,
            metrics={name: _statistics([getattr(summary, name) for _, summary in replicates])
                     for name in _METRICS},
        ))
    seeds = sorted({entry["seed"] for entry in plan})
    warnings = [] if len(seeds) >= 2 else ["只有一个独立训练种子，样本标准差不可估计，结果不能代表稳定性。"]
    return dict(
        schema_version=1, source_directory=str(root), source_plan=str(root / "ablation_plan.json"),
        expected_runs=len(plan), completed_runs=len(plan), training_seeds=seeds,
        aggregations=sorted({entry["aggregation"] for entry in plan}),
        training_configuration=reference_training,
        scenario_configurations=list(reference_scenarios.values()),
        warnings=warnings, rows=rows,
    )


def _csv_rows(summaries: list[dict]) -> list[dict]:
    """统一 CSV 展平规则，让保存与续跑核验使用同一输出协议。"""

    rows = []
    for row in summaries:
        flat = {key: value for key, value in row.items() if key != "metrics"}
        flat["training_seeds"] = json.dumps(flat["training_seeds"])
        for name, statistics in row["metrics"].items():
            flat.update({f"{name}_{key}": value for key, value in statistics.items()})
        rows.append(flat)
    return rows


def save_ablation_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """将汇总保存到新目录，返回 JSON/CSV/PNG 路径；已有目录一律拒绝覆盖。"""

    from .plotting import plot_ablation_summary

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    paths = {"json": directory / "ablation_summary.json",
             "csv": directory / "ablation_summary.csv",
             "figure": directory / "ablation_comparison.png"}
    paths["json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    rows = _csv_rows(report["rows"])
    with paths["csv"].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_ablation_summary(report["rows"], paths["figure"])
    return paths
