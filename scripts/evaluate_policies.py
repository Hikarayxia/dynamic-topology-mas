"""规则基线、学习策略、动态故障和规模泛化评估入口。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import SharedGraphActorCritic
from evaluation.evaluator import (
    ExperimentScenario,
    LinkFaultSpec,
    NodeFaultSpec,
    compare_with_nominal,
    evaluate_scenario,
    summarize_scenario,
)
from evaluation.plotting import plot_evaluation_summary, plot_replacement_summary
from evaluation.reporting import save_evaluation_results
from interaction import CommunicationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行独立弹性与规模泛化评估")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--agent-counts", default="3,4,6")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--coverage-memory", action=argparse.BooleanOptionalAction, default=None,
                        help="不指定时按检查点自动选择；显式指定必须与检查点一致")
    parser.add_argument("--observation-mode", choices=("communicated", "oracle"), default="communicated")
    parser.add_argument("--neighbor-state-ttl", type=float, default=1.0)
    parser.add_argument("--packet-loss-rate", type=float, default=0.0)
    parser.add_argument("--base-delay", type=float, default=0.0)
    parser.add_argument("--delay-jitter", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation")
    return parser.parse_args()


def _agent_counts(raw: str) -> list[int]:
    counts = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("agent-counts 必须是逗号分隔的正整数")
    return counts


def _scenarios(args: argparse.Namespace) -> list[ExperimentScenario]:
    scenarios: list[ExperimentScenario] = []
    total_time = args.max_steps * 0.2
    link_start = total_time * 0.25
    node_start = total_time * 0.55
    fault_duration = max(0.4, total_time * 0.1)
    for count in _agent_counts(args.agent_counts):
        scenarios.append(
            ExperimentScenario(
                name="nominal",
                agent_count=count,
                episodes=args.episodes,
                base_seed=args.seed,
                max_steps=args.max_steps,
            )
        )
        link_faults = (
            (LinkFaultSpec("agent-0", "agent-1", link_start, fault_duration),)
            if count >= 2
            else ()
        )
        node_faults = (
            NodeFaultSpec(f"agent-{count - 1}", node_start, fault_duration),
        )
        scenarios.append(
            ExperimentScenario(
                name="link_node_fault",
                agent_count=count,
                episodes=args.episodes,
                base_seed=args.seed,
                max_steps=args.max_steps,
                link_faults=link_faults,
                node_faults=node_faults,
            )
        )
        if any((args.packet_loss_rate, args.base_delay, args.delay_jitter, args.noise_std)):
            scenarios.append(ExperimentScenario(
                name="communication_fault",
                agent_count=count, episodes=args.episodes, base_seed=args.seed,
                max_steps=args.max_steps,
                communication_config=CommunicationConfig(
                    packet_loss_rate=args.packet_loss_rate, base_delay=args.base_delay,
                    delay_jitter=args.delay_jitter, noise_std=args.noise_std,
                ),
            ))
    # nominal 始终是理想通信；拓扑故障组与通信故障组分开，便于归因。
    return [replace(
        scenario, neighbor_observation_mode=args.observation_mode,
        neighbor_state_ttl=args.neighbor_state_ttl,
        coverage_memory=bool(getattr(args, "coverage_memory", False)),
    ) for scenario in scenarios]


def _load_model(
    checkpoint: Path, hidden_dim: int, *, checkpoint_bytes: bytes | None = None,
) -> SharedGraphActorCritic:
    """从检查点重建推理模型；可传入已读取字节，使加载内容与来源指纹严格一致。"""

    source = BytesIO(checkpoint_bytes) if checkpoint_bytes is not None else checkpoint
    data = torch.load(source, map_location="cpu", weights_only=False)
    config = data.get(
        "model_config",
        {
            "self_dim": 6,
            "neighbor_dim": 5,
            "task_dim": 2,
            "action_dim": 2,
            "hidden_dim": hidden_dim,
        },
    )
    model = SharedGraphActorCritic(**config)
    model.load_state_dict(data["model"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes 和 max-steps 必须为正整数")
    # 同一份不可变字节既用于加载又用于算指纹，避免文件被并发替换后来源误标。
    checkpoint_bytes = args.checkpoint.read_bytes() if args.checkpoint else None
    model = (_load_model(args.checkpoint, args.hidden_dim, checkpoint_bytes=checkpoint_bytes)
             if args.checkpoint else None)
    if model is not None:
        enabled = model.observation_profile == "local_coverage_v1"
        if args.coverage_memory is not None and args.coverage_memory != enabled:
            raise ValueError("显式 coverage-memory 与检查点观测配置不一致")
        args.coverage_memory = enabled
    else:
        args.coverage_memory = bool(args.coverage_memory)
    # 规则基线与协作补位策略始终使用相同场景比较；只有显式提供检查点时才加入
    # 学习策略，避免未训练模型被误当成有效实验组。
    policies = ["rule", "replacement"]
    if model is not None:
        policies.append("learned")
    all_episodes = []
    summaries = []
    scenarios = _scenarios(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scenario_config.json").write_text(
        json.dumps([asdict(scenario) for scenario in scenarios], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for scenario in scenarios:
        for policy in policies:
            episodes = evaluate_scenario(scenario, policy=policy, model=model)
            all_episodes.extend(episodes)
            summaries.append(summarize_scenario(episodes))
            summary = summaries[-1]
            print(
                f"{policy}/{scenario.name}/N={scenario.agent_count}: "
                f"success={summary.success_rate:.3f}, "
                f"reward={summary.mean_cumulative_reward:.3f}, "
                f"recovery={summary.mean_recovery_time}, "
                f"replacement_success={summary.mean_replacement_success_rate}"
            )

    summaries = compare_with_nominal(summaries)
    paths = save_evaluation_results(args.output_dir, all_episodes, summaries)
    provenance = dict(
        schema_version=1,
        checkpoint_sha256=(hashlib.sha256(checkpoint_bytes).hexdigest()
                           if checkpoint_bytes is not None else None),
        model_config=model.architecture_config() if model is not None else None,
        deterministic=True,
        result_sha256={name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
                       for name in ("scenario_config.json", "episode_metrics.jsonl")},
    )
    # 只在实际评估完成后记录来源。该清单检测误配与意外改动，不是防恶意篡改签名。
    (args.output_dir / "evaluation_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    figure = plot_evaluation_summary(summaries, args.output_dir / "evaluation_summary.png")
    replacement_figure = plot_replacement_summary(
        summaries, args.output_dir / "replacement_summary.png"
    )
    print(f"逐回合指标: {paths['episodes']}")
    print(f"汇总表: {paths['summary_csv']}")
    print(f"评估图: {figure}")
    print(f"补位专项图: {replacement_figure}")


if __name__ == "__main__":
    main()
