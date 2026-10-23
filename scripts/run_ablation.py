"""顺序运行多种聚合器和独立训练种子，保存每组的命令及独立评估。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Graph-MAPPO 多种子消融实验矩阵")
    parser.add_argument("--aggregations", default="none,mean,attention")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-agents", type=int, default=5)
    parser.add_argument("--agent-counts", default="5,10,20,50")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--packet-loss-rate", type=float, default=0.3)
    parser.add_argument("--base-delay", type=float, default=0.2)
    parser.add_argument("--delay-jitter", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--neighbor-state-ttl", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "ablation")
    parser.add_argument("--dry-run", action="store_true", help="只写入实验计划，不执行训练")
    args = parser.parse_args()
    aggregations = list(dict.fromkeys(value.strip() for value in args.aggregations.split(",")))
    seeds = list(dict.fromkeys(int(value.strip()) for value in args.seeds.split(",")))
    counts = [int(value.strip()) for value in args.agent_counts.split(",")]
    if not set(aggregations) <= {"none", "mean", "attention"}:
        parser.error("aggregations 只能包含 none、mean、attention")
    if min(seeds) < 0 or min(counts) <= 0:
        parser.error("训练种子必须非负，测试智能体数必须为正")
    if min(args.train_agents, args.updates, args.rollout_steps, args.update_epochs,
           args.max_steps, args.hidden_dim, args.episodes) <= 0:
        parser.error("智能体数、训练步数、模型维度和回合数必须为正")
    communication = []
    for name in ("packet_loss_rate", "base_delay", "delay_jitter", "noise_std", "neighbor_state_ttl"):
        communication.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    plan = []
    for aggregation in aggregations:
        for seed in seeds:
            run_dir = args.output_dir.resolve() / aggregation / f"seed_{seed}"
            train_dir, evaluation_dir = run_dir / "training", run_dir / "evaluation"
            if train_dir.exists() or evaluation_dir.exists():
                parser.error(f"已有实验输出，请更换 output-dir: {run_dir}")
            train = [
                sys.executable, str(PROJECT_ROOT / "scripts" / "train_mappo.py"),
                "--aggregation", aggregation, "--seed", str(seed),
                "--agents", str(args.train_agents), "--updates", str(args.updates),
                "--rollout-steps", str(args.rollout_steps),
                "--update-epochs", str(args.update_epochs),
                "--max-steps", str(args.max_steps), "--hidden-dim", str(args.hidden_dim),
                "--output-dir", str(train_dir), *communication,
            ]
            evaluate = [
                sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_policies.py"),
                "--checkpoint", str(train_dir / "mappo_checkpoint.pt"),
                "--seed", str(args.eval_seed), "--episodes", str(args.episodes),
                "--agent-counts", args.agent_counts, "--max-steps", str(args.max_steps),
                "--output-dir", str(evaluation_dir), *communication,
            ]
            plan.append({"aggregation": aggregation, "seed": seed, "train": train, "evaluate": evaluate})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "ablation_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"实验计划: {plan_path}，共 {len(plan)} 个独立训练组", flush=True)
    if args.dry_run:
        return
    for run in plan:
        print(f"开始 {run['aggregation']} / seed={run['seed']}", flush=True)
        subprocess.run(run["train"], cwd=PROJECT_ROOT, check=True)
        subprocess.run(run["evaluate"], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
