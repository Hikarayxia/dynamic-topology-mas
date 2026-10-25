"""记录真实运动轨迹及边界指标，使用固定模型进行独立评估，不训练模型。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments import Continuous2DSearchEnv
from evaluation.evaluator import compare_with_nominal, run_episode, summarize_scenario
from evaluation.reporting import save_evaluation_results
from evaluation.trajectory import plot_motion, summarize_motion
from scripts.evaluate_policies import _load_model, _scenarios


def main() -> None:
    """写入全新目录；汇总所有回合，但只保存指定数量的完整轨迹以控制磁盘占用。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--agent-counts", default="4")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--trajectory-episodes", type=int, default=1)
    parser.add_argument("--observation-mode", choices=("communicated", "oracle"), default="communicated")
    parser.add_argument("--neighbor-state-ttl", type=float, default=1.0)
    for flag in ("packet-loss-rate", "base-delay", "delay-jitter", "noise-std"):
        parser.add_argument("--" + flag, type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if (args.episodes <= 0 or args.max_steps <= 0 or args.seed < 0
                or not 0 <= args.trajectory_episodes <= args.episodes):
            raise ValueError("回合/步数必须为正，种子非负，轨迹回合数必须在 0 到总回合数之间")
        if args.output_dir.exists():
            raise ValueError("运动评估目录已存在，请使用新目录")
        source = args.checkpoint.read_bytes() if args.checkpoint else None
        model = _load_model(args.checkpoint, 64, checkpoint_bytes=source) if source is not None else None
        args.coverage_memory = model is not None and model.observation_profile == "local_coverage_v1"
        scenarios = _scenarios(args)
        policies = ["rule", "replacement"] + (["learned"] if model is not None else [])
        # 原实现无版本字段时明确标注旧语义，不把改名或重跑的文件当作旧结果。
        profile = getattr(Continuous2DSearchEnv, "MOTION_PROFILE", "command_velocity_v1")
        sources = {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in (
            "environments/continuous_2d.py", "evaluation/evaluator.py", "evaluation/trajectory.py",
            "scripts/evaluate_motion.py",
        )}
        args.output_dir.mkdir(parents=True, exist_ok=False)
        trajectory_dir = args.output_dir / "trajectories"
        trajectory_dir.mkdir()
        scenario_path = args.output_dir / "scenario_config.json"
        scenario_path.write_text(json.dumps([asdict(row) for row in scenarios], ensure_ascii=False, indent=2), encoding="utf-8")
        episodes, summaries, motion_rows, files = [], [], [], [scenario_path]
        for scenario in scenarios:
            for policy in policies:
                group = []
                for episode in range(scenario.episodes):
                    trajectory = []
                    metrics = run_episode(scenario, policy=policy, episode=episode, model=model, trajectory=trajectory)
                    group.append(metrics)
                    motion_rows.append(dict(scenario=scenario.name, policy=policy, agent_count=scenario.agent_count,
                                            episode=episode, seed=metrics.seed, **summarize_motion(trajectory)))
                    if episode < args.trajectory_episodes:
                        stem = f"{scenario.name}_{policy}_n{scenario.agent_count}_episode{episode}"
                        path = trajectory_dir / (stem + ".jsonl")
                        with path.open("x", encoding="utf-8") as stream:
                            for row in trajectory:
                                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        files.append(path)
                        # 有模型时重点画学习策略；规则/补位的完整记录仍可单独重画。
                        if policy == "learned" or model is None:
                            files.append(plot_motion(trajectory, trajectory_dir / (stem + ".png")))
                episodes.extend(group)
                summaries.append(summarize_scenario(group))
                print(f"{profile}: {scenario.name}/{policy} success={summaries[-1].success_rate:.3f}, "
                      f"reward={summaries[-1].mean_cumulative_reward:.3f}", flush=True)
        paths = save_evaluation_results(args.output_dir, episodes, compare_with_nominal(summaries))
        files.extend(paths.values())
        motion_path = args.output_dir / "motion_metrics.jsonl"
        with motion_path.open("x", encoding="utf-8") as stream:
            for row in motion_rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        csv_path = args.output_dir / "motion_metrics.csv"
        with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(motion_rows[0]))
            writer.writeheader()
            writer.writerows(motion_rows)
        files.extend([motion_path, csv_path])
        provenance = dict(
            schema_version=1, checkpoint_sha256=hashlib.sha256(source).hexdigest() if source is not None else None,
            model_config=model.architecture_config() if model is not None else None,
            deterministic=True, motion_profile=profile, code_sha256=sources,
            trajectory_episodes=args.trajectory_episodes,
            result_sha256={path.relative_to(args.output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in files},
        )
        (args.output_dir / "evaluation_provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(1, f"运动评估未完成: {exc}\n")
    print(f"运动评估已保存: {args.output_dir}")


if __name__ == "__main__":
    main()
