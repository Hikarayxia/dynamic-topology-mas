"""顺序运行多种聚合器和独立训练种子，保存每组的命令及独立评估。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@contextmanager
def experiment_lock(directory: Path):
    """以操作系统文件锁保护整个实验目录；退出后释放，锁文件保留以避免竞争。

    参数为实验根目录。Windows 使用字节锁，其他平台使用 flock；不靠 PID 或
    锁文件是否存在判断进程，避免异常退出后遗留的文件永久阻止续跑。
    """

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".ablation.lock"
    if path.is_symlink():
        raise ValueError("实验锁不能是符号链接")
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("该目录已有实验运行中，请等待其结束") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _event(root: Path, event: str, **details) -> None:
    """追加阶段事件而不覆盖历史，便于无人值守时追踪中断位置。"""

    path = root / "ablation_events.jsonl"
    if path.is_symlink():
        raise ValueError("实验事件日志不能是符号链接")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),
                                    event=event, **details), ensure_ascii=False) + "\n")


def _run_directory(root: Path, entry: dict) -> Path:
    path = (root / entry["aggregation"] / f"seed_{entry['seed']}").resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("训练组路径超出了实验目录，拒绝执行或归档")
    return path


def _archive(path: Path, root: Path) -> None:
    """把中断阶段原样移到同级 attempts/，不删除或覆盖任何已有结果。"""

    if (path.is_symlink() or path.resolve() == root.resolve()
            or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError(f"归档路径不在实验目录内: {path}")
    if not path.exists():
        return
    destination_dir = path.parent / "attempts"
    if destination_dir.is_symlink():
        raise ValueError("归档目录不能是符号链接")
    destination_dir.mkdir(exist_ok=True)
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    destination = destination_dir / f"{path.name}-{suffix}"
    # 两个已解析路径都受同一实验根目录约束，rename 不会递归删除原数据。
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("归档目标超出了实验目录")
    path.rename(destination)
    _event(root, "archived", source=str(path), destination=str(destination))
    print(f"保留中断结果: {destination}", flush=True)


def _check_resume_plan(root: Path, current: list[dict]) -> None:
    """核对旧计划与当前参数；只允许程序位置、输出目录和检查点路径变化。"""

    from evaluation.ablation import _options, _plan

    if not (root / "ablation_plan.json").is_file():
        raise ValueError("续跑需要原始实验计划 ablation_plan.json")

    def normalized(plan):
        return {(entry["aggregation"], entry["seed"]): {
            stage: {key: value for key, value in _options(entry[stage]).items()
                    if key not in {"output_dir", "checkpoint"}}
            for stage in ("train", "evaluate")
        } for entry in plan}

    if normalized(_plan(root)) != normalized(current):
        raise ValueError("续跑参数与原实验计划不一致，请沿用首次运行的参数或使用新输出目录")


def _training_complete(root: Path, entry: dict) -> tuple[bool, str]:
    """同时校验日志预算及真实检查点，防止只因存在 .pt 文件就跳过训练。"""

    import torch
    from algorithms import SharedGraphActorCritic
    from evaluation.ablation import _read_json, _training_signature

    run = _run_directory(root, entry)
    try:
        _training_signature(run, entry)
        config = _read_json(run / "training/run_config.json")
        # 不反序列化自定义 Python 对象；续跑只需要本框架保存的张量和基础配置。
        checkpoint = torch.load(run / "training/mappo_checkpoint.pt", map_location="cpu", weights_only=True)
        expected_steps = config["arguments"]["updates"] * config["training"]["rollout_steps"]
        if (not isinstance(checkpoint, dict) or checkpoint.get("model_config") != config["model"]
                or checkpoint.get("config") != config["training"]
                or checkpoint.get("environment_steps") != expected_steps
                or not isinstance(checkpoint.get("optimizer"), dict)):
            raise ValueError("检查点与实际训练配置或采样预算不一致")
        model = SharedGraphActorCritic(**config["model"])
        model.load_state_dict(checkpoint["model"])
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError("检查点包含非有限模型参数")
    except Exception as exc:
        # 损坏或未写完的训练只能归档重跑，不能伪装成可复用的完成结果。
        return False, str(exc)
    return True, ""


def _evaluation_complete(root: Path, entry: dict) -> tuple[bool, str]:
    """复用汇总器的场景、种子、回合完整性检查，不以图片存在代替有效数据。"""

    from evaluation.ablation import _episodes, _scenarios

    run = _run_directory(root, entry)
    try:
        _episodes(run, _scenarios(run, entry))
    except Exception as exc:
        return False, str(exc)
    return True, ""


def _execute_stage(root: Path, entry: dict, stage: str) -> None:
    """归档失败阶段后执行当前项目生成的命令，成功返回仍需检查实际产物。"""

    label = "training" if stage == "train" else "evaluation"
    _archive(_run_directory(root, entry) / label, root)
    identity = dict(aggregation=entry["aggregation"], seed=entry["seed"], stage=label)
    _event(root, "stage_started", **identity)
    try:
        subprocess.run(entry[stage], cwd=PROJECT_ROOT, check=True,
                       env={**os.environ, "PYTHONUTF8": "1"})
        check = _training_complete if stage == "train" else _evaluation_complete
        complete, reason = check(root, entry)
        if not complete:
            raise ValueError(f"{label} 子进程已退出，但结果未通过校验: {reason}")
    except BaseException as exc:
        _event(root, "stage_failed", **identity, error=str(exc))
        raise
    _event(root, "stage_completed", **identity)


def _summary_complete(directory: Path, report: dict) -> bool:
    """只复用与当前数据一致且主要文件齐全的汇总，其余汇总原样归档后重建。"""

    from PIL import Image
    from evaluation.ablation import _csv_rows, _read_json

    try:
        if _read_json(directory / "ablation_summary.json") != report:
            return False
        with (directory / "ablation_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            saved_rows = list(csv.DictReader(stream))
        expected_rows = [{key: "" if value is None else str(value) for key, value in row.items()}
                         for row in _csv_rows(report["rows"])]
        if saved_rows != expected_rows:
            return False
        with Image.open(directory / "ablation_comparison.png") as image:
            if image.format != "PNG":
                return False
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


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
    parser.add_argument("--resume", action="store_true", help="沿用原配置续跑，校验并复用完整阶段")
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
    root = args.output_dir.resolve()
    summary_dir = root / "summary"
    try:
        with experiment_lock(root):
            if args.resume:
                _check_resume_plan(root, plan)
            else:
                if summary_dir.exists() or any(
                    (_run_directory(root, entry) / stage).exists()
                    for entry in plan for stage in ("training", "evaluation")
                ):
                    raise ValueError("已有实验输出，请显式使用 --resume 或更换 output-dir")
                (root / "ablation_plan.json").write_text(
                    json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"实验计划: {root / 'ablation_plan.json'}，共 {len(plan)} 个独立训练组", flush=True)
            if args.dry_run:
                if args.resume:
                    for entry in plan:
                        trained, _ = _training_complete(root, entry)
                        evaluated = trained and _evaluation_complete(root, entry)[0]
                        print(f"预览 {entry['aggregation']}/seed={entry['seed']}: "
                              f"训练={'跳过' if trained else '重跑'}，评估={'跳过' if evaluated else '重跑'}")
                return
            _event(root, "matrix_started", resume=args.resume, groups=len(plan))
            for entry in plan:
                print(f"开始 {entry['aggregation']} / seed={entry['seed']}", flush=True)
                trained = args.resume and _training_complete(root, entry)[0]
                if trained:
                    print("跳过训练：配置、日志及检查点已通过校验", flush=True)
                    _event(root, "stage_skipped", aggregation=entry["aggregation"],
                           seed=entry["seed"], stage="training")
                else:
                    # 新训练不能复用旧模型的评估，即使旧评估文件本身完整也必须归档。
                    _archive(_run_directory(root, entry) / "evaluation", root)
                    _execute_stage(root, entry, "train")
                evaluated = trained and _evaluation_complete(root, entry)[0]
                if evaluated:
                    print("跳过评估：场景、配对种子及回合数据已通过校验", flush=True)
                    _event(root, "stage_skipped", aggregation=entry["aggregation"],
                           seed=entry["seed"], stage="evaluation")
                else:
                    _execute_stage(root, entry, "evaluate")

            from evaluation.ablation import save_ablation_report, summarize_ablation

            report = summarize_ablation(root)
            if args.resume and _summary_complete(summary_dir, report):
                print(f"跳过汇总：已有结果与当前数据一致，目录 {summary_dir}", flush=True)
            else:
                _archive(summary_dir, root)
                paths = save_ablation_report(report, summary_dir)
                print(f"跨训练种子汇总表: {paths['csv']}", flush=True)
                print(f"消融对比图: {paths['figure']}", flush=True)
            for warning in report["warnings"]:
                print(f"统计提示: {warning}", flush=True)
            _event(root, "matrix_completed", groups=len(plan))
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"实验未完成: {exc}\n")


if __name__ == "__main__":
    main()
