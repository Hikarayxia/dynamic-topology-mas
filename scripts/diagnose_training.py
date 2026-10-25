"""从检查点与实际训练配置复现独立采样诊断，不训练或覆盖源模型。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import SharedGraphActorCritic
from environments import Continuous2DConfig, Continuous2DSearchEnv
from evaluation.ablation import _finite_float, _reject_constant
from interaction import CommunicationConfig
from scripts.train_mappo import build_environment
from training import MAPPOConfig, MAPPOTrainer
from training.diagnostics import diagnose_rollout


def run_diagnostic(checkpoint: Path, run_config: Path, *, seed: int, rollout_steps: int | None = None,
                   allow_motion_shift: bool = False) -> dict:
    """加载同一份源字节并采样，返回带来源指纹的报告；不写磁盘。

    参数为已保存的模型、训练配置、独立诊断种子和可选采样步数。环境沿用原
    训练入口的故障及通信设置；allow_motion_shift 显式允许并标记运动语义变化，
    此时不再声称复现原训练环境。不读取评估成绩选样本，不恢复优化器或续训。
    """

    if seed < 0 or (rollout_steps is not None and rollout_steps <= 0):
        raise ValueError("种子必须非负，诊断采样步数必须为正")
    checkpoint_bytes, config_bytes = checkpoint.read_bytes(), run_config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8-sig"), parse_constant=_reject_constant, parse_float=_finite_float)
    training_profile = config.get("motion_profile", "command_velocity_v1")
    current_profile = Continuous2DSearchEnv.MOTION_PROFILE
    if training_profile not in {"command_velocity_v1", current_profile}:
        raise ValueError("不支持的训练运动语义版本")
    motion_shift = training_profile != current_profile
    if motion_shift and not allow_motion_shift:
        raise ValueError("运动语义与原训练不同；显式 --allow-motion-shift 才可执行跨环境诊断")
    data = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    model = SharedGraphActorCritic(**data["model_config"])
    model.load_state_dict(data["model"])
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("检查点参数包含非有限数值")
    planned_model = {**config["model"]}
    planned_model.setdefault("aggregation", "mean")
    if planned_model.get("observation_profile") == "legacy":
        planned_model.pop("observation_profile")
    if model.architecture_config() != planned_model:
        raise ValueError("检查点模型与训练配置不一致")
    training = MAPPOConfig(**config["training"])
    if "config" in data:
        if asdict(MAPPOConfig(**data["config"])) != asdict(training):
            raise ValueError("检查点优化配置与训练记录不一致")
    elif data.get("environment_steps") != 0 or data.get("initialization_seed") != config["arguments"]["seed"]:
        raise ValueError("无优化配置的检查点必须是有来源标记的初始模型")
    arguments = config["arguments"]
    for flag in ("link_fault", "node_fault"):
        if type(arguments[flag]) is not bool:
            raise ValueError("故障开关必须为布尔值")
    if (type(arguments["agents"]) is not int or arguments["agents"] <= 0
            or type(arguments["max_steps"]) is not int or arguments["max_steps"] <= 0):
        raise ValueError("节点数与回合步数必须为正整数")
    communication = CommunicationConfig(**config["communication"])
    environment = build_environment(
        model, arguments["agents"], arguments["max_steps"], communication_config=communication,
        observation_mode=arguments["observation_mode"], neighbor_state_ttl=arguments["neighbor_state_ttl"],
        link_fault=arguments["link_fault"], node_fault=arguments["node_fault"],
    )
    expected_environment = Continuous2DConfig(**config["environment"])
    if asdict(environment.config) != asdict(expected_environment):
        raise ValueError("重建环境与记录配置不一致，不能将不同采样协议误标为同一次诊断")
    if arguments.get("coverage_memory", False) != environment.config.coverage_memory:
        raise ValueError("记忆开关与实际模型不一致")
    if rollout_steps is not None:
        training = replace(training, rollout_steps=rollout_steps)
    # 在重建模型之后播种，避免新旧输入维度消耗不同的初始化随机数。
    torch.manual_seed(seed)
    trainer = MAPPOTrainer(model, training, seed=seed)
    rollout, finished, successful = trainer.collect_rollout(environment)
    result = diagnose_rollout(trainer, rollout)
    return dict(
        schema_version=1, checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        motion_profile=current_profile, training_motion_profile=training_profile, motion_shift=motion_shift,
        run_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        checkpoint_environment_steps=data.get("environment_steps"),
        model_config=model.architecture_config(), diagnostic_seed=seed,
        diagnostic_training_config=asdict(training), environment=asdict(environment.config),
        communication=asdict(communication), link_fault=arguments["link_fault"], node_fault=arguments["node_fault"],
        episodes_finished=finished, successful_episodes=successful,
        torch_version=str(torch.__version__), numpy_version=np.__version__, diagnostics=result,
        warnings=(["运动语义已改变：本次为旧模型在新环境中的诊断，不是原训练环境的复现。"] if motion_shift else [])
        + ["诊断轨迹按随机策略采样，动作敏感性按确定性策略计算，均不替代独立任务评估。",
                  "共享梯度比较不含熵项；梯度大小和夹角是局部信号，不证明退化的唯一原因。",
                  "覆盖标志清零是输入消融，可能离开训练分布，不表示历史信息的任务收益。"],
    )


def main() -> None:
    """把诊断报告写入全新目录；输入不兼容或目标已存在时保留原始文件。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--allow-motion-shift", action="store_true", help="显式允许旧模型在新运动语义下诊断，并记录变化")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            raise ValueError("诊断输出目录已存在，请使用新目录")
        report = run_diagnostic(args.checkpoint, args.run_config, seed=args.seed, rollout_steps=args.rollout_steps,
                                allow_motion_shift=args.allow_motion_shift)
        # 先完成数值与来源检查，再创建目录，失败时不留下貌似完成的报告。
        content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        target = args.output_dir / "training_diagnostics.json"
        target.write_text(content, encoding="utf-8")
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(1, f"诊断未完成: {exc}\n")
    print(f"独立采样诊断完成: {target}")


if __name__ == "__main__":
    main()
