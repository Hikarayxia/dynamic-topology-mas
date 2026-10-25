"""检查实际轨迹的输入和优化信号；不执行参数更新，不把诊断值当作任务成绩。"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from algorithms import SharedGraphActorCritic
from core import Observation
from .mappo import MAPPOTrainer, RolloutStep


def gradient_comparison(actor: Tensor, weighted_value: Tensor) -> dict[str, float | None]:
    """比较同一参数集合上的两条梯度，返回范数、比值和余弦相似度。

    参数是按相同顺序展平的梯度；价值梯度应已乘 value_coef。零 Actor 梯度使
    比值无定义，任一零梯度使夹角无定义，均返回 None，避免伪造为零或无穷。
    """

    if actor.shape != weighted_value.shape or actor.ndim != 1:
        raise ValueError("梯度必须是同形状的一维向量")
    if not torch.isfinite(actor).all() or not torch.isfinite(weighted_value).all():
        raise ValueError("梯度包含非有限数值")
    actor, weighted_value = actor.double(), weighted_value.double()
    actor_norm, value_norm = float(actor.norm()), float(weighted_value.norm())
    cosine = (float(torch.dot(actor, weighted_value) / (actor_norm * value_norm))
              if actor_norm > 0 and value_norm > 0 else None)
    return dict(actor_norm=actor_norm, weighted_value_norm=value_norm,
                value_actor_ratio=value_norm / actor_norm if actor_norm > 0 else None,
                cosine_similarity=max(-1.0, min(1.0, cosine)) if cosine is not None else None)


def coverage_diagnostics(
    model: SharedGraphActorCritic, observations: Sequence[Observation],
) -> dict | None:
    """统计本机记忆变化及动作敏感性；旧观测返回 None。

    参数是模型和真实采样观测。只对诊断副本清零 9 个覆盖标志，保留地图有效
    掩码、原任务特征和邻居输入，比较确定性动作。该输入消融可能离开训练分布，
    动作差值只表示敏感性，不证明这些记忆有益或具有独立因果贡献。
    """

    if model.observation_profile != "local_coverage_v1":
        return None
    if not observations:
        raise ValueError("覆盖诊断需要非空观测")
    features = np.stack([observation.task_features for observation in observations])
    history, mask = features[:, 2:11], features[:, 11:20]
    if features.shape[1] != 20 or not np.isin(features[:, 2:], [0, 1]).all():
        raise ValueError("覆盖诊断需要 local_coverage_v1 的二值特征")
    if np.any(history > mask):
        raise ValueError("地图外部不能标记为已覆盖")
    mask_groups: dict[tuple, set] = {}
    for values, valid in zip(history, mask):
        mask_groups.setdefault(tuple(valid), set()).add(tuple(values))
    counterfactual = []
    for observation in observations:
        changed = observation.task_features.copy()
        changed[2:11] = 0
        counterfactual.append(replace(observation, task_features=changed))
    with torch.no_grad():
        original = model.act(observations, deterministic=True)[0]
        altered = model.act(counterfactual, deterministic=True)[0]
        difference = (original - altered).abs()
    return dict(
        observation_count=len(observations),
        covered_valid_fraction=float(history.sum() / mask.sum()) if mask.sum() else None,
        fully_covered_patch_fraction=float(np.all(history == mask, axis=1).mean()),
        unique_history_patterns=len({tuple(row) for row in history}),
        mask_groups_with_history_variation=sum(len(values) > 1 for values in mask_groups.values()),
        history_zero_action_mean_abs_change=float(difference.mean()),
        history_zero_action_max_abs_change=float(difference.max()),
    )


def diagnose_rollout(trainer: MAPPOTrainer, rollout: Sequence[RolloutStep]) -> dict:
    """返回当前模型在已有轨迹上的更新前诊断，不修改模型、优化器或轨迹。

    复用训练器的 GAE、价值损失和第一轮 PPO 裁剪目标，分别计算 Actor 与加权
    Critic 对共享编码器的梯度。autograd.grad 不写入 parameter.grad，也不调用
    optimizer.step；熵梯度不计入两者比较。结果对应本批随机采样，不能外推为
    所有训练更新的梯度关系或泛化表现。
    """

    if not rollout:
        raise ValueError("诊断轨迹不能为空")
    advantages, returns = trainer._advantages(rollout)
    actor_terms, values, log_errors, ratio_errors, actions = [], [], [], [], []
    for index, step in enumerate(rollout):
        log_probs, _, value = trainer.model.evaluate_actions(step.observations, step.actions)
        delta = log_probs - step.log_probs.to(trainer.device)
        ratio = delta.exp()
        actor_terms.append(-torch.minimum(
            ratio * advantages[index],
            ratio.clamp(1 - trainer.config.clip_ratio, 1 + trainer.config.clip_ratio) * advantages[index],
        ))
        values.append(value.reshape(()))
        log_errors.append(delta.detach().abs())
        ratio_errors.append((ratio.detach() - 1).abs())
        with torch.no_grad():
            actions.append(trainer.model.act(step.observations, deterministic=True)[0])
    actor_loss = torch.cat(actor_terms).mean()
    value_loss = trainer.value_loss(torch.stack(values), returns)
    parameters = tuple(trainer.model.encoder.parameters())
    actor_gradients = torch.autograd.grad(actor_loss, parameters, retain_graph=True, allow_unused=True)
    value_gradients = torch.autograd.grad(trainer.config.value_coef * value_loss, parameters, allow_unused=True)
    # 无邻域消融中的消息编码器可能未参与运算，缺失梯度按该参数的零贡献处理。
    actor_vector = torch.cat([(gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
                              for gradient, parameter in zip(actor_gradients, parameters)])
    value_vector = torch.cat([(gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
                              for gradient, parameter in zip(value_gradients, parameters)])
    joint_actions = torch.cat(actions)
    spread = torch.cat([(action - action.mean(dim=0)).square().sum(dim=1) for action in actions])
    observations = [observation for step in rollout for observation in step.observations]
    return dict(
        rollout_steps=len(rollout), agent_observations=len(observations),
        sampled_team_reward_sum=sum(step.reward for step in rollout),
        max_preupdate_log_probability_error=float(torch.cat(log_errors).max()),
        max_preupdate_ratio_error=float(torch.cat(ratio_errors).max()),
        actor_loss=float(actor_loss.detach()), value_loss=float(value_loss.detach()),
        shared_gradients=gradient_comparison(actor_vector, value_vector),
        deterministic_mean_action_norm=float(joint_actions.norm(dim=1).mean()),
        deterministic_agent_action_spread_rms=float(spread.mean().sqrt()),
        deterministic_saturated_component_fraction=float((joint_actions.abs() >= 0.99).float().mean()),
        coverage=coverage_diagnostics(trainer.model, observations),
    )
