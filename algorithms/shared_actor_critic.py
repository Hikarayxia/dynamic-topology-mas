"""参数共享的连续动作 Actor 与集中式 Critic。"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from core.observation import Observation

from .graph_encoder import GraphObservationEncoder, batch_observations


class SharedGraphActorCritic(nn.Module):
    """面向动态规模智能体群的共享 Actor-Critic。

    Actor 对每个智能体的局部图表示独立输出动作分布；Critic 对当时全部在线
    智能体表示取均值，形成集中式全局状态。均值池化不会绑定智能体顺序和数量，
    同一个模型可在不同规模和拓扑上复用。
    """

    def __init__(
        self,
        self_dim: int,
        neighbor_dim: int,
        task_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 64,
        aggregation: str = "mean",
        observation_profile: str = "legacy",
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim 必须为正整数")
        if observation_profile not in {"legacy", "local_coverage_v1"}:
            raise ValueError("不支持的 observation_profile")
        if observation_profile == "local_coverage_v1" and (self_dim, neighbor_dim, task_dim) != (6, 5, 20):
            raise ValueError("local_coverage_v1 要求自身/邻居/任务维度为 6/5/20")
        self.observation_profile = observation_profile
        self.action_dim = action_dim
        self.encoder = GraphObservationEncoder(
            self_dim, neighbor_dim, task_dim, hidden_dim, aggregation
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim)
        )
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    @property
    def device(self) -> torch.device:
        """返回模型参数所在设备。"""

        return next(self.parameters()).device

    def architecture_config(self) -> dict[str, int | str]:
        """返回重建模型所需维度，随检查点保存以避免评估配置错配。"""

        config = {
            "self_dim": self.encoder.self_dim,
            "neighbor_dim": self.encoder.neighbor_dim,
            "task_dim": self.encoder.task_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.encoder.hidden_dim,
            "aggregation": self.encoder.aggregation,
        }
        # 旧模型保持原配置字典不变；只有启用新输入的检查点才显式标记版本。
        if self.observation_profile != "legacy":
            config["observation_profile"] = self.observation_profile
        return config

    def encode(self, observations: Sequence[Observation]) -> Tensor:
        """编码一个时间步内所有在线智能体的观测。"""

        return self.encoder(batch_observations(observations, device=self.device))

    def _distribution(self, embeddings: Tensor) -> Normal:
        mean = self.actor_mean(embeddings)
        std = self.actor_log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _squashed_log_prob(distribution: Normal, raw_actions: Tensor) -> Tensor:
        bounded = torch.tanh(raw_actions)
        correction = torch.log(1.0 - bounded.square() + 1e-6)
        return (distribution.log_prob(raw_actions) - correction).sum(dim=-1)

    def act(
        self,
        observations: Sequence[Observation],
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """生成有界动作、动作对数概率和集中式状态价值。"""

        embeddings = self.encode(observations)
        distribution = self._distribution(embeddings)
        raw_actions = distribution.mean if deterministic else distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_probs = self._squashed_log_prob(distribution, raw_actions)
        value = self.critic(embeddings.mean(dim=0, keepdim=True)).squeeze()
        return actions, log_probs, value

    def evaluate_actions(
        self, observations: Sequence[Observation], actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """重新计算旧动作概率、近似熵和集中式价值，供 PPO 更新使用。"""

        embeddings = self.encode(observations)
        distribution = self._distribution(embeddings)
        bounded = actions.to(self.device).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_actions = torch.atanh(bounded)
        log_probs = self._squashed_log_prob(distribution, raw_actions)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.critic(embeddings.mean(dim=0, keepdim=True)).squeeze()
        return log_probs, entropy, value

    def value(self, observations: Sequence[Observation]) -> Tensor:
        """只计算集中式价值，用于 GAE 末端自举。"""

        embeddings = self.encode(observations)
        return self.critic(embeddings.mean(dim=0, keepdim=True)).squeeze()
