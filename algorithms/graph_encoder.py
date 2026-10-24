"""将数量可变的邻域观测转换为固定维度图表示。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from core.observation import Observation


@dataclass(frozen=True)
class GraphBatch:
    """经过填充的图观测批次。

    ``neighbor_mask`` 标出有效邻居快照，确保填充值不会参与聚合。通信观测中的
    快照可短暂保留到 TTL 到期，消息年龄独立于距离权重传入注意力网络。
    """

    self_features: Tensor
    neighbor_features: Tensor
    neighbor_weights: Tensor
    neighbor_mask: Tensor
    task_features: Tensor
    neighbor_ages: Tensor | None = None

    def to(self, device: torch.device | str) -> "GraphBatch":
        """返回移动到目标设备的新批次。"""

        return GraphBatch(*(
            value.to(device) if value is not None else None
            for value in self.__dict__.values()
        ))


def batch_observations(
    observations: Sequence[Observation],
    *,
    device: torch.device | str = "cpu",
) -> GraphBatch:
    """把可变邻居观测填充成张量批次。

    参数:
        observations: 同一任务定义下的一组智能体观测。
        device: 张量所在设备。

    返回值:
        可直接输入图编码器的 ``GraphBatch``。
    """

    if not observations:
        raise ValueError("observations 不能为空")
    self_dim = observations[0].self_features.size
    neighbor_dim = observations[0].neighbor_feature_dim
    task_dim = observations[0].task_features.size
    if any(obs.self_features.size != self_dim for obs in observations):
        raise ValueError("批次中的自身特征维度必须一致")
    if any(obs.neighbor_feature_dim != neighbor_dim for obs in observations):
        raise ValueError("批次中的邻居特征维度必须一致")
    if any(obs.task_features.size != task_dim for obs in observations):
        raise ValueError("批次中的任务特征维度必须一致")

    batch_size = len(observations)
    # 至少保留一个填充槽，统一处理所有智能体都暂时失联的情况。
    max_neighbors = max(1, max(len(obs.neighbors) for obs in observations))
    self_features = torch.zeros((batch_size, self_dim), dtype=torch.float32)
    neighbor_features = torch.zeros(
        (batch_size, max_neighbors, neighbor_dim), dtype=torch.float32
    )
    neighbor_weights = torch.zeros((batch_size, max_neighbors), dtype=torch.float32)
    neighbor_mask = torch.zeros((batch_size, max_neighbors), dtype=torch.bool)
    neighbor_ages = torch.zeros((batch_size, max_neighbors), dtype=torch.float32)
    task_features = torch.zeros((batch_size, task_dim), dtype=torch.float32)

    for row, observation in enumerate(observations):
        self_features[row] = torch.as_tensor(observation.self_features)
        task_features[row] = torch.as_tensor(observation.task_features)
        for column, neighbor in enumerate(observation.neighbors):
            neighbor_features[row, column] = torch.as_tensor(neighbor.features)
            neighbor_weights[row, column] = float(neighbor.link_weight)
            neighbor_mask[row, column] = True
            neighbor_ages[row, column] = float(neighbor.metadata.get("message_age", 0.0))

    return GraphBatch(
        self_features.to(device),
        neighbor_features.to(device),
        neighbor_weights.to(device),
        neighbor_mask.to(device),
        task_features.to(device),
        neighbor_ages.to(device),
    )


class GraphObservationEncoder(nn.Module):
    """以均值、通信质量注意力或无邻居输入完成可变规模编码。

    所有边复用同一个邻居编码器，均值与注意力均对排列不敏感。模型参数量不随
    邻居数或智能体总数变化；实际泛化效果仍需通过独立实验验证。
    """

    def __init__(
        self,
        self_dim: int,
        neighbor_dim: int,
        task_dim: int,
        hidden_dim: int = 64,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if min(self_dim, neighbor_dim, task_dim, hidden_dim) <= 0:
            raise ValueError("编码器维度必须为正整数")
        if aggregation not in {"mean", "attention", "none"}:
            raise ValueError("aggregation 必须为 mean、attention 或 none")
        self.aggregation = aggregation
        self.self_dim = self_dim
        self.neighbor_dim = neighbor_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim
        self.self_encoder = nn.Sequential(nn.Linear(self_dim, hidden_dim), nn.Tanh())
        self.message_encoder = nn.Sequential(
            nn.Linear(neighbor_dim, hidden_dim), nn.Tanh()
        )
        self.task_encoder = nn.Sequential(nn.Linear(task_dim, hidden_dim), nn.Tanh())
        if aggregation == "attention":
            # 质量输入来自本地已接收信息：距离权重和消息年龄，不读取真实丢包率。
            self.attention_scorer = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.Tanh()
        )

    def forward(self, batch: GraphBatch) -> Tensor:
        """返回形状为 ``[批量大小, hidden_dim]`` 的局部图表示。"""

        if batch.self_features.shape[-1] != self.self_dim:
            raise ValueError("自身特征维度与编码器配置不一致")
        if batch.neighbor_features.shape[-1] != self.neighbor_dim:
            raise ValueError("邻居特征维度与编码器配置不一致")
        if batch.task_features.shape[-1] != self.task_dim:
            raise ValueError("任务特征维度与编码器配置不一致")

        self_embedding = self.self_encoder(batch.self_features)
        messages = self.message_encoder(batch.neighbor_features)
        if self.aggregation == "none":
            neighbor_embedding = torch.zeros_like(self_embedding)
        else:
            effective_weights = self.aggregation_weights(batch, self_embedding, messages)
            neighbor_embedding = (messages * effective_weights.unsqueeze(-1)).sum(dim=1)
        task_embedding = self.task_encoder(batch.task_features)
        return self.output(
            torch.cat((self_embedding, neighbor_embedding, task_embedding), dim=-1)
        )

    def aggregation_weights(
        self, batch: GraphBatch, self_embedding: Tensor, messages: Tensor
    ) -> Tensor:
        """返回归一化邻居权重；全失联时全部为零且梯度保持有限。"""

        mask = batch.neighbor_mask.to(batch.neighbor_weights.dtype)
        if self.aggregation == "none":
            return torch.zeros_like(batch.neighbor_weights)
        if self.aggregation == "attention":
            ages = batch.neighbor_ages
            if ages is None:
                ages = torch.zeros_like(batch.neighbor_weights)
            own = self_embedding.unsqueeze(1).expand_as(messages)
            quality = torch.stack((batch.neighbor_weights, ages), dim=-1)
            logits = self.attention_scorer(torch.cat((own, messages, quality), dim=-1)).squeeze(-1)
            # 使用有限最小值，避免空邻域 softmax(-inf, ...) 产生 NaN。
            logits = logits.masked_fill(~batch.neighbor_mask, torch.finfo(logits.dtype).min)
            effective_weights = torch.softmax(logits, dim=1) * mask
        else:
            effective_weights = batch.neighbor_weights * mask
        weight_sum = effective_weights.sum(dim=1, keepdim=True)
        return effective_weights / weight_sum.clamp_min(1e-8)
