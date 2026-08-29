"""图分类的多空间门控相似度模块。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from src.baselines.gcl_baselines import GraphEncoder


VARIANTS = (
    "baseline",
    "semantic",
    "topology",
    "distribution",
    "uniform",
    "global",
    "gate",
    "boundary_gate",
    "full",
)


class MultiSpaceGateClassifier(nn.Module):
    """GIN骨架、三个互补相似度空间与样本对级边界感知门控。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        topology_dim: int,
        prototypes: int = 16,
        gate_hidden_dim: int = 32,
        assignment_temperature: float = 0.2,
    ):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim, layers)
        encoder_dim = self.encoder.output_dim
        self.semantic_projection = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.topology_projection = nn.Sequential(
            nn.Linear(topology_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_projection = nn.Linear(encoder_dim, hidden_dim)
        self.node_prototypes = nn.Parameter(torch.randn(prototypes, hidden_dim) / math.sqrt(hidden_dim))
        self.distribution_projection = nn.Sequential(
            nn.Linear(prototypes, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
        self.global_gate_logits = nn.Parameter(torch.zeros(3))
        self.pair_gate = nn.Sequential(
            nn.Linear(7, gate_hidden_dim), nn.ReLU(),
            nn.Linear(gate_hidden_dim, 3),
        )
        self.assignment_temperature = assignment_temperature

    def load_graphcl_encoder(self, state: dict[str, Tensor]) -> None:
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"GraphCL编码器参数不匹配：missing={missing}, unexpected={unexpected}"
            )

    def forward(self, data: Data) -> tuple[Tensor, dict[str, Tensor]]:
        graph, nodes = self.encoder(data)
        logits = self.classifier(graph)
        semantic = F.normalize(self.semantic_projection(graph), dim=-1)
        topology = F.normalize(self.topology_projection(data.topology), dim=-1)

        node_code = F.normalize(self.node_projection(nodes), dim=-1)
        prototypes = F.normalize(self.node_prototypes, dim=-1)
        assignment = (node_code @ prototypes.t() / self.assignment_temperature).softmax(dim=-1)
        histogram = global_mean_pool(assignment, data.batch)
        distribution = F.normalize(self.distribution_projection(histogram), dim=-1)
        return logits, {
            "graph": graph,
            "semantic": semantic,
            "topology": topology,
            "distribution": distribution,
        }

    @staticmethod
    def _similarities(representations: dict[str, Tensor]) -> Tensor:
        return MultiSpaceGateClassifier._cross_similarities(representations, representations)

    @staticmethod
    def _cross_similarities(
        query: dict[str, Tensor], bank: dict[str, Tensor]
    ) -> Tensor:
        return torch.stack([
            query["semantic"] @ bank["semantic"].t(),
            query["topology"] @ bank["topology"].t(),
            query["distribution"] @ bank["distribution"].t(),
        ], dim=-1).add(1).div(2).clamp(1e-6, 1 - 1e-6)

    @staticmethod
    def _uncertainty(logits: Tensor) -> Tensor:
        probabilities = logits.detach().softmax(dim=-1)
        uncertainty = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=-1)
        return uncertainty / math.log(probabilities.size(-1))

    def fused_similarity(
        self,
        logits: Tensor,
        representations: dict[str, Tensor],
        variant: str,
    ) -> tuple[Tensor, Tensor]:
        similarities = self._similarities(representations)
        if variant == "semantic":
            weights = similarities.new_tensor([1.0, 0.0, 0.0]).expand_as(similarities)
        elif variant == "topology":
            weights = similarities.new_tensor([0.0, 1.0, 0.0]).expand_as(similarities)
        elif variant == "distribution":
            weights = similarities.new_tensor([0.0, 0.0, 1.0]).expand_as(similarities)
        elif variant == "uniform":
            weights = similarities.new_full(similarities.shape, 1 / 3)
        elif variant == "global":
            weights = self.global_gate_logits.softmax(dim=0).expand_as(similarities)
        elif variant in {"gate", "boundary_gate", "full"}:
            semantic = representations["semantic"]
            difference = (semantic[:, None, :] - semantic[None, :, :]).abs().mean(dim=-1)
            product = (semantic[:, None, :] * semantic[None, :, :]).mean(dim=-1)
            uncertainty = self._uncertainty(logits)
            if variant == "gate":
                uncertainty = torch.zeros_like(uncertainty)
            uncertainty_mean = (uncertainty[:, None] + uncertainty[None, :]) / 2
            uncertainty_gap = (uncertainty[:, None] - uncertainty[None, :]).abs()
            gate_input = torch.cat([
                similarities,
                uncertainty_mean.unsqueeze(-1),
                uncertainty_gap.unsqueeze(-1),
                difference.unsqueeze(-1),
                product.unsqueeze(-1),
            ], dim=-1)
            weights = self.pair_gate(gate_input).softmax(dim=-1)
        else:
            raise ValueError(f"不支持的相似度变体：{variant}")
        return (weights * similarities).sum(dim=-1), weights

    def cross_fused_similarity(
        self,
        query_logits: Tensor,
        query: dict[str, Tensor],
        bank_logits: Tensor,
        bank: dict[str, Tensor],
        variant: str,
    ) -> tuple[Tensor, Tensor]:
        """计算查询图到仅含训练图的样本库之间的门控相似度。"""
        similarities = self._cross_similarities(query, bank)
        if variant == "semantic":
            weights = similarities.new_tensor([1.0, 0.0, 0.0]).expand_as(similarities)
        elif variant == "topology":
            weights = similarities.new_tensor([0.0, 1.0, 0.0]).expand_as(similarities)
        elif variant == "distribution":
            weights = similarities.new_tensor([0.0, 0.0, 1.0]).expand_as(similarities)
        elif variant == "uniform":
            weights = similarities.new_full(similarities.shape, 1 / 3)
        elif variant == "global":
            weights = self.global_gate_logits.softmax(dim=0).expand_as(similarities)
        elif variant in {"gate", "boundary_gate", "full"}:
            query_semantic = query["semantic"]
            bank_semantic = bank["semantic"]
            difference = (
                query_semantic[:, None, :] - bank_semantic[None, :, :]
            ).abs().mean(dim=-1)
            product = (
                query_semantic[:, None, :] * bank_semantic[None, :, :]
            ).mean(dim=-1)
            query_uncertainty = self._uncertainty(query_logits)
            bank_uncertainty = self._uncertainty(bank_logits)
            if variant == "gate":
                query_uncertainty = torch.zeros_like(query_uncertainty)
                bank_uncertainty = torch.zeros_like(bank_uncertainty)
            uncertainty_mean = (
                query_uncertainty[:, None] + bank_uncertainty[None, :]
            ) / 2
            uncertainty_gap = (
                query_uncertainty[:, None] - bank_uncertainty[None, :]
            ).abs()
            gate_input = torch.cat([
                similarities,
                uncertainty_mean.unsqueeze(-1),
                uncertainty_gap.unsqueeze(-1),
                difference.unsqueeze(-1),
                product.unsqueeze(-1),
            ], dim=-1)
            weights = self.pair_gate(gate_input).softmax(dim=-1)
        else:
            raise ValueError(f"不支持的相似度变体：{variant}")
        return (weights * similarities).sum(dim=-1), weights

    @staticmethod
    def _hard_pair_mask(labels: Tensor, semantic: Tensor, fraction: float) -> Tensor:
        count = labels.numel()
        upper = torch.triu(torch.ones(count, count, dtype=torch.bool, device=labels.device), diagonal=1)
        same = labels[:, None] == labels[None, :]
        mask = torch.zeros_like(upper)
        semantic = semantic.detach()
        positive_indices = (upper & same).nonzero(as_tuple=False)
        negative_indices = (upper & ~same).nonzero(as_tuple=False)
        if positive_indices.numel():
            values = semantic[positive_indices[:, 0], positive_indices[:, 1]]
            keep = max(1, math.ceil(values.numel() * fraction))
            selected = positive_indices[values.topk(keep, largest=False).indices]
            mask[selected[:, 0], selected[:, 1]] = True
        if negative_indices.numel():
            values = semantic[negative_indices[:, 0], negative_indices[:, 1]]
            keep = max(1, math.ceil(values.numel() * fraction))
            selected = negative_indices[values.topk(keep, largest=True).indices]
            mask[selected[:, 0], selected[:, 1]] = True
        return mask

    def training_loss(
        self,
        logits: Tensor,
        representations: dict[str, Tensor],
        labels: Tensor,
        variant: str,
        similarity_weight: float = 0.5,
        relation_weight: float = 0.2,
        gate_weight: float = 0.01,
        hard_fraction: float = 0.25,
        negative_margin: float = 0.5,
    ) -> tuple[Tensor, dict[str, float]]:
        classification_loss = F.cross_entropy(logits, labels)
        zero = classification_loss.new_zeros(())
        if variant == "baseline" or labels.numel() < 2:
            return classification_loss, {
                "classification": float(classification_loss.detach()),
                "similarity": 0.0,
                "relation": 0.0,
                "gate": 0.0,
            }

        fused, weights = self.fused_similarity(logits, representations, variant)
        same = labels[:, None] == labels[None, :]
        upper = torch.triu(torch.ones_like(same), diagonal=1)
        if variant == "full":
            pair_mask = self._hard_pair_mask(labels, self._similarities(representations)[..., 0], hard_fraction)
        else:
            pair_mask = upper
        targets = same[pair_mask].float()
        predictions = fused[pair_mask]
        positive_count = targets.sum().clamp_min(1)
        negative_count = (1 - targets).sum().clamp_min(1)
        pair_weights = targets * (0.5 / positive_count) + (1 - targets) * (0.5 / negative_count)
        similarity_loss = (
            F.binary_cross_entropy(predictions, targets, reduction="none") * pair_weights
        ).sum()

        semantic_distance = 1 - self._similarities(representations)[..., 0]
        positive_relation = (
            fused.detach() * semantic_distance * same.float()
        )[pair_mask]
        negative_relation = (
            (1 - fused.detach())
            * F.relu(negative_margin - semantic_distance).square()
            * (~same).float()
        )[pair_mask]
        relation_loss = positive_relation.sum() / positive_count + negative_relation.sum() / negative_count

        gate_loss = zero
        if variant in {"gate", "boundary_gate", "full"}:
            mean_weights = weights[pair_mask].mean(dim=0)
            gate_loss = (mean_weights - 1 / 3).square().sum()

        total = (
            classification_loss
            + similarity_weight * similarity_loss
            + relation_weight * relation_loss
            + gate_weight * gate_loss
        )
        return total, {
            "classification": float(classification_loss.detach()),
            "similarity": float(similarity_loss.detach()),
            "relation": float(relation_loss.detach()),
            "gate": float(gate_loss.detach()),
        }
