"""ReToGCL 的 A→B→C 链接式 Pipeline 实验变体。

本文件刻意不修改原 ``ModularTripleResearchIdeaBalanceGCL``。新变体让增强阶段
产生的扰动元信息进入软洁净估计，再把洁净表示与置信度送入拓扑条件投影。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from src.models.modular_triple_research_ideas import ModularTripleResearchIdeaBalanceGCL


class ReToGCLLinkedPipeline(ModularTripleResearchIdeaBalanceGCL):
    """保留原 ReToGCL 参数主干，新增可审计的增强→清洗→投影信息流。"""

    protocol = "retogcl_linked_augmentation_reliability_topology_v1"

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
    ) -> None:
        super().__init__(
            ("D7-I10", "D7-I01", "D10-I04"),
            in_dim,
            hidden_dim,
            layers,
            num_classes,
            idea_parameters,
        )
        parameters = {
            idea_id: values
            for idea_id, values in zip(self.idea_ids, idea_parameters, strict=True)
        }
        self.clean_parameters = parameters["D7-I01"]
        self.topology_parameters = parameters["D10-I04"]
        self.augmentation_parameters = parameters["D7-I10"]
        self.strata_count = int(round(self.topology_parameters["strata_count"]))
        self.strata_router = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, self.strata_count),
        )
        self.strata_embeddings = nn.Parameter(
            torch.randn(self.strata_count, self.output_dim) * 0.01
        )
        # 残差门使链接算子从保守状态开始训练，并为后续消融提供单独开关。
        self.link_logit = nn.Parameter(torch.tensor(-1.0))

    @staticmethod
    def _graph_mean(values: Tensor, batch: Tensor, graph_count: int) -> Tensor:
        return scatter(values, batch, dim=0, dim_size=graph_count, reduce="mean")

    def _augmentation_trace(self, original: Data, augmented: Data) -> Tensor:
        """计算每张图的属性、节点和边保留信息，不依赖标签。"""
        graph_count = int(original.batch.max().item()) + 1
        feature_scale = original.x.abs().mean(dim=-1).clamp_min(1e-6)
        relative_change = (
            (original.x - augmented.x).abs().mean(dim=-1) / feature_scale
        ).clamp(0.0, 1.0)
        feature_retention = 1.0 - self._graph_mean(
            relative_change, original.batch, graph_count
        )
        node_changed = ((original.x - augmented.x).abs().sum(dim=-1) > 1e-8).float()
        node_retention = 1.0 - self._graph_mean(
            node_changed, original.batch, graph_count
        )

        original_source = original.edge_index[0]
        augmented_source = augmented.edge_index[0]
        original_edges = scatter(
            torch.ones_like(original_source, dtype=original.x.dtype),
            original.batch[original_source], dim=0, dim_size=graph_count, reduce="sum",
        )
        augmented_edges = scatter(
            torch.ones_like(augmented_source, dtype=original.x.dtype),
            augmented.batch[augmented_source], dim=0, dim_size=graph_count, reduce="sum",
        )
        edge_retention = (augmented_edges / original_edges.clamp_min(1.0)).clamp(0.0, 1.0)
        overall_retention = (
            feature_retention + node_retention + edge_retention
        ) / 3.0
        severity = 1.0 - overall_retention
        return torch.stack(
            (feature_retention, node_retention, edge_retention, overall_retention, severity),
            dim=-1,
        )

    def prepare_views(self, batch: Data) -> tuple[Data, Data]:
        """复用原 ReToGCL 增强，并把 A 阶段信息附加到两个视图。"""
        first, second = super().prepare_views(batch)
        first.augmentation_trace = self._augmentation_trace(batch, first)
        second.augmentation_trace = self._augmentation_trace(batch, second)
        return first, second

    def _soft_clean(
        self,
        first_bundle: dict[str, Tensor],
        second_bundle: dict[str, Tensor],
        first_trace: Tensor,
        second_trace: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """B 阶段：联合增强元信息与表示证据产生样本×视图软洁净矩阵。"""
        first_views = first_bundle["views"]
        second_views = second_bundle["views"]
        evidence = self.second._view_evidence(first_views, second_views)
        agreement, stability, diversity = evidence.unbind(dim=-1)
        base_clean = (1.0 + stability) / (
            2.0 + stability + (1.0 - agreement).clamp_min(0.0)
        )

        mean_trace = 0.5 * (first_trace + second_trace)
        feature_prior = mean_trace[:, 0]
        node_prior = mean_trace[:, 1]
        edge_prior = mean_trace[:, 2]
        overall_prior = mean_trace[:, 3]
        topology_ratio = float(self.clean_parameters["topology_evidence_ratio"])
        topology_prior = (
            topology_ratio * edge_prior
            + (1.0 - topology_ratio) * 0.5 * (feature_prior + node_prior)
        )
        prior = torch.stack(
            (node_prior, edge_prior, topology_prior, overall_prior), dim=-1
        )
        prior_strength = float(self.clean_parameters["cleanliness_beta_prior_strength"])
        clean = (base_clean + prior_strength * prior) / (1.0 + prior_strength)
        clean = clean.clamp_min(float(self.clean_parameters["clean_probability_floor"]))
        weights = clean / clean.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        first_clean = (weights.unsqueeze(-1) * first_views).sum(dim=1)
        second_clean = (weights.unsqueeze(-1) * second_views).sum(dim=1)
        confidence = (weights * clean).sum(dim=-1)
        prior_weights = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        reliability_loss = (
            -(weights * agreement).sum(dim=-1).mean()
            + 0.1 * F.mse_loss(weights, prior_weights)
        )
        return first_clean, second_clean, weights, confidence, reliability_loss

    @staticmethod
    def _topology_features(
        bundle: dict[str, Tensor],
        trace: Tensor,
        clean_weights: Tensor,
        confidence: Tensor,
    ) -> Tensor:
        scales = bundle["scales"]
        cycle_view = bundle["views"][:, 2]
        scale_first = F.cosine_similarity(scales[:, 0], scales[:, 1])
        scale_second = F.cosine_similarity(scales[:, 1], scales[:, 2])
        scale_span = F.cosine_similarity(scales[:, 0], scales[:, 2])
        return torch.stack(
            (
                confidence,
                trace[:, 0],
                trace[:, 2],
                trace[:, 3],
                clean_weights[:, 2],
                cycle_view.mean(dim=-1),
                scale_first - scale_second,
                scale_span,
            ),
            dim=-1,
        )

    @staticmethod
    def _soft_centroid(representation: Tensor, assignment: Tensor) -> Tensor:
        centroids = assignment.t() @ representation
        centroids = centroids / assignment.sum(dim=0).unsqueeze(-1).clamp_min(1e-6)
        return assignment @ centroids

    def _topology_projection(
        self,
        first_clean: Tensor,
        second_clean: Tensor,
        first_bundle: dict[str, Tensor],
        second_bundle: dict[str, Tensor],
        first_trace: Tensor,
        second_trace: Tensor,
        clean_weights: Tensor,
        confidence: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """C 阶段：以 B 输出为输入执行软拓扑分层、桥接和残差投影。"""
        first_features = self._topology_features(
            first_bundle, first_trace, clean_weights, confidence
        )
        second_features = self._topology_features(
            second_bundle, second_trace, clean_weights, confidence
        )
        first_assignment = self.strata_router(first_features).softmax(dim=-1)
        second_assignment = self.strata_router(second_features).softmax(dim=-1)
        first_local = self._soft_centroid(first_clean, first_assignment)
        second_local = self._soft_centroid(second_clean, second_assignment)

        bridge_weight = float(self.topology_parameters["bridge_weight"])
        first_bridged = (1.0 - bridge_weight) * first_clean + bridge_weight * first_local
        second_bridged = (1.0 - bridge_weight) * second_clean + bridge_weight * second_local
        first_context = first_assignment @ self.strata_embeddings
        second_context = second_assignment @ self.strata_embeddings
        first_candidate = self.third.bottleneck(first_bridged) + first_context
        second_candidate = self.third.bottleneck(second_bridged) + second_context

        # 原 ReToGCL 的拓扑条件投影作为残差基座，新链路只学习其修正量。
        first_sign = first_bundle["views"][:, 2].mean(dim=-1, keepdim=True).sign()
        second_sign = second_bundle["views"][:, 2].mean(dim=-1, keepdim=True).sign()
        first_base = self.third.bottleneck(first_clean) * first_sign
        second_base = self.third.bottleneck(second_clean) * second_sign
        link_strength = torch.sigmoid(self.link_logit)
        gate = (link_strength * confidence).unsqueeze(-1)
        first_projected = first_base + gate * (first_candidate - first_base)
        second_projected = second_base + gate * (second_candidate - second_base)

        clean_distance = torch.cdist(
            F.normalize(first_clean, dim=-1), F.normalize(first_clean, dim=-1)
        )
        projected_distance = torch.cdist(
            F.normalize(first_projected, dim=-1),
            F.normalize(first_projected, dim=-1),
        )
        distance_loss = F.mse_loss(projected_distance, clean_distance.detach())
        assignment_loss = F.mse_loss(first_assignment, second_assignment)
        average_assignment = first_assignment.mean(dim=0)
        balance_loss = (
            average_assignment
            * (average_assignment.clamp_min(1e-8).log() + math.log(self.strata_count))
        ).sum()
        topology_loss = distance_loss + 0.1 * assignment_loss + 0.01 * balance_loss
        return (
            first_projected,
            second_projected,
            first_assignment,
            second_assignment,
            topology_loss,
        )

    def _project_hard_negatives(
        self,
        hard_input: Tensor,
        first_assignment: Tensor,
        confidence: Tensor,
        first_bundle: dict[str, Tensor],
    ) -> Tensor:
        context = first_assignment @ self.strata_embeddings
        candidate = self.third.bottleneck(hard_input) + context.unsqueeze(1)
        sign = first_bundle["views"][:, 2].mean(dim=-1, keepdim=True).sign().unsqueeze(1)
        base = self.third.bottleneck(hard_input) * sign
        gate = (torch.sigmoid(self.link_logit) * confidence).view(-1, 1, 1)
        return base + gate * (candidate - base)

    def idea_loss(self, first_data: Data, second_data: Data) -> tuple[Tensor, dict[str, float]]:
        first_history = self.encoder(first_data)
        second_history = self.encoder(second_data)
        first_bundle = self.second._bundle_from_history(first_data, first_history)
        second_bundle = self.second._bundle_from_history(second_data, second_history)
        first_topology_bundle = self.third._bundle_from_history(first_data, first_history)
        second_topology_bundle = self.third._bundle_from_history(second_data, second_history)

        first_clean, second_clean, weights, confidence, reliability_loss = self._soft_clean(
            first_bundle,
            second_bundle,
            first_data.augmentation_trace,
            second_data.augmentation_trace,
        )
        routing = self.first.semantic_router(
            first_clean, self.first.semantic_prototypes
        )
        second_default_projection = F.normalize(
            self.first.projector(second_clean), dim=-1
        )
        positive_input = self.first.positive_builder(
            first_clean, second_clean, second_default_projection, routing
        )

        anchor, positive, first_assignment, _, topology_loss = self._topology_projection(
            first_clean,
            positive_input,
            first_topology_bundle,
            second_topology_bundle,
            first_data.augmentation_trace,
            second_data.augmentation_trace,
            weights,
            confidence,
        )
        hard_input = self.first.negative_builder(first_clean, routing)
        hard = self._project_hard_negatives(
            hard_input, first_assignment, confidence, first_topology_bundle
        )
        objective = self.first.objective(
            F.normalize(anchor, dim=-1),
            F.normalize(positive, dim=-1),
            F.normalize(hard, dim=-1),
            self._temperature(),
        )

        reliability_gain = self.second._parameter_controls()[2]
        topology_gain = self.third._parameter_controls()[2]
        auxiliary = (
            self.second.auxiliary_weight * reliability_gain * reliability_loss
            + self.third.auxiliary_weight * topology_gain * topology_loss
        )
        total = objective + auxiliary
        mean_retention = 0.5 * (
            first_data.augmentation_trace[:, 3].mean()
            + second_data.augmentation_trace[:, 3].mean()
        )
        return total, {
            "module_objective": float(objective.detach()),
            "auxiliary_total": float(auxiliary.detach()),
            "reliability_loss": float(reliability_loss.detach()),
            "topology_loss": float(topology_loss.detach()),
            "mean_clean_confidence": float(confidence.mean().detach()),
            "mean_augmentation_retention": float(mean_retention.detach()),
            "effective_strata": float(
                torch.exp(
                    -(first_assignment.mean(0) * first_assignment.mean(0).clamp_min(1e-8).log()).sum()
                ).detach()
            ),
            "link_strength": float(torch.sigmoid(self.link_logit).detach()),
        }


def build_retogcl_linked_pipeline(
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
) -> ReToGCLLinkedPipeline:
    return ReToGCLLinkedPipeline(
        in_dim, hidden_dim, layers, num_classes, idea_parameters
    )
