"""ReToGCL 的十套轻量 A→B→C Pipeline 策略。

所有策略以原 ReToGCL 分支为残差基座，仅改变跨阶段信息连接方式；原模型和
``ReToGCLLinkedPipeline`` 强连接实验均不在本文件中修改。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from src.models.modular_triple_research_ideas import ModularTripleResearchIdeaBalanceGCL


@dataclass(frozen=True)
class PipelineStrategy:
    strategy_id: str
    name: str
    description: str
    gate_mode: str
    gate_bias: float
    clean_residual: float = 0.25
    auxiliary_weight: float = 0.01
    detach_gate: bool = False
    topology_selective: bool = False
    sparse_strata: bool = False
    adaptive_bridge: bool = False
    warmup_epochs: int = 0


PIPELINE_STRATEGIES: dict[str, PipelineStrategy] = {
    "P01": PipelineStrategy(
        "P01", "微残差传递", "增强轨迹进入洁净先验，再以5%残差送入拓扑投影。",
        "fixed", -3.0, clean_residual=0.20, auxiliary_weight=0.0,
    ),
    "P02": PipelineStrategy(
        "P02", "置信残差门", "洁净置信度连续控制链接表示对原路径的修正。",
        "confidence", -2.2, clean_residual=0.25,
    ),
    "P03": PipelineStrategy(
        "P03", "双路径学习融合", "原ReToGCL路径与链接路径并行，由样本级门网络融合。",
        "learned", -1.5, clean_residual=0.30,
    ),
    "P04": PipelineStrategy(
        "P04", "不确定性旁路", "低置信样本绕过链接路径，高置信样本才进入后续投影。",
        "uncertainty_bypass", -1.2, clean_residual=0.30,
    ),
    "P05": PipelineStrategy(
        "P05", "停止梯度可靠门", "可靠性仅作为控制信号，阻断下游目标反向操纵洁净门。",
        "confidence", -1.8, clean_residual=0.25, detach_gate=True,
    ),
    "P06": PipelineStrategy(
        "P06", "拓扑选择门", "只让环视图洁净证据强的样本产生跨阶段修正。",
        "topology", -1.5, clean_residual=0.30, topology_selective=True,
    ),
    "P07": PipelineStrategy(
        "P07", "边稳定联合门", "增强后的边保留率与洁净置信度共同控制投影修正。",
        "edge_stability", -1.5, clean_residual=0.30,
    ),
    "P08": PipelineStrategy(
        "P08", "课程式连接", "前十轮保持原路径，之后逐渐开放A→B→C连接。",
        "curriculum", -1.2, clean_residual=0.30, warmup_epochs=10,
    ),
    "P09": PipelineStrategy(
        "P09", "稀疏拓扑专家残差", "选择每个样本权重最高的两个拓扑层作为小残差专家。",
        "confidence", -1.8, clean_residual=0.25, sparse_strata=True,
    ),
    "P10": PipelineStrategy(
        "P10", "一致性自适应桥接", "仅对跨视图一致且可靠的样本进行弱局部质心桥接。",
        "agreement", -1.8, clean_residual=0.25, adaptive_bridge=True,
    ),
}


class ReToGCLPipelineStrategy(ModularTripleResearchIdeaBalanceGCL):
    """以原 ReToGCL 为主路径、以策略化链接为受控残差的统一模型。"""

    protocol = "retogcl_ten_lightweight_pipeline_strategies_v1"

    def __init__(
        self,
        strategy_id: str,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
    ) -> None:
        if strategy_id not in PIPELINE_STRATEGIES:
            raise ValueError(f"未知 Pipeline 策略：{strategy_id}")
        super().__init__(
            ("D7-I10", "D7-I01", "D10-I04"),
            in_dim, hidden_dim, layers, num_classes, idea_parameters,
        )
        self.strategy = PIPELINE_STRATEGIES[strategy_id]
        parameters = {
            idea_id: values
            for idea_id, values in zip(self.idea_ids, idea_parameters, strict=True)
        }
        self.clean_parameters = parameters["D7-I01"]
        self.topology_parameters = parameters["D10-I04"]
        self.strata_count = int(round(self.topology_parameters["strata_count"]))

        # 新参数的初始化不能改变原模型之后用于随机增强的全局随机数序列。
        cpu_rng_state = torch.random.get_rng_state()
        self.link_logit = nn.Parameter(torch.tensor(self.strategy.gate_bias))
        self.gate_network = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.strata_router = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, self.strata_count)
        )
        self.strata_embeddings = nn.Parameter(
            torch.randn(self.strata_count, self.output_dim) * 0.005
        )
        with torch.no_grad():
            nn.init.zeros_(self.gate_network[-1].weight)
            self.gate_network[-1].bias.fill_(self.strategy.gate_bias)
        torch.random.set_rng_state(cpu_rng_state)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @staticmethod
    def _graph_mean(values: Tensor, batch: Tensor, graph_count: int) -> Tensor:
        return scatter(values, batch, dim=0, dim_size=graph_count, reduce="mean")

    def _augmentation_trace(self, original: Data, augmented: Data) -> Tensor:
        graph_count = int(original.batch.max().item()) + 1
        feature_scale = original.x.abs().mean(-1).clamp_min(1e-6)
        change = ((original.x - augmented.x).abs().mean(-1) / feature_scale).clamp(0, 1)
        feature_retention = 1 - self._graph_mean(change, original.batch, graph_count)
        node_change = ((original.x - augmented.x).abs().sum(-1) > 1e-8).float()
        node_retention = 1 - self._graph_mean(node_change, original.batch, graph_count)
        original_source, augmented_source = original.edge_index[0], augmented.edge_index[0]
        original_edges = scatter(
            torch.ones_like(original_source, dtype=original.x.dtype),
            original.batch[original_source], dim=0, dim_size=graph_count, reduce="sum",
        )
        augmented_edges = scatter(
            torch.ones_like(augmented_source, dtype=original.x.dtype),
            augmented.batch[augmented_source], dim=0, dim_size=graph_count, reduce="sum",
        )
        edge_retention = (augmented_edges / original_edges.clamp_min(1)).clamp(0, 1)
        overall = (feature_retention + node_retention + edge_retention) / 3
        return torch.stack(
            (feature_retention, node_retention, edge_retention, overall, 1 - overall), -1
        )

    def prepare_views(self, batch: Data) -> tuple[Data, Data]:
        first, second = super().prepare_views(batch)
        first.augmentation_trace = self._augmentation_trace(batch, first)
        second.augmentation_trace = self._augmentation_trace(batch, second)
        return first, second

    def _trace_conditioned_clean(
        self,
        first_bundle: dict[str, Tensor],
        second_bundle: dict[str, Tensor],
        first_trace: Tensor,
        second_trace: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        first_views, second_views = first_bundle["views"], second_bundle["views"]
        agreement, stability, _ = self.second._view_evidence(
            first_views, second_views
        ).unbind(-1)
        base_clean = (1 + stability) / (
            2 + stability + (1 - agreement).clamp_min(0)
        )
        trace = 0.5 * (first_trace + second_trace)
        topology_ratio = float(self.clean_parameters["topology_evidence_ratio"])
        topology_prior = (
            topology_ratio * trace[:, 2]
            + (1 - topology_ratio) * 0.5 * (trace[:, 0] + trace[:, 1])
        )
        prior = torch.stack(
            (trace[:, 1], trace[:, 2], topology_prior, trace[:, 3]), -1
        )
        strength = float(self.clean_parameters["cleanliness_beta_prior_strength"])
        clean = ((base_clean + strength * prior) / (1 + strength)).clamp_min(
            float(self.clean_parameters["clean_probability_floor"])
        )
        weights = clean / clean.sum(-1, keepdim=True).clamp_min(1e-6)
        weighted_first = (weights.unsqueeze(-1) * first_views).sum(1)
        weighted_second = (weights.unsqueeze(-1) * second_views).sum(1)
        residual = self.strategy.clean_residual
        first_clean = first_bundle["graph"] + residual * (
            weighted_first - first_bundle["graph"]
        )
        second_clean = second_bundle["graph"] + residual * (
            weighted_second - second_bundle["graph"]
        )
        confidence = (weights * clean).sum(-1)
        auxiliary = -(weights * agreement).sum(-1).mean()
        return first_clean, second_clean, weights, confidence, agreement, auxiliary

    @staticmethod
    def _strategy_features(
        topology_bundle: dict[str, Tensor],
        trace: Tensor,
        weights: Tensor,
        confidence: Tensor,
        agreement: Tensor,
    ) -> Tensor:
        scales = topology_bundle["scales"]
        return torch.stack(
            (
                confidence,
                trace[:, 0], trace[:, 2], trace[:, 3],
                weights[:, 2], agreement.mean(-1),
                F.cosine_similarity(scales[:, 0], scales[:, 1]),
                F.cosine_similarity(scales[:, 0], scales[:, 2]),
            ), -1,
        )

    @staticmethod
    def _replace_graph(bundle: dict[str, Tensor], graph: Tensor) -> dict[str, Tensor]:
        return {**bundle, "graph": graph}

    def _base_path(
        self,
        first_graph: Tensor,
        second_graph: Tensor,
        first_clean_bundle: dict[str, Tensor],
        second_clean_bundle: dict[str, Tensor],
        first_topology_bundle: dict[str, Tensor],
        second_topology_bundle: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        routed_first, _, clean_auxiliary = self.second._module_transform(
            first_clean_bundle, second_clean_bundle
        )
        routing = self.first.semantic_router(
            routed_first, self.first.semantic_prototypes
        )
        hard = self.first.negative_builder(first_graph, routing)
        anchor, positive, topology_auxiliary = self.third._module_transform(
            self._replace_graph(first_topology_bundle, first_graph),
            self._replace_graph(second_topology_bundle, second_graph),
        )
        auxiliary = (
            self.second.auxiliary_weight
            * self.second._parameter_controls()[2]
            * clean_auxiliary
            + self.third.auxiliary_weight
            * self.third._parameter_controls()[2]
            * topology_auxiliary
        )
        return anchor, positive, hard, auxiliary

    def _linked_path(
        self,
        first_clean: Tensor,
        second_clean: Tensor,
        weights: Tensor,
        confidence: Tensor,
        agreement: Tensor,
        first_trace: Tensor,
        second_trace: Tensor,
        first_topology_bundle: dict[str, Tensor],
        second_topology_bundle: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.strategy.topology_selective:
            topology_gate = weights[:, 2:3]
            first_clean = first_topology_bundle["graph"] + topology_gate * (
                first_clean - first_topology_bundle["graph"]
            )
            second_clean = second_topology_bundle["graph"] + topology_gate * (
                second_clean - second_topology_bundle["graph"]
            )

        first_features = self._strategy_features(
            first_topology_bundle, first_trace, weights, confidence, agreement
        )
        second_features = self._strategy_features(
            second_topology_bundle, second_trace, weights, confidence, agreement
        )
        first_assignment = self.strata_router(first_features).softmax(-1)
        second_assignment = self.strata_router(second_features).softmax(-1)

        if self.strategy.sparse_strata:
            first_top = first_assignment.topk(min(2, self.strata_count), dim=-1).indices
            second_top = second_assignment.topk(min(2, self.strata_count), dim=-1).indices
            first_mask = torch.zeros_like(first_assignment).scatter_(1, first_top, 1.0)
            second_mask = torch.zeros_like(second_assignment).scatter_(1, second_top, 1.0)
            first_assignment = first_assignment * first_mask
            second_assignment = second_assignment * second_mask
            first_assignment = first_assignment / first_assignment.sum(-1, keepdim=True).clamp_min(1e-6)
            second_assignment = second_assignment / second_assignment.sum(-1, keepdim=True).clamp_min(1e-6)

        if self.strategy.adaptive_bridge:
            first_centroids = first_assignment.t() @ first_clean
            second_centroids = second_assignment.t() @ second_clean
            first_centroids /= first_assignment.sum(0).unsqueeze(-1).clamp_min(1e-6)
            second_centroids /= second_assignment.sum(0).unsqueeze(-1).clamp_min(1e-6)
            agreement_gate = ((agreement.mean(-1) + 1) / 2).clamp(0, 1)
            bridge = (0.1 * confidence * agreement_gate).unsqueeze(-1)
            first_clean = first_clean + bridge * (
                first_assignment @ first_centroids - first_clean
            )
            second_clean = second_clean + bridge * (
                second_assignment @ second_centroids - second_clean
            )

        routing = self.first.semantic_router(
            first_clean, self.first.semantic_prototypes
        )
        second_default = F.normalize(self.first.projector(second_clean), dim=-1)
        positive_input = self.first.positive_builder(
            first_clean, second_clean, second_default, routing
        )
        hard = self.first.negative_builder(first_clean, routing)
        anchor, positive, topology_auxiliary = self.third._module_transform(
            self._replace_graph(first_topology_bundle, first_clean),
            self._replace_graph(second_topology_bundle, positive_input),
        )
        if self.strategy.sparse_strata:
            anchor = anchor + 0.05 * (first_assignment @ self.strata_embeddings)
            positive = positive + 0.05 * (second_assignment @ self.strata_embeddings)
        assignment_loss = F.mse_loss(first_assignment, second_assignment)
        linked_auxiliary = topology_auxiliary + 0.1 * assignment_loss
        return anchor, positive, hard, linked_auxiliary, first_features

    def _mix_weight(
        self,
        features: Tensor,
        confidence: Tensor,
        weights: Tensor,
        agreement: Tensor,
        first_trace: Tensor,
        second_trace: Tensor,
    ) -> Tensor:
        mode = self.strategy.gate_mode
        base_gate = torch.sigmoid(self.link_logit)
        if mode == "fixed":
            mix = 0.05 * 0.5 * (first_trace[:, 3] + second_trace[:, 3])
        elif mode == "confidence":
            mix = base_gate * confidence
        elif mode == "learned":
            mix = torch.sigmoid(self.gate_network(features)).squeeze(-1)
        elif mode == "uncertainty_bypass":
            mix = base_gate * torch.sigmoid((confidence - 0.75) / 0.05)
        elif mode == "topology":
            mix = base_gate * confidence * weights[:, 2]
        elif mode == "edge_stability":
            edge = (first_trace[:, 2] * second_trace[:, 2]).clamp_min(0).sqrt()
            mix = base_gate * confidence * edge
        elif mode == "curriculum":
            schedule = max(0.0, min(1.0, (self.current_epoch - self.strategy.warmup_epochs) / 20.0))
            mix = base_gate * confidence * schedule
        elif mode == "agreement":
            agreement_gate = ((agreement.mean(-1) + 1) / 2).clamp(0, 1)
            mix = base_gate * confidence * agreement_gate
        else:
            raise RuntimeError(mode)
        return mix.detach() if self.strategy.detach_gate else mix

    def idea_loss(self, first_data: Data, second_data: Data) -> tuple[Tensor, dict[str, float]]:
        first_history = self.encoder(first_data)
        second_history = self.encoder(second_data)
        first_graph_bundle = self.first._bundle_from_history(first_data, first_history)
        second_graph_bundle = self.first._bundle_from_history(second_data, second_history)
        first_clean_bundle = self.second._bundle_from_history(first_data, first_history)
        second_clean_bundle = self.second._bundle_from_history(second_data, second_history)
        first_topology_bundle = self.third._bundle_from_history(first_data, first_history)
        second_topology_bundle = self.third._bundle_from_history(second_data, second_history)
        first_graph, second_graph = first_graph_bundle["graph"], second_graph_bundle["graph"]

        base_anchor, base_positive, base_hard, base_auxiliary = self._base_path(
            first_graph, second_graph,
            first_clean_bundle, second_clean_bundle,
            first_topology_bundle, second_topology_bundle,
        )
        first_clean, second_clean, weights, confidence, agreement, clean_auxiliary = (
            self._trace_conditioned_clean(
                first_clean_bundle, second_clean_bundle,
                first_data.augmentation_trace, second_data.augmentation_trace,
            )
        )
        linked_anchor, linked_positive, linked_hard, linked_auxiliary, features = self._linked_path(
            first_clean, second_clean, weights, confidence, agreement,
            first_data.augmentation_trace, second_data.augmentation_trace,
            first_topology_bundle, second_topology_bundle,
        )
        mix = self._mix_weight(
            features, confidence, weights, agreement,
            first_data.augmentation_trace, second_data.augmentation_trace,
        ).clamp(0, 1)
        anchor = base_anchor + mix.unsqueeze(-1) * (linked_anchor - base_anchor)
        positive = base_positive + mix.unsqueeze(-1) * (linked_positive - base_positive)
        hard = base_hard + mix.view(-1, 1, 1) * (linked_hard - base_hard)
        objective = self.first.objective(
            F.normalize(anchor, dim=-1), F.normalize(positive, dim=-1),
            F.normalize(hard, dim=-1), self._temperature(),
        )
        extra_auxiliary = self.strategy.auxiliary_weight * (
            linked_auxiliary + 0.1 * clean_auxiliary
        )
        total = objective + base_auxiliary + extra_auxiliary
        return total, {
            "module_objective": float(objective.detach()),
            "base_auxiliary": float(base_auxiliary.detach()),
            "linked_auxiliary": float(extra_auxiliary.detach()),
            "mean_mix": float(mix.mean().detach()),
            "mean_clean_confidence": float(confidence.mean().detach()),
            "mean_augmentation_retention": float(
                0.5 * (
                    first_data.augmentation_trace[:, 3].mean()
                    + second_data.augmentation_trace[:, 3].mean()
                ).detach()
            ),
        }


def build_retogcl_pipeline_strategy(
    strategy_id: str,
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
) -> ReToGCLPipelineStrategy:
    return ReToGCLPipelineStrategy(
        strategy_id, in_dim, hidden_dim, layers, num_classes, idea_parameters
    )
