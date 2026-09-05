"""围绕 T077 的可拒绝、自适应 Pipeline 实验模型。

本文件不修改 T077。所有候选均以 T077 的计算图为残差基座，只在一个或两个
明确接口上改变信息连接；强度为零时应严格退化为 T077。模型同时暴露连接门、
表示修正量和跨模块证据，供逐轮实验诊断使用。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from src.models.retogcl_t077_micro_tuning import (
    T077MicroConfiguration,
    ReToGCLT077MicroTuning,
    _clip_correction,
    _orthogonal_alternative,
    _spherical_alternative,
    _tanh_correction,
)


ROUTE_ACTIONS = (
    "route_delete_clean",
    "route_view_consensus",
    "route_residual_clip",
    "route_residual_tanh",
    "route_spherical",
    "route_orthogonal",
)
TOPOLOGY_INPUT_ACTIONS = (
    "topology_positive_semantic",
    "topology_anchor_clean",
    "topology_second_clean",
    "topology_both_clean",
    "topology_clean_semantic",
    "topology_view_consensus",
)
TOPOLOGY_OUTPUT_ACTIONS = (
    "topology_anchor_bypass",
    "topology_positive_bypass",
    "topology_both_bypass",
    "topology_output_clip",
    "topology_output_tanh",
    "topology_output_orthogonal",
)
HARD_SOURCE_ACTIONS = (
    "hard_anchor_clean",
    "hard_route_raw",
    "hard_clean_raw_route",
    "hard_anchor_consensus",
)
HARD_PROJECTION_ACTIONS = (
    "hard_projection_raw",
    "hard_projection_unit_gain",
    "hard_projection_clean_gate",
    "hard_projection_topology_gate",
    "hard_projection_agreement_gate",
    "hard_projection_joint_gate",
    "hard_projection_clip",
    "hard_projection_orthogonal",
)
CLEAN_AUXILIARY_ACTIONS = (
    "clean_auxiliary_delete",
    "clean_auxiliary_detach",
)
TOPOLOGY_AUXILIARY_ACTIONS = (
    "topology_auxiliary_delete",
    "topology_auxiliary_detach",
)

ACTION_SLOT: dict[str, str] = {
    **{name: "route" for name in ROUTE_ACTIONS},
    **{name: "topology_input" for name in TOPOLOGY_INPUT_ACTIONS},
    **{name: "topology_output" for name in TOPOLOGY_OUTPUT_ACTIONS},
    **{name: "hard_source" for name in HARD_SOURCE_ACTIONS},
    **{name: "hard_projection" for name in HARD_PROJECTION_ACTIONS},
    **{name: "clean_auxiliary" for name in CLEAN_AUXILIARY_ACTIONS},
    **{name: "topology_auxiliary" for name in TOPOLOGY_AUXILIARY_ACTIONS},
}
ALL_ACTIONS = frozenset(("none", *ACTION_SLOT))
GATE_MODES = frozenset({
    "fixed",
    "clean",
    "dirty",
    "agreement",
    "retention",
    "topology_prior",
    "prototype_margin",
    "joint",
    "uncertainty",
    "learned_scalar",
    "learned_sample",
})
GRADIENT_POLICIES = frozenset({
    "full",
    "detach_gate",
    "detach_delta",
    "detach_both",
    "orthogonal_delta",
    "clip_delta",
})
SCHEDULES = frozenset({
    "constant",
    "warmup5",
    "warmup10",
    "warmup20",
    "decay",
    "late",
})


@dataclass(frozen=True)
class T077AdaptivePipelineConfiguration:
    """一个至多含两条跨模块连接的候选配置。"""

    primary_action: str
    primary_strength: float = 0.25
    primary_gate: str = "fixed"
    secondary_action: str = "none"
    secondary_strength: float = 0.0
    secondary_gate: str = "fixed"
    gradient_policy: str = "full"
    schedule: str = "constant"

    def __post_init__(self) -> None:
        if self.primary_action not in ALL_ACTIONS - {"none"}:
            raise ValueError(f"未知主连接：{self.primary_action}")
        if self.secondary_action not in ALL_ACTIONS:
            raise ValueError(f"未知次连接：{self.secondary_action}")
        if self.primary_gate not in GATE_MODES or self.secondary_gate not in GATE_MODES:
            raise ValueError("未知门控证据")
        if self.gradient_policy not in GRADIENT_POLICIES:
            raise ValueError(f"未知梯度策略：{self.gradient_policy}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"未知开放日程：{self.schedule}")
        if not 0 <= self.primary_strength <= 1.5:
            raise ValueError("primary_strength 必须位于 [0, 1.5]")
        if not 0 <= self.secondary_strength <= 1.5:
            raise ValueError("secondary_strength 必须位于 [0, 1.5]")
        if self.secondary_action == "none" and not math.isclose(
            self.secondary_strength, 0.0, abs_tol=1e-12
        ):
            raise ValueError("没有次连接时 secondary_strength 必须为 0")
        if self.secondary_action != "none":
            if math.isclose(self.secondary_strength, 0.0, abs_tol=1e-12):
                raise ValueError("启用次连接时 secondary_strength 必须大于 0")
            if ACTION_SLOT[self.primary_action] == ACTION_SLOT[self.secondary_action]:
                raise ValueError("两条连接不能修改同一个 Pipeline 接口")

    def actions(self) -> tuple[tuple[str, float, str], ...]:
        values = [(self.primary_action, self.primary_strength, self.primary_gate)]
        if self.secondary_action != "none":
            values.append((
                self.secondary_action,
                self.secondary_strength,
                self.secondary_gate,
            ))
        return tuple(values)


def _score(left: Tensor, right: Tensor) -> Tensor:
    return ((F.cosine_similarity(left, right, dim=-1) + 1.0) * 0.5).clamp(0, 1)


def _broadcast_gate(gate: Tensor, value: Tensor) -> Tensor:
    while gate.ndim < value.ndim:
        gate = gate.unsqueeze(-1)
    return gate


class ReToGCLT077AdaptivePipeline(ReToGCLT077MicroTuning):
    """T077 主路径加至多两条弱连接，并记录其训练证据。"""

    protocol = "retogcl_t077_adaptive_pipeline_v1"

    def __init__(
        self,
        configuration: T077AdaptivePipelineConfiguration,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
    ) -> None:
        self.pipeline_configuration = configuration
        super().__init__(
            T077MicroConfiguration(),
            in_dim,
            hidden_dim,
            layers,
            num_classes,
            idea_parameters,
        )
        # 新门的构造不能推进后续图增强所依赖的随机数状态。
        rng_state = torch.random.get_rng_state()
        self.pipeline_gate_logit = nn.Parameter(torch.tensor(-1.3862944))
        self.pipeline_gate_network = nn.Sequential(
            nn.Linear(8, 12),
            nn.Tanh(),
            nn.Linear(12, 1),
        )
        with torch.no_grad():
            nn.init.zeros_(self.pipeline_gate_network[-1].weight)
            self.pipeline_gate_network[-1].bias.fill_(-1.3862944)
        torch.random.set_rng_state(rng_state)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @staticmethod
    def _graph_mean(values: Tensor, batch: Tensor, graph_count: int) -> Tensor:
        return scatter(values, batch, dim=0, dim_size=graph_count, reduce="mean")

    def _augmentation_trace(self, original: Data, augmented: Data) -> Tensor:
        graph_count = int(original.batch.max().item()) + 1
        feature_scale = original.x.abs().mean(-1).clamp_min(1e-6)
        feature_change = (
            (original.x - augmented.x).abs().mean(-1) / feature_scale
        ).clamp(0, 1)
        feature_retention = 1 - self._graph_mean(
            feature_change, original.batch, graph_count
        )
        node_change = ((original.x - augmented.x).abs().sum(-1) > 1e-8).float()
        node_retention = 1 - self._graph_mean(
            node_change, original.batch, graph_count
        )
        original_source = original.edge_index[0]
        augmented_source = augmented.edge_index[0]
        original_edges = scatter(
            torch.ones_like(original_source, dtype=original.x.dtype),
            original.batch[original_source],
            dim=0,
            dim_size=graph_count,
            reduce="sum",
        )
        augmented_edges = scatter(
            torch.ones_like(augmented_source, dtype=original.x.dtype),
            augmented.batch[augmented_source],
            dim=0,
            dim_size=graph_count,
            reduce="sum",
        )
        edge_retention = (augmented_edges / original_edges.clamp_min(1)).clamp(0, 1)
        overall = (feature_retention + node_retention + edge_retention) / 3
        return torch.stack(
            (feature_retention, node_retention, edge_retention, overall), dim=-1
        )

    def prepare_views(self, batch: Data) -> tuple[Data, Data]:
        first, second = super().prepare_views(batch)
        first.augmentation_trace = self._augmentation_trace(batch, first)
        second.augmentation_trace = self._augmentation_trace(batch, second)
        return first, second

    def _schedule_scale(self) -> float:
        epoch = max(self.current_epoch, 1)
        mode = self.pipeline_configuration.schedule
        if mode == "constant":
            return 1.0
        if mode == "warmup5":
            return min(1.0, epoch / 5.0)
        if mode == "warmup10":
            return min(1.0, epoch / 10.0)
        if mode == "warmup20":
            return min(1.0, epoch / 20.0)
        if mode == "decay":
            return max(0.2, 1.0 - 0.8 * (epoch - 1) / 39.0)
        return 0.0 if epoch <= 10 else min(1.0, (epoch - 10) / 10.0)

    def _evidence_features(
        self,
        first_graph: Tensor,
        second_graph: Tensor,
        first_clean: Tensor,
        second_clean: Tensor,
        first_bundle: dict[str, Tensor],
        second_bundle: dict[str, Tensor],
        first_trace: Tensor,
        second_trace: Tensor,
    ) -> Tensor:
        clean = 0.5 * (
            _score(first_graph, first_clean) + _score(second_graph, second_clean)
        )
        agreement = _score(first_graph, second_graph)
        retention = 0.5 * (first_trace[:, 3] + second_trace[:, 3])
        topology_prior = 0.5 * (
            _score(first_bundle["scales"][:, 0], first_bundle["scales"][:, 2])
            + _score(second_bundle["scales"][:, 0], second_bundle["scales"][:, 2])
        )
        prototypes = F.normalize(self.first.semantic_prototypes, dim=-1)
        logits = F.normalize(first_clean, dim=-1) @ prototypes.t()
        if logits.size(1) > 1:
            top = logits.topk(2, dim=-1).values
            margin = torch.sigmoid(4 * (top[:, 0] - top[:, 1]))
        else:
            margin = torch.ones_like(clean)
        cycle_agreement = _score(
            first_bundle["views"][:, 2], second_bundle["views"][:, 2]
        )
        norm_ratio = first_clean.norm(dim=-1).clamp_min(1e-6) / first_graph.norm(
            dim=-1
        ).clamp_min(1e-6)
        magnitude_match = torch.exp(-norm_ratio.log().abs())
        return torch.stack((
            clean,
            agreement,
            retention,
            topology_prior,
            margin,
            cycle_agreement,
            1 - clean,
            magnitude_match,
        ), dim=-1)

    def _gate(self, mode: str, strength: float, features: Tensor) -> Tensor:
        if mode == "fixed":
            evidence = features.new_ones(features.size(0))
        elif mode == "clean":
            evidence = features[:, 0]
        elif mode == "dirty":
            evidence = features[:, 6]
        elif mode == "agreement":
            evidence = features[:, 1]
        elif mode == "retention":
            evidence = features[:, 2]
        elif mode == "topology_prior":
            evidence = features[:, 3]
        elif mode == "prototype_margin":
            evidence = features[:, 4]
        elif mode == "joint":
            evidence = (
                features[:, 0] * features[:, 1] * features[:, 3]
            ).clamp_min(1e-12).pow(1 / 3)
        elif mode == "uncertainty":
            evidence = 4 * features[:, 0] * (1 - features[:, 0])
        elif mode == "learned_scalar":
            evidence = torch.sigmoid(self.pipeline_gate_logit).expand(features.size(0))
        elif mode == "learned_sample":
            evidence = torch.sigmoid(self.pipeline_gate_network(features)).squeeze(-1)
        else:
            raise RuntimeError(mode)
        gate = (strength * self._schedule_scale() * evidence).clamp(0, 1)
        if self.pipeline_configuration.gradient_policy in {"detach_gate", "detach_both"}:
            gate = gate.detach()
        return gate

    def _link(self, base: Tensor, alternative: Tensor, gate: Tensor) -> Tensor:
        delta = alternative - base
        policy = self.pipeline_configuration.gradient_policy
        if policy in {"detach_delta", "detach_both"}:
            delta = delta.detach()
        elif policy == "orthogonal_delta":
            parallel = (delta * base).sum(-1, keepdim=True) / base.pow(2).sum(
                -1, keepdim=True
            ).clamp_min(1e-6)
            delta = delta - parallel * base
        elif policy == "clip_delta":
            base_norm = base.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            delta = delta * (0.25 * base_norm / delta_norm).clamp_max(1.0)
        return base + _broadcast_gate(gate, base) * delta

    @staticmethod
    def _actions_for_slot(
        actions: tuple[tuple[str, float, str], ...], slot: str
    ) -> tuple[tuple[str, float, str], ...]:
        return tuple(entry for entry in actions if ACTION_SLOT[entry[0]] == slot)

    @staticmethod
    def _record_correction(
        records: list[Tensor], base: Tensor, modified: Tensor
    ) -> None:
        denominator = base.norm(dim=-1).mean().detach().clamp_min(1e-6)
        records.append((modified - base).norm(dim=-1).mean().detach() / denominator)

    def idea_loss(self, first_data: Data, second_data: Data):
        self._clean_score = None
        self._topology_score = None
        actions = self.pipeline_configuration.actions()
        gates: list[Tensor] = []
        corrections: list[Tensor] = []

        first_history = self.encoder(first_data)
        second_history = self.encoder(second_data)
        bundles = {
            model.target_module: (
                model._bundle_from_history(first_data, first_history),
                model._bundle_from_history(second_data, second_history),
            )
            for model in self._idea_models
        }
        first_graph = self.first._bundle_from_history(
            first_data, first_history
        )["graph"]
        second_graph = self.first._bundle_from_history(
            second_data, second_history
        )["graph"]

        clean_model = self._by_module["M4"]
        first_clean, second_clean, clean_auxiliary = self._apply_idea_transform(
            clean_model,
            *bundles["M4"],
            first_graph,
            second_graph,
        )
        features = self._evidence_features(
            first_graph,
            second_graph,
            first_clean,
            second_clean,
            bundles["M7"][0],
            bundles["M7"][1],
            first_data.augmentation_trace,
            second_data.augmentation_trace,
        )

        routing_source = first_clean
        for action, strength, gate_mode in self._actions_for_slot(actions, "route"):
            alternatives = {
                "route_delete_clean": first_graph,
                "route_view_consensus": 0.5 * (first_clean + second_clean),
                "route_residual_clip": _clip_correction(first_graph, first_clean),
                "route_residual_tanh": _tanh_correction(first_graph, first_clean),
                "route_spherical": _spherical_alternative(first_graph, first_clean),
                "route_orthogonal": _orthogonal_alternative(first_graph, first_clean),
            }
            gate = self._gate(gate_mode, strength, features)
            modified = self._link(routing_source, alternatives[action], gate)
            self._record_correction(corrections, routing_source, modified)
            routing_source = modified
            gates.append(gate.detach())
        routing = self.first.semantic_router(
            routing_source, self.first.semantic_prototypes
        )
        raw_routing = None

        # T077 虽然构造了语义正样本，但 M7 默认仍接第二个原始图表示。保留这次
        # 调用可确保随机中性掩码与 T077 推进完全相同的随机数状态。
        projected_second = F.normalize(self.first.projector(second_graph), dim=-1)
        positive_input = self.first.positive_builder(
            first_graph, second_graph, projected_second, routing
        )

        hard_input = self.first.negative_builder(first_graph, routing)
        for action, strength, gate_mode in self._actions_for_slot(
            actions, "hard_source"
        ):
            if raw_routing is None:
                raw_routing = self.first.semantic_router(
                    first_graph, self.first.semantic_prototypes
                )
            if action == "hard_anchor_clean":
                alternative = self.first.negative_builder(first_clean, routing)
            elif action == "hard_route_raw":
                alternative = self.first.negative_builder(first_graph, raw_routing)
            elif action == "hard_clean_raw_route":
                alternative = self.first.negative_builder(first_clean, raw_routing)
            else:
                alternative = self.first.negative_builder(
                    0.5 * (first_clean + second_clean), routing
                )
            gate = self._gate(gate_mode, strength, features)
            modified = self._link(hard_input, alternative, gate)
            self._record_correction(corrections, hard_input, modified)
            hard_input = modified
            gates.append(gate.detach())

        topology_first, topology_second = first_graph, second_graph
        for action, strength, gate_mode in self._actions_for_slot(
            actions, "topology_input"
        ):
            alternatives = {
                "topology_positive_semantic": (first_graph, positive_input),
                "topology_anchor_clean": (first_clean, second_graph),
                "topology_second_clean": (first_graph, second_clean),
                "topology_both_clean": (first_clean, second_clean),
                "topology_clean_semantic": (first_clean, positive_input),
                "topology_view_consensus": (
                    0.5 * (first_graph + first_clean),
                    0.5 * (second_graph + second_clean),
                ),
            }
            gate = self._gate(gate_mode, strength, features)
            alternative_first, alternative_second = alternatives[action]
            modified_first = self._link(topology_first, alternative_first, gate)
            modified_second = self._link(topology_second, alternative_second, gate)
            self._record_correction(corrections, topology_first, modified_first)
            self._record_correction(corrections, topology_second, modified_second)
            topology_first, topology_second = modified_first, modified_second
            gates.append(gate.detach())

        topology_model = self._by_module["M7"]
        anchor, positive, topology_auxiliary = self._apply_idea_transform(
            topology_model,
            *bundles["M7"],
            topology_first,
            topology_second,
        )
        for action, strength, gate_mode in self._actions_for_slot(
            actions, "topology_output"
        ):
            if action == "topology_anchor_bypass":
                alternative_anchor, alternative_positive = topology_first, positive
            elif action == "topology_positive_bypass":
                alternative_anchor, alternative_positive = anchor, topology_second
            elif action == "topology_both_bypass":
                alternative_anchor, alternative_positive = topology_first, topology_second
            elif action == "topology_output_clip":
                alternative_anchor = _clip_correction(topology_first, anchor)
                alternative_positive = _clip_correction(topology_second, positive)
            elif action == "topology_output_tanh":
                alternative_anchor = _tanh_correction(topology_first, anchor)
                alternative_positive = _tanh_correction(topology_second, positive)
            else:
                alternative_anchor = _orthogonal_alternative(topology_first, anchor)
                alternative_positive = _orthogonal_alternative(topology_second, positive)
            gate = self._gate(gate_mode, strength, features)
            modified_anchor = self._link(anchor, alternative_anchor, gate)
            modified_positive = self._link(positive, alternative_positive, gate)
            self._record_correction(corrections, anchor, modified_anchor)
            self._record_correction(corrections, positive, modified_positive)
            anchor, positive = modified_anchor, modified_positive
            gates.append(gate.detach())

        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        hard = self._m7_hard_negative(
            topology_model, hard_input, bundles["M7"][0]
        )
        topology_confidence = 0.5 * (
            _score(topology_first, anchor) + _score(topology_second, positive)
        )
        raw_hard = F.normalize(hard_input, dim=-1)
        cycle_mean = bundles["M7"][0]["views"][:, 2].mean(-1, keepdim=True)
        strata = cycle_mean.sign()
        unit_target = topology_model.bottleneck(hard_input) * strata.unsqueeze(1)
        unit_hard = F.normalize(hard_input + (unit_target - hard_input), dim=-1)
        for action, strength, gate_mode in self._actions_for_slot(
            actions, "hard_projection"
        ):
            clean_gate = features[:, 0].view(-1, 1, 1)
            topology_gate = topology_confidence.view(-1, 1, 1)
            agreement_gate = features[:, 1].view(-1, 1, 1)
            hard_correction = hard - raw_hard
            if action == "hard_projection_raw":
                alternative = raw_hard
            elif action == "hard_projection_unit_gain":
                alternative = unit_hard
            elif action == "hard_projection_clean_gate":
                alternative = F.normalize(raw_hard + clean_gate * hard_correction, dim=-1)
            elif action == "hard_projection_topology_gate":
                alternative = F.normalize(
                    raw_hard + topology_gate * hard_correction, dim=-1
                )
            elif action == "hard_projection_agreement_gate":
                alternative = F.normalize(
                    raw_hard + agreement_gate * hard_correction, dim=-1
                )
            elif action == "hard_projection_joint_gate":
                joint = (clean_gate * topology_gate * agreement_gate).clamp_min(
                    1e-12
                ).pow(1 / 3)
                alternative = F.normalize(raw_hard + joint * hard_correction, dim=-1)
            elif action == "hard_projection_clip":
                alternative = F.normalize(_clip_correction(raw_hard, hard), dim=-1)
            else:
                alternative = F.normalize(
                    _orthogonal_alternative(raw_hard, hard), dim=-1
                )
            gate = self._gate(gate_mode, strength, features)
            modified = F.normalize(self._link(hard, alternative, gate), dim=-1)
            self._record_correction(corrections, hard, modified)
            hard = modified
            gates.append(gate.detach())

        clean_weighted = self._weighted_auxiliary(clean_model, clean_auxiliary)
        topology_weighted = self._weighted_auxiliary(
            topology_model, topology_auxiliary
        )
        for action, strength, gate_mode in self._actions_for_slot(
            actions, "clean_auxiliary"
        ):
            gate = self._gate(gate_mode, strength, features).mean()
            alternative = (
                clean_weighted.detach()
                if action == "clean_auxiliary_detach"
                else clean_weighted.new_zeros(())
            )
            clean_weighted = clean_weighted + gate * (alternative - clean_weighted)
            gates.append(gate.detach().reshape(1))
        for action, strength, gate_mode in self._actions_for_slot(
            actions, "topology_auxiliary"
        ):
            gate = self._gate(gate_mode, strength, features).mean()
            alternative = (
                topology_weighted.detach()
                if action == "topology_auxiliary_detach"
                else topology_weighted.new_zeros(())
            )
            topology_weighted = topology_weighted + gate * (
                alternative - topology_weighted
            )
            gates.append(gate.detach().reshape(1))

        objective = self.first.objective(
            anchor, positive, hard, self._temperature()
        )
        auxiliary_total = clean_weighted + topology_weighted
        total = objective + auxiliary_total
        all_gates = torch.cat([value.reshape(-1) for value in gates])
        mean_correction = (
            torch.stack(corrections).mean()
            if corrections
            else total.new_zeros(())
        )
        return total, {
            "module_objective": float(objective.detach()),
            "auxiliary_total": float(auxiliary_total.detach()),
            "first_auxiliary": 0.0,
            "second_auxiliary": float(clean_auxiliary.detach()),
            "third_auxiliary": float(topology_auxiliary.detach()),
            "temperature": float(self._temperature()),
            "pipeline_gate_mean": float(all_gates.mean()),
            "pipeline_gate_std": float(
                all_gates.std(unbiased=False) if all_gates.numel() > 1 else 0.0
            ),
            "pipeline_relative_correction": float(mean_correction),
            "clean_confidence": float(features[:, 0].mean().detach()),
            "view_agreement": float(features[:, 1].mean().detach()),
            "augmentation_retention": float(features[:, 2].mean().detach()),
            "topology_prior": float(features[:, 3].mean().detach()),
            "prototype_margin": float(features[:, 4].mean().detach()),
            "topology_confidence": float(topology_confidence.mean().detach()),
        }


def build_retogcl_t077_adaptive_pipeline(
    configuration: T077AdaptivePipelineConfiguration,
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
) -> ReToGCLT077AdaptivePipeline:
    return ReToGCLT077AdaptivePipeline(
        configuration,
        in_dim,
        hidden_dim,
        layers,
        num_classes,
        idea_parameters,
    )
