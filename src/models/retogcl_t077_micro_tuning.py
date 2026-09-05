"""以 T077 为不可变基准的 ReToGCL 小步结构与参数微调模型。

每个配置相对 T077 只允许改变一个、最多两个字段。结构策略统一用
``micro_mode + micro_strength`` 表示，并通过残差插值保证改动从 T077 连续出发。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from src.models.modular_research_ideas import ModularResearchIdeaBalanceGCL
from src.models.retogcl_single_point_optimizations import (
    ReToGCLSinglePointOptimization,
)


T077_TEMPERATURE = 0.045
T077_HARD_ALIGNMENT_GAIN = 4.0
T077_AUXILIARY_SCALE = 1.0
T077_WEIGHT_DECAY = 5e-4


HARD_MODES = (
    "hard_clean_gate",
    "hard_dirty_gate",
    "hard_topology_gate",
    "hard_joint_geometric_gate",
    "hard_joint_min_gate",
    "hard_gate_agreement",
    "hard_residual_cosine_gate",
    "hard_residual_distance_gate",
    "hard_spread_gate",
    "hard_correction_clip",
    "hard_correction_tanh",
    "hard_spherical_blend",
    "hard_orthogonal_correction",
    "hard_target_center",
    "hard_joint_center",
    "hard_target_standardize",
    "hard_soft_cycle_strata",
    "hard_multiview_strata",
    "hard_clean_logit_scale",
    "hard_uncertain_logit_scale",
)


PIPELINE_MODES = (
    "pipe_clean_to_topology",
    "pipe_dirty_to_topology",
    "pipe_joint_to_topology",
    "pipe_agreement_to_topology",
    "pipe_anchor_clean_gate",
    "pipe_positive_clean_gate",
    "pipe_topology_residual_clip",
    "pipe_topology_residual_tanh",
    "pipe_topology_spherical",
    "pipe_topology_orthogonal",
    "pipe_topology_center",
    "pipe_topology_standardize",
    "pipe_clean_residual_clip",
    "pipe_clean_residual_tanh",
    "pipe_clean_spherical",
    "pipe_clean_orthogonal",
    "pipe_clean_routing_shrink",
    "pipe_topology_shrink",
    "pipe_clean_aux_shrink",
    "pipe_topology_aux_shrink",
)


ALL_MICRO_MODES = frozenset(("t077", *HARD_MODES, *PIPELINE_MODES))


@dataclass(frozen=True)
class T077MicroConfiguration:
    """单个小步候选；默认值严格等于已经完成实验的 T077。"""

    micro_mode: str = "t077"
    micro_strength: float = 1.0
    temperature: float = T077_TEMPERATURE
    hard_alignment_gain: float = T077_HARD_ALIGNMENT_GAIN
    auxiliary_scale: float = T077_AUXILIARY_SCALE
    weight_decay: float = T077_WEIGHT_DECAY

    def __post_init__(self) -> None:
        if self.micro_mode not in ALL_MICRO_MODES:
            raise ValueError(f"未知 T077 微调策略：{self.micro_mode}")
        if not 0 <= self.micro_strength <= 2:
            raise ValueError("micro_strength 必须位于 [0, 2]")
        if self.temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if self.hard_alignment_gain < 0:
            raise ValueError("hard_alignment_gain 必须大于等于 0")
        if self.auxiliary_scale < 0:
            raise ValueError("auxiliary_scale 必须大于等于 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay 必须大于等于 0")
        if self.micro_mode == "t077" and not math.isclose(
            self.micro_strength, 1.0, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("未启用结构策略时不能单独修改 micro_strength")
        if len(self.changed_parameters()) > 2:
            raise ValueError("每个版本相对 T077 最多改变两个字段")

    def changed_parameters(self) -> tuple[str, ...]:
        defaults: dict[str, str | float] = {
            "micro_mode": "t077",
            "micro_strength": 1.0,
            "temperature": T077_TEMPERATURE,
            "hard_alignment_gain": T077_HARD_ALIGNMENT_GAIN,
            "auxiliary_scale": T077_AUXILIARY_SCALE,
            "weight_decay": T077_WEIGHT_DECAY,
        }
        changed = []
        for name, default in defaults.items():
            value = getattr(self, name)
            if isinstance(default, str):
                different = value != default
            else:
                different = not math.isclose(
                    float(value), default, rel_tol=1e-9, abs_tol=1e-12
                )
            if different:
                changed.append(name)
        return tuple(changed)


def _score(left: Tensor, right: Tensor) -> Tensor:
    """把两个表示的余弦一致性稳定映射到 [0, 1]。"""

    return ((F.cosine_similarity(left, right, dim=-1) + 1.0) * 0.5).clamp(0, 1)


def _blend(original: Tensor, alternative: Tensor, strength: float) -> Tensor:
    return original + strength * (alternative - original)


def _clip_correction(base: Tensor, transformed: Tensor) -> Tensor:
    correction = transformed - base
    base_norm = base.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    correction_norm = correction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return base + correction * (base_norm / correction_norm).clamp_max(1.0)


def _tanh_correction(base: Tensor, transformed: Tensor) -> Tensor:
    scale = base.norm(dim=-1, keepdim=True).div(
        math.sqrt(max(base.size(-1), 1))
    ).clamp_min(1e-6)
    return base + scale * torch.tanh((transformed - base) / scale)


def _spherical_alternative(base: Tensor, transformed: Tensor) -> Tensor:
    direction = F.normalize(base, dim=-1) + F.normalize(transformed, dim=-1)
    return F.normalize(direction, dim=-1) * base.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)


def _orthogonal_alternative(base: Tensor, transformed: Tensor) -> Tensor:
    correction = transformed - base
    parallel = (correction * base).sum(-1, keepdim=True) / base.pow(2).sum(
        -1, keepdim=True
    ).clamp_min(1e-6)
    return base + correction - parallel * base


def _center_last_set(values: Tensor) -> Tensor:
    if values.ndim < 3:
        return values - values.mean(0, keepdim=True)
    return values - values.mean(1, keepdim=True)


def _standardize_last(values: Tensor) -> Tensor:
    centered = values - values.mean(-1, keepdim=True)
    return centered / centered.pow(2).mean(
        -1, keepdim=True
    ).clamp_min(1e-12).sqrt()


class ReToGCLT077MicroTuning(ReToGCLSinglePointOptimization):
    """保留 T077 主体，只在指定接口注入一个连续小改动。"""

    protocol = "retogcl_t077_micro_tuning_v1"

    def __init__(
        self,
        configuration: T077MicroConfiguration,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
    ) -> None:
        self.micro_configuration = configuration
        self._clean_score: Tensor | None = None
        self._topology_score: Tensor | None = None
        super().__init__(
            "O07", in_dim, hidden_dim, layers, num_classes, idea_parameters
        )

    def _temperature(self) -> float:
        return self.micro_configuration.temperature

    def idea_loss(self, first_data, second_data):
        self._clean_score = None
        self._topology_score = None
        return super().idea_loss(first_data, second_data)

    def _weighted_auxiliary(
        self,
        model: ModularResearchIdeaBalanceGCL,
        auxiliary: Tensor,
    ) -> Tensor:
        if math.isclose(
            self.micro_configuration.auxiliary_scale, 0.0, abs_tol=1e-12
        ):
            return auxiliary.new_zeros(())
        weighted = (
            self.micro_configuration.auxiliary_scale
            * super()._weighted_auxiliary(model, auxiliary)
        )
        mode = self.micro_configuration.micro_mode
        strength = self.micro_configuration.micro_strength
        if mode == "pipe_clean_aux_shrink" and model.target_module == "M4":
            weighted = (1.0 - 0.5 * strength) * weighted
        elif mode == "pipe_topology_aux_shrink" and model.target_module == "M7":
            weighted = (1.0 - 0.5 * strength) * weighted
        return weighted

    def _apply_idea_transform(
        self,
        model: ModularResearchIdeaBalanceGCL,
        first_bundle: dict[str, Tensor],
        second_bundle: dict[str, Tensor],
        first_graph: Tensor,
        second_graph: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        first, second, auxiliary = super()._apply_idea_transform(
            model, first_bundle, second_bundle, first_graph, second_graph
        )
        mode = self.micro_configuration.micro_mode
        strength = self.micro_configuration.micro_strength

        if model.target_module == "M4":
            self._clean_score = 0.5 * (
                _score(first_graph, first) + _score(second_graph, second)
            )
            alternative_first, alternative_second = first, second
            if mode == "pipe_clean_residual_clip":
                alternative_first = _clip_correction(first_graph, first)
                alternative_second = _clip_correction(second_graph, second)
            elif mode == "pipe_clean_residual_tanh":
                alternative_first = _tanh_correction(first_graph, first)
                alternative_second = _tanh_correction(second_graph, second)
            elif mode == "pipe_clean_spherical":
                alternative_first = _spherical_alternative(first_graph, first)
                alternative_second = _spherical_alternative(second_graph, second)
            elif mode == "pipe_clean_orthogonal":
                alternative_first = _orthogonal_alternative(first_graph, first)
                alternative_second = _orthogonal_alternative(second_graph, second)
            elif mode == "pipe_clean_routing_shrink":
                alternative_first = 0.5 * (first_graph + first)
                alternative_second = 0.5 * (second_graph + second)
            if mode in {
                "pipe_clean_residual_clip", "pipe_clean_residual_tanh",
                "pipe_clean_spherical", "pipe_clean_orthogonal",
                "pipe_clean_routing_shrink",
            }:
                first = _blend(first, alternative_first, strength)
                second = _blend(second, alternative_second, strength)

        if model.target_module == "M7":
            self._topology_score = 0.5 * (
                _score(first_graph, first) + _score(second_graph, second)
            )
            clean = self._clean_score
            if clean is None:
                clean = torch.ones_like(self._topology_score)
            topology = self._topology_score
            alternative_first, alternative_second = first, second
            if mode in {
                "pipe_clean_to_topology", "pipe_dirty_to_topology",
                "pipe_joint_to_topology", "pipe_agreement_to_topology",
                "pipe_anchor_clean_gate", "pipe_positive_clean_gate",
            }:
                if mode == "pipe_clean_to_topology":
                    gate = clean
                elif mode == "pipe_dirty_to_topology":
                    gate = 1.0 - clean
                elif mode == "pipe_joint_to_topology":
                    gate = (clean * topology).clamp_min(1e-12).sqrt()
                else:
                    gate = (1.0 - (clean - topology).abs()).clamp(0, 1)
                gated_first = first_graph + gate.unsqueeze(-1) * (first - first_graph)
                gated_second = second_graph + gate.unsqueeze(-1) * (second - second_graph)
                if mode != "pipe_positive_clean_gate":
                    alternative_first = gated_first
                if mode != "pipe_anchor_clean_gate":
                    alternative_second = gated_second
            elif mode == "pipe_topology_residual_clip":
                alternative_first = _clip_correction(first_graph, first)
                alternative_second = _clip_correction(second_graph, second)
            elif mode == "pipe_topology_residual_tanh":
                alternative_first = _tanh_correction(first_graph, first)
                alternative_second = _tanh_correction(second_graph, second)
            elif mode == "pipe_topology_spherical":
                alternative_first = _spherical_alternative(first_graph, first)
                alternative_second = _spherical_alternative(second_graph, second)
            elif mode == "pipe_topology_orthogonal":
                alternative_first = _orthogonal_alternative(first_graph, first)
                alternative_second = _orthogonal_alternative(second_graph, second)
            elif mode == "pipe_topology_center":
                alternative_first = _center_last_set(first)
                alternative_second = _center_last_set(second)
            elif mode == "pipe_topology_standardize":
                alternative_first = _standardize_last(first)
                alternative_second = _standardize_last(second)
            elif mode == "pipe_topology_shrink":
                alternative_first = 0.5 * (first_graph + first)
                alternative_second = 0.5 * (second_graph + second)
            if mode in PIPELINE_MODES and mode not in {
                "pipe_clean_residual_clip", "pipe_clean_residual_tanh",
                "pipe_clean_spherical", "pipe_clean_orthogonal",
                "pipe_clean_routing_shrink", "pipe_clean_aux_shrink",
                "pipe_topology_aux_shrink",
            }:
                first = _blend(first, alternative_first, strength)
                second = _blend(second, alternative_second, strength)
        return first, second, auxiliary

    def _m7_hard_negative(
        self,
        model: ModularResearchIdeaBalanceGCL,
        hard_input: Tensor,
        first_bundle: dict[str, Tensor],
    ) -> Tensor:
        configuration = self.micro_configuration
        mode = configuration.micro_mode
        strength = configuration.micro_strength
        cycle_mean = first_bundle["views"][:, 2].mean(-1, keepdim=True)
        strata = cycle_mean.sign()
        raw = hard_input
        target = model.bottleneck(raw) * strata.unsqueeze(1)
        mapped = raw + configuration.hard_alignment_gain * (target - raw)
        if mode not in HARD_MODES:
            return F.normalize(mapped, dim=-1)

        clean = self._clean_score
        topology = self._topology_score
        if clean is None:
            clean = raw.new_ones(raw.size(0))
        if topology is None:
            topology = raw.new_ones(raw.size(0))
        clean_gate = clean.view(-1, 1, 1)
        topology_gate = topology.view(-1, 1, 1)
        correction = mapped - raw

        if mode == "hard_clean_gate":
            alternative = raw + clean_gate * correction
        elif mode == "hard_dirty_gate":
            alternative = raw + (1.0 - clean_gate) * correction
        elif mode == "hard_topology_gate":
            alternative = raw + topology_gate * correction
        elif mode == "hard_joint_geometric_gate":
            alternative = raw + (
                clean_gate * topology_gate
            ).clamp_min(1e-12).sqrt() * correction
        elif mode == "hard_joint_min_gate":
            alternative = raw + torch.minimum(clean_gate, topology_gate) * correction
        elif mode == "hard_gate_agreement":
            gate = (1.0 - (clean_gate - topology_gate).abs()).clamp(0, 1)
            alternative = raw + gate * correction
        elif mode == "hard_residual_cosine_gate":
            gate = _score(raw, target).unsqueeze(-1)
            alternative = raw + gate * correction
        elif mode == "hard_residual_distance_gate":
            distance = (target - raw).pow(2).mean(
                -1, keepdim=True
            ).clamp_min(1e-12).sqrt()
            alternative = raw + torch.exp(-distance) * correction
        elif mode == "hard_spread_gate":
            spread = raw.var(1, unbiased=False).mean(
                -1, keepdim=True
            ).clamp_min(1e-12).sqrt()
            gate = spread.div(spread.mean().detach().clamp_min(1e-6)).sigmoid()
            alternative = raw + gate.unsqueeze(-1) * correction
        elif mode == "hard_correction_clip":
            alternative = _clip_correction(raw, mapped)
        elif mode == "hard_correction_tanh":
            alternative = _tanh_correction(raw, mapped)
        elif mode == "hard_spherical_blend":
            alternative = _spherical_alternative(raw, mapped)
        elif mode == "hard_orthogonal_correction":
            alternative = _orthogonal_alternative(raw, mapped)
        elif mode == "hard_target_center":
            centered = _center_last_set(target)
            alternative = raw + configuration.hard_alignment_gain * (centered - raw)
        elif mode == "hard_joint_center":
            centered_raw = _center_last_set(raw)
            centered_target = _center_last_set(target)
            alternative = centered_raw + configuration.hard_alignment_gain * (
                centered_target - centered_raw
            )
        elif mode == "hard_target_standardize":
            standardized = _standardize_last(target)
            target_norm = target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            standardized = F.normalize(standardized, dim=-1) * target_norm
            alternative = raw + configuration.hard_alignment_gain * (
                standardized - raw
            )
        elif mode == "hard_soft_cycle_strata":
            soft_target = model.bottleneck(raw) * torch.tanh(
                4.0 * cycle_mean
            ).unsqueeze(1)
            alternative = raw + configuration.hard_alignment_gain * (
                soft_target - raw
            )
        elif mode == "hard_multiview_strata":
            multiview = first_bundle["views"][:, :3].mean((1, 2), keepdim=True)
            multi_target = model.bottleneck(raw) * multiview.sign()
            alternative = raw + configuration.hard_alignment_gain * (
                multi_target - raw
            )
        elif mode == "hard_clean_logit_scale":
            normalized = F.normalize(mapped, dim=-1)
            return normalized * (1.0 + strength * (clean_gate - 0.5))
        else:  # hard_uncertain_logit_scale
            normalized = F.normalize(mapped, dim=-1)
            return normalized * (1.0 + strength * (0.5 - clean_gate))

        return F.normalize(_blend(mapped, alternative, strength), dim=-1)


def build_retogcl_t077_micro_tuning(
    configuration: T077MicroConfiguration,
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
) -> ReToGCLT077MicroTuning:
    return ReToGCLT077MicroTuning(
        configuration,
        in_dim,
        hidden_dim,
        layers,
        num_classes,
        idea_parameters,
    )
