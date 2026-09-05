"""O07（困难负样本度量对齐）的十个单因素参数版本。

所有版本都保留 ReToGCL 的三个创新模块及 O07 数据流，只相对正式 O07
改变一个参数，以便判断性能下降来自度量强度、温度还是正则化/容量。
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
import torch.nn.functional as F

from src.models.modular_research_ideas import ModularResearchIdeaBalanceGCL
from src.models.retogcl_single_point_optimizations import (
    ReToGCLSinglePointOptimization,
)


@dataclass(frozen=True)
class O07ParameterVariant:
    variant_id: str
    name: str
    changed_parameter: str
    hard_alignment_gain: float = 4.0
    temperature: float | None = None
    auxiliary_scale: float = 1.0
    weight_decay: float = 1e-5
    layers: int = 3


O07_PARAMETER_VARIANTS: dict[str, O07ParameterVariant] = {
    "P01": O07ParameterVariant(
        "P01", "对齐增益 0.5", "hard_alignment_gain: 4.0 -> 0.5",
        hard_alignment_gain=0.5,
    ),
    "P02": O07ParameterVariant(
        "P02", "对齐增益 1.0", "hard_alignment_gain: 4.0 -> 1.0",
        hard_alignment_gain=1.0,
    ),
    "P03": O07ParameterVariant(
        "P03", "对齐增益 2.0", "hard_alignment_gain: 4.0 -> 2.0",
        hard_alignment_gain=2.0,
    ),
    "P04": O07ParameterVariant(
        "P04", "对齐增益 3.0", "hard_alignment_gain: 4.0 -> 3.0",
        hard_alignment_gain=3.0,
    ),
    "P05": O07ParameterVariant(
        "P05", "温度 0.06", "contrastive_temperature: 0.08944 -> 0.06",
        temperature=0.06,
    ),
    "P06": O07ParameterVariant(
        "P06", "温度 0.12", "contrastive_temperature: 0.08944 -> 0.12",
        temperature=0.12,
    ),
    "P07": O07ParameterVariant(
        "P07", "温度 0.16", "contrastive_temperature: 0.08944 -> 0.16",
        temperature=0.16,
    ),
    "P08": O07ParameterVariant(
        "P08", "辅助约束减半", "auxiliary_scale: 1.0 -> 0.5",
        auxiliary_scale=0.5,
    ),
    "P09": O07ParameterVariant(
        "P09", "增强权重衰减", "weight_decay: 1e-5 -> 1e-4",
        weight_decay=1e-4,
    ),
    "P10": O07ParameterVariant(
        "P10", "两层编码器", "encoder_layers: 3 -> 2",
        layers=2,
    ),
}


class ReToGCLO07ParameterVariant(ReToGCLSinglePointOptimization):
    """在正式 O07 上应用恰好一个参数改动。"""

    protocol = "retogcl_o07_parameter_variant_temperature006_v2"

    def __init__(
        self,
        variant_id: str,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
    ) -> None:
        if variant_id not in O07_PARAMETER_VARIANTS:
            raise ValueError(f"未知 O07 参数版本：{variant_id}")
        self.parameter_variant = O07_PARAMETER_VARIANTS[variant_id]
        super().__init__(
            "O07", in_dim, hidden_dim, layers, num_classes, idea_parameters
        )

    def _temperature(self) -> float:
        if self.parameter_variant.temperature is not None:
            return self.parameter_variant.temperature
        return super()._temperature()

    def _weighted_auxiliary(
        self,
        model: ModularResearchIdeaBalanceGCL,
        auxiliary: Tensor,
    ) -> Tensor:
        return (
            self.parameter_variant.auxiliary_scale
            * super()._weighted_auxiliary(model, auxiliary)
        )

    def _m7_hard_negative(
        self,
        model: ModularResearchIdeaBalanceGCL,
        hard_input: Tensor,
        first_bundle: dict[str, Tensor],
    ) -> Tensor:
        strata = first_bundle["views"][:, 2].mean(-1, keepdim=True).sign()
        mapped = model.bottleneck(hard_input) * strata.unsqueeze(1)
        gain = self.parameter_variant.hard_alignment_gain
        mapped = hard_input + gain * (mapped - hard_input)
        return F.normalize(mapped, dim=-1)


def build_retogcl_o07_parameter_variant(
    variant_id: str,
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
) -> ReToGCLO07ParameterVariant:
    return ReToGCLO07ParameterVariant(
        variant_id, in_dim, hidden_dim, layers, num_classes, idea_parameters
    )
