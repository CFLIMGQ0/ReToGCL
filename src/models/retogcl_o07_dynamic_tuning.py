"""以 temperature=0.06 为默认点的 O07 动态调参模型。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from torch import Tensor
import torch.nn.functional as F

from src.models.modular_research_ideas import ModularResearchIdeaBalanceGCL
from src.models.retogcl_single_point_optimizations import (
    O07_DEFAULT_TEMPERATURE,
    ReToGCLSinglePointOptimization,
)


@dataclass(frozen=True)
class O07DynamicConfiguration:
    temperature: float = O07_DEFAULT_TEMPERATURE
    hard_alignment_gain: float = 4.0
    auxiliary_scale: float = 1.0
    weight_decay: float = 1e-5

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if self.hard_alignment_gain < 0:
            raise ValueError("hard_alignment_gain 必须大于等于 0")
        if self.auxiliary_scale < 0:
            raise ValueError("auxiliary_scale 必须大于等于 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay 必须大于等于 0")
        if len(self.changed_parameters()) > 2:
            raise ValueError("单个版本相对 P05 默认配置最多只能改变两个参数")

    def changed_parameters(self) -> tuple[str, ...]:
        defaults = {
            "temperature": O07_DEFAULT_TEMPERATURE,
            "hard_alignment_gain": 4.0,
            "auxiliary_scale": 1.0,
            "weight_decay": 1e-5,
        }
        return tuple(
            name
            for name, default in defaults.items()
            if not math.isclose(
                float(getattr(self, name)), default, rel_tol=1e-9, abs_tol=1e-12
            )
        )


class ReToGCLO07DynamicTuning(ReToGCLSinglePointOptimization):
    """保留 O07 结构，只注入显式调参配置。"""

    protocol = "retogcl_o07_dynamic_tuning_v1"

    def __init__(
        self,
        configuration: O07DynamicConfiguration,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
    ) -> None:
        self.tuning_configuration = configuration
        super().__init__(
            "O07", in_dim, hidden_dim, layers, num_classes, idea_parameters
        )

    def _temperature(self) -> float:
        return self.tuning_configuration.temperature

    def _weighted_auxiliary(
        self,
        model: ModularResearchIdeaBalanceGCL,
        auxiliary: Tensor,
    ) -> Tensor:
        if math.isclose(
            self.tuning_configuration.auxiliary_scale, 0.0, abs_tol=1e-12
        ):
            # 真正的辅助项消融必须切断该分支；先计算再乘 0 仍可能传播
            # 0×NaN，Tox21 的边界配置会因此产生非有限梯度。
            return auxiliary.new_zeros(())
        return (
            self.tuning_configuration.auxiliary_scale
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
        gain = self.tuning_configuration.hard_alignment_gain
        mapped = hard_input + gain * (mapped - hard_input)
        return F.normalize(mapped, dim=-1)


def build_retogcl_o07_dynamic_tuning(
    configuration: O07DynamicConfiguration,
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
) -> ReToGCLO07DynamicTuning:
    return ReToGCLO07DynamicTuning(
        configuration, in_dim, hidden_dim, layers, num_classes, idea_parameters
    )
