"""把单个研究 idea 接入 BalanceGCL 的真实 M1--M8 数据流。

旧版 ``ResearchIdeaBalanceGCL`` 把所有 idea 都实现成编码后的并行损失分支。本模块
只复用其中已经审计过的可微算子，并根据 ``model.yaml`` 的目标模块把算子输出接回
BalanceGCL 主流水线。它用于重新进行单 idea 的 ABC 筛选；旧 checkpoint 不兼容。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data

from src.baselines.gcl_baselines import augment_graph
from src.models.research_ideas import ResearchIdeaBalanceGCL, _symmetric_nce


MODULE_IDEAS: dict[str, frozenset[str]] = {
    "M1": frozenset({
        "D7-I06", "D7-I10", "D8-I06",
        *(f"D11-I{index:02d}" for index in range(1, 11)),
    }),
    "M2": frozenset({
        *(f"D2-I{index:02d}" for index in range(1, 11)),
        "D3-I01", "D3-I04", "D3-I10",
    }),
    "M3": frozenset({
        "D3-I06", "D3-I08",
        *(f"D4-I{index:02d}" for index in range(1, 11)),
        *(f"D5-I{index:02d}" for index in range(1, 11)),
    }),
    "M4": frozenset({
        *(f"D1-I{index:02d}" for index in range(1, 11)),
        "D3-I07", "D7-I01", "D7-I02", "D7-I03", "D7-I04", "D7-I05",
        "D7-I07", "D7-I08", "D7-I09", "D8-I04", "D8-I08", "D8-I09",
    }),
    "M5": frozenset({
        "D3-I02", *(f"D6-I{index:02d}" for index in range(1, 11)),
        "D8-I03", "D8-I05", "D8-I07",
    }),
    "M6": frozenset({"D3-I05", "D3-I09", "D8-I02"}),
    "M7": frozenset(f"D10-I{index:02d}" for index in range(1, 11)),
    "M8": frozenset({
        "D3-I03", "D8-I01", "D8-I10",
        *(f"D9-I{index:02d}" for index in range(1, 11)),
    }),
}

IDEA_TO_MODULE = {
    idea_id: module_id
    for module_id, idea_ids in MODULE_IDEAS.items()
    for idea_id in idea_ids
}

if len(IDEA_TO_MODULE) != 110:
    raise RuntimeError(f"模块映射必须覆盖 110 个 idea，实际 {len(IDEA_TO_MODULE)}")


@dataclass(frozen=True)
class ModularDiagnostics:
    target_module: str
    base_objective: float
    module_objective: float
    auxiliary: float


class ModularResearchIdeaBalanceGCL(ResearchIdeaBalanceGCL):
    """单 idea 模块替换模型；任何时刻只有一个 M1--M8 位置被替换。"""

    fidelity = "module_position_faithful_differentiable_implementation"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_module = IDEA_TO_MODULE[self.spec.idea_id]

    def _parameter_controls(self) -> tuple[float, float, float]:
        """返回三个专属参数相对各自默认点的稳定控制量。

        控制量只在被替换模块内部使用：A 控制主算子，B 控制该算子的分辨率，
        C 控制文档指定的约束。它们不再形成额外的通用 idea 损失分支。
        """

        return self.idea_parameter_multipliers

    def prepare_views(self, batch: Data) -> tuple[Data, Data]:
        """M1 在编码前生成增强图；其余模块沿用 BalanceGCL 默认增强。"""

        first = augment_graph(batch, "edge_drop", 0.2)
        second = augment_graph(batch, "attr_mask", 0.2)
        if self.target_module != "M1":
            return first, second

        a_gain, b_resolution, c_gain = self._parameter_controls()
        strength = min(0.8, max(0.01, 0.2 * a_gain))
        secondary_strength = min(
            0.8,
            max(0.01, strength * max(c_gain, 0.05) / math.sqrt(max(b_resolution, 1e-3))),
        )
        # M1 的改变发生在 Data 上并先于 GIN。不同研究族保留各自的输入操作次序。
        if self.spec.family == 11:
            if self.spec.variant in {2, 4, 6, 9}:
                first = augment_graph(batch, "edge_drop", strength)
                second = augment_graph(first, "attr_mask", secondary_strength)
            elif self.spec.variant in {5, 8, 10}:
                first = augment_graph(batch, "attr_mask", strength)
                second = augment_graph(first, "edge_drop", min(0.8, strength * max(c_gain, 0.1)))
            else:
                first = augment_graph(batch, "node_drop", strength)
                second = augment_graph(batch, "subgraph", secondary_strength)
        elif self.spec.idea_id == "D7-I06":
            first = augment_graph(augment_graph(batch, "attr_mask", strength), "edge_drop", secondary_strength)
            second = augment_graph(augment_graph(batch, "edge_drop", secondary_strength), "attr_mask", strength)
        elif self.spec.idea_id == "D7-I10":
            adversarial = min(0.8, strength * max(b_resolution, 0.25))
            first = augment_graph(batch, "attr_mask", adversarial)
            second = augment_graph(batch, "subgraph", min(0.8, adversarial * max(c_gain, 0.25)))
        else:  # D8-I06：顺、逆操作保留同一节点编号，便于等变恢复。
            first = augment_graph(batch, "attr_mask", strength)
            second = Data(
                x=batch.x + (batch.x - first.x) * min(
                    1.0, max(0.0, c_gain / math.sqrt(max(b_resolution, 1e-3)))
                ),
                edge_index=batch.edge_index,
                batch=batch.batch,
                y=batch.y,
            )
        return first, second

    def _module_transform(
        self,
        first: dict[str, Tensor],
        second: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        method = getattr(self, f"_family{self.spec.family}")
        transformed_first, transformed_second, auxiliary = method(
            first, second, self.spec.variant
        )
        a_gain, _, _ = self._parameter_controls()
        if math.isclose(a_gain, 0.0, abs_tol=1e-12):
            return first["graph"], second["graph"], auxiliary
        return (
            first["graph"] + a_gain * (transformed_first - first["graph"]),
            second["graph"] + a_gain * (transformed_second - second["graph"]),
            auxiliary,
        )

    def _pipeline_loss(
        self,
        first_representation: Tensor,
        second_representation: Tensor,
        transformed_first: Tensor,
        transformed_second: Tensor,
        auxiliary: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """从 idea 的目标模块继续执行下游模块，保证存在真实数据依赖。"""

        module_id = self.target_module
        if module_id in {"M2", "M3"}:
            first_representation = transformed_first
            second_representation = transformed_second

        routing_source = transformed_first if module_id == "M4" else first_representation
        routing = self.semantic_router(routing_source, self.semantic_prototypes)

        if module_id == "M7":
            anchor = F.normalize(transformed_first, dim=-1)
            second_projected = F.normalize(transformed_second, dim=-1)
        else:
            anchor = F.normalize(self.projector(first_representation), dim=-1)
            second_projected = F.normalize(self.projector(second_representation), dim=-1)

        if module_id == "M5":
            positive_input = transformed_second
        else:
            positive_input = self.positive_builder(
                first_representation,
                second_representation,
                second_projected,
                routing,
            )

        if module_id == "M7":
            positive = second_projected
        else:
            positive = F.normalize(self.projector(positive_input), dim=-1)

        if module_id == "M6":
            hard_input = transformed_second.roll(1, 0).unsqueeze(1)
        else:
            hard_input = self.negative_builder(first_representation, routing)
        hard = (
            F.normalize(hard_input, dim=-1)
            if module_id == "M7"
            else F.normalize(self.projector(hard_input), dim=-1)
        )

        _, b_resolution, c_gain = self._parameter_controls()
        temperature = 0.2 / math.sqrt(max(b_resolution, 1e-3))
        if module_id == "M8":
            # M8 替换默认目标，而不是把新目标附加到默认目标上。
            objective = _symmetric_nce(
                transformed_first,
                transformed_second,
                temperature=temperature,
            )
        else:
            objective = self.objective(anchor, positive, hard, temperature)
        auxiliary_term = (
            auxiliary.new_zeros(())
            if math.isclose(c_gain, 0.0, abs_tol=1e-12)
            else c_gain * auxiliary
        )
        return objective + self.auxiliary_weight * auxiliary_term, objective

    def idea_loss(self, first_data: Data, second_data: Data) -> tuple[Tensor, dict[str, float]]:
        first = self._bundle(first_data)
        second = self._bundle(second_data)
        base_objective = super().contrastive_loss(first["graph"], second["graph"], 0.2)

        if self.target_module == "M1":
            # 图已经在 prepare_views 中被替换；此处只执行默认 M2--M8。
            total = base_objective
            module_objective = base_objective
            auxiliary = base_objective.new_zeros(())
        else:
            transformed_first, transformed_second, auxiliary = self._module_transform(first, second)
            total, module_objective = self._pipeline_loss(
                first["graph"], second["graph"], transformed_first, transformed_second, auxiliary
            )

        diagnostics = ModularDiagnostics(
            target_module=self.target_module,
            base_objective=float(base_objective.detach()),
            module_objective=float(module_objective.detach()),
            auxiliary=float(auxiliary.detach()),
        )
        return total, {
            "target_module": diagnostics.target_module,
            "base_objective": diagnostics.base_objective,
            "module_objective": diagnostics.module_objective,
            "auxiliary": diagnostics.auxiliary,
            "fidelity": self.fidelity,
        }


def build_modular_research_idea_model(
    idea_id: str,
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    idea_parameters: dict[str, float] | None = None,
) -> ModularResearchIdeaBalanceGCL:
    return ModularResearchIdeaBalanceGCL(
        idea_id,
        in_dim,
        hidden_dim,
        layers,
        num_classes,
        auxiliary_weight=0.25,
        idea_temperature=0.2,
        operator_mix=1.0,
        idea_parameters=idea_parameters,
    )
