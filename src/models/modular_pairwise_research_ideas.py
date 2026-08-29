"""在 BalanceGCL 的两个不同模块位置联合接入两个研究 idea。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.models.modular_research_ideas import (
    IDEA_TO_MODULE,
    ModularResearchIdeaBalanceGCL,
    build_modular_research_idea_model,
)
from src.models.research_ideas import _symmetric_nce


MODULE_ORDER = {f"M{index}": index for index in range(1, 9)}


class ModularPairwiseResearchIdeaBalanceGCL(nn.Module):
    """共享主干，并按真实流水线位置顺序执行两个不冲突的 idea。"""

    fidelity = "module_position_faithful_differentiable_pairwise_implementation"

    def __init__(
        self,
        first_idea: str,
        second_idea: str,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        first_parameters: dict[str, float],
        second_parameters: dict[str, float],
    ):
        super().__init__()
        first_module = IDEA_TO_MODULE[first_idea]
        second_module = IDEA_TO_MODULE[second_idea]
        if first_module == second_module:
            raise ValueError(
                f"同一 pipeline 模块的 idea 不允许组合：{first_idea}+{second_idea} ({first_module})"
            )

        entries = sorted(
            (
                (first_module, first_idea, first_parameters),
                (second_module, second_idea, second_parameters),
            ),
            key=lambda item: (MODULE_ORDER[item[0]], item[1]),
        )
        self.first = build_modular_research_idea_model(
            entries[0][1], in_dim, hidden_dim, layers, num_classes, entries[0][2]
        )
        self.second = build_modular_research_idea_model(
            entries[1][1], in_dim, hidden_dim, layers, num_classes, entries[1][2]
        )

        # 两个模块属于同一个 BalanceGCL：编码器和原始 M2--M8 组件只能各有一套。
        for name in (
            "encoder",
            "projector",
            "semantic_prototypes",
            "readout",
            "semantic_router",
            "positive_builder",
            "negative_builder",
            "objective",
        ):
            setattr(self.second, name, getattr(self.first, name))

        self.idea_ids = (entries[0][1], entries[1][1])
        self.target_modules = (entries[0][0], entries[1][0])
        self.output_dim = self.first.output_dim
        self._by_module = {
            self.first.target_module: self.first,
            self.second.target_module: self.second,
        }

    @property
    def encoder(self):
        return self.first.encoder

    def encode(self, data: Data) -> Tensor:
        return self.first._bundle(data)["graph"]

    def prepare_views(self, batch: Data) -> tuple[Data, Data]:
        module = self._by_module.get("M1")
        return (module or self.first).prepare_views(batch)

    @staticmethod
    def _with_graph(bundle: dict[str, Tensor], graph: Tensor) -> dict[str, Tensor]:
        return {**bundle, "graph": graph}

    def _apply_idea_transform(
        self,
        model: ModularResearchIdeaBalanceGCL,
        first_bundle: dict[str, Tensor],
        second_bundle: dict[str, Tensor],
        first_graph: Tensor,
        second_graph: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return model._module_transform(
            self._with_graph(first_bundle, first_graph),
            self._with_graph(second_bundle, second_graph),
        )

    def _temperature(self) -> float:
        # M1 的 B 已在图增强强度中生效；其余模块的 B 共同控制下游关系分辨率。
        product = 1.0
        for model in (self.first, self.second):
            if model.target_module != "M1":
                product *= max(model._parameter_controls()[1], 1e-3)
        return 0.2 / math.sqrt(max(product, 1e-3))

    def idea_loss(self, first_data: Data, second_data: Data) -> tuple[Tensor, dict[str, float]]:
        first_history = self.encoder(first_data)
        second_history = self.encoder(second_data)
        bundles = {
            model.target_module: (
                model._bundle_from_history(first_data, first_history),
                model._bundle_from_history(second_data, second_history),
            )
            for model in (self.first, self.second)
        }
        first_graph = self.first._bundle_from_history(first_data, first_history)["graph"]
        second_graph = self.first._bundle_from_history(second_data, second_history)["graph"]
        auxiliaries: list[tuple[ModularResearchIdeaBalanceGCL, Tensor]] = []

        # M2/M3 改写编码后的图表示，并依次传给所有下游模块。
        for module_id in ("M2", "M3"):
            model = self._by_module.get(module_id)
            if model is None:
                continue
            first_graph, second_graph, auxiliary = self._apply_idea_transform(
                model, *bundles[module_id], first_graph, second_graph
            )
            auxiliaries.append((model, auxiliary))

        # M4 只替换语义路由输入。
        routing_source = first_graph
        model = self._by_module.get("M4")
        if model is not None:
            transformed_first, _, auxiliary = self._apply_idea_transform(
                model, *bundles["M4"], first_graph, second_graph
            )
            routing_source = transformed_first
            auxiliaries.append((model, auxiliary))
        routing = self.first.semantic_router(routing_source, self.first.semantic_prototypes)

        projected_second = F.normalize(self.first.projector(second_graph), dim=-1)
        positive_input = self.first.positive_builder(
            first_graph, second_graph, projected_second, routing
        )
        model = self._by_module.get("M5")
        if model is not None:
            _, positive_input, auxiliary = self._apply_idea_transform(
                model, *bundles["M5"], first_graph, second_graph
            )
            auxiliaries.append((model, auxiliary))

        hard_input = self.first.negative_builder(first_graph, routing)
        model = self._by_module.get("M6")
        if model is not None:
            _, transformed_second, auxiliary = self._apply_idea_transform(
                model, *bundles["M6"], first_graph, second_graph
            )
            hard_input = transformed_second.roll(1, 0).unsqueeze(1)
            auxiliaries.append((model, auxiliary))

        # M7 替换投影阶段；若 M5 已替换正样本，则其输出继续进入 M7。
        model = self._by_module.get("M7")
        if model is not None:
            m7_second = positive_input if "M5" in self._by_module else second_graph
            anchor, positive, auxiliary = self._apply_idea_transform(
                model, *bundles["M7"], first_graph, m7_second
            )
            anchor = F.normalize(anchor, dim=-1)
            positive = F.normalize(positive, dim=-1)
            hard = F.normalize(hard_input, dim=-1)
            auxiliaries.append((model, auxiliary))
        else:
            anchor = F.normalize(self.first.projector(first_graph), dim=-1)
            positive = F.normalize(self.first.projector(positive_input), dim=-1)
            hard = F.normalize(self.first.projector(hard_input), dim=-1)

        temperature = self._temperature()
        model = self._by_module.get("M8")
        if model is not None:
            objective_first, objective_second, auxiliary = self._apply_idea_transform(
                model, *bundles["M8"], anchor, positive
            )
            objective = _symmetric_nce(
                objective_first, objective_second, temperature=temperature
            )
            auxiliaries.append((model, auxiliary))
        else:
            objective = self.first.objective(anchor, positive, hard, temperature)

        auxiliary_total = objective.new_zeros(())
        auxiliary_values = {self.idea_ids[0]: 0.0, self.idea_ids[1]: 0.0}
        for idea_model, auxiliary in auxiliaries:
            c_gain = idea_model._parameter_controls()[2]
            weighted = idea_model.auxiliary_weight * c_gain * auxiliary
            auxiliary_total = auxiliary_total + weighted
            auxiliary_values[idea_model.spec.idea_id] = float(auxiliary.detach())
        total = objective + auxiliary_total
        return total, {
            "module_objective": float(objective.detach()),
            "auxiliary_total": float(auxiliary_total.detach()),
            "first_auxiliary": auxiliary_values[self.idea_ids[0]],
            "second_auxiliary": auxiliary_values[self.idea_ids[1]],
            "temperature": float(temperature),
        }


def build_modular_pairwise_research_idea_model(
    first_idea: str,
    second_idea: str,
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    first_parameters: dict[str, float],
    second_parameters: dict[str, float],
) -> ModularPairwiseResearchIdeaBalanceGCL:
    return ModularPairwiseResearchIdeaBalanceGCL(
        first_idea,
        second_idea,
        in_dim,
        hidden_dim,
        layers,
        num_classes,
        first_parameters,
        second_parameters,
    )
