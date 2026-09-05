"""在 BalanceGCL 的三个不同模块位置联合接入三个研究 idea。"""

from __future__ import annotations

import math

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


class ModularTripleResearchIdeaBalanceGCL(nn.Module):
    """共享主干，并按真实流水线位置顺序执行三个互不冲突的 idea。"""

    fidelity = "module_position_faithful_differentiable_triple_implementation"

    def __init__(
        self,
        idea_ids: tuple[str, str, str],
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        idea_parameters: tuple[
            dict[str, float], dict[str, float], dict[str, float]
        ],
    ):
        super().__init__()
        if len(idea_ids) != 3 or len(idea_parameters) != 3:
            raise ValueError("三模块模型必须提供恰好三个 idea 及其参数")

        entries = sorted(
            (
                (IDEA_TO_MODULE[idea_id], idea_id, parameters)
                for idea_id, parameters in zip(idea_ids, idea_parameters, strict=True)
            ),
            key=lambda item: (MODULE_ORDER[item[0]], item[1]),
        )
        modules = [entry[0] for entry in entries]
        if len(set(modules)) != 3:
            raise ValueError(
                f"三模块组合要求三个不同 pipeline 模块：{idea_ids} -> {modules}"
            )

        self.first = build_modular_research_idea_model(
            entries[0][1], in_dim, hidden_dim, layers, num_classes, entries[0][2]
        )
        self.second = build_modular_research_idea_model(
            entries[1][1], in_dim, hidden_dim, layers, num_classes, entries[1][2]
        )
        self.third = build_modular_research_idea_model(
            entries[2][1], in_dim, hidden_dim, layers, num_classes, entries[2][2]
        )

        # 三个模块属于同一个 BalanceGCL，主干与默认 M2--M8 组件只能各保留一套。
        shared_names = (
            "encoder",
            "projector",
            "semantic_prototypes",
            "readout",
            "semantic_router",
            "positive_builder",
            "negative_builder",
            "objective",
        )
        for model in (self.second, self.third):
            for name in shared_names:
                setattr(model, name, getattr(self.first, name))

        self.idea_ids = tuple(entry[1] for entry in entries)
        self.target_modules = tuple(entry[0] for entry in entries)
        self.output_dim = self.first.output_dim
        self._by_module = {
            model.target_module: model for model in self._idea_models
        }

    @property
    def _idea_models(self) -> tuple[ModularResearchIdeaBalanceGCL, ...]:
        return self.first, self.second, self.third

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
        product = 1.0
        for model in self._idea_models:
            if model.target_module != "M1":
                product *= max(model._parameter_controls()[1], 1e-3)
        return 0.2 / math.sqrt(max(product, 1e-3))

    def _m7_hard_negative(
        self,
        model: ModularResearchIdeaBalanceGCL,
        hard_input: Tensor,
        first_bundle: dict[str, Tensor],
    ) -> Tensor:
        """M7 对困难负样本的默认处理；默认值保持旧公式不变。"""

        del model, first_bundle
        return F.normalize(hard_input, dim=-1)

    def _weighted_auxiliary(
        self,
        model: ModularResearchIdeaBalanceGCL,
        auxiliary: Tensor,
    ) -> Tensor:
        """计算单个 idea 的辅助项；默认值保持旧权重公式不变。"""

        c_gain = model._parameter_controls()[2]
        return model.auxiliary_weight * c_gain * auxiliary

    def idea_loss(self, first_data: Data, second_data: Data) -> tuple[Tensor, dict[str, float]]:
        first_history = self.encoder(first_data)
        second_history = self.encoder(second_data)
        bundles = {
            model.target_module: (
                model._bundle_from_history(first_data, first_history),
                model._bundle_from_history(second_data, second_history),
            )
            for model in self._idea_models
        }
        first_graph = self.first._bundle_from_history(first_data, first_history)["graph"]
        second_graph = self.first._bundle_from_history(second_data, second_history)["graph"]
        auxiliaries: list[tuple[ModularResearchIdeaBalanceGCL, Tensor]] = []

        # 上游表示模块按 M2、M3 的真实顺序改写图表示。
        for module_id in ("M2", "M3"):
            model = self._by_module.get(module_id)
            if model is None:
                continue
            first_graph, second_graph, auxiliary = self._apply_idea_transform(
                model, *bundles[module_id], first_graph, second_graph
            )
            auxiliaries.append((model, auxiliary))

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

        model = self._by_module.get("M7")
        if model is not None:
            m7_second = positive_input if "M5" in self._by_module else second_graph
            anchor, positive, auxiliary = self._apply_idea_transform(
                model, *bundles["M7"], first_graph, m7_second
            )
            anchor = F.normalize(anchor, dim=-1)
            positive = F.normalize(positive, dim=-1)
            hard = self._m7_hard_negative(model, hard_input, bundles["M7"][0])
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
        auxiliary_values = {idea_id: 0.0 for idea_id in self.idea_ids}
        for idea_model, auxiliary in auxiliaries:
            weighted = self._weighted_auxiliary(idea_model, auxiliary)
            auxiliary_total = auxiliary_total + weighted
            auxiliary_values[idea_model.spec.idea_id] = float(auxiliary.detach())
        total = objective + auxiliary_total
        return total, {
            "module_objective": float(objective.detach()),
            "auxiliary_total": float(auxiliary_total.detach()),
            "first_auxiliary": auxiliary_values[self.idea_ids[0]],
            "second_auxiliary": auxiliary_values[self.idea_ids[1]],
            "third_auxiliary": auxiliary_values[self.idea_ids[2]],
            "temperature": float(temperature),
        }


def build_modular_triple_research_idea_model(
    idea_ids: tuple[str, str, str],
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
) -> ModularTripleResearchIdeaBalanceGCL:
    return ModularTripleResearchIdeaBalanceGCL(
        idea_ids,
        in_dim,
        hidden_dim,
        layers,
        num_classes,
        idea_parameters,
    )
