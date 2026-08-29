"""共享编码器的两个研究 idea 联合原型。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from src.models.research_ideas import ResearchIdeaBalanceGCL, build_research_idea_model


class PairwiseResearchIdeaBalanceGCL(nn.Module):
    """固定两个 idea 的最优参数，在同一主干上联合优化两个专属分支。"""

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
        auxiliary_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.first = build_research_idea_model(
            first_idea,
            in_dim,
            hidden_dim,
            layers,
            num_classes,
            auxiliary_weight=auxiliary_weight,
            idea_parameters=first_parameters,
        )
        self.second = build_research_idea_model(
            second_idea,
            in_dim,
            hidden_dim,
            layers,
            num_classes,
            auxiliary_weight=auxiliary_weight,
            idea_parameters=second_parameters,
        )
        # 两个 idea 是同一模型上的模块，不是两个独立编码器的集成。
        self.second.encoder = self.first.encoder
        self.second.projector = self.first.projector
        self.second.semantic_prototypes = self.first.semantic_prototypes
        self.output_dim = self.first.output_dim
        self.auxiliary_weight = float(auxiliary_weight)

    @property
    def idea_ids(self) -> tuple[str, str]:
        return self.first.spec.idea_id, self.second.spec.idea_id

    def encode(self, data: Data) -> Tensor:
        return self.first.encode(data)

    def idea_loss(self, first_data: Data, second_data: Data) -> tuple[Tensor, dict[str, float]]:
        # 主干只前向一次；两个分支从同一节点历史构造各自视图。
        first_history = self.first.encoder(first_data)
        second_history = self.first.encoder(second_data)
        first_a = self.first._bundle_from_history(first_data, first_history)
        first_b = self.first._bundle_from_history(second_data, second_history)
        second_a = self.second._bundle_from_history(first_data, first_history)
        second_b = self.second._bundle_from_history(second_data, second_history)

        base = self.first.contrastive_loss(first_a["graph"], first_b["graph"], 0.2)
        first_contrast, first_auxiliary, first_diagnostics = self.first.idea_branch_terms(
            first_a, first_b
        )
        second_contrast, second_auxiliary, second_diagnostics = self.second.idea_branch_terms(
            second_a, second_b
        )
        first_term = first_contrast + first_auxiliary
        second_term = second_contrast + second_auxiliary
        total = base + self.auxiliary_weight * (first_term + second_term)
        diagnostics = {
            "base": float(base.detach()),
            "first_idea_term": float(first_term.detach()),
            "second_idea_term": float(second_term.detach()),
            "first_idea_contrast": first_diagnostics["idea_contrast"],
            "second_idea_contrast": second_diagnostics["idea_contrast"],
            "first_auxiliary": first_diagnostics["auxiliary"],
            "second_auxiliary": second_diagnostics["auxiliary"],
            "auxiliary_weight_per_idea": self.auxiliary_weight,
        }
        return total, diagnostics


def build_pairwise_research_idea_model(
    first_idea: str,
    second_idea: str,
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    first_parameters: dict[str, float],
    second_parameters: dict[str, float],
    auxiliary_weight: float = 0.25,
) -> PairwiseResearchIdeaBalanceGCL:
    return PairwiseResearchIdeaBalanceGCL(
        first_idea,
        second_idea,
        in_dim,
        hidden_dim,
        layers,
        num_classes,
        first_parameters,
        second_parameters,
        auxiliary_weight,
    )
