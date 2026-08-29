"""BalanceGCL 的可替换算法模块。

本文件只拆分原有计算，不改变公式。带参数的编码器仍由调用方构造，投影头
保持 ``nn.Sequential`` 的参数编号，以兼容已有 ``projector.0`` 和
``projector.2`` checkpoint 键；其余模块均为无状态算子。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool


@dataclass(frozen=True)
class BalanceGCLSemanticRouting:
    """语义原型路由器输出，供正、负样本构造器共同使用。"""

    pseudo_class: Tensor
    own_prototype: Tensor
    alternative_prototypes: Tensor
    positive_mask: Tensor
    negative_mask: Tensor
    neutral_mask: Tensor


class BalanceGCLReadout(nn.Module):
    """M3：逐层求和池化并拼接为图表示。"""

    def forward(self, history: Tensor, batch: Tensor) -> Tensor:
        return torch.cat(
            [global_add_pool(history[:, layer], batch) for layer in range(history.size(1))],
            dim=-1,
        )


class BalanceGCLProjectionHead(nn.Sequential):
    """M7：保持原参数顺序和键名的两层投影头。"""

    def __init__(self, dimension: int) -> None:
        super().__init__(
            nn.Linear(dimension, dimension),
            nn.ReLU(),
            nn.Linear(dimension, dimension),
        )


class BalanceGCLSemanticRouter(nn.Module):
    """M4：根据可学习原型产生伪类和逐维语义关系。"""

    def forward(self, representation: Tensor, semantic_prototypes: Tensor) -> BalanceGCLSemanticRouting:
        prototypes = F.normalize(semantic_prototypes, dim=-1)
        pseudo_class = (F.normalize(representation, dim=-1) @ prototypes.t()).argmax(dim=-1)

        own = prototypes[pseudo_class]
        alternatives = prototypes.unsqueeze(0).expand(representation.size(0), -1, -1)
        class_mask = F.one_hot(pseudo_class, prototypes.size(0)).bool()
        alternatives = alternatives[~class_mask].view(
            representation.size(0), prototypes.size(0) - 1, -1
        )

        positive_contribution = representation * own
        negative_contribution = (representation.unsqueeze(1) * alternatives).mean(dim=1)
        positive_mask = positive_contribution > negative_contribution + 0.01
        negative_mask = negative_contribution > positive_contribution + 0.01
        neutral_mask = ~(positive_mask | negative_mask)
        return BalanceGCLSemanticRouting(
            pseudo_class=pseudo_class,
            own_prototype=own,
            alternative_prototypes=alternatives,
            positive_mask=positive_mask,
            negative_mask=negative_mask,
            neutral_mask=neutral_mask,
        )


class BalanceGCLPositiveBuilder(nn.Module):
    """M5：按三分区语义掩码构造正样本。"""

    def forward(
        self,
        anchor: Tensor,
        second_view: Tensor,
        projected_second_view: Tensor,
        routing: BalanceGCLSemanticRouting,
    ) -> Tensor:
        semantic_positive = second_view.clone()
        semantic_positive = semantic_positive + 0.1 * routing.positive_mask * anchor
        semantic_positive = semantic_positive - 0.1 * routing.negative_mask * anchor
        semantic_positive = semantic_positive + 0.05 * routing.neutral_mask * torch.randn_like(anchor)
        return 0.5 * projected_second_view + 0.5 * semantic_positive


class BalanceGCLNegativeBuilder(nn.Module):
    """M6：沿其他类别原型方向生成反事实困难负样本。"""

    def forward(self, anchor: Tensor, routing: BalanceGCLSemanticRouting) -> Tensor:
        return anchor.unsqueeze(1) + 0.2 * (
            routing.alternative_prototypes - anchor.unsqueeze(1)
        )


class BalanceGCLContrastiveObjective(nn.Module):
    """M8：联合 Batch 负样本与反事实困难负样本的对比目标。"""

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        hard_negatives: Tensor,
        temperature: float,
    ) -> Tensor:
        positive_logit = (anchor * positive).sum(dim=-1, keepdim=True) / temperature
        batch_logits = anchor @ positive.t() / temperature
        hard_logits = torch.einsum("bd,bkd->bk", anchor, hard_negatives) / temperature
        denominator = torch.cat([batch_logits, hard_logits], dim=1).logsumexp(dim=1)
        return -(positive_logit.squeeze(1) - denominator).mean()
