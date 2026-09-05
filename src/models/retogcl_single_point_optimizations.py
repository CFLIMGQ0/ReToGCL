"""ReToGCL 全模型的十个严格单点优化候选。

每个候选都从同一个 ``D7-I10 + D7-I01 + D10-I04`` ReToGCL 构造，只替换
一个组件或一条连接。候选之间不累计，便于把性能差异归因到唯一改动。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn.norm import GraphNorm

from src.models.modular_research_ideas import ModularResearchIdeaBalanceGCL
from src.models.modular_triple_research_ideas import (
    ModularTripleResearchIdeaBalanceGCL,
)


# 第一轮 10 版本实验确认 0.06 优于原先由 ABC 参数推导出的 0.08944；
# 从 v2 起，O07 的正式默认温度固定为 0.06。旧结果目录仍保留为历史对照。
O07_DEFAULT_TEMPERATURE = 0.06


@dataclass(frozen=True)
class SinglePointOptimization:
    optimization_id: str
    name: str
    component: str
    change: str


OPTIMIZATIONS: dict[str, SinglePointOptimization] = {
    "O01": SinglePointOptimization(
        "O01", "GIN 恒等残差", "structure_encoder",
        "每层 GIN 更新改为等权恒等残差，其他参数和逐层输出不变。",
    ),
    "O02": SinglePointOptimization(
        "O02", "逐图 GraphNorm", "encoder_normalization",
        "仅把 GIN 的 BatchNorm 替换为按单图归一化的 GraphNorm。",
    ),
    "O03": SinglePointOptimization(
        "O03", "逐层均值读出", "graph_readout",
        "仅把每层节点求和池化替换成均值池化，仍按层拼接。",
    ),
    "O04": SinglePointOptimization(
        "O04", "平方根图规模读出", "graph_readout",
        "仅把每层求和除以节点数平方根，折中保留规模和密度信息。",
    ),
    "O05": SinglePointOptimization(
        "O05", "删除洁净辅助项", "auxiliary_objective",
        "仅移除 D7-I01 洁净辅助损失，洁净表示和路由仍完整保留。",
    ),
    "O06": SinglePointOptimization(
        "O06", "限制拓扑投影过冲", "projection_metric",
        "仅把 D10-I04 的表示残差增益从 4 限制为 1。",
    ),
    "O07": SinglePointOptimization(
        "O07", "困难负样本度量对齐", "hard_negative_metric",
        "仅让困难负样本通过与锚点相同的 D10-I04 拓扑切空间映射。",
    ),
    "O08": SinglePointOptimization(
        "O08", "双向平衡对比目标", "contrastive_objective",
        "仅把单向平衡对比损失替换为锚点与正样本双向平均。",
    ),
    "O09": SinglePointOptimization(
        "O09", "最相似批负样本剔除", "contrastive_objective",
        "仅从每个锚点的分母中剔除一个最相似非对角批负样本。",
    ),
    "O10": SinglePointOptimization(
        "O10", "对齐感知温度", "contrastive_temperature",
        "仅按当前正对齐置信度在原温度附近逐样本调整温度。",
    ),
}


class _ForwardGINVariant(nn.Module):
    """复用原 GIN 参数，仅改变每层前向连接。"""

    def __init__(self, source: nn.Module, mode: str) -> None:
        super().__init__()
        self.source = source
        self.mode = mode

    def forward(self, data: Data) -> Tensor:
        x = self.source.input_projection(data.x)
        history = [x]
        for conv, norm in zip(self.source.convs, self.source.norms):
            update = F.relu(norm(conv(x, data.edge_index)))
            if self.mode == "residual":
                x = (x + update) / math.sqrt(2.0)
            else:
                raise RuntimeError(self.mode)
            history.append(x)
        return torch.stack(history, dim=1)


class _GraphNormGIN(nn.Module):
    """保留原 GIN 卷积，仅替换归一化层。"""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.input_projection = source.input_projection
        self.convs = source.convs
        hidden_dim = source.input_projection.out_features
        self.norms = nn.ModuleList(
            GraphNorm(hidden_dim) for _ in range(len(source.convs))
        )

    def forward(self, data: Data) -> Tensor:
        x = self.input_projection(data.x)
        history = [x]
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, data.edge_index), data.batch))
            history.append(x)
        return torch.stack(history, dim=1)


class _MeanReadout(nn.Module):
    def forward(self, history: Tensor, batch: Tensor) -> Tensor:
        return torch.cat(
            [global_mean_pool(history[:, layer], batch) for layer in range(history.size(1))],
            dim=-1,
        )


class _SqrtSizeReadout(nn.Module):
    def forward(self, history: Tensor, batch: Tensor) -> Tensor:
        graph_count = int(batch.max().item()) + 1
        scale = torch.bincount(batch, minlength=graph_count).to(history.dtype)
        scale = scale.sqrt().clamp_min(1).unsqueeze(-1)
        return torch.cat(
            [
                global_add_pool(history[:, layer], batch, size=graph_count) / scale
                for layer in range(history.size(1))
            ],
            dim=-1,
        )


def _directional_objective(
    anchor: Tensor,
    positive: Tensor,
    hard_negatives: Tensor,
    temperature: float | Tensor,
) -> Tensor:
    if isinstance(temperature, Tensor) and temperature.ndim:
        divisor = temperature.unsqueeze(-1)
    else:
        divisor = temperature
    positive_logit = (anchor * positive).sum(-1, keepdim=True) / divisor
    batch_logits = (anchor @ positive.t()) / divisor
    hard_logits = torch.einsum("bd,bkd->bk", anchor, hard_negatives) / divisor
    denominator = torch.cat((batch_logits, hard_logits), dim=1).logsumexp(dim=1)
    return -(positive_logit.squeeze(1) - denominator).mean()


class _SymmetricObjective(nn.Module):
    def forward(
        self, anchor: Tensor, positive: Tensor, hard_negatives: Tensor,
        temperature: float,
    ) -> Tensor:
        forward = _directional_objective(
            anchor, positive, hard_negatives, temperature
        )
        backward = _directional_objective(
            positive, anchor, hard_negatives, temperature
        )
        return 0.5 * (forward + backward)


class _TrimNearestBatchNegativeObjective(nn.Module):
    def forward(
        self, anchor: Tensor, positive: Tensor, hard_negatives: Tensor,
        temperature: float,
    ) -> Tensor:
        positive_logit = (anchor * positive).sum(-1, keepdim=True) / temperature
        batch_logits = anchor @ positive.t() / temperature
        if batch_logits.size(0) > 1:
            diagonal = torch.eye(
                batch_logits.size(0), dtype=torch.bool, device=batch_logits.device
            )
            nearest = batch_logits.masked_fill(diagonal, -torch.inf).argmax(-1)
            batch_logits = batch_logits.clone()
            batch_logits[
                torch.arange(batch_logits.size(0), device=batch_logits.device), nearest
            ] = -torch.inf
        hard_logits = (
            torch.einsum("bd,bkd->bk", anchor, hard_negatives) / temperature
        )
        denominator = torch.cat((batch_logits, hard_logits), dim=1).logsumexp(1)
        return -(positive_logit.squeeze(1) - denominator).mean()


class _AlignmentAwareTemperatureObjective(nn.Module):
    def forward(
        self, anchor: Tensor, positive: Tensor, hard_negatives: Tensor,
        temperature: float,
    ) -> Tensor:
        confidence = ((anchor * positive).sum(-1).detach() + 1.0) * 0.5
        adaptive = (
            temperature * (1.25 - 0.5 * confidence.clamp(0, 1))
        ).clamp(min=0.05, max=0.30)
        return _directional_objective(anchor, positive, hard_negatives, adaptive)


class ReToGCLSinglePointOptimization(ModularTripleResearchIdeaBalanceGCL):
    """相对原 ReToGCL 恰好应用一个改动。"""

    protocol = "retogcl_single_point_optimization_v1"

    def __init__(
        self,
        optimization_id: str,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
    ) -> None:
        if optimization_id not in OPTIMIZATIONS:
            raise ValueError(f"未知单点优化：{optimization_id}")
        super().__init__(
            ("D7-I10", "D7-I01", "D10-I04"),
            in_dim, hidden_dim, layers, num_classes, idea_parameters,
        )
        self.optimization = OPTIMIZATIONS[optimization_id]

        if optimization_id == "O01":
            self._replace_shared("encoder", _ForwardGINVariant(self.encoder, "residual"))
        elif optimization_id == "O02":
            self._replace_shared("encoder", _GraphNormGIN(self.encoder))
        elif optimization_id == "O03":
            self._replace_shared("readout", _MeanReadout())
        elif optimization_id == "O04":
            self._replace_shared("readout", _SqrtSizeReadout())
        elif optimization_id == "O08":
            self._replace_shared("objective", _SymmetricObjective())
        elif optimization_id == "O09":
            self._replace_shared("objective", _TrimNearestBatchNegativeObjective())
        elif optimization_id == "O10":
            self._replace_shared("objective", _AlignmentAwareTemperatureObjective())

    def _replace_shared(self, name: str, module: nn.Module) -> None:
        for model in self._idea_models:
            setattr(model, name, module)

    def _temperature(self) -> float:
        if self.optimization.optimization_id == "O07":
            return O07_DEFAULT_TEMPERATURE
        return super()._temperature()

    def _weighted_auxiliary(
        self,
        model: ModularResearchIdeaBalanceGCL,
        auxiliary: Tensor,
    ) -> Tensor:
        if (
            self.optimization.optimization_id == "O05"
            and model.spec.idea_id == "D7-I01"
        ):
            return auxiliary.new_zeros(())
        return super()._weighted_auxiliary(model, auxiliary)

    def _apply_idea_transform(
        self,
        model: ModularResearchIdeaBalanceGCL,
        first_bundle: dict[str, Tensor],
        second_bundle: dict[str, Tensor],
        first_graph: Tensor,
        second_graph: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if (
            self.optimization.optimization_id != "O06"
            or model.spec.idea_id != "D10-I04"
        ):
            return super()._apply_idea_transform(
                model, first_bundle, second_bundle, first_graph, second_graph
            )

        first = self._with_graph(first_bundle, first_graph)
        second = self._with_graph(second_bundle, second_graph)
        transformed_first, transformed_second, auxiliary = model._family10(
            first, second, model.spec.variant
        )
        # 原最优 ABC 的倍率为 4；本候选只验证是否应限制为单位残差。
        gain = 1.0
        return (
            first_graph + gain * (transformed_first - first_graph),
            second_graph + gain * (transformed_second - second_graph),
            auxiliary,
        )

    def _m7_hard_negative(
        self,
        model: ModularResearchIdeaBalanceGCL,
        hard_input: Tensor,
        first_bundle: dict[str, Tensor],
    ) -> Tensor:
        if self.optimization.optimization_id != "O07":
            return super()._m7_hard_negative(model, hard_input, first_bundle)

        strata = first_bundle["views"][:, 2].mean(-1, keepdim=True).sign()
        mapped = model.bottleneck(hard_input) * strata.unsqueeze(1)
        gain = model._parameter_controls()[0]
        mapped = hard_input + gain * (mapped - hard_input)
        return F.normalize(mapped, dim=-1)


def build_retogcl_single_point_optimization(
    optimization_id: str,
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    idea_parameters: tuple[dict[str, float], dict[str, float], dict[str, float]],
) -> ReToGCLSinglePointOptimization:
    return ReToGCLSinglePointOptimization(
        optimization_id, in_dim, hidden_dim, layers, num_classes,
        idea_parameters,
    )
