"""五个新增 SOTA 方法的 PyTorch Geometric 兼容实现。

官方仓库原样保存在 ``src/sota_sources``。本文件仅负责把论文中的核心模块接入
项目统一的数据读取和分层五折协议，避免修改第三方源码和混淆审计提交。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from torch_geometric.utils import scatter, to_dense_batch

from src.models.balancegcl_modules import (
    BalanceGCLContrastiveObjective,
    BalanceGCLNegativeBuilder,
    BalanceGCLPositiveBuilder,
    BalanceGCLProjectionHead,
    BalanceGCLReadout,
    BalanceGCLSemanticRouter,
)


def _gin_conv(in_dim: int, out_dim: int) -> GINConv:
    return GINConv(
        nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    )


class GINHistory(nn.Module):
    """返回每层节点激活，供 HISTOGRAPH 和其他方法复用。"""

    def __init__(self, in_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.input_projection = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([_gin_conv(hidden_dim, hidden_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(layers)])

    def forward(self, data: Data) -> Tensor:
        x = self.input_projection(data.x)
        history = [x]
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, data.edge_index)))
            history.append(x)
        return torch.stack(history, dim=1)


class HISTOGRAPH(nn.Module):
    """论文的逐层注意力、节点注意力及均值池化残差。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim 必须能被 heads 整除")
        self.backbone = GINHistory(in_dim, hidden_dim, layers)
        self.history_projection = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.node_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _positional_encoding(length: int, dim: int, device: torch.device) -> Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        scales = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        encoding = torch.zeros(length, dim, device=device)
        encoding[:, 0::2] = torch.sin(positions * scales)
        encoding[:, 1::2] = torch.cos(positions * scales[: encoding[:, 1::2].shape[1]])
        return encoding

    def encode(self, data: Data) -> Tensor:
        history = self.backbone(data)
        history = self.history_projection(history)
        history = history + self._positional_encoding(
            history.size(1), history.size(2), history.device
        ).unsqueeze(0)

        query = self.query(history[:, -1])
        keys = self.key(history)
        node_layer_scores = (query.unsqueeze(1) * keys).sum(dim=-1) / math.sqrt(history.size(-1))
        graph_layer_scores = global_mean_pool(node_layer_scores, data.batch)
        denominator = graph_layer_scores.sum(dim=-1, keepdim=True)
        signed_epsilon = torch.where(denominator >= 0, 1e-6, -1e-6)
        denominator = torch.where(denominator.abs() < 1e-6, signed_epsilon, denominator)
        layer_weights = graph_layer_scores / denominator
        node_weights = layer_weights[data.batch]
        historical_nodes = (history * node_weights.unsqueeze(-1)).sum(dim=1)

        dense_nodes, valid = to_dense_batch(historical_nodes, data.batch)
        attended, _ = self.node_attention(
            dense_nodes, dense_nodes, dense_nodes, key_padding_mask=~valid
        )
        attended = self.layer_norm(attended + dense_nodes)
        graph_history = (attended * valid.unsqueeze(-1)).sum(dim=1)
        graph_history = graph_history / valid.sum(dim=1, keepdim=True).clamp_min(1)
        graph_last = global_mean_pool(history[:, -1], data.batch)
        alpha = torch.sigmoid(self.mix_logit)
        return alpha * graph_history + (1 - alpha) * graph_last

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        representation = self.encode(data)
        return self.classifier(representation), representation, representation.new_zeros(())


class UniImb(nn.Module):
    """多尺度拓扑编码、个性化扰动和动态平衡原型。"""

    def __init__(
        self,
        in_dim: int,
        topology_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        prototypes: int = 16,
        topk_perception: int = 16,
        topk_balance: int = 8,
    ):
        super().__init__()
        self.feature_projection = nn.Linear(in_dim + topology_dim, hidden_dim)
        self.local_projection = nn.Linear(topology_dim, hidden_dim)
        self.feature_convs = nn.ModuleList(
            [_gin_conv(2 * hidden_dim, hidden_dim) for _ in range(layers)]
        )
        self.local_convs = nn.ModuleList(
            [_gin_conv(hidden_dim, hidden_dim) for _ in range(layers)]
        )
        self.feature_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(layers)])
        self.local_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(layers)])
        self.edge_drop_predictor = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        self.feature_mask_predictor = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        self.prototypes = nn.Parameter(torch.randn(prototypes, hidden_dim) / math.sqrt(hidden_dim))
        self.prototype_bias = nn.Parameter(torch.zeros(prototypes))
        self.topk_perception = topk_perception
        self.topk_balance = min(topk_balance, prototypes)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def _personalized_view(self, data: Data) -> tuple[Tensor, Tensor]:
        x, edge_index = data.x, data.edge_index
        if not self.training:
            return x, edge_index
        graph_count = int(data.batch.max()) + 1
        source = edge_index[0]
        degrees = scatter(
            torch.ones_like(source, dtype=x.dtype), source, dim=0,
            dim_size=x.size(0), reduce="sum",
        )
        mean_degree = scatter(degrees, data.batch, dim=0, dim_size=graph_count, reduce="mean")
        edge_probability = torch.sigmoid(self.edge_drop_predictor(mean_degree[:, None])).squeeze(-1)
        edge_probability = edge_probability.clamp(0.02, 0.5)
        edge_graph = data.batch[source]
        edge_keep = torch.rand(edge_index.size(1), device=x.device) > edge_probability[edge_graph]

        graph_scale = scatter(x.abs().mean(dim=-1), data.batch, dim=0, dim_size=graph_count, reduce="mean")
        feature_probability = torch.sigmoid(
            self.feature_mask_predictor(graph_scale[:, None])
        ).squeeze(-1).clamp(0.02, 0.5)
        node_keep = torch.rand(x.size(0), device=x.device) > feature_probability[data.batch]
        return x * node_keep.unsqueeze(-1), edge_index[:, edge_keep]

    def encode(self, data: Data) -> Tensor:
        x, edge_index = self._personalized_view(data)
        lap_pe = getattr(data, "lap_pe")
        rw_pe = getattr(data, "rw_pe")
        feature = self.feature_projection(torch.cat([x, lap_pe], dim=-1))
        local = self.local_projection(rw_pe)
        for feature_conv, local_conv, feature_norm, local_norm in zip(
            self.feature_convs, self.local_convs, self.feature_norms, self.local_norms
        ):
            feature = F.relu(feature_norm(feature_conv(torch.cat([feature, local], dim=-1), edge_index)))
            local = F.relu(local_norm(local_conv(local, edge_index)))
        return global_add_pool(feature, data.batch)

    def _dynamic_balanced_prototypes(self, representation: Tensor) -> tuple[Tensor, Tensor]:
        count = representation.size(0)
        if self.training:
            permutation = torch.randperm(count, device=representation.device)
            coefficient = torch.rand(count, 1, device=representation.device)
            mixed = coefficient * representation + (1 - coefficient) * representation[permutation]
            all_graphs = torch.cat([representation, mixed], dim=0)
        else:
            all_graphs = representation

        perception = self.prototypes @ all_graphs.t() / math.sqrt(all_graphs.size(-1))
        perception_k = min(self.topk_perception, all_graphs.size(0))
        perception_mask = torch.zeros_like(perception, dtype=torch.bool)
        perception_mask.scatter_(1, perception.topk(perception_k, dim=1).indices, True)
        perception = perception.masked_fill(~perception_mask, -torch.inf).softmax(dim=1)
        dynamic = perception @ all_graphs

        affinity = all_graphs @ dynamic.t() / math.sqrt(all_graphs.size(-1))
        affinity = affinity + self.prototype_bias
        balance_mask = torch.zeros_like(affinity, dtype=torch.bool)
        balance_mask.scatter_(1, affinity.topk(self.topk_balance, dim=1).indices, True)
        weights = torch.sigmoid(affinity).masked_fill(~balance_mask, 0)
        enhanced = all_graphs + weights @ dynamic / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        soft_load = affinity.softmax(dim=-1).mean(dim=0)
        uniform = torch.full_like(soft_load, 1 / soft_load.numel())
        balance_loss = F.mse_loss(soft_load, uniform)
        return enhanced[:count], balance_loss

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        representation = self.encode(data)
        enhanced, balance_loss = self._dynamic_balanced_prototypes(representation)
        return self.classifier(enhanced), enhanced, balance_loss


class DiffLift(nn.Module):
    """以可微自适应超边提升和超图消息传递实例化 DiffLift。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        k_max: int = 5,
    ):
        super().__init__()
        if k_max < 2:
            raise ValueError("k_max 至少为 2")
        self.k_max = k_max
        self.input_projection = nn.Linear(in_dim, hidden_dim)
        self.gnn = nn.ModuleList([_gin_conv(hidden_dim, hidden_dim) for _ in range(layers)])
        self.gnn_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(layers)])
        self.size_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, k_max - 1)
        )
        self.accept_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.tnn_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
             for _ in range(2)]
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _lift_batch(
        self, nodes: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """并行构造一个 mini-batch 中的候选高阶单元。

        所有图先填充为稠密张量，掩码保证不同图之间没有距离或消息交互。
        这与逐图版本使用相同的自适应 k、Gumbel-Softmax 和 Bernoulli-ST，
        但避免每轮为数千张图启动独立的小 CUDA kernel。
        """
        batch_size, node_count, hidden_dim = nodes.shape
        neighbor_count = min(self.k_max - 1, max(node_count - 1, 0))
        if neighbor_count == 0:
            empty_membership = nodes.new_zeros(batch_size, node_count, 0)
            empty_nearest = torch.empty(
                batch_size, node_count, 0, dtype=torch.long, device=nodes.device
            )
            return empty_membership, empty_nearest, valid.to(nodes.dtype), nodes

        distances = torch.cdist(nodes, nodes)
        distances = distances.masked_fill(~valid[:, None, :], torch.inf)
        diagonal = torch.eye(node_count, dtype=torch.bool, device=nodes.device).unsqueeze(0)
        distances = distances.masked_fill(diagonal, torch.inf)
        nearest = distances.topk(neighbor_count, largest=False).indices
        nearest_valid = valid.gather(1, nearest.flatten(1)).view_as(nearest)

        logits = self.size_network(nodes)[..., :neighbor_count]
        if self.training:
            size_choice = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        else:
            size_choice = F.one_hot(
                logits.argmax(dim=-1), num_classes=neighbor_count
            ).to(logits.dtype)
        # 第 r 个近邻在选择大小 >= r+1 时加入超边。
        membership = torch.flip(
            torch.cumsum(torch.flip(size_choice, dims=[-1]), dim=-1), dims=[-1]
        )
        membership = membership * nearest_valid * valid.unsqueeze(-1)

        gather_index = nearest.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
        expanded_nodes = nodes.unsqueeze(1).expand(-1, node_count, -1, -1)
        neighbor_nodes = expanded_nodes.gather(2, gather_index)
        candidate = nodes + (neighbor_nodes * membership.unsqueeze(-1)).sum(dim=2)
        candidate = candidate / (1 + membership.sum(dim=2, keepdim=True))
        probability = torch.sigmoid(self.accept_network(candidate)).squeeze(-1)
        probability = probability * valid
        if self.training:
            sampled = torch.bernoulli(probability)
            gate = sampled + probability - probability.detach()
        else:
            gate = (probability >= 0.5).to(probability.dtype) * valid
            no_selected_cell = (gate.sum(dim=1, keepdim=True) == 0) & valid.any(
                dim=1, keepdim=True
            )
            gate = torch.where(no_selected_cell, probability, gate)
        return membership, nearest, gate, candidate

    @staticmethod
    def _hypergraph_message(
        nodes: Tensor,
        membership: Tensor,
        nearest: Tensor,
        gate: Tensor,
        candidate: Tensor,
    ) -> Tensor:
        if nearest.numel() == 0:
            return nodes
        candidate = candidate * gate.unsqueeze(-1)
        aggregate = candidate.clone()
        normalizer = gate.clone()
        for rank in range(nearest.size(2)):
            weight = gate * membership[:, :, rank]
            target = nearest[:, :, rank]
            aggregate.scatter_add_(
                1,
                target.unsqueeze(-1).expand(-1, -1, nodes.size(-1)),
                candidate * membership[:, :, rank : rank + 1],
            )
            normalizer.scatter_add_(1, target, weight)
        return aggregate / normalizer.unsqueeze(-1).clamp_min(1e-6)

    def encode(self, data: Data) -> Tensor:
        nodes = self.input_projection(data.x)
        for conv, norm in zip(self.gnn, self.gnn_norms):
            nodes = F.relu(norm(conv(nodes, data.edge_index)))

        graph_nodes, valid = to_dense_batch(nodes, data.batch)
        membership, nearest, gate, candidate = self._lift_batch(graph_nodes, valid)
        for layer in self.tnn_layers:
            message = self._hypergraph_message(
                graph_nodes, membership, nearest, gate, candidate
            )
            graph_nodes = graph_nodes + layer(torch.cat([graph_nodes, message], dim=-1))
            candidate = graph_nodes + (
                graph_nodes.unsqueeze(1)
                .expand(-1, graph_nodes.size(1), -1, -1)
                .gather(
                    2,
                    nearest.unsqueeze(-1).expand(-1, -1, -1, graph_nodes.size(-1)),
                )
                * membership.unsqueeze(-1)
            ).sum(dim=2)
            candidate = candidate / (1 + membership.sum(dim=2, keepdim=True))
        output = graph_nodes[valid]
        return global_mean_pool(output, data.batch)

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        representation = self.encode(data)
        return self.classifier(representation), representation, representation.new_zeros(())


class BalancedGCL(nn.Module):
    """BalanceGCL 的平衡反事实负样本与三分区语义正样本适配。"""

    def __init__(self, in_dim: int, hidden_dim: int, layers: int, num_classes: int):
        super().__init__()
        self.encoder = GINHistory(in_dim, hidden_dim, layers)
        self.output_dim = hidden_dim * (layers + 1)
        self.projector = BalanceGCLProjectionHead(self.output_dim)
        self.semantic_prototypes = nn.Parameter(
            torch.randn(num_classes, self.output_dim) / math.sqrt(self.output_dim)
        )
        # 无状态模块放在带参数组件之后构造，以保持旧版随机初始化顺序和
        # checkpoint 参数键。每个属性都是后续研究 idea 的唯一替换接口。
        self.readout = BalanceGCLReadout()
        self.semantic_router = BalanceGCLSemanticRouter()
        self.positive_builder = BalanceGCLPositiveBuilder()
        self.negative_builder = BalanceGCLNegativeBuilder()
        self.objective = BalanceGCLContrastiveObjective()

    def encode(self, data: Data) -> Tensor:
        history = self.encoder(data)
        return self.readout(history, data.batch)

    def contrastive_loss(self, first: Tensor, second: Tensor, temperature: float = 0.2) -> Tensor:
        anchor = F.normalize(self.projector(first), dim=-1)
        positive_base = F.normalize(self.projector(second), dim=-1)
        routing = self.semantic_router(first, self.semantic_prototypes)
        positive_input = self.positive_builder(first, second, positive_base, routing)
        hard_negative_input = self.negative_builder(first, routing)
        positive = F.normalize(self.projector(positive_input), dim=-1)
        hard = F.normalize(self.projector(hard_negative_input), dim=-1)
        return self.objective(anchor, positive, hard, temperature)


class CubicSplineLinear(nn.Module):
    """Khan-GCL 所需的三阶 B 样条 KAN 连接层。"""

    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, order: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.order = order
        core = torch.linspace(-2.0, 2.0, grid_size + 1)
        spacing = core[1] - core[0]
        left = core[0] - spacing * torch.arange(order, 0, -1)
        right = core[-1] + spacing * torch.arange(1, order + 1)
        self.register_buffer("knots", torch.cat([left, core, right]))
        basis_count = self.knots.numel() - order - 1
        self.coefficients = nn.Parameter(torch.randn(in_dim, out_dim, basis_count) * 0.05)
        self.base_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.base_weight)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def _basis(self, x: Tensor) -> Tensor:
        x = x.clamp(float(self.knots[0]) + 1e-6, float(self.knots[-1]) - 1e-6)
        basis = ((x.unsqueeze(-1) >= self.knots[:-1]) & (x.unsqueeze(-1) < self.knots[1:])).to(x.dtype)
        for degree in range(1, self.order + 1):
            left_den = self.knots[degree:-1] - self.knots[: -degree - 1]
            right_den = self.knots[degree + 1 :] - self.knots[1:-degree]
            left = (x.unsqueeze(-1) - self.knots[: -degree - 1]) / left_den.clamp_min(1e-12)
            right = (self.knots[degree + 1 :] - x.unsqueeze(-1)) / right_den.clamp_min(1e-12)
            basis = left * basis[..., :-1] + right * basis[..., 1:]
        return basis

    def forward(self, x: Tensor) -> Tensor:
        spline = torch.einsum("nib,iob->no", self._basis(x), self.coefficients)
        return F.silu(x) @ self.base_weight + spline + self.bias

    def critical_scores(self) -> Tensor:
        discriminative = self.coefficients.var(dim=-1, unbiased=False).mean(dim=0)
        flattened = self.coefficients.permute(1, 0, 2).flatten(1)
        independent = []
        for index in range(self.out_dim):
            others = torch.cat([flattened[:index], flattened[index + 1 :]], dim=0)
            if others.numel() == 0:
                independent.append(flattened[index].norm())
                continue
            solution = torch.linalg.lstsq(others.t(), flattened[index]).solution
            reconstructed = solution @ others
            independent.append((flattened[index] - reconstructed).norm())
        independent_score = torch.stack(independent)
        return discriminative + independent_score


class SplineMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.first = CubicSplineLinear(in_dim, hidden_dim)
        self.second = CubicSplineLinear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.second(F.relu(self.first(x)))


class KhanGCL(nn.Module):
    """KAN-GIN 编码器、CKFI 和定向潜空间硬负样本。"""

    def __init__(self, in_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(layers):
            current_in = in_dim if layer == 0 else hidden_dim
            self.convs.append(GINConv(SplineMLP(current_in, hidden_dim)))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.output_dim = hidden_dim * layers
        self.projector = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim), nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def encode(self, data: Data) -> Tensor:
        x = data.x
        graphs = []
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, data.edge_index)))
            graphs.append(global_add_pool(x, data.batch))
        return torch.cat(graphs, dim=-1)

    def critical_scores(self) -> Tensor:
        scores = [conv.nn.second.critical_scores() for conv in self.convs]
        score = torch.cat(scores)
        return (score - score.min()) / (score.max() - score.min()).clamp_min(1e-6)

    def contrastive_loss(self, first: Tensor, second: Tensor, temperature: float = 0.2) -> Tensor:
        first_projected = F.normalize(self.projector(first), dim=-1)
        second_projected = F.normalize(self.projector(second), dim=-1)
        logits = first_projected @ second_projected.t() / temperature
        labels = torch.arange(first.size(0), device=first.device)
        base_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

        score = self.critical_scores().unsqueeze(0).expand_as(first)
        perturbation = torch.normal(mean=0.2 * score, std=0.05)
        perturbation = perturbation * torch.where(
            torch.rand_like(perturbation) < 0.5, -torch.ones_like(perturbation), torch.ones_like(perturbation)
        )
        hard_negative = F.normalize(self.projector((first + perturbation).detach()), dim=-1)
        hard_negative_loss = (first_projected * hard_negative).sum(dim=-1).mean()
        return base_loss + hard_negative_loss


SUPERVISED_METHODS = {"histograph", "uniimb", "difflift"}
CONTRASTIVE_METHODS = {"balancegcl", "khangcl"}


def build_supervised_model(
    method: str,
    in_dim: int,
    hidden_dim: int,
    layers: int,
    num_classes: int,
    topology_dim: int,
) -> nn.Module:
    if method == "histograph":
        heads = 4 if hidden_dim % 4 == 0 else 1
        return HISTOGRAPH(in_dim, hidden_dim, layers, num_classes, heads=heads)
    if method == "uniimb":
        prototypes = 16 if num_classes <= 2 else 32
        return UniImb(
            in_dim, topology_dim, hidden_dim, layers, num_classes,
            prototypes=prototypes,
            topk_perception=min(16, prototypes),
            topk_balance=min(8, prototypes),
        )
    if method == "difflift":
        return DiffLift(in_dim, hidden_dim, layers, num_classes)
    raise ValueError(f"未知监督方法：{method}")


def build_contrastive_model(
    method: str, in_dim: int, hidden_dim: int, layers: int, num_classes: int
) -> nn.Module:
    if method == "balancegcl":
        return BalancedGCL(in_dim, hidden_dim, layers, num_classes)
    if method == "khangcl":
        return KhanGCL(in_dim, hidden_dim, layers)
    raise ValueError(f"未知对比方法：{method}")
