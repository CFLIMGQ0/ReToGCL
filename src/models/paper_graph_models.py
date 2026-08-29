"""十篇新增论文模型的统一 PyG 图分类适配层。

第三方官方源码保存在 ``src/sota_sources``，本文件不修改这些仓库。适配层保留
各论文的核心计算单元，并提供统一的 ``forward -> (logits, embedding, regularizer)``
接口，供同一套分层五折协议调用。
"""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import add_self_loops, degree, scatter


METHODS = (
    "maxcutpool",
    "simplicial_mp",
    "del",
    "cellclat",
    "plstm",
    "abstaingnn",
    "edgeprompt",
    "dualprism",
    "invgnn",
    "genvsexp",
)
CONTRASTIVE_METHODS = {"cellclat"}
PROMPT_METHODS = {"edgeprompt"}
SUPERVISED_METHODS = set(METHODS) - CONTRASTIVE_METHODS - PROMPT_METHODS


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )


def _gin(in_dim: int, out_dim: int) -> GINConv:
    return GINConv(_mlp(in_dim, out_dim), train_eps=True)


class GraphHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MaxCutPoolClassifier(nn.Module):
    """GIN + 特征感知 MaxCut 打分、稀疏选择和辅助割损失。"""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.pre = _gin(in_dim, hidden_dim)
        self.score_convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, normalize=False),
            GCNConv(hidden_dim, hidden_dim, normalize=False),
            GCNConv(hidden_dim, hidden_dim, normalize=False),
        ])
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.mix = nn.Linear(2 * hidden_dim, hidden_dim)
        self.head = GraphHead(hidden_dim, num_classes)

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        x = F.elu(self.pre(data.x, data.edge_index))
        score_x = x
        for conv in self.score_convs:
            score_x = torch.tanh(conv(score_x, data.edge_index))
        scores = torch.tanh(self.score(score_x)).squeeze(-1)

        selected = []
        graph_count = int(data.batch.max()) + 1
        for graph_id in range(graph_count):
            indices = (data.batch == graph_id).nonzero(as_tuple=False).flatten()
            count = max(1, math.floor(indices.numel() * 0.5))
            selected.append(indices[scores[indices].topk(count).indices])
        selected = torch.cat(selected)
        pooled_cut = global_add_pool(
            x[selected] * scores[selected].abs().unsqueeze(-1), data.batch[selected], size=graph_count
        )
        pooled_all = global_mean_pool(x, data.batch, size=graph_count)
        embedding = F.relu(self.mix(torch.cat([pooled_cut, pooled_all], dim=-1)))

        source, target = data.edge_index
        edge_products = scores[source] * scores[target]
        edge_batch = data.batch[source]
        cut_sum = scatter(edge_products, edge_batch, dim=0, dim_size=graph_count, reduce="sum")
        edge_count = scatter(torch.ones_like(edge_products), edge_batch, dim=0, dim_size=graph_count, reduce="sum")
        cut_loss = (cut_sum / edge_count.clamp_min(1)).mean()
        return self.head(embedding), embedding, cut_loss


class CellMessageLayer(nn.Module):
    """在 0-胞腔（节点）与 1-胞腔（边）之间传递边界/上边界消息。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_update = _mlp(3 * hidden_dim, hidden_dim)
        self.node_update = _mlp(2 * hidden_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: Tensor, edges: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        new_edges = self.edge_update(torch.cat([edges, nodes[source], nodes[target]], dim=-1))
        new_edges = self.edge_norm(edges + new_edges)
        incident = scatter(new_edges, target, dim=0, dim_size=nodes.size(0), reduce="mean")
        incident = incident + scatter(new_edges, source, dim=0, dim_size=nodes.size(0), reduce="mean")
        new_nodes = self.node_update(torch.cat([nodes, incident], dim=-1))
        return self.node_norm(nodes + new_nodes), new_edges


class SimplicialMessagePassingClassifier(nn.Module):
    """基于关系结构的节点/边胞腔消息传递图分类器。"""

    def __init__(self, in_dim: int, hidden_dim: int, layers: int, num_classes: int):
        super().__init__()
        self.node_projection = nn.Linear(in_dim, hidden_dim)
        self.edge_projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.layers = nn.ModuleList([CellMessageLayer(hidden_dim) for _ in range(layers)])
        self.fuse = nn.Linear(2 * hidden_dim, hidden_dim)
        self.head = GraphHead(hidden_dim, num_classes)

    def encode(self, data: Data) -> Tensor:
        nodes = F.relu(self.node_projection(data.x))
        source, target = data.edge_index
        edges = F.relu(self.edge_projection(torch.cat([nodes[source], nodes[target]], dim=-1)))
        for layer in self.layers:
            nodes, edges = layer(nodes, edges, data.edge_index)
        graph_count = int(data.batch.max()) + 1
        edge_batch = data.batch[source]
        node_pool = global_add_pool(nodes, data.batch, size=graph_count)
        edge_pool = global_mean_pool(edges, edge_batch, size=graph_count)
        return F.relu(self.fuse(torch.cat([node_pool, edge_pool], dim=-1)))

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        embedding = self.encode(data)
        return self.head(embedding), embedding, embedding.new_zeros(())


class DistributionalEdgeConv(MessagePassing):
    def __init__(self, hidden_dim: int):
        super().__init__(aggr="add")
        self.layout_net = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.update_net = _mlp(2 * hidden_dim, hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        source, target = edge_index
        deg = degree(source, x.size(0), dtype=x.dtype).clamp_min(1)
        degree_gap = (deg[source] - deg[target]).abs() / (deg[source] + deg[target])
        # 完全相同的端点特征会产生零距离；直接对零做 sqrt 的反向传播会出现
        # 无穷梯度，并在后续层扩散成 NaN。下界只用于数值稳定，不改变布局排序。
        feature_gap = (x[source] - x[target]).pow(2).mean(dim=-1).clamp_min(1e-12).sqrt()
        cosine = 1 - F.cosine_similarity(x[source], x[target], dim=-1)
        radial = 1 / (1 + feature_gap)
        layout = torch.stack([degree_gap, feature_gap, cosine, radial], dim=-1)
        return self.propagate(edge_index, x=x, layout=layout)

    def message(self, x_j: Tensor, layout: Tensor) -> Tensor:
        return x_j * torch.sigmoid(self.layout_net(layout))

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return self.update_net(torch.cat([x, aggr_out], dim=-1))


class DELClassifier(nn.Module):
    """以多统计量边布局分布调制消息的 DEL 适配。"""

    def __init__(self, in_dim: int, hidden_dim: int, layers: int, num_classes: int):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([DistributionalEdgeConv(hidden_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(layers)])
        self.head = GraphHead(hidden_dim, num_classes)

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        x = F.relu(self.input(data.x))
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, data.edge_index)))
        embedding = global_add_pool(x, data.batch)
        return self.head(embedding), embedding, embedding.new_zeros(())


class CellCLAT(nn.Module):
    """胞腔编码器、可学习冗余裁剪及 SimGRACE 风格对比目标。"""

    def __init__(self, in_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.encoder = SimplicialMessagePassingClassifier(in_dim, hidden_dim, layers, hidden_dim)
        self.cell_selector = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.projector = _mlp(hidden_dim, hidden_dim)

    def encode(self, data: Data, trim: bool = False) -> Tensor:
        nodes = F.relu(self.encoder.node_projection(data.x))
        source, target = data.edge_index
        edges = F.relu(self.encoder.edge_projection(torch.cat([nodes[source], nodes[target]], dim=-1)))
        if trim:
            keep = torch.sigmoid(self.cell_selector(edges))
            edges = edges * F.dropout(keep, p=0.2, training=self.training)
        for layer in self.encoder.layers:
            nodes, edges = layer(nodes, edges, data.edge_index)
        graph_count = int(data.batch.max()) + 1
        edge_batch = data.batch[source]
        return F.relu(self.encoder.fuse(torch.cat([
            global_add_pool(nodes, data.batch, size=graph_count),
            global_mean_pool(edges, edge_batch, size=graph_count),
        ], dim=-1)))

    def contrastive_loss(self, data: Data) -> Tensor:
        first = self.projector(self.encode(data, trim=True))
        second = self.projector(self.encode(data, trim=True))
        first, second = F.normalize(first, dim=-1), F.normalize(second, dim=-1)
        logits = first @ second.t() / 0.2
        labels = torch.arange(logits.size(0), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


class SourceTransitionMarkLayer(nn.Module):
    """稳定的 Source/Transition/Mark 门控 DAG 消息层。"""

    def __init__(self, hidden_dim: int, reverse: bool):
        super().__init__()
        self.reverse = reverse
        self.source = nn.Linear(hidden_dim, hidden_dim)
        self.transition = nn.Linear(2 * hidden_dim, hidden_dim)
        self.mark = nn.Linear(hidden_dim, hidden_dim)
        self.direct = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        source, target = edge_index
        forward = source < target
        mask = ~forward if self.reverse else forward
        source, target = source[mask], target[mask]
        if source.numel() == 0:
            return x
        transition = torch.sigmoid(self.transition(torch.cat([x[source], x[target]], dim=-1)))
        message = torch.sigmoid(self.source(x[source])) * transition * self.value(x[source])
        aggregate = scatter(message, target, dim=0, dim_size=x.size(0), reduce="mean")
        update = torch.sigmoid(self.mark(x)) * aggregate + torch.sigmoid(self.direct(x)) * self.value(x)
        return self.norm(x + update)


class pLSTMClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int, num_classes: int):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            SourceTransitionMarkLayer(hidden_dim, reverse=bool(index % 2)) for index in range(layers)
        ])
        self.head = GraphHead(hidden_dim, num_classes)

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        x = self.input(data.x)
        deg = degree(data.edge_index[0], x.size(0), dtype=x.dtype).unsqueeze(-1)
        scales = torch.exp(torch.arange(0, x.size(1), 2, device=x.device) * (-math.log(10000) / x.size(1)))
        positional = torch.zeros_like(x)
        positional[:, 0::2] = torch.sin(deg * scales)
        positional[:, 1::2] = torch.cos(deg * scales[: positional[:, 1::2].size(1)])
        x = x + positional
        for layer in self.layers:
            x = layer(x, data.edge_index)
        embedding = global_mean_pool(x, data.batch)
        return self.head(embedding), embedding, embedding.new_zeros(())


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden_dim)] + [GCNConv(hidden_dim, hidden_dim) for _ in range(layers - 1)])

    def forward(self, data: Data) -> tuple[Tensor, Tensor]:
        x = data.x
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))
        return x, global_add_pool(x, data.batch)


class AbstainGNN(nn.Module):
    """三层预测函数与一层图级拒识函数。"""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.predictor = GCNEncoder(in_dim, hidden_dim, 3)
        self.confidence = GCNEncoder(in_dim, hidden_dim, 1)
        self.head = GraphHead(hidden_dim, num_classes)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        _, embedding = self.predictor(data)
        _, confidence_embedding = self.confidence(data)
        logits = self.head(embedding)
        confidence = torch.sigmoid(self.confidence_head(confidence_embedding)).squeeze(-1)
        target = logits.detach().softmax(dim=-1).amax(dim=-1)
        calibration = F.binary_cross_entropy(confidence, target)
        return logits, embedding, calibration


class EdgePromptConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, edge_index: Tensor, edge_prompt: Tensor | None) -> Tensor:
        return self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_prompt=edge_prompt))

    def message(self, x_j: Tensor, edge_prompt: Tensor | None) -> Tensor:
        return F.relu(x_j if edge_prompt is None else x_j + edge_prompt)


class EdgePromptClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int, num_classes: int, anchors: int = 5):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (layers - 1)
        self.convs = nn.ModuleList([EdgePromptConv(dims[i], hidden_dim) for i in range(layers)])
        self.anchor_prompts = nn.ParameterList([nn.Parameter(torch.randn(anchors, dim) / math.sqrt(dim)) for dim in dims])
        self.prompt_weights = nn.ModuleList([nn.Linear(2 * dim, anchors) for dim in dims])
        self.projector = _mlp(hidden_dim, hidden_dim)
        self.head = GraphHead(hidden_dim, num_classes)

    def encode(self, data: Data, use_prompt: bool) -> Tensor:
        x = data.x
        source, target = data.edge_index
        for index, conv in enumerate(self.convs):
            prompt = None
            if use_prompt:
                weights = F.softmax(F.leaky_relu(self.prompt_weights[index](torch.cat([x[source], x[target]], dim=-1))), dim=-1)
                prompt = weights @ self.anchor_prompts[index]
            x = F.relu(conv(x, data.edge_index, prompt))
        return global_mean_pool(x, data.batch)

    def pretrain_loss(self, first: Data, second: Data) -> Tensor:
        left = F.normalize(self.projector(self.encode(first, use_prompt=False)), dim=-1)
        right = F.normalize(self.projector(self.encode(second, use_prompt=False)), dim=-1)
        logits = left @ right.t() / 0.2
        labels = torch.arange(logits.size(0), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

    def freeze_encoder(self) -> None:
        for module in (self.convs, self.projector):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        embedding = self.encode(data, use_prompt=True)
        return self.head(embedding), embedding, embedding.new_zeros(())


class DualPrismClassifier(nn.Module):
    """用于谱域图增强训练的五层 GIN 分类器。"""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.convs = nn.ModuleList([_gin(in_dim, hidden_dim)] + [_gin(hidden_dim, hidden_dim) for _ in range(4)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(5)])
        self.head = GraphHead(hidden_dim, num_classes)

    @staticmethod
    def spectral_view(data: Data, probability: float = 0.2) -> Data:
        """以归一化度频率为谱代理，屏蔽高频边并保持批归属不变。"""
        output = copy.copy(data)
        source, target = data.edge_index
        deg = degree(source, data.num_nodes, dtype=torch.float32).clamp_min(1)
        frequency = (deg[source] - deg[target]).abs() / (deg[source] + deg[target])
        random_mask = torch.rand_like(frequency) > probability * (0.5 + frequency)
        if random_mask.any():
            output.edge_index = data.edge_index[:, random_mask]
            if getattr(data, "edge_attr", None) is not None:
                output.edge_attr = data.edge_attr[random_mask]
        return output

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        x = data.x
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, data.edge_index)))
        embedding = global_mean_pool(x, data.batch)
        return self.head(embedding), embedding, embedding.new_zeros(())


class Invertible1x1(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        orthogonal = torch.linalg.qr(torch.randn(dim, dim)).Q
        lu, pivots = torch.linalg.lu_factor(orthogonal)
        p, lower, upper = torch.lu_unpack(lu, pivots)
        self.register_buffer("permutation", p)
        self.lower = nn.Parameter(lower)
        self.diagonal = nn.Parameter(upper.diag())
        self.upper = nn.Parameter(torch.triu(upper, diagonal=1))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        dim = x.size(-1)
        lower = torch.tril(self.lower, diagonal=-1) + torch.eye(dim, device=x.device)
        upper = torch.triu(self.upper, diagonal=1) + torch.diag(self.diagonal)
        return x @ (self.permutation @ lower @ upper) + self.bias


class InvGNNClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int, num_classes: int):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.transforms = nn.ModuleList([Invertible1x1(hidden_dim) for _ in range(layers)])
        self.head = GraphHead(hidden_dim, num_classes)

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
        source, target = edge_index
        deg = degree(target, data.num_nodes, dtype=data.x.dtype).clamp_min(1)
        norm = deg[source].rsqrt() * deg[target].rsqrt()
        x = self.input(data.x)
        for transform in self.transforms:
            x = torch.sigmoid(transform(x))
            x = scatter(x[source] * norm.unsqueeze(-1), target, dim=0, dim_size=x.size(0), reduce="sum")
        embedding = global_add_pool(x, data.batch)
        regularizer = sum(-torch.log(layer.diagonal.abs().clamp_min(1e-6)).mean() for layer in self.transforms)
        return self.head(embedding), embedding, 1e-4 * regularizer


class GenVsExpClassifier(nn.Module):
    """论文真实图实验采用的 MPNN + 任务相关 cycle-basis 结构编码。"""

    def __init__(self, in_dim: int, hidden_dim: int, layers: int, num_classes: int):
        super().__init__()
        self.input = nn.Linear(in_dim + 2, hidden_dim)
        self.convs = nn.ModuleList([_gin(hidden_dim, hidden_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(layers)])
        self.head = GraphHead(hidden_dim, num_classes)

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        cycle_pe = getattr(data, "cycle_pe", data.x.new_zeros((data.num_nodes, 2)))
        x = F.relu(self.input(torch.cat([data.x, cycle_pe], dim=-1)))
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, data.edge_index)))
        embedding = global_add_pool(x, data.batch)
        return self.head(embedding), embedding, embedding.new_zeros(())


def build_model(method: str, in_dim: int, hidden_dim: int, layers: int, num_classes: int) -> nn.Module:
    if method == "maxcutpool":
        return MaxCutPoolClassifier(in_dim, hidden_dim, num_classes)
    if method == "simplicial_mp":
        return SimplicialMessagePassingClassifier(in_dim, hidden_dim, layers, num_classes)
    if method == "del":
        return DELClassifier(in_dim, hidden_dim, layers, num_classes)
    if method == "cellclat":
        return CellCLAT(in_dim, hidden_dim, layers)
    if method == "plstm":
        return pLSTMClassifier(in_dim, hidden_dim, layers, num_classes)
    if method == "abstaingnn":
        return AbstainGNN(in_dim, hidden_dim, num_classes)
    if method == "edgeprompt":
        return EdgePromptClassifier(in_dim, hidden_dim, layers, num_classes)
    if method == "dualprism":
        return DualPrismClassifier(in_dim, hidden_dim, num_classes)
    if method == "invgnn":
        return InvGNNClassifier(in_dim, hidden_dim, layers, num_classes)
    if method == "genvsexp":
        return GenVsExpClassifier(in_dim, hidden_dim, layers, num_classes)
    raise ValueError(f"未知方法：{method}")
