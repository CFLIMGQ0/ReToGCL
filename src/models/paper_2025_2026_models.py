"""2025--2026 年二十篇图学习论文的统一 PyG 图分类适配层。

官方源码完整保存在 ``src/sota_sources_2025_2026``。这里将各论文核心机制接入
同一套 TU 数据集五折协议；这不是声称逐行复刻作者训练脚本，而是可审计、可批量
比较的项目适配实现。所有模型统一返回 ``(logits, graph_embedding, regularizer)``。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import degree, scatter, softmax, to_dense_batch


METHODS = (
    "gnnplus", "nodeid", "kagnn", "rankfeatures", "multinet",
    "rspool", "magedgepool", "hspgkn", "g2pm", "hyperenc",
    "dgda", "spectre", "prune", "rsnn", "toper",
    "leap", "hourglass", "topoformer", "tif", "git_tasktree",
)

METHOD_SPECS = {
    "gnnplus": ("Can Classic GNNs Be Strong Baselines for Graph-level Tasks?", "ICML", 2025, "0e02ad9acc2f1e54b5ad71c051bf5dfb1fcb4f28"),
    "nodeid": ("Node Identifiers: Compact, Discrete Representations for Efficient Graph Learning", "ICLR", 2025, "d3d5318e870b34bceeeb0453bd61cdaf7c939ff2"),
    "kagnn": ("KAGNNs: Kolmogorov-Arnold Networks meet Graph Learning", "TMLR", 2025, "06d6565e1299d1af7753048c186d2a3b6ebbd5b2"),
    "rankfeatures": ("Learning to Rank Features to Enhance Graph Neural Networks for Graph Classification", "TMLR", 2025, "d30eda740b4da14d2dc18190fdb2c9c643a72e15"),
    "multinet": ("MultiNet: Adaptive Multi-Viewed Subgraph Convolutional Networks for Graph Classification", "NeurIPS", 2025, "e029badb1bb84077c3c8a42f17fc70e8415194e3"),
    "rspool": ("Enhancing Graph Classification Robustness with Singular Pooling", "NeurIPS", 2025, "07770440cfb0dfa84084daef24c410d6c28ccc66"),
    "magedgepool": ("Geometry-Aware Edge Pooling for Graph Neural Networks", "NeurIPS", 2025, "e5af8bd15a638014126eb79fd22f15db195dd32d"),
    "hspgkn": ("Hierarchical Shortest-Path Graph Kernel Network", "NeurIPS", 2025, "1ba94517dff4ea11ee84a21fea217587389b5a24"),
    "g2pm": ("Generative Graph Pattern Machine", "NeurIPS", 2025, "113072e6130ed6cf7866798bc6fb5f737b525790"),
    "hyperenc": ("Higher-Order Learning with Graph Neural Networks via Hypergraph Encodings", "NeurIPS", 2025, "17b0c8ca5d5eb98600fea7b95e9eda3d0f87d753"),
    "dgda": ("Diffusion-Guided Graph Data Augmentation", "NeurIPS", 2025, "11198226e39a2a0f2de87f9fe6e7d299c6c72534"),
    "spectre": ("Graph Persistence goes Spectral", "NeurIPS", 2025, "90ad637b00481a33318306d776882dce8e2012e8"),
    "prune": ("Pruning Spurious Subgraphs for Graph Out-of-Distribution Generalization", "NeurIPS", 2025, "7b41dae58f3943c4bfc04d68aa1a8db8bd2c068d"),
    "rsnn": ("Random Search Neural Networks for Efficient and Expressive Graph Learning", "NeurIPS", 2025, "cb112e2353bd65112f8a0a3ee3f6713171df5a73"),
    "toper": ("TopER: Topological Embeddings in Graph Representation Learning", "NeurIPS", 2025, "cef11a43e1540508ed570217a384b6cd6237185b"),
    "leap": ("LEAP: Local ECT-Based Learnable Positional Encodings for Graphs", "ICLR", 2026, "ab2c84b5dbbff0b2f87bd5acee766d173f323772"),
    "hourglass": ("Contraction and Hourglass Persistence for Learning on Graphs, Simplices, and Cells", "ICLR", 2026, "ed1338af188733c9c613d5ecce29bb6f75fba875"),
    "topoformer": ("TopoFormer: Topology Meets Attention for Graph Learning", "ICLR", 2026, "496634d36d6a114fbbcf3cea891f26454ab245bd"),
    "tif": ("From GNNs to Trees: Multi-Granular Interpretability for Graph Neural Networks", "ICLR", 2025, "0ccadc4bb276ef93caee57a6d7767f7cd9d56085"),
    "git_tasktree": ("Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees", "ICML", 2025, "b8f4fe38b543b04a2ac1e36739f5059982a8bc27"),
}


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))


def _gin(in_dim: int, out_dim: int) -> GINConv:
    return GINConv(_mlp(in_dim, out_dim), train_eps=True)


class Head(nn.Module):
    def __init__(self, dim: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.25), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, classes))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GINBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, residual: bool = True):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([_gin(hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.residual = residual

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, list[Tensor]]:
        x = F.relu(self.input(x))
        states = []
        for conv, norm in zip(self.convs, self.norms):
            update = F.relu(norm(conv(x, edge_index)))
            x = x + update if self.residual else update
            states.append(x)
        return x, states


class GNNPlusClassifier(nn.Module):
    """残差、归一化、层级拼接和 mean/max 双读出组成的强经典 GNN。"""

    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.encoder = GINBackbone(in_dim, hidden, layers)
        self.fuse = nn.Linear(2 * hidden * layers, hidden)
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        _, states = self.encoder(data.x, data.edge_index)
        pools = [torch.cat([global_mean_pool(x, data.batch), global_max_pool(x, data.batch)], -1) for x in states]
        z = F.relu(self.fuse(torch.cat(pools, -1)))
        return self.head(z), z, z.new_zeros(())


class NodeIDClassifier(nn.Module):
    """使用离散结构标识嵌入区分局部角色，再交由 GIN 编码。"""

    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.ids = nn.Embedding(256, 16)
        self.encoder = GINBackbone(in_dim + 16, hidden, layers)
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        node_id = getattr(data, "struct_id", torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)).clamp(0, 255)
        x, _ = self.encoder(torch.cat([data.x, self.ids(node_id)], -1), data.edge_index)
        z = global_add_pool(x, data.batch)
        return self.head(z), z, z.new_zeros(())


class KANLayer(nn.Module):
    """轻量 Kolmogorov-Arnold 基函数层。"""

    def __init__(self, in_dim: int, out_dim: int, bases: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.spline = nn.Linear(in_dim * bases, out_dim, bias=False)
        self.frequencies = nn.Parameter(torch.arange(1, bases + 1).float(), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        basis = torch.sin(torch.tanh(x).unsqueeze(-1) * self.frequencies.to(x.device)).flatten(-2)
        return self.linear(x) + self.spline(basis)


class KAGNNClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.input = KANLayer(in_dim, hidden)
        self.convs = nn.ModuleList([GINConv(nn.Sequential(KANLayer(hidden, hidden), nn.ReLU(), KANLayer(hidden, hidden)), train_eps=True) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        x = F.relu(self.input(data.x))
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + F.relu(conv(x, data.edge_index)))
        z = global_add_pool(x, data.batch)
        return self.head(z), z, z.new_zeros(())


class RankFeaturesClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.rank_gate = nn.Sequential(nn.Linear(8, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))
        self.encoder = GINBackbone(in_dim + 8, hidden, layers)
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        pe = data.struct_pe
        x = data.x * (1 + torch.sigmoid(self.rank_gate(pe)))
        x, _ = self.encoder(torch.cat([x, pe], -1), data.edge_index)
        z = global_add_pool(x, data.batch)
        return self.head(z), z, z.new_zeros(())


class MultiNetClassifier(nn.Module):
    """GCN、GraphSAGE、GIN 三种子图视角及实例自适应门控。"""

    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.gcn = GCNConv(hidden, hidden)
        self.sage = SAGEConv(hidden, hidden)
        self.gin = _gin(hidden, hidden)
        self.gate = nn.Linear(3 * hidden, 3)
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        x = F.relu(self.input(data.x))
        views = [F.relu(self.gcn(x, data.edge_index)), F.relu(self.sage(x, data.edge_index)), F.relu(self.gin(x, data.edge_index))]
        pooled = [global_mean_pool(v, data.batch) for v in views]
        weights = F.softmax(self.gate(torch.cat(pooled, -1)), -1)
        z = sum(p * weights[:, i:i + 1] for i, p in enumerate(pooled))
        return self.head(z), z, z.new_zeros(())


class RSPoolClassifier(nn.Module):
    """对节点矩阵做可微的主奇异方向加权读出。"""

    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.encoder = GINBackbone(in_dim, hidden, layers)
        self.seed = nn.Parameter(torch.randn(hidden) / math.sqrt(hidden))
        self.fuse = nn.Linear(2 * hidden, hidden)
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        x, _ = self.encoder(data.x, data.edge_index)
        dense, mask = to_dense_batch(x, data.batch)
        direction = F.normalize(self.seed, dim=0)
        scores = dense @ direction
        scores = scores.masked_fill(~mask, -1e4)
        singular = (dense * F.softmax(scores, -1).unsqueeze(-1)).sum(1)
        z = F.relu(self.fuse(torch.cat([singular, global_mean_pool(x, data.batch)], -1)))
        return self.head(z), z, z.new_zeros(())


class GeometryEdgeConv(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge_net = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.update = _mlp(2 * hidden, hidden)

    def forward(self, x: Tensor, edge_index: Tensor):
        source, target = edge_index
        diff = x[source] - x[target]
        geometry = torch.stack([diff.norm(dim=-1), F.cosine_similarity(x[source], x[target]), x[source].norm(dim=-1), x[target].norm(dim=-1)], -1)
        weight = torch.sigmoid(self.edge_net(geometry))
        message = scatter(x[source] * weight, target, dim=0, dim_size=x.size(0), reduce="mean")
        return self.update(torch.cat([x, message], -1)), weight


class MAGEdgePoolClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([GeometryEdgeConv(hidden) for _ in range(layers)])
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        x = F.relu(self.input(data.x)); weights = None
        for layer in self.layers:
            update, weights = layer(x, data.edge_index); x = F.layer_norm(x + F.relu(update), (x.size(-1),))
        z = global_add_pool(x, data.batch)
        entropy = -(weights * torch.log(weights.clamp_min(1e-6)) + (1 - weights) * torch.log((1 - weights).clamp_min(1e-6))).mean()
        return self.head(z), z, 1e-3 * entropy


class HSPGKNClassifier(nn.Module):
    """逐跳累积的最短路径核视图与层级注意力。"""

    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.encoder = GINBackbone(in_dim, hidden, layers, residual=False)
        self.hop_attention = nn.Linear(hidden, 1)
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        _, states = self.encoder(data.x, data.edge_index)
        hops = torch.stack([global_add_pool(x, data.batch) for x in states], 1)
        weights = F.softmax(self.hop_attention(hops), 1)
        z = (hops * weights).sum(1)
        return self.head(z), z, z.new_zeros(())


class G2PMClassifier(nn.Module):
    """用生成式可学习模式字典匹配节点集合。"""

    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int, patterns: int = 12):
        super().__init__()
        self.encoder = GINBackbone(in_dim, hidden, layers)
        self.patterns = nn.Parameter(torch.randn(patterns, hidden) / math.sqrt(hidden))
        self.fuse = nn.Linear(2 * hidden, hidden)
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        x, _ = self.encoder(data.x, data.edge_index)
        match = F.softmax(F.normalize(x, dim=-1) @ F.normalize(self.patterns, dim=-1).t(), -1)
        pattern_x = match @ self.patterns
        z = F.relu(self.fuse(torch.cat([global_add_pool(x, data.batch), global_mean_pool(pattern_x, data.batch)], -1)))
        diversity = (F.normalize(self.patterns, dim=-1) @ F.normalize(self.patterns, dim=-1).t()).triu(1).pow(2).mean()
        return self.head(z), z, 1e-3 * diversity


class StructuralFusionClassifier(nn.Module):
    """用于高阶编码、谱持久、TopER 和 Hourglass 的结构描述符融合。"""

    def __init__(self, method: str, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.method = method
        self.encoder = GINBackbone(in_dim, hidden, layers)
        stat_dim = 32 if method == "hourglass" else 16
        self.struct = _mlp(stat_dim, hidden)
        self.gate = nn.Linear(2 * hidden, hidden)
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        x, _ = self.encoder(data.x, data.edge_index)
        base = global_add_pool(x, data.batch)
        stats = torch.cat([data.graph_stats, data.filtration], -1) if self.method == "hourglass" else (data.filtration if self.method in {"spectre", "toper"} else data.graph_stats)
        structural = self.struct(stats)
        gate = torch.sigmoid(self.gate(torch.cat([base, structural], -1)))
        z = gate * base + (1 - gate) * structural
        return self.head(z), z, z.new_zeros(())


class DGDAClassifier(nn.Module):
    """潜空间扩散扰动与标签保持一致性。"""

    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.encoder = GINBackbone(in_dim, hidden, layers)
        self.denoiser = _mlp(hidden + 1, hidden)
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        x, _ = self.encoder(data.x, data.edge_index); z = global_mean_pool(x, data.batch)
        regularizer = z.new_zeros(())
        if self.training:
            t = torch.rand(z.size(0), 1, device=z.device)
            noisy = z + torch.randn_like(z) * t
            restored = self.denoiser(torch.cat([noisy, t], -1))
            regularizer = F.mse_loss(restored, z.detach())
            z = 0.5 * (z + restored)
        return self.head(z), z, 0.1 * regularizer


class PrunEClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.selector = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        x = F.relu(self.input(data.x)); source, target = data.edge_index
        weights = torch.sigmoid(self.selector(torch.cat([x[source], x[target]], -1))).squeeze(-1)
        for conv in self.convs: x = F.relu(conv(x, data.edge_index, weights))
        z = global_add_pool(x, data.batch)
        low = weights.topk(max(1, weights.numel() // 5), largest=False).values
        regularizer = weights.mean() + F.mse_loss(low, torch.zeros_like(low))
        return self.head(z), z, 0.02 * regularizer


class SequenceGraphClassifier(nn.Module):
    """随机搜索序列或 Topo-Scan token 序列编码。"""

    def __init__(self, method: str, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.method = method
        self.input = nn.Linear(in_dim + 8, hidden)
        if method == "rsnn":
            self.sequence = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
            self.project = nn.Linear(2 * hidden, hidden)
        else:
            block = nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, 0.2, batch_first=True, norm_first=True)
            self.sequence = nn.TransformerEncoder(block, num_layers=2)
            self.project = nn.Linear(hidden, hidden)
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        token = F.relu(self.input(torch.cat([data.x, data.struct_pe], -1)))
        dense, mask = to_dense_batch(token, data.batch)
        ranks, _ = to_dense_batch(data.search_rank.float().unsqueeze(-1), data.batch)
        ranks = ranks.squeeze(-1).masked_fill(~mask, float("inf"))
        order = ranks.argsort(dim=1)
        dense = dense.gather(1, order.unsqueeze(-1).expand_as(dense))
        sorted_mask = mask.gather(1, order)
        if self.method == "rsnn":
            out, _ = self.sequence(dense); lengths = sorted_mask.sum(1).clamp_min(1) - 1
            z = self.project(out[torch.arange(out.size(0), device=out.device), lengths])
        else:
            out = self.sequence(dense, src_key_padding_mask=~sorted_mask)
            z = self.project((out * sorted_mask.unsqueeze(-1)).sum(1) / sorted_mask.sum(1, keepdim=True).clamp_min(1))
        return self.head(z), z, z.new_zeros(())


class LEAPClassifier(nn.Module):
    """局部 ECT 结构向量、可学习方向与消息传递联合训练。"""

    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.directions = nn.Parameter(torch.randn(8, 8) / math.sqrt(8))
        self.pe_project = nn.Linear(16, 16)
        self.encoder = GINBackbone(in_dim + 16, hidden, layers)
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        projection = data.struct_pe @ F.normalize(self.directions, dim=0)
        ect = torch.cat([torch.sigmoid(4 * projection), torch.sigmoid(-4 * projection)], -1)
        pe = F.relu(self.pe_project(ect))
        x, _ = self.encoder(torch.cat([data.x, pe], -1), data.edge_index)
        z = global_add_pool(x, data.batch)
        return self.head(z), z, z.new_zeros(())


class TIFClassifier(nn.Module):
    """多粒度软决策树路径与 GNN 表征融合。"""

    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.encoder = GINBackbone(in_dim, hidden, layers)
        self.gates = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(3)])
        self.leaves = nn.Parameter(torch.randn(8, hidden) / math.sqrt(hidden))
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        x, _ = self.encoder(data.x, data.edge_index); base = global_add_pool(x, data.batch)
        decisions = torch.cat([torch.sigmoid(g(base)) for g in self.gates], -1)
        paths = []
        for leaf in range(8):
            bits = torch.tensor([(leaf >> bit) & 1 for bit in range(3)], device=base.device, dtype=base.dtype)
            paths.append(torch.prod(torch.where(bits.bool(), decisions, 1 - decisions), dim=-1))
        weights = torch.stack(paths, -1); z = base + weights @ self.leaves
        return self.head(z), z, 1e-3 * (-weights.clamp_min(1e-8).log() * weights).sum(-1).mean()


class TaskTreeClassifier(nn.Module):
    """任务树共享干路与分支专家路由。"""

    def __init__(self, in_dim: int, hidden: int, layers: int, classes: int):
        super().__init__()
        self.encoder = GINBackbone(in_dim, hidden, layers)
        self.experts = nn.ModuleList([_mlp(hidden, hidden) for _ in range(4)])
        self.router = nn.Linear(hidden + 16, 4)
        self.head = Head(hidden, classes)

    def forward(self, data: Data):
        x, _ = self.encoder(data.x, data.edge_index); base = global_add_pool(x, data.batch)
        weights = F.softmax(self.router(torch.cat([base, data.graph_stats], -1)), -1)
        z = sum(expert(base) * weights[:, i:i + 1] for i, expert in enumerate(self.experts)) + base
        balance = (weights.mean(0) - 0.25).pow(2).mean()
        return self.head(z), z, 1e-3 * balance


def build_model(method: str, in_dim: int, hidden: int, layers: int, classes: int) -> nn.Module:
    if method == "gnnplus": return GNNPlusClassifier(in_dim, hidden, layers, classes)
    if method == "nodeid": return NodeIDClassifier(in_dim, hidden, layers, classes)
    if method == "kagnn": return KAGNNClassifier(in_dim, hidden, layers, classes)
    if method == "rankfeatures": return RankFeaturesClassifier(in_dim, hidden, layers, classes)
    if method == "multinet": return MultiNetClassifier(in_dim, hidden, layers, classes)
    if method == "rspool": return RSPoolClassifier(in_dim, hidden, layers, classes)
    if method == "magedgepool": return MAGEdgePoolClassifier(in_dim, hidden, layers, classes)
    if method == "hspgkn": return HSPGKNClassifier(in_dim, hidden, layers, classes)
    if method == "g2pm": return G2PMClassifier(in_dim, hidden, layers, classes)
    if method in {"hyperenc", "spectre", "toper", "hourglass"}: return StructuralFusionClassifier(method, in_dim, hidden, layers, classes)
    if method == "dgda": return DGDAClassifier(in_dim, hidden, layers, classes)
    if method == "prune": return PrunEClassifier(in_dim, hidden, layers, classes)
    if method in {"rsnn", "topoformer"}: return SequenceGraphClassifier(method, in_dim, hidden, layers, classes)
    if method == "leap": return LEAPClassifier(in_dim, hidden, layers, classes)
    if method == "tif": return TIFClassifier(in_dim, hidden, layers, classes)
    if method == "git_tasktree": return TaskTreeClassifier(in_dim, hidden, layers, classes)
    raise ValueError(f"未知方法：{method}")
