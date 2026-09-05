"""六种深度图聚类方法的图级适配实现。

原论文均面向单张属性图的节点聚类。本文件保留各方法的核心无监督目标，
但把聚类对象改为一个批次中的独立图，并以 GIN/平滑编码器得到图表示。
第三方作者源码原样保存在 ``src/sota_sources_dgc``，本文件不修改其内容。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import degree, negative_sampling, scatter


METHODS = ("thesaurus", "arln", "mhgc", "scgc", "rdsa", "dcgc")

METHOD_SPECS = {
    "thesaurus": {
        "title": "THESAURUS: Contrastive Graph Clustering by Swapping Fused Gromov-Wasserstein Couplings",
        "venue": "AAAI",
        "year": 2025,
        "source": "paper_reimplementation",
        "source_commit": None,
    },
    "arln": {
        "title": "Deep Graph Clustering via Aligning Representation Learning",
        "venue": "Neural Networks",
        "year": 2025,
        "source": "paper_reimplementation",
        "source_commit": None,
    },
    "mhgc": {
        "title": "MHGC: Multi-scale Hard Sample Mining for Contrastive Deep Graph Clustering",
        "venue": "Information Processing & Management",
        "year": 2025,
        "source": "official_source_adaptation",
        "source_commit": "875bb84d38ccb03a6abb9e8eb5fab436e0015f0b",
    },
    "scgc": {
        "title": "Simple Contrastive Graph Clustering",
        "venue": "IEEE TNNLS",
        "year": 2023,
        "source": "official_source_adaptation",
        "source_commit": "ab029e452e6fc91c29163ebf4ff2050208351839",
    },
    "rdsa": {
        "title": "RDSA: A Robust Deep Graph Clustering Framework via Dual Soft Assignment",
        "venue": "DASFAA",
        "year": 2025,
        "source": "official_source_adaptation",
        "source_commit": "0d6a14f9a0144f602698659bae7e0e8091f1ef31",
    },
    "dcgc": {
        "title": "Dual-Center Graph Clustering with Neighbor Distribution",
        "venue": "ECAI",
        "year": 2025,
        "source": "official_source_adaptation",
        "source_commit": "d7282bcc2a0d0b252d08e3151895ec59641f9e7f",
    },
}


def _mlp(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))


def _gin(dim: int) -> GINConv:
    return GINConv(_mlp(dim), train_eps=True)


def _safe_normalize(value: Tensor, dim: int = -1) -> Tensor:
    return F.normalize(value, dim=dim, eps=1e-8)


def _target_distribution(q: Tensor) -> Tensor:
    weight = q.square() / q.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _student_assignment(z: Tensor, centers: Tensor) -> Tensor:
    distance = (z.unsqueeze(1) - centers.unsqueeze(0)).square().sum(dim=-1)
    numerator = (1.0 + distance).reciprocal()
    return numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _symmetric_kl(first: Tensor, second: Tensor) -> Tensor:
    first = first.clamp_min(1e-8)
    second = second.clamp_min(1e-8)
    return 0.5 * (
        F.kl_div(first.log(), second.detach(), reduction="batchmean")
        + F.kl_div(second.log(), first.detach(), reduction="batchmean")
    )


def _nt_xent(first: Tensor, second: Tensor, temperature: float = 0.2) -> Tensor:
    if first.size(0) < 2:
        return first.new_zeros(())
    first = _safe_normalize(first)
    second = _safe_normalize(second)
    logits = first @ second.t() / temperature
    labels = torch.arange(first.size(0), device=first.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _barlow_loss(first: Tensor, second: Tensor, off_diagonal_weight: float = 0.005) -> Tensor:
    if first.size(0) < 2:
        return first.new_zeros(())
    first = (first - first.mean(0)) / first.std(0, unbiased=False).clamp_min(1e-4)
    second = (second - second.mean(0)) / second.std(0, unbiased=False).clamp_min(1e-4)
    correlation = first.t() @ second / first.size(0)
    diagonal = (correlation.diagonal() - 1).square().mean()
    off_diagonal = correlation.square().sum() - correlation.diagonal().square().sum()
    off_diagonal = off_diagonal / max(correlation.numel() - correlation.size(0), 1)
    return diagonal + off_diagonal_weight * off_diagonal


def _drop_view(data: Data, edge_drop: float, feature_drop: float) -> tuple[Tensor, Tensor]:
    device = data.x.device
    feature_mask = torch.rand_like(data.x) >= feature_drop
    x = data.x * feature_mask
    if data.edge_index.size(1) == 0:
        return x, data.edge_index
    keep = torch.rand(data.edge_index.size(1), device=device) >= edge_drop
    if not keep.any():
        keep[torch.randint(data.edge_index.size(1), (1,), device=device)] = True
    return x, data.edge_index[:, keep]


def _edge_reconstruction_loss(z: Tensor, edge_index: Tensor, maximum_edges: int = 8192) -> Tensor:
    if z.size(0) < 2 or edge_index.size(1) == 0:
        return z.new_zeros(())
    if edge_index.size(1) > maximum_edges:
        selection = torch.randperm(edge_index.size(1), device=z.device)[:maximum_edges]
        positive_edges = edge_index[:, selection]
    else:
        positive_edges = edge_index
    negative_edges = negative_sampling(
        edge_index, num_nodes=z.size(0), num_neg_samples=positive_edges.size(1),
        method="sparse",
    )
    positive = (z[positive_edges[0]] * z[positive_edges[1]]).sum(-1)
    negative = (z[negative_edges[0]] * z[negative_edges[1]]).sum(-1)
    return 0.5 * (
        F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
        + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
    )


def _feature_reconstruction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """按特征列尺度归一化，避免分子类别编码的数值范围支配目标函数。"""
    scale = target.detach().square().mean(dim=0, keepdim=True).sqrt().clamp_min(1.0)
    return F.smooth_l1_loss(prediction / scale, target / scale)


def _knn_affinity(z: Tensor, neighbors: int = 8) -> Tensor:
    count = z.size(0)
    if count <= 1:
        return z.new_eye(count)
    # kNN 拓扑作为当前轮次的伪结构，不反传离散邻居选择的梯度。
    similarity = (_safe_normalize(z) @ _safe_normalize(z).t()).clamp_min(0).detach()
    similarity.fill_diagonal_(0)
    k = min(neighbors, count - 1)
    indices = similarity.topk(k, dim=1).indices
    mask = torch.zeros_like(similarity, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    affinity = similarity.masked_fill(~mask, 0)
    affinity = torch.maximum(affinity, affinity.t())
    affinity = affinity + torch.eye(count, device=z.device, dtype=z.dtype)
    return affinity / affinity.sum(dim=1, keepdim=True).clamp_min(1e-8)


class GINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([_gin(hidden_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(layers)])

    def node_states(self, x: Tensor, edge_index: Tensor) -> list[Tensor]:
        x = F.relu(self.input(x))
        states = [x]
        for conv, norm in zip(self.convs, self.norms):
            update = F.relu(norm(conv(x, edge_index)))
            x = x + update
            states.append(x)
        return states

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        return global_mean_pool(self.node_states(x, edge_index)[-1], batch)


class BaseDGCAdapter(nn.Module):
    method: str

    def encode(self, data: Data) -> Tensor:
        raise NotImplementedError

    def ssl_loss(self, data: Data) -> tuple[Tensor, dict[str, float]]:
        raise NotImplementedError


class ThesaurusAdapter(BaseDGCAdapter):
    """语义原型、平衡分配、跨视图交换预测和原型图动量更新。"""

    method = "thesaurus"

    def __init__(self, in_dim: int, hidden_dim: int, layers: int, clusters: int):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden_dim, layers)
        self.projector = _mlp(hidden_dim)
        prototype_count = max(8, min(32, clusters * 4))
        self.prototypes = nn.Parameter(torch.randn(prototype_count, hidden_dim) / math.sqrt(hidden_dim))
        self.register_buffer("prototype_graph", torch.eye(prototype_count))
        self.register_buffer("prototype_marginal", torch.full((prototype_count,), 1 / prototype_count))
        self.temperature = 0.2
        self.structure_weight = 0.5
        self.momentum = 0.9

    @staticmethod
    @torch.no_grad()
    def _sinkhorn(scores: Tensor, iterations: int = 4) -> Tensor:
        assignment = torch.exp(scores - scores.max()).t()
        assignment /= assignment.sum().clamp_min(1e-8)
        prototypes, samples = assignment.shape
        for _ in range(iterations):
            assignment /= assignment.sum(dim=1, keepdim=True).clamp_min(1e-8)
            assignment /= prototypes
            assignment /= assignment.sum(dim=0, keepdim=True).clamp_min(1e-8)
            assignment /= samples
        return (assignment * samples).t()

    def _view(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        return _safe_normalize(self.projector(self.encoder(x, edge_index, batch)))

    def encode(self, data: Data) -> Tensor:
        return self._view(data.x, data.edge_index, data.batch)

    def ssl_loss(self, data: Data) -> tuple[Tensor, dict[str, float]]:
        x1, edge1 = _drop_view(data, 0.2, 0.2)
        x2, edge2 = _drop_view(data, 0.2, 0.2)
        z1, z2 = self._view(x1, edge1, data.batch), self._view(x2, edge2, data.batch)
        prototypes = _safe_normalize(self.prototypes)
        score1, score2 = z1 @ prototypes.t() / self.temperature, z2 @ prototypes.t() / self.temperature
        target1, target2 = self._sinkhorn(score1), self._sinkhorn(score2)
        swapped = -0.5 * (
            (target1 * F.log_softmax(score2, dim=-1)).sum(-1).mean()
            + (target2 * F.log_softmax(score1, dim=-1)).sum(-1).mean()
        )

        sample_structure = ((z1 @ z1.t()) + (z2 @ z2.t())).mul(0.25).add(0.5).clamp(0, 1)
        # 后面会动量更新缓冲区，克隆可避免原位更新破坏本轮反向图。
        prototype_graph = self.prototype_graph.detach().clone()
        predicted_structure = target1 @ prototype_graph @ target1.t()
        structural = F.mse_loss(predicted_structure, sample_structure.detach())
        with torch.no_grad():
            joint = target1.t() @ sample_structure @ target1
            normalizer = target1.sum(0).outer(target1.sum(0)).clamp_min(1e-8)
            current_graph = (joint / normalizer).clamp(0, 1)
            current_marginal = target1.mean(0)
            self.prototype_graph.mul_(self.momentum).add_(current_graph, alpha=1 - self.momentum)
            self.prototype_marginal.mul_(self.momentum).add_(current_marginal, alpha=1 - self.momentum)
        loss = swapped + self.structure_weight * structural
        return loss, {"swapped_assignment": float(swapped.detach()), "fgw_structure": float(structural.detach())}


class ARLNAdapter(BaseDGCAdapter):
    """AE/GAE 双视图、实例/特征对齐及分配概率对齐。"""

    method = "arln"

    def __init__(self, in_dim: int, hidden_dim: int, layers: int, clusters: int):
        super().__init__()
        self.attribute_encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), _mlp(hidden_dim))
        self.attribute_decoder = nn.Linear(hidden_dim, in_dim)
        self.graph_encoder = GINEncoder(in_dim, hidden_dim, layers)
        self.assignment_attribute = nn.Linear(hidden_dim, clusters)
        self.assignment_graph = nn.Linear(hidden_dim, clusters)
        self.fusion_gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def _views(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        attribute_nodes = self.attribute_encoder(data.x)
        attribute_graph = global_mean_pool(attribute_nodes, data.batch)
        graph_nodes = self.graph_encoder.node_states(data.x, data.edge_index)[-1]
        structure_graph = global_mean_pool(graph_nodes, data.batch)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([attribute_graph, structure_graph], dim=-1)))
        fused = gate * attribute_graph + (1 - gate) * structure_graph
        return attribute_graph, structure_graph, fused

    def encode(self, data: Data) -> Tensor:
        return self._views(data)[-1]

    def ssl_loss(self, data: Data) -> tuple[Tensor, dict[str, float]]:
        attribute, structure, _ = self._views(data)
        attribute_nodes = self.attribute_encoder(data.x)
        reconstruction = _feature_reconstruction_loss(self.attribute_decoder(attribute_nodes), data.x)
        structure_nodes = self.graph_encoder.node_states(data.x, data.edge_index)[-1]
        graph_reconstruction = _edge_reconstruction_loss(structure_nodes, data.edge_index)
        instance = _nt_xent(attribute, structure)
        feature = _barlow_loss(attribute, structure)
        q_attribute = F.softmax(self.assignment_attribute(attribute) / 0.5, dim=-1)
        q_structure = F.softmax(self.assignment_graph(structure) / 0.5, dim=-1)
        assignment = _symmetric_kl(q_attribute, q_structure)
        loss = 0.1 * (reconstruction + graph_reconstruction) + instance + 0.1 * feature + 0.5 * assignment
        return loss, {
            "reconstruction": float((reconstruction + graph_reconstruction).detach()),
            "instance": float(instance.detach()),
            "feature": float(feature.detach()),
            "assignment": float(assignment.detach()),
        }


class MHGCAdapter(BaseDGCAdapter):
    """多尺度表示、难样本后验加权和聚类目标分布。"""

    method = "mhgc"

    def __init__(self, in_dim: int, hidden_dim: int, layers: int, clusters: int):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden_dim, layers)
        self.cluster_centers = nn.Parameter(torch.randn(clusters, hidden_dim) / math.sqrt(hidden_dim))
        self.scale_gate = nn.Linear(hidden_dim * 3, 3)
        self.temperature = 0.2

    def _scales(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> list[Tensor]:
        states = self.encoder.node_states(x, edge_index)
        indices = sorted({0, len(states) // 2, len(states) - 1})
        scales = [global_mean_pool(states[index], batch) for index in indices]
        while len(scales) < 3:
            scales.append(scales[-1])
        return scales[:3]

    def _fuse(self, scales: list[Tensor]) -> Tensor:
        weights = F.softmax(self.scale_gate(torch.cat(scales, dim=-1)), dim=-1)
        return sum(scale * weights[:, index:index + 1] for index, scale in enumerate(scales))

    def encode(self, data: Data) -> Tensor:
        return self._fuse(self._scales(data.x, data.edge_index, data.batch))

    def _hard_contrast(self, first: Tensor, second: Tensor, pseudo: Tensor) -> Tensor:
        if first.size(0) < 2:
            return first.new_zeros(())
        first, second = _safe_normalize(first), _safe_normalize(second)
        similarity = first @ second.t()
        same = pseudo.unsqueeze(1).eq(pseudo.unsqueeze(0)).float()
        normalized = (similarity.detach() + 1) / 2
        difficulty = (same - normalized).abs().clamp(1e-4, 1 - 1e-4)
        # 两个 Beta 分量的解析后验，作为作者 BMM 难样本挖掘的批量稳定实现。
        easy_likelihood = difficulty.pow(1) * (1 - difficulty).pow(4)
        hard_likelihood = difficulty.pow(4) * (1 - difficulty).pow(1)
        hard_posterior = hard_likelihood / (easy_likelihood + hard_likelihood + 1e-8)
        weight = 0.5 + hard_posterior
        logits = similarity / self.temperature + weight.log()
        labels = torch.arange(first.size(0), device=first.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def ssl_loss(self, data: Data) -> tuple[Tensor, dict[str, float]]:
        x1, edge1 = _drop_view(data, 0.2, 0.1)
        x2, edge2 = _drop_view(data, 0.2, 0.1)
        scales1 = self._scales(x1, edge1, data.batch)
        scales2 = self._scales(x2, edge2, data.batch)
        fused = 0.5 * (self._fuse(scales1) + self._fuse(scales2))
        q = _student_assignment(fused, self.cluster_centers)
        pseudo = q.detach().argmax(-1)
        scale_losses = [self._hard_contrast(first, second, pseudo) for first, second in zip(scales1, scales2)]
        multiscale = torch.stack(scale_losses).mean()
        target = _target_distribution(q).detach()
        cluster = F.kl_div(q.clamp_min(1e-8).log(), target, reduction="batchmean")
        loss = multiscale + 0.2 * cluster
        return loss, {"multi_scale_hard": float(multiscale.detach()), "cluster": float(cluster.detach())}


class SCGCAdapter(BaseDGCAdapter):
    """拉普拉斯平滑、双线性头、高斯扰动与结构相似度重构。"""

    method = "scgc"

    def __init__(self, in_dim: int, hidden_dim: int, layers: int, clusters: int):
        super().__init__()
        del clusters
        self.layers = max(1, layers)
        self.first = nn.Linear(in_dim, hidden_dim)
        self.second = nn.Linear(in_dim, hidden_dim)
        self.noise_sigma = 0.01

    def _smooth(self, x: Tensor, edge_index: Tensor) -> Tensor:
        source, target = edge_index
        for _ in range(self.layers):
            messages = scatter(x[source], target, dim=0, dim_size=x.size(0), reduce="sum")
            counts = degree(target, x.size(0), dtype=x.dtype).unsqueeze(-1)
            x = (x + messages) / (1 + counts)
        return x

    def _nodes(self, data: Data, noisy: bool) -> tuple[Tensor, Tensor]:
        smooth = self._smooth(data.x, data.edge_index)
        first = _safe_normalize(self.first(smooth))
        second = _safe_normalize(self.second(smooth))
        if noisy:
            second = _safe_normalize(second + torch.randn_like(second) * self.noise_sigma)
        return first, second

    def encode(self, data: Data) -> Tensor:
        first, second = self._nodes(data, noisy=False)
        return global_mean_pool(0.5 * (first + second), data.batch)

    def ssl_loss(self, data: Data) -> tuple[Tensor, dict[str, float]]:
        first, second = self._nodes(data, noisy=True)
        structure = _edge_reconstruction_loss(first, data.edge_index) + _edge_reconstruction_loss(second, data.edge_index)
        graph_first = global_mean_pool(first, data.batch)
        graph_second = global_mean_pool(second, data.batch)
        consistency = _nt_xent(graph_first, graph_second)
        loss = structure + 0.2 * consistency
        return loss, {"structure": float(structure.detach()), "consistency": float(consistency.detach())}


class RDSAAdapter(BaseDGCAdapter):
    """属性/拓扑融合编码、结构软分配和地标软分配。"""

    method = "rdsa"

    def __init__(self, in_dim: int, hidden_dim: int, layers: int, clusters: int):
        super().__init__()
        self.attribute = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), _mlp(hidden_dim))
        self.decoder = nn.Linear(hidden_dim, in_dim)
        self.graph = GINEncoder(in_dim, hidden_dim, layers)
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))
        self.landmarks = nn.Parameter(torch.randn(clusters, hidden_dim) / math.sqrt(hidden_dim))

    def _representation(self, data: Data) -> tuple[Tensor, Tensor]:
        attribute_nodes = self.attribute(data.x)
        graph_nodes = self.graph.node_states(data.x, data.edge_index)[-1]
        sigma = torch.sigmoid(self.fusion_logit)
        nodes = sigma * attribute_nodes + (1 - sigma) * graph_nodes
        return nodes, global_mean_pool(nodes, data.batch)

    def encode(self, data: Data) -> Tensor:
        return self._representation(data)[1]

    def ssl_loss(self, data: Data) -> tuple[Tensor, dict[str, float]]:
        nodes, graphs = self._representation(data)
        reconstruction = _feature_reconstruction_loss(self.decoder(self.attribute(data.x)), data.x)
        node_assignment = _student_assignment(graphs, self.landmarks)
        affinity = _knn_affinity(graphs)
        structure_assignment = affinity @ node_assignment
        structure_assignment = structure_assignment / structure_assignment.sum(-1, keepdim=True).clamp_min(1e-8)
        target = _target_distribution(node_assignment).detach()
        dual = 0.5 * (
            F.kl_div(node_assignment.clamp_min(1e-8).log(), target, reduction="batchmean")
            + F.kl_div(structure_assignment.clamp_min(1e-8).log(), target, reduction="batchmean")
        )
        if graphs.size(0) > 1:
            adjacency = 0.5 * (affinity + affinity.t())
            degree_vector = adjacency.sum(1)
            total = adjacency.sum().clamp_min(1e-8)
            modularity_matrix = adjacency - degree_vector.outer(degree_vector) / total
            modularity = -(node_assignment.t() @ modularity_matrix @ node_assignment).trace() / total
        else:
            modularity = graphs.new_zeros(())
        loss = 0.2 * reconstruction + dual + 0.2 * modularity
        return loss, {
            "reconstruction": float(reconstruction.detach()),
            "dual_assignment": float(dual.detach()),
            "modularity": float(modularity.detach()),
        }


class DCGCAdapter(BaseDGCAdapter):
    """自适应低/高频滤波、邻居难负样本与特征/邻居双中心。"""

    method = "dcgc"

    def __init__(self, in_dim: int, hidden_dim: int, layers: int, clusters: int):
        super().__init__()
        del layers
        self.input = nn.Linear(in_dim, hidden_dim)
        self.low_view = nn.Linear(hidden_dim, hidden_dim)
        self.high_view = nn.Linear(hidden_dim, hidden_dim)
        self.feature_centers = nn.Parameter(torch.randn(clusters, hidden_dim) / math.sqrt(hidden_dim))
        self.neighbor_centers = nn.Parameter(torch.eye(clusters))
        self.balance_logit = nn.Parameter(torch.tensor(0.0))
        self.temperature = 0.5
        self.threshold = 0.5

    def _filter(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        base = F.relu(self.input(data.x))
        source, target = data.edge_index
        messages = scatter(base[source], target, dim=0, dim_size=base.size(0), reduce="sum")
        counts = degree(target, base.size(0), dtype=base.dtype).unsqueeze(-1)
        low = (base + messages) / (1 + counts)
        high = base - low
        node_first = _safe_normalize(self.low_view(low))
        node_second = _safe_normalize(self.high_view(high))
        graph_first = global_mean_pool(node_first, data.batch)
        graph_second = global_mean_pool(node_second, data.batch)
        graph = 0.5 * (graph_first + graph_second)
        return graph_first, graph_second, graph

    def encode(self, data: Data) -> Tensor:
        return self._filter(data)[-1]

    def ssl_loss(self, data: Data) -> tuple[Tensor, dict[str, float]]:
        first, second, graph = self._filter(data)
        affinity = _knn_affinity(graph)
        q = _student_assignment(graph, self.feature_centers)
        neighbor_distribution = affinity @ q
        neighbor_distribution = neighbor_distribution / neighbor_distribution.sum(-1, keepdim=True).clamp_min(1e-8)
        f = _student_assignment(neighbor_distribution, self.neighbor_centers)
        target_q, target_f = _target_distribution(q).detach(), _target_distribution(f).detach()
        balance = torch.sigmoid(self.balance_logit)
        dual = balance * F.kl_div(q.clamp_min(1e-8).log(), target_q, reduction="batchmean")
        dual = dual + (1 - balance) * F.kl_div(f.clamp_min(1e-8).log(), target_f, reduction="batchmean")

        if graph.size(0) > 1:
            similarity = _safe_normalize(first) @ _safe_normalize(second).t()
            embedding_similarity = (similarity.detach() + 1) / 2
            neighbor_similarity = _safe_normalize(neighbor_distribution) @ _safe_normalize(neighbor_distribution).t()
            neighbor_target = (neighbor_similarity > self.threshold).float()
            hard_weight = 0.5 + (neighbor_target - embedding_similarity).abs()
            logits = similarity / self.temperature + hard_weight.log()
            labels = torch.arange(graph.size(0), device=graph.device)
            contrast = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            normalized_graph = _safe_normalize(graph)
            reconstruction = F.binary_cross_entropy(
                ((normalized_graph @ normalized_graph.t()) + 1).mul(0.5).clamp(1e-6, 1 - 1e-6),
                (affinity > 0).float(),
            )
        else:
            contrast = graph.new_zeros(())
            reconstruction = graph.new_zeros(())
        loss = contrast + 0.2 * reconstruction + 0.5 * dual
        return loss, {
            "hard_contrast": float(contrast.detach()),
            "structure_reconstruction": float(reconstruction.detach()),
            "dual_center": float(dual.detach()),
        }


def build_model(
    method: str,
    in_dim: int,
    hidden_dim: int,
    layers: int,
    clusters: int,
) -> BaseDGCAdapter:
    builders = {
        "thesaurus": ThesaurusAdapter,
        "arln": ARLNAdapter,
        "mhgc": MHGCAdapter,
        "scgc": SCGCAdapter,
        "rdsa": RDSAAdapter,
        "dcgc": DCGCAdapter,
    }
    if method not in builders:
        raise ValueError(f"未知方法：{method}")
    return builders[method](in_dim, hidden_dim, layers, clusters)
