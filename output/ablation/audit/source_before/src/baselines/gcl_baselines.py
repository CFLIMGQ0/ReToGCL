"""GraphCL、JOAOv2、RGCL 和 SimGRACE 的统一 PyG 实现。"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, global_add_pool


AUGMENTATIONS = ("identity", "node_drop", "edge_drop", "subgraph", "attr_mask")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_features(data: Data) -> Data:
    if data.x is None or data.x.size(-1) == 0:
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)
    else:
        data.x = data.x.float()
    return data


def augment_graph(data: Data, kind: str, ratio: float = 0.2) -> Data:
    """生成保留节点编号与 batch 对齐关系的图增强视图。"""
    x = data.x.clone()
    edge_index = data.edge_index
    num_nodes = x.size(0)

    if kind == "identity":
        new_edges = edge_index
    elif kind == "attr_mask":
        count = max(1, int(num_nodes * ratio))
        masked = torch.randperm(num_nodes, device=x.device)[:count]
        x[masked] = 0
        new_edges = edge_index
    elif kind == "edge_drop":
        keep = torch.rand(edge_index.size(1), device=x.device) >= ratio
        new_edges = edge_index[:, keep]
    elif kind in {"node_drop", "subgraph"}:
        batch = data.batch
        graph_count = int(batch.max()) + 1
        counts = torch.bincount(batch, minlength=graph_count)
        keep_counts = (counts.float() * (1 - ratio)).floor().long().clamp_min(1)
        if kind == "node_drop":
            scores = torch.rand(num_nodes, device=x.device)
        else:
            # 带随机扰动的节点度优先保留稠密区域。
            degree = torch.bincount(edge_index[0], minlength=num_nodes).float()
            scores = degree + torch.rand(num_nodes, device=x.device)

        # 先按分数降序，再稳定地按图编号分组，得到每张图内部的分数排名。
        score_order = torch.argsort(scores, descending=True, stable=True)
        order = score_order[torch.argsort(batch[score_order], stable=True)]
        starts = torch.cumsum(counts, dim=0) - counts
        ranks = torch.arange(num_nodes, device=x.device) - torch.repeat_interleave(
            starts, counts
        )
        keep_sorted = ranks < keep_counts[batch[order]]
        keep_node = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        keep_node[order] = keep_sorted
        edge_keep = keep_node[edge_index[0]] & keep_node[edge_index[1]]
        new_edges = edge_index[:, edge_keep]
        x[~keep_node] = 0
    else:
        raise ValueError(f"未知增强类型：{kind}")

    return Data(x=x, edge_index=new_edges, batch=data.batch, y=data.y)


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(layers):
            input_dim = in_dim if layer == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.output_dim = hidden_dim * layers

    def forward(self, data: Data) -> tuple[Tensor, Tensor]:
        x = data.x
        node_layers = []
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, data.edge_index)))
            node_layers.append(x)
        graph_layers = [global_add_pool(h, data.batch) for h in node_layers]
        return torch.cat(graph_layers, dim=-1), torch.cat(node_layers, dim=-1)


class NodeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(layers):
            input_dim = in_dim if layer == 0 else hidden_dim
            self.convs.append(GCNConv(input_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.output_dim = hidden_dim * layers

    def forward(self, data: Data) -> tuple[Tensor, Tensor]:
        x = data.x
        layers = []
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, data.edge_index)))
            layers.append(x)
        node_repr = torch.cat(layers, dim=-1)
        return node_repr, node_repr


class GCLModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        task: str,
        method: str,
    ):
        super().__init__()
        encoder_type = GraphEncoder if task == "graph" else NodeEncoder
        self.encoder = encoder_type(in_dim, hidden_dim, layers)
        dim = self.encoder.output_dim
        head_count = len(AUGMENTATIONS) if method == "joaov2" else 1
        self.projectors = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)) for _ in range(head_count)]
        )
        self.rationale = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.task = task
        self.method = method

    def encode(self, data: Data) -> tuple[Tensor, Tensor]:
        return self.encoder(data)

    def project(self, representation: Tensor, head: int = 0) -> Tensor:
        return self.projectors[head](representation)

    def rationale_views(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        anchor, nodes = self.encode(data)
        weights = torch.sigmoid(self.rationale(nodes))
        if self.task == "graph":
            rationale = global_add_pool(nodes * weights, data.batch)
            complement = global_add_pool(nodes * (1 - weights), data.batch)
        else:
            rationale = nodes * (0.9 + 0.2 * weights)
            complement = nodes * (1.1 - 0.2 * weights)
        return self.project(anchor), self.project(rationale), self.project(complement)


def info_nce(first: Tensor, second: Tensor, temperature: float = 0.2, max_samples: int = 2048) -> Tensor:
    if first.size(0) > max_samples:
        indices = torch.randperm(first.size(0), device=first.device)[:max_samples]
        first, second = first[indices], second[indices]
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    logits = first @ second.t() / temperature
    labels = torch.arange(first.size(0), device=first.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def rgcl_loss(anchor: Tensor, rationale: Tensor, complement: Tensor, temperature: float = 0.2) -> Tensor:
    if anchor.size(0) > 2048:
        indices = torch.randperm(anchor.size(0), device=anchor.device)[:2048]
        anchor, rationale, complement = anchor[indices], rationale[indices], complement[indices]
    anchor = F.normalize(anchor, dim=-1)
    rationale = F.normalize(rationale, dim=-1)
    complement = F.normalize(complement, dim=-1)
    positive_matrix = torch.exp(anchor @ rationale.t() / temperature)
    complement_matrix = torch.exp(anchor @ complement.t() / temperature)
    positive = positive_matrix.diag()
    ratio_main = positive / positive_matrix.sum(dim=1).clamp_min(1e-12)
    ratio_complement = positive / (positive + complement_matrix.sum(dim=1)).clamp_min(1e-12)
    return -torch.log((ratio_main + 0.1 * ratio_complement).clamp_min(1e-12)).mean()


def perturb_model(source: GCLModel, eta: float) -> GCLModel:
    target = copy.deepcopy(source)
    with torch.no_grad():
        source_params = dict(source.named_parameters())
        for name, parameter in target.named_parameters():
            original = source_params[name]
            if name.startswith("projectors"):
                parameter.copy_(original)
            else:
                # unbiased=False 同时兼容只有一个元素的标量参数。
                scale = original.std(unbiased=False).clamp_min(1e-12)
                parameter.copy_(original + eta * torch.randn_like(original) * scale)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return target


def project_simplex(values: np.ndarray) -> np.ndarray:
    """将向量投影到概率单纯形。"""
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1
    indices = np.arange(1, values.size + 1)
    valid = ordered - cumulative / indices > 0
    rho = indices[valid][-1]
    theta = cumulative[valid][-1] / rho
    projected = np.maximum(values - theta, 0)
    return projected / projected.sum()


@dataclass
class TrainState:
    augmentation_probabilities: np.ndarray | None = None
