"""诊断语义、拓扑和节点分布三个空间对图类别关系的辨识能力。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.evaluation import classification_metrics, stratified_folds, summarize
from src.baselines.gcl_baselines import GCLModel, ensure_features, set_seed


DATASETS = ("NCI1", "PROTEINS", "COLLAB", "MUTAG", "COLORS-3")
SPACES = ("semantic", "topology", "distribution", "uniform_fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/baselines/checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("outputs/multispace/diagnostics.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/multispace/cache"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--prototypes", type=int, default=16)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--prototype-fit-nodes", type=int, default=50000)
    parser.add_argument("--max-pairs", type=int, default=200000)
    parser.add_argument("--boundary-fraction", type=float, default=0.2)
    parser.add_argument("--neighbors", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{int(np.argmax(free))}")


def topology_descriptor(data) -> Tensor:
    """返回不依赖节点排列的轻量结构统计量。"""
    node_count = int(data.num_nodes)
    edge_count = int(data.edge_index.size(1) // 2)
    denominator = max(node_count - 1, 1)
    degree = torch.bincount(data.edge_index[0], minlength=node_count).float()
    normalized_degree = degree / denominator
    quantiles = torch.quantile(normalized_degree, torch.tensor([0.25, 0.5, 0.75]))
    histogram = torch.histc(normalized_degree, bins=16, min=0.0, max=1.0)
    histogram = histogram / max(node_count, 1)
    density = 2 * edge_count / max(node_count * (node_count - 1), 1)
    cycle_rank = max(edge_count - node_count + 1, 0) / max(node_count, 1)
    scalars = torch.tensor([
        math.log1p(node_count),
        math.log1p(edge_count),
        density,
        float(normalized_degree.mean()),
        float(normalized_degree.std(unbiased=False)),
        float(normalized_degree.max()) if node_count else 0.0,
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
        float((degree == 0).float().mean()),
        float((degree == 1).float().mean()),
        cycle_rank,
    ])
    return torch.cat([scalars, histogram])


@torch.no_grad()
def extract_features(
    dataset_name: str,
    data_root: Path,
    checkpoint_dir: Path,
    cache_dir: Path,
    device: torch.device,
    hidden_dim: int,
    layers: int,
    batch_size: int,
    force: bool,
) -> dict[str, Tensor]:
    cache_path = cache_dir / f"graphcl_{dataset_name}.pt"
    if cache_path.exists() and not force:
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    dataset = TUDataset(
        root=str(data_root), name=dataset_name, cleaned=False,
        use_node_attr=dataset_name == "COLORS-3",
    )
    dataset.transform = ensure_features
    model = GCLModel(
        max(1, dataset.num_node_features), hidden_dim, layers, "graph", "graphcl"
    ).to(device)
    checkpoint_path = checkpoint_dir / f"graphcl_{dataset_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"缺少 GraphCL checkpoint：{checkpoint_path}")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    model.eval()

    semantic_parts: list[Tensor] = []
    node_parts: list[Tensor] = []
    node_graph_parts: list[Tensor] = []
    label_parts: list[Tensor] = []
    graph_offset = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in tqdm(loader, desc=f"{dataset_name} 提取GraphCL表示", unit="批"):
        batch = batch.to(device)
        semantic, nodes = model.encode(batch)
        semantic_parts.append(semantic.cpu())
        node_parts.append(nodes.cpu())
        node_graph_parts.append(batch.batch.cpu() + graph_offset)
        label_parts.append(batch.y.view(-1).long().cpu())
        graph_offset += batch.num_graphs

    topology = torch.stack([
        topology_descriptor(ensure_features(dataset[index]))
        for index in tqdm(range(len(dataset)), desc=f"{dataset_name} 提取拓扑特征", unit="图")
    ])
    content = {
        "semantic": torch.cat(semantic_parts),
        "nodes": torch.cat(node_parts),
        "node_graph": torch.cat(node_graph_parts),
        "topology": topology,
        "labels": torch.cat(label_parts),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(content, cache_path)
    return content


def train_linear_probe(
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = ((train_x - mean) / std).to(device)
    test_x = ((test_x - mean) / std).to(device)
    train_y = train_y.to(device)
    torch.manual_seed(seed)
    classifier = nn.Linear(train_x.size(1), num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(classifier(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probability = classifier(test_x).softmax(dim=-1).cpu()
    return probability.argmax(dim=-1), probability


def fit_kmeans(
    values: Tensor,
    clusters: int,
    iterations: int,
    max_points: int,
    seed: int,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    if values.size(0) > max_points:
        values = values[torch.randperm(values.size(0), generator=generator)[:max_points]]
    clusters = min(clusters, values.size(0))
    centers = values[torch.randperm(values.size(0), generator=generator)[:clusters]].clone()
    for _ in range(iterations):
        assignment = torch.cdist(values, centers).argmin(dim=1)
        updated = centers.clone()
        for cluster in range(clusters):
            members = values[assignment == cluster]
            if members.numel():
                updated[cluster] = members.mean(dim=0)
        if torch.allclose(updated, centers, atol=1e-4, rtol=1e-4):
            centers = updated
            break
        centers = updated
    return centers


def prototype_histograms(nodes: Tensor, node_graph: Tensor, centers: Tensor, graph_count: int) -> Tensor:
    histograms = torch.zeros(graph_count, centers.size(0))
    chunk_size = 8192
    for start in range(0, nodes.size(0), chunk_size):
        stop = min(start + chunk_size, nodes.size(0))
        assignment = torch.cdist(nodes[start:stop], centers).argmin(dim=1)
        flat = node_graph[start:stop] * centers.size(0) + assignment
        histograms.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
    histograms = histograms / histograms.sum(dim=1, keepdim=True).clamp_min(1)
    return histograms


def rbf_similarity(train: Tensor, test: Tensor) -> Tensor:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    train = (train - mean) / std
    test = (test - mean) / std
    generator = torch.Generator().manual_seed(17)
    sample = train
    if train.size(0) > 1024:
        sample = train[torch.randperm(train.size(0), generator=generator)[:1024]]
    squared_sample = torch.cdist(sample, sample).square()
    positive = squared_sample[squared_sample > 0]
    bandwidth = positive.median().clamp_min(1e-6) if positive.numel() else torch.tensor(1.0)
    return torch.exp(-torch.cdist(test, test).square() / bandwidth)


def semantic_similarity(values: Tensor) -> Tensor:
    values = F.normalize(values, dim=-1)
    return ((values @ values.t()) + 1) / 2


def distribution_similarity(histograms: Tensor) -> Tensor:
    roots = histograms.clamp_min(0).sqrt()
    return (roots @ roots.t()).clamp(0, 1)


def sampled_pair_values(
    similarity: Tensor, labels: Tensor, max_pairs: int, seed: int
) -> tuple[Tensor, Tensor]:
    upper = torch.triu_indices(labels.numel(), labels.numel(), offset=1)
    pair_labels = (labels[upper[0]] == labels[upper[1]])
    positive = pair_labels.nonzero(as_tuple=False).flatten()
    negative = (~pair_labels).nonzero(as_tuple=False).flatten()
    count = min(positive.numel(), negative.numel(), max_pairs // 2)
    generator = torch.Generator().manual_seed(seed)
    positive = positive[torch.randperm(positive.numel(), generator=generator)[:count]]
    negative = negative[torch.randperm(negative.numel(), generator=generator)[:count]]
    selected = torch.cat([positive, negative])
    scores = similarity[upper[0, selected], upper[1, selected]]
    truth = torch.cat([torch.ones(count), torch.zeros(count)])
    return scores, truth


def binary_auc(scores: Tensor, truth: Tensor) -> float:
    """使用含并列平均秩的 Mann-Whitney 统计量计算二分类 AUC。"""
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty(scores.numel(), dtype=torch.float64)
    start = 0
    while start < scores.numel():
        stop = start + 1
        while stop < scores.numel() and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    positive = truth.bool()
    positive_count = int(positive.sum())
    negative_count = truth.numel() - positive_count
    if not positive_count or not negative_count:
        return float("nan")
    statistic = ranks[positive].sum() - positive_count * (positive_count + 1) / 2
    return float(statistic / (positive_count * negative_count))


def recall_at_k(similarity: Tensor, labels: Tensor, k: int, subset: Tensor | None = None) -> float:
    k = min(k, max(labels.numel() - 1, 1))
    adjusted = similarity.clone()
    adjusted.fill_diagonal_(-float("inf"))
    neighbors = adjusted.topk(k, dim=1).indices
    scores = (labels[neighbors] == labels.unsqueeze(1)).float().mean(dim=1)
    if subset is not None:
        scores = scores[subset]
    return float(scores.mean()) if scores.numel() else float("nan")


def pair_metrics(
    similarity: Tensor,
    labels: Tensor,
    boundary: Tensor,
    neighbors: list[int],
    max_pairs: int,
    seed: int,
) -> dict[str, float]:
    scores, truth = sampled_pair_values(similarity, labels, max_pairs, seed)
    positive = scores[truth.bool()]
    negative = scores[~truth.bool()]
    result = {
        "pair_auc": binary_auc(scores, truth),
        "positive_mean": float(positive.mean()),
        "negative_mean": float(negative.mean()),
        "positive_negative_gap": float(positive.mean() - negative.mean()),
    }
    for k in neighbors:
        result[f"recall_at_{k}"] = recall_at_k(similarity, labels, k)
        result[f"boundary_recall_at_{k}"] = recall_at_k(similarity, labels, k, boundary)
    return result


def summarize_folds(folds: list[dict[str, float]]) -> dict[str, dict[str, float | list[float]]]:
    keys = folds[0].keys()
    return {key: summarize([fold[key] for fold in folds]) for key in keys}


def diagnose_dataset(args: argparse.Namespace, dataset_name: str, device: torch.device) -> dict:
    features = extract_features(
        dataset_name, args.data_root, args.checkpoint_dir, args.cache_dir,
        device, args.hidden_dim, args.layers, args.batch_size, args.force_extract,
    )
    semantic = features["semantic"].float()
    topology = features["topology"].float()
    nodes = features["nodes"].float()
    node_graph = features["node_graph"].long()
    labels = features["labels"].long()
    splits = stratified_folds(labels, args.folds, args.seed)
    fold_results = {space: [] for space in SPACES}
    classifier_results = []
    correlations = []

    for fold, test_indices in enumerate(tqdm(splits, desc=f"{dataset_name} 五折诊断", unit="折"), start=1):
        train_mask = torch.ones(labels.numel(), dtype=torch.bool)
        train_mask[test_indices] = False
        train_indices = train_mask.nonzero(as_tuple=False).flatten()
        predictions, probabilities = train_linear_probe(
            semantic[train_indices], labels[train_indices], semantic[test_indices],
            int(labels.max()) + 1, device, args.probe_epochs, args.seed + fold,
        )
        test_labels = labels[test_indices]
        entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=1)
        entropy = entropy / math.log(probabilities.size(1))
        boundary_count = max(1, math.ceil(test_indices.numel() * args.boundary_fraction))
        boundary = entropy.topk(boundary_count).indices
        classifier = classification_metrics(test_labels, predictions)
        classifier["boundary_accuracy"] = float(
            (predictions[boundary] == test_labels[boundary]).float().mean()
        )
        classifier["mean_uncertainty"] = float(entropy.mean())
        classifier_results.append(classifier)

        train_nodes = nodes[train_mask[node_graph]]
        centers = fit_kmeans(
            train_nodes, args.prototypes, args.kmeans_iterations,
            args.prototype_fit_nodes, args.seed + fold,
        )
        histograms = prototype_histograms(nodes, node_graph, centers, labels.numel())
        similarities = {
            "semantic": semantic_similarity(semantic[test_indices]),
            "topology": rbf_similarity(topology[train_indices], topology[test_indices]),
            "distribution": distribution_similarity(histograms[test_indices]),
        }
        similarities["uniform_fusion"] = sum(similarities.values()) / 3
        for space, matrix in similarities.items():
            fold_results[space].append(pair_metrics(
                matrix, test_labels, boundary, args.neighbors,
                args.max_pairs, args.seed + fold,
            ))

        pair_scores = {}
        for space in ("semantic", "topology", "distribution"):
            scores, _ = sampled_pair_values(
                similarities[space], test_labels, args.max_pairs, args.seed + fold
            )
            pair_scores[space] = scores
        correlations.append({
            "semantic_topology": float(torch.corrcoef(torch.stack([
                pair_scores["semantic"], pair_scores["topology"]
            ]))[0, 1]),
            "semantic_distribution": float(torch.corrcoef(torch.stack([
                pair_scores["semantic"], pair_scores["distribution"]
            ]))[0, 1]),
            "topology_distribution": float(torch.corrcoef(torch.stack([
                pair_scores["topology"], pair_scores["distribution"]
            ]))[0, 1]),
        })

    return {
        "dataset": dataset_name,
        "samples": labels.numel(),
        "classes": labels.unique().numel(),
        "classifier": summarize_folds(classifier_results),
        "spaces": {space: summarize_folds(values) for space, values in fold_results.items()},
        "space_correlations": summarize_folds(correlations),
    }


def save_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.data_root = resolve_path(args.data_root)
    args.checkpoint_dir = resolve_path(args.checkpoint_dir)
    args.output = resolve_path(args.output)
    args.cache_dir = resolve_path(args.cache_dir)
    if not 0 < args.boundary_fraction <= 1:
        raise SystemExit("--boundary-fraction 必须在 (0, 1] 范围内。")
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"使用设备：{device}")
    result = {
        "metadata": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "backbone": "GraphCL-GIN",
            "folds": args.folds,
            "seed": args.seed,
            "spaces": list(SPACES),
            "leakage_control": "fold-local topology scaling, node prototypes and classifier",
        },
        "datasets": [],
    }
    for dataset_name in tqdm(args.datasets, desc="多空间诊断总进度", unit="数据集"):
        result["datasets"].append(diagnose_dataset(args, dataset_name, device))
        save_json(args.output, result)
    print(f"诊断完成：{args.output}")


if __name__ == "__main__":
    main()
