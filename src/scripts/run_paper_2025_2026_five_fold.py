"""运行二十篇 2025--2026 年论文模型的 TU 图分类五折实验。"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, scatter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.evaluation import METRIC_KEYS, classification_metrics, stratified_folds, summarize
from src.baselines.gcl_baselines import ensure_features
from src.scripts.graph_classification_datasets import (
    GRAPH_CLASSIFICATION_DATASETS,
    MULTITASK_DATASETS,
    aggregate_multitask_results,
    dataset_protocol,
    load_graphs as load_base_graphs,
    multitask_graphs,
)
from src.models.paper_2025_2026_models import METHODS, METHOD_SPECS, build_model


DATASETS = GRAPH_CLASSIFICATION_DATASETS
STRUCT_CACHE_ROOT = PROJECT_ROOT / "outputs/paper_sota_2025_2026/preprocess_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_sota_2025_2026"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{int(np.argmax(free))}")


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _dfs_ranks(num_nodes: int, edge_index: Tensor, deg: Tensor) -> Tensor:
    adjacency = [[] for _ in range(num_nodes)]
    for source, target in edge_index.t().tolist():
        adjacency[source].append(target)
    starts = sorted(range(num_nodes), key=lambda i: (-float(deg[i]), i))
    visited, order = set(), []
    for start in starts:
        if start in visited:
            continue
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node); order.append(node)
            neighbors = sorted(adjacency[node], key=lambda i: (float(deg[i]), -i))
            stack.extend(neighbors)
    ranks = torch.empty(num_nodes, dtype=torch.long)
    ranks[torch.tensor(order)] = torch.arange(num_nodes)
    return ranks


def attach_structural_features(graph: Data) -> Data:
    num_nodes = graph.num_nodes
    source, target = graph.edge_index
    deg = degree(source, num_nodes, dtype=torch.float32)
    max_degree = deg.max().clamp_min(1)
    neighbor_degree = scatter(deg[source], target, dim=0, dim_size=num_nodes, reduce="mean")
    feature_norm = graph.x.norm(dim=-1)
    feature_scale = feature_norm.max().clamp_min(1)
    score = deg + 1e-3 * feature_norm
    rank = torch.empty(num_nodes)
    rank[torch.argsort(score, stable=True)] = torch.linspace(0, 1, num_nodes)
    graph.struct_pe = torch.stack([
        deg / max_degree,
        torch.log1p(deg) / torch.log1p(max_degree),
        neighbor_degree / max_degree,
        (deg - neighbor_degree).abs() / max_degree,
        deg.rsqrt().where(deg > 0, torch.zeros_like(deg)),
        feature_norm / feature_scale,
        rank,
        (deg >= 2).float(),
    ], -1)
    graph.struct_id = ((deg.long() * 31 + neighbor_degree.round().long() * 17 + (rank * 31).long()) % 256)
    graph.search_rank = _dfs_ranks(num_nodes, graph.edge_index, deg)

    edge_count = int(graph.edge_index.size(1))
    undirected_edges = edge_count / 2
    density = 0.0 if num_nodes < 2 else undirected_edges / (num_nodes * (num_nodes - 1) / 2)
    edge_gap = (deg[source] - deg[target]).abs().mean() / max_degree if edge_count else deg.new_zeros(())
    self_loops = (source == target).float().mean() if edge_count else deg.new_zeros(())
    x_mean, x_std = graph.x.mean(), graph.x.std(unbiased=False)
    graph.graph_stats = torch.tensor([[
        math.log1p(num_nodes) / 8,
        math.log1p(undirected_edges) / 10,
        density,
        float(deg.mean() / max(num_nodes, 1)),
        float(deg.std(unbiased=False) / max(num_nodes, 1)),
        float(deg.max() / max(num_nodes, 1)),
        float(deg.min() / max(num_nodes, 1)),
        float((deg == 0).float().mean()),
        float((deg == 1).float().mean()),
        float((deg == 2).float().mean()),
        float((deg >= 3).float().mean()),
        max(0.0, (undirected_edges - num_nodes + 1) / max(num_nodes, 1)),
        float(x_mean), float(x_std), float(edge_gap), float(self_loops),
    ]], dtype=torch.float32)

    quantiles = torch.linspace(0.125, 1.0, 8)
    thresholds = torch.quantile(deg, quantiles)
    node_curve, edge_curve = [], []
    for threshold in thresholds:
        node_curve.append((deg <= threshold).float().mean())
        edge_curve.append(((deg[source] <= threshold) & (deg[target] <= threshold)).float().mean() if edge_count else deg.new_zeros(()))
    graph.filtration = torch.stack(node_curve + edge_curve).view(1, 16)
    return graph


def load_graphs(root: Path, name: str) -> tuple[list[Data], int, int]:
    cache = STRUCT_CACHE_ROOT / f"{name}.pt"
    if cache.exists():
        value = torch.load(cache, map_location="cpu", weights_only=False)
        return value["graphs"], value["in_dim"], value["num_classes"]
    base_graphs, in_dim, num_classes = load_base_graphs(root, name)
    graphs = []
    for graph in tqdm(base_graphs, desc=f"{name} 结构预处理", unit="图", leave=False):
        graphs.append(attach_structural_features(graph))
    value = {"graphs": graphs, "in_dim": in_dim, "num_classes": num_classes}
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    # 多模型并发预处理同一数据集时，最后一个完整缓存原子覆盖即可。
    temporary.replace(cache)
    return value["graphs"], value["in_dim"], value["num_classes"]


def inner_split(labels: Tensor, train_pool: Tensor, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_parts, validation_parts = [], []
    for label in labels[train_pool].unique(sorted=True):
        indices = train_pool[labels[train_pool] == label]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        validation_count = min(max(1, round(indices.numel() * 0.1)), max(indices.numel() - 1, 1))
        validation_parts.append(indices[:validation_count])
        train_parts.append(indices[validation_count:])
    return torch.cat(train_parts), torch.cat(validation_parts)


def method_batch_size(method: str, dataset: str, requested: int) -> int:
    if dataset.startswith(("ABIDE", "ADHD200")) and method in {"rsnn", "topoformer", "rspool"}:
        return min(requested, 8)
    if dataset.startswith(("ABIDE", "ADHD200")):
        return min(requested, 32)
    if dataset == "COLLAB" and method in {"rsnn", "topoformer", "rspool"}:
        return min(requested, 12)
    if dataset == "COLLAB":
        return min(requested, 32)
    if method in {"rsnn", "topoformer", "rspool"}:
        return min(requested, 32)
    return requested


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval(); labels, predictions = [], []
    for data in loader:
        data = data.to(device)
        logits, _, _ = model(data)
        labels.append(data.y.cpu()); predictions.append(logits.argmax(-1).cpu())
    return classification_metrics(torch.cat(labels), torch.cat(predictions))


def train_fold(args: argparse.Namespace, method: str, dataset_name: str, graphs: list[Data], in_dim: int, classes: int, fold: int, train_indices: Tensor, validation_indices: Tensor, test_indices: Tensor, device: torch.device) -> dict[str, float]:
    batch_size = method_batch_size(method, dataset_name, args.batch_size)
    train_loader = DataLoader([graphs[int(i)] for i in train_indices], batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader([graphs[int(i)] for i in validation_indices], batch_size=batch_size)
    test_loader = DataLoader([graphs[int(i)] for i in test_indices], batch_size=batch_size)
    checkpoint = args.output_dir / "checkpoints" / method / f"{dataset_name}_fold{fold}.pt"
    log_path = args.output_dir / "logs" / method / f"{dataset_name}_fold{fold}.jsonl"
    set_seed(args.seed + fold)
    model = build_model(method, in_dim, args.hidden_dim, args.layers, classes).to(device)
    if checkpoint.exists() and not args.force:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_validation, stale, best_state = -1.0, 0, None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            progress = tqdm(range(1, args.epochs + 1), desc=f"{method}/{dataset_name}/折{fold}", unit="轮", leave=False)
            for epoch in progress:
                model.train(); total_loss = total_graphs = 0
                for data in train_loader:
                    data = data.to(device); optimizer.zero_grad()
                    logits, _, regularizer = model(data)
                    loss = F.cross_entropy(logits, data.y) + regularizer
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"{method}/{dataset_name}/折{fold} 出现非有限损失")
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
                    total_loss += float(loss.detach()) * data.num_graphs; total_graphs += data.num_graphs
                validation = evaluate(model, validation_loader, device)["accuracy"]
                mean_loss = total_loss / max(total_graphs, 1)
                log_file.write(json.dumps({"epoch": epoch, "loss": mean_loss, "validation_accuracy": validation}, ensure_ascii=False) + "\n"); log_file.flush()
                progress.set_postfix(loss=f"{mean_loss:.4f}", val=f"{validation:.4f}")
                if validation > best_validation:
                    best_validation, stale = validation, 0
                    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                else:
                    stale += 1
                if stale >= args.patience:
                    break
        if best_state is None:
            raise RuntimeError("训练未产生可用 checkpoint")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best_state, "best_validation_accuracy": best_validation}, checkpoint)
        model.load_state_dict(best_state)
    return evaluate(model, test_loader, device)


def run_five_fold(args: argparse.Namespace, method: str, dataset_name: str, graphs: list[Data], in_dim: int, classes: int, device: torch.device) -> dict:
    labels = torch.tensor([int(graph.y) for graph in graphs])
    test_folds = stratified_folds(labels, args.folds, args.seed)
    fold_metrics = {key: [] for key in METRIC_KEYS}; all_indices = torch.arange(len(graphs))
    for fold, test_indices in enumerate(test_folds, start=1):
        train_pool = all_indices[~torch.isin(all_indices, test_indices)]
        train_indices, validation_indices = inner_split(labels, train_pool, args.seed + fold)
        current = train_fold(args, method, dataset_name, graphs, in_dim, classes, fold, train_indices, validation_indices, test_indices, device)
        for key in METRIC_KEYS:
            fold_metrics[key].append(current[key])
    return {"training": "supervised_end_to_end_per_fold", **{f"{key}_percent": summarize(values) for key, values in fold_metrics.items()}}


def merge_results(output_dir: Path, folds: int = 5) -> Path:
    experiments = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output_dir / "jobs").glob("*.json"))]
    result = {
        "metadata": {
            "updated_at": datetime.now(timezone.utc).isoformat(), "methods": list(METHODS), "datasets": list(DATASETS),
            "folds": folds, "metrics": [f"{key}_percent" for key in METRIC_KEYS],
            "implementation": "project_pyg_adapter_from_2025_2026_papers_and_official_sources",
            "papers": {method: {"title": spec[0], "venue": spec[1], "year": spec[2], "source_commit": spec[3]} for method, spec in METHOD_SPECS.items()},
        },
        "experiments": sorted(experiments, key=lambda item: (item["method"], item["dataset"])),
    }
    path = output_dir / "five_fold_results.json"; save_json(path, result); return path


def smoke_test(args: argparse.Namespace, device: torch.device) -> None:
    dataset_name = args.datasets[0]
    graphs, in_dim, classes = load_graphs(args.data_root, dataset_name)
    batch = next(iter(DataLoader(graphs[: min(8, len(graphs))], batch_size=min(8, len(graphs))))).to(device)
    for method in tqdm(args.methods, desc="二十模型冒烟测试", unit="模型"):
        set_seed(args.seed); model = build_model(method, in_dim, args.hidden_dim, args.layers, classes).to(device); model.train()
        logits, embedding, regularizer = model(batch); loss = F.cross_entropy(logits, batch.y) + regularizer
        loss.backward()
        if logits.shape != (batch.num_graphs, classes) or not torch.isfinite(loss):
            raise RuntimeError(f"{method} 冒烟测试失败")
        print(f"通过 {method}: logits={tuple(logits.shape)}, embedding={tuple(embedding.shape)}, loss={float(loss):.4f}")


def main() -> None:
    args = parse_args()
    args.data_root = (PROJECT_ROOT / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    if args.merge_only:
        print(f"已汇总：{merge_results(args.output_dir, args.folds)}"); return
    device = choose_device(args.device); print(f"使用设备：{device}")
    if args.smoke_test:
        smoke_test(args, device); return
    combinations = [(method, dataset) for method in args.methods for dataset in args.datasets]
    for method, dataset_name in tqdm(combinations, desc="二十论文实验总进度", unit="组"):
        job_path = args.output_dir / "jobs" / f"{method}_{dataset_name}.json"
        if job_path.exists() and not args.force:
            print(f"复用已有结果：{method}/{dataset_name}"); continue
        graphs, in_dim, classes = load_graphs(args.data_root, dataset_name)
        task_graph_sets = (
            [multitask_graphs(graphs, task_index) for task_index in range(graphs[0].task_y.numel())]
            if dataset_name in MULTITASK_DATASETS else [graphs]
        )
        task_results = []
        for task_index, current_graphs in enumerate(task_graph_sets):
            current_name = (
                f"{dataset_name}__task{task_index:02d}"
                if dataset_name in MULTITASK_DATASETS else dataset_name
            )
            current = run_five_fold(args, method, current_name, current_graphs, in_dim, classes, device)
            if dataset_name in MULTITASK_DATASETS:
                current["task_index"] = task_index
                current["sample_count"] = len(current_graphs)
            task_results.append(current)
        result = (
            aggregate_multitask_results(dataset_name, task_results)
            if dataset_name in MULTITASK_DATASETS else task_results[0]
        )
        title, venue, year, commit = METHOD_SPECS[method]
        result.update({"method": method, "dataset": dataset_name, "task": "graph_classification", "fold_protocol": f"stratified_{args.folds}_fold_with_inner_validation", "seed": args.seed, "paper": title, "venue": venue, "year": year, "source_commit": commit, "implementation": "project_pyg_adapter", "dataset_protocol": dataset_protocol(dataset_name)})
        save_json(job_path, result)
        print(f"{method}/{dataset_name}: {result['accuracy_percent']['mean']:.2f} ± {result['accuracy_percent']['std']:.2f}")
    if not args.no_merge:
        print(f"已汇总：{merge_results(args.output_dir, args.folds)}")


if __name__ == "__main__":
    main()
