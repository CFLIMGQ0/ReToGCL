"""在四个共同 TU 图分类数据集上运行五个新增 SOTA 的分层五折实验。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.evaluation import (
    METRIC_KEYS,
    classification_metrics,
    linear_probe_five_fold_metrics,
    stratified_folds,
    summarize,
)
from src.baselines.gcl_baselines import augment_graph, ensure_features, set_seed
from src.scripts.graph_classification_datasets import (
    GRAPH_CLASSIFICATION_DATASETS,
    MULTITASK_DATASETS,
    aggregate_multitask_results,
    dataset_protocol,
    load_graphs,
    multitask_graphs,
)
from src.models.sota_graph_models import (
    CONTRASTIVE_METHODS,
    SUPERVISED_METHODS,
    build_contrastive_model,
    build_supervised_model,
)


METHODS = ("histograph", "uniimb", "difflift", "balancegcl", "khangcl")
DATASETS = GRAPH_CLASSIFICATION_DATASETS
SOURCE_COMMITS = {
    "histograph": "e6e1d01cb0ffbf145bb06631c8d115f517e9db22",
    "uniimb": "fa36c7bde799d03dae66af5a3ec70dac85c0c642",
    "difflift": "899e74a4f857b4fa24d13341547dc8de4b2fcd95",
    "balancegcl": "6d68ce70eac1b8507fa5d87c3acfc4e4b2c1d68d",
    "khangcl": "249b7b18878af7ecfb7461285715d4a5109e388f",
}
DEFAULT_EPOCHS = {
    "histograph": 40,
    "uniimb": 40,
    "difflift": 40,
    "balancegcl": 40,
    "khangcl": 40,
}
DEFAULT_LR = {
    "histograph": 1e-3,
    "uniimb": 1e-3,
    "difflift": 1e-3,
    "balancegcl": 1e-3,
    "khangcl": 1e-3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sota"))
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda:0 等")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None, help="固定外层五折索引的随机种子")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--topology-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练或预训练轮数")
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--refresh-metrics", action="store_true",
        help="复用 checkpoint，但重新计算并覆盖已有任务的评测指标",
    )
    parser.add_argument("--merge-only", action="store_true", help="仅汇总已有单任务结果")
    parser.add_argument("--no-merge", action="store_true", help="并行子任务结束时不写共享汇总文件")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{int(np.argmax(free))}")


def save_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _topology_features(graph: Data, dimension: int) -> tuple[torch.Tensor, torch.Tensor]:
    node_count = graph.num_nodes
    adjacency = to_dense_adj(graph.edge_index, max_num_nodes=node_count)[0].float().cpu()
    adjacency = (adjacency > 0).float()
    degree = adjacency.sum(dim=1).clamp_min(1)
    transition = adjacency / degree[:, None]
    power = torch.eye(node_count)
    returns = []
    for _ in range(dimension):
        power = power @ transition
        returns.append(power.diag())
    rw_pe = torch.stack(returns, dim=1)

    inv_sqrt = degree.rsqrt()
    laplacian = torch.eye(node_count) - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    available = min(dimension, max(node_count - 1, 0))
    lap_pe = torch.zeros(node_count, dimension)
    if available:
        selected_values = eigenvalues[1 : available + 1]
        selected_vectors = eigenvectors[:, 1 : available + 1]
        # 对特征向量正负号不变，对应论文中的 l(h, λ)+l(-h, λ)。
        lap_pe[:, :available] = selected_vectors.abs() * selected_values.unsqueeze(0)
    return rw_pe, lap_pe


def attach_topology_features(
    graphs: list[Data], dataset_name: str, dimension: int, output_dir: Path
) -> list[Data]:
    cache_path = output_dir / "topology_cache" / f"{dataset_name}_dim{dimension}.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if len(cached) == len(graphs):
            for graph, features in zip(graphs, cached):
                graph.rw_pe, graph.lap_pe = features["rw_pe"], features["lap_pe"]
            return graphs

    cached = []
    for graph in tqdm(graphs, desc=f"{dataset_name} 拓扑编码", unit="图"):
        rw_pe, lap_pe = _topology_features(graph, dimension)
        graph.rw_pe, graph.lap_pe = rw_pe, lap_pe
        cached.append({"rw_pe": rw_pe, "lap_pe": lap_pe})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, cache_path)
    return graphs


def _inner_validation_indices(labels: torch.Tensor, indices: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    validation_parts = []
    training_parts = []
    for label in labels[indices].unique(sorted=True):
        class_indices = indices[labels[indices] == label]
        class_indices = class_indices[torch.randperm(class_indices.numel(), generator=generator)]
        validation_count = max(1, round(class_indices.numel() * 0.1))
        validation_parts.append(class_indices[:validation_count])
        training_parts.append(class_indices[validation_count:])
    return torch.cat(training_parts), torch.cat(validation_parts)


@torch.no_grad()
def evaluate_supervised(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for data in loader:
        data = data.to(device)
        logits, _, _ = model(data)
        correct += int((logits.argmax(dim=-1) == data.y).sum())
        total += data.num_graphs
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_supervised_metrics(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    labels, predictions = [], []
    for data in loader:
        data = data.to(device)
        logits, _, _ = model(data)
        labels.append(data.y.cpu())
        predictions.append(logits.argmax(dim=-1).cpu())
    return classification_metrics(torch.cat(labels), torch.cat(predictions))


def run_supervised(
    args: argparse.Namespace,
    method: str,
    dataset_name: str,
    graphs: list[Data],
    in_dim: int,
    num_classes: int,
    device: torch.device,
) -> dict:
    labels = torch.tensor([int(graph.y) for graph in graphs])
    split_seed = args.seed if args.split_seed is None else args.split_seed
    test_folds = stratified_folds(labels, args.folds, split_seed)
    epochs = args.epochs or DEFAULT_EPOCHS[method]
    batch_size = min(args.batch_size, 32 if dataset_name == "COLLAB" else args.batch_size)
    fold_metrics = {key: [] for key in METRIC_KEYS}

    for fold_index, test_indices in enumerate(test_folds, start=1):
        checkpoint = args.output_dir / "checkpoints" / method / f"{dataset_name}_fold{fold_index}.pt"
        log_path = args.output_dir / "logs" / method / f"{dataset_name}_fold{fold_index}.jsonl"
        all_indices = torch.arange(len(graphs))
        train_pool = all_indices[~torch.isin(all_indices, test_indices)]
        train_indices, val_indices = _inner_validation_indices(
            labels, train_pool, split_seed + fold_index
        )
        train_loader = DataLoader([graphs[i] for i in train_indices], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader([graphs[i] for i in val_indices], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader([graphs[i] for i in test_indices], batch_size=batch_size, shuffle=False)

        set_seed(args.seed + fold_index)
        model = build_supervised_model(
            method, in_dim, args.hidden_dim, args.layers, num_classes, args.topology_dim
        ).to(device)
        if checkpoint.exists() and not args.force:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state["model"])
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=DEFAULT_LR[method], weight_decay=1e-4
            )
            best_validation = -1.0
            stale_epochs = 0
            best_state = None
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                progress = tqdm(
                    range(1, epochs + 1),
                    desc=f"{method}/{dataset_name}/折{fold_index}", unit="轮", leave=False,
                )
                for epoch in progress:
                    model.train()
                    total_loss = total_graphs = 0
                    for data in train_loader:
                        data = data.to(device)
                        optimizer.zero_grad()
                        logits, _, regularizer = model(data)
                        loss = F.cross_entropy(logits, data.y) + 0.1 * regularizer
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                        total_loss += float(loss.detach()) * data.num_graphs
                        total_graphs += data.num_graphs
                    validation = evaluate_supervised(model, val_loader, device)
                    mean_loss = total_loss / max(total_graphs, 1)
                    record = {"epoch": epoch, "loss": mean_loss, "validation_accuracy": validation}
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_file.flush()
                    progress.set_postfix(loss=f"{mean_loss:.4f}", val=f"{validation:.4f}")
                    if validation > best_validation:
                        best_validation = validation
                        stale_epochs = 0
                        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    else:
                        stale_epochs += 1
                    if stale_epochs >= args.patience:
                        break
            if best_state is None:
                raise RuntimeError("训练未产生可用 checkpoint")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": best_state,
                    "method": method,
                    "dataset": dataset_name,
                    "fold": fold_index,
                    "best_validation_accuracy": best_validation,
                },
                checkpoint,
            )
            model.load_state_dict(best_state)
        current_metrics = evaluate_supervised_metrics(model, test_loader, device)
        for key in METRIC_KEYS:
            fold_metrics[key].append(current_metrics[key])

    return {
        "method": method,
        "dataset": dataset_name,
        "task": "graph_classification",
        "training": "supervised_end_to_end_per_fold",
        "train_epochs_max": epochs,
        "fold_protocol": "stratified_5_fold_with_inner_validation",
        "accuracy_percent": summarize(fold_metrics["accuracy"]),
        "nmi_percent": summarize(fold_metrics["nmi"]),
        "ari_percent": summarize(fold_metrics["ari"]),
        "macro_f1_percent": summarize(fold_metrics["macro_f1"]),
    }


@torch.no_grad()
def extract_contrastive_embeddings(
    model: torch.nn.Module, graphs: list[Data], batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings, labels = [], []
    for data in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        data = data.to(device)
        embeddings.append(model.encode(data).cpu())
        labels.append(data.y.cpu())
    return torch.cat(embeddings), torch.cat(labels)


def run_contrastive(
    args: argparse.Namespace,
    method: str,
    dataset_name: str,
    graphs: list[Data],
    in_dim: int,
    num_classes: int,
    device: torch.device,
) -> dict:
    epochs = args.epochs or DEFAULT_EPOCHS[method]
    batch_size = min(args.batch_size, 64 if dataset_name == "COLLAB" else args.batch_size)
    checkpoint = args.output_dir / "checkpoints" / method / f"{dataset_name}.pt"
    log_path = args.output_dir / "logs" / method / f"{dataset_name}.jsonl"
    set_seed(args.seed)
    model = build_contrastive_model(
        method, in_dim, args.hidden_dim, args.layers, num_classes
    ).to(device)
    if checkpoint.exists() and not args.force:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR[method], weight_decay=1e-5)
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            progress = tqdm(range(1, epochs + 1), desc=f"{method}/{dataset_name}", unit="轮")
            for epoch in progress:
                model.train()
                total_loss = total_graphs = 0
                for data in loader:
                    data = data.to(device)
                    first_kind, second_kind = np.random.choice(
                        ("node_drop", "edge_drop", "attr_mask", "subgraph"), size=2
                    )
                    first = model.encode(augment_graph(data, str(first_kind), 0.2))
                    second = model.encode(augment_graph(data, str(second_kind), 0.2))
                    loss = model.contrastive_loss(first, second)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += float(loss.detach()) * data.num_graphs
                    total_graphs += data.num_graphs
                mean_loss = total_loss / max(total_graphs, 1)
                log_file.write(json.dumps({"epoch": epoch, "loss": mean_loss}) + "\n")
                log_file.flush()
                progress.set_postfix(loss=f"{mean_loss:.4f}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "method": method, "dataset": dataset_name, "epochs": epochs},
            checkpoint,
        )

    embeddings, labels = extract_contrastive_embeddings(model, graphs, batch_size, device)
    fold_metrics = linear_probe_five_fold_metrics(
        embeddings, labels, device=device, seed=args.seed,
        split_seed=args.split_seed,
        folds=args.folds, epochs=args.probe_epochs,
    )
    return {
        "method": method,
        "dataset": dataset_name,
        "task": "graph_classification",
        "training": "self_supervised_pretrain_then_linear_probe",
        "pretrain_epochs": epochs,
        "fold_protocol": "stratified_5_fold_linear_probe",
        "accuracy_percent": summarize(fold_metrics["accuracy"]),
        "nmi_percent": summarize(fold_metrics["nmi"]),
        "ari_percent": summarize(fold_metrics["ari"]),
        "macro_f1_percent": summarize(fold_metrics["macro_f1"]),
    }


def merge_results(output_dir: Path) -> Path:
    experiments = []
    for path in sorted((output_dir / "jobs").glob("*.json")):
        experiments.append(json.loads(path.read_text(encoding="utf-8")))
    result = {
        "metadata": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "methods": list(METHODS),
            "datasets": list(DATASETS),
            "metrics": ["accuracy_percent", "nmi_percent", "ari_percent", "macro_f1_percent"],
            "implementation": "project_pyg_adapter_from_paper_and_official_source",
            "source_commits": SOURCE_COMMITS,
        },
        "experiments": sorted(experiments, key=lambda item: (item["method"], item["dataset"])),
    }
    path = output_dir / "five_fold_results.json"
    save_json(path, result)
    return path


def main() -> None:
    args = parse_args()
    args.data_root = (PROJECT_ROOT / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    if args.merge_only:
        print(f"已汇总：{merge_results(args.output_dir)}")
        return
    device = choose_device(args.device)
    print(f"使用设备：{device}")

    combinations = [(method, dataset) for method in args.methods for dataset in args.datasets]
    for method, dataset_name in tqdm(combinations, desc="SOTA 实验总进度", unit="组"):
        job_path = args.output_dir / "jobs" / f"{method}_{dataset_name}.json"
        if job_path.exists() and not args.force and not args.refresh_metrics:
            existing_result = json.loads(job_path.read_text(encoding="utf-8"))
            required_metrics = {
                "accuracy_percent", "nmi_percent", "ari_percent", "macro_f1_percent"
            }
            if required_metrics.issubset(existing_result):
                print(f"复用已有四指标结果：{method}/{dataset_name}")
                continue
        graphs, in_dim, num_classes = load_graphs(args.data_root, dataset_name)
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
            if method == "uniimb":
                current_graphs = attach_topology_features(
                    current_graphs, current_name, args.topology_dim, args.output_dir
                )
            if method in SUPERVISED_METHODS:
                current = run_supervised(
                    args, method, current_name, current_graphs, in_dim, num_classes, device
                )
            elif method in CONTRASTIVE_METHODS:
                current = run_contrastive(
                    args, method, current_name, current_graphs, in_dim, num_classes, device
                )
            else:
                raise ValueError(method)
            if dataset_name in MULTITASK_DATASETS:
                current["task_index"] = task_index
                current["sample_count"] = len(current_graphs)
            task_results.append(current)
        result = (
            aggregate_multitask_results(dataset_name, task_results)
            if dataset_name in MULTITASK_DATASETS else task_results[0]
        )
        result.update({"method": method, "dataset": dataset_name})
        result.update(
            {
                "seed": args.seed,
                "split_seed": args.seed if args.split_seed is None else args.split_seed,
                "source_commit": SOURCE_COMMITS[method],
                "implementation": "project_pyg_adapter",
                "dataset_protocol": dataset_protocol(dataset_name),
            }
        )
        save_json(job_path, result)
        print(
            f"{method}/{dataset_name}: "
            f"{result['accuracy_percent']['mean']:.2f} ± {result['accuracy_percent']['std']:.2f}"
        )
    if not args.no_merge:
        print(f"已汇总：{merge_results(args.output_dir)}")


if __name__ == "__main__":
    main()
