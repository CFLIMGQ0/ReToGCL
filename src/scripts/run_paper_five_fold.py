"""运行十篇新增论文模型在五个 TU 图分类数据集上的统一五折实验。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.evaluation import METRIC_KEYS, classification_metrics, linear_probe_five_fold_metrics, stratified_folds, summarize
from src.baselines.gcl_baselines import augment_graph, ensure_features, set_seed
from src.scripts.graph_classification_datasets import (
    GRAPH_CLASSIFICATION_DATASETS,
    MULTITASK_DATASETS,
    aggregate_multitask_results,
    dataset_protocol,
    load_graphs,
    multitask_graphs,
)
from src.models.paper_graph_models import (
    CONTRASTIVE_METHODS,
    METHODS,
    PROMPT_METHODS,
    SUPERVISED_METHODS,
    build_model,
)


DATASETS = GRAPH_CLASSIFICATION_DATASETS
SOURCE_COMMITS = {
    "maxcutpool": "109b50b929bcd42c8b7a4bf4045d32fe28e7bf7d",
    "simplicial_mp": "28a9ca589db43b35193906527d10ce02ff921886",
    "del": "bc53614708784f9d7f6fc0617c4dbcba1db1aa45",
    "cellclat": "960b14d7aeb4328c70c2dbd24149f14e369d8adc",
    "plstm": "ac539b61890a793be873472f16d33b7d78f994d4",
    "abstaingnn": "7c533b3099f7a0e0678843c3d677708d438f4778",
    "edgeprompt": "fa3d7f4cf42054c4ed88067cb49ae64b31f4ce8e",
    "dualprism": "01b26b3b7dcfd92fd33158287d6ad7671b66b845",
    "invgnn": "c33cb6f39ab2ac71e017d61b109cacdac23e7140",
    "genvsexp": "15b95864803183b102fd358e33fdaaf77db8dffe",
}
DEFAULT_EPOCHS = {
    "maxcutpool": 40,
    "simplicial_mp": 40,
    "del": 40,
    "cellclat": 40,
    "plstm": 40,
    "abstaingnn": 50,
    "edgeprompt": 40,
    "dualprism": 40,
    "invgnn": 40,
    "genvsexp": 40,
}
DEFAULT_LR = {
    "maxcutpool": 1e-3,
    "simplicial_mp": 1e-3,
    "del": 1e-3,
    "cellclat": 1e-3,
    "plstm": 1e-3,
    "abstaingnn": 1e-3,
    "edgeprompt": 1e-3,
    "dualprism": 1e-3,
    "invgnn": 1e-3,
    "genvsexp": 1e-3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_sota"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None, help="固定外层五折索引的随机种子")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{int(np.argmax(free))}")


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def attach_cycle_basis_features(graphs: list[Data], dataset_name: str, output_dir: Path) -> list[Data]:
    cache = output_dir / "preprocess_cache" / f"{dataset_name}_cycle_basis_3_4.pt"
    if cache.exists():
        features = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        features = []
        for graph in tqdm(graphs, desc=f"{dataset_name} cycle-basis 编码", unit="图"):
            network = nx.Graph()
            network.add_nodes_from(range(graph.num_nodes))
            network.add_edges_from(graph.edge_index.t().tolist())
            counts = torch.zeros(graph.num_nodes, 2)
            for cycle in nx.cycle_basis(network):
                if len(cycle) in (3, 4):
                    counts[cycle, len(cycle) - 3] += 1
            scale = counts.amax(dim=0, keepdim=True).clamp_min(1)
            features.append(counts / scale)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, cache)
    if len(features) != len(graphs):
        raise RuntimeError(f"{dataset_name} cycle-basis 缓存长度不一致")
    for graph, feature in zip(graphs, features):
        graph.cycle_pe = feature
    return graphs


def inner_split(labels: Tensor, train_pool: Tensor, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_parts, validation_parts = [], []
    for label in labels[train_pool].unique(sorted=True):
        indices = train_pool[labels[train_pool] == label]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        count = min(max(1, round(indices.numel() * 0.1)), max(indices.numel() - 1, 1))
        validation_parts.append(indices[:count])
        train_parts.append(indices[count:])
    return torch.cat(train_parts), torch.cat(validation_parts)


def method_batch_size(method: str, dataset: str, requested: int) -> int:
    if dataset == "COLLAB" and method in {"maxcutpool", "simplicial_mp", "cellclat", "plstm"}:
        return min(requested, 8)
    if dataset == "COLLAB":
        return min(requested, 32)
    if method in {"simplicial_mp", "cellclat"}:
        return min(requested, 32)
    return requested


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    labels, predictions = [], []
    for data in loader:
        data = data.to(device)
        logits, _, _ = model(data)
        labels.append(data.y.cpu())
        predictions.append(logits.argmax(dim=-1).cpu())
    labels_tensor, prediction_tensor = torch.cat(labels), torch.cat(predictions)
    metrics = classification_metrics(labels_tensor, prediction_tensor)
    return metrics["accuracy"], metrics


def train_supervised_fold(
    args: argparse.Namespace,
    method: str,
    dataset_name: str,
    graphs: list[Data],
    in_dim: int,
    num_classes: int,
    fold_index: int,
    train_indices: Tensor,
    validation_indices: Tensor,
    test_indices: Tensor,
    device: torch.device,
    initial_state: dict[str, Tensor] | None = None,
) -> dict[str, float]:
    batch_size = method_batch_size(method, dataset_name, args.batch_size)
    train_loader = DataLoader([graphs[int(i)] for i in train_indices], batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader([graphs[int(i)] for i in validation_indices], batch_size=batch_size)
    test_loader = DataLoader([graphs[int(i)] for i in test_indices], batch_size=batch_size)
    checkpoint = args.output_dir / "checkpoints" / method / f"{dataset_name}_fold{fold_index}.pt"
    log_path = args.output_dir / "logs" / method / f"{dataset_name}_fold{fold_index}.jsonl"

    set_seed(args.seed + fold_index)
    model = build_model(method, in_dim, args.hidden_dim, args.layers, num_classes).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=False)
        model.freeze_encoder()
    if checkpoint.exists() and not args.force:
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
    else:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=DEFAULT_LR[method], weight_decay=1e-4)
        best_validation, stale, best_state = -1.0, 0, None
        epochs = args.epochs or DEFAULT_EPOCHS[method]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            progress = tqdm(range(1, epochs + 1), desc=f"{method}/{dataset_name}/折{fold_index}", unit="轮", leave=False)
            for epoch in progress:
                model.train()
                total_loss = total_graphs = 0
                for data in train_loader:
                    data = data.to(device)
                    if method == "dualprism":
                        data = model.spectral_view(data)
                    optimizer.zero_grad()
                    logits, _, regularizer = model(data)
                    regularizer_weight = 0.1
                    if method == "abstaingnn" and epoch <= max(1, int(epochs * 0.7)):
                        regularizer_weight = 0.0
                    loss = F.cross_entropy(logits, data.y) + regularizer_weight * regularizer
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"{method}/{dataset_name}/折{fold_index} 出现非有限损失"
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                    optimizer.step()
                    total_loss += float(loss.detach()) * data.num_graphs
                    total_graphs += data.num_graphs
                validation_accuracy, _ = evaluate(model, validation_loader, device)
                mean_loss = total_loss / max(total_graphs, 1)
                record = {"epoch": epoch, "loss": mean_loss, "validation_accuracy": validation_accuracy}
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                progress.set_postfix(loss=f"{mean_loss:.4f}", val=f"{validation_accuracy:.4f}")
                if validation_accuracy > best_validation:
                    best_validation, stale = validation_accuracy, 0
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
    return evaluate(model, test_loader, device)[1]


def run_supervised(args: argparse.Namespace, method: str, dataset_name: str, graphs: list[Data], in_dim: int, num_classes: int, device: torch.device, initial_state: dict[str, Tensor] | None = None) -> dict:
    labels = torch.tensor([int(graph.y) for graph in graphs])
    split_seed = args.seed if args.split_seed is None else args.split_seed
    test_folds = stratified_folds(labels, args.folds, split_seed)
    fold_metrics = {key: [] for key in METRIC_KEYS}
    all_indices = torch.arange(len(graphs))
    for fold_index, test_indices in enumerate(test_folds, start=1):
        train_pool = all_indices[~torch.isin(all_indices, test_indices)]
        train_indices, validation_indices = inner_split(labels, train_pool, split_seed + fold_index)
        current = train_supervised_fold(
            args, method, dataset_name, graphs, in_dim, num_classes, fold_index,
            train_indices, validation_indices, test_indices, device, initial_state,
        )
        for key in METRIC_KEYS:
            fold_metrics[key].append(current[key])
    return {
        "training": "supervised_end_to_end_per_fold" if initial_state is None else "self_supervised_pretrain_then_prompt_tuning_per_fold",
        **{f"{key if key != 'macro_f1' else 'macro_f1'}_percent": summarize(values) for key, values in fold_metrics.items()},
    }


@torch.no_grad()
def extract_embeddings(model: torch.nn.Module, graphs: list[Data], batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    model.eval()
    embeddings, labels = [], []
    for data in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        data = data.to(device)
        embeddings.append(model.encode(data).cpu())
        labels.append(data.y.cpu())
    return torch.cat(embeddings), torch.cat(labels)


def run_cellclat(args: argparse.Namespace, dataset_name: str, graphs: list[Data], in_dim: int, num_classes: int, device: torch.device) -> dict:
    method = "cellclat"
    batch_size = method_batch_size(method, dataset_name, args.batch_size)
    checkpoint = args.output_dir / "checkpoints" / method / f"{dataset_name}.pt"
    log_path = args.output_dir / "logs" / method / f"{dataset_name}.jsonl"
    set_seed(args.seed)
    model = build_model(method, in_dim, args.hidden_dim, args.layers, num_classes).to(device)
    if checkpoint.exists() and not args.force:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR[method], weight_decay=1e-5)
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        epochs = args.epochs or DEFAULT_EPOCHS[method]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            for epoch in tqdm(range(1, epochs + 1), desc=f"{method}/{dataset_name}", unit="轮"):
                model.train()
                losses = []
                for data in loader:
                    if data.num_graphs < 2:
                        continue
                    data = data.to(device)
                    optimizer.zero_grad()
                    loss = model.contrastive_loss(data)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"{method}/{dataset_name} 出现非有限损失")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                mean_loss = float(np.mean(losses)) if losses else 0.0
                log_file.write(json.dumps({"epoch": epoch, "loss": mean_loss}) + "\n")
                log_file.flush()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, checkpoint)
    embeddings, labels = extract_embeddings(model, graphs, batch_size, device)
    metrics = linear_probe_five_fold_metrics(
        embeddings, labels, device=device, seed=args.seed, split_seed=args.split_seed,
        folds=args.folds, epochs=args.probe_epochs,
    )
    return {"training": "self_supervised_cellular_pretrain_then_linear_probe", **{f"{key}_percent": summarize(values) for key, values in metrics.items()}}


def pretrain_edgeprompt(args: argparse.Namespace, dataset_name: str, graphs: list[Data], in_dim: int, num_classes: int, device: torch.device) -> dict[str, Tensor]:
    method = "edgeprompt"
    checkpoint = args.output_dir / "checkpoints" / method / f"{dataset_name}_pretrain.pt"
    model = build_model(method, in_dim, args.hidden_dim, args.layers, num_classes).to(device)
    if checkpoint.exists() and not args.force:
        return torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    batch_size = method_batch_size(method, dataset_name, args.batch_size)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(list(model.convs.parameters()) + list(model.projector.parameters()), lr=1e-3)
    epochs = args.epochs or DEFAULT_EPOCHS[method]
    for _ in tqdm(range(epochs), desc=f"edgeprompt/{dataset_name}/GraphCL预训练", unit="轮"):
        model.train()
        for data in loader:
            if data.num_graphs < 2:
                continue
            data = data.to(device)
            first = augment_graph(data, "edge_drop", 0.2)
            second = augment_graph(data, "attr_mask", 0.2)
            optimizer.zero_grad()
            loss = model.pretrain_loss(first, second)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method}/{dataset_name} 预训练出现非有限损失")
            loss.backward()
            optimizer.step()
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state}, checkpoint)
    return state


def merge_results(output_dir: Path) -> Path:
    experiments = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output_dir / "jobs").glob("*.json"))]
    result = {
        "metadata": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "methods": list(METHODS),
            "datasets": list(DATASETS),
            "folds": 5,
            "metrics": ["accuracy_percent", "nmi_percent", "ari_percent", "macro_f1_percent"],
            "implementation": "project_pyg_adapter_from_papers_and_official_sources",
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
    for method, dataset_name in tqdm(combinations, desc="十论文实验总进度", unit="组"):
        job_path = args.output_dir / "jobs" / f"{method}_{dataset_name}.json"
        if job_path.exists() and not args.force:
            print(f"复用已有结果：{method}/{dataset_name}")
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
            if method == "genvsexp":
                current_graphs = attach_cycle_basis_features(current_graphs, current_name, args.output_dir)
            if method in SUPERVISED_METHODS:
                current = run_supervised(args, method, current_name, current_graphs, in_dim, num_classes, device)
            elif method in CONTRASTIVE_METHODS:
                current = run_cellclat(args, current_name, current_graphs, in_dim, num_classes, device)
            elif method in PROMPT_METHODS:
                state = pretrain_edgeprompt(args, current_name, current_graphs, in_dim, num_classes, device)
                current = run_supervised(args, method, current_name, current_graphs, in_dim, num_classes, device, state)
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
        result.update({
            "method": method,
            "dataset": dataset_name,
            "task": "graph_classification",
            "fold_protocol": "stratified_5_fold_with_inner_validation_or_linear_probe",
            "seed": args.seed,
            "split_seed": args.seed if args.split_seed is None else args.split_seed,
            "source_commit": SOURCE_COMMITS[method],
            "implementation": "project_pyg_adapter",
            "dataset_protocol": dataset_protocol(dataset_name),
        })
        save_json(job_path, result)
        print(f"{method}/{dataset_name}: {result['accuracy_percent']['mean']:.2f} ± {result['accuracy_percent']['std']:.2f}")
    if not args.no_merge:
        print(f"已汇总：{merge_results(args.output_dir)}")


if __name__ == "__main__":
    main()
