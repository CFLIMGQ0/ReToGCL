#!/usr/bin/env python3
"""运行随机 GIN、监督 GIN 与四种经典图对比学习基线。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.evaluation import METRIC_KEYS, classification_metrics, linear_probe_five_fold_metrics, stratified_folds, summarize
from src.baselines.gcl_baselines import AUGMENTATIONS, GCLModel, GraphEncoder, set_seed
from src.scripts.graph_classification_datasets import MULTITASK_DATASETS, aggregate_multitask_results, dataset_protocol, load_graphs, multitask_graphs
from src.scripts.run_gcl_baselines import train_epoch, update_joao_probabilities
from src.scripts.run_sota_five_fold import choose_device, save_json


METHODS = ("random_gin_linear", "supervised_gin", "graphcl", "joaov2", "rgcl", "simgrace")
DATASETS = ("ADHD200", "BACE", "BBBP", "Tox21", "AIDS", "BZR")
DEFAULT_EPOCHS = {"graphcl": 20, "joaov2": 40, "rgcl": 40, "simgrace": 20}
DEFAULT_LR = {"graphcl": 0.01, "joaov2": 0.001, "rgcl": 0.01, "simgrace": 0.01}


class SupervisedGIN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int, classes: int):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim, layers)
        self.classifier = nn.Linear(self.encoder.output_dim, classes)

    def forward(self, data) -> torch.Tensor:
        graph, _ = self.encoder(data)
        return self.classifier(graph)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classic_direct_baselines_9x6_20260904/classic"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def inner_split(labels: torch.Tensor, indices: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train, validation = [], []
    for label in labels[indices].unique(sorted=True):
        current = indices[labels[indices] == label]
        current = current[torch.randperm(current.numel(), generator=generator)]
        count = max(1, round(current.numel() * 0.1))
        validation.append(current[:count])
        train.append(current[count:])
    return torch.cat(train), torch.cat(validation)


@torch.no_grad()
def supervised_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels, predictions = [], []
    for batch in loader:
        batch = batch.to(device)
        labels.append(batch.y.cpu())
        predictions.append(model(batch).argmax(dim=-1).cpu())
    return classification_metrics(torch.cat(labels), torch.cat(predictions))


def supervised_five_fold(args: argparse.Namespace, graphs: list, in_dim: int, classes: int, device: torch.device, dataset_key: str) -> dict:
    labels = torch.tensor([int(graph.y.item()) for graph in graphs])
    test_folds = stratified_folds(labels, 5, args.split_seed)
    all_indices = torch.arange(len(graphs))
    fold_metrics = {metric: [] for metric in METRIC_KEYS}
    epochs = args.epochs or 40
    batch_size = min(args.batch_size, 64 if args.dataset == "Tox21" else args.batch_size)
    for fold, test_indices in enumerate(test_folds, start=1):
        train_pool = all_indices[~torch.isin(all_indices, test_indices)]
        train_indices, validation_indices = inner_split(labels, train_pool, args.split_seed + fold)
        train_loader = DataLoader([graphs[int(i)] for i in train_indices], batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader([graphs[int(i)] for i in validation_indices], batch_size=batch_size)
        test_loader = DataLoader([graphs[int(i)] for i in test_indices], batch_size=batch_size)
        set_seed(args.seed + fold)
        model = SupervisedGIN(in_dim, args.hidden_dim, args.layers, classes).to(device)
        checkpoint = args.output_dir / "checkpoints/supervised_gin" / f"{dataset_key}_fold{fold}.pt"
        if checkpoint.exists() and not args.force:
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            best_accuracy, stale, best_state = -1.0, 0, None
            for _ in tqdm(range(epochs), desc=f"supervised_gin/{dataset_key}/折{fold}", unit="轮", leave=False):
                model.train()
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(batch), batch.y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                accuracy = supervised_metrics(model, validation_loader, device)["accuracy"]
                if accuracy > best_accuracy:
                    best_accuracy, stale = accuracy, 0
                    best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
                else:
                    stale += 1
                if stale >= args.patience:
                    break
            if best_state is None:
                raise RuntimeError("监督式 GIN 未产生可用参数")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": best_state, "seed": args.seed, "split_seed": args.split_seed}, checkpoint)
            model.load_state_dict(best_state)
        current = supervised_metrics(model, test_loader, device)
        for metric in METRIC_KEYS:
            fold_metrics[metric].append(current[metric])
    return {f"{metric}_percent": summarize(values) for metric, values in fold_metrics.items()}


@torch.no_grad()
def embeddings(model: GCLModel, graphs: list, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    values, labels = [], []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        representation, _ = model.encode(batch)
        values.append(representation.cpu())
        labels.append(batch.y.cpu())
    return torch.cat(values), torch.cat(labels)


def ssl_embeddings(args: argparse.Namespace, graphs: list, in_dim: int, device: torch.device) -> torch.Tensor:
    model_method = "graphcl" if args.method == "random_gin_linear" else args.method
    set_seed(args.seed)
    model = GCLModel(in_dim, args.hidden_dim, args.layers, "graph", model_method).to(device)
    checkpoint = args.output_dir / "checkpoints" / args.method / f"{args.dataset}.pt"
    if args.method != "random_gin_linear" and (not checkpoint.exists() or args.force):
        optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR[args.method])
        loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)
        probabilities = np.ones(len(AUGMENTATIONS), dtype=np.float64) / len(AUGMENTATIONS)
        epochs = args.epochs or DEFAULT_EPOCHS[args.method]
        for _ in tqdm(range(epochs), desc=f"{args.method}/{args.dataset}", unit="轮"):
            train_epoch(model, loader, optimizer, args.method, device, 0.2, 1.0, probabilities)
            if args.method == "joaov2":
                probabilities = update_joao_probabilities(model, DataLoader(graphs, batch_size=args.batch_size, shuffle=True), device, 0.2, probabilities, 0.1)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "seed": args.seed, "split_seed": args.split_seed}, checkpoint)
    elif args.method != "random_gin_linear":
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    return embeddings(model, graphs, args.batch_size, device)[0]


def probe_metrics(args: argparse.Namespace, graphs: list, representation: torch.Tensor, device: torch.device) -> dict:
    if args.dataset not in MULTITASK_DATASETS:
        labels = torch.tensor([int(graph.y.item()) for graph in graphs])
        metrics = linear_probe_five_fold_metrics(
            representation, labels, device=device, seed=args.seed, split_seed=args.split_seed,
            folds=5, epochs=args.probe_epochs,
        )
        return {f"{metric}_percent": summarize(values) for metric, values in metrics.items()}
    task_y = torch.cat([graph.task_y for graph in graphs], dim=0)
    task_results = []
    for task_index in tqdm(range(task_y.size(1)), desc=f"{args.dataset} 多任务线性评测", unit="任务"):
        mask = torch.isfinite(task_y[:, task_index])
        metrics = linear_probe_five_fold_metrics(
            representation[mask], task_y[mask, task_index].long(), device=device,
            seed=args.seed, split_seed=args.split_seed, folds=5, epochs=args.probe_epochs,
        )
        task_results.append({
            "task_index": task_index, "sample_count": int(mask.sum()),
            **{f"{metric}_percent": summarize(values) for metric, values in metrics.items()},
        })
    return aggregate_multitask_results(args.dataset, task_results)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root if args.data_root.is_absolute() else (PROJECT_ROOT / args.data_root).resolve()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir).resolve()
    result_path = args.output_dir / "jobs" / f"{args.method}_{args.dataset}.json"
    if result_path.exists() and not args.force:
        value = json.loads(result_path.read_text(encoding="utf-8"))
        if value.get("status") == "completed":
            print(f"已存在，跳过：{result_path}")
            return
    device = choose_device(args.device)
    graphs, in_dim, classes = load_graphs(args.data_root, args.dataset)
    if args.method == "supervised_gin":
        if args.dataset in MULTITASK_DATASETS:
            task_results = []
            for task_index in tqdm(range(graphs[0].task_y.numel()), desc=f"{args.dataset} 多任务监督评测", unit="任务"):
                current = multitask_graphs(graphs, task_index)
                metrics = supervised_five_fold(args, current, in_dim, 2, device, f"{args.dataset}__task{task_index:02d}")
                task_results.append({"task_index": task_index, "sample_count": len(current), **metrics})
            metric_result = aggregate_multitask_results(args.dataset, task_results)
        else:
            metric_result = supervised_five_fold(args, graphs, in_dim, classes, device, args.dataset)
        training = "supervised_end_to_end_stratified_five_fold"
    else:
        representation = ssl_embeddings(args, graphs, in_dim, device)
        metric_result = probe_metrics(args, graphs, representation, device)
        training = "random_frozen_GIN_then_linear_probe" if args.method == "random_gin_linear" else "self_supervised_pretrain_then_linear_probe"
    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(), "status": "completed",
        "method": args.method, "dataset": args.dataset, "training": training,
        "seed": args.seed, "split_seed": args.split_seed, "folds": 5,
        "dataset_protocol": dataset_protocol(args.dataset), **metric_result,
    }
    save_json(result_path, result)
    print(f"完成：{result_path}")


if __name__ == "__main__":
    main()
