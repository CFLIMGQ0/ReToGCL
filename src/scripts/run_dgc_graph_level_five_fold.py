"""运行六种深度图聚类方法的图级无监督预训练与五折线性评测。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.evaluation import linear_probe_five_fold_metrics, summarize
from src.models.dgc_graph_level_models import METHODS, METHOD_SPECS, BaseDGCAdapter, build_model
from src.scripts.graph_classification_datasets import (
    GRAPH_CLASSIFICATION_DATASETS,
    MULTITASK_DATASETS,
    aggregate_multitask_results,
    dataset_protocol,
    load_graphs,
)


DATASETS = GRAPH_CLASSIFICATION_DATASETS
PROTOCOL = "dgc_graph_level_adaptation_v1_unsupervised_pretrain_then_stratified_5_fold_linear_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dgc_graph_level_6x18"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{int(np.argmax(free))}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def effective_batch_size(dataset: str, requested: int) -> int:
    if dataset in {"ABIDE", "ADHD200"}:
        return min(requested, 16)
    if dataset == "COLLAB":
        return min(requested, 32)
    return requested


@torch.no_grad()
def extract_embeddings(
    model: BaseDGCAdapter,
    graphs: list,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings, labels = [], []
    for batch in tqdm(
        DataLoader(graphs, batch_size=batch_size, shuffle=False),
        desc="提取图表示", unit="批", leave=False,
    ):
        batch = batch.to(device)
        embedding = model.encode(batch)
        if not torch.isfinite(embedding).all():
            raise RuntimeError("提取的图表示包含非有限值")
        embeddings.append(embedding.cpu())
        labels.append(batch.y.cpu())
    return torch.cat(embeddings), torch.cat(labels).long()


def evaluate_embeddings(
    embeddings: torch.Tensor,
    graphs: list,
    labels: torch.Tensor,
    dataset: str,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    if dataset not in MULTITASK_DATASETS:
        metrics = linear_probe_five_fold_metrics(
            embeddings, labels, device=device, seed=args.seed,
            folds=args.folds, epochs=args.probe_epochs,
        )
        return {
            "accuracy_percent": summarize(metrics["accuracy"]),
            "nmi_percent": summarize(metrics["nmi"]),
            "ari_percent": summarize(metrics["ari"]),
            "macro_f1_percent": summarize(metrics["macro_f1"]),
        }

    task_labels = torch.cat([graph.task_y for graph in graphs], dim=0)
    task_results = []
    for task_index in tqdm(
        range(task_labels.size(1)), desc=f"{dataset} 多任务五折评测", unit="任务",
    ):
        mask = torch.isfinite(task_labels[:, task_index])
        metrics = linear_probe_five_fold_metrics(
            embeddings[mask], task_labels[mask, task_index].long(),
            device=device, seed=args.seed, folds=args.folds, epochs=args.probe_epochs,
        )
        task_results.append({
            "task_index": task_index,
            "sample_count": int(mask.sum()),
            "accuracy_percent": summarize(metrics["accuracy"]),
            "nmi_percent": summarize(metrics["nmi"]),
            "ari_percent": summarize(metrics["ari"]),
            "macro_f1_percent": summarize(metrics["macro_f1"]),
        })
    return aggregate_multitask_results(dataset, task_results)


def train_one(
    args: argparse.Namespace,
    method: str,
    dataset: str,
    graphs: list,
    in_dim: int,
    clusters: int,
    device: torch.device,
) -> BaseDGCAdapter:
    checkpoint = args.output_dir / "checkpoints" / method / f"{dataset}.pt"
    expected = {
        "protocol": PROTOCOL,
        "method": method,
        "dataset": dataset,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "epochs": args.epochs,
    }
    set_seed(args.seed)
    model = build_model(method, in_dim, args.hidden_dim, args.layers, clusters).to(device)
    if checkpoint.exists() and not args.force:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        differences = {key: (state.get(key), value) for key, value in expected.items() if state.get(key) != value}
        if differences:
            raise RuntimeError(f"拒绝复用不兼容 checkpoint：{differences}")
        model.load_state_dict(state["model"])
        return model

    batch_size = effective_batch_size(dataset, args.batch_size)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    log_path = args.output_dir / "logs" / method / f"{dataset}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        progress = tqdm(range(1, args.epochs + 1), desc=f"{method}/{dataset}", unit="轮")
        for epoch in progress:
            model.train()
            total_loss = 0.0
            graph_count = 0
            diagnostic_sums: dict[str, float] = {}
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, diagnostics = model.ssl_loss(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{method}/{dataset}/epoch={epoch} 出现非有限损失")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(gradient):
                    raise RuntimeError(f"{method}/{dataset}/epoch={epoch} 出现非有限梯度")
                optimizer.step()
                count = int(batch.num_graphs)
                total_loss += float(loss.detach()) * count
                graph_count += count
                for key, value in diagnostics.items():
                    diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + value * count
            record = {
                "epoch": epoch,
                "loss": total_loss / max(graph_count, 1),
                "gradient_norm": float(gradient),
                "diagnostics": {
                    key: value / max(graph_count, 1) for key, value in diagnostic_sums.items()
                },
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
            progress.set_postfix(loss=f"{record['loss']:.4f}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**expected, "model": model.state_dict()}, checkpoint)
    return model


def run_job(args: argparse.Namespace, method: str, dataset: str, device: torch.device) -> Path:
    result_path = args.output_dir / "jobs" / f"{method}__{dataset}.json"
    if result_path.exists() and not args.force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("protocol") == PROTOCOL:
            print(f"已存在，跳过：{method}/{dataset}")
            return result_path
        raise RuntimeError(f"拒绝复用不兼容结果：{result_path}")

    graphs, in_dim, clusters = load_graphs(args.data_root, dataset)
    model = train_one(args, method, dataset, graphs, in_dim, clusters, device)
    batch_size = effective_batch_size(dataset, args.batch_size)
    embeddings, labels = extract_embeddings(model, graphs, batch_size, device)
    metrics = evaluate_embeddings(embeddings, graphs, labels, dataset, device, args)
    spec = METHOD_SPECS[method]
    result = {
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "method": method,
        "paper": spec["title"],
        "venue": spec["venue"],
        "year": spec["year"],
        "source": spec["source"],
        "source_commit": spec["source_commit"],
        "implementation": "graph_level_adaptation_of_node_clustering_method",
        "dataset": dataset,
        "dataset_protocol": dataset_protocol(dataset),
        "sample_count": len(graphs),
        "in_dim": in_dim,
        "cluster_count": clusters,
        **metrics,
        "task": "graph_level_embedding_evaluation",
        "labels_used_in_encoder_training": False,
        "encoder_training": {
            "labels_used_in_encoder_training": False,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "batch_size": batch_size,
            "seed": args.seed,
        },
        "evaluation": f"stratified_{args.folds}_fold_linear_probe",
    }
    save_json(result_path, result)
    print(
        f"{method}/{dataset}: Accuracy {result['accuracy_percent']['mean']:.2f} ± "
        f"{result['accuracy_percent']['std']:.2f}"
    )
    return result_path


def merge_results(output_dir: Path) -> Path:
    jobs = []
    for path in sorted((output_dir / "jobs").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("protocol") == PROTOCOL and value.get("status") == "completed":
            jobs.append(value)
    merged = {
        "metadata": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL,
            "methods": list(METHODS),
            "datasets": list(DATASETS),
            "expected_jobs": len(METHODS) * len(DATASETS),
            "completed_jobs": len(jobs),
            "metrics": ["accuracy_percent", "nmi_percent", "ari_percent", "macro_f1_percent"],
            "adaptation_warning": "原论文是节点聚类；本结果是独立图样本上的图级无监督适配，不是原论文协议。",
            "papers": METHOD_SPECS,
        },
        "experiments": sorted(jobs, key=lambda item: (item["method"], item["dataset"])),
    }
    path = output_dir / "five_fold_results.json"
    save_json(path, merged)
    return path


def smoke_test(args: argparse.Namespace, device: torch.device) -> None:
    dataset = args.datasets[0]
    graphs, in_dim, clusters = load_graphs(args.data_root, dataset)
    batch = next(iter(DataLoader(graphs[: min(12, len(graphs))], batch_size=min(12, len(graphs))))).to(device)
    for method in tqdm(args.methods, desc="六模型冒烟测试", unit="模型"):
        set_seed(args.seed)
        model = build_model(method, in_dim, args.hidden_dim, args.layers, clusters).to(device)
        model.train()
        loss, diagnostics = model.ssl_loss(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{method} 冒烟测试损失非有限")
        loss.backward()
        embedding = model.encode(batch)
        if embedding.shape != (batch.num_graphs, args.hidden_dim) or not torch.isfinite(embedding).all():
            raise RuntimeError(f"{method} 冒烟测试表示异常：{tuple(embedding.shape)}")
        print(f"通过 {method}: loss={float(loss.detach()):.5f}, embedding={tuple(embedding.shape)}, {diagnostics}")


def main() -> None:
    args = parse_args()
    args.data_root = resolve(args.data_root)
    args.output_dir = resolve(args.output_dir)
    if args.merge_only:
        print(f"已汇总：{merge_results(args.output_dir)}")
        return
    device = choose_device(args.device)
    print(f"使用设备：{device}")
    if args.smoke_test:
        smoke_test(args, device)
        return
    combinations = [(method, dataset) for method in args.methods for dataset in args.datasets]
    for method, dataset in tqdm(combinations, desc="六模型×十八数据集", unit="组"):
        run_job(args, method, dataset, device)
    if not args.no_merge:
        print(f"已汇总：{merge_results(args.output_dir)}")


if __name__ == "__main__":
    main()
