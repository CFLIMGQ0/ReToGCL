"""运行多空间门控相似度模块的分层五折消融实验。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.evaluation import METRIC_KEYS, classification_metrics, stratified_folds, summarize
from src.baselines.gcl_baselines import ensure_features, set_seed
from src.models.multispace_similarity import MultiSpaceGateClassifier, VARIANTS
from src.scripts.diagnose_multispace_similarity import topology_descriptor


DATASETS = ("NCI1", "PROTEINS", "COLLAB", "MUTAG", "COLORS-3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/baselines/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multispace"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prototypes", type=int, default=16)
    parser.add_argument("--similarity-weight", type=float, default=0.5)
    parser.add_argument("--relation-weight", type=float, default=0.2)
    parser.add_argument("--gate-weight", type=float, default=0.01)
    parser.add_argument("--hard-fraction", type=float, default=0.25)
    parser.add_argument("--negative-margin", type=float, default=0.5)
    parser.add_argument("--retrieval-k", type=int, default=10)
    parser.add_argument("--retrieval-weight", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
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


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_graphs(root: Path, name: str) -> tuple[list[Data], int, int, Tensor]:
    dataset = TUDataset(root=str(root), name=name, cleaned=False, use_node_attr=name == "COLORS-3")
    graphs, topology = [], []
    for index in tqdm(range(len(dataset)), desc=f"{name} 准备多空间特征", unit="图"):
        graph = ensure_features(dataset[index].clone())
        graph.y = graph.y.long().view(1)
        graphs.append(graph)
        topology.append(topology_descriptor(graph))
    return graphs, max(1, dataset.num_node_features), dataset.num_classes, torch.stack(topology)


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


def fold_graphs(
    graphs: list[Data], topology: Tensor, train_indices: Tensor, indices: Tensor
) -> list[Data]:
    mean = topology[train_indices].mean(dim=0, keepdim=True)
    std = topology[train_indices].std(dim=0, keepdim=True).clamp_min(1e-6)
    normalized = (topology - mean) / std
    selected = []
    for index in indices.tolist():
        graph = graphs[index].clone()
        graph.topology = normalized[index].view(1, -1)
        selected.append(graph)
    return selected


@torch.no_grad()
def collect_outputs(
    model: MultiSpaceGateClassifier, loader: DataLoader, device: torch.device
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    model.eval()
    logits, labels = [], []
    representations: dict[str, list[Tensor]] = {
        "graph": [], "semantic": [], "topology": [], "distribution": [],
    }
    for data in loader:
        data = data.to(device)
        current_logits, current_representations = model(data)
        logits.append(current_logits.cpu())
        labels.append(data.y.cpu())
        for key in representations:
            representations[key].append(current_representations[key].cpu())
    return (
        torch.cat(logits),
        {key: torch.cat(parts) for key, parts in representations.items()},
        torch.cat(labels),
    )


@torch.no_grad()
def evaluate(
    model: MultiSpaceGateClassifier,
    loader: DataLoader,
    device: torch.device,
    bank_loader: DataLoader | None = None,
    variant: str = "baseline",
    retrieval_k: int = 10,
    retrieval_weight: float = 0.5,
) -> dict[str, float]:
    query_logits, query_representations, labels = collect_outputs(model, loader, device)
    probabilities = query_logits.softmax(dim=-1)
    if bank_loader is not None and variant != "baseline":
        bank_logits, bank_representations, bank_labels = collect_outputs(model, bank_loader, device)
        corrected_parts = []
        for start in range(0, labels.numel(), 256):
            stop = min(start + 256, labels.numel())
            query_chunk = {
                key: value[start:stop].to(device)
                for key, value in query_representations.items()
            }
            bank_chunk = {key: value.to(device) for key, value in bank_representations.items()}
            similarity, _ = model.cross_fused_similarity(
                query_logits[start:stop].to(device), query_chunk,
                bank_logits.to(device), bank_chunk, variant,
            )
            k = min(retrieval_k, bank_labels.numel())
            values, indices = similarity.topk(k, dim=1)
            neighbor_labels = bank_labels.to(device)[indices]
            votes = F.one_hot(
                neighbor_labels, num_classes=query_logits.size(1)
            ).float()
            votes = (votes * values.unsqueeze(-1)).sum(dim=1)
            votes = votes / votes.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            base = probabilities[start:stop].to(device)
            uncertainty = model._uncertainty(query_logits[start:stop].to(device))
            mixture = (retrieval_weight * uncertainty).clamp(0, 1).unsqueeze(-1)
            corrected_parts.append(((1 - mixture) * base + mixture * votes).cpu())
        probabilities = torch.cat(corrected_parts)
    return classification_metrics(labels, probabilities.argmax(dim=-1))


def build_model(
    args: argparse.Namespace,
    in_dim: int,
    num_classes: int,
    topology_dim: int,
    graphcl_state: dict[str, Tensor],
    device: torch.device,
) -> MultiSpaceGateClassifier:
    model = MultiSpaceGateClassifier(
        in_dim, args.hidden_dim, args.layers, num_classes, topology_dim,
        prototypes=args.prototypes,
    ).to(device)
    model.load_graphcl_encoder(graphcl_state)
    return model


def train_fold(
    args: argparse.Namespace,
    variant: str,
    dataset_name: str,
    graphs: list[Data],
    topology: Tensor,
    in_dim: int,
    num_classes: int,
    train_indices: Tensor,
    validation_indices: Tensor,
    test_indices: Tensor,
    fold: int,
    graphcl_state: dict[str, Tensor],
    device: torch.device,
) -> dict[str, float]:
    checkpoint = args.output_dir / "checkpoints" / variant / f"{dataset_name}_fold{fold}.pt"
    log_path = args.output_dir / "logs" / variant / f"{dataset_name}_fold{fold}.jsonl"
    batch_size = min(args.batch_size, 32) if dataset_name == "COLLAB" else args.batch_size
    train_loader = DataLoader(
        fold_graphs(graphs, topology, train_indices, train_indices),
        batch_size=batch_size, shuffle=True,
    )
    validation_loader = DataLoader(
        fold_graphs(graphs, topology, train_indices, validation_indices),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        fold_graphs(graphs, topology, train_indices, test_indices),
        batch_size=batch_size,
    )
    set_seed(args.seed + fold)
    model = build_model(
        args, in_dim, num_classes, topology.size(1), graphcl_state, device
    )
    if checkpoint.exists() and not args.force:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
        return evaluate(
            model, test_loader, device, train_loader, variant,
            args.retrieval_k, args.retrieval_weight,
        )

    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": args.encoder_learning_rate},
        {"params": other_parameters, "lr": args.learning_rate},
    ], weight_decay=args.weight_decay)
    best_validation, stale, best_state = -1.0, 0, None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        progress = tqdm(
            range(1, args.epochs + 1),
            desc=f"MSGS/{variant}/{dataset_name}/折{fold}", unit="轮", leave=False,
        )
        for epoch in progress:
            model.train()
            totals = {"loss": 0.0, "classification": 0.0, "similarity": 0.0, "relation": 0.0, "gate": 0.0}
            graph_count = 0
            active_variant = variant
            if epoch <= args.warmup_epochs and variant in {"gate", "boundary_gate", "full"}:
                active_variant = "uniform"
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                logits, representations = model(data)
                loss, parts = model.training_loss(
                    logits, representations, data.y, active_variant,
                    similarity_weight=args.similarity_weight,
                    relation_weight=args.relation_weight,
                    gate_weight=args.gate_weight,
                    hard_fraction=args.hard_fraction,
                    negative_margin=args.negative_margin,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                current_count = data.num_graphs
                graph_count += current_count
                totals["loss"] += float(loss.detach()) * current_count
                for key, value in parts.items():
                    totals[key] += value * current_count
            validation = evaluate(
                model, validation_loader, device, train_loader, variant,
                args.retrieval_k, args.retrieval_weight,
            )
            record = {
                "epoch": epoch,
                **{key: value / max(graph_count, 1) for key, value in totals.items()},
                "validation_accuracy": validation["accuracy"],
                "active_variant": active_variant,
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
            progress.set_postfix(loss=f"{record['loss']:.4f}", val=f"{validation['accuracy']:.4f}")
            if validation["accuracy"] > best_validation:
                best_validation, stale = validation["accuracy"], 0
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("训练未产生可用参数")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "best_validation_accuracy": best_validation}, checkpoint)
    model.load_state_dict(best_state)
    return evaluate(
        model, test_loader, device, train_loader, variant,
        args.retrieval_k, args.retrieval_weight,
    )


def run_experiment(
    args: argparse.Namespace,
    variant: str,
    dataset_name: str,
    graphs: list[Data],
    topology: Tensor,
    in_dim: int,
    num_classes: int,
    graphcl_state: dict[str, Tensor],
    device: torch.device,
) -> dict:
    labels = torch.tensor([int(graph.y) for graph in graphs])
    test_folds = stratified_folds(labels, args.folds, args.seed)
    all_indices = torch.arange(len(graphs))
    fold_metrics = {key: [] for key in METRIC_KEYS}
    for fold, test_indices in enumerate(test_folds, start=1):
        train_pool = all_indices[~torch.isin(all_indices, test_indices)]
        train_indices, validation_indices = inner_split(labels, train_pool, args.seed + fold)
        metrics = train_fold(
            args, variant, dataset_name, graphs, topology, in_dim, num_classes,
            train_indices, validation_indices, test_indices, fold, graphcl_state, device,
        )
        for key in METRIC_KEYS:
            fold_metrics[key].append(metrics[key])
    return {
        "method": f"msgs_{variant}",
        "variant": variant,
        "dataset": dataset_name,
        "task": "graph_classification",
        "training": "GraphCL_initialization_then_supervised_multispace_finetuning_and_train_bank_retrieval",
        "fold_protocol": "stratified_5_fold_with_inner_validation",
        "seed": args.seed,
        **{f"{key}_percent": summarize(values) for key, values in fold_metrics.items()},
    }


def merge_results(output_dir: Path) -> Path:
    experiments = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "jobs").glob("*.json"))
    ]
    result = {
        "metadata": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "variants": list(VARIANTS),
            "datasets": list(DATASETS),
            "metrics": ["accuracy_percent", "nmi_percent", "ari_percent", "macro_f1_percent"],
            "implementation": "MSGS GraphCL-GIN prototype",
        },
        "experiments": sorted(experiments, key=lambda item: (item["variant"], item["dataset"])),
    }
    path = output_dir / "five_fold_results.json"
    save_json(path, result)
    return path


def main() -> None:
    args = parse_args()
    args.data_root = resolve_path(args.data_root)
    args.checkpoint_dir = resolve_path(args.checkpoint_dir)
    args.output_dir = resolve_path(args.output_dir)
    if args.merge_only:
        print(f"已汇总：{merge_results(args.output_dir)}")
        return
    device = choose_device(args.device)
    print(f"使用设备：{device}")
    for dataset_name in tqdm(args.datasets, desc="MSGS 数据集进度", unit="数据集"):
        graphs, in_dim, num_classes, topology = load_graphs(args.data_root, dataset_name)
        graphcl_checkpoint = args.checkpoint_dir / f"graphcl_{dataset_name}.pt"
        graphcl_state = torch.load(
            graphcl_checkpoint, map_location="cpu", weights_only=True
        )["model"]
        for variant in tqdm(args.variants, desc=f"{dataset_name} 消融进度", unit="配置", leave=False):
            job_path = args.output_dir / "jobs" / f"{variant}_{dataset_name}.json"
            if job_path.exists() and not args.force:
                continue
            result = run_experiment(
                args, variant, dataset_name, graphs, topology,
                in_dim, num_classes, graphcl_state, device,
            )
            save_json(job_path, result)
            print(
                f"{variant}/{dataset_name}: "
                f"{result['accuracy_percent']['mean']:.2f} ± {result['accuracy_percent']['std']:.2f}"
            )
    print(f"已汇总：{merge_results(args.output_dir)}")


if __name__ == "__main__":
    main()
