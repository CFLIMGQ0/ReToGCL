"""在项目数据集上预训练四个 GCL 基线并执行分层五折评测。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.evaluation import linear_probe_five_fold_metrics, summarize
from src.baselines.gcl_baselines import (
    AUGMENTATIONS,
    GCLModel,
    augment_graph,
    ensure_features,
    info_nce,
    perturb_model,
    project_simplex,
    rgcl_loss,
    set_seed,
)


METHODS = ("graphcl", "joaov2", "rgcl", "simgrace")
GRAPH_DATASETS = (
    "NCI1", "PROTEINS", "COLLAB", "MUTAG", "COLORS-3",
    "PTC_MR", "Mutagenicity",
)
NODE_DATASETS = ("Cora", "CiteSeer", "PubMed")
ALL_DATASETS = GRAPH_DATASETS + NODE_DATASETS
DEFAULT_EPOCHS = {"graphcl": 20, "joaov2": 40, "rgcl": 40, "simgrace": 20}
DEFAULT_LR = {"graphcl": 0.01, "joaov2": 0.001, "rgcl": 0.01, "simgrace": 0.01}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--datasets", nargs="+", choices=ALL_DATASETS, default=list(GRAPH_DATASETS)
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda、cuda:0 或 cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=None, help="覆盖论文默认预训练轮数")
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--augmentation-ratio", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=1.0, help="SimGRACE 参数扰动系数")
    parser.add_argument("--gamma", type=float, default=0.1, help="JOAOv2 概率正则系数")
    parser.add_argument("--force", action="store_true", help="忽略已有 checkpoint 并重新预训练")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free_memory = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{int(np.argmax(free_memory))}")


def load_dataset(root: Path, name: str) -> tuple[str, object, int]:
    if name in GRAPH_DATASETS:
        dataset = TUDataset(
            root=str(root), name=name, cleaned=False,
            use_node_attr=name == "COLORS-3",
        )
        dataset.transform = ensure_features
        return "graph", dataset, max(1, dataset.num_node_features)
    dataset = Planetoid(root=str(root), name=name, split="public")
    data = ensure_features(dataset[0])
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    return "node", data, data.num_node_features


def make_loader(task: str, dataset: object, batch_size: int, shuffle: bool) -> Iterable[Data]:
    if task == "graph":
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return [dataset]


def move_batch(data: Data, device: torch.device) -> Data:
    data = ensure_features(data)
    if getattr(data, "batch", None) is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    return data.to(device)


def train_epoch(
    model: GCLModel,
    loader: Iterable[Data],
    optimizer: torch.optim.Optimizer,
    method: str,
    device: torch.device,
    ratio: float,
    eta: float,
    augmentation_probabilities: np.ndarray,
) -> float:
    model.train()
    total_loss, total_examples = 0.0, 0
    for raw_data in loader:
        data = move_batch(raw_data, device)
        optimizer.zero_grad()

        if method == "graphcl":
            first_kind, second_kind = np.random.choice(AUGMENTATIONS[1:], size=2, replace=True)
            first, _ = model.encode(augment_graph(data, str(first_kind), ratio))
            second, _ = model.encode(augment_graph(data, str(second_kind), ratio))
            loss = info_nce(model.project(first), model.project(second))
        elif method == "joaov2":
            selected = int(np.random.choice(len(AUGMENTATIONS), p=augmentation_probabilities))
            anchor, _ = model.encode(data)
            augmented, _ = model.encode(augment_graph(data, AUGMENTATIONS[selected], ratio))
            loss = info_nce(model.project(anchor, 0), model.project(augmented, selected))
        elif method == "rgcl":
            anchor, rationale, complement = model.rationale_views(data)
            loss = rgcl_loss(anchor, rationale, complement)
        elif method == "simgrace":
            perturbed = perturb_model(model, eta).to(device)
            with torch.no_grad():
                target, _ = perturbed.encode(data)
                target = perturbed.project(target)
            online, _ = model.encode(data)
            loss = info_nce(model.project(online), target)
            del perturbed
        else:
            raise ValueError(method)

        loss.backward()
        optimizer.step()
        examples = data.num_graphs if model.task == "graph" else data.num_nodes
        total_loss += float(loss.detach()) * examples
        total_examples += examples
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def update_joao_probabilities(
    model: GCLModel,
    loader: Iterable[Data],
    device: torch.device,
    ratio: float,
    probabilities: np.ndarray,
    gamma: float,
) -> np.ndarray:
    model.eval()
    first_batch = move_batch(next(iter(loader)), device)
    anchor, _ = model.encode(first_batch)
    losses = []
    for index, kind in enumerate(AUGMENTATIONS):
        augmented, _ = model.encode(augment_graph(first_batch, kind, ratio))
        losses.append(float(info_nce(model.project(anchor, 0), model.project(augmented, index))))
    values = probabilities + np.asarray(losses) - gamma * (probabilities - 1 / len(AUGMENTATIONS))
    return project_simplex(values)


@torch.no_grad()
def extract_embeddings(
    model: GCLModel,
    task: str,
    dataset: object,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings, labels = [], []
    for raw_data in make_loader(task, dataset, batch_size, shuffle=False):
        data = move_batch(raw_data, device)
        representation, _ = model.encode(data)
        embeddings.append(representation.cpu())
        labels.append(data.y.view(-1).cpu())
    return torch.cat(embeddings), torch.cat(labels)


def save_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_experiment(args: argparse.Namespace, method: str, dataset_name: str, device: torch.device) -> dict:
    set_seed(args.seed)
    task, dataset, in_dim = load_dataset(args.data_root, dataset_name)
    batch_size = args.batch_size
    if dataset_name == "COLLAB":
        batch_size = min(batch_size, 64)

    model = GCLModel(in_dim, args.hidden_dim, args.layers, task, method).to(device)
    epochs = args.epochs or DEFAULT_EPOCHS[method]
    checkpoint = args.output_dir / "checkpoints" / f"{method}_{dataset_name}.pt"
    log_path = args.output_dir / "logs" / f"{method}_{dataset_name}.jsonl"
    probabilities = np.ones(len(AUGMENTATIONS), dtype=np.float64) / len(AUGMENTATIONS)

    if checkpoint.exists() and not args.force:
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        probabilities = np.asarray(saved.get("augmentation_probabilities", probabilities))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR[method])
        loader = make_loader(task, dataset, batch_size, shuffle=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            progress = tqdm(range(1, epochs + 1), desc=f"{method}/{dataset_name}", unit="轮")
            for epoch in progress:
                loss = train_epoch(
                    model, loader, optimizer, method, device,
                    args.augmentation_ratio, args.eta, probabilities,
                )
                if method == "joaov2":
                    probabilities = update_joao_probabilities(
                        model, make_loader(task, dataset, batch_size, shuffle=True),
                        device, args.augmentation_ratio, probabilities, args.gamma,
                    )
                record = {
                    "epoch": epoch,
                    "loss": loss,
                    "augmentation_probabilities": probabilities.tolist() if method == "joaov2" else None,
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                progress.set_postfix(loss=f"{loss:.4f}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "method": method,
                "dataset": dataset_name,
                "epochs": epochs,
                "augmentation_probabilities": probabilities.tolist(),
            },
            checkpoint,
        )

    embeddings, labels = extract_embeddings(model, task, dataset, batch_size, device)
    fold_metrics = linear_probe_five_fold_metrics(
        embeddings, labels, device=device, seed=args.seed, epochs=args.probe_epochs,
    )
    return {
        "method": method,
        "dataset": dataset_name,
        "task": f"{task}_classification",
        "pretrain_epochs": epochs,
        "seed": args.seed,
        "fold_protocol": "stratified_5_fold_linear_probe",
        "accuracy_percent": summarize(fold_metrics["accuracy"]),
        "nmi_percent": summarize(fold_metrics["nmi"]),
        "ari_percent": summarize(fold_metrics["ari"]),
        "macro_f1_percent": summarize(fold_metrics["macro_f1"]),
        "augmentation_probabilities": probabilities.tolist() if method == "joaov2" else None,
        "checkpoint": str(checkpoint),
    }


def main() -> None:
    args = parse_args()
    args.data_root = (PROJECT_ROOT / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    device = choose_device(args.device)
    print(f"使用设备：{device}")

    result_path = args.output_dir / "five_fold_results.json"
    if result_path.exists():
        results = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        results = {"metadata": {}, "experiments": []}
    results["metadata"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "methods": list(METHODS),
        "datasets": list(ALL_DATASETS),
        "metrics": ["accuracy_percent", "nmi_percent", "ari_percent", "macro_f1_percent"],
    }

    existing = {(item["method"], item["dataset"]): item for item in results["experiments"]}
    combinations = [(method, dataset) for method in args.methods for dataset in args.datasets]
    for method, dataset in tqdm(combinations, desc="基线实验总进度", unit="组"):
        result = run_experiment(args, method, dataset, device)
        existing[(method, dataset)] = result
        results["experiments"] = sorted(existing.values(), key=lambda item: (item["method"], item["dataset"]))
        save_json(result_path, results)
        print(
            f"{method}/{dataset}: "
            f"{result['accuracy_percent']['mean']:.2f} ± {result['accuracy_percent']['std']:.2f}"
        )


if __name__ == "__main__":
    main()
