"""运行一套 ReToGCL 轻量 Pipeline 策略并进行五折线性评测。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.evaluation import linear_probe_five_fold_metrics, summarize
from src.baselines.gcl_baselines import set_seed
from src.models.retogcl_pipeline_strategies import (
    PIPELINE_STRATEGIES,
    build_retogcl_pipeline_strategy,
)
from src.scripts.graph_classification_datasets import (
    GRAPH_CLASSIFICATION_DATASETS,
    MULTITASK_DATASETS,
    aggregate_multitask_results,
    dataset_protocol,
    load_graphs,
)
from src.scripts.run_research_ideas import extract_embeddings
from src.scripts.run_sota_five_fold import choose_device, save_json


PROTOCOL = "retogcl_ten_lightweight_pipeline_strategies_v1"
IDEA_PARAMETERS = (
    {
        "adversarial_augmentation_budget": 0.1,
        "minimum_clean_ratio": 0.6,
        "bilevel_update_interval": 2.0,
    },
    {
        "cleanliness_beta_prior_strength": 0.5,
        "clean_probability_floor": 0.05,
        "topology_evidence_ratio": 0.5,
    },
    {
        "strata_count": 16.0,
        "bridge_weight": 0.5,
        "tangent_dimension": 32.0,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=PIPELINE_STRATEGIES, required=True)
    parser.add_argument("--dataset", choices=GRAPH_CLASSIFICATION_DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/retogcl_pipeline_strategies_10x18"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None, help="固定外层五折索引的随机种子")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def evaluate_embeddings(
    graphs: list,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    dataset: str,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    if dataset not in MULTITASK_DATASETS:
        metrics = linear_probe_five_fold_metrics(
            embeddings, labels, device=device, seed=args.seed,
            split_seed=args.split_seed,
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
        range(task_labels.size(1)), desc=f"{dataset} 多任务评测", unit="任务"
    ):
        mask = torch.isfinite(task_labels[:, task_index])
        metrics = linear_probe_five_fold_metrics(
            embeddings[mask], task_labels[mask, task_index].long(),
            device=device, seed=args.seed, split_seed=args.split_seed, folds=args.folds,
            epochs=args.probe_epochs,
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


def main() -> None:
    args = parse_args()
    args.data_root = resolve(args.data_root)
    args.output_dir = resolve(args.output_dir)
    strategy = PIPELINE_STRATEGIES[args.strategy]
    model_name = f"ReToGCL-{args.strategy}"
    result_path = args.output_dir / "jobs" / f"{args.strategy}__{args.dataset}.json"
    if result_path.exists() and not args.force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("protocol") == PROTOCOL:
            print(f"已存在，跳过：{result_path}")
            return
        raise RuntimeError(f"拒绝复用不兼容结果：{result_path}")

    device = choose_device(args.device)
    graphs, in_dim, classes = load_graphs(args.data_root, args.dataset)
    set_seed(args.seed)
    model = build_retogcl_pipeline_strategy(
        args.strategy, in_dim, args.hidden_dim, args.layers, classes,
        IDEA_PARAMETERS,
    ).to(device)
    checkpoint = args.output_dir / "checkpoints" / args.strategy / f"{args.dataset}.pt"
    log_path = args.output_dir / "logs" / args.strategy / f"{args.dataset}.jsonl"
    batch_size = min(args.batch_size, 64 if args.dataset == "COLLAB" else args.batch_size)
    expected = {
        "protocol": PROTOCOL,
        "strategy": args.strategy,
        "dataset": args.dataset,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "epochs": args.epochs,
        "idea_parameters": list(IDEA_PARAMETERS),
    }

    if checkpoint.exists() and not args.force:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        differences = {
            key: (state.get(key), value)
            for key, value in expected.items() if state.get(key) != value
        }
        if differences:
            raise RuntimeError(f"拒绝不兼容 checkpoint：{differences}")
        model.load_state_dict(state["model"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            progress = tqdm(
                range(1, args.epochs + 1),
                desc=f"{model_name}/{args.dataset}", unit="轮",
            )
            for epoch in progress:
                model.set_epoch(epoch)
                model.train()
                totals: dict[str, float] = {}
                graph_count = 0
                for batch in loader:
                    batch = batch.to(device)
                    first, second = model.prepare_views(batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss, diagnostics = model.idea_loss(first, second)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"{args.strategy}/{args.dataset}/epoch={epoch} 损失非有限")
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    if not torch.isfinite(gradient):
                        raise RuntimeError(f"{args.strategy}/{args.dataset}/epoch={epoch} 梯度非有限")
                    optimizer.step()
                    count = batch.num_graphs
                    totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * count
                    for key, value in diagnostics.items():
                        totals[key] = totals.get(key, 0.0) + float(value) * count
                    graph_count += count
                record = {
                    "epoch": epoch,
                    **{key: value / max(graph_count, 1) for key, value in totals.items()},
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                progress.set_postfix(
                    loss=f"{record['loss']:.4f}",
                    clean=f"{record['mean_clean_confidence']:.3f}",
                    mix=f"{record['mean_mix']:.3f}",
                )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), **expected}, checkpoint)

    embeddings, labels = extract_embeddings(model, graphs, batch_size, device)
    metric_result = evaluate_embeddings(
        graphs, embeddings, labels, args.dataset, device, args
    )
    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "model": model_name,
        "strategy": args.strategy,
        "strategy_name": strategy.name,
        "strategy_description": strategy.description,
        "strategy_configuration": strategy.__dict__,
        "dataset": args.dataset,
        "dataset_protocol": dataset_protocol(args.dataset),
        "protocol": PROTOCOL,
        "base_model": "ReToGCL_D7-I10_D7-I01_D10-I04_preserved_separately",
        "training": "self_supervised_pretrain_then_stratified_five_fold_linear_probe",
        "configuration": {
            "seed": args.seed,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "probe_epochs": args.probe_epochs,
            "folds": args.folds,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "idea_parameters": list(IDEA_PARAMETERS),
        },
        **metric_result,
    }
    save_json(result_path, result)
    print(f"完成：{result_path}")


if __name__ == "__main__":
    main()
