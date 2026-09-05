#!/usr/bin/env python3
"""运行一个 T077 自适应 Pipeline 配置并执行图分类数据集五折评测。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.gcl_baselines import set_seed
from src.models.retogcl_t077_adaptive_pipeline import (
    T077AdaptivePipelineConfiguration,
    build_retogcl_t077_adaptive_pipeline,
)
from src.scripts.graph_classification_datasets import (
    GRAPH_CLASSIFICATION_DATASETS,
    dataset_protocol,
    load_graphs,
)
from src.scripts.run_research_ideas import extract_embeddings
from src.scripts.run_retogcl_pipeline_strategy_five_fold import (
    IDEA_PARAMETERS,
    evaluate_embeddings,
)
from src.scripts.run_sota_five_fold import choose_device, save_json


TARGET_DATASETS = GRAPH_CLASSIFICATION_DATASETS
PROTOCOL = "retogcl_t077_adaptive_pipeline_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--dataset", choices=TARGET_DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, required=True)
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


def parameter_group(name: str) -> str:
    if "pipeline_gate" in name:
        return "pipeline_gate"
    if "semantic_prototypes" in name:
        return "semantic_prototypes"
    if ".encoder." in name:
        return "encoder"
    if ".projector." in name:
        return "projector"
    if name.startswith("third.bottleneck"):
        return "topology_bottleneck"
    if name.startswith("second."):
        return "clean_module"
    if name.startswith("third."):
        return "topology_module"
    return "other"


def gradient_group_norms(model: torch.nn.Module) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        totals[parameter_group(name)] += float(
            parameter.grad.detach().float().pow(2).sum()
        )
    return {name: math.sqrt(value) for name, value in totals.items()}


def parameter_evidence(
    model: torch.nn.Module,
    initial: dict[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, float]]:
    norms: dict[str, float] = defaultdict(float)
    drifts: dict[str, float] = defaultdict(float)
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        current = parameter.detach().float()
        norms[group] += float(current.pow(2).sum())
        drifts[group] += float((current - initial[name]).pow(2).sum())
    return (
        {name: math.sqrt(value) for name, value in norms.items()},
        {name: math.sqrt(value) for name, value in drifts.items()},
    )


def training_evidence(records: list[dict]) -> dict:
    last = records[-1]
    losses = [float(record["loss"]) for record in records]
    gradients = [float(record["gradient_norm"]) for record in records]
    tail = losses[-min(10, len(losses)):]
    tail_mean = sum(tail) / len(tail)
    tail_std = math.sqrt(sum((value - tail_mean) ** 2 for value in tail) / len(tail))
    return {
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "last_ten_loss_std": tail_std,
        "maximum_gradient_norm": max(gradients),
        "mean_gradient_norm": sum(gradients) / len(gradients),
        "clipped_batch_ratio": sum(
            int(record["clipped_batches"] > 0) for record in records
        ) / len(records),
        "final_parameter_norms": last["parameter_norms"],
        "final_parameter_drifts": last["parameter_drifts"],
        "final_pipeline_gate_mean": last.get("pipeline_gate_mean", 0.0),
        "final_pipeline_gate_std": last.get("pipeline_gate_std", 0.0),
        "final_pipeline_relative_correction": last.get(
            "pipeline_relative_correction", 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    args.manifest = resolve(args.manifest)
    args.data_root = resolve(args.data_root)
    args.output_dir = resolve(args.output_dir)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("manifest 协议不兼容")
    entries = {entry["config_id"]: entry for entry in manifest.get("configs", [])}
    if args.config_id not in entries:
        raise ValueError(f"manifest 中不存在 {args.config_id}")
    entry = entries[args.config_id]
    configuration = T077AdaptivePipelineConfiguration(**entry["configuration"])
    result_path = args.output_dir / "jobs" / f"{args.config_id}__{args.dataset}.json"
    if result_path.exists() and not args.force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("protocol") == PROTOCOL:
            print(f"已存在，跳过：{result_path}")
            return
        raise RuntimeError(f"拒绝复用不兼容结果：{result_path}")

    device = choose_device(args.device)
    graphs, in_dim, classes = load_graphs(args.data_root, args.dataset)
    set_seed(args.seed)
    model = build_retogcl_t077_adaptive_pipeline(
        configuration,
        in_dim,
        args.hidden_dim,
        args.layers,
        classes,
        IDEA_PARAMETERS,
    ).to(device)
    checkpoint = args.output_dir / "checkpoints" / args.config_id / f"{args.dataset}.pt"
    log_path = args.output_dir / "logs" / args.config_id / f"{args.dataset}.jsonl"
    expected = {
        "protocol": PROTOCOL,
        "config_id": args.config_id,
        "dataset": args.dataset,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "epochs": args.epochs,
        "pipeline_configuration": entry["configuration"],
        "idea_parameters": list(IDEA_PARAMETERS),
    }
    records: list[dict] = []
    if checkpoint.exists() and not args.force:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        differences = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if differences:
            raise RuntimeError(f"拒绝不兼容 checkpoint：{differences}")
        model.load_state_dict(state["model"])
        records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-3, weight_decay=5e-4
        )
        batch_size = min(
            args.batch_size,
            64 if args.dataset == "COLLAB" else args.batch_size,
        )
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        initial = {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            progress = tqdm(
                range(1, args.epochs + 1),
                desc=f"{args.config_id}/{args.dataset}",
                unit="轮",
            )
            for epoch in progress:
                model.set_epoch(epoch)
                model.train()
                totals: dict[str, float] = defaultdict(float)
                group_gradient_totals: dict[str, float] = defaultdict(float)
                graph_count = 0
                batch_count = 0
                clipped_batches = 0
                for batch in loader:
                    batch = batch.to(device)
                    first, second = model.prepare_views(batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss, diagnostics = model.idea_loss(first, second)
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"{args.config_id}/{args.dataset}/epoch={epoch} 损失非有限"
                        )
                    loss.backward()
                    group_gradients = gradient_group_norms(model)
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    if not torch.isfinite(gradient):
                        raise RuntimeError(
                            f"{args.config_id}/{args.dataset}/epoch={epoch} 梯度非有限"
                        )
                    clipped_batches += int(float(gradient) > 5.0)
                    optimizer.step()
                    count = batch.num_graphs
                    totals["loss"] += float(loss.detach()) * count
                    totals["gradient_norm"] += float(gradient) * count
                    for key, value in diagnostics.items():
                        if isinstance(value, (int, float)):
                            totals[key] += float(value) * count
                    for key, value in group_gradients.items():
                        group_gradient_totals[key] += value
                    graph_count += count
                    batch_count += 1
                parameter_norms, parameter_drifts = parameter_evidence(model, initial)
                record = {
                    "epoch": epoch,
                    **{
                        key: value / max(graph_count, 1)
                        for key, value in totals.items()
                    },
                    "clipped_batches": clipped_batches,
                    "batch_count": batch_count,
                    "group_gradient_norms": {
                        key: value / max(batch_count, 1)
                        for key, value in group_gradient_totals.items()
                    },
                    "parameter_norms": parameter_norms,
                    "parameter_drifts": parameter_drifts,
                }
                records.append(record)
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                progress.set_postfix(
                    loss=f"{record['loss']:.4f}",
                    gate=f"{record.get('pipeline_gate_mean', 0):.3f}",
                    grad=f"{record['gradient_norm']:.2f}",
                )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), **expected}, checkpoint)

    if len(records) != args.epochs:
        raise ValueError(
            f"训练日志应有 {args.epochs} 轮，实际 {len(records)} 轮"
        )
    batch_size = min(
        args.batch_size,
        64 if args.dataset == "COLLAB" else args.batch_size,
    )
    embeddings, labels = extract_embeddings(model, graphs, batch_size, device)
    metric_result = evaluate_embeddings(
        graphs, embeddings, labels, args.dataset, device, args
    )
    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "model": f"ReToGCL-T077-Pipeline-{args.config_id}",
        "config_id": args.config_id,
        "stage": entry["stage"],
        "focus": entry["focus"],
        "pipeline_configuration": entry["configuration"],
        "dataset": args.dataset,
        "dataset_protocol": dataset_protocol(args.dataset),
        "protocol": PROTOCOL,
        "base_model": "ReToGCL-O07-T077",
        "training": "self_supervised_pretrain_then_stratified_five_fold_linear_probe",
        "training_evidence": training_evidence(records),
        "configuration": {
            "seed": args.seed,
            "split_seed": args.seed if args.split_seed is None else args.split_seed,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "probe_epochs": args.probe_epochs,
            "folds": args.folds,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "weight_decay": 5e-4,
            "idea_parameters": list(IDEA_PARAMETERS),
        },
        **metric_result,
    }
    save_json(result_path, result)
    print(f"完成：{result_path}")


if __name__ == "__main__":
    main()
