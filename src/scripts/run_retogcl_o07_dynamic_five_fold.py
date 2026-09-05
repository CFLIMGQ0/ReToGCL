"""运行一个 manifest 中的 O07 动态调参版本并执行五折评测。"""

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

from src.baselines.gcl_baselines import set_seed
from src.models.retogcl_o07_dynamic_tuning import (
    O07DynamicConfiguration,
    build_retogcl_o07_dynamic_tuning,
)
from src.scripts.graph_classification_datasets import dataset_protocol, load_graphs
from src.scripts.run_research_ideas import extract_embeddings
from src.scripts.run_retogcl_pipeline_strategy_five_fold import (
    IDEA_PARAMETERS,
    evaluate_embeddings,
)
from src.scripts.run_sota_five_fold import choose_device, save_json


TARGET_DATASETS = ("ADHD200", "BACE", "BBBP", "Tox21", "AIDS", "BZR")
PROTOCOL = "retogcl_o07_dynamic_tuning_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--dataset", choices=TARGET_DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
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


def main() -> None:
    args = parse_args()
    args.manifest = resolve(args.manifest)
    args.data_root = resolve(args.data_root)
    args.output_dir = resolve(args.output_dir)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("manifest 协议不兼容")
    entries = {
        entry["config_id"]: entry for entry in manifest.get("configs", [])
    }
    if args.config_id not in entries:
        raise ValueError(f"manifest 中不存在 {args.config_id}")
    entry = entries[args.config_id]
    configuration = O07DynamicConfiguration(**entry["configuration"])
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
    model = build_retogcl_o07_dynamic_tuning(
        configuration, in_dim, args.hidden_dim, args.layers, classes,
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
        "tuning_configuration": entry["configuration"],
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
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-3,
            weight_decay=configuration.weight_decay,
        )
        loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            progress = tqdm(
                range(1, args.epochs + 1),
                desc=f"{args.config_id}/{args.dataset}", unit="轮",
            )
            for epoch in progress:
                model.train()
                totals: dict[str, float] = {}
                graph_count = 0
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
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    if not torch.isfinite(gradient):
                        raise RuntimeError(
                            f"{args.config_id}/{args.dataset}/epoch={epoch} 梯度非有限"
                        )
                    optimizer.step()
                    count = batch.num_graphs
                    totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * count
                    for key, value in diagnostics.items():
                        totals[key] = totals.get(key, 0.0) + float(value) * count
                    graph_count += count
                record = {
                    "epoch": epoch,
                    **{
                        key: value / max(graph_count, 1)
                        for key, value in totals.items()
                    },
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                progress.set_postfix(loss=f"{record['loss']:.4f}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), **expected}, checkpoint)

    embeddings, labels = extract_embeddings(
        model, graphs, args.batch_size, device
    )
    metric_result = evaluate_embeddings(
        graphs, embeddings, labels, args.dataset, device, args
    )
    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "model": f"ReToGCL-O07-{args.config_id}",
        "config_id": args.config_id,
        "stage": entry["stage"],
        "focus": entry["focus"],
        "changed_parameters": entry["changed_parameters"],
        "tuning_configuration": entry["configuration"],
        "dataset": args.dataset,
        "dataset_protocol": dataset_protocol(args.dataset),
        "protocol": PROTOCOL,
        "base_model": "ReToGCL-O07-P05-temperature-0.06",
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
            "weight_decay": configuration.weight_decay,
            "idea_parameters": list(IDEA_PARAMETERS),
        },
        **metric_result,
    }
    save_json(result_path, result)
    print(f"完成：{result_path}")


if __name__ == "__main__":
    main()
