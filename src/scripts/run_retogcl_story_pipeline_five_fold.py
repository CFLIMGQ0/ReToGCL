"""运行一套故事化 ReToGCL Pipeline，并进行五折线性评测。"""

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
from src.models.retogcl_story_pipelines import (
    STORY_PIPELINES,
    build_retogcl_story_pipeline,
)
from src.scripts.graph_classification_datasets import dataset_protocol, load_graphs
from src.scripts.run_research_ideas import extract_embeddings
from src.scripts.run_retogcl_pipeline_strategy_five_fold import (
    IDEA_PARAMETERS,
    evaluate_embeddings,
)
from src.scripts.run_sota_five_fold import choose_device, save_json


TARGET_DATASETS = ("ADHD200", "BACE", "BBBP", "Tox21", "AIDS", "BZR")
PROTOCOL = "retogcl_story_pipelines_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=STORY_PIPELINES, required=True)
    parser.add_argument("--dataset", choices=TARGET_DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/retogcl_story_pipelines_8x6"),
    )
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
    args.data_root = resolve(args.data_root)
    args.output_dir = resolve(args.output_dir)
    pipeline = STORY_PIPELINES[args.pipeline]
    model_name = f"ReToGCL-Story-{args.pipeline}"
    result_path = args.output_dir / "jobs" / f"{args.pipeline}__{args.dataset}.json"
    if result_path.exists() and not args.force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("protocol") == PROTOCOL:
            print(f"已存在，跳过：{result_path}")
            return
        raise RuntimeError(f"拒绝复用不兼容结果：{result_path}")

    device = choose_device(args.device)
    graphs, in_dim, classes = load_graphs(args.data_root, args.dataset)
    set_seed(args.seed)
    model = build_retogcl_story_pipeline(
        args.pipeline, in_dim, args.hidden_dim, args.layers, classes,
        IDEA_PARAMETERS,
    ).to(device)
    checkpoint = (
        args.output_dir / "checkpoints" / args.pipeline / f"{args.dataset}.pt"
    )
    log_path = args.output_dir / "logs" / args.pipeline / f"{args.dataset}.jsonl"
    expected = {
        "protocol": PROTOCOL,
        "pipeline": args.pipeline,
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
        loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)
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
                        raise RuntimeError(
                            f"{args.pipeline}/{args.dataset}/epoch={epoch} 损失非有限"
                        )
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    if not torch.isfinite(gradient):
                        raise RuntimeError(
                            f"{args.pipeline}/{args.dataset}/epoch={epoch} 梯度非有限"
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
                progress.set_postfix(
                    loss=f"{record['loss']:.4f}",
                    mix=f"{record['mean_mix']:.3f}",
                    topo=f"{record['mean_topology_trust']:.3f}",
                )
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
        "model": model_name,
        "pipeline": args.pipeline,
        "pipeline_name": pipeline.name,
        "pipeline_story": pipeline.story,
        "pipeline_configuration": pipeline.__dict__,
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
