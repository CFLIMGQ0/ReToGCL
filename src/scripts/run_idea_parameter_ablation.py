"""运行 idea 模块参数的单因素五折消融实验。"""

from __future__ import annotations

import argparse
import gc
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

from src.baselines.evaluation import METRIC_KEYS, linear_probe_five_fold_metrics, summarize
from src.baselines.gcl_baselines import set_seed
from src.models.idea_parameter_ablation import CONFIG_BY_ID, PARAMETER_CONFIGS
from src.models.research_ideas import IDEA_IDS, build_research_idea_model, get_idea_spec
from src.scripts.run_research_ideas import _augment_pair, extract_embeddings
from src.scripts.run_sota_five_fold import choose_device, load_graphs, save_json


DATASETS = ("MUTAG", "PTC_MR", "Mutagenicity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ideas", nargs="+", choices=IDEA_IDS, default=list(IDEA_IDS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--config-ids", nargs="+", choices=tuple(CONFIG_BY_ID),
        default=[config.config_id for config in PARAMETER_CONFIGS],
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("IDEA_PARA"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    return parser.parse_args()


def resolved_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = args.data_root if args.data_root.is_absolute() else PROJECT_ROOT / args.data_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    return data_root.resolve(), output_dir.resolve()


def build_model(args: argparse.Namespace, idea_id: str, in_dim: int, classes: int, config_id: str):
    parameters = CONFIG_BY_ID[config_id].parameters
    return build_research_idea_model(
        idea_id, in_dim, args.hidden_dim, args.layers, classes,
        auxiliary_weight=parameters["idea_loss_weight"],
        idea_temperature=parameters["idea_temperature"],
        operator_mix=parameters["operator_mix"],
    )


def smoke_test(args: argparse.Namespace, data_root: Path, device: torch.device) -> None:
    graphs, in_dim, classes = load_graphs(data_root, args.datasets[0])
    batch = next(iter(DataLoader(graphs[:8], batch_size=8))).to(device)
    combinations = [(idea, config) for idea in args.ideas for config in args.config_ids]
    for offset, (idea_id, config_id) in enumerate(tqdm(combinations, desc="参数消融冒烟", unit="配置")):
        set_seed(args.seed + offset)
        model = build_model(args, idea_id, in_dim, classes, config_id).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        first, second = _augment_pair(batch)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.idea_loss(first, second)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{idea_id}/{config_id} 出现非有限损失")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(gradient):
            raise RuntimeError(f"{idea_id}/{config_id} 出现非有限梯度")
        optimizer.step()
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_one(args: argparse.Namespace, idea_id: str, dataset_name: str, config_id: str, data_root: Path, output_dir: Path, device: torch.device) -> dict:
    graphs, in_dim, classes = load_graphs(data_root, dataset_name)
    config = CONFIG_BY_ID[config_id]
    set_seed(args.seed)
    model = build_model(args, idea_id, in_dim, classes, config_id).to(device)
    checkpoint = output_dir / "checkpoints" / idea_id / dataset_name / f"{config_id}.pt"
    log_path = output_dir / "logs" / idea_id / dataset_name / f"{config_id}.jsonl"
    batch_size = min(args.batch_size, 128)

    if checkpoint.exists() and not args.force:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            progress = tqdm(
                range(1, args.epochs + 1),
                desc=f"{idea_id}/{dataset_name}/{config_id}", unit="轮", leave=False,
            )
            for epoch in progress:
                model.train(); total_loss = total_graphs = 0
                for batch in loader:
                    batch = batch.to(device)
                    first, second = _augment_pair(batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss, diagnostics = model.idea_loss(first, second)
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"{idea_id}/{dataset_name}/{config_id}/epoch={epoch} 出现非有限损失"
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += float(loss.detach()) * batch.num_graphs
                    total_graphs += batch.num_graphs
                record = {"epoch": epoch, "loss": total_loss / max(total_graphs, 1), **diagnostics}
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush(); progress.set_postfix(loss=f"{record['loss']:.4f}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(), "idea_id": idea_id, "dataset": dataset_name,
            "config_id": config_id, "parameters": config.parameters, "epochs": args.epochs,
        }, checkpoint)

    embeddings, labels = extract_embeddings(model, graphs, batch_size, device)
    metrics = linear_probe_five_fold_metrics(
        embeddings, labels, device=device, seed=args.seed,
        folds=args.folds, epochs=args.probe_epochs,
    )
    spec = get_idea_spec(idea_id)
    return {
        "idea_id": idea_id, "family": spec.family_name, "revision": spec.revision,
        "dataset": dataset_name, "config_id": config_id,
        "varied_parameter": config.varied_parameter,
        "varied_value": config.varied_value, "parameters": config.parameters,
        "protocol": "single_factor_idea_module_only",
        "fixed_non_idea_parameters": {
            "seed": args.seed, "hidden_dim": args.hidden_dim, "layers": args.layers,
            "batch_size": args.batch_size, "epochs": args.epochs,
            "probe_epochs": args.probe_epochs, "folds": args.folds,
            "optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 1e-5,
            "base_contrastive_temperature": 0.2,
        },
        **{f"{key}_percent": summarize(values) for key, values in metrics.items()},
    }


def merge_results(output_dir: Path) -> Path:
    experiments = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in tqdm(sorted((output_dir / "jobs").glob("*.json")), desc="汇总参数消融", unit="任务")
    ]
    result = {
        "metadata": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": "single_factor_idea_module_only",
            "experiment_count": len(experiments),
            "idea_count": len({item["idea_id"] for item in experiments}),
            "dataset_count": len({item["dataset"] for item in experiments}),
            "config_count": len({item["config_id"] for item in experiments}),
            "metrics": list(METRIC_KEYS),
        },
        "experiments": sorted(
            experiments, key=lambda item: (item["idea_id"], item["dataset"], item["config_id"])
        ),
    }
    path = output_dir / "five_fold_results.json"; save_json(path, result); return path


def main() -> None:
    args = parse_args(); data_root, output_dir = resolved_paths(args)
    if args.merge_only:
        print(f"已汇总：{merge_results(output_dir)}"); return
    device = choose_device(args.device); print(f"使用设备：{device}")
    if args.smoke_test:
        smoke_test(args, data_root, device); print("参数消融冒烟全部通过"); return
    combinations = [
        (idea, dataset, config)
        for idea in args.ideas for dataset in args.datasets for config in args.config_ids
    ]
    for idea_id, dataset_name, config_id in tqdm(combinations, desc="参数消融总进度", unit="任务"):
        job_path = output_dir / "jobs" / f"{idea_id}_{dataset_name}_{config_id}.json"
        if job_path.exists() and not args.force:
            continue
        result = run_one(args, idea_id, dataset_name, config_id, data_root, output_dir, device)
        save_json(job_path, result)
    if not args.no_merge:
        print(f"已汇总：{merge_results(output_dir)}")


if __name__ == "__main__":
    main()
