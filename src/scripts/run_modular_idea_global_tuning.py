"""在真实模块位置对单个 idea 做跨七数据集 A→B→C 顺序调参。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.evaluation import METRIC_KEYS, linear_probe_five_fold_metrics, summarize
from src.baselines.gcl_baselines import set_seed
from src.models.idea_parameter_registry import parameter_space
from src.models.modular_research_ideas import (
    IDEA_TO_MODULE,
    build_modular_research_idea_model,
)
from src.models.research_ideas import IDEA_IDS
from src.scripts.run_research_ideas import DATASETS, extract_embeddings
from src.scripts.run_sota_five_fold import choose_device, load_graphs, save_json


PROTOCOL = "modular_cross_dataset_sequential_A_then_B_then_C_v1"
FIDELITY = "module_position_faithful_differentiable_implementation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", choices=IDEA_IDS, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("IDEA_PARA_MODULAR"))
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
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def configuration_id(args: argparse.Namespace, parameters: dict[str, float]) -> str:
    payload = {
        "protocol": PROTOCOL,
        "idea": args.idea,
        "parameters": parameters,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "epochs": args.epochs,
        "probe_epochs": args.probe_epochs,
        "folds": args.folds,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def build_model(args: argparse.Namespace, in_dim: int, classes: int, parameters: dict[str, float]):
    return build_modular_research_idea_model(
        args.idea,
        in_dim,
        args.hidden_dim,
        args.layers,
        classes,
        idea_parameters=parameters,
    )


def selection_key(dataset_results: dict[str, dict[str, Any]]) -> tuple[float, ...]:
    """七数据集等权、四指标等权；平分时按 Accuracy、F1、NMI、ARI。"""

    per_metric = []
    for metric in METRIC_KEYS:
        per_metric.append(
            sum(float(dataset_results[dataset][f"{metric}_percent"]["mean"]) for dataset in DATASETS)
            / len(DATASETS)
        )
    return (
        sum(per_metric) / len(per_metric),
        per_metric[0],
        per_metric[3],
        per_metric[1],
        per_metric[2],
    )


def run_trial(
    args: argparse.Namespace,
    dataset: str,
    graphs,
    in_dim: int,
    classes: int,
    parameters: dict[str, float],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    config_id = configuration_id(args, parameters)
    trial_path = output_dir / "trials" / args.idea / dataset / f"{config_id}.json"
    if trial_path.exists() and not args.force:
        result = json.loads(trial_path.read_text(encoding="utf-8"))
        if result.get("protocol") == PROTOCOL and result.get("fidelity") == FIDELITY:
            return result

    set_seed(args.seed)
    model = build_model(args, in_dim, classes, parameters).to(device)
    checkpoint = output_dir / "checkpoints" / args.idea / dataset / f"{config_id}.pt"
    log_path = output_dir / "logs" / args.idea / dataset / f"{config_id}.jsonl"
    batch_size = min(args.batch_size, 64 if dataset == "COLLAB" else args.batch_size)

    if checkpoint.exists() and not args.force:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state.get("protocol") != PROTOCOL or state.get("parameters") != parameters:
            raise RuntimeError(f"拒绝不兼容 checkpoint：{checkpoint}")
        model.load_state_dict(state["model"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            progress = tqdm(
                range(1, args.epochs + 1),
                desc=f"{args.idea}/{dataset}/{config_id}",
                unit="轮",
                leave=False,
            )
            for epoch in progress:
                model.train()
                total_loss = 0.0
                total_graphs = 0
                diagnostics: dict[str, Any] = {}
                for batch in loader:
                    batch = batch.to(device)
                    first, second = model.prepare_views(batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss, diagnostics = model.idea_loss(first, second)
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"{args.idea}/{dataset}/{config_id}/epoch={epoch} 损失非有限"
                        )
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    if not torch.isfinite(gradient):
                        raise RuntimeError(
                            f"{args.idea}/{dataset}/{config_id}/epoch={epoch} 梯度非有限"
                        )
                    optimizer.step()
                    total_loss += float(loss.detach()) * batch.num_graphs
                    total_graphs += batch.num_graphs
                record = {
                    "epoch": epoch,
                    "loss": total_loss / max(total_graphs, 1),
                    **diagnostics,
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                progress.set_postfix(loss=f"{record['loss']:.4f}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "protocol": PROTOCOL,
                "fidelity": FIDELITY,
                "idea_id": args.idea,
                "target_module": IDEA_TO_MODULE[args.idea],
                "dataset": dataset,
                "config_id": config_id,
                "parameters": parameters,
            },
            checkpoint,
        )

    embeddings, labels = extract_embeddings(model, graphs, batch_size, device)
    metrics = linear_probe_five_fold_metrics(
        embeddings,
        labels,
        device=device,
        seed=args.seed,
        folds=args.folds,
        epochs=args.probe_epochs,
    )
    result: dict[str, Any] = {
        "idea_id": args.idea,
        "target_module": IDEA_TO_MODULE[args.idea],
        "dataset": dataset,
        "config_id": config_id,
        "parameters": parameters,
        "protocol": PROTOCOL,
        "fidelity": FIDELITY,
        "fixed_non_idea_parameters": {
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
        },
        **{f"{key}_percent": summarize(values) for key, values in metrics.items()},
    }
    save_json(trial_path, result)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def smoke_test(args: argparse.Namespace, data_root: Path, device: torch.device) -> None:
    graphs, in_dim, classes = load_graphs(data_root, "MUTAG")
    batch = next(iter(DataLoader(graphs[:8], batch_size=min(8, len(graphs))))).to(device)
    space = parameter_space(args.idea)
    tested = 0
    for parameter in space.parameters:
        for value in parameter.values:
            parameters = dict(space.defaults)
            parameters[parameter.name] = float(value)
            set_seed(args.seed)
            model = build_model(args, in_dim, classes, parameters).to(device)
            first, second = model.prepare_views(batch)
            loss, diagnostics = model.idea_loss(first, second)
            if diagnostics["target_module"] != IDEA_TO_MODULE[args.idea]:
                raise RuntimeError(f"{args.idea} 接入模块不一致")
            if not torch.isfinite(loss):
                raise RuntimeError(f"{args.idea}/{parameter.name}={value} 损失非有限")
            loss.backward()
            gradients = [item.grad for item in model.parameters() if item.requires_grad and item.grad is not None]
            if not gradients or not all(torch.isfinite(item).all() for item in gradients):
                raise RuntimeError(f"{args.idea}/{parameter.name}={value} 梯度异常")
            tested += 1
            del model
    print(f"{args.idea} 模块化冒烟通过：target={IDEA_TO_MODULE[args.idea]}，参数点={tested}")


def main() -> None:
    args = parse_args()
    data_root = resolve(args.data_root)
    output_dir = resolve(args.output_dir)
    device = choose_device(args.device)
    if args.smoke_test:
        smoke_test(args, data_root, device)
        return

    final_path = output_dir / "jobs" / f"{args.idea}.json"
    if final_path.exists() and not args.force:
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("protocol") == PROTOCOL:
            print(f"已完成，跳过：{final_path}")
            return

    datasets = {
        dataset: load_graphs(data_root, dataset)
        for dataset in tqdm(DATASETS, desc=f"{args.idea}/加载数据", unit="数据集")
    }
    space = parameter_space(args.idea)
    selected = dict(space.defaults)
    stages = []
    for stage_index, parameter in enumerate(space.parameters):
        candidates = []
        for value in parameter.values:
            parameters = dict(selected)
            parameters[parameter.name] = float(value)
            dataset_results = {}
            for dataset in DATASETS:
                graphs, in_dim, classes = datasets[dataset]
                dataset_results[dataset] = run_trial(
                    args,
                    dataset,
                    graphs,
                    in_dim,
                    classes,
                    parameters,
                    output_dir,
                    device,
                )
            key = selection_key(dataset_results)
            candidates.append({
                "value": float(value),
                "parameters": parameters,
                "global_selection_score": key[0],
                "dataset_results": dataset_results,
            })
        best = max(candidates, key=lambda item: selection_key(item["dataset_results"]))
        selected[parameter.name] = float(best["value"])
        stages.append({
            "stage": "ABC"[stage_index],
            "parameter": parameter.name,
            "candidates": candidates,
            "selected_value": float(best["value"]),
            "selected_parameters_after_stage": dict(selected),
        })
        save_json(
            output_dir / "sequences" / f"{args.idea}.json",
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "idea_id": args.idea,
                "target_module": IDEA_TO_MODULE[args.idea],
                "status": f"stage_{'ABC'[stage_index]}_completed",
                "protocol": PROTOCOL,
                "fidelity": FIDELITY,
                "selected_parameters": selected,
                "stages": stages,
            },
        )

    final_results = {}
    for dataset in DATASETS:
        graphs, in_dim, classes = datasets[dataset]
        final_results[dataset] = run_trial(
            args, dataset, graphs, in_dim, classes, selected, output_dir, device
        )
    save_json(
        final_path,
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "idea_id": args.idea,
            "target_module": IDEA_TO_MODULE[args.idea],
            "status": "completed",
            "protocol": PROTOCOL,
            "fidelity": FIDELITY,
            "selected_parameters": selected,
            "selection_score": selection_key(final_results)[0],
            "stages": stages,
            "final_results": final_results,
        },
    )
    print(f"{args.idea} 模块化 ABC 完成：{selected}")


if __name__ == "__main__":
    main()
