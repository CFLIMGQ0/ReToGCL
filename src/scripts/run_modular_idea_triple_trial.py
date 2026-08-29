"""运行一个固定 ABC 参数的跨模块三 idea、单数据集五折任务。"""

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
from src.models.idea_parameter_registry import parameter_space
from src.models.modular_research_ideas import IDEA_TO_MODULE
from src.models.modular_triple_research_ideas import (
    MODULE_ORDER,
    build_modular_triple_research_idea_model,
)
from src.scripts.run_research_ideas import extract_embeddings
from src.scripts.run_sota_five_fold import choose_device, load_graphs, save_json
from src.scripts.graph_classification_datasets import (
    MULTITASK_DATASETS,
    aggregate_multitask_results,
    dataset_protocol,
)


PROTOCOL = "modular_cross_module_triple_from_top5_pairwise_fixed_ABC_v1"
FIDELITY = "module_position_faithful_differentiable_triple_implementation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-json")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("IDEA_TRIPLE_MODULAR_TOP5"))
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


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_task(manifest: Path, task_id: str) -> dict:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    matches = [task for task in payload["tasks"] if task["task_id"] == task_id]
    if len(matches) != 1:
        raise RuntimeError(f"任务 {task_id} 在清单中出现 {len(matches)} 次")
    return matches[0]


def validate_parameters(idea_id: str, values: dict) -> dict[str, float]:
    space = parameter_space(idea_id)
    expected = {parameter.name for parameter in space.parameters}
    if set(values) != expected:
        raise RuntimeError(f"{idea_id} 参数字段异常")
    normalized = {name: float(value) for name, value in values.items()}
    for parameter in space.parameters:
        value = normalized[parameter.name]
        if not any(abs(value - float(candidate)) < 1e-12 for candidate in parameter.values):
            raise RuntimeError(f"{idea_id}/{parameter.name}={value} 不在候选空间中")
    return normalized


def main() -> None:
    args = parse_args()
    manifest = resolve_path(args.manifest)
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    result_path = output_dir / "jobs" / f"{args.task_id}.json"
    if result_path.exists() and not args.force:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "completed" and result.get("protocol") == PROTOCOL:
            print(f"已存在，跳过：{result_path}")
            return
        raise RuntimeError(f"拒绝复用不兼容结果：{result_path}")

    task = json.loads(args.task_json) if args.task_json else load_task(manifest, args.task_id)
    if task.get("task_id") != args.task_id:
        raise RuntimeError("命令行 task-id 与任务 JSON 不一致")
    idea_ids = tuple(str(idea_id) for idea_id in task["idea_ids"])
    if len(idea_ids) != 3:
        raise RuntimeError("三模块任务必须包含三个 idea")
    modules = tuple(IDEA_TO_MODULE[idea_id] for idea_id in idea_ids)
    if len(set(modules)) != 3:
        raise RuntimeError(f"同模块 idea 不允许组合：{idea_ids} -> {modules}")
    if tuple(task.get("modules", ())) != modules:
        raise RuntimeError("任务清单中的模块映射与代码不一致")
    if tuple(sorted(zip(modules, idea_ids), key=lambda x: (MODULE_ORDER[x[0]], x[1]))) != tuple(
        zip(modules, idea_ids)
    ):
        raise RuntimeError("三模块任务没有按 M1 到 M8 排序")
    idea_parameters = tuple(
        validate_parameters(idea_id, values)
        for idea_id, values in zip(idea_ids, task["idea_parameters"], strict=True)
    )
    dataset = str(task["dataset"])
    device = choose_device(args.device)
    print(
        f"跨模块三组合任务：{args.task_id}，模块：{'+'.join(modules)}，设备：{device}"
    )

    graphs, in_dim, classes = load_graphs(data_root, dataset)
    set_seed(args.seed)
    model = build_modular_triple_research_idea_model(
        idea_ids,
        in_dim,
        args.hidden_dim,
        args.layers,
        classes,
        idea_parameters,
    ).to(device)
    checkpoint = output_dir / "checkpoints" / task["triple_id"] / f"{dataset}.pt"
    log_path = output_dir / "logs" / task["triple_id"] / f"{dataset}.jsonl"
    batch_size = min(args.batch_size, 64 if dataset == "COLLAB" else args.batch_size)

    expected_checkpoint = {
        "protocol": PROTOCOL,
        "fidelity": FIDELITY,
        "idea_ids": list(idea_ids),
        "modules": list(modules),
        "idea_parameters": list(idea_parameters),
        "dataset": dataset,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "epochs": args.epochs,
    }
    if checkpoint.exists() and not args.force:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected_checkpoint.items()
            if state.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"拒绝不兼容 checkpoint：{checkpoint}，差异={mismatches}")
        model.load_state_dict(state["model"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            progress = tqdm(range(1, args.epochs + 1), desc=args.task_id, unit="轮", leave=False)
            for epoch in progress:
                model.train()
                total_loss = 0.0
                total_graphs = 0
                diagnostics_sum: dict[str, float] = {}
                for batch in loader:
                    batch = batch.to(device)
                    first, second = model.prepare_views(batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss, diagnostics = model.idea_loss(first, second)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"{args.task_id}/epoch={epoch} 损失非有限")
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    if not torch.isfinite(gradient):
                        raise RuntimeError(f"{args.task_id}/epoch={epoch} 梯度非有限")
                    optimizer.step()
                    total_loss += float(loss.detach()) * batch.num_graphs
                    total_graphs += batch.num_graphs
                    for key, value in diagnostics.items():
                        diagnostics_sum[key] = diagnostics_sum.get(key, 0.0) + float(value) * batch.num_graphs
                record = {
                    "epoch": epoch,
                    "loss": total_loss / max(total_graphs, 1),
                    **{
                        key: value / max(total_graphs, 1)
                        for key, value in diagnostics_sum.items()
                    },
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                progress.set_postfix(loss=f"{record['loss']:.4f}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), **expected_checkpoint}, checkpoint)

    embeddings, labels = extract_embeddings(model, graphs, batch_size, device)
    if dataset in MULTITASK_DATASETS:
        task_labels = torch.cat([graph.task_y for graph in graphs], dim=0)
        task_results = []
        for task_index in range(task_labels.size(1)):
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
        metric_result = aggregate_multitask_results(dataset, task_results)
    else:
        metrics = linear_probe_five_fold_metrics(
            embeddings,
            labels,
            device=device,
            seed=args.seed,
            folds=args.folds,
            epochs=args.probe_epochs,
        )
        metric_result = {
            "accuracy_percent": summarize(metrics["accuracy"]),
            "nmi_percent": summarize(metrics["nmi"]),
            "ari_percent": summarize(metrics["ari"]),
            "macro_f1_percent": summarize(metrics["macro_f1"]),
        }
    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": args.task_id,
        "triple_id": task["triple_id"],
        "status": "completed",
        "idea_ids": list(idea_ids),
        "modules": list(modules),
        "idea_parameters": list(idea_parameters),
        "source_top_pairs": list(task["source_top_pairs"]),
        "dataset": dataset,
        "dataset_protocol": dataset_protocol(dataset),
        "protocol": PROTOCOL,
        "training": "self_supervised_pretrain_then_stratified_five_fold_linear_probe",
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
            "auxiliary_weight_per_idea": 0.25,
        },
        **metric_result,
    }
    save_json(result_path, result)
    print(f"跨模块三组合任务完成：{result_path}")


if __name__ == "__main__":
    main()
