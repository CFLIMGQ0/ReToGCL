"""运行一个固定全局最优 ABC 的双 idea、单数据集五折任务。"""

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
from src.models.pairwise_research_ideas import build_pairwise_research_idea_model
from src.models.research_ideas import get_idea_spec
from src.scripts.run_research_ideas import _augment_pair, extract_embeddings
from src.scripts.run_sota_five_fold import choose_device, load_graphs, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--task-json",
        help="由启动器直接传入单任务 JSON，避免每个工作进程重复解析大型清单",
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("IDEA_PAIRWISE"))
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
        raise RuntimeError(f"任务 {task_id} 在清单中出现{len(matches)}次")
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
        print(f"已存在，跳过：{result_path}")
        return

    task = json.loads(args.task_json) if args.task_json else load_task(manifest, args.task_id)
    if task.get("task_id") != args.task_id:
        raise RuntimeError("命令行 task-id 与任务 JSON 不一致")
    first_idea = str(task["first_idea"])
    second_idea = str(task["second_idea"])
    if get_idea_spec(first_idea).family == get_idea_spec(second_idea).family:
        raise RuntimeError(f"同方向 idea 不允许组合：{first_idea}+{second_idea}")
    first_parameters = validate_parameters(first_idea, task["first_parameters"])
    second_parameters = validate_parameters(second_idea, task["second_parameters"])
    dataset = str(task["dataset"])
    device = choose_device(args.device)
    print(f"组合任务：{args.task_id}，设备：{device}")

    graphs, in_dim, classes = load_graphs(data_root, dataset)
    set_seed(args.seed)
    model = build_pairwise_research_idea_model(
        first_idea,
        second_idea,
        in_dim,
        args.hidden_dim,
        args.layers,
        classes,
        first_parameters,
        second_parameters,
    ).to(device)
    checkpoint = output_dir / "checkpoints" / task["pair_id"] / f"{dataset}.pt"
    log_path = output_dir / "logs" / task["pair_id"] / f"{dataset}.jsonl"
    batch_size = min(args.batch_size, 64 if dataset == "COLLAB" else args.batch_size)

    if checkpoint.exists() and not args.force:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
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
                    first, second = _augment_pair(batch)
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
                        diagnostics_sum[key] = diagnostics_sum.get(key, 0.0) + value * batch.num_graphs
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
        torch.save(
            {
                "model": model.state_dict(),
                "task_id": args.task_id,
                "pair_id": task["pair_id"],
                "first_idea": first_idea,
                "second_idea": second_idea,
                "first_parameters": first_parameters,
                "second_parameters": second_parameters,
                "dataset": dataset,
                "epochs": args.epochs,
                "protocol": "shared_encoder_joint_two_idea_branches",
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
    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": args.task_id,
        "pair_id": task["pair_id"],
        "status": "completed",
        "first_idea": first_idea,
        "second_idea": second_idea,
        "first_family": task["first_family"],
        "second_family": task["second_family"],
        "first_parameters": first_parameters,
        "second_parameters": second_parameters,
        "dataset": dataset,
        "protocol": "shared_encoder_joint_two_idea_branches_fixed_global_ABC",
        "training": "self_supervised_pretrain_then_stratified_five_fold_linear_probe",
        "fidelity": "differentiable_pairwise_prototype",
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
        "accuracy_percent": summarize(metrics["accuracy"]),
        "nmi_percent": summarize(metrics["nmi"]),
        "ari_percent": summarize(metrics["ari"]),
        "macro_f1_percent": summarize(metrics["macro_f1"]),
    }
    save_json(result_path, result)
    print(f"两两组合任务完成：{result_path}")


if __name__ == "__main__":
    main()
