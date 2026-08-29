"""运行 110 个 BalanceGCL 研究原型的冒烟测试或单机五折实验。"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.evaluation import METRIC_KEYS, linear_probe_five_fold_metrics, summarize
from src.baselines.gcl_baselines import augment_graph, set_seed
from src.models.research_ideas import IDEA_IDS, build_research_idea_model, get_idea_spec
from src.scripts.run_sota_five_fold import choose_device, load_graphs, save_json


DATASETS = (
    "NCI1", "PROTEINS", "COLLAB", "MUTAG", "COLORS-3",
    "PTC_MR", "Mutagenicity",
)
AUGMENTATIONS = ("node_drop", "edge_drop", "attr_mask", "subgraph")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ideas", nargs="+", choices=IDEA_IDS, default=list(IDEA_IDS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research_ideas"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="每个 idea 仅执行一次前向、反向和更新")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    return parser.parse_args()


def _paths(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = args.data_root if args.data_root.is_absolute() else PROJECT_ROOT / args.data_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    return data_root.resolve(), output_dir.resolve()


def _augment_pair(data):
    choices = np.random.choice(AUGMENTATIONS, size=2, replace=False)
    return augment_graph(data, str(choices[0]), 0.2), augment_graph(data, str(choices[1]), 0.2)


@torch.no_grad()
def extract_embeddings(model, graphs, batch_size: int, device: torch.device):
    model.eval()
    embeddings, labels = [], []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        embeddings.append(model.encode(batch).cpu())
        labels.append(batch.y.cpu())
    return torch.cat(embeddings), torch.cat(labels)


def smoke_test(args: argparse.Namespace, data_root: Path, output_dir: Path, device: torch.device) -> Path:
    dataset_name = args.datasets[0]
    graphs, in_dim, num_classes = load_graphs(data_root, dataset_name)
    batch_size = min(args.batch_size, 8, len(graphs))
    batch = next(iter(DataLoader(graphs[: max(batch_size, 2)], batch_size=batch_size, shuffle=False))).to(device)
    records = []
    for offset, idea_id in enumerate(tqdm(args.ideas, desc="110 个 idea 冒烟测试", unit="idea")):
        set_seed(args.seed + offset)
        model = build_research_idea_model(
            idea_id, in_dim, args.hidden_dim, args.layers, num_classes,
            args.auxiliary_weight,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        try:
            first, second = _augment_pair(batch)
            optimizer.zero_grad(set_to_none=True)
            loss, diagnostics = model.idea_loss(first, second)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"损失不是有限值：{float(loss.detach())}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"梯度不是有限值：{float(gradient_norm)}")
            optimizer.step()
            records.append({
                "idea_id": idea_id,
                "family": get_idea_spec(idea_id).family_name,
                "revision": get_idea_spec(idea_id).revision,
                "status": "passed",
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                **diagnostics,
            })
        except Exception as error:  # 将全部失败收集后一次报告。
            records.append({
                "idea_id": idea_id,
                "family": get_idea_spec(idea_id).family_name,
                "revision": get_idea_spec(idea_id).revision,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            })
        finally:
            del model, optimizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_name,
            "device": str(device),
            "total": len(records),
            "passed": sum(item["status"] == "passed" for item in records),
            "failed": sum(item["status"] == "failed" for item in records),
            "fidelity": "differentiable_prototype",
        },
        "ideas": records,
    }
    path = output_dir / "smoke_test_results.json"
    save_json(path, report)
    failures = [item for item in records if item["status"] == "failed"]
    if failures:
        summary = "\n".join(f"- {item['idea_id']}: {item['error']}" for item in failures)
        raise RuntimeError(f"{len(failures)} 个 idea 冒烟失败：\n{summary}\n完整报告：{path}")
    return path


def run_experiment(
    args: argparse.Namespace,
    idea_id: str,
    dataset_name: str,
    graphs,
    in_dim: int,
    num_classes: int,
    output_dir: Path,
    device: torch.device,
) -> dict:
    set_seed(args.seed)
    model = build_research_idea_model(
        idea_id, in_dim, args.hidden_dim, args.layers, num_classes,
        args.auxiliary_weight,
    ).to(device)
    checkpoint = output_dir / "checkpoints" / idea_id / f"{dataset_name}.pt"
    log_path = output_dir / "logs" / idea_id / f"{dataset_name}.jsonl"
    batch_size = min(args.batch_size, 64 if dataset_name == "COLLAB" else args.batch_size)
    if checkpoint.exists() and not args.force:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            progress = tqdm(range(1, args.epochs + 1), desc=f"{idea_id}/{dataset_name}", unit="轮")
            for epoch in progress:
                model.train()
                total_loss = total_graphs = 0
                diagnostics_sum = {"base": 0.0, "idea_contrast": 0.0, "auxiliary": 0.0}
                for batch in loader:
                    batch = batch.to(device)
                    first, second = _augment_pair(batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss, diagnostics = model.idea_loss(first, second)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"{idea_id}/{dataset_name}/epoch={epoch} 出现非有限损失")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += float(loss.detach()) * batch.num_graphs
                    total_graphs += batch.num_graphs
                    for key in diagnostics_sum:
                        diagnostics_sum[key] += diagnostics[key] * batch.num_graphs
                record = {
                    "epoch": epoch,
                    "loss": total_loss / max(total_graphs, 1),
                    **{key: value / max(total_graphs, 1) for key, value in diagnostics_sum.items()},
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                progress.set_postfix(loss=f"{record['loss']:.4f}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(), "idea_id": idea_id,
            "dataset": dataset_name, "epochs": args.epochs,
            "fidelity": "differentiable_prototype",
            "revision": get_idea_spec(idea_id).revision,
        }, checkpoint)

    embeddings, labels = extract_embeddings(model, graphs, batch_size, device)
    metrics = linear_probe_five_fold_metrics(
        embeddings, labels, device=device, seed=args.seed,
        folds=args.folds, epochs=args.probe_epochs,
    )
    return {
        "idea_id": idea_id,
        "family": get_idea_spec(idea_id).family_name,
        "revision": get_idea_spec(idea_id).revision,
        "dataset": dataset_name,
        "task": "graph_classification",
        "training": "self_supervised_pretrain_then_stratified_linear_probe",
        "folds": args.folds,
        "pretrain_epochs": args.epochs,
        "fidelity": "differentiable_prototype",
        "accuracy_percent": summarize(metrics["accuracy"]),
        "nmi_percent": summarize(metrics["nmi"]),
        "ari_percent": summarize(metrics["ari"]),
        "macro_f1_percent": summarize(metrics["macro_f1"]),
    }


def merge_results(output_dir: Path) -> Path:
    experiments = []
    for path in tqdm(sorted((output_dir / "jobs").glob("*.json")), desc="汇总研究实验", unit="任务"):
        experiments.append(json.loads(path.read_text(encoding="utf-8")))
    result = {
        "metadata": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "idea_count": len({item["idea_id"] for item in experiments}),
            "dataset_count": len({item["dataset"] for item in experiments}),
            "experiment_count": len(experiments),
            "metrics": list(METRIC_KEYS),
            "fidelity": "differentiable_prototype",
        },
        "experiments": sorted(experiments, key=lambda item: (item["idea_id"], item["dataset"])),
    }
    path = output_dir / "five_fold_results.json"
    save_json(path, result)
    return path


def main() -> None:
    args = parse_args()
    data_root, output_dir = _paths(args)
    if args.merge_only:
        print(f"已汇总：{merge_results(output_dir)}")
        return
    device = choose_device(args.device)
    print(f"使用设备：{device}")
    if args.smoke_test:
        print(f"全部冒烟通过：{smoke_test(args, data_root, output_dir, device)}")
        return

    combinations = [(idea_id, dataset) for idea_id in args.ideas for dataset in args.datasets]
    for idea_id, dataset_name in tqdm(combinations, desc="研究 idea 实验总进度", unit="任务"):
        job_path = output_dir / "jobs" / f"{idea_id}_{dataset_name}.json"
        if job_path.exists() and not args.force:
            continue
        graphs, in_dim, num_classes = load_graphs(data_root, dataset_name)
        result = run_experiment(
            args, idea_id, dataset_name, graphs, in_dim, num_classes,
            output_dir, device,
        )
        save_json(job_path, result)
        print(
            f"{idea_id}/{dataset_name}: Acc={result['accuracy_percent']['mean']:.2f}, "
            f"NMI={result['nmi_percent']['mean']:.2f}, "
            f"ARI={result['ari_percent']['mean']:.2f}, "
            f"F1={result['macro_f1_percent']['mean']:.2f}"
        )
    if not args.no_merge:
        print(f"已汇总：{merge_results(output_dir)}")


if __name__ == "__main__":
    main()
