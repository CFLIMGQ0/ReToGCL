#!/usr/bin/env python3
"""固化六个正式数据集的分层五折索引，供多随机种子配对复核。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.evaluation import stratified_folds
from src.scripts.graph_classification_datasets import MULTITASK_DATASETS, load_graphs


DATASETS = ("ADHD200", "BACE", "BBBP", "Tox21", "AIDS", "BZR")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def fold_record(labels: torch.Tensor, global_indices: torch.Tensor, split_seed: int) -> dict:
    local_folds = stratified_folds(labels.long(), 5, split_seed)
    folds = [global_indices[indices].tolist() for indices in local_folds]
    return {
        "sample_count": int(labels.numel()),
        "class_counts": {str(int(label)): int((labels == label).sum()) for label in labels.unique(sorted=True)},
        "fold_indices": folds,
        "fold_sizes": [len(fold) for fold in folds],
        "indices_sha256": digest(folds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/multiseed_validation_6x6x10_20260904/fixed_five_fold_indices.json"))
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    data_root = args.data_root if args.data_root.is_absolute() else (PROJECT_ROOT / args.data_root).resolve()
    output = args.output if args.output.is_absolute() else (PROJECT_ROOT / args.output).resolve()
    records = {}
    for dataset in tqdm(DATASETS, desc="固化五折索引", unit="数据集"):
        graphs, _, _ = load_graphs(data_root, dataset)
        global_indices = torch.arange(len(graphs))
        if dataset in MULTITASK_DATASETS:
            task_y = torch.cat([graph.task_y for graph in graphs], dim=0)
            tasks = []
            for task_index in range(task_y.size(1)):
                mask = torch.isfinite(task_y[:, task_index])
                tasks.append({
                    "task_index": task_index,
                    **fold_record(task_y[mask, task_index], global_indices[mask], args.split_seed),
                })
            records[dataset] = {"task_count": len(tasks), "tasks": tasks}
        else:
            labels = torch.tensor([int(graph.y.item()) for graph in graphs])
            records[dataset] = fold_record(labels, global_indices, args.split_seed)
    value = {
        "metadata": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "split_seed": args.split_seed, "folds": 5,
            "protocol": "stratified_5_fold_shared_by_all_models_and_pretraining_seeds",
        },
        "datasets": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"固定五折索引已写入：{output}")


if __name__ == "__main__":
    main()
