#!/usr/bin/env python3
"""汇总六模型十随机种子结果，并计算 95% 置信区间与配对差值。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("ADHD200", "BACE", "BBBP", "Tox21", "AIDS", "BZR")
METHODS = ("retogcl", "pl069", "balancegcl", "khangcl", "del", "simplicial_mp")
METRICS = ("accuracy_percent", "nmi_percent", "ari_percent", "macro_f1_percent")
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def path_for(root: Path, method: str, dataset: str, seed: int) -> Path:
    base = root / f"seed_{seed:03d}"
    if method == "retogcl":
        return base / "retogcl/jobs/D7-I10__D7-I01__D10-I04__{}.json".format(dataset)
    if method == "pl069":
        return base / f"pl069/jobs/PL069__{dataset}.json"
    if method in {"balancegcl", "khangcl"}:
        return base / f"sota/jobs/{method}_{dataset}.json"
    return base / f"paper_sota/jobs/{method}_{dataset}.json"


def interval(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    count = len(array)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    critical = T_CRITICAL_95.get(count, 1.96)
    margin = critical * std / math.sqrt(count) if count > 1 else 0.0
    return {
        "n": count, "values": [round(float(value), 4) for value in array],
        "mean": round(mean, 4), "std": round(std, 4),
        "ci95_low": round(mean - margin, 4), "ci95_high": round(mean + margin, 4),
        "ci_method": "two_sided_Student_t",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/multiseed_validation_6x6x10_20260904"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()
    root = args.output_root if args.output_root.is_absolute() else (PROJECT_ROOT / args.output_root).resolve()
    records = {}
    missing = []
    for method in tqdm(METHODS, desc="汇总多种子", unit="模型"):
        for dataset in DATASETS:
            items = {}
            for seed in args.seeds:
                path = path_for(root, method, dataset, seed)
                if not path.exists():
                    missing.append({"method": method, "dataset": dataset, "seed": seed, "path": str(path)})
                    continue
                value = json.loads(path.read_text(encoding="utf-8"))
                if all(metric in value for metric in METRICS):
                    items[seed] = value
                else:
                    missing.append({"method": method, "dataset": dataset, "seed": seed, "path": str(path), "reason": "指标不完整"})
            records[(method, dataset)] = items

    aggregate = {}
    for (method, dataset), items in records.items():
        aggregate[f"{method}/{dataset}"] = {
            "method": method, "dataset": dataset, "seeds": sorted(items),
            **{
                metric: interval([items[seed][metric]["mean"] for seed in sorted(items)])
                for metric in METRICS if items
            },
        }

    paired = []
    for dataset in DATASETS:
        for first_index, first in enumerate(METHODS):
            for second in METHODS[first_index + 1:]:
                first_items, second_items = records[(first, dataset)], records[(second, dataset)]
                shared = sorted(set(first_items) & set(second_items))
                item = {"dataset": dataset, "first": first, "second": second, "shared_seeds": shared}
                if shared:
                    for metric in METRICS:
                        item[metric] = interval([
                            first_items[seed][metric]["mean"] - second_items[seed][metric]["mean"]
                            for seed in shared
                        ])
                paired.append(item)

    result = {
        "metadata": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": "10_pretraining_or_training_seeds_fixed_stratified_5_fold",
            "requested_seeds": args.seeds, "split_seed": 42,
            "confidence_interval": "two-sided Student t over seed-level five-fold means",
            "paired_difference": "first minus second, paired by identical seed and outer-fold indices",
        },
        "aggregate": aggregate, "paired_differences": paired, "missing": missing,
    }
    path = root / "multiseed_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已汇总 {sum(len(items) for items in records.values())}/360 个种子任务；缺失 {len(missing)}：{path}")


if __name__ == "__main__":
    main()
