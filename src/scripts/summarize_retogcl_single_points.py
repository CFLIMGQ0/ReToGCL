#!/usr/bin/env python3
"""审计并汇总 ReToGCL 十个单点优化的六数据集结果。"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.retogcl_single_point_optimizations import OPTIMIZATIONS
from src.scripts.summarize_retogcl_story_pipelines import (
    DATASETS, METRICS, SOTAS, base_path, benchmark_root, comparison,
    load_json, means, sota_path,
)


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "retogcl_single_point_10x6"


def validate_result(path: Path, optimization: str, dataset: str) -> dict:
    result = load_json(path)
    if result.get("status") != "completed":
        raise ValueError(f"{path}: status 不是 completed")
    if (
        result.get("protocol") != "retogcl_single_point_optimization_v1"
        or result.get("optimization") != optimization
        or result.get("dataset") != dataset
    ):
        raise ValueError(f"{path}: 协议或任务字段不一致")
    means(result)
    if dataset == "Tox21":
        tasks = result.get("per_task_results", [])
        if len(tasks) != 12:
            raise ValueError(f"{path}: Tox21 不是 12 个官方任务")
        if any(
            len(task[metric].get("folds", [])) != 5
            for task in tasks for metric in METRICS
        ):
            raise ValueError(f"{path}: Tox21 子任务不是五折")
    elif any(len(result[metric].get("folds", [])) != 5 for metric in METRICS):
        raise ValueError(f"{path}: 不是完整五折")
    return result


def main() -> None:
    results = {
        optimization: {
            dataset: validate_result(
                OUTPUT_ROOT / "jobs" / f"{optimization}__{dataset}.json",
                optimization, dataset,
            )
            for dataset in DATASETS
        }
        for optimization in OPTIMIZATIONS
    }
    checkpoints = len(list((OUTPUT_ROOT / "checkpoints").glob("*/*.pt")))
    logs = len(list((OUTPUT_ROOT / "logs").glob("*/*.jsonl")))
    if checkpoints != 60 or logs != 60:
        raise ValueError(f"产物不完整：checkpoint={checkpoints}, log={logs}")
    if any(sum(1 for _ in path.open(encoding="utf-8")) != 40 for path in (OUTPUT_ROOT / "logs").glob("*/*.jsonl")):
        raise ValueError("存在训练日志不足 40 轮")

    base = {dataset: load_json(base_path(PROJECT_ROOT, dataset)) for dataset in DATASETS}
    s03 = {
        dataset: load_json(
            PROJECT_ROOT / "outputs" / "retogcl_story_pipelines_8x6"
            / "jobs" / f"S03__{dataset}.json"
        )
        for dataset in DATASETS
    }
    sotas = {
        dataset: {
            name: load_json(sota_path(PROJECT_ROOT, dataset, group, stem))
            for name, (group, stem) in SOTAS.items()
        }
        for dataset in DATASETS
    }

    rows = []
    for optimization, spec in OPTIMIZATIONS.items():
        row = {
            "id": optimization, "name": spec.name,
            "base": [0, 0, 0], "s03": [0, 0, 0],
            "base_all4": [], "sota_strict": 0, "sota_metrics": 0,
            "deltas": [], "worst_delta": math.inf,
        }
        for dataset in DATASETS:
            candidate = means(results[optimization][dataset])
            baseline = means(base[dataset])
            story = means(s03[dataset])
            base_cmp = comparison(candidate, baseline)
            story_cmp = comparison(candidate, story)
            row["base"] = [a + b for a, b in zip(row["base"], base_cmp)]
            row["s03"] = [a + b for a, b in zip(row["s03"], story_cmp)]
            if base_cmp[0] == 4:
                row["base_all4"].append(dataset)
            delta = [a - b for a, b in zip(candidate, baseline)]
            row["deltas"].extend(delta)
            row["worst_delta"] = min(row["worst_delta"], *delta)
            for sota in sotas[dataset].values():
                cmp = comparison(candidate, means(sota))
                row["sota_metrics"] += cmp[0]
                row["sota_strict"] += int(cmp[0] == 4)
        row["avg_delta"] = sum(row["deltas"]) / len(row["deltas"])
        rows.append(row)

    rows.sort(
        key=lambda row: (
            len(row["base_all4"]), row["base"][0], row["sota_strict"],
            row["sota_metrics"], row["avg_delta"],
        ), reverse=True,
    )
    print("AUDIT\tresults=60\tcheckpoints=60\tlogs=60\tepochs=40\tfailures=0")
    print("RANKING")
    print("id\tname\tbase_W-T-L/24\tbase_all4\tavg_delta\tworst_delta\ts03_W-T-L/24\tsota_strict/78\tsota_metrics/312")
    for row in rows:
        print(
            f'{row["id"]}\t{row["name"]}\t{"-".join(map(str,row["base"]))}\t'
            f'{",".join(row["base_all4"]) or "-"}\t{row["avg_delta"]:+.4f}\t'
            f'{row["worst_delta"]:+.4f}\t{"-".join(map(str,row["s03"]))}\t'
            f'{row["sota_strict"]}\t{row["sota_metrics"]}'
        )

    print("DETAIL")
    print("id\tdataset\taccuracy\tnmi\tari\tmacro_f1\tbase_wins\tdeltas")
    for row in rows:
        optimization = row["id"]
        for dataset in DATASETS:
            candidate = means(results[optimization][dataset])
            baseline = means(base[dataset])
            cmp = comparison(candidate, baseline)
            values = "\t".join(f"{value:.4f}" for value in candidate)
            deltas = ",".join(f"{a-b:+.4f}" for a, b in zip(candidate, baseline))
            print(
                f"{optimization}\t{dataset}\t{values}\t{cmp[0]}\t{deltas}"
            )


if __name__ == "__main__":
    main()
