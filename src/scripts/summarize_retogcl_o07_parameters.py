#!/usr/bin/env python3
"""审计并汇总 O07 十个单因素参数版本的六数据集结果。"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.retogcl_o07_parameter_variants import O07_PARAMETER_VARIANTS
from src.scripts.summarize_retogcl_story_pipelines import (
    DATASETS,
    METRICS,
    SOTAS,
    base_path,
    comparison,
    load_json,
    means,
    sota_path,
)


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "retogcl_o07_parameter_10x6"
O07_ROOT = PROJECT_ROOT / "outputs" / "retogcl_single_point_10x6" / "jobs"
PROTOCOL = "retogcl_o07_parameter_variant_v1"


def validate_result(path: Path, variant: str, dataset: str) -> dict:
    result = load_json(path)
    if result.get("status") != "completed":
        raise ValueError(f"{path}: status 不是 completed")
    if (
        result.get("protocol") != PROTOCOL
        or result.get("variant") != variant
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
        variant: {
            dataset: validate_result(
                OUTPUT_ROOT / "jobs" / f"{variant}__{dataset}.json",
                variant,
                dataset,
            )
            for dataset in DATASETS
        }
        for variant in O07_PARAMETER_VARIANTS
    }
    checkpoints = len(list((OUTPUT_ROOT / "checkpoints").glob("*/*.pt")))
    logs = list((OUTPUT_ROOT / "logs").glob("*/*.jsonl"))
    if checkpoints != 60 or len(logs) != 60:
        raise ValueError(
            f"产物不完整：checkpoint={checkpoints}, log={len(logs)}"
        )
    if any(sum(1 for _ in path.open(encoding="utf-8")) != 40 for path in logs):
        raise ValueError("存在训练日志不足 40 轮")

    bases = {
        dataset: load_json(base_path(PROJECT_ROOT, dataset))
        for dataset in DATASETS
    }
    o07s = {
        dataset: load_json(O07_ROOT / f"O07__{dataset}.json")
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
    for variant, spec in O07_PARAMETER_VARIANTS.items():
        row = {
            "variant": variant,
            "name": spec.name,
            "o07": [0, 0, 0],
            "base": [0, 0, 0],
            "o07_all4": [],
            "base_all4": [],
            "delta_o07": [],
            "sota_strict": 0,
            "sota_metrics": 0,
        }
        for dataset in DATASETS:
            candidate = means(results[variant][dataset])
            o07 = means(o07s[dataset])
            base = means(bases[dataset])
            o07_cmp = comparison(candidate, o07)
            base_cmp = comparison(candidate, base)
            row["o07"] = [a + b for a, b in zip(row["o07"], o07_cmp)]
            row["base"] = [a + b for a, b in zip(row["base"], base_cmp)]
            if o07_cmp[0] == 4:
                row["o07_all4"].append(dataset)
            if base_cmp[0] == 4:
                row["base_all4"].append(dataset)
            row["delta_o07"].extend(
                left - right for left, right in zip(candidate, o07)
            )
            for sota in sotas[dataset].values():
                cmp = comparison(candidate, means(sota))
                row["sota_metrics"] += cmp[0]
                row["sota_strict"] += int(cmp[0] == 4)
        row["avg_delta_o07"] = sum(row["delta_o07"]) / len(row["delta_o07"])
        row["worst_delta_o07"] = min(row["delta_o07"])
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["o07"][0], len(row["o07_all4"]), row["avg_delta_o07"]
        ),
        reverse=True,
    )
    print("AUDIT\tresults=60\tcheckpoints=60\tlogs=60\tepochs=40\tfailures=0")
    print("RANKING")
    print(
        "variant\tname\to07_W-T-L/24\to07_all4\tavg_delta_o07\t"
        "worst_delta_o07\tbase_W-T-L/24\tbase_all4\t"
        "sota_strict/78\tsota_metrics/312"
    )
    for row in rows:
        print(
            f'{row["variant"]}\t{row["name"]}\t'
            f'{"-".join(map(str, row["o07"]))}\t'
            f'{",".join(row["o07_all4"]) or "-"}\t'
            f'{row["avg_delta_o07"]:+.4f}\t{row["worst_delta_o07"]:+.4f}\t'
            f'{"-".join(map(str, row["base"]))}\t'
            f'{",".join(row["base_all4"]) or "-"}\t'
            f'{row["sota_strict"]}\t{row["sota_metrics"]}'
        )

    print("DETAIL")
    print("variant\tdataset\taccuracy\tnmi\tari\tmacro_f1")
    for row in rows:
        variant = row["variant"]
        for dataset in DATASETS:
            values = "\t".join(
                f"{value:.4f}" for value in means(results[variant][dataset])
            )
            print(f"{variant}\t{dataset}\t{values}")


if __name__ == "__main__":
    main()
