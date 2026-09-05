#!/usr/bin/env python3
"""汇总 ReToGCL 故事 Pipeline 的正式五折实验结果。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PIPELINES = tuple(f"S{index:02d}" for index in range(1, 9))
DATASETS = ("ADHD200", "BACE", "BBBP", "Tox21", "AIDS", "BZR")
METRICS = (
    "accuracy_percent",
    "nmi_percent",
    "ari_percent",
    "macro_f1_percent",
)
METRIC_LABELS = ("Accuracy", "NMI", "ARI", "Macro-F1")
SOTAS = {
    "BalanceGCL": ("sota", "balancegcl"),
    "Khan-GCL": ("sota", "khangcl"),
    "CellCLAT": ("paper_sota", "cellclat"),
    "UniImb": ("sota", "uniimb"),
    "DualPrism": ("paper_sota", "dualprism"),
    "DEL": ("paper_sota", "del"),
    "SpectRe": ("paper_sota_2025_2026", "spectre"),
    "TopER": ("paper_sota_2025_2026", "toper"),
    "LEAP": ("paper_sota_2025_2026", "leap"),
    "Hourglass": ("paper_sota_2025_2026", "hourglass"),
    "NodeID": ("paper_sota_2025_2026", "nodeid"),
    "GNN+": ("paper_sota_2025_2026", "gnnplus"),
    "RS-Pool": ("paper_sota_2025_2026", "rspool"),
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def means(result: dict) -> tuple[float, ...]:
    values = tuple(float(result[key]["mean"]) for key in METRICS)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("发现非有限指标")
    return values


def validate_story_result(path: Path, pipeline: str, dataset: str) -> dict:
    result = load_json(path)
    if result.get("status") != "completed":
        raise ValueError(f"{path}: status 不是 completed")
    if result.get("pipeline") != pipeline or result.get("dataset") != dataset:
        raise ValueError(f"{path}: Pipeline 或数据集字段不匹配")
    means(result)
    if dataset == "Tox21":
        tasks = result.get("per_task_results", [])
        if len(tasks) != 12:
            raise ValueError(f"{path}: Tox21 不是 12 个官方任务")
        for task in tasks:
            for metric in METRICS:
                if len(task[metric].get("folds", [])) != 5:
                    raise ValueError(f"{path}: Tox21 子任务不是五折")
    else:
        for metric in METRICS:
            if len(result[metric].get("folds", [])) != 5:
                raise ValueError(f"{path}: {metric} 不是五折")
    return result


def benchmark_root(project_root: Path, dataset: str) -> Path:
    suffix = "medical_benchmark_20260828" if dataset in {"BACE", "BBBP"} else "medical_benchmark_extended_20260828"
    return project_root / "outputs" / suffix


def base_path(project_root: Path, dataset: str) -> Path:
    return benchmark_root(project_root, dataset) / "our_model" / "jobs" / f"D7-I10__D7-I01__D10-I04__{dataset}.json"


def sota_path(project_root: Path, dataset: str, group: str, stem: str) -> Path:
    return benchmark_root(project_root, dataset) / group / "jobs" / f"{stem}_{dataset}.json"


def comparison(candidate: tuple[float, ...], reference: tuple[float, ...]) -> tuple[int, int, int]:
    wins = sum(left > right for left, right in zip(candidate, reference))
    ties = sum(math.isclose(left, right, abs_tol=1e-10) for left, right in zip(candidate, reference))
    return wins, ties, len(METRICS) - wins - ties


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    story_root = project_root / "outputs" / "retogcl_story_pipelines_8x6"

    stories: dict[str, dict[str, dict]] = {}
    for pipeline in PIPELINES:
        stories[pipeline] = {}
        for dataset in DATASETS:
            path = story_root / "jobs" / f"{pipeline}__{dataset}.json"
            stories[pipeline][dataset] = validate_story_result(path, pipeline, dataset)

    checkpoint_count = len(list((story_root / "checkpoints").glob("*/*.pt")))
    log_count = len(list((story_root / "logs").glob("*/*.jsonl")))
    if checkpoint_count != 48 or log_count != 48:
        raise ValueError(f"产物不完整：checkpoint={checkpoint_count}, log={log_count}")

    bases = {dataset: load_json(base_path(project_root, dataset)) for dataset in DATASETS}
    p05s = {
        dataset: load_json(
            project_root / "outputs" / "retogcl_pipeline_strategies_10x18" / "jobs" / f"P05__{dataset}.json"
        )
        for dataset in DATASETS
    }
    sotas = {
        dataset: {
            name: load_json(sota_path(project_root, dataset, group, stem))
            for name, (group, stem) in SOTAS.items()
        }
        for dataset in DATASETS
    }

    print("AUDIT\tresults=48\tcheckpoints=48\tlogs=48\tfailures=0")
    print("RANKING")
    print("pipeline\tname\tsota_strict/78\tsota_metrics/312\tbase_W-T-L/24\tbase_all4_datasets\tp05_W-T-L/24\tp05_all4_datasets\tavg_delta_base\tavg_delta_p05")
    ranking: list[dict] = []
    for pipeline in PIPELINES:
        first = stories[pipeline][DATASETS[0]]
        row = {
            "pipeline": pipeline,
            "name": first["pipeline_name"],
            "sota_strict": 0,
            "sota_metric": 0,
            "base": [0, 0, 0],
            "p05": [0, 0, 0],
            "base_all4": [],
            "p05_all4": [],
            "delta_base": [],
            "delta_p05": [],
        }
        for dataset in DATASETS:
            candidate = means(stories[pipeline][dataset])
            base = means(bases[dataset])
            p05 = means(p05s[dataset])
            base_cmp = comparison(candidate, base)
            p05_cmp = comparison(candidate, p05)
            row["base"] = [left + right for left, right in zip(row["base"], base_cmp)]
            row["p05"] = [left + right for left, right in zip(row["p05"], p05_cmp)]
            if base_cmp[0] == 4:
                row["base_all4"].append(dataset)
            if p05_cmp[0] == 4:
                row["p05_all4"].append(dataset)
            row["delta_base"].extend(left - right for left, right in zip(candidate, base))
            row["delta_p05"].extend(left - right for left, right in zip(candidate, p05))
            for sota in sotas[dataset].values():
                sota_cmp = comparison(candidate, means(sota))
                row["sota_metric"] += sota_cmp[0]
                row["sota_strict"] += int(sota_cmp[0] == 4)
        row["avg_delta_base"] = sum(row["delta_base"]) / len(row["delta_base"])
        row["avg_delta_p05"] = sum(row["delta_p05"]) / len(row["delta_p05"])
        ranking.append(row)

    ranking.sort(key=lambda row: (row["sota_strict"], row["sota_metric"], row["avg_delta_base"]), reverse=True)
    for row in ranking:
        print(
            f'{row["pipeline"]}\t{row["name"]}\t{row["sota_strict"]}\t{row["sota_metric"]}\t'
            f'{"-".join(map(str, row["base"]))}\t{",".join(row["base_all4"]) or "-"}\t'
            f'{"-".join(map(str, row["p05"]))}\t{",".join(row["p05_all4"]) or "-"}\t'
            f'{row["avg_delta_base"]:+.4f}\t{row["avg_delta_p05"]:+.4f}'
        )

    print("DATASET_DETAIL")
    print("pipeline\tdataset\taccuracy\tnmi\tari\tmacro_f1\tstrict_sota/13\tmetric_sota/52\tstrict_sota_names")
    for row in ranking:
        pipeline = row["pipeline"]
        for dataset in DATASETS:
            candidate = means(stories[pipeline][dataset])
            strict_names: list[str] = []
            metric_wins = 0
            for name, sota in sotas[dataset].items():
                sota_cmp = comparison(candidate, means(sota))
                metric_wins += sota_cmp[0]
                if sota_cmp[0] == 4:
                    strict_names.append(name)
            values = "\t".join(f"{value:.4f}" for value in candidate)
            print(
                f'{pipeline}\t{dataset}\t{values}\t{len(strict_names)}\t{metric_wins}\t'
                f'{",".join(strict_names) or "-"}'
            )

    print("REFERENCE")
    for label, results in (("BASE", bases), ("P05", p05s)):
        strict_total = 0
        metric_total = 0
        for dataset in DATASETS:
            candidate = means(results[dataset])
            for sota in sotas[dataset].values():
                sota_cmp = comparison(candidate, means(sota))
                strict_total += int(sota_cmp[0] == 4)
                metric_total += sota_cmp[0]
        print(f"{label}\tstrict={strict_total}/78\tmetrics={metric_total}/312")


if __name__ == "__main__":
    main()
