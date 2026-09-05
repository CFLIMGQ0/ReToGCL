#!/usr/bin/env python3
"""完整审计并汇总 T077 的 100 个自适应微调版本。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.prepare_retogcl_t077_micro_stage import (
    PROTOCOL,
    rank_key,
    score_configuration,
)
from src.scripts.summarize_retogcl_story_pipelines import (
    DATASETS,
    METRICS,
    METRIC_LABELS,
    means,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retogcl_t077_micro_100x6"),
    )
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_result(path: Path, config_id: str, dataset: str) -> dict:
    result = load_json(path)
    if result.get("status") != "completed":
        raise ValueError(f"{path}: status 不是 completed")
    if result.get("protocol") != PROTOCOL:
        raise ValueError(f"{path}: protocol 不兼容")
    if result.get("config_id") != config_id or result.get("dataset") != dataset:
        raise ValueError(f"{path}: 配置或数据集字段不匹配")
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


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    manifest = load_json(output_dir / "manifest.json")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("manifest 协议不兼容")
    configs = manifest.get("configs", [])
    if len(configs) != 100:
        raise ValueError(f"配置不是 100 个，实际 {len(configs)}")
    if len(manifest.get("stages", [])) != 5:
        raise ValueError("阶段不是 5 个")

    for entry in configs:
        for dataset in DATASETS:
            validate_result(
                output_dir / "jobs" / f"{entry['config_id']}__{dataset}.json",
                entry["config_id"],
                dataset,
            )
    checkpoints = list((output_dir / "checkpoints").glob("*/*.pt"))
    logs = list((output_dir / "logs").glob("*/*.jsonl"))
    if len(checkpoints) != 600 or len(logs) != 600:
        raise ValueError(
            f"产物不完整：checkpoint={len(checkpoints)}, log={len(logs)}"
        )

    ranking = [
        {
            **entry,
            "score": score_configuration(output_dir, entry["config_id"]),
        }
        for entry in configs
    ]
    ranking.sort(key=rank_key, reverse=True)
    print(
        "AUDIT\tconfigs=100\tdatasets=6\tresults=600\tcheckpoints=600\t"
        "logs=600\tpretrain_epochs=40\tfolds=5\tfailures=0"
    )
    print("RANKING")
    print(
        "rank\tconfig\tstage\tmode\tstrength\tchanged\t"
        "T077_W-T-L/24\tall4/6\tmin_dataset_wins\tworst_dataset_mean\t"
        "worst_metric\tmean_delta\tstrict_sota/78\tmetric_sota/312"
    )
    for rank, item in enumerate(ranking[: max(1, args.top)], 1):
        score = item["score"]
        configuration = item["configuration"]
        print(
            f"{rank}\t{item['config_id']}\t{item['stage']}\t"
            f"{configuration['micro_mode']}\t{configuration['micro_strength']}\t"
            f"{','.join(item['changed_parameters'])}\t"
            f"{score['wins_vs_t077']}-{score['ties_vs_t077']}-"
            f"{score['losses_vs_t077']}\t"
            f"{score['all4_datasets_vs_t077']}\t"
            f"{score['minimum_metric_wins_per_dataset']}\t"
            f"{score['worst_dataset_mean_delta']:+.4f}\t"
            f"{score['worst_delta_vs_t077']:+.4f}\t"
            f"{score['mean_delta_vs_t077']:+.4f}\t"
            f"{score['strict_sota_wins']}\t{score['metric_sota_wins']}"
        )

    best = ranking[0]
    print("BEST_CONFIGURATION")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("BEST_DATASET_DETAIL")
    print(
        "dataset\tmetric\tT077_mean\tT077_std\tbest_mean\tbest_std\tdelta"
    )
    reference_root = (
        PROJECT_ROOT / "outputs" / "retogcl_o07_dynamic_100x6" / "jobs"
    )
    for dataset in DATASETS:
        result = load_json(
            output_dir / "jobs" / f"{best['config_id']}__{dataset}.json"
        )
        reference = load_json(reference_root / f"T077__{dataset}.json")
        for metric, label in zip(METRICS, METRIC_LABELS, strict=True):
            candidate_mean = float(result[metric]["mean"])
            candidate_std = float(result[metric]["std"])
            reference_mean = float(reference[metric]["mean"])
            reference_std = float(reference[metric]["std"])
            if not all(math.isfinite(value) for value in (
                candidate_mean, candidate_std, reference_mean, reference_std
            )):
                raise ValueError("发现非有限结果")
            print(
                f"{dataset}\t{label}\t{reference_mean:.4f}\t"
                f"{reference_std:.4f}\t{candidate_mean:.4f}\t"
                f"{candidate_std:.4f}\t{candidate_mean-reference_mean:+.4f}"
            )


if __name__ == "__main__":
    main()
