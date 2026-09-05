#!/usr/bin/env python3
"""审计并汇总 O07 的 100 个动态调参版本。"""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.prepare_retogcl_o07_dynamic_stage import (
    PROTOCOL,
    rank_key,
    score_configuration,
)
from src.scripts.summarize_retogcl_story_pipelines import DATASETS, METRICS, means


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "retogcl_o07_dynamic_100x6"


def main() -> None:
    manifest = json.loads(
        (OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    configs = manifest.get("configs", [])
    if len(configs) != 100 or len({entry["config_id"] for entry in configs}) != 100:
        raise ValueError("manifest 不是 100 个唯一配置")
    if any(len(entry["changed_parameters"]) > 2 for entry in configs):
        raise ValueError("存在超过两个改动参数的配置")

    results = list((OUTPUT_ROOT / "jobs").glob("*.json"))
    checkpoints = list((OUTPUT_ROOT / "checkpoints").glob("*/*.pt"))
    logs = list((OUTPUT_ROOT / "logs").glob("*/*.jsonl"))
    if not (len(results) == len(checkpoints) == len(logs) == 600):
        raise ValueError(
            f"产物不完整：results={len(results)}, "
            f"checkpoints={len(checkpoints)}, logs={len(logs)}"
        )
    if any(sum(1 for _ in path.open(encoding="utf-8")) != 40 for path in logs):
        raise ValueError("存在训练日志不足 40 轮")

    for entry in configs:
        for dataset in DATASETS:
            path = OUTPUT_ROOT / "jobs" / f"{entry['config_id']}__{dataset}.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            if (
                result.get("status") != "completed"
                or result.get("protocol") != PROTOCOL
                or result.get("config_id") != entry["config_id"]
                or result.get("dataset") != dataset
            ):
                raise ValueError(f"结果字段不一致：{path}")
            means(result)
            if dataset == "Tox21":
                tasks = result.get("per_task_results", [])
                if len(tasks) != 12 or any(
                    len(task[metric].get("folds", [])) != 5
                    for task in tasks for metric in METRICS
                ):
                    raise ValueError(f"Tox21 子任务不完整：{path}")
            elif any(len(result[metric].get("folds", [])) != 5 for metric in METRICS):
                raise ValueError(f"五折不完整：{path}")

    ranking = [
        {**entry, "score": score_configuration(OUTPUT_ROOT, entry["config_id"])}
        for entry in configs
    ]
    ranking.sort(key=rank_key, reverse=True)
    print(
        "AUDIT\tconfigs=100\tdatasets=6\tresults=600\tcheckpoints=600\t"
        "logs=600\tepochs=40\tfolds=5\tfailures=0"
    )
    print("RANKING")
    print(
        "rank\tconfig_id\tstage\ttemperature\thard_gain\taux_scale\t"
        "weight_decay\tstrict_sota/78\tmetric_sota/312\t"
        "p05_W-T-L/24\tp05_all4\tmean_delta_p05\tworst_delta_p05"
    )
    for index, entry in enumerate(ranking, 1):
        config = entry["configuration"]
        score = entry["score"]
        print(
            f"{index}\t{entry['config_id']}\t{entry['stage']}\t"
            f"{config['temperature']}\t{config['hard_alignment_gain']}\t"
            f"{config['auxiliary_scale']}\t{config['weight_decay']}\t"
            f"{score['strict_sota_wins']}\t{score['metric_sota_wins']}\t"
            f"{score['wins_vs_p05']}-{score['ties_vs_p05']}-"
            f"{score['losses_vs_p05']}\t{score['all4_datasets_vs_p05']}\t"
            f"{score['mean_delta_vs_p05']:+.4f}\t"
            f"{score['worst_delta_vs_p05']:+.4f}"
        )


if __name__ == "__main__":
    main()
