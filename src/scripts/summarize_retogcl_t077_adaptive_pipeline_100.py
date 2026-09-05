#!/usr/bin/env python3
"""审计并汇总 T077 的 100 个自适应 Pipeline 实验。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.prepare_retogcl_t077_adaptive_pipeline_stage import (
    PROTOCOL,
    REFERENCE_ROOT,
    rank_key,
    score_configuration,
)
from src.scripts.summarize_retogcl_story_pipelines import (
    DATASETS,
    METRICS,
    METRIC_LABELS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retogcl_t077_adaptive_pipeline_100x6"),
    )
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_result(path: Path, config_id: str, dataset: str) -> dict:
    result = load_json(path)
    if result.get("status") != "completed" or result.get("protocol") != PROTOCOL:
        raise ValueError(f"结果不完整或协议错误：{path}")
    if result.get("config_id") != config_id or result.get("dataset") != dataset:
        raise ValueError(f"配置/数据集字段错误：{path}")
    if dataset == "Tox21":
        tasks = result.get("per_task_results", [])
        if len(tasks) != 12:
            raise ValueError(f"Tox21 不是 12 个官方任务：{path}")
        for task in tasks:
            for metric in METRICS:
                if len(task[metric].get("folds", [])) != 5:
                    raise ValueError(f"Tox21 子任务不是五折：{path}/{metric}")
    for metric in METRICS:
        value = result[metric]
        if dataset != "Tox21" and len(value.get("folds", [])) != 5:
            raise ValueError(f"不是五折结果：{path}/{metric}")
        if not math.isfinite(float(value["mean"])):
            raise ValueError(f"发现非有限指标：{path}/{metric}")
    evidence = result.get("training_evidence", {})
    if not evidence or not all(
        math.isfinite(float(evidence[key]))
        for key in (
            "final_loss", "maximum_gradient_norm",
            "final_pipeline_gate_mean", "final_pipeline_relative_correction",
        )
    ):
        raise ValueError(f"训练证据不完整：{path}")
    return result


def validate_training_artifacts(
    output_dir: Path, config_id: str, dataset: str
) -> None:
    log_path = output_dir / "logs" / config_id / f"{dataset}.jsonl"
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 40 or [record["epoch"] for record in records] != list(range(1, 41)):
        raise ValueError(f"训练日志不是连续 40 轮：{log_path}")
    for record in records:
        for key in (
            "loss", "gradient_norm", "pipeline_gate_mean",
            "pipeline_relative_correction",
        ):
            if not math.isfinite(float(record[key])):
                raise ValueError(f"训练日志含非有限值：{log_path}/{key}")
    checkpoint_path = output_dir / "checkpoints" / config_id / f"{dataset}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("protocol") != PROTOCOL
        or checkpoint.get("config_id") != config_id
        or checkpoint.get("dataset") != dataset
        or checkpoint.get("epochs") != 40
    ):
        raise ValueError(f"checkpoint 元数据错误：{checkpoint_path}")
    for name, value in checkpoint["model"].items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise ValueError(f"checkpoint 含非有限参数：{checkpoint_path}/{name}")


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    manifest = load_json(output_dir / "manifest.json")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("manifest 协议不兼容")
    configs = manifest.get("configs", [])
    if len(configs) != 100 or len(manifest.get("stages", [])) != 10:
        raise ValueError(
            f"应为 100 配置/10 阶段，实际 {len(configs)}/{len(manifest.get('stages', []))}"
        )
    progress = tqdm(total=600, desc="审计训练产物", unit="任务")
    for entry in configs:
        for dataset in DATASETS:
            validate_result(
                output_dir / "jobs" / f"{entry['config_id']}__{dataset}.json",
                entry["config_id"],
                dataset,
            )
            validate_training_artifacts(
                output_dir, entry["config_id"], dataset
            )
            progress.update(1)
    progress.close()
    checkpoints = list((output_dir / "checkpoints").glob("*/*.pt"))
    logs = list((output_dir / "logs").glob("*/*.jsonl"))
    if len(checkpoints) != 600 or len(logs) != 600:
        raise ValueError(
            f"产物不完整：checkpoint={len(checkpoints)}, log={len(logs)}"
        )

    ranking = [
        {**entry, "score": score_configuration(output_dir, entry["config_id"])}
        for entry in configs
    ]
    ranking.sort(key=rank_key, reverse=True)
    report = {
        "audit": {
            "configs": 100,
            "datasets": 6,
            "results": 600,
            "checkpoints": 600,
            "logs": 600,
            "pretrain_epochs": 40,
            "folds": 5,
            "failures": 0,
        },
        "best": ranking[0],
        "ranking": ranking,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "审计通过\t配置=100\t数据集=6\t结果=600\tcheckpoint=600\t"
        "日志=600\t预训练=40轮\t五折\t失败=0"
    )
    print(
        "排名\t配置\t阶段\t主连接\t次连接\tT077胜-平-负/24\t"
        "全胜数据集\t最弱数据集胜项\t最弱均值变化\t平均变化\t严格SOTA\t指标SOTA\t"
        "门均值\t修正量\t最大梯度"
    )
    for index, item in enumerate(ranking[:max(1, args.top)], 1):
        config = item["configuration"]
        score = item["score"]
        print(
            f"{index}\t{item['config_id']}\t{item['stage']}\t"
            f"{config['primary_action']}\t{config['secondary_action']}\t"
            f"{score['wins_vs_t077']}-{score['ties_vs_t077']}-{score['losses_vs_t077']}\t"
            f"{score['all4_datasets_vs_t077']}\t"
            f"{score['minimum_metric_wins_per_dataset']}\t"
            f"{score['worst_dataset_mean_delta']:+.4f}\t"
            f"{score['mean_delta_vs_t077']:+.4f}\t"
            f"{score['strict_sota_wins']}\t{score['metric_sota_wins']}\t"
            f"{score['mean_pipeline_gate']:.4f}\t"
            f"{score['mean_pipeline_correction']:.4f}\t"
            f"{score['maximum_gradient_norm']:.3f}"
        )

    best = ranking[0]
    print("最佳配置")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("最佳配置逐数据集指标")
    print("数据集\t指标\tT077均值\tT077标准差\t候选均值\t候选标准差\t变化")
    for dataset in DATASETS:
        candidate = load_json(
            output_dir / "jobs" / f"{best['config_id']}__{dataset}.json"
        )
        reference = load_json(REFERENCE_ROOT / f"T077__{dataset}.json")
        for metric, label in zip(METRICS, METRIC_LABELS, strict=True):
            left = candidate[metric]
            right = reference[metric]
            print(
                f"{dataset}\t{label}\t{float(right['mean']):.4f}\t"
                f"{float(right['std']):.4f}\t{float(left['mean']):.4f}\t"
                f"{float(left['std']):.4f}\t"
                f"{float(left['mean'])-float(right['mean']):+.4f}"
            )


if __name__ == "__main__":
    main()
