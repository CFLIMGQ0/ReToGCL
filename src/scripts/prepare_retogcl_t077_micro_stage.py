#!/usr/bin/env python3
"""根据上一阶段结果，逐步生成 T077 的 100 个小步微调配置。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.retogcl_t077_micro_tuning import (
    HARD_MODES,
    PIPELINE_MODES,
    T077MicroConfiguration,
)
from src.scripts.summarize_retogcl_story_pipelines import (
    DATASETS,
    SOTAS,
    comparison,
    means,
    sota_path,
)


PROTOCOL = "retogcl_t077_micro_tuning_v1"
REFERENCE_ROOT = PROJECT_ROOT / "outputs" / "retogcl_o07_dynamic_100x6" / "jobs"
DEFAULT = T077MicroConfiguration()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=range(1, 6), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retogcl_t077_micro_100x6"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def configuration_dict(**changes: str | float) -> dict[str, str | float]:
    values: dict[str, str | float] = {
        "micro_mode": DEFAULT.micro_mode,
        "micro_strength": DEFAULT.micro_strength,
        "temperature": DEFAULT.temperature,
        "hard_alignment_gain": DEFAULT.hard_alignment_gain,
        "auxiliary_scale": DEFAULT.auxiliary_scale,
        "weight_decay": DEFAULT.weight_decay,
    }
    values.update(changes)
    parsed = T077MicroConfiguration(**values)
    return {
        "micro_mode": parsed.micro_mode,
        "micro_strength": parsed.micro_strength,
        "temperature": parsed.temperature,
        "hard_alignment_gain": parsed.hard_alignment_gain,
        "auxiliary_scale": parsed.auxiliary_scale,
        "weight_decay": parsed.weight_decay,
    }


def stage_one() -> tuple[str, list[dict], dict]:
    return (
        "hard_negative_micro_connections",
        [
            configuration_dict(micro_mode=mode, micro_strength=0.25)
            for mode in HARD_MODES
        ],
        {
            "anchor": "T077",
            "reason": "逐个测试 20 种困难负样本的小步度量修正，改动强度固定为 0.25。",
        },
    )


def reference_result(dataset: str) -> dict:
    return load_json(REFERENCE_ROOT / f"T077__{dataset}.json")


def score_configuration(output_dir: Path, config_id: str) -> dict:
    wins = ties = losses = all4 = strict_sota = metric_sota = 0
    deltas: list[float] = []
    dataset_wins: dict[str, int] = {}
    dataset_mean_deltas: dict[str, float] = {}
    for dataset in DATASETS:
        candidate = load_json(output_dir / "jobs" / f"{config_id}__{dataset}.json")
        if candidate.get("status") != "completed":
            raise ValueError(f"{config_id}/{dataset} 尚未完成")
        candidate_means = means(candidate)
        reference_means = means(reference_result(dataset))
        cmp = comparison(candidate_means, reference_means)
        wins += cmp[0]
        ties += cmp[1]
        losses += cmp[2]
        dataset_wins[dataset] = cmp[0]
        all4 += int(cmp[0] == 4)
        current_deltas = [
            left - right for left, right in zip(candidate_means, reference_means)
        ]
        deltas.extend(current_deltas)
        dataset_mean_deltas[dataset] = sum(current_deltas) / len(current_deltas)
        for group, stem in SOTAS.values():
            sota = load_json(sota_path(PROJECT_ROOT, dataset, group, stem))
            sota_cmp = comparison(candidate_means, means(sota))
            metric_sota += sota_cmp[0]
            strict_sota += int(sota_cmp[0] == 4)
    return {
        "wins_vs_t077": wins,
        "ties_vs_t077": ties,
        "losses_vs_t077": losses,
        "all4_datasets_vs_t077": all4,
        "minimum_metric_wins_per_dataset": min(dataset_wins.values()),
        "worst_dataset_mean_delta": min(dataset_mean_deltas.values()),
        "mean_delta_vs_t077": sum(deltas) / len(deltas),
        "worst_delta_vs_t077": min(deltas),
        "strict_sota_wins": strict_sota,
        "metric_sota_wins": metric_sota,
        "dataset_wins": dataset_wins,
        "dataset_mean_deltas": dataset_mean_deltas,
    }


def rank_key(item: dict) -> tuple:
    score = item["score"]
    return (
        score["all4_datasets_vs_t077"],
        score["minimum_metric_wins_per_dataset"],
        score["wins_vs_t077"],
        score["worst_dataset_mean_delta"],
        score["worst_delta_vs_t077"],
        score["mean_delta_vs_t077"],
        score["strict_sota_wins"],
        score["metric_sota_wins"],
    )


def completed_stage_ranking(
    manifest: dict, output_dir: Path, stages: tuple[int, ...]
) -> list[dict]:
    entries = []
    for entry in manifest["configs"]:
        if entry["stage"] not in stages:
            continue
        entries.append({
            **entry,
            "score": score_configuration(output_dir, entry["config_id"]),
        })
    expected = 20 * len(stages)
    if len(entries) != expected:
        raise ValueError(f"阶段 {stages} 应有 {expected} 个完整配置，实际 {len(entries)}")
    return sorted(entries, key=rank_key, reverse=True)


def stage_two(anchor: dict) -> tuple[str, list[dict], dict]:
    mode = anchor["configuration"]["micro_mode"]
    strengths = (
        0.025, 0.05, 0.075, 0.10, 0.15,
        0.20, 0.30, 0.35, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90, 1.00,
        1.10, 1.20, 1.35, 1.50, 1.75,
    )
    return (
        f"refine_{mode}",
        [configuration_dict(micro_mode=mode, micro_strength=value) for value in strengths],
        {
            "anchor": anchor["config_id"],
            "anchor_score": anchor["score"],
            "reason": "保留第一阶段覆盖最稳的结构策略，单独细化其连续注入强度。",
        },
    )


def stage_three() -> tuple[str, list[dict], dict]:
    configurations = []
    for value in (0.0425, 0.0435, 0.0440, 0.0460, 0.0475):
        configurations.append(configuration_dict(temperature=value))
    for value in (3.4, 3.7, 3.9, 4.1, 4.3):
        configurations.append(configuration_dict(hard_alignment_gain=value))
    for value in (0.75, 0.875, 0.95, 1.05, 1.125):
        configurations.append(configuration_dict(auxiliary_scale=value))
    for value in (3.5e-4, 4.25e-4, 5.75e-4, 7e-4, 9e-4):
        configurations.append(configuration_dict(weight_decay=value))
    return (
        "t077_scalar_micro_search",
        configurations,
        {
            "anchor": "T077",
            "reason": "只在 T077 邻域分别微调温度、对齐增益、辅助权重和权重衰减。",
        },
    )


def stage_four() -> tuple[str, list[dict], dict]:
    return (
        "three_module_pipeline_micro_connections",
        [
            configuration_dict(micro_mode=mode, micro_strength=0.25)
            for mode in PIPELINE_MODES
        ],
        {
            "anchor": "T077",
            "reason": "逐个测试洁净路由、拓扑映射和辅助项之间的 20 种小步连接。",
        },
    )


def _signature(configuration: dict) -> tuple:
    return tuple(configuration[name] for name in sorted(configuration))


def stage_five(
    anchor: dict,
    ranking: list[dict],
    existing: set[tuple],
) -> tuple[str, list[dict], dict]:
    configuration = anchor["configuration"]
    changed = anchor["changed_parameters"]
    candidates: list[dict] = []
    if "micro_mode" in changed:
        mode = configuration["micro_mode"]
        center = float(configuration["micro_strength"])
        raw_values = [
            max(0.001, min(2.0, center * factor))
            for factor in (
                0.10, 0.20, 0.30, 0.40, 0.50,
                0.60, 0.70, 0.80, 0.90, 0.95,
                1.05, 1.10, 1.20, 1.30, 1.40,
                1.50, 1.65, 1.80, 2.00, 2.25,
                2.50, 2.75, 3.00, 3.50, 4.00,
            )
        ]
        for value in raw_values:
            candidate = configuration_dict(
                micro_mode=mode,
                micro_strength=round(value, 6),
            )
            if _signature(candidate) not in existing:
                candidates.append(candidate)
    else:
        parameter = changed[0]
        center = float(configuration[parameter])
        # 单参数候选若在不同数据集上互补，最后阶段允许把两个已经验证过的
        # 小改动组成二维局部网格；总改动字段仍严格不超过两个。
        weak_weights = {
            dataset: 4 - wins
            for dataset, wins in anchor["score"]["dataset_wins"].items()
        }
        secondary_candidates = []
        for item in ranking:
            item_changed = item["changed_parameters"]
            if len(item_changed) != 1 or item_changed[0] == parameter:
                continue
            if item["configuration"]["micro_mode"] != "t077":
                continue
            complement = sum(
                weak_weights[dataset] * wins
                for dataset, wins in item["score"]["dataset_wins"].items()
            )
            secondary_candidates.append((complement, rank_key(item), item))
        if secondary_candidates:
            secondary_candidates.sort(reverse=True, key=lambda value: (value[0], value[1]))
            secondary = secondary_candidates[0][2]
            secondary_parameter = secondary["changed_parameters"][0]
            secondary_center = float(
                secondary["configuration"][secondary_parameter]
            )
            primary_values = [center * factor for factor in (0.85, 0.925, 1.0, 1.075, 1.15)]
            secondary_values = [
                secondary_center * factor for factor in (0.90, 0.975, 1.0, 1.05)
            ]
            for primary_value in primary_values:
                for secondary_value in secondary_values:
                    candidate = configuration_dict(**{
                        parameter: primary_value,
                        secondary_parameter: secondary_value,
                    })
                    if _signature(candidate) not in existing:
                        candidates.append(candidate)
        else:
            secondary = None
            secondary_parameter = None
            factors = (
                0.70, 0.75, 0.80, 0.825, 0.85,
                0.875, 0.90, 0.925, 0.95, 0.975,
                1.025, 1.05, 1.075, 1.10, 1.125,
                1.15, 1.175, 1.20, 1.25, 1.30,
                1.35, 1.40, 1.50,
            )
            for factor in factors:
                value = center * factor
                candidate = configuration_dict(**{parameter: value})
                if _signature(candidate) not in existing:
                    candidates.append(candidate)
    unique: list[dict] = []
    seen: set[tuple] = set()
    for candidate in candidates:
        signature = _signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(candidate)
    if len(unique) < 20:
        raise ValueError(f"最终局部候选不足 20 个，只有 {len(unique)} 个")
    unique.sort(
        key=lambda candidate: sum(
            abs(float(candidate[name]) - float(configuration[name]))
            for name in candidate
            if not isinstance(candidate[name], str)
        )
    )
    secondary_note = None
    if "micro_mode" not in changed and secondary is not None:
        secondary_note = {
            "config_id": secondary["config_id"],
            "parameter": secondary_parameter,
            "score": secondary["score"],
        }
    return (
        f"final_local_refinement_{anchor['config_id']}",
        unique[:20],
        {
            "anchor": anchor["config_id"],
            "anchor_score": anchor["score"],
            "complementary_anchor": secondary_note,
            "reason": (
                "从前四阶段整体最稳候选出发；结构候选只细化注入强度，"
                "单参数候选则与弱数据集上最互补的另一参数组成二参数局部网格。"
            ),
        },
    )


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("protocol") != PROTOCOL:
            raise ValueError("已有 manifest 协议不兼容")
    else:
        manifest = {
            "protocol": PROTOCOL,
            "base_model": "ReToGCL-O07-T077",
            "default_configuration": configuration_dict(),
            "selection_rule": [],
            "configs": [],
            "stages": [],
        }
    manifest["selection_rule"] = [
        "all4_datasets_vs_t077",
        "minimum_metric_wins_per_dataset",
        "wins_vs_t077",
        "worst_dataset_mean_delta",
        "worst_delta_vs_t077",
        "mean_delta_vs_t077",
        "strict_sota_wins",
        "metric_sota_wins",
    ]
    if any(entry["stage"] == args.stage for entry in manifest["configs"]):
        print(f"阶段 {args.stage} 已存在，不重复生成")
        return
    if args.stage != len(manifest["stages"]) + 1:
        raise ValueError("阶段必须按 1 到 5 顺序生成")

    if args.stage == 1:
        focus, configurations, selection = stage_one()
    elif args.stage == 2:
        anchor = completed_stage_ranking(manifest, output_dir, (1,))[0]
        focus, configurations, selection = stage_two(anchor)
    elif args.stage == 3:
        completed_stage_ranking(manifest, output_dir, (1, 2))
        focus, configurations, selection = stage_three()
    elif args.stage == 4:
        completed_stage_ranking(manifest, output_dir, (1, 2, 3))
        focus, configurations, selection = stage_four()
    else:
        ranking = completed_stage_ranking(manifest, output_dir, (1, 2, 3, 4))
        existing = {_signature(entry["configuration"]) for entry in manifest["configs"]}
        focus, configurations, selection = stage_five(
            ranking[0], ranking, existing
        )

    if len(configurations) != 20:
        raise ValueError(f"每阶段必须生成 20 个版本，实际 {len(configurations)}")
    start = len(manifest["configs"]) + 1
    entries = []
    for offset, configuration in enumerate(configurations):
        parsed = T077MicroConfiguration(**configuration)
        entries.append({
            "config_id": f"M{start + offset:03d}",
            "stage": args.stage,
            "focus": focus,
            "configuration": configuration,
            "changed_parameters": list(parsed.changed_parameters()),
        })
    manifest["configs"].extend(entries)
    manifest["stages"].append({
        "stage": args.stage,
        "focus": focus,
        "selection": selection,
        "config_ids": [entry["config_id"] for entry in entries],
    })
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(f"已生成阶段 {args.stage}：{entries[0]['config_id']}--{entries[-1]['config_id']}")


if __name__ == "__main__":
    main()
