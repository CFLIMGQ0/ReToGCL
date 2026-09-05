#!/usr/bin/env python3
"""依据上一阶段六数据集五折结果，生成 O07 下一阶段动态调参配置。"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.retogcl_o07_dynamic_tuning import O07DynamicConfiguration
from src.scripts.summarize_retogcl_story_pipelines import (
    DATASETS,
    SOTAS,
    comparison,
    means,
    sota_path,
)


PROTOCOL = "retogcl_o07_dynamic_tuning_v1"
DEFAULT = O07DynamicConfiguration()
DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT / "outputs" / "retogcl_o07_parameter_10x6" / "jobs"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=range(1, 6), required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/retogcl_o07_dynamic_100x6"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def configuration_dict(**changes: float) -> dict[str, float]:
    values = {
        "temperature": DEFAULT.temperature,
        "hard_alignment_gain": DEFAULT.hard_alignment_gain,
        "auxiliary_scale": DEFAULT.auxiliary_scale,
        "weight_decay": DEFAULT.weight_decay,
    }
    values.update(changes)
    configuration = O07DynamicConfiguration(**values)
    return {
        "temperature": configuration.temperature,
        "hard_alignment_gain": configuration.hard_alignment_gain,
        "auxiliary_scale": configuration.auxiliary_scale,
        "weight_decay": configuration.weight_decay,
    }


def stage_one() -> tuple[str, list[dict[str, float]], dict]:
    temperatures = (
        0.025, 0.030, 0.035, 0.040, 0.045,
        0.050, 0.054, 0.057, 0.063, 0.066,
        0.070, 0.075, 0.080, 0.090, 0.100,
        0.115, 0.130, 0.150, 0.180, 0.220,
    )
    return (
        "temperature_global_search",
        [configuration_dict(temperature=value) for value in temperatures],
        {"anchor": "P05", "reason": "先在 0.06 新默认点周围及两侧搜索温度。"},
    )


def score_configuration(output_dir: Path, config_id: str) -> dict:
    wins = ties = losses = all4 = strict_sota = metric_sota = 0
    deltas: list[float] = []
    for dataset in DATASETS:
        candidate = load_json(output_dir / "jobs" / f"{config_id}__{dataset}.json")
        if candidate.get("status") != "completed":
            raise ValueError(f"{config_id}/{dataset} 尚未完成")
        reference = load_json(DEFAULT_RESULT_ROOT / f"P05__{dataset}.json")
        candidate_means = means(candidate)
        reference_means = means(reference)
        cmp = comparison(candidate_means, reference_means)
        wins += cmp[0]
        ties += cmp[1]
        losses += cmp[2]
        all4 += int(cmp[0] == 4)
        deltas.extend(
            left - right for left, right in zip(candidate_means, reference_means)
        )
        for group, stem in SOTAS.values():
            sota = load_json(sota_path(PROJECT_ROOT, dataset, group, stem))
            sota_cmp = comparison(candidate_means, means(sota))
            metric_sota += sota_cmp[0]
            strict_sota += int(sota_cmp[0] == 4)
    return {
        "wins_vs_p05": wins,
        "ties_vs_p05": ties,
        "losses_vs_p05": losses,
        "all4_datasets_vs_p05": all4,
        "mean_delta_vs_p05": sum(deltas) / len(deltas),
        "worst_delta_vs_p05": min(deltas),
        "strict_sota_wins": strict_sota,
        "metric_sota_wins": metric_sota,
    }


def rank_key(item: dict) -> tuple:
    score = item["score"]
    return (
        score["strict_sota_wins"],
        score["metric_sota_wins"],
        score["wins_vs_p05"],
        score["all4_datasets_vs_p05"],
        score["mean_delta_vs_p05"],
        score["worst_delta_vs_p05"],
    )


def completed_stage_ranking(manifest: dict, output_dir: Path, stage: int) -> list[dict]:
    entries = []
    for entry in manifest["configs"]:
        if entry["stage"] != stage:
            continue
        entries.append({**entry, "score": score_configuration(output_dir, entry["config_id"])})
    if len(entries) != 20:
        raise ValueError(f"阶段 {stage} 应有 20 个完整配置，实际 {len(entries)}")
    return sorted(entries, key=rank_key, reverse=True)


def stage_two(anchor: dict) -> tuple[str, list[dict[str, float]], dict]:
    gains = (
        0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75,
        2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.5, 5.5, 7.0,
    )
    temperature = anchor["configuration"]["temperature"]
    return (
        "hard_alignment_gain_search",
        [
            configuration_dict(
                temperature=temperature, hard_alignment_gain=value
            )
            for value in gains
        ],
        {
            "anchor": anchor["config_id"],
            "anchor_score": anchor["score"],
            "reason": "固定阶段一最优温度，仅搜索困难负样本度量对齐增益。",
        },
    )


def stage_three(anchor: dict) -> tuple[str, list[dict[str, float]], dict]:
    scales = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0,
    )
    temperature = anchor["configuration"]["temperature"]
    return (
        "auxiliary_scale_search",
        [
            configuration_dict(temperature=temperature, auxiliary_scale=value)
            for value in scales
        ],
        {
            "anchor": anchor["config_id"],
            "anchor_score": anchor["score"],
            "reason": "固定阶段一最优温度，仅搜索三个创新模块辅助约束的统一倍率。",
        },
    )


def stage_four(anchor: dict) -> tuple[str, list[dict[str, float]], dict]:
    decays = (
        0.0, 1e-7, 3e-7, 1e-6, 2e-6, 3e-6, 5e-6, 7e-6,
        1.5e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4,
        2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3,
    )
    temperature = anchor["configuration"]["temperature"]
    return (
        "weight_decay_search",
        [
            configuration_dict(temperature=temperature, weight_decay=value)
            for value in decays
        ],
        {
            "anchor": anchor["config_id"],
            "anchor_score": anchor["score"],
            "reason": "固定阶段一最优温度，仅搜索 Adam 权重衰减。",
        },
    )


def local_values(value: float, parameter: str) -> list[float]:
    if parameter == "weight_decay":
        if math.isclose(value, 0.0, abs_tol=1e-16):
            return [0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6]
        return sorted({value * factor for factor in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)})
    lower = 0.0
    return sorted({max(lower, value * factor) for factor in (0.6, 0.8, 0.9, 1.0, 1.1, 1.25)})


def stage_five(best_secondary: dict, existing: set[tuple]) -> tuple[str, list[dict[str, float]], dict]:
    configuration = best_secondary["configuration"]
    secondary = next(
        name for name in ("hard_alignment_gain", "auxiliary_scale", "weight_decay")
        if not math.isclose(
            float(configuration[name]), float(getattr(DEFAULT, name)),
            rel_tol=1e-9, abs_tol=1e-12,
        )
    )
    temperature = configuration["temperature"]
    temperatures = sorted({
        round(max(0.005, temperature * factor), 6)
        for factor in (0.75, 0.85, 0.925, 1.0, 1.075, 1.15, 1.3)
    })
    secondary_values = local_values(float(configuration[secondary]), secondary)
    candidates = []
    for temp in temperatures:
        for value in secondary_values:
            config = configuration_dict(temperature=temp, **{secondary: value})
            signature = tuple(config[name] for name in sorted(config))
            if signature in existing:
                continue
            candidates.append(config)
    candidates.sort(
        key=lambda config: (
            abs(config["temperature"] - temperature),
            abs(config[secondary] - configuration[secondary]),
        )
    )
    if len(candidates) < 20:
        raise ValueError("局部二参数候选不足 20 个")
    return (
        f"joint_local_refinement_temperature_{secondary}",
        candidates[:20],
        {
            "anchor": best_secondary["config_id"],
            "anchor_score": best_secondary["score"],
            "secondary_parameter": secondary,
            "reason": "只在目前最优的温度—次参数二维邻域做最后局部搜索。",
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
            "base_model": "ReToGCL-O07-P05",
            "default_configuration": configuration_dict(),
            "selection_rule": [],
            "configs": [],
            "stages": [],
        }
    # 最终目标是论文比较中的广覆盖：先最大化对 13 个 SOTA 的完整四指标
    # 胜出数量，再看单指标覆盖；相对 P05 的广泛提升用于后续破同分。
    manifest["selection_rule"] = [
        "strict_sota_wins",
        "metric_sota_wins",
        "wins_vs_p05",
        "all4_datasets_vs_p05",
        "mean_delta_vs_p05",
        "worst_delta_vs_p05",
    ]
    if any(entry["stage"] == args.stage for entry in manifest["configs"]):
        print(f"阶段 {args.stage} 已存在，不重复生成")
        return
    if args.stage != len(manifest["stages"]) + 1:
        raise ValueError("阶段必须按 1 到 5 顺序生成")

    if args.stage == 1:
        focus, configurations, selection = stage_one()
    else:
        stage_one_ranking = completed_stage_ranking(manifest, output_dir, 1)
        if args.stage == 2:
            focus, configurations, selection = stage_two(stage_one_ranking[0])
        elif args.stage == 3:
            focus, configurations, selection = stage_three(stage_one_ranking[0])
        elif args.stage == 4:
            focus, configurations, selection = stage_four(stage_one_ranking[0])
        else:
            secondary_rankings = []
            for stage in (2, 3, 4):
                secondary_rankings.extend(completed_stage_ranking(manifest, output_dir, stage))
            secondary_rankings.sort(key=rank_key, reverse=True)
            existing = {
                tuple(entry["configuration"][name] for name in sorted(entry["configuration"]))
                for entry in manifest["configs"]
            }
            focus, configurations, selection = stage_five(
                secondary_rankings[0], existing
            )

    start = len(manifest["configs"]) + 1
    entries = []
    for offset, configuration in enumerate(configurations):
        parsed = O07DynamicConfiguration(**configuration)
        entries.append({
            "config_id": f"T{start + offset:03d}",
            "stage": args.stage,
            "focus": focus,
            "configuration": configuration,
            "changed_parameters": list(parsed.changed_parameters()),
        })
    if len(entries) != 20:
        raise ValueError(f"每阶段必须生成 20 个版本，实际 {len(entries)}")
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
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(
        f"已生成阶段 {args.stage}：{focus}，"
        f"配置 {entries[0]['config_id']}--{entries[-1]['config_id']}"
    )


if __name__ == "__main__":
    main()
