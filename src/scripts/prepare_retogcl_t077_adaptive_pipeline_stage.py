#!/usr/bin/env python3
"""依据已完成结果逐轮生成 100 个 T077 Pipeline 候选。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.retogcl_t077_adaptive_pipeline import (
    ACTION_SLOT,
    T077AdaptivePipelineConfiguration,
)
from src.scripts.summarize_retogcl_story_pipelines import (
    DATASETS,
    METRICS,
    SOTAS,
    comparison,
    means,
    sota_path,
)


PROTOCOL = "retogcl_t077_adaptive_pipeline_v1"
REFERENCE_ROOT = PROJECT_ROOT / "outputs/retogcl_o07_dynamic_100x6/jobs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=range(1, 11), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retogcl_t077_adaptive_pipeline_100x6"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def reference_result(dataset: str) -> dict:
    return load_json(REFERENCE_ROOT / f"T077__{dataset}.json")


def score_configuration(output_dir: Path, config_id: str) -> dict:
    wins = ties = losses = all4 = strict_sota = metric_sota = 0
    deltas: list[float] = []
    dataset_wins: dict[str, int] = {}
    dataset_mean_deltas: dict[str, float] = {}
    metric_deltas: dict[str, list[float]] = {}
    clipped_ratios: list[float] = []
    gradient_norms: list[float] = []
    correction_sizes: list[float] = []
    gate_means: list[float] = []
    for dataset in DATASETS:
        candidate = load_json(output_dir / "jobs" / f"{config_id}__{dataset}.json")
        if candidate.get("status") != "completed":
            raise ValueError(f"{config_id}/{dataset} 尚未完成")
        candidate_means = means(candidate)
        reference_means = means(reference_result(dataset))
        current = [
            left - right for left, right in zip(candidate_means, reference_means)
        ]
        cmp = comparison(candidate_means, reference_means)
        wins += cmp[0]
        ties += cmp[1]
        losses += cmp[2]
        all4 += int(cmp[0] == 4)
        dataset_wins[dataset] = cmp[0]
        dataset_mean_deltas[dataset] = sum(current) / len(current)
        metric_deltas[dataset] = current
        deltas.extend(current)
        for group, stem in SOTAS.values():
            sota = load_json(sota_path(PROJECT_ROOT, dataset, group, stem))
            sota_cmp = comparison(candidate_means, means(sota))
            metric_sota += sota_cmp[0]
            strict_sota += int(sota_cmp[0] == 4)
        evidence = candidate["training_evidence"]
        clipped_ratios.append(float(evidence["clipped_batch_ratio"]))
        gradient_norms.append(float(evidence["maximum_gradient_norm"]))
        correction_sizes.append(float(evidence["final_pipeline_relative_correction"]))
        gate_means.append(float(evidence["final_pipeline_gate_mean"]))
    return {
        "wins_vs_t077": wins,
        "ties_vs_t077": ties,
        "losses_vs_t077": losses,
        "all4_datasets_vs_t077": all4,
        "minimum_metric_wins_per_dataset": min(dataset_wins.values()),
        "covered_datasets": sum(value > 0 for value in dataset_wins.values()),
        "positive_mean_datasets": sum(
            value > 0 for value in dataset_mean_deltas.values()
        ),
        "worst_dataset_mean_delta": min(dataset_mean_deltas.values()),
        "mean_delta_vs_t077": sum(deltas) / len(deltas),
        "worst_delta_vs_t077": min(deltas),
        "strict_sota_wins": strict_sota,
        "metric_sota_wins": metric_sota,
        "dataset_wins": dataset_wins,
        "dataset_mean_deltas": dataset_mean_deltas,
        "metric_deltas": metric_deltas,
        "mean_clipped_ratio": sum(clipped_ratios) / len(clipped_ratios),
        "maximum_gradient_norm": max(gradient_norms),
        "mean_pipeline_correction": sum(correction_sizes) / len(correction_sizes),
        "mean_pipeline_gate": sum(gate_means) / len(gate_means),
    }


def rank_key(item: dict) -> tuple:
    score = item["score"]
    return (
        score["minimum_metric_wins_per_dataset"],
        score["covered_datasets"],
        score["positive_mean_datasets"],
        score["worst_dataset_mean_delta"],
        score["worst_delta_vs_t077"],
        score["wins_vs_t077"],
        score["all4_datasets_vs_t077"],
        score["mean_delta_vs_t077"],
        score["strict_sota_wins"],
        score["metric_sota_wins"],
        -score["mean_clipped_ratio"],
    )


def completed_ranking(manifest: dict, output_dir: Path) -> list[dict]:
    ranking = []
    for entry in manifest["configs"]:
        if any(
            not (output_dir / "jobs" / f"{entry['config_id']}__{dataset}.json").exists()
            for dataset in DATASETS
        ):
            continue
        ranking.append({
            **entry,
            "score": score_configuration(output_dir, entry["config_id"]),
        })
    return sorted(ranking, key=rank_key, reverse=True)


def configuration(**values) -> T077AdaptivePipelineConfiguration:
    return T077AdaptivePipelineConfiguration(**values)


def stage_one() -> tuple[str, list[T077AdaptivePipelineConfiguration], dict]:
    actions = (
        ("route_delete_clean", 0.20),
        ("topology_positive_semantic", 0.10),
        ("topology_anchor_clean", 0.10),
        ("topology_second_clean", 0.10),
        ("topology_clean_semantic", 0.10),
        ("hard_anchor_clean", 0.15),
        ("hard_route_raw", 0.20),
        ("hard_projection_raw", 0.10),
        ("clean_auxiliary_delete", 0.25),
        ("topology_auxiliary_delete", 0.25),
    )
    return (
        "单连接发现",
        [configuration(primary_action=action, primary_strength=strength) for action, strength in actions],
        {
            "anchor": "T077",
            "reason": "分别弱化、补接或删除一个真实接口，定位六数据集共同受益的连接位置。",
        },
    )


def _single_action(entry: dict) -> bool:
    return entry["configuration"]["secondary_action"] == "none"


def stage_two(ranking: list[dict]) -> tuple[str, list[T077AdaptivePipelineConfiguration], dict]:
    anchors = []
    seen = set()
    for item in ranking:
        action = item["configuration"]["primary_action"]
        if not _single_action(item) or action in seen:
            continue
        anchors.append(item)
        seen.add(action)
        if len(anchors) == 2:
            break
    values = (0.025, 0.05, 0.10, 0.35, 0.50)
    configs = []
    for anchor in anchors:
        base = T077AdaptivePipelineConfiguration(**anchor["configuration"])
        configs.extend(replace(base, primary_strength=value) for value in values)
    return (
        "双候选连接强度细化",
        configs,
        {
            "anchors": [item["config_id"] for item in anchors],
            "anchor_scores": [item["score"] for item in anchors],
            "reason": "对第一轮最稳且动作不同的两条连接分别做局部剂量搜索。",
        },
    )


def stage_three(ranking: list[dict]) -> tuple[str, list[T077AdaptivePipelineConfiguration], dict]:
    anchor = next(item for item in ranking if _single_action(item))
    base = T077AdaptivePipelineConfiguration(**anchor["configuration"])
    gates = (
        "clean", "dirty", "agreement", "retention", "topology_prior",
        "prototype_margin", "joint", "uncertainty", "learned_scalar", "learned_sample",
    )
    return (
        "样本级门控证据搜索",
        [replace(base, primary_gate=gate) for gate in gates],
        {
            "anchor": anchor["config_id"],
            "anchor_score": anchor["score"],
            "reason": "保持连接位置和强度不变，只比较十种可拒绝门证据。",
        },
    )


def stage_four(ranking: list[dict]) -> tuple[str, list[T077AdaptivePipelineConfiguration], dict]:
    anchor = next(item for item in ranking if _single_action(item))
    base = T077AdaptivePipelineConfiguration(**anchor["configuration"])
    configs = [
        replace(base, gradient_policy=value)
        for value in (
            "detach_gate", "detach_delta", "detach_both", "orthogonal_delta", "clip_delta"
        )
    ]
    configs.extend(
        replace(base, schedule=value)
        for value in ("warmup5", "warmup10", "warmup20", "decay", "late")
    )
    return (
        "梯度隔离与开放日程",
        configs,
        {
            "anchor": anchor["config_id"],
            "anchor_score": anchor["score"],
            "reason": "针对原型和拓扑瓶颈漂移，单独测试梯度阻断、修正约束和课程开放。",
        },
    )


def _control_from_anchor(anchor: dict) -> dict:
    values = anchor["configuration"]
    return {
        "primary_strength": values["primary_strength"],
        "primary_gate": values["primary_gate"],
        "gradient_policy": values["gradient_policy"],
        "schedule": values["schedule"],
    }


def action_sweep(
    ranking: list[dict], focus: str, actions: tuple[str, ...], reason: str
) -> tuple[str, list[T077AdaptivePipelineConfiguration], dict]:
    anchor = ranking[0]
    controls = _control_from_anchor(anchor)
    configs = [configuration(primary_action=action, **controls) for action in actions]
    return (
        focus,
        configs,
        {
            "anchor": anchor["config_id"],
            "anchor_score": anchor["score"],
            "reason": reason,
        },
    )


def stage_five(ranking: list[dict]):
    return action_sweep(
        ranking,
        "语义正样本到拓扑层的连接搜索",
        (
            "topology_positive_semantic", "topology_anchor_clean",
            "topology_second_clean", "topology_both_clean",
            "topology_clean_semantic", "topology_view_consensus",
            "topology_anchor_bypass", "topology_positive_bypass",
            "topology_output_clip", "topology_output_orthogonal",
        ),
        "重点修复 T077 中已构造语义正样本却未进入 M7 的断点，并比较清洗表示的接入位置。",
    )


def stage_six(ranking: list[dict]):
    return action_sweep(
        ranking,
        "路由与困难负样本连接搜索",
        (
            "route_delete_clean", "route_view_consensus", "route_residual_clip",
            "route_orthogonal", "hard_anchor_clean", "hard_route_raw",
            "hard_projection_raw", "hard_projection_unit_gain",
            "hard_projection_clean_gate", "hard_projection_joint_gate",
        ),
        "分别检查洁净路由、困难负样本来源和 T077 拓扑映射闭环，避免三处同时漂移。",
    )


def stage_seven(ranking: list[dict]):
    return action_sweep(
        ranking,
        "拓扑输出与辅助梯度消融",
        (
            "topology_anchor_bypass", "topology_positive_bypass",
            "topology_both_bypass", "topology_output_clip",
            "topology_output_tanh", "topology_output_orthogonal",
            "clean_auxiliary_delete", "clean_auxiliary_detach",
            "topology_auxiliary_delete", "topology_auxiliary_detach",
        ),
        "用删除和停止梯度区分性能损失来自表示连接还是辅助目标，并限制拓扑修正幅度。",
    )


def additive_pair_key(first: dict, second: dict) -> tuple:
    combined = []
    minimum_dataset_wins = 4
    all4 = 0
    for dataset in DATASETS:
        values = [
            left + right
            for left, right in zip(
                first["score"]["metric_deltas"][dataset],
                second["score"]["metric_deltas"][dataset],
            )
        ]
        wins = sum(value > 0 for value in values)
        minimum_dataset_wins = min(minimum_dataset_wins, wins)
        all4 += int(wins == 4)
        combined.extend(values)
    return (
        all4,
        minimum_dataset_wins,
        sum(value > 0 for value in combined),
        min(combined),
        sum(combined) / len(combined),
    )


def stage_eight(ranking: list[dict]):
    singles = [item for item in ranking if _single_action(item)][:35]
    pairs = []
    for first, second in itertools.combinations(singles, 2):
        first_action = first["configuration"]["primary_action"]
        second_action = second["configuration"]["primary_action"]
        if first_action == second_action:
            continue
        if ACTION_SLOT[first_action] == ACTION_SLOT[second_action]:
            continue
        pairs.append((additive_pair_key(first, second), first, second))
    pairs.sort(key=lambda value: value[0], reverse=True)
    configs = []
    sources = []
    seen_actions = set()
    for proxy, first, second in pairs:
        one = first["configuration"]
        two = second["configuration"]
        signature = tuple(sorted((one["primary_action"], two["primary_action"])))
        if signature in seen_actions:
            continue
        seen_actions.add(signature)
        configs.append(configuration(
            primary_action=one["primary_action"],
            primary_strength=one["primary_strength"],
            primary_gate=one["primary_gate"],
            secondary_action=two["primary_action"],
            secondary_strength=two["primary_strength"],
            secondary_gate=two["primary_gate"],
            gradient_policy=one["gradient_policy"],
            schedule=one["schedule"],
        ))
        sources.append({
            "first": first["config_id"],
            "second": second["config_id"],
            "additive_proxy": proxy,
        })
        if len(configs) == 10:
            break
    return (
        "互补双连接组合",
        configs,
        {
            "sources": sources,
            "reason": "仅组合不同真实接口；按两条单连接在弱数据集上的加性互补性预筛。",
        },
    )


def stage_nine(ranking: list[dict]):
    anchor = next(
        item for item in ranking
        if item["configuration"]["secondary_action"] != "none"
    )
    base = T077AdaptivePipelineConfiguration(**anchor["configuration"])
    factors = (
        (0.50, 1.00), (0.75, 1.00), (0.90, 1.00),
        (1.10, 1.00), (1.25, 1.00),
        (1.00, 0.50), (1.00, 0.75), (1.00, 0.90),
        (1.00, 1.10), (1.00, 1.25),
    )
    configs = [
        replace(
            base,
            primary_strength=min(1.5, base.primary_strength * first),
            secondary_strength=min(1.5, base.secondary_strength * second),
        )
        for first, second in factors
    ]
    return (
        "最佳双连接逐边强度细化",
        configs,
        {
            "anchor": anchor["config_id"],
            "anchor_score": anchor["score"],
            "reason": "一次只改变一条连接的剂量，分辨两条边各自的有效区间。",
        },
    )


def stage_ten(ranking: list[dict]):
    anchor = next(
        (item for item in ranking if item["configuration"]["secondary_action"] != "none"),
        ranking[0],
    )
    base = T077AdaptivePipelineConfiguration(**anchor["configuration"])
    configs = [
        replace(base, primary_gate="clean"),
        replace(base, primary_gate="agreement"),
        replace(base, primary_gate="joint"),
        replace(base, primary_gate="learned_sample"),
    ]
    if base.secondary_action != "none":
        configs.extend([
            replace(base, secondary_gate="clean"),
            replace(base, secondary_gate="joint"),
        ])
    else:
        configs.extend([
            replace(base, primary_gate="retention"),
            replace(base, primary_gate="topology_prior"),
        ])
    configs.extend([
        replace(base, gradient_policy="detach_gate"),
        replace(base, gradient_policy="detach_delta"),
        replace(base, schedule="warmup10"),
        replace(base, schedule="decay"),
    ])
    return (
        "最终门控与梯度闭环复核",
        configs,
        {
            "anchor": anchor["config_id"],
            "anchor_score": anchor["score"],
            "reason": "固定最终结构，只微调每条边的拒绝证据、梯度回传和开放日程。",
        },
    )


def signature(config: T077AdaptivePipelineConfiguration) -> tuple:
    return tuple(asdict(config).items())


def unique_ten(
    candidates: list[T077AdaptivePipelineConfiguration], existing: set[tuple]
) -> list[T077AdaptivePipelineConfiguration]:
    unique = []
    seen = set(existing)
    for candidate in candidates:
        key = signature(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    # 某轮可能恰好生成历史配置；只细调主边强度补齐，不引入第三条连接。
    base = candidates[0]
    for strength in (
        0.015, 0.035, 0.065, 0.085, 0.125, 0.175, 0.225,
        0.275, 0.325, 0.425, 0.60, 0.75, 0.90, 1.10, 1.30,
    ):
        if len(unique) == 10:
            break
        candidate = replace(base, primary_strength=strength)
        key = signature(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    if len(unique) != 10:
        raise ValueError(f"去重后候选不是 10 个，实际 {len(unique)}")
    return unique


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("protocol") != PROTOCOL:
            raise ValueError("已有 manifest 协议不兼容")
    else:
        manifest = {
            "protocol": PROTOCOL,
            "base_model": "ReToGCL-O07-T077",
            "selection_rule": [
                "all4_datasets_vs_t077",
                "minimum_metric_wins_per_dataset",
                "wins_vs_t077",
                "worst_dataset_mean_delta",
                "worst_delta_vs_t077",
                "mean_delta_vs_t077",
                "strict_sota_wins",
                "metric_sota_wins",
                "training_stability",
            ],
            "configs": [],
            "stages": [],
        }
    manifest["selection_rule"] = [
        "minimum_metric_wins_per_dataset",
        "covered_datasets",
        "positive_mean_datasets",
        "worst_dataset_mean_delta",
        "worst_delta_vs_t077",
        "wins_vs_t077",
        "all4_datasets_vs_t077",
        "mean_delta_vs_t077",
        "strict_sota_wins",
        "metric_sota_wins",
        "training_stability",
    ]
    if any(entry["stage"] == args.stage for entry in manifest["configs"]):
        print(f"阶段 {args.stage} 已存在，不重复生成")
        return
    if args.stage != len(manifest["stages"]) + 1:
        raise ValueError("阶段必须按 1 到 10 顺序生成")

    ranking = completed_ranking(manifest, output_dir)
    expected_completed = (args.stage - 1) * 10
    if len(ranking) != expected_completed:
        raise ValueError(
            f"生成阶段 {args.stage} 前应完成 {expected_completed} 个配置，实际 {len(ranking)}"
        )
    if args.stage == 1:
        focus, candidates, selection = stage_one()
    elif args.stage == 2:
        focus, candidates, selection = stage_two(ranking)
    elif args.stage == 3:
        focus, candidates, selection = stage_three(ranking)
    elif args.stage == 4:
        focus, candidates, selection = stage_four(ranking)
    elif args.stage == 5:
        focus, candidates, selection = stage_five(ranking)
    elif args.stage == 6:
        focus, candidates, selection = stage_six(ranking)
    elif args.stage == 7:
        focus, candidates, selection = stage_seven(ranking)
    elif args.stage == 8:
        focus, candidates, selection = stage_eight(ranking)
    elif args.stage == 9:
        focus, candidates, selection = stage_nine(ranking)
    else:
        focus, candidates, selection = stage_ten(ranking)

    existing = {
        tuple(entry["configuration"].items()) for entry in manifest["configs"]
    }
    candidates = unique_ten(candidates, existing)
    start = len(manifest["configs"]) + 1
    entries = [
        {
            "config_id": f"PL{start + offset:03d}",
            "stage": args.stage,
            "focus": focus,
            "configuration": asdict(candidate),
        }
        for offset, candidate in enumerate(candidates)
    ]
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
        f"已生成阶段 {args.stage}：{entries[0]['config_id']}--{entries[-1]['config_id']}"
    )


if __name__ == "__main__":
    main()
