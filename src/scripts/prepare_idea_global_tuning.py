"""准备并汇总跨七数据集的全局 A→B→C 顺序调参实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.idea_parameter_registry import parameter_space

METRICS = ("accuracy", "nmi", "ari", "macro_f1")
METRIC_KEYS = {
    "accuracy": "accuracy_percent",
    "nmi": "nmi_percent",
    "ari": "ari_percent",
    "macro_f1": "macro_f1_percent",
}
DATASETS = (
    "NCI1",
    "PROTEINS",
    "COLLAB",
    "MUTAG",
    "COLORS-3",
    "PTC_MR",
    "Mutagenicity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("B", "C", "finalize"),
        help="准备 B/C 阶段，或在 C 阶段齐全后生成最终默认配置",
    )
    parser.add_argument("--source-dir", type=Path, default=Path("IDEA_PARA"))
    parser.add_argument("--output-dir", type=Path, default=Path("IDEA_PARA_GLOBAL"))
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parameter_key(parameters: dict[str, float]) -> str:
    normalized = {name: float(value) for name, value in sorted(parameters.items())}
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def configuration_id(parameters: dict[str, float]) -> str:
    return hashlib.sha1(parameter_key(parameters).encode("utf-8")).hexdigest()[:12]


def metric_payload(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {METRIC_KEYS[metric]: record[METRIC_KEYS[metric]] for metric in METRICS}


def metric_means(record: dict[str, Any]) -> dict[str, float]:
    return {
        metric: float(record[METRIC_KEYS[metric]]["mean"])
        for metric in METRICS
    }


def composite_score(record: dict[str, Any]) -> float:
    return sum(metric_means(record).values()) / len(METRICS)


def same_metrics(first: dict[str, Any], second: dict[str, Any]) -> bool:
    for metric in METRICS:
        key = METRIC_KEYS[metric]
        if first[key] != second[key]:
            return False
    return True


def cache_record(
    cache: dict[tuple[str, str, str], dict[str, Any]],
    *,
    idea_id: str,
    dataset: str,
    parameters: dict[str, float],
    record: dict[str, Any],
    source: str,
) -> None:
    key = (idea_id, dataset, parameter_key(parameters))
    normalized = {
        "idea_id": idea_id,
        "dataset": dataset,
        "parameters": {name: float(value) for name, value in parameters.items()},
        "source": source,
        **metric_payload(record),
    }
    existing = cache.get(key)
    if existing is not None and not same_metrics(existing, normalized):
        raise RuntimeError(
            f"同一配置出现冲突结果：{idea_id}/{dataset}/{configuration_id(parameters)}"
        )
    cache[key] = normalized


def load_source_jobs(
    source_dir: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    paths = sorted((source_dir / "jobs").glob("*.json"))
    if len(paths) != 770:
        raise RuntimeError(f"源结果不完整：{len(paths)}/770")
    jobs: dict[tuple[str, str], dict[str, Any]] = {}
    cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        job = read_json(path)
        if job.get("status") != "completed":
            raise RuntimeError(f"源任务未完成：{path}")
        idea_id = str(job["idea_id"])
        dataset = str(job["dataset"])
        jobs[(idea_id, dataset)] = job
        for stage in job["stages"]:
            parameter = str(stage["parameter"])
            inherited = {
                name: float(value)
                for name, value in stage["inherited_parameters"].items()
            }
            for candidate in stage["candidates"]:
                parameters = dict(inherited)
                parameters[parameter] = float(candidate["value"])
                cache_record(
                    cache,
                    idea_id=idea_id,
                    dataset=dataset,
                    parameters=parameters,
                    record=candidate,
                    source=f"{path}:stage_{stage['stage']}",
                )
    expected_pairs = {
        (f"D{family}-I{variant:02d}", dataset)
        for family in range(1, 12)
        for variant in range(1, 11)
        for dataset in DATASETS
    }
    if set(jobs) != expected_pairs:
        missing = sorted(expected_pairs - set(jobs))
        extra = sorted(set(jobs) - expected_pairs)
        raise RuntimeError(f"源任务覆盖异常：missing={missing[:5]}, extra={extra[:5]}")
    return jobs, cache


def load_global_jobs(
    output_dir: Path,
    cache: dict[tuple[str, str, str], dict[str, Any]],
) -> int:
    count = 0
    for path in sorted((output_dir / "jobs").glob("*.json")):
        payload = read_json(path)
        if payload.get("status") != "completed":
            continue
        result = payload["result"]
        parameters = {
            name: float(value) for name, value in result["parameters"].items()
        }
        cache_record(
            cache,
            idea_id=str(result["idea_id"]),
            dataset=str(result["dataset"]),
            parameters=parameters,
            record=result,
            source=str(path),
        )
        count += 1
    return count


def idea_ids(jobs: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    return sorted(
        {idea_id for idea_id, _ in jobs},
        key=lambda value: tuple(int(part[1:]) for part in value.split("-")),
    )


def parameter_layout(
    jobs: dict[tuple[str, str], dict[str, Any]], idea_id: str
) -> tuple[list[str], list[list[float]], dict[str, float]]:
    job = jobs[(idea_id, DATASETS[0])]
    names = [str(stage["parameter"]) for stage in job["stages"]]
    values = [
        [float(candidate["value"]) for candidate in stage["candidates"]]
        for stage in job["stages"]
    ]
    registry = parameter_space(idea_id)
    defaults = registry.defaults
    registry_names = [parameter.name for parameter in registry.parameters]
    registry_values = [list(parameter.values) for parameter in registry.parameters]
    if names != registry_names or values != registry_values:
        raise RuntimeError(f"{idea_id} 的结果参数空间与 IDEA-PARA.md 不一致")
    if set(defaults) != set(names):
        raise RuntimeError(f"无法恢复 {idea_id} 的三个默认参数")
    return names, values, defaults


def lookup(
    cache: dict[tuple[str, str, str], dict[str, Any]],
    idea_id: str,
    dataset: str,
    parameters: dict[str, float],
) -> dict[str, Any] | None:
    return cache.get((idea_id, dataset, parameter_key(parameters)))


def aggregate_candidates(
    *,
    idea_id: str,
    stage: str,
    parameter: str,
    candidates: list[tuple[float, dict[str, dict[str, Any]]]],
    preferred_value: float,
) -> dict[str, Any]:
    if len(candidates) != 5:
        raise RuntimeError(f"{idea_id}/stage_{stage} 候选值不是5个")
    for value, records in candidates:
        if set(records) != set(DATASETS):
            missing = sorted(set(DATASETS) - set(records))
            raise RuntimeError(f"{idea_id}/stage_{stage}/{value} 缺数据集：{missing}")

    raw_scores: dict[float, dict[str, float]] = {}
    metric_scores: dict[float, dict[str, dict[str, float]]] = {}
    for value, records in candidates:
        raw_scores[value] = {
            dataset: composite_score(records[dataset]) for dataset in DATASETS
        }
        metric_scores[value] = {
            dataset: metric_means(records[dataset]) for dataset in DATASETS
        }

    normalized: dict[float, dict[str, float]] = {value: {} for value, _ in candidates}
    ranks: dict[float, dict[str, int]] = {value: {} for value, _ in candidates}
    for dataset in DATASETS:
        values_for_dataset = {
            value: raw_scores[value][dataset] for value, _ in candidates
        }
        minimum = min(values_for_dataset.values())
        maximum = max(values_for_dataset.values())
        denominator = maximum - minimum
        ordered = sorted(
            values_for_dataset,
            key=lambda value: (
                values_for_dataset[value],
                metric_scores[value][dataset]["accuracy"],
                metric_scores[value][dataset]["macro_f1"],
                metric_scores[value][dataset]["nmi"],
                metric_scores[value][dataset]["ari"],
            ),
            reverse=True,
        )
        for rank, value in enumerate(ordered, start=1):
            ranks[value][dataset] = rank
            normalized[value][dataset] = (
                0.5 if denominator < 1e-12 else (values_for_dataset[value] - minimum) / denominator
            )

    summaries = []
    values_in_order = [value for value, _ in candidates]
    for value, records in candidates:
        average_metrics = {
            metric: sum(metric_scores[value][dataset][metric] for dataset in DATASETS)
            / len(DATASETS)
            for metric in METRICS
        }
        summary = {
            "value": value,
            "mean_normalized_composite": sum(normalized[value].values()) / len(DATASETS),
            "mean_rank": sum(ranks[value].values()) / len(DATASETS),
            "mean_raw_composite": sum(raw_scores[value].values()) / len(DATASETS),
            "mean_metrics": average_metrics,
            "per_dataset": {
                dataset: {
                    "composite": raw_scores[value][dataset],
                    "normalized_composite": normalized[value][dataset],
                    "rank": ranks[value][dataset],
                    "metrics": metric_scores[value][dataset],
                    "source": records[dataset]["source"],
                }
                for dataset in DATASETS
            },
        }
        summaries.append(summary)

    def selection_key(summary: dict[str, Any]) -> tuple[float, ...]:
        metrics = summary["mean_metrics"]
        return (
            float(summary["mean_normalized_composite"]),
            float(summary["mean_raw_composite"]),
            float(metrics["accuracy"]),
            float(metrics["macro_f1"]),
            float(metrics["nmi"]),
            float(metrics["ari"]),
            float(abs(float(summary["value"]) - preferred_value) < 1e-12),
            -float(values_in_order.index(float(summary["value"]))),
        )

    selected = max(summaries, key=selection_key)
    return {
        "idea_id": idea_id,
        "stage": stage,
        "parameter": parameter,
        "selected_value": float(selected["value"]),
        "selection_rule": {
            "within_dataset": "mean(Accuracy,NMI,ARI,Macro-F1)",
            "across_datasets_primary": "七数据集等权的候选内 min-max 归一化综合分均值",
            "tie_break": [
                "更高七数据集原始综合分均值",
                "Accuracy",
                "Macro-F1",
                "NMI",
                "ARI",
                "当前文档默认值",
                "文档候选顺序",
            ],
        },
        "candidates": summaries,
    }


def collect_stage_candidates(
    *,
    jobs: dict[tuple[str, str], dict[str, Any]],
    cache: dict[tuple[str, str, str], dict[str, Any]],
    idea_id: str,
    stage_index: int,
    inherited: dict[str, float],
) -> tuple[str, list[tuple[float, dict[str, dict[str, Any]]]], list[dict[str, Any]]]:
    names, values_by_stage, defaults = parameter_layout(jobs, idea_id)
    parameter = names[stage_index]
    tasks: list[dict[str, Any]] = []
    candidates: list[tuple[float, dict[str, dict[str, Any]]]] = []
    for value in values_by_stage[stage_index]:
        records: dict[str, dict[str, Any]] = {}
        for dataset in DATASETS:
            parameters = dict(defaults)
            parameters.update(inherited)
            parameters[parameter] = float(value)
            record = lookup(cache, idea_id, dataset, parameters)
            if record is None:
                task_id = f"{chr(65 + stage_index)}__{idea_id}__{dataset}__{configuration_id(parameters)}"
                tasks.append(
                    {
                        "task_id": task_id,
                        "stage": chr(65 + stage_index),
                        "idea_id": idea_id,
                        "dataset": dataset,
                        "parameters": parameters,
                    }
                )
            else:
                records[dataset] = record
        candidates.append((float(value), records))
    return parameter, candidates, tasks


def source_stage_a_candidates(
    jobs: dict[tuple[str, str], dict[str, Any]], idea_id: str
) -> tuple[str, list[tuple[float, dict[str, dict[str, Any]]]]]:
    parameter = str(jobs[(idea_id, DATASETS[0])]["stages"][0]["parameter"])
    values = [
        float(candidate["value"])
        for candidate in jobs[(idea_id, DATASETS[0])]["stages"][0]["candidates"]
    ]
    candidates = []
    for value in values:
        records = {}
        for dataset in DATASETS:
            stage = jobs[(idea_id, dataset)]["stages"][0]
            candidate = next(
                item for item in stage["candidates"] if float(item["value"]) == value
            )
            records[dataset] = {
                "idea_id": idea_id,
                "dataset": dataset,
                "parameters": {
                    **{
                        name: float(item)
                        for name, item in stage["inherited_parameters"].items()
                    },
                    parameter: value,
                },
                "source": f"IDEA_PARA/jobs/{idea_id}_{dataset}.json:stage_A",
                **metric_payload(candidate),
            }
        candidates.append((value, records))
    return parameter, candidates


def load_selection(output_dir: Path, stage: str) -> dict[str, dict[str, Any]]:
    path = output_dir / "selections" / f"stage_{stage}.json"
    if not path.exists():
        raise RuntimeError(f"缺少阶段选择：{path}")
    payload = read_json(path)
    records = payload["records"]
    return {str(record["idea_id"]): record for record in records}


def write_selection(output_dir: Path, stage: str, records: list[dict[str, Any]]) -> None:
    write_json(
        output_dir / "selections" / f"stage_{stage}.json",
        {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "idea_count": len(records),
                "dataset_count": len(DATASETS),
                "selection_scope": "cross_dataset_global_sequential",
            },
            "records": records,
        },
    )


def write_manifest(output_dir: Path, stage: str, tasks: list[dict[str, Any]]) -> Path:
    task_ids = [task["task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError(f"stage_{stage} 任务ID重复")
    path = output_dir / "manifests" / f"stage_{stage}.json"
    write_json(
        path,
        {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "task_count": len(tasks),
                "shard_count": 3,
                "host202_shards": [0],
                "host204_shards": [1, 2],
                "jobs_per_device": 2,
            },
            "tasks": tasks,
        },
    )
    return path


def prepare_b(
    jobs: dict[tuple[str, str], dict[str, Any]],
    cache: dict[tuple[str, str, str], dict[str, Any]],
    output_dir: Path,
) -> None:
    selections = []
    tasks: list[dict[str, Any]] = []
    for idea_id in idea_ids(jobs):
        parameter, candidates = source_stage_a_candidates(jobs, idea_id)
        _, _, defaults = parameter_layout(jobs, idea_id)
        selection = aggregate_candidates(
            idea_id=idea_id,
            stage="A",
            parameter=parameter,
            candidates=candidates,
            preferred_value=float(defaults[parameter]),
        )
        selections.append(selection)
        _, _, missing = collect_stage_candidates(
            jobs=jobs,
            cache=cache,
            idea_id=idea_id,
            stage_index=1,
            inherited={parameter: float(selection["selected_value"])},
        )
        tasks.extend(missing)
    write_selection(output_dir, "A", selections)
    path = write_manifest(output_dir, "B", tasks)
    print(f"全局 A 已选择：{len(selections)} 个 idea")
    print(f"B 阶段需补跑：{len(tasks)} 个配置，manifest={path}")


def prepare_c(
    jobs: dict[tuple[str, str], dict[str, Any]],
    cache: dict[tuple[str, str, str], dict[str, Any]],
    output_dir: Path,
) -> None:
    stage_a = load_selection(output_dir, "A")
    selections = []
    tasks: list[dict[str, Any]] = []
    incomplete = []
    for idea_id in idea_ids(jobs):
        names, _, _ = parameter_layout(jobs, idea_id)
        inherited_a = {names[0]: float(stage_a[idea_id]["selected_value"])}
        parameter, candidates, missing_b = collect_stage_candidates(
            jobs=jobs,
            cache=cache,
            idea_id=idea_id,
            stage_index=1,
            inherited=inherited_a,
        )
        if missing_b:
            incomplete.extend(missing_b)
            continue
        selection = aggregate_candidates(
            idea_id=idea_id,
            stage="B",
            parameter=parameter,
            candidates=candidates,
            preferred_value=float(parameter_layout(jobs, idea_id)[2][parameter]),
        )
        selections.append(selection)
        inherited_ab = {
            **inherited_a,
            parameter: float(selection["selected_value"]),
        }
        _, _, missing_c = collect_stage_candidates(
            jobs=jobs,
            cache=cache,
            idea_id=idea_id,
            stage_index=2,
            inherited=inherited_ab,
        )
        tasks.extend(missing_c)
    if incomplete:
        raise RuntimeError(f"B 阶段尚缺 {len(incomplete)} 个配置，不能准备 C")
    write_selection(output_dir, "B", selections)
    path = write_manifest(output_dir, "C", tasks)
    print(f"全局 B 已选择：{len(selections)} 个 idea")
    print(f"C 阶段需补跑：{len(tasks)} 个配置，manifest={path}")


def finalize(
    jobs: dict[tuple[str, str], dict[str, Any]],
    cache: dict[tuple[str, str, str], dict[str, Any]],
    output_dir: Path,
) -> None:
    stage_a = load_selection(output_dir, "A")
    stage_b = load_selection(output_dir, "B")
    selections = []
    defaults = []
    incomplete = []
    for idea_id in idea_ids(jobs):
        names, _, base_defaults = parameter_layout(jobs, idea_id)
        inherited = {
            names[0]: float(stage_a[idea_id]["selected_value"]),
            names[1]: float(stage_b[idea_id]["selected_value"]),
        }
        parameter, candidates, missing = collect_stage_candidates(
            jobs=jobs,
            cache=cache,
            idea_id=idea_id,
            stage_index=2,
            inherited=inherited,
        )
        if missing:
            incomplete.extend(missing)
            continue
        selection = aggregate_candidates(
            idea_id=idea_id,
            stage="C",
            parameter=parameter,
            candidates=candidates,
            preferred_value=float(base_defaults[parameter]),
        )
        selections.append(selection)
        parameters = {
            **inherited,
            parameter: float(selection["selected_value"]),
        }
        per_dataset = {}
        for dataset in DATASETS:
            record = lookup(cache, idea_id, dataset, parameters)
            if record is None:
                raise RuntimeError(f"最终配置缺结果：{idea_id}/{dataset}")
            per_dataset[dataset] = {
                "selection_score": composite_score(record),
                **metric_payload(record),
                "source": record["source"],
            }
        defaults.append(
            {
                "idea_id": idea_id,
                "title": parameter_space(idea_id).title,
                "family": jobs[(idea_id, DATASETS[0])]["family"],
                "parameters": parameters,
                "initial_document_defaults": base_defaults,
                "per_dataset": per_dataset,
                "fidelity": "differentiable_prototype",
            }
        )
    if incomplete:
        raise RuntimeError(f"C 阶段尚缺 {len(incomplete)} 个配置，不能生成默认配置")
    write_selection(output_dir, "C", selections)
    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "idea_count": len(defaults),
            "dataset_count": len(DATASETS),
            "protocol": "cross_dataset_global_sequential_A_then_B_then_C",
            "selection_rule": "七数据集等权的候选内 min-max 归一化四指标综合分",
            "seed": 42,
            "folds": 5,
            "fidelity": "differentiable_prototype",
        },
        "defaults": defaults,
    }
    write_json(output_dir / "GLOBAL_DEFAULTS.json", payload)

    lines = [
        "# 110 个 Idea 的七数据集全局默认参数",
        "",
        "- 协议：跨七数据集全局顺序贪心 A→B→C。",
        "- 数据集：NCI1、PROTEINS、COLLAB、MUTAG、COLORS-3、PTC_MR、Mutagenicity。",
        "- 选择：每个数据集内四指标等权，再对五个候选做 min-max 归一化，七数据集等权平均。",
        "- 不包含显著性分析或额外随机种子复验。",
        "",
        "| Idea | A | B | C |",
        "|---|---|---|---|",
    ]
    for item in defaults:
        parameters = list(item["parameters"].items())
        lines.append(
            f"| {item['idea_id']} | `{parameters[0][0]}={parameters[0][1]}` | "
            f"`{parameters[1][0]}={parameters[1][1]}` | "
            f"`{parameters[2][0]}={parameters[2][1]}` |"
        )
    markdown = output_dir / "GLOBAL_DEFAULTS.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"全局 C 已选择：{len(selections)} 个 idea")
    print(f"默认配置完成：{output_dir / 'GLOBAL_DEFAULTS.json'}")
    print(f"默认配置表完成：{markdown}")


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs, cache = load_source_jobs(source_dir)
    loaded = load_global_jobs(output_dir, cache)
    print(f"读取源缓存 {len(cache) - loaded} 项，新增全局实验缓存 {loaded} 项")
    if args.stage == "B":
        prepare_b(jobs, cache, output_dir)
    elif args.stage == "C":
        prepare_c(jobs, cache, output_dir)
    else:
        finalize(jobs, cache, output_dir)


if __name__ == "__main__":
    main()
