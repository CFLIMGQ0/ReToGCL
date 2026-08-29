"""从两两组合当前前五名扩展第三个不同模块 idea，并生成七数据集任务。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.idea_parameter_registry import parameter_space
from src.models.modular_research_ideas import IDEA_TO_MODULE
from src.scripts.run_research_ideas import DATASETS


PROTOCOL = "modular_cross_module_triple_from_top5_pairwise_fixed_ABC_v1"
FIDELITY = "module_position_faithful_differentiable_triple_implementation"
MODULE_ORDER = {f"M{index}": index for index in range(1, 9)}
TOP_FIVE_PAIRS = (
    "D7-I10__D10-I04",
    "D4-I06__D10-I04",
    "D2-I03__D10-I04",
    "D11-I01__D10-I10",
    "D4-I08__D10-I03",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--defaults",
        type=Path,
        default=Path("IDEA_PARA_MODULAR/DEFAULT_PARAMETERS.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("IDEA_TRIPLE_MODULAR_TOP5/manifest.json"),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_parameters(idea_id: str, values: dict) -> dict[str, float]:
    space = parameter_space(idea_id)
    expected = {parameter.name for parameter in space.parameters}
    if set(values) != expected:
        raise RuntimeError(f"{idea_id} 的 ABC 最优参数字段异常")
    normalized = {name: float(value) for name, value in values.items()}
    for parameter in space.parameters:
        value = normalized[parameter.name]
        if not any(abs(value - float(candidate)) < 1e-12 for candidate in parameter.values):
            raise RuntimeError(f"{idea_id}/{parameter.name}={value} 不在候选空间中")
    return normalized


def canonical_ideas(idea_ids: tuple[str, str, str]) -> tuple[str, str, str]:
    return tuple(
        sorted(idea_ids, key=lambda idea_id: (MODULE_ORDER[IDEA_TO_MODULE[idea_id]], idea_id))
    )


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    defaults_path = resolve_path(args.defaults)
    output_path = resolve_path(args.output)
    defaults_payload = json.loads(defaults_path.read_text(encoding="utf-8"))
    if defaults_payload.get("metadata", {}).get("idea_count") != 110:
        raise RuntimeError("ABC 默认参数清单没有完整覆盖 110 个 idea")

    defaults = {}
    for record in defaults_payload["defaults"]:
        idea_id = str(record["idea_id"])
        if record.get("target_module") != IDEA_TO_MODULE[idea_id]:
            raise RuntimeError(f"{idea_id} 的模块映射与当前代码不一致")
        defaults[idea_id] = validate_parameters(idea_id, record["parameters"])
    if len(defaults) != 110:
        raise RuntimeError(f"ABC 默认参数数量异常：{len(defaults)}")

    triple_sources: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for pair_id in TOP_FIVE_PAIRS:
        pair = tuple(pair_id.split("__"))
        if len(pair) != 2 or any(idea_id not in defaults for idea_id in pair):
            raise RuntimeError(f"非法两两组合 ID：{pair_id}")
        occupied = {IDEA_TO_MODULE[idea_id] for idea_id in pair}
        if len(occupied) != 2:
            raise RuntimeError(f"前五名中存在同模块组合：{pair_id}")
        for third_idea in defaults:
            if IDEA_TO_MODULE[third_idea] in occupied:
                continue
            triple = canonical_ideas((pair[0], pair[1], third_idea))
            triple_sources[triple].add(pair_id)

    if len(triple_sources) != 414:
        raise RuntimeError(f"前五名扩展后的三模块组合数异常：{len(triple_sources)}")

    tasks = []
    for ideas in tqdm(
        sorted(triple_sources, key=lambda ids: tuple((MODULE_ORDER[IDEA_TO_MODULE[x]], x) for x in ids)),
        desc="生成前五名扩展三模块任务",
        unit="组合",
    ):
        modules = tuple(IDEA_TO_MODULE[idea_id] for idea_id in ideas)
        if len(set(modules)) != 3:
            raise RuntimeError(f"三模块去重失败：{ideas} -> {modules}")
        triple_id = "__".join(ideas)
        for dataset in DATASETS:
            tasks.append(
                {
                    "task_id": f"{triple_id}__{dataset}",
                    "triple_id": triple_id,
                    "idea_ids": list(ideas),
                    "modules": list(modules),
                    "idea_parameters": [defaults[idea_id] for idea_id in ideas],
                    "source_top_pairs": sorted(triple_sources[ideas]),
                    "dataset": dataset,
                }
            )

    if len(tasks) != 2_898 or len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError(f"三模块任务数量或唯一性异常：{len(tasks)}")

    manifest = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL,
            "fidelity": FIDELITY,
            "source_defaults": str(defaults_path.relative_to(PROJECT_ROOT)),
            "source_pairwise_protocol": "modular_cross_module_pairwise_fixed_ABC_v1",
            "selection_rule": "按35个SOTA、7数据集、4指标的四指标全胜数排序得到两两组合前五名",
            "source_top_pairs": list(TOP_FIVE_PAIRS),
            "triple_count": len(triple_sources),
            "dataset_count": len(DATASETS),
            "task_count": len(tasks),
            "datasets": list(DATASETS),
            "folds": 5,
            "constraint": "三个 idea 的 target_module 必须两两不同，并按 M1 到 M8 顺序执行",
        },
        "tasks": tasks,
    }
    atomic_write(output_path, manifest)
    print(f"三模块清单已生成：triples={len(triple_sources)}，tasks={len(tasks)}，{output_path}")


if __name__ == "__main__":
    main()
