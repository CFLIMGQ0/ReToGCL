"""由110个 idea 的全局最优 ABC 参数生成跨方向两两组合任务。"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.idea_parameter_registry import parameter_space
from src.models.research_ideas import get_idea_spec
from src.scripts.run_research_ideas import DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--defaults",
        type=Path,
        default=Path("IDEA_PARA_GLOBAL/GLOBAL_DEFAULTS.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("IDEA_PAIRWISE/manifest.json"))
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_parameters(idea_id: str, values: dict) -> dict[str, float]:
    space = parameter_space(idea_id)
    expected = {parameter.name for parameter in space.parameters}
    if set(values) != expected:
        raise RuntimeError(f"{idea_id} 的全局最优参数字段不完整")
    normalized = {name: float(value) for name, value in values.items()}
    for parameter in space.parameters:
        value = normalized[parameter.name]
        if not any(abs(value - float(candidate)) < 1e-12 for candidate in parameter.values):
            raise RuntimeError(f"{idea_id}/{parameter.name}={value} 不在候选空间中")
    return normalized


def main() -> None:
    args = parse_args()
    defaults_path = resolve_path(args.defaults)
    output_path = resolve_path(args.output)
    payload = json.loads(defaults_path.read_text(encoding="utf-8"))
    records = payload.get("defaults", [])
    if len(records) != 110:
        raise RuntimeError(f"预期110个全局默认配置，实际为{len(records)}个")

    defaults = {}
    for record in records:
        idea_id = str(record["idea_id"])
        if idea_id in defaults:
            raise RuntimeError(f"全局默认配置重复：{idea_id}")
        defaults[idea_id] = {
            "idea_id": idea_id,
            "family": get_idea_spec(idea_id).family,
            "family_name": get_idea_spec(idea_id).family_name,
            "parameters": validate_parameters(idea_id, record["parameters"]),
        }

    ideas = sorted(defaults, key=lambda idea: (get_idea_spec(idea).family, idea))
    pairs = [
        (first, second)
        for first, second in itertools.combinations(ideas, 2)
        if defaults[first]["family"] != defaults[second]["family"]
    ]
    if len(pairs) != 5500:
        raise RuntimeError(f"合法跨方向组合应为5500个，实际为{len(pairs)}个")

    tasks = []
    for first, second in tqdm(pairs, desc="生成两两组合任务", unit="组合"):
        pair_id = f"{first}__{second}"
        for dataset in DATASETS:
            tasks.append(
                {
                    "task_id": f"{pair_id}__{dataset}",
                    "pair_id": pair_id,
                    "first_idea": first,
                    "second_idea": second,
                    "first_family": defaults[first]["family_name"],
                    "second_family": defaults[second]["family_name"],
                    "first_parameters": defaults[first]["parameters"],
                    "second_parameters": defaults[second]["parameters"],
                    "dataset": dataset,
                }
            )
    if len(tasks) != 38_500 or len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError("两两组合任务数量或唯一性校验失败")

    manifest = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": "cross_family_pairwise_fixed_global_ABC_shared_encoder",
            "source_defaults": str(defaults_path),
            "idea_count": len(defaults),
            "family_count": len({item["family"] for item in defaults.values()}),
            "pair_count": len(pairs),
            "dataset_count": len(DATASETS),
            "task_count": len(tasks),
            "datasets": list(DATASETS),
            "folds": 5,
            "fidelity": "differentiable_pairwise_prototype",
        },
        "tasks": tasks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(f"两两组合清单完成：pairs={len(pairs)}，tasks={len(tasks)}，path={output_path}")


if __name__ == "__main__":
    main()
