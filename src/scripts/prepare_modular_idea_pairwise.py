"""由模块化 ABC 结果生成统一默认参数和跨模块两两组合任务。"""

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
from src.models.modular_research_ideas import IDEA_TO_MODULE, MODULE_IDEAS
from src.models.research_ideas import IDEA_IDS, get_idea_spec
from src.scripts.run_research_ideas import DATASETS


ABC_PROTOCOL = "modular_cross_dataset_sequential_A_then_B_then_C_v1"
ABC_FIDELITY = "module_position_faithful_differentiable_implementation"
PAIR_PROTOCOL = "modular_cross_module_pairwise_fixed_ABC_v1"
PAIR_FIDELITY = "module_position_faithful_differentiable_pairwise_implementation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abc-jobs", type=Path, default=Path("IDEA_PARA_MODULAR/jobs"))
    parser.add_argument(
        "--defaults-output",
        type=Path,
        default=Path("IDEA_PARA_MODULAR/DEFAULT_PARAMETERS.json"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("IDEA_PAIRWISE_MODULAR/manifest.json"),
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


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    abc_jobs = resolve_path(args.abc_jobs)
    defaults_output = resolve_path(args.defaults_output)
    manifest_output = resolve_path(args.manifest_output)
    now = datetime.now(timezone.utc).isoformat()

    defaults: dict[str, dict] = {}
    for idea_id in tqdm(IDEA_IDS, desc="核验模块化 ABC 最优参数", unit="idea"):
        path = abc_jobs / f"{idea_id}.json"
        if not path.exists():
            raise RuntimeError(f"缺少 ABC 结果：{path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "idea_id": idea_id,
            "target_module": IDEA_TO_MODULE[idea_id],
            "status": "completed",
            "protocol": ABC_PROTOCOL,
            "fidelity": ABC_FIDELITY,
        }
        mismatches = {
            key: (record.get(key), value)
            for key, value in expected.items()
            if record.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"{path} 元数据不兼容：{mismatches}")
        if set(record.get("final_results", {})) != set(DATASETS):
            raise RuntimeError(f"{idea_id} 的七数据集最终结果不完整")
        defaults[idea_id] = {
            "idea_id": idea_id,
            "target_module": IDEA_TO_MODULE[idea_id],
            "family": get_idea_spec(idea_id).family,
            "family_name": get_idea_spec(idea_id).family_name,
            "parameters": validate_parameters(idea_id, record["selected_parameters"]),
            "selection_score": float(record["selection_score"]),
            "source_job": str(path.relative_to(PROJECT_ROOT)),
        }

    defaults_payload = {
        "metadata": {
            "created_at": now,
            "source_protocol": ABC_PROTOCOL,
            "source_fidelity": ABC_FIDELITY,
            "selection_scope": "七个数据集等权、四项指标等权的顺序 A→B→C 最优配置",
            "idea_count": len(defaults),
            "module_counts": {
                module_id: len(idea_ids) for module_id, idea_ids in MODULE_IDEAS.items()
            },
        },
        "defaults": [defaults[idea_id] for idea_id in IDEA_IDS],
    }
    atomic_write(defaults_output, defaults_payload)

    ideas = sorted(IDEA_IDS, key=lambda idea_id: (int(IDEA_TO_MODULE[idea_id][1:]), idea_id))
    pairs = [
        (first, second)
        for first, second in itertools.combinations(ideas, 2)
        if IDEA_TO_MODULE[first] != IDEA_TO_MODULE[second]
    ]
    expected_pairs = len(ideas) * (len(ideas) - 1) // 2 - sum(
        len(idea_ids) * (len(idea_ids) - 1) // 2 for idea_ids in MODULE_IDEAS.values()
    )
    if len(pairs) != expected_pairs or expected_pairs != 5_160:
        raise RuntimeError(f"跨模块组合数量异常：actual={len(pairs)}, expected={expected_pairs}")

    tasks = []
    for first, second in tqdm(pairs, desc="生成跨模块组合任务", unit="组合"):
        pair_id = f"{first}__{second}"
        for dataset in DATASETS:
            tasks.append(
                {
                    "task_id": f"{pair_id}__{dataset}",
                    "pair_id": pair_id,
                    "first_idea": first,
                    "second_idea": second,
                    "first_module": IDEA_TO_MODULE[first],
                    "second_module": IDEA_TO_MODULE[second],
                    "first_parameters": defaults[first]["parameters"],
                    "second_parameters": defaults[second]["parameters"],
                    "dataset": dataset,
                }
            )
    if len(tasks) != 36_120 or len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError("跨模块组合任务数量或唯一性校验失败")

    manifest = {
        "metadata": {
            "created_at": now,
            "protocol": PAIR_PROTOCOL,
            "fidelity": PAIR_FIDELITY,
            "source_defaults": str(defaults_output.relative_to(PROJECT_ROOT)),
            "idea_count": len(defaults),
            "module_count": len(MODULE_IDEAS),
            "pair_count": len(pairs),
            "dataset_count": len(DATASETS),
            "task_count": len(tasks),
            "datasets": list(DATASETS),
            "folds": 5,
            "constraint": "只组合 target_module 不同的两个 idea",
        },
        "tasks": tasks,
    }
    atomic_write(manifest_output, manifest)
    print(
        f"配置已更新：ideas={len(defaults)}，跨模块 pairs={len(pairs)}，"
        f"tasks={len(tasks)}，manifest={manifest_output}"
    )


if __name__ == "__main__":
    main()
