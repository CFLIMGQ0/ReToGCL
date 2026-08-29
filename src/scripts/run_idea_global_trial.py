"""运行跨数据集全局顺序调参中的一个缺失配置。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.idea_parameter_registry import parameter_space
from src.scripts.run_idea_sequential_tuning import run_trial
from src.scripts.run_sota_five_fold import choose_device, load_graphs, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("IDEA_PARA_GLOBAL"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_task(manifest_path: Path, task_id: str) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [task for task in payload["tasks"] if task["task_id"] == task_id]
    if len(matches) != 1:
        raise RuntimeError(f"任务 {task_id} 在 manifest 中出现 {len(matches)} 次")
    return matches[0]


def validate_parameters(idea_id: str, parameters: dict[str, float]) -> dict[str, float]:
    space = parameter_space(idea_id)
    expected = {parameter.name for parameter in space.parameters}
    if set(parameters) != expected:
        raise RuntimeError(
            f"{idea_id} 参数字段异常：actual={sorted(parameters)}, expected={sorted(expected)}"
        )
    normalized = {name: float(value) for name, value in parameters.items()}
    for parameter in space.parameters:
        value = normalized[parameter.name]
        if not any(abs(value - float(candidate)) < 1e-12 for candidate in parameter.values):
            raise RuntimeError(f"{idea_id}/{parameter.name}={value} 不在候选空间中")
    return normalized


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    task = load_task(manifest_path, args.task_id)
    idea_id = str(task["idea_id"])
    dataset = str(task["dataset"])
    parameters = validate_parameters(idea_id, task["parameters"])
    result_path = output_dir / "jobs" / f"{args.task_id}.json"
    if result_path.exists() and not args.force:
        print(f"已存在，跳过：{result_path}")
        return

    args.idea = idea_id
    args.dataset = dataset
    device = choose_device(args.device)
    print(f"任务：{args.task_id}，设备：{device}")
    graphs, in_dim, classes = load_graphs(data_root, dataset)
    result = run_trial(
        args,
        graphs,
        in_dim,
        classes,
        parameters,
        output_dir,
        device,
    )
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": args.task_id,
        "stage": task["stage"],
        "status": "completed",
        "result": result,
        "fidelity": "differentiable_prototype",
    }
    save_json(result_path, payload)
    print(f"全局顺序调参配置完成：{result_path}")


if __name__ == "__main__":
    main()
