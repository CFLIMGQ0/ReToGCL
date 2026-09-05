"""将六模型×十八数据集任务按分片并行调度到多张 GPU。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "src/scripts/run_dgc_graph_level_five_fold.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dgc_graph_level_models import METHODS
from src.scripts.graph_classification_datasets import GRAPH_CLASSIFICATION_DATASETS


DATASETS = GRAPH_CLASSIFICATION_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, required=True)
    parser.add_argument("--jobs-per-device", type=int, default=2)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--task-pairs", nargs="+", default=None, metavar="METHOD/DATASET",
        help="只运行明确列出的任务对；设置后忽略 methods、datasets 和分片筛选。",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dgc_graph_level_6x18"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--min-free-memory-mib", type=int, default=3500)
    parser.add_argument("--poll-seconds", type=int, default=3)
    parser.add_argument("--tag", default="local")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def free_memory() -> dict[int, int]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits",
    ], text=True)
    return {int(line.split(",")[0]): int(line.split(",")[1]) for line in output.splitlines()}


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    library = str(Path(sys.executable).resolve().parent.parent / "lib")
    current = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = f"{library}:{current}" if current else library
    return environment


def save_status(
    path: Path,
    total: int,
    queue: list[tuple[str, str]],
    running: dict,
    completed: int,
    failures: list,
) -> None:
    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "queued": len(queue),
        "running_count": len(running),
        "completed": completed,
        "running": {
            f"gpu{device}:slot{slot}": value[1]
            for (device, slot), value in running.items()
        },
        "failures": [
            {"task": task, "exit_code": code, "log": str(log)}
            for task, code, log in failures
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    visible = torch.cuda.device_count()
    invalid = [device for device in args.devices if not 0 <= device < visible]
    if invalid:
        raise SystemExit(f"请求的 GPU {invalid} 不可见；当前仅检测到 {visible} 张 GPU。")
    if args.jobs_per_device < 1:
        raise SystemExit("--jobs-per-device 必须大于等于 1。")
    if args.shard_count < 1 or any(not 0 <= index < args.shard_count for index in args.shard_indices):
        raise SystemExit("分片参数非法。")

    if args.task_pairs:
        queue = []
        for value in args.task_pairs:
            try:
                method, dataset = value.split("/", 1)
            except ValueError as error:
                raise SystemExit(f"任务对格式错误：{value}，应为 METHOD/DATASET。") from error
            if method not in METHODS or dataset not in DATASETS:
                raise SystemExit(f"未知任务对：{value}")
            queue.append((method, dataset))
        if len(set(queue)) != len(queue):
            raise SystemExit("--task-pairs 中存在重复任务。")
    else:
        combinations = [(method, dataset) for method in args.methods for dataset in args.datasets]
        selected = set(args.shard_indices)
        queue = [
            task for index, task in enumerate(combinations)
            if index % args.shard_count in selected
        ]
    total = len(queue)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir).resolve()
    log_dir = output_dir / "launcher_logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"launcher_status_{args.tag}.json"
    environment = child_environment()
    running: dict[tuple[int, int], tuple[subprocess.Popen, str, Path, object]] = {}
    failures: list[tuple[str, int, Path]] = []
    completed = 0
    progress = tqdm(total=total, desc=f"{args.tag} DGC 图级适配", unit="组")

    while queue or running:
        memory = free_memory()
        for device in args.devices:
            for slot in range(args.jobs_per_device):
                key = (device, slot)
                if key in running or not queue:
                    continue
                if memory.get(device, 0) < args.min_free_memory_mib:
                    break
                method, dataset = queue.pop(0)
                task = f"{method}/{dataset}"
                log_path = log_dir / f"{method}_{dataset}_gpu{device}_slot{slot}.log"
                handle = log_path.open("w", encoding="utf-8")
                command = [
                    sys.executable, str(RUNNER),
                    "--methods", method,
                    "--datasets", dataset,
                    "--device", f"cuda:{device}",
                    "--output-dir", str(output_dir),
                    "--epochs", str(args.epochs),
                    "--probe-epochs", str(args.probe_epochs),
                    "--batch-size", str(args.batch_size),
                    "--hidden-dim", str(args.hidden_dim),
                    "--layers", str(args.layers),
                    "--no-merge",
                ]
                if args.force:
                    command.append("--force")
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, stdout=handle,
                    stderr=subprocess.STDOUT, env=environment,
                )
                running[key] = (process, task, log_path, handle)

        finished = []
        for key, (process, task, log_path, handle) in running.items():
            code = process.poll()
            if code is None:
                continue
            handle.close()
            if code:
                failures.append((task, code, log_path))
            else:
                completed += 1
            finished.append(key)
            progress.update(1)
        for key in finished:
            del running[key]
        progress.set_postfix(等待=len(queue), 运行=len(running), 完成=completed, 失败=len(failures))
        save_status(status_path, total, queue, running, completed, failures)
        if queue or running:
            time.sleep(max(1, args.poll_seconds))
    progress.close()
    if failures:
        raise SystemExit("存在失败任务：\n" + "\n".join(
            f"- {task}: exit={code}, log={log}" for task, code, log in failures
        ))


if __name__ == "__main__":
    main()
