"""在多张 GPU 上并行运行 10 个 ReToGCL 单点优化×6个数据集。"""

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
DATASETS = ("ADHD200", "BACE", "BBBP", "Tox21", "AIDS", "BZR")
OPTIMIZATIONS = tuple(f"O{index:02d}" for index in range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-device", type=int, default=3)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-indices", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/retogcl_single_point_10x6"),
    )
    parser.add_argument("--min-free-memory-mib", type=int, default=3000)
    parser.add_argument("--poll-seconds", type=int, default=3)
    parser.add_argument("--tag", default="single_point_host202")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def free_memory() -> dict[int, int]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True)
    return {
        int(line.split(",")[0]): int(line.split(",")[1])
        for line in output.splitlines()
    }


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    library = str(Path(sys.executable).resolve().parent.parent / "lib")
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        f"{library}:{existing}" if existing else library
    )
    return environment


def write_status(
    path: Path, total: int, queue: list[tuple[str, str]], running: dict,
    completed: int, skipped: int, failures: list,
) -> None:
    value = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "queued": len(queue),
        "running_count": len(running),
        "completed": completed,
        "skipped": skipped,
        "running": {
            f"gpu{gpu}:slot{slot}": f"{item[1]}/{item[2]}"
            for (gpu, slot), item in running.items()
        },
        "failures": [
            {
                "optimization": optimization, "dataset": dataset,
                "exit_code": code, "log": str(log),
            }
            for optimization, dataset, code, log in failures
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    visible = torch.cuda.device_count()
    invalid = [device for device in args.devices if not 0 <= device < visible]
    if invalid:
        raise SystemExit(
            f"请求的 GPU {invalid} 不可见；当前仅检测到 {visible} 张 GPU。"
        )

    selected = set(args.shard_indices)
    all_tasks = [
        (optimization, dataset)
        for optimization_index, optimization in enumerate(OPTIMIZATIONS)
        for dataset_index, dataset in enumerate(DATASETS)
        if (optimization_index + dataset_index) % args.shard_count in selected
    ]
    output_dir = (
        args.output_dir if args.output_dir.is_absolute()
        else (PROJECT_ROOT / args.output_dir).resolve()
    )
    queue = []
    skipped = 0
    for optimization, dataset in all_tasks:
        result_path = output_dir / "jobs" / f"{optimization}__{dataset}.json"
        if result_path.exists() and not args.force:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                result = {}
            if result.get("status") == "completed":
                skipped += 1
                continue
        queue.append((optimization, dataset))

    total = len(all_tasks)
    log_dir = output_dir / "launcher_logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"launcher_status_{args.tag}.json"
    environment = child_environment()
    running: dict[
        tuple[int, int], tuple[subprocess.Popen, str, str, Path, object]
    ] = {}
    failures = []
    completed = 0
    progress = tqdm(
        total=total, initial=skipped, desc=f"{args.tag} 单点优化", unit="任务"
    )

    while queue or running:
        memory = free_memory()
        for device in args.devices:
            for slot in range(args.jobs_per_device):
                key = (device, slot)
                if key in running or not queue:
                    continue
                if memory.get(device, 0) < args.min_free_memory_mib:
                    break
                optimization, dataset = queue.pop(0)
                log_path = (
                    log_dir
                    / f"{optimization}__{dataset}_gpu{device}_slot{slot}.log"
                )
                handle = log_path.open("w", encoding="utf-8")
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "src/scripts/run_retogcl_single_point_five_fold.py"),
                    "--optimization", optimization,
                    "--dataset", dataset,
                    "--data-root", str(PROJECT_ROOT / "datasets/main_data"),
                    "--output-dir", str(output_dir),
                    "--device", f"cuda:{device}",
                ]
                if args.force:
                    command.append("--force")
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, stdout=handle,
                    stderr=subprocess.STDOUT, env=environment,
                )
                running[key] = (
                    process, optimization, dataset, log_path, handle
                )

        finished = []
        for key, (
            process, optimization, dataset, log_path, handle
        ) in running.items():
            code = process.poll()
            if code is None:
                continue
            handle.close()
            if code:
                failures.append((optimization, dataset, code, log_path))
            else:
                completed += 1
            finished.append(key)
            progress.update(1)
        for key in finished:
            del running[key]
        progress.set_postfix(
            等待=len(queue), 运行=len(running), 完成=completed,
            跳过=skipped, 失败=len(failures),
        )
        write_status(
            status_path, total, queue, running, completed, skipped, failures
        )
        if queue or running:
            time.sleep(max(1, args.poll_seconds))
    progress.close()
    if failures:
        raise SystemExit("存在失败任务：\n" + "\n".join(
            f"- {optimization}/{dataset}: exit={code}, log={log}"
            for optimization, dataset, code, log in failures
        ))


if __name__ == "__main__":
    main()
