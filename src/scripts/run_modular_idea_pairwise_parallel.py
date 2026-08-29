"""在指定 GPU 上按配置的每卡进程数运行跨模块两两 idea 组合。"""

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
WORKER = PROJECT_ROOT / "src/scripts/run_modular_idea_pairwise_trial.py"
PROTOCOL = "modular_cross_module_pairwise_fixed_ABC_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", type=int, required=True)
    parser.add_argument("--jobs-per-device", type=int, default=2)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--output-dir", type=Path, default=Path("IDEA_PAIRWISE_MODULAR"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--min-free-memory-mib", type=int, default=4000)
    parser.add_argument("--poll-seconds", type=int, default=3)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def gpu_free_memory() -> dict[int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {
        int(line.split(",")[0].strip()): int(line.split(",")[1].strip())
        for line in output.splitlines()
    }


def valid_result(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "completed" and payload.get("protocol") == PROTOCOL


def write_status(
    path: Path,
    *,
    total: int,
    queue: list,
    running: dict,
    completed: int,
    skipped: int,
    failures: list,
) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": PROTOCOL,
        "total": total,
        "queued": len(queue),
        "running_count": len(running),
        "completed_this_run": completed,
        "skipped_existing": skipped,
        "running": {
            f"gpu{gpu}:slot{slot}": item[1]["task_id"]
            for (gpu, slot), item in running.items()
        },
        "failures": [
            {"task_id": task["task_id"], "exit_code": code, "log": str(log)}
            for task, code, log in failures
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.jobs_per_device < 1:
        raise SystemExit("jobs-per-device 必须大于等于 1")
    if args.shard_count < 1 or any(
        index < 0 or index >= args.shard_count for index in args.shard_indices
    ):
        raise SystemExit("分片参数不合法")
    if len(set(args.shard_indices)) != len(args.shard_indices):
        raise SystemExit("shard-indices 不允许重复")
    visible = torch.cuda.device_count()
    invalid = [device for device in args.devices if not 0 <= device < visible]
    if invalid:
        raise SystemExit(f"GPU {invalid} 不可见；当前仅检测到 {visible} 张 GPU")

    manifest = resolve_path(args.manifest)
    output_dir = resolve_path(args.output_dir)
    data_root = resolve_path(args.data_root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("metadata", {}).get("protocol") != PROTOCOL:
        raise SystemExit("任务清单协议不兼容")
    tasks = payload["tasks"]
    selected_shards = set(args.shard_indices)
    selected = [
        task for index, task in enumerate(tasks)
        if index % args.shard_count in selected_shards
    ]
    total = len(selected)
    queue = [
        task
        for task in selected
        if not valid_result(output_dir / "jobs" / f"{task['task_id']}.json")
    ]
    skipped = total - len(queue)

    running: dict[tuple[int, int], tuple] = {}
    failures = []
    completed = 0
    log_dir = output_dir / "launcher_logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"launcher_status_{args.tag}.json"
    progress = tqdm(total=total, initial=skipped, desc=args.tag, unit="任务")
    while queue or running:
        memory = gpu_free_memory()
        for device in args.devices:
            for slot in range(args.jobs_per_device):
                key = (device, slot)
                if key in running or not queue:
                    continue
                if memory.get(device, 0) < args.min_free_memory_mib:
                    break
                task = queue.pop(0)
                result_path = output_dir / "jobs" / f"{task['task_id']}.json"
                if valid_result(result_path) and not args.force:
                    skipped += 1
                    progress.update(1)
                    continue
                log_path = log_dir / f"{task['task_id']}__gpu{device}_slot{slot}.log"
                command = [
                    sys.executable,
                    str(WORKER),
                    "--manifest", str(manifest),
                    "--task-id", task["task_id"],
                    "--task-json", json.dumps(task, ensure_ascii=True, separators=(",", ":")),
                    "--data-root", str(data_root),
                    "--output-dir", str(output_dir),
                    "--device", f"cuda:{device}",
                    "--epochs", str(args.epochs),
                    "--probe-epochs", str(args.probe_epochs),
                ]
                if args.force:
                    command.append("--force")
                handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
                running[key] = process, task, log_path, handle

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
        progress.set_postfix(
            等待=len(queue), 运行=len(running), 完成=completed, 跳过=skipped, 失败=len(failures)
        )
        write_status(
            status_path,
            total=total,
            queue=queue,
            running=running,
            completed=completed,
            skipped=skipped,
            failures=failures,
        )
        if queue or running:
            time.sleep(max(1, args.poll_seconds))
    progress.close()
    if failures:
        details = "\n".join(
            f"- {task['task_id']}: exit={code}, log={log_path}"
            for task, code, log_path in failures
        )
        raise SystemExit(f"以下跨模块两两组合任务失败：\n{details}")


if __name__ == "__main__":
    main()
