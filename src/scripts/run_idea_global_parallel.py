"""在本机 GPU 上按 manifest 并行运行全局顺序调参缺失配置。"""

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
WORKER = PROJECT_ROOT / "src/scripts/run_idea_global_trial.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", type=int, required=True)
    parser.add_argument("--jobs-per-device", type=int, default=2)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--output-dir", type=Path, default=Path("IDEA_PARA_GLOBAL"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--min-free-memory-mib", type=int, default=4000)
    parser.add_argument("--poll-seconds", type=int, default=3)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--redistribute-missing",
        action="store_true",
        help="先排除已有结果，再对缺失任务重新分片；用于多主机断点续跑",
    )
    parser.add_argument(
        "--prefilter-existing",
        action="store_true",
        help="保留原分片归属，但在启动前一次性排除已有结果",
    )
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


def write_status(
    path: Path,
    *,
    stage: str,
    total: int,
    queue: list[dict],
    running: dict,
    completed: int,
    skipped: int,
    failures: list,
) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "total": total,
        "queued": len(queue),
        "running_count": len(running),
        "completed": completed,
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
    if args.jobs_per_device != 2:
        raise SystemExit("本轮要求每张 GPU 固定两个 Lim 任务，请使用 --jobs-per-device 2")
    if args.shard_count < 1 or any(
        index < 0 or index >= args.shard_count for index in args.shard_indices
    ):
        raise SystemExit("分片参数不合法")
    visible = torch.cuda.device_count()
    invalid = [device for device in args.devices if not 0 <= device < visible]
    if invalid:
        raise SystemExit(f"GPU {invalid} 不可见；当前仅检测到 {visible} 张 GPU")

    manifest = resolve_path(args.manifest)
    output_dir = resolve_path(args.output_dir)
    data_root = resolve_path(args.data_root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    stage = str(payload["metadata"]["stage"])
    all_tasks = payload["tasks"]
    selected_shards = set(args.shard_indices)
    if args.redistribute_missing:
        missing_tasks = [
            task
            for task in all_tasks
            if not (output_dir / "jobs" / f"{task['task_id']}.json").exists()
        ]
        queue = [
            task
            for index, task in enumerate(missing_tasks)
            if index % args.shard_count in selected_shards
        ]
        total = len(queue)
        initial_skipped = 0
    else:
        selected_tasks = [
            task for index, task in enumerate(all_tasks)
            if index % args.shard_count in selected_shards
        ]
        total = len(selected_tasks)
        if args.prefilter_existing and not args.force:
            queue = [
                task
                for task in selected_tasks
                if not (output_dir / "jobs" / f"{task['task_id']}.json").exists()
            ]
            initial_skipped = total - len(queue)
        else:
            queue = selected_tasks
            initial_skipped = 0
    running: dict[tuple[int, int], tuple] = {}
    failures = []
    completed = 0
    skipped = initial_skipped
    log_dir = output_dir / "launcher_logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"launcher_status_{args.tag}.json"
    progress = tqdm(total=total, desc=f"{args.tag}/stage_{stage}", unit="配置")
    if skipped:
        progress.update(skipped)

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
                if result_path.exists() and not args.force:
                    skipped += 1
                    progress.update(1)
                    continue
                log_path = log_dir / f"{task['task_id']}__gpu{device}_slot{slot}.log"
                command = [
                    sys.executable,
                    str(WORKER),
                    "--manifest",
                    str(manifest),
                    "--task-id",
                    task["task_id"],
                    "--data-root",
                    str(data_root),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    f"cuda:{device}",
                    "--epochs",
                    str(args.epochs),
                    "--probe-epochs",
                    str(args.probe_epochs),
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
        progress.set_postfix(
            等待=len(queue), 运行=len(running), 完成=completed, 跳过=skipped, 失败=len(failures)
        )
        write_status(
            status_path,
            stage=stage,
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
        raise SystemExit(f"以下全局顺序调参任务失败：\n{details}")


if __name__ == "__main__":
    main()
