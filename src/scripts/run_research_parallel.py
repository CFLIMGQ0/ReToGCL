"""在当前主机的多张 GPU 上并行调度研究 idea × 数据集任务。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "src/scripts/run_research_ideas.py"
IDEA_IDS = tuple(f"D{family}-I{variant:02d}" for family in range(1, 12) for variant in range(1, 11))
DATASETS = (
    "NCI1", "PROTEINS", "COLLAB", "MUTAG", "COLORS-3",
    "PTC_MR", "Mutagenicity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, required=True)
    parser.add_argument("--jobs-per-device", type=int, default=3)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--ideas", nargs="+", choices=IDEA_IDS, default=list(IDEA_IDS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research_ideas"))
    parser.add_argument("--wait-for-free", action="store_true")
    parser.add_argument("--max-memory-used-mib", type=int, default=2000)
    parser.add_argument("--max-utilization", type=int, default=15)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def gpu_is_available(device: int, args: argparse.Namespace) -> bool:
    query = subprocess.run(
        [
            "nvidia-smi", f"--id={device}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    memory_used, utilization = (int(value.strip()) for value in query.split(","))
    return memory_used <= args.max_memory_used_mib and utilization <= args.max_utilization


def main() -> None:
    args = parse_args()
    if args.jobs_per_device < 1:
        raise SystemExit("--jobs-per-device 必须至少为 1")
    if args.shard_count < 1:
        raise SystemExit("--shard-count 必须至少为 1")
    if any(index < 0 or index >= args.shard_count for index in args.shard_indices):
        raise SystemExit("--shard-indices 必须位于 [0, shard-count) 范围")
    visible = torch.cuda.device_count()
    invalid = [index for index in args.devices if index < 0 or index >= visible]
    if invalid:
        raise SystemExit(f"请求的 GPU {invalid} 不可见；当前仅检测到 {visible} 张 GPU。")

    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    all_tasks = [(idea, dataset) for idea in args.ideas for dataset in args.datasets]
    selected_shards = set(args.shard_indices)
    queue = [
        task for index, task in enumerate(all_tasks)
        if index % args.shard_count in selected_shards
    ]
    slots = [(device, slot) for device in args.devices for slot in range(args.jobs_per_device)]
    running: dict[tuple[int, int], tuple[subprocess.Popen, object, str, Path]] = {}
    failures = []
    log_dir = output_dir / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=len(queue), desc="研究任务并行调度", unit="任务")
    admitted_devices: set[int] = set()
    last_gpu_poll = {device: 0.0 for device in args.devices}

    while queue or running:
        for device in args.devices:
            device_slots = [slot for slot in slots if slot[0] == device]
            occupied = [slot for slot in device_slots if slot in running]
            if not occupied and device not in admitted_devices:
                if args.wait_for_free:
                    now = time.monotonic()
                    if now - last_gpu_poll[device] < args.poll_seconds:
                        continue
                    last_gpu_poll[device] = now
                    if not gpu_is_available(device, args):
                        continue
                admitted_devices.add(device)
            if device not in admitted_devices:
                continue
            for _, slot_index in device_slots:
                slot = (device, slot_index)
                if slot in running or not queue:
                    continue
                idea, dataset = queue.pop(0)
                job_path = output_dir / "jobs" / f"{idea}_{dataset}.json"
                if job_path.exists() and not args.force:
                    progress.update(1)
                    continue
                log_path = log_dir / f"{idea}_{dataset}_gpu{device}_slot{slot_index}.log"
                command = [
                    sys.executable, str(RUNNER), "--ideas", idea, "--datasets", dataset,
                    "--device", f"cuda:{device}", "--epochs", str(args.epochs),
                    "--probe-epochs", str(args.probe_epochs), "--output-dir", str(output_dir),
                    "--no-merge",
                ]
                if args.force:
                    command.append("--force")
                handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, stdout=handle,
                    stderr=subprocess.STDOUT, env=os.environ.copy(),
                )
                running[slot] = (process, handle, f"{idea}/{dataset}", log_path)

        completed = []
        for slot, (process, handle, name, log_path) in running.items():
            code = process.poll()
            if code is None:
                continue
            handle.close()
            if code:
                failures.append((name, code, log_path))
            completed.append(slot)
            progress.update(1)
        for slot in completed:
            del running[slot]
        progress.set_postfix(running=len(running), failed=len(failures))
        if running and not completed:
            time.sleep(1)
        elif args.wait_for_free and queue and not running:
            time.sleep(max(1, args.poll_seconds))

    progress.close()
    subprocess.run(
        [sys.executable, str(RUNNER), "--merge-only", "--output-dir", str(output_dir)],
        cwd=PROJECT_ROOT, check=True,
    )
    if failures:
        details = "\n".join(f"- {name}: exit={code}, log={path}" for name, code, path in failures)
        raise SystemExit(f"以下任务失败：\n{details}")


if __name__ == "__main__":
    main()
