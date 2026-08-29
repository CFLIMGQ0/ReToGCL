"""把十论文、五数据集的实验组合分配到可用 GPU，并支持等待繁忙 GPU。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "src/scripts/run_paper_five_fold.py"
METHODS = (
    "maxcutpool", "simplicial_mp", "del", "cellclat", "plstm",
    "abstaingnn", "edgeprompt", "dualprism", "invgnn", "genvsexp",
)
DATASETS = (
    "MUTAG", "PROTEINS", "NCI1", "COLORS-3", "COLLAB",
    "PTC_MR", "Mutagenicity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--jobs-per-device", type=int, default=1,
                        help="每张 GPU 同时运行的任务数")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wait-for-free", action="store_true")
    parser.add_argument("--max-memory-used-mib", type=int, default=2000)
    parser.add_argument("--max-utilization", type=int, default=15)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def gpu_stats() -> dict[int, tuple[int, int]]:
    command = [
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    stats = {}
    for line in output.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        stats[index] = (memory, utilization)
    return stats


def write_status(
    path: Path,
    queue: list[tuple[str, str]],
    running: dict[tuple[int, int], tuple],
    completed: int,
    failures: list,
) -> None:
    value = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "queued": len(queue),
        "running": {
            f"gpu{device}:slot{slot}": name
            for (device, slot), (_, name, _) in running.items()
        },
        "completed": completed,
        "failures": [{"task": name, "exit_code": code, "log": str(log)} for name, code, log in failures],
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.jobs_per_device < 1:
        raise SystemExit("--jobs-per-device 必须大于等于 1。")
    visible = torch.cuda.device_count()
    invalid = [device for device in args.devices if device < 0 or device >= visible]
    if invalid:
        raise SystemExit(f"请求的 GPU {invalid} 不可见；当前仅检测到 {visible} 张 GPU。")

    queue = [(method, dataset) for method in args.methods for dataset in args.datasets]
    running: dict[tuple[int, int], tuple[subprocess.Popen, str, Path]] = {}
    failures: list[tuple[str, int, Path]] = []
    completed = 0
    output_dir = PROJECT_ROOT / "outputs/paper_sota"
    log_dir = output_dir / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "launcher_status.json"
    progress = tqdm(total=len(queue), desc="论文模型任务调度", unit="组")

    while queue or running:
        stats = gpu_stats()
        for device in args.devices:
            device_slots = [slot for gpu, slot in running if gpu == device]
            if not queue or len(device_slots) >= args.jobs_per_device:
                continue
            if not device_slots:
                memory, utilization = stats[device]
                if args.wait_for_free and (
                    memory > args.max_memory_used_mib or utilization > args.max_utilization
                ):
                    continue

            free_slots = [
                slot for slot in range(args.jobs_per_device)
                if (device, slot) not in running
            ]
            for slot in free_slots:
                if not queue:
                    break
                method, dataset = queue.pop(0)
                name = f"{method}/{dataset}"
                log_path = log_dir / f"{method}_{dataset}_gpu{device}_slot{slot}.log"
                command = [
                    sys.executable, str(RUNNER), "--methods", method, "--datasets", dataset,
                    "--device", f"cuda:{device}", "--probe-epochs", str(args.probe_epochs), "--no-merge",
                ]
                if args.epochs is not None:
                    command.extend(["--epochs", str(args.epochs)])
                if args.force:
                    command.append("--force")
                log_handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, stdout=log_handle, stderr=subprocess.STDOUT
                )
                process._xmlg_log_handle = log_handle  # type: ignore[attr-defined]
                running[(device, slot)] = (process, name, log_path)

        finished = []
        for device_slot, (process, name, log_path) in running.items():
            return_code = process.poll()
            if return_code is None:
                continue
            process._xmlg_log_handle.close()  # type: ignore[attr-defined]
            if return_code:
                failures.append((name, return_code, log_path))
            completed += 1
            finished.append(device_slot)
            progress.update(1)
        for device_slot in finished:
            del running[device_slot]
        progress.set_postfix(等待=len(queue), 运行=len(running), 失败=len(failures))
        write_status(status_path, queue, running, completed, failures)
        if queue or running:
            time.sleep(max(1, args.poll_seconds))

    progress.close()
    subprocess.run([sys.executable, str(RUNNER), "--merge-only"], cwd=PROJECT_ROOT, check=True)
    if failures:
        details = "\n".join(f"- {name}: exit={code}, log={log}" for name, code, log in failures)
        raise SystemExit(f"以下任务失败：\n{details}")


if __name__ == "__main__":
    main()
