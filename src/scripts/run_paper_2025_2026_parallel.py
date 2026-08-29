"""将二十篇论文、五数据集实验并发分配到本机可见 GPU。"""

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
RUNNER = PROJECT_ROOT / "src/scripts/run_paper_2025_2026_five_fold.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.paper_2025_2026_models import METHODS

DATASETS = (
    "MUTAG", "PROTEINS", "NCI1", "COLORS-3", "COLLAB",
    "PTC_MR", "Mutagenicity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, required=True)
    parser.add_argument("--jobs-per-device", type=int, default=3)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-free-memory-mib", type=int, default=2500)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tag", default="local")
    return parser.parse_args()


def free_memory() -> dict[int, int]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"
    ], text=True)
    return {int(line.split(",")[0]): int(line.split(",")[1]) for line in output.splitlines()}


def write_status(path: Path, queue: list[tuple[str, str]], running: dict, completed: int, failures: list) -> None:
    value = {
        "updated_at": datetime.now(timezone.utc).isoformat(), "queued": len(queue), "completed": completed,
        "running": {f"gpu{gpu}:slot{slot}": item[1] for (gpu, slot), item in running.items()},
        "failures": [{"task": name, "exit_code": code, "log": str(log)} for name, code, log in failures],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp"); temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"); temporary.replace(path)


def main() -> None:
    args = parse_args()
    visible = torch.cuda.device_count()
    invalid = [device for device in args.devices if not 0 <= device < visible]
    if invalid:
        raise SystemExit(f"请求的 GPU {invalid} 不可见；当前仅检测到 {visible} 张 GPU。")
    if args.jobs_per_device < 1:
        raise SystemExit("--jobs-per-device 必须大于等于 1。")
    queue = [(method, dataset) for method in args.methods for dataset in args.datasets]
    total = len(queue); running = {}; failures = []; completed = 0
    output_dir = PROJECT_ROOT / "outputs/paper_sota_2025_2026"
    log_dir = output_dir / "launcher_logs" / args.tag; log_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"launcher_status_{args.tag}.json"
    progress = tqdm(total=total, desc=f"{args.tag} 论文任务调度", unit="组")
    while queue or running:
        memory = free_memory()
        for device in args.devices:
            free_slots = [slot for slot in range(args.jobs_per_device) if (device, slot) not in running]
            for slot in free_slots:
                if not queue or memory.get(device, 0) < args.min_free_memory_mib:
                    break
                method, dataset = queue.pop(0); name = f"{method}/{dataset}"
                log_path = log_dir / f"{method}_{dataset}_gpu{device}_slot{slot}.log"
                command = [
                    sys.executable, str(RUNNER), "--methods", method, "--datasets", dataset,
                    "--device", f"cuda:{device}", "--epochs", str(args.epochs), "--patience", str(args.patience), "--no-merge",
                ]
                if args.force:
                    command.append("--force")
                handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy())
                process._paper_log_handle = handle  # type: ignore[attr-defined]
                running[(device, slot)] = (process, name, log_path)
        finished = []
        for device_slot, (process, name, log_path) in running.items():
            return_code = process.poll()
            if return_code is None:
                continue
            process._paper_log_handle.close()  # type: ignore[attr-defined]
            if return_code:
                failures.append((name, return_code, log_path))
            completed += 1; finished.append(device_slot); progress.update(1)
        for device_slot in finished:
            del running[device_slot]
        progress.set_postfix(等待=len(queue), 运行=len(running), 失败=len(failures)); write_status(status_path, queue, running, completed, failures)
        if queue or running:
            time.sleep(max(1, args.poll_seconds))
    progress.close()
    if failures:
        details = "\n".join(f"- {name}: exit={code}, log={log}" for name, code, log in failures)
        raise SystemExit(f"以下任务失败：\n{details}")


if __name__ == "__main__":
    main()
