"""在本机可见 GPU 上并行调度 110 个 idea 的参数消融。"""

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.idea_parameter_ablation import PARAMETER_CONFIGS
from src.models.research_ideas import IDEA_IDS

RUNNER = PROJECT_ROOT / "src/scripts/run_idea_parameter_ablation.py"
DATASETS = ("MUTAG", "PTC_MR", "Mutagenicity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, required=True)
    parser.add_argument("--jobs-per-device", type=int, default=3)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--ideas", nargs="+", choices=IDEA_IDS, default=list(IDEA_IDS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output-dir", type=Path, default=Path("IDEA_PARA"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--min-free-memory-mib", type=int, default=2500)
    parser.add_argument("--poll-seconds", type=int, default=2)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def gpu_free_memory() -> dict[int, int]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"
    ], text=True)
    return {int(line.split(",")[0]): int(line.split(",")[1]) for line in output.splitlines()}


def write_status(path: Path, queue: list, running: dict, completed: int, skipped: int, failures: list) -> None:
    value = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "queued": len(queue), "running_count": len(running),
        "completed": completed, "skipped_existing": skipped,
        "running": {
            f"gpu{gpu}:slot{slot}": "/".join(item[1])
            for (gpu, slot), item in running.items()
        },
        "failures": [
            {"task": "/".join(task), "exit_code": code, "log": str(log)}
            for task, code, log in failures
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.jobs_per_device < 1 or args.shard_count < 1:
        raise SystemExit("jobs-per-device 和 shard-count 必须至少为 1")
    if any(index < 0 or index >= args.shard_count for index in args.shard_indices):
        raise SystemExit("shard-indices 超出合法范围")
    visible = torch.cuda.device_count()
    invalid = [device for device in args.devices if not 0 <= device < visible]
    if invalid:
        raise SystemExit(f"请求的 GPU {invalid} 不可见；当前仅检测到 {visible} 张 GPU。")
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    all_tasks = [
        (idea, dataset, config.config_id)
        for idea in args.ideas for dataset in args.datasets for config in PARAMETER_CONFIGS
    ]
    selected = set(args.shard_indices)
    queue = [task for index, task in enumerate(all_tasks) if index % args.shard_count in selected]
    total = len(queue); running = {}; failures = []; completed = skipped = 0
    log_dir = output_dir / "launcher_logs" / args.tag; log_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"launcher_status_{args.tag}.json"
    progress = tqdm(total=total, desc=f"{args.tag} idea 参数消融", unit="任务")
    while queue or running:
        memory = gpu_free_memory()
        for device in args.devices:
            for slot in range(args.jobs_per_device):
                key = (device, slot)
                if key in running or not queue:
                    continue
                if memory.get(device, 0) < args.min_free_memory_mib:
                    break
                task = queue.pop(0); idea, dataset, config_id = task
                result = output_dir / "jobs" / f"{idea}_{dataset}_{config_id}.json"
                if result.exists() and not args.force:
                    skipped += 1; progress.update(1); continue
                log_path = log_dir / f"{idea}_{dataset}_{config_id}_gpu{device}_slot{slot}.log"
                command = [
                    sys.executable, str(RUNNER), "--ideas", idea, "--datasets", dataset,
                    "--config-ids", config_id, "--device", f"cuda:{device}",
                    "--output-dir", str(output_dir), "--epochs", str(args.epochs),
                    "--probe-epochs", str(args.probe_epochs), "--no-merge",
                ]
                if args.force:
                    command.append("--force")
                handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT,
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
            finished.append(key); progress.update(1)
        for key in finished:
            del running[key]
        progress.set_postfix(等待=len(queue), 运行=len(running), 完成=completed, 失败=len(failures))
        write_status(status_path, queue, running, completed, skipped, failures)
        if queue or running:
            time.sleep(max(1, args.poll_seconds))
    progress.close()
    if failures:
        details = "\n".join(f"- {'/'.join(task)}: exit={code}, log={log}" for task, code, log in failures)
        raise SystemExit(f"以下参数消融任务失败：\n{details}")


if __name__ == "__main__":
    main()
