#!/usr/bin/env python3
"""在指定 GPU 分片上并行运行一阶段 T077 Pipeline 实验。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("ADHD200", "BACE", "BBBP", "Tox21", "AIDS", "BZR")
PROTOCOL = "retogcl_t077_adaptive_pipeline_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-device", type=int, default=3)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-free-memory-mib", type=int, default=3500)
    parser.add_argument("--poll-seconds", type=int, default=3)
    parser.add_argument("--tag", default="t077_adaptive_pipeline")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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


def write_status(path: Path, total: int, queue: list, running: dict,
                 completed: int, skipped: int, failures: list) -> None:
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
                "config_id": config_id,
                "dataset": dataset,
                "exit_code": code,
                "log": str(log),
            }
            for config_id, dataset, code, log in failures
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
    args.manifest = resolve(args.manifest)
    args.output_dir = resolve(args.output_dir)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("manifest 协议不兼容")
    configs = [entry for entry in manifest["configs"] if entry["stage"] == args.stage]
    if len(configs) != 10:
        raise ValueError(f"阶段 {args.stage} 配置数不是 10")
    visible = torch.cuda.device_count()
    invalid = [device for device in args.devices if not 0 <= device < visible]
    if invalid:
        raise SystemExit(f"请求的 GPU {invalid} 不可见；当前仅检测到 {visible} 张 GPU。")

    selected = set(args.shard_indices)
    all_tasks = [
        (entry["config_id"], dataset)
        for config_index, entry in enumerate(configs)
        for dataset_index, dataset in enumerate(DATASETS)
        if (config_index + dataset_index) % args.shard_count in selected
    ]
    queue = []
    skipped = 0
    for config_id, dataset in all_tasks:
        path = args.output_dir / "jobs" / f"{config_id}__{dataset}.json"
        if path.exists() and not args.force:
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                result = {}
            if result.get("status") == "completed" and result.get("protocol") == PROTOCOL:
                skipped += 1
                continue
        queue.append((config_id, dataset))
    # 先启动需要 12 个标签任务分别五折评测的 Tox21，避免长任务在队尾集中到
    # 同一张卡，导致其他 GPU 已空闲而整轮仍等待尾项。
    queue.sort(key=lambda item: (item[1] != "Tox21", item[0], item[1]))

    total = len(all_tasks)
    log_dir = args.output_dir / "launcher_logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / f"launcher_status_{args.tag}.json"
    environment = child_environment()
    running = {}
    failures = []
    completed = 0
    progress = tqdm(total=total, initial=skipped, desc=args.tag, unit="任务")
    while queue or running:
        memory = free_memory()
        # 按 slot 后按 GPU 轮转，确保排在队首的长任务不会先填满同一张卡。
        for slot in range(args.jobs_per_device):
            for device in args.devices:
                key = (device, slot)
                if key in running or not queue:
                    continue
                if memory.get(device, 0) < args.min_free_memory_mib:
                    break
                config_id, dataset = queue.pop(0)
                log_path = log_dir / f"{config_id}__{dataset}_gpu{device}_slot{slot}.log"
                handle = log_path.open("w", encoding="utf-8")
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "src/scripts/run_retogcl_t077_adaptive_pipeline_five_fold.py"),
                    "--manifest", str(args.manifest),
                    "--config-id", config_id,
                    "--dataset", dataset,
                    "--data-root", str(PROJECT_ROOT / "datasets/main_data"),
                    "--output-dir", str(args.output_dir),
                    "--device", f"cuda:{device}",
                ]
                if args.force:
                    command.append("--force")
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                running[key] = (process, config_id, dataset, log_path, handle)

        finished = []
        for key, (process, config_id, dataset, log_path, handle) in running.items():
            code = process.poll()
            if code is None:
                continue
            handle.close()
            if code:
                failures.append((config_id, dataset, code, log_path))
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
            f"- {config_id}/{dataset}: exit={code}, log={log}"
            for config_id, dataset, code, log in failures
        ))


if __name__ == "__main__":
    main()
