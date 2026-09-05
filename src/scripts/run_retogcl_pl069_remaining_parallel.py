#!/usr/bin/env python3
"""在多张 GPU 上补跑 CITA-GCL 尚未覆盖的十二个数据集。

保留 PL069 实验编号和历史输出路径，不因更名重复运行已完成任务。
"""

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
PROTOCOL = "retogcl_t077_adaptive_pipeline_v1"
CONFIG_ID = "PL069"
DATASETS = (
    "SIDER",
    "HIV",
    "COLLAB",
    "ABIDE",
    "NCI1",
    "PROTEINS",
    "MUTAG",
    "COLORS-3",
    "PTC_MR",
    "Mutagenicity",
    "ClinTox",
    "COX2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-device", type=int, default=1)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/retogcl_t077_adaptive_pipeline_100x6/manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retogcl_pl069_remaining_12x5fold"),
    )
    parser.add_argument("--min-free-memory-mib", type=int, default=11000)
    parser.add_argument("--max-utilization", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--tag", default="host202")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def gpu_state() -> dict[int, tuple[int, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    state = {}
    for line in output.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        state[index] = (memory, utilization)
    return state


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    library = str(Path(sys.executable).resolve().parent.parent / "lib")
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        f"{library}:{existing}" if existing else library
    )
    return environment


def is_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("status") == "completed"
        and result.get("protocol") == PROTOCOL
        and result.get("config_id") == CONFIG_ID
    )


def write_status(
    path: Path,
    total: int,
    queue: list[str],
    running: dict,
    completed: int,
    skipped: int,
    failures: list,
) -> None:
    value = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config_id": CONFIG_ID,
        "total": total,
        "queued": len(queue),
        "running_count": len(running),
        "completed": completed,
        "skipped": skipped,
        "running": {
            f"gpu{gpu}:slot{slot}": item[1]
            for (gpu, slot), item in running.items()
        },
        "failures": [
            {"dataset": dataset, "exit_code": code, "log": str(log)}
            for dataset, code, log in failures
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
    visible = torch.cuda.device_count()
    invalid = [device for device in args.devices if not 0 <= device < visible]
    if invalid:
        raise SystemExit(
            f"请求的 GPU {invalid} 不可见；当前仅检测到 {visible} 张 GPU。"
        )
    queue = []
    skipped = 0
    for dataset in DATASETS:
        result_path = args.output_dir / "jobs" / f"{CONFIG_ID}__{dataset}.json"
        if not args.force and is_completed(result_path):
            skipped += 1
        else:
            queue.append(dataset)

    log_dir = args.output_dir / "launcher_logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / f"launcher_status_{args.tag}.json"
    environment = child_environment()
    running = {}
    failures = []
    completed = 0
    progress = tqdm(
        total=len(DATASETS), initial=skipped, desc=f"CITA-GCL/{args.tag}", unit="数据集"
    )
    while queue or running:
        state = gpu_state()
        for slot in range(args.jobs_per_device):
            for device in args.devices:
                key = (device, slot)
                if key in running or not queue:
                    continue
                free_memory, utilization = state.get(device, (0, 100))
                if (
                    free_memory < args.min_free_memory_mib
                    or utilization > args.max_utilization
                ):
                    continue
                dataset = queue.pop(0)
                log_path = log_dir / f"{CONFIG_ID}__{dataset}_gpu{device}_slot{slot}.log"
                handle = log_path.open("w", encoding="utf-8")
                command = [
                    sys.executable,
                    str(
                        PROJECT_ROOT
                        / "src/scripts/run_retogcl_t077_adaptive_pipeline_five_fold.py"
                    ),
                    "--manifest",
                    str(args.manifest),
                    "--config-id",
                    CONFIG_ID,
                    "--dataset",
                    dataset,
                    "--data-root",
                    str(PROJECT_ROOT / "datasets/main_data"),
                    "--output-dir",
                    str(args.output_dir),
                    "--device",
                    f"cuda:{device}",
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
                running[key] = (process, dataset, log_path, handle)

        finished = []
        for key, (process, dataset, log_path, handle) in running.items():
            code = process.poll()
            if code is None:
                continue
            handle.close()
            if code:
                failures.append((dataset, code, log_path))
            else:
                completed += 1
            finished.append(key)
            progress.update(1)
        for key in finished:
            del running[key]
        progress.set_postfix(
            等待=len(queue),
            运行=len(running),
            完成=completed,
            跳过=skipped,
            失败=len(failures),
        )
        write_status(
            status_path,
            len(DATASETS),
            queue,
            running,
            completed,
            skipped,
            failures,
        )
        if queue or running:
            time.sleep(max(1, args.poll_seconds))
    progress.close()
    if failures:
        raise SystemExit(
            "存在失败任务：\n"
            + "\n".join(
                f"- {dataset}: exit={code}, log={log}"
                for dataset, code, log in failures
            )
        )


if __name__ == "__main__":
    main()
