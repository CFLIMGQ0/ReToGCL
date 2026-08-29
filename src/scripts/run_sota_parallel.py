"""将 SOTA 模型/数据集组合分配到多张 GPU 并行执行。"""

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
RUNNER = PROJECT_ROOT / "src/scripts/run_sota_five_fold.py"
METHODS = ("histograph", "uniimb", "difflift", "balancegcl", "khangcl")
DATASETS = (
    "NCI1", "PROTEINS", "COLLAB", "MUTAG", "COLORS-3",
    "PTC_MR", "Mutagenicity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible_count = torch.cuda.device_count()
    invalid = [device for device in args.devices if device < 0 or device >= visible_count]
    if invalid:
        raise SystemExit(
            f"请求的 GPU {invalid} 不可见；当前仅检测到 {visible_count} 张 GPU。"
        )

    queue = [(method, dataset) for method in args.methods for dataset in args.datasets]
    running: dict[int, tuple[subprocess.Popen, str, Path]] = {}
    log_dir = PROJECT_ROOT / "outputs/sota/launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=len(queue), desc="四卡任务调度", unit="组")
    failures = []

    while queue or running:
        for device in args.devices:
            if device in running or not queue:
                continue
            method, dataset = queue.pop(0)
            log_path = log_dir / f"{method}_{dataset}_gpu{device}.log"
            command = [
                sys.executable, str(RUNNER),
                "--methods", method,
                "--datasets", dataset,
                "--device", f"cuda:{device}",
                "--probe-epochs", str(args.probe_epochs),
                "--no-merge",
            ]
            if args.epochs is not None:
                command.extend(["--epochs", str(args.epochs)])
            if args.force:
                command.append("--force")
            if args.refresh_metrics:
                command.append("--refresh-metrics")
            log_handle = log_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, stdout=log_handle,
                stderr=subprocess.STDOUT, env=environment,
            )
            process._xmlg_log_handle = log_handle  # type: ignore[attr-defined]
            running[device] = (process, f"{method}/{dataset}", log_path)

        completed = []
        for device, (process, name, log_path) in running.items():
            return_code = process.poll()
            if return_code is None:
                continue
            process._xmlg_log_handle.close()  # type: ignore[attr-defined]
            if return_code:
                failures.append((name, return_code, log_path))
            completed.append(device)
            progress.update(1)
            progress.set_postfix(running=len(running) - len(completed), failed=len(failures))
        for device in completed:
            del running[device]
        if running and not completed:
            time.sleep(1)

    progress.close()
    subprocess.run(
        [sys.executable, str(RUNNER), "--merge-only"],
        cwd=PROJECT_ROOT, check=True,
    )
    if failures:
        details = "\n".join(
            f"- {name}: exit={code}, log={path}" for name, code, path in failures
        )
        raise SystemExit(f"以下任务失败：\n{details}")


if __name__ == "__main__":
    main()
