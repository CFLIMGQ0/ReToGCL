#!/usr/bin/env python3
"""并行运行六个正式数据集上的九个经典直接基线。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.run_new_medical5_selected_parallel import ORIGINAL_ID, ORIGINAL_TEMPLATE


DATASETS = ("ADHD200", "BACE", "BBBP", "Tox21", "AIDS", "BZR")
METHODS = (
    "random_gin_linear", "supervised_gin", "graphcl", "joaov2", "rgcl", "simgrace",
    "balancegcl", "retogcl", "pl069",
)


@dataclass(frozen=True)
class Task:
    method: str
    dataset: str

    @property
    def task_id(self) -> str:
        return f"{self.method}/{self.dataset}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-device", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--output-root", type=Path, default=Path("outputs/classic_direct_baselines_9x6_20260904"))
    parser.add_argument("--min-free-memory-mib", type=int, default=6500)
    parser.add_argument("--max-utilization", type=int, default=90)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--tag", default="host202")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def shard(task: Task, count: int) -> int:
    return int.from_bytes(hashlib.sha256(task.task_id.encode()).digest()[:8], "big") % count


def result_path(task: Task, root: Path) -> Path:
    if task.method in {"random_gin_linear", "supervised_gin", "graphcl", "joaov2", "rgcl", "simgrace"}:
        return root / f"classic/jobs/{task.method}_{task.dataset}.json"
    if task.method == "balancegcl":
        return root / f"sota/jobs/balancegcl_{task.dataset}.json"
    if task.method == "retogcl":
        return root / f"retogcl/jobs/{ORIGINAL_ID}__{task.dataset}.json"
    return root / f"pl069/jobs/PL069__{task.dataset}.json"


def write_manifest(root: Path) -> Path:
    path = root / "retogcl_manifest.json"
    value = {
        "metadata": {"protocol": "modular_cross_module_triple_from_top5_pairwise_fixed_ABC_v1", "description": "经典直接基线中的 ReToGCL", "folds": 5, "seed": 42},
        "tasks": [{**ORIGINAL_TEMPLATE, "task_id": f"{ORIGINAL_ID}__{dataset}", "dataset": dataset} for dataset in DATASETS],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def command(task: Task, root: Path, manifest: Path, device: int, force: bool) -> list[str]:
    common = ["--data-root", str(PROJECT_ROOT / "datasets/main_data"), "--device", f"cuda:{device}", "--seed", "42", "--split-seed", "42"]
    if task.method in {"random_gin_linear", "supervised_gin", "graphcl", "joaov2", "rgcl", "simgrace"}:
        result = [sys.executable, str(PROJECT_ROOT / "src/scripts/run_classic_baseline_five_fold.py"), "--method", task.method, "--dataset", task.dataset, "--output-dir", str(root / "classic"), *common]
    elif task.method == "balancegcl":
        result = [sys.executable, str(PROJECT_ROOT / "src/scripts/run_sota_five_fold.py"), "--methods", "balancegcl", "--datasets", task.dataset, "--output-dir", str(root / "sota"), "--no-merge", *common]
    elif task.method == "retogcl":
        result = [sys.executable, str(PROJECT_ROOT / "src/scripts/run_modular_idea_triple_trial.py"), "--manifest", str(manifest), "--task-id", f"{ORIGINAL_ID}__{task.dataset}", "--output-dir", str(root / "retogcl"), *common]
    else:
        result = [sys.executable, str(PROJECT_ROOT / "src/scripts/run_retogcl_t077_adaptive_pipeline_five_fold.py"), "--manifest", str(PROJECT_ROOT / "outputs/retogcl_t077_adaptive_pipeline_100x6/manifest.json"), "--config-id", "PL069", "--dataset", task.dataset, "--output-dir", str(root / "pl069"), *common]
    if force:
        result.append("--force")
    return result


def gpu_state() -> dict[int, tuple[int, int]]:
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    return {int(parts[0]): (int(parts[1]), int(parts[2])) for line in output.splitlines() if (parts := [value.strip() for value in line.split(",")])}


def child_environment() -> dict[str, str]:
    result = os.environ.copy()
    library = str(Path(sys.executable).resolve().parent.parent / "lib")
    result["LD_LIBRARY_PATH"] = f"{library}:{result['LD_LIBRARY_PATH']}" if result.get("LD_LIBRARY_PATH") else library
    return result


def complete(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return all(metric in value for metric in ("accuracy_percent", "nmi_percent", "ari_percent", "macro_f1_percent"))
    except Exception:
        return False


def save_status(path: Path, total: int, queue: list[Task], running: dict, completed: int, skipped: int, failures: list) -> None:
    value = {
        "updated_at": datetime.now(timezone.utc).isoformat(), "total": total, "queued": len(queue),
        "running_count": len(running), "completed_this_run": completed, "skipped_existing": skipped,
        "running": {f"gpu{key[0]}:slot{key[1]}": item[1].task_id for key, item in running.items()},
        "failures": [{"task": task.task_id, "exit_code": code, "log": str(log)} for task, code, log in failures],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if any(device < 0 or device >= torch.cuda.device_count() for device in args.devices):
        raise SystemExit(f"GPU 参数无效；当前可见 {torch.cuda.device_count()} 张卡")
    root = args.output_root if args.output_root.is_absolute() else (PROJECT_ROOT / args.output_root).resolve()
    manifest = write_manifest(root)
    selected = set(args.shard_indices)
    queue = [task for task in (Task(method, dataset) for method in METHODS for dataset in DATASETS) if shard(task, args.shard_count) in selected]
    total = len(queue)
    running, failures = {}, []
    completed = skipped = 0
    log_dir = root / "launcher_logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    status = root / f"launcher_status_{args.tag}.json"
    env = child_environment()
    progress = tqdm(total=total, desc=f"{args.tag} 经典直接基线", unit="任务")
    while queue or running:
        state = gpu_state()
        for device in args.devices:
            for slot in range(args.jobs_per_device):
                key = (device, slot)
                if key in running or not queue:
                    continue
                free, utilization = state[device]
                if free < args.min_free_memory_mib or utilization > args.max_utilization:
                    continue
                task = queue.pop(0)
                path = result_path(task, root)
                if not args.force and complete(path):
                    skipped += 1
                    progress.update(1)
                    continue
                log_path = log_dir / f"{task.method}__{task.dataset}.log"
                stream = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(command(task, root, manifest, device, args.force), cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
                running[key] = (process, task, log_path, stream)
                state[device] = (max(0, free - args.min_free_memory_mib), utilization)
        for key, (process, task, log_path, stream) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del running[key]
            completed += int(code == 0)
            if code != 0:
                failures.append((task, code, log_path))
            progress.update(1)
        save_status(status, total, queue, running, completed, skipped, failures)
        if queue or running:
            time.sleep(args.poll_seconds)
    progress.close()
    save_status(status, total, queue, running, completed, skipped, failures)
    if failures:
        raise SystemExit(f"有 {len(failures)} 个任务失败，详见 {status}")


if __name__ == "__main__":
    main()
