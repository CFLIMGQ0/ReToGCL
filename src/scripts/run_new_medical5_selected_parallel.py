#!/usr/bin/env python3
"""并行运行 5 个新增 TDC 数据集上的 ReToGCL、CITA-GCL 与 13 个选定 SOTA。

CITA-GCL 的内部实验编号仍为 PL069，结果路径与历史任务保持兼容。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
DATASETS = ("hERG_Karim", "CYP2D6_Veith", "CYP3A4_Veith", "Pgp_Broccatelli", "DILI")
ORIGINAL_ID = "D7-I10__D7-I01__D10-I04"
ORIGINAL_TEMPLATE = {
    "triple_id": ORIGINAL_ID,
    "idea_ids": ["D7-I10", "D7-I01", "D10-I04"],
    "modules": ["M1", "M4", "M7"],
    "idea_parameters": [
        {"adversarial_augmentation_budget": 0.1, "minimum_clean_ratio": 0.6, "bilevel_update_interval": 2.0},
        {"cleanliness_beta_prior_strength": 0.5, "clean_probability_floor": 0.05, "topology_evidence_ratio": 0.5},
        {"strata_count": 16.0, "bridge_weight": 0.5, "tangent_dimension": 32.0},
    ],
    "source_top_pairs": ["D7-I10__D10-I04"],
}
METHODS = (
    ("sota", "balancegcl", "run_sota_five_fold.py"),
    ("sota", "khangcl", "run_sota_five_fold.py"),
    ("paper_sota", "cellclat", "run_paper_five_fold.py"),
    ("sota", "uniimb", "run_sota_five_fold.py"),
    ("paper_sota", "dualprism", "run_paper_five_fold.py"),
    ("paper_sota", "del", "run_paper_five_fold.py"),
    ("paper_sota_2025_2026", "spectre", "run_paper_2025_2026_five_fold.py"),
    ("paper_sota_2025_2026", "toper", "run_paper_2025_2026_five_fold.py"),
    ("paper_sota_2025_2026", "leap", "run_paper_2025_2026_five_fold.py"),
    ("paper_sota_2025_2026", "hourglass", "run_paper_2025_2026_five_fold.py"),
    ("paper_sota_2025_2026", "nodeid", "run_paper_2025_2026_five_fold.py"),
    ("paper_sota_2025_2026", "gnnplus", "run_paper_2025_2026_five_fold.py"),
    ("paper_sota_2025_2026", "rspool", "run_paper_2025_2026_five_fold.py"),
)


@dataclass(frozen=True)
class Task:
    category: str
    method: str
    dataset: str
    runner: str

    @property
    def task_id(self) -> str:
        return f"{self.category}/{self.method}/{self.dataset}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-device", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--output-root", type=Path, default=Path("outputs/new_medical5_selected15_20260904"))
    parser.add_argument("--min-free-memory-mib", type=int, default=7000)
    parser.add_argument("--max-utilization", type=int, default=85)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--tag", default="host202")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def all_tasks() -> list[Task]:
    tasks = [Task("retogcl", "original", dataset, "run_modular_idea_triple_trial.py") for dataset in DATASETS]
    tasks += [Task("pl069", "PL069", dataset, "run_retogcl_t077_adaptive_pipeline_five_fold.py") for dataset in DATASETS]
    tasks += [Task(category, method, dataset, runner) for category, method, runner in METHODS for dataset in DATASETS]
    return tasks


def result_path(task: Task, output_root: Path) -> Path:
    if task.category == "retogcl":
        return output_root / "retogcl" / "jobs" / f"{ORIGINAL_ID}__{task.dataset}.json"
    if task.category == "pl069":
        return output_root / "pl069" / "jobs" / f"PL069__{task.dataset}.json"
    return output_root / task.category / "jobs" / f"{task.method}_{task.dataset}.json"


def write_original_manifest(output_root: Path) -> Path:
    path = output_root / "retogcl_manifest.json"
    tasks = []
    for dataset in DATASETS:
        tasks.append({
            **ORIGINAL_TEMPLATE,
            "task_id": f"{ORIGINAL_ID}__{dataset}",
            "dataset": dataset,
        })
    value = {
        "metadata": {
            "protocol": "modular_cross_module_triple_from_top5_pairwise_fixed_ABC_v1",
            "description": "ReToGCL 原始模型在 5 个新增 TDC 医学数据集上的统一五折清单",
            "folds": 5,
            "seed": 42,
        },
        "tasks": tasks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def command(task: Task, output_root: Path, original_manifest: Path, device: int, force: bool) -> list[str]:
    common = ["--data-root", str(PROJECT_ROOT / "datasets/main_data"), "--device", f"cuda:{device}"]
    if task.category == "retogcl":
        result = [
            sys.executable, str(PROJECT_ROOT / "src/scripts" / task.runner),
            "--manifest", str(original_manifest),
            "--task-id", f"{ORIGINAL_ID}__{task.dataset}",
            "--output-dir", str(output_root / "retogcl"),
            *common,
        ]
    elif task.category == "pl069":
        result = [
            sys.executable, str(PROJECT_ROOT / "src/scripts" / task.runner),
            "--manifest", str(PROJECT_ROOT / "outputs/retogcl_t077_adaptive_pipeline_100x6/manifest.json"),
            "--config-id", "PL069", "--dataset", task.dataset,
            "--output-dir", str(output_root / "pl069"),
            *common,
        ]
    else:
        result = [
            sys.executable, str(PROJECT_ROOT / "src/scripts" / task.runner),
            "--methods", task.method, "--datasets", task.dataset,
            "--output-dir", str(output_root / task.category), "--no-merge",
            *common,
        ]
    if force:
        result.append("--force")
    return result


def gpu_state() -> dict[int, tuple[int, int]]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    state = {}
    for line in output.splitlines():
        index, free, utilization = (int(value.strip()) for value in line.split(","))
        state[index] = (free, utilization)
    return state


def environment() -> dict[str, str]:
    result = os.environ.copy()
    prefix = str(Path(sys.executable).resolve().parent.parent / "lib")
    current = result.get("LD_LIBRARY_PATH")
    result["LD_LIBRARY_PATH"] = f"{prefix}:{current}" if current else prefix
    return result


def save_status(path: Path, total: int, queue: list[Task], running: dict, completed: int, skipped: int, failures: list) -> None:
    value = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total": total, "queued": len(queue), "running_count": len(running),
        "completed_this_run": completed, "skipped_existing": skipped,
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
    output_root = args.output_root if args.output_root.is_absolute() else (PROJECT_ROOT / args.output_root).resolve()
    original_manifest = write_original_manifest(output_root)
    selected = set(args.shard_indices)
    queue = [task for index, task in enumerate(all_tasks()) if index % args.shard_count in selected]
    total = len(queue)
    running: dict[tuple[int, int], tuple[subprocess.Popen, Task, Path, object]] = {}
    failures = []
    completed = skipped = 0
    log_dir = output_root / "launcher_logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    status = output_root / f"launcher_status_{args.tag}.json"
    child_env = environment()
    progress = tqdm(total=total, desc=f"{args.tag} 新医学5数据集", unit="任务")
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
                path = result_path(task, output_root)
                if path.exists() and not args.force:
                    try:
                        if json.loads(path.read_text(encoding="utf-8")).get("status", "completed") == "completed":
                            skipped += 1
                            progress.update(1)
                            continue
                    except Exception:
                        pass
                log_path = log_dir / f"{task.category}__{task.method}__{task.dataset}.log"
                stream = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command(task, output_root, original_manifest, device, args.force),
                    cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, env=child_env,
                )
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
