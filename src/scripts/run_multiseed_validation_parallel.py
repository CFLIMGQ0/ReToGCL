#!/usr/bin/env python3
"""并行运行六模型、六数据集、十随机种子的固定五折复核。"""

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
METHODS = ("retogcl", "pl069", "balancegcl", "khangcl", "del", "simplicial_mp")


@dataclass(frozen=True)
class Task:
    method: str
    dataset: str
    seed: int

    @property
    def task_id(self) -> str:
        return f"{self.method}/{self.dataset}/seed_{self.seed:03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-device", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--output-root", type=Path, default=Path("outputs/multiseed_validation_6x6x10_20260904"))
    parser.add_argument("--min-free-memory-mib", type=int, default=6500)
    parser.add_argument("--max-utilization", type=int, default=90)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--tag", default="host202")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def tasks(seeds: list[int]) -> list[Task]:
    return [Task(method, dataset, seed) for seed in seeds for method in METHODS for dataset in DATASETS]


def shard(task: Task, count: int) -> int:
    digest = hashlib.sha256(task.task_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def seed_root(output_root: Path, seed: int) -> Path:
    return output_root / f"seed_{seed:03d}"


def result_path(task: Task, output_root: Path) -> Path:
    root = seed_root(output_root, task.seed)
    if task.method == "retogcl":
        return root / "retogcl/jobs" / f"{ORIGINAL_ID}__{task.dataset}.json"
    if task.method == "pl069":
        return root / "pl069/jobs" / f"PL069__{task.dataset}.json"
    if task.method in {"balancegcl", "khangcl"}:
        return root / "sota/jobs" / f"{task.method}_{task.dataset}.json"
    return root / "paper_sota/jobs" / f"{task.method}_{task.dataset}.json"


def write_manifest(output_root: Path) -> Path:
    path = output_root / "retogcl_manifest.json"
    value = {
        "metadata": {
            "protocol": "modular_cross_module_triple_from_top5_pairwise_fixed_ABC_v1",
            "description": "ReToGCL 十随机种子固定外层五折复核",
            "folds": 5,
        },
        "tasks": [
            {**ORIGINAL_TEMPLATE, "task_id": f"{ORIGINAL_ID}__{dataset}", "dataset": dataset}
            for dataset in DATASETS
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def command(task: Task, output_root: Path, manifest: Path, split_seed: int, device: int, force: bool) -> list[str]:
    root = seed_root(output_root, task.seed)
    common = [
        "--data-root", str(PROJECT_ROOT / "datasets/main_data"),
        "--device", f"cuda:{device}", "--seed", str(task.seed),
        "--split-seed", str(split_seed),
    ]
    if task.method == "retogcl":
        result = [
            sys.executable, str(PROJECT_ROOT / "src/scripts/run_modular_idea_triple_trial.py"),
            "--manifest", str(manifest), "--task-id", f"{ORIGINAL_ID}__{task.dataset}",
            "--output-dir", str(root / "retogcl"), *common,
        ]
    elif task.method == "pl069":
        result = [
            sys.executable, str(PROJECT_ROOT / "src/scripts/run_retogcl_t077_adaptive_pipeline_five_fold.py"),
            "--manifest", str(PROJECT_ROOT / "outputs/retogcl_t077_adaptive_pipeline_100x6/manifest.json"),
            "--config-id", "PL069", "--dataset", task.dataset,
            "--output-dir", str(root / "pl069"), *common,
        ]
    elif task.method in {"balancegcl", "khangcl"}:
        result = [
            sys.executable, str(PROJECT_ROOT / "src/scripts/run_sota_five_fold.py"),
            "--methods", task.method, "--datasets", task.dataset,
            "--output-dir", str(root / "sota"), "--no-merge", *common,
        ]
    else:
        result = [
            sys.executable, str(PROJECT_ROOT / "src/scripts/run_paper_five_fold.py"),
            "--methods", task.method, "--datasets", task.dataset,
            "--output-dir", str(root / "paper_sota"), "--no-merge", *common,
        ]
    if force:
        result.append("--force")
    return result


def gpu_state() -> dict[int, tuple[int, int]]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True)
    result = {}
    for line in output.splitlines():
        index, free, utilization = (int(value.strip()) for value in line.split(","))
        result[index] = (free, utilization)
    return result


def child_environment() -> dict[str, str]:
    result = os.environ.copy()
    library = str(Path(sys.executable).resolve().parent.parent / "lib")
    current = result.get("LD_LIBRARY_PATH")
    result["LD_LIBRARY_PATH"] = f"{library}:{current}" if current else library
    return result


def complete_result(path: Path, task: Task, split_seed: int) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        configuration = result.get("configuration", result.get("fixed_non_idea_parameters", {}))
        return (
            int(result.get("seed", configuration.get("seed", -1))) == task.seed
            and int(result.get("split_seed", configuration.get("split_seed", -1))) == split_seed
            and all(key in result for key in ("accuracy_percent", "nmi_percent", "ari_percent", "macro_f1_percent"))
        )
    except Exception:
        return False


def save_status(path: Path, total: int, queue: list[Task], running: dict, completed: int, skipped: int, failures: list) -> None:
    value = {
        "updated_at": datetime.now(timezone.utc).isoformat(), "total": total,
        "queued": len(queue), "running_count": len(running),
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
    manifest = write_manifest(output_root)
    selected = set(args.shard_indices)
    queue = [task for task in tasks(args.seeds) if shard(task, args.shard_count) in selected]
    total = len(queue)
    running = {}
    failures = []
    completed = skipped = 0
    log_dir = output_root / "launcher_logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    status = output_root / f"launcher_status_{args.tag}.json"
    env = child_environment()
    progress = tqdm(total=total, desc=f"{args.tag} 多种子复核", unit="任务")
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
                if not args.force and complete_result(path, task, args.split_seed):
                    skipped += 1
                    progress.update(1)
                    continue
                log_path = log_dir / f"{task.method}__{task.dataset}__seed{task.seed:03d}.log"
                stream = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command(task, output_root, manifest, args.split_seed, device, args.force),
                    cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env,
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
    subprocess.run([sys.executable, str(PROJECT_ROOT / "src/scripts/summarize_multiseed_validation.py"), "--output-root", str(output_root)], cwd=PROJECT_ROOT, env=env, check=False)
    if failures:
        raise SystemExit(f"有 {len(failures)} 个任务失败，详见 {status}")


if __name__ == "__main__":
    main()
