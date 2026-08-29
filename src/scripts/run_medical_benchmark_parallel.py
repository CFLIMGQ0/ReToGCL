"""并行运行论文主模型和 35 个 SOTA 的医学图数据集五折实验。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.paper_2025_2026_models import METHODS as RECENT_METHODS


DATASETS = (
    "ClinTox", "BACE", "BBBP", "ABIDE", "ADHD200", "SIDER", "HIV",
    "AIDS", "Tox21", "BZR", "COX2",
)
CORE_METHODS = ("histograph", "uniimb", "difflift", "balancegcl", "khangcl")
PAPER_METHODS = (
    "maxcutpool", "simplicial_mp", "del", "cellclat", "plstm",
    "abstaingnn", "edgeprompt", "dualprism", "invgnn", "genvsexp",
)
OUR_METHOD = "D7-I10__D7-I01__D10-I04"
OUR_TASK_TEMPLATE = {
    "triple_id": OUR_METHOD,
    "idea_ids": ["D7-I10", "D7-I01", "D10-I04"],
    "modules": ["M1", "M4", "M7"],
    "idea_parameters": [
        {"adversarial_augmentation_budget": 0.1, "minimum_clean_ratio": 0.6, "bilevel_update_interval": 2.0},
        {"cleanliness_beta_prior_strength": 0.5, "clean_probability_floor": 0.05, "topology_evidence_ratio": 0.5},
        {"strata_count": 16.0, "bridge_weight": 0.5, "tangent_dimension": 32.0},
    ],
    "source_top_pairs": ["D7-I10__D10-I04"],
}


@dataclass(frozen=True)
class Task:
    category: str
    method: str
    dataset: str

    @property
    def name(self) -> str:
        return f"{self.category}/{self.method}/{self.dataset}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs-per-device", type=int, default=3)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/medical_benchmark_20260828"),
    )
    parser.add_argument("--min-free-memory-mib", type=int, default=3000)
    parser.add_argument("--poll-seconds", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tag", default="host202")
    return parser.parse_args()


def build_tasks(datasets: list[str]) -> list[Task]:
    tasks = [Task("our_model", OUR_METHOD, dataset) for dataset in datasets]
    tasks.extend(Task("sota_core", method, dataset) for method in CORE_METHODS for dataset in datasets)
    tasks.extend(Task("paper_sota", method, dataset) for method in PAPER_METHODS for dataset in datasets)
    tasks.extend(Task("recent_sota", method, dataset) for method in RECENT_METHODS for dataset in datasets)
    return tasks


def result_path(task: Task, output_root: Path) -> Path:
    directory = {
        "our_model": "our_model",
        "sota_core": "sota",
        "paper_sota": "paper_sota",
        "recent_sota": "paper_sota_2025_2026",
    }[task.category]
    if task.category == "our_model":
        filename = f"{OUR_METHOD}__{task.dataset}.json"
    else:
        filename = f"{task.method}_{task.dataset}.json"
    return output_root / directory / "jobs" / filename


def command_for(task: Task, output_root: Path, device: int, force: bool) -> list[str]:
    if task.category == "our_model":
        task_payload = {
            **OUR_TASK_TEMPLATE,
            "task_id": f"{OUR_METHOD}__{task.dataset}",
            "dataset": task.dataset,
        }
        command = [
            sys.executable,
            str(PROJECT_ROOT / "src/scripts/run_modular_idea_triple_trial.py"),
            "--manifest", str(PROJECT_ROOT / "MEDICAL_BENCHMARK/manifest.json"),
            "--task-id", f"{OUR_METHOD}__{task.dataset}",
            "--task-json", json.dumps(task_payload, ensure_ascii=False, separators=(",", ":")),
            "--data-root", str(PROJECT_ROOT / "datasets/main_data"),
            "--output-dir", str(output_root / "our_model"),
            "--device", f"cuda:{device}",
        ]
    else:
        runner, directory = {
            "sota_core": ("run_sota_five_fold.py", "sota"),
            "paper_sota": ("run_paper_five_fold.py", "paper_sota"),
            "recent_sota": ("run_paper_2025_2026_five_fold.py", "paper_sota_2025_2026"),
        }[task.category]
        command = [
            sys.executable, str(PROJECT_ROOT / "src/scripts" / runner),
            "--methods", task.method,
            "--datasets", task.dataset,
            "--data-root", str(PROJECT_ROOT / "datasets/main_data"),
            "--output-dir", str(output_root / directory),
            "--device", f"cuda:{device}",
            "--no-merge",
        ]
    if force:
        command.append("--force")
    return command


def free_memory() -> dict[int, int]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits",
    ], text=True)
    return {int(line.split(",")[0]): int(line.split(",")[1]) for line in output.splitlines()}


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment_root = Path(sys.executable).resolve().parent.parent
    environment_library = str(environment_root / "lib")
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        f"{environment_library}:{existing}" if existing else environment_library
    )
    return environment


def write_status(path: Path, total: int, queue: list[Task], running: dict, completed: int, skipped: int, failures: list) -> None:
    value = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_model_dataset_tasks": total,
        "queued": len(queue),
        "running_count": len(running),
        "completed": completed,
        "skipped_existing": skipped,
        "running": {
            f"gpu{gpu}:slot{slot}": item[1].name
            for (gpu, slot), item in running.items()
        },
        "failures": [
            {"task": task.name, "exit_code": code, "log": str(log)}
            for task, code, log in failures
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    visible = torch.cuda.device_count()
    invalid = [device for device in args.devices if not 0 <= device < visible]
    if invalid:
        raise SystemExit(f"请求的 GPU {invalid} 不可见；当前仅检测到 {visible} 张 GPU。")
    if args.jobs_per_device < 1:
        raise SystemExit("--jobs-per-device 必须至少为 1。")
    if args.shard_count < 1:
        raise SystemExit("--shard-count 必须至少为 1。")
    if any(index < 0 or index >= args.shard_count for index in args.shard_indices):
        raise SystemExit("--shard-indices 超出合法范围。")

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    selected_shards = set(args.shard_indices)
    all_tasks = build_tasks(args.datasets)
    queue = [
        task for index, task in enumerate(all_tasks)
        if index % args.shard_count in selected_shards
    ]
    total = len(queue)
    running: dict[tuple[int, int], tuple[subprocess.Popen, Task, Path, object]] = {}
    failures: list[tuple[Task, int, Path]] = []
    completed = skipped = 0
    log_dir = output_root / "launcher_logs" / args.tag
    status_path = output_root / f"launcher_status_{args.tag}.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=total, desc=f"{args.tag} 医学基准", unit="任务")
    environment = child_environment()

    while queue or running:
        memory = free_memory()
        for device in args.devices:
            for slot in range(args.jobs_per_device):
                key = (device, slot)
                if key in running or not queue:
                    continue
                if memory.get(device, 0) < args.min_free_memory_mib:
                    break
                task = queue.pop(0)
                if result_path(task, output_root).exists() and not args.force:
                    skipped += 1
                    progress.update(1)
                    continue
                log_path = log_dir / f"{task.category}_{task.method}_{task.dataset}_gpu{device}_slot{slot}.log"
                handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command_for(task, output_root, device, args.force),
                    cwd=PROJECT_ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
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
            finished.append(key)
            progress.update(1)
        for key in finished:
            del running[key]
        progress.set_postfix(等待=len(queue), 运行=len(running), 完成=completed, 失败=len(failures))
        write_status(status_path, total, queue, running, completed, skipped, failures)
        if queue or running:
            time.sleep(max(1, args.poll_seconds))

    progress.close()
    if failures:
        details = "\n".join(
            f"- {task.name}: exit={code}, log={log}" for task, code, log in failures
        )
        raise SystemExit(f"以下任务失败：\n{details}")


if __name__ == "__main__":
    main()
