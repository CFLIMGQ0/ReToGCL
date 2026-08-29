"""在多 GPU 上调度 MUTAG、PTC_MR、Mutagenicity 的全部模型实验。"""

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
DATASETS = ("MUTAG", "PTC_MR", "Mutagenicity")
BASELINES = ("graphcl", "joaov2", "rgcl", "simgrace")
SOTA_CORE = ("histograph", "uniimb", "difflift", "balancegcl", "khangcl")
PAPER_SOTA = (
    "maxcutpool", "simplicial_mp", "del", "cellclat", "plstm",
    "abstaingnn", "edgeprompt", "dualprism", "invgnn", "genvsexp",
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.paper_2025_2026_models import METHODS as PAPER_2025_2026
from src.models.research_ideas import IDEA_IDS


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
    parser.add_argument("--devices", nargs="+", type=int, required=True)
    parser.add_argument("--jobs-per-device", type=int, default=3)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--categories", nargs="+",
        choices=("baselines", "sota_core", "paper_sota", "paper_2025_2026", "ideas"),
        default=["baselines", "sota_core", "paper_sota", "paper_2025_2026", "ideas"],
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/three_datasets_20260812"))
    parser.add_argument("--tag", default="local")
    parser.add_argument("--poll-seconds", type=int, default=2)
    parser.add_argument("--min-free-memory-mib", type=int, default=2500)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_tasks(args: argparse.Namespace) -> list[Task]:
    groups = {
        "baselines": BASELINES,
        "sota_core": SOTA_CORE,
        "paper_sota": PAPER_SOTA,
        "paper_2025_2026": tuple(PAPER_2025_2026),
        "ideas": tuple(IDEA_IDS),
    }
    all_tasks = [
        Task(category, method, dataset)
        for category in args.categories
        for method in groups[category]
        for dataset in args.datasets
    ]
    selected = set(args.shard_indices)
    return [task for index, task in enumerate(all_tasks) if index % args.shard_count in selected]


def result_path(task: Task, output_root: Path) -> Path:
    if task.category == "baselines":
        return output_root / "baselines" / "jobs" / f"{task.method}_{task.dataset}" / "five_fold_results.json"
    directory = {
        "sota_core": "sota",
        "paper_sota": "paper_sota",
        "paper_2025_2026": "paper_sota_2025_2026",
        "ideas": "research_ideas",
    }[task.category]
    key = task.method if task.category != "ideas" else task.method
    return output_root / directory / "jobs" / f"{key}_{task.dataset}.json"


def command_for(task: Task, output_root: Path, device: int, force: bool) -> list[str]:
    category_config = {
        "sota_core": ("run_sota_five_fold.py", "sota", "--methods"),
        "paper_sota": ("run_paper_five_fold.py", "paper_sota", "--methods"),
        "paper_2025_2026": (
            "run_paper_2025_2026_five_fold.py", "paper_sota_2025_2026", "--methods"
        ),
        "ideas": ("run_research_ideas.py", "research_ideas", "--ideas"),
    }
    if task.category == "baselines":
        job_output = output_root / "baselines" / "jobs" / f"{task.method}_{task.dataset}"
        command = [
            sys.executable, str(PROJECT_ROOT / "src/scripts/run_gcl_baselines.py"),
            "--methods", task.method, "--datasets", task.dataset,
            "--device", f"cuda:{device}", "--output-dir", str(job_output),
        ]
    else:
        runner, directory, method_flag = category_config[task.category]
        command = [
            sys.executable, str(PROJECT_ROOT / "src/scripts" / runner),
            method_flag, task.method, "--datasets", task.dataset,
            "--device", f"cuda:{device}",
            "--output-dir", str(output_root / directory), "--no-merge",
        ]
    if force:
        command.append("--force")
    return command


def free_memory() -> dict[int, int]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"
    ], text=True)
    return {int(line.split(",")[0]): int(line.split(",")[1]) for line in output.splitlines()}


def write_status(path: Path, queue: list[Task], running: dict, completed: int, skipped: int, failures: list) -> None:
    value = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "queued": len(queue), "running_count": len(running),
        "completed": completed, "skipped_existing": skipped,
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
    if args.jobs_per_device < 1 or args.shard_count < 1:
        raise SystemExit("jobs-per-device 和 shard-count 必须至少为 1。")
    if any(index < 0 or index >= args.shard_count for index in args.shard_indices):
        raise SystemExit("shard-indices 超出合法范围。")
    visible = torch.cuda.device_count()
    invalid = [device for device in args.devices if not 0 <= device < visible]
    if invalid:
        raise SystemExit(f"请求的 GPU {invalid} 不可见；当前仅检测到 {visible} 张 GPU。")

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    queue = build_tasks(args)
    total = len(queue)
    running: dict[tuple[int, int], tuple[subprocess.Popen, Task, Path, object]] = {}
    failures: list[tuple[Task, int, Path]] = []
    completed = skipped = 0
    log_dir = output_root / "launcher_logs" / args.tag
    status_path = output_root / f"launcher_status_{args.tag}.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=total, desc=f"{args.tag} 三数据集总调度", unit="任务")

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
                    cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT,
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
            finished.append(key)
            progress.update(1)
        for key in finished:
            del running[key]
        progress.set_postfix(等待=len(queue), 运行=len(running), 完成=completed, 失败=len(failures))
        write_status(status_path, queue, running, completed, skipped, failures)
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
