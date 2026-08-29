"""将110个模块化单 idea ABC 任务分配到本机多张 GPU。"""

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
WORKER = PROJECT_ROOT / "src/scripts/run_modular_idea_global_tuning.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.research_ideas import IDEA_IDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, required=True)
    parser.add_argument("--jobs-per-device", type=int, default=2)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-indices", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--include-ideas",
        nargs="+",
        choices=IDEA_IDS,
        help="显式指定本 launcher 负责的 idea；设置后忽略分片选择",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("IDEA_PARA_MODULAR"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--min-free-memory-mib", type=int, default=4000)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--adopt-running",
        action="append",
        default=[],
        metavar="GPU:SLOT:PID:IDEA",
        help="接管重启前仍在运行的 worker，避免重复启动；可重复提供",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def gpu_free_memory() -> dict[int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {
        int(line.split(",")[0].strip()): int(line.split(",")[1].strip())
        for line in output.splitlines()
    }


class AdoptedProcess:
    """只读监控一个已存在的 worker PID。"""

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> int | None:
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return 0
        return None


def parse_adopted(args: argparse.Namespace) -> dict[tuple[int, int], tuple[int, str]]:
    adopted = {}
    for token in args.adopt_running:
        try:
            device_text, slot_text, pid_text, idea_id = token.split(":", 3)
            key = (int(device_text), int(slot_text))
            pid = int(pid_text)
        except ValueError as error:
            raise SystemExit(f"无效 --adopt-running：{token}") from error
        if key[0] not in args.devices or not 0 <= key[1] < args.jobs_per_device:
            raise SystemExit(f"接管槽位不属于当前 launcher：{token}")
        if idea_id not in IDEA_IDS or key in adopted:
            raise SystemExit(f"接管任务异常：{token}")
        try:
            os.kill(pid, 0)
        except ProcessLookupError as error:
            raise SystemExit(f"接管 PID 已不存在：{token}") from error
        adopted[key] = (pid, idea_id)
    return adopted


def write_status(path: Path, queue, running, completed, skipped, failures, total) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "queued": len(queue),
        "running_count": len(running),
        "completed": completed,
        "skipped_existing": skipped,
        "running": {
            f"gpu{gpu}:slot{slot}": item[1]
            for (gpu, slot), item in running.items()
        },
        "failures": [
            {"idea_id": idea_id, "exit_code": code, "log": str(log)}
            for idea_id, code, log in failures
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.jobs_per_device < 1:
        raise SystemExit("--jobs-per-device 必须大于0")
    if any(index < 0 or index >= args.shard_count for index in args.shard_indices):
        raise SystemExit("分片参数不合法")
    visible = torch.cuda.device_count()
    invalid = [device for device in args.devices if not 0 <= device < visible]
    if invalid:
        raise SystemExit(f"GPU {invalid} 不可见；当前仅检测到 {visible} 张 GPU")

    output_dir = resolve(args.output_dir)
    data_root = resolve(args.data_root)
    adopted_specs = parse_adopted(args)
    adopted_ideas = {idea_id for _, idea_id in adopted_specs.values()}
    if args.include_ideas:
        included = set(args.include_ideas)
        if len(included) != len(args.include_ideas):
            raise SystemExit("--include-ideas 中存在重复 ID")
        selected = [idea_id for idea_id in IDEA_IDS if idea_id in included]
    else:
        selected_shards = set(args.shard_indices)
        selected = [
            idea_id for index, idea_id in enumerate(IDEA_IDS)
            if index % args.shard_count in selected_shards
        ]
    queue = []
    skipped = 0
    for idea_id in selected:
        result_path = output_dir / "jobs" / f"{idea_id}.json"
        if result_path.exists() and not args.force and not args.smoke_test:
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                if payload.get("status") == "completed" and str(payload.get("protocol", "")).startswith("modular_"):
                    skipped += 1
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        if idea_id not in adopted_ideas:
            queue.append(idea_id)

    total = len(selected)
    running = {
        key: (
            AdoptedProcess(pid),
            idea_id,
            output_dir / "launcher_logs" / args.tag / f"adopted_{idea_id}.log",
            None,
            True,
        )
        for key, (pid, idea_id) in adopted_specs.items()
    }
    completed = 0
    failures = []
    log_dir = output_dir / "launcher_logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"launcher_status_{args.tag}.json"
    progress = tqdm(total=total, initial=skipped, desc=args.tag, unit="idea")

    while queue or running:
        free_memory = gpu_free_memory()
        for device in args.devices:
            for slot in range(args.jobs_per_device):
                key = (device, slot)
                if key in running or not queue:
                    continue
                if free_memory.get(device, 0) < args.min_free_memory_mib:
                    break
                idea_id = queue.pop(0)
                log_path = log_dir / f"{idea_id}__gpu{device}_slot{slot}.log"
                command = [
                    sys.executable,
                    str(WORKER),
                    "--idea", idea_id,
                    "--data-root", str(data_root),
                    "--output-dir", str(output_dir),
                    "--device", f"cuda:{device}",
                    "--epochs", str(args.epochs),
                    "--probe-epochs", str(args.probe_epochs),
                ]
                if args.force:
                    command.append("--force")
                if args.smoke_test:
                    command.append("--smoke-test")
                handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
                running[key] = (process, idea_id, log_path, handle, False)

        finished = []
        for key, (process, idea_id, log_path, handle, adopted) in running.items():
            code = process.poll()
            if code is None:
                continue
            if handle is not None:
                handle.close()
            result_path = output_dir / "jobs" / f"{idea_id}.json"
            if adopted and not result_path.exists():
                # 被接管进程若未形成完整 idea 结果，则放回队列断点续跑。
                queue.append(idea_id)
            elif code:
                failures.append((idea_id, code, log_path))
                progress.update(1)
            else:
                completed += 1
                progress.update(1)
            finished.append(key)
        for key in finished:
            del running[key]
        progress.set_postfix(等待=len(queue), 运行=len(running), 完成=completed, 失败=len(failures))
        write_status(status_path, queue, running, completed, skipped, failures, total)
        if queue or running:
            time.sleep(max(1, args.poll_seconds))
    progress.close()
    if failures:
        raise SystemExit(f"{len(failures)} 个 idea 失败；请查看 {status_path}")


if __name__ == "__main__":
    main()
