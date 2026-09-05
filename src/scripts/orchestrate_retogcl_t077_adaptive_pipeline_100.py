#!/usr/bin/env python3
"""在 202/204 六张 GPU 上逐轮闭环运行 100 个 T077 Pipeline。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "retogcl_t077_adaptive_pipeline_v1"
DATASETS = ("ADHD200", "BACE", "BBBP", "Tox21", "AIDS", "BZR")
SYNC_FILES = (
    "src/models/retogcl_t077_adaptive_pipeline.py",
    "src/scripts/prepare_retogcl_t077_adaptive_pipeline_stage.py",
    "src/scripts/run_retogcl_t077_adaptive_pipeline_five_fold.py",
    "src/scripts/run_retogcl_t077_adaptive_pipeline_parallel.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--end-stage", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retogcl_t077_adaptive_pipeline_100x6"),
    )
    parser.add_argument("--remote", default="xmlg204")
    parser.add_argument("--remote-root", default="/home/Lim/Project-xmlg")
    parser.add_argument("--jobs-per-device", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=15)
    return parser.parse_args()


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=PROJECT_ROOT, check=check, text=True)


def sync_code(args: argparse.Namespace) -> None:
    run(["rsync", "-aR", *SYNC_FILES, f"{args.remote}:{args.remote_root}/"])


def sync_output(args: argparse.Namespace, output_dir: Path, *, both: bool) -> None:
    relative = output_dir.relative_to(PROJECT_ROOT)
    remote_output = f"{args.remote_root}/{relative}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run(["ssh", args.remote, "mkdir", "-p", remote_output])
    if both:
        # 已完成 JSON 采用原子替换；双向同步可安全支持中断续跑。
        for directory in ("jobs", "checkpoints", "logs"):
            local = output_dir / directory
            local.mkdir(parents=True, exist_ok=True)
            run([
                "rsync", "-a", f"{args.remote}:{remote_output}/{directory}/",
                f"{local}/",
            ], check=False)
            run([
                "rsync", "-a", f"{local}/",
                f"{args.remote}:{remote_output}/{directory}/",
            ], check=False)


def ensure_stage(stage: int, output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    exists = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol") != PROTOCOL:
            raise ValueError("本地 manifest 协议不兼容")
        exists = any(entry["stage"] == stage for entry in manifest["configs"])
    if not exists:
        run([
            sys.executable,
            "src/scripts/prepare_retogcl_t077_adaptive_pipeline_stage.py",
            "--stage", str(stage),
            "--output-dir", str(output_dir),
        ])


def sync_manifest(args: argparse.Namespace, output_dir: Path) -> None:
    relative = output_dir.relative_to(PROJECT_ROOT)
    remote_output = f"{args.remote_root}/{relative}"
    run([
        "rsync", "-a", str(output_dir / "manifest.json"),
        f"{args.remote}:{remote_output}/manifest.json",
    ])


def result_count(output_dir: Path, stage: int) -> int:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = {
        entry["config_id"] for entry in manifest["configs"]
        if entry["stage"] == stage
    }
    count = 0
    for config_id in ids:
        for dataset in DATASETS:
            path = output_dir / "jobs" / f"{config_id}__{dataset}.json"
            if not path.exists():
                continue
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            count += int(
                result.get("status") == "completed"
                and result.get("protocol") == PROTOCOL
            )
    return count


def launch_stage(args: argparse.Namespace, output_dir: Path, stage: int) -> None:
    relative = output_dir.relative_to(PROJECT_ROOT)
    remote_output = f"{args.remote_root}/{relative}"
    local_log = output_dir / f"orchestrator_stage_{stage}_host202.log"
    remote_log = output_dir / f"orchestrator_stage_{stage}_host204_ssh.log"
    local_command = [
        sys.executable,
        "src/scripts/run_retogcl_t077_adaptive_pipeline_parallel.py",
        "--manifest", str(output_dir / "manifest.json"),
        "--stage", str(stage),
        "--devices", "0", "1",
        "--jobs-per-device", str(args.jobs_per_device),
        "--shard-count", "3", "--shard-indices", "0",
        "--output-dir", str(output_dir),
        "--min-free-memory-mib", "3500",
        "--poll-seconds", "3",
        "--tag", f"pipeline_stage{stage}_host202",
    ]
    remote_command = (
        f"cd {args.remote_root} && /home/Lim/conda/envs/myenv/bin/python "
        "src/scripts/run_retogcl_t077_adaptive_pipeline_parallel.py "
        f"--manifest {remote_output}/manifest.json --stage {stage} "
        "--devices 0 1 2 3 "
        f"--jobs-per-device {args.jobs_per_device} "
        "--shard-count 3 --shard-indices 1 2 "
        f"--output-dir {remote_output} --min-free-memory-mib 3500 "
        f"--poll-seconds 3 --tag pipeline_stage{stage}_host204"
    )
    local_log.parent.mkdir(parents=True, exist_ok=True)
    with local_log.open("w", encoding="utf-8") as local_handle, remote_log.open(
        "w", encoding="utf-8"
    ) as remote_handle:
        local_process = subprocess.Popen(
            local_command,
            cwd=PROJECT_ROOT,
            stdout=local_handle,
            stderr=subprocess.STDOUT,
        )
        remote_process = subprocess.Popen(
            ["ssh", args.remote, remote_command],
            cwd=PROJECT_ROOT,
            stdout=remote_handle,
            stderr=subprocess.STDOUT,
        )
        progress = tqdm(total=60, initial=result_count(output_dir, stage),
                        desc=f"阶段 {stage}/10", unit="任务")
        previous = progress.n
        while local_process.poll() is None or remote_process.poll() is None:
            time.sleep(max(3, args.poll_seconds))
            sync_output(args, output_dir, both=False)
            # 只拉取远端已完成结果，避免轮询期间搬运 checkpoint。
            run([
                "rsync", "-a",
                f"{args.remote}:{remote_output}/jobs/",
                f"{output_dir / 'jobs'}/",
            ], check=False)
            current = result_count(output_dir, stage)
            progress.update(max(0, current - previous))
            previous = current
            progress.set_postfix(
                本地退出=local_process.poll(), 远端退出=remote_process.poll()
            )
        local_code = local_process.wait()
        remote_code = remote_process.wait()
        progress.close()
    sync_output(args, output_dir, both=True)
    if local_code or remote_code:
        raise RuntimeError(
            f"阶段 {stage} 启动器失败：202={local_code}, 204={remote_code}"
        )
    completed = result_count(output_dir, stage)
    if completed != 60:
        raise RuntimeError(f"阶段 {stage} 应完成 60 个任务，实际 {completed}")
    print(f"阶段 {stage} 已完成并汇回：60/60")


def main() -> None:
    args = parse_args()
    if not 1 <= args.start_stage <= args.end_stage <= 10:
        raise ValueError("阶段范围必须位于 1--10")
    output_dir = (
        args.output_dir if args.output_dir.is_absolute()
        else (PROJECT_ROOT / args.output_dir).resolve()
    )
    if PROJECT_ROOT not in output_dir.parents:
        raise ValueError("output-dir 必须位于项目目录内")
    sync_code(args)
    sync_output(args, output_dir, both=True)
    for stage in range(args.start_stage, args.end_stage + 1):
        ensure_stage(stage, output_dir)
        sync_manifest(args, output_dir)
        launch_stage(args, output_dir, stage)
    print("100 个 Pipeline × 6 数据集实验全部完成。")


if __name__ == "__main__":
    main()
