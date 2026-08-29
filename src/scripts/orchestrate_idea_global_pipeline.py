"""监控全局 B 阶段，自动继续 C 阶段并生成110套统一默认配置。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARE = PROJECT_ROOT / "src/scripts/prepare_idea_global_tuning.py"
LAUNCHER = PROJECT_ROOT / "src/scripts/run_idea_global_parallel.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("IDEA_PARA_GLOBAL"))
    parser.add_argument("--source-dir", type=Path, default=Path("IDEA_PARA"))
    parser.add_argument("--remote", default="Lim@172.16.170.204")
    parser.add_argument(
        "--identity",
        type=Path,
        default=Path("/home/Lim/.ssh/id_ed25519_project4_pool_204"),
    )
    parser.add_argument("--remote-root", default="/home/Lim/Project-xmlg")
    parser.add_argument("--remote-python", default="/home/Lim/conda/envs/myenv/bin/python")
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_status(path: Path, state: str, **details: Any) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        **details,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_local_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def ssh_prefix(args: argparse.Namespace) -> list[str]:
    return [
        "ssh",
        "-i",
        str(args.identity),
        "-o",
        "IdentitiesOnly=yes",
        args.remote,
    ]


def read_remote_status(args: argparse.Namespace, relative_path: str) -> dict[str, Any] | None:
    command = [
        *ssh_prefix(args),
        f"test -f {args.remote_root}/{relative_path} && cat {args.remote_root}/{relative_path}",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def stage_finished(status: dict[str, Any] | None) -> bool:
    if not status or status.get("failures"):
        return False
    total = int(status.get("total", -1))
    finished = int(status.get("completed", 0)) + int(status.get("skipped_existing", 0))
    return (
        total >= 0
        and int(status.get("queued", -1)) == 0
        and int(status.get("running_count", -1)) == 0
        and finished == total
    )


def wait_for_stage(
    args: argparse.Namespace,
    output_dir: Path,
    coordinator_status: Path,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    local_path = output_dir / f"launcher_status_stage_{stage}_host202.json"
    remote_relative = f"IDEA_PARA_GLOBAL/launcher_status_stage_{stage}_host204.json"
    while True:
        local = read_local_status(local_path)
        remote = read_remote_status(args, remote_relative)
        failures = []
        if local:
            failures.extend(local.get("failures", []))
        if remote:
            failures.extend(remote.get("failures", []))
        write_status(
            coordinator_status,
            f"waiting_stage_{stage}",
            host202=local,
            host204=remote,
        )
        if failures:
            raise RuntimeError(f"stage_{stage} 检测到 {len(failures)} 个失败任务")
        if stage_finished(local) and stage_finished(remote):
            return local, remote
        time.sleep(max(5, args.poll_seconds))


def sync_remote_jobs(args: argparse.Namespace, output_dir: Path) -> None:
    subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            f"ssh -i {args.identity} -o IdentitiesOnly=yes",
            f"{args.remote}:{args.remote_root}/IDEA_PARA_GLOBAL/jobs/",
            f"{output_dir}/jobs/",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def prepare_stage(
    stage: str,
    source_dir: Path,
    output_dir: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            stage,
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def sync_c_inputs(args: argparse.Namespace, output_dir: Path) -> None:
    subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            f"ssh -i {args.identity} -o IdentitiesOnly=yes",
            str(output_dir / "manifests" / "stage_C.json"),
            f"{args.remote}:{args.remote_root}/IDEA_PARA_GLOBAL/manifests/",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            f"ssh -i {args.identity} -o IdentitiesOnly=yes",
            str(output_dir / "selections" / "stage_B.json"),
            f"{args.remote}:{args.remote_root}/IDEA_PARA_GLOBAL/selections/",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def launch_stage_c_remote(args: argparse.Namespace) -> None:
    command = (
        f"cd {args.remote_root} && "
        f"setsid -f {args.remote_python} src/scripts/run_idea_global_parallel.py "
        "--manifest IDEA_PARA_GLOBAL/manifests/stage_C.json "
        "--devices 0 1 2 3 --jobs-per-device 2 "
        "--shard-count 3 --shard-indices 1 2 "
        "--output-dir IDEA_PARA_GLOBAL --epochs 40 --probe-epochs 200 "
        "--min-free-memory-mib 4000 --poll-seconds 3 --tag stage_C_host204 "
        "> IDEA_PARA_GLOBAL/launcher_stage_C_host204.out 2>&1 < /dev/null"
    )
    subprocess.run([*ssh_prefix(args), command], cwd=PROJECT_ROOT, check=True)


def launch_stage_c_local(output_dir: Path) -> tuple[subprocess.Popen, Any]:
    log_path = output_dir / "launcher_stage_C_host202.out"
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            str(LAUNCHER),
            "--manifest",
            str(output_dir / "manifests" / "stage_C.json"),
            "--devices",
            "0",
            "1",
            "--jobs-per-device",
            "2",
            "--shard-count",
            "3",
            "--shard-indices",
            "0",
            "--output-dir",
            str(output_dir),
            "--epochs",
            "40",
            "--probe-epochs",
            "200",
            "--min-free-memory-mib",
            "4000",
            "--poll-seconds",
            "3",
            "--tag",
            "stage_C_host202",
        ],
        cwd=PROJECT_ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, handle


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    source_dir = resolve_path(args.source_dir)
    coordinator_status = output_dir / "pipeline_status.json"
    local_c_process = None
    local_c_handle = None
    try:
        wait_for_stage(args, output_dir, coordinator_status, "B")
        write_status(coordinator_status, "syncing_stage_B")
        sync_remote_jobs(args, output_dir)
        write_status(coordinator_status, "preparing_stage_C")
        prepare_stage("C", source_dir, output_dir)
        sync_c_inputs(args, output_dir)
        write_status(coordinator_status, "launching_stage_C")
        launch_stage_c_remote(args)
        local_c_process, local_c_handle = launch_stage_c_local(output_dir)
        wait_for_stage(args, output_dir, coordinator_status, "C")
        if local_c_process.wait(timeout=30) != 0:
            raise RuntimeError("202 的 stage_C launcher 非零退出")
        local_c_handle.close()
        local_c_handle = None
        write_status(coordinator_status, "syncing_stage_C")
        sync_remote_jobs(args, output_dir)
        write_status(coordinator_status, "finalizing")
        prepare_stage("finalize", source_dir, output_dir)
        write_status(
            coordinator_status,
            "complete",
            defaults=str(output_dir / "GLOBAL_DEFAULTS.json"),
        )
    except Exception as error:
        write_status(coordinator_status, "failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if local_c_handle is not None:
            local_c_handle.close()


if __name__ == "__main__":
    main()
