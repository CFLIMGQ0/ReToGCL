"""等待双主机参数消融结束，自动同步、汇总并执行显著性分析。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("IDEA_PARA"))
    parser.add_argument("--remote", default="Lim@172.16.170.204")
    parser.add_argument("--identity", type=Path, default=Path("/home/Lim/.ssh/id_ed25519_project4_pool_204"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def remote_status(args: argparse.Namespace) -> dict | None:
    command = [
        "ssh", "-i", str(args.identity), "-o", "IdentitiesOnly=yes", args.remote,
        "cat /home/Lim/Project-xmlg/IDEA_PARA/launcher_status_host204.json",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    return None if result.returncode else json.loads(result.stdout)


def finished(status: dict | None) -> bool:
    return bool(status and status["queued"] == 0 and status["running_count"] == 0)


def write_status(path: Path, state: str, local: dict | None, remote: dict | None) -> None:
    value = {
        "updated_at": datetime.now(timezone.utc).isoformat(), "state": state,
        "host202": local, "host204": remote,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "finalizer_status.json"
    while True:
        local = read_json(output_dir / "launcher_status_host202.json")
        remote = remote_status(args)
        failures = (local or {}).get("failures", []) + (remote or {}).get("failures", [])
        if failures:
            write_status(status_path, "failed_tasks_detected", local, remote)
            raise SystemExit("检测到失败任务，停止自动汇总")
        if finished(local) and finished(remote):
            break
        write_status(status_path, "waiting", local, remote)
        time.sleep(max(5, args.poll_seconds))

    write_status(status_path, "syncing", local, remote)
    subprocess.run([
        "rsync", "-a", "-e",
        f"ssh -i {args.identity} -o IdentitiesOnly=yes",
        f"{args.remote}:/home/Lim/Project-xmlg/IDEA_PARA/", f"{output_dir}/",
    ], check=True, cwd=PROJECT_ROOT)
    write_status(status_path, "merging", local, remote)
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "src/scripts/run_idea_parameter_ablation.py"),
        "--merge-only", "--output-dir", str(output_dir),
    ], check=True, cwd=PROJECT_ROOT)
    write_status(status_path, "analyzing", local, remote)
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "src/scripts/analyze_idea_parameter_ablation.py"),
        "--input-dir", str(output_dir), "--expected", "4950",
    ], check=True, cwd=PROJECT_ROOT)
    write_status(status_path, "complete", local, remote)


if __name__ == "__main__":
    main()
