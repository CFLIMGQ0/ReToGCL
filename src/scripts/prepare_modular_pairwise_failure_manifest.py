"""从一个或多个启动器状态中提取失败任务，生成独立重试清单。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL = "modular_cross_module_pairwise_fixed_ABC_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    failed_ids: set[str] = set()
    sources = []
    for path in args.status:
        status = json.loads(path.read_text(encoding="utf-8"))
        if status.get("stage") != PROTOCOL:
            raise RuntimeError(f"状态协议不兼容：{path}")
        sources.append(str(path))
        failed_ids.update(item["task_id"] for item in status.get("failures", []))
    missing = sorted(failed_ids - tasks_by_id.keys())
    if missing:
        raise RuntimeError(f"原始清单缺少失败任务：{missing[:5]}")
    tasks = [task for task in manifest["tasks"] if task["task_id"] in failed_ids]
    if len(tasks) != len(failed_ids):
        raise RuntimeError("失败任务清单存在重复或遗漏")
    output = {
        "metadata": {
            **manifest["metadata"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "retry_failed_tasks_on_host204",
            "source_statuses": sources,
            "task_count": len(tasks),
        },
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(f"失败重试清单已生成：{len(tasks)} 个任务，{args.output}")


if __name__ == "__main__":
    main()
