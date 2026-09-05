#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/Lim/Project-xmlg
REMOTE_ROOT=/home/Lim/Project-xmlg
cd "$PROJECT_ROOT"

# 只依据三阶段的任务记录判断结束，避免 tmux 会话前缀匹配或 PID 复用。
STATUS_CHECK='import json,sys
from pathlib import Path
root=Path(sys.argv[1]); tag=sys.argv[2]
stages=("new_medical5_selected15_20260904", "classic_direct_baselines_9x6_20260904", "multiseed_validation_6x6x10_20260904")
for stage in stages:
    path=root/"outputs"/stage/f"launcher_status_{tag}.json"
    if not path.exists(): sys.exit(1)
    value=json.loads(path.read_text())
    settled=value.get("completed_this_run",0)+value.get("skipped_existing",0)+len(value.get("failures",[]))
    if value.get("queued",1) or value.get("running_count",1) or settled!=value.get("total"):
        sys.exit(1)
'

while ! ssh -o BatchMode=yes -o ConnectTimeout=10 xmlg204 \
  "/home/Lim/conda/envs/myenv/bin/python - '$REMOTE_ROOT' host204" <<< "$STATUS_CHECK"; do
  date -u '+等待 204 实验队列完成：%Y-%m-%d %H:%M:%S UTC'
  sleep 60
done

while ! /home/Lim/conda/envs/myenv/bin/python - "$PROJECT_ROOT" host202 <<< "$STATUS_CHECK"; do
  date -u '+等待 202 实验队列完成：%Y-%m-%d %H:%M:%S UTC'
  sleep 60
done

for directory in \
  outputs/new_medical5_selected15_20260904 \
  outputs/classic_direct_baselines_9x6_20260904 \
  outputs/multiseed_validation_6x6x10_20260904; do
  rsync -az --ignore-existing --info=progress2 --exclude='*.tmp' \
    "xmlg204:$REMOTE_ROOT/$directory/" "$PROJECT_ROOT/$directory/"
done

export LD_LIBRARY_PATH=/home/Lim/conda/envs/myenv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
/home/Lim/conda/envs/myenv/bin/python src/scripts/summarize_multiseed_validation.py \
  --output-root outputs/multiseed_validation_6x6x10_20260904
/home/Lim/conda/envs/myenv/bin/python src/scripts/update_table_campaign_results.py
date -u '+204 结果已回收并完成汇总：%Y-%m-%d %H:%M:%S UTC'
