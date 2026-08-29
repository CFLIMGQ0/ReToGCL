#!/usr/bin/env bash
# 恢复模块化两两组合：202 每卡 3 个任务，204 每卡 1 个任务。

set -euo pipefail

project_root="/home/Lim/Project-xmlg"
local_python="/home/Lim/conda/envs/myenv/bin/python"
remote_host="xmlg204"
remote_root="/home/Lim/Project-xmlg"
remote_python="/xmlg/Lim/conda/envs/myenv/bin/python"
remote_output="/xmlg/Lim/Project-xmlg_runtime/IDEA_PAIRWISE_MODULAR"

cd "$project_root"

local_jobs=$(ps -eo pid=,comm=,args= | awk \
  '$2 ~ /^python/ && $0 ~ /src\/scripts\/run_modular_idea_pairwise_(parallel|trial)\.py/ {print}')
if [ -n "$local_jobs" ]; then
  echo "202 已存在模块化组合任务，为避免重复，本次未启动。"
  echo "$local_jobs"
  exit 1
fi
remote_jobs=$(ssh "$remote_host" "ps -eo pid=,comm=,args= | awk \
  '\$2 ~ /^python/ && \$0 ~ /src\\/scripts\\/run_modular_idea_pairwise_(parallel|trial)\\.py/ {print}'")
if [ -n "$remote_jobs" ]; then
  echo "204 已存在模块化组合任务，为避免重复，本次未启动。"
  echo "$remote_jobs"
  exit 1
fi

ssh "$remote_host" "mkdir -p '$remote_root/src/models' '$remote_root/src/scripts' '$remote_root/IDEA_PARA_MODULAR' '$remote_root/IDEA_PAIRWISE_MODULAR' '$remote_output'"
mkdir -p IDEA_PAIRWISE_MODULAR/jobs
# 先合并两台主机已完成结果，随后两个启动器都会自动跳过这些任务。
rsync -a "$remote_host:$remote_output/jobs/" IDEA_PAIRWISE_MODULAR/jobs/
rsync -a IDEA_PAIRWISE_MODULAR/jobs/ "$remote_host:$remote_output/jobs/"
rsync -aR \
  ./model.yaml \
  ./src/models/modular_research_ideas.py \
  ./src/models/modular_pairwise_research_ideas.py \
  ./src/scripts/prepare_modular_idea_pairwise.py \
  ./src/scripts/run_modular_idea_pairwise_trial.py \
  ./src/scripts/run_modular_idea_pairwise_parallel.py \
  ./IDEA_PARA_MODULAR/DEFAULT_PARAMETERS.json \
  ./IDEA_PAIRWISE_MODULAR/manifest.json \
  "$remote_host:$remote_root/"

ssh "$remote_host" "cd '$remote_root' && setsid -f '$remote_python' src/scripts/run_modular_idea_pairwise_parallel.py \
  --manifest IDEA_PAIRWISE_MODULAR/manifest.json \
  --devices 0 1 2 3 --jobs-per-device 1 \
  --shard-count 5 --shard-indices 3 4 \
  --output-dir '$remote_output' \
  --data-root datasets/main_data \
  --epochs 40 --probe-epochs 200 \
  --min-free-memory-mib 4000 --poll-seconds 3 \
  --tag modular_pairwise_resume_host204_1x \
  > '$remote_output/launcher_modular_pairwise_resume_host204_1x.out' 2>&1 < /dev/null"

setsid -f "$local_python" src/scripts/run_modular_idea_pairwise_parallel.py \
  --manifest IDEA_PAIRWISE_MODULAR/manifest.json \
  --devices 0 1 --jobs-per-device 3 \
  --shard-count 5 --shard-indices 0 1 2 \
  --output-dir IDEA_PAIRWISE_MODULAR \
  --data-root datasets/main_data \
  --epochs 40 --probe-epochs 200 \
  --min-free-memory-mib 4000 --poll-seconds 3 \
  --tag modular_pairwise_resume_host202_3x \
  > IDEA_PAIRWISE_MODULAR/launcher_modular_pairwise_resume_host202_3x.out 2>&1 < /dev/null

echo "已恢复：202 使用 GPU0/1、每卡 3 个任务；204 使用 GPU0/1/2/3、每卡 1 个任务。"
