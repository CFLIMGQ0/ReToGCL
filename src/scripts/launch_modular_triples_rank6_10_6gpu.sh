#!/usr/bin/env bash
# 二二组合第6至10名扩展三模块：202 两卡×3，204 四卡×3，按算力1:2分片。

set -euo pipefail

project_root="/home/Lim/Project-xmlg"
local_python="/home/Lim/conda/envs/myenv/bin/python"
remote_host="xmlg204"
remote_python="/xmlg/Lim/conda/envs/myenv/bin/python"
remote_output="/xmlg/Lim/Project-xmlg_runtime/IDEA_TRIPLE_MODULAR_RANK6_10"
manifest="IDEA_TRIPLE_MODULAR_RANK6_10/manifest.json"
local_tag="triple_rank6_10_host202_3x"
remote_tag="triple_rank6_10_host204_3x"

cd "$project_root"

local_running=$(pgrep -af '^[^ ]*python .*run_modular_idea_triple_(parallel|trial)\.py' || true)
remote_running=$(ssh "$remote_host" \
  "pgrep -af '^[^ ]*python .*run_modular_idea_triple_(parallel|trial)\\.py' || true")
if [[ -n "$local_running" || -n "$remote_running" ]]; then
  echo "已存在三模块任务，为避免重复，本次未启动。"
  [[ -n "$local_running" ]] && echo "202：$local_running"
  [[ -n "$remote_running" ]] && echo "204：$remote_running"
  exit 0
fi

rsync -a --relative \
  ./src/models/modular_triple_research_ideas.py \
  ./src/scripts/run_modular_idea_triple_trial.py \
  ./src/scripts/run_modular_idea_triple_parallel.py \
  ./src/scripts/prepare_modular_idea_triples_rank6_10.py \
  ./src/scripts/launch_modular_triples_rank6_10_6gpu.sh \
  ./IDEA_TRIPLE_MODULAR_RANK6_10/manifest.json \
  "$remote_host:$project_root/"

ssh "$remote_host" "mkdir -p '$remote_output' && cd '$project_root' && \
  setsid -f '$remote_python' src/scripts/run_modular_idea_triple_parallel.py \
    --manifest '$manifest' \
    --devices 0 1 2 3 --jobs-per-device 3 \
    --shard-count 3 --shard-indices 1 2 \
    --output-dir '$remote_output' \
    --data-root datasets/main_data \
    --epochs 40 --probe-epochs 200 \
    --min-free-memory-mib 4000 --poll-seconds 3 \
    --tag '$remote_tag' \
    > '$remote_output/launcher_${remote_tag}.out' 2>&1 < /dev/null"

mkdir -p IDEA_TRIPLE_MODULAR_RANK6_10
setsid -f "$local_python" src/scripts/run_modular_idea_triple_parallel.py \
  --manifest "$manifest" \
  --devices 0 1 --jobs-per-device 3 \
  --shard-count 3 --shard-indices 0 \
  --output-dir IDEA_TRIPLE_MODULAR_RANK6_10 \
  --data-root datasets/main_data \
  --epochs 40 --probe-epochs 200 \
  --min-free-memory-mib 4000 --poll-seconds 3 \
  --tag "$local_tag" \
  > "IDEA_TRIPLE_MODULAR_RANK6_10/launcher_${local_tag}.out" 2>&1 < /dev/null

echo "第6至10名三模块实验已启动：202=6 个并行任务（1/3），204=12 个并行任务（2/3）。"
