#!/usr/bin/env bash
# 204 优先重跑失败任务，结束后自动恢复其原始主体分片。

set -u

project_root="/home/Lim/Project-xmlg"
python_bin="/xmlg/Lim/conda/envs/myenv/bin/python"
output_dir="/xmlg/Lim/Project-xmlg_runtime/IDEA_PAIRWISE_MODULAR"

cd "$project_root" || exit 1

"$python_bin" src/scripts/run_modular_idea_pairwise_parallel.py \
  --manifest IDEA_PAIRWISE_MODULAR/failure_retry_manifest.json \
  --devices 0 1 2 3 --jobs-per-device 1 \
  --shard-count 1 --shard-indices 0 \
  --output-dir "$output_dir" \
  --data-root datasets/main_data \
  --epochs 40 --probe-epochs 200 \
  --min-free-memory-mib 4000 --poll-seconds 3 \
  --tag modular_pairwise_failure_retry_host204_1x
retry_exit=$?

echo "失败清单阶段退出码：$retry_exit；开始恢复 204 主体分片。"
"$python_bin" src/scripts/run_modular_idea_pairwise_parallel.py \
  --manifest IDEA_PAIRWISE_MODULAR/manifest.json \
  --devices 0 1 2 3 --jobs-per-device 1 \
  --shard-count 5 --shard-indices 3 4 \
  --output-dir "$output_dir" \
  --data-root datasets/main_data \
  --epochs 40 --probe-epochs 200 \
  --min-free-memory-mib 4000 --poll-seconds 3 \
  --tag modular_pairwise_resume_after_failure_host204_1x
