#!/usr/bin/env bash
set -u

PROJECT_ROOT=/home/Lim/Project-xmlg
PYTHON_BIN=/home/Lim/conda/envs/myenv/bin/python
export LD_LIBRARY_PATH=/home/Lim/conda/envs/myenv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$PROJECT_ROOT"

HOST_TAG=${1:-host202}
if [[ "$HOST_TAG" == "host204" ]]; then
  DEVICES=(0 1 2 3)
  SHARDS=(2 3 4 5)
else
  DEVICES=(0 1)
  SHARDS=(0 1)
fi

"$PYTHON_BIN" src/scripts/run_new_medical5_selected_parallel.py \
  --devices "${DEVICES[@]}" --jobs-per-device 1 \
  --shard-count 6 --shard-indices "${SHARDS[@]}" --tag "$HOST_TAG"
"$PYTHON_BIN" src/scripts/run_classic_direct_baselines_parallel.py \
  --devices "${DEVICES[@]}" --jobs-per-device 1 \
  --shard-count 6 --shard-indices "${SHARDS[@]}" --tag "$HOST_TAG"
"$PYTHON_BIN" src/scripts/run_multiseed_validation_parallel.py \
  --devices "${DEVICES[@]}" --jobs-per-device 1 \
  --shard-count 6 --shard-indices "${SHARDS[@]}" --tag "$HOST_TAG"
