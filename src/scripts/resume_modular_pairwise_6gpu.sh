#!/usr/bin/env bash
# 统一恢复入口：202 每卡 3 个任务，204 每卡 1 个任务。

set -euo pipefail

project_root="/home/Lim/Project-xmlg"
cd "$project_root"
exec bash src/scripts/launch_modular_pairwise_4gpu.sh
