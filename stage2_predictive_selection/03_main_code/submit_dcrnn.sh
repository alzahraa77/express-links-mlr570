#!/usr/bin/env bash
# Submit the DCRNN comparison: METR-LA, original / express_links / forecast_only / random_matched,
# seeds 0-2 (12 runs) over 3 GPU workers.
#
#   bash submit_dcrnn.sh
#
# Options (environment variables):
#   DATASETS="METR-LA PEMS-BAY"   also run PEMS-BAY
#   MODES="original express_links forecast_only gtr random_matched virtual_node"   all six set-ups
#   WORKERS=2                      fewer GPU jobs
set -euo pipefail
cd "$(dirname "$0")"
export EXPRESS_PYTHON=${EXPRESS_PYTHON:-/shared/$USER/envs/express-links/bin/python}
DATASETS=${DATASETS:-METR-LA}
MODES=${MODES:-original express_links forecast_only random_matched}
SEEDS=${SEEDS:-0 1 2}
WORKERS=${WORKERS:-3}
mkdir -p replication_logs
[[ -f run_replication.py && -f replication_core.py ]] || { echo "Run this from the replication code folder"; exit 1; }

tasks=()
for d in $DATASETS; do for s in $SEEDS; do for m in $MODES; do tasks+=("$d:$m:$s"); done; done; done
for ((w = 0; w < WORKERS; w++)); do
  mine=()
  for ((i = w; i < ${#tasks[@]}; i += WORKERS)); do mine+=("${tasks[$i]}"); done
  [[ ${#mine[@]} -eq 0 ]] && continue
  job=$(sbatch --parsable run_dcrnn_worker.sbatch "${mine[@]}")
  echo "DCRNN worker ${job%%;*}: ${mine[*]}"
done
echo "Watch: squeue -u $USER      Logs: replication_logs/dcrnn-*.out"
