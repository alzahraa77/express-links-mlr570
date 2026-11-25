#!/usr/bin/env bash
# Submit (1) the over-squashing test and (2) the second-model (DCGRU) comparison.
#
#   bash submit_diagnostics.sh
#
# Defaults: over-squashing test on METR-LA with original + express_links + random_matched models
# (seeds 0-2), and DCGRU on METR-LA for original, express_links, forecast_only, random_matched
# (seeds 0-2) split over 3 GPU workers. That is 4 jobs in total, within the cluster job limit.
# Options (environment variables):
#   DATASETS="METR-LA PEMS-BAY"   also train DCGRU on PEMS-BAY
#   MODES="original express_links forecast_only gtr random_matched virtual_node"   all six setups
#   WORKERS=2                      fewer GPU jobs
set -euo pipefail
cd "$(dirname "$0")"
export EXPRESS_PYTHON=${EXPRESS_PYTHON:-/shared/$USER/envs/express-links/bin/python}
DATASETS=${DATASETS:-METR-LA}
MODES=${MODES:-original express_links forecast_only random_matched}
SEEDS=${SEEDS:-0 1 2}
WORKERS=${WORKERS:-3}
mkdir -p replication_logs

job=$(sbatch --parsable oversquash_test.sbatch --modes original express_links random_matched)
echo "over-squashing test: job ${job%%;*}"

tasks=()
for d in $DATASETS; do for s in $SEEDS; do for m in $MODES; do tasks+=("$d:$m:$s"); done; done; done
for ((w = 0; w < WORKERS; w++)); do
  mine=()
  for ((i = w; i < ${#tasks[@]}; i += WORKERS)); do mine+=("${tasks[$i]}"); done
  [[ ${#mine[@]} -eq 0 ]] && continue
  job=$(sbatch --parsable run_other_worker.sbatch "${mine[@]}")
  echo "DCGRU worker ${job%%;*}: ${mine[*]}"
done
echo "Watch: squeue -u $USER      Logs: replication_logs/oversquash-*.out, replication_logs/dcgru-*.out"
