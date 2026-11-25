#!/usr/bin/env bash
# Submit the 36 replication runs as a few GPU worker jobs (for clusters with a low job-count limit).
#
#   bash submit_workers.sh                 # 3 workers, start immediately (prepare + reuse must be finished)
#   bash submit_workers.sh 60500           # 3 workers, start after job 60500 (the reuse job) succeeds
#   WORKERS=2 bash submit_workers.sh 60500
#
# Tasks are dealt round-robin so every worker gets a similar mix of PEMS-BAY and METR-LA runs.
# Rerunning this later is safe: finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")"
PROJECT=${PROJECT:-/shared/$USER/express_links_replication}
CACHE=${CACHE:-/shared/$USER/express_links_cache}
export EXPRESS_PYTHON=${EXPRESS_PYTHON:-/shared/$USER/envs/express-links/bin/python}
WORKERS=${WORKERS:-3}
after=${1:-}
mkdir -p replication_logs
dep=()
[[ -n "$after" ]] && dep=(--dependency=afterok:$after)
for ((w = 0; w < WORKERS; w++)); do
  tasks=()
  for ((t = w; t < 36; t += WORKERS)); do tasks+=("$t"); done
  job=$(sbatch --parsable "${dep[@]}" run_worker.sbatch "$PROJECT" "$CACHE" "${tasks[@]}")
  echo "worker job ${job%%;*}: tasks ${tasks[*]}"
done
echo "Watch: squeue -u $USER      Logs: replication_logs/worker-*.out"
