#!/usr/bin/env bash
# One GPU worker that trains several second-model (DCGRU) runs one after another.
# Each task is DATASET:MODE:SEED, e.g. METR-LA:forecast_only:1. Finished runs are skipped.
#SBATCH --job-name=el-dcgru
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=replication_logs/dcgru-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
project=${PROJECT:-/shared/$USER/express_links_replication}
cache=${CACHE:-/shared/$USER/express_links_cache}
python=${EXPRESS_PYTHON:-/shared/$USER/envs/express-links/bin/python}
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
echo "DCGRU worker $SLURM_JOB_ID on $(hostname), tasks: $*"
failed=()
for task in "$@"; do
  IFS=: read -r dataset mode seed <<< "$task"
  if [[ -f "$project/other_model_output/dcgru/full/$dataset/$mode/seed_$seed/result.json" ]]; then
    echo "== $task: already finished, skipping"; continue
  fi
  echo "== $task: starting $(date)"
  if ! "$python" -u run_other_model.py --dataset "$dataset" --mode "$mode" --seed "$seed" \
        --project-dir "$project" --cache "$cache"; then
    echo "!! $task FAILED"; failed+=("$task")
  fi
done
echo "Worker finished $(date). Failed tasks: ${failed[*]:-none}"
