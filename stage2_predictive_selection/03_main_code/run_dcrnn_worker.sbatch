#!/usr/bin/env bash
# One GPU worker that trains several DCRNN runs one after another.
# Each task is DATASET:MODE:SEED, e.g. METR-LA:forecast_only:1. Finished runs are skipped and
# unfinished runs resume from their last epoch, so resubmitting the same command is safe.
#SBATCH --job-name=el-dcrnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=replication_logs/dcrnn-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
project=${PROJECT:-/shared/$USER/express_links_replication}
cache=${CACHE:-/shared/$USER/express_links_cache}
python=${EXPRESS_PYTHON:-/shared/$USER/envs/express-links/bin/python}
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
echo "DCRNN worker $SLURM_JOB_ID on $(hostname), tasks: $*"
if ! "$python" -c "import yaml" 2>/dev/null; then
  echo "Installing pyyaml (needed to read the DCRNN config files)"
  "$python" -m pip install --quiet pyyaml || { echo "!! pyyaml install failed"; exit 1; }
fi
failed=()
for task in "$@"; do
  IFS=: read -r dataset mode seed <<< "$task"
  if [[ -f "$project/other_model_output/dcrnn/full/$dataset/$mode/seed_$seed/result.json" ]]; then
    echo "== $task: already finished, skipping"; continue
  fi
  echo "== $task: starting $(date)"
  if ! "$python" -u run_dcrnn.py --dataset "$dataset" --mode "$mode" --seed "$seed" \
        --project-dir "$project" --cache "$cache"; then
    echo "!! $task FAILED"; failed+=("$task")
  fi
done
echo "Worker finished $(date). Failed tasks: ${failed[*]:-none}"
