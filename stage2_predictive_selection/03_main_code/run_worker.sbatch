#!/usr/bin/env bash
# One GPU worker that runs several replication tasks one after another.
# Task numbers use the same mapping as run_replication.sbatch:
#   0-17 PEMS-BAY, 18-35 METR-LA; seed = (task/6)%3; mode = task%6.
# Finished runs (result.json present) are skipped, so a worker can simply be resubmitted.
#SBATCH --job-name=el-worker
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=replication_logs/worker-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
project=${1:?Pass a project directory}
cache=${2:?Pass the cache directory}
shift 2
python=${EXPRESS_PYTHON:-/shared/$USER/envs/express-links/bin/python}
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
datasets=(PEMS-BAY METR-LA)
modes=(original express_links forecast_only gtr random_matched virtual_node)
echo "Worker $SLURM_JOB_ID on $(hostname), tasks: $*"
failed=()
for task in "$@"; do
  dataset=${datasets[$((task / 18))]}
  seed=$(((task / 6) % 3))
  mode=${modes[$((task % 6))]}
  if [[ -f "$project/pilot_output/full/$dataset/$mode/seed_$seed/result.json" ]]; then
    echo "== task $task $dataset $mode seed $seed: already finished, skipping"
    continue
  fi
  echo "== task $task $dataset $mode seed $seed: starting $(date)"
  if ! "$python" -u run_replication.py --dataset "$dataset" --profile full --mode "$mode" --seed "$seed" \
        --project-dir "$project" --cache "$cache"; then
    echo "!! task $task $dataset $mode seed $seed FAILED"
    failed+=("$task")
  fi
done
echo "Worker finished $(date). Failed tasks: ${failed[*]:-none}"
