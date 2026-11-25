#!/usr/bin/env bash
# One step of the Forecast-only tuning. Called by submit_tuning.sh; not meant to be run by hand.
#   plan STAGE | worker STAGE INDEX COUNT | decide STAGE | final
#SBATCH --job-name=el-tune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=tuning_logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
project=${PROJECT:-/shared/$USER/express_links_replication}
cache=${CACHE:-/shared/$USER/express_links_cache}
python=${EXPRESS_PYTHON:-/shared/$USER/envs/express-links/bin/python}
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
common=(--project-dir "$project" --cache "$cache")
step=$1; shift
echo "el-tune $step $* on $(hostname), $(date)"
case "$step" in
  plan)   exec "$python" -u tune_forecast_only.py plan --stage "$1" "${common[@]}" ;;
  decide) exec "$python" -u tune_forecast_only.py decide --stage "$1" "${common[@]}" ;;
  final)  exec "$python" -u tune_forecast_only.py final "${common[@]}" ;;
  worker)
    stage=$1 index=$2 count=$3
    tasks="$project/tuning_output/METR-LA/tasks/$stage.txt"
    [[ -f $tasks ]] || { echo "No task list $tasks"; exit 1; }
    failed=0
    while read -r n label seed; do
      (( (n - 1) % count == index )) || continue
      echo "== $label seed $seed: $(date)"
      "$python" -u tune_forecast_only.py train --label "$label" --seed "$seed" "${common[@]}" || { echo "!! $label seed $seed FAILED"; failed=1; }
    done < <(nl -ba -w1 -s' ' "$tasks")
    echo "Worker $index finished $(date)"
    exit $failed ;;
  *) echo "unknown step $step"; exit 2 ;;
esac
