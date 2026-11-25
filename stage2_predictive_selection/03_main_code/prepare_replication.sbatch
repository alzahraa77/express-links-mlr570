#!/usr/bin/env bash
#SBATCH --job-name=el-prepare
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=replication_logs/prepare-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
project=${1:?Pass a project directory}
cache=${2:-"$project/cache"}
python=${EXPRESS_PYTHON:-/shared/$USER/envs/express-links/bin/python}
if [[ ! -x "$python" ]]; then python=python3; fi
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
for dataset in PEMS-BAY METR-LA; do
  "$python" -u run_replication.py --dataset "$dataset" --profile audit --project-dir "$project" --cache "$cache"
done
