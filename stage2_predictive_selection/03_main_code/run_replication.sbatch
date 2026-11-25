#!/usr/bin/env bash
#SBATCH --job-name=el-replicate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --output=replication_logs/train-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
project=${1:?Pass a project directory}
cache=${2:-"$project/cache"}
python=${EXPRESS_PYTHON:-/shared/$USER/envs/express-links/bin/python}
if [[ ! -x "$python" ]]; then python=python3; fi
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
task=${SLURM_ARRAY_TASK_ID:?Submit as an array task}
datasets=(PEMS-BAY METR-LA)
modes=(original express_links forecast_only gtr random_matched virtual_node)
dataset=${datasets[$((task / 18))]}
seed=$(((task / 6) % 3))
mode=${modes[$((task % 6))]}
"$python" -u run_replication.py --dataset "$dataset" --profile full --mode "$mode" --seed "$seed" --project-dir "$project" --cache "$cache"
