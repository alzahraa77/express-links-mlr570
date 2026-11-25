#!/usr/bin/env bash
#SBATCH --job-name=el-reuse
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=replication_logs/reuse-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
project=${1:?Pass the replication project directory}
pilot=${2:?Pass the pilot project directory}
python=${EXPRESS_PYTHON:-/shared/$USER/envs/express-links/bin/python}
"$python" -u reuse_metr_seed0.py --project-dir "$project" --pilot-dir "$pilot"
