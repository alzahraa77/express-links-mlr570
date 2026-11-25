#!/usr/bin/env bash
#SBATCH --job-name=el-diagnostic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostic_logs/el-diagnostic-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
p="${1:-/shared/$USER/express_links_replication}"
cache="${2:-/shared/$USER/express_links_cache}"
python="${EXPRESS_PYTHON:-/shared/$USER/envs/express-links/bin/python}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
"$python" -u run_diagnostic.py --project-dir "$p" --cache "$cache" --output "$p/diagnostic"
