#!/usr/bin/env bash
# Over-squashing diagnostic on METR-LA with the saved Graph WaveNet models (no training).
#   sbatch oversquash_test.sbatch                                   # original model, seeds 0-2
#   sbatch oversquash_test.sbatch --modes original express_links random_matched   # also the primary rule
#SBATCH --job-name=el-oversquash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=replication_logs/oversquash-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
python=${EXPRESS_PYTHON:-/shared/$USER/envs/express-links/bin/python}
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
"$python" -u oversquash_test.py "$@"
