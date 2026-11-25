#!/bin/bash
# Slurm job for the Express Links Step 2 pilot.
# Everything after the script name is passed to express_links_step2_pilot.py, e.g.
#   sbatch run_pilot.sbatch --profile smoke      # quick check
#   sbatch run_pilot.sbatch                      # full pilot: METR-LA, all six modes, seed 0
#   sbatch run_pilot.sbatch --seeds 1 2          # more seeds later (finished runs are skipped)
#
#SBATCH --job-name=el-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=pilot_logs/%x-%j.out

set -euo pipefail

ENV=/shared/$USER/envs/express-links
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$ENV/bin/python" -u express_links_step2_pilot.py "$@"

echo "Finished $(date)"
