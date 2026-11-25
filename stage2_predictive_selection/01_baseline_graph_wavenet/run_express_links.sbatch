#!/bin/bash
# Slurm job for Express Links Step 1.
# Everything after the script name is passed to express_links_step1.py, e.g.
#   sbatch run_express_links.sbatch --profile full
#   sbatch run_express_links.sbatch --profile full --datasets METR-LA --modes paper_full
#
#SBATCH --job-name=express-links
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=express_links_logs/%x-%j.out
#SBATCH --partition=gpu

set -euo pipefail

ENV=/shared/$USER/envs/express-links
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$ENV/bin/python" -u express_links_step1.py "$@"

echo "Finished $(date)"
