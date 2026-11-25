#!/usr/bin/env bash
# Submit the whole replication on a single NVIDIA A10G GPU (Slurm): PEMS-BAY and METR-LA, seeds 0, 1 and 2, all six modes.
#
#   cd /shared/$USER/express_links_replication_code
#   bash submit_all.sh
#
# Order of jobs:
#   1. el-prepare  (CPU)  builds and freezes the link graphs for both datasets from the existing data cache
#   2. el-reuse    (CPU)  copies the finished METR-LA seed-0 pilot checkpoints (no retraining)
#   3. el-replicate (GPU array 0-35, 3 at a time)
#        0-17  PEMS-BAY seeds 0,1,2 (trained)
#        18-23 METR-LA seed 0      (re-scored from the pilot checkpoints, a few minutes each)
#        24-35 METR-LA seeds 1,2   (trained)
set -euo pipefail
cd "$(dirname "$0")"

PROJECT=${PROJECT:-/shared/$USER/express_links_replication}
CACHE=${CACHE:-/shared/$USER/express_links_cache}
PILOT=${PILOT:-/shared/$USER/express_links_pilot}
export EXPRESS_PYTHON=${EXPRESS_PYTHON:-/shared/$USER/envs/express-links/bin/python}
CONCURRENT=${CONCURRENT:-3}

for f in "$CACHE/data/metr-la.h5" "$CACHE/data/pems-bay.h5" "$CACHE/data/adj_mx.pkl" "$CACHE/data/adj_mx_bay.pkl" "$EXPRESS_PYTHON"; do
  [[ -e "$f" ]] || { echo "Missing: $f"; exit 1; }
done
[[ -d "$PILOT/pilot_output/full/METR-LA" ]] || echo "Note: no pilot results at $PILOT, METR-LA seed 0 will be retrained."

mkdir -p replication_logs "$PROJECT"
prep=$(sbatch --parsable prepare_replication.sbatch "$PROJECT" "$CACHE"); prep=${prep%%;*}
reuse=$(sbatch --parsable --dependency=afterok:$prep reuse_metr_seed0.sbatch "$PROJECT" "$PILOT"); reuse=${reuse%%;*}
train=$(sbatch --parsable --array=0-35%$CONCURRENT --dependency=afterok:$reuse run_replication.sbatch "$PROJECT" "$CACHE"); train=${train%%;*}

echo "prepare  job $prep"
echo "reuse    job $reuse  (after $prep)"
echo "training job $train  (array 0-35, $CONCURRENT at a time, after $reuse)"
echo "Watch: squeue -u $USER      Logs: replication_logs/"
echo "When all done: $EXPRESS_PYTHON collect_replication.py --project-dir $PROJECT"
