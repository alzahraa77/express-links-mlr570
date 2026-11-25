#!/bin/bash
# Submit the six full runs as separate GPU jobs (2 datasets x 3 modes).
# Shorter jobs fit time limits more easily and can run side by side if GPUs are free.
#
#   bash submit_all.sh            # full profile, seed 0
#   SEEDS="0 1 2" bash submit_all.sh
#   bash submit_all.sh smoke      # quick check of all six as jobs
#
# Run the download step on the login node first:
#   /shared/$USER/envs/express-links/bin/python -u express_links_step1.py --download-only

set -euo pipefail
PROFILE="${1:-full}"
SEEDS="${SEEDS:-0}"
cd "$(dirname "$0")"
mkdir -p express_links_logs

for file in metr-la.h5 pems-bay.h5 adj_mx.pkl adj_mx_bay.pkl; do
  if [ ! -f "/shared/$USER/express_links_cache/data/$file" ]; then
    echo "Missing $file. Run first:  /shared/$USER/envs/express-links/bin/python -u express_links_step1.py --download-only"
    exit 1
  fi
done

for dataset in METR-LA PEMS-BAY; do
  for mode in paper_full paper_fixed express_fixed; do
    # shellcheck disable=SC2086
    job=$(sbatch --parsable --job-name="el-$dataset-$mode" run_express_links.sbatch \
          --profile "$PROFILE" --datasets "$dataset" --modes "$mode" --seeds $SEEDS)
    echo "$dataset $mode -> job $job"
  done
done
echo "Check with: squeue -u $USER     Logs: express_links_logs/"
