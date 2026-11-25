#!/usr/bin/env bash
# Submit one tuning stage: plan -> 3 GPU workers -> decision, chained with Slurm dependencies.
#
#   bash submit_tuning.sh grid      # then weights, inner, alloc, confirm (one at a time)
#   bash submit_tuning.sh final     # opens the METR-LA test split once
#
# Start the next stage only after the previous stage's decide log says "Decision saved".
set -euo pipefail
cd "$(dirname "$0")"
stage=${1:?usage: bash submit_tuning.sh grid|weights|inner|alloc|confirm|final}
workers=${WORKERS:-3}
mkdir -p tuning_logs
for f in run_replication.py replication_core.py tune_forecast_only.py tune_job.sbatch; do
  [[ -f $f ]] || { echo "Missing $f. Run this from the replication code folder."; exit 1; }
done
if [[ $stage == final ]]; then
  read -r -p "This scores the METR-LA test split. Have grid, weights, inner, alloc and confirm all been decided? [yes/no] " ok
  [[ $ok == yes ]] || exit 1
  job=$(sbatch --parsable --job-name=el-tune-final tune_job.sbatch final)
  echo "Final test job ${job%%;*}. Log: tuning_logs/el-tune-final-${job%%;*}.out"
  exit 0
fi
plan=$(sbatch --parsable ${AFTER:+--dependency=$AFTER} --job-name="el-plan-$stage" tune_job.sbatch plan "$stage")
plan=${plan%%;*}
ids=()
for ((w = 0; w < workers; w++)); do
  j=$(sbatch --parsable --job-name="el-$stage-w$w" --dependency=afterok:$plan tune_job.sbatch worker "$stage" "$w" "$workers")
  ids+=("${j%%;*}")
done
dep=$(IFS=:; echo "${ids[*]}")
decide=$(sbatch --parsable --job-name="el-decide-$stage" --dependency=afterok:$dep tune_job.sbatch decide "$stage")
echo "Stage $stage: plan $plan -> workers ${ids[*]} -> decision ${decide%%;*}"
echo "Watch:  squeue -u $USER"
echo "Result: cat tuning_logs/el-decide-$stage-${decide%%;*}.out"
