# Express Links replication on PEMS-BAY and METR-LA

The six comparison arms and graph selector are unchanged from the pilot. Only the shared evaluation metric and job controls were added. Seed 0, 1 and 2 are paired across all six modes per dataset. The fixed graph selector runs once per dataset on training data, then all seeds reuse its saved graphs. The original model is rerun to pair it with the same seed and current environment. The virtual hub has a larger arc budget and remains a separate global-hub control.

The new metric `target_low_mae60` is MAE in mph on observed ground truth below the dataset's first training-derived speed-band cutoff, at 60 minutes, restricted to the original Express Links' selected target roads for **every** model. It is distinct from `band1_mae60` (slow readings on all roads) and `express_target_mae60` (all speeds on selected roads). The collector verifies matching sensor order, timestamps, readings, missing masks, cutoff and target set via a fingerprint. The speed slice does not label jam onsets.

## 1. Upload and submit

Unzip into a new HPC folder, then run from inside it. Pick a persistent directory outside the upload folder. The live pilot output is not reused or overwritten.

```bash
mkdir -p replication_logs
PROJECT=/shared/$USER/express_links_replication
CACHE="$PROJECT/cache"

PREP=$(sbatch --parsable prepare_replication.sbatch "$PROJECT" "$CACHE")
sbatch --array=0-35%3 --dependency="afterok:${PREP%%;*}" run_replication.sbatch "$PROJECT" "$CACHE"
```

When you have an existing Step 2 data cache, set `CACHE` to its absolute path before both submissions to reuse the HDF5 files and graph data. If downloads are restricted and the HDF5 files are elsewhere, use `--manual-data-dir` with `run_replication.py --profile audit` in the allocated job. The scripts request one GPU per training task and up to three concurrent tasks. Change the concurrency limit if required by the lab allocation. If the lab needs a specific GPU partition or account, use the same Slurm options as your working pilot job. The scripts prefer `/opt/hpc-env/gpu311/bin/python3`; set `EXPRESS_PYTHON` to your CUDA Python executable if different.

Array indices 0–17 are PEMS-BAY and 18–35 are METR-LA. Six consecutive indices cover each seed, with the same mode order shown in `run_replication.sbatch`. To decide what to run in stages, start with indices 0–5 (PEMS-BAY seed 0) or 0–17 (all PEMS-BAY seeds). The remaining METR-LA block repeats seeds with the same automatic metric. Submit one task per condition to avoid concurrent writes to a single run.

## 2. Collect after the jobs finish

```bash
python3 collect_replication.py --project-dir "$PROJECT"
```

For only the three-seed PEMS-BAY block, append `--dataset PEMS-BAY`. For a one-seed readout append `--seeds 0`. The collector refuses to summarize missing arms or mismatched evaluation data. It saves per-seed results in `pilot_output/full/comparison.csv` and paired averages, standard deviations, and delta versus each seed's original graph in `pilot_output/full/seed_summary.csv`. A negative delta means lower error. Compare Express Links to forecast-only on the shared target-low-speed metric before deciding whether resistance helps. Review the overall and all-speed target metrics alongside it. Seed repeats measure training variation under a fixed test period, so do not treat them as independent traffic samples.
