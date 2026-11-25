# Forecast-only tuning (METR-LA, Graph WaveNet)

Rules: `PREREGISTERED_TUNING.txt`. Put these files in the replication code folder
(`/shared/$USER/express_links_replication_code`), next to `run_replication.py` and
`replication_core.py`.

| Stage | What runs | New GPU runs (about 50 min each) |
|---|---|---|
| grid | 12 set-ups + original, seeds 0-1 | about 22 (finished replication runs are reused) |
| weights | 0.5x and 2x on the 2 best set-ups | 8 |
| inner | original graph on the first 80% of training | 3 |
| alloc | standard vs error-based, up to 3 links per road | 4 |
| confirm | seeds 0-4 of original, current Forecast-only, final, matched random | about 12 |
| final | scores the test split once | none (scoring only) |

**Upload this version only after the grid decision log says "Decision saved".** The finished grid
results are reused as they are.

Each `bash submit_tuning.sh STAGE` submits: plan (builds the graphs) → 3 GPU workers → decide.
Start the next stage after the decide log says "Decision saved".
No test predictions or forecast scores are made before `final` (the shared data audit still counts
missing values and speed bands in every split). `final` locks the test plan first and saves a
PASS / FAIL / NO TUNED IMPROVEMENT verdict in `decisions/final_result.json`.

Outputs: `/shared/$USER/express_links_replication/tuning_output/METR-LA/`
(`graphs/`, `decisions/`, `val/`, `test/`, `final_test_results.csv`).
