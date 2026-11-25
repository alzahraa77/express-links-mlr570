# Run the frozen METR-LA diagnostic

This package uses the nine existing best checkpoints on HPC. No training or downloads. The package includes the frozen Markdown decision rule from 19 November. Compare it with any other protocol file in your project before submitting, and retain both if they differ. The implementation follows the Markdown rule shipped here.

1. Unzip into a new directory on HPC and open that directory.
2. Submit from the directory containing `run_diagnostic.sbatch`:

   ```bash
   mkdir -p diagnostic_logs
   sbatch run_diagnostic.sbatch
   ```

   The defaults use the previous replication project `/shared/$USER/express_links_replication`, cache `/shared/$USER/express_links_cache` and Python at `/shared/$USER/envs/express-links/bin/python`. For different paths, pass the project directory and cache in that order after the sbatch filename. Set `EXPRESS_PYTHON` before submitting to choose a different interpreter.

3. Once Slurm finishes, read the result:

   ```bash
   cat "/shared/$USER/express_links_replication/diagnostic/diagnostic_result.json"
   ```

The output includes coverage, per-seed reliance, paired contrasts and intervals, and a pass or inconclusive status. `paired_day_sums.json` stores the exact aggregates used for resampling. The script checks source-data fingerprints, graph and link files, model settings, and the saved original 60-minute predictions. As an engineering verification, the clean prediction check allows 0.001 mph absolute error plus a relative tolerance of 0.00001 for GPU floating-point differences. The diagnostic computes both clean and perturbed forecasts on the same hardware and does not change the +0.01 mph decision threshold. It calculates the secondary comparison only if the primary gate passes.

The case-matching preflight can run on CPU with `--preflight` after providing the three paths described by `run_diagnostic.py --help`. It does not need the checkpoints. Matching features are standardized using their training-window standard deviations. For the original weighted neighbor mean, the directed road graph is symmetrized as in the selector, excluding self-loops. If a test window lacks five distinct eligible validation dates, it is excluded from every arm under the frozen rule.
