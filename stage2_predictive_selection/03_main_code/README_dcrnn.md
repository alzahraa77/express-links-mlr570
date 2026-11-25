# Second model: published DCRNN (replaces the home-made DCGRU)

## Where the model comes from

- DCRNN paper: Li, Yu, Shahabi and Liu, *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*, ICLR 2018. https://arxiv.org/abs/1707.01926
- Official code (TensorFlow, by the first author): https://github.com/liyaguang/DCRNN
- PyTorch code used here, the version the official README links to: https://github.com/chnsh/DCRNN_PyTorch
  Fixed commit `d92490b808ba5c5be2f23d427d96e9a56b066d7f`, MIT licence.

`setup_dcrnn.sh` clones that commit into the cache. `run_dcrnn.py` refuses to run if the clone is at a different commit or has any local edit. The model (`DCRNNModel` with its DCGRU cells), its graph supports (`dual_random_walk`), its loss (`masked_mae_loss`) and all model and training settings (`data/model/dcrnn_la.yaml` / `dcrnn_bay.yaml`) are taken from the repository. Nothing in the model is written by us.

## Same steps as Graph WaveNet

| Step | Graph WaveNet (run_replication.py) | DCRNN (run_dcrnn.py) |
|---|---|---|
| Model code | official repo, fixed commit 6b162e80 | official-linked PyTorch repo, fixed commit d92490b |
| Model and training settings | the authors' (lr 0.001, dropout 0.3, ...) | the authors' yaml (lr 0.01, eps 1e-3, decay x0.1 at 20/30/40/50, curriculum learning, patience 50) |
| Data, split, gap-filling, scaler | strict pipeline | same function, same values |
| Inputs | 12 steps of speed + time of day | same arrays, reshaped to DCRNN's layout |
| Road graph and link graphs | frozen graphs, no re-selection | same frozen files |
| Message direction | road v → road w through A[v, w] | same (DCRNN's random-walk supports applied the same way) |
| Batches | 64, padded with the last window, shuffled with seed + 1009 × epoch | same |
| Epochs and checkpoint | 100, best validation MAE | 100 (stops early only if validation does not improve for 50 epochs, as in DCRNN's config), best validation MAE |
| Test slice and metrics | mae60, band1_mae60, express_target_mae60, target_low_mae60 | same code; evaluation_sha256 must match |
| Set-ups and seeds | 6 set-ups × seeds 0, 1, 2 | default: original, express_links, forecast_only, random_matched × seeds 0, 1, 2 (all six possible) |

Two small things differ from running the repository's own `dcrnn_train_pytorch.py` script:
1. Batches come from our pipeline instead of the repository's npz loader, so both models see identical data.
2. The repository's training loop re-creates its optimiser after the first batch, which leaves the learning-rate decay in its config attached to the old optimiser (so the decay never happens). Here the decay in the config is applied to the optimiser that trains the model.

`lib/utils.py` imports TensorFlow for helpers that only the TensorFlow version uses. A placeholder is used while that file is imported, so TensorFlow does not need to be installed.

## Check before trusting the results

On the original graph, DCRNN should come out close to its published METR-LA numbers (paper, METR-LA MAE: 2.77 at 15 min, 3.15 at 30 min, 3.60 at 60 min). The strict split and gap-filling are slightly different, so expect small differences, not a match to the second decimal. `--collect` prints this line first.

## Commands (HPC, from /shared/$USER/express_links_replication_code)

```bash
bash setup_dcrnn.sh                 # once, on the login node
bash submit_dcrnn.sh                # 3 GPU workers, 12 runs
squeue -u $USER
tail -f replication_logs/dcrnn-*.out
/shared/$USER/envs/express-links/bin/python run_dcrnn.py --collect     # when all finish
```

Results go to `/shared/$USER/express_links_replication/other_model_output/dcrnn/full/`. If a worker hits the 24-hour limit, run `bash submit_dcrnn.sh` again: finished runs are skipped and unfinished ones continue from their last epoch.
