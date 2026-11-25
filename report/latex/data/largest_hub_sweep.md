# Results: sd

| Setting | Seeds | RMSE avg (1-20) | RMSE steps 15-20 | Δ RMSE 15-20 vs baseline | MAE steps 15-20 | MAPE steps 15-20 | Extra params | Trainable params | s / epoch | Epochs | Train min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Distance graph (baseline) | 2 | 39.62 ± 0.52 | 46.97 ± 0.48 |  | 25.55 ± 0.10 | 18.4 ± 0.3% | 0 | 508,564 | 14 | 50 | 11 |
| Adaptive, 10 hubs | 3 | 40.65 ± 0.29 | 48.18 ± 0.36 | +3.01% ± 1.67 | 26.71 ± 0.35 | 19.4 ± 0.7% |  | 528,204 | 13 | 47 | 10 |
| Semi-adaptive (Cao), 1 hub | 3 | 39.46 ± 0.42 | 47.03 ± 0.54 | -0.36% ± 0.10 | 25.79 ± 0.26 | 18.6 ± 0.2% |  | 523,416 | 14 | 50 | 11 |
| Semi-adaptive (Cao), 2 hubs | 3 | 39.07 ± 0.61 | 46.07 ± 0.97 | -1.80% ± 3.55 | 25.51 ± 0.68 | 18.2 ± 0.6% |  | 523,948 | 12 | 49 | 10 |
| Semi-adaptive (Cao), 5 hubs | 3 | 39.33 ± 0.63 | 47.03 ± 1.06 | -0.94% ± 3.01 | 26.01 ± 0.84 | 18.4 ± 0.5% |  | 525,544 | 13 | 50 | 11 |
| Semi-adaptive (Cao), 10 hubs | 3 | 39.39 ± 0.81 | 46.79 ± 0.93 | -0.95% ± 3.20 | 25.51 ± 0.28 | 18.8 ± 0.0% |  | 528,204 | 15 | 50 | 13 |
| Semi-adaptive (Cao), 20 hubs | 3 | 39.89 ± 1.21 | 47.80 ± 1.87 | +3.24% ± 3.08 | 26.09 ± 0.99 | 18.8 ± 0.6% |  | 533,524 | 15 | 50 | 13 |

## For reference: Cao et al. (2025) Table 2 (LargeST-SD, one run each)

| Setting (Cao et al. Table 2) | RMSE avg | RMSE avg 75-100 min | MAPE avg | MAPE 75-100 min |
|---|---|---|---|---|
| Distance graph (baseline) | 39.73 | 45.15 | 15.9% | 18.3% |
| All-ones, 1 hub | 39.29 | 45.05 | 15.5% | 17.8% |
| Adaptive, 1 hub | 41.31 | 47.74 | 16.5% | 18.4% |
| Adaptive, 2 hubs | 41.12 | 47.68 | 16.4% | 18.4% |
| Adaptive, 5 hubs | 41.31 | 47.76 | 16.3% | 18.2% |
| Adaptive, 10 hubs | 41.00 | 47.54 | 16.4% | 18.4% |
| Adaptive, 20 hubs | 41.18 | 47.47 | 16.2% | 18.1% |
| Semi-adaptive (Cao), 1 hub | 40.03 | 45.88 | 15.9% | 17.8% |
| Semi-adaptive (Cao), 2 hubs | 39.56 | 44.61 | 16.0% | 17.8% |
| Semi-adaptive (Cao), 5 hubs | 41.53 | 47.95 | 17.2% | 19.2% |
| Semi-adaptive (Cao), 10 hubs | 37.82 | 42.32 | 15.4% | 17.3% |
| Semi-adaptive (Cao), 20 hubs | 40.82 | 46.89 | 16.4% | 18.7% |
