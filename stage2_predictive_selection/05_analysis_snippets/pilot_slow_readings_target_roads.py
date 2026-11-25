#!/usr/bin/env python3
"""Pilot (METR-LA, seed 0): 60-minute MAE on SLOW readings (below the training band-1
speed threshold) at the 25 Express Links target roads, for all six setups.
Checks that every run uses the same sensors, timestamps, targets and observed mask first.

Run from the pilot output folder:
  cd /shared/g00112835/express_links_pilot/pilot_output
  python3 /path/to/pilot_slow_readings_target_roads.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

links = pd.read_csv("graphs/METR-LA/express_links_links.csv", dtype=str)
cutoff = json.loads(Path("audit/METR-LA/preprocessing.json").read_text())[
    "speed_band_cutpoints_training"
][0]

modes = ["original", "express_links", "forecast_only", "gtr",
         "random_matched", "virtual_node"]
root = Path("full/METR-LA")

def load(mode):
    return np.load(root / mode / "seed_0/predictions_12_horizons.npz")

ref = load("original")
ids = ref["sensor_id"].astype(str).tolist()
cols = [ids.index(s) for s in links["target_sensor"]]
truth = ref["targets"][:, -1, cols]
mask = ref["target_observed"][:, -1, cols] & (truth < cutoff)
assert ref["horizon_minutes"][-1] == 60 and mask.any()

print(f"60-minute forecasts | {len(cols)} roads | {mask.sum():,} slow readings")
for mode in modes:
    with load(mode) as run:
        for key in ("sensor_id", "target_timestamp", "targets", "target_observed"):
            assert np.array_equal(run[key], ref[key]), (mode, key)
        error = np.abs(run["predictions"][:, -1, cols] - truth)
        print(f"{mode:16s} MAE = {error[mask].mean():.6f}")
