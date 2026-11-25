#!/usr/bin/env python3
"""Reuse the finished METR-LA seed-0 pilot runs in the replication, without retraining.

For each of the six modes, this copies the pilot's 100-epoch checkpoint into the
replication folder. The replication's array task for that run (indices 18-23) then
finds a finished checkpoint, skips training, and only evaluates it with the new
target_low_mae60 metric. Training code is identical between the pilot and the
replication (only the protocol version number and the extra metric differ).

A checkpoint is copied only if the pilot run used the same data file, road graph,
link graph, input scaler, epoch count and software versions as the replication.
If any check fails, nothing is copied for that mode and the array task retrains
it from scratch, so the replication stays valid either way.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch

MODES = ("original", "express_links", "forecast_only", "gtr", "random_matched", "virtual_node")
COPY = ("last_resume.pt", "best_weights.pt", "learning_curve.csv")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project-dir", type=Path, required=True, help="replication project folder")
parser.add_argument("--pilot-dir", type=Path, required=True, help="pilot project folder (holds pilot_output/)")
args = parser.parse_args()

new = args.project_dir.expanduser().resolve() / "pilot_output"
pilot = args.pilot_dir.expanduser().resolve() / "pilot_output" / "full" / "METR-LA"
frozen = json.loads((new / "graphs" / "METR-LA" / "frozen_graphs.json").read_text())
source = json.loads((new / "audit" / "METR-LA" / "source_audit.json").read_text())
prep = json.loads((new / "audit" / "METR-LA" / "preprocessing.json").read_text())
versions = dict(torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__)

reused = 0
for mode in MODES:
    src = pilot / mode / "seed_0"
    dest = new / "full" / "METR-LA" / mode / "seed_0"
    if (dest / "result.json").exists() or (dest / "config.json").exists():
        print(f"{mode}: replication run already exists, leaving it alone")
        continue
    if not (src / "config.json").exists() or not all((src / f).exists() for f in COPY):
        print(f"{mode}: pilot run incomplete or missing, will retrain")
        continue
    cfg = json.loads((src / "config.json").read_text())
    checks = {
        "profile is full": cfg["profile"] == "full",
        "seed is 0": cfg["seed"] == 0,
        "mode matches": cfg["mode"] == mode,
        "100 epochs": cfg["epochs"] == 100,
        "trained on GPU": cfg["device"] == "cuda",
        "no adaptive graph": cfg["adaptive"] is False,
        "same link graph": cfg["graph_variant_sha256"] == frozen[mode]["adjacency_sha256"],
        "same speed file": cfg["speeds_sha256"] == source["speeds_sha256"],
        "same road graph file": cfg["adjacency_sha256"] == source["adjacency_sha256"],
        "same scaler mean": abs(cfg["mean"] - prep["mean"]) <= 1e-9 * abs(prep["mean"]),
        "same scaler std": abs(cfg["std"] - prep["std"]) <= 1e-9 * abs(prep["std"]),
        **{f"same {k} version": cfg[k] == v for k, v in versions.items()},
    }
    state = torch.load(src / "last_resume.pt", map_location="cpu", weights_only=True)
    checks["checkpoint finished 100 epochs"] = len(state["history"]) == 100
    del state
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print(f"{mode}: NOT reused ({', '.join(failed)}), will retrain")
        continue
    dest.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        tmp = dest / (name + ".copying")
        shutil.copy2(src / name, tmp)
        tmp.replace(dest / name)
    pilot_result = json.loads((src / "result.json").read_text())
    (dest / "REUSED_FROM_PILOT.txt").write_text(
        f"Checkpoint copied from {src}\n"
        f"Pilot mae60 = {pilot_result['mae60']} (the re-evaluation should give the same value)\n")
    print(f"{mode}: reused pilot checkpoint (pilot mae60 {pilot_result['mae60']:.6f})")
    reused += 1

print(f"Reused {reused} of {len(MODES)} METR-LA seed-0 runs.")
sys.exit(0)
