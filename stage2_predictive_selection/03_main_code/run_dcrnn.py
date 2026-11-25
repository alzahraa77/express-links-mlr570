#!/usr/bin/env python3
"""Second model for the Express Links comparison: the published DCRNN, not a re-implementation.

Model code: the PyTorch DCRNN by chnsh, the implementation the DCRNN author links from the
official repository (github.com/liyaguang/DCRNN, README). It is cloned unchanged at a fixed
commit by setup_dcrnn.sh, and this script imports DCRNNModel, its DCGRU cells, its graph
supports (dual_random_walk) and its masked_mae_loss directly from that clone. Model and training
settings are read from the repository's own config files (data/model/dcrnn_la.yaml and
dcrnn_bay.yaml): 2 layers x 64 units, diffusion step 2, seq2seq decoder with curriculum
learning, Adam lr 0.01 (eps 1e-3), learning rate x0.1 at epochs 20/30/40/50, gradient clipping 5,
100 epochs, early stopping with patience 50, best-validation checkpoint.

The steps around the model are exactly those used for Graph WaveNet in run_replication.py:
the same strict preprocessing and scaler, the same input windows and time-of-day feature,
the same padded batches and per-epoch shuffle seeds, the same FROZEN link graphs (no
re-selection), the same skipping of label-free batches, and the same test slice and metrics
(mae60, band1_mae60, express_target_mae60, target_low_mae60; same evaluation_sha256).
As with Graph WaveNet, the authors' model and training settings are used and only the data
pipeline, graphs and evaluation are ours.

Two things differ from running the repository's dcrnn_train_pytorch.py as a script:
- Its data loader reads its own npz files; here the batches come from the replication pipeline.
- Its supervisor re-creates the Adam optimiser after the first batch (the cells create their
  weights on first use) but leaves the learning-rate schedule attached to the discarded
  optimiser, so the configured decay never takes effect. Here the weights are created with one
  forward pass first (as the supervisor's own _setup_graph does), and the schedule from the
  config is applied to the optimiser that trains the model.
lib/utils.py imports TensorFlow for TF-only helpers that the PyTorch path never calls, so a
placeholder module is registered to avoid installing TensorFlow. No repository file is edited.

  python -u run_dcrnn.py --dataset METR-LA --mode original --seed 0
  python -u run_dcrnn.py --collect          # summary vs Graph WaveNet and the decision rule
"""
import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import replication_core

MODEL = "dcrnn"
DCRNN_REPO = "https://github.com/chnsh/DCRNN_PyTorch"
DCRNN_COMMIT = "d92490b808ba5c5be2f23d427d96e9a56b066d7f"
MODES = ("original", "express_links", "forecast_only", "gtr", "random_matched", "virtual_node")
METRIC_KEYS = ("target_low_mae60", "express_target_mae60", "band1_mae60", "mae60")
user = os.environ.get("USER", "user")
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", choices=["METR-LA", "PEMS-BAY"], default="METR-LA")
parser.add_argument("--mode", choices=MODES, default="original")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--profile", choices=["smoke", "full"], default="full")
parser.add_argument("--project-dir", type=Path, default=Path(f"/shared/{user}/express_links_replication"))
parser.add_argument("--cache", type=Path, default=Path(f"/shared/{user}/express_links_cache"))
parser.add_argument("--collect", action="store_true", help="only summarise finished runs")
args = parser.parse_args()
OUT = args.project_dir / "other_model_output" / MODEL
REPO = args.cache / f"dcrnn-pytorch-{DCRNN_COMMIT[:7]}"


# ---------------------------------------------------------------- collect
def read(path):
    return json.loads(path.read_text())


def collect():
    gwn = args.project_dir / "pilot_output" / "full" / "comparison.csv"
    gwn = pd.read_csv(gwn) if gwn.exists() else None
    rows = []
    for result_path in sorted(OUT.glob("full/*/*/seed_*/result.json")):
        cfg = read(result_path.parent / "config.json")
        rows.append(dict(dataset=cfg["dataset"], mode=cfg["mode"], seed=cfg["seed"], **read(result_path)))
    if not rows:
        raise SystemExit(f"No finished {MODEL} runs under {OUT / 'full'}")
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "full" / "comparison.csv", index=False)
    pd.set_option("display.width", 200)
    for dataset, group in table.groupby("dataset"):
        if gwn is not None:
            signatures = set(gwn[gwn.dataset == dataset].evaluation_sha256) | set(group.evaluation_sha256)
            print(f"\n{dataset}: test slice identical to Graph WaveNet runs: {len(signatures) == 1}")
        base = group[group["mode"] == "original"]
        if len(base):
            print(f"{dataset} DCRNN original graph, mean over seeds: MAE 15/30/60 min {base.mae15.mean():.3f} / "
                  f"{base.mae30.mean():.3f} / {base.mae60.mean():.3f} (paper METR-LA 2.77 / 3.15 / 3.60), "
                  f"best epochs {sorted(base.best_epoch)}")
        for label, frame in ((MODEL.upper(), group), ("Graph WaveNet", None if gwn is None else gwn[gwn.dataset == dataset])):
            if frame is None or frame.empty:
                continue
            seeds = sorted(set(group.seed))
            frame = frame[frame.seed.isin(seeds) & frame["mode"].isin(set(group["mode"]))]
            ref = frame[frame["mode"] == "original"].set_index("seed")
            print(f"\n=== {dataset} · {label} · seeds {seeds}: mean, change vs original [seeds better than original]")
            lines = []
            for mode, part in frame.groupby("mode"):
                part = part.set_index("seed")
                line = {"mode": mode, "n": len(part)}
                for key in METRIC_KEYS:
                    common = part.index.intersection(ref.index)
                    delta = (part.loc[common, key] - ref.loc[common, key]) if len(common) else pd.Series(dtype=float)
                    wins = "" if mode == "original" else f" [{int((delta < 0).sum())}/{len(delta)}]"
                    line[key] = f"{part[key].mean():.3f} {delta.mean():+.3f}{wins}" if len(delta) else f"{part[key].mean():.3f}"
                lines.append(line)
            print(pd.DataFrame(lines).set_index("mode").to_string())
            main = frame.set_index(["mode", "seed"])["target_low_mae60"]
            for method in ("forecast_only", "express_links"):
                if not {method, "random_matched", "original"} <= set(frame["mode"]):
                    continue
                m, r, o = main[method], main["random_matched"], main["original"]
                common = m.index.intersection(r.index).intersection(o.index)
                d = m[common] - r[common]
                wins = int((d < 0).sum())
                holds = wins >= 2 and m[common].mean() < o[common].mean()
                print(f"{method} - random_matched on target_low_mae60, per seed: "
                      f"{', '.join(f'{v:+.3f}' for v in d)}; beats random on {wins}/{len(d)} seeds; "
                      f"3-seed mean {m[common].mean():.3f} vs original {o[common].mean():.3f} -> "
                      f"rule {'HOLDS' if holds else 'does not hold'}")


if args.collect:
    collect()
    raise SystemExit(0)


# ---------------------------------------------------------------- the published DCRNN code
def load_dcrnn():
    if not (REPO / "model" / "pytorch" / "dcrnn_model.py").exists():
        raise SystemExit(f"DCRNN code not found in {REPO}; run setup_dcrnn.sh first.")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != DCRNN_COMMIT:
        raise SystemExit(f"{REPO} is at {head or 'unknown'}, expected {DCRNN_COMMIT}")
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{REPO} has local edits; re-clone it with setup_dcrnn.sh:\n{dirty}")
    import importlib.util
    placeholder = "tensorflow" not in sys.modules and importlib.util.find_spec("tensorflow") is None
    if placeholder:
        # Present only while lib/utils.py is imported; its TF-only helpers are never called.
        sys.modules["tensorflow"] = types.ModuleType("tensorflow")
    sys.path.insert(0, str(REPO))
    import yaml
    try:
        from model.pytorch import dcrnn_model, loss
        from lib import utils as dcrnn_utils
    finally:
        if placeholder:
            del sys.modules["tensorflow"]
    name = {"METR-LA": "dcrnn_la.yaml", "PEMS-BAY": "dcrnn_bay.yaml"}[args.dataset]
    settings = yaml.safe_load((REPO / "data" / "model" / name).read_text())
    return dcrnn_model, loss.masked_mae_loss, dcrnn_utils, settings, name


dcrnn_model, masked_mae_loss, dcrnn_utils, settings, settings_file = load_dcrnn()
MODEL_KW, TRAIN_KW = dict(settings["model"]), settings["train"]

# ---------------------------------------------------------------- data, as for Graph WaveNet
R = replication_core.load(args.dataset, args.project_dir, args.cache, profile=args.profile)
DEVICE = R["DEVICE"]
if DEVICE != dcrnn_model.device:
    raise SystemExit(f"Device mismatch: pipeline {DEVICE}, DCRNN code {dcrnn_model.device}")
run_dir = OUT / args.profile / args.dataset / args.mode / f"seed_{args.seed}"
run_dir.mkdir(parents=True, exist_ok=True)
source, prepared, graphs, targets = replication_core.prepare(R, args.dataset, run_dir / "audit")
data = R["variant_data"](prepared, graphs[args.mode])
data["shared_targets"] = targets
nodes = len(data["ids"])
adj = np.asarray(data["adj"], dtype=np.float32)      # float32, as in the repository's pickle
MODEL_KW["num_nodes"] = nodes                       # the virtual-node graph has one extra node
assert MODEL_KW["seq_len"] == 12 and MODEL_KW["horizon"] == 12 and MODEL_KW["input_dim"] == 2
assert MODEL_KW["output_dim"] == 1 and MODEL_KW["filter_type"] == "dual_random_walk"
assert settings["data"]["batch_size"] == R["BATCH"] == 64
epochs = 1 if args.profile == "smoke" else int(TRAIN_KW["epochs"])
scaler = data["scaler"]

config = dict(model=MODEL, code_repository=DCRNN_REPO, code_commit=DCRNN_COMMIT, settings_file=settings_file,
              model_settings=MODEL_KW, train_settings=TRAIN_KW, dataset=args.dataset, mode=args.mode,
              seed=args.seed, profile=args.profile, epochs=epochs, batch_size=R["BATCH"],
              mean=scaler.mean, std=scaler.std, graph_variant_sha256=R["graph_fingerprint"](data["adj"]),
              torch=str(torch.__version__), speeds_sha256=source[-1]["speeds_sha256"],
              adjacency_sha256=source[-1]["adjacency_sha256"])
manifest = run_dir / "config.json"
if manifest.exists():
    old = json.loads(manifest.read_text())
    changed = [k for k in config if k not in ("mean", "std") and old.get(k) != config[k]]
    changed += [k for k in ("mean", "std") if abs(old[k] - config[k]) > 1e-9 * abs(config[k])]
    if changed:
        raise RuntimeError(f"Run settings changed ({changed}); use a new output folder: {run_dir}")
R["save_json"](manifest, config)


def to_dcrnn(x):
    """Pipeline batch [batch, 2, node, 12] -> repository layout (12, batch, node * 2)."""
    return x.permute(3, 0, 2, 1).reshape(12, x.shape[0], nodes * 2)


def scaled_labels(y):
    """Raw labels [batch, node, 12] -> (12, batch, node), scaled as the repository's loader does
    (missing zeros included), for curriculum learning."""
    return scaler.transform(y).permute(2, 0, 1).contiguous()


def raw_labels(y):
    return y.permute(2, 0, 1).contiguous()


def predict(model, x):
    return scaler.inverse_transform(model(to_dcrnn(x)))  # (12, batch, node), mph


# ---------------------------------------------------------------- model and optimiser
logger = logging.getLogger(f"dcrnn.{args.dataset}.{args.mode}.{args.seed}")
logging.basicConfig(level=logging.INFO, format="%(message)s")
R["seed_everywhere"](args.seed)
model = dcrnn_model.DCRNNModel(adj, logger, **MODEL_KW).to(DEVICE)
# The DCGRU cells create their weights on first use; build them with one validation batch,
# as the repository's _setup_graph does, before creating the optimiser.
model.eval()
with torch.no_grad():
    x, _ = R["make_batch"](data, next(iter(R["origin_batches"](data["split"]["val"]))))
    model(to_dcrnn(x))
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_KW["base_lr"], eps=TRAIN_KW["epsilon"])
schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=TRAIN_KW["steps"],
                                                gamma=TRAIN_KW["lr_decay_ratio"])
clip, patience = TRAIN_KW["max_grad_norm"], TRAIN_KW["patience"]

last, best_file = run_dir / "last_resume.pt", run_dir / "best_weights.pt"
history, best, best_epoch, best_weights, wait, batches_seen = [], float("inf"), 0, None, 0, 0
if last.exists():
    state = torch.load(last, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    schedule.load_state_dict(state["schedule"])
    history, best, best_epoch = state["history"], state["best"], state["best_epoch"]
    wait, batches_seen = state["wait"], state["batches_seen"]
    best_weights = {k: v.detach().cpu() for k, v in state["best_weights"].items()}
    del state
    print(f"Resuming {args.dataset}/{args.mode} seed {args.seed} from epoch {len(history)}", flush=True)
print(f"DCRNN {DCRNN_COMMIT[:7]} ({settings_file}): {n_params} parameters, {nodes} nodes, "
      f"{len(data['split']['train'])} training windows", flush=True)

stopped_early = bool(history) and wait >= patience
for epoch in range(len(history) + 1, epochs + 1):
    if stopped_early:
        break
    epoch_seed = (args.seed + 1009 * epoch) % (2**32)
    R["seed_everywhere"](epoch_seed)                 # also seeds the curriculum draws (np.random)
    started = time.monotonic()
    model.train()
    train_losses, val_losses = [], []
    for step, ids in enumerate(R["origin_batches"](data["split"]["train"], shuffle_seed=epoch_seed)):
        x, y = R["make_batch"](data, ids)
        if (y != 0).any():
            optimizer.zero_grad()
            out = model(to_dcrnn(x), scaled_labels(y), batches_seen)
            loss = masked_mae_loss(scaler.inverse_transform(out), raw_labels(y))
            batches_seen += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_losses.append(loss.item())
        if args.profile == "smoke" and step == 1:
            break
    schedule.step()
    model.eval()
    with torch.no_grad():
        for step, ids in enumerate(R["origin_batches"](data["split"]["val"])):
            x, y = R["make_batch"](data, ids)
            if (y != 0).any():
                val_losses.append(masked_mae_loss(predict(model, x), raw_labels(y)).item())
            if args.profile == "smoke":
                break
    assert train_losses and val_losses and np.isfinite(train_losses + val_losses).all()
    val = float(np.mean(val_losses))
    history.append(dict(epoch=epoch, train_MAE=float(np.mean(train_losses)), val_MAE=val,
                        lr=optimizer.param_groups[0]["lr"], seconds=time.monotonic() - started))
    if val < best:
        best, best_epoch, wait = val, epoch, 0
        best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
    R["atomic_torch_save"](dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                schedule=schedule.state_dict(), best_weights=best_weights, history=history,
                                best=best, best_epoch=best_epoch, wait=wait, batches_seen=batches_seen), last)
    print(f"{MODEL} {args.dataset} {args.mode} seed={args.seed} epoch {epoch}/{epochs} "
          f"train={history[-1]['train_MAE']:.4f} val={val:.4f} lr={history[-1]['lr']:.2e} "
          f"sec={history[-1]['seconds']:.1f}", flush=True)
    if wait == patience:
        print(f"Early stopping at epoch {epoch} (patience {patience}), as in the repository", flush=True)
        stopped_early = True

R["atomic_torch_save"](best_weights, best_file)
pd.DataFrame(history).to_csv(run_dir / "learning_curve.csv", index=False)
model.load_state_dict(best_weights)
model.eval()

# ---------------------------------------------------------------- evaluation, identical to run_replication.py
origins = data["split"]["test"][:128] if args.profile == "smoke" else data["split"]["test"]
forecast = []
with torch.inference_mode():
    for ids in R["origin_batches"](origins):
        x, _ = R["make_batch"](data, ids)
        forecast.append(predict(model, x).permute(1, 0, 2).cpu().numpy())   # [batch, 12, node]
pred = np.concatenate(forecast, axis=0)[:len(origins)]
target_rows = origins[:, None] + R["Y_OFF"]
truth = data["labels"][target_rows]
mask = truth != 0
real = data["real_node_count"]
pred, truth, mask = pred[:, :, :real], truth[:, :, :real], mask[:, :, :real]
assert np.isfinite(pred).all() and pred.shape == truth.shape
dates = data["index"].to_numpy(dtype="datetime64[ns]")
np.savez_compressed(run_dir / "predictions_12_horizons.npz", predictions=pred, targets=truth,
                    target_observed=mask, issue_timestamp=dates[origins], target_timestamp=dates[target_rows],
                    sensor_id=np.array(data["ids"][:real], dtype=str), horizon_minutes=R["Y_OFF"] * 5)
score = R["score_predictions"]
bins = data["bins"]
slow_cutoff = float(bins[0])
target_idx = np.flatnonzero(np.isin(np.asarray(data["ids"])[:real], targets))
assert len(target_idx) == len(targets)
h = -1
low_all = mask[:, h] & (np.searchsorted(bins, truth[:, h], side="right") == 0)
shared_mask = mask[:, h, target_idx] & (truth[:, h, target_idx] < slow_cutoff)
digest = hashlib.sha256()
for payload in (np.asarray(data["ids"][:real], dtype="<U32"), dates[origins].astype("<M8[ns]"),
                dates[target_rows].astype("<M8[ns]"), truth.astype("<f4", copy=False), mask,
                np.asarray(targets, dtype="<U32"), np.asarray([slow_cutoff], dtype="<f8"), shared_mask):
    digest.update(np.ascontiguousarray(payload).tobytes())
per_horizon = [score(pred[:, i], truth[:, i], mask[:, i]) for i in range(12)]
result = {
    "model": MODEL, "code_commit": DCRNN_COMMIT, "best_epoch": best_epoch, "best_validation_MAE": best,
    "epochs_run": len(history), "stopped_early": stopped_early, "test_origins": len(origins),
    "smoke": args.profile == "smoke", "added_arcs": data["added_arcs"], "parameters": n_params,
    "mae60": float(score(pred[:, h], truth[:, h], mask[:, h])["MAE"]),
    "band1_mae60": float(score(pred[:, h], truth[:, h], low_all)["MAE"]),
    "band1_n60": int(low_all.sum()),
    "express_target_mae60": float(score(pred[:, h, target_idx], truth[:, h, target_idx], mask[:, h, target_idx])["MAE"]),
    "target_low_mae60": float(score(pred[:, h, target_idx], truth[:, h, target_idx], shared_mask)["MAE"]),
    "target_low_n60": int(shared_mask.sum()),
    "target_sensor_n": len(target_idx), "low_speed_cutoff": slow_cutoff,
    "evaluation_sha256": digest.hexdigest(),
    "mae15": float(per_horizon[2]["MAE"]), "mae30": float(per_horizon[5]["MAE"]),
    "rmse60": float(per_horizon[11]["RMSE"]), "mape60_pct": float(per_horizon[11]["MAPE_pct"]),
    "mae_mean_12": float(np.mean([r["MAE"] for r in per_horizon])),
    "median_train_epoch_seconds": float(np.median([r["seconds"] for r in history])),
}
R["save_json"](run_dir / "result.json.tmp", result)
(run_dir / "result.json.tmp").replace(run_dir / "result.json")
print(json.dumps(result, indent=2), flush=True)
