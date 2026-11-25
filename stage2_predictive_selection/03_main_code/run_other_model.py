#!/usr/bin/env python3
"""Repeat the forecasting comparison with a second graph model: is the effect Graph WaveNet's,
or does the link-selection method help a different architecture too?

Model: DCGRU, a diffusion-convolution GRU encoder in the style of DCRNN (Li et al., ICLR 2018):
two recurrent layers of 64 units read the 12 inputs; each gate mixes every road with its
1- and 2-step neighbours over the forward and backward transition matrices; a linear layer
outputs all 12 horizons. It is recurrent, unlike Graph WaveNet's dilated convolutions, and has
no learned adaptive graph. It is a simplified DCRNN (no seq2seq decoder), not the original.

Everything else matches the replication exactly: same strict preprocessing, inputs, batching
order and epoch seeds, the same FROZEN link graphs (no re-selection), the same double-transition
supports and message direction, masked MAE loss, Adam (lr 0.001, weight decay 0.0001), gradient
clipping at 5, 100 epochs, best-validation checkpoint, and the same test metrics
(mae60, band1_mae60, express_target_mae60, target_low_mae60) on the same test slice.

  python -u run_other_model.py --dataset METR-LA --mode original --seed 0
  python -u run_other_model.py --collect                 # summary vs Graph WaveNet
"""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import replication_core

MODEL = "dcgru"
MODES = ("original", "express_links", "forecast_only", "gtr", "random_matched", "virtual_node")
METRIC_KEYS = ("target_low_mae60", "express_target_mae60", "band1_mae60", "mae60")
user = os.environ.get("USER", "user")
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", choices=["METR-LA", "PEMS-BAY"], default="METR-LA")
parser.add_argument("--mode", choices=MODES, default="original")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--profile", choices=["smoke", "full"], default="full")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--hidden", type=int, default=64)
parser.add_argument("--project-dir", type=Path, default=Path(f"/shared/{user}/express_links_replication"))
parser.add_argument("--cache", type=Path, default=Path(f"/shared/{user}/express_links_cache"))
parser.add_argument("--collect", action="store_true", help="only summarise finished runs")
args = parser.parse_args()
OUT = args.project_dir / "other_model_output" / MODEL


# ---------------------------------------------------------------- model
class DiffusionConv(nn.Module):
    """[x, P_f x, P_f^2 x, P_b x, P_b^2 x] -> linear, with Graph WaveNet's message direction:
    road v sends to road w through support[v, w] (einsum 'bvf,vw->bwf')."""

    def __init__(self, supports, in_dim, out_dim, order=2, bias_init=0.0):
        super().__init__()
        for i, support in enumerate(supports):
            self.register_buffer(f"support_{i}", support)
        self.n_supports, self.order = len(supports), order
        self.linear = nn.Linear(in_dim * (1 + self.n_supports * order), out_dim)
        nn.init.constant_(self.linear.bias, bias_init)

    def forward(self, x):
        terms = [x]
        for i in range(self.n_supports):
            support, h = getattr(self, f"support_{i}"), x
            for _ in range(self.order):
                h = torch.einsum("bvf,vw->bwf", h, support)
                terms.append(h)
        return self.linear(torch.cat(terms, dim=-1))


class DCGRUCell(nn.Module):
    def __init__(self, supports, in_dim, hidden):
        super().__init__()
        self.gates = DiffusionConv(supports, in_dim + hidden, 2 * hidden, bias_init=1.0)
        self.candidate = DiffusionConv(supports, in_dim + hidden, hidden)

    def forward(self, x, h):
        r, u = torch.sigmoid(self.gates(torch.cat([x, h], dim=-1))).chunk(2, dim=-1)
        c = torch.tanh(self.candidate(torch.cat([x, r * h], dim=-1)))
        return u * h + (1 - u) * c


class DCGRU(nn.Module):
    def __init__(self, supports, in_dim=2, hidden=64, layers=2, horizon=12):
        super().__init__()
        self.hidden = hidden
        self.cells = nn.ModuleList(DCGRUCell(supports, in_dim if i == 0 else hidden, hidden) for i in range(layers))
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):                      # x: [batch, 2, roads, 12] as built by make_batch
        batch, _, roads, steps = x.shape
        sequence = x.permute(0, 3, 2, 1)       # [batch, 12, roads, 2]
        states = [x.new_zeros(batch, roads, self.hidden) for _ in self.cells]
        for t in range(steps):
            inp = sequence[:, t]
            for i, cell in enumerate(self.cells):
                states[i] = cell(inp, states[i])
                inp = states[i]
        return self.head(inp)                  # [batch, roads, 12], scaled speed


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
        for label, frame in ((MODEL.upper(), group), ("Graph WaveNet", None if gwn is None else gwn[gwn.dataset == dataset])):
            if frame is None or frame.empty:
                continue
            seeds = sorted(set(group.seed))
            frame = frame[frame.seed.isin(seeds) & frame["mode"].isin(set(group["mode"]))]
            base = frame[frame["mode"] == "original"].set_index("seed")
            print(f"\n=== {dataset} · {label} · seeds {seeds}: mean, change vs original [seeds better than original]")
            lines = []
            for mode, part in frame.groupby("mode"):
                part = part.set_index("seed")
                line = {"mode": mode, "n": len(part)}
                for key in METRIC_KEYS:
                    common = part.index.intersection(base.index)
                    delta = (part.loc[common, key] - base.loc[common, key]) if len(common) else pd.Series(dtype=float)
                    wins = "" if mode == "original" else f" [{int((delta < 0).sum())}/{len(delta)}]"
                    line[key] = f"{part[key].mean():.3f} {delta.mean():+.3f}{wins}" if len(delta) else f"{part[key].mean():.3f}"
                lines.append(line)
            print(pd.DataFrame(lines).set_index("mode").to_string())
            if {"forecast_only", "random_matched"} <= set(frame["mode"]):
                fo = frame[frame["mode"] == "forecast_only"].set_index("seed")["target_low_mae60"]
                rm = frame[frame["mode"] == "random_matched"].set_index("seed")["target_low_mae60"]
                common = fo.index.intersection(rm.index)
                d = fo[common] - rm[common]
                print(f"forecast_only - random_matched on target_low_mae60, per seed: "
                      f"{', '.join(f'{v:+.3f}' for v in d)} (mean {d.mean():+.3f})")


if args.collect:
    collect()
    raise SystemExit(0)


# ---------------------------------------------------------------- train and evaluate one run
R = replication_core.load(args.dataset, args.project_dir, args.cache, profile=args.profile)
DEVICE = R["DEVICE"]
run_dir = OUT / args.profile / args.dataset / args.mode / f"seed_{args.seed}"
run_dir.mkdir(parents=True, exist_ok=True)
source, prepared, graphs, targets = replication_core.prepare(R, args.dataset, run_dir / "audit")
data = R["variant_data"](prepared, graphs[args.mode])
data["shared_targets"] = targets
epochs = 1 if args.profile == "smoke" else args.epochs
config = dict(model=MODEL, dataset=args.dataset, mode=args.mode, seed=args.seed, profile=args.profile,
              epochs=epochs, hidden=args.hidden, layers=2, diffusion_order=2, batch_size=R["BATCH"],
              learning_rate=.001, weight_decay=.0001, gradient_clip=5, adjacency="doubletransition",
              mean=data["scaler"].mean, std=data["scaler"].std,
              graph_variant_sha256=R["graph_fingerprint"](data["adj"]), torch=str(torch.__version__),
              speeds_sha256=source[-1]["speeds_sha256"])
manifest = run_dir / "config.json"
if manifest.exists():
    old = json.loads(manifest.read_text())
    changed = [k for k in config if k not in ("mean", "std") and old.get(k) != config[k]]
    changed += [k for k in ("mean", "std") if abs(old[k] - config[k]) > 1e-9 * abs(config[k])]
    if changed:
        raise RuntimeError(f"Run settings changed ({changed}); use a new output folder: {run_dir}")
R["save_json"](manifest, config)

R["seed_everywhere"](args.seed)
supports = [torch.as_tensor(np.asarray(R["gwn_util"].asym_adj(a)), dtype=torch.float32, device=DEVICE)
            for a in (data["adj"], data["adj"].T)]
model = DCGRU(supports, hidden=args.hidden).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
scaler, loss_fn = data["scaler"], R["gwn_util"].masked_mae

last, best_file = run_dir / "last_resume.pt", run_dir / "best_weights.pt"
history, best, best_epoch, best_weights = [], float("inf"), 0, None
if last.exists():
    state = torch.load(last, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    history, best, best_epoch = state["history"], state["best"], state["best_epoch"]
    best_weights = {k: v.detach().cpu() for k, v in state["best_weights"].items()}
    print(f"Resuming {args.dataset}/{args.mode} seed {args.seed} from epoch {len(history)}", flush=True)

for epoch in range(len(history) + 1, epochs + 1):
    epoch_seed = (args.seed + 1009 * epoch) % (2**32)
    R["seed_everywhere"](epoch_seed)
    started = time.monotonic()
    model.train()
    train_losses, val_losses = [], []
    for step, ids in enumerate(R["origin_batches"](data["split"]["train"], shuffle_seed=epoch_seed)):
        x, y = R["make_batch"](data, ids)
        if (y != 0).any():
            optimizer.zero_grad()
            loss = loss_fn(scaler.inverse_transform(model(x)), y, 0.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_losses.append(loss.item())
        if args.profile == "smoke" and step == 1:
            break
    model.eval()
    with torch.no_grad():
        for step, ids in enumerate(R["origin_batches"](data["split"]["val"])):
            x, y = R["make_batch"](data, ids)
            if (y != 0).any():
                val_losses.append(loss_fn(scaler.inverse_transform(model(x)), y, 0.0).item())
            if args.profile == "smoke":
                break
    assert train_losses and val_losses and np.isfinite(train_losses + val_losses).all()
    val = float(np.mean(val_losses))
    history.append(dict(epoch=epoch, train_MAE=float(np.mean(train_losses)), val_MAE=val,
                        seconds=time.monotonic() - started))
    if val < best:
        best, best_epoch = val, epoch
        best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    R["atomic_torch_save"](dict(model=model.state_dict(), optimizer=optimizer.state_dict(), best_weights=best_weights,
                                history=history, best=best, best_epoch=best_epoch), last)
    print(f"{MODEL} {args.dataset} {args.mode} seed={args.seed} epoch {epoch}/{epochs} "
          f"train={history[-1]['train_MAE']:.4f} val={val:.4f} sec={history[-1]['seconds']:.1f}", flush=True)

R["atomic_torch_save"](best_weights, best_file)
pd.DataFrame(history).to_csv(run_dir / "learning_curve.csv", index=False)
model.load_state_dict(best_weights)
model.eval()

# Evaluation: identical slicing and metrics to run_experiment in run_replication.py.
origins = data["split"]["test"][:128] if args.profile == "smoke" else data["split"]["test"]
forecast = []
with torch.inference_mode():
    for ids in R["origin_batches"](origins):
        x, _ = R["make_batch"](data, ids)
        forecast.append(scaler.inverse_transform(model(x)).transpose(1, 2).cpu().numpy())   # [batch, 12, roads]
pred = np.concatenate(forecast, axis=0)[:len(origins)]
target_rows = origins[:, None] + R["Y_OFF"]
truth = data["labels"][target_rows]
mask = truth != 0
real = data["real_node_count"]
pred, truth, mask = pred[:, :, :real], truth[:, :, :real], mask[:, :, :real]
assert np.isfinite(pred).all() and pred.shape == truth.shape
score = R["score_predictions"]
bins = data["bins"]
slow_cutoff = float(bins[0])
target_idx = np.flatnonzero(np.isin(np.asarray(data["ids"])[:real], targets))
assert len(target_idx) == len(targets)
h = -1
low_all = mask[:, h] & (np.searchsorted(bins, truth[:, h], side="right") == 0)
shared_mask = mask[:, h, target_idx] & (truth[:, h, target_idx] < slow_cutoff)
dates = data["index"].to_numpy(dtype="datetime64[ns]")
digest = hashlib.sha256()
for payload in (np.asarray(data["ids"][:real], dtype="<U32"), dates[origins].astype("<M8[ns]"),
                dates[target_rows].astype("<M8[ns]"), truth.astype("<f4", copy=False), mask,
                np.asarray(targets, dtype="<U32"), np.asarray([slow_cutoff], dtype="<f8"), shared_mask):
    digest.update(np.ascontiguousarray(payload).tobytes())
result = {
    "model": MODEL, "best_epoch": best_epoch, "best_validation_MAE": best, "test_origins": len(origins),
    "smoke": args.profile == "smoke", "added_arcs": data["added_arcs"],
    "mae60": float(score(pred[:, h], truth[:, h], mask[:, h])["MAE"]),
    "band1_mae60": float(score(pred[:, h], truth[:, h], low_all)["MAE"]),
    "band1_n60": int(low_all.sum()),
    "express_target_mae60": float(score(pred[:, h, target_idx], truth[:, h, target_idx], mask[:, h, target_idx])["MAE"]),
    "target_low_mae60": float(score(pred[:, h, target_idx], truth[:, h, target_idx], shared_mask)["MAE"]),
    "target_low_n60": int(shared_mask.sum()),
    "target_sensor_n": len(target_idx), "low_speed_cutoff": slow_cutoff,
    "evaluation_sha256": digest.hexdigest(),
    "mae_mean_12": float(np.mean([score(pred[:, i], truth[:, i], mask[:, i])["MAE"] for i in range(12)])),
    "median_train_epoch_seconds": float(np.median([r["seconds"] for r in history])),
}
R["save_json"](run_dir / "result.json.tmp", result)
(run_dir / "result.json.tmp").replace(run_dir / "result.json")
print(json.dumps(result, indent=2), flush=True)
