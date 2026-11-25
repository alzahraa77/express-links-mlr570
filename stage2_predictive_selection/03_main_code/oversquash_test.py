#!/usr/bin/env python3
"""Over-squashing test on METR-LA (PREREGISTERED_DIAGNOSTIC.txt), using saved models only.

For every source road and every saved model, the source's 12 input readings are replaced by the
same road's readings at the same weekday and time of day from an earlier week inside the TRAINING
period. The 60-minute forecast at every other road is then compared with the unswapped forecast:
  Use         = MAE with the swap - MAE without it   (positive = the real source readings helped)
  Sensitivity = mean |forecast change|
  Gradient    = mean |d forecast(target, 60 min) / d input(source)|, summed over the 12 inputs
All three are stored for every (source, target) pair and every test day, then:

SECONDARY RULE (original model): among pairs with positive held-out ridge gain (training data
only), pairs at the same hop distance with ridge gain within 20% are matched one-to-one, one
below and one above that hop distance's median resistance. C = Use(low) - Use(high).
Supported if C > 0 with a day-bootstrap 95% interval excluding 0 on at least 2 of 3 seeds.
Fewer than 10 matched pairs = INCONCLUSIVE.

PRIMARY RULE (only if --modes includes express_links and random_matched):
  A = Use(express_links model, EL pairs) - Use(original model, EL pairs)
  B = Use(random_matched model, RAND pairs) - Use(original model, RAND pairs)
Supported if A - B > 0 with a day-bootstrap 95% interval excluding 0 on at least 2 of 3 seeds.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import replication_core

WEEK = 7 * 288          # five-minute steps in one week
H60 = 11                 # index of the 60-minute horizon
user = os.environ.get("USER", "user")
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--project-dir", type=Path, default=Path(f"/shared/{user}/express_links_replication"))
parser.add_argument("--cache", type=Path, default=Path(f"/shared/{user}/express_links_cache"))
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--modes", nargs="+", default=["original"],
                    choices=["original", "express_links", "random_matched"])
parser.add_argument("--grad-stride", type=int, default=4, help="use every k-th test window for gradients")
parser.add_argument("--bootstrap", type=int, default=1000)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--test-windows", type=int, default=0,
                    help="quick checks only: use the first N test windows (0 = all; decisions need all)")
args = parser.parse_args()
assert "original" in args.modes, "The original model is always needed as the reference."

dataset = "METR-LA"
out = args.output or (args.project_dir / "diagnostic_output" / "oversquash")
out.mkdir(parents=True, exist_ok=True)
R = replication_core.load(dataset, args.project_dir, args.cache)
DEVICE = R["DEVICE"]
if DEVICE.type != "cuda":
    print("WARNING: no GPU found; this will be slow.", flush=True)
source, prepared, graphs, targets = replication_core.prepare(R, dataset, out / "audit_scratch")
raw, index = source[0], source[1]
n_roads = len(prepared["ids"])
step = pd.Timedelta(minutes=5)
assert (index[1:] - index[:-1] == step).all(), "METR-LA should have no timestamp gaps"

# ---------------------------------------------------------------- held-out ridge gain, every pair
def all_pair_gain(source, prepared, eligible):
    """Same ridge set-up as predictive_scores in run_replication.py, scored for EVERY eligible pair."""
    raw, index, tod, ids, adj, _ = source
    end = prepared["summary"]["raw_train_end_exclusive"]
    cut = int(.8 * end)
    origins = prepared["split"]["train"]
    first, later = origins[origins + 12 < cut], origins[origins + 1 >= cut]
    fit = pd.DataFrame(raw[:end]).where(raw[:end] > 0)
    observed = fit.iloc[:cut].to_numpy()
    filled = fit.ffill(limit=3).fillna(fit.iloc[:cut].median().fillna(np.nanmedian(observed))).to_numpy()
    neighborhood = adj.astype(float).copy()
    np.fill_diagonal(neighborhood, 0)
    neighborhood += neighborhood.T.copy()
    degree = neighborhood.sum(axis=1)
    normalized = neighborhood / np.maximum(degree[:, None], np.finfo(float).eps)
    neighbor_speed = filled @ normalized.T
    neighbor_speed[:, degree == 0] = filled[:, degree == 0]
    all_origins = np.concatenate([first, later])
    source_speed = (filled[all_origins] + filled[all_origins - 1] + filled[all_origins - 2]) / 3
    sincos = np.stack([np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)], axis=1)
    alpha = R["RIDGE_ALPHA"]
    gain = np.full((len(ids), len(ids)), np.nan)
    for target in range(len(ids)):
        candidates = np.flatnonzero(eligible[:, target])
        if not len(candidates):
            continue
        labels = raw[all_origins + 12, target]
        train_mask, score_mask = labels[:len(first)] > 0, labels[len(first):] > 0
        if train_mask.sum() < 100 or score_mask.sum() < 100:
            continue
        local = np.column_stack((filled[all_origins[:, None] + np.arange(-11, 1), target],
                                 neighbor_speed[all_origins, target], sincos[all_origins]))
        x, xv = local[:len(first)][train_mask], local[len(first):][score_mask]
        mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
        x = np.column_stack((np.ones(len(x)), (x - mean) / std))
        xv = np.column_stack((np.ones(len(xv)), (xv - mean) / std))
        y, yv = labels[:len(first)][train_mask], labels[len(first):][score_mask]
        penalty = np.diag(np.r_[0, np.full(x.shape[1] - 1, alpha)])
        z = source_speed[:len(first)][train_mask][:, candidates]
        zv = source_speed[len(first):][score_mask][:, candidates]
        z_mean, z_std = z.mean(axis=0), np.maximum(z.std(axis=0), 1e-8)
        z, zv = (z - z_mean) / z_std, (zv - z_mean) / z_std
        gram = x.T @ x + penalty
        beta = np.linalg.solve(gram, x.T @ y)
        cross = x.T @ z
        source_beta = np.linalg.solve(gram, cross)
        num = (y - x @ beta) @ z
        denom = (z * z).sum(axis=0) + alpha - (cross * source_beta).sum(axis=0)
        gamma = num / np.maximum(denom, alpha * 0.5)
        residual = yv - xv @ beta
        pred_gain = (zv - xv @ source_beta) * gamma
        gain[candidates, target] = np.abs(residual).mean() - np.abs(residual[:, None] - pred_gain).mean(axis=0)
    return gain

eligible, hops, _, resistance = R["graph_candidates"](prepared["adj"])
started = time.monotonic()
gain = all_pair_gain(source, prepared, eligible)
print(f"Ridge gain for {np.isfinite(gain).sum()} eligible pairs in {time.monotonic() - started:.0f}s", flush=True)

# ---------------------------------------------------------------- test windows and swap offsets
origins = prepared["split"]["test"]
if args.test_windows:
    origins = origins[:args.test_windows]
    print(f"QUICK CHECK: only {len(origins)} test windows; not valid for the decision rules", flush=True)
n = len(origins)
truth = prepared["labels"][origins + 12][:, :n_roads]            # 60-minute targets
observed = truth != 0
slow_cutoff = float(prepared["bins"][0])
slow = observed & (truth < slow_cutoff)
day_labels, day_of_window = np.unique(index[origins].normalize(), return_inverse=True)
n_days = len(day_labels)
onehot = np.eye(n_days)[day_of_window]                           # [n, days]
count_all = onehot.T @ observed                                  # [days, roads]
count_slow = onehot.T @ slow

train_end = prepared["summary"]["raw_train_end_exclusive"]
rows = origins[:, None] + R["X_OFF"]                             # [n, 12]
obs_raw = raw > 0
offset = np.zeros((n, n_roads), dtype=int)                       # weeks back, per window and source road
fallback = np.zeros((n, n_roads), dtype=bool)
first_valid = np.zeros(n, dtype=int)
for k in range(1, 30):
    back = rows - k * WEEK
    in_train = (back[:, 0] >= 0) & (back[:, -1] < train_end)
    first_valid[(first_valid == 0) & in_train] = k
    complete = in_train[:, None] & obs_raw[np.clip(back, 0, None)].all(axis=1)
    offset[(offset == 0) & complete] = k
missing = offset == 0
assert (first_valid > 0).all(), "Some test windows have no training-period week to borrow from"
offset[missing] = np.repeat(first_valid[:, None], n_roads, axis=1)[missing]
fallback[missing] = True
dates = index.to_numpy()
assert np.all(dates[rows] - dates[rows - WEEK] == np.timedelta64(7, 'D'))
print(f"Swap offsets: median {np.median(offset):.0f} weeks; "
      f"{100 * fallback.mean():.2f}% of swaps had no fully observed week and use the nearest training week",
      flush=True)

# ---------------------------------------------------------------- per-model measurements
def load_model(view, mode, seed):
    run_dir = R["OUTPUT"] / "full" / dataset / mode / f"seed_{seed}"
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["graph_variant_sha256"] == R["graph_fingerprint"](view["adj"]), f"{mode}/{seed}: graph mismatch"
    for key in ("mean", "std"):
        assert abs(cfg[key] - getattr(view["scaler"], key)) <= 1e-9 * abs(cfg[key]), f"{mode}/{seed}: scaler mismatch"
    supports = [torch.as_tensor(np.asarray(R["gwn_util"].asym_adj(a)), dtype=torch.float32, device=DEVICE)
                for a in (view["adj"], view["adj"].T)]
    fit = R["gwn_engine"].trainer(view["scaler"], 2, 12, len(view["ids"]), 32, .3, .001, .0001,
                                  DEVICE, supports, True, False, None)
    fit.model.load_state_dict(torch.load(run_dir / "best_weights.pt", map_location=DEVICE, weights_only=True))
    fit.model.eval()
    return fit.model


def measure(mode, seed):
    view = R["variant_data"](prepared, graphs[mode])
    model = load_model(view, mode, seed)
    mean, std = view["scaler"].mean, view["scaler"].std
    batches = [np.arange(i, min(i + 64, n)) for i in range(0, n, 64)]
    inputs = [R["make_batch"](view, origins[b])[0] for b in batches]
    with torch.inference_mode():
        base = np.concatenate([(model(x)[:, H60, :n_roads, 0] * std + mean).cpu().numpy() for x in inputs])
    base_err = np.abs(base - truth)
    use = np.zeros((n_roads, n_days, n_roads))
    use_slow = np.zeros((n_roads, n_days, n_roads))
    sens = np.zeros((n_roads, n_days, n_roads))
    scaled = torch.as_tensor(view["scaled"], device=DEVICE)
    started = time.monotonic()
    with torch.inference_mode():
        for s in range(n_roads):
            replacement = scaled[torch.as_tensor(rows - offset[:, s:s + 1] * WEEK, device=DEVICE), s]  # [n, 12]
            swapped = []
            for b, x in zip(batches, inputs):
                xs = x.clone()
                xs[:, 0, s, :] = replacement[b]
                swapped.append((model(xs)[:, H60, :n_roads, 0] * std + mean).cpu().numpy())
            swapped = np.concatenate(swapped)
            diff = (np.abs(swapped - truth) - base_err)
            use[s] = onehot.T @ (diff * observed)
            use_slow[s] = onehot.T @ (diff * slow)
            sens[s] = onehot.T @ (np.abs(swapped - base) * observed)
            if s % 50 == 0:
                print(f"  {mode} seed {seed}: source {s + 1}/{n_roads} ({time.monotonic() - started:.0f}s)", flush=True)
    grad = np.zeros((n_roads, n_roads))
    picked = np.arange(0, n, args.grad_stride)
    for i in range(0, len(picked), 64):
        x = R["make_batch"](view, origins[picked[i:i + 64]])[0].requires_grad_(True)
        out_all = model(x)[:, H60, :n_roads, 0] * std
        for t in range(n_roads):
            g, = torch.autograd.grad(out_all[:, t].sum(), x, retain_graph=t < n_roads - 1)
            grad[:, t] += g[:, 0, :n_roads, :].abs().sum(dim=(0, 2)).cpu().numpy()
    grad /= len(picked)
    np.savez_compressed(out / f"{mode}_seed_{seed}.npz", use=use, use_slow=use_slow, sens=sens, grad=grad,
                        base_mae=(base_err * observed).sum(0) / observed.sum(0))
    print(f"{mode} seed {seed} done in {time.monotonic() - started:.0f}s", flush=True)
    return dict(use=use, use_slow=use_slow, sens=sens, grad=grad)


# ---------------------------------------------------------------- matching for the secondary rule
useful = eligible & np.isfinite(gain) & (gain > 0)
low_side = np.zeros_like(useful)
for h in np.unique(hops[eligible]):
    stratum = eligible & (hops == h)
    median = np.median(resistance[stratum])
    low_side |= stratum & (resistance <= median)
from bisect import bisect_left
from collections import defaultdict


def match_pairs(same_target):
    """One-to-one: each low-resistance useful pair gets the unused high-resistance useful pair with
    the same hop distance (and, optionally, the same target) whose gain is closest, within 20%."""
    groups = defaultdict(list)
    for b in zip(*np.nonzero(useful & ~low_side)):
        groups[(hops[b], b[1]) if same_target else hops[b]].append(b)
    table = {}
    for key, members in groups.items():
        members.sort(key=lambda p: gain[p])
        table[key] = ([gain[p] for p in members], members, np.zeros(len(members), dtype=bool))
    matched = []
    for a in sorted(zip(*np.nonzero(useful & low_side)), key=lambda p: (-gain[p], p)):
        key = (hops[a], a[1]) if same_target else hops[a]
        if key not in table:
            continue
        gains, members, used = table[key]
        i = bisect_left(gains, gain[a])
        left, right = i - 1, i
        while left >= 0 and used[left]:
            left -= 1
        while right < len(gains) and used[right]:
            right += 1
        options = [j for j in (left, right) if 0 <= j < len(gains)]
        if not options:
            continue
        j = min(options, key=lambda j: abs(gains[j] - gain[a]))
        if abs(gains[j] - gain[a]) <= 0.2 * max(gains[j], gain[a]):
            used[j] = True
            matched.append((a, members[j]))
    return matched


pairs = [match_pairs(False), match_pairs(True)]
matched, matched_same_target = pairs
print(f"Matched pairs: {len(matched)} (same target: {len(matched_same_target)})", flush=True)
pd.DataFrame([dict(low_source=prepared["ids"][a[0]], low_target=prepared["ids"][a[1]], high_source=prepared["ids"][b[0]],
                   high_target=prepared["ids"][b[1]], hops=int(hops[a]), low_gain=gain[a], high_gain=gain[b],
                   low_resistance=resistance[a], high_resistance=resistance[b]) for a, b in matched]
             ).to_csv(out / "matched_pairs.csv", index=False)

rng = np.random.default_rng(12345)
weights = rng.multinomial(n_days, np.full(n_days, 1 / n_days), size=args.bootstrap).T   # [days, B]


def use_of(m, key, count, pair_list):
    """Use per pair: point value and bootstrap replicates (resampling whole test days)."""
    s = np.array([p[0] for p in pair_list]); t = np.array([p[1] for p in pair_list])
    sums, counts = m[key][s, :, t], count[:, t].T                # [pairs, days]
    point = sums.sum(1) / np.maximum(counts.sum(1), 1)
    boot = (sums @ weights) / np.maximum(counts @ weights, 1)
    return point, boot


def summarise(point, boot):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(value=float(point), ci_low=float(lo), ci_high=float(hi), excludes_zero=bool(lo > 0 or hi < 0))


results = []
el_pairs = [(a, b) for a, b, _ in graphs["express_links"][2]]
rand_pairs = [(a, b) for a, b, _ in graphs["random_matched"][2]]
for seed in args.seeds:
    measured = {mode: measure(mode, seed) for mode in args.modes}
    orig = measured["original"]
    for label, match in (("matched", matched), ("matched_same_target", matched_same_target)):
        if not match:
            continue
        for key, count in (("use", count_all), ("use_slow", count_slow)):
            pl, bl = use_of(orig, key, count, [a for a, _ in match])
            ph, bh = use_of(orig, key, count, [b for _, b in match])
            results.append(dict(seed=seed, test=f"secondary_C_{label}", measure=key, pairs=len(match),
                                **summarise(pl.mean() - ph.mean(), (bl - bh).mean(0))))
        g_low = np.mean([orig["grad"][a] for a, _ in match]); g_high = np.mean([orig["grad"][b] for _, b in match])
        results.append(dict(seed=seed, test=f"secondary_C_{label}", measure="gradient_low_minus_high",
                            pairs=len(match), value=float(g_low - g_high)))
    for name, plist in (("EL_pairs", el_pairs), ("RAND_pairs", rand_pairs)):
        p, b = use_of(orig, "use", count_all, plist)
        results.append(dict(seed=seed, test=f"original_model_use_{name}", measure="use", pairs=len(plist),
                            **summarise(p.mean(), b.mean(0))))
    if {"express_links", "random_matched"} <= set(args.modes):
        pe, be = use_of(measured["express_links"], "use", count_all, el_pairs)
        po, bo = use_of(orig, "use", count_all, el_pairs)
        pr, br = use_of(measured["random_matched"], "use", count_all, rand_pairs)
        pq, bq = use_of(orig, "use", count_all, rand_pairs)
        a_point, a_boot = pe.mean() - po.mean(), be.mean(0) - bo.mean(0)
        b_point, b_boot = pr.mean() - pq.mean(), br.mean(0) - bq.mean(0)
        results.append(dict(seed=seed, test="primary_A", measure="use", pairs=25, **summarise(a_point, a_boot)))
        results.append(dict(seed=seed, test="primary_B", measure="use", pairs=25, **summarise(b_point, b_boot)))
        results.append(dict(seed=seed, test="primary_A_minus_B", measure="use", pairs=25,
                            **summarise(a_point - b_point, a_boot - b_boot)))
    pd.DataFrame(results).to_csv(out / "oversquash_results.csv", index=False)
    del measured
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

table = pd.DataFrame(results)
pd.set_option("display.width", 200)
print("\n" + table.round(5).to_string(index=False))

def verdict(test, measure="use"):
    rows = table[(table.test == test) & (table.measure == measure)]
    if rows.empty:
        return "NOT RUN"
    wins = int(((rows.value > 0) & rows.excludes_zero).sum())
    detail = f"({wins} of {len(rows)} seeds: >0 with interval excluding 0)"
    if len(rows) < 3:
        return f"NEEDS 3 SEEDS {detail}"
    return f"{'SUPPORTED' if wins >= 2 else 'NOT SUPPORTED'} {detail}"

print("\n=== Pre-registered decisions ===")
if len(matched) < 10:
    print(f"Secondary rule (over-squashing in the original graph): INCONCLUSIVE, only {len(matched)} matched pairs")
else:
    print(f"Secondary rule (over-squashing in the original graph), {len(matched)} matched pairs: "
          + verdict("secondary_C_matched"))
print("  robustness, same-target matching:", verdict("secondary_C_matched_same_target"))
print("  slow readings only:", verdict("secondary_C_matched", "use_slow"))
print("Primary rule (targeted shortcuts add useful use beyond random):", verdict("primary_A_minus_B"))
json.dump({"matched_pairs": len(matched), "matched_same_target": len(matched_same_target),
           "swap_fallback_percent": float(100 * fallback.mean()), "slow_cutoff": slow_cutoff,
           "test_days": int(n_days), "seeds": args.seeds, "modes": args.modes},
          open(out / "oversquash_run_info.json", "w"), indent=2)
print("Saved to", out)
