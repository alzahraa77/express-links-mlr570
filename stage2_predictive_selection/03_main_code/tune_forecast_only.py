#!/usr/bin/env python3
"""Tune the Forecast-only link selector on METR-LA validation data (Graph WaveNet), then test once.

Rules: PREREGISTERED_TUNING.txt. Stages, in order (each needs the previous stage's decision):

  grid     12 set-ups: minimum hops 3/2/1 x 10/25/50/100 links, weight 1x median, seeds 0-1,
           plus the original graph as a reference
  weights  0.5x and 2x weight on the two best eligible grid set-ups, seeds 0-1
  inner    original graph trained on the first 80% of training; its low-speed errors on the
           last 20% of training rank the hardest roads (seeds 0-2)
  alloc    up to 3 links per road: standard (by forecast gain) vs error-based (hardest roads
           first), same budget, hops and weight as the winner, seeds 0-1
  confirm  remaining training for the final comparison: seeds 0-4 of original, current
           Forecast-only, the frozen final set-up and links randomly matched to it
  final    opens the METR-LA test split once and reports the comparison with day-bootstrap
           intervals

  python tune_forecast_only.py plan  --stage grid      # build graphs + task list for a stage
  python tune_forecast_only.py train --label fo_h3_b25_w1 --seed 0
  python tune_forecast_only.py decide --stage grid     # apply the pre-registered rule
  python tune_forecast_only.py final

Training and scoring call run_replication.py's own run_experiment unchanged. During tuning,
run_experiment's scoring slot is pointed at the validation split, so no test predictions or
forecast scores are made before `final` (the shared data audit still counts missing values
and speed bands in every split). In `final`, the trained checkpoints are scored on the real test split
with the same code, so test numbers are directly comparable with the replication.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import replication_core

user = os.environ.get("USER", "user")
STAGES = ("grid", "weights", "inner", "alloc", "confirm", "final")
HOPS, BUDGETS, WEIGHT_MULTS = (3, 2, 1), (10, 25, 50, 100), (0.5, 2.0)
BASELINE = "fo_h3_b25_w1"            # current Forecast-only (must reproduce the frozen graph)
OVERALL_LIMIT, MIN_GAIN = 0.01, 0.02  # mph, applied to unrounded seed means
TUNE_SEEDS, INNER_SEEDS, FINAL_SEEDS = (0, 1), (0, 1, 2), (0, 1, 2, 3, 4)
PER_ROAD, MIN_LOW_READINGS, MIN_LOW_DAYS = 3, 30, 3
BOOT_DRAWS, BOOT_SEED = 10000, 20261123

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("command", choices=["plan", "train", "decide", "final", "status"])
parser.add_argument("--stage", choices=STAGES)
parser.add_argument("--label")
parser.add_argument("--seed", type=int)
parser.add_argument("--dataset", default="METR-LA", choices=["METR-LA"])
parser.add_argument("--project-dir", type=Path, default=Path(f"/shared/{user}/express_links_replication"))
parser.add_argument("--cache", type=Path, default=Path(f"/shared/{user}/express_links_cache"))
args = parser.parse_args()

ROOT = args.project_dir / "tuning_output" / args.dataset
GRAPHS, DECISIONS, TASKS = ROOT / "graphs", ROOT / "decisions", ROOT / "tasks"
for folder in (GRAPHS, DECISIONS, TASKS):
    folder.mkdir(parents=True, exist_ok=True)


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float))
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def decision(stage):
    path = DECISIONS / f"{stage}.json"
    if not path.exists():
        raise SystemExit(f"Stage '{stage}' has no decision yet. Run its jobs and `decide --stage {stage}` first.")
    return read_json(path)


def label_for(hops, budget, mult, alloc="fo"):
    return f"{alloc}_h{hops}_b{budget}_w{mult:g}"


# ---------------------------------------------------------------- pipeline (replication code, unchanged)
R = replication_core.load(args.dataset, args.project_dir, args.cache)
PROFILE = R["PROFILE"]                      # "full" on the HPC; "smoke" only for local tests
source, prepared, frozen, el_targets = replication_core.prepare(R, args.dataset, ROOT / "audit")
ids, adj = list(prepared["ids"]), prepared["adj"]
off_diag = ~np.eye(len(adj), dtype=bool)
MEDIAN_WEIGHT = float(np.median(adj[(adj > 0) & off_diag]))
fingerprint = R["graph_fingerprint"]


def inner_split():
    """The selector's own chronological split of the training windows (fit block / last 20%)."""
    end = prepared["summary"]["raw_train_end_exclusive"]
    cut = int(.8 * end)
    origins = prepared["split"]["train"]
    return origins[origins + 12 < cut], origins[origins + 1 >= cut]


_inner = {}


def inner_prepared():
    """Preprocessing for the hardest-road model, fitted on the first 80% of training only.

    Same formulas as run_replication.prepare_dataset (strict): causal forward fill up to 3
    steps, then per-road medians; scaler from the training input windows with their
    multiplicity; low-speed cutoff = lowest training speed quartile. Here every fitted
    statistic uses only rows before the selector's cut, so the last 20% of training, which
    is used to rank roads, never informs the preprocessing."""
    if not _inner:
        raw = source[0]
        end = prepared["summary"]["raw_train_end_exclusive"]
        cut = int(.8 * end)
        first, later = inner_split()
        valid = np.isfinite(raw) & (raw > 0)
        fit_values = raw[:cut][valid[:cut]]
        masked = pd.DataFrame(raw).where(valid)
        medians = masked.iloc[:cut].median().fillna(np.median(fit_values))
        inputs = masked.ffill(limit=3).fillna(medians).to_numpy()
        assert np.array_equal(inputs[valid], raw[valid]) and np.isfinite(inputs).all()
        counts = np.bincount((first[:, None] + R["X_OFF"]).ravel(), minlength=len(raw))
        assert counts[cut:].sum() == 0, "inner training windows reach past the cut"
        weights = counts / counts.sum()
        mean = float(weights @ inputs.mean(axis=1))
        std = float(np.sqrt(weights @ np.square(inputs - mean).mean(axis=1)))
        bins = np.unique(np.quantile(fit_values, [.25, .5, .75]))
        _inner.update(prepared)
        _inner.update(scaled=((inputs - mean) / std).astype(np.float32),
                      scaler=R["gwn_util"].StandardScaler(mean, std), bins=bins,
                      split=dict(train=first, val=later, test=later))
    return _inner


_scores = {}


def ranked_candidates(hops):
    """Forecast-only candidate links for a minimum-hop threshold: per target road, the 16-link
    shortlist by fit-block gain, scored by held-out gain on the last 20% of training
    (run_replication.predictive_scores, unchanged)."""
    if hops not in _scores:
        R["MIN_HOPS"] = hops
        eligible, hop_matrix, _, resistance = R["graph_candidates"](adj)
        scored, _ = R["predictive_scores"](source, prepared, eligible, resistance)
        R["MIN_HOPS"] = 3
        _scores[hops] = (scored["forecast_only"], eligible, hop_matrix, resistance)
    return _scores[hops]


def select_links(hops, budget, per_road=1, road_order=None):
    rows, eligible, *_ = ranked_candidates(hops)
    rows = [r for r in rows if r[2] > 0]                     # positive held-out gain only
    chosen, per_target = [], {}
    if road_order is None:                                   # standard: best gains first
        for a, b, score in rows:
            if len(chosen) == budget:
                break
            if per_target.get(b, 0) < per_road:
                chosen.append((a, b, score))
                per_target[b] = per_target.get(b, 0) + 1
    else:                                                    # error-based: hardest roads first
        by_road = {}
        for a, b, score in rows:
            by_road.setdefault(b, []).append((a, b, score))
        for b in road_order:
            for link in by_road.get(b, [])[:per_road]:
                if len(chosen) < budget:
                    chosen.append(link)
    if len(chosen) < budget:
        raise RuntimeError(f"Only {len(chosen)} positive-gain links available for hops>={hops}, budget {budget}")
    return chosen, eligible


def write_graph(label, links, mult, hops, extra=None):
    _, _, hop_matrix, resistance = ranked_candidates(hops)
    weight = MEDIAN_WEIGHT * mult
    graph = adj.copy()
    rows = []
    for a, b, score in links:
        assert a != b and graph[a, b] == 0, (label, a, b)
        graph[a, b] = weight
        rows.append(dict(source_sensor=ids[a], target_sensor=ids[b], weight=weight, score=score,
                         original_hops=int(hop_matrix[a, b]), original_pair_resistance=float(resistance[a, b])))
    R["transition_direction_check"](graph, [(a, b) for a, b, _ in links])
    assert np.count_nonzero(graph) - np.count_nonzero(adj) == len(links)
    np.savez_compressed(GRAPHS / f"{label}.npz", adjacency=graph, sensor_id=np.asarray(ids, dtype=str))
    pd.DataFrame(rows).to_csv(GRAPHS / f"{label}_links.csv", index=False)
    meta = dict(label=label, hops=hops, budget=len(links), weight_mult=mult, weight=weight,
                targets=len({b for _, b, _ in links}), adjacency_sha256=fingerprint(graph), **(extra or {}))
    save_json(GRAPHS / f"{label}.json", meta)
    return meta


def build(label):
    """Create the graph for a label if it does not exist yet."""
    if (GRAPHS / f"{label}.json").exists():
        return read_json(GRAPHS / f"{label}.json")
    if label == "original" or label == "inner_original":
        np.savez_compressed(GRAPHS / f"{label}.npz", adjacency=adj, sensor_id=np.asarray(ids, dtype=str))
        meta = dict(label=label, budget=0, adjacency_sha256=fingerprint(adj))
        save_json(GRAPHS / f"{label}.json", meta)
        return meta
    if label.startswith("random_"):
        base = label[len("random_"):]                       # random_<final label>
        meta = read_json(GRAPHS / f"{base}.json")
        base_links = pd.read_csv(GRAPHS / f"{base}_links.csv", dtype={"source_sensor": str, "target_sensor": str})
        _, eligible, *_ = ranked_candidates(meta["hops"])
        rng, links, used = np.random.default_rng(0), [], set()
        for target in base_links.target_sensor:            # same target roads, random eligible sources
            b = ids.index(target)
            pool = [a for a in np.flatnonzero(eligible[:, b]) if (int(a), b) not in used]
            a = int(rng.choice(pool))
            used.add((a, b))
            links.append((a, b, None))
        return write_graph(label, links, meta["weight_mult"], meta["hops"], dict(matched_to=base))
    kind, h, b, w = label.split("_")
    hops, budget, mult = int(h[1:]), int(b[1:]), float(w[1:])
    if kind == "fo":
        links, _ = select_links(hops, budget)
        meta = write_graph(label, links, mult, hops)
        if label == BASELINE:
            assert meta["adjacency_sha256"] == fingerprint(frozen["forecast_only"][0]), \
                "Rebuilt 25-link Forecast-only graph differs from the frozen replication graph"
        return meta
    if kind == "std3":
        links, _ = select_links(hops, budget, per_road=PER_ROAD)
        return write_graph(label, links, mult, hops)
    if kind == "err3":
        order = decision("inner")["road_order"]
        links, _ = select_links(hops, budget, per_road=PER_ROAD, road_order=[ids.index(r) for r in order])
        return write_graph(label, links, mult, hops, dict(allocation="hardest roads first"))
    raise ValueError(label)


def load_graph(label):
    meta = read_json(GRAPHS / f"{label}.json")
    with np.load(GRAPHS / f"{label}.npz") as payload:
        graph = payload["adjacency"]
    assert fingerprint(graph) == meta["adjacency_sha256"]
    links = pd.read_csv(GRAPHS / f"{label}_links.csv", dtype=str) if (GRAPHS / f"{label}_links.csv").exists() else []
    edges = [(ids.index(a), ids.index(b), None) for a, b in zip(links.source_sensor, links.target_sensor)] if len(links) else []
    return graph, edges


# ---------------------------------------------------------------- training and scoring
def view_for(label, scoring):
    graph, edges = load_graph(label)
    if label == "inner_original":                           # first 80% of training only (see inner_prepared)
        view = R["variant_data"](inner_prepared(), (graph, ids, edges))
        view["shared_targets"] = el_targets
        return view
    view = R["variant_data"](prepared, (graph, ids, edges))
    view["shared_targets"] = el_targets                     # the original 25-road metric
    split = dict(view["split"])
    if scoring == "val":
        split["test"] = split["val"]                        # scoring slot -> validation
    view["split"] = split
    return view


def run_dir(scoring, label, seed):
    return ROOT / scoring / PROFILE / args.dataset / label / f"seed_{seed}"


REPLICATION_NAMES = {"original": "original", BASELINE: "forecast_only"}


def adopt_replication_checkpoint(label, seed, dest):
    """Reuse a finished replication run of the same graph and seed instead of retraining."""
    name = REPLICATION_NAMES.get(label)
    src = args.project_dir / "pilot_output" / PROFILE / args.dataset / str(name) / f"seed_{seed}"
    if name is None or not (src / "last_resume.pt").exists() or not (src / "config.json").exists():
        return False
    cfg = read_json(src / "config.json")
    graph, _ = load_graph(label)
    state = torch.load(src / "last_resume.pt", map_location="cpu", weights_only=True)
    done = len(state["history"]) == cfg["epochs"]
    if cfg.get("graph_variant_sha256") != fingerprint(graph) or not done:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "last_resume.pt", dest / "last_resume.pt")
    (dest / "ADOPTED_FROM.txt").write_text(str(src) + "\n")
    print(f"Reusing finished replication checkpoint {src}", flush=True)
    return True


def train(label, seed, scoring="val"):
    out = run_dir(scoring, label, seed)
    if (out / "result.json").exists():
        print(f"{label} seed {seed} ({scoring}): already finished")
        return read_json(out / "result.json")
    if scoring == "test":
        src = run_dir("val", label, seed) / "last_resume.pt"
        assert src.exists(), f"Train {label} seed {seed} first"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / "last_resume.pt")           # resume at epoch 100 -> score only
    elif label != "inner_original" and not (out / "last_resume.pt").exists():
        adopt_replication_checkpoint(label, seed, out)
    R["OUTPUT"] = ROOT / scoring                            # run_experiment writes OUTPUT/PROFILE/...
    R["run_experiment"](view_for(label, scoring), source[-1], label, seed)
    return read_json(out / "result.json")


# ---------------------------------------------------------------- stage plans
def stage_tasks(stage):
    if stage == "grid":
        labels = ["original"] + [label_for(h, b, 1) for h in HOPS for b in BUDGETS]
        return [(l, s) for l in labels for s in TUNE_SEEDS]
    if stage == "weights":
        return [(label_for(c["hops"], c["budget"], m), s) for c in decision("grid")["top2"]
                for m in WEIGHT_MULTS for s in TUNE_SEEDS]
    if stage == "inner":
        return [("inner_original", s) for s in INNER_SEEDS]
    if stage == "alloc":
        w = decision("weights")["winner"]
        return [(f"{kind}_h{w['hops']}_b{w['budget']}_w{w['weight_mult']:g}", s)
                for kind in ("std3", "err3") for s in TUNE_SEEDS]
    if stage == "confirm":
        final = decision("alloc")["final"]
        labels = list(dict.fromkeys(["original", BASELINE, final, f"random_{final}"]))
        return [(l, s) for l in labels for s in FINAL_SEEDS]
    raise ValueError(stage)


def plan(stage):
    if stage == "final":
        raise SystemExit("`final` has no training tasks; run `final` after `decide --stage confirm`.")
    tasks = stage_tasks(stage)
    for label in dict.fromkeys(l for l, _ in tasks):
        meta = build(label)
        print(f"graph {label}: {meta.get('budget', 0)} links, weight {meta.get('weight', 0):.4f}, "
              f"{meta.get('targets', 0)} target roads")
    (TASKS / f"{stage}.txt").write_text("".join(f"{l} {s}\n" for l, s in tasks))
    print(f"{len(tasks)} tasks -> {TASKS / f'{stage}.txt'}")


# ---------------------------------------------------------------- decisions (validation only)
def val_table(labels, seeds):
    rows = []
    for label in labels:
        for seed in seeds:
            path = run_dir("val", label, seed) / "result.json"
            if not path.exists():
                raise SystemExit(f"Missing result: {label} seed {seed}. Finish the stage's jobs first.")
            r = read_json(path)
            rows.append(dict(label=label, seed=seed, band1=r["band1_mae60"], overall=r["mae60"],
                             roads25=r["target_low_mae60"], best_epoch=r["best_epoch"]))
    table = pd.DataFrame(rows)
    return table, table.groupby("label")[["band1", "overall", "roads25"]].mean()


def show(means, base, title):
    view = means.copy()
    for k in ("band1", "overall", "roads25"):
        view[f"d_{k}"] = view[k] - means.loc[base, k]
    print(f"\n=== {title} (validation, mean of seeds; d_ = change vs {base})")
    print(view.sort_values("band1").to_string(float_format=lambda v: f"{v:+.4f}" if abs(v) < 1 else f"{v:.4f}"))


def eligible_rows(means):
    limit = means.loc[BASELINE, "overall"] + OVERALL_LIMIT
    return means[(means.overall <= limit) & (means.index != "original")]


def parse(label):
    kind, h, b, w = label.split("_")
    return dict(label=label, hops=int(h[1:]), budget=int(b[1:]), weight_mult=float(w[1:]))


def decide(stage):
    if stage == "grid":
        labels = [l for l, _ in dict.fromkeys(stage_tasks("grid"))]
        labels = list(dict.fromkeys(labels))
        table, means = val_table(labels, TUNE_SEEDS)
        show(means, BASELINE, "Grid")
        ok = eligible_rows(means).sort_values("band1")
        top2 = [parse(l) for l in ok.index[:2]]
        result = dict(eligible=list(ok.index), top2=top2, means=means.reset_index().to_dict("records"))
        print("\nEligible (overall MAE within +0.01 of current Forecast-only):", list(ok.index))
        print("Two best eligible set-ups get the weight test:", [t["label"] for t in top2])
    elif stage == "weights":
        grid = [r["label"] for r in decision("grid")["means"]]
        labels = list(dict.fromkeys(grid + [l for l, _ in stage_tasks("weights")]))
        table, means = val_table(labels, TUNE_SEEDS)
        show(means, BASELINE, "Grid + weights")
        ok = eligible_rows(means).sort_values("band1")
        best = ok.index[0]
        gain = means.loc[BASELINE, "band1"] - means.loc[best, "band1"]
        winner = parse(best) if best != BASELINE and gain >= MIN_GAIN else parse(BASELINE)
        result = dict(best_eligible=best, gain_vs_current=gain, winner=winner,
                      means=means.reset_index().to_dict("records"))
        print(f"\nBest eligible: {best}, band-1 gain vs current Forecast-only {gain:+.5f} mph "
              f"(needs >= {MIN_GAIN}) -> winner: {winner['label']}")
    elif stage == "inner":
        per_road = []
        cutoff = float(inner_prepared()["bins"][0])        # first-80% cutoff, as the inner model
        for seed in INNER_SEEDS:
            with np.load(run_dir("val", "inner_original", seed) / "predictions_12_horizons.npz") as p:
                pred, truth, obs = p["predictions"][:, -1], p["targets"][:, -1], p["target_observed"][:, -1]
                days = pd.to_datetime(p["target_timestamp"][:, -1]).normalize()
            low = obs & (truth < cutoff)
            err = np.where(low, np.abs(pred - truth), np.nan)
            n_days = [len(np.unique(days[low[:, j]])) for j in range(low.shape[1])]
            per_road.append(pd.DataFrame(dict(road=ids, seed=seed, mae=np.nanmean(err, axis=0),
                                              n=low.sum(axis=0), days=n_days)))
        roads = pd.concat(per_road).groupby("road").agg(mae=("mae", "mean"), n=("n", "first"), days=("days", "first"))
        roads["enough"] = (roads.n >= MIN_LOW_READINGS) & (roads.days >= MIN_LOW_DAYS)
        roads = roads.sort_values(["enough", "mae"], ascending=[False, False])
        save_json(DECISIONS / "inner_roads.json", roads.reset_index().to_dict("records"))
        result = dict(road_order=list(roads.index), roads_with_enough_data=int(roads.enough.sum()),
                      cutoff=cutoff)
        print(roads.head(15).to_string())
        print(f"\n{int(roads.enough.sum())} of {len(roads)} roads have >= {MIN_LOW_READINGS} low-speed readings "
              f"on >= {MIN_LOW_DAYS} days in the last 20% of training; the rest are ranked last.")
    elif stage == "alloc":
        w = decision("weights")["winner"]
        labels = [w["label"]] + [l for l, _ in dict.fromkeys(stage_tasks("alloc"))]
        labels = list(dict.fromkeys([BASELINE] + labels))
        table, means = val_table(labels, TUNE_SEEDS)
        show(means, BASELINE, "Allocation")
        err, std = [l for l in labels if l.startswith("err3")][0], [l for l in labels if l.startswith("std3")][0]
        limit = means.loc[BASELINE, "overall"] + OVERALL_LIMIT
        better = min(means.loc[w["label"], "band1"], means.loc[std, "band1"]) - means.loc[err, "band1"]
        adopt = better >= MIN_GAIN and means.loc[err, "overall"] <= limit
        final = err if adopt else w["label"]
        result = dict(winner=w["label"], standard=std, error_based=err, error_gain=better,
                      error_based_adopted=bool(adopt), final=final, means=means.reset_index().to_dict("records"))
        print(f"\nError-based vs best of (winner, standard-3): {better:+.5f} mph (needs >= {MIN_GAIN}, "
              f"overall within limit: {means.loc[err, 'overall'] <= limit}) -> FINAL SET-UP: {final}")
    elif stage == "confirm":
        table, means = val_table([l for l, _ in dict.fromkeys(stage_tasks("confirm"))], FINAL_SEEDS)
        show(means, BASELINE, "Final set-ups, seeds 0-4")
        result = dict(ready_for_test=True)
    save_json(DECISIONS / f"{stage}.json", result)
    print(f"\nDecision saved: {DECISIONS / f'{stage}.json'}")


# ---------------------------------------------------------------- final test (opened once)
def day_sums(label, metric_mask):
    """Per test day, summed absolute 60-minute error and count, for each seed."""
    out = []
    for seed in FINAL_SEEDS:
        with np.load(run_dir("test", label, seed) / "predictions_12_horizons.npz") as p:
            pred, truth, obs = p["predictions"][:, -1], p["targets"][:, -1], p["target_observed"][:, -1]
            days = pd.to_datetime(p["target_timestamp"][:, -1]).normalize()
        mask = metric_mask(truth, obs)
        frame = pd.DataFrame(dict(day=np.repeat(days, mask.shape[1])[mask.ravel()],
                                  err=np.abs(pred - truth)[mask]))
        out.append(frame.groupby("day").err.agg(["sum", "count"]))
    return out


def paired_interval(a, b, metric_mask):
    sa, sb = day_sums(a, metric_mask), day_sums(b, metric_mask)
    days = sa[0].index
    diff = np.mean([x["sum"].reindex(days).to_numpy() - y["sum"].reindex(days).to_numpy() for x, y in zip(sa, sb)], axis=0)
    count = sa[0]["count"].reindex(days).to_numpy()
    for x, y in zip(sa, sb):
        assert np.array_equal(x["count"].reindex(days), y["count"].reindex(days))
    point = diff.sum() / count.sum()
    draws = np.random.default_rng(BOOT_SEED).integers(len(days), size=(BOOT_DRAWS, len(days)))
    boot = diff[draws].sum(axis=1) / count[draws].sum(axis=1)
    return point, np.percentile(boot, [2.5, 97.5]), len(days)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_inputs(final_label, labels):
    """Everything the test comparison depends on, fixed before any test prediction is made."""
    graphs = {}
    for label in labels:
        meta = read_json(GRAPHS / f"{label}.json")
        graph, _ = load_graph(label)                        # also re-checks the stored fingerprint
        graphs[label] = meta["adjacency_sha256"]
    checkpoints = {f"{label}/seed_{seed}": file_sha(run_dir("val", label, seed) / "last_resume.pt")
                   for label in labels for seed in FINAL_SEEDS}
    return dict(final=final_label, current=BASELINE, comparisons=labels, seeds=list(FINAL_SEEDS),
                graph_sha256=graphs, checkpoint_sha256=checkpoints, metric="band1_mae60 (60 min, all roads)",
                overall_limit_mph=OVERALL_LIMIT, interval="paired 95% day-block bootstrap",
                boot_draws=BOOT_DRAWS, boot_seed=BOOT_SEED,
                rule=("tuned set-up passes if the upper 95% bound of (final - current Forecast-only) and of "
                      "(final - matched random) on band-1 MAE are both below 0, and mean overall MAE over "
                      "seeds is at most 0.01 mph above current Forecast-only. If no tuned set-up won on "
                      "validation, the result is 'no tuned improvement' and current Forecast-only vs its "
                      "matched random is reported as a separate yes/no (upper bound below 0)."),
                decisions={s: file_sha(DECISIONS / f"{s}.json") for s in ("grid", "weights", "inner", "alloc", "confirm")})


def final():
    decision("confirm")
    final_label = decision("alloc")["final"]
    labels = list(dict.fromkeys(["original", BASELINE, final_label, f"random_{final_label}"]))
    lock = DECISIONS / "final_lock.json"
    current = frozen_inputs(final_label, labels)
    if lock.exists():
        locked = read_json(lock)
        changed = [k for k in current if locked.get(k) != current[k]]
        if changed:
            raise SystemExit(f"Inputs differ from the locked test plan ({changed}); refusing to score. "
                             f"Lock file: {lock}")
        print("Resuming with the locked test plan.")
    else:
        save_json(lock, current)
        print(f"Test plan locked before scoring: {lock}")
    tuned = final_label != BASELINE
    if not tuned:
        print("No tuned improvement: no set-up beat current Forecast-only on validation.")
    rows = []
    for label in labels:
        for seed in FINAL_SEEDS:
            r = train(label, seed, scoring="test")
            rows.append(dict(label=label, seed=seed, band1=r["band1_mae60"], overall=r["mae60"],
                             roads25=r["target_low_mae60"], evaluation=r["evaluation_sha256"]))
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "final_test_results.csv", index=False)
    assert table.evaluation.nunique() == 1, "Test slices differ between runs"
    cutoff = float(prepared["bins"][0])
    el_idx = np.flatnonzero(np.isin(ids, el_targets))
    metrics = {
        "band1 (all roads, low speed)": lambda t, o: o & (t < cutoff),
        "overall (all roads)": lambda t, o: o,
        "25 roads, low speed": lambda t, o: o & (t < cutoff) & np.isin(np.arange(t.shape[1]), el_idx)[None, :],
    }
    means = table.groupby("label")[["band1", "overall", "roads25"]].mean().reindex(labels)
    print(f"\n=== METR-LA TEST, seeds 0-4, final set-up = {final_label}")
    print(means.to_string(float_format=lambda v: f"{v:.4f}"))
    per_seed = table.pivot(index="seed", columns="label", values="band1")
    intervals = {}
    for other in [l for l in labels if l != final_label]:
        print(f"\n{final_label} minus {other}:")
        d = per_seed[final_label] - per_seed[other]
        print(f"  band1 per seed: {', '.join(f'{v:+.3f}' for v in d)}  (better on {int((d < 0).sum())}/5)")
        for name, mask in metrics.items():
            point, ci, n = paired_interval(final_label, other, mask)
            intervals[(other, name)] = (point, ci)
            print(f"  {name}: {point:+.4f} mph, 95% day-bootstrap [{ci[0]:+.4f}, {ci[1]:+.4f}] over {n} days")
    band = "band1 (all roads, low speed)"
    random_label = f"random_{final_label}"
    beats_random = bool(intervals[(random_label, band)][1][1] < 0)
    if tuned:
        beats_current = bool(intervals[(BASELINE, band)][1][1] < 0)
        overall_gap = float(means.loc[final_label, "overall"] - means.loc[BASELINE, "overall"])
        overall_ok = overall_gap <= OVERALL_LIMIT
        passed = beats_current and beats_random and overall_ok
        verdict = dict(result="PASS" if passed else "FAIL", tuned_improvement=True,
                       beats_current_forecast_only=beats_current, beats_matched_random=beats_random,
                       overall_gap_mph=overall_gap, overall_within_limit=overall_ok)
    else:
        verdict = dict(result="NO TUNED IMPROVEMENT", tuned_improvement=False,
                       current_forecast_only_beats_its_matched_random=beats_random)
    verdict.update(final=final_label, intervals={f"{a} | {b}": dict(point=p, ci95=list(c))
                                                 for (a, b), (p, c) in intervals.items()})
    save_json(DECISIONS / "final_result.json", verdict)
    print(f"\nRESULT: {verdict['result']}  " +
          ", ".join(f"{k}={v}" for k, v in verdict.items() if k not in ("result", "intervals", "final")))
    print(f"Saved: {DECISIONS / 'final_result.json'}")


if __name__ == "__main__":
    if args.command == "plan":
        plan(args.stage)
    elif args.command == "train":
        build(args.label)
        train(args.label, args.seed)
    elif args.command == "decide":
        decide(args.stage)
    elif args.command == "final":
        final()
    elif args.command == "status":
        for f in sorted(DECISIONS.glob("*.json")):
            print(f.name)
