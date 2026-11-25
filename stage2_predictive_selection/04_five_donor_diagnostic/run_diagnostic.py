#!/usr/bin/env python3
"""Frozen METR-LA source-replacement analysis, with no model training."""
import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data_support as data
from selector_support import graph_candidates, graph_fingerprint, predictive_scores

MODES = ("original", "express_links", "random_matched")
SEEDS = (0, 1, 2)
MARGIN = .01
BOOT_SEED = 20261119
DRAWS = 10_000
MIN_TARGETS, MIN_DAYS, DONOR_COUNT = 15, 7, 5
EXPECTED_GRAPH = "d17f1885980935364f2ee1f3409f1083ed264e711a07a9e22aa14fdd7d945cce"
EXPECTED_SPEED = "e00a068ca5ec0abcf225b5a044bf962cc1c32c4fef31bad8187c04f342ef0990"
EXPECTED_LINKS = Path(__file__).with_name("METR-LA_Audited_Express_Links.csv")

@dataclass
class Cases:
    source: int
    target: int
    origins: np.ndarray
    donors: np.ndarray

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_json(path):
    return json.loads(path.read_text())

def as_graph(path, ids):
    with np.load(path, allow_pickle=False) as arr:
        require(arr["sensor_id"].astype(str).tolist() == ids, f"Sensor order differs: {path}")
        return arr["adjacency"]

def link_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def setup(args):
    base = args.project_dir / "pilot_output"
    frozen = base / "graphs" / "METR-LA"
    cache = args.cache
    repo = cache / "gwn-6b162e80"
    require((repo / "model.py").exists(), f"Missing prepared Graph WaveNet code in {repo}")
    sys.path.insert(0, str(repo))
    import engine
    import util
    data.gwn_util = util
    data.DEVICE = torch.device("cuda" if torch.cuda.is_available() and not args.preflight else "cpu")
    if data.DEVICE.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    src = data.validate_data("METR-LA", cache / "data/metr-la.h5", cache / "data/adj_mx.pkl")
    require(src[-1]["speeds_sha256"] == EXPECTED_SPEED, "HPC speed file differs from the frozen audit.")
    saved_audit = read_json(base / "audit/METR-LA/preprocessing.json")
    prep = data.prepare_dataset("METR-LA", src, "strict", args.output / "preprocessing_check")
    for key in ["raw_train_end_exclusive", "raw_val_end_exclusive", "speed_band_cutpoints_training"]:
        require(prep["summary"][key] == saved_audit[key], f"Saved audit differs at {key}")
    adj, ids = prep["adj"], prep["ids"]
    require(graph_fingerprint(adj) == EXPECTED_GRAPH, "HPC original graph differs from the frozen audit.")
    adj_by_mode = {m: as_graph(frozen / f"{m}_graph.npz", ids) for m in MODES}
    manifest = read_json(frozen / "frozen_graphs.json")
    for m in MODES:
        for ending, key in [("graph.npz", "file_sha256"), ("links.csv", "links_sha256")]:
            path = frozen / f"{m}_{ending}"
            require(data.sha256(path) == manifest[m][key], f"Changed frozen file: {path}")
    require(np.array_equal(adj_by_mode["original"], adj), "Frozen original graph differs from the source graph.")
    indices = {name:i for i,name in enumerate(ids)}
    links = link_rows(frozen / "express_links_links.csv")
    audited = link_rows(EXPECTED_LINKS)
    key = lambda x: (x["source_sensor"], x["target_sensor"])
    require([key(r) for r in links] == [key(r) for r in audited], "HPC Express Links pairs differ from the audited pairs.")
    require(np.allclose([float(r["score"]) for r in links], [float(r["score"]) for r in audited],
                        rtol=1e-8, atol=1e-9), "Express Links fit-block scores differ from the audit.")
    pairs = [(indices[r["source_sensor"]], indices[r["target_sensor"]], float(r["score"]))
             for r in links if float(r["score"]) > 0]
    require(len(pairs) == 25 and len({t for _,t,_ in pairs}) == 25, "Expected 25 positive links with distinct targets.")
    for m in ("express_links", "random_matched"):
        g = adj_by_mode[m]
        records = link_rows(frozen / f"{m}_links.csv")
        new_arcs = set(zip(*np.where((adj == 0) & (g > 0))))
        stated = {(indices[r["source_sensor"]], indices[r["target_sensor"]]) for r in records}
        require(new_arcs == stated and len(stated) == 25, f"{m} has different arcs from its CSV")
        require({b for _,b in stated} == {b for _,b,_ in pairs}, f"{m} does not share the target set")
        require(np.array_equal(g[adj != 0], adj[adj != 0]), f"{m} changes original edges.")
    return src, prep, adj_by_mode, pairs, engine, util

def build_matcher(src, prep):
    raw, idx, *_ = src
    val, test = prep["split"]["val"], prep["split"]["test"]
    offsets = np.arange(-11, 1)
    minute = np.asarray(idx.hour * 60 + idx.minute)
    weekend = np.asarray(idx.dayofweek >= 5)
    days = idx.values.astype("datetime64[D]")
    missing = raw <= 0
    weights = prep["adj"].astype(float).copy()
    np.fill_diagonal(weights, 0)
    weights += weights.T.copy()
    weighted = weights / np.maximum(weights.sum(axis=1)[:, None], np.finfo(float).eps)
    # Prepared speeds are scaled by one global affine transform within the dataset.
    # Standardizing each matching feature on training origins removes that common scale.
    speeds = prep["scaled"]
    train = prep["split"]["train"]
    cutoff = float(prep["bins"][0])

    def for_pair(source, target):
        usable = ~missing[:, source][np.arange(len(raw))[:, None][11:] + offsets].any(axis=1)
        donor = val[usable[val - 11]]
        truth = raw[test + 12, target]
        origins = test[usable[test - 11] & (truth > 0) & (truth < cutoff)]
        def features(rows):
            return np.column_stack([speeds[rows[:, None] + offsets, target],
                                    speeds[rows] @ weighted[target]])
        scale = np.maximum(features(train).std(axis=0, dtype=np.float64), 1e-8)
        dv, tv = features(donor) / scale, features(origins) / scale
        chosen_origins, chosen_donors = [], []
        for k, t in enumerate(origins):
            diff = np.abs(minute[donor] - minute[t])
            eligible = np.flatnonzero((weekend[donor] == weekend[t]) & (np.minimum(diff, 1440-diff) <= 30))
            if len(np.unique(days[donor[eligible]])) < DONOR_COUNT:
                continue
            distance = np.square(dv[eligible] - tv[k]).sum(axis=1)
            rank = eligible[np.lexsort((donor[eligible], distance))]
            used, chosen = set(), []
            for j in rank:
                d = days[donor[j]]
                if d in used:
                    continue
                used.add(d)
                chosen.append(donor[j])
                if len(chosen) == DONOR_COUNT:
                    break
            chosen_origins.append(t)
            chosen_donors.append(chosen)
        return Cases(source, target, np.asarray(chosen_origins, dtype=int),
                     np.asarray(chosen_donors, dtype=int).reshape(-1, DONOR_COUNT))

    return for_pair, days, cutoff

def coverage(cases, days, ids):
    count = {ids[c.target]:len(c.origins) for c in cases if len(c.origins)}
    selected = np.concatenate([c.origins for c in cases]) if cases else np.array([], dtype=int)
    date, day_count = np.unique(days[selected+12], return_counts=True)
    return dict(roads=len(count), cases=len(selected), days=len(date),
                by_road=count, by_day={str(d):int(n) for d,n in zip(date, day_count)},
                adequate=len(count) >= MIN_TARGETS and len(date) >= MIN_DAYS)

def read_model(base, mode, seed, src, prep, graph, engine, util):
    root = base / "full/METR-LA" / mode / f"seed_{seed}"
    cfg = read_json(root / "config.json")
    require((cfg["dataset"],cfg["mode"],cfg["seed"],cfg["profile"]) == ("METR-LA",mode,seed,"full"),
            f"Wrong run identity: {root}")
    require(cfg["graph_variant_sha256"] == graph_fingerprint(graph), f"Graph does not match {root}")
    require(cfg["speeds_sha256"] == src[-1]["speeds_sha256"] and cfg["adjacency_sha256"] == src[-1]["adjacency_sha256"],
            f"Source files do not match {root}")
    require(cfg["code_commit"] == "6b162e80c59a1d494809252eca055cff93dc66b1"
            and cfg["graph_commit"] == "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
            and cfg["epochs"] == 100 and cfg["adjacency"] == "doubletransition" and not cfg["adaptive"]
            and cfg["batch_size"] == 64 and cfg["input_steps"] == cfg["output_steps"] == 12
            and cfg["diffusion_order"] == 2 and cfg["hidden_channels"] == 32,
            f"Model configuration differs: {root}")
    scaler = prep["scaler"]
    require(np.allclose([scaler.mean, scaler.std], [cfg["mean"],cfg["std"]], rtol=1e-9, atol=0),
            f"Scaler differs: {root}")
    scaler = util.StandardScaler(cfg["mean"],cfg["std"])
    supports = [torch.as_tensor(np.asarray(util.asym_adj(a)), dtype=torch.float32, device=data.DEVICE)
                for a in (graph, graph.T)]
    model = engine.gwnet(data.DEVICE, len(graph), cfg["dropout"], supports=supports,
                        gcn_bool=True, addaptadj=False, aptinit=None, in_dim=2, out_dim=12,
                        residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512).to(data.DEVICE)
    checkpoint = root / "best_weights.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=data.DEVICE, weights_only=True))
    model.eval()
    return model, scaler, root, data.sha256(checkpoint)

def predict_last(model, scaler, prep, origins, source=None, donors=None):
    chunks=[]
    for left in range(0, len(origins), 64):
        stop = left + 64
        x, _ = data.make_batch(prep, origins[left:stop])
        if source is not None:
            rows = donors[left:stop,None] + data.X_OFF
            swaps = torch.as_tensor(prep["scaled"][rows,source], device=data.DEVICE, dtype=x.dtype)
            x[:,0,source,:] = swaps
        out = model(x)
        chunks.append(scaler.inverse_transform(out[:, -1, :, 0]).cpu().numpy())
    return np.concatenate(chunks)

def verify_and_clean(model, scaler, prep, root):
    origins = prep["split"]["test"]
    dates = prep["index"].to_numpy(dtype="datetime64[ns]")
    with np.load(root / "predictions_12_horizons.npz", allow_pickle=False) as z:
        require(z["sensor_id"].astype(str).tolist() == prep["ids"], f"Sensor order differs: {root}")
        require(np.array_equal(z["issue_timestamp"],dates[origins]) and
                np.array_equal(z["target_timestamp"][:,-1],dates[origins+12]) and z["horizon_minutes"][-1]==60,
                f"Saved prediction times differ: {root}")
        raw = prep["labels"][origins+12]
        require(np.array_equal(z["targets"][:,-1],raw) and np.array_equal(z["target_observed"][:,-1],raw>0),
                f"Saved labels or missing mask differ: {root}")
        saved = z["predictions"][:,-1].copy()
    clean = predict_last(model,scaler,prep,origins)
    require(np.allclose(clean,saved,rtol=1e-5,atol=.001),f"Checkpoint did not reproduce forecasts: {root}")
    out = np.full((len(dates),len(prep["ids"])), np.nan, dtype=np.float32)
    out[origins] = clean
    return out,float(np.abs(clean-saved).max())

def accumulate(model, scaler, prep, cases, days, clean):
    sums, counts = {}, {}
    for c in cases:
        n = len(c.origins)
        if not n:
            continue
        orig = np.repeat(c.origins,DONOR_COUNT)
        donors = c.donors.ravel()
        altered=[]
        # Store only the destination output for the replacement.
        for i in range(0,len(orig),256):
            altered.append(predict_last(model,scaler,prep,orig[i:i+256],c.source,donors[i:i+256])[:,c.target])
        y = prep["labels"][c.origins+12,c.target].astype(float)
        delta = np.abs(np.concatenate(altered).reshape(n,DONOR_COUNT)-y[:,None]).mean(axis=1)
        delta -= np.abs(clean[c.origins,c.target]-y)
        selected_days = days[c.origins+12]
        for day in np.unique(selected_days):
            m=selected_days==day
            key=str(day)
            sums[key]=sums.get(key,0.)+float(delta[m].sum())
            counts[key]=counts.get(key,0)+int(m.sum())
    return sums, counts

def bootstrap(left, right):
    for a,b in zip(left,right):
        require(a[1] == b[1] == left[0][1], "Paired day counts differ between arms or seeds.")
    order = sorted(left[0][1])
    n = np.array([left[0][1][d] for d in order])
    diff = np.array([[a[0][d]-b[0][d] for d in order] for a,b in zip(left,right)])
    points = diff.sum(axis=1)/n.sum()
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(len(order),size=(DRAWS,len(order)))
    distribution = (diff[:,draws].sum(axis=2)/n[draws].sum(axis=1)).mean(axis=0)
    lo,hi=np.percentile(distribution,[2.5,97.5])
    wins=int(np.sum(points > MARGIN))
    return dict(delta_mph=float(points.mean()), ci95=[float(lo),float(hi)],
                seed_deltas_mph=points.tolist(), seeds_above_margin=wins,
                passes=bool(lo > MARGIN and wins >= 2))

def control_matches(src,prep,pairs,matcher):
    eligible,hops,_,res=graph_candidates(prep["adj"])
    scores,_=predictive_scores(src,prep,eligible,res,all_candidates=True)
    gains={(i,j):score for i,j,score in scores["control"]}
    high,low,matches=[],[],[]
    for source,target,score in pairs:
        mask=eligible[:,target]
        r=res[mask,target]
        options=[]
        for other in np.flatnonzero(mask & (hops[:,target]==hops[source,target])):
            g=gains.get((int(other),target),-np.inf)
            if g>0 and abs(g-score)<=max(.05,.25*score) and np.mean(r <= res[other,target]) <= .5:
                options.append((abs(g-score),prep["ids"][other],int(other),g))
        if not options:
            continue
        _,_,other,g=min(options)
        a,b=matcher(source,target),matcher(other,target)
        same,ia,ib=np.intersect1d(a.origins,b.origins,return_indices=True)
        high.append(Cases(source,target,same,a.donors[ia]))
        low.append(Cases(other,target,same,b.donors[ib]))
        matches.append(dict(source=prep["ids"][source], target=prep["ids"][target],
                            control_source=prep["ids"][other], hops=int(hops[source,target]),
                            high_gain=score, control_gain=g, shared_cases=len(same)))
    return high,low,matches

def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir",required=True,type=Path)
    parser.add_argument("--cache",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--preflight",action="store_true",help="Verify graph and donor coverage without opening checkpoints.")
    args=parser.parse_args()
    if not args.preflight and not torch.cuda.is_available():
        raise SystemExit("Run the checkpoint diagnostic on an allocated CUDA node, or use --preflight.")
    args.output.mkdir(parents=True,exist_ok=True)
    result_path = args.output/"diagnostic_result.json"
    if not args.preflight and result_path.exists():
        raise SystemExit("A diagnostic result already exists. Choose a new output directory to run again.")
    src,prep,graphs,pairs,engine,util=setup(args)
    matcher,days,cutoff=build_matcher(src,prep)
    cases=[matcher(a,b) for a,b,_ in pairs]
    cov=coverage(cases,days,prep["ids"])
    base=args.project_dir/"pilot_output"
    diagnostic=dict(dataset="METR-LA", source_speed_sha256=src[-1]["speeds_sha256"],
                    original_graph_sha256=EXPECTED_GRAPH, donor_count=DONOR_COUNT, low_speed_cutoff_mph=cutoff,
                    margin_mph=MARGIN, bootstrap_draws=DRAWS, bootstrap_seed=BOOT_SEED, primary_coverage=cov)
    save(args.output/"preflight.json",diagnostic)
    print({k:cov[k] for k in ("roads","days","cases","adequate")},flush=True)
    if args.preflight:
        return
    if not cov["adequate"]:
        diagnostic["primary_status"]="inconclusive_coverage"
        save(result_path,diagnostic)
        return
    by_mode = {mode:[] for mode in MODES}
    diagnostics=[]
    protocol = Path(__file__).with_name("Express_Links_Diagnostic_Decision_Rule.md")
    diagnostic["decision_rule_sha256"]=data.sha256(protocol)
    with torch.inference_mode():
        for mode in MODES:
            for seed in SEEDS:
                model,scaler,root,sha=read_model(base,mode,seed,src,prep,graphs[mode],engine,util)
                clean,check=verify_and_clean(model,scaler,prep,root)
                sums,counts=accumulate(model,scaler,prep,cases,days,clean)
                by_mode[mode].append((sums,counts))
                diagnostics.append(dict(mode=mode,seed=seed,clean_reproduction_max_mph=check,
                                        checkpoint_sha256=sha, reliance_mph=sum(sums.values())/sum(counts.values())))
                print(diagnostics[-1],flush=True)
                del model,clean
    diagnostic["seed_runs"]=diagnostics
    diagnostic["primary_contrasts"]={mode:bootstrap(by_mode["express_links"],by_mode[mode])
                                     for mode in ("original","random_matched")}
    positive=np.mean([item["reliance_mph"] for item in diagnostics if item["mode"]=="express_links"]) > 0
    passed=bool(positive and all(v["passes"] for v in diagnostic["primary_contrasts"].values()))
    diagnostic["primary_status"]="pass" if passed else "not_passed"
    diagnostic["express_links_average_reliance_positive"]=bool(positive)
    save(args.output/"primary_result.json",diagnostic)
    secondary_days = []
    if passed:
        high,low,pairs_out=control_matches(src,prep,pairs,matcher)
        cov2=coverage(high,days,prep["ids"])
        diagnostic["bottleneck_coverage"] = cov2
        diagnostic["bottleneck_pairs"] = pairs_out
        if cov2["adequate"]:
            left,right=[],[]
            with torch.inference_mode():
                for seed in SEEDS:
                    m,s,r,_=read_model(base,"original",seed,src,prep,graphs["original"],engine,util)
                    clean,_=verify_and_clean(m,s,prep,r)
                    low_day = accumulate(m,s,prep,low,days,clean)
                    high_day = accumulate(m,s,prep,high,days,clean)
                    left.append(low_day)
                    right.append(high_day)
                    secondary_days.append(dict(seed=seed,low=low_day,high=high_day))
                    del m,clean
            diagnostic["bottleneck_contrast"]=bootstrap(left,right)
            diagnostic["bottleneck_status"]="pass" if diagnostic["bottleneck_contrast"]["passes"] else "not_passed"
        else:
            diagnostic["bottleneck_status"]="inconclusive_coverage"
    else:
        diagnostic["bottleneck_status"]="not_run_primary_gate"
    save(args.output/"paired_day_sums.json",{"primary":by_mode,"bottleneck":secondary_days})
    save(result_path,diagnostic)
    print("Diagnostic:", result_path, diagnostic["primary_status"], diagnostic["bottleneck_status"],flush=True)

if __name__=="__main__":
    main()
