#!/usr/bin/env python3
"""Where do each method's 25 links sit in the road graph?

Uses the same definitions as run_replication.py (graph_candidates): the road graph is made
symmetric, hops are unweighted shortest paths on it, and effective resistance comes from
the pseudo-inverse of its Laplacian. Eligible pairs are those the selectors could choose:
at least 3 hops apart and not already directly linked.

For each method it reports how far apart its source and target roads are, and where each
link's resistance falls among all eligible pairs (percentile 0 = least resistance, the
best-connected pairs; 100 = most resistance, the weakest connections).
Nothing is modified; it only reads the frozen graphs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

MIN_HOPS = 3
METHODS = ("express_links", "forecast_only", "gtr", "random_matched")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project-dir", type=Path, default=Path("/shared/g00112835/express_links_replication"))
args = parser.parse_args()
graphs = args.project_dir / "pilot_output" / "graphs"

pd.set_option("display.width", 200)
for dataset in ("METR-LA", "PEMS-BAY"):
    folder = graphs / dataset
    if not folder.exists():
        print(f"{dataset}: no graphs found in {folder}")
        continue
    g = np.load(folder / "original_graph.npz")
    adj = np.asarray(g["adjacency"], dtype=float)
    ids = [str(s) for s in g["sensor_id"]]
    index = {s: i for i, s in enumerate(ids)}

    undirected = (adj + adj.T) / 2
    np.fill_diagonal(undirected, 0)
    hops = shortest_path(undirected, directed=False, unweighted=True)
    eligible = np.isfinite(hops) & (hops >= MIN_HOPS) & (adj == 0)
    laplacian = np.diag(undirected.sum(axis=1)) - undirected
    inverse = np.linalg.pinv(laplacian, hermitian=True, rcond=1e-10)
    d = np.diag(inverse)
    resistance = np.maximum(d[:, None] + d[None, :] - 2 * inverse, 0)
    pool = np.sort(resistance[eligible])
    pool_hops = hops[eligible]

    print(f"\n===== {dataset}: {len(ids)} roads, {eligible.sum()} eligible pairs")
    print(f"All eligible pairs: hops median {np.median(pool_hops):.0f} (range {pool_hops.min():.0f}-{pool_hops.max():.0f}), "
          f"resistance median {np.median(pool):.2f}")

    rows = []
    for method in METHODS:
        path = folder / f"{method}_links.csv"
        if not path.exists():
            continue
        links = pd.read_csv(path, dtype={"source_sensor": str, "target_sensor": str})
        a = links.source_sensor.map(index).to_numpy()
        b = links.target_sensor.map(index).to_numpy()
        h, r = hops[a, b], resistance[a, b]
        if "original_pair_resistance" in links and links.original_pair_resistance.notna().all():
            gap = np.abs(r - links.original_pair_resistance.to_numpy(dtype=float)).max()
            assert gap < 1e-6, f"{dataset}/{method}: recomputed resistance differs from the CSV by {gap}"
        pct = 100 * np.searchsorted(pool, r, side="right") / len(pool)
        sources = links.source_sensor.value_counts()
        rows.append({
            "method": method,
            "links": len(links),
            "distinct_sources": len(sources),
            "max_links_per_source": int(sources.max()),
            "hops_median": float(np.median(h)),
            "hops_min": int(h.min()),
            "hops_max": int(h.max()),
            "links_3_4_hops": int((h <= 4).sum()),
            "resistance_median": float(np.median(r)),
            "resistance_pct_median": float(np.median(pct)),
            "links_in_top_quarter_resistance": int((pct > 75).sum()),
            "links_in_bottom_half_resistance": int((pct <= 50).sum()),
        })
    table = pd.DataFrame(rows).set_index("method")
    print(table.round(2).T.to_string())
