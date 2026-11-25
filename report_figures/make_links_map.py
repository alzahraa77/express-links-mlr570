"""Draws figs/links_map.pdf: the 25 links of Express Links, Forecast-only and matched random on the METR-LA road graph.
Reads the frozen graphs in ../stage2_predictive_selection/02_pilot/reference_graphs_METR-LA/ and graph_sensor_locations.csv
(a copy of data/sensor_graph/graph_sensor_locations.csv from chnsh/DCRNN_PyTorch)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REF = _os.path.join(HERE, "..", "stage2_predictive_selection", "02_pilot", "reference_graphs_METR-LA") + _os.sep
_os.makedirs(_os.path.join(HERE, "figs"), exist_ok=True)
LOC = _os.path.join(HERE, "graph_sensor_locations.csv")  # from chnsh/DCRNN_PyTorch, data/sensor_graph/
C = {"Express Links": "#2a78d6", "Forecast-only": "#eb6834", "Matched random": "#1baf7a", "Original": "#8a8984"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "axes.titlesize": 8, "legend.fontsize": 7.5})
modes = [("Express Links", "express_links"), ("Forecast-only", "forecast_only"), ("Matched random", "random_matched")]

# map (full width)
g = np.load(REF + "original_graph.npz"); ids = list(g["sensor_id"]); A = g["adjacency"]
loc = pd.read_csv(LOC, dtype={"sensor_id": str}).set_index("sensor_id").loc[ids]
x, y = loc.longitude.to_numpy(), loc.latitude.to_numpy()
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), sharex=True, sharey=True)
ii, jj = np.nonzero((A > 0) & ~np.eye(len(A), dtype=bool))
for ax, (name, key) in zip(axes, modes):
    for i, j in zip(ii, jj):
        ax.plot([x[i], x[j]], [y[i], y[j]], color=GRID, lw=0.45, zorder=1)
    ax.scatter(x, y, s=2.5, color="#b4b3ad", zorder=2, linewidths=0)
    L = pd.read_csv(REF + f"{key}_links.csv", dtype={"source_sensor": str, "target_sensor": str})
    for s, t in zip(L.source_sensor, L.target_sensor):
        a, b = ids.index(s), ids.index(t)
        ax.add_patch(FancyArrowPatch((x[a], y[a]), (x[b], y[b]), arrowstyle="-|>", mutation_scale=5,
                     color=C[name], lw=0.8, connectionstyle="arc3,rad=0.15", zorder=3, alpha=0.9))
    tg = [ids.index(t) for t in L.target_sensor]
    ax.scatter(x[tg], y[tg], s=9, color=C[name], edgecolors="white", linewidths=0.5, zorder=4)
    ax.set_title(f"{name} (median {L.original_hops.median():.0f} hops)", color=INK)
    ax.set_xlabel("Longitude"); ax.set_aspect(1 / np.cos(np.radians(y.mean())))
    ax.set_xticks([-118.5, -118.4, -118.3, -118.2])
axes[0].set_ylabel("Latitude")
fig.savefig(_os.path.join(HERE, "figs", "links_map.pdf")); plt.close(fig)
