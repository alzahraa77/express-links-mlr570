import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figs"
FIGURES.mkdir(exist_ok=True)
C = {"Express Links": "#2a78d6", "Forecast-only": "#eb6834", "Matched random": "#1baf7a", "Original": "#8a8984"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "axes.titlesize": 8, "legend.fontsize": 7.5})
# two models (full width)
model_columns = ["target_low_mae60", "band1_mae60", "mae60"]
gwn = pd.read_csv(DATA / "replication.csv").query("dataset == 'METR-LA'")
dcrnn = pd.read_csv(DATA / "second_models.csv").query("model == 'dcrnn'")
mode_labels = {"express_links": "Express Links", "forecast_only": "Forecast-only",
               "random_matched": "Matched random"}
delta = {}
for model, scores, mode_column in [("Graph WaveNet", gwn, "mode"), ("DCRNN", dcrnn, "graph")]:
    means = scores.groupby(mode_column)[model_columns].mean()
    delta[model] = {label: (means.loc[mode] - means.loc["original"]).tolist()
                    for mode, label in mode_labels.items()}
titles = ["(a) Low-speed, 25 target roads (primary)", "(b) Low-speed, all roads (Band 1)", "(c) All readings, all roads"]
methods = ["Express Links", "Forecast-only", "Matched random"]
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
for k, ax in enumerate(axes):
    vals = [delta[mo][m][k] for mo in delta for m in methods]; span = max(vals) - min(vals); off = span * 0.035
    for m_i, m in enumerate(methods):
        for g_i, model in enumerate(delta):
            d = delta[model][m][k]; xpos = g_i * 4 + m_i
            ax.bar(xpos, d, width=0.86, color=C[m], label=m if (k == 0 and g_i == 0) else None)
            ax.text(xpos, d + (off if d >= 0 else -off), f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top",
                    fontsize=5.3, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([1, 5]); ax.set_xticklabels(list(delta)); ax.tick_params(axis="x", length=0)
    ax.set_title(titles[k], color=INK); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_ylim(min(vals) - span * 0.2, max(vals) + span * 0.2)
axes[0].set_ylabel("Change in 60-min MAE (mph)")
fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
fig.savefig(FIGURES / "two_models.pdf"); plt.close(fig)

# tuning heatmap (full width)
hops, budgets = [3, 2, 1], [10, 25, 50, 100]
validation = json.loads((DATA / "tuning_validation_decisions.json").read_text())
means = pd.DataFrame(validation["grid"]["means"]).set_index("label")
metric_cols = ["band1", "overall"]
baseline = means.loc["fo_h3_b25_w1", metric_cols]
grid = {(h, b): (means.loc[f"fo_h{h}_b{b}_w1", metric_cols] - baseline).tolist()
        for h in hops for b in budgets}
fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.95), gridspec_kw=dict(wspace=0.62))
for ax, k, title in zip(axes, [0, 1], ["(a) Band 1 MAE, change vs. current Forecast-only", "(b) Overall MAE, change vs. current Forecast-only"]):
    M = np.array([[grid[(h, b)][k] for b in budgets] for h in hops]); lim = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for r in range(3):
        for c in range(4):
            ax.text(c, r, f"{M[r, c]:+.3f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(M[r, c]) > 0.6 * lim else INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{b} links" for b in budgets])
    ax.set_yticks(range(3)); ax.set_yticklabels([f"min. {h} hop{'s' if h > 1 else ''}" for h in hops])
    ax.set_title(title, color=INK)
    for s in ax.spines.values(): s.set_visible(False)
    ax.add_patch(plt.Rectangle((0.5, -0.5), 1, 1, fill=False, ec=INK, lw=1.4))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label("mph", fontsize=6.5)
fig.savefig(FIGURES / "tuning_grid.pdf"); plt.close(fig)

# final test forest (column)
decision = json.loads((DATA / "tuning_final_decision.json").read_text())
contrasts = decision["intervals"]
comparators = [("original", "vs. original graph"), ("fo_h3_b25_w1", "vs. current Forecast-only"),
               (f"random_{decision['final']}", "vs. matched random")]
slices = ["band1 (all roads, low speed)", "overall (all roads)", "25 roads, low speed"]
rows = []
for mode, caption in comparators:
    row = [caption]
    for metric in slices:
        summary = contrasts[f"{mode} | {metric}"]
        row.extend([summary["point"], *summary["ci95"]])
    rows.append(row)
mets = [("Band 1, all roads (primary)", 1, "#eb6834"), ("Overall, all roads", 4, "#8a8984"), ("Low-speed, 25 roads", 7, "#2a78d6")]
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for r, row in enumerate(rows):
    for m, (name, j, col) in enumerate(mets):
        yy = r * 4 + m
        ax.errorbar(row[j], yy, xerr=[[row[j] - row[j+1]], [row[j+2] - row[j]]], fmt="o", ms=3, color=col,
                    elinewidth=1.1, capsize=2, label=name if r == 0 else None)
ax.axvline(0, color=INK2, lw=0.7)
ax.set_yticks([1, 5, 9]); ax.set_yticklabels(["vs. original\ngraph", "vs. current\nForecast-only", "vs. matched\nrandom"]); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Tuned minus comparison, 60-min MAE (mph)")
ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.25), ncol=2, frameon=False, fontsize=6.3)
fig.savefig(FIGURES / "final_test.pdf"); plt.close(fig)
print("ok")

labels = {
    "express": "Greedy lead–lag", "express_links": "Ridge Express Links",
    "forecast_only": "Forecast-only", "random": "Random",
    "random_matched": "Matched random", "virtual_node": "Virtual node",
    "map_only": "Resistance only", "corr_only": "Correlation only",
    "same_side": "Same-side", "gtr": "Greedy total resistance",
}
colors = {"express": "#1666a3", "express_links": "#1666a3",
          "forecast_only": "#a84c0e", "virtual_node": "#20705a"}


def paired_seeds(source, mode_column, metric, modes, filename):
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.25), sharey=True)
    for axis, dataset in zip(axes, ["METR-LA", "PEMS-BAY"]):
        data = source[source.dataset == dataset]
        by_seed = data.pivot(index=mode_column, columns="seed", values=metric)
        differences = by_seed.subtract(by_seed.loc["original"], axis="columns")
        for row, mode in enumerate(modes):
            color = colors.get(mode, "#5d6772")
            points = differences.loc[mode].to_numpy()
            axis.scatter(points, row + np.array([-.17, 0, .17]), s=12, marker="o",
                         facecolors="white", edgecolors=color, linewidth=.75)
            axis.scatter(points.mean(), row, s=27, marker="D", color=color, zorder=3)
        axis.axvline(0, color=INK2, linewidth=.7)
        axis.set_title(dataset, fontweight="bold")
        axis.set_xlabel("Change vs. original graph (mph)")
        axis.set_yticks(range(len(modes)), [labels[mode] for mode in modes])
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", color=GRID, linewidth=.5)
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    figure.tight_layout(w_pad=1.8)
    figure.savefig(FIGURES / filename)
    plt.close(figure)


paired_seeds(pd.read_csv(DATA / "original_study.csv"), "graph", "mae_60",
             ["express", "random", "map_only", "corr_only", "same_side", "virtual_node"],
             "stage_one.pdf")
paired_seeds(pd.read_csv(DATA / "replication.csv"), "mode", "target_low_mae60",
             ["forecast_only", "express_links", "random_matched", "gtr", "virtual_node"],
             "replication_seeds.pdf")

d1 = pd.read_csv(DATA / "diagnostic_week_swap.csv")
d1 = d1[d1.test == "primary_A_minus_B"].sort_values("seed")
d2 = json.loads((DATA / "diagnostic_five_donor.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.83))
axes[0].set_title("D1 · Week swap", fontweight="bold")
axes[1].set_title("D2 · Five matched donors", fontweight="bold")
for i, row in enumerate(d1.itertuples()):
    axes[0].errorbar(row.value, i,
                     xerr=[[row.value - row.ci_low], [row.ci_high - row.value]],
                     color="#1666a3", fmt="o", capsize=2, ms=3.5)
axes[0].set_yticks(range(3), ["Seed 0", "Seed 1", "Seed 2"])
axes[0].axvline(0, linewidth=.9, color="#a84c0e", linestyle="--")
for i, key in enumerate(["original", "random_matched"]):
    row = d2["primary_contrasts"][key]
    value, (lo, hi) = row["delta_mph"], row["ci95"]
    axes[1].errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt="o",
                     capsize=2, ms=3.5, color="#1666a3")
axes[1].set_yticks(range(2), ["vs. original graph", "vs. matched random"])
axes[1].axvline(0, linewidth=.6, color=INK2)
axes[1].axvline(.01, linewidth=.9, color="#a84c0e", linestyle="--")
for axis in axes:
    axis.set_xlabel("Added error contrast (mph)")
    axis.invert_yaxis()
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color=GRID, linewidth=.5)
    axis.set_axisbelow(True)
fig.tight_layout(w_pad=1.8)
fig.savefig(FIGURES / "mechanism.pdf")
plt.close(fig)

horizons = pd.read_csv(DATA / "replication_all_horizons.csv")
modes = ["original", "express_links", "forecast_only", "gtr", "random_matched", "virtual_node"]
styles = [
    ("Original", "#444444", "--", "o"),
    ("Ridge Express Links", "#1d67ac", "-", "^"),
    ("Forecast-only", "#b75310", "-", "s"),
    ("GTR", "#753e99", ":", "P"),
    ("Matched random", "#84651b", "-.", "v"),
    ("Virtual node", "#167d5e", "-", "D"),
]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharey=True)
for axis, dataset in zip(axes, ["METR-LA", "PEMS-BAY"]):
    means = horizons[horizons.dataset == dataset].groupby(["graph", "horizon_minutes"]).MAE.mean()
    for mode, (label, color, line, marker) in zip(modes, styles):
        curve = (means.loc[mode] - means.loc["original"]).sort_index()
        axis.plot(curve.index, curve, color=color, ls=line, marker=marker, ms=2.5,
                  lw=1, markevery=[2, 5, 11], label=label)
    axis.set_title(dataset, fontweight="bold")
    axis.set_xticks([5, 15, 30, 45, 60])
    axis.set_xlabel("Forecast horizon (minutes)")
    axis.grid(axis="y", color=GRID, linewidth=.5)
    axis.set_axisbelow(True)
axes[0].set_ylabel("MAE change vs. original (mph)")
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(.5, -.005), handlelength=2.2)
fig.tight_layout(rect=(0, .13, 1, 1))
fig.savefig(FIGURES / "horizon_profiles.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# How the Stage 1 selection rules spend 25 links (single column)
props = pd.read_csv(DATA / "stage1_link_properties.csv")
rules = [("random", "Random", "#8a8984", "o"), ("map_only", "Resistance only", "#7a5aa6", "s"),
         ("corr_only", "Correlation only", "#1baf7a", "^"), ("same_side", "Same-side", "#c9a227", "v"),
         ("express", "Lead--lag selector", "#2a78d6", "D")]
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.85), sharey=True)
for ax, ds in zip(axes, ["METR-LA", "PEMS-BAY"]):
    sub = props[props.dataset == ds].set_index("version")
    for key, label, col, mk in rules:
        r = sub.loc[key]
        ax.scatter(float(r.mean_rho), float(r.total_R_saved_pct), s=26, color=col, marker=mk,
                   edgecolor="white", linewidth=0.5, zorder=3, label=label.replace("--", "–"))
    ax.set_title(ds, color=INK); ax.set_xlim(0, 0.65); ax.set_ylim(0, 50)
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("Total resistance\nreduction (%)")
fig.supxlabel(r"Mean lead–lag $\rho$ of the 25 added links", fontsize=7.5, y=0.02)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.17), fontsize=6.3, handletextpad=0.2, columnspacing=0.8)
fig.tight_layout()
fig.savefig(FIGURES / "stage1_tradeoff.pdf"); plt.close(fig)

# Properties of the links chosen by the ridge-based selectors (single column)
links = pd.read_csv(DATA / "selected_links.csv")
sel = [("express_links", "Express Links", C["Express Links"]), ("forecast_only", "Forecast-only", C["Forecast-only"]),
       ("gtr", "GTR", "#7a5aa6"), ("random_matched", "Matched random", C["Matched random"])]
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.45))
rng = np.random.default_rng(0)
ax = axes[0]
for d_i, ds in enumerate(["METR-LA", "PEMS-BAY"]):
    for m_i, (mode, label, col) in enumerate(sel):
        v = links[(links.dataset == ds) & (links.graph == mode)].original_pair_resistance.to_numpy()
        x = d_i * 5 + m_i
        ax.scatter(x + rng.uniform(-0.22, 0.22, len(v)), v, s=4, color=col, alpha=0.55, linewidth=0,
                   label=label if d_i == 0 else None)
        ax.plot([x - 0.35, x + 0.35], [np.median(v)] * 2, color=INK, lw=1.1)
ax.set_xticks([1.5, 6.5]); ax.set_xticklabels(["METR-LA", "PEMS-BAY"]); ax.tick_params(axis="x", length=0)
ax.set_ylabel(r"Pair resistance $R(u,v)$"); ax.set_title("(a) Connectivity of chosen pairs", color=INK)
ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax = axes[1]
for d_i, ds in enumerate(["METR-LA", "PEMS-BAY"]):
    for m_i, (mode, label, col) in enumerate(sel[:2]):
        v = links[(links.dataset == ds) & (links.graph == mode)].score.to_numpy()
        x = d_i * 3 + m_i
        ax.scatter(x + rng.uniform(-0.22, 0.22, len(v)), v, s=4, color=col, alpha=0.55, linewidth=0)
        ax.plot([x - 0.35, x + 0.35], [np.median(v)] * 2, color=INK, lw=1.1)
ax.set_xticks([0.5, 3.5]); ax.set_xticklabels(["METR", "PEMS"]); ax.tick_params(axis="x", length=0)
ax.set_ylabel("Held-out gain $g$ (mph)"); ax.set_ylim(0, None); ax.set_title("(b) Predictive gain", color=INK)
ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
h = [Line2D([], [], ls="", marker="o", ms=3.5, color=col) for _, _, col in sel]; l = [lab for _, lab, _ in sel]
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1), fontsize=6.3,
           handletextpad=0.1, columnspacing=0.7)
fig.savefig(FIGURES / "link_properties.pdf"); plt.close(fig)
