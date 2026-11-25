"""Generate supplementary LaTeX tables from the packaged original outputs."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NAMES = {
    "original": "Original", "express_links": "Ridge Express Links",
    "forecast_only": "Forecast-only", "gtr": "GTR",
    "random_matched": "Matched random", "virtual_node": "Virtual node",
    "express": "Greedy lead--lag", "random": "Random",
    "map_only": "Resistance only", "corr_only": "Correlation only",
    "same_side": "Same-side",
}
DATASETS = ("METR-LA", "PEMS-BAY")
SPARSE = ("original", "express_links", "forecast_only", "gtr", "random_matched", "virtual_node")
ORIGINAL = ("original", "random", "map_only", "corr_only", "same_side", "express", "virtual_node")
METRICS = ("MAE", "RMSE", "MAPE_pct")


def table(caption, label, headers, rows):
    """Use full-width tables with compact numeric headers."""
    return "\n".join([
        r"\begin{table}[H]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}",
        r"\begin{tabular}{@{}" + "ll" + "r" * (len(headers) - 2) + "@{}}",
        r"\toprule", " & ".join(headers) + r" \\", r"\midrule",
        *[" & ".join(row) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def horizon_rows(horizons, averages):
    results = []
    for dataset in DATASETS:
        for i, mode in enumerate(SPARSE):
            metrics = [
                horizons.loc[(dataset, mode, h)].tolist() for h in (15, 30, 60)
            ] + [averages.loc[(dataset, mode)].tolist()]
            triples = [" / ".join(
                f"{score:.{2 if i == 2 else 3}f}" for i, score in enumerate(scores)
            ) for scores in metrics]
            results.append([dataset if i == 0 else "", NAMES[mode], *triples])
    return results


def main():
    horizons = pd.read_csv(DATA / "replication_all_horizons.csv")
    averages = pd.read_csv(DATA / "replication_mean_12.csv")
    for scores, index in [(horizons, ["dataset", "graph", "horizon_minutes"]),
                          (averages, ["dataset", "graph"])]:
        assert scores.groupby(index).seed.nunique().eq(3).all()
    horizons = horizons.groupby(["dataset", "graph", "horizon_minutes"])[list(METRICS)].mean()
    averages = averages.groupby(["dataset", "graph"])[list(METRICS)].mean()
    body = [table(
        "Six-way Graph WaveNet comparison on all observed test readings. Entries are MAE / RMSE / MAPE, in mph / mph / percent. Each metric averages three seeds. The final column uses the arithmetic mean over the 12 horizons.",
        "horizons", ["Dataset", "Graph", "15 minutes", "30 minutes", "60 minutes", "12-horizon mean"],
        horizon_rows(horizons, averages),
    )]
    original = pd.read_csv(DATA / "original_study.csv")
    fields = ["jam_recall_30", "jam_precision_30", "jam_recall_60", "jam_precision_60"]
    events = original.groupby(["dataset", "graph"])[fields].mean()
    rows = [
        [dataset if i == 0 else "", NAMES[mode],
         *[f"{100*v:.2f}" for v in events.loc[(dataset, mode)]]]
        for dataset in DATASETS for i, mode in enumerate(ORIGINAL)
    ]
    body.append(table(
        "Jam-warning recall and precision (percent) on all roads in the original greedy lead--lag study, averaged over three seeds. These scores require prediction of a jam onset and use the original event rule. The PEMS-BAY tolerance check in the main text measures speed error at observed events.",
        "events", ["Dataset", "Graph", "Recall at 30 min", "Precision at 30 min",
                   "Recall at 60 min", "Precision at 60 min"], rows,
    ))
    models = pd.read_csv(DATA / "second_models.csv")
    means = models.groupby(["model", "graph"])[["target_low_mae60", "band1_mae60", "mae60", "mae_mean_12"]].mean()
    rows = [
        [model.upper() if i == 0 else "", NAMES[mode],
         *[f"{v:.3f}" for v in means.loc[(model, mode)]]]
        for model in ("dcrnn", "dcgru") for i, mode in enumerate(("original", "express_links", "forecast_only", "random_matched"))
    ]
    body.append(table(
        "DCRNN and supplementary DCGRU results on METR-LA. MAE in mph, means over three seeds. The first metric is the shared low-speed target-road slice, Band 1 includes low-speed readings on all roads, and overall results include all observed speeds.",
        "second_models", ["Model", "Graph", "25 roads, low speed", "Band 1", "Overall 60 min", "Overall 12-horizon mean"], rows,
    ))
    (ROOT / "supplement_horizons.tex").write_text(body[0])
    (ROOT / "supplement_other.tex").write_text("\n".join(body[1:]))


if __name__ == "__main__":
    main()
