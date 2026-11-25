#!/usr/bin/env python3
"""Collect complete paired-seed comparisons without loading forecast arrays."""
import argparse
import csv
import json
import statistics
from pathlib import Path

MODES = ("original", "express_links", "forecast_only", "gtr", "random_matched", "virtual_node")
METRICS = ("mae60", "band1_mae60", "express_target_mae60", "target_low_mae60")
DATASETS = ("METR-LA", "PEMS-BAY")
KEYS = ("epochs", "batch_size", "adaptive", "input_steps", "output_steps", "dropout",
        "learning_rate", "weight_decay", "hidden_channels", "diffusion_order", "gradient_clip",
        "adjacency", "code_commit", "graph_commit", "mean", "std", "notebook_protocol_version",
        "speeds_sha256", "adjacency_sha256")

def read(path):
    return json.loads(path.read_text())

def write(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)

def collect(root, datasets, seeds, profile):
    base = root / "pilot_output"
    output = base / profile
    rows, aggregate, missing = [], [], []
    for dataset in datasets:
        graph_dir = base / "graphs" / dataset
        frozen = read(graph_dir / "frozen_graphs.json")
        graph_info = read(graph_dir / "selector_summary.json")
        checks, signatures = set(), set()
        for seed in seeds:
            for mode in MODES:
                folder = output / dataset / mode / f"seed_{seed}"
                if not (folder / "result.json").exists():
                    missing.append(f"{dataset}/{mode}/seed_{seed}")
                    continue
                result, config = read(folder / "result.json"), read(folder / "config.json")
                assert config["profile"] == profile and config["seed"] == seed and config["mode"] == mode
                assert config["dataset"] == dataset and result["smoke"] == (profile == "smoke")
                assert config["graph_variant_sha256"] == frozen[mode]["adjacency_sha256"]
                assert result["added_arcs"] == graph_info["added_arcs"][mode]
                checks.add(tuple(config[k] for k in KEYS))
                signatures.add((result["evaluation_sha256"], result["target_low_n60"],
                                result["test_origins"], result["target_sensor_n"]))
                rows.append({"dataset": dataset, "mode": mode, "seed": seed, **result})
        if len(signatures) > 1 or len(checks) > 1:
            raise RuntimeError(f"Different evaluation samples or model settings found on {dataset}.")
    if missing:
        raise SystemExit("Waiting for completed runs: " + ", ".join(missing))

    for dataset in datasets:
        group = [r for r in rows if r["dataset"] == dataset]
        original = {r["seed"]:r for r in group if r["mode"] == "original"}
        forecast_only = {r["seed"]:r for r in group if r["mode"] == "forecast_only"}
        for row in group:
            for k in METRICS:
                row["delta_" + k] = row[k] - original[row["seed"]][k]
                row["delta_vs_forecast_only_" + k] = row[k] - forecast_only[row["seed"]][k]
        for mode in MODES:
            selected = [r for r in group if r["mode"] == mode]
            for key in METRICS:
                values = [r[key] for r in selected]
                deltas = [r["delta_" + key] for r in selected]
                versus_forecast = [r["delta_vs_forecast_only_" + key] for r in selected]
                aggregate.append(dict(dataset=dataset, mode=mode, metric=key, seeds=len(selected),
                                      mean=statistics.mean(values),
                                      sd=statistics.stdev(values) if len(values)>1 else "",
                                      mean_delta_vs_original=statistics.mean(deltas),
                                      paired_wins_vs_original=(sum(x < 0 for x in deltas) if mode != "original" else ""),
                                      mean_delta_vs_forecast_only=statistics.mean(versus_forecast),
                                      paired_wins_vs_forecast_only=(sum(x < 0 for x in versus_forecast)
                                                                     if mode != "forecast_only" else "")))

    write(output / "comparison.csv", rows)
    write(output / "seed_summary.csv", aggregate)
    print(f"Wrote {len(rows)} verified runs to {output / 'comparison.csv'}")
    print(f"Wrote paired summaries to {output / 'seed_summary.csv'}")
    for dataset in datasets:
        print(f"{dataset} · low-speed MAE on the shared selected roads (lower is better)")
        filtered = [r for r in aggregate if r["dataset"] == dataset and r["metric"] == "target_low_mae60"]
        for r in sorted(filtered, key=lambda row:row["mean"]):
            print(f" {r['mode']:16} {r['mean']:.6f}  delta {r['mean_delta_vs_original']:+.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--dataset", choices=DATASETS, nargs="+", default=DATASETS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    collect(args.project_dir, tuple(dict.fromkeys(args.dataset)), tuple(dict.fromkeys(args.seeds)), args.profile)
