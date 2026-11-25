#!/usr/bin/env python3
"""Express Links · Step 1 · Graph WaveNet baselines (standalone version of the notebook).

Same data checks, splits, model and training as Express_Links_Step1_Graph_WaveNet.ipynb.
Differences from the notebook, all about running it outside Colab/Jupyter:
  * settings come from command-line options instead of the first code cell
  * files go to --root (default /shared/$USER) instead of /content
  * no Google Drive / Colab download cell
  * --download-only fetches data and code (run it once on the login node, which has internet)
  * the combined summary files are rebuilt from every finished run folder,
    so several jobs can run at the same time without overwriting each other's rows

Examples:
  python -u express_links_step1.py --download-only
  python -u express_links_step1.py --profile smoke
  python -u express_links_step1.py                  # full run (default)
  python -u express_links_step1.py --profile full --datasets METR-LA --modes paper_full
  python -u express_links_step1.py --profile full --collect      # rebuild summary CSVs only
"""
from pathlib import Path
import gc, hashlib, importlib, json, os, random, shutil, subprocess, sys, tarfile, time, urllib.request, zipfile
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch


import argparse

parser = argparse.ArgumentParser(description="Express Links Step 1: Graph WaveNet baselines")
parser.add_argument("--profile", choices=["smoke", "audit", "full"], default="full",
                    help="smoke = quick pipeline check, audit = data only, full = 100 epochs")
parser.add_argument("--datasets", nargs="+", choices=["METR-LA", "PEMS-BAY"], default=["METR-LA", "PEMS-BAY"])
parser.add_argument("--modes", nargs="+", choices=["paper_full", "paper_fixed", "express_fixed"],
                    default=["paper_full", "paper_fixed", "express_fixed"])
parser.add_argument("--seeds", nargs="+", type=int, default=[0])
parser.add_argument("--root", type=Path, default=Path(os.environ.get("EXPRESS_ROOT", f"/shared/{os.environ.get('USER', 'user')}")),
                    help="folder that will hold express_links_cache/ and express_links_output/")
parser.add_argument("--download-source", choices=["auto", "official", "tsl"], default="auto")
parser.add_argument("--manual-data-dir", default=None, help="folder with metr-la.h5 / pems-bay.h5 or the author archive")
parser.add_argument("--allow-cpu-full", action="store_true")
parser.add_argument("--download-only", action="store_true", help="fetch data, graphs and author code, then stop")
parser.add_argument("--collect", action="store_true", help="only rebuild the combined summary CSVs")
args = parser.parse_args()

PROFILE = args.profile
DATASETS = list(dict.fromkeys(args.datasets))
MODES = list(dict.fromkeys(args.modes))
SEEDS = list(dict.fromkeys(args.seeds))
DOWNLOAD_SOURCE = args.download_source
MANUAL_DATA_DIR = args.manual_data_dir
ALLOW_CPU_FULL = args.allow_cpu_full

assert PROFILE in {"smoke", "audit", "full"}
CACHE = args.root / "express_links_cache"
OUTPUT = args.root / "express_links_output"
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)


def atomic_csv(frame, dest):
    tmp = dest.with_name(dest.name + f".{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(dest)


def collect_summaries():
    """Combine every finished run under OUTPUT/PROFILE into the two summary files."""
    base = OUTPUT / PROFILE
    scores, comparisons = [], []
    for run_dir in sorted(base.glob("*/*/seed_*")):
        metrics, config = run_dir / "metrics_15_30_60.csv", run_dir / "config.json"
        if metrics.exists() and config.exists():
            cfg = json.loads(config.read_text())
            frame = pd.read_csv(metrics)
            frame.insert(0, "profile", cfg["profile"])
            frame.insert(0, "seed", cfg["seed"])
            frame.insert(0, "mode", cfg["mode"])
            frame.insert(0, "dataset", cfg["dataset"])
            scores.append(frame)
        if (run_dir / "paper_comparison.csv").exists():
            comparisons.append(pd.read_csv(run_dir / "paper_comparison.csv"))
    if scores:
        atomic_csv(pd.concat(scores, ignore_index=True), base / "summary_15_30_60.csv")
    if comparisons:
        atomic_csv(pd.concat(comparisons, ignore_index=True), base / "comparison_to_paper.csv")
    print(f"Collected {len(scores)} finished run(s) into {base}", flush=True)


if args.collect:
    collect_summaries()
    sys.exit(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if PROFILE == "full" and DEVICE.type != "cuda" and not ALLOW_CPU_FULL:
    raise RuntimeError("Full profile requires a GPU by default. In Colab, change the runtime hardware accelerator.")
if DEVICE.type == "cpu":
    torch.set_num_threads(min(4, torch.get_num_threads()))
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
print({"python": sys.version.split()[0], "torch": torch.__version__, "device": str(DEVICE),
       "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None})

# ---- 1 · Fetch the author releases
import gdown

GWN_REF = "6b162e80c59a1d494809252eca055cff93dc66b1"
DCRNN_REF = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
GWN_ROOT = f"https://raw.githubusercontent.com/nnzhan/Graph-WaveNet/{GWN_REF}"
GRAPH_ROOT = f"https://raw.githubusercontent.com/liyaguang/DCRNN/{DCRNN_REF}/data/sensor_graph"
DATA_ID = "10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
TSL_ROOT = "https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/traffic"
assert DOWNLOAD_SOURCE in {"auto", "official", "tsl"}
SPECS = {
    "METR-LA": ("metr-la.h5", "adj_mx.pkl", 34272, 207, 1515),
    "PEMS-BAY": ("pems-bay.h5", "adj_mx_bay.pkl", 52116, 325, 2369),
}
assert DATASETS and set(DATASETS) <= SPECS.keys()
assert MODES and set(MODES) <= {"paper_full", "paper_fixed", "express_fixed"}

def fetch(url, dest):
    if not dest.is_file():
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as output:
            shutil.copyfileobj(response, output)
        tmp.replace(dest)
    return dest

def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def extract_requested(archive, destinations, provenance):
    installed = []
    def copy_one(name, open_member):
        base = Path(name).name.lower().replace("_", "-")
        if base in destinations and not destinations[base].exists():
            dest = destinations[base]
            tmp = dest.with_suffix(".part")
            with open_member() as source, tmp.open("wb") as output:
                shutil.copyfileobj(source, output)
            tmp.replace(dest)
            dest.with_suffix(".source").write_text(provenance)
            installed.append(dest)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as container:
            for member in container.infolist():
                if not member.is_dir():
                    copy_one(member.filename, lambda: container.open(member))
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as container:
            for member in container:
                if member.isfile():
                    copy_one(member.name, lambda: container.extractfile(member))
    return installed

data_dir = CACHE / "data"
repo = CACHE / ("gwn-" + GWN_REF[:8])
for directory in (data_dir, repo):
    directory.mkdir(exist_ok=True)

paths = {SPECS[key][0]: data_dir / SPECS[key][0] for key in DATASETS}
if MANUAL_DATA_DIR is not None:
    manual = Path(MANUAL_DATA_DIR)
    if not manual.is_dir():
        raise FileNotFoundError(manual)
    for name, dest in paths.items():
        files = [manual / name, manual / name.replace("-", "_")]
        present = next((p for p in files if p.is_file()), None)
        if present is not None:
            if present.resolve() != dest.resolve():
                shutil.copyfile(present, dest)
            dest.with_suffix(".source").write_text("manual H5 (user supplied)")
    for archive in manual.iterdir():
        if archive.is_file() and archive.name.lower().endswith((".zip", ".tar", ".tgz", ".tar.gz")):
            extract_requested(archive, paths, "manual archive (user supplied)")
else:
    for dest in paths.values():
        marker = dest.with_suffix(".source")
        provenance = marker.read_text() if marker.exists() else ""
        if dest.is_file() and ((DOWNLOAD_SOURCE == "official" and provenance != "DCRNN author Drive") or
                               (DOWNLOAD_SOURCE == "tsl" and not provenance.startswith("TSL "))) :
            dest.unlink()
            marker.unlink(missing_ok=True)
if any(not p.is_file() for p in paths.values()) and DOWNLOAD_SOURCE in {"auto", "official"}:
    archive = CACHE / "official_dcrnn_archive"
    try:
        if not archive.exists():
            tmp = archive.with_suffix(".part")
            tmp.unlink(missing_ok=True)
            subprocess.run([sys.executable, "-m", "gdown", f"https://drive.google.com/uc?id={DATA_ID}",
                            "-O", str(tmp)], check=True, timeout=150)
            if not (zipfile.is_zipfile(tmp) or tarfile.is_tarfile(tmp)):
                raise RuntimeError("The author download was not an archive.")
            tmp.replace(archive)
        extract_requested(archive, paths, "DCRNN author Drive")
        if any(not p.exists() for p in paths.values()):
            raise RuntimeError("The author archive did not contain the expected H5 files.")
    except Exception as exc:
        if DOWNLOAD_SOURCE == "official":
            raise RuntimeError("Download the official archive linked above and set MANUAL_DATA_DIR.") from exc
        print(f"Author Drive path unavailable ({type(exc).__name__}). Trying the research mirror.")
if any(not p.is_file() for p in paths.values()) and DOWNLOAD_SOURCE in {"auto", "tsl"}:
    for name, path in paths.items():
        if not path.is_file():
            archive_name = name.removesuffix(".h5").replace("-", "_") + ".zip"
            mirror_url = f"{TSL_ROOT}/{archive_name}"
            archive = fetch(mirror_url, CACHE / archive_name)
            extract_requested(archive, {name: path}, f"TSL v1.0.0: {mirror_url}")

for name, path in paths.items():
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}. Use MANUAL_DATA_DIR with the author release.")
    with path.open("rb") as handle:
        assert handle.read(8) == b"\x89HDF\r\n\x1a\n", f"Invalid HDF5 header: {path}"

for source_file in ("model.py", "engine.py", "util.py", "generate_training_data.py", "LICENSE"):
    remote = fetch(f"{GWN_ROOT}/{source_file}", repo / ("original_" + source_file))
    source = remote.read_text()
    if source_file == "model.py":
        assert source.count("nn.Conv1d(") == 3, "Unexpected author model revision"
        source = source.replace("nn.Conv1d(", "nn.Conv2d(")
    target = repo / source_file
    if not target.exists() or target.read_text() != source:  # avoid rewriting while another job imports it
        target.write_text(source)
for dataset in DATASETS:
    filename = SPECS[dataset][1]
    fetch(f"{GRAPH_ROOT}/{filename}", data_dir / filename)

sys.path.insert(0, str(repo))
for name in ("engine", "model", "util", "generate_training_data"):
    if name in sys.modules:
        del sys.modules[name]
import engine as gwn_engine
import util as gwn_util
from generate_training_data import generate_graph_seq2seq_io_data
print("Author files ready.")
if args.download_only:
    print("Download finished. Cache:", CACHE)
    sys.exit(0)

# ---- 2 · Audit, clean and split each dataset
X_OFF = np.arange(-11, 1)
Y_OFF = np.arange(1, 13)
HIGHLIGHTS = [15, 30, 60]
BATCH = 64
STEP = pd.Timedelta(minutes=5)

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))

def validate_data(dataset, speeds_path, adjacency_path, enforce_reference=True):
    _, _, n_rows, n_nodes, n_edges = SPECS[dataset]
    frame = pd.read_hdf(speeds_path)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert not frame.index.hasnans and frame.index.is_unique and frame.index.is_monotonic_increasing
    rows_before = len(frame)
    dropped = []
    if dataset == "PEMS-BAY":
        clock_rows = pd.date_range("2017-03-12 02:00", periods=12, freq="5min")
        if frame.index.tz is None and clock_rows.isin(frame.index).all() and frame.loc[clock_rows].isna().all().all():
            dropped = [str(t) for t in clock_rows]
            frame = frame.drop(clock_rows)
    gaps = frame.index[1:] - frame.index[:-1]
    unusual = np.flatnonzero(gaps != STEP) + 1
    if len(unusual):
        allowed = (dataset == "PEMS-BAY" and len(unusual) == 1 and
                   frame.index[unusual[0]-1] == pd.Timestamp("2017-03-12 01:55") and
                   frame.index[unusual[0]] == pd.Timestamp("2017-03-12 03:00"))
        assert allowed, f"{dataset}: unexpected timestamp gap(s): {frame.index[unusual].tolist()[:5]}"
    sensors = frame.columns.map(str)
    assert sensors.is_unique and all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes)
    ids, index_map, adj = gwn_util.load_pickle(adjacency_path)
    ids = list(map(str, ids))
    assert ids == list(sensors), f"{dataset}: HDF columns and graph order differ. Verify the official release."
    assert {str(k): int(v) for k, v in index_map.items()} == dict(zip(ids, range(len(ids))))
    adj = np.asarray(adj)
    raw = frame.to_numpy(dtype=np.float64)
    assert adj.shape == (len(ids), len(ids))
    assert np.isfinite(adj).all() and (adj >= 0).all()
    off = adj.copy()
    np.fill_diagonal(off, 0)
    assert np.isfinite(raw).all() and (raw >= 0).all(), "Unexpected nonfinite or negative official speed. Inspect the source file."
    if enforce_reference:
        assert raw.shape == (n_rows, n_nodes), f"{dataset}: unexpected shape {raw.shape}"
        assert np.count_nonzero(off) == n_edges, f"{dataset}: edge count differs from Table 1"
    missing = raw == 0
    on_day = frame.index.values
    tod = (on_day - on_day.astype("datetime64[D]")) / np.timedelta64(1, "D")
    audit = {
        "dataset": dataset, "timesteps": len(raw), "sensors": len(ids), "start": str(frame.index[0]),
        "end": str(frame.index[-1]), "timestep_minutes": 5, "off_diagonal_edges": int(np.count_nonzero(off)),
        "self_loops": int(np.count_nonzero(np.diag(adj))),
        "weak_components": int(connected_components(csr_matrix(off), directed=True, connection="weak")[0]),
        "zero_incoming_off_diagonal": int((~(off != 0).any(axis=0)).sum()),
        "zero_outgoing_off_diagonal": int((~(off != 0).any(axis=1)).sum()),
        "asymmetric_weight_entries": int(np.count_nonzero(adj != adj.T)),
        "raw_missing_fraction": float(missing.mean()),
        "entirely_missing_timestamps": int(missing.all(axis=1).sum()),
        "exact_duplicate_timestamp_count": 0, "nonfinite_count": 0, "negative_count": 0,
        "source_input_rows": rows_before, "dropped_clock_change_rows": dropped,
        "clock_gap_positions": unusual.tolist(),
        "clock_gap_after": [str(frame.index[i]) for i in unusual],
        "speed_source": speeds_path.with_suffix(".source").read_text() if speeds_path.with_suffix(".source").exists() else "prepopulated cache (provenance not recorded)",
        "positive_speed_quantiles": dict(zip(["min", "p01", "median", "p99", "max"],
             np.quantile(raw[~missing], [0, .01, .5, .99, 1]).tolist())),
        "speeds_sha256": sha256(speeds_path), "adjacency_sha256": sha256(adjacency_path),
    }
    return raw, frame.index, tod, ids, adj, audit

def make_splits(n_rows, protocol):
    origins = np.arange(11, n_rows - 12)
    if protocol == "author":
        n_train, n_test = round(len(origins) * .7), round(len(origins) * .2)
        split = dict(zip(("train", "val", "test"), np.split(origins, [n_train, len(origins) - n_test])))
        boundaries = {"split_surface": "window origins", "n_candidate_origins": len(origins)}
    elif protocol == "strict":
        t1, t2 = round(n_rows * .7), n_rows - round(n_rows * .2)
        bounds = [(0, t1), (t1, t2), (t2, n_rows)]
        split = {k: origins[(origins + 1 >= lo) & (origins + 12 < hi)]
                 for k, (lo, hi) in zip(("train", "val", "test"), bounds)}
        boundaries = {"split_surface": "raw target timestamps", "raw_train_end_exclusive": t1,
                      "raw_val_end_exclusive": t2, "n_candidate_origins": len(origins)}
    else:
        raise ValueError(protocol)
    selectors = list(split.values())
    assert all(len(s) and np.all(np.diff(s) == 1) for s in selectors)
    assert all(a[-1] < b[0] for a, b in zip(selectors, selectors[1:]))
    overlap = [max(0, int(a[-1] + 12 - b[0])) for a, b in zip(selectors, selectors[1:])]
    assert overlap == ([11, 11] if protocol == "author" else [0, 0])
    boundaries.update({"shared_target_timestamps_train_val": overlap[0],
                       "shared_target_timestamps_val_test": overlap[1],
                       "discarded_boundary_origins": len(origins) - sum(map(len, selectors))})
    return split, boundaries

def prepare_dataset(dataset, source, protocol, audit_dir):
    raw, index, tod, ids, adj, source_audit = source
    split, summary = make_splits(len(raw), protocol)
    excluded = 0
    if protocol == "strict":
        for key, origins in split.items():
            keep = np.ones(len(origins), dtype=bool)
            for gap in source_audit["clock_gap_positions"]:
                keep &= ~((origins - 11 < gap) & (origins + 12 >= gap))
            excluded += int((~keep).sum())
            split[key] = origins[keep]
            assert len(split[key])
    summary["discarded_clock_gap_origins"] = excluded
    # Fit imputers and the regime audit on unique timestamps available for training.
    fit_end = summary.get("raw_train_end_exclusive", int(split["train"][-1] + 12 + 1))
    valid_raw = np.isfinite(raw) & (raw > 0)
    fit_values = raw[:fit_end][valid_raw[:fit_end]]
    assert fit_values.size, "Training partition has no observed speeds."
    missing = ~valid_raw
    inputs = raw.copy()
    if protocol == "strict":
        masked = pd.DataFrame(raw).where(valid_raw)
        medians = masked.iloc[:fit_end].median().fillna(np.median(fit_values))
        inputs = masked.ffill(limit=3).fillna(medians).to_numpy()
        assert np.array_equal(inputs[valid_raw], raw[valid_raw])
    assert np.isfinite(inputs).all()
    # Count each row with its multiplicity across all train input windows, as upstream.
    counts = np.bincount((split["train"][:, None] + X_OFF).ravel(), minlength=len(raw))
    weights = counts / counts.sum()
    mean = float(weights @ inputs.mean(axis=1))
    std = float(np.sqrt(weights @ np.square(inputs - mean).mean(axis=1)))
    assert np.isfinite([mean, std]).all() and std > 0
    scaled = ((inputs - mean) / std).astype(np.float32)
    labels = np.where(valid_raw, raw, 0).astype(np.float32)
    assert np.isfinite(scaled).all() and np.isfinite(labels).all()
    bins = np.unique(np.quantile(fit_values, [.25, .5, .75]))
    edges = np.concatenate(([-np.inf], bins, [np.inf]))
    split_rows, sensor_rows, coverage_rows, regime_rows = [], [], [], []
    for key, origins in split.items():
        target = np.unique((origins[:, None] + Y_OFF).ravel())
        y = raw[target]
        m = missing[target]
        times = index[target]
        assert (~m).any()
        lag_fractions = [(~missing[origins + lag]).mean().item() for lag in Y_OFF]
        split_rows.append({
            "split": key, "windows": len(origins), "first_origin": str(index[origins[0]]),
            "last_origin": str(index[origins[-1]]), "first_target": str(times[0]), "last_target": str(times[-1]),
            "unique_target_timestamps": len(times), "missing_target_fraction": float(m.mean()),
            "missing_input_fraction": float(np.bincount((origins[:, None] + X_OFF).ravel(), minlength=len(raw)) @ missing.mean(axis=1) / (len(origins) * 12)),
            **{f"observed_fraction_h{5*h}": v for h, v in enumerate(lag_fractions, 1)},
        })
        sensor_rows.extend({"split": key, "sensor_id": sensor, "missing_target_fraction": frac}
                           for sensor, frac in zip(ids, m.mean(axis=0)))
        for field, values in (("hour", times.hour), ("weekday", times.dayofweek)):
            for value in np.unique(values):
                selected = m[values == value]
                coverage_rows.append({"split": key, "group": field, "value": int(value),
                                      "valid_count": int((~selected).sum()), "missing_fraction": float(selected.mean())})
        band_counts = np.histogram(y[~m], bins=edges)[0]
        regime_rows.extend({"split": key, "band": i + 1, "n_valid": int(count),
                            "share_of_valid": float(count / band_counts.sum())}
                           for i, count in enumerate(band_counts))
    summary.update({"protocol": protocol, "dataset": dataset, "input_scaler_fit": "train input windows only",
                    "mean": mean, "std": std, "imputed_unique_input_slots": int(missing.sum()) if protocol == "strict" else 0,
                    "speed_band_cutpoints_training": bins.tolist(), "sensor_ids": ids})
    audit_dir.mkdir(parents=True, exist_ok=True)
    save_json(audit_dir / "source_audit.json", source_audit)
    save_json(audit_dir / "preprocessing.json", summary)
    pd.DataFrame(split_rows).to_csv(audit_dir / "splits.csv", index=False)
    pd.DataFrame(sensor_rows).to_csv(audit_dir / "missing_by_sensor.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(audit_dir / "time_coverage.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(audit_dir / "speed_regimes.csv", index=False)
    np.savez_compressed(audit_dir / "validated_graph.npz", weights=adj, sensor_ids=np.asarray(ids, dtype=str))
    print(dataset, protocol, "missing zeros", round(source_audit["raw_missing_fraction"] * 100, 3), "%")
    print(pd.DataFrame(split_rows)[["split", "windows", "first_target", "last_target", "missing_target_fraction"]].to_string(index=False))
    return dict(dataset=dataset, index=index, tod=tod.astype(np.float32), ids=ids, split=split, adj=adj,
                scaler=gwn_util.StandardScaler(mean, std), scaled=scaled, labels=labels, bins=bins, summary=summary)

def make_batch(data, origins):
    rows = origins[:, None] + X_OFF
    speed = data["scaled"][rows]
    tod = np.broadcast_to(data["tod"][rows, None], speed.shape)
    x = np.stack((speed, tod), axis=1)  # [batch, 2, input-time, node]
    x = torch.from_numpy(x).transpose(2, 3).contiguous().to(DEVICE)
    y = data["labels"][origins[:, None] + Y_OFF]  # [batch, horizon, node]
    y = torch.from_numpy(y).transpose(1, 2).contiguous().to(DEVICE)
    return x, y

def origin_batches(origins, shuffle_seed=None):
    # Match the author's right padding with copies of the last origin.
    ids = np.pad(origins, (0, (-len(origins)) % BATCH), mode="edge")
    if shuffle_seed is not None:
        ids = np.random.RandomState(shuffle_seed).permutation(ids)
    for start in range(0, len(ids), BATCH):
        yield ids[start:start + BATCH]

def validate_window_code():
    # Compare the notebook's indexing, time feature, overlap and scaler with official code.
    fixture = pd.DataFrame(np.arange(720, dtype=float).reshape(240, 3),
                           index=pd.date_range("2024-01-01", periods=240, freq="5min"))
    x, y = generate_graph_seq2seq_io_data(fixture, X_OFF, Y_OFF)
    split, _ = make_splits(len(fixture), "author")
    make_splits(len(fixture), "strict")
    train = split["train"]
    raw = fixture.to_numpy()
    counts = np.bincount((train[:, None] + X_OFF).ravel(), minlength=len(raw))
    avg = (counts / counts.sum()) @ raw.mean(axis=1)
    sd = np.sqrt((counts / counts.sum()) @ ((raw - avg) ** 2).mean(axis=1))
    np.testing.assert_allclose([avg, sd], [x[:len(train), ..., 0].mean(), x[:len(train), ..., 0].std()])
    fixture_data = dict(scaled=((raw - avg) / sd).astype(np.float32),
                        tod=np.arange(len(raw), dtype=np.float32) / 288, labels=raw.astype(np.float32))
    bx, by = make_batch(fixture_data, train[:5])
    ref_x = x[:5].copy()
    ref_x[..., 0] = (ref_x[..., 0] - avg) / sd
    np.testing.assert_allclose(bx.cpu().numpy().transpose(0, 3, 2, 1), ref_x, atol=1e-7)
    np.testing.assert_array_equal(by.cpu().numpy().transpose(0, 2, 1), y[:5, ..., 0])

validate_window_code()

# ---- 3 · Train, select on validation, and export test forecasts
PAPER_T2 = {
    "METR-LA": {15: (2.69, 5.15, 6.90), 30: (3.07, 6.22, 8.37), 60: (3.53, 7.37, 10.01)},
    "PEMS-BAY": {15: (1.30, 2.74, 2.73), 30: (1.63, 3.70, 3.67), 60: (1.95, 4.52, 4.63)},
}
PAPER_T3 = {
    ("METR-LA", "paper_full"): (3.04, 6.09, 8.23),
    ("METR-LA", "paper_fixed"): (3.08, 6.13, 8.25),
    ("PEMS-BAY", "paper_full"): (1.58, 3.52, 3.55),
    ("PEMS-BAY", "paper_fixed"): (1.59, 3.55, 3.57),
}
METRICS = ("MAE", "RMSE", "MAPE_pct")

def atomic_torch_save(value, dest):
    tmp = dest.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(dest)

def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

def score_predictions(pred, truth, mask):
    if not mask.any():
        return dict.fromkeys(METRICS, np.nan)
    y = truth[mask].astype(np.float64)
    err = np.abs(pred[mask].astype(np.float64) - y)
    return dict(zip(METRICS, [err.mean(), np.sqrt(np.square(err).mean()), 100 * (err / np.abs(y)).mean()]))

def run_experiment(data, source_audit, mode, seed):
    dataset = data["dataset"]
    run_dir = OUTPUT / PROFILE / dataset / mode / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs = 1 if PROFILE == "smoke" else 100
    adaptive = mode == "paper_full"
    config = dict(dataset=dataset, profile=PROFILE, mode=mode, seed=int(seed), epochs=epochs, batch_size=BATCH,
                  adaptive=adaptive, embedding_init="normal (author code)" if adaptive else None,
                  input_steps=12, output_steps=12, dropout=.3, learning_rate=.001, weight_decay=.0001,
                  hidden_channels=32, diffusion_order=2, gradient_clip=5, adjacency="doubletransition",
                  code_commit=GWN_REF, graph_commit=DCRNN_REF, mean=data["scaler"].mean, std=data["scaler"].std,
                  notebook_protocol_version=1, device=str(DEVICE), torch=str(torch.__version__),
                  numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                  speeds_sha256=source_audit["speeds_sha256"], adjacency_sha256=source_audit["adjacency_sha256"])
    manifest = run_dir / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise RuntimeError(f"Run config changed. Choose a new output directory before resuming: {run_dir}")
    save_json(manifest, config)
    seed_everywhere(seed)
    supports = [torch.as_tensor(np.asarray(gwn_util.asym_adj(a)), dtype=torch.float32, device=DEVICE)
                for a in (data["adj"], data["adj"].T)]
    for support in supports:
        row_sum = support.sum(dim=1)
        assert torch.isfinite(support).all() and torch.all((row_sum == 0) | torch.isclose(row_sum, torch.ones_like(row_sum)))
    fit = gwn_engine.trainer(data["scaler"], 2, 12, len(data["ids"]), 32,
                             .3, .001, .0001, DEVICE, supports, True, adaptive, None)
    assert fit.model.receptive_field == 13
    assert hasattr(fit.model, "nodevec1") == adaptive
    assert len(fit.model.gconv) == 8
    last, best_file = run_dir / "last_resume.pt", run_dir / "best_weights.pt"
    history, best, best_epoch = [], float("inf"), 0
    if last.exists():
        state = torch.load(last, map_location=DEVICE, weights_only=True)
        fit.model.load_state_dict(state["model"])
        fit.optimizer.load_state_dict(state["optimizer"])
        history, best, best_epoch = state["history"], state["best"], state["best_epoch"]
        # Last checkpoint stores the best weights too, so resume can regenerate the sidecar atomically.
        best_weights = {k: v.detach().cpu() for k, v in state["best_weights"].items()}
        del state
        print(f"Resuming {dataset}/{mode} from epoch {len(history)}")
    else:
        best_weights = None

    for epoch in range(len(history) + 1, epochs + 1):
        epoch_seed = (seed + 1009 * epoch) % (2**32)
        seed_everywhere(epoch_seed)
        started = time.monotonic()
        train_losses, val_losses = [], []
        for step, ids in enumerate(origin_batches(data["split"]["train"], shuffle_seed=epoch_seed)):
            x, y = make_batch(data, ids)
            if (y != 0).any():
                train_losses.append(fit.train(x, y)[0])
            if PROFILE == "smoke" and step == 1:
                break
        fit.model.eval()
        with torch.no_grad():
            for step, ids in enumerate(origin_batches(data["split"]["val"])):
                x, y = make_batch(data, ids)
                if (y != 0).any():
                    val_losses.append(fit.eval(x, y)[0])
                if PROFILE == "smoke":
                    break
        assert train_losses and val_losses and np.isfinite(train_losses + val_losses).all()
        loss = float(np.mean(val_losses))
        history.append(dict(epoch=epoch, train_MAE=float(np.mean(train_losses)), val_MAE=loss,
                            seconds=time.monotonic() - started))
        if loss < best:
            best, best_epoch = loss, epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in fit.model.state_dict().items()}
        atomic_torch_save(dict(model=fit.model.state_dict(), optimizer=fit.optimizer.state_dict(),
                               best_weights=best_weights, history=history, best=best, best_epoch=best_epoch), last)
        print(f"{dataset} {mode} seed={seed} epoch {epoch}/{epochs} train={history[-1]['train_MAE']:.4f} val={loss:.4f} sec={history[-1]['seconds']:.1f}", flush=True)

    atomic_torch_save(best_weights, best_file)
    pd.DataFrame(history).to_csv(run_dir / "learning_curve.csv", index=False)
    fit.model.load_state_dict(best_weights)
    fit.model.eval()
    origins = data["split"]["test"][:128] if PROFILE == "smoke" else data["split"]["test"]
    forecast = []
    with torch.inference_mode():
        for ids in origin_batches(origins):
            inputs, _ = make_batch(data, ids)
            preds = data["scaler"].inverse_transform(fit.model(inputs).squeeze(-1))
            assert preds.shape == (len(ids), 12, len(data["ids"])), preds.shape
            forecast.append(preds.cpu().numpy())
    pred = np.concatenate(forecast, axis=0)[:len(origins)]
    target_rows = origins[:, None] + Y_OFF
    truth = data["labels"][target_rows]
    mask = truth != 0
    assert np.isfinite(pred).all() and pred.shape == truth.shape
    dates = data["index"].to_numpy(dtype="datetime64[ns]")
    if mode == "express_fixed":
        assert np.all(dates[target_rows] - dates[origins, None] == (Y_OFF * 5).astype("timedelta64[m]"))
    np.savez_compressed(run_dir / "predictions_12_horizons.npz", predictions=pred, targets=truth,
                        target_observed=mask, issue_timestamp=dates[origins],
                        target_timestamp=dates[target_rows], sensor_id=np.array(data["ids"], dtype=str),
                        horizon_minutes=Y_OFF * 5)
    rows, regime_rows = [], []
    for i, minutes in enumerate(Y_OFF * 5):
        m = mask[:, i]
        p, y = pred[:, i], truth[:, i]
        rows.append(dict(horizon_minutes=minutes, valid_target_count=int(m.sum()), **score_predictions(p, y, m)))
        for band in range(len(data["bins"]) + 1):
            select = m & (np.searchsorted(data["bins"], y, side="right") == band)
            regime_rows.append(dict(horizon_minutes=minutes, band=band+1, valid_target_count=int(select.sum()),
                                    **score_predictions(p, y, select)))
    scores = pd.DataFrame(rows)
    assert scores[list(METRICS)].notna().all().all()
    scores.to_csv(run_dir / "metrics_all_horizons.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(run_dir / "metrics_speed_bands.csv", index=False)
    mean = scores[list(METRICS)].mean().to_dict()
    pd.DataFrame([mean]).to_csv(run_dir / "metrics_mean_12.csv", index=False)
    selected = scores[scores.horizon_minutes.isin(HIGHLIGHTS)].copy()
    selected.to_csv(run_dir / "metrics_15_30_60.csv", index=False)
    comparison = []
    if PROFILE == "full" and (dataset, mode) in PAPER_T3:
        refs = [("Table 3", "mean_12", mean, PAPER_T3[dataset, mode])]
        if mode == "paper_full":
            refs.extend(("Table 2", str(int(row.horizon_minutes)), row._asdict(), PAPER_T2[dataset][int(row.horizon_minutes)])
                        for row in selected.itertuples(index=False))
        for table, horizon, actual, published in refs:
            for metric, value in zip(METRICS, published):
                comparison.append(dict(dataset=dataset, mode=mode, seed=seed, source=table, horizon=horizon,
                                       metric=metric, reproduced=actual[metric], published=value,
                                       delta=actual[metric] - value))
        pd.DataFrame(comparison).to_csv(run_dir / "paper_comparison.csv", index=False)
    save_json(run_dir / "result.json", {"best_epoch": best_epoch, "best_validation_MAE": best,
                                       "test_origins": len(origins), "smoke": PROFILE == "smoke"})
    print(selected.to_string(index=False))
    return [{"dataset": dataset, "mode": mode, "seed": seed, "profile": PROFILE, **r} for r in selected.to_dict("records")], comparison

# ---- Run
all_scores, all_comparisons = [], []
for dataset in DATASETS:
    h5, graph, *_ = SPECS[dataset]
    source = validate_data(dataset, data_dir / h5, data_dir / graph)
    for protocol in ("author", "strict"):
        selected_modes = [m for m in MODES if (m == "express_fixed") == (protocol == "strict")]
        if not selected_modes and PROFILE != "audit":
            continue
        prepared = prepare_dataset(dataset, source, protocol, OUTPUT / "audit" / dataset / protocol)
        if PROFILE != "audit":
            for mode in selected_modes:
                for seed in SEEDS:
                    scores, comparisons = run_experiment(prepared, source[-1], mode, seed)
                    all_scores.extend(scores)
                    all_comparisons.extend(comparisons)
        del prepared
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    del source

if PROFILE != "audit":
    collect_summaries()
print("Outputs:", OUTPUT)
