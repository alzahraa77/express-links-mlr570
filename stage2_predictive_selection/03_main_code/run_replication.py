#!/usr/bin/env python3

from pathlib import Path
import importlib
import importlib.util
import subprocess
import sys

# Compatible with managed kernels that disable --user installs.
package_dir = Path.home() / "express_links_packages"
if str(package_dir) not in sys.path:
    sys.path.append(str(package_dir))
required = {"gdown": "gdown==5.2.0", "tables": "tables"}
missing = [p for module, p in required.items() if importlib.util.find_spec(module) is None]
if missing:
    package_dir.mkdir(exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", str(package_dir), *missing])
    importlib.invalidate_caches()


from pathlib import Path
import gc, hashlib, importlib, json, os, random, shutil, subprocess, sys, tarfile, time, urllib.request, zipfile
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
try:
    import torch
except ImportError as exc:
    raise RuntimeError("Select the lab's PyTorch Python kernel before running this notebook.") from exc


from argparse import ArgumentParser
from uuid import uuid4

parser = ArgumentParser(description="Replicate the Express Links comparison on a GPU.")
parser.add_argument("--dataset", choices=["METR-LA", "PEMS-BAY"], required=True)
parser.add_argument("--profile", choices=["audit", "smoke", "full"], default="smoke")
parser.add_argument("--mode", choices=["all", "original", "express_links", "forecast_only", "gtr",
                                         "random_matched", "virtual_node"], default="all")
parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
parser.add_argument("--project-dir", type=Path, default=Path.cwd() / "express_links_replication")
parser.add_argument("--cache", type=Path)
parser.add_argument("--manual-data-dir", type=Path)
parser.add_argument("--download-source", choices=["auto", "official", "tsl"], default="auto")
args = parser.parse_args()
PROFILE, DATASETS = args.profile, [args.dataset]
MODES = (["original", "express_links", "forecast_only", "gtr", "random_matched", "virtual_node"]
         if args.mode == "all" else [args.mode])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if PROFILE == "full" and DEVICE.type != "cuda":
    raise SystemExit("Full runs require a CUDA GPU. Submit through Slurm on the lab GPU partition.")
if DEVICE.type == "cpu":
    torch.set_num_threads(min(4, torch.get_num_threads()))
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
PROJECT_DIR = args.project_dir.expanduser().resolve()
CACHE = (args.cache or PROJECT_DIR / "cache").expanduser().resolve()
OUTPUT = PROJECT_DIR / "pilot_output"
for path in (CACHE, OUTPUT):
    path.mkdir(parents=True, exist_ok=True)
MANUAL_DATA_DIR = args.manual_data_dir
DOWNLOAD_SOURCE = args.download_source
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "", flush=True)


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
assert MODES and set(MODES) <= {"original", "express_links", "forecast_only", "gtr", "random_matched", "virtual_node"}

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

if PROFILE == "full":
    graph_meta = OUTPUT / "graphs" / args.dataset / "selector_summary.json"
    required_cache = [data_dir / f for f in SPECS[args.dataset][:2]]
    required_cache += [repo / ("original_" + f) for f in
                       ("model.py", "engine.py", "util.py", "generate_training_data.py", "LICENSE")]
    if (not graph_meta.exists() or not (graph_meta.parent / "frozen_graphs.json").exists()
            or not all(p.is_file() for p in required_cache)):
        raise RuntimeError("Run this dataset with --profile audit first to freeze graphs and prepare cache.")
    # Training workers consume the audited cache without replacing any source data.
    MANUAL_DATA_DIR = None
    DOWNLOAD_SOURCE = "auto"

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
                            "-O", str(tmp)], check=True, timeout=150,
                           env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
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
    local = repo / source_file
    if not local.exists() or local.read_text() != source:
        staging = repo / (source_file + "." + uuid4().hex + ".tmp")
        staging.write_text(source)
        staging.replace(local)
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

import hashlib
import time
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

PILOT_MODES = ("original", "express_links", "forecast_only", "gtr", "random_matched", "virtual_node")
SHORTLIST, LINK_BUDGET, MIN_HOPS = 16, 25, 3
RIDGE_ALPHA = 1.0

def graph_fingerprint(adj):
    return hashlib.sha256(np.ascontiguousarray(adj, dtype="<f8").tobytes()).hexdigest()

def transition_direction_check(adj, links):
    # Graph WaveNet computes einsum('ncvl,vw->ncwl', x, support).
    # A impulse at source v reaches destination w through support[v,w].
    for source, target in links:
        impulse = np.eye(1, len(adj), source).ravel()
        assert (impulse @ adj)[target] > 0

def graph_candidates(adj):
    undirected = (adj.astype(float) + adj.T.astype(float)) / 2
    np.fill_diagonal(undirected, 0)
    hops = shortest_path(undirected, directed=False, unweighted=True)
    eligible = np.isfinite(hops) & (hops >= MIN_HOPS) & (adj == 0)
    laplacian = np.diag(undirected.sum(axis=1)) - undirected
    inverse = np.linalg.pinv(laplacian, hermitian=True, rcond=1e-10)
    d = np.diag(inverse)
    resistance = np.maximum(d[:, None] + d[None, :] - 2 * inverse, 0)
    return eligible, hops, inverse, resistance

def predictive_scores(source, prepared, eligible, resistance):
    raw, index, tod, ids, adj, _ = source
    end = prepared["summary"]["raw_train_end_exclusive"]
    cut = int(.8 * end)
    origins = prepared["split"]["train"]
    first = origins[origins + 12 < cut]
    later = origins[origins + 1 >= cut]
    assert len(first) and len(later) and later[-1] + 12 < end
    fit = pd.DataFrame(raw[:end]).where(raw[:end] > 0)
    observed = fit.iloc[:cut].to_numpy()
    fallback = np.nanmedian(observed)
    filled = fit.ffill(limit=3).fillna(fit.iloc[:cut].median().fillna(fallback)).to_numpy()
    neighborhood = adj.astype(float).copy()
    np.fill_diagonal(neighborhood, 0)
    neighborhood += neighborhood.T.copy()
    degree = neighborhood.sum(axis=1)
    normalized = neighborhood / np.maximum(degree[:, None], np.finfo(float).eps)
    neighbor_speed = filled @ normalized.T
    neighbor_speed[:, degree == 0] = filled[:, degree == 0]
    all_origins = np.concatenate([first, later])
    source_speed = (filled[all_origins] + filled[all_origins-1] + filled[all_origins-2]) / 3
    sincos = np.stack([np.sin(2*np.pi*tod), np.cos(2*np.pi*tod)], axis=1)
    rows = {"express_links": [], "forecast_only": []}
    for target in range(len(ids)):
        candidates = np.flatnonzero(eligible[:, target])
        if not len(candidates):
            continue
        labels = raw[all_origins + 12, target]
        train_mask = labels[:len(first)] > 0
        score_mask = labels[len(first):] > 0
        if train_mask.sum() < 100 or score_mask.sum() < 100:
            continue
        local = np.column_stack((filled[all_origins[:, None] + np.arange(-11, 1), target],
                                 neighbor_speed[all_origins, target], sincos[all_origins]))
        x = local[:len(first)][train_mask]
        xv = local[len(first):][score_mask]
        mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
        x = np.column_stack((np.ones(len(x)), (x-mean)/std))
        xv = np.column_stack((np.ones(len(xv)), (xv-mean)/std))
        y, yv = labels[:len(first)][train_mask], labels[len(first):][score_mask]
        penalty = np.diag(np.r_[0, np.full(x.shape[1]-1, RIDGE_ALPHA)])
        z = source_speed[:len(first)][train_mask][:, candidates]
        zv = source_speed[len(first):][score_mask][:, candidates]
        z_mean, z_std = z.mean(axis=0), np.maximum(z.std(axis=0), 1e-8)
        z, zv = (z-z_mean)/z_std, (zv-z_mean)/z_std
        gram = x.T @ x + penalty
        beta = np.linalg.solve(gram, x.T @ y)
        cross = x.T @ z
        source_beta = np.linalg.solve(gram, cross)
        num = (y - x @ beta) @ z
        denom = (z*z).sum(axis=0) + RIDGE_ALPHA - (cross*source_beta).sum(axis=0)
        gamma = num / np.maximum(denom, RIDGE_ALPHA*0.5)
        # Each column is the independent augmented ridge fit for one candidate.
        # Rank the no-resistance shortlist using fit-block objective gain only.
        fit_gain = num * gamma
        shortlists = {
            "express_links": np.argsort(-resistance[candidates, target], kind="stable")[:SHORTLIST],
            "forecast_only": np.argsort(-fit_gain, kind="stable")[:SHORTLIST],
        }
        baseline_error = np.abs(yv - xv @ beta)
        for mode, ix in shortlists.items():
            pred_gain = (zv[:, ix] - xv @ source_beta[:, ix]) * gamma[ix]
            gain = baseline_error.mean() - np.abs((yv - xv @ beta)[:, None] - pred_gain).mean(axis=0)
            for position, score in zip(ix, gain):
                rows[mode].append((int(candidates[position]), target, float(score)))
    result = {mode: sorted(values, key=lambda a: (-a[2], a[1], a[0])) for mode, values in rows.items()}
    timing = {"fit_last_target": str(index[first[-1]+12]), "ranking_first_target": str(index[later[0]+12]),
              "ranking_last_target": str(index[later[-1]+12]), "test_labels_used": False,
              "fit_origins": len(first), "ranking_origins": len(later)}
    return result, timing

def at_most_one_per_target(rows):
    selected, targets = [], set()
    for a, b, score in rows:
        if score > 0 and b not in targets:
            selected.append((a, b, score))
            targets.add(b)
    return selected

def gtr_control(eligible, inverse, count, weight, random_seed=0):
    # Weighted total-resistance drop, evaluated on the same undirected surrogate.
    # Choose a single directed arc per selected pair using a fixed random orientation.
    n = len(inverse)
    rng = np.random.default_rng(random_seed)
    allowed = np.triu(eligible | eligible.T, 1)
    out = []
    squared = inverse @ inverse
    for _ in range(count):
        d, q = np.diag(inverse), np.diag(squared)
        resistance = np.maximum(d[:, None] + d[None, :] - 2*inverse, 0)
        norm_sq = np.maximum(q[:, None] + q[None, :] - 2*squared, 0)
        # Each directed edge contributes weight/2 under the shared mean symmetrization.
        w = weight / 2
        gain = np.where(allowed, n*w*norm_sq/(1+w*resistance), -np.inf)
        a, b = np.unravel_index(np.argmax(gain), gain.shape)
        assert np.isfinite(gain[a, b]), "Too few eligible GTR links."
        a_to_b = bool(eligible[a, b] and (not eligible[b, a] or rng.integers(2) == 0))
        src, dst = (a,b) if a_to_b else (b,a)
        out.append((int(src), int(dst), float(gain[a,b])))
        column = inverse[:, a] - inverse[:, b]
        scale = w/(1+w*resistance[a,b])
        projected = inverse @ column
        outer = np.outer(column, column)
        squared = squared - scale*(np.outer(projected, column) + np.outer(column, projected)) + scale**2*(column @ column)*outer
        inverse = inverse - scale*outer
        allowed[a,b] = False
    return out

def build_pilot_graphs(source, prepared, directory):
    start = time.monotonic()
    adj, ids = prepared["adj"], prepared["ids"]
    eligible, hops, inverse, resistance = graph_candidates(adj)
    graph_seconds = time.monotonic() - start
    scored, inner = predictive_scores(source, prepared, eligible, resistance)
    score_seconds = time.monotonic() - start - graph_seconds
    ranked = {method: at_most_one_per_target(rows) for method, rows in scored.items()}
    count = min(LINK_BUDGET, *(len(rows) for rows in ranked.values()))
    if count == 0:
        raise RuntimeError("No positive shortlist passes the pilot's scoring rule.")
    lists = {method: rows[:count] for method, rows in ranked.items()}
    off = ~np.eye(len(adj), dtype=bool)
    weight = float(np.median(adj[(adj > 0) & off]))
    t = time.monotonic()
    lists["gtr"] = gtr_control(eligible, inverse, count, weight)
    gtr_seconds = time.monotonic()-t
    random = np.random.default_rng(0)
    lists["random_matched"] = [(int(random.choice(np.flatnonzero(eligible[:, b]))), b, None)
                               for _, b, _ in lists["express_links"]]
    n = len(ids)
    lists["original"] = []
    lists["virtual_node"] = ([(a, n, None) for a in range(n)] + [(n, b, None) for b in range(n)])
    directory.mkdir(parents=True, exist_ok=True)
    graphs = {}
    for method, links in lists.items():
        hub = method == "virtual_node"
        graph = np.pad(adj, ((0,1),(0,1))) if hub else adj.copy()
        named = list(ids) + (["__virtual_hub__"] if hub else [])
        rows = []
        for a, b, score in links:
            assert a != b and graph[a,b] == 0
            graph[a,b] = weight
            rows.append(dict(source_sensor=named[a], target_sensor=named[b], weight=weight, score=score,
                             original_hops=None if hub else int(hops[a,b]),
                             original_pair_resistance=None if hub else float(resistance[a,b])))
        assert np.array_equal(graph[:n, :n][adj != 0], adj[adj != 0])
        assert np.count_nonzero(graph) - np.count_nonzero(adj) == len(links)
        transition_direction_check(graph, [(a,b) for a,b,_ in links])
        pd.DataFrame(rows, columns=["source_sensor","target_sensor","weight","score","original_hops",
                                  "original_pair_resistance"]).to_csv(directory/f"{method}_links.csv", index=False)
        np.savez_compressed(directory/f"{method}_graph.npz", adjacency=graph, sensor_id=np.asarray(named, dtype=str))
        graphs[method] = (graph, named, links)
    breakdown = dict(inner_split=inner, arc_budget=count, link_weight=weight, min_hops=MIN_HOPS, shortlist=SHORTLIST,
                     positive_targets={k:len(v) for k,v in ranked.items()}, eligible_directed_pairs=int(eligible.sum()),
                     original_graph_sha256=graph_fingerprint(adj), source_speed_sha256=source[-1]["speeds_sha256"],
                     gtr_selection_seconds=gtr_seconds, shared_graph_preparation_seconds=graph_seconds,
                     joint_forecast_screening_seconds=score_seconds, total_selector_seconds=time.monotonic()-start,
                     added_arcs={mode:len(links) for mode,(_,_,links) in graphs.items()})
    save_json(directory/"selector_summary.json",breakdown)
    print(prepared["dataset"], breakdown, flush=True)
    return graphs

def variant_data(prepared, variant):
    graph, ids, links = variant
    view = {**prepared, "adj":graph, "ids":ids, "real_node_count":len(prepared["ids"]), "added_arcs":len(links)}
    if len(ids) != len(prepared["ids"]):
        # A neutral virtual input with no supervised target.
        view["scaled"] = np.pad(prepared["scaled"], ((0,0),(0,1)))
        view["labels"] = np.pad(prepared["labels"], ((0,0),(0,1)))
    return view


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
    adaptive = False
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    config = dict(dataset=dataset, profile=PROFILE, mode=mode, seed=int(seed), epochs=epochs, batch_size=BATCH,
                  adaptive=adaptive, embedding_init="normal (author code)" if adaptive else None,
                  input_steps=12, output_steps=12, dropout=.3, learning_rate=.001, weight_decay=.0001,
                  hidden_channels=32, diffusion_order=2, gradient_clip=5, adjacency="doubletransition",
                  code_commit=GWN_REF, graph_commit=DCRNN_REF, mean=data["scaler"].mean, std=data["scaler"].std,
                  notebook_protocol_version=2, graph_variant_sha256=graph_fingerprint(data["adj"]), device=str(DEVICE), torch=str(torch.__version__),
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
    real_nodes = data["real_node_count"]
    pred, truth, mask = pred[:, :, :real_nodes], truth[:, :, :real_nodes], mask[:, :, :real_nodes]
    dates = data["index"].to_numpy(dtype="datetime64[ns]")
    assert np.all(dates[target_rows] - dates[origins, None] == (Y_OFF * 5).astype("timedelta64[m]"))
    np.savez_compressed(run_dir / "predictions_12_horizons.npz", predictions=pred, targets=truth,
                        target_observed=mask, issue_timestamp=dates[origins],
                        target_timestamp=dates[target_rows], sensor_id=np.array(data["ids"][:real_nodes], dtype=str),
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
    target_ids = data["shared_targets"]
    target_idx = np.flatnonzero(np.isin(np.asarray(data["ids"])[:real_nodes], target_ids))
    assert len(target_idx) == len(target_ids)
    err = score_predictions(pred[:, -1, target_idx], truth[:, -1, target_idx], mask[:, -1, target_idx])
    if not len(data["bins"]):
        raise ValueError("Training data did not define a low-speed band.")
    slow_cutoff = float(data["bins"][0])
    shared_truth = truth[:, -1, target_idx]
    shared_mask = mask[:, -1, target_idx] & (shared_truth < slow_cutoff)
    if not shared_mask.any():
        raise ValueError("No observed low-speed targets on the shared evaluation slice.")
    shared_pred = pred[:, -1, target_idx]
    shared_low_error = score_predictions(shared_pred, shared_truth, shared_mask)
    # Hash sample identity, readings, cutoff and the low-speed mask across arms and seeds.
    digest = hashlib.sha256()
    for payload in (np.asarray(data["ids"][:real_nodes], dtype="<U32"),
                    dates[origins].astype("<M8[ns]"), dates[target_rows].astype("<M8[ns]"),
                    truth.astype("<f4", copy=False), mask,
                    np.asarray(target_ids, dtype="<U32"), np.asarray([slow_cutoff], dtype="<f8"), shared_mask):
        digest.update(np.ascontiguousarray(payload).tobytes())
    low = pd.DataFrame(regime_rows).query("horizon_minutes == 60 and band == 1").iloc[0]
    result = {
        "best_epoch": best_epoch, "best_validation_MAE": best, "test_origins": len(origins),
        "smoke": PROFILE == "smoke", "added_arcs": data["added_arcs"],
        "mae60": float(scores.iloc[-1]["MAE"]),
        "band1_mae60": None if pd.isna(low["MAE"]) else float(low["MAE"]),
        "band1_n60": int(low["valid_target_count"]),
        "target_low_mae60": float(shared_low_error["MAE"]),
        "target_low_n60": int(shared_mask.sum()),
        "target_sensor_n": len(target_idx),
        "low_speed_cutoff": slow_cutoff,
        "evaluation_sha256": digest.hexdigest(),
        "express_target_mae60": None if np.isnan(err["MAE"]) else float(err["MAE"]),
        "median_train_epoch_seconds": float(np.median([r["seconds"] for r in history])),
        "peak_allocated_mib_this_session": float(torch.cuda.max_memory_allocated(DEVICE) / 2**20) if DEVICE.type == "cuda" else None,
    }
    temp_result = run_dir / "result.json.tmp"
    save_json(temp_result, result)
    temp_result.replace(run_dir / "result.json")
    print(selected.to_string(index=False))
    return selected


def load_frozen_graphs(folder, audit, original):
    summary = json.loads((folder / "selector_summary.json").read_text())
    assert summary["source_speed_sha256"] == audit["speeds_sha256"]
    assert summary["original_graph_sha256"] == graph_fingerprint(original)
    manifest = json.loads((folder / "frozen_graphs.json").read_text())
    graphs = {}
    for mode in PILOT_MODES:
        with np.load(folder / f"{mode}_graph.npz") as payload:
            graph = payload["adjacency"]
            ids = payload["sensor_id"].astype(str).tolist()
        links = pd.read_csv(folder / f"{mode}_links.csv", dtype={"source_sensor":str,"target_sensor":str})
        index = {name:i for i,name in enumerate(ids)}
        edges = [(index[a], index[b], None) for a,b in zip(links.source_sensor, links.target_sensor)]
        assert manifest[mode]["adjacency_sha256"] == graph_fingerprint(graph)
        assert manifest[mode]["links_sha256"] == sha256(folder / f"{mode}_links.csv")
        assert manifest[mode]["file_sha256"] == sha256(folder / f"{mode}_graph.npz")
        assert len(edges) == summary["added_arcs"][mode]
        graphs[mode] = graph, ids, edges
    return graphs

dataset = args.dataset
h5, graph_name, *_ = SPECS[dataset]
source = validate_data(dataset, data_dir / h5, data_dir / graph_name)
folder = OUTPUT / "graphs" / dataset
audit_dir = OUTPUT / "audit" / dataset
if PROFILE == "full":
    # Unique work audit path for parallel workers.
    audit_dir = OUTPUT / "worker_audit" / dataset / args.mode / f"seed_{args.seed}"
prepared = prepare_dataset(dataset, source, "strict", audit_dir)
if PROFILE == "audit" or not (folder / "frozen_graphs.json").exists():
    graphs = build_pilot_graphs(source, prepared, folder)
    save_json(folder / "frozen_graphs.json", {
        mode: {"adjacency_sha256": graph_fingerprint(graph),
               "links_sha256": sha256(folder / f"{mode}_links.csv"),
               "file_sha256": sha256(folder / f"{mode}_graph.npz")}
        for mode, (graph, _, _) in graphs.items()
    })
else:
    graphs = load_frozen_graphs(folder, source[-1], prepared["adj"])
if PROFILE != "audit":
    targets = [prepared["ids"][b] for _, b, _ in graphs["express_links"][2]]
    for mode in MODES:
        view = variant_data(prepared, graphs[mode])
        view["shared_targets"] = targets
        run_experiment(view, source[-1], mode, args.seed)
        del view
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
print("Outputs:", OUTPUT, flush=True)
