from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

SPECS = {"METR-LA": ("metr-la.h5", "adj_mx.pkl", 34272, 207, 1515)}

def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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

