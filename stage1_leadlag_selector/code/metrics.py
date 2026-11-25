'''
Express Links · step 9 · metrics.py  (owner: Mohammed)

One scoring function that turns a run's forecasts into a results row:
  - masked MAE, RMSE, MAPE at 15, 30 and 60 minutes (all roads, and roads behind a bottleneck)
  - jam recall and precision at 30 and 60 minutes (all roads, and roads behind a bottleneck)

What it reads
  run_dir/test_predictions.npz   from a training run (step 3 / step 12): pred (S, 12, N) in mph, t_anchor (S,)
  data_dir/test.npz              from step 2 / step 8: x, y (S, 12, N, 2), t_index, t_anchor
  behind-a-bottleneck subset, one of:
    data_dir/bottleneck/regions.npz          step 5 (METR-LA, PEMS-BAY): per-road flag behind_bottleneck
    raw/toy/<dataset>_jam_arrivals.csv       step 8 (toy cities): per-jam flag 'crossed' (the jam came from the other district)
    raw/sharjah/<dataset>_jam_arrivals.csv   step 11 (Sharjah): the same file and flag

Depends on other people's steps only for inputs, not for code:
  step 7  (Al Zahraa)  express graph           -> needed for the week-2 checkpoint run, not for scoring
  step 10 (Al Zahraa)  seven graph versions    -> each version is just another run folder to score
  step 12 (together)   the 42 + 42 runs        -> score_many() loops over them

The jam rule (proposal 3.4)
  A jam starts at road n, time t when the speed at t is at least 12.4 mph (20 km/h) below the highest
  reading of the previous 15 minutes, and stays that low for the next 15 minutes. Zero readings are
  missing data: a jam is never declared on them.

The warning rule
  A forecast made at anchor a warns of a jam at road n, H minutes ahead, when the jam rule, applied to
  the series [last 15 min observed, then the 12 predicted steps], finds a jam START at step H: the
  predicted speed is at least 12.4 mph below the highest of the 3 steps before it, it was not already
  that low one step earlier, and it stays that low for the predicted steps that follow (3 steps at
  30 min; at 60 min the forecast ends there, so only the step itself is checked).
  Recall(H)    = share of real jam starts t for which some forecast made at a in [t-H-TOL, t-H+TOL] warned.
  Precision(H) = share of warnings (a, n) for which a real jam starts at road n in [a+H-TOL, a+H+TOL].
  TOL is 15 minutes (3 steps), the same granularity as the jam rule.
'''
import os, json
import numpy as np
import pandas as pd

STEP_MIN   = 5
JAM_DROP   = 12.4          # mph  (20 km/h)
WITHIN     = 3             # the drop happens within 15 min
LASTS      = 3             # and lasts 15 min
TOL        = 3             # timing tolerance for warnings, in steps (15 min)
HORIZONS   = {15: 2, 30: 5, 60: 11}       # minutes -> index in the 12-step forecast
WARN_H     = (30, 60)


# ---------------------------------------------------------------------------
# 1. Standard errors, masked like Graph WaveNet (null value 0 = missing reading)
# ---------------------------------------------------------------------------
def masked_errors(pred, true, null_val=0.0):
    '''MAE, RMSE, MAPE (in %) over entries where true != null_val.'''
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    mask = true != null_val
    if not mask.any():
        return dict(mae=np.nan, rmse=np.nan, mape=np.nan)
    err = pred[mask] - true[mask]
    return dict(mae=float(np.abs(err).mean()),
                rmse=float(np.sqrt((err ** 2).mean())),
                mape=float((np.abs(err) / np.abs(true[mask])).mean() * 100))


# ---------------------------------------------------------------------------
# 2. The jam rule on an observed speed table (T, N)
# ---------------------------------------------------------------------------
def jam_state(speed):
    '''True at (t, n) when the jam rule holds there. speed: (T, N), zeros = missing.'''
    T, N = speed.shape
    state = np.zeros((T, N), bool)
    for t in range(WITHIN, T - LASTS):
        prev = speed[t - WITHIN:t]
        after = speed[t:t + LASTS + 1]
        ok = (prev > 0).all(axis=0) & (after > 0).all(axis=0)       # no missing readings involved
        level = prev.max(axis=0) - JAM_DROP
        state[t] = ok & (after <= level[None, :] + 1e-6).all(axis=0)
    return state


def jam_starts(speed):
    '''Jam starts: the first step of each run of jam_state. Returns a DataFrame with columns t, n.'''
    state = jam_state(speed)
    start = state & ~np.vstack([np.zeros((1, state.shape[1]), bool), state[:-1]])
    t, n = np.nonzero(start)
    return pd.DataFrame({'t': t, 'n': n})


# ---------------------------------------------------------------------------
# 3. Rebuild the test-period speed table from test.npz (no raw file needed)
# ---------------------------------------------------------------------------
def test_series(test_npz):
    '''Observed speeds (T_test, N) for the rows the test samples cover, plus the row offset.
    Row r of the result is raw row r + offset.'''
    x, y, t_index = test_npz['x'][..., 0], test_npz['y'][..., 0], test_npz['t_index']
    seq_in, seq_out = x.shape[1], y.shape[1]
    first, last = int(t_index[0]) - (seq_in - 1), int(t_index[-1]) + seq_out
    speed = np.full((last - first + 1, x.shape[2]), np.nan, np.float32)
    speed[0:seq_in] = x[0]
    for s in range(len(t_index)):
        r = int(t_index[s]) - first
        speed[r + 1:r + 1 + seq_out] = y[s]
    assert not np.isnan(speed).any(), 'test samples are not consecutive'
    return speed, first


# ---------------------------------------------------------------------------
# 4. Warnings and their scoring
# ---------------------------------------------------------------------------
def warnings_at(pred, x_obs, h_index):
    '''(S, N) bool: the forecast made at anchor s predicts a jam STARTING h_index+1 steps ahead.
    The jam rule is applied to the series [last 15 min observed, then the 12 predicted steps]:
    the predicted speed at that step is >= 12.4 mph below the highest of the 3 steps before it,
    it was not already that low one step earlier, and it stays that low for the next 3 predicted
    steps (fewer at the end of the forecast: at 60 min only the step itself can be checked).
    pred: (S, 12, N) mph; x_obs: (S, 12, N) observed inputs (last row = anchor).'''
    series = np.concatenate([x_obs[:, -WITHIN:, :], pred], axis=1)       # (S, 3 + 12, N)
    k = WITHIN + h_index
    prev = series[:, k - WITHIN:k, :]
    after = series[:, k:min(k + LASTS + 1, series.shape[1]), :]
    ok = (prev > 0).all(axis=1)
    level = prev.max(axis=1) - JAM_DROP
    not_jammed_yet = prev[:, -1, :] > level + 1e-6
    return ok & not_jammed_yet & (after <= level[:, None, :] + 1e-6).all(axis=1)


def recall_precision(jams, warn, t_index, h_steps, road_mask=None, jam_mask=None):
    '''jams: DataFrame (t, n) of real jam starts in raw-row units; warn: (S, N) warnings whose anchors
    are raw rows t_index. road_mask (N,) or jam_mask (len(jams),) restrict the subset.'''
    S, N = warn.shape
    anchor_row = {int(r): s for s, r in enumerate(t_index)}
    rows = np.asarray(t_index)

    # a lookup of jam starts per road, for precision
    starts_by_road = {n: np.sort(g['t'].values) for n, g in jams.groupby('n')}

    # recall: for each real jam, any warning anchored in [t-h-TOL, t-h+TOL]?
    keep = np.ones(len(jams), bool)
    if road_mask is not None:
        keep &= road_mask[jams['n'].values]
    if jam_mask is not None:
        keep &= jam_mask
    hit = []
    for t, n in jams.loc[keep, ['t', 'n']].values:
        found = False
        for a in range(t - h_steps - TOL, t - h_steps + TOL + 1):
            s = anchor_row.get(a)
            if s is not None and warn[s, n]:
                found = True
                break
        hit.append(found)
    n_jams = int(keep.sum())
    recall = float(np.mean(hit)) if n_jams else np.nan

    # precision: for each warning, a real jam start at that road in [a+h-TOL, a+h+TOL]?
    # Only defined for a fixed set of roads. For a per-jam subset (toy cities: "the jam came from the
    # other district") a false warning belongs to no jam, so precision is reported as NaN there.
    if jam_mask is not None:
        n_warn = 0
        return dict(recall=recall, precision=np.nan, jams=n_jams, warnings=n_warn,
                    hits=int(np.sum(hit)) if n_jams else 0, correct=0)
    ws, wn = np.nonzero(warn)
    if road_mask is not None:
        sel = road_mask[wn]
        ws, wn = ws[sel], wn[sel]
    correct = []
    for s, n in zip(ws, wn):
        target = rows[s] + h_steps
        st = starts_by_road.get(n)
        correct.append(st is not None and np.any((st >= target - TOL) & (st <= target + TOL)))
    n_warn = len(correct)
    precision = float(np.mean(correct)) if n_warn else np.nan
    return dict(recall=recall, precision=precision, jams=n_jams, warnings=n_warn,
                hits=int(np.sum(hit)) if n_jams else 0, correct=int(np.sum(correct)) if n_warn else 0)


# ---------------------------------------------------------------------------
# 5. The behind-a-bottleneck subset
# ---------------------------------------------------------------------------
def load_subset(data_dir, jams=None, row_offset=0, raw_toy_dir=None):
    '''Returns (road_mask or None, jam_mask or None, source description).
    Benchmarks: step 5 regions.npz gives a per-road flag.
    Toy cities: step 8 jam_arrivals.csv says, per logged jam, whether it crossed the bottleneck.'''
    reg = os.path.join(data_dir, 'bottleneck', 'regions.npz')
    if os.path.exists(reg):
        r = np.load(reg, allow_pickle=True)
        return r['behind_bottleneck'].astype(bool), None, 'step 5 regions.npz (behind_bottleneck)'

    name = os.path.basename(os.path.normpath(data_dir))
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(data_dir)), 'raw')
    # toy cities live in raw/toy, Sharjah in raw/sharjah (step 11): look in both
    folders = [raw_toy_dir] if raw_toy_dir else []
    folders += [os.path.join(raw_dir, 'toy'), os.path.join(raw_dir, 'sharjah')]
    arr_path = next((os.path.join(d, f'{name}_jam_arrivals.csv') for d in folders
                     if os.path.exists(os.path.join(d, f'{name}_jam_arrivals.csv'))), '')
    if arr_path and jams is not None:
        arr = pd.read_csv(arr_path)
        # match each detected jam start to the nearest logged arrival at the same road (within 2*TOL)
        crossed = np.zeros(len(jams), bool)
        matched = np.zeros(len(jams), bool)
        by_road = {n: g.sort_values('arrival_row') for n, g in arr.groupby('road')}
        for i, (t, n) in enumerate(jams[['t', 'n']].values):
            g = by_road.get(n)
            if g is None:
                continue
            d = np.abs(g['arrival_row'].values - (t + row_offset))
            k = int(np.argmin(d))
            if d[k] <= 2 * TOL:
                matched[i] = True
                crossed[i] = bool(g['crossed'].values[k])
        return None, crossed, f'step 8 {name}_jam_arrivals.csv (crossed; {matched.mean():.0%} of detected jams matched to the log)'
    return None, None, 'none found: behind-bottleneck columns will be empty'


# ---------------------------------------------------------------------------
# 6. Score one run
# ---------------------------------------------------------------------------
def score_run(run_dir, data_dir, raw_toy_dir=None, save=True):
    '''Returns one results row (dict) and writes run_dir/score.json.'''
    P = np.load(os.path.join(run_dir, 'test_predictions.npz'))
    D = np.load(os.path.join(data_dir, 'test.npz'))
    pred = P['pred'].astype(np.float32)                    # (S, 12, N) mph
    x_obs, y_true, t_index = D['x'][..., 0], D['y'][..., 0], D['t_index'].astype(int)
    assert pred.shape == y_true.shape, f'pred {pred.shape} vs test y {y_true.shape}'
    assert np.array_equal(P['t_anchor'], D['t_anchor']), 'run and test set have different anchors'

    speed, offset = test_series(D)
    jams = jam_starts(speed)
    jams['t'] = jams['t'] + offset                          # raw-row units, like t_index
    road_mask, jam_mask, subset_src = load_subset(data_dir, jams, 0, raw_toy_dir)

    row = {}
    cfg = os.path.join(run_dir, 'results.json')
    if os.path.exists(cfg):
        with open(cfg) as f:
            c = json.load(f)
        row.update({k: c.get(k) for k in ('dataset', 'graph', 'seed')})
    row.update(run=run_dir, subset_source=subset_src, test_samples=int(len(t_index)), roads=int(pred.shape[2]))

    # standard errors
    for mins, h in HORIZONS.items():
        e = masked_errors(pred[:, h], y_true[:, h])
        row.update({f'{k}_{mins}': v for k, v in e.items()})
        if road_mask is not None:
            e = masked_errors(pred[:, h][:, road_mask], y_true[:, h][:, road_mask])
            row.update({f'{k}_{mins}_behind': v for k, v in e.items()})
    e = masked_errors(pred, y_true)
    row.update({f'{k}_avg': v for k, v in e.items()})

    # jam warnings
    for mins in WARN_H:
        h = HORIZONS[mins]
        warn = warnings_at(pred, x_obs, h)
        r = recall_precision(jams, warn, t_index, h + 1)
        row.update({f'jam_recall_{mins}': r['recall'], f'jam_precision_{mins}': r['precision'],
                    f'jams_{mins}': r['jams'], f'warnings_{mins}': r['warnings']})
        if road_mask is not None or jam_mask is not None:
            r = recall_precision(jams, warn, t_index, h + 1, road_mask, jam_mask)
            row.update({f'jam_recall_{mins}_behind': r['recall'], f'jam_precision_{mins}_behind': r['precision'],
                        f'jams_{mins}_behind': r['jams'], f'warnings_{mins}_behind': r['warnings']})
    row['real_jams_in_test'] = int(len(jams))

    if save:
        with open(os.path.join(run_dir, 'score.json'), 'w') as f:
            json.dump(row, f, indent=2)
    return row


def score_many(run_dirs, data_root, raw_toy_dir=None, table_path=None):
    '''Score several run folders (each must have results.json naming its dataset) into one table.'''
    rows = []
    for rd in run_dirs:
        with open(os.path.join(rd, 'results.json')) as f:
            ds = json.load(f)['dataset']
        rows.append(score_run(rd, os.path.join(data_root, ds), raw_toy_dir))
    table = pd.DataFrame(rows)
    if table_path:
        if os.path.exists(table_path):
            old = pd.read_csv(table_path)
            old = old[~old['run'].isin(table['run'])]
            table = pd.concat([old, table], ignore_index=True)
        table.to_csv(table_path, index=False)
    return table


def summarise(table, by=('dataset', 'graph')):
    '''Mean (std) over seeds for the headline columns.'''
    cols = [c for c in table.columns if c.startswith(('mae_', 'rmse_', 'mape_', 'jam_'))]
    g = table.groupby(list(by))[cols]
    m, s = g.mean(), g.std()
    out = m.copy().astype(object)
    for c in cols:
        out[c] = [f'{a:.3f} ({b:.3f})' if not np.isnan(b) else f'{a:.3f}' for a, b in zip(m[c], s[c])]
    out['seeds'] = g.size()
    return out
