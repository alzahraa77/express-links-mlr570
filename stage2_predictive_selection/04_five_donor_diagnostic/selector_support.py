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

def predictive_scores(source, prepared, eligible, resistance, all_candidates=False):
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
    if all_candidates:
        rows["control"] = []
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
        if all_candidates:
            shortlists["control"] = np.arange(len(candidates))
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

