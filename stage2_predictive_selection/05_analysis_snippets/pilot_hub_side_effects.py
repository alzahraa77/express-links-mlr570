#!/usr/bin/env python3
"""Pilot (METR-LA, seed 0): did the main source roads and their normal neighbours get worse?"""
import numpy as np
base = '/shared/g00112835/express_links_pilot/pilot_output'
g = np.load(f'{base}/graphs/METR-LA/original_graph.npz'); A = g['adjacency']; ids = list(g['sensor_id'])
def road_mae(mode):
    z = np.load(f'{base}/full/METR-LA/{mode}/seed_0/predictions_12_horizons.npz')
    e = np.abs(z['predictions'][:, -1] - z['targets'][:, -1]); m = z['target_observed'][:, -1]
    return np.array([e[m[:, j], j].mean() for j in range(e.shape[1])])
o, x = road_mae('original'), road_mae('express_links')
for hub in ['717825', '763995', '717823']:
    i = ids.index(hub)
    nb = [j for j in np.flatnonzero((A[i] > 0) | (A[:, i] > 0)) if j != i]
    print(hub, '| own change', round(x[i] - o[i], 3), '| normal neighbours', len(nb), 'mean change', round((x[nb] - o[nb]).mean(), 3))
print('all 207 roads mean change', round((x - o).mean(), 3))
