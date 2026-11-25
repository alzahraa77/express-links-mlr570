#!/usr/bin/env python3
"""Pilot (METR-LA, seed 0): 60-minute MAE on each of the 25 Express Links target roads,
for original, express_links and random_matched, and the average change per source road."""
import numpy as np, pandas as pd
base = '/shared/g00112835/express_links_pilot/pilot_output'
links = pd.read_csv(f'{base}/graphs/METR-LA/express_links_links.csv', dtype=str)
def per_road(mode):
    z = np.load(f'{base}/full/METR-LA/{mode}/seed_0/predictions_12_horizons.npz')
    ids = list(z['sensor_id'])
    p, y, m = z['predictions'][:, -1], z['targets'][:, -1], z['target_observed'][:, -1]
    err = np.abs(p - y)
    return pd.Series({s: err[m[:, ids.index(s)], ids.index(s)].mean() for s in links.target_sensor})
d = pd.DataFrame({k: per_road(k) for k in ['original', 'express_links', 'random_matched']})
d['source'] = links.set_index('target_sensor').source_sensor
d['change_express'] = d.express_links - d.original
d['change_random'] = d.random_matched - d.original
print(d.round(3).sort_values('change_express').to_string())
print('Express Links improved', (d.change_express < 0).sum(), 'of', len(d), 'target roads')
print(d.groupby('source').change_express.agg(['count', 'mean']).round(3))
