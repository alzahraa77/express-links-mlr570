#!/usr/bin/env python3
"""Replication (both datasets, seeds 0-2): summary tables after collect_replication.py has run."""
import pandas as pd
base = '/shared/g00112835/express_links_replication/pilot_output/full'
s = pd.read_csv(f'{base}/seed_summary.csv')
c = pd.read_csv(f'{base}/comparison.csv')
pd.set_option('display.width', 200)
for ds in ['METR-LA', 'PEMS-BAY']:
    print(f'\n===== {ds}: mean over 3 seeds (sd), change vs original, seeds better than original')
    t = s[s.dataset == ds].copy()
    t['cell'] = t.apply(lambda r: f"{r['mean']:.3f} ({r['sd']:.3f}) {r['mean_delta_vs_original']:+.3f} [{'' if pd.isna(r['paired_wins_vs_original']) else int(r['paired_wins_vs_original'])}/3]", axis=1)
    print(t.pivot(index='mode', columns='metric', values='cell')[['target_low_mae60', 'express_target_mae60', 'band1_mae60', 'mae60']].to_string())
    print(f'\n{ds}: main metric per seed')
    print(c[c.dataset == ds].pivot(index='mode', columns='seed', values='target_low_mae60').round(3).to_string())
r = c[(c.dataset == 'METR-LA') & (c.seed == 0)].set_index('mode')['mae60']
print('\nMETR-LA seed 0 re-scored mae60 (should equal the pilot):')
print(r.round(6).to_string())
