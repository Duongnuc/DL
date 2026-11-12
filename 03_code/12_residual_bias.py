"""12_residual_bias.py   (03_code, numbers for the text of step 7)
PURPOSE    Mean residual (measured - predicted, 1e-6/MPa) of each model on the pooled test readings, by load duration
           (< 10, 10-100, > 100 days): describes the residual figures (shrinkage-paper Figs 8 and 11).
WRITES     04_results/level_<LEVEL>/04_out_of_fold/residual_bias.csv
"""
import pandas as pd

import common as C

rows = []
for creep in C.CREEP_TYPES:
    p = pd.read_csv(C.RESULTS / '04_out_of_fold' / f'pred_{creep}.csv')
    p['duration'] = pd.cut(p.dt, [0, 10, 100, 1e5], labels=['<10', '10-100', '>100'])
    for m in ['B4TW-SCC', 'DT', 'RF', 'ET', 'GB', 'XGB']:
        for b, g in p.groupby('duration', observed=True):
            rows.append(dict(creep=creep, model=m, duration=b, readings=len(g), mean_residual=round((g.J - g[m]).mean(), 1)))
out = pd.DataFrame(rows)
out.to_csv(C.RESULTS / '04_out_of_fold' / 'residual_bias.csv', index=False)
print(out.pivot_table(index=['creep', 'duration'], columns='model', values='mean_residual', observed=True).to_string())
