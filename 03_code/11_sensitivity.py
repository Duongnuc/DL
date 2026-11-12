"""11_sensitivity.py   (03_code, numbers for the text of step 7)
PURPOSE    Effect of each hyperparameter in the first tuning pass on all curves: range (max - min) of the mean
           validation NRMSE over its grid, per model and creep type (the shrinkage paper reports the same quantity).
WRITES     04_results/level_<LEVEL>/03_tuning/sensitivity_pass1.csv
"""
import pandas as pd

import common as C

rows = []
for creep in C.CREEP_TYPES:
    for m in ['DT', 'RF', 'ET', 'GB', 'XGB']:
        t = pd.read_csv(C.RESULTS / '03_tuning' / creep / 'all' / f'{m}_trace.csv')
        t = t[t['pass'] == 1]
        for prm, g in t.groupby('param'):
            rows.append(dict(creep=creep, model=m, param=prm, nrmse_range=round(g.val_nrmse.max() - g.val_nrmse.min(), 1)))
out = pd.DataFrame(rows)
out.to_csv(C.RESULTS / '03_tuning' / 'sensitivity_pass1.csv', index=False)
print(out.pivot_table(index=['creep', 'param'], columns='model', values='nrmse_range').to_string())
