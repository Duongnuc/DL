"""10_grouping_check.py   (03_code, defence check; not for the paper)
PURPOSE    How much do the test scores depend on the validation design? With the parameters tuned on all curves
           (so not nested), compares 5-fold grouped by curve (as in the shrinkage paper) with grouped by mix.
WRITES     04_results/04_out_of_fold/grouping_check.csv
"""
import importlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import common as C

T = importlib.import_module('03_tune')
rows = []
for creep in C.CREEP_TYPES:
    d, _ = C.load_folds(creep)
    X, y = d[C.FEATURES[creep]].to_numpy(float), d['J'].to_numpy(float)
    for m in ['DT', 'RF', 'ET', 'GB', 'XGB']:
        prm = json.load(open(T.OUT / creep / 'all' / f'{m}_best.json'))['params']
        row = dict(creep=creep, model=m)
        for g in ['curve', 'mix']:
            p = np.empty(len(y))
            for tr, te in GroupKFold(C.N_OUTER, shuffle=True, random_state=C.SEED).split(X, y, d[g]):
                p[te] = T.make(m, prm).fit(X[tr], y[tr]).predict(X[te])
            row[f'R2_by_{g}_not_nested'] = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        rows.append(row)
r = pd.DataFrame(rows).round(2)
r.to_csv(C.RESULTS / '04_out_of_fold' / 'grouping_check.csv', index=False)
print(r.to_string(index=False))
