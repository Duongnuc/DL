"""15_quick_check.py   (03_code)
PURPOSE    Short version of the full run (owner 2026-11-13: "bản rút gọn"): skips the tuning (03_tune.py, ~1.5 h) and
           uses the hyperparameters already saved in 04_results/level_<level>/03_tuning/<creep>/outer<k>. For each
           outer fold it refits the five algorithms on the training part, predicts the test part and computes
           B4TW-SCC, then prints test R2 and NRMSE next to the saved values of the paper (Tables 4-6). Writes nothing.
RUN        python -X utf8 15_quick_check.py              (both levels, about 1-2 min)
           CREEP_LEVEL=mix python -X utf8 15_quick_check.py   (one level)
"""
import importlib
import json
import os
import subprocess
import sys
import time

if 'CREEP_LEVEL' not in os.environ:                      # run both levels, each in its own process
    for level in ('curve', 'mix'):
        r = subprocess.run([sys.executable, '-X', 'utf8', __file__], env={**os.environ, 'CREEP_LEVEL': level})
        if r.returncode:
            sys.exit(r.returncode)
    sys.exit(0)

import numpy as np
import pandas as pd

import common as C
from b4tw_scc_advisor import b4tw_scc_advisor

T = importlib.import_module('03_tune')
MODELS = ['DT', 'RF', 'ET', 'GB', 'XGB']
TOL = 0.005                                               # largest accepted difference in test R2


def r2_nrmse(y, p):
    q1, q3 = np.percentile(y, [25, 75])
    rmse = np.sqrt(np.mean((p - y) ** 2))
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), 100 * rmse / (q3 - q1)


t0 = time.time()
worst = 0.0
print(f'== level "{C.LEVEL}" ({"test curves held out" if C.LEVEL == "curve" else "whole mixes held out"})')
for creep in C.CREEP_TYPES:
    d, curves = C.load_folds(creep)
    X, y = d[C.FEATURES[creep]].to_numpy(float), d['J'].to_numpy(float)
    pred = pd.DataFrame(index=d.index, columns=['B4TW-SCC'] + MODELS, dtype=float)
    for cid, idx in d.groupby('curve').indices.items():
        pred.iloc[idx, 0] = b4tw_scc_advisor(curves.loc[cid], d['dt'].to_numpy()[idx])
    for k in range(C.N_OUTER):
        te = (d['fold'] == k).to_numpy()
        for m in MODELS:
            prm = json.load(open(T.OUT / creep / f'outer{k}' / f'{m}_best.json'))['params']
            pred.loc[te, m] = T.make(m, prm).fit(X[~te], y[~te]).predict(X[te])
    saved = pd.read_csv(C.RESULTS / '04_out_of_fold' / f'accuracy_{creep}.csv').set_index('model')
    rows = []
    for m in pred.columns:
        r2, nr = r2_nrmse(y, pred[m].to_numpy())
        diff = abs(r2 - saved.loc[m, 'test_R2'])
        worst = max(worst, diff)
        rows.append(dict(model=m, R2_now=round(r2, 3), R2_saved=saved.loc[m, 'test_R2'], NRMSE_now=round(nr, 1),
                         NRMSE_saved=round(saved.loc[m, 'test_NRMSE'], 1)))
    print(f'-- {creep} creep: {d.curve.nunique()} curves, {len(d)} readings')
    print(pd.DataFrame(rows).to_string(index=False))
verdict = 'matches the saved results' if worst <= TOL else f'DIFFERS from the saved results (max |dR2| {worst:.3f})'
print(f'== level "{C.LEVEL}": {verdict}; {time.time() - t0:.0f} s\n')
sys.exit(0 if worst <= TOL else 1)
