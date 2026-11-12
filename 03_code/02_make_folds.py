"""02_make_folds.py   (03_code, step 2 of PLAN.md)
PURPOSE    Assigns every curve to one of 5 outer folds, grouped by mix (all curves of a mix in one fold), with
           GroupKFold(shuffle=True, random_state=2). Each fold is once the test part; the other four are the
           training part, tuned in step 3 by an inner grouped 5-fold.
WRITES     04_results/02_folds/folds_basic.csv, folds_total.csv (curve, mix, fold), summary.txt
DONE WHEN  no mix in more than one fold (assert); curves, readings and mixes per fold printed.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import common as C

OUT = C.RESULTS / '02_folds'
OUT.mkdir(parents=True, exist_ok=True)

lines = []
for creep in C.CREEP_TYPES:
    d, curves = C.load(creep)
    fold = np.full(len(d), -1)
    gkf = GroupKFold(n_splits=C.N_OUTER, shuffle=True, random_state=C.SEED)
    for k, (tr, te) in enumerate(gkf.split(d, groups=d[C.GROUP])):
        assert not set(d[C.GROUP].iloc[tr]) & set(d[C.GROUP].iloc[te]), 'group on both sides'
        fold[te] = k
    assert (fold >= 0).all()
    d['fold'] = fold
    f = d.groupby('curve').agg(mix=('mix', 'first'), fold=('fold', 'first'), n_fold=('fold', 'nunique'),
                               readings=('J', 'size')).reset_index()
    assert (f.n_fold == 1).all(), 'curve split over folds'
    if C.GROUP == 'mix':
        assert (f.groupby('mix').fold.nunique() == 1).all(), 'mix split over folds'
    f.drop(columns='n_fold').to_csv(C.folds_path(creep), index=False)

    s = f.groupby('fold').agg(curves=('curve', 'size'), readings=('readings', 'sum'), mixes=('mix', 'nunique'))
    s['test_share_%'] = (100 * s.readings / s.readings.sum()).round(1)
    lines.append(f'== {creep}: {len(f)} curves, {len(d)} readings, {f.mix.nunique()} mixes')
    lines.append(s.to_string())
    for k in range(C.N_OUTER):   # inner-fold size check: mixes available for the inner 5-fold
        lines.append(f'   outer fold {k}: training part {f[f.fold != k].mix.nunique()} mixes '
                     f'({f[f.fold != k].shape[0]} curves)')
txt = '\n'.join(lines)
(OUT / 'summary.txt').write_text(txt, encoding='utf-8')
print(txt)
