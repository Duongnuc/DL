"""05_random_search_check.py   (03_code, step 3 defence check; not for the paper)
PURPOSE    Does one-at-a-time tuning miss better settings? Draws 60 random combinations from the same grids
           (Bergstra & Bengio 2012) and scores them on the same grouped 5-fold of all curves; compares the best
           validation NRMSE with that of the sequential tuning.
WRITES     04_results/03_tuning/random_search_check.csv
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

import common as C
import importlib
T = importlib.import_module('03_tune')

N_DRAWS = 60


def run(creep, model):
    d, _ = C.load_folds(creep)
    X, y = d[C.FEATURES[creep]].to_numpy(float), d['J'].to_numpy(float)
    folds = list(GroupKFold(n_splits=C.N_INNER, shuffle=True, random_state=C.SEED).split(X, y, d[C.GROUP]))
    p = X.shape[1]
    rng = np.random.default_rng(C.SEED)
    g = T.grids(model, p)
    best = (np.inf, None)
    for _ in range(N_DRAWS):
        prm = {**T.defaults(model, p), **{k: v[rng.integers(len(v))] for k, v in g.items()}}
        s = T.cv_score(model, prm, X, y, folds)[0]
        if s < best[0]:
            best = (s, prm)
    seq = json.load(open(T.OUT / creep / 'all' / f'{model}_best.json'))
    return dict(creep=creep, model=model, sequential=round(seq['val_nrmse'], 1), random_60=round(best[0], 1),
                default=round(seq['default_val_nrmse'], 1), random_best_params=str(best[1]))


if __name__ == '__main__':
    res = Parallel(n_jobs=7)(delayed(run)(c, m) for c in C.CREEP_TYPES for m in ['DT', 'RF', 'ET', 'GB', 'XGB'])
    r = pd.DataFrame(res)
    r['random_minus_sequential'] = (r.random_60 - r.sequential).round(1)
    r.to_csv(T.OUT / 'random_search_check.csv', index=False)
    print(r.drop(columns='random_best_params').to_string(index=False))
