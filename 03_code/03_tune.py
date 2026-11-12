"""03_tune.py   (03_code, step 3 of PLAN.md)
PURPOSE    Sequential (one-at-a-time) tuning as in the shrinkage paper: starting from the package defaults, each
           hyperparameter in turn is varied over a grid with the others fixed and set to the value with the lowest
           mean validation NRMSE of a grouped 5-fold (groups = mixes); then a second pass over all of them.
           Runs on all curves ('all': parameter table and max_depth figure) and on the training part of each
           outer fold ('outer0'..'outer4': used in step 4 for the out-of-fold test).
WRITES     04_results/03_tuning/<creep>/<set>/<model>_trace.csv, <model>_best.json
USAGE      python 03_tune.py                     # everything, in parallel
           python 03_tune.py basic all DT        # one job (smoke test)
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import common as C

OUT = C.RESULTS / '03_tuning'
MODELS = ['DT', 'RF', 'ET', 'GB', 'XGB']
DEPTH = list(range(1, 21))
LEAVES = [None, 5, 10, 20, 30, 50, 70, 100, 150, 200]
SPLIT = [2, 3, 4, 5, 6, 8, 10, 15, 20]
LEAF = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
NTREES = [50, 100, 200, 300, 500]
LR = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
SUB = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def grids(model, p):
    """Ordered grids; max_depth first (shrinkage paper). p = number of inputs."""
    if model == 'DT':
        return {'max_depth': DEPTH + [None], 'max_leaf_nodes': LEAVES, 'min_samples_split': SPLIT,
                'min_samples_leaf': LEAF}
    if model in ('RF', 'ET'):
        return {'max_depth': DEPTH + [None], 'max_features': list(range(1, p + 1)), 'max_leaf_nodes': LEAVES,
                'min_samples_split': SPLIT, 'min_samples_leaf': LEAF}
    if model == 'GB':
        return {'max_depth': DEPTH, 'n_estimators': NTREES, 'learning_rate': LR, 'max_leaf_nodes': LEAVES,
                'min_samples_split': SPLIT, 'min_samples_leaf': LEAF, 'subsample': SUB}
    return {'max_depth': DEPTH, 'n_estimators': NTREES, 'learning_rate': LR, 'min_child_weight': [1, 2, 3, 5, 7, 10],
            'subsample': SUB, 'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}


def defaults(model, p):
    if model == 'DT':
        return dict(max_depth=None, max_leaf_nodes=None, min_samples_split=2, min_samples_leaf=1)
    if model in ('RF', 'ET'):
        return dict(n_estimators=100, max_depth=None, max_features=p, max_leaf_nodes=None, min_samples_split=2,
                    min_samples_leaf=1)
    if model == 'GB':
        return dict(n_estimators=100, learning_rate=0.1, max_depth=3, max_leaf_nodes=None, min_samples_split=2,
                    min_samples_leaf=1, subsample=1.0)
    return dict(n_estimators=100, learning_rate=0.3, max_depth=6, min_child_weight=1, subsample=1.0,
                colsample_bytree=1.0)


def make(model, prm):
    if model == 'DT':
        return DecisionTreeRegressor(random_state=C.SEED, **prm)
    if model == 'RF':
        return RandomForestRegressor(random_state=C.SEED, n_jobs=1, **prm)
    if model == 'ET':
        return ExtraTreesRegressor(random_state=C.SEED, n_jobs=1, **prm)
    if model == 'GB':
        return GradientBoostingRegressor(random_state=C.SEED, **prm)
    return XGBRegressor(random_state=C.SEED, n_jobs=1, tree_method='hist', **prm)


def nrmse(y, p):
    q1, q3 = np.percentile(y, [25, 75])
    return 100 * np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)) / (q3 - q1)


def cv_score(model, prm, X, y, folds):
    va, tr = [], []
    for itr, iva in folds:
        m = make(model, prm).fit(X[itr], y[itr])
        va.append(nrmse(y[iva], m.predict(X[iva])))
        tr.append(nrmse(y[itr], m.predict(X[itr])))
    return float(np.mean(va)), float(np.std(va)), float(np.mean(tr))


def tune(creep, set_name, model):
    d, _ = C.load_folds(creep)
    if set_name != 'all':
        d = d[d['fold'] != int(set_name[-1])].reset_index(drop=True)
    X, y = d[C.FEATURES[creep]].to_numpy(float), d['J'].to_numpy(float)
    folds = list(GroupKFold(n_splits=C.N_INNER, shuffle=True, random_state=C.SEED).split(X, y, d[C.GROUP]))
    p = X.shape[1]
    prm, rows, t0 = defaults(model, p), [], time.time()
    base = cv_score(model, prm, X, y, folds)
    rows.append(dict(pass_=0, param='default', value='', val_nrmse=base[0], val_sd=base[1], train_nrmse=base[2]))
    for pss in (1, 2):
        for name, grid in grids(model, p).items():
            res = []
            for v in grid:
                s = cv_score(model, {**prm, name: v}, X, y, folds)
                res.append((v, *s))
                rows.append(dict(pass_=pss, param=name, value=str(v), val_nrmse=s[0], val_sd=s[1], train_nrmse=s[2]))
            prm[name] = min(res, key=lambda r: r[1])[0]          # lowest mean validation NRMSE; ties -> first
    final = cv_score(model, prm, X, y, folds)
    out = OUT / creep / set_name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).rename(columns={'pass_': 'pass'}).to_csv(out / f'{model}_trace.csv', index=False)
    json.dump(dict(params=prm, val_nrmse=final[0], val_sd=final[1], train_nrmse=final[2], default_val_nrmse=base[0],
                   n_readings=len(d), n_curves=int(d['curve'].nunique()), n_mixes=int(d['mix'].nunique()),
                   seconds=round(time.time() - t0, 1)), open(out / f'{model}_best.json', 'w'), indent=1)
    return f'{creep} {set_name} {model}: val NRMSE {base[0]:.1f} -> {final[0]:.1f} %  ({time.time() - t0:.0f} s)'


if __name__ == '__main__':
    if len(sys.argv) == 4:
        print(tune(*sys.argv[1:]))
    else:
        jobs = [(c, s, m) for c in C.CREEP_TYPES for s in ['all'] + [f'outer{k}' for k in range(C.N_OUTER)]
                for m in MODELS if not (OUT / c / s / f'{m}_best.json').exists()]
        jobs.sort(key=lambda j: j[2] in ('GB', 'RF', 'ET'), reverse=True)   # slow ones first
        for msg in Parallel(n_jobs=7, verbose=0, return_as='generator')(delayed(tune)(*j) for j in jobs):
            print(msg, flush=True)
