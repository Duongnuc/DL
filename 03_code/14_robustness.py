"""14_robustness.py   (03_code, extra analyses approved by the owner on 2026-11-13: "Chạy cả 3")
PURPOSE    Three simple robustness checks of the main result (the design of steps 2-4 is unchanged):
  seeds    Repeat the whole nested cross-validation (outer folds, inner folds, model randomness) with 10 seeds
           (1-10; seed 2 is the paper). Sources: Bouckaert & Frank 2004 (10 x repeated CV for replicability);
           Bouthillier et al. 2021 (vary every source of randomness, report how often A beats B).
           Per seed: pooled test R2, NRMSE, RMSE, MAE of every model and the RMSE difference from B4TW-SCC.
  random   Readings split at random (no grouping), same nested tuning, seed 2: how much a random split inflates
           the scores (Roberts et al. 2017, Box 2; Kapoor & Narayanan 2023). Level-independent.
  mono     With the paper's outer models (seed 2), predict every test curve on a dense load-duration grid and
           count curves whose predicted compliance decreases with time (physical inconsistency, Daw et al. PGNN
           §4.3). A decrease = a drop of more than 0.5 % between two successive grid points.
CHECK      Seed 2 should reproduce 04_out_of_fold/pred_<creep>.csv of the paper; the largest difference is written
           to reproduction_check.txt (another OS or CPU may differ slightly).
READS      02_data/cleaned_curves.csv, cleaned_readings.csv; 04_results/level_<LEVEL>/04_out_of_fold/pred_*.csv
WRITES     04_results/level_<LEVEL>/07_robustness/seeds/seed_<s>/<creep>/outer<k>/<model>_best.json,
           seeds/seed_<s>/pred_<creep>.csv, seeds_metrics.csv, random_split_metrics.csv, monotonic.csv, summary.txt
USAGE      CREEP_LEVEL=curve python 14_robustness.py seeds  [--seeds 1-10] [--jobs N]
           CREEP_LEVEL=mix   python 14_robustness.py seeds
           CREEP_LEVEL=curve python 14_robustness.py random
           CREEP_LEVEL=curve python 14_robustness.py mono
           CREEP_LEVEL=curve python 14_robustness.py smoke      # 1 model, 1 fold, short grids: seconds
           Finished jobs (a *_best.json on disk) are skipped, so a stopped run can be restarted.
"""
import argparse
import importlib
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, KFold

import common as C
from b4tw_scc_advisor import b4tw_scc_advisor

T = importlib.import_module('03_tune')
OUT = C.RESULTS / '07_robustness'
MODELS = T.MODELS
ALL = ['B4TW-SCC'] + MODELS


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    q1, q3 = np.percentile(y, [25, 75])
    rmse = np.sqrt(np.mean((p - y) ** 2))
    return dict(R2=1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), NRMSE=100 * rmse / (q3 - q1),
                RMSE=rmse, MAE=np.mean(np.abs(p - y)))


def make(model, prm, seed):
    """As 03_tune.make, with the seed as an argument."""
    m = T.make(model, prm)
    m.set_params(random_state=seed)
    return m


def splitter(group, n, seed):
    """'curve'/'mix': GroupKFold as in steps 2-3; 'none': readings at random."""
    if group == 'none':
        return lambda X, g: KFold(n_splits=n, shuffle=True, random_state=seed).split(X)
    return lambda X, g: GroupKFold(n_splits=n, shuffle=True, random_state=seed).split(X, groups=g)


def outer_folds(creep, group, seed):
    """Outer fold (0-4) of every reading, as 02_make_folds.py does for seed 2."""
    d, curves = C.load(creep)
    fold = np.full(len(d), -1)
    g = d[group] if group != 'none' else None
    for k, (_, te) in enumerate(splitter(group, C.N_OUTER, seed)(d, g)):
        fold[te] = k
    assert (fold >= 0).all()
    d['fold'] = fold
    return d, curves


def tune_job(creep, group, seed, k, model, out, grids=None):
    """Sequential tuning of 03_tune.tune on the training part of outer fold k, with this seed."""
    path = out / creep / f'outer{k}' / f'{model}_best.json'
    if path.exists():
        return f'skip {path.parent.name} {model}'
    t0 = time.time()
    d, _ = outer_folds(creep, group, seed)
    d = d[d['fold'] != k].reset_index(drop=True)
    X, y = d[C.FEATURES[creep]].to_numpy(float), d['J'].to_numpy(float)
    g = d[group] if group != 'none' else None
    folds = list(splitter(group, C.N_INNER, seed)(X, g))

    def score(prm):
        va = []
        for itr, iva in folds:
            m = make(model, prm, seed).fit(X[itr], y[itr])
            va.append(T.nrmse(y[iva], m.predict(X[iva])))
        return float(np.mean(va))

    p = X.shape[1]
    prm = T.defaults(model, p)
    for _ in (1, 2):
        for name, grid in (grids or T.grids(model, p)).items():
            res = [(v, score({**prm, name: v})) for v in grid]
            prm[name] = min(res, key=lambda r: r[1])[0]               # lowest mean validation NRMSE; ties -> first
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(params=prm, seed=seed, group=group, seconds=round(time.time() - t0, 1)), open(path, 'w'), indent=1)
    return f'{creep} seed {seed} outer{k} {model}: {time.time() - t0:.0f} s'


def predict(creep, group, seed, out, models=MODELS):
    """Out-of-fold predictions with the tuned parameters, as 06_out_of_fold.py."""
    d, curves = outer_folds(creep, group, seed)
    X, y = d[C.FEATURES[creep]].to_numpy(float), d['J'].to_numpy(float)
    pred = d[['curve', 'mix', 'fold', 'dt', 'J']].copy()
    pred['B4TW-SCC'] = np.nan
    for cid, idx in d.groupby('curve').indices.items():
        pred.loc[idx, 'B4TW-SCC'] = b4tw_scc_advisor(curves.loc[cid], d['dt'].to_numpy()[idx])
    for m in models:
        pred[m] = np.nan
        for k in range(C.N_OUTER):
            te = (d['fold'] == k).to_numpy()
            prm = json.load(open(out / creep / f'outer{k}' / f'{m}_best.json'))['params']
            pred.loc[te, m] = make(m, prm, seed).fit(X[~te], y[~te]).predict(X[te])
    pred.to_csv(out / f'pred_{creep}.csv', index=False)
    return pred


def score_rows(pred, **tag):
    y = pred['J']
    b = metrics(y, pred['B4TW-SCC'])
    return [dict(**tag, model=m, **{f'test_{k}': v for k, v in metrics(y, pred[m]).items()},
                 RMSE_minus_B4TW=metrics(y, pred[m])['RMSE'] - b['RMSE']) for m in ALL if m in pred]


def run_jobs(jobs, n_jobs):
    jobs.sort(key=lambda j: j[4] in ('GB', 'RF', 'ET'), reverse=True)        # slow ones first
    for msg in Parallel(n_jobs=n_jobs, return_as='generator')(delayed(tune_job)(*j) for j in jobs):
        print(msg, flush=True)


def seeds(seed_list, n_jobs):
    group = C.GROUP
    base = OUT / 'seeds'
    jobs = [(c, group, s, k, m, base / f'seed_{s}') for s in seed_list for c in C.CREEP_TYPES
            for k in range(C.N_OUTER) for m in MODELS]
    run_jobs(jobs, n_jobs)
    rows = []
    for s in seed_list:
        for c in C.CREEP_TYPES:
            pred = predict(c, group, s, base / f'seed_{s}')
            if s == C.SEED:                                                 # reproduces the paper
                ref = pd.read_csv(C.RESULTS / '04_out_of_fold' / f'pred_{c}.csv')
                diff = np.abs(pred[ALL].to_numpy(float) - ref[ALL].to_numpy(float)).max()
                note = 'reproduces' if diff < 1e-6 else 'WARNING: differs from'   # another OS/CPU may differ slightly
                msg = f'seed {s} {c}: {note} the paper predictions (max abs difference {diff:.2e})'
                print(msg, flush=True)
                with open(OUT / 'reproduction_check.txt', 'a', encoding='utf-8') as fh:
                    fh.write(msg + '\n')
            rows += score_rows(pred, level=C.LEVEL, creep=c, seed=s)
    r = pd.DataFrame(rows)
    r.round(4).to_csv(OUT / 'seeds_metrics.csv', index=False)
    s = (r.groupby(['creep', 'model'])
          .agg(R2_mean=('test_R2', 'mean'), R2_min=('test_R2', 'min'), R2_max=('test_R2', 'max'),
               NRMSE_mean=('test_NRMSE', 'mean'), NRMSE_min=('test_NRMSE', 'min'), NRMSE_max=('test_NRMSE', 'max'),
               seeds_better_than_B4TW=('RMSE_minus_B4TW', lambda v: int((v < 0).sum())), n_seeds=('seed', 'size')))
    return f'== {C.LEVEL} level, seeds {seed_list}: test scores over seeds\n' + s.round(3).to_string()


def random_split(n_jobs):
    base = OUT / 'random_split'
    jobs = [(c, 'none', C.SEED, k, m, base) for c in C.CREEP_TYPES for k in range(C.N_OUTER) for m in MODELS]
    run_jobs(jobs, n_jobs)
    rows = []
    for c in C.CREEP_TYPES:
        rows += score_rows(predict(c, 'none', C.SEED, base), split='readings at random', creep=c, seed=C.SEED)
        ref = pd.read_csv(C.RESULTS / '04_out_of_fold' / f'pred_{c}.csv')
        rows += score_rows(ref, split=f'grouped by {C.GROUP} (paper)', creep=c, seed=C.SEED)
    r = pd.DataFrame(rows)
    r.round(4).to_csv(OUT / 'random_split_metrics.csv', index=False)
    return '== random split of readings vs the paper split (test R2, NRMSE %)\n' + \
        r.pivot_table(index=['creep', 'model'], columns='split', values=['test_R2', 'test_NRMSE']).round(2).to_string()


def monotonic():
    """Dense grid of load durations for every test curve; share of curves with a predicted decrease > 0.5 %."""
    T6 = C.RESULTS / '03_tuning'
    rows = []
    for creep in C.CREEP_TYPES:
        d, curves = C.load_folds(creep)
        X, y = d[C.FEATURES[creep]].to_numpy(float), d['J'].to_numpy(float)
        fitted = {(k, m): make(m, json.load(open(T6 / creep / f'outer{k}' / f'{m}_best.json'))['params'], C.SEED)
                  .fit(X[(d['fold'] != k).to_numpy()], y[(d['fold'] != k).to_numpy()])
                  for k in range(C.N_OUTER) for m in MODELS}
        for cid, g in d.groupby('curve'):
            k = int(g['fold'].iloc[0])
            grid = np.geomspace(max(g['dt'].min(), 0.01), g['dt'].max(), 60)
            Xg = np.repeat(g[C.FEATURES[creep]].to_numpy(float)[:1], len(grid), axis=0)
            Xg[:, C.FEATURES[creep].index('dt')] = grid
            for m in ALL:
                p = (b4tw_scc_advisor(curves.loc[cid], grid) if m == 'B4TW-SCC' else fitted[(k, m)].predict(Xg))
                drop = np.max(-(np.diff(p) / p[:-1]))                      # largest relative decrease
                rows.append(dict(creep=creep, curve=cid, model=m, max_drop_pct=100 * max(drop, 0.0),
                                 decreasing=bool(drop > 0.005)))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / 'monotonic_per_curve.csv', index=False)
    s = r.groupby(['creep', 'model']).agg(curves=('curve', 'size'), decreasing=('decreasing', 'sum'),
                                          median_max_drop_pct=('max_drop_pct', 'median'))
    s['share_%'] = (100 * s.decreasing / s.curves).round(1)
    s.to_csv(OUT / 'monotonic.csv')
    return f'== {C.LEVEL} level: test curves whose predicted J drops by > 0.5 % on a 60-point duration grid\n' + \
        s.round(2).to_string()


def smoke():
    """Seconds: one model, one fold, short grids; checks paths, data and the B4TW-SCC import."""
    base = OUT / 'smoke'
    short = {'max_depth': [3, 8], 'min_samples_leaf': [1, 5]}
    print(tune_job('basic', C.GROUP, C.SEED, 0, 'DT', base, grids=short))
    d, _ = outer_folds('basic', C.GROUP, C.SEED)
    ref = pd.read_csv(C.folds_path('basic')) if C.folds_path('basic').exists() else None
    if ref is not None:                                                    # same outer folds as the paper
        f = d.groupby('curve')['fold'].first()
        assert (f.reindex(ref['curve']).to_numpy() == ref['fold'].to_numpy()).all(), 'outer folds differ'
        print('outer folds of seed 2 match 02_folds/folds_basic.csv')
    return 'smoke ok'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['seeds', 'random', 'mono', 'smoke'])
    ap.add_argument('--seeds', default='1-10')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lo, hi = map(int, a.seeds.split('-')) if '-' in a.seeds else (int(a.seeds), int(a.seeds))
    t0 = time.time()
    txt = {'seeds': lambda: seeds(list(range(lo, hi + 1)), a.jobs), 'random': lambda: random_split(a.jobs),
           'mono': monotonic, 'smoke': smoke}[a.what]()
    txt += f'\n({time.time() - t0:.0f} s)\n'
    print(txt)
    if a.what != 'smoke':
        with open(OUT / 'summary.txt', 'a', encoding='utf-8') as fh:
            fh.write(txt + '\n')
