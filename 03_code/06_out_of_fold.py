"""06_out_of_fold.py   (03_code, step 4 of PLAN.md)
PURPOSE    For each outer fold k: refit each model with the parameters tuned on the training part (outer<k>) and
           predict the test part, so every reading gets one prediction from a model that never saw its mix.
           B4TW-SCC (advisor's code) is computed for every reading. Accuracy as in the shrinkage paper tables:
           training (mean over the 5 training parts) and test (pooled out-of-fold) R2, NRMSE, RMSE, MAE.
           Uncertainty of the test difference model - B4TW-SCC: bootstrap over mixes (defence note).
WRITES     04_results/04_out_of_fold/pred_<creep>.csv, accuracy_<creep>.csv, per_fold_<creep>.csv,
           bootstrap_vs_b4tw.csv, summary.txt
"""
import importlib
import json

import numpy as np
import pandas as pd

import common as C
from b4tw_scc_advisor import b4tw_scc_advisor

T = importlib.import_module('03_tune')
OUT = C.RESULTS / '04_out_of_fold'
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ['DT', 'RF', 'ET', 'GB', 'XGB']
ALL = ['B4TW-SCC'] + MODELS
N_BOOT = 2000


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    q1, q3 = np.percentile(y, [25, 75])
    rmse = np.sqrt(np.mean((p - y) ** 2))
    return dict(R2=1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), NRMSE=100 * rmse / (q3 - q1),
                RMSE=rmse, MAE=np.mean(np.abs(p - y)))


lines, boot_rows = [], []
for creep in C.CREEP_TYPES:
    d, curves = C.load_folds(creep)
    X, y = d[C.FEATURES[creep]].to_numpy(float), d['J'].to_numpy(float)
    pred = d[['curve', 'mix', 'fold', 'dt', 'J']].copy()
    pred['B4TW-SCC'] = np.nan
    for cid, idx in d.groupby('curve').indices.items():
        pred.loc[idx, 'B4TW-SCC'] = b4tw_scc_advisor(curves.loc[cid], d['dt'].to_numpy()[idx])
    train_rows, fold_rows = [], []
    for m in MODELS:
        pred[m] = np.nan
    for k in range(C.N_OUTER):
        te = (d['fold'] == k).to_numpy()
        tr = ~te
        assert not set(d.loc[tr, C.GROUP]) & set(d.loc[te, C.GROUP])
        for m in MODELS:
            prm = json.load(open(T.OUT / creep / f'outer{k}' / f'{m}_best.json'))['params']
            model = T.make(m, prm).fit(X[tr], y[tr])
            pred.loc[te, m] = model.predict(X[te])
            train_rows.append(dict(model=m, fold=k, **metrics(y[tr], model.predict(X[tr]))))
        train_rows.append(dict(model='B4TW-SCC', fold=k, **metrics(y[tr], pred.loc[tr, 'B4TW-SCC'])))
        for m in ALL:
            fold_rows.append(dict(model=m, fold=k, n_curves=int(d.loc[te, 'curve'].nunique()),
                                  **metrics(y[te], pred.loc[te, m])))
    assert pred[ALL].notna().all().all(), 'missing prediction'
    pred.to_csv(OUT / f'pred_{creep}.csv', index=False)

    tr_mean = pd.DataFrame(train_rows).groupby('model')[['R2', 'NRMSE']].mean()
    acc = pd.DataFrame([dict(model=m, train_R2=tr_mean.loc[m, 'R2'], train_NRMSE=tr_mean.loc[m, 'NRMSE'],
                             **{f'test_{k}': v for k, v in metrics(y, pred[m]).items()}) for m in ALL])
    acc.round(3).to_csv(OUT / f'accuracy_{creep}.csv', index=False)
    pf = pd.DataFrame(fold_rows)
    pf.round(3).to_csv(OUT / f'per_fold_{creep}.csv', index=False)
    rng_ = pf.groupby('model')['R2'].agg(['min', 'max'])

    # bootstrap over mixes: test RMSE of model minus that of B4TW-SCC
    mixes = pred['mix'].unique()
    groups = {mx: np.flatnonzero(pred['mix'].to_numpy() == mx) for mx in mixes}
    rng = np.random.default_rng(C.SEED)
    diffs = {m: [] for m in MODELS}
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[mx] for mx in rng.choice(mixes, len(mixes), replace=True)])
        yb = y[idx]
        rb = np.sqrt(np.mean((pred['B4TW-SCC'].to_numpy()[idx] - yb) ** 2))
        for m in MODELS:
            diffs[m].append(np.sqrt(np.mean((pred[m].to_numpy()[idx] - yb) ** 2)) - rb)
    for m in MODELS:
        lo, hi = np.percentile(diffs[m], [2.5, 97.5])
        boot_rows.append(dict(creep=creep, model=m, RMSE_diff_vs_B4TW=metrics(y, pred[m])['RMSE'] -
                              metrics(y, pred['B4TW-SCC'])['RMSE'], ci95_low=lo, ci95_high=hi,
                              share_ML_better=float(np.mean(np.array(diffs[m]) < 0))))

    # creep part: J minus its value at the first reading of the curve, for measured and predicted (first reading
    # excluded). Explains where the error comes from: level at loading vs growth with time (owner option A).
    q = pred.sort_values(['curve', 'dt']).copy()
    first = q.groupby('curve')[['J'] + ALL].transform('first')
    keep = q['dt'] > q.groupby('curve')['dt'].transform('min')
    cp = q.loc[keep, ['curve', 'mix', 'dt']].copy()
    for col in ['J'] + ALL:
        cp[col] = q.loc[keep, col] - first.loc[keep, col]
    cp.to_csv(OUT / f'pred_creep_part_{creep}.csv', index=False)
    yc = cp['J'].to_numpy()
    accp = pd.DataFrame([dict(model=m, **{f'creep_part_{k}': v for k, v in metrics(yc, cp[m]).items()}) for m in ALL])
    accp.round(3).to_csv(OUT / f'accuracy_creep_part_{creep}.csv', index=False)
    cmix = cp['mix'].unique()
    cgroups = {mx: np.flatnonzero(cp['mix'].to_numpy() == mx) for mx in cmix}
    cdiffs = {m: [] for m in MODELS}
    for _ in range(N_BOOT):
        idx = np.concatenate([cgroups[mx] for mx in rng.choice(cmix, len(cmix), replace=True)])
        rb = np.sqrt(np.mean((cp['B4TW-SCC'].to_numpy()[idx] - yc[idx]) ** 2))
        for m in MODELS:
            cdiffs[m].append(np.sqrt(np.mean((cp[m].to_numpy()[idx] - yc[idx]) ** 2)) - rb)
    for m in MODELS:
        lo, hi = np.percentile(cdiffs[m], [2.5, 97.5])
        boot_rows.append(dict(creep=creep + ' (creep part)', model=m,
                              RMSE_diff_vs_B4TW=metrics(yc, cp[m])['RMSE'] - metrics(yc, cp['B4TW-SCC'])['RMSE'],
                              ci95_low=lo, ci95_high=hi, share_ML_better=float(np.mean(np.array(cdiffs[m]) < 0))))

    a = acc.copy()
    a['test R2 range over folds'] = [f"{rng_.loc[m, 'min']:.2f}-{rng_.loc[m, 'max']:.2f}" for m in a.model]
    lines += [f'== {creep}: {len(d)} readings, {d.curve.nunique()} curves, {d.mix.nunique()} mixes',
              a.round(2).to_string(index=False), '-- creep part (J minus first reading)',
              accp.round(2).to_string(index=False), '']

b = pd.DataFrame(boot_rows)
b.round(3).to_csv(OUT / 'bootstrap_vs_b4tw.csv', index=False)
lines += ['== test RMSE (1e-6/MPa) of model minus B4TW-SCC; 95 % interval from 2000 bootstrap draws of mixes',
          b.round(2).to_string(index=False)]
txt = '\n'.join(lines)
(OUT / 'summary.txt').write_text(txt, encoding='utf-8')
print(txt)
