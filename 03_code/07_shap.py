"""07_shap.py   (03_code, step 5 of PLAN.md)
PURPOSE    Input importance as in the shrinkage paper (Fig. 3/5): mean |SHAP| per input on the test curves,
           normalised so that the most important input = 100. Exact TreeSHAP (TreeExplainer; Lundberg et al. 2020,
           Nat. Mach. Intell. 2:56) of each outer-fold model on its own test part, so every reading is explained by
           a model that never saw its mix. Models are refitted exactly as in 06_out_of_fold.py; their predictions
           are checked against the saved ones.
WRITES     04_results/05_shap/shap_<creep>.csv (relative importance), shap_<creep>.png, summary.txt
"""
import importlib
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

import common as C

matplotlib.use('Agg')
T = importlib.import_module('03_tune')
OUT = C.RESULTS / '05_shap'
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ['DT', 'RF', 'ET', 'GB', 'XGB']
NAMES = {'DT': 'Decision tree', 'RF': 'Random forest', 'ET': 'Extra trees', 'GB': 'Gradient boosting', 'XGB': 'XGBoost'}
TEX = {'w/cm': 'w/cm', 'cm': 'cm', 'fc28': r'$f_{cm28}$', 'T': 'T', "t'": r"$t'$", 'dt': r"$t-t'$", 'V/S': 'V/S',
       'RH_test': 'RH', 't0': r'$t_0$'}

lines, direction = [], []   # direction: Spearman of input vs its SHAP value (sign of the learned effect)
for creep in C.CREEP_TYPES:
    d, _ = C.load_folds(creep)
    feats = C.FEATURES[creep]
    X, y = d[feats].to_numpy(float), d['J'].to_numpy(float)
    saved = pd.read_csv(C.RESULTS / '04_out_of_fold' / f'pred_{creep}.csv')
    imp = {}
    for m in MODELS:
        S = np.zeros_like(X)
        p = np.zeros(len(y))
        for k in range(C.N_OUTER):
            te = (d['fold'] == k).to_numpy()
            prm = json.load(open(T.OUT / creep / f'outer{k}' / f'{m}_best.json'))['params']
            model = T.make(m, prm).fit(X[~te], y[~te])
            p[te] = model.predict(X[te])
            S[te] = shap.TreeExplainer(model).shap_values(X[te])
        assert np.allclose(p, saved[m].to_numpy(), atol=1e-6), f'{creep} {m}: refit differs from step 4'
        v = np.abs(S).mean(axis=0)
        imp[m] = 100 * v / v.max()
        direction.append(dict(creep=creep, model=m, **{f: round(spearmanr(X[:, i], S[:, i]).statistic, 2)
                                                       for i, f in enumerate(feats)}))
    tab = pd.DataFrame(imp, index=feats).round(1)
    tab.to_csv(OUT / f'shap_{creep}.csv')
    tab['mean rank'] = tab.rank(ascending=False).mean(axis=1).round(1)
    lines += [f'== {creep}: relative mean |SHAP| on test curves (max = 100)', tab.sort_values('mean rank').to_string(), '']

    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for ax, m in zip(axs.flat, MODELS):
        s = pd.Series(imp[m], index=feats).sort_values()
        ax.barh([TEX[f] for f in s.index], s.values, color='#1f77b4')
        ax.set_title(f'mean |SHAP|, test ({NAMES[m]})')
        ax.set_xlabel('Relative importance (%)')
        ax.set_xlim(0, 105)
    axs.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(OUT / f'shap_{creep}.png', dpi=200)
    plt.close(fig)

dr = pd.DataFrame(direction)
dr.to_csv(OUT / 'direction_check.csv', index=False)
lines += ['== direction check (Spearman, input vs its SHAP value)', dr.to_string(index=False)]
txt = '\n'.join(lines)
(OUT / 'summary.txt').write_text(txt, encoding='utf-8')
print(txt)
