"""04_tuning_outputs.py   (03_code, step 3 of PLAN.md, outputs)
PURPOSE    Parameter table and max_depth figure from the tuning on all curves, in the forms of the shrinkage paper
           (Fig. 4/6: NRMSE of training and validation CV folds against max_depth, selected value dashed red;
           Tables 2/3: selected hyperparameters per model). Also the spread of the selected values over the 5 outer
           training parts (defence note, not for the paper).
READS      04_results/03_tuning/<creep>/<set>/<model>_trace.csv, _best.json
WRITES     04_results/03_tuning/params_<creep>.csv, params_outer_<creep>.csv, max_depth_<creep>.png, summary.txt
"""
import json

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import common as C

matplotlib.use('Agg')
IN = C.RESULTS / '03_tuning'
MODELS = ['DT', 'RF', 'ET', 'GB', 'XGB']
NAMES = {'DT': 'Decision tree', 'RF': 'Random forest', 'ET': 'Extra trees', 'GB': 'Gradient boosting',
         'XGB': 'XGBoost'}
ROWS = ['n_estimators', 'max_features', 'max_depth', 'max_leaf_nodes', 'min_samples_split', 'min_samples_leaf',
        'learning_rate', 'subsample', 'colsample_bytree', 'min_child_weight']


def best(creep, s, m):
    return json.load(open(IN / creep / s / f'{m}_best.json'))


lines = []
for creep in C.CREEP_TYPES:
    tab = pd.DataFrame(index=ROWS + ['val NRMSE default (%)', 'val NRMSE tuned (%)'], columns=[NAMES[m] for m in MODELS])
    for m in MODELS:
        b = best(creep, 'all', m)
        for k, v in b['params'].items():
            tab.loc[k, NAMES[m]] = 'None' if v is None else v
        tab.loc['val NRMSE default (%)', NAMES[m]] = round(b['default_val_nrmse'], 1)
        tab.loc['val NRMSE tuned (%)', NAMES[m]] = round(b['val_nrmse'], 1)
    tab = tab.fillna('-')
    tab.to_csv(IN / f'params_{creep}.csv')
    lines += [f'== {creep}: selected on all curves (grouped 5-fold)', tab.to_string(), '']

    outer = []
    for m in MODELS:
        for k in range(C.N_OUTER):
            b = best(creep, f'outer{k}', m)
            outer.append(dict(model=m, fold=k, **{p: b['params'].get(p) for p in ROWS}, val_nrmse=round(b['val_nrmse'], 1)))
    o = pd.DataFrame(outer)
    o.to_csv(IN / f'params_outer_{creep}.csv', index=False)
    lines += [f'-- {creep}: selected in each outer training part', o.to_string(index=False), '']

    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for ax, m in zip(axs.flat, MODELS):
        t = pd.read_csv(IN / creep / 'all' / f'{m}_trace.csv')
        t = t[(t['pass'] == 2) & (t['param'] == 'max_depth')].dropna(subset=['value']).copy()   # None read as NaN
        t['value'] = t['value'].astype(float).astype(int)
        ax.plot(t.value, t.train_nrmse, 'o-', ms=4, label='Training (CV folds)')
        ax.plot(t.value, t.val_nrmse, 's-', ms=4, label='Validation (CV folds)')
        sel = best(creep, 'all', m)['params']['max_depth']
        if sel is not None:
            ax.axvline(sel, color='red', ls='--', lw=1.2, label='selected')
        ax.set_title(NAMES[m])
        ax.set_xlabel('max_depth')
        ax.set_ylabel('NRMSE (%)')
        ax.legend(fontsize=8)
    axs.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(IN / f'max_depth_{creep}.png', dpi=200)
    plt.close(fig)

txt = '\n'.join(lines)
(IN / 'summary.txt').write_text(txt, encoding='utf-8')
print(txt)
