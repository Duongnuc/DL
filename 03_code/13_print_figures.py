"""13_print_figures.py   (03_code, after steps 3-6 of PLAN.md)
PURPOSE    Figures 2-11 of the paper redrawn at print size (owner 2026-11-13, variant B: "Duyệt theo đề xuất").
           Same data and chart forms as 04_tuning_outputs.py, 07_shap.py and 08_figures.py (advisor's shrinkage
           paper forms); only the drawing changes: drawn at the printed size (12 cm wide, curves 15.5 cm), Times New
           Roman 8 pt, nothing below 7 pt on paper, axis titles on outer panels only, panel title = model name,
           legend once, parity metrics in the empty corner with the points drawn over them, B4TW-SCC grey dashed
           in the curve panels. Parity dashed lines as in the advisor's code: predicted = 0.6 and 1/0.6 x measured.
           Cold-read by two fresh readers on 2026-11-13 (_control/language_check_2026-11-12/BLIND_READ_2026-11-13.md).
READS      04_results/level_<LEVEL>/03_tuning, 05_shap, 04_out_of_fold, 06_figures/representative_curves.csv
WRITES     the same PNG files the earlier scripts wrote (overwrites them): 05_shap/shap_<creep>.png,
           03_tuning/max_depth_<creep>.png, 06_figures/fig_{scatter,residual,curves}_<creep>.png
"""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

matplotlib.use('Agg')
CM = 1 / 2.54
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'mathtext.fontset': 'stix'})
R = C.RESULTS
ML = ['DT', 'RF', 'ET', 'GB', 'XGB']
ALL = ['B4TW-SCC'] + ML
NAMES = {'B4TW-SCC': 'B4TW-SCC', 'DT': 'Decision tree', 'RF': 'Random forest', 'ET': 'Extra trees',
         'GB': 'Gradient boosting', 'XGB': 'XGBoost'}
FCM = r'$\mathregular{f_{cm28}}$'                     # upright, as in the text of the paper
TEX = {'w/cm': 'w/cm', 'cm': 'cm', 'fc28': FCM, 'T': 'T', "t'": 't′', 'dt': 't − t′', 'V/S': 'V/S',
       'RH_test': 'RH', 't0': r'$\mathregular{t_0}$'}
UNIT = r'($10^{-6}$/MPa)'
COL = dict(zip(ML, ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']))   # colours of the earlier figures
TP = "t'"


def save(fig, path):
    fig.savefig(path, dpi=300)
    plt.close(fig)


def outer_labels(axs, xl, yl):
    for ax in axs.flat:
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel(xl)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(yl)


def metrics(y, p):
    q1, q3 = np.percentile(y, [25, 75])
    rmse = np.sqrt(np.mean((p - y) ** 2))
    return dict(R2=1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), NRMSE=100 * rmse / (q3 - q1),
                RMSE=rmse, MAE=np.mean(np.abs(p - y)))


for creep in C.CREEP_TYPES:
    # SHAP importance (Figs 2, 4); five panels, last-row x labels plus the panel above the empty slot
    imp = pd.read_csv(R / '05_shap' / f'shap_{creep}.csv', index_col=0)
    fig, axs = plt.subplots(2, 3, figsize=(12 * CM, 7 * CM), sharex=True)
    for ax, m in zip(axs.flat, ML):
        s = imp[m].sort_values()
        ax.barh([TEX[f] for f in s.index], s.values, color='#1f77b4', height=0.7)
        ax.set_title(NAMES[m])
        ax.set_xlim(0, 105)
    axs.flat[-1].axis('off')
    for ax in (axs[1, 0], axs[1, 1], axs[0, 2]):
        ax.set_xlabel('Relative importance (%)')
        ax.tick_params(labelbottom=True)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)
    save(fig, R / '05_shap' / f'shap_{creep}.png')

    # NRMSE against max_depth (Figs 3, 5)
    fig, axs = plt.subplots(2, 3, figsize=(12 * CM, 7 * CM))
    for ax, m in zip(axs.flat, ML):
        t = pd.read_csv(R / '03_tuning' / creep / 'all' / f'{m}_trace.csv')
        t = t[(t['pass'] == 2) & (t['param'] == 'max_depth')].dropna(subset=['value']).copy()   # None read as NaN
        t['value'] = t['value'].astype(float).astype(int)
        ax.plot(t.value, t.train_nrmse, 'o-', ms=2.5, lw=0.9, label='Training (CV folds)')
        ax.plot(t.value, t.val_nrmse, 's-', ms=2.5, lw=0.9, label='Validation (CV folds)')
        sel = json.load(open(R / '03_tuning' / creep / 'all' / f'{m}_best.json'))['params']['max_depth']
        if sel is not None:
            ax.axvline(sel, color='red', ls='--', lw=0.9, label='Selected')
        ax.set_title(NAMES[m])
    axs.flat[-1].axis('off')
    h, lab = axs.flat[0].get_legend_handles_labels()
    axs.flat[-1].legend(h, lab, loc='center', frameon=False, fontsize=8)
    for ax in (axs[1, 0], axs[1, 1], axs[0, 2]):
        ax.set_xlabel('max_depth')
    for ax in (axs[0, 0], axs[1, 0]):
        ax.set_ylabel('NRMSE (%)')
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)
    save(fig, R / '03_tuning' / f'max_depth_{creep}.png')

    # predicted against measured (Figs 6, 9)
    p = pd.read_csv(R / '04_out_of_fold' / f'pred_{creep}.csv')
    y = p['J'].to_numpy()
    lim = 1.05 * max(y.max(), p[ALL].to_numpy().max())
    fig, axs = plt.subplots(2, 3, figsize=(12 * CM, 8 * CM), sharex=True, sharey=True)
    for ax, m in zip(axs.flat, ALL):
        s = metrics(y, p[m].to_numpy())
        ax.scatter(y, p[m], s=2, color='blue', alpha=0.6, linewidths=0, zorder=4)
        x = np.array([0, lim])
        ax.plot(x, x, 'k-', lw=0.7, zorder=2)
        ax.plot(x, x / 0.6, 'r--', lw=0.7, zorder=2)
        ax.plot(x, 0.6 * x, 'r--', lw=0.7, zorder=2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.set_title(NAMES[m])
        ax.text(0.04, 0.97, f"$R^2$: {s['R2']:.2f}\nNRMSE: {s['NRMSE']:.1f}%\nRMSE: {s['RMSE']:.1f}\nMAE: {s['MAE']:.1f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=7, linespacing=1.1, zorder=3,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.8))
    outer_labels(axs, f'Measured {UNIT}', f'Predicted {UNIT}')
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    save(fig, R / '06_figures' / f'fig_scatter_{creep}.png')

    # residual against load duration (Figs 7, 10)
    fig, axs = plt.subplots(2, 3, figsize=(12 * CM, 7 * CM), sharey=True, sharex=True)
    ymax = np.abs(y[:, None] - p[ALL].to_numpy()).max() * 1.05
    for ax, m in zip(axs.flat, ALL):
        ax.scatter(p['dt'], y - p[m], s=2, color='blue', alpha=0.6, linewidths=0)
        ax.axhline(0, color='k', lw=0.7)
        ax.set_xscale('log')
        ax.set_ylim(-ymax, ymax)
        ax.set_title(NAMES[m])
    outer_labels(axs, 't − t′ (days)', f'Residual {UNIT}')
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    save(fig, R / '06_figures' / f'fig_residual_{creep}.png')

    # representative test curves (Figs 8, 11): the curves chosen by 08_figures.py
    rep = pd.read_csv(R / '06_figures' / 'representative_curves.csv').query('creep == @creep')
    d = C.load(creep)[0]
    fig, axs = plt.subplots(1, 3, figsize=(15.5 * CM, 4.9 * CM))
    for k, (ax, cid) in enumerate(zip(axs, rep.curve)):
        g = p[p.curve == cid].sort_values('dt')
        ax.plot(g.dt, g.J, 'ko', ms=3, label='Measured', zorder=5)
        ax.plot(g.dt, g['B4TW-SCC'], '--', color='0.25', lw=1.0, label='B4TW-SCC')
        for m in ML:
            ax.plot(g.dt, g[m], '-', lw=0.9, label=NAMES[m], color=COL[m])
        ax.set_xscale('log')
        cin = d.query('curve == @cid').iloc[0]
        ax.set_title(f'({"abc"[k]}) w/cm = {cin["w/cm"]:.2f}, {FCM} = {cin["fc28"]:.0f} MPa, '
                     f't′ = {cin.loc[TP]:.0f} days', fontsize=7)
        ax.set_xlabel('t − t′ (days)')
    axs[0].set_ylabel(f'Creep compliance J {UNIT}')
    hd, lb = axs[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc='lower center', ncol=7, frameon=False, fontsize=7, columnspacing=1.0, handlelength=1.6,
               handletextpad=0.4)
    fig.tight_layout(pad=0.3, w_pad=0.5, rect=(0, 0.11, 1, 1))
    save(fig, R / '06_figures' / f'fig_curves_{creep}.png')
print('print-size figures written to', R)
