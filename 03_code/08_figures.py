"""08_figures.py   (03_code, step 6 of PLAN.md)
PURPOSE    Figures and Table 1 in the forms of the advisor's shrinkage paper:
           Fig. 2   database by creep type (a) and cement type (b)                 [shrinkage Fig. 2]
           Table 1  range of the main parameters of the curves used                [shrinkage Table 1]
           scatter  predicted vs measured, test curves, 6 panels, 1:1 and +/-40 %  [shrinkage Fig. 7/10]
           residual measured - predicted vs load duration, 6 panels              [shrinkage Fig. 8/11]
           curves   3 representative test curves, all models                    [shrinkage Fig. 9/12]
           Representative curves are chosen by rule, not by eye: among test curves with >= 6 readings, the curves
           at the 25th, 50th and 75th percentile of the per-curve RMSE of extra trees (best ML model in step 4).
READS      02_data/Creep Database SCC.xlsx, 04_results/04_out_of_fold/pred_<creep>.csv
WRITES     04_results/06_figures/*.png, table1_ranges.csv, representative_curves.csv
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

TP = "t'"

matplotlib.use('Agg')
OUT = C.RESULTS / '06_figures'
OUT.mkdir(parents=True, exist_ok=True)
ALL = ['B4TW-SCC', 'DT', 'RF', 'ET', 'GB', 'XGB']
NAMES = {'B4TW-SCC': 'B4TW-SCC', 'DT': 'Decision tree', 'RF': 'Random forest', 'ET': 'Extra trees',
         'GB': 'Gradient boosting', 'XGB': 'XGBoost'}
UNIT = r'($10^{-6}$/MPa)'


def metrics(y, p):
    q1, q3 = np.percentile(y, [25, 75])
    rmse = np.sqrt(np.mean((p - y) ** 2))
    return dict(R2=1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), NRMSE=100 * rmse / (q3 - q1),
                RMSE=rmse, MAE=np.mean(np.abs(p - y)))


# ---------------------------------------------------------------- Fig. 2: database
info = pd.read_excel(C.DATA / 'Creep Database SCC.xlsx', 'Creep Info')
plt.rcParams['font.family'] = 'Times New Roman'
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, (col, order, labels, title) in zip(axs, [
        ('type', ['total', 'basic'], ['Total creep', 'Basic creep'], 'Creep types'),
        ('Cem_type', [1, 2, 3], ['R type', 'SL type', 'RS type'], 'Cement types')]):
    n = info[col].value_counts().reindex(order).to_numpy()
    for k, (v, lab, c) in enumerate(zip(n, labels, ['#0070C0', '#FF0000', '#00B050'])):
        ax.bar(k, v, width=0.5, color=c, label=lab)
        ax.text(k, v + 8, str(v), ha='center', fontsize=10)
    ax.set_xticks([])
    ax.set_xlabel(title, fontweight='bold')
    ax.set_ylabel('Number of data sets')
    ax.set_ylim(0, 1.15 * n.max())
    ax.yaxis.grid(True, color='0.85')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=9)
axs[0].set_title('(a)', loc='left')
axs[1].set_title('(b)', loc='left')
fig.tight_layout()
fig.savefig(OUT / 'fig02_database.png', dpi=200)
plt.close(fig)
plt.rcParams['font.family'] = 'DejaVu Sans'

# ---------------------------------------------------------------- Table 1: ranges
rows = []
for creep in C.CREEP_TYPES:
    d, c = C.load(creep)
    c['ca'] = c['Coarse_agg']
    dd = d.groupby('curve')['dt'].agg(['min', 'max'])
    for col, lab in [('fc28', 'fcm28 (MPa)'), ('w/cm', 'w/cm'), ('cm', 'cm (kg/m3)'), ('a/cm', 'a/cm'),
                     ('ca', 'ca (kg/m3)'), ('V/S', 'V/S (mm)'), ('T', 'T (C)'), ('RH_test', 'RH'),
                     ("t'", "t' (days)"), ('t0', 't0 (days)')]:
        if creep == 'basic' and col in ('RH_test', 't0', 'V/S'):
            continue
        v = c[col]
        rows.append(dict(creep=creep, parameter=lab, min=v.min(), max=v.max(), mean=v.mean()))
    rows.append(dict(creep=creep, parameter="t - t' (days), last reading", min=dd['max'].min(), max=dd['max'].max(),
                     mean=dd['max'].mean()))
t1 = pd.DataFrame(rows)
t1.round(2).to_csv(OUT / 'table1_ranges.csv', index=False)

# ---------------------------------------------------------------- scatter, residual, curves
rep_rows = []
for creep in C.CREEP_TYPES:
    p = pd.read_csv(C.RESULTS / '04_out_of_fold' / f'pred_{creep}.csv')
    y = p['J'].to_numpy()
    lim = 1.05 * max(y.max(), p[ALL].to_numpy().max())

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for ax, m in zip(axs.flat, ALL):
        s = metrics(y, p[m].to_numpy())
        ax.scatter(y, p[m], s=8, color='blue', alpha=0.7)
        x = np.array([0, lim])
        ax.plot(x, x, 'k-', lw=1)
        ax.plot(x, x / 0.6, 'r--', lw=1)
        ax.plot(x, 0.6 * x, 'r--', lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.set_title(f'{NAMES[m]} (test curves)')
        ax.set_xlabel(f'Measured {UNIT}')
        ax.set_ylabel(f'Predicted {UNIT}')
        ax.text(0.97, 0.03, f"Data sets: {p.curve.nunique()}\nData points: {len(p)}\n$R^2$: {s['R2']:.2f}\n"
                f"NRMSE: {s['NRMSE']:.1f}%\nRMSE: {s['RMSE']:.1f}\nMAE: {s['MAE']:.1f}", transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, bbox=dict(facecolor='white', edgecolor='0.7'))
    fig.tight_layout()
    fig.savefig(OUT / f'fig_scatter_{creep}.png', dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    ymax = np.abs(y[:, None] - p[ALL].to_numpy()).max() * 1.05
    for ax, m in zip(axs.flat, ALL):
        ax.scatter(p['dt'], y - p[m], s=8, color='blue', alpha=0.7)
        ax.axhline(0, color='k', lw=1)
        ax.set_xscale('log')
        ax.set_ylim(-ymax, ymax)
        ax.set_title(f'{NAMES[m]} (test curves)')
        ax.set_xlabel(r"$t-t'$ (days)")
        ax.set_ylabel(f'Residual error {UNIT}')
    fig.tight_layout()
    fig.savefig(OUT / f'fig_residual_{creep}.png', dpi=200)
    plt.close(fig)

    per = p.groupby('curve').apply(lambda g: pd.Series(dict(n=len(g), rmse_ET=np.sqrt(np.mean((g.ET - g.J) ** 2)))),
                                   include_groups=False)
    per = per[per.n >= 6].sort_values('rmse_ET')
    picks = [per.index[int(round(q * (len(per) - 1)))] for q in (0.25, 0.5, 0.75)]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for k, (ax, cid) in enumerate(zip(axs, picks)):
        g = p[p.curve == cid].sort_values('dt')
        ax.plot(g.dt, g.J, 'ko', ms=6, label='Measured', zorder=5)
        for m in ALL:
            ax.plot(g.dt, g[m], '-', lw=1.4, label=NAMES[m])
        ax.set_xscale('log')
        cin = C.load(creep)[0].query('curve == @cid').iloc[0]      # panel title: mix and loading age, no curve code
        ax.set_title(f'({"abc"[k]}) w/cm = {cin["w/cm"]:.2f}, $f_{{cm28}}$ = {cin["fc28"]:.0f} MPa, '
                     f"$t'$ = {cin.loc[TP]:.0f} days", fontsize=10)
        ax.set_xlabel(r"$t-t'$ (days)")
        rep_rows.append(dict(creep=creep, panel='abc'[k], curve=cid, readings=len(g),
                             percentile=(25, 50, 75)[k], rmse_ET=per.loc[cid, 'rmse_ET']))
    axs[0].set_ylabel(f'Creep compliance J {UNIT}')
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f'fig_curves_{creep}.png', dpi=200)
    plt.close(fig)

pd.DataFrame(rep_rows).round(2).to_csv(OUT / 'representative_curves.csv', index=False)
print(t1.round(2).to_string(index=False))
print(pd.DataFrame(rep_rows).round(2).to_string(index=False))
