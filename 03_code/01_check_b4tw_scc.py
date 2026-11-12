"""01_check_b4tw_scc.py   (03_code, step 1 of PLAN.md)
PURPOSE    Computes B4TW-SCC with the advisor's code (b4tw_scc_advisor.py) for every reading of the cleaned data and
           compares it, curve by curve, with the V1 function empirical_models.b4tw_scc (read-only V1 package).
READS      02_data/cleaned_curves.csv, 02_data/cleaned_readings.csv
WRITES     04_results/01_b4tw_check/per_reading.csv, per_curve.csv, summary.txt
DONE WHEN  every curve differs by < 0.1 %, or each larger difference has a named cause.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
V1_SETTINGS = pathlib.Path(r'E:\05deep L\02_Creep_SCC_ML\creep_SCC_JOMC_2026-11-10\02_code_and_data'
                           r'\creep_scc_code\2_code\0_settings')
sys.path[:0] = [str(HERE), str(V1_SETTINGS)]
from b4tw_scc_advisor import b4tw_scc_advisor   # noqa: E402
from empirical_models import b4tw_scc, admixture_class, scores   # noqa: E402

OUT = ROOT / '04_results' / '01_b4tw_check'
OUT.mkdir(parents=True, exist_ok=True)

curves = pd.read_csv(ROOT / '02_data' / 'cleaned_curves.csv').set_index('curve')
readings = pd.read_csv(ROOT / '02_data' / 'cleaned_readings.csv')


def advisor_class(r):
    """Admixture class as the advisor's Table4C_cal reads it."""
    sl, fa, sf = r['Slag'] > 0, r['FlyAsh'] > 0, r['SF'] > 0
    if sl and not fa and not sf:
        return 'slag'
    if fa and not sl and not sf:
        return 'fly ash'
    if sl and fa and not sf:
        return 'slag+fly ash'
    if sf and not sl and not fa:
        return 'silica fume'
    return 'none'


rows, per_curve = [], []
for cid, g in readings.groupby('curve', sort=False):
    r = curves.loc[cid]
    info = {}
    ja = b4tw_scc_advisor(r, g['dt'].to_numpy(), info)
    jv = b4tw_scc(r, g['dt'].to_numpy())
    d = 100 * (ja - jv) / jv
    rows.append(pd.DataFrame(dict(curve=cid, type=r['type'], dt=g['dt'].to_numpy(), J_measured=g['J'].to_numpy(),
                                  J_advisor=ja, J_V1=jv, diff_pct=d)))
    causes = []
    if np.isfinite(r.get('Unit_weight', np.nan)) and abs(r['Unit_weight'] - 2350) > 1:
        causes.append(f"V1 uses unit weight {r['Unit_weight']:.0f} instead of 2350")
    if advisor_class(r) != admixture_class(r):
        causes.append(f"admixture class: advisor '{advisor_class(r)}', V1 '{admixture_class(r)}'")
    if info['negative_Cd_points']:
        causes.append(f"{info['negative_Cd_points']} points with negative drying-creep term")
    per_curve.append(dict(curve=cid, type=r['type'], n=len(g), max_abs_diff_pct=np.nanmax(np.abs(d)),
                          cause='; '.join(causes) or ('6/7 vs 0.85 in E(t)' if np.nanmax(np.abs(d)) < 0.1 else 'UNEXPLAINED')))

pr = pd.concat(rows, ignore_index=True)
pc = pd.DataFrame(per_curve)
pr.to_csv(OUT / 'per_reading.csv', index=False)
pc.to_csv(OUT / 'per_curve.csv', index=False)

lines = []
for typ, g in pc.groupby('type'):
    big = g[g.max_abs_diff_pct >= 0.1]
    lines.append(f'{typ}: {len(g)} curves, {g.n.sum()} readings; max |diff| {g.max_abs_diff_pct.max():.3f} %; '
                 f'curves >= 0.1 %: {len(big)}')
    for cause, gg in big.groupby('cause'):
        lines.append(f'   {len(gg):3d} curves (max {gg.max_abs_diff_pct.max():.2f} %): {cause}')
    x = pr[pr.type == typ]
    for col in ['J_advisor', 'J_V1']:
        s = scores(x.J_measured, x[col])
        lines.append(f'   {col:9s} vs measured: R2 {s["R2"]:.3f}, NRMSE(IQR) {s["NRMSE_IQR"]:.1f} %, RMSE {s["RMSE"]:.2f}')
lines.append(f'NaN predictions: {int(pr.J_advisor.isna().sum())}')
txt = '\n'.join(lines)
(OUT / 'summary.txt').write_text(txt, encoding='utf-8')
print(txt)
