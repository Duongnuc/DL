"""09_fig01_process.py   (03_code, step 6 of PLAN.md)
PURPOSE    Fig. 1: nested grouped cross-validation. Layout of the shrinkage paper's Fig. 1 (data column ->
           algorithm ellipses -> evaluation), drawn compact (owner 2026-11-12: "Hình 1 gọn"): 16 cm wide,
           Times New Roman, muted colours. The fold column names the split unit (curve or mix, common.GROUP).
WRITES     04_results/level_<LEVEL>/06_figures/fig01_process.png
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

import common as C

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9})
INK, BOX, HI, ELL = '#333333', '#dfe7f1', '#f6d7a7', '#2f4f7f'
UNIT = {'curve': 'Creep curves', 'mix': 'Mixes'}[C.GROUP]

fig, ax = plt.subplots(figsize=(16 / 2.54, 5.6 / 2.54))
ax.set_xlim(0, 17)
ax.set_ylim(0, 5.8)
ax.axis('off')


def box(x, y, w, h, t, fc=BOX):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=INK, lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center')


def arr(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, color=INK, lw=0.9, shrinkA=0, shrinkB=0))


ax.text(1.9, 5.45, f'{UNIT} (5 folds)', ha='center', fontweight='bold')
for k in range(5):
    box(0.4, 4.4 - k * 0.9, 3.0, 0.68, 'test fold' if k == 4 else f'training fold {k + 1}', fc=HI if k == 4 else BOX)
ax.text(6.9, 5.45, 'Inner 5-fold', ha='center', fontweight='bold')
box(5.3, 3.85, 3.2, 0.75, 'inner training')
box(5.3, 2.85, 3.2, 0.75, 'inner validation')
arr((3.45, 3.6), (5.25, 3.6))
ax.add_patch(Ellipse((12.0, 3.7), 4.6, 1.3, fc=ELL, ec=INK, lw=0.8))
ax.text(12.0, 3.7, 'hyperparameters with the\nlowest validation NRMSE', ha='center', va='center', color='white',
        fontsize=8)
arr((8.55, 3.7), (9.68, 3.7))
ax.add_patch(Ellipse((12.0, 1.95), 4.6, 1.1, fc=ELL, ec=INK, lw=0.8))
ax.text(12.0, 1.95, 'refit on the four\ntraining folds', ha='center', va='center', color='white', fontsize=8)
arr((12.0, 3.04), (12.0, 2.52))
box(8.9, 0.1, 6.2, 0.75, 'prediction of the test curves')
arr((12.0, 1.39), (12.0, 0.87))
arr((3.45, 0.74), (8.85, 0.48))
ax.text(15.6, 2.9, 'repeated\nfor each\nfold and\nalgorithm', ha='center', va='center', style='italic', fontsize=8)
fig.savefig(C.RESULTS / '06_figures' / 'fig01_process.png', dpi=300, bbox_inches='tight')
