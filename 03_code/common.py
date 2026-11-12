"""common.py   (03_code, shared module)
PURPOSE    Paths, inputs, seed and the data loader shared by the V2 scripts (decisions in ../PLAN.md).
"""
import os
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / '02_data'
# Test level (owner 2026-11-12, option 2): 'curve' = mixes already tested (split by curve, as the advisor asked and
# as in the shrinkage paper; main level); 'mix' = new mixes (all curves of a mix on one side; extra table).
LEVEL = os.environ.get('CREEP_LEVEL', 'curve')
assert LEVEL in ('curve', 'mix')
GROUP = LEVEL                # column used as the group in every split
RESULTS = ROOT / '04_results' / f'level_{LEVEL}'

SEED = 2                 # random_state, as in the shrinkage paper
N_OUTER = 5              # outer grouped folds: every curve is a test curve once
N_INNER = 5              # inner grouped folds for tuning
CREEP_TYPES = ['basic', 'total']

# Inputs (card 3b). Column names of cleaned_curves.csv; 'dt' = t - t' (load duration, days).
FEATURES = {
    'basic': ['w/cm', 'cm', 'fc28', 'T', "t'", 'dt'],
    'total': ['w/cm', 'cm', 'fc28', 'T', "t'", 'dt', 'V/S', 'RH_test', 't0'],
}
LABELS = {'w/cm': 'w/cm', 'cm': 'cm', 'fc28': 'fcm28', 'T': 'T', "t'": "t'", 'dt': "t - t'",
          'V/S': 'V/S', 'RH_test': 'RH', 't0': 't0'}
MIX_COLS = ['Ref', 'w/cm', 'cm', 'fc28', 'a/cm']   # curves of one mix differ only in loading age
FC_RANGE = (17.0, 110.0)   # MPa; range of the advisor's B4TW-SCC scripts and of the shrinkage paper (owner 2026-11-12:
                           # drops 4 basic + 4 total curves of one Persson mix at 124/141 MPa)


def mix_id(curves):
    """Mix label per curve: same reference and mix proportions."""
    return curves[MIX_COLS].round(3).astype(str).agg('|'.join, axis=1)


def load(creep):
    """Readings of one creep type joined with their curve inputs; columns: curve, mix, inputs, J."""
    c = pd.read_csv(DATA / 'cleaned_curves.csv')
    c = c[(c['type'] == creep) & c['fc28'].between(*FC_RANGE)].copy()
    c['mix'] = mix_id(c)
    r = pd.read_csv(DATA / 'cleaned_readings.csv')
    cols = ['curve', 'mix'] + [f for f in FEATURES[creep] if f != 'dt']
    d = r.merge(c[cols], on='curve', how='inner')
    return d[['curve', 'mix'] + FEATURES[creep] + ['J']].reset_index(drop=True), c.set_index('curve')


def folds_path(creep):
    return RESULTS / '02_folds' / f'folds_{creep}.csv'


def load_folds(creep):
    """Readings with their outer fold number (0-4)."""
    d, c = load(creep)
    f = pd.read_csv(folds_path(creep))
    d = d.merge(f[['curve', 'fold']], on='curve', how='left')
    assert d['fold'].notna().all()
    return d, c
