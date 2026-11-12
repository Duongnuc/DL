"""b4tw_scc_advisor.py   (03_code, shared module)
PURPOSE    Creep compliance J (1e-6/MPa) of model B4TW-SCC, ported line by line from the advisor's scripts
           'B4TW-SCC total creep 4734.py' and 'B4TW-SCC basic creep 4734.py' (02_data/advisor_B4TW-SCC_code_2026-11-10).
           Only the input changed: one row of cleaned_curves.csv instead of a row of the SQL table SCC_Creep_Info
           (SQL column k = CSV column k-1; names checked: w/cm, a/cm, cm, SF, FlyAsh, Slag, NP, Filler, fc28, V/S,
           t', t0, Tcur, T, RH_test, Curing, Shape_factor, Cem_type, agg_type, region_type, Coarse_agg).
           The goodness-of-fit and plotting parts of the scripts are not ported.
USAGE      J = b4tw_scc_advisor(row, dt)   # dt = t - t' (days); total or basic from row['type']
"""
import math

import numpy as np


# ---------------------------------------------------------------- tables, as in the advisor's scripts
def ktaua_cal(aggtype, basic):
    v = {1: 0.06, 2: 0.59, 3: 1.8, 4: 2.8 if basic else 2.7, 5: 4.0}
    return v[aggtype]


def kea_cal(aggtype, basic):
    v = {1: 0.76, 2: 0.71, 3: 0.95, 4: 1.7 if basic else 2.0, 5: 1.05}
    return v[aggtype]


def Kh_cal(h):
    if h <= 0.98:
        return 1 - h ** 3
    elif h == 1:
        return -0.2
    elif 0.98 < h < 1:
        return 12.94 * (1 - h) - 0.2
    raise ValueError('Humidity error (must be 0~1)')


def Ks_cal(geo):
    return {1: 1, 2: 1.15, 3: 1.25, 4: 1.3, 5: 1.55}[geo]


def Table1_cal(cement):   # coarse-aggregate version
    return {1: [0.016, -0.30, -0.04, -0.05, 440e-6, -0.6, 1.10, 0.07, 0.35],
            2: [0.010, -0.30, 4.50, 2.50, 400e-6, -0.6, 3.20, 0.07, 0.80],
            3: [0.060, -0.30, -2.60, -2.00, 860e-6, -2.6, 0.40, 0.07, 0.46]}[cement]


def Table3_cal(cement):   # "4734" values
    return {1: [1.30, 50.0e-3, 40.0e-3, 4.0e-3, 700e-6, 7, 2.0, 1.1, 0.2, -0.8, 0.3, -0.7, 2.58, -0.76, -0.5, 4.00],
            2: [0.90, 30.0e-3, 40.0e-3, 3.4e-3, 300e-6, 7, 2.5, 1.1, 0.7, -0.8, 0.3, -0.7, 2.58, -0.86, -0.5, 1.00],
            3: [0.60, 17.4e-3, 39.9e-3, 3.4e-3, 94.6e-6, 1, 3.0, 1.0, 1.0, -1.1, 0.4, -0.9, 2.45, -0.85, -1.0, 0.78]}[cement]


def Table4S_cal(Slag, FlyAsh, SiO2, NP, Filler, wcm):
    if Slag == 0 and FlyAsh == 0 and SiO2 == 0 and NP == 0 and Filler > 0:
        return [1.20, 0.60, 1.00, 1.00]
    elif Slag > 0 and FlyAsh == 0 and SiO2 == 0 and NP == 0:
        return [1.30, 0.80, 0.85, 1.00]
    elif Slag == 0 and FlyAsh > 0 and SiO2 == 0 and NP == 0:
        return [1.05, 0.90, 1.15, 0.95]
    elif Slag > 0 and FlyAsh > 0 and SiO2 == 0 and NP == 0:
        return [1.00, 0.70, 1.10, 1.00]
    elif Slag == 0 and FlyAsh == 0 and SiO2 > 0 and NP == 0:
        return [1.00, 0.90, 1.00, 1.00]
    elif Slag > 0 and FlyAsh == 0 and SiO2 > 0 and NP == 0:
        return [1.50, 1.00, 0.90, 1.10]
    elif Slag == 0 and FlyAsh > 0 and SiO2 > 0 and NP == 0:
        return [0.95, 0.65, 1.15, 0.85]
    elif Slag > 0 and FlyAsh > 0 and SiO2 > 0 and NP == 0:
        return [0.95, 0.60, 0.85, 1.15]
    elif NP > 0:
        return [0.95, 0.70, 0.90, 1.00]
    return [1.00, 1.00, 1.00, 1.00]


def Table4C_cal(Slag, FlyAsh, SiO2, wcm, rtype):
    if Slag > 0 and FlyAsh == 0 and SiO2 == 0:
        return [1.0, 1.0, 1.3, 0.9]
    elif Slag == 0 and FlyAsh > 0 and SiO2 == 0:
        return [1.0, 1.0, 1.4, 0.9]
    elif Slag > 0 and FlyAsh > 0 and SiO2 == 0:
        return [1.2, 1.0, 3.5, 1.2]
    elif Slag == 0 and FlyAsh == 0 and SiO2 > 0:
        return [1.0, 1.0, 1.4, 0.9]
    return [1.0, 1.0, 1.0, 1.0]


def Table5_cal(cement, curing):
    return {(1, 1): [1.0, 0.95], (1, 3): [4.0, 0.85], (2, 1): [1.0, 0.95], (2, 3): [4.0, 0.85],
            (3, 1): [0.7, 0.98], (3, 3): [2.3, 0.92]}[(cement, curing)]


def BT(T):
    return math.exp(4000 * (1 / 293. - 1. / (T + 273)))


# ---------------------------------------------------------------- one curve
def b4tw_scc_advisor(r, dt, info=None):
    """J (1e-6/MPa) at load durations dt for curve row r. info (dict) receives intermediate values if given."""
    basic = r['type'] == 'basic'
    wcm, acm, cm = float(r['w/cm']), float(r['a/cm']), float(r['cm'])
    SiO2, FlyAsh, Slag = float(r['SF']), float(r['FlyAsh']), float(r['Slag'])
    NP, Filler = float(r['NP']), float(r['Filler'])
    fc = float(r['fc28'])
    D = float(r['V/S']) * 2
    tp, t0, Tcur, T = float(r["t'"]), float(r['t0']), float(r['Tcur']), float(r['T'])
    h = float(r['RH_test'])
    curing = int(r['Curing'])
    if curing == 2:
        curing = 3
    geo, cement, aggtype = int(r['Shape_factor']), int(r['Cem_type']), int(r['agg_type'])
    rtype = float(r['region_type'])
    ca = float(r['Coarse_agg'])

    ktaua, kea = ktaua_cal(aggtype, basic), kea_cal(aggtype, basic)
    Kh, Ks = Kh_cal(h), Ks_cal(geo)
    BTh, BTs, BTc = BT(Tcur), BT(T), BT(T)

    t = np.asarray(dt, float) + tp
    t0wave = t0 * BTh
    tphat = t0 * BTh + (tp - t0) * BTs

    taucem, ptaua, ptauw, ptauc, ecem, pea, pew, pec, peca = Table1_cal(cement)
    (p1, p2, p3, p4, p5, p5H, p2w, p2c, p2ca, p3a, p3w, p4a, p4w, p5e, p5a, p5w) = Table3_cal(cement)
    Table4S = Table4S_cal(Slag, FlyAsh, SiO2, NP, wcm, rtype)   # argument order as in the advisor's call
    fshcem = Table4S[2]                                          # (Filler <- wcm); fshcem/fshtau do not enter J
    a, b = Table5_cal(cement, curing)
    fp2, fp3, fp4, fp5 = Table4C_cal(Slag, FlyAsh, SiO2, wcm, rtype)

    tau0 = taucem * ((acm / 3.4) ** ptaua) * ((wcm / 0.36) ** ptauw) * ((4.6 * cm / 2350) ** ptauc)
    taush = tau0 * ktaua * ((Ks * D) ** 2)
    e0 = ecem * ((acm / 3.4) ** pea) * ((wcm / 0.36) ** pew) * ((4.6 * cm / 2350) ** pec) * ((ca / 840) ** peca)

    fcmtp = (tp / (a + b * tp)) * fc
    Ecmtp = (fcmtp ** 0.5) * 4734
    E28 = (fc ** 0.5) * 4734

    def E_cal(tt):
        return E28 * (tt / (4 + (6. / 7) * tt)) ** 0.5

    esh8 = -1 * e0 * kea * E_cal(7 * BTh + 600 * BTs) / E_cal(t0wave + taush * BTs)

    that = tphat + (t - tp) * BTc
    q1 = p1 / Ecmtp
    q2 = fp2 * p2 * ((wcm / 0.36) ** p2w) * ((4.6 * cm / 2350) ** p2c) * ((ca / 840) ** p2ca) / 1000
    q3 = fp3 * p3 * q2 * ((acm / 3.4) ** p3a) * ((wcm / 0.36) ** p3w)
    q4 = fp4 * p4 * ((acm / 3.4) ** p4a) * ((wcm / 0.36) ** p4w) / 1000
    r_ = 1.7 * (tphat ** 0.12) + 8
    Z = np.log(1 + (that - tphat) ** 0.1) * tphat ** -0.5
    Qf = 1. / (0.086 * tphat ** (2. / 9) + 1.21 * tphat ** (4. / 9))
    Q = Qf * (1 + (Qf / Z) ** r_) ** (-1 / r_)
    q5 = fp5 * p5 * ((acm / 3.4) ** p5a) * ((wcm / 0.36) ** p5w) * (math.fabs(Kh * esh8)) ** p5e / 1000
    C0 = (q2 * Q + q3 * np.log(1 + (that - tphat) ** 0.1) + q4 * np.log(that / tphat)) * 1e6

    J = q1 * 1e6 + C0
    neg = 0
    if not basic:
        tp0hat = max(tphat, t0wave)
        H = 1 - (1 - h) * np.tanh(((that - t0wave) / taush) ** 0.5)
        Hc = 1 - (1 - h) * math.tanh(((tp0hat - t0wave) / taush) ** 0.5)
        inner = np.exp(-p5H * H) - math.exp(-p5H * Hc)
        on = that >= tp0hat
        neg = int(np.sum(on & (inner < 0)))     # the advisor's script would stop here (complex number)
        Cd = np.where(on, q5 * np.sqrt(np.where(inner < 0, np.nan, inner)) * 1e6, 0.0)
        J = J + Cd
    if info is not None:
        info.update(q1=q1 * 1e6, q2=q2, q3=q3, q4=q4, q5=q5, taush=taush, esh8=esh8, Ecmtp=Ecmtp,
                    fp=(fp2, fp3, fp4, fp5), negative_Cd_points=neg)
    return J
