# TODO / Technical notes to fix later

## 1. Redundant time features (t0, dt_log1p, age_days, age_log1p)

**Found while**: reviewing the MLP_Tapered feature-importance chart (top 20).

**Issue**: the four features `t0`, `dt_log1p`, `age_days`, `age_log1p` are
all fed into the model at the same time, but they are not 4 independent
sources of information — they're just 2 underlying quantities (`t0` = the
concrete's age when shrinkage measurement started; `dt_days` = days elapsed
since then) re-encoded into 4 different forms:

```
age_days   = t0 + dt_days
age_log1p  = log1p(age_days) = log1p(t0 + dt_days)
dt_log1p   = log1p(dt_days)
```

Since `dt_days = expm1(dt_log1p)`, knowing just `t0` and `dt_log1p` is
enough to derive `age_days` and `age_log1p` — no new information there.

(Raw `dt_days` is already excluded from the feature set via
`EXCLUDE_COLUMNS` in 01_annbasic.py:105, but the 4 features derived
from it/`t0` above are still kept in the feature set.)

**Observed impact**:
- Doesn't cause errors or wrong results with a nonlinear model (MLP), but
  it dilutes permutation feature importance: the time signal gets "shared"
  across 4 features instead of concentrated in one, making the importance
  chart harder to read and potentially understating how important the
  time-related feature cluster really is.
- For any linear/regression models in the pipeline, this could be
  significant multicollinearity.

**Proposed fix (not yet done, needs a scope review before changing)**:
- Reduce to 1-2 representative features, e.g. keep only `t0` + `dt_log1p`
  (or just `age_log1p` alone if absolute age matters more than relative
  age), and drop the remaining derived features from
  `EXCLUDE_COLUMNS`/the feature set.
- This same feature-engineering block is duplicated across **all files
  that build these features** (at least 01, 05, 06, 07, 09 — see
  `out['dt_log1p'] = ...`, `out['age_days'] = ...`,
  `out['age_log1p'] = ...` in each file), so any fix needs to be applied
  consistently across all of them, not just one file.
- Re-run the full k-fold pipeline after the change to compare
  R2/NRMSE before vs. after — don't change blindly without verification.
