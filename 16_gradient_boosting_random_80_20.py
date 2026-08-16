"""Teacher check 1/2: sklearn GradientBoostingRegressor with RANDOM 80:20 row split.

Purpose
-------
Re-run Gradient Boosting with the exact model hyperparameters supplied by the
supervisor, using an 80:20 random split by measurement ROW.  This is an explicit
sanity-check experiment; it is intentionally different from the paper's main
GID-aware evaluation and may place points from the same shrinkage curve in both
train and test.

IMPORTANT
---------
- This file NEVER reads ``data