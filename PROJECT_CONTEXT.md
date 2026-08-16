# Project context and change log

Last reviewed: 2026-08-15

## Purpose and structure

- This is a self-contained deep-learning pipeline for predicting concrete shrinkage (SCC).
- `01`–`10` train/evaluate individual models; `11` is empirical-formula baselines;
  `12` compares results; `13` provides ResMLP explainability; `14` creates report figures;
  `15` checks Optuna-budget sensitivity.
- Each Python file is designed to run independently from `data_raw/Database SCC.xlsx` and writes
  intermediate artifacts to `_cache/<model>/`; main paper figures go to `output_figures/`.
- Recommended execution order: `01`–`10`, then `11`, `12`, `13`–`14`, and finally `15`.
- There is no Git repository in this folder, so file timestamps and this log are the available
  local change history.

## Current data/experiment snapshot

- Canonical filtered sequence data: 788 valid GID curves and 6,741 measurement points;
  each curve has 5–22 points (median 8).
- Group-aware splitting is used so rows from a GID do not leak between train/validation/test.
- Existing cached completed runs are present for `annbasic` and `mlp_tapered`; no cached LSTM run
  was present when this note was created.
- Known open issue: duplicated time encodings (`t0`, `dt_log1p`, `age_days`, `age_log1p`) may
  dilute feature importance. Details and a proposed, unimplemented fix are in `TODO_NOTES.md`.

## LSTM / sequence-model run policy

- `07_gru.py`, `08_lstm.py`, and `09_tcn.py` intentionally share the K-fold policy:
  seed 42, 5 folds, 8 Optuna trials/fold, up to 80 epochs, early-stop patience 12,
  batch size 16.
- In `08_lstm.py`, those values apply only to K-fold. The holdout pass is explicitly lighter:
  3 trials, 60 epochs, patience 10, batch size 16.
- `15_optuna_sensitivity.py` reruns the top DL models with 30 trials/fold to check whether the
  conclusions remain stable when the tuning budget is increased.
- Technical assessment (2026-08-15): all studies call `optuna.create_study()` without an explicit
  sampler. Optuna's default TPE sampler uses 10 random startup trials; therefore the current
  8-trial-per-fold budget does not reach its adaptive TPE phase. It is a lightweight screening
  budget, not a robust hyperparameter-optimization budget. The Optuna sampler also has no explicit
  seed in this code, so `SEED=42` fixes the data/training randomness but does not fully reproduce
  the sequence of suggested hyperparameters.

## Literature audit for answering the supervisor's LSTM-parameter question

- `paper/A time-series deep learning model for predicting concrete shrinkage and.pdf` is the
  closest domain reference (Fan et al., *Construction and Building Materials*, 2024). It supports
  LSTM as a time-series baseline for concrete shrinkage/creep. Its own experiment uses a five-step
  input window, Adam, initial learning rate `1e-4`, batch size 100 and 100 epochs, with an 80:20
  train/validation split. These values are not transferable defaults because its sample unit and
  split protocol differ from this project.
- `paper/LSTM A Search Space Odyssey.pdf` (Greff et al., 2017) is the methodological reference
  for LSTM tuning. It uses validation for early stopping, a cap of 150 epochs with patience 15,
  and 200 random trials per model/dataset. It finds learning rate to be the most important tuned
  hyperparameter, followed by hidden-layer size.
- Consequently, only `hidden_dim`, `num_layers`, `dropout`, `lr`, and `wd` should be described as
  LSTM architecture/optimization hyperparameters in this project. `SEED`, `N_SPLITS`,
  `N_TRIALS_KFOLD`, max epochs, patience, and batch size are experiment-protocol settings.
  The exact current values `42, 5, 8, 80, 12, 16` are project choices, not a set copied from either
  reference; in particular, 8 trials/fold is a light screening budget.
- Literature-comparison note (2026-08-15): Fan et al. (2024) overlaps at the broad level of
  time-series DL for concrete shrinkage, but it is not the same study. It uses an autoregressive
  sliding window including historical target values and proposes GRU + self-attention for
  shrinkage/creep. This project focuses on SCC, compares a wider benchmark of DL/GBDT/empirical
  models, and evaluates sequence models at GID level. The paper groups windows by test identifier,
  then states that its preprocessed samples are shuffled and split 80:20. Because adjacent
  sliding windows from one identifier overlap and share static features, this is a strong
  group-leakage risk; its normalization also appears to occur before that split. Do not state the
  leak as proven without the authors' code, but describe it as an optimistic-validation risk.

## Parameter-source audit

- `PARAMETER_SOURCE_AUDIT.md` maps every model in files `01`–`11` to: its tuned parameter
  families, the original architecture/API source to cite, the current project search space, and
  whether the numerical range is literature-prescribed or an internal experimental choice.

## Change log

| Date | Change | Notes |
|---|---|---|
| 2026-08-15 | Created project context | Baseline snapshot after reviewing the folder and `08_lstm.py`. |
| 2026-08-15 | Added tuning audit | Current 8-trial budget is below Optuna TPE's default 10 startup trials; LSTM has no cached completed run to evaluate. |
| 2026-08-15 | Read two LSTM references | Recorded which current settings are literature-backed versus project-specific. |
| 2026-08-15 | Added parameter-source audit | Mapped source references and internal search ranges for models 01–11. |

## Maintenance rule

When changing data policy, features, model settings, or outputs, add one concise row to the
change log with date, files changed, reason, and whether affected models were rerun. Read this
file and `TODO_NOTES.md` first in a future task; inspect only the files relevant to the new work.
