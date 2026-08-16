"""15 - Optuna Budget Sensitivity: rerun the top-3 DL models (ResMLP, GRU, LSTM
- per the pooled OOF of file 12) with Optuna 30 trials/fold instead of 8 trials/fold
as in the official files 03/07/08.

This file addresses the limitation "small tuning budget (8 trials/fold)" in the
paper: if the model ranking and the gap with GBDT do not change significantly when
increasing to 30 trials, the paper's conclusion is considered ROBUST to the tuning
budget chosen for Optuna.

This file MERGES (without omission) the content from buoc_23_optuna_sensitivity.py,
plus the shared pipeline common_tabular.py, common_sequence_group.py and
buoc_01_code_loc_data.py. Uses the SAME GroupKFold5, seed 42, and the same preprocessing as
files 03/07/08/12 - only N_TRIALS=30 differs.

Intermediate CSV/JSON files are written to the hidden directory _cache/sensitivity_t30/
(so as not to overwrite the official 8-trial results in _cache/<model>/kfold/ of
files 03/07/08). The final comparison table is written to output_tables/ (the
VISIBLE directory, since this is a deliverable that needs to be cross-checked in the paper/thesis).

WARNING: this file runs Optuna 30 trials/fold x 5 folds x 3 models - SIGNIFICANTLY
LONGER than the 8-trial files (01-10). Should be run separately, not run together
with other files at the same time.

Run:  python 15_optuna_sensitivity.py
"""

from __future__ import annotations

# =============================================================================
# PART 1 - SHARED PIPELINE FOR TABULAR MODELS (merged from common_tabular.py)
# NOTE: must be placed before the data-preparation part because this block imports torch BEFORE
# numpy/pandas/sklearn (required on Windows to avoid WinError 1114 when
# loading torch's c10.dll if numpy/MKL was already loaded beforehand).
# =============================================================================

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable

# Disable Dynamo (PyTorch's JIT compiler) to avoid errors on some systems
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
# Limit CPU thread count = 1 to ensure reproducible results (reproducibility)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# IMPORTANT: torch must be imported BEFORE numpy/sklearn/optuna.
# On Windows, if numpy/MKL is loaded first, loading torch's c10.dll
# can fail with "[WinError 1114] DLL initialization routine failed".
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    # If PyTorch cannot be imported, create dummy (fallback) classes so the file still loads
    # without a syntax error. A clear error will be raised at actual run time.
    class _TorchFallback:
        Tensor = object

    class _NNFallback:
        class Module:
            pass
        class Sequential:
            def __init__(self, *a, **kw): pass
        class Linear:
            def __init__(self, *a, **kw): pass
        class ReLU:
            def __init__(self, *a, **kw): pass
        class Dropout:
            def __init__(self, *a, **kw): pass
        class BatchNorm1d:
            def __init__(self, *a, **kw): pass

    class _OptimFallback:
        pass

    torch = _TorchFallback()
    nn = _NNFallback()
    optim = _OptimFallback()
    TORCH_IMPORT_ERROR = exc

# Limit PyTorch thread count = 1 to avoid resource contention and ensure reproducibility
if TORCH_IMPORT_ERROR is None:
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

# The scientific stack is imported AFTER torch (see note above about WinError 1114).
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Columns to exclude from the feature set: group column, target column, auxiliary columns
EXCLUDE_COLUMNS = ["GID", "dt_days", "y_shrinkage_total_mm_per_m", "y_log1p", "seq_ok"]
# Categorical columns will be one-hot encoded for the DL model
DL_ONE_HOT_COLUMNS = ["agg_type", "region_type", "Shape_factor"]


# ---------------------------------------------------------------------------
# Config & dataclass
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BenchmarkConfig:
    """Config class holding all parameters to run a benchmark for a DL model.

    Explanation of each parameter (for beginners):
        data_path   : Path to the CSV file with input data (already preprocessed).
        output_dir  : Directory to save results (summary CSV, best params, metadata).
        split_mode  : How to split the data. "group" = split by GID group (recommended
                      since it avoids data leakage between samples in the same group).
                      "random" = split randomly by row.
        seeds       : List of seed values. Each seed runs one full pipeline
                      to evaluate stability.
        n_trials    : Number of Optuna trials to find the best hyperparameters.
                      More trials increase the chance of finding an optimal parameter set,
                      but take more time.
        max_epochs  : Maximum epochs when training each trial. If the model doesn't
                      improve sooner, it stops early via early stopping.
        patience    : Number of consecutive epochs where val loss doesn't decrease before
                      stopping early. Helps avoid overtraining (overfitting).
        dl_batch_size: Mini-batch size. Small batch -> more frequent updates
                      but noisier. Large batch -> more stable but more memory.
        device      : Compute device. "cpu" = use CPU, "cuda" = use GPU if available.
    """
    data_path: Path
    output_dir: Path
    split_mode: str = "group"
    seeds: list[int] = field(default_factory=lambda: [42])
    n_trials: int = 3
    max_epochs: int = 100
    patience: int = 15
    dl_batch_size: int = 256
    device: str = "cpu"
    # Number of folds for group k-fold cross-validation (only used by run_tabular_kfold).
    n_splits: int = 5


@dataclass(slots=True)
class SplitIndices:
    """Stores the indices of rows belonging to the train, val, test sets.

    After splitting the data, we don't copy the data, we only store row indices.
    To get the actual data, use df.iloc[train_idx] to access it.
    """
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


# ---------------------------------------------------------------------------
# Set seed (ensures reproducible results)
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set the seed for all libraries so results can be reproduced.

    Why do we need a seed?
    - DL algorithms have many random steps: weight initialization,
      batch shuffling, data splitting.
    - Without a fixed seed, each run would give different results, making it
      hard to compare models or debug.
    - By setting the same seed, we ensure every random step happens identically,
      making results fully reproducible.

    Parameters:
        seed: Integer used as the seed. 42 is commonly used by convention.
    """
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_IMPORT_ERROR is None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def portable_path(path: Path) -> str:
    """Convert an absolute path to a path relative to the package's root directory.

    Helps metadata files have short, readable paths that are portable across machines.
    """
    package_root = Path(__file__).resolve().parent
    try:
        return path.resolve().relative_to(package_root).as_posix()
    except Exception:
        return path.as_posix()


def build_torch_runtime_message(model_name: str, exc: Exception | None) -> str:
    """Build a detailed error message when PyTorch cannot be initialized.

    Includes a suggested fix: create a separate environment for Python 3.11
    with torch and spyder-kernels.
    """
    base = f"Could not initialize PyTorch for {model_name}."
    detail = "" if exc is None else f" Loi goc: {exc}."
    hint = (
        " Tip: run the DL files in a separate Python 3.11 environment, "
        "install torch and spyder-kernels, then point Spyder to that environment "
        "instead of using spyder-runtime."
    )
    return f"{base}{detail}{hint}"


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute evaluation metrics for the regression task.

    Metrics computed:
        N             : Number of samples.
        R2            : Coefficient of determination (1.0 = perfect, 0.0 = as bad as predicting the mean).
        RMSE          : Root mean squared error (same unit as target).
        MAE           : Mean absolute error (less sensitive to outliers than RMSE).
        NRMSE_IQR_pct : RMSE normalized by the interquartile range (IQR), expressed as a percentage.
                        Helps compare error across datasets with different ranges.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    # IQR = distance between the 75th and 25th percentile, used to normalize RMSE
    iqr = float(np.percentile(y_true, 75) - np.percentile(y_true, 25))
    nrmse_iqr_pct = 0.0 if iqr == 0 else float(rmse / iqr * 100.0)
    return {"N": int(y_true.shape[0]), "R2": r2, "RMSE": rmse, "MAE": mae, "NRMSE_IQR_pct": nrmse_iqr_pct}


def aggregate_results(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results from multiple runs (multiple seeds) into a table of mean and standard deviation.

    Group by (Split, Family, Model) then compute mean and std for each numeric metric.
    Helps assess model stability across multiple runs.
    """
    if summary_df.empty:
        return pd.DataFrame()
    group_cols = ["Split", "Family", "Model"]
    # Filter out numeric columns (skip Seed, Split, Family, Model)
    numeric_cols = [
        col for col in summary_df.columns
        if col not in {"Seed", "Split", "Family", "Model"}
        and pd.api.types.is_numeric_dtype(summary_df[col])
    ]
    grouped = summary_df.groupby(group_cols)
    agg = grouped[numeric_cols].agg(["mean", "std"])
    # Name columns like "RMSE_mean", "RMSE_std" for readability
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()
    # Add a column counting the number of runs per group
    agg.insert(len(group_cols), "runs", grouped.size().to_numpy())
    return agg


# ---------------------------------------------------------------------------
# Data Splitting
# ---------------------------------------------------------------------------
def make_quantile_bins(values: pd.Series, q: int = 5) -> pd.Series | None:
    """Bin a continuous value into groups (bins) by quantile for stratified splitting.

    Stratified split ensures the target distribution is similar across train/val/test.
    Returns None if there isn't enough data to split (e.g. too many duplicate values).
    """
    values = pd.to_numeric(values, errors="coerce")
    if values.isna().all():
        return None
    bins = pd.qcut(values, q=q, labels=False, duplicates="drop")
    counts = pd.Series(bins).value_counts(dropna=False)
    # Need at least 2 samples per bin for stratified train_test_split to work
    if counts.empty or counts.min() < 2:
        return None
    return pd.Series(bins)


def random_row_split(df: pd.DataFrame, seed: int) -> SplitIndices:
    """Randomly split data by row into train (70%), val (15%), test (15%).

    This approach is simple but has a drawback: samples from the same GID group may end up
    in both train and test, causing data leakage. Use group_stratified_split
    instead when the data has group structure.
    """
    indices = np.arange(len(df))
    # Stratify by fc28 to distribute concrete strength evenly across the splits
    strat = make_quantile_bins(df["fc28"]) if "fc28" in df.columns else None
    strat_values = None if strat is None else strat.to_numpy()
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=seed, shuffle=True, stratify=strat_values)
    # Split the remainder (30%) into val (15%) and test (15%)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    temp_indices = np.arange(len(temp_idx))
    temp_strat = make_quantile_bins(temp_df["fc28"]) if "fc28" in temp_df.columns else None
    temp_strat_values = None if temp_strat is None else temp_strat.to_numpy()
    val_local, test_local = train_test_split(temp_indices, test_size=0.5, random_state=seed, shuffle=True, stratify=temp_strat_values)
    return SplitIndices(train_idx=np.asarray(train_idx), val_idx=np.asarray(temp_idx[val_local]), test_idx=np.asarray(temp_idx[test_local]))


def group_stratified_split(df: pd.DataFrame, seed: int) -> SplitIndices:
    """Split data by GID (Group ID) into train/val/test.

    Why split by GID?
    - Each GID represents a group of concrete samples from the same batch/conditions.
    - If split randomly by row, samples from the same GID could end up in both train and test.
      The model would "see" information from that group during training, leading to
      falsely optimistic evaluation (data leakage).
    - Splitting by GID ensures ALL samples in a group stay in the same split (train OR
      val OR test), correctly reflecting the ability to generalize to unseen groups.

    Procedure:
        1. Compute the median fc28 for each GID to create a group-representative feature.
        2. Split the GIDs into 70% train / 15% val / 15% test (stratified by fc28).
        3. Assign all rows belonging to each GID to the corresponding split.
    """
    # Step 1: Compute median fc28 for each GID group
    group_stats = df.groupby("GID", as_index=False).agg(fc28=("fc28", "median")).sort_values("GID").reset_index(drop=True)
    strat = make_quantile_bins(group_stats["fc28"])
    strat_values = None if strat is None else strat.to_numpy()
    all_groups = group_stats["GID"].to_numpy()
    # Step 2: Split GIDs into train (70%) and the remainder (30%)
    train_groups, temp_groups = train_test_split(all_groups, test_size=0.3, random_state=seed, shuffle=True, stratify=strat_values)
    # Split the remainder into val and test (each ~15%)
    temp_stats = group_stats[group_stats["GID"].isin(temp_groups)].reset_index(drop=True)
    temp_strat = make_quantile_bins(temp_stats["fc28"])
    temp_strat_values = None if temp_strat is None else temp_strat.to_numpy()
    val_groups, test_groups = train_test_split(temp_stats["GID"].to_numpy(), test_size=0.5, random_state=seed, shuffle=True, stratify=temp_strat_values)
    # Step 3: Assign row indices based on the allocated GID groups
    gid_values = df["GID"].to_numpy()
    return SplitIndices(
        train_idx=np.where(np.isin(gid_values, train_groups))[0],
        val_idx=np.where(np.isin(gid_values, val_groups))[0],
        test_idx=np.where(np.isin(gid_values, test_groups))[0],
    )


def group_kfold_split_indices(df: pd.DataFrame, n_splits: int, seed: int, val_fraction: float = 0.1765) -> list[tuple[SplitIndices, int]]:
    """Create folds for GROUP K-FOLD cross-validation (stratified by fc28).

    Why do we need group k-fold?
    - A single holdout split (70/15/15) only gives ONE test estimate -> prone to luck
      depending on which group ends up in test. Reviewer comment (#2) requires cross-validation to
      assess generalization ability more reliably.
    - Group k-fold ensures EVERY GID group appears in the test set EXACTLY ONCE (across K
      folds), and never appears in both train and test within the same fold ->
      no data leakage between samples in the same group.

    Procedure:
        1. Each GID is represented by the group's median fc28; split into quantile bins to stratify.
        2. StratifiedKFold over the list of GIDs -> assigns each GID to exactly 1 test fold.
           (Falls back to plain KFold if stratification isn't possible.)
        3. For each fold k: test = GIDs of fold k; the remainder is split into train/val
           (val ~ val_fraction of the remainder, stratified by fc28 if possible).

    Parameters:
        n_splits     : number of folds K (e.g. 5).
        seed         : random seed for splitting folds (reproducible).
        val_fraction : fraction of groups taken out for validation FROM the remainder after
                       removing test. Default 0.1765 ~ makes val approximately 15% of the total when K=5
                       (0.1765 x 0.8 ~ 0.141).

    Returns:
        A list of [(SplitIndices, fold_id)] with fold_id running from 0..K-1.
    """
    group_stats = df.groupby("GID", as_index=False).agg(fc28=("fc28", "median")).sort_values("GID").reset_index(drop=True)
    all_groups = group_stats["GID"].to_numpy()
    strat = make_quantile_bins(group_stats["fc28"])
    gid_values = df["GID"].to_numpy()

    # Step 2: assign each GID to a test fold
    if strat is not None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_iter = splitter.split(all_groups, strat.to_numpy())
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_iter = splitter.split(all_groups)

    folds: list[tuple[SplitIndices, int]] = []
    for fold_id, (rest_pos, test_pos) in enumerate(fold_iter):
        test_groups = all_groups[test_pos]
        rest_groups = all_groups[rest_pos]
        # Step 3: split train/val from the remainder (stratified by fc28 of the rest)
        rest_stats = group_stats.iloc[rest_pos].reset_index(drop=True)
        rest_strat = make_quantile_bins(rest_stats["fc28"])
        rest_strat_values = None if rest_strat is None else rest_strat.to_numpy()
        train_groups, val_groups = train_test_split(
            rest_groups, test_size=val_fraction, random_state=seed, shuffle=True, stratify=rest_strat_values,
        )
        folds.append((
            SplitIndices(
                train_idx=np.where(np.isin(gid_values, train_groups))[0],
                val_idx=np.where(np.isin(gid_values, val_groups))[0],
                test_idx=np.where(np.isin(gid_values, test_groups))[0],
            ),
            fold_id,
        ))
    return folds


def make_split_indices(df: pd.DataFrame, split_mode: str, seed: int) -> tuple[SplitIndices, str]:
    """Dispatcher function: chooses how to split the data based on split_mode.

    Returns a tuple (SplitIndices, split_name) to record in the results.
    """
    if split_mode == "group":
        return group_stratified_split(df, seed), "GroupWise"
    if split_mode == "random":
        return random_row_split(df, seed), "RandomRowWise"
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def split_diagnostics(df: pd.DataFrame, split_idx: SplitIndices, split_name: str, seed: int) -> dict[str, Any]:
    """Compute diagnostic statistics about how the data was split.

    Checks whether any GID group is duplicated across splits (should be 0).
    If overlap > 0 it means there is data leakage, the split should be checked again.
    """
    train_gids = set(df.iloc[split_idx.train_idx]["GID"].tolist())
    val_gids = set(df.iloc[split_idx.val_idx]["GID"].tolist())
    test_gids = set(df.iloc[split_idx.test_idx]["GID"].tolist())
    # Count GID groups duplicated across splits (ideally = 0)
    overlap_tv = len(train_gids & val_gids)
    overlap_tt = len(train_gids & test_gids)
    overlap_vt = len(val_gids & test_gids)
    return {
        "Seed": seed, "Split": split_name,
        "Train_Rows": int(len(split_idx.train_idx)), "Val_Rows": int(len(split_idx.val_idx)), "Test_Rows": int(len(split_idx.test_idx)),
        "Train_Groups": int(len(train_gids)), "Val_Groups": int(len(val_gids)), "Test_Groups": int(len(test_gids)),
        "Overlap_Train_Val_Groups": overlap_tv, "Overlap_Train_Test_Groups": overlap_tt, "Overlap_Val_Test_Groups": overlap_vt,
        "Any_GID_Overlap": int(overlap_tv > 0 or overlap_tt > 0 or overlap_vt > 0),
    }


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------
def split_frames(df: pd.DataFrame, split_idx: SplitIndices) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the original DataFrame into 3 sub-DataFrames (train, val, test) based on the split indices.

    reset_index(drop=True) so indices restart at 0, avoiding errors when accessed later.
    """
    return (
        df.iloc[split_idx.train_idx].reset_index(drop=True),
        df.iloc[split_idx.val_idx].reset_index(drop=True),
        df.iloc[split_idx.test_idx].reset_index(drop=True),
    )


def impute_train_median(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols: list[str]) -> None:
    """Fill missing values (NaN) in numeric columns with the MEDIAN computed ONLY on the train set.

    Why put this step here (at experiment time) instead of at data-build time?
    - If the median is computed on the ENTIRE dataset before splitting into train/test, information from
      val/test (the median value) would "leak" into the train imputation ->
      falsely optimistic evaluation. This is a form of preprocessing data leakage.
    - Correct approach: fit the median ONLY on train, then apply (transform) it to train,
      val and test alike. Exactly the same principle as StandardScaler.

    This function modifies the 3 passed-in DataFrames in place.
    If a column is entirely NaN on train (median = NaN), fill with 0.0 as a fallback.
    """
    if not numeric_cols:
        return
    med = train_df[numeric_cols].median(numeric_only=True)
    # Fallback: any column whose median is still NaN (entirely NaN on train) -> fill with 0.0
    med = med.fillna(0.0)
    for frame in (train_df, val_df, test_df):
        frame[numeric_cols] = frame[numeric_cols].fillna(med)


def scale_numeric_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize numeric features using StandardScaler.

    StandardScaler transforms each feature to mean = 0 and standard deviation = 1.
    This matters a lot for neural networks because:
    - It puts features on the same scale, preventing large-magnitude features from dominating small ones.
    - It helps the optimization process (gradient descent) converge faster.

    IMPORTANT NOTE: fit_transform only on the train set, then transform on val/test.
    Fitting on val/test as well would cause data leakage.
    """
    if not feature_cols:
        return tuple(np.empty((len(d), 0), dtype=np.float32) for d in (train_df, val_df, test_df))
    scaler = StandardScaler()
    # Fit only on train to avoid leaking information from val/test
    X_train = scaler.fit_transform(train_df[feature_cols].to_numpy()).astype(np.float32)
    X_val = scaler.transform(val_df[feature_cols].to_numpy()).astype(np.float32)
    X_test = scaler.transform(test_df[feature_cols].to_numpy()).astype(np.float32)
    return X_train, X_val, X_test


def encode_dl_categoricals(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, categorical_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """One-hot encode categorical features for the DL model.

    One-hot encoding turns each categorical value into a binary column (0 or 1).
    Example: column "agg_type" with 3 values {1, 2, 3} becomes 3 columns: agg_type__1, agg_type__2, agg_type__3.

    The list of categories is taken only from the train set to avoid data leakage.

    Returns:
        A tuple of (train array, val array, test array, list of encoded column names).
    """
    if not categorical_cols:
        return (*(np.empty((len(d), 0), dtype=np.float32) for d in (train_df, val_df, test_df)), [])
    train_blocks, val_blocks, test_blocks = [], [], []
    encoded_feature_cols: list[str] = []
    for col in categorical_cols:
        # Convert to numeric and get the list of unique values from the train set
        train_values = pd.to_numeric(train_df[col], errors="coerce")
        categories = sorted(pd.Series(train_values).dropna().unique().tolist())
        if not categories:
            continue
        # Build encoded column names like "column_name__value"
        for cat in categories:
            label = str(int(cat)) if float(cat).is_integer() else str(cat)
            encoded_feature_cols.append(f"{col}__{label}")
        def build_block(frame: pd.DataFrame, cats=categories) -> np.ndarray:
            """Build a one-hot array for one categorical column of a DataFrame."""
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy()
            return np.column_stack([(values == cat).astype(np.float32) for cat in cats])
        train_blocks.append(build_block(train_df))
        val_blocks.append(build_block(val_df))
        test_blocks.append(build_block(test_df))
    if not train_blocks:
        return (*(np.empty((len(d), 0), dtype=np.float32) for d in (train_df, val_df, test_df)), [])
    # Concatenate all one-hot blocks into a single array
    return (np.concatenate(train_blocks, axis=1), np.concatenate(val_blocks, axis=1), np.concatenate(test_blocks, axis=1), encoded_feature_cols)


def prepare_arrays_dl(df: pd.DataFrame, split_idx: SplitIndices) -> tuple[np.ndarray, ...]:
    """Full preprocessing pipeline: from the original DataFrame to NumPy arrays ready for DL.

    Detailed procedure:
        1. Remove non-feature columns (GID, target, auxiliary columns).
        2. Split the DataFrame into train/val/test based on the split indices.
        3. Classify columns into numeric and categorical.
        4. Scale numeric features with StandardScaler (fit on train).
        5. One-hot encode categorical features (categories from train).
        6. Concatenate numeric and one-hot features into the final feature array.
        7. Take the target as column "y_log1p" = log1p(y_shrinkage_total_mm_per_m).
           log1p is used because:
           - Shrinkage values are right-skewed.
           - log1p makes the distribution closer to normal, easier for the model to learn.
           - After prediction, use expm1 to convert back to the original value.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
    """
    # Step 1: Determine feature columns (remove unused columns)
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    # Step 2: Split the DataFrame into 3 parts
    train_df, val_df, test_df = split_frames(df, split_idx)
    # Step 3: Classify columns
    categorical_cols = [col for col in DL_ONE_HOT_COLUMNS if col in feature_cols]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    # Step 3b: Fill missing values (NaN) using the median computed ONLY on train (avoids leakage).
    # Imputation was moved from the data-build step to here to guarantee train-only.
    impute_train_median(train_df, val_df, test_df, numeric_cols)
    # Step 4: Scale numeric features
    X_train_num, X_val_num, X_test_num = scale_numeric_features(train_df, val_df, test_df, numeric_cols)
    # Step 5: One-hot encode categorical features
    X_train_cat, X_val_cat, X_test_cat, enc_cols = encode_dl_categoricals(train_df, val_df, test_df, categorical_cols)
    # Step 6: Concatenate numeric and one-hot features
    def concat(*blocks):
        """Concatenate feature arrays, skipping empty arrays."""
        nonempty = [b for b in blocks if b.shape[1] > 0]
        return np.concatenate(nonempty, axis=1).astype(np.float32) if nonempty else np.empty((blocks[0].shape[0], 0), dtype=np.float32)
    X_train = concat(X_train_num, X_train_cat)
    X_val = concat(X_val_num, X_val_cat)
    X_test = concat(X_test_num, X_test_cat)
    feature_names = numeric_cols + enc_cols
    # Step 7: Take the log1p-transformed target
    y_train = train_df["y_log1p"].to_numpy(dtype=np.float32)
    y_val = val_df["y_log1p"].to_numpy(dtype=np.float32)
    y_test = test_df["y_log1p"].to_numpy(dtype=np.float32)
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def to_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    """Convert a NumPy array to a PyTorch tensor (float32) on the specified device (CPU or GPU)."""
    return torch.tensor(array, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Training the PyTorch model
# ---------------------------------------------------------------------------
def train_model(
    model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    params: dict[str, Any], max_epochs: int, patience: int, dl_batch_size: int,
) -> tuple[nn.Module, int, list[float]]:
    """Train a PyTorch model with mini-batches, AdamW, ReduceLROnPlateau, and early stopping.

    Detailed explanation for beginners:

    1. Mini-batch training:
       - Instead of feeding the entire dataset into the model at once (memory-intensive),
         we split the data into small batches (mini-batches) and update weights after each batch.
       - Each epoch, the batch order is shuffled randomly so the model
         doesn't learn the data order.

    2. AdamW optimizer:
       - The most popular optimization algorithm for DL today.
       - Combines Adam (adaptive learning rate) with weight decay (penalizing large weights)
         to avoid overfitting. Parameters: lr = learning rate, wd = weight decay.

    3. ReduceLROnPlateau scheduler:
       - Automatically reduces the learning rate when val loss stops decreasing after some epochs.
       - factor=0.5: halves the lr. patience=patience//3: waits a few epochs before reducing.
       - min_lr=1e-5: never reduces below this threshold.

    4. Early stopping:
       - Stops training early when val loss doesn't improve after 'patience' consecutive epochs.
       - Saves the best model state (best_state) to restore after stopping.
       - Helps avoid overfitting: the model learning train too well but performing poorly on val/test.

    5. Gradient clipping (clip_grad_norm_ max_norm=1.0):
       - Clips gradient magnitude to avoid "exploding gradients",
         making training more stable.

    Parameters:
        model        : The PyTorch model (nn.Module) to train.
        X_train      : Feature tensor for the training set.
        y_train      : Target tensor for the training set.
        X_val        : Feature tensor for the validation set.
        y_val        : Target tensor for the validation set.
        params       : Dict of hyperparameters (lr, wd, ...).
        max_epochs   : Maximum number of epochs.
        patience     : Number of epochs to wait before early stopping.
        dl_batch_size: Mini-batch size.

    Returns:
        (best_model, best_epoch, val_loss_history): the model with the best weights,
        the epoch reached, and the val_loss of EACH epoch run (used to plot the learning curve later).
    """
    model.to(X_train.device)
    optimizer = optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["wd"]))
    # Scheduler that reduces lr when val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(4, patience // 3), min_lr=1e-5)
    loss_fn = nn.MSELoss()  # Loss function: Mean Squared Error
    n_train = int(X_train.shape[0])
    batch_size = max(1, min(int(dl_batch_size), n_train))
    # Track the best state to restore on early stopping
    best_loss, best_epoch, best_state, patience_count = float("inf"), 0, None, 0
    val_loss_history: list[float] = []
    for epoch in range(max_epochs):
        model.train()  # Enable training mode (activates Dropout, BatchNorm train mode)
        # Randomly shuffle sample order each epoch
        perm = torch.randperm(n_train, device=X_train.device)
        batches = list(perm.split(batch_size))
        # If the last batch has only 1 sample, merge it into the previous batch (BatchNorm needs at least 2 samples)
        if len(batches) >= 2 and batches[-1].numel() == 1:
            batches[-2] = torch.cat([batches[-2], batches[-1]], dim=0)
            batches.pop()
        for idx in batches:
            optimizer.zero_grad(set_to_none=True)  # Clear old gradients (saves memory)
            loss = loss_fn(model(X_train[idx]), y_train[idx])  # Compute loss on the batch
            loss.backward()  # Backpropagate to compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradient
            optimizer.step()  # Update weights
        model.eval()  # Enable evaluation mode (disables Dropout, BatchNorm eval mode)
        with torch.no_grad():  # No need to compute gradients during evaluation (saves memory)
            val_loss = float(loss_fn(model(X_val), y_val).item())
        val_loss_history.append(val_loss)
        scheduler.step(val_loss)  # Update the learning rate based on val loss
        # Check whether val loss improved
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            # Save a copy of the best weights (cloned to CPU to save GPU memory)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break  # Early stop: val loss hasn't decreased after 'patience' epochs
    # Restore the best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, val_loss_history


def predict_log(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Predict log1p(y) values from the trained model.

    Returns a 1D NumPy array. The result is still on the log scale,
    call np.expm1() to convert it back to the original value.
    """
    model.eval()
    with torch.no_grad():
        return model(X).detach().cpu().numpy().reshape(-1)


def permutation_importance_tabular(
    model: nn.Module, X_val_t: torch.Tensor, y_val: np.ndarray,
    feature_names: list[str], device: str, n_repeats: int = 3, seed: int = 0,
) -> list[float]:
    """Measure the importance of each feature via permutation importance.

    Idea: shuffle the values of ONE feature column across rows,
    breaking the relationship between that feature and the target, then measure how much
    val MSE increases compared to the baseline. The more important the feature -> shuffling it
    increases MSE the most. Reuses the ALREADY-TRAINED model of the current
    fold (no retraining), only needs multiple forward-passes on val.
    """
    baseline_mse = float(mean_squared_error(y_val, predict_log(model, X_val_t)))
    rng = np.random.default_rng(seed)
    X_val_np = X_val_t.detach().cpu().numpy()
    importances: list[float] = []
    for j in range(len(feature_names)):
        increases = []
        for _ in range(n_repeats):
            X_perm = X_val_np.copy()
            rng.shuffle(X_perm[:, j])
            perm_mse = float(mean_squared_error(y_val, predict_log(model, to_tensor(X_perm, device))))
            increases.append(perm_mse - baseline_mse)
        importances.append(float(np.mean(increases)))
    return importances


# ---------------------------------------------------------------------------
# Result Summary
# ---------------------------------------------------------------------------
def summarize_result(
    model_name: str, split_name: str, seed: int, n_features: int,
    selection_mse_log: float, epochs_used: int,
    y_train_raw: np.ndarray, pred_train_raw: np.ndarray,
    y_val_raw: np.ndarray, pred_val_raw: np.ndarray,
    y_test_raw: np.ndarray, pred_test_raw: np.ndarray,
) -> dict[str, Any]:
    """Build a single summary result row for one run (one seed).

    Compute metrics on all 3 sets (train, val, test) on the original scale (after expm1).
    Also includes auxiliary info: number of features, number of epochs used, MSE on the log scale.
    """
    t = regression_metrics(y_train_raw, pred_train_raw)
    v = regression_metrics(y_val_raw, pred_val_raw)
    te = regression_metrics(y_test_raw, pred_test_raw)
    return {
        "Seed": seed, "Split": split_name, "Family": "DL", "Model": model_name,
        "n_features": n_features, "epochs_used": epochs_used, "Selection_MSE_log": selection_mse_log,
        "Train_N": t["N"], "Train_R2": t["R2"], "Train_RMSE": t["RMSE"], "Train_MAE": t["MAE"], "Train_NRMSE_IQR_pct": t["NRMSE_IQR_pct"],
        "Val_N": v["N"], "Val_R2": v["R2"], "Val_RMSE": v["RMSE"], "Val_MAE": v["MAE"], "Val_NRMSE_IQR_pct": v["NRMSE_IQR_pct"],
        "Test_N": te["N"], "Test_R2": te["R2"], "Test_RMSE": te["RMSE"], "Test_MAE": te["MAE"], "Test_NRMSE_IQR_pct": te["NRMSE_IQR_pct"],
    }


# ---------------------------------------------------------------------------
# Skip & Output Writers
# ---------------------------------------------------------------------------
def write_skip_outputs(cfg: BenchmarkConfig, model_name: str, reason: str) -> dict[str, Path]:
    """Write empty output files when the model is skipped (e.g. PyTorch import failed).

    Still creates the full directory structure and metadata file so downstream steps don't fail.
    The metadata file records the "skipped" status and the reason.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: cfg.output_dir / f for k, f in [
        ("summary", "results_summary.csv"), ("aggregate", "results_aggregate.csv"),
        ("split_summary", "split_summary.csv"), ("best_params", "best_params.json"),
        ("metadata", "run_metadata.json"),
    ]}
    # Write an empty CSV file and an empty JSON file
    for k in ("summary", "aggregate", "split_summary"):
        paths[k].write_text("", encoding="utf-8")
    paths["best_params"].write_text(json.dumps({}, indent=2), encoding="utf-8")
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name, "status": "skipped", "skip_reason": reason,
        "config": {
            "data_path": portable_path(cfg.data_path), "output_dir": portable_path(cfg.output_dir),
            "split_mode": cfg.split_mode, "seeds": cfg.seeds, "n_trials": cfg.n_trials,
            "max_epochs": cfg.max_epochs, "patience": cfg.patience,
            "dl_batch_size": cfg.dl_batch_size, "device": cfg.device,
        },
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def write_completed_outputs(
    cfg: BenchmarkConfig, model_name: str,
    summary_rows: list[dict], split_rows: list[dict], best_params: dict,
) -> dict[str, Path]:
    """Write all results after the benchmark completes successfully.

    Output files:
        results_summary.csv   : Detailed results per seed (one row = one run).
        results_aggregate.csv : Mean/std aggregated across seeds.
        split_summary.csv     : Diagnostic statistics about how the data was split.
        best_params.json      : Best hyperparameters found by Optuna for each seed.
        run_metadata.json     : Info about the run (time, config, status).
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    aggregate_df = aggregate_results(summary_df)
    split_df = pd.DataFrame(split_rows)
    paths = {
        "summary": cfg.output_dir / "results_summary.csv",
        "aggregate": cfg.output_dir / "results_aggregate.csv",
        "split_summary": cfg.output_dir / "split_summary.csv",
        "best_params": cfg.output_dir / "best_params.json",
        "metadata": cfg.output_dir / "run_metadata.json",
    }
    # Use utf-8-sig for CSV so Excel opens Vietnamese text correctly
    summary_df.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(paths["aggregate"], index=False, encoding="utf-8-sig")
    split_df.to_csv(paths["split_summary"], index=False, encoding="utf-8-sig")
    paths["best_params"].write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name, "status": "completed",
        "config": {"data_path": portable_path(cfg.data_path), "output_dir": portable_path(cfg.output_dir),
                   "split_mode": cfg.split_mode, "seeds": cfg.seeds, "n_trials": cfg.n_trials,
                   "max_epochs": cfg.max_epochs, "patience": cfg.patience,
                   "dl_batch_size": cfg.dl_batch_size, "device": cfg.device},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


# ---------------------------------------------------------------------------
# Evaluate a single split (shared between holdout and k-fold CV)
# ---------------------------------------------------------------------------
def _fit_eval_one_split(
    cfg: BenchmarkConfig,
    model_name: str,
    build_model_fn: Callable[[int, dict[str, Any]], nn.Module],
    search_space_fn: Callable[[optuna.Trial], dict[str, Any]],
    df: pd.DataFrame,
    split_idx: SplitIndices,
    split_label: str,
    seed: int,
) -> dict[str, Any]:
    """Preprocessing + Optuna + training + evaluation for ONE split (train/val/test).

    This is the core logic extracted for shared use by both:
      - run_tabular_benchmark (70/15/15 holdout split per seed), and
      - run_tabular_kfold (group k-fold cross-validation, called once per fold).

    Returns a dict with:
        row        : one result row (train/val/test metrics on the original scale).
        params     : best hyperparameters found by Optuna for this split.
        y_test_raw : test target array (original scale) -- used for plotting later.
        pred_test_raw : test prediction array (original scale) -- used for residual/scatter plots.
        gid_test   : GID array of the test rows (for out-of-fold traceability).

    Raises an exception if training fails; the caller decides how to handle it.
    """
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_arrays_dl(df, split_idx)

    X_train_t = to_tensor(X_train, cfg.device)
    y_train_t = to_tensor(y_train.reshape(-1, 1), cfg.device)
    X_val_t = to_tensor(X_val, cfg.device)
    y_val_t = to_tensor(y_val.reshape(-1, 1), cfg.device)
    X_test_t = to_tensor(X_test, cfg.device)

    _best: dict[str, Any] = {"val": float("inf")}

    def objective(trial: optuna.Trial) -> float:
        params = search_space_fn(trial)
        m = build_model_fn(X_train.shape[1], params)
        m, ep, hist = train_model(m, X_train_t, y_train_t, X_val_t, y_val_t, params, cfg.max_epochs, cfg.patience, cfg.dl_batch_size)
        val_mse = float(mean_squared_error(y_val, predict_log(m, X_val_t)))
        if val_mse < _best["val"]:
            _best["val"] = val_mse
            _best["state"] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            _best["params"] = params
            _best["epochs"] = ep
            _best["history"] = hist
        return val_mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)
    params = _best.get("params", dict(study.best_params))
    epochs_used = _best.get("epochs", 0)
    loss_history = _best.get("history", [])
    trials_df = study.trials_dataframe(attrs=("number", "value", "params"))

    model = build_model_fn(X_train.shape[1], params)
    if "state" in _best:
        model.load_state_dict(_best["state"])
        model = model.to(cfg.device)
    else:
        model, epochs_used, loss_history = train_model(model, X_train_t, y_train_t, X_val_t, y_val_t, params, cfg.max_epochs, cfg.patience, cfg.dl_batch_size)

    pred_train_log = predict_log(model, X_train_t)
    pred_val_log = predict_log(model, X_val_t)
    pred_test_log = predict_log(model, X_test_t)
    selection_mse_log = float(mean_squared_error(y_val, pred_val_log))

    log_cap = float(np.max(y_train)) * 1.2
    pred_train_log = np.clip(pred_train_log, 0.0, log_cap)
    pred_val_log = np.clip(pred_val_log, 0.0, log_cap)
    pred_test_log = np.clip(pred_test_log, 0.0, log_cap)

    y_test_raw = np.expm1(y_test)
    pred_test_raw = np.expm1(pred_test_log)
    row = summarize_result(
        model_name=model_name, split_name=split_label, seed=seed,
        n_features=len(feature_cols), selection_mse_log=selection_mse_log, epochs_used=epochs_used,
        y_train_raw=np.expm1(y_train), pred_train_raw=np.expm1(pred_train_log),
        y_val_raw=np.expm1(y_val), pred_val_raw=np.expm1(pred_val_log),
        y_test_raw=y_test_raw, pred_test_raw=pred_test_raw,
    )
    gid_test = df.iloc[split_idx.test_idx]["GID"].to_numpy() if "GID" in df.columns else np.arange(len(y_test_raw))
    feat_importance = permutation_importance_tabular(model, X_val_t, y_val, feature_cols, cfg.device, seed=seed)
    return {"row": row, "params": params, "y_test_raw": y_test_raw, "pred_test_raw": pred_test_raw,
            "gid_test": gid_test, "loss_history": loss_history, "trials_df": trials_df,
            "feature_names": feature_cols, "feature_importance": feat_importance}


# ---------------------------------------------------------------------------
# Main benchmark-running function for tabular PyTorch models
# ---------------------------------------------------------------------------
def run_tabular_benchmark(
    cfg: BenchmarkConfig,
    model_name: str,
    build_model_fn: Callable[[int, dict[str, Any]], nn.Module],
    search_space_fn: Callable[[optuna.Trial], dict[str, Any]],
) -> dict[str, Path]:
    """Run the full benchmark pipeline for any tabular DL model.

    This is the central function that each model file (ANN, MLP, ...) will call.
    It performs the entire process from reading data to writing results.

    Detailed procedure:
        1. Check whether PyTorch can be imported. If not -> write "skipped" results.
        2. Read the CSV data.
        3. For each seed:
           a. Set the seed to ensure reproducibility.
           b. Split the data into train/val/test.
           c. Preprocess: scale, one-hot, take the log1p target.
           d. Convert to PyTorch tensors.
           e. Use Optuna to optimize hyperparameters:
              - Optuna is an automatic hyperparameter optimization library.
              - Each trial, Optuna proposes a set of parameters (lr, wd, number of layers, etc.).
              - Train the model with that parameter set, evaluate on val.
              - Optuna learns from previous trials to propose better parameters.
              - After n_trials, take the model with the lowest val MSE.
           f. IMPORTANT: Directly save the state_dict of the best model from Optuna,
              do NOT retrain from scratch. Reason: a new random weight initialization
              would give a different result, losing the best model already found.
           g. Predict on train/val/test (on the log scale).
           h. Clip predictions: bound predicted values within a reasonable range before
              converting back to the original scale with expm1. For example, if pred_log = 10
              then expm1(10) = 22026, while real data only reaches ~1000 mm/m (log ~6.9).
              Clipping at 120% of max(y_train) avoids meaningless exploding values.
           i. Convert back to the original scale with expm1 then compute metrics.
        4. Write all results to CSV and JSON files.

    Parameters:
        cfg             : A BenchmarkConfig object holding the full configuration.
        model_name      : Model name (e.g. "ANN", "ResMLP") recorded in the results.
        build_model_fn  : Model-building function, takes (num_features, param_dict) -> nn.Module.
        search_space_fn : Function defining the Optuna search space, takes (trial) -> param_dict.

    Returns:
        A dict mapping output file name -> the Path of each output file written.
    """
    # Check whether PyTorch is available
    if TORCH_IMPORT_ERROR is not None:
        return write_skip_outputs(cfg, model_name, build_torch_runtime_message(model_name, TORCH_IMPORT_ERROR))

    df = pd.read_csv(cfg.data_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    best_params: dict[str, dict[str, Any]] = {}

    for seed in cfg.seeds:
        set_seed(seed)  # Ensure reproducibility: data split, weight init, batch shuffling
        split_idx, split_label = make_split_indices(df, cfg.split_mode, seed)
        split_rows.append(split_diagnostics(df, split_idx, split_label, seed))
        try:
            result = _fit_eval_one_split(cfg, model_name, build_model_fn, search_space_fn, df, split_idx, split_label, seed)
        except Exception as exc:
            return write_skip_outputs(cfg, model_name, f"Loi khi train {model_name}. Loi goc: {exc}")
        summary_rows.append(result["row"])
        best_params[str(seed)] = result["params"]

    # Write all results to file
    return write_completed_outputs(cfg, model_name, summary_rows, split_rows, best_params)


# ---------------------------------------------------------------------------
# Group K-Fold Cross-Validation (addresses reviewer comment #2)
# ---------------------------------------------------------------------------
def run_tabular_kfold(
    cfg: BenchmarkConfig,
    model_name: str,
    build_model_fn: Callable[[int, dict[str, Any]], nn.Module],
    search_space_fn: Callable[[optuna.Trial], dict[str, Any]],
) -> dict[str, Path]:
    """Run GROUP K-FOLD cross-validation for a tabular DL model.

    Unlike run_tabular_benchmark (a single holdout split per seed), this function
    splits the data into cfg.n_splits folds by GID group. Each fold in turn serves as the
    test set once; the remainder is split into train/val and hyperparameters are tuned with Optuna.

    Output written:
        kfold_summary.csv     : one row per fold (train/val/test metrics).
        kfold_aggregate.csv   : mean +/- standard deviation ACROSS FOLDS.
        oof_predictions.csv   : out-of-fold predictions (each test GID appears once)
                                -- used for scatter & residual plots over the full dataset.
        split_summary.csv     : diagnostics of the split for each fold (checks GID overlap).
        best_params.json      : best hyperparameters for each fold.
        run_metadata.json     : metadata (config, status).

    Note: uses cfg.seeds[0] as the seed for splitting folds, for reproducibility.
    """
    if TORCH_IMPORT_ERROR is not None:
        return write_skip_outputs(cfg, model_name, build_torch_runtime_message(model_name, TORCH_IMPORT_ERROR))

    df = pd.read_csv(cfg.data_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.seeds[0] if cfg.seeds else 42
    set_seed(seed)

    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    best_params: dict[str, dict[str, Any]] = {}
    oof_frames: list[pd.DataFrame] = []
    # Label "GroupKFold" = group k-fold stratified by fc28 (StratifiedKFold on GID,
    # shuffled by seed), NOT the sklearn.GroupKFold class.
    split_label = f"GroupKFold{cfg.n_splits}"

    loss_history_rows: list[dict[str, Any]] = []
    trials_frames: list[pd.DataFrame] = []
    feat_imp_rows: list[dict[str, Any]] = []
    folds = group_kfold_split_indices(df, cfg.n_splits, seed)
    for split_idx, fold_id in folds:
        diag = split_diagnostics(df, split_idx, split_label, seed)
        diag["Fold"] = fold_id
        split_rows.append(diag)
        try:
            result = _fit_eval_one_split(cfg, model_name, build_model_fn, search_space_fn, df, split_idx, split_label, seed)
        except Exception as exc:
            return write_skip_outputs(cfg, model_name, f"Loi khi train {model_name} (fold {fold_id}). Loi goc: {exc}")
        row = result["row"]
        row["Fold"] = fold_id
        summary_rows.append(row)
        best_params[f"fold_{fold_id}"] = result["params"]
        # Gather out-of-fold predictions
        oof_frames.append(pd.DataFrame({
            "GID": result["gid_test"],
            "Fold": fold_id,
            "y_true": result["y_test_raw"],
            "y_pred": result["pred_test_raw"],
        }))
        # Collect per-epoch val_loss history of the best model for this fold (for the learning curve)
        for i, vloss in enumerate(result.get("loss_history", [])):
            loss_history_rows.append({"Fold": fold_id, "epoch": i + 1, "val_loss": vloss})
        # Collect Optuna trial history for this fold (for optimization history / param importance)
        trials_df_fold = result.get("trials_df")
        if trials_df_fold is not None and not trials_df_fold.empty:
            trials_df_fold = trials_df_fold.copy()
            trials_df_fold["Fold"] = fold_id
            trials_frames.append(trials_df_fold)
        # Collect feature importance (permutation) for this fold
        for fname, fval in zip(result.get("feature_names", []), result.get("feature_importance", [])):
            feat_imp_rows.append({"Fold": fold_id, "Feature": fname, "Importance": fval})

    # Write k-fold results
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    # Aggregate across folds: drop the Fold column from the mean/std computation for cleanliness
    aggregate_df = aggregate_results(summary_df.drop(columns=["Fold"], errors="ignore"))
    split_df = pd.DataFrame(split_rows)
    oof_df = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    loss_history_df = pd.DataFrame(loss_history_rows)
    trials_df = pd.concat(trials_frames, ignore_index=True) if trials_frames else pd.DataFrame()
    feat_imp_df = pd.DataFrame(feat_imp_rows)

    paths = {
        "summary": cfg.output_dir / "kfold_summary.csv",
        "aggregate": cfg.output_dir / "kfold_aggregate.csv",
        "oof": cfg.output_dir / "oof_predictions.csv",
        "split_summary": cfg.output_dir / "split_summary.csv",
        "best_params": cfg.output_dir / "best_params.json",
        "metadata": cfg.output_dir / "run_metadata.json",
        "loss_history": cfg.output_dir / "loss_history.csv",
        "optuna_trials": cfg.output_dir / "optuna_trials.csv",
        "feature_importance": cfg.output_dir / "feature_importance.csv",
    }
    summary_df.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(paths["aggregate"], index=False, encoding="utf-8-sig")
    oof_df.to_csv(paths["oof"], index=False, encoding="utf-8-sig")
    split_df.to_csv(paths["split_summary"], index=False, encoding="utf-8-sig")
    loss_history_df.to_csv(paths["loss_history"], index=False, encoding="utf-8-sig")
    trials_df.to_csv(paths["optuna_trials"], index=False, encoding="utf-8-sig")
    feat_imp_df.to_csv(paths["feature_importance"], index=False, encoding="utf-8-sig")
    paths["best_params"].write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name, "status": "completed", "evaluation": "group_kfold",
        "config": {"data_path": portable_path(cfg.data_path), "output_dir": portable_path(cfg.output_dir),
                   "split_mode": "group_kfold", "n_splits": cfg.n_splits, "seed": seed, "n_trials": cfg.n_trials,
                   "max_epochs": cfg.max_epochs, "patience": cfg.patience,
                   "dl_batch_size": cfg.dl_batch_size, "device": cfg.device},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


# =============================================================================
# PART 1b - PRINT RESULTS TO CONSOLE (merged from common_console_summary.py)
# =============================================================================

import json
from pathlib import Path
from typing import Any

import pandas as pd


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Safely read a CSV file, returning an empty DataFrame if the file doesn't exist or errors.

    This function checks whether the file exists before reading.
    If the file is missing or has a format error, returns an empty DataFrame
    instead of raising an exception, so the print flow isn't interrupted.

    Parameters:
        path: Path to the CSV file to read.

    Returns:
        pd.DataFrame: The CSV file's content, or an empty DataFrame on error.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _round_if_number(value: Any, digits: int = 4) -> Any:
    """Round a value if it's numeric, leave it unchanged otherwise.

    Used to format output nicely when printing to the console.
    String or None values are returned unchanged.

    Parameters:
        value: The value to check and round.
        digits: Number of decimal digits to keep (default 4).

    Returns:
        The rounded value if numeric, or the original value otherwise.
    """
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def print_model_console_summary(model_name: str, outputs: dict[str, Any]) -> None:
    """Print a summary of results and best parameters to the console for a quick read.

    This function takes the output dict returned by the run_benchmark function of each model file,
    then reads the aggregate CSV file and best_params JSON to extract information.

    The printed metrics include:
    - Average R2 on the train, validation, test sets
    - Average RMSE on the train, validation, test sets
    - Average MAE on the train, validation, test sets
    - Average number of epochs used and average number of features

    In addition, the function also prints the list of main output files so the user
    knows where the detailed results are saved.

    Parameters:
        model_name: Model name shown in the header (e.g. 'LSTM', 'XGBoost').
        outputs: Dict containing paths to output files such as 'aggregate',
                 'best_params', 'summary', 'split_summary', 'metadata'.
    """
    print("")
    print("=" * 72)
    print(f"RESULT SUMMARY - {model_name}")
    print("=" * 72)

    # Get the aggregate CSV file path and best_params JSON file path from the output dict.
    aggregate_path = Path(outputs["aggregate"]) if "aggregate" in outputs else None
    best_params_path = Path(outputs["best_params"]) if "best_params" in outputs else None

    # Read the aggregate CSV file with the average evaluation metrics.
    aggregate_df = safe_read_csv(aggregate_path) if aggregate_path else pd.DataFrame()
    if not aggregate_df.empty:
        # Take the first row (usually only 1 row) and print each metric.
        row = aggregate_df.iloc[0].to_dict()
        print("Main metrics:")
        for key in [
            "Train_R2_mean",
            "Val_R2_mean",
            "Test_R2_mean",
            "Train_RMSE_mean",
            "Val_RMSE_mean",
            "Test_RMSE_mean",
            "Train_MAE_mean",
            "Val_MAE_mean",
            "Test_MAE_mean",
            "epochs_used_mean",
            "n_features_mean",
        ]:
            if key in row:
                print(f"- {key}: {_round_if_number(row[key])}")
    else:
        print("Could not read results_aggregate.csv, or the file is empty.")

    # Read the best_params JSON file with the best parameter set for each seed.
    if best_params_path and best_params_path.exists():
        try:
            best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
        except Exception:
            best_params = {}
        if isinstance(best_params, dict) and best_params:
            print("Best parameters:")
            for seed, params in best_params.items():
                print(f"- Seed {seed}:")
                if isinstance(params, dict):
                    for name, value in params.items():
                        print(f"  - {name}: {_round_if_number(value)}")
                else:
                    print(f"  - {params}")
        else:
            print("Could not read best_params.json.")

    # Print the list of main output files so the user knows where results are saved.
    print("Main output files:")
    for key in ["summary", "aggregate", "split_summary", "best_params", "metadata"]:
        if key in outputs:
            print(f"- {key}: {outputs[key]}")
    print("=" * 72)
    print("")



def print_kfold_console_summary(model_name: str, outputs: dict[str, Any]) -> None:
    """Print a summary of GROUP K-FOLD results to the console (each fold + mean +/- std).

    Reads kfold_summary.csv (each row = 1 fold) and kfold_aggregate.csv (mean
    +/- standard deviation across folds), then prints to the console for a quick check
    without opening the file. This works similarly to print_model_console_summary above,
    but adds the standard deviation (_std) and a per-fold detail table.
    """
    print("")
    print("=" * 72)
    print(f"GROUP K-FOLD SUMMARY - {model_name}")
    print("=" * 72)

    summary_path = Path(outputs["summary"]) if "summary" in outputs else None
    aggregate_path = Path(outputs["aggregate"]) if "aggregate" in outputs else None

    summary_df = safe_read_csv(summary_path) if summary_path else pd.DataFrame()
    if not summary_df.empty:
        cols = [c for c in ["Fold", "Test_R2", "Test_RMSE", "Test_MAE", "epochs_used"] if c in summary_df.columns]
        if cols:
            print("Per-fold detail:")
            print(summary_df[cols].to_string(index=False))
    else:
        print("Could not read kfold_summary.csv or the file is empty.")

    aggregate_df = safe_read_csv(aggregate_path) if aggregate_path else pd.DataFrame()
    if not aggregate_df.empty:
        row = aggregate_df.iloc[0].to_dict()
        print("\nMean +/- standard deviation across folds:")
        for base in ["Train_R2", "Val_R2", "Test_R2", "Train_RMSE", "Val_RMSE", "Test_RMSE",
                     "Train_MAE", "Val_MAE", "Test_MAE"]:
            mean_key, std_key = f"{base}_mean", f"{base}_std"
            if mean_key in row:
                mean_v = _round_if_number(row[mean_key])
                std_v = _round_if_number(row.get(std_key))
                print(f"- {base}: {mean_v} +/- {std_v}")
        if "epochs_used_mean" in row:
            print(f"- epochs_used: {_round_if_number(row['epochs_used_mean'])}")
        if "n_features_mean" in row:
            print(f"- n_features: {_round_if_number(row['n_features_mean'])}")
    else:
        print("Could not read kfold_aggregate.csv or the file is empty.")

    print("Main output files:")
    for key in ["summary", "aggregate", "oof", "split_summary", "best_params", "metadata"]:
        if key in outputs:
            print(f"- {key}: {outputs[key]}")
    print("=" * 72)
    print("")


# =============================================================================
# PART 2 - DATA PREPARATION (merged from buoc_01_code_loc_data.py)
# =============================================================================
from dataclasses import dataclass, field
from pathlib import Path

# List of columns to exclude when training the model.
# These are identifier columns (GID), the raw time column (dt_days),
# the target column (y_shrinkage_total_mm_per_m), the transformed target column (y_log1p),
# and the sequence-marker column (seq_ok) -- not input features.
EXCLUDE_COLUMNS = ['GID', 'dt_days', 'y_shrinkage_total_mm_per_m', 'y_log1p', 'seq_ok']

# Default list of Deep Learning models supported by the package.
# Includes both point-wise models (ANN, MLP, ResMLP, FT_Transformer, TabNet)
# and sequence models (GRU, LSTM).
DEFAULT_DL_MODELS = ['ANNBasic', 'MLP_Tapered', 'ResMLP', 'FT_Transformer', 'TabNet', 'GRU', 'LSTM']
DEFAULT_MODELS = DEFAULT_DL_MODELS

@dataclass(slots=True)
class DataPipelineConfig:
    """Configuration for the data-filtering pipeline.

    Contains all parameters controlling the filtering process, including:
    - Source Excel file name and output file names
    - Monotonic-curve filter thresholds (drop_curve_if_any_decrease, eps_y_decrease)
    - Minimum points per curve (min_points_per_curve, min_points_for_seq)
    - Whether to median-impute non-core columns (median_impute_noncore)
    - List of columns that must have a value (core_required)

    The core_required columns include:
    - dt_days: time elapsed since measurement started (days)
    - y_shrinkage_total_mm_per_m: total shrinkage value (mm/m) -- this is the target
    - w_cm: water/cement ratio
    - cm: cement content (kg/m3)
    - fc28: 28-day compressive strength (MPa)
    - V_S: volume/surface-area ratio (mm)
    - RH_test: relative humidity during the test
    - T: temperature during the test (deg C)
    - t0: age of the concrete when shrinkage measurement started (days)
    """
    target_xlsx_name: str = 'Database SCC.xlsx'
    out_filtered: str = 'data_train_total_filtered.csv'
    out_point: str = 'data_train_total_ready.csv'
    out_seq: str = 'data_train_total_ready_seq.csv'
    out_report: str = 'build_report.json'
    drop_curve_if_any_decrease: bool = True
    eps_y_decrease: float = 0.0
    min_points_per_curve: int = 2
    min_points_for_seq: int = 5
    # DEFAULT False: do NOT impute at build time to avoid data leakage.
    # Median filling is moved to experiment time (fit median ONLY on the train fold)
    # in common_tabular.impute_train_median / common_sequence_group.build_feature_frames.
    # NaN values are kept in the CSV so the pipeline can impute correctly per fold.
    median_impute_noncore: bool = False
    core_required: list[str] = field(default_factory=lambda: ['dt_days', 'y_shrinkage_total_mm_per_m', 'w_cm', 'cm', 'fc28', 'V_S', 'RH_test', 'T', 't0'])

@dataclass(slots=True)
class ExperimentConfig:
    """Configuration for a benchmark experiment.

    Contains parameters controlling the model training and evaluation process:
    - data_path: path to the filtered CSV data file
    - output_dir: directory to write results
    - split_mode: how to split the data (e.g. 'random', 'group')
    - seeds: list of seeds to repeat the experiment, ensuring reproducibility
    - models: list of model names to run
    - dl_training_style: DL training style ('standard' or custom)
    - dl_report_mode: reporting mode ('fair' uses the best epoch on val)
    - dl_preprocessing: how to preprocess the data for DL
    - n_trials: number of hyperparameter optimization trials (Optuna)
    - max_epochs: maximum epochs during training
    - patience: number of epochs to wait before early stopping
    - dl_batch_size: batch size when training DL
    - device: compute device ('cpu' or 'cuda')
    """
    data_path: Path
    output_dir: Path
    split_mode: str
    seeds: list[int] = field(default_factory=lambda: [42])
    models: list[str] = field(default_factory=lambda: DEFAULT_MODELS.copy())
    dl_training_style: str = 'standard'
    dl_report_mode: str = 'fair'
    dl_preprocessing: str = 'standard'
    n_trials: int = 25
    max_epochs: int = 300
    patience: int = 25
    dl_batch_size: int = 256
    device: str = 'cpu'
import json
from pathlib import Path
import numpy as np
import pandas as pd

def identical_numeric(a: pd.Series, b: pd.Series, tol: float=1e-09) -> bool:
    """Check whether two numeric columns have identical values.

    Used to compare column T (test temperature) and Tcur (curing temperature).
    If these two columns are identical, drop Tcur to avoid duplicate information
    (multicollinearity) when feeding into the model.

    Parameters:
        a: first Series column to compare.
        b: second Series column to compare.
        tol: allowed tolerance when comparing (default 1e-9).

    Returns:
        True if all numeric values in both columns are equal (within tol).
    """
    a_num = pd.to_numeric(a, errors='coerce')
    b_num = pd.to_numeric(b, errors='coerce')
    mask = a_num.notna() & b_num.notna()
    if not mask.any():
        return False
    return bool(((a_num[mask] - b_num[mask]).abs() <= tol).all())

def build_datasets(xlsx_path: Path, output_dir: Path, cfg: DataPipelineConfig | None=None) -> dict[str, Path]:
    """Main function of the data-build step.

    This is the central function that runs the entire data-filtering pipeline.
    The input is the path to the original Excel workbook; the output is filtered,
    normalized CSV files ready for the model, along with a
    build_report.json file recording the filtering process in detail.

    The pipeline consists of these steps:
    1. Read two sheets from Excel: 'Shrinkage Info' and 'Shrinkage Data'
    2. Apply the metadata filter to the Info sheet
    3. Merge the two tables based on the 'File' column
    4. Normalize column names and create new features
    5. Remove duplicates, remove short curves, remove non-monotonic curves
    6. Drop rows missing core columns, impute median for non-core columns
    7. Write out 4 output files

    Parameters:
        xlsx_path: Path to the original Excel file (Database SCC.xlsx).
        output_dir: Directory to write the output CSV and report files.
        cfg: Pipeline config; defaults are used if None.

    Returns:
        A dict with paths to 4 output files: 'filtered', 'point', 'seq', 'report'.
    """
    cfg = cfg or DataPipelineConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    # === STEP 1: Read the two data sheets from the original Excel workbook ===
    # The 'Shrinkage Info' sheet holds the concrete mix-design info (one row = 1 sample/curve).
    # The 'Shrinkage Data' sheet holds the shrinkage-over-time measurements (one row = 1 data point).
    info = pd.read_excel(xlsx_path, sheet_name='Shrinkage Info')
    data = pd.read_excel(xlsx_path, sheet_name='Shrinkage Data')

    # Record the number of curves and raw data points before filtering.
    raw_curve_count = int(info['File'].nunique()) if 'File' in info.columns else int(len(info))
    raw_point_count = int(len(data))

    # Normalize the File column (strip extra whitespace) for accurate merging.
    info['File'] = info['File'].astype(str).str.strip()
    data['File'] = data['File'].astype(str).str.strip()
    if 'type' in info.columns:
        info['type'] = info['type'].astype(str).str.strip().str.lower()

    # === STEP 2: Filter metadata according to the finalized policy ===
    # Only keep samples that satisfy ALL of the following conditions:
    # - type == 'total': only total shrinkage (not autogenous or drying shrinkage separately)
    # - Curing == 3: standard curing regime (moist curing)
    # - Not s_036: excludes the s_036 sample group due to abnormal data
    # - RH_test >= 0.4: relative humidity during the test must be >= 40%
    # - Coarse_agg > 0 and Fine_agg > 0: must have both coarse and fine aggregate
    # - fc28 in [17, 110]: 28-day compressive strength within a reasonable range (MPa)
    # - Tcur in [20, 30]: curing temperature within the standard range (deg C)
    # - V/S > 11.3: volume/surface-area ratio large enough (mm)
    # - Cem_type == 1: only Type I Portland cement
    mask = (info['type'] == 'total') & (info['Curing'] == 3) & ~info['File'].str.startswith('s_036', na=False) & (pd.to_numeric(info['RH_test'], errors='coerce') >= 0.4) & (pd.to_numeric(info['Coarse_agg'], errors='coerce') > 0) & (pd.to_numeric(info['Fine_agg'], errors='coerce') > 0) & pd.to_numeric(info['fc28'], errors='coerce').between(17, 110, inclusive='both') & pd.to_numeric(info['Tcur'], errors='coerce').between(20, 30, inclusive='both') & (pd.to_numeric(info['V/S'], errors='coerce') > 11.3) & (pd.to_numeric(info['Cem_type'], errors='coerce') == 1)
    info_keep = info.loc[mask].copy()

    # Filter the Data table: keep only rows whose File is in the filtered list from Info.
    data_keep = data[data['File'].isin(info_keep['File'])].copy()

    # Record the count after metadata filtering.
    kept_curve_count = int(info_keep['File'].nunique()) if 'File' in info_keep.columns else int(len(info_keep))
    kept_point_count = int(len(data_keep))
    s036_total = int(info['File'].astype(str).str.startswith('s_036', na=False).sum())
    s036_kept = int(info_keep['File'].astype(str).str.startswith('s_036', na=False).sum())

    # === STEP 3: Determine required columns and check required columns exist ===
    # Required columns from the Info table: File, ID, w/cm, cm, fc28, V/S, RH_test, T, Tcur, t0.
    info_base_cols = ['File', 'ID', 'w/cm', 'cm', 'fc28', 'V/S', 'RH_test', 'T', 'Tcur', 't0']
    # Required columns from the Data table: File, dt (days), Shrinkage strain (mm/m).
    data_base_cols = ['File', 'dt (days)', 'Shrinkage strain (mm/m)']
    # Optional columns -- may or may not be present in the workbook.
    optional_info_cols = ['FlyAsh', 'Slag', 'Filler', 'SF', 'water', 'Fine_agg', 'Coarse_agg', 'agg_total', 'sandrate', 'agg_Dmax', 'Super (kg)', 'VA(kg)', 'AEA (l)', 'Shape_factor', 'r', 'h', 'a/cm', 'agg_type', 'region_type']

    # Check whether the required columns exist in the data.
    missing_info = [col for col in info_base_cols if col not in info_keep.columns]
    missing_data = [col for col in data_base_cols if col not in data_keep.columns]
    if missing_info:
        raise KeyError(f"Missing required columns in 'Shrinkage Info': {missing_info}")
    if missing_data:
        raise KeyError(f"Missing required columns in 'Shrinkage Data': {missing_data}")

    # === STEP 4: Merge the two tables and normalize columns ===
    # Merge the Data table with the Info table based on the 'File' column (inner join).
    info_cols = info_base_cols + [col for col in optional_info_cols if col in info_keep.columns]
    merged = data_keep[data_base_cols].merge(info_keep[info_cols], on='File', how='inner')

    # Check whether columns T and Tcur are identical.
    # If identical, drop Tcur to avoid duplicate information (multicollinearity).
    drop_tcur = False
    if 'T' in merged.columns and 'Tcur' in merged.columns:
        drop_tcur = identical_numeric(merged['T'], merged['Tcur'])

    # Build the output DataFrame with normalized column names.
    # The GID column is the unique key for each curve (File + ID), used for groupby.
    out = pd.DataFrame()
    file_part = merged['File'].astype(str).str.strip()
    id_part = pd.to_numeric(merged['ID'], errors='coerce').fillna(-1).astype(int)
    out['GID'] = file_part + '_' + id_part.astype(str)

    # dt_days column: time elapsed since shrinkage measurement started (days).
    out['dt_days'] = pd.to_numeric(merged['dt (days)'], errors='coerce')
    # y_shrinkage_total_mm_per_m column: total shrinkage value (mm/m) -- absolute value taken.
    out['y_shrinkage_total_mm_per_m'] = pd.to_numeric(merged['Shrinkage strain (mm/m)'], errors='coerce').abs()

    # Main feature columns from the Info table, renamed to fit Python conventions.
    out['w_cm'] = pd.to_numeric(merged['w/cm'], errors='coerce')
    out['cm'] = pd.to_numeric(merged['cm'], errors='coerce')
    out['fc28'] = pd.to_numeric(merged['fc28'], errors='coerce')
    out['V_S'] = pd.to_numeric(merged['V/S'], errors='coerce')
    out['RH_test'] = pd.to_numeric(merged['RH_test'], errors='coerce')
    out['T'] = pd.to_numeric(merged['T'], errors='coerce')
    out['t0'] = pd.to_numeric(merged['t0'], errors='coerce')

    # Only keep Tcur if it differs from T (avoids multicollinearity).
    if not drop_tcur and 'Tcur' in merged.columns:
        out['Tcur'] = pd.to_numeric(merged['Tcur'], errors='coerce')

    # Add optional columns (if present in the workbook).
    def add_optional(col_src: str, col_out: str | None=None) -> None:
        """Add an optional column to the output DataFrame if it exists in the merged data."""
        target = col_out or col_src
        if col_src in merged.columns:
            out[target] = pd.to_numeric(merged[col_src], errors='coerce')
    add_optional('FlyAsh')
    add_optional('Slag')
    add_optional('Filler')
    add_optional('SF')
    add_optional('water')
    add_optional('Fine_agg')
    add_optional('Coarse_agg')
    add_optional('agg_total')
    add_optional('sandrate')
    add_optional('agg_Dmax')
    add_optional('Super (kg)', 'Super_kg')
    add_optional('VA(kg)', 'VA_kg')
    add_optional('AEA (l)', 'AEA_l')
    add_optional('Shape_factor')
    add_optional('r')
    add_optional('h')
    add_optional('a/cm', 'a_cm')
    add_optional('agg_type')
    add_optional('region_type')

    # === Create new features (feature engineering) ===
    # dt_log1p: log(1 + dt_days) -- the log transform helps the model handle
    # the right-skewed distribution of dt_days better.
    out['dt_log1p'] = np.log1p(out['dt_days'].astype(float))
    # age_days: the concrete's actual age = t0 + dt_days.
    out['age_days'] = out['t0'].astype(float) + out['dt_days'].astype(float)
    # age_log1p: log(1 + age_days) -- similar to dt_log1p.
    out['age_log1p'] = np.log1p(out['age_days'].clip(lower=0.0))

    # Compute the total supplementary cementitious material (SCM) and OPC (pure Portland cement) ratio.
    scm_cols = [col for col in ['FlyAsh', 'Slag', 'Filler', 'SF'] if col in out.columns]
    if scm_cols:
        out['SCM_sum'] = out[scm_cols].sum(axis=1, skipna=True)
        out['OPC_pct'] = 100.0 - out['SCM_sum']
        # Compute the mass (kg) of each supplementary component based on the cm content.
        if 'cm' in out.columns:
            for col in scm_cols:
                out[f'{col}_kg'] = out['cm'] * (out[col] / 100.0)
            out['OPC_kg'] = out['cm'] * (out['OPC_pct'] / 100.0)

    # Compute the fine-aggregate/cement and coarse-aggregate/cement ratios.
    if 'Fine_agg' in out.columns and 'cm' in out.columns:
        out['FineAgg_cm'] = out['Fine_agg'] / out['cm']
    if 'Coarse_agg' in out.columns and 'cm' in out.columns:
        out['CoarseAgg_cm'] = out['Coarse_agg'] / out['cm']

    # === STEP 5: Remove duplicates, short curves, and non-monotonic curves ===

    # Remove duplicate records (same GID and dt_days), keeping the first record.
    out = out.drop_duplicates(subset=['GID', 'dt_days'], keep='first').copy()
    out = out.sort_values(['GID', 'dt_days']).reset_index(drop=True)
    dedup_curve_count = int(out['GID'].nunique())
    dedup_point_count = int(len(out))

    # Remove curves with too few data points (fewer than min_points_per_curve).
    # A curve that's too short doesn't have enough information for the model to learn the shrinkage trend.
    counts = out.groupby('GID')['dt_days'].size()
    good_gids = set(counts[counts >= cfg.min_points_per_curve].index)
    out = out[out['GID'].isin(good_gids)].copy()
    min_curve_count = int(out['GID'].nunique())
    min_point_count = int(len(out))
    dropped_for_min_points = int(dedup_curve_count - min_curve_count)

    # Remove curves that violate monotonicity.
    # Total shrinkage should theoretically increase monotonically over time.
    # If y decreases as dt increases, this could be a measurement error or abnormal data.
    removed_nonmonotonic = 0
    if cfg.drop_curve_if_any_decrease:
        tmp = out[['GID', 'dt_days', 'y_shrinkage_total_mm_per_m']].copy()
        tmp = tmp.sort_values(['GID', 'dt_days'], kind='mergesort')
        tmp['d_dt'] = tmp.groupby('GID')['dt_days'].diff()
        tmp['d_y'] = tmp.groupby('GID')['y_shrinkage_total_mm_per_m'].diff()
        # Flag steps where dt increases but y decreases (violates monotonicity).
        bad_step = (tmp['d_dt'] > 0) & (tmp['d_y'] < -cfg.eps_y_decrease)
        bad_gids = tmp.loc[bad_step, 'GID'].unique()
        removed_nonmonotonic = int(len(bad_gids))
        out = out[~out['GID'].isin(bad_gids)].copy()
        out = out.sort_values(['GID', 'dt_days']).reset_index(drop=True)
    monotonic_curve_count = int(out['GID'].nunique())
    monotonic_point_count = int(len(out))

    # Flag which curves have enough points for the sequence model (>= min_points_for_seq).
    # The GRU/LSTM models need long-enough curves to learn the pattern over time.
    counts_seq = out.groupby('GID')['dt_days'].size()
    seq_ok = set(counts_seq[counts_seq >= cfg.min_points_for_seq].index)
    out['seq_ok'] = out['GID'].isin(seq_ok).astype(int)
    seq_ok_curve_count = int(out.loc[out['seq_ok'] == 1, 'GID'].nunique())

    # === STEP 6: Handle missing values ===

    # Cast required core columns to numeric type.
    for col in cfg.core_required:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    # Remove rows missing a value in a core (required) column.
    # Core columns are essential information and cannot be imputed.
    core_present = [col for col in cfg.core_required if col in out.columns]
    before_dropna_points = int(len(out))
    before_dropna_curves = int(out['GID'].nunique())
    out = out.dropna(subset=core_present).copy()
    after_dropna_points = int(len(out))
    after_dropna_curves = int(out['GID'].nunique())

    # Impute median for non-core (optional) columns if enabled.
    # Non-core columns may have missing values; the median is used to fill them in,
    # keeping the model from losing data while remaining statistically reasonable.
    imputed_columns: list[str] = []
    if cfg.median_impute_noncore:
        num_cols = [col for col in out.columns if pd.api.types.is_numeric_dtype(out[col])]
        noncore = [col for col in num_cols if col not in core_present and col != 'seq_ok']
        if noncore:
            med = out[noncore].median(numeric_only=True)
            out[noncore] = out[noncore].fillna(med)
            imputed_columns = sorted(noncore)

    # Re-sort by GID and dt_days, reset index.
    out = out.sort_values(['GID', 'dt_days']).reset_index(drop=True)
    final_curve_count = int(out['GID'].nunique())
    final_point_count = int(len(out))

    # === STEP 7: Write output files ===

    filtered_path = output_dir / cfg.out_filtered
    point_path = output_dir / cfg.out_point
    seq_path = output_dir / cfg.out_seq
    report_path = output_dir / cfg.out_report

    # File 1: data_train_total_filtered.csv - the fully filtered data.
    out.to_csv(filtered_path, index=False, encoding='utf-8-sig')

    # File 2: data_train_total_ready.csv - adds the y_log1p column for the point-wise model.
    # y_log1p = log(1 + y) makes the target distribution closer to normal, improving training.
    point_ready = out.copy()
    point_ready['y_log1p'] = np.log1p(pd.to_numeric(point_ready['y_shrinkage_total_mm_per_m'], errors='coerce').clip(lower=0.0))
    point_ready.to_csv(point_path, index=False, encoding='utf-8-sig')

    # File 3: data_train_total_ready_seq.csv - only curves long enough for the sequence model.
    seq_ready = out[out['seq_ok'] == 1].copy()
    seq_ready.to_csv(seq_path, index=False, encoding='utf-8-sig')

    # File 4: build_report.json - a detailed report of the filtering process.
    # Records the filter policy, sample counts at each step, and imputation info
    # to ensure reproducibility and traceability when needed.
    report = {'xlsx_path': str(xlsx_path), 'filter_policy': {'type': 'total', 'curing_equals': 3, 'exclude_file_prefix': 's_036', 'rh_test_min': 0.4, 'coarse_agg_gt': 0, 'fine_agg_gt': 0, 'fc28_range': [17, 110], 'tcur_range': [20, 30], 'v_s_gt': 11.3, 'cem_type_equals': 1, 'keep_s036': False, 'drop_curve_if_any_decrease': cfg.drop_curve_if_any_decrease, 'eps_y_decrease': cfg.eps_y_decrease, 'min_points_per_curve': cfg.min_points_per_curve, 'min_points_for_seq': cfg.min_points_for_seq}, 'counts': {'raw_curves': raw_curve_count, 'raw_points': raw_point_count, 'after_metadata_filter_curves': kept_curve_count, 'after_metadata_filter_points': kept_point_count, 'after_drop_duplicates_curves': dedup_curve_count, 'after_drop_duplicates_points': dedup_point_count, 'after_min_points_curves': min_curve_count, 'after_min_points_points': min_point_count, 'after_monotonicity_curves': monotonic_curve_count, 'after_monotonicity_points': monotonic_point_count, 'before_dropna_curves': before_dropna_curves, 'before_dropna_points': before_dropna_points, 'after_dropna_curves': after_dropna_curves, 'after_dropna_points': after_dropna_points, 'final_curves': final_curve_count, 'final_points': final_point_count, 'seq_ok_curves': seq_ok_curve_count, 'seq_ok_points': int(len(seq_ready))}, 'drops': {'s036_rows_in_info': s036_total, 's036_rows_kept_after_filter': s036_kept, 'dropped_for_min_points_curves': dropped_for_min_points, 'removed_nonmonotonic_curves': removed_nonmonotonic, 'dropped_for_missing_core_points': before_dropna_points - after_dropna_points}, 'imputation': {'median_impute_noncore': cfg.median_impute_noncore, 'imputed_columns': imputed_columns}, 'notes': ['This report documents the current canonical filtering policy.', 'Curing and Cem_type are hard-filtered in the current policy to keep the benchmark within the supported data regime.', 's_036 is excluded in the current policy.', 'Train-only preprocessing is enforced later at experiment time, not during dataset build.', 'Median imputation of non-core columns is now disabled at build time (NaN are kept); imputation is performed per train-fold inside the model pipeline to avoid preprocessing leakage.']}
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return {'filtered': filtered_path, 'point': point_path, 'seq': seq_path, 'report': report_path}
from pathlib import Path


# =============================================================================
# PART 3 - Addition that DOES NOT CLASH IN NAME, from common_sequence_group.py: allows running
# run_sequence_kfold() (GRU/LSTM) again alongside run_tabular_kfold() (ResMLP)
# within THE SAME FILE.
# =============================================================================
from torch.utils.data import Dataset, DataLoader

SEQ_ONE_HOT_COLUMNS = DL_ONE_HOT_COLUMNS


@dataclass(slots=True)
class SequenceBenchmarkConfig:
    """Configuration for one run of a sequence model benchmark.

    This is a dataclass containing all the parameters needed to run an experiment.
    The user just needs to create a SequenceBenchmarkConfig object then pass
    it into the run_sequence_benchmark() function.

    Parameters:
        data_path: Path to the CSV file with preprocessed input data.
        output_dir: Directory to save results (CSV, JSON) after the run finishes.
        model_name: Model name (e.g. "LSTM", "GRU", "Transformer"),
            used when writing the results report.
        seeds: List of random seeds. Each seed produces a different data split
            and training run, helping evaluate the model's stability.
            Default is [42].
        n_trials: Number of hyperparameter trials via Optuna.
            Optuna will run n_trials times, trying a different parameter set each time,
            then choose the best one. Default is 1 (no tuning, runs fast).
        max_epochs: Maximum number of epochs for each training run. An epoch is one
            pass through the entire training set. Default is 10.
        patience: Number of consecutive epochs with no val loss improvement before stopping early
            (early stopping). Helps avoid overfitting. Default is 3.
        batch_size: Number of sequences per batch during training.
            A larger batch is faster but uses more RAM. Default is 16.
        device: Compute device ("cpu" or "cuda"). Default is "cpu".
    """

    data_path: Path
    output_dir: Path
    model_name: str
    seeds: list[int] = field(default_factory=lambda: [42])
    n_trials: int = 1
    max_epochs: int = 10
    patience: int = 3
    batch_size: int = 16
    device: str = "cpu"
    # Number of folds for group k-fold cross-validation (only used by run_sequence_kfold).
    n_splits: int = 5



def write_skip_outputs_seq(cfg: SequenceBenchmarkConfig, reason: str) -> dict[str, Path]:
    """Writes empty output files when the benchmark is skipped.

    When an error occurs (e.g. insufficient memory, invalid data),
    this function creates empty CSV/JSON files so the pipeline doesn't break.
    The metadata file records the skip reason (skip_reason) so the user
    can check it later.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.output_dir / "results_summary.csv"
    aggregate_path = cfg.output_dir / "results_aggregate.csv"
    split_summary_path = cfg.output_dir / "split_summary.csv"
    best_params_path = cfg.output_dir / "best_params.json"
    metadata_path = cfg.output_dir / "run_metadata.json"
    summary_path.write_text("", encoding="utf-8")
    aggregate_path.write_text("", encoding="utf-8")
    split_summary_path.write_text("", encoding="utf-8")
    best_params_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": cfg.model_name,
        "status": "skipped",
        "skip_reason": reason,
        "data_path": portable_path(cfg.data_path),
        "output_dir": portable_path(cfg.output_dir),
        "seeds": cfg.seeds,
        "n_trials": cfg.n_trials,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "batch_size": cfg.batch_size,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "summary": summary_path,
        "aggregate": aggregate_path,
        "split_summary": split_summary_path,
        "best_params": best_params_path,
        "metadata": metadata_path,
    }



def build_feature_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Builds the feature matrix for each set.

    Procedure:
    1. Removes the columns in EXCLUDE_COLUMNS (target, GID, etc.).
    2. Normalizes (StandardScaler) the numeric columns: fit ONLY on train,
       then transform both val and test. This is the golden rule to avoid
       data leakage - val/test must not influence the fitting process.
    3. One-hot encoding for categorical columns. Ensures val/test have the same
       columns as train (reindex), filling 0 for missing values.

    Returns: (train_feat, val_feat, test_feat, list_of_column_names)
    """
    feature_cols = [col for col in train_df.columns if col not in EXCLUDE_COLUMNS]
    categorical_cols = [col for col in SEQ_ONE_HOT_COLUMNS if col in feature_cols]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    # Fill missing values (NaN) using the median computed ONLY on train (avoids leakage).
    # Imputation was moved from the data-build step to here to guarantee train-only.
    if numeric_cols:
        med = train_df[numeric_cols].median(numeric_only=True).fillna(0.0)
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        train_df[numeric_cols] = train_df[numeric_cols].fillna(med)
        val_df[numeric_cols] = val_df[numeric_cols].fillna(med)
        test_df[numeric_cols] = test_df[numeric_cols].fillna(med)

    # Normalize numeric columns: fit on train, transform on all
    scaler = StandardScaler()
    train_num = pd.DataFrame(
        scaler.fit_transform(train_df[numeric_cols]),
        columns=numeric_cols,
        index=train_df.index,
    ) if numeric_cols else pd.DataFrame(index=train_df.index)
    val_num = pd.DataFrame(
        scaler.transform(val_df[numeric_cols]),
        columns=numeric_cols,
        index=val_df.index,
    ) if numeric_cols else pd.DataFrame(index=val_df.index)
    test_num = pd.DataFrame(
        scaler.transform(test_df[numeric_cols]),
        columns=numeric_cols,
        index=test_df.index,
    ) if numeric_cols else pd.DataFrame(index=test_df.index)

    # One-hot encoding for categorical columns
    if categorical_cols:
        train_cat = pd.get_dummies(train_df[categorical_cols].astype("category"), prefix=categorical_cols, dtype=np.float32)
        val_cat = pd.get_dummies(val_df[categorical_cols].astype("category"), prefix=categorical_cols, dtype=np.float32)
        test_cat = pd.get_dummies(test_df[categorical_cols].astype("category"), prefix=categorical_cols, dtype=np.float32)
        # Ensure val/test have the same columns as train, fill 0 for missing columns
        val_cat = val_cat.reindex(columns=train_cat.columns, fill_value=0.0)
        test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0.0)
    else:
        train_cat = pd.DataFrame(index=train_df.index)
        val_cat = pd.DataFrame(index=val_df.index)
        test_cat = pd.DataFrame(index=test_df.index)

    # Concatenate numeric and categorical columns into one complete feature matrix
    train_feat = pd.concat([train_num, train_cat], axis=1).astype(np.float32)
    val_feat = pd.concat([val_num, val_cat], axis=1).astype(np.float32)
    test_feat = pd.concat([test_num, test_cat], axis=1).astype(np.float32)

    return train_feat, val_feat, test_feat, list(train_feat.columns)


def _group_to_sequences(df: pd.DataFrame, features_df: pd.DataFrame, use_log1p: bool = True) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Converts a DataFrame into a list of (X, y) sequences per GID.

    Each GID forms a time series: rows are sorted by dt_days
    (measurement day), forming a sequence of time steps.

    log1p transform for the target:
        If use_log1p=True (default), the target y is converted to log1p:
            y_transformed = log(1 + clip(y, 0))

        Why is log1p needed?
        - The shrinkage value has a right-skewed distribution:
          most values are small, a few values are very large.
        - log1p compresses the value range, making the gradient more stable during training.
        - Without log1p, large values would produce very large gradients, potentially
          preventing the model from converging (exploding gradient).
        - clip(y, 0) ensures there are no negative values before taking the log.
        - Matches the tabular pipeline to ensure the same target transform.

        When computing the final metric, expm1 (the inverse of log1p) must be used
        to convert back to the original scale (mm/m).
    """
    sequences_x: list[np.ndarray] = []
    sequences_y: list[np.ndarray] = []
    gids: list[str] = []
    ordered = df.copy()
    # Create a __row_id__ column to map between the original df and features_df after sorting
    ordered["__row_id__"] = np.arange(len(ordered))
    # Sort by GID and dt_days to create the time series in the correct order
    ordered = ordered.sort_values(["GID", "dt_days"]).reset_index(drop=True)
    feature_lookup = features_df.copy()
    feature_lookup["__row_id__"] = np.arange(len(feature_lookup))
    # Merge features by __row_id__ to keep the correct order after sorting
    ordered_features = ordered[["__row_id__"]].merge(feature_lookup, on="__row_id__", how="left").drop(columns="__row_id__")
    for gid, group_idx in ordered.groupby("GID").groups.items():
        idx = list(group_idx)
        x = ordered_features.iloc[idx].to_numpy(dtype=np.float32)
        # Use log1p(clip(y, 0)) instead of the raw value so the loss is smaller, the model learns better.
        # Use clip(lower=0) matching the tabular pipeline to ensure the same target transform.
        y_raw = ordered.iloc[idx]["y_shrinkage_total_mm_per_m"].to_numpy(dtype=np.float32)
        y = np.log1p(np.clip(y_raw, 0.0, None)) if use_log1p else y_raw
        sequences_x.append(x)
        sequences_y.append(y)
        gids.append(gid)
    return sequences_x, sequences_y, gids


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequence data.

    This is a wrapper class so the PyTorch DataLoader can fetch each sequence
    by index. Each element is a pair (X, y) with:
        - X: 2D array (n_time_steps, n_features)
        - y: 1D array (n_time_steps,) containing the target at each step
    """

    def __init__(self, sequences_x: list[np.ndarray], sequences_y: list[np.ndarray]) -> None:
        self.sequences_x = sequences_x
        self.sequences_y = sequences_y

    def __len__(self) -> int:
        return len(self.sequences_x)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.sequences_x[idx], self.sequences_y[idx]


def collate_sequences(batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combines multiple sequences of different lengths into a batch via padding and mask.

    Problem: sequences in the same batch can have different lengths
    (e.g. GID A has 5 steps, GID B has 8 steps). PyTorch requires the tensor
    to have a fixed size in each dimension.

    Solution - Padding and Mask:
    1. Padding: Add zeros to the end of shorter sequences so all of them have the same length
       as the longest sequence in the batch. This 0 value is fake, has no meaning.
    2. Mask: Create a binary (0/1) tensor marking which positions are real data (1)
       and which positions are padding (0). When computing loss, only compute on positions where mask=1.

    Returns:
        - batch_x: tensor (batch_size, max_len, feat_dim) - padded features
        - batch_y: tensor (batch_size, max_len) - padded target
        - batch_mask: tensor (batch_size, max_len) - mask marking real positions
    """
    xs, ys = zip(*batch)
    # Find the longest length in the batch to pad the shorter sequences
    max_len = max(item.shape[0] for item in xs)
    feat_dim = xs[0].shape[1]
    # Initialize with 0: padding positions will automatically be 0
    batch_x = np.zeros((len(xs), max_len, feat_dim), dtype=np.float32)
    batch_y = np.zeros((len(xs), max_len), dtype=np.float32)
    batch_mask = np.zeros((len(xs), max_len), dtype=np.float32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        length = x.shape[0]
        # Only fill real data into the beginning part, leave the rest as 0
        batch_x[i, :length, :] = x
        batch_y[i, :length] = y
        # Mark positions with real data = 1.0
        batch_mask[i, :length] = 1.0
    return (
        torch.tensor(batch_x, dtype=torch.float32),
        torch.tensor(batch_y, dtype=torch.float32),
        torch.tensor(batch_mask, dtype=torch.float32),
    )


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes Mean Squared Error only on positions with real data (mask=1).

    Why is a mask needed?
    Because sequences in a batch have different lengths, they have been padded with 0.
    If MSE were computed on padding positions too, the model would be "penalized" for predicting
    something other than 0 at those padding positions - this is meaningless and would bias
    the training process. The mask helps skip padding positions, computing loss only on
    real data.

    Formula: sum((pred - target)^2 * mask) / sum(mask)
    clamp_min(1.0) avoids dividing by 0 when there are no valid positions.
    """
    diff = (pred - target) ** 2
    # Multiply by mask: padding positions (mask=0) will have loss = 0
    diff = diff * mask
    # Divide by the total mask (number of real positions) instead of the total number of elements
    return diff.sum() / mask.sum().clamp_min(1.0)


def evaluate_sequence_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the model over the whole DataLoader and collects predictions.

    Sets the model to eval mode (disables dropout, batch norm uses running stats)
    and uses torch.no_grad() to skip gradient computation, saving memory.
    Only takes values at positions where mask > 0.5 (real data, not padding).

    Returns: (y_true, y_pred) as 1D arrays in log1p space.
    """
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y, batch_mask in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            pred = model(batch_x)
            # Only take predictions at positions with real data (not padding)
            valid_pred = pred[batch_mask > 0.5].detach().cpu().numpy()
            valid_true = batch_y[batch_mask > 0.5].detach().cpu().numpy()
            preds.append(valid_pred)
            trues.append(valid_true)
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(trues), np.concatenate(preds)


def permutation_importance_sequence(
    model: nn.Module, val_x: list[np.ndarray], val_y: list[np.ndarray],
    feature_names: list[str], batch_size: int, device: str,
    n_repeats: int = 3, seed: int = 0,
) -> list[float]:
    """Measures feature importance via permutation importance for the sequence model.

    For each feature, shuffle its value ALONG THE TIME AXIS
    within each sequence (keeping each sample's length unchanged, only breaking the time
    order of that particular feature), then measure how much the MSE on val increases relative to
    the baseline. Reuses the ALREADY-TRAINED model of the current fold (does not
    retrain).
    """
    baseline_loader = DataLoader(SequenceDataset(val_x, val_y), batch_size=batch_size,
                                  shuffle=False, collate_fn=collate_sequences)
    y_true, y_pred = evaluate_sequence_model(model, baseline_loader, device)
    if y_true.size == 0:
        return [0.0 for _ in feature_names]
    baseline_mse = float(mean_squared_error(y_true, y_pred))
    rng = np.random.default_rng(seed)
    importances: list[float] = []
    for j in range(len(feature_names)):
        increases = []
        for _ in range(n_repeats):
            perm_x = []
            for seq in val_x:
                seq_perm = seq.copy()
                rng.shuffle(seq_perm[:, j])
                perm_x.append(seq_perm)
            perm_loader = DataLoader(SequenceDataset(perm_x, val_y), batch_size=batch_size,
                                      shuffle=False, collate_fn=collate_sequences)
            _, perm_pred = evaluate_sequence_model(model, perm_loader, device)
            perm_mse = float(mean_squared_error(y_true, perm_pred))
            increases.append(perm_mse - baseline_mse)
        importances.append(float(np.mean(increases)))
    return importances


def train_sequence_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    params: dict[str, Any],
    max_epochs: int,
    patience: int,
    device: str,
) -> tuple[nn.Module, int, float, list[float]]:
    """Trains the sequence model with overfitting-prevention techniques.

    This function uses several important deep learning techniques:

    1. AdamW optimizer:
       - Adam is the most popular optimization algorithm for DL, combining momentum
         and adaptive learning rate (automatically adjusting the learning speed for each parameter).
       - AdamW is an improved version with decoupled weight decay,
         giving more effective regularization than the original Adam.
       - lr (learning rate): learning speed, determines the step size when updating weights.
       - weight_decay (wd): penalty coefficient for large weights, helps avoid overfitting.

    2. ReduceLROnPlateau scheduler:
       - Automatically reduces the learning rate when val loss stops improving.
       - factor=0.5: each time it reduces, new lr = old lr * 0.5.
       - patience=max(2, patience//3): how many epochs to wait before reducing.
       - Helps the model "fine-tune" once it's close to convergence.

    3. Gradient clipping (clip_grad_norm_, max_norm=1.0):
       - Clips the gradient if its norm exceeds 1.0.
       - Why? In sequence models (RNN, LSTM), the gradient can explode
         (exploding gradient) when propagated through many time steps.
       - Clipping helps stabilize the training process.

    4. Early stopping:
       - Stops training early if val loss doesn't improve after 'patience' epochs.
       - Why? After a while, the model starts "memorizing" the training set instead of
         "learning" the general pattern (overfitting). Val loss will increase.
       - Saves the best weights (best_state) and restores them after stopping.

    Returns: (best_model, best_epoch, best_val_loss, val_loss_history)
    val_loss_history contains the val_loss of EACH epoch that ran (for plotting the learning curve).
    """
    model = model.to(device)
    # Initialize AdamW with learning rate and weight decay from Optuna
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["wd"]),
    )
    # Scheduler automatically reduces lr when val loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, patience // 3),
    )
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    val_loss_history: list[float] = []

    for epoch in range(1, max_epochs + 1):
        # Training mode: enables dropout, batch norm uses batch stats
        model.train()
        for batch_x, batch_y, batch_mask in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            # Clear old gradients before computing new ones
            optimizer.zero_grad()
            pred = model(batch_x)
            # Compute loss only on real positions (not padding)
            loss = masked_mse(pred, batch_y, batch_mask)
            # Backpropagate to compute the gradient
            loss.backward()
            # Clip the gradient to prevent exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Update the weights
            optimizer.step()

        # Evaluate on the validation set after each epoch
        val_true, val_pred = evaluate_sequence_model(model, val_loader, device)
        val_mse = float(mean_squared_error(val_true, val_pred))
        val_loss_history.append(val_mse)
        # Inform the scheduler of the current val loss
        scheduler.step(val_mse)
        if val_mse < best_val:
            # Found a better model, save its weights
            best_val = val_mse
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            # Val loss did not improve, increment the wait counter
            wait += 1
            if wait >= patience:
                # Waited long enough, stop early (early stopping)
                break

    # Restore the best weights (not the final weights)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_val, val_loss_history



def split_diagnostics_seq(df: pd.DataFrame, split_idx: SplitIndices, seed: int) -> dict[str, Any]:
    """Creates a diagnostic report about the data split.

    Checks the number of rows, number of GID groups, and in particular whether there is
    GID overlap between the sets (Overlap). If Overlap > 0 it means there is data leakage
    and the split logic needs to be fixed.
    """
    train_gids = set(df.iloc[split_idx.train_idx]["GID"].tolist())
    val_gids = set(df.iloc[split_idx.val_idx]["GID"].tolist())
    test_gids = set(df.iloc[split_idx.test_idx]["GID"].tolist())
    return {
        "Seed": seed,
        "Split": "GroupWise",
        "Train_Rows": int(len(split_idx.train_idx)),
        "Val_Rows": int(len(split_idx.val_idx)),
        "Test_Rows": int(len(split_idx.test_idx)),
        "Train_Groups": int(len(train_gids)),
        "Val_Groups": int(len(val_gids)),
        "Test_Groups": int(len(test_gids)),
        "Overlap_Train_Val_Groups": int(len(train_gids & val_gids)),
        "Overlap_Train_Test_Groups": int(len(train_gids & test_gids)),
        "Overlap_Val_Test_Groups": int(len(val_gids & test_gids)),
        "Any_GID_Overlap": int(bool((train_gids & val_gids) or (train_gids & test_gids) or (val_gids & test_gids))),
    }



def prepare_sequence_splits(
    data_path: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Reads data, filters valid sequences, splits by GID, and builds features.

    This is a utility function that groups the preprocessing steps into a single call:
    1. Read the CSV and filter to keep only rows with seq_ok=1 (valid sequences).
    2. Split by GID (group_stratified_split).
    3. Build the feature matrix (build_feature_frames).
    4. Create the diagnostic report (split_diagnostics).
    """
    df = pd.read_csv(data_path)
    # Keep only rows with a valid sequence (seq_ok=1)
    df = df[df["seq_ok"] == 1].copy()
    split_idx = group_stratified_split(df, seed)
    train_df, val_df, test_df = split_frames(df, split_idx)
    train_feat, val_feat, test_feat, feature_cols = build_feature_frames(train_df, val_df, test_df)
    diag = split_diagnostics_seq(df, split_idx, seed)
    return train_df, val_df, test_df, feature_cols, diag, train_feat, val_feat, test_feat



def run_sequence_benchmark(
    cfg: SequenceBenchmarkConfig,
    build_model_fn: Callable[[int, dict[str, Any]], nn.Module],
    suggest_params_fn: Callable[[optuna.Trial], dict[str, Any]],
) -> dict[str, Path]:
    """Runs the entire benchmark pipeline for a sequence model.

    This is the main function orchestrating the whole process:

    Optuna hyperparameter search process:
        - Optuna is a hyperparameter optimization library.
        - Each "trial" is one attempt with a different parameter set (lr, hidden_size, etc.).
        - suggest_params_fn: a function provided by the user, asking Optuna to suggest
          a value for each parameter (e.g. lr in the range [1e-4, 1e-2]).
        - Optuna picks the parameter set with the lowest val loss.

    Saving the best model (WITHOUT retraining from scratch):
        - Reason: if retrained with the same parameters but different random
          weight initialization, the result could be very different. So, the
          state_dict of the best model found during the Optuna process is saved directly.

    Converting predictions from log-space back to the original scale:
        - The model predicts in log1p space.
        - Clip the predictions to [0, log_cap] before expm1.
        - log_cap = 120% of train_y's max value in log-space.
        - Why clip? If the model predicts a value that is too large in log-space,
          expm1 would produce an extremely large value (exploding), corrupting the metric.
        - expm1 is the inverse of log1p: expm1(log1p(x)) = x.

    Parameters:
        cfg: benchmark configuration (see SequenceBenchmarkConfig).
        build_model_fn: model-building function, receives (n_features, params) -> nn.Module.
        suggest_params_fn: function suggesting parameters for Optuna.

    Returns: dict mapping file name -> result file path.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    best_params_rows: dict[str, dict[str, Any]] = {}

    try:
        for seed in cfg.seeds:
            set_seed(seed)
            train_df, val_df, test_df, feature_cols, diag, train_feat, val_feat, test_feat = prepare_sequence_splits(cfg.data_path, seed)
            split_rows.append(diag)
            train_x, train_y, _ = _group_to_sequences(train_df, train_feat)
            val_x, val_y, _ = _group_to_sequences(val_df, val_feat)
            test_x, test_y, _ = _group_to_sequences(test_df, test_feat)

            train_loader = DataLoader(SequenceDataset(train_x, train_y), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_sequences)
            val_loader = DataLoader(SequenceDataset(val_x, val_y), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_sequences)
            test_loader = DataLoader(SequenceDataset(test_x, test_y), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_sequences)

            # Save the best model from Optuna (WITHOUT retraining from scratch).
            # Reason: retraining from scratch with different random weights -> the result could be very different.
            _best_state: dict[str, Any] = {}
            _best_info: dict[str, Any] = {"epoch": 0, "val": float("inf")}

            def objective(trial: optuna.Trial) -> float:
                """Objective function for Optuna: trains the model and returns val loss."""
                params = suggest_params_fn(trial)
                model = build_model_fn(len(feature_cols), params)
                model, ep, val_score, hist = train_sequence_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    params=params,
                    max_epochs=cfg.max_epochs,
                    patience=cfg.patience,
                    device=cfg.device,
                )
                # Save the state_dict of the model with the lowest val loss across all trials
                if val_score < _best_info["val"]:
                    _best_info["val"] = val_score
                    _best_info["epoch"] = ep
                    _best_info["params"] = params
                    _best_info["history"] = hist
                    _best_state.clear()
                    _best_state.update({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                return val_score

            # Create an Optuna study aiming to minimize val loss
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)

            best_params = _best_info.get("params", study.best_trial.params)
            best_params_rows[str(seed)] = best_params
            best_epoch = _best_info["epoch"]
            best_val = _best_info["val"]

            # Restore the best model instead of retraining
            best_model = build_model_fn(len(feature_cols), best_params)
            if _best_state:
                best_model.load_state_dict(_best_state)
                best_model = best_model.to(cfg.device)
            else:
                # Fallback case: if no saved state exists, retrain
                best_model, best_epoch, best_val, _ = train_sequence_model(
                    model=best_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    params=best_params,
                    max_epochs=cfg.max_epochs,
                    patience=cfg.patience,
                    device=cfg.device,
                )

            # Combine the metrics of the 3 sets (train/val/test) into a single result row
            row: dict[str, Any] = {
                "Seed": seed,
                "Split": "GroupWise",
                "Family": "DL",
                "Model": cfg.model_name,
                "n_features": len(feature_cols),
                "epochs_used": best_epoch,
                "Selection_MSE_log": float(best_val),
            }
            # Bound in log-space: 120% of train_y's max value.
            # Purpose: avoid expm1 exploding when the model predicts too large a value.
            all_train_y = np.concatenate(train_y)
            log_cap = float(np.max(all_train_y)) * 1.2 if len(all_train_y) > 0 else 10.0

            for split_name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
                y_true_log, y_pred_log = evaluate_sequence_model(best_model, loader, cfg.device)
                # Clip the prediction before expm1 to avoid exploding values
                y_pred_log = np.clip(y_pred_log, 0.0, log_cap)
                # Convert from log-space back to the original scale (mm/m) to compute the metric
                y_true = np.expm1(y_true_log)
                y_pred = np.expm1(y_pred_log)
                metrics = regression_metrics(y_true, y_pred)
                for key, value in metrics.items():
                    row[f"{split_name}_{key}"] = value
            summary_rows.append(row)
    except Exception as exc:
        return write_skip_outputs_seq(cfg, f"Insufficient memory or an error occurred while training the sequence model. Original error: {exc}")

    # Create a combined DataFrame from all seeds
    wide_df = pd.DataFrame(summary_rows)
    aggregate_df = aggregate_results(wide_df)
    split_df = pd.DataFrame(split_rows)

    # Write the results to CSV and JSON files
    summary_path = cfg.output_dir / "results_summary.csv"
    aggregate_path = cfg.output_dir / "results_aggregate.csv"
    split_summary_path = cfg.output_dir / "split_summary.csv"
    best_params_path = cfg.output_dir / "best_params.json"
    metadata_path = cfg.output_dir / "run_metadata.json"

    wide_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    split_df.to_csv(split_summary_path, index=False, encoding="utf-8-sig")
    best_params_path.write_text(json.dumps(best_params_rows, indent=2), encoding="utf-8")
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": cfg.model_name,
        "data_path": portable_path(cfg.data_path),
        "output_dir": portable_path(cfg.output_dir),
        "seeds": cfg.seeds,
        "n_trials": cfg.n_trials,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "batch_size": cfg.batch_size,
        "note": "Group-only sequence benchmark on seq_ok curves.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "summary": summary_path,
        "aggregate": aggregate_path,
        "split_summary": split_summary_path,
        "best_params": best_params_path,
        "metadata": metadata_path,
    }



def _predict_sequences_per_gid(model: nn.Module, seqs_x: list[np.ndarray], gids: list[str], device: str) -> tuple[np.ndarray, np.ndarray]:
    """Predicts each sequence (by GID) one at a time, returning (gid_per_point, pred_log_per_point).

    Evaluates each sequence separately (batch size = 1) to preserve the correct point -> GID mapping,
    for collecting out-of-fold predictions at the individual measurement-point level (for scatter/residual plots).
    """
    model.eval()
    gid_points: list[str] = []
    pred_points: list[np.ndarray] = []
    with torch.no_grad():
        for x, gid in zip(seqs_x, gids):
            t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)  # (1, T, D)
            p = model(t).squeeze(0).detach().cpu().numpy().reshape(-1)  # (T,)
            pred_points.append(p)
            gid_points.extend([gid] * len(p))
    if not pred_points:
        return np.array([], dtype=object), np.array([])
    return np.array(gid_points, dtype=object), np.concatenate(pred_points)



def run_sequence_kfold(
    cfg: SequenceBenchmarkConfig,
    build_model_fn: Callable[[int, dict[str, Any]], nn.Module],
    suggest_params_fn: Callable[[optuna.Trial], dict[str, Any]],
) -> dict[str, Path]:
    """Runs GROUP K-FOLD cross-validation for a sequence model (GRU/LSTM).

    Unlike run_sequence_benchmark (a single holdout split per seed), this function
    splits GID into cfg.n_splits folds. Each fold takes a turn as the test set; the rest
    is split into train/val and hyperparameters are optimized with Optuna. Each fold uses its own
    StandardScaler + imputation fit ON THAT FOLD'S TRAIN set (no data leakage).

    Results written out (mirroring the tabular pipeline):
        kfold_summary.csv   : each row = one fold (train/val/test metrics).
        kfold_aggregate.csv : mean +/- standard deviation ACROSS FOLDS.
        oof_predictions.csv : out-of-fold predictions at the measurement-point level (GID, Fold, y_true, y_pred).
        split_summary.csv   : per-fold data split diagnostics (checks GID overlap).
        best_params.json    : best hyperparameters of each fold.
        run_metadata.json   : metadata.

    Uses cfg.seeds[0] as the fold-split seed for reproducibility.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.seeds[0] if cfg.seeds else 42
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    best_params_rows: dict[str, dict[str, Any]] = {}
    oof_frames: list[pd.DataFrame] = []
    loss_history_rows: list[dict[str, Any]] = []
    trials_frames: list[pd.DataFrame] = []
    feat_imp_rows: list[dict[str, Any]] = []
    # Label "GroupKFold" = group k-fold stratified by fc28 (StratifiedKFold on GID,
    # shuffled by seed), NOT the sklearn.GroupKFold class.
    split_label = f"GroupKFold{cfg.n_splits}"

    try:
        df = pd.read_csv(cfg.data_path)
        df = df[df["seq_ok"] == 1].copy().reset_index(drop=True)
        set_seed(seed)
        folds = group_kfold_split_indices(df, cfg.n_splits, seed)

        for split_idx, fold_id in folds:
            diag = split_diagnostics_seq(df, split_idx, seed)
            diag["Split"] = split_label
            diag["Fold"] = fold_id
            split_rows.append(diag)

            train_df, val_df, test_df = split_frames(df, split_idx)
            train_feat, val_feat, test_feat, feature_cols = build_feature_frames(train_df, val_df, test_df)
            train_x, train_y, _ = _group_to_sequences(train_df, train_feat)
            val_x, val_y, _ = _group_to_sequences(val_df, val_feat)
            test_x, test_y, test_gids = _group_to_sequences(test_df, test_feat)

            train_loader = DataLoader(SequenceDataset(train_x, train_y), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_sequences)
            val_loader = DataLoader(SequenceDataset(val_x, val_y), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_sequences)
            test_loader = DataLoader(SequenceDataset(test_x, test_y), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_sequences)

            _best_state: dict[str, Any] = {}
            _best_info: dict[str, Any] = {"epoch": 0, "val": float("inf")}

            def objective(trial: optuna.Trial) -> float:
                params = suggest_params_fn(trial)
                model = build_model_fn(len(feature_cols), params)
                model, ep, val_score, hist = train_sequence_model(
                    model=model, train_loader=train_loader, val_loader=val_loader,
                    params=params, max_epochs=cfg.max_epochs, patience=cfg.patience, device=cfg.device,
                )
                if val_score < _best_info["val"]:
                    _best_info["val"] = val_score
                    _best_info["epoch"] = ep
                    _best_info["params"] = params
                    _best_info["history"] = hist
                    _best_state.clear()
                    _best_state.update({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                return val_score

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)
            trials_df_fold = study.trials_dataframe(attrs=("number", "value", "params"))
            if not trials_df_fold.empty:
                trials_df_fold = trials_df_fold.copy()
                trials_df_fold["Fold"] = fold_id
                trials_frames.append(trials_df_fold)

            best_params = _best_info.get("params", study.best_trial.params)
            best_params_rows[f"fold_{fold_id}"] = best_params
            best_epoch = _best_info["epoch"]
            best_val = _best_info["val"]

            best_model = build_model_fn(len(feature_cols), best_params)
            if _best_state:
                best_model.load_state_dict(_best_state)
                best_model = best_model.to(cfg.device)
            else:
                best_model, best_epoch, best_val, _ = train_sequence_model(
                    model=best_model, train_loader=train_loader, val_loader=val_loader,
                    params=best_params, max_epochs=cfg.max_epochs, patience=cfg.patience, device=cfg.device,
                )

            row: dict[str, Any] = {
                "Seed": seed, "Split": split_label, "Family": "DL", "Model": cfg.model_name,
                "Fold": fold_id, "n_features": len(feature_cols),
                "epochs_used": best_epoch, "Selection_MSE_log": float(best_val),
            }
            all_train_y = np.concatenate(train_y)
            log_cap = float(np.max(all_train_y)) * 1.2 if len(all_train_y) > 0 else 10.0

            for split_name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
                y_true_log, y_pred_log = evaluate_sequence_model(best_model, loader, cfg.device)
                y_pred_log = np.clip(y_pred_log, 0.0, log_cap)
                y_true = np.expm1(y_true_log)
                y_pred = np.expm1(y_pred_log)
                metrics = regression_metrics(y_true, y_pred)
                for key, value in metrics.items():
                    row[f"{split_name}_{key}"] = value
            summary_rows.append(row)

            # Collect the per-epoch val_loss history of this fold's best model (for plotting the learning curve)
            for i, vloss in enumerate(_best_info.get("history", [])):
                loss_history_rows.append({"Fold": fold_id, "epoch": i + 1, "val_loss": vloss})

            # Measure feature importance (permutation importance) using this fold's trained model
            feat_importance = permutation_importance_sequence(
                best_model, val_x, val_y, feature_cols, cfg.batch_size, cfg.device, seed=seed)
            for fname, fval in zip(feature_cols, feat_importance):
                feat_imp_rows.append({"Fold": fold_id, "Feature": fname, "Importance": fval})

            # Out-of-fold predictions at the measurement-point level (keeping the correct GID for each point)
            gid_pts, pred_log_pts = _predict_sequences_per_gid(best_model, test_x, test_gids, cfg.device)
            true_log_pts = np.concatenate(test_y) if test_y else np.array([])
            pred_log_pts = np.clip(pred_log_pts, 0.0, log_cap)
            oof_frames.append(pd.DataFrame({
                "GID": gid_pts,
                "Fold": fold_id,
                "y_true": np.expm1(true_log_pts),
                "y_pred": np.expm1(pred_log_pts),
            }))
    except Exception as exc:
        return write_skip_outputs_seq(cfg, f"Error while running sequence k-fold ({cfg.model_name}). Original error: {exc}")

    summary_df = pd.DataFrame(summary_rows)
    aggregate_df = aggregate_results(summary_df.drop(columns=["Fold"], errors="ignore"))
    split_df = pd.DataFrame(split_rows)
    oof_df = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    loss_history_df = pd.DataFrame(loss_history_rows)
    trials_df = pd.concat(trials_frames, ignore_index=True) if trials_frames else pd.DataFrame()
    feat_imp_df = pd.DataFrame(feat_imp_rows)

    paths = {
        "summary": cfg.output_dir / "kfold_summary.csv",
        "aggregate": cfg.output_dir / "kfold_aggregate.csv",
        "oof": cfg.output_dir / "oof_predictions.csv",
        "split_summary": cfg.output_dir / "split_summary.csv",
        "best_params": cfg.output_dir / "best_params.json",
        "metadata": cfg.output_dir / "run_metadata.json",
        "loss_history": cfg.output_dir / "loss_history.csv",
        "optuna_trials": cfg.output_dir / "optuna_trials.csv",
        "feature_importance": cfg.output_dir / "feature_importance.csv",
    }
    summary_df.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(paths["aggregate"], index=False, encoding="utf-8-sig")
    oof_df.to_csv(paths["oof"], index=False, encoding="utf-8-sig")
    split_df.to_csv(paths["split_summary"], index=False, encoding="utf-8-sig")
    loss_history_df.to_csv(paths["loss_history"], index=False, encoding="utf-8-sig")
    trials_df.to_csv(paths["optuna_trials"], index=False, encoding="utf-8-sig")
    feat_imp_df.to_csv(paths["feature_importance"], index=False, encoding="utf-8-sig")
    paths["best_params"].write_text(json.dumps(best_params_rows, indent=2), encoding="utf-8")
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": cfg.model_name, "status": "completed", "evaluation": "group_kfold",
        "data_path": portable_path(cfg.data_path), "output_dir": portable_path(cfg.output_dir),
        "n_splits": cfg.n_splits, "seed": seed, "n_trials": cfg.n_trials,
        "max_epochs": cfg.max_epochs, "patience": cfg.patience, "batch_size": cfg.batch_size,
        "note": "Group k-fold sequence cross-validation on seq_ok curves.",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths



# =============================================================================
# PART 4 - Architecture + search space of the top-3 DL models (per the pooled OOF of
# file 12): ResMLP (tabular), GRU, LSTM (sequence). Renamed build_model/
# search_space/suggest_params for each model to avoid overwrite-conflicts within the same file.
# =============================================================================
# ---------------------------------------------------------------------------
# Network architecture — Defines the PyTorch classes
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """A residual block — the basic building unit of ResMLP.

    Each ResBlock performs a 2-layer transform then adds the result to the original input
    (skip connection). This is the core mechanism that lets ResMLP outperform
    plain MLP.

    Detailed data flow (forward pass):
    ───────────────────────────────────────
    Input x (d-dimensional vector)
      │
      ├──── save a copy of x ──────────────────────────────────┐
      ↓                                                      │
    FC₁(d→d): linear transform = W₁·x + b₁          │
      ↓       (learns to recombine features)            │
    BN₁(d): batch normalize → stabilizes the value distribution     │
      ↓                                                      │
    ReLU: max(0, val) → introduces nonlinearity                   │
      ↓   (without nonlinearity, 2 FC layers = 1 FC layer)        │
    Dropout(p): randomly disable p% of neurons → reduce overfitting │
      ↓                                                      │
    FC₂(d→d): second linear transform = W₂·h + b₂         │
      ↓       (further refine the features)                │
    BN₂(d): normalize again before adding                │
      ↓                                                      │
    ╔══════════════════════════════════════╗                  │
    ║  ADD element-wise: h + x            ║ ←───────────────┘
    ║  (add corresponding elements)       ║
    ╚══════════════════════════════════════╝
      ↓
    ReLU: final nonlinear activation
      ↓
    Output (d-dimensional vector, same size as input)

    Why 2 FC layers instead of 1?
    ───────────────────────────────────────
    - 1 FC layer + skip = a linear transform added to the input = still nearly
      linear, limited representational power.
    - 2 FC layers + ReLU in between = a more complex NONLINEAR transform, can
      approximate more functions (per the Universal Approximation Theorem).

    Constructor parameters:
    ─────────────────
        d       (int):   Dimension of the input AND output vector. Must be
                         equal for the skip-connection addition to be valid (adding 2
                         vectors of the same size). Usually 96-224.
        dropout (float): Fraction of neurons randomly disabled on each forward pass,
                         e.g. 0.1 = disable 10%. Only active during training,
                         during eval all neurons are on.
    """

    def __init__(self, d: int, dropout: float) -> None:
        super().__init__()
        # --- Layer 1: first linear transform ---
        # FC₁ takes a d-dimensional vector, outputs a d-dimensional vector.
        # Weight matrix W₁ has size d×d, i.e. d² parameters to learn.
        # Example d=160: 160×160 = 25,600 parameters just for this layer.
        self.fc1 = nn.Linear(d, d)

        # BatchNorm normalizes FC₁'s output to mean ≈ 0, standard
        # deviation ≈ 1 WITHIN EACH BATCH. Reason: if values are too large/small,
        # ReLU will saturate (all 0 or all large positive) → poor learning.
        # BN also has a mild regularization effect since each batch has different
        # statistics → adds noise → reduces overfitting.
        self.bn1 = nn.BatchNorm1d(d)

        # ReLU (Rectified Linear Unit): f(x) = max(0, x)
        # The simplest and most common activation function. Why choose ReLU:
        # - Extremely fast to compute (just a comparison to 0)
        # - Gradient = 1 when x > 0 → no vanishing gradient on the positive half
        # - Good enough for tabular data (no need for a more complex function like GELU)
        self.act = nn.ReLU()

        # Dropout: on each forward pass, randomly pick p% of neurons and set them = 0.
        # Why? Forces the network to NOT depend on any single neuron.
        # Like teaching a group of students: if each session you randomly make
        # 10% of students absent, everyone must learn to cover for each other → the group is stronger
        # than a group where only 1-2 skilled people "carry the team".
        self.drop = nn.Dropout(dropout)

        # --- Layer 2: second linear transform ---
        # This layer further refines the result after ReLU introduced nonlinearity.
        # The output will be added directly to the original input (skip connection),
        # so the size must stay d→d.
        self.fc2 = nn.Linear(d, d)

        # BN₂ normalizes again BEFORE adding to the original x.
        # Reason: if FC₂'s output has a value much larger than x, the addition
        # would be "dominated" by one side → losing the meaning of the skip connection.
        # BN₂ keeps the value at the same "scale" as x → the addition is
        # more balanced.
        self.bn2 = nn.BatchNorm1d(d)

    def forward(self, x):
        # --- Transform branch: x passes through 2 FC layers to produce the "residual" h ---
        # Layer 1: FC₁ → BN₁ → ReLU → Dropout
        # This order (Linear→BN→Activation→Dropout) is a common convention
        # in deep learning, called the "pre-activation" pattern.
        h = self.drop(self.act(self.bn1(self.fc1(x))))

        # Layer 2: FC₂ → BN₂ (NO ReLU here, since ReLU is applied
        # AFTER the skip-connection addition — if ReLU were here + ReLU after
        # = 2 consecutive ReLUs, pointless since ReLU(ReLU(x)) = ReLU(x))
        h = self.bn2(self.fc2(h))

        # --- Skip connection: add the original input x to the residual h ---
        # This is the single most important line of code in the whole file!
        # x + h: element-wise addition, requires x and h to have the same size (d-dim).
        # The final ReLU activates the sum. The result has the same size d
        # → can be fed into the next ResBlock.
        return self.act(x + h)


class ResMLP(nn.Module):
    """Residual MLP — the main deep learning model for tabular data.

    The model chains 3 parts: Input Block → N ResBlocks → Head, forming
    a "pipeline" processing data from raw input to the final prediction.

    Overall data flow (forward pass):
    ────────────────────────────────────────
    Input: in_dim-dimensional vector (each dimension = 1 data feature)
      ↓
    [Input Block] FC(in_dim→d) → BN → ReLU → Dropout
      ↓  Purpose: "lift" the dimension from in_dim to d to be compatible
      ↓  with the ResBlocks that follow (ResBlock requires a d-dimensional input)
    [ResBlock_1] 2×FC + skip connection (d→d)
      ↓
    [ResBlock_2] 2×FC + skip connection (d→d)
      ↓
    ... (N ResBlocks total)
      ↓
    [ResBlock_N] 2×FC + skip connection (d→d)
      ↓
    [Head] FC(d→1): compress into 1 number = predicted value (regression)
      ↓
    Output: 1 real number (scalar) per data sample in the batch

    Constructor parameters:
    ─────────────────
        in_dim  (int):   Number of input features. Example: data with 30
                         feature columns → in_dim = 30. This number depends on
                         the data preprocessing step, it is NOT a hyperparameter
                         to tune.
        d       (int):   Dimension (width) of all hidden layers in the model.
                         Larger = "wider" network = more parameters =
                         can learn more complex patterns, but also more prone to
                         overfitting and slower. Search range: 96-224.
        blocks  (int):   Number of stacked ResBlocks. More = "deeper" network
                         = can learn more abstract representations. Thanks to skip
                         connections, adding blocks doesn't hurt. Range: 2-4.
        dropout (float): Fraction of neurons randomly disabled, e.g. 0.1 = 10%.
                         Larger value = stronger regularization = less
                         overfitting but harder to converge. Range: 0.05-0.25.
    """

    def __init__(self, in_dim: int, d: int, blocks: int, dropout: float) -> None:
        super().__init__()
        # --- Input Block: convert dimension from in_dim to d ---
        # Why is this block needed? Because in_dim (original feature count, e.g. 30)
        # is usually different from d (hidden layer dimension, e.g. 160). ResBlock requires
        # input and output to have the same size d (so the skip-connection addition works).
        # So FC(in_dim→d) is needed to "lift the dimension" before entering ResBlock.
        # BN + ReLU + Dropout help stabilize and regularize from the very start.
        self.inp = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- N stacked ResBlocks: extract progressively more abstract features ---
        # nn.Sequential chains the blocks in sequence: output of block 1 → input of block 2.
        # Each block keeps the same size d → any number can be stacked.
        # The number of blocks (2-4) is optimized by Optuna: too few → underfit, too many →
        # overfitting on small datasets (since it adds parameters without adding
        # new information from the data).
        self.blocks = nn.Sequential(*[ResBlock(d, dropout) for _ in range(blocks)])

        # --- Head: the final prediction layer ---
        # Compresses from d dimensions down to a single number. This is a regression
        # task, so there is NO activation function after FC — the output is
        # any real number (can be negative, positive, or 0).
        # If it were a classification task, we would add Sigmoid (2 classes) or
        # Softmax (multi-class) here, but it's not needed here.
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        # The forward flow is very compact thanks to nn.Sequential:
        # 1. self.inp(x): Input Block transforms x from in_dim dims → d dims
        # 2. self.blocks(...): Chain of N ResBlocks, each d→d with skip
        # 3. self.head(...): FC(d→1) produces 1 predicted number
        # The whole process happens automatically when calling model(x) via PyTorch.
        return self.head(self.blocks(self.inp(x)))


def build_resmlp(in_dim: int, params: dict[str, Any]) -> nn.Module:
    """Create a ResMLP instance from the hyperparameter dictionary.

    This function is the "bridge" between Optuna (returns a param dict) and ResMLP
    (takes individual parameters). Needs int/float casting since Optuna may
    return a numpy type that isn't directly compatible with PyTorch.

    Parameters:
        in_dim (int):          Number of input features (from the data).
        params (dict):         Dictionary containing hyperparameters from Optuna:
                               - "res_d": hidden layer dimension
                               - "res_blocks": number of ResBlocks
                               - "dropout": dropout rate

    Returns:
        nn.Module: Initialized ResMLP instance, ready to train.
    """
    return ResMLP(
        in_dim=in_dim,
        d=int(params["res_d"]),
        blocks=int(params["res_blocks"]),
        dropout=float(params["dropout"]),
    )


def search_space_resmlp(trial: optuna.Trial) -> dict[str, Any]:
    return {
        # ── lr (learning rate) ──
        # Determines the step size on each weight update.
        # log=True: search on a logarithmic scale (3e-4, 6e-4, 1e-3, 2e-3, 3e-3)
        # since learning rate is usually distributed on a log scale — the difference between
        # 1e-4 and 2e-4 matters much more than between 2e-3 and 2.001e-3.
        # Too small (3e-4): converges very slowly, may need thousands of epochs.
        # Too large (3e-3): weights "jump" too far → loss oscillates, doesn't converge.
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),

        # ── wd (weight decay) ──
        # A form of regularization: on each update step, weights are multiplied by
        # (1 - wd), pulling them toward 0. Effect: prevents weights from growing too large →
        # the model generalizes better on new data (reduces overfitting).
        # log=True: since wd is also distributed on a log scale.
        # Too small (1e-6): almost no regularization.
        # Too large (3e-4): weights are forced too close to 0, the network loses learning ability.
        "wd": trial.suggest_float("wd", 1e-6, 3e-4, log=True),

        # ── dropout (random neuron-disable rate) ──
        # Range 0.05-0.25 suits tabular data.
        # Tabular data usually has fewer samples than images → overfitting happens easily →
        # dropout is needed, but not too high since information is lost.
        # 0.05: only disables 5% → light regularization → suited to large datasets.
        # 0.25: disables 25% → strong regularization → suited to small, overfit-prone datasets.
        "dropout": trial.suggest_float("dropout", 0.05, 0.25),

        # ── res_d (hidden layer dimension) ──
        # Determines the network's "width". All hidden layers and
        # ResBlocks use this same size.
        # step=16: only tries 96, 112, 128, 144, 160, 176, 192, 208, 224
        #          (9 values). Multiples of 16 make optimal use of GPU hardware
        #          (Tensor Cores work most efficiently with sizes that are multiples of 16).
        # 96:  narrow network, few parameters (~50K), fast but may underfit.
        # 224: wide network, many parameters (~300K), powerful but slower and prone to overfitting.
        "res_d": trial.suggest_int("res_d", 96, 224, step=16),

        # ── res_blocks (number of stacked ResBlocks) ──
        # Determines the network's "depth".
        # 2: shallow network → 4 FC layers (2 blocks × 2 layers) + input block + head
        #    = 6 transform layers total. Suited to simple data.
        # 4: deep network → 8 FC layers + input + head = 10 transform layers.
        #    Needs more data to train effectively.
        # Thanks to skip connections, 4 blocks are NOT worse than 2 blocks (worst case:
        # 2 extra blocks become identity → a 4-block network ≈ a 2-block network).
        "res_blocks": trial.suggest_int("res_blocks", 2, 4),
    }



class GRURegressor(nn.Module):
    """GRU regression model using a sequence-to-sequence (seq2seq) architecture.

    This class encapsulates the entire GRU neural network architecture used to predict the
    shrinkage value at EVERY time step in the sequence (not just the last step).

    Internal architecture:
    --------------------
    1. GRU layer (nn.GRU):
       - Receives the input feature sequence, processing each time step sequentially
       - At each step, applies 2 gates (reset + update) to update the hidden state
       - Can be stacked into multiple layers (num_layers > 1) to learn more complex representations
       - When num_layers > 1: the lower layer's output is the upper layer's input,
         dropout is applied BETWEEN layers (not applied at the last layer)

    2. FC Head (nn.Linear):
       - Fully connected layer converting hidden state -> predicted value
       - Maps from hidden_dim dimensions -> 1 dimension (1 shrinkage value)
       - Applied with the SAME SET OF WEIGHTS for every time step
         (parameter sharing, helps reduce overfitting)

    Data sizes (tensor shapes):
    ------------------------------------
    Input x:
        Shape: (batch_size, seq_len, input_dim)
        - batch_size: number of sequences processed simultaneously in 1 batch
          Example: batch_size=16 means 16 sample groups (GID) at once
        - seq_len:    sequence length (number of time steps), already padded to the same
                      length within 1 batch. Example: 50 measurement days
        - input_dim:  number of features at each time step
                      Example: [age, temperature, humidity, cement_ratio, ...] = 8 features

    Hidden state after GRU:
        Shape: (batch_size, seq_len, hidden_dim)
        - hidden_dim: size of the hidden state vector, chosen by the user
          Example: hidden_dim=64 means each time step has a 64-dimensional vector
                 containing a "summary" of the entire past up to that point

    Output (predictions):
        Shape: (batch_size, seq_len)
        - Each element is 1 predicted shrinkage value at the corresponding time step

    Constructor parameters:
    -----------------
    input_dim  (int):   Number of input features at each time step.
                        Automatically determined from the training data.
    hidden_dim (int):   Size of the hidden state. A larger value allows the model to
                        learn more complex patterns, but increases computation time
                        and overfitting risk. Commonly tried: 32, 48, 64.
    num_layers (int):   Number of stacked GRU layers. 1 layer is usually enough for simple
                        data; 2 layers help learn more abstract representations but are slower.
    dropout    (float): Dropout rate (0.0 to 1.0) BETWEEN GRU layers.
                        Dropout randomly "disables" some neurons during
                        training to fight overfitting. Only has an effect when
                        num_layers > 1 (because dropout is applied between layers, not
                        applied at the last layer).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        # Call the parent class nn.Module's __init__ - required so PyTorch manages
        # the parameters (weights, biases) and supports .to(device), .parameters(), etc.
        super().__init__()

        # ---------------------------------------------------------------
        # Main GRU layer: processes the sequence over time steps.
        #
        # Parameters:
        #   input_size  = input_dim  : number of input features per step
        #   hidden_size = hidden_dim : size of the hidden state vector
        #   num_layers  : number of stacked GRU layers (stacked GRU)
        #   dropout     : dropout rate between layers (0 if only 1 layer,
        #                 because PyTorch requires dropout=0 when num_layers=1)
        #   batch_first = True : convention that the input has shape (batch, seq, feat)
        #                 instead of (seq, batch, feat). This is more intuitive
        #                 and matches how we organize the data.
        # ---------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # ---------------------------------------------------------------
        # FC Head (Fully Connected Head): a linear layer mapping
        # the hidden state (hidden_dim dimensions) -> 1 predicted value.
        #
        # Example: if hidden_dim = 64, this layer has:
        #   - Weight matrix W: shape (1, 64) -> 64 parameters
        #   - Bias b: shape (1) -> 1 parameter
        #   - Total: 65 learnable parameters
        #
        # This layer is applied SEPARATELY at each time step but
        # uses the SAME weights (weight sharing), helping the model be consistent
        # and reducing the number of parameters to learn.
        # ---------------------------------------------------------------
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Forward pass - computes the prediction from the input.

        This is the function PyTorch calls automatically when we write model(x).
        It describes HOW data flows through the model from input to output.

        Args:
            x: Input tensor, shape (batch_size, seq_len, input_dim).
               Each element x[i, t, :] is the feature vector of sequence i
               at time step t.

        Returns:
            Prediction tensor, shape (batch_size, seq_len).
            Each element output[i, t] is the predicted shrinkage value of sequence
            i at time step t.
        """
        # Step 1: Pass the sequence through GRU
        # -----------------------------------------------------------
        # self.gru(x) returns a tuple of 2 parts:
        #   - hidden: (batch, seq_len, hidden_dim) - the hidden state AT EVERY step
        #             This is the "short-term memory" containing accumulated information from the
        #             start of the sequence to the current step.
        #   - _     : (num_layers, batch, hidden_dim) - the final hidden state
        #             of each GRU layer. We don't need this value so it's assigned to _.
        # -----------------------------------------------------------
        hidden, _ = self.gru(x)

        # Step 2: Map the hidden state -> predicted value via the FC head
        # -----------------------------------------------------------
        # self.head(hidden): (batch, seq_len, hidden_dim) -> (batch, seq_len, 1)
        #   The Linear layer applies a matrix multiplication + bias to each hidden vector.
        # .squeeze(-1): removes the last dimension (the dimension with size=1)
        #   (batch, seq_len, 1) -> (batch, seq_len)
        #   Result: each time step has exactly 1 predicted value.
        # -----------------------------------------------------------
        return self.head(hidden).squeeze(-1)


def build_gru(input_dim: int, params: dict) -> nn.Module:
    """Builds a GRU model from the hyperparameters suggested by Optuna.

    This function is the "bridge" between Optuna (hyperparameter search) and the model architecture.
    Every time Optuna tries a new parameter set, this function is called to create the
    corresponding model, which is then trained and evaluated.

    Args:
        input_dim: Number of input features (automatically determined from the data).
        params:    Dict containing the hyperparameters suggested by suggest_params().
                   Must contain: hidden_dim, num_layers, dropout.

    Returns:
        nn.Module: A GRURegressor instance ready for training.
    """
    return GRURegressor(
        input_dim=input_dim,
        hidden_dim=int(params["hidden_dim"]),   # Cast to int since Optuna may return a float
        num_layers=int(params["num_layers"]),    # Number of GRU layers (1 or 2)
        dropout=float(params["dropout"]),        # Dropout rate (0.05 to 0.30)
    )


def suggest_params_gru(trial) -> dict:
    """Defines the hyperparameter search space for Optuna.

    Optuna calls this function every trial (attempt) to get a new parameter set.
    Each parameter is chosen from a reasonable value range, and Optuna will gradually
    focus on the region that gives the best results (Bayesian optimization).

    Search space:
    ---------------------
    hidden_dim (categorical: [32, 48, 64]):
        Hidden state size. Chosen from 3 fixed values instead of a continuous range
        to reduce the search space. 32 = lightest, 64 = most powerful.
        For shrinkage data (not overly complex pattern), 32-64 is usually enough.

    num_layers (int: 1 to 2):
        Number of stacked GRU layers. 1 layer = simple, fast. 2 layers = learn deeper
        representations but slower and more prone to overfitting.
        3+ layers is not tried because shrinkage data is usually not large/complex enough.

    dropout (float: 0.05 to 0.30):
        Fraction of neurons randomly "disabled" during training. Helps fight overfitting.
        0.05 = light regularization, 0.30 = strong regularization.
        Only has an effect when num_layers > 1.

    lr (float: 1e-4 to 3e-3, log scale):
        Learning rate - learning speed. The MOST IMPORTANT parameter affecting the
        training process. Log scale because the optimal learning rate is usually
        distributed logarithmically (1e-4 and 1e-3 are "as far apart" as 1e-3 and 1e-2).

    wd (float: 1e-7 to 1e-3, log scale):
        Weight decay - L2 regularization penalty coefficient. Prevents weights from growing too large,
        helping the model generalize better. Log scale for the same reason as lr.

    Args:
        trial: Optuna trial object, provides suggest_* methods
               to choose parameter values.

    Returns:
        dict: A hyperparameter set for one trial.
    """
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 48, 64]),
        "num_layers": trial.suggest_int("num_layers", 1, 2),
        "dropout": trial.suggest_float("dropout", 0.05, 0.30),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-7, 1e-3, log=True),
    }



class LSTMRegressor(nn.Module):
    """LSTM regression model using a sequence-to-sequence (seq2seq) architecture.

    This class encapsulates the entire LSTM neural network architecture used to predict
    the shrinkage value at EVERY time step in the sequence (not just the last step).

    Internal architecture:
    --------------------
    1. LSTM layer (nn.LSTM):
       - Receives the input feature sequence, processing each time step sequentially
       - At each step, applies 3 gates (forget + input + output) and updates the
         cell state to manage long-term memory
       - Can be stacked into multiple layers (num_layers > 1) to learn deeper representations
       - When num_layers > 1: the lower layer's output is the upper layer's input, dropout is
         applied BETWEEN layers (not applied at the last layer)

    2. FC Head (nn.Linear):
       - Fully connected layer converting hidden state -> predicted
         shrinkage value
       - Mapping: hidden_dim dimensions -> 1 dimension (1 scalar value)
       - Uses the SAME set of weights for every time step (weight sharing)

    Structural comparison with GRU (buoc_07):
       - GRU returns: (output, h_n) - only a hidden state
       - LSTM returns: (output, (h_n, c_n)) - BOTH hidden state AND cell state
       - However, the FC head part is identical: it only uses output (the hidden state at
         each step) to predict, not directly using the cell state or h_n/c_n

    Data sizes (tensor shapes):
    ------------------------------------
    Input x:
        Shape: (batch_size, seq_len, input_dim)
        - batch_size: number of sequences processed simultaneously. E.g. 16 sample groups (GID) at once
        - seq_len:    sequence length (number of time steps), already padded to the same length.
                      E.g. 50 measurement days, shorter sequences are padded with zeros
        - input_dim:  number of features per step. E.g. [age, temperature, humidity, ...] = 8

    Hidden state after LSTM:
        output shape: (batch_size, seq_len, hidden_dim)
        - Contains the hidden state at EVERY time step (used for seq2seq prediction)
        h_n shape: (num_layers, batch_size, hidden_dim)
        - The FINAL hidden state of each layer (not used in seq2seq)
        c_n shape: (num_layers, batch_size, hidden_dim)
        - The FINAL cell state of each layer (not used in seq2seq)

    Output (predictions):
        Shape: (batch_size, seq_len)
        - Each element is 1 predicted shrinkage value at the corresponding time step

    Constructor parameters:
    -----------------
    input_dim  (int):   Number of input features per time step.
                        Automatically determined from the training data.
    hidden_dim (int):   Size of the hidden state AND cell state (both the same
                        size). A larger value -> a more powerful model but slower
                        and more prone to overfitting. Commonly tried: 32, 48, 64.
    num_layers (int):   Number of stacked LSTM layers. 1 layer is usually enough for
                        simple problems; 2 layers help learn more abstract representations.
    dropout    (float): Dropout rate (0.0 to 1.0) BETWEEN LSTM layers.
                        Randomly "disables" neurons during training to fight overfitting.
                        Only has an effect when num_layers > 1 (PyTorch does not apply
                        dropout at the last layer).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        # Call the parent class nn.Module's __init__ - required so PyTorch manages
        # weights and supports .to(device), .parameters(), .state_dict(), etc.
        super().__init__()

        # ---------------------------------------------------------------
        # Main LSTM layer: processes the sequence over time steps.
        #
        # DIFFERENCE from GRU (buoc_07):
        #   - LSTM maintains BOTH hidden state (h) AND cell state (c) at each step
        #   - Has 3 gates instead of 2 -> about 33% more parameters
        #   - With the same hidden_dim=64: GRU has ~3*64*64 = 12,288 gate parameters,
        #     LSTM has ~4*64*64 = 16,384 gate parameters (4 matrices instead of 3)
        #
        # Parameters:
        #   input_size  = input_dim  : number of input features per step
        #   hidden_size = hidden_dim : size of the hidden state AND cell state
        #   num_layers  : number of stacked LSTM layers (stacked LSTM)
        #   dropout     : dropout rate between layers (0 if only 1 layer)
        #   batch_first = True : the input has shape (batch, seq, feat) instead of
        #                 (seq, batch, feat). This is more intuitive.
        # ---------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # ---------------------------------------------------------------
        # FC Head: a linear layer mapping the hidden state -> 1 predicted value.
        # Exactly identical to GRU (buoc_07): the same Linear(hidden_dim, 1) structure
        # since both GRU and LSTM output a hidden state of the same size.
        # ---------------------------------------------------------------
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Forward pass - computes the prediction from the input.

        This is the function PyTorch calls automatically when we write model(x).
        It describes HOW data flows through the model from input to output.

        DIFFERENCE from GRU:
        - GRU: self.gru(x) returns (output, h_n)
        - LSTM: self.lstm(x) returns (output, (h_n, c_n))
          where c_n is the final cell state (long-term memory)
        - However, we only use output (the hidden state at every step) so
          the code looks nearly identical.

        Args:
            x: Input tensor, shape (batch_size, seq_len, input_dim).
               Each element x[i, t, :] is the feature vector of sequence i
               at time step t.

        Returns:
            Prediction tensor, shape (batch_size, seq_len).
            Each element output[i, t] is the predicted shrinkage value of sequence
            i at time step t.
        """
        # Step 1: Pass the sequence through LSTM
        # -----------------------------------------------------------
        # self.lstm(x) returns a tuple of 2 parts:
        #   - hidden: (batch, seq_len, hidden_dim) - the hidden state AT EVERY step
        #             This is the main output, containing information "filtered" from the cell state
        #             via the output gate, ready for prediction.
        #   - _     : tuple (h_n, c_n) - the final states:
        #       h_n: (num_layers, batch, hidden_dim) - the final hidden state of each layer
        #       c_n: (num_layers, batch, hidden_dim) - the final cell state of each layer
        #     We don't need h_n and c_n because we use a seq2seq architecture (predicting at
        #     every step, not just the last one).
        # -----------------------------------------------------------
        hidden, _ = self.lstm(x)  # hidden: (batch, seq_len, hidden_dim)

        # Step 2: Map the hidden state -> predicted value via the FC head
        # -----------------------------------------------------------
        # self.head(hidden): (batch, seq_len, hidden_dim) -> (batch, seq_len, 1)
        #   The Linear layer applies y = W * h + b to each hidden vector at each
        #   time step, using the SAME set of weights W and bias b.
        # .squeeze(-1): removes the last dimension of size=1
        #   (batch, seq_len, 1) -> (batch, seq_len)
        #   Result: a 2D tensor, each element is 1 predicted shrinkage value.
        # -----------------------------------------------------------
        return self.head(hidden).squeeze(-1)  # (batch, seq_len)


def build_lstm(input_dim: int, params: dict) -> nn.Module:
    """Builds an LSTM model from the hyperparameters suggested by Optuna.

    This function is the "bridge" between Optuna (hyperparameter search) and the model architecture.
    Every time Optuna tries a new parameter set, this function creates the corresponding
    LSTM model to train and evaluate.

    Args:
        input_dim: Number of input features (automatically determined from the data).
        params:    Dict containing the hyperparameters suggested by suggest_params().
                   Must contain: hidden_dim, num_layers, dropout.

    Returns:
        nn.Module: An LSTMRegressor instance ready for training.
    """
    return LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=int(params["hidden_dim"]),   # Cast to int since Optuna may return a float
        num_layers=int(params["num_layers"]),    # Number of LSTM layers (1 or 2)
        dropout=float(params["dropout"]),        # Dropout rate (0.05 to 0.30)
    )


def suggest_params_lstm(trial) -> dict:
    """Defines the hyperparameter search space for Optuna.

    Optuna calls this function every trial (attempt) to get a new parameter set.
    Each parameter is chosen from a reasonable value range, and Optuna gradually focuses
    on the region that gives the best results (Bayesian optimization with TPE).

    NOTE: This search space is IDENTICAL to GRU (buoc_07). This is INTENTIONAL:
    - Same search space -> a fair comparison between the 2 architectures
    - Differences in the results reflect ARCHITECTURAL differences, not
      hyperparameter differences

    Search space:
    ---------------------
    hidden_dim (categorical: [32, 48, 64]):
        Size of the hidden state AND cell state (both the same size in LSTM).
        Chosen from 3 fixed values. 32 = light, 64 = powerful.
        Note: with the same hidden_dim, LSTM has more parameters than GRU (~33%) because it has
        1 extra gate (3 gates vs 2 gates) and a cell state.

    num_layers (int: 1 to 2):
        Number of stacked LSTM layers. 1 layer is already powerful enough for many problems.
        2 layers allow learning more abstract representations but risk overfitting.

    dropout (float: 0.05 to 0.30):
        Dropout rate between layers. Especially important for LSTM since LSTM
        has more parameters than GRU -> needs stronger regularization to avoid overfitting.

    lr (float: 1e-4 to 3e-3, log scale):
        Learning rate - learning speed. Log scale because the optimal value is usually distributed
        logarithmically. This is the parameter with the MOST INFLUENCE on the result.

    wd (float: 1e-7 to 1e-3, log scale):
        Weight decay - L2 penalty coefficient. Prevents weights from growing too large, improving
        generalization ability.

    Args:
        trial: Optuna trial object, provides suggest_* methods
               to intelligently choose parameter values.

    Returns:
        dict: A hyperparameter set for one trial.
    """
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 48, 64]),
        "num_layers": trial.suggest_int("num_layers", 1, 2),
        "dropout": trial.suggest_float("dropout", 0.05, 0.30),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-7, 1e-3, log=True),
    }



# =============================================================================
# PART 5 - MAIN ORCHESTRATION (hand-written): rerun Group K-Fold 5 folds, seed 42,
# BUT with Optuna 30 trials/fold (instead of 8 trials as in the official files 03/07/08
# ), to check whether the conclusion is robust to the tuning budget or not.
# =============================================================================
SEED = 42
DEVICE = "cpu"
N_SPLITS = 5
N_TRIALS_T30 = 30
TAB_MAX_EPOCHS = 120
TAB_PATIENCE = 20
SEQ_MAX_EPOCHS = 80
SEQ_PATIENCE = 12
SEQ_BATCH = 16


def run_resmlp_t30(data_path: Path, out_dir: Path) -> dict[str, Path]:
    cfg = BenchmarkConfig(
        data_path=data_path, output_dir=out_dir, split_mode="group",
        seeds=[SEED], n_splits=N_SPLITS, n_trials=N_TRIALS_T30,
        max_epochs=TAB_MAX_EPOCHS, patience=TAB_PATIENCE,
    )
    return run_tabular_kfold(cfg, "ResMLP", build_resmlp, search_space_resmlp)


def run_seq_t30(model_name: str, build_fn, suggest_fn, data_path: Path, out_dir: Path) -> dict[str, Path]:
    cfg = SequenceBenchmarkConfig(
        data_path=data_path, output_dir=out_dir, model_name=model_name,
        seeds=[SEED], n_splits=N_SPLITS, n_trials=N_TRIALS_T30,
        max_epochs=SEQ_MAX_EPOCHS, patience=SEQ_PATIENCE, batch_size=SEQ_BATCH, device=DEVICE,
    )
    return run_sequence_kfold(cfg, build_fn, suggest_fn)


def main() -> None:
    if TORCH_IMPORT_ERROR is not None:
        print(build_torch_runtime_message("Optuna sensitivity (30 trials)", TORCH_IMPORT_ERROR))
        return

    package_root = Path(__file__).resolve().parent
    xlsx_path = package_root / "data_raw" / "Database SCC.xlsx"
    cache_root = package_root / "_cache" / "sensitivity_t30"
    tab_dir = package_root / "output_tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("15_OPTUNA_SENSITIVITY: Rerunning ResMLP/GRU/LSTM with 30 trials/fold (instead of 8)")
    print("Goal: check whether the ranking & gap with GBDT change significantly when")
    print("increasing the Optuna tuning budget or not (cross-checked against files 03/07/08).")
    print("=" * 72)

    data_dir = cache_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_outputs = build_datasets(xlsx_path=xlsx_path, output_dir=data_dir, cfg=DataPipelineConfig())
    tab_path = data_outputs["point"]
    seq_path = data_outputs["seq"]

    runners = [
        ("ResMLP", lambda: run_resmlp_t30(tab_path, cache_root / "resmlp")),
        ("GRU", lambda: run_seq_t30("GRU", build_gru, suggest_params_gru, seq_path, cache_root / "gru")),
        ("LSTM", lambda: run_seq_t30("LSTM", build_lstm, suggest_params_lstm, seq_path, cache_root / "lstm")),
    ]

    results = []
    timing = {}
    for name, fn in runners:
        print(f"\n{'=' * 72}\n[{name}] Group K-Fold {N_SPLITS} fold, Optuna {N_TRIALS_T30} trial/fold\n{'=' * 72}")
        t0 = time.time()
        outputs = fn()
        timing[name] = round((time.time() - t0) / 60.0, 1)
        print_kfold_console_summary(name, outputs)
        agg = safe_read_csv(Path(outputs["aggregate"]))
        if not agg.empty:
            row = agg.iloc[0].to_dict()
            row["Model"] = name
            results.append(row)
        print(f"  -> {name} finished after {timing[name]} min", flush=True)

    if results:
        cmp_df = pd.DataFrame(results)
        cmp_df = cmp_df[["Model"] + [c for c in cmp_df.columns if c != "Model"]]
        cmp_path = tab_dir / "table_optuna_sensitivity_t30.csv"
        cmp_df.to_csv(cmp_path, index=False, encoding="utf-8-sig")
        print("\n" + "=" * 72)
        print("30-TRIAL/FOLD COMPARISON (manually cross-checked against the official 8-trial results")
        print("in file 03 ResMLP / 07 GRU / 08 LSTM, Test_R2_mean column).")
        print("=" * 72)
        print(cmp_df.to_string(index=False))
        print(f"\nSaved the comparison table: {cmp_path}")

    print("\n" + "=" * 72)
    print("File 15 complete (Optuna budget sensitivity).")
    print("=" * 72)


if __name__ == "__main__":
    import time as _t
    _t0 = _t.time()
    main()
    print(f"\n[runtime] {(_t.time() - _t0) / 60.0:.2f} min", flush=True)

