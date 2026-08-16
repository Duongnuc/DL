"""Merged file - 02_mlp_tapered.py

This file MERGES TOGETHER (without omission) content from the original files in the research
original: buoc_03_mlp_tapered.py, plus the shared pipeline common_tabular.py,
common_console_summary.py and buoc_01_code_loc_data.py.

This file is SELF-CONTAINED AND RUNNABLE (does not import any other .py file in the replacement bundle),
and covers all 4 parts of the work:
  1. Prepare data from the raw Excel file (filtering + feature engineering) - from buoc_01.
  2. Define the model architecture + Optuna hyperparameter search space - from buoc_03_mlp_tapered.py.
  3. Run HOLDOUT (single train/val/test split) AND GROUP K-FOLD (5 fold) to
     evaluate stability, using the shared engine from common_tabular.py.
  4. Print all results (R2/RMSE/MAE on train/val/test, best params) to
     the console - not just write to file.

Intermediate CSV/JSON files (models, out-of-fold predictions, best_params, ...) are still
written to the hidden _cache/mlp_tapered/ directory so downstream summary files (12, 13) can
read them back without retraining from scratch.

Run: python 02_mlp_tapered.py
"""

from __future__ import annotations

# =============================================================================
# PART 1/4 - SHARED PIPELINE FOR TABULAR MODELS (merged from common_tabular.py)
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
# PART 2/4 - DATA PREPARATION (merged from buoc_01_code_loc_data.py)
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
# PART 2b - PRINT RESULTS TO CONSOLE (merged from common_console_summary.py)
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



def plot_kfold_prediction_figures(model_name: str, model_key: str, kfold_outputs: dict[str, Any], fig_dir: Path) -> None:
    """Draw 2 figures from the out-of-fold (OOF) results of THIS model ONLY:
      - <key>_pred_vs_measured.png : scatter of predicted vs measured (OOF), y=x line.
      - <key>_residuals.png        : residual vs predicted + residual histogram.

    Only reads the already-available oof_predictions.csv (no retrain) so it runs very fast,
    call right after group k-fold completes.
    """
    oof_path = kfold_outputs.get("oof")
    if oof_path is None or not Path(oof_path).exists():
        print(f"[warning] oof_predictions.csv not found, skipping prediction plot for {model_name}.")
        return
    oof = safe_read_csv(Path(oof_path))
    if oof.empty or "y_true" not in oof.columns or "y_pred" not in oof.columns:
        print(f"[warning] oof_predictions.csv is empty or missing y_true/y_pred columns, skipping plot for {model_name}.")
        return

    import matplotlib
    try:
        get_ipython()  # type: ignore[name-defined]  # available if running in Spyder/Jupyter (IPython)
        in_ipython = True
    except NameError:
        in_ipython = False
    if not in_ipython:
        # Running via "python file.py" in a plain terminal: set Agg (no screen needed,
        # never hangs waiting for a window to close) - just saves PNG.
        matplotlib.use("Agg")
    # If running in Spyder/Jupyter, KEEP its default inline backend so
    # plt.show() below displays the figure right in the Plots pane / notebook output, non-blocking.
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    yt = oof["y_true"].to_numpy()
    yp = oof["y_pred"].to_numpy()
    m = regression_metrics(yt, yp)

    # Figure 1: scatter of predicted vs measured (OOF) - standardized per the convention of the
    # reference paper (Nguyen, Nguyen & Hoang 2025, Fig. 5/6/9/10): y=x line SOLID black,
    # +-40% error band dashed red labeled directly on the line, metrics box (point count,
    # R2, NRMSE) placed in the plot's inner corner, and equal aspect ratio, with
    # both axes spanning 0->max so the y=x line is always exactly 45 degrees.
    hi = float(np.nanpercentile(yt, 99.5))
    lim = (0.0, hi * 1.05)
    tol = 0.40

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(yt, yp, s=8, alpha=0.3, color="#3b6fb6", edgecolors="none", zorder=2)
    ax.plot(lim, lim, color="black", lw=1.3, zorder=3)
    ax.plot(lim, [v * (1 + tol) for v in lim], color="red", ls="--", lw=1, zorder=3)
    ax.plot(lim, [v * (1 - tol) for v in lim], color="red", ls="--", lw=1, zorder=3)
    ax.text(lim[1] * 0.60, lim[1] * 0.60 * (1 + tol), f"+{tol * 100:.0f}%",
            color="red", fontsize=9, ha="left", va="bottom")
    ax.text(lim[1] * 0.60, lim[1] * 0.60 * (1 - tol), f"-{tol * 100:.0f}%",
            color="red", fontsize=9, ha="left", va="top")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Measured OOF value (mustrain)")
    ax.set_ylabel("Predicted OOF value (mustrain)")
    ax.set_title(f"{model_name} - Predicted OOF vs Measured")
    stats_text = f"N points: {m['N']}\nR2: {m['R2']:.2f}\nNRMSE: {m['NRMSE_IQR_pct']:.2f}%"
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#888888", alpha=0.9), zorder=4)
    fig.tight_layout()
    pred_path = fig_dir / f"{model_key}_pred_vs_measured.png"
    fig.savefig(pred_path, dpi=200, bbox_inches="tight")
    if in_ipython:
        plt.show()
    plt.close(fig)
    print(f"  [OK] predicted vs measured figure: {pred_path}")

    # Figure 2: residual analysis
    resid = yp - yt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    hi_p = float(np.nanpercentile(yp, 99.5))
    axes[0].scatter(yp, resid, s=8, alpha=0.3, color="#3b6fb6", edgecolors="none")
    axes[0].axhline(0, color="red", lw=1.2)
    axes[0].set_xlim(0, hi_p * 1.05)
    axes[0].set_xlabel("Predicted sugar content (mustrain)")
    axes[0].set_ylabel("Residual (predicted - measured)")
    axes[0].set_title("Residual vs Predicted")

    mu, sd = float(np.mean(resid)), float(np.std(resid))
    axes[1].hist(resid, bins=50, color="#3b6fb6", alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="red", lw=1.2)
    axes[1].set_xlabel("Residual (mustrain)")
    axes[1].set_ylabel("Tan suat")
    axes[1].set_title(f"Residual distribution (mu={mu:.1f}, sigma={sd:.1f})")

    fig.suptitle(f"{model_name} - Residual analysis (out-of-fold)")
    fig.tight_layout()
    resid_path = fig_dir / f"{model_key}_residuals.png"
    fig.savefig(resid_path, dpi=200, bbox_inches="tight")
    if in_ipython:
        plt.show()
    plt.close(fig)
    print(f"  [OK] residual figure: {resid_path}")

    # Figure 3: stability across folds (fold stability) - reads back kfold_summary.csv
    # already available (no retrain), shows whether Test R2/RMSE fluctuates a lot or a little across
    # folds - complements the OOF figure above (OOF merges all folds into one).
    summary_path = kfold_outputs.get("summary")
    if summary_path is None or not Path(summary_path).exists():
        print(f"  [warning] kfold_summary.csv not found, skipping fold-stability figure for {model_name}.")
        return
    summary_df = safe_read_csv(Path(summary_path))
    if summary_df.empty or "Fold" not in summary_df.columns or "Test_R2" not in summary_df.columns:
        print(f"  [warning] kfold_summary.csv is empty or missing Fold/Test_R2 columns, skipping fold-stability figure for {model_name}.")
        return

    folds = [str(f) for f in summary_df["Fold"].tolist()]
    r2_vals = summary_df["Test_R2"].to_numpy()
    rmse_vals = summary_df["Test_RMSE"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(folds, r2_vals, color="#3b6fb6", alpha=0.85)
    axes[0].axhline(float(np.mean(r2_vals)), color="red", lw=1.2, ls="--",
                     label=f"Mean = {np.mean(r2_vals):.3f}")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Test R2")
    axes[0].set_title("Test R2 by fold")
    axes[0].legend(fontsize=9)

    axes[1].bar(folds, rmse_vals, color="#e07b39", alpha=0.85)
    axes[1].axhline(float(np.mean(rmse_vals)), color="red", lw=1.2, ls="--",
                     label=f"Mean = {np.mean(rmse_vals):.3f}")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Test RMSE")
    axes[1].set_title("Test RMSE by fold")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"{model_name} - Stability across folds")
    fig.tight_layout()
    fold_path = fig_dir / f"{model_key}_fold_stability.png"
    fig.savefig(fold_path, dpi=200, bbox_inches="tight")
    if in_ipython:
        plt.show()
    plt.close(fig)
    print(f"  [OK] fold-stability figure: {fold_path}")

    # Figure 4: learning curve (val_loss over epoch) - reads back loss_history.csv already available
    # (no retrain), shows whether the model converges fast/slow and shows signs of overfit/underfit
    # or not (each fold is one line, epoch stops at that fold's early-stopping point).
    loss_hist_path = kfold_outputs.get("loss_history")
    if loss_hist_path is None or not Path(loss_hist_path).exists():
        print(f"  [warning] loss_history.csv not found, skipping learning-curve figure for {model_name}.")
        return
    loss_hist_df = safe_read_csv(Path(loss_hist_path))
    if loss_hist_df.empty or "Fold" not in loss_hist_df.columns or "val_loss" not in loss_hist_df.columns:
        print(f"  [warning] loss_history.csv is empty or missing Fold/val_loss columns, skipping learning-curve figure for {model_name}.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for fold_id, grp in loss_hist_df.groupby("Fold"):
        grp = grp.sort_values("epoch")
        ax.plot(grp["epoch"], grp["val_loss"], marker="o", markersize=2.5, lw=1.2,
                label=f"Fold {fold_id}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val loss")
    ax.set_title(f"{model_name} - Learning curve (val loss by epoch, per fold)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    curve_path = fig_dir / f"{model_key}_learning_curve.png"
    fig.savefig(curve_path, dpi=200, bbox_inches="tight")
    if in_ipython:
        plt.show()
    plt.close(fig)
    print(f"  [OK] learning-curve figure: {curve_path}")

    # Figure 5: Optuna optimization history (optimization history) + hyperparameter importance.
    # Reads back optuna_trials.csv already available (no retrain). Hyperparameter importance
    # is estimated via |Pearson correlation| between each hyperparameter and the objective value
    # (val MSE), NOT using optuna's fANOVA since only the CSV remains, no original Study.
    trials_path = kfold_outputs.get("optuna_trials")
    if trials_path is None or not Path(trials_path).exists():
        print(f"  [warning] optuna_trials.csv not found, skipping Optuna figure for {model_name}.")
        return
    trials_df = safe_read_csv(Path(trials_path))
    if trials_df.empty or "value" not in trials_df.columns:
        print(f"  [warning] optuna_trials.csv is empty or missing the value column, skipping Optuna figure for {model_name}.")
        return

    param_cols = [c for c in trials_df.columns if c.startswith("params_")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fold_col = "Fold" if "Fold" in trials_df.columns else None
    if fold_col:
        for fold_id, grp in trials_df.groupby(fold_col):
            grp = grp.sort_values("number")
            axes[0].scatter(grp["number"], grp["value"], s=18, alpha=0.6, label=f"Fold {fold_id}")
            axes[0].plot(grp["number"], grp["value"].cummin(), lw=1.2, alpha=0.8)
        axes[0].legend(fontsize=8, ncol=2)
    else:
        grp = trials_df.sort_values("number")
        axes[0].scatter(grp["number"], grp["value"], s=18, alpha=0.6, color="#3b6fb6")
        axes[0].plot(grp["number"], grp["value"].cummin(), lw=1.2, color="red")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Objective value (val MSE, log-space)")
    axes[0].set_title("Optuna optimization history (lower = better)")

    importances: dict[str, float] = {}
    for col in param_cols:
        vals = pd.to_numeric(trials_df[col], errors="coerce")
        if vals.notna().sum() < 3 or vals.nunique() < 2:
            continue
        corr = trials_df["value"].corr(vals)
        if pd.notna(corr):
            importances[col.replace("params_", "")] = abs(float(corr))

    if importances:
        items = sorted(importances.items(), key=lambda kv: kv[1])
        axes[1].barh([k for k, _ in items], [v for _, v in items], color="#3b6fb6", alpha=0.85)
        axes[1].set_xlabel("|Pearson correlation| with objective value")
        axes[1].set_title("Hyperparameter importance (correlation estimate)")
    else:
        axes[1].text(0.5, 0.5, "Not enough data to estimate", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_axis_off()

    fig.suptitle(f"{model_name} - Optuna: optimization history & hyperparameter importance")
    fig.tight_layout()
    optuna_path = fig_dir / f"{model_key}_optuna_history.png"
    fig.savefig(optuna_path, dpi=200, bbox_inches="tight")
    if in_ipython:
        plt.show()
    plt.close(fig)
    print(f"  [OK] Optuna history figure: {optuna_path}")

    # Figure 6: importance of each feature - reads back feature_importance.csv already available
    # (no retrain). For DL models (tabular/sequence): permutation importance (how much
    # val MSE increases when shuffling 1 feature column, reusing the already-trained model
    # of each fold). For GBDT (XGBoost/LightGBM): the library's built-in "gain" importance.
    # Both are aggregated per fold then averaged +/- standard deviation, sorted descending,
    # only plotting the top 20 most important features for readability.
    feat_imp_path = kfold_outputs.get("feature_importance")
    if feat_imp_path is None or not Path(feat_imp_path).exists():
        print(f"  [warning] feature_importance.csv not found, skipping feature-importance figure for {model_name}.")
        return
    feat_imp_df = safe_read_csv(Path(feat_imp_path))
    if feat_imp_df.empty or "Feature" not in feat_imp_df.columns or "Importance" not in feat_imp_df.columns:
        print(f"  [warning] feature_importance.csv is empty or missing Feature/Importance columns, skipping feature-importance figure for {model_name}.")
        return

    stats = (feat_imp_df.groupby("Feature")["Importance"]
             .agg(["mean", "std"]).fillna(0.0).sort_values("mean", ascending=True))
    top = stats.tail(20)

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.32 * len(top))))
    ax.barh(top.index.astype(str), top["mean"], xerr=top["std"], color="#3b6fb6", alpha=0.85,
            error_kw=dict(ecolor="#555555", elinewidth=0.9, capsize=2))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Importance (permutation: MSE increase when shuffled; GBDT: gain)")
    ax.set_title(f"{model_name} - Feature importance (top {len(top)}, mean +/- std across folds)")
    fig.tight_layout()
    feat_imp_fig_path = fig_dir / f"{model_key}_feature_importance.png"
    fig.savefig(feat_imp_fig_path, dpi=200, bbox_inches="tight")
    if in_ipython:
        plt.show()
    plt.close(fig)
    print(f"  [OK] feature-importance figure: {feat_imp_fig_path}")


# =============================================================================
# PART 3/4 - MODEL ARCHITECTURE + SEARCH SPACE (merged from buoc_03_mlp_tapered.py)
# =============================================================================
"""
MLP_Tapered model -- Multi-Layer Perceptron with a "funnel" architecture (decreasing node count).
=======================================================================================

Full name : MLP_Tapered (Multi-Layer Perceptron -- Tapered / Funnel-shaped)
Purpose    : Used in the benchmark pipeline comparing tabular models.
Goal    : Binary classification on tabular data.

1. DETAILED ARCHITECTURE (Text Diagram)
-------------------------------------
Assume in_dim=30 (30 input features), h1=256, dropout=0.2:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Input (batch_size, 30)                                         │
    │       │                                                         │
    │       ▼                                                         │
    │  ┌─────────────┐                                                │
    │  │ Linear(30→256) │  ← Layer 1: expand the feature space          │
    │  │ BatchNorm(256) │  ← Normalize so the gradient doesn't explode  │
    │  │ ReLU()         │  ← Nonlinear activation function             │
    │  │ Dropout(0.2)   │  ← Randomly disable 20% of nodes to reduce overfit│
    │  └───────┬───────┘                                              │
    │          ▼                                                      │
    │  ┌─────────────┐                                                │
    │  │ Linear(256→128)│  ← Layer 2: start compressing info (h1/2)   │
    │  │ BatchNorm(128) │  ← Normalize again to stabilize training     │
    │  │ ReLU()         │  ← Nonlinear activation                      │
    │  │ Dropout(0.2)   │  ← Continue reducing overfit                  │
    │  └───────┬───────┘                                              │
    │          ▼                                                      │
    │  ┌─────────────┐                                                │
    │  │ Linear(128→64) │  ← Layer 3: compress further (h1/4), NO BN   │
    │  │ ReLU()         │  ← Nonlinear activation                      │
    │  │ Dropout(0.2)   │  ← Reduce overfit                            │
    │  └───────┬───────┘                                              │
    │          ▼                                                      │
    │  ┌─────────────┐                                                │
    │  │ Linear(64→1)   │  ← Output layer: 1 node for the binary task│
    │  └───────┬───────┘                                              │
    │          ▼                                                      │
    │  Output (batch_size, 1) -- logit, before sigmoid             │
    └──────────────────────────────────────────────────────────────────┘

    Overall data flow:
        30 → 256 → 128 → 64 → 1
        (wide)  (progressively compressed)  (1 output value)

2. COMPARISON WITH ANNBasic (buoc_02)
----------------------------------
    ┌────────────────┬──────────────────────┬───────────────────────────┐
    │ Criterion       │ ANNBasic             │ MLP_Tapered               │
    ├────────────────┼──────────────────────┼───────────────────────────┤
    │ Nodes per layer│ Equal (h, h, h)     │ Decreasing (h, h/2, h/4)   │
    │ BatchNorm      │ None                 │ Present in first 2 layers │
    │ First layer    │ Small (~64-128)     │ Wider (~64-512)            │
    │ Idea           │ Simple, uniform      │ Funnel: wide capture, gradual compress │
    │ Overfit control│ Dropout only         │ Dropout + BatchNorm       │
    │ Param count    │ More (uniform layers)│ Fewer (later layers small)│
    └────────────────┴──────────────────────┴───────────────────────────┘

    Why is the funnel architecture better than a uniform one?
    - A wide first layer lets the network "see" many different feature combinations.
    - Progressively narrower later layers force the network to "select" the most important
      information, similar to how the human brain processes: gather then synthesize.
    - BatchNorm keeps intermediate values from being too large/small, so the gradient
      propagated backward (backpropagation) is more stable and training converges faster.

3. TUNING METHOD (Hyperparameter Tuning)
---------------------------------------
    - Uses Optuna (a Bayesian optimization library) to automatically find:
        + h1 (taper_h1): number of nodes in the first layer (64-512)
        + dropout: node-disable rate (0.05-0.35)
        + lr (learning rate): learning speed (2e-4 to 3e-3)
        + wd (weight decay): weight penalty coefficient (1e-6 to 5e-4)
    - Layers h2, h3 are computed automatically from h1 (h1/2, h1/4), so only
      1 parameter h1 needs to be tuned instead of tuning 3 layers separately -> smaller search space.

4. ADVANTAGES
-----------
    + Fewer parameters than a uniform network thanks to progressively smaller layers -> faster training.
    + BatchNorm makes training more stable, less sensitive to learning rate.
    + The funnel architecture matches intuition: extract then compress information.
    + Only need to tune 1 parameter h1, h2 and h3 are derived automatically -> smaller search space.

5. DISADVANTAGES
--------------
    - BatchNorm behaves differently in train vs eval mode, needs attention when deploying.
    - If h1 is too large, the first layer may overfit before later layers can compress.
    - Fixed architecture (halved each layer), less flexible than tuning each layer separately.
    - With very small data, BatchNorm can introduce noise (small batch -> mean/variance
      statistics are inaccurate).

Run standalone:  python buoc_03_mlp_tapered.py
"""


MODEL_NAME = "MLP_Tapered"


# ---------------------------------------------------------------------------
# Network architecture -- Defines the layer structure of the model
# ---------------------------------------------------------------------------
class MLP_Tapered(nn.Module):
    """MLP funnel architecture: node count decreases through each hidden layer (h1 -> h1/2 -> h1/4 -> 1).

    Core idea:
        A wide first layer to "scan" many combinations of input features, then subsequent
        layers narrow progressively to force the network to "distill" the most important
        information, finally compressing down to 1 number (probability of the positive class).

    Input shape:
        x : Tensor with shape (batch_size, in_dim)
            Each row is 1 data sample, each column is 1 feature.
            Example: for the Breast Cancer dataset, in_dim=30 (30 features
            such as mean radius, perimeter, area, etc.).

    Output shape:
        Tensor with shape (batch_size, 1)
            Logit value (before sigmoid). The BCEWithLogitsLoss function will
            apply sigmoid internally, so we do NOT need to call sigmoid here.
            Reason: combining sigmoid + BCE into 1 function gives more numerically
            stable computation.

    Constructor parameters (Parameters):
        in_dim : int
            Number of input features. Must match the number of columns in the data
            after preprocessing (removing the target column, encoding categoricals, etc.).
        h1 : int
            Number of neurons in the first hidden layer -- the widest layer.
            Layers h2 and h3 are computed automatically: h2 = h1 // 2, h3 = h2 // 2.
            Example: h1=256 -> h2=128 -> h3=64.
            A larger h1 gives the network richer representational power but costs more memory
            and is more prone to overfitting; a smaller value is the opposite.
        dropout : float (0.0 to 1.0)
            Fraction of nodes randomly disabled on each forward pass during training.
            Example: dropout=0.2 means 20% of nodes are disabled -> the network cannot
            "rely" on a few specific nodes -> forced to learn more general features
            -> reduces overfit. During eval/predict, dropout is automatically disabled.

    Forward pass (data flow through the network):
        1. Layer 1: Linear(in_dim -> h1) -> BatchNorm -> ReLU -> Dropout
           Expand the space, normalize, nonlinear activation, reduce overfit.
        2. Layer 2: Linear(h1 -> h2) -> BatchNorm -> ReLU -> Dropout
           Start compressing, still normalize since gradients in middle layers are prone to instability.
        3. Layer 3: Linear(h2 -> h3) -> ReLU -> Dropout
           Compress further. NO BatchNorm since this layer is already small, adding BN could
           introduce noise (batch statistics are less accurate at small dimensions).
        4. Output layer: Linear(h3 -> 1)
           Produces 1 logit value. No activation since the loss function handles it.
    """

    def __init__(self, in_dim: int, h1: int, dropout: float) -> None:
        super().__init__()

        # --- Compute node counts for layers 2 and 3 by halving ---
        # Use integer division (//) to ensure the node count is an integer.
        # max(..., 8) and max(..., 4) set a minimum threshold to avoid cases where
        # h1 is too small (e.g. h1=8), resulting in h2=4, h3=2 -- too few nodes would make
        # the network lose representational capacity (expressiveness), unable to learn anything.
        h2 = max(h1 // 2, 8)   # Layer 2: half of h1, minimum 8 nodes
        h3 = max(h2 // 2, 4)   # Layer 3: half of h2, minimum 4 nodes

        # --- Build the network using nn.Sequential ---
        # nn.Sequential stacks layers in order; when calling forward, data
        # passes through each layer top to bottom, no need to write forward manually.
        self.net = nn.Sequential(
            # === HIDDEN LAYER 1 (widest -- "scans" features) ===
            nn.Linear(in_dim, h1),  # Linear transform: y = Wx + b, W has shape (h1, in_dim)
            nn.BatchNorm1d(h1),     # Normalize output per batch: (x - mean) / std, then scale + shift.
                                    # Why needed? Because if Linear's output has very large/small values,
                                    # the gradient will explode or vanish -> training is slow or doesn't converge.
            nn.ReLU(),              # ReLU(x) = max(0, x). Why use ReLU?
                                    # (1) Fast to compute, (2) gradient = 0 or 1 so less prone to vanishing,
                                    # (3) introduces nonlinearity so the network can learn complex relationships.
            nn.Dropout(dropout),    # Randomly disable a fraction of nodes. Why place it AFTER activation?
                                    # Because we want to disable already-activated nodes, not disable before
                                    # knowing which nodes are "lit" (value > 0 after ReLU).

            # === HIDDEN LAYER 2 (compress down to h1/2) ===
            nn.Linear(h1, h2),      # Reduce dimension from h1 to h2 -- forces the network to select info
            nn.BatchNorm1d(h2),     # Still need normalization since middle layers are prone to unstable gradients
            nn.ReLU(),              # Nonlinear activation -- same as layer 1
            nn.Dropout(dropout),    # Reduce overfit -- same as layer 1

            # === HIDDEN LAYER 3 (compress further to h1/4 -- NO BatchNorm) ===
            nn.Linear(h2, h3),      # Compress further from h2 to h3
                                    # Why NO BatchNorm here? Because h3 is usually small (16-64 nodes),
                                    # at small dimensions, the batch's mean/var statistics aren't representative,
                                    # BatchNorm could introduce noise rather than help.
            nn.ReLU(),              # Nonlinear activation
            nn.Dropout(dropout),    # Reduce overfit

            # === OUTPUT LAYER (1 node for the binary task) ===
            nn.Linear(h3, 1),       # Produces 1 logit value. NO activation function here because:
                                    # (1) BCEWithLogitsLoss integrates sigmoid internally,
                                    # (2) Separating out sigmoid can cause numerical issues (log(sigmoid(x))
                                    #     can be -inf when x is very negative, but BCEWithLogitsLoss
                                    #     uses a more stable formula to avoid this issue).
        )

    def forward(self, x):
        """Pass data through the entire network and return the logit.

        Just need to call self.net(x) since nn.Sequential already handles passing
        data through each layer in the order defined in __init__.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Search space -- Hyperparameter search space
# ---------------------------------------------------------------------------
def search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Define the search range for each hyperparameter.

    Optuna calls this function each trial, each time choosing a different set of values
    within the defined range, then trains the model with that set of values
    to see if the result is good or bad, and learns to choose the next set more intelligently
    (Bayesian optimization instead of random trial).

    Tuned parameters:
        lr (learning rate):
            Range: 2e-4 to 3e-3, logarithmic distribution.
            Why log=True? Because learning rate operates on an order-of-magnitude basis (1e-4 vs 1e-3
            differ by a factor of 10), a log distribution lets Optuna sample evenly at each order of magnitude
            instead of concentrating on the large region (with uniform, most values
            would fall between 1e-3 and 3e-3, skipping the small region).

        wd (weight decay):
            Range: 1e-6 to 5e-4, logarithmic distribution.
            Weight decay adds a small amount to the loss: loss += wd * sum(w^2).
            Effect: forces weights to stay small -> simpler model -> reduces
            overfit. A small value (1e-6) barely penalizes, a large value (5e-4)
            penalizes more heavily. Also uses log=True for the same reason as lr.

        dropout (node-disable rate):
            Range: 0.05 to 0.35, uniform distribution.
            Why not log? Because dropout operates linearly (0.1 and 0.2
            differ by exactly 0.1, not by a doubling effect).
            0.05 = almost nothing disabled -> prone to overfit but learns fast.
            0.35 = 35% of nodes disabled -> strong overfit control but may underfit.

        taper_h1 (number of nodes in the first hidden layer):
            Range: 64 to 512 nodes.
            This is the MOST IMPORTANT parameter of the funnel architecture since h2 and h3
            are computed automatically from h1 (h2 = h1/2, h3 = h1/4).
            64 nodes: small, fast network but limited representational power.
            512 nodes: large, powerful network but slower and prone to overfitting on small data.
    """
    return {
        "lr": trial.suggest_float("lr", 2e-4, 3e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 5e-4, log=True),
        "dropout": trial.suggest_float("dropout", 0.05, 0.35),
        "taper_h1": trial.suggest_int("taper_h1", 64, 512),  # Number of nodes in the first layer, determines the whole architecture
    }


def build_model(in_dim: int, params: dict[str, Any]) -> nn.Module:
    """Initialize the MLP_Tapered model from the hyperparameter set chosen by Optuna.

    This function is the "bridge" between Optuna and the MLP_Tapered class: Optuna returns a dict
    containing numeric values, this function converts them to the correct data type (int, float)
    then passes them into the MLP_Tapered constructor.

    Why int(params["taper_h1"])? Because Optuna's suggest_int can return
    numpy.int64 in some cases, cast to plain Python int to avoid errors.
    """
    return MLP_Tapered(in_dim=in_dim, h1=int(params["taper_h1"]), dropout=float(params["dropout"]))


# ---------------------------------------------------------------------------
# Main -- entry point when running the file directly
# ---------------------------------------------------------------------------


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, Path]:
    """Run the full holdout pipeline: tune Optuna -> train -> evaluate -> save results."""
    return run_tabular_benchmark(cfg, MODEL_NAME, build_model, search_space)


# =============================================================================
# PART 4/4 - ACTUAL RUN: HOLDOUT + GROUP K-FOLD + PRINT RESULTS
# =============================================================================
SEED = 42
N_SPLITS = 5
N_TRIALS_KFOLD = 8
TAB_MAX_EPOCHS = 120
TAB_PATIENCE = 20


def main() -> None:
    if TORCH_IMPORT_ERROR is not None:
        print(build_torch_runtime_message(MODEL_NAME, TORCH_IMPORT_ERROR))
        return

    package_root = Path(__file__).resolve().parent
    xlsx_path = package_root / "data_raw" / "Database SCC.xlsx"
    cache_dir = package_root / "_cache" / "mlp_tapered"

    print("=" * 72)
    print(f"[{MODEL_NAME}] STEP 1/4: Data preparation (filtering + feature engineering)")
    print("=" * 72)
    data_dir = cache_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_outputs = build_datasets(xlsx_path=xlsx_path, output_dir=data_dir, cfg=DataPipelineConfig())
    data_path = data_outputs["point"]
    print(f"Finished preparing data: {data_path}")

    print("\n" + "=" * 72)
    print(f"[{MODEL_NAME}] STEP 2/4: Holdout (single train/val/test split, Optuna 4 trials)")
    print("=" * 72)
    holdout_cfg = BenchmarkConfig(
        data_path=data_path, output_dir=cache_dir / "holdout",
        seeds=[SEED], n_trials=4, max_epochs=100, patience=15,
    )
    holdout_outputs = run_benchmark(holdout_cfg)
    print_model_console_summary(MODEL_NAME, holdout_outputs)

    print("\n" + "=" * 72)
    print(f"[{MODEL_NAME}] STEP 3/4: Group K-Fold ({N_SPLITS} fold, Optuna {N_TRIALS_KFOLD} trials)")
    print("=" * 72)
    kfold_cfg = BenchmarkConfig(
        data_path=data_path, output_dir=cache_dir / "kfold", split_mode="group",
        seeds=[SEED], n_splits=N_SPLITS, n_trials=N_TRIALS_KFOLD,
        max_epochs=TAB_MAX_EPOCHS, patience=TAB_PATIENCE,
    )
    kfold_outputs = run_tabular_kfold(kfold_cfg, MODEL_NAME, build_model, search_space)
    print_kfold_console_summary(MODEL_NAME, kfold_outputs)

    fig_dir = package_root / "output_figures"
    plot_kfold_prediction_figures(MODEL_NAME, "mlp_tapered", kfold_outputs, fig_dir)


    print(f"\nFinished running the full pipeline for {MODEL_NAME} (holdout + k-fold).")


if __name__ == "__main__":
    import time as _t
    _t0 = _t.time()
    main()
    print(f"\n[runtime] {(_t.time() - _t0) / 60.0:.2f} min", flush=True)
