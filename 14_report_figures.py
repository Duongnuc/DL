"""14 - All figures/tables for the scientific paper: data description (Table 1,
distribution histograms, correlation matrix) and the shrinkage development curve over
time (measured vs LSTM-predicted on the holdout test set).

This file MERGES TOGETHER (without omission) content from buoc_17_data_and_curves.py, plus
the shared pipeline common_tabular.py, common_sequence_group.py and
buoc_01_code_loc_data.py.

This file LIGHTLY RETRAINS 1 LSTM on a group-split holdout (seed 42), using the
REPRESENTATIVE hyperparameters saved in _cache/lstm/kfold/best_params.json of file 08
- if that file has not been run yet, use reasonable default parameters (will not crash).

Figures are written to output_figures/, data tables are written to output_tables/ (both
are VISIBLE directories, not hidden, since these are deliverables to submit for the paper/thesis).

Run:  python 14_report_figures.py
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
    detail = "" if exc is None else f" Original error: {exc}."
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
            return write_skip_outputs(cfg, model_name, f"Error while training {model_name}. Original error: {exc}")
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
            return write_skip_outputs(cfg, model_name, f"Error while training {model_name} (fold {fold_id}). Original error: {exc}")
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
# PART 3 - Preprocessing specific to SEQUENCE (Dataset/DataLoader) + train_sequence_model
# (with early stopping, unlike train_sequence_history used in file 13), merged from
# common_sequence_group.py
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

SEQ_ONE_HOT_COLUMNS = DL_ONE_HOT_COLUMNS


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



# =============================================================================
# PART 4 - LSTM architecture (class + build_lstm only; reuses the REPRESENTATIVE
# hyperparameters cached from the k-fold of file 08, does not rerun Optuna)
# =============================================================================
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



# =============================================================================
# PART 5 - (A) Data description: value-range table + histogram + correlation
# =============================================================================
# Display labels (with units) for the variables.
LABELS = {
    "y_shrinkage_total_mm_per_m": "Total shrinkage (×10⁻⁶)",
    "dt_days": "Drying time t–t₀ (days)",
    "fc28": "28-day strength f_cm28 (MPa)",
    "w_cm": "W/CM ratio (w/cm)",
    "cm": "CM content (kg/m³)",
    "V_S": "Volume/surface ratio V/S (mm)",
    "RH_test": "Relative humidity RH",
    "T": "Test temperature T (°C)",
    "t0": "Age at start of drying t₀ (days)",
    "Tcur": "Curing temperature (°C)",
    "age_days": "Sample age (days)",
    "water": "Water (kg/m³)",
    "FlyAsh": "Fly ash (%)",
    "Slag": "Slag (%)",
    "agg_total": "Total aggregate (kg/m³)",
    "sandrate": "Sand ratio",
}
HIST_VARS = ["y_shrinkage_total_mm_per_m", "dt_days", "fc28", "w_cm",
             "cm", "V_S", "RH_test", "T", "t0"]
RANGE_VARS = ["fc28", "cm", "w_cm", "V_S", "RH_test", "T", "Tcur", "t0",
              "dt_days", "age_days", "y_shrinkage_total_mm_per_m"]
CORR_VARS = ["y_shrinkage_total_mm_per_m", "dt_days", "age_days", "fc28", "w_cm",
             "cm", "water", "V_S", "RH_test", "T", "Tcur", "t0",
             "FlyAsh", "Slag", "agg_total", "sandrate"]


def short_label(col: str) -> str:
    return LABELS.get(col, col)



def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in RANGE_VARS:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        rows.append({
            "Variable": short_label(c),
            "Column": c,
            "N": int(s.shape[0]),
            "Min": round(float(s.min()), 4),
            "Max": round(float(s.max()), 4),
            "Mean": round(float(s.mean()), 4),
            "Std": round(float(s.std()), 4),
        })
    return pd.DataFrame(rows)



def fig_distributions(df: pd.DataFrame, fig_dir: Path) -> Path:
    n = len(HIST_VARS)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, HIST_VARS):
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        ax.hist(s, bins=35, color="#3b6fb6", edgecolor="white", alpha=0.85)
        ax.axvline(s.mean(), color="#c0504d", ls="--", lw=1.6,
                   label=f"Mean = {s.mean():.1f}")
        ax.set_xlabel(short_label(c), fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"SCC shrinkage data distribution (N = {len(df)} measurement points, {df['GID'].nunique()} samples)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    out = fig_dir / "fig10_data_distributions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out



def fig_correlation(df: pd.DataFrame, fig_dir: Path) -> Path:
    cols = [c for c in CORR_VARS if c in df.columns]
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    labels = [short_label(c) for c in cols]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.to_numpy()[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black" if abs(v) < 0.6 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation coefficient", fontsize=10)
    ax.set_title("Correlation matrix of variables (entire database)", fontsize=13)
    fig.tight_layout()
    out = fig_dir / "fig11_correlation_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out



# =============================================================================
# PART 6 - (B) Shrinkage development curve over time: measured vs predicted (LSTM)
# =============================================================================
N_CURVES = 6  # number of representative samples for the development curve


def predict_curve(model, test_df, test_feat, gid, log_cap):
    """Predict the shrinkage curve for 1 GID, return (dt_days, y_meas, y_pred)."""
    mask = test_df["GID"] == gid
    sub = test_df[mask].sort_values("dt_days")
    order = sub.index.to_numpy()
    x = test_feat.loc[order].to_numpy(dtype=np.float32)
    dt = sub["dt_days"].to_numpy(dtype=float)
    y_meas = sub["y_shrinkage_total_mm_per_m"].to_numpy(dtype=float)
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
        plog = model(t).squeeze(0).cpu().numpy().reshape(-1)
    plog = np.clip(plog, 0.0, log_cap)
    y_pred = np.expm1(plog)
    return dt, y_meas, y_pred



def pick_representative_gids(test_df, min_len=8, k=N_CURVES):
    """Select k test GIDs with sufficiently long sequences, spread evenly by end-of-period shrinkage magnitude."""
    info = []
    for gid, sub in test_df.groupby("GID"):
        if len(sub) < min_len:
            continue
        last = sub.sort_values("dt_days")["y_shrinkage_total_mm_per_m"].iloc[-1]
        info.append((gid, len(sub), float(last)))
    if len(info) < k:  # relax the condition if not enough
        info = []
        for gid, sub in test_df.groupby("GID"):
            last = sub.sort_values("dt_days")["y_shrinkage_total_mm_per_m"].iloc[-1]
            info.append((gid, len(sub), float(last)))
    info.sort(key=lambda r: r[2])  # sort by end-of-period shrinkage
    # take k samples spread evenly by magnitude percentile
    idx = np.linspace(0, len(info) - 1, k).round().astype(int)
    return [info[i][0] for i in idx]



def fig_time_curves(model, test_df, test_feat, log_cap, test_metrics, fig_dir: Path):
    gids = pick_representative_gids(test_df)
    ncol = 3
    nrow = int(np.ceil(len(gids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, gid in zip(axes, gids):
        dt, ym, yp = predict_curve(model, test_df, test_feat, gid, log_cap)
        sub = test_df[test_df["GID"] == gid].iloc[0]
        r2 = r2_score(ym, yp) if len(ym) > 2 else float("nan")
        ax.plot(dt, ym, "o", color="#1f3b6f", ms=6, label="Measured (experiment)")
        ax.plot(dt, yp, "-", color="#c0504d", lw=2.2, label="Predicted (LSTM)")
        ax.set_xscale("symlog")
        ax.set_xlabel("Drying time t–t₀ (days)", fontsize=10)
        ax.set_ylabel("Shrinkage (×10⁻⁶)", fontsize=10)
        ax.set_title(f"GID {gid} | f_cm28={sub['fc28']:.0f} MPa, "
                     f"w/cm={sub['w_cm']:.2f}\nSample R² = {r2:.3f}", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper left")
    for ax in axes[len(gids):]:
        ax.axis("off")
    fig.suptitle(f"Shrinkage development curve over time — measured vs predicted (LSTM, holdout test samples)\n"
                 f"Global test: R² = {test_metrics['R2']:.3f}, RMSE = {test_metrics['RMSE']:.1f} ×10⁻⁶",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = fig_dir / "fig12_shrinkage_time_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out, gids



# =============================================================================
# PART 7 - MAIN ORCHESTRATION (hand-written, not taken from the original file)
# =============================================================================
SEED = 42
DEVICE = "cpu"
SEQ_EPOCHS = 120
SEQ_PATIENCE = 25
SEQ_BATCH = 16
FOLD_LSTM = "fold_2"
LSTM_DEFAULT = {"hidden_dim": 48, "num_layers": 1, "dropout": 0.15, "lr": 5e-4, "wd": 1e-5}


def load_representative_params(cache_root: Path, key: str, fold: str, default: dict) -> dict:
    p = cache_root / key / "kfold" / "best_params.json"
    if not p.exists():
        print(f"  [warning] {p} not found, using default parameters for {key}.")
        return default
    try:
        params = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default
    return params.get(fold, next(iter(params.values()), default))


def train_lstm(seq_df: pd.DataFrame, params: dict) -> tuple:
    """Train LSTM (representative hyperparameters) on train + early stopping on val.

    Return (model, test_df, test_feat, log_cap, test_metrics).
    """
    set_seed(SEED)
    df = seq_df[seq_df["seq_ok"] == 1].copy().reset_index(drop=True)
    split_idx, _ = make_split_indices(df, "group", SEED)
    train_df, val_df, test_df = split_frames(df, split_idx)
    tr_feat, va_feat, te_feat, feats = build_feature_frames(train_df, val_df, test_df)

    tr_x, tr_y, _ = _group_to_sequences(train_df, tr_feat)
    va_x, va_y, _ = _group_to_sequences(val_df, va_feat)
    te_x, te_y, te_gids = _group_to_sequences(test_df, te_feat)

    train_loader = DataLoader(SequenceDataset(tr_x, tr_y), batch_size=SEQ_BATCH,
                              shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(SequenceDataset(va_x, va_y), batch_size=SEQ_BATCH,
                            shuffle=False, collate_fn=collate_sequences)
    test_loader = DataLoader(SequenceDataset(te_x, te_y), batch_size=SEQ_BATCH,
                             shuffle=False, collate_fn=collate_sequences)

    set_seed(SEED)
    model = build_lstm(len(feats), params)
    model, best_epoch, best_val, _ = train_sequence_model(
        model, train_loader, val_loader, params, SEQ_EPOCHS, SEQ_PATIENCE, DEVICE)

    all_train_y = np.concatenate(tr_y)
    log_cap = float(np.max(all_train_y)) * 1.2 if len(all_train_y) else 10.0

    yt_log, yp_log = evaluate_sequence_model(model, test_loader, DEVICE)
    yp_log = np.clip(yp_log, 0.0, log_cap)
    yt, yp = np.expm1(yt_log), np.expm1(yp_log)
    test_metrics = {
        "R2": float(r2_score(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "best_epoch": best_epoch,
    }
    return model, test_df, te_feat, log_cap, test_metrics


def main() -> None:
    if TORCH_IMPORT_ERROR is not None:
        print(build_torch_runtime_message("LSTM (report figures)", TORCH_IMPORT_ERROR))
        return

    package_root = Path(__file__).resolve().parent
    xlsx_path = package_root / "data_raw" / "Database SCC.xlsx"
    cache_root = package_root / "_cache"
    fig_dir = package_root / "output_figures"
    tab_dir = package_root / "output_tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("14_REPORT_FIGURES: Data description (A) + shrinkage curve over time (B)")
    print("=" * 72)
    data_dir = cache_root / "figures" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_outputs = build_datasets(xlsx_path=xlsx_path, output_dir=data_dir, cfg=DataPipelineConfig())
    tab_df = pd.read_csv(data_outputs["point"])
    print(f"Data: {len(tab_df)} measurement points | {tab_df['GID'].nunique()} samples (GID)")

    summ = make_summary_table(tab_df)
    tab_path = tab_dir / "table1_data_summary.csv"
    summ.to_csv(tab_path, index=False, encoding="utf-8-sig")
    print(f"Saved value-range table (Table 1): {tab_path}")
    print(summ[["Column", "N", "Min", "Max", "Mean", "Std"]].to_string(index=False))

    f10 = fig_distributions(tab_df, fig_dir)
    f11 = fig_correlation(tab_df, fig_dir)
    print(f"Saved {f10.name}, {f11.name}")

    print("\nTraining LSTM for the shrinkage curve over time ...")
    seq_df = pd.read_csv(data_outputs["seq"])
    params = load_representative_params(cache_root, "lstm", FOLD_LSTM, LSTM_DEFAULT)
    print(f"Representative hyperparameters ({FOLD_LSTM}): {params}")
    model, test_df, test_feat, log_cap, tm = train_lstm(seq_df, params)
    print(f"  LSTM best_epoch={tm['best_epoch']} | Test R2={tm['R2']:.3f} RMSE={tm['RMSE']:.1f}")
    f12, gids = fig_time_curves(model, test_df, test_feat, log_cap, tm, fig_dir)
    print(f"Saved {f12.name} for the representative GIDs: {gids}")

    print("\n" + "=" * 72)
    print(f"File 14 complete. Figures: {fig_dir} | Tables: {tab_dir}")
    print("=" * 72)


if __name__ == "__main__":
    import time as _t
    _t0 = _t.time()
    main()
    print(f"\n[runtime] {(_t.time() - _t0) / 60.0:.2f} min", flush=True)

