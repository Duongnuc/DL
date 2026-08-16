"""Merged file - 09_tcn.py

This file MERGES TOGETHER (without omission) content from the original files in the research pipeline
original: buoc_14_tcn.py, plus the shared pipeline common_sequence_group.py,
common_console_summary.py and buoc_01_code_loc_data.py.

This file is SELF-CONTAINED AND RUNNABLE (does not import any other .py file in the replacement bundle),
and covers all 4 parts of the work:
  1. Prepare sequence-shaped data from the raw Excel file - from buoc_01.
  2. Model architecture definition + Optuna hyperparameter search space - from buoc_14_tcn.py.
  3. Run HOLDOUT (single train/val/test split by GID) AND GROUP K-FOLD (5 fold)
     to evaluate stability, using the shared engine of common_sequence_group.py.
  4. Print all results (R2/RMSE/MAE on train/val/test, best params) to
     the console - not just write to file.

Intermediate CSV/JSON files (models, out-of-fold predictions, best_params, ...) are still
written to the hidden _cache/tcn/ directory so downstream summary files (12, 13) can
read them back without retraining from scratch.

Run: python 09_tcn.py
"""

from __future__ import annotations

# =============================================================================
# PART 1/4 - SHARED PIPELINE FOR SEQUENCE MODELS (merged from common_sequence_group.py)
# NOTE: must be placed before the data-preparation part because this block imports torch BEFORE
# numpy/pandas/sklearn (required on Windows to avoid WinError 1114 when
# loading torch's c10.dll if numpy/MKL was already loaded beforehand).
# =============================================================================

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable

# Disable Dynamo (PyTorch 2.x's JIT compiler) since it is not needed on CPU
# and can cause errors or be slower in some cases.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
# Limit the number of threads to avoid resource contention when running multiple experiments in parallel.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# IMPORTANT: torch must be imported BEFORE numpy/sklearn/optuna.
# On Windows, if numpy/MKL is loaded first, loading torch's c10.dll
# can fail with "[WinError 1114] DLL initialization routine failed".
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Set PyTorch threads = 1 to avoid multi-threading overhead on a small CPU.
# If a RuntimeError occurs (e.g. already set previously), ignore it.
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


# Columns to exclude from the feature set:
# - GID: group code, used to split the data, not a feature
# - dt_days: time (days), already embedded in the sequence order
# - y_shrinkage_total_mm_per_m: target variable, must not be used as input
# - y_log1p: log-transformed target variable, also must not be used as input
# - seq_ok: flag marking a valid sequence, only used for filtering
EXCLUDE_COLUMNS = ["GID", "dt_days", "y_shrinkage_total_mm_per_m", "y_log1p", "seq_ok"]
# Categorical columns will be converted to one-hot encoding.
SEQ_ONE_HOT_COLUMNS = ["agg_type", "region_type", "Shape_factor"]


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


@dataclass(slots=True)
class SplitIndices:
    """Stores the index of rows belonging to the train, val, test sets.

    After splitting the data by group (GID), each row in the original DataFrame
    is assigned to one of the three sets. This dataclass stores those indices
    so the data can be extracted later.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def portable_path(path: Path) -> str:
    """Converts an absolute path into a path relative to the package root directory.

    Purpose: save into the metadata JSON so the results can be read on
    different machines (portable). If the relative path cannot be computed,
    return a POSIX-style path (using / separators).
    """
    package_root = Path(__file__).resolve().parent
    try:
        return path.resolve().relative_to(package_root).as_posix()
    except Exception:
        return path.as_posix()


def write_skip_outputs(cfg: SequenceBenchmarkConfig, reason: str) -> dict[str, Path]:
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


def set_seed(seed: int) -> None:
    """Sets the random seed for all libraries so results can be reproduced.

    Deep learning has many sources of randomness: weight initialization, data shuffling,
    dropout, etc. Setting the same seed helps reruns produce the same result (reproducibility).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes evaluation metrics for a regression model.

    Returns a dict containing:
        - N: number of samples
        - R2: coefficient of determination (1.0 is perfect, can be negative if predictions are very bad)
        - RMSE: root mean squared error (same unit as y)
        - MAE: mean absolute error (same unit as y)
        - NRMSE_IQR_pct: RMSE normalized by the interquartile range (%), helps compare
          across datasets with different distributions
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    iqr = float(np.percentile(y_true, 75) - np.percentile(y_true, 25))
    nrmse_iqr_pct = 0.0 if iqr == 0 else float(rmse / iqr * 100.0)
    return {
        "N": int(y_true.shape[0]),
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "NRMSE_IQR_pct": nrmse_iqr_pct,
    }


def aggregate_results(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates results from multiple seeds into a mean and standard deviation.

    When running multiple seeds, each seed produces one result row. This function groups
    by (Split, Family, Model) then computes mean and std for each metric,
    helping evaluate whether the model is stable.
    """
    if summary_df.empty:
        return pd.DataFrame()
    group_cols = ["Split", "Family", "Model"]
    numeric_cols = [
        col
        for col in summary_df.columns
        if col not in {"Seed", "Split", "Family", "Model"}
        and pd.api.types.is_numeric_dtype(summary_df[col])
    ]
    grouped = summary_df.groupby(group_cols)
    agg = grouped[numeric_cols].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()
    agg.insert(len(group_cols), "runs", grouped.size().to_numpy())
    return agg


def make_quantile_bins(values: pd.Series, q: int = 5) -> pd.Series | None:
    """Splits values into q quantile-based groups for use in a stratified split.

    Stratified split means splitting the data so that each set (train/val/test)
    has a similar distribution. This function creates group labels (0, 1, ..., q-1) based
    on quantiles. Returns None if there isn't enough data to split (each group needs
    at least 2 elements).
    """
    values = pd.to_numeric(values, errors="coerce")
    if values.isna().all():
        return None
    bins = pd.qcut(values, q=q, labels=False, duplicates="drop")
    counts = pd.Series(bins).value_counts(dropna=False)
    if counts.empty or counts.min() < 2:
        return None
    return pd.Series(bins)


def group_stratified_split(df: pd.DataFrame, seed: int) -> SplitIndices:
    """Splits the data into train/val/test by GID group (group-wise split).

    This is the most important step to avoid data leakage in the sequence problem:
    - Each GID (experimental sample group) appears in only ONE set.
    - If the same GID is present in both train and test, the model will "memorize" instead
      of "learn", leading to a biased evaluation (hidden overfitting).

    Procedure:
    1. Compute the median fc28 of each GID to use as the stratification strategy.
       This ensures the sets have similar fc28 distributions.
    2. Split GID into train (70%) and temporary (temp, 30%) via stratified split.
    3. Split temp further into val (15%) and test (15%) via stratified split.
    4. Map back from GID to the row indices in the original DataFrame.
    """
    # Compute median fc28 for each GID to use as the stratification criterion
    group_stats = (
        df.groupby("GID", as_index=False)
        .agg(fc28=("fc28", "median"))
        .sort_values("GID")
        .reset_index(drop=True)
    )
    strat = make_quantile_bins(group_stats["fc28"])
    strat_values = None if strat is None else strat.to_numpy()
    all_groups = group_stats["GID"].to_numpy()
    # Step 1: split off 70% train, 30% temporary (temp)
    train_groups, temp_groups = train_test_split(
        all_groups,
        test_size=0.3,
        random_state=seed,
        shuffle=True,
        stratify=strat_values,
    )
    # Step 2: split the 30% temp into 15% val + 15% test
    temp_stats = group_stats[group_stats["GID"].isin(temp_groups)].reset_index(drop=True)
    temp_strat = make_quantile_bins(temp_stats["fc28"])
    temp_strat_values = None if temp_strat is None else temp_strat.to_numpy()
    val_groups, test_groups = train_test_split(
        temp_stats["GID"].to_numpy(),
        test_size=0.5,
        random_state=seed,
        shuffle=True,
        stratify=temp_strat_values,
    )
    # Reverse mapping: from the GID list of each set, find the corresponding row indices
    gid_values = df["GID"].to_numpy()
    return SplitIndices(
        train_idx=np.where(np.isin(gid_values, train_groups))[0],
        val_idx=np.where(np.isin(gid_values, val_groups))[0],
        test_idx=np.where(np.isin(gid_values, test_groups))[0],
    )


def group_kfold_split_indices(df: pd.DataFrame, n_splits: int, seed: int, val_fraction: float = 0.1765) -> list[tuple[SplitIndices, int]]:
    """Creates folds for GROUP K-FOLD cross-validation on sequence data.

    A copy of the logic in common_tabular but used for the sequence pipeline:
    - Each GID (sample group) belongs to the test set EXACTLY ONCE across K folds, never
      appearing in both train and test within the same fold -> no data leakage.
    - Stratified by each GID's median fc28 so the folds have balanced distributions.

    Process for each fold k:
        test = GIDs of fold k; the rest is split into train/val (val is approximately val_fraction
        of the remainder, stratified by fc28 if possible).

    Returns: a list of [(SplitIndices, fold_id)] with fold_id running 0..K-1.
    """
    group_stats = df.groupby("GID", as_index=False).agg(fc28=("fc28", "median")).sort_values("GID").reset_index(drop=True)
    all_groups = group_stats["GID"].to_numpy()
    strat = make_quantile_bins(group_stats["fc28"])
    gid_values = df["GID"].to_numpy()

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


def split_diagnostics(df: pd.DataFrame, split_idx: SplitIndices, seed: int) -> dict[str, Any]:
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


def split_frames(df: pd.DataFrame, split_idx: SplitIndices) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extracts the sub-DataFrame for each of the train, val, test sets from the split indices.

    Resets the index so the sub-DataFrames have a contiguous index starting at 0,
    avoiding errors when accessed by position (iloc) later.
    """
    train_df = df.iloc[split_idx.train_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx.val_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx.test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


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
    diag = split_diagnostics(df, split_idx, seed)
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
        return write_skip_outputs(cfg, f"Insufficient memory or an error occurred while training the sequence model. Original error: {exc}")

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
            diag = split_diagnostics(df, split_idx, seed)
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
        return write_skip_outputs(cfg, f"Error while running sequence k-fold ({cfg.model_name}). Original error: {exc}")

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


# common_sequence_group.py imports torch WITHOUT try/except (unlike common_tabular.py):
# if execution reaches this point it means torch was already imported successfully above, so it will always be None.
TORCH_IMPORT_ERROR: Exception | None = None
SEQ_IMPORT_ERROR: Exception | None = None

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
# PART 3/4 - MODEL ARCHITECTURE + SEARCH SPACE (merged from buoc_14_tcn.py)
# =============================================================================
"""Step 14 - TCN (Temporal Convolutional Network) model - 1D CNN for shrinkage sequences.

=================================================================================
WHAT IS TCN?
=================================================================================
TCN (Temporal Convolutional Network) is a 1D CNN architecture
for sequence data, proposed by Bai, Kolter & Koltun (2018).
Instead of processing step-by-step sequentially like RNNs (GRU/LSTM), TCN uses 1D CONVOLUTION
sliding along the time axis. This is a REPRESENTATIVE DL family quite different from RNNs, adding
to the GRU/LSTM comparison in the paper.

Three core components:

1) CAUSAL CONVOLUTION:
   - The output at time t depends ONLY on the input at t and steps BEFORE t
     (no "peeking" at the future). Implemented by padding on the LEFT then
     trimming the excess on the right (Chomp). Suited to forecasting shrinkage over time.

2) DILATED CONVOLUTION:
   - The dilation factor increases exponentially across layers: 1, 2, 4, 8...
   - This lets the receptive field grow EXPONENTIALLY with the number of layers,
     letting it "see far" into the past without needing too many layers -> capturing the
     long-term trend of the shrinkage curve.

3) RESIDUAL BLOCK:
   - Each block = [dilated Conv -> normalization -> ReLU -> Dropout] x2, plus a skip
     connection. The skip connection helps stabilize deep training (preventing
     vanishing gradients) - exactly what ANNBasic LACKS, which is why it can blow up on OOD data.

=================================================================================
WHY weight_norm INSTEAD OF BatchNorm?
=================================================================================
Sequences are padded at the END to form a batch. BatchNorm computes statistics over the whole batch and
time axis -> padding positions (value 0) would distort the statistics. weight_norm
normalizes the WEIGHTS (independent of data/padding), so it is more stable with padded
sequences + masking. This is the standard choice in the original TCN paper.

=================================================================================
PREVENTING OVERFITTING (lessons learned from TCN overfitting previously)
=================================================================================
Shrinkage sequence data has few GID groups -> TCN easily overfits. So the search space keeps:
  - Few channels (16-40), moderate dilation (2-4 layers).
  - High dropout floor (0.1-0.4), strong weight decay (up to 3e-3).
If it still overfits on k-fold -> report it honestly, as with ANNBasic/TabNet.

Run standalone:  python buoc_14_tcn.py
"""


from torch.nn.utils import weight_norm  # TCN uses weight_norm instead of BatchNorm (sequences have padding)

MODEL_NAME = "TCN"


def build_runtime_error_message() -> str:
    return (
        f"Cannot initialize the {MODEL_NAME} model. "
        f"torch error: {TORCH_IMPORT_ERROR}. helper error: {SEQ_IMPORT_ERROR}. "
        "Suggestion: use a dedicated DL environment, install torch."
    )


# ---------------------------------------------------------------------------
# TCN architecture
# ---------------------------------------------------------------------------
class Chomp1d(nn.Module):
    """Trim the last `chomp_size` elements along the time axis.

    After padding (k-1)*dilation on the left to preserve causality, the Conv1d output length
    is (k-1)*dilation elements longer than the input at the end. Chomp trims this excess
    so the output has the same length as the input AND does not use future information.
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: (batch, channels, time). Trim chomp_size elements off the end if present.
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """A TCN residual block: [dilated Conv -> Chomp -> ReLU -> Dropout] x2 + skip."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation  # left padding to preserve causality
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )
        # Skip connection: if the number of input channels != output channels, use a 1x1 conv to match dimensions.
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNRegressor(nn.Module):
    """TCN seq2seq: predicts shrinkage at EVERY time step.

    Parameters:
        input_dim   (int):   number of input features per step.
        num_channels(int):   number of hidden channels per convolutional layer.
        levels      (int):   number of TemporalBlocks (dilation = 1,2,4,... by layer).
        kernel_size (int):   convolution window size.
        dropout     (float): dropout rate within each block.
    """

    def __init__(self, input_dim: int, num_channels: int, levels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for i in range(levels):
            dilation = 2 ** i  # dilation increases exponentially
            layers.append(TemporalBlock(prev, num_channels, kernel_size, dilation, dropout))
            prev = num_channels
        self.tcn = nn.Sequential(*layers)
        # FC head: maps num_channels -> 1 at every time step (using a 1x1 Conv1d).
        self.head = nn.Conv1d(num_channels, 1, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)
        y = self.tcn(x)          # (batch, num_channels, seq_len)
        y = self.head(y)         # (batch, 1, seq_len)
        return y.squeeze(1)      # (batch, seq_len)


def build_model(input_dim: int, params: dict) -> nn.Module:
    """Build a TCNRegressor from the hyperparameters proposed by Optuna."""
    return TCNRegressor(
        input_dim=input_dim,
        num_channels=int(params["num_channels"]),
        levels=int(params["levels"]),
        kernel_size=int(params["kernel_size"]),
        dropout=float(params["dropout"]),
    )


def suggest_params(trial) -> dict:
    """Search space for TCN - regularize strongly to prevent overfitting on sequences with few groups."""
    return {
        # few channels to limit capacity (prevent overfitting)
        "num_channels": trial.suggest_categorical("num_channels", [16, 24, 32, 40]),
        # number of layers -> receptive field grows exponentially
        "levels": trial.suggest_int("levels", 2, 4),
        # convolution window
        "kernel_size": trial.suggest_categorical("kernel_size", [2, 3]),
        # high dropout floor
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        # learning rate (log)
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        # STRONG weight decay (log) to keep weights small, stable on OOD test data
        "wd": trial.suggest_float("wd", 1e-5, 3e-3, log=True),
    }


def run_benchmark(cfg: SequenceBenchmarkConfig) -> dict[str, Path]:
    """Run the full holdout pipeline: tune Optuna -> train -> evaluate -> save results."""
    if TORCH_IMPORT_ERROR is not None or SEQ_IMPORT_ERROR is not None:
        print(build_runtime_error_message())
        return write_skip_outputs(cfg, build_runtime_error_message())
    return run_sequence_benchmark(cfg, build_model, suggest_params)


# =============================================================================
# PART 4/4 - ACTUAL RUN: HOLDOUT + GROUP K-FOLD + PRINT RESULTS
# =============================================================================
SEED = 42
N_SPLITS = 5
N_TRIALS_KFOLD = 8
SEQ_MAX_EPOCHS = 80
SEQ_PATIENCE = 12
SEQ_BATCH_SIZE = 16


def main() -> None:
    if TORCH_IMPORT_ERROR is not None or SEQ_IMPORT_ERROR is not None:
        print(build_runtime_error_message())
        return

    package_root = Path(__file__).resolve().parent
    xlsx_path = package_root / "data_raw" / "Database SCC.xlsx"
    cache_dir = package_root / "_cache" / "tcn"

    print("=" * 72)
    print(f"[{MODEL_NAME}] STEP 1/3: Prepare sequence-shaped data (filter + feature engineering)")
    print("=" * 72)
    data_dir = cache_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_outputs = build_datasets(xlsx_path=xlsx_path, output_dir=data_dir, cfg=DataPipelineConfig())
    data_path = data_outputs["seq"]
    print(f"Finished preparing data: {data_path}")

    print("\n" + "=" * 72)
    print(f"[{MODEL_NAME}] STEP 2/3: Holdout (train/val/test once by GID, Optuna 3 trials)")
    print("=" * 72)
    holdout_cfg = SequenceBenchmarkConfig(
        data_path=data_path, output_dir=cache_dir / "holdout", model_name=MODEL_NAME,
        seeds=[SEED], n_trials=3, max_epochs=60, patience=10,
        batch_size=16,
    )
    holdout_outputs = run_benchmark(holdout_cfg)
    print_model_console_summary(MODEL_NAME, holdout_outputs)

    print("\n" + "=" * 72)
    print(f"[{MODEL_NAME}] STEP 3/3: Group K-Fold ({N_SPLITS} fold, Optuna {N_TRIALS_KFOLD} trials)")
    print("=" * 72)
    kfold_cfg = SequenceBenchmarkConfig(
        data_path=data_path, output_dir=cache_dir / "kfold", model_name=MODEL_NAME,
        seeds=[SEED], n_splits=N_SPLITS, n_trials=N_TRIALS_KFOLD,
        max_epochs=SEQ_MAX_EPOCHS, patience=SEQ_PATIENCE, batch_size=SEQ_BATCH_SIZE,
    )
    kfold_outputs = run_sequence_kfold(kfold_cfg, build_model, suggest_params)
    print_kfold_console_summary(MODEL_NAME, kfold_outputs)

    fig_dir = package_root / "output_figures"
    plot_kfold_prediction_figures(MODEL_NAME, "tcn", kfold_outputs, fig_dir)

    print(f"\nFinished running the full pipeline for {MODEL_NAME} (holdout + k-fold).")


if __name__ == "__main__":
    import time as _t
    _t0 = _t.time()
    main()
    print(f"\n[runtime] {(_t.time() - _t0) / 60.0:.2f} min", flush=True)
