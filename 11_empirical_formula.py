"""Merged file - 11_empirical_formula.py

This file MERGES (without omission) content from buoc_20_empirical_baseline.py
(a benchmark using empirical physical formulas: ACI 209R-92, B3, B4, fib
MC2010), plus buoc_01_code_loc_data.py to self-prepare the data.

Unlike files 01-10 (machine learning models that need torch/Optuna), this file ONLY uses
numpy/pandas to compute the current design formulas on EACH row of data, then
score R2/RMSE/MAE to compare fairly against the deep learning models (step
12_tong_hop_so_sanh.py will read this result back). Since it doesn't use torch, this file
does NOT need the shared pipeline common_tabular.py/common_sequence_group.py.

This file RUNS STANDALONE (does not import any other .py file in this translated
package), and consists of 3 parts:
  1. Prepare data from the raw Excel file (filtering + feature engineering) - from buoc_01.
  2. Define the 5 empirical formulas (ACI 209R-92, B3, B4, B4 drying only, fib
     MC2010) - from buoc_20_empirical_baseline.py.
  3. Compute predictions for each row, score R2/RMSE/MAE for both the standard version and the
     calibrated version (multiplied by a least-squares-optimal factor), over the entire data AND
     over the seq_ok=1 subset (the same set of points GRU/LSTM/TCN are scored OOF on, for a
     fair comparison), then PRINT THE RESULTS to the console.

Intermediate CSV files are still written to the hidden _cache/empirical/ directory so the downstream summary file
(12_tong_hop_so_sanh.py) can read them back without needing to recompute
from scratch.

Run: python 11_empirical_formula.py
"""

from __future__ import annotations

# =============================================================================
# PART 1/3 - DATA PREPARATION (merged from buoc_01_code_loc_data.py)
# Already contains "import numpy as np", "import pandas as pd", "from pathlib
# import Path" - no torch needed so no need to import TAB_ENGINE/SEQ_ENGINE first.
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
# PART 2/3 - 5 EMPIRICAL FORMULAS (merged from buoc_20_empirical_baseline.py)
# =============================================================================
"""Step 20 - Empirical formula benchmark: ACI 209R-92, B3, B4, fib MC2010.

The most common objection to a machine-learning shrinkage prediction paper: where exactly does the deep learning model outperform
the current design formulas (ACI 209, B3, B4, fib MC2010)? This is not
an ML model, but an empirical physical formula, so it serves as a benchmark
without breaking the "deep learning only" constraint.

Fair comparison method: compute each formula for EVERY row in the dataset, then
score R2/RMSE against the measured value, comparing against the pooled out-of-fold metric of the
deep learning models (every GID is tested exactly once within the group 5-fold).

Formula sources (cross-checked against the originals, PDFs stored in _tmp_refs/):
  - ACI 209R-92: ACI Committee 209 report.
  - B3: Bažant & Baweja, ACI SP-194 / RILEM recommendation (drying shrinkage only).
  - B4: RILEM TC-242-MDC, Mater Struct 48:753-770 (2015) - drying + autogenous.
  - fib MC2010: Eqs. 5.1-76..83 - basic (autogenous) + drying.

Honest assumptions (stated explicitly in the paper):
  - No slump/air content data (ACI: coefficient = 1.0).
  - Cement type: B3 Type I (α1=1), B4 type R, MC2010 class 42.5N; moist curing
    (α2=1). B4 aggregate: k_τa = k_εa = 1.0 (parent rock type unknown per Table 6).
  - w/c taken as w/cm and c as the total binder content (SCC uses a lot of SCM).
  - Two variants reported per formula: (a) standard; (b) calibrated, scaled entirely
    by a least-squares-optimal factor on the data (the best possible advantage
    that can be given to the empirical formula).

Run:  python buoc_20_empirical_baseline.py --smoke    (check 1 curve)
      python buoc_20_empirical_baseline.py            (run everything)
"""


def aci209_shrinkage(row: pd.Series) -> float:
    """ACI 209R-92 shrinkage strain (×10^-6) for one row (one time point).

    eps_sh(t,tc) = (t-tc)/(f + (t-tc)) * eps_shu
    eps_shu = 780e-6 * gamma_sh, gamma_sh = product of the correction factors.
    """
    dt = float(row["dt_days"])           # drying time t - tc (days)
    if dt <= 0:
        return 0.0
    tc = float(row["t0"])                # age at start of drying (days)

    # Time: f = 35 for 7-day moist curing (ACI standard value)
    f = 35.0
    time_ratio = dt / (f + dt)

    # Moist-curing coefficient based on the age at start of drying tc (ACI 209 Table)
    cp_t = np.array([1.0, 3.0, 7.0, 14.0, 28.0, 90.0])
    cp_v = np.array([1.2, 1.1, 1.0, 0.93, 0.86, 0.75])
    gamma_cp = float(np.interp(tc, cp_t, cp_v))

    # Relative humidity coefficient h (%): RH_test is a fraction -> multiply by 100
    h = float(row["RH_test"]) * 100.0
    if h <= 80.0:
        gamma_rh = 1.40 - 0.0102 * h
    else:
        gamma_rh = 3.00 - 0.030 * h
    gamma_rh = max(gamma_rh, 0.0)

    # Volume-to-surface ratio coefficient V/S (mm)
    vs = float(row["V_S"])
    gamma_vs = 1.2 * np.exp(-0.00472 * vs)

    # Fine aggregate ratio coefficient psi (%): sandrate is a fraction -> multiply by 100
    psi = float(row["sandrate"]) * 100.0
    if psi <= 50.0:
        gamma_psi = 0.30 + 0.014 * psi
    else:
        gamma_psi = 0.90 + 0.002 * psi

    # Binder content coefficient c (kg/m3)
    c = float(row["cm"])
    gamma_c = 0.75 + 0.00061 * c

    # No slump or air content -> coefficient = 1.0 (assumption)
    gamma_s = 1.0
    gamma_alpha = 1.0

    gamma_sh = gamma_cp * gamma_rh * gamma_vs * gamma_psi * gamma_c * gamma_s * gamma_alpha
    eps_shu = 780.0 * gamma_sh          # units ×10^-6
    return time_ratio * eps_shu


# Shape factor k_s (B3 Eq. 1.12 = B4): 1 infinite slab, 2 infinite cylinder,
# 3 infinite square prism, 4 sphere, 5 cube - matches the Shape_factor column encoding.
KS_MAP = {1: 1.00, 2: 1.15, 3: 1.25, 4: 1.30, 5: 1.55}


def b3_shrinkage(row: pd.Series) -> float:
    """B3 (Bažant-Baweja, SI) DRYING shrinkage (×10^-6); Type I, moist curing."""
    dt = float(row["dt_days"])
    if dt <= 0:
        return 0.0
    t0 = float(row["t0"])
    w = float(row["water"])              # kg/m3
    fc = float(row["fc28"])              # MPa (mean strength)

    eps_sinf = 0.019 * w ** 2.1 * fc ** (-0.28) + 270.0   # ×10^-6, alpha1=alpha2=1
    ks = KS_MAP[int(row["Shape_factor"])]
    d_cm = 2.0 * float(row["V_S"]) / 10.0                 # D = 2v/s, cm
    kt = 8.5 * t0 ** (-0.08) * fc ** (-0.25)              # ngay/cm^2 (Eq. 1.20)
    tau_sh = kt * (ks * d_cm) ** 2

    def e_rel(t: float) -> float:                          # E(t)/E28 (Eq. 1.13)
        return np.sqrt(t / (4.0 + 0.85 * t))

    eps_shinf = eps_sinf * e_rel(607.0) / e_rel(t0 + tau_sh)
    h = float(row["RH_test"])                              # fraction, <=0.98 in the data
    kh = 1.0 - h ** 3
    s_t = np.tanh(np.sqrt(dt / tau_sh))
    return eps_shinf * kh * s_t


# B4 (RILEM TC-242-MDC 2015), cement type R; k_tau_a = k_eps_a = 1.0
B4_RHO = 2350.0
B4_DRY = dict(tau_cem=0.016, p_ta=-0.33, p_tw=-0.06, p_tc=-0.10,
              eps_cem=360.0, p_ea=-0.80, p_ew=1.10, p_ec=0.11)
B4_AU = dict(tau_au_cem=1.00, r_tw=3.00, r_t=-4.50, r_alpha=1.00,
             eps_au_cem=210.0, r_ea=-0.75, r_ew=-3.50)


def _b4_parts(row: pd.Series) -> tuple[float, float]:
    """B4: returns (drying, autogenous) ×10^-6, measured from time t0."""
    dt = float(row["dt_days"])
    if dt <= 0:
        return 0.0, 0.0
    t0 = float(row["t0"])
    ac = float(row["a_cm"])
    wc = float(row["w_cm"])
    c = float(row["cm"])

    # --- kho ---
    p = B4_DRY
    fa = (ac / 6.0, wc / 0.38, 6.5 * c / B4_RHO)
    eps0 = p["eps_cem"] * fa[0] ** p["p_ea"] * fa[1] ** p["p_ew"] * fa[2] ** p["p_ec"]
    tau0 = p["tau_cem"] * fa[0] ** p["p_ta"] * fa[1] ** p["p_tw"] * fa[2] ** p["p_tc"]
    ks = KS_MAP[int(row["Shape_factor"])]
    d_mm = 2.0 * float(row["V_S"])                         # D = 2V/S, mm
    tau_sh = tau0 * (ks * d_mm) ** 2                       # k_tau_a = 1

    def e_rel(t: float) -> float:                          # E(t)/E28, B4 uses 6/7
        return np.sqrt(t / (4.0 + (6.0 / 7.0) * t))

    eps_shinf = eps0 * e_rel(607.0) / e_rel(t0 + tau_sh)   # k_eps_a = 1
    h = float(row["RH_test"])
    kh = 1.0 - h ** 3
    eps_dry = eps_shinf * kh * np.tanh(np.sqrt(dt / tau_sh))

    # --- autogenous (computed by absolute age t, minus the part that occurred before t0) ---
    a = B4_AU
    eps_au_inf = a["eps_au_cem"] * (ac / 6.0) ** a["r_ea"] * (wc / 0.38) ** a["r_ew"]
    tau_au = a["tau_au_cem"] * (wc / 0.38) ** a["r_tw"]
    alpha = a["r_alpha"] * wc / 0.38

    def eps_au(t: float) -> float:
        return eps_au_inf * (1.0 + (tau_au / t) ** alpha) ** a["r_t"]

    return eps_dry, eps_au(t0 + dt) - eps_au(t0)


def b4_shrinkage(row: pd.Series) -> float:
    """B4 TOTAL shrinkage = drying + autogenous."""
    dry, au = _b4_parts(row)
    return dry + au


def b4_drying_shrinkage(row: pd.Series) -> float:
    """B4 DRYING part ONLY - the autogenous part extrapolates very strongly when w/c << 0.38."""
    return _b4_parts(row)[0]


# fib MC2010, lop xi mang 42.5N: alpha_bs=700, alpha_ds1=4, alpha_ds2=0.012
MC_ABS, MC_ADS1, MC_ADS2 = 700.0, 4.0, 0.012


def mc2010_shrinkage(row: pd.Series) -> float:
    """fib MC2010 TOTAL shrinkage = basic + drying (×10^-6), measured from time ts=t0."""
    dt = float(row["dt_days"])
    if dt <= 0:
        return 0.0
    ts = float(row["t0"])
    fcm = float(row["fc28"])

    # basic (autogenous), Eq. 5.1-78..79 - measured from ts so subtract beta_bs(ts)
    eps_cbs0 = MC_ABS * ((0.1 * fcm) / (6.0 + 0.1 * fcm)) ** 2.5

    def beta_bs(t: float) -> float:
        return 1.0 - np.exp(-0.2 * np.sqrt(t))

    eps_basic = eps_cbs0 * (beta_bs(ts + dt) - beta_bs(ts))

    # kho, Eq. 5.1-80..83
    eps_cds0 = (220.0 + 110.0 * MC_ADS1) * np.exp(-MC_ADS2 * fcm)
    rh = float(row["RH_test"]) * 100.0
    beta_s1 = min((35.0 / fcm) ** 0.1, 1.0)
    if rh < 99.0 * beta_s1:
        beta_rh = 1.55 * (1.0 - (rh / 100.0) ** 3)
    else:
        beta_rh = -0.25                                    # swelling case
    h_mm = 2.0 * float(row["V_S"])                         # h = 2Ac/u
    beta_ds = np.sqrt(dt / (0.035 * h_mm ** 2 + dt))
    return eps_basic + eps_cds0 * beta_rh * beta_ds


EMPIRICAL_MODELS = {
    "ACI 209R-92": aci209_shrinkage,
    "B3": b3_shrinkage,
    "B4": b4_shrinkage,
    "B4 (drying only)": b4_drying_shrinkage,
    "fib MC2010": mc2010_shrinkage,
}



def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Simple R2/RMSE/MAE (different from regression_metrics in common_tabular.py:
    no NRMSE_IQR_pct since this isn't an ML pipeline with train/val/test)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    resid = y_true - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "N": int(len(y_true))}


# =============================================================================
# PART 3/3 - ACTUAL RUN: COMPUTE FORMULAS + SCORE + PRINT RESULTS
# =============================================================================
def main() -> None:
    package_root = Path(__file__).resolve().parent
    xlsx_path = package_root / "data_raw" / "Database SCC.xlsx"
    cache_dir = package_root / "_cache" / "empirical"

    print("=" * 72)
    print("[Empirical formula] STEP 1/2: Prepare data (filtering + feature engineering)")
    print("=" * 72)
    data_dir = cache_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_outputs = build_datasets(xlsx_path=xlsx_path, output_dir=data_dir, cfg=DataPipelineConfig())
    df = pd.read_csv(data_outputs["point"])
    seq = pd.read_csv(data_outputs["seq"])
    y = df["y_shrinkage_total_mm_per_m"].to_numpy(float)
    print(f"Finished preparing data: {data_outputs['point']} ({len(df)} points)")

    print("\n" + "=" * 72)
    print("[Empirical formula] STEP 2/2: Compute 5 formulas + score R2/RMSE/MAE")
    print("=" * 72)
    seq_gids = set(seq.loc[seq["seq_ok"] == 1, "GID"].unique())
    mask = df["GID"].isin(seq_gids).to_numpy()
    print(f"Seq subset (same points GRU/LSTM/TCN score OOF): {int(mask.sum())}/{len(df)} points")

    out_dir = cache_dir / "ketqua"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    out_pred = df[["GID", "dt_days", "y_shrinkage_total_mm_per_m"]].copy()
    for name, fn in EMPIRICAL_MODELS.items():
        pred = df.apply(fn, axis=1).to_numpy(float)
        k = float(np.dot(y, pred) / np.dot(pred, pred)) if np.dot(pred, pred) > 0 else 1.0
        y_s, pred_s = y[mask], pred[mask]
        k_s = float(np.dot(y_s, pred_s) / np.dot(pred_s, pred_s)) if np.dot(pred_s, pred_s) > 0 else 1.0
        rows += [
            {"Model": f"{name} (standard)", **metrics(y, pred), "scale": 1.0},
            {"Model": f"{name} (calibrated)", **metrics(y, k * pred), "scale": k},
            {"Model": f"{name} (standard, seq subset)", **metrics(y_s, pred_s), "scale": 1.0},
            {"Model": f"{name} (calibrated, seq subset)", **metrics(y_s, k_s * pred_s), "scale": k_s},
        ]
        tag = name.lower().replace(" ", "_").replace("-", "_")
        out_pred[f"{tag}_raw"] = pred
        out_pred[f"{tag}_cal"] = k * pred

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "empirical_metrics.csv", index=False, encoding="utf-8-sig")
    out_pred.to_csv(out_dir / "empirical_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n=== Empirical formulas on the entire dataset (comparison with DL OOF) ===")
    print(res.to_string(index=False))
    print(f"\nWritten: {out_dir}")
    print("\nFinished running the entire pipeline for the empirical formulas.")


if __name__ == "__main__":
    import time as _t
    _t0 = _t.time()
    main()
    print(f"\n[runtime] {(_t.time() - _t0) / 60.0:.2f} min", flush=True)
