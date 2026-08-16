"""Teacher check 1/2: Gradient Boosting with RANDOM 80:20 row split.

This script uses the already-filtered point dataset ``data_train_total_ready.csv``.
It does NOT read ``Database SCC.xlsx`` and does NOT rerun the raw-data filtering step.

The model hyperparameters are exactly those supplied by the supervisor.  Only the
train/test split is controlled here: 80% of measurement rows for training and 20%
for testing.  Because rows from one GID can appear on both sides, this experiment
is intentionally a leakage-prone sanity check and is NOT the paper's main protocol.

Run:
    python 16_gradient_boosting_random_80_20.py
or explicitly point to an already-filtered CSV:
    python 16_gradient_boosting_random_80_20.py --data path/to/data_train_total_ready.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
TARGET = "y_shrinkage_total_mm_per_m"
EXCLUDE_COLUMNS = {"GID", "dt_days", TARGET, "y_log1p", "seq_ok"}
CATEGORICAL_COLUMNS = ["agg_type", "region_type", "Shape_factor"]
SPLIT_SEED = 42

# EXACT parameters supplied by the supervisor.
PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0,
    "learning_rate": 0.1,
    "max_leaf_nodes": 40,
    "max_features": 0.8,
    "random_state": None,
}


def locate_filtered_csv(cli_path: str | None) -> Path:
    """Find an existing filtered point CSV without touching raw Excel data."""
    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Filtered CSV not found: {p}")
        return p

    preferred = [
        ROOT / "_cache" / "gbdt" / "data" / "data_train_total_ready.csv",
        ROOT / "_cache" / "resmlp" / "data" / "data_train_total_ready.csv",
        ROOT / "_cache" / "annbasic" / "data" / "data_train_total_ready.csv",
        ROOT / "_cache" / "lstm" / "data" / "data_train_total_ready.csv",
    ]
    for p in preferred:
        if p.exists():
            return p

    matches = sorted((ROOT / "_cache").glob("**/data_train_total_ready.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "No already-filtered data_train_total_ready.csv was found under _cache/. "
        "Pass it explicitly with --data. This script intentionally does not read raw Excel."
    )


def make_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Train-only median imputation + train-defined one-hot encoding."""
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLUMNS]
    categorical = [c for c in CATEGORICAL_COLUMNS if c in feature_cols]
    numeric = [c for c in feature_cols if c not in categorical]

    train_num = train_df[numeric].apply(pd.to_numeric, errors="coerce").copy()
    test_num = test_df[numeric].apply(pd.to_numeric, errors="coerce").copy()
    med = train_num.median(numeric_only=True).fillna(0.0)
    train_num = train_num.fillna(med)
    test_num = test_num.fillna(med)

    train_blocks = [train_num.to_numpy(dtype=np.float64)]
    test_blocks = [test_num.to_numpy(dtype=np.float64)]
    names = list(numeric)

    for col in categorical:
        tr = train_df[col].astype("string").fillna("<NA>")
        te = test_df[col].astype("string").fillna("<NA>")
        categories = sorted(tr.unique().tolist())
        if not categories:
            continue
        train_blocks.append(np.column_stack([(tr == cat).to_numpy(dtype=float) for cat in categories]))
        test_blocks.append(np.column_stack([(te == cat).to_numpy(dtype=float) for cat in categories]))
        names.extend([f"{col}__{cat}" for cat in categories])

    return np.column_stack(train_blocks), np.column_stack(test_blocks), names


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    iqr = float(np.percentile(y_true, 75) - np.percentile(y_true, 25))
    nrmse_iqr = np.nan if iqr == 0 else rmse / iqr * 100.0
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "NRMSE_IQR_pct": float(nrmse_iqr)}


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, m: dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(y_true, y_pred, s=9, alpha=0.45)
    lim = max(float(np.max(y_true)), float(np.max(y_pred))) * 1.03
    x = np.linspace(0, lim, 300)
    ax.plot(x, x, linewidth=1.2, label="1:1")
    ax.plot(x, 1.4 * x, linestyle="--", linewidth=1.0, label="+40%")
    ax.plot(x, 0.6 * x, linestyle="--", linewidth=1.0, label="-40%")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Measured shrinkage")
    ax.set_ylabel("Predicted shrinkage")
    ax.set_title("Gradient Boosting — Random 80:20 row split")
    ax.text(
        0.03, 0.97,
        f"Test points: {len(y_true)}\nR²: {m['R2']:.3f}\nRMSE: {m['RMSE']:.2f}\nNRMSE(IQR): {m['NRMSE_IQR_pct']:.2f}%",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to existing data_train_total_ready.csv")
    args = parser.parse_args()

    data_path = locate_filtered_csv(args.data)
    df = pd.read_csv(data_path)
    if TARGET not in df.columns or "GID" not in df.columns:
        raise KeyError(f"Filtered CSV must contain GID and {TARGET}")

    # RANDOM ROW 80:20. Fixed split seed is used only to make this comparison reproducible.
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=SPLIT_SEED, shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train, X_test, feature_names = make_features(train_df, test_df)
    y_train_raw = pd.to_numeric(train_df[TARGET], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test_df[TARGET], errors="coerce").to_numpy(dtype=float)

    # Match the current project pipeline: learn log1p(target), report metrics on original scale.
    y_train_log = np.log1p(np.clip(y_train_raw, 0.0, None))
    model = GradientBoostingRegressor(**PARAMS)
    model.fit(X_train, y_train_log)
    pred_log = model.predict(X_test)
    log_cap = float(np.max(y_train_log)) * 1.2
    y_pred = np.expm1(np.clip(pred_log, 0.0, log_cap))

    m = metrics(y_test, y_pred)
    train_gids = set(train_df["GID"].astype(str))
    test_gids = set(test_df["GID"].astype(str))
    gid_overlap = len(train_gids & test_gids)

    out_tables = ROOT / "output_tables"
    out_figures = ROOT / "output_figures"
    out_tables.mkdir(exist_ok=True)
    out_figures.mkdir(exist_ok=True)

    pd.DataFrame({
        "GID": test_df["GID"].astype(str),
        "y_true": y_test,
        "y_pred": y_pred,
    }).to_csv(out_tables / "gradient_boosting_random_80_20_predictions.csv", index=False)

    summary = {
        "split": "RandomRow80_20",
        "data_path": str(data_path),
        "N_total": len(df),
        "N_train": len(train_df),
        "N_test": len(test_df),
        "train_GID": train_df["GID"].nunique(),
        "test_GID": test_df["GID"].nunique(),
        "GID_overlap_train_test": gid_overlap,
        "n_features_after_encoding": len(feature_names),
        **m,
    }
    pd.DataFrame([summary]).to_csv(
        out_tables / "gradient_boosting_random_80_20_metrics.csv", index=False
    )
    plot_predictions(
        y_test, y_pred, m,
        out_figures / "gradient_boosting_random_80_20_pred_vs_measured.png",
    )

    print("=" * 72)
    print("Gradient Boosting — RANDOM 80:20 ROW SPLIT")
    print("=" * 72)
    print(f"Filtered data : {data_path}")
    print(f"Rows          : train={len(train_df)}, test={len(test_df)}")
    print(f"GID overlap   : {gid_overlap}  <-- expected for row-random split")
    print(f"Features      : {len(feature_names)}")
    print(f"R2            : {m['R2']:.6f}")
    print(f"RMSE          : {m['RMSE']:.6f}")
    print(f"MAE           : {m['MAE']:.6f}")
    print(f"NRMSE (IQR)   : {m['NRMSE_IQR_pct']:.3f}%")
    print("Model params  :", PARAMS)
    print("Outputs       : output_tables/gradient_boosting_random_80_20_*.csv")
    print("                output_figures/gradient_boosting_random_80_20_pred_vs_measured.png")


if __name__ == "__main__":
    main()
