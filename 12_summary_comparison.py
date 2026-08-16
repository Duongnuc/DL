"""Merged file - 12_summary_comparison.py

This file MERGES (without omission) content from 3 original files in the research
pipeline: buoc_12_kfold_driver.py (the comparison-table section - NOT the k-fold
run section, since each model already runs its own k-fold in its own file 01-10),
buoc_22_conformal.py (conformal prediction intervals from out-of-fold residuals), and
buoc_25_bootstrap_ci.py (bootstrap CI clustered by GID + significant pairwise
thong ke).

Unlike files 01-11 (which prepare their own data + train their own models), this file ONLY
READS BACK results already saved in the hidden _cache/<model>/kfold/ directory of files
01-10 (and _cache/empirical/ of file 11) - it does NOT retrain anything. Therefore
FILES 01-11 MUST BE RUN FIRST (at least a few files, to have comparison data), then
only then run this file.

This file RUNS STANDALONE (does not import any other .py file in this translated
package), and consists of 3 parts:
  1. Assemble the Group K-Fold comparison table for all models (mean R2/RMSE/MAE
     +/- standard deviation across 5 folds), cross-checked against the empirical formulas.
  2. Compute conformal prediction intervals (split-conformal, leave-fold-out) for each
     model with at least 2 folds - answering the question "how large can the error be".
  3. Bootstrap 2000 times clustered by GID to get 95% confidence intervals for R2/RMSE, and
     pairwise comparisons between models to see whether the difference is
     statistically significant.

Intermediate CSV results are written to _cache/tong_hop/ (00_kfold_comparison.csv,
conformal/conformal_summary.csv, bootstrap/bootstrap_model_ci.csv,
bootstrap/bootstrap_pairwise.csv).

Run: python 12_summary_comparison.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PACKAGE_ROOT / "_cache"

# Subdirectories (inside _cache/) contain kfold_aggregate.csv + oof_predictions.csv
# for each model, written by files 01-10 once group k-fold has finished running.
MODEL_SUBDIRS = {
    "ANNBasic": "annbasic/kfold",
    "MLP_Tapered": "mlp_tapered/kfold",
    "ResMLP": "resmlp/kfold",
    "FT_Transformer": "ft_transformer/kfold",
    "TabNet": "tabnet/kfold",
    "DCNv2": "dcnv2/kfold",
    "GRU": "gru/kfold",
    "LSTM": "lstm/kfold",
    "TCN": "tcn/kfold",
    "XGBoost": "gbdt/kfold/xgboost",
    "LightGBM": "gbdt/kfold/lightgbm",
}
EMPIRICAL_METRICS_PATH = CACHE_ROOT / "empirical" / "ketqua" / "empirical_metrics.csv"

# Model pairs for statistical comparison (same as buoc_25), keeping only pairs with scientific significance
PAIRS = [
    ("LSTM", "GRU"), ("LSTM", "ResMLP"), ("GRU", "ResMLP"),
    ("LightGBM", "LSTM"), ("LightGBM", "XGBoost"), ("LightGBM", "ResMLP"),
]

# Conformal (buoc_22)
CONFORMAL_ALPHAS = [0.10, 0.05]     # 90% and 95%
CONFORMAL_EPS = 10.0                # minimum y_hat threshold for the normalized variant

# Bootstrap (buoc_25)
BOOT_B = 2000
BOOT_SEED = 42
BOOT_ALPHA = 0.05


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def oof_path(name: str) -> Path:
    return CACHE_ROOT / MODEL_SUBDIRS[name] / "oof_predictions.csv"


def agg_path(name: str) -> Path:
    return CACHE_ROOT / MODEL_SUBDIRS[name] / "kfold_aggregate.csv"


# =============================================================================
# PART 1: Assemble the k-fold comparison table for all models (same as buoc_12, but ONLY
# READS BACK existing results, does not rerun) + cross-checks against the empirical formulas
# (buoc_20/file 11, the calibrated variant on the seq subset, for a fair comparison).
# =============================================================================
def run_comparison_table(out_dir: Path) -> pd.DataFrame:
    frames = []
    missing = []
    for name in MODEL_SUBDIRS:
        df = safe_read_csv(agg_path(name))
        if df.empty:
            missing.append(name)
            continue
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if missing:
        print(f"[note] no k-fold results yet for: {', '.join(missing)} "
              "(please run the corresponding file first, e.g. python 07_gru.py)")

    combined.to_csv(out_dir / "00_kfold_comparison.csv", index=False, encoding="utf-8-sig")
    print("=== Group K-Fold comparison table (read back from _cache/, not rerun) ===")
    if not combined.empty:
        cols = [c for c in ["Model", "Test_R2_mean", "Test_R2_std", "Test_RMSE_mean",
                            "Test_MAE_mean", "epochs_used_mean"] if c in combined.columns]
        if cols:
            print(combined[cols].sort_values("Test_R2_mean", ascending=False).to_string(index=False))
    else:
        print("No model has k-fold results in _cache/ yet.")

    emp = safe_read_csv(EMPIRICAL_METRICS_PATH)
    if not emp.empty:
        emp_best = emp[emp["Model"].str.contains(r"calibrated, seq subset", regex=True)]
        print("\n=== Cross-check against empirical formulas (calibrated, seq_ok=1 subset) ===")
        print(emp_best[["Model", "R2", "RMSE", "MAE", "N"]].to_string(index=False))
    else:
        print("\n[note] no empirical-formula results yet (please run 11_empirical_formula.py first)")

    print(f"\nComparison table: {out_dir / '00_kfold_comparison.csv'}")
    return combined


# =============================================================================
# PART 2: Conformal prediction intervals (same as buoc_22) - only applies to models
# with OOF predictions by fold (needs >=2 folds to calibrate leave-fold-out).
# =============================================================================
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected quantile: ceil((n+1)(1-alpha))/n, method='higher'."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def eval_conformal(name: str, oof: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    rows: list[dict] = []
    detail = oof.copy()
    y, yh, fold = oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(), oof["Fold"].to_numpy()
    abs_res = np.abs(y - yh)
    ok = yh > CONFORMAL_EPS
    rel_res = np.where(ok, abs_res / np.maximum(yh, CONFORMAL_EPS), np.nan)

    for alpha in CONFORMAL_ALPHAS:
        for method in ("tuyet_doi", "chuan_hoa"):
            lo = np.full(len(y), np.nan)
            hi = np.full(len(y), np.nan)
            for k in np.unique(fold):
                test_m = fold == k
                cal_m = ~test_m
                if method == "tuyet_doi":
                    q = conformal_quantile(abs_res[cal_m], alpha)
                    lo[test_m] = yh[test_m] - q
                    hi[test_m] = yh[test_m] + q
                else:
                    cal_scores = rel_res[cal_m & ok]
                    q = conformal_quantile(cal_scores, alpha)
                    lo[test_m] = yh[test_m] * (1.0 - q)
                    hi[test_m] = yh[test_m] * (1.0 + q)
            lo = np.maximum(lo, 0.0)   # shrinkage cannot be negative
            covered = (y >= lo) & (y <= hi)
            width = hi - lo
            rows.append({
                "Model": name, "alpha": alpha, "muc_phu_danh_dinh": 1.0 - alpha,
                "phuong_phap": method,
                "do_phu_thuc_te": float(np.mean(covered)),
                "rong_TB": float(np.mean(width)),
                "rong_trung_vi": float(np.median(width)),
                "N": int(len(y)),
            })
            if alpha == 0.10:
                detail[f"lo90_{method}"] = lo
                detail[f"hi90_{method}"] = hi
    return rows, detail


def run_conformal(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for name in MODEL_SUBDIRS:
        f = oof_path(name)
        if not f.exists():
            continue
        oof = safe_read_csv(f)
        if oof.empty or "y_pred" not in oof.columns:
            continue
        if oof["Fold"].nunique() < 2:
            print(f"[skipped] {name}: only has {oof['Fold'].nunique()} fold "
                  "(needs >=2 to calibrate leave-fold-out)")
            continue
        rows, detail = eval_conformal(name, oof)
        all_rows.extend(rows)
        detail.to_csv(out_dir / f"intervals_{name}.csv", index=False, encoding="utf-8-sig")

    if not all_rows:
        print("No model has enough data yet to compute conformal intervals (needs oof_predictions.csv with >=2 folds).")
        return
    res = pd.DataFrame(all_rows).sort_values(["alpha", "phuong_phap", "Model"])
    res.to_csv(out_dir / "conformal_summary.csv", index=False, encoding="utf-8-sig")

    print("=== Conformal intervals from out-of-fold residuals (leave-fold-out, group-honest) ===")
    for alpha in CONFORMAL_ALPHAS:
        for method in ("tuyet_doi", "chuan_hoa"):
            block = res[(res["alpha"] == alpha) & (res["phuong_phap"] == method)]
            if block.empty:
                continue
            print(f"\n--- nominal coverage level {1-alpha:.0%}, method {method} ---")
            print(block[["Model", "do_phu_thuc_te", "rong_TB", "rong_trung_vi", "N"]]
                  .to_string(index=False))
    print(f"\nConformal results saved at: {out_dir}")


# =============================================================================
# PART 3: Bootstrap CI clustered by GID + pairwise comparison (same as buoc_25) - only
# applies to models that already have oof_predictions.csv (GID, y_true, y_pred).
# =============================================================================
def load_gid_stats(name: str) -> pd.DataFrame:
    """Read OOF and aggregate into sufficient statistics by GID."""
    df = safe_read_csv(oof_path(name))
    if df.empty:
        return pd.DataFrame()
    err = df["y_true"] - df["y_pred"]
    g = pd.DataFrame({
        "GID": df["GID"], "n": 1.0, "sy": df["y_true"],
        "sy2": df["y_true"] ** 2, "sse": err ** 2, "sae": err.abs(),
    }).groupby("GID").sum()
    return g


def pooled_metrics(stats: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, ...]:
    """stats: (G,5) columns [n, sy, sy2, sse, sae]; idx: (B,G) GID indices for resampling."""
    agg = stats[idx].sum(axis=1)
    n, sy, sy2, sse, sae = agg.T
    ss_tot = sy2 - sy ** 2 / n
    r2 = 1.0 - sse / ss_tot
    return r2, np.sqrt(sse / n), sae / n


def bootstrap_ci(x: np.ndarray) -> tuple[float, float]:
    return (float(np.percentile(x, 100 * BOOT_ALPHA / 2)),
            float(np.percentile(x, 100 * (1 - BOOT_ALPHA / 2))))


def run_bootstrap(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    gid_stats = {name: load_gid_stats(name) for name in MODEL_SUBDIRS}
    gid_stats = {name: g for name, g in gid_stats.items() if not g.empty}
    if not gid_stats:
        print("No model has oof_predictions.csv to bootstrap yet.")
        return

    rows = []
    for name, g in gid_stats.items():
        stats = g.to_numpy()
        G = len(g)
        idx = rng.integers(0, G, size=(BOOT_B, G))
        r2, rmse, mae = pooled_metrics(stats, idx)
        r2_pt, rmse_pt, mae_pt = pooled_metrics(stats, np.arange(G)[None, :])
        lo_r2, hi_r2 = bootstrap_ci(r2)
        lo_rm, hi_rm = bootstrap_ci(rmse)
        rows.append({
            "Model": name, "n_GID": G,
            "R2": round(float(r2_pt[0]), 4), "R2_lo": round(lo_r2, 4), "R2_hi": round(hi_r2, 4),
            "RMSE": round(float(rmse_pt[0]), 2), "RMSE_lo": round(lo_rm, 2), "RMSE_hi": round(hi_rm, 2),
            "MAE": round(float(mae_pt[0]), 2),
        })
    ci_df = pd.DataFrame(rows).sort_values("R2", ascending=False)
    ci_df.to_csv(out_dir / "bootstrap_model_ci.csv", index=False, encoding="utf-8-sig")
    print(f"=== 95% CI by model (GID-clustered bootstrap, B={BOOT_B}) ===")
    print(ci_df.to_string(index=False))

    pw_rows = []
    for a, b in PAIRS:
        if a not in gid_stats or b not in gid_stats:
            print(f"[skipped pair] {a} vs {b}: missing OOF data")
            continue
        ga, gb = gid_stats[a], gid_stats[b]
        common = ga.index.intersection(gb.index)
        if len(common) == 0:
            continue
        sa = ga.loc[common].to_numpy()
        sb = gb.loc[common].to_numpy()
        G = len(common)
        idx = rng.integers(0, G, size=(BOOT_B, G))
        r2a, _, _ = pooled_metrics(sa, idx)
        r2b, _, _ = pooled_metrics(sb, idx)
        d = r2a - r2b
        d_pt = (pooled_metrics(sa, np.arange(G)[None, :])[0]
                - pooled_metrics(sb, np.arange(G)[None, :])[0])[0]
        lo, hi = bootstrap_ci(d)
        pw_rows.append({
            "A": a, "B": b, "n_GID_chung": G,
            "dR2": round(float(d_pt), 4), "dR2_lo": round(lo, 4), "dR2_hi": round(hi, 4),
            "P(A>B)": round(float((d > 0).mean()), 3),
            "co_y_nghia_5%": bool(lo > 0 or hi < 0),
        })
    if pw_rows:
        pw_df = pd.DataFrame(pw_rows)
        pw_df.to_csv(out_dir / "bootstrap_pairwise.csv", index=False, encoding="utf-8-sig")
        print("\n=== Pairwise comparison (R2 difference, same bootstrap sample, common GID set) ===")
        print(pw_df.to_string(index=False))
    print(f"\nBootstrap results saved at: {out_dir}")


# =============================================================================
# ACTUAL RUN: assemble comparison table + conformal + bootstrap CI
# =============================================================================
def main() -> None:
    out_root = CACHE_ROOT / "tong_hop"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("[Summary comparison] PART 1/3: Group K-Fold comparison table")
    print("=" * 72)
    run_comparison_table(out_root)

    print("\n" + "=" * 72)
    print("[Summary comparison] PART 2/3: Conformal prediction intervals")
    print("=" * 72)
    run_conformal(out_root / "conformal")

    print("\n" + "=" * 72)
    print("[Summary comparison] PART 3/3: Bootstrap CI + pairwise comparison")
    print("=" * 72)
    run_bootstrap(out_root / "bootstrap")

    print("\nFinished running the entire summary-comparison pipeline.")


if __name__ == "__main__":
    import time as _t
    _t0 = _t.time()
    main()
    print(f"\n[runtime] {(_t.time() - _t0) / 60.0:.2f} min", flush=True)

