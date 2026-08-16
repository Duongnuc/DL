import sys
import os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
m = importlib.import_module("10_gbdt")

# Monkey-patch tiny-scale constants for a fast smoke test.
m.N_SPLITS = 2
m.N_TRIALS = 1
m.MAX_ROUNDS = 50
m.EARLY_STOP = 10

package_root = Path(__file__).resolve().parent.parent
xlsx_path = package_root / "data_raw" / "Database SCC.xlsx"
cache_dir = package_root / "_cache" / "_smoke_gbdt"

print("=" * 72)
print("[GBDT] SMOKE STEP 1: Prepare data")
print("=" * 72)
data_dir = cache_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)
data_outputs = m.build_datasets(xlsx_path=xlsx_path, output_dir=data_dir, cfg=m.DataPipelineConfig())
data_path = data_outputs["point"]
print(f"Finished preparing data: {data_path}")
import pandas as pd
df = pd.read_csv(data_path)

print("\n" + "=" * 72)
print("[GBDT] SMOKE STEP 2: Group K-Fold (tiny) - XGBoost")
print("=" * 72)
kfold_outputs = m.run_gbdt_kfold("XGBoost", df, data_path, cache_dir / "kfold" / "xgboost", m.N_TRIALS, folds_limit=2)
m.print_kfold_console_summary("XGBoost", kfold_outputs)
fig_dir = cache_dir / "output_figures"
m.plot_kfold_prediction_figures("XGBoost", "xgboost", kfold_outputs, fig_dir)

print("\n" + "=" * 72)
print("[GBDT] SMOKE STEP 3: Robustness check (tiny) - source split, XGBoost")
print("=" * 72)
src = m.source_of(df["GID"])
print(f"Data: {len(df)} points, {df['GID'].nunique()} GID, {src.nunique()} sources")
folds_src = m.source_kfold_split_indices(df, m.N_SPLITS, m.SEED)
robust_result = m.run_robust_kfold(
    "XGBoost", df, folds_src, m.SEED,
    cache_dir / "robustness_src" / "xgboost", f"SourceKFold{m.N_SPLITS}")
print("Robustness result:", robust_result)

print("\nSMOKE TEST DONE.")
print("Figures created:", list(fig_dir.glob("*.png")) if fig_dir.exists() else "NONE")
