import sys
import os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
m = importlib.import_module("07_gru")

package_root = Path(__file__).resolve().parent.parent
xlsx_path = package_root / "data_raw" / "Database SCC.xlsx"
cache_dir = package_root / "_cache" / "_smoke_gru"

print("=" * 72)
print(f"[{m.MODEL_NAME}] SMOKE STEP 1: Prepare sequence-shaped data")
print("=" * 72)
data_dir = cache_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)
data_outputs = m.build_datasets(xlsx_path=xlsx_path, output_dir=data_dir, cfg=m.DataPipelineConfig())
data_path = data_outputs["seq"]
print(f"Finished preparing data: {data_path}")

print("\n" + "=" * 72)
print(f"[{m.MODEL_NAME}] SMOKE STEP 2: Holdout (tiny)")
print("=" * 72)
holdout_cfg = m.SequenceBenchmarkConfig(
    data_path=data_path, output_dir=cache_dir / "holdout", model_name=m.MODEL_NAME,
    seeds=[42], n_trials=1, max_epochs=2, patience=1, batch_size=16,
)
holdout_outputs = m.run_benchmark(holdout_cfg)
m.print_model_console_summary(m.MODEL_NAME, holdout_outputs)

print("\n" + "=" * 72)
print(f"[{m.MODEL_NAME}] SMOKE STEP 3: Group K-Fold (tiny)")
print("=" * 72)
kfold_cfg = m.SequenceBenchmarkConfig(
    data_path=data_path, output_dir=cache_dir / "kfold", model_name=m.MODEL_NAME,
    seeds=[42], n_splits=2, n_trials=1, max_epochs=2, patience=1, batch_size=16,
)
kfold_outputs = m.run_sequence_kfold(kfold_cfg, m.build_model, m.suggest_params)
m.print_kfold_console_summary(m.MODEL_NAME, kfold_outputs)

fig_dir = cache_dir / "output_figures"
m.plot_kfold_prediction_figures(m.MODEL_NAME, "gru", kfold_outputs, fig_dir)

print("\nSMOKE TEST DONE.")
print("Figures created:", list(fig_dir.glob("*.png")) if fig_dir.exists() else "NONE")
