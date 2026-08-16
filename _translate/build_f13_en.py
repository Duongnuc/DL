# -*- coding: utf-8 -*-
"""Build f13_new_en.txt from f13_new.txt using master_dict.tsv + manual translations
for lines not already covered by the shared dictionary. Manual translations are
keyed by 1-based line INDEX into f13_new.txt (not by VN text) to avoid any
transcription mismatch on Vietnamese/Unicode source text.
"""

vn_path = "f13_new.txt"
dict_path = "master_dict.tsv"
out_path = "f13_new_en.txt"

lines_vn = open(vn_path, encoding="utf-8").readlines()

d = {}
with open(dict_path, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if "\t" not in line:
            continue
        vn, en = line.split("\t", 1)
        d[vn] = en

# manual[idx] = English text (no trailing newline) for candidate line #idx (1-based)
manual = {}

# --- Header docstring (1-14) ---
manual[1] = '"""13 - Explain the ResMLP model with SHAP + PDP (reviewer comment #4) and learning'
manual[2] = 'curves for ResMLP/ANNBasic/LSTM (reviewer comment #3).'
manual[3] = 'This file MERGES TOGETHER (without omission) content from buoc_15_shap_pdp.py and'
manual[4] = 'buoc_16_learning_curves.py, plus the shared pipeline from common_tabular.py'
manual[5] = 'and buoc_01_code_loc_data.py.'
manual[6] = 'This file LIGHTLY RETRAINS on 1 group-stratified holdout (seed 42), using'
manual[7] = 'REPRESENTATIVE hyperparameters saved in _cache/<model>/kfold/best_params.json of'
manual[8] = 'files 01 (ANNBasic), 03 (ResMLP), 08 (LSTM) - if those files have not been run yet, uses'
manual[9] = 'reasonable default parameters (will not crash).'
manual[10] = 'Requires 2 extra packages beyond the standard requirements: shap, matplotlib.'
manual[11] = 'Figures are written to the output_figures/ directory (a VISIBLE directory, not hidden, since this is'
manual[12] = 'a deliverable to submit for the paper/thesis).'
manual[13] = 'Run:  python 13_resmlp_explainability.py'
manual[14] = '# PART 1 - SHARED PIPELINE FOR TABULAR MODELS (merged from common_tabular.py)'

# --- Part 2 header (342) ---
manual[342] = '# PART 2 - DATA PREPARATION (merged from buoc_01_code_loc_data.py)'

# --- Part 3 header (481-482) ---
manual[481] = '# PART 3 - Preprocessing specific to SEQUENCE (Dataset/DataLoader), a supplementary part'
manual[482] = '# that DOES NOT CLASH IN NAME with TAB_ENGINE, merged from common_sequence_group.py'

# --- Part 4 header (559-561) ---
manual[559] = '# PART 4 - Model architecture: ResMLP, ANNBasic, LSTM (only class + build_*;'
manual[560] = '# search_space is NOT reused since this file does not rerun Optuna, only lightly'
manual[561] = '# retrains using REPRESENTATIVE hyperparameters cached from k-fold of files 01/03/08)'

# --- Pure box-drawing / math-symbol lines: identical VN == EN (false positives) ---
for idx in (568, 570, 572, 582, 585, 586, 595, 645, 647, 650, 651, 653):
    manual[idx] = lines_vn[idx - 1].rstrip("\n")

# --- Part 5 header + feature labels (900-921) ---
manual[900] = '# PART 5 - SHAP + PDP + 2D-PDP for ResMLP (merged from buoc_15_shap_pdp.py,'
manual[901] = '# reviewer comment #4: feature importance and interaction between variables)'
manual[902] = '# Display names for the main features (nicer-looking on the figures).'
manual[903] = '    "w_cm": "N/CKD ratio (w/cm)",'
manual[904] = '    "cm": "Binder content (kg/m³)",'
manual[905] = '    "fc28": "28-day strength fc28 (MPa)",'
manual[906] = '    "V_S": "Volume/Surface ratio V/S",'
manual[907] = '    "RH_test": "Ambient humidity RH (%)",'
manual[908] = '    "T": "Ambient temperature T (°C)",'
manual[909] = '    "t0": "Age at start of measurement t0 (days)",'
manual[910] = '    "Tcur": "Curing temperature Tcur (°C)",'
manual[911] = '    "FlyAsh": "Fly ash ratio (%)",'
manual[912] = '    "Slag": "Slag ratio (%)",'
manual[913] = '    "water": "Water content (kg/m³)",'
manual[914] = '    "Fine_agg": "Fine aggregate (kg/m³)",'
manual[915] = '    "Coarse_agg": "Coarse aggregate (kg/m³)",'
manual[916] = '    "sandrate": "Sand ratio (%)",'
manual[917] = '    "age_days": "Sample age (days)",'
manual[918] = '    "age_log1p": "log(1+sample age)",'
manual[919] = '    "dt_log1p": "log(1+measurement time)",'
manual[920] = '    "OPC_pct": "OPC cement ratio (%)",'
manual[921] = '    "Super_kg": "Superplasticizer (kg/m³)",'

manual[922] = '# Preprocessing — KEEP the scaler to invert the standardization for the PDP axis (original units)'
manual[923] = '    """Mirror prepare_arrays_dl but also return the scaler + list of columns.'
manual[924] = '    Need scaler.mean_/scale_ to plot the PDP in ORIGINAL UNITS (invert StandardScaler) for numeric'
manual[925] = '    variables. The logic must EXACTLY MATCH common_tabular.prepare_arrays_dl so the model'
manual[926] = '    receives the same input distribution as during the benchmark.'
manual[927] = '# Manual Partial Dependence (original units μstrain)'
manual[928] = '    """Predict on the original μstrain scale (clip log_cap as during the benchmark, then expm1)."""'
manual[929] = '    """1D PDP for feature feat_idx.'
manual[930] = '    Sweep feat_idx across the [p5, p95] range (standardized scale); for each value: assign it to'
manual[931] = '    ALL background samples, predict, and average in μstrain. The x-axis is converted back to'
manual[932] = '    the original units if it is a numeric variable (idx < n_numeric).'
manual[933] = '    """2D-PDP for the pair (idx_a, idx_b): heatmap of average predictions (μstrain)."""'

# Pure-code false positives (934-937, 943-945)
for idx in (934, 935, 936, 937, 943, 944, 945):
    manual[idx] = lines_vn[idx - 1].rstrip("\n")

manual[938] = '# Figure'
manual[939] = '    plt.title("SHAP beeswarm — feature contribution to the shrinkage prediction (ResMLP)",'
manual[940] = '    plt.xlabel("SHAP value (effect on log shrinkage)")'
manual[941] = '    plt.xlabel("Mean |SHAP| importance")'
manual[942] = '    plt.title("Feature importance ranking (ResMLP, group holdout)", fontsize=13)'
manual[946] = '        ax.set_ylabel("Predicted shrinkage (μstrain)")'
manual[947] = '    fig.suptitle("1D PDP — partial effect of the main features (ResMLP)",'
manual[948] = lines_vn[947].rstrip("\n")  # pure code f-string, no VN words
manual[949] = '        fig.colorbar(im, ax=ax, label="Predicted shrinkage (μstrain)")'
manual[950] = '    fig.suptitle("2D-PDP — interaction between the feature pair (ResMLP)", fontsize=14, y=1.02)'

# --- Part 6 (951-965) ---
manual[951] = '# PART 6 - Learning curves over epochs for ResMLP/ANNBasic/LSTM (merged from'
manual[952] = '# buoc_16_learning_curves.py, reviewer comment #3: evidence of convergence & overfitting)'
manual[953] = '# TABULAR trainer — records train/val MSE history (log-space) every epoch'
manual[954] = '    # merge the last batch if it has only 1 sample (BatchNorm needs >=2)'
for idx in (955, 956, 957, 958, 959):
    manual[idx] = lines_vn[idx - 1].rstrip("\n")
manual[960] = '# SEQUENCE trainer — records train/val masked-MSE history (log-space) every epoch'
for idx in (961, 962):
    manual[idx] = lines_vn[idx - 1].rstrip("\n")
manual[963] = lines_vn[962].rstrip("\n")  # pure code (ax.plot ... label="Validation MSE (log-space)")
manual[964] = '                   label=f"Best val epoch = {r[\'best\']}")'
manual[965] = '    fig.suptitle("Learning curves — convergence & evidence of overfitting (group holdout, seed 42)",'

# --- Part 7 (966-993) ---
manual[966] = '# PART 7 - Orchestration: lightly retrain on 1 group-stratified holdout (seed 42),'
manual[967] = '# using REPRESENTATIVE hyperparameters read from _cache/<model>/kfold/best_params.json of'
manual[968] = '# files 01 (ANNBasic), 03 (ResMLP), 08 (LSTM); if those files have not been run yet'
manual[969] = '# then use reasonable default parameters (safe fallback).'
manual[970] = '    """Read best_params.json from the cached k-fold of files 01/03/08, take the'
manual[971] = '    representative fold config. If k-fold for that model has not been run, use default parameters."""'
manual[972] = '        print(f"  [warning] {p} not found, using default parameters for {key}.")'
manual[973] = '    print("PART A: SHAP + PDP + 2D-PDP for ResMLP (reviewer comment #4)")'
manual[974] = '    print(f"Number of features: {len(feature_names)} ({n_numeric} numeric + "'
manual[975] = '    print(f"Representative hyperparameters ({FOLD_RESMLP}): {params}")'
manual[976] = '    print("\\nTop 12 most important features (mean |SHAP|):")'
manual[977] = '    print("Saved fig06_shap_beeswarm.png, fig06b_shap_bar.png")'
manual[978] = '    print("Saved fig07_pdp_main.png for:",'
manual[979] = '        print("Saved fig08_pdp2d.png for the pair:",'
manual[980] = '    print("PART B: Learning curves - convergence & evidence of overfitting (reviewer comment #3)")'
manual[981] = lines_vn[980].rstrip("\n")  # pure code
manual[982] = lines_vn[981].rstrip("\n")  # pure code (print f-string, no VN words)
manual[983] = '    results.append({"name": "ResMLP (tabular)", "note": "BatchNorm + skip -> stable",'
manual[984] = lines_vn[983].rstrip("\n")  # pure code
manual[985] = lines_vn[984].rstrip("\n")  # pure code
manual[986] = '    results.append({"name": "ANNBasic (tabular)", "note": "Plain MLP, no BatchNorm/skip -> prone to overfitting",'
manual[987] = lines_vn[986].rstrip("\n")  # pure code
manual[988] = lines_vn[987].rstrip("\n")  # pure code
manual[989] = '    results.append({"name": "LSTM (sequence)", "note": "Gated RNN -> best for sequences",'
manual[990] = '    print(f"\\nSaved: {out}")'
manual[991] = '    print("13_RESMLP_EXPLAINABILITY: SHAP/PDP (buoc 15) + Learning curves (buoc 16)")'
manual[992] = '    print(f"Data prepared: {len(tab_df)} points (tabular), {len(seq_df)} points (sequence)")'
manual[993] = '    print(f"File 13 complete. All figures in: {fig_dir}")'

out_lines = []
missing = []
for i, raw in enumerate(lines_vn):
    idx = i + 1
    has_nl = raw.endswith("\n")
    key = raw[:-1] if has_nl else raw
    if idx in manual:
        en = manual[idx]
    elif key in d:
        en = d[key]
    else:
        missing.append((idx, key))
        en = key  # placeholder
    out_lines.append(en + ("\n" if has_nl else ""))

print("total lines:", len(lines_vn))
print("manual entries:", len(manual))
print("missing:", len(missing))
for i, k in missing[:80]:
    print(i, repr(k))

with open(out_path, "w", encoding="utf-8", newline="") as f:
    f.writelines(out_lines)
print("wrote", out_path, "lines:", len(out_lines))
