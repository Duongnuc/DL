# Paper V2 (simple, JOMC): plan approved 2026-11-10

Source of truth for the rewrite. Changes go here, not into a second copy.

## Goal
Rewrite the creep SCC paper in the frame of the advisor's shrinkage paper
(`E:\Downloads\Comparative study of machine learning methods on predicting shrinkage of SCC (revised).docx`):
five tree models (DT, RF, ET, GB, XGB), hyperparameters tuned one at a time, SHAP, comparison with B4TW-SCC.
Total creep and basic creep. Target journal: JOMC. The 17-page hybrid/Ensemble paper stays untouched as V1
(`manuscript\`, may go to another journal).

## Decisions (owner cards, 2026-11-10)
| Card | Decision |
|---|---|
| 1 | Frame of the shrinkage paper (sections, ~12 figures, ~5 tables) |
| 2 | Cleaned data: total 147 curves / 1,398 readings, basic 51 / 445, cement type R (`1_data\cleaned_curves.csv`, `cleaned_readings.csv` of the V1 code package) |
| 3a | Outer grouped 5-fold cross-validation: each curve predicted once by models that never saw it; groups = mixes (all curves of one mix on one side): basic 28, total 116. Tuning inside each outer training part by inner grouped 5-fold. Reason: single 80/20 split ranks models by chance (V1, 30 splits: basic GB test R² −0.08 to 0.96; best model per split ET 11, XGB 8, GB 8, DT 3) |
| 3b | Inputs. Basic (6): w/cm, cm, fcm28, T, t′, t − t′. Total (9): + V/S, RH, t0. Not used: E28 (measured on 24/51 and 95/147 curves only), Tcur (= T on 100 % / 99 % of curves), a/cm (r = −0.94 with cm); basic without t0 (= t′), RH, V/S |
| 4 | Run basic creep; owner decides keep or drop after seeing the numbers (stop point 1) |
| 5 | B4TW-SCC from the advisor's code (`E:\Downloads\B4TW-SCC total creep 4734.py`, `...basic creep 4734.py`) ported from SQL to CSV, checked curve by curve against V1 `empirical_models.b4tw_scc` |
| 6 | V1 kept as is; V2 replaces it for JOMC |

## Method details
- Output: compliance J (10⁻⁶/MPa), elastic part included, as in B4TW-SCC.
- Models: scikit-learn DT, RF, ET, GB; xgboost XGB; n_estimators 100; random_state 2.
- Tuning: as in the shrinkage paper, one hyperparameter at a time over a grid (max_depth 1–20 first, then
  max_leaf_nodes, min_samples_split, min_samples_leaf, learning_rate 0.01–0.3, subsample, colsample_bytree,
  min_child_weight), then a second pass; criterion = mean validation NRMSE of the inner grouped 5-fold.
- Reported hyperparameter table and max_depth figure: tuning on all curves by grouped 5-fold (the final models).
  Reported accuracy: pooled out-of-fold predictions of the outer 5-fold (every curve a test curve).
- Metrics: R², NRMSE = RMSE / IQR of measured J, RMSE, MAE; training scores = mean over outer folds.
- SHAP: TreeExplainer of each outer-fold model on its test part, mean |SHAP| normalized to 100.

## Steps (each ends on a check)
| # | Step | Done when |
|---|---|---|
| 1 | Port B4TW-SCC (advisor code) to CSV input | per-curve difference to V1 function < 0.1 %, or each difference explained |
| 2 | Build mix groups and outer folds | printed: curves, readings, mixes per fold; no mix on both sides (assert) |
| 3 | Sequential tuning | parameter table + max_depth curves (training vs validation NRMSE) per model and creep type |
| 4 | Out-of-fold predictions, ML and B4TW-SCC | table R², NRMSE, RMSE, MAE (training, test) → **stop point 1: owner decides on basic creep** |
| 5 | SHAP | importance figure per creep type |
| 6 | Figures in the shrinkage-paper forms: database, max_depth, SHAP, predicted vs measured (6 panels, ±40 % lines), residual vs load duration, representative test curves | each figure opened and inspected |
| 7 | Draft in the shrinkage-paper frame | every number traced to a result file → **stop point 2: owner reads the draft** |

## Likely committee questions (defence notes; the paper keeps one sentence + citation per choice)
| Question | Answer | Source |
|---|---|---|
| Why not a random 80/20 split of readings? | Readings of one curve (and curves of one mix) are not independent; a random split tests interpolation along known curves | Roberts et al. 2017, Ecography 40:913 (block CV for structured data); Daou 2022 (readings of one experiment kept in one set); advisor's shrinkage paper (curve-level split) |
| Why group by mix, not by curve? | Curves of one mix differ only in loading age (checked: 9 basic and 14 total multi-curve mixes, only t′ and stress ratio vary) | own data check, step 2 |
| Why 5-fold outer CV instead of one 80/20 split? | One split ranks models by chance with 51 basic curves (V1, 30 splits: GB test R² −0.08 to 0.96); outer CV tests every curve once | V1 results; Vabalas et al. 2019 PLOS ONE 14:e0224365 |
| Why tune inside each outer fold (nested)? | Tuning on the test curves biases the error downward (selection bias) | Cawley & Talbot 2010, JMLR 11:2079; Varma & Simon 2006 |
| Why one-at-a-time tuning, not grid/random/Bayesian search? | Advisor's protocol (shrinkage paper); gives the sensitivity curves (e.g. max_depth) that show under- and over-fitting; a second pass catches interactions. A random-search check on the same folds is kept in the results (not in the paper) | shrinkage paper §3; Bergstra & Bengio 2012 JMLR 13:281 for random search |
| Which hyperparameters matter? | Decision tree: minimum samples per leaf / split; random forest: number of features per split (mtry = max_features); boosting: learning rate. All are in the grids (max_features added for RF and ET, absent from the shrinkage paper) | Probst, Boulesteix & Bischl 2019, JMLR 20(53), §5.1.2 and Table 3 |
| Why not tune the number of trees of RF / ET? | More trees never increase the MSE-type error; set to a feasible number (100) | Probst & Boulesteix 2018, JMLR 18(181) |
| Why tune n_estimators for GB / XGB? | In boosting the number of trees and the learning rate trade off; both in the grid | Friedman 2001, Ann. Stat. 29:1189 |
| Why NRMSE normalised by the IQR? | Scale-free, robust to the few large compliances; same metric as the advisor's papers | Nguyen et al. 2025 JSCBM; shrinkage paper Eq. (11) |

## Progress
- Step 1 done 2026-11-10 (`03_code/b4tw_scc_advisor.py`, `01_check_b4tw_scc.py`, results `04_results/01_b4tw_check`):
  157 of 198 curves within 0.1 % of V1 (cause: 6/7 vs 0.85 in E(t)); 33 curves 0.1–1.3 % (V1 used the measured unit
  weight, advisor's code 2350); 8 curves up to 23 % (silica fume together with slag or fly ash: advisor's code
  applies no admixture factor, V1 applied the slag / fly ash factors; published Table 5 lists only the four single
  classes). Overall scores almost equal (basic R² 0.755 vs 0.753, total 0.483 vs 0.481). V2 uses the advisor's code.
- Step 2 done 2026-11-10 (`03_code/common.py`, `02_make_folds.py`, results `04_results/02_folds`): mix = same
  reference and mix proportions (28 basic, 116 total; same count when grouped by author or by proportions alone);
  inside a mix only the loading age (and stress ratio) varies. GroupKFold(5, shuffle, random_state 2): basic folds
  5–6 mixes, 6–16 curves, 12–29 % of readings; total 23–24 mixes, 27–31 curves, 18–23 %. Asserts passed: no mix or
  curve in two folds; no missing input.
- Step 3 done 2026-11-11 (`03_tune.py`, `04_tuning_outputs.py`, `05_random_search_check.py`; results
  `04_results/03_tuning`): 60 tuning runs (2 creep types × all + 5 outer training parts × 5 models). Grids as in
  `03_tune.py`; added to the shrinkage-paper set: max_features for RF/ET (Probst 2019), n_estimators for GB/XGB
  (Friedman 2001). Mean validation NRMSE, all curves, default → tuned: basic DT 77.2→72.6, RF 70.3→68.3,
  ET 66.3→64.2, GB 63.1→57.0, XGB 72.9→62.8; total DT 82.5→71.3, RF 63.7→60.3, ET 62.4→60.2, GB 60.9→59.4,
  XGB 60.4→60.3. Random search (60 draws, same folds) within −1.9 to +2.2 points of the sequential result.
  Selected values vary between outer training parts (e.g. basic DT max_depth 3–10): validation curves are flat
  beyond the selected depth, so the choice matters little.
- Step 4 done 2026-11-11 (`06_out_of_fold.py`; results `04_results/04_out_of_fold`). Pooled out-of-fold test:
  basic B4TW-SCC R² 0.75 / NRMSE 44.5 %, ML 0.15–0.46 / 66.2–82.7 % (ET best); total B4TW-SCC 0.48 / 67.9 %,
  ML 0.38–0.52 / 65.1–74.6 % (ET best). Mix bootstrap (2000 draws), test RMSE ML − B4TW-SCC: basic +6.8 to +12.0
  (all 95 % intervals above 0: B4TW-SCC better); total −0.95 to +2.3 (all intervals include 0: no difference).
  Defence check (same tuned parameters, no nesting): grouping by curve instead of by mix raises test R² by
  0.08–0.18 (basic ET 0.48→0.66, GB 0.60→0.68; total ET 0.55→0.64, GB 0.57→0.71); tuning on all curves then
  testing (not nested) gives basic GB 0.60 vs nested 0.45 (selection bias, Cawley & Talbot 2010).
  B4TW-SCC was calibrated on this database (advisor's JSTCE 2026), so its test curves are not new to it.
  → STOP POINT 1 passed 2026-11-11: owner keeps basic creep (finding: with 28 mixes ML does not replace
  B4TW-SCC for basic creep; for total creep ML matches it).
- Step 5 done 2026-11-11 (`07_shap.py`; results `04_results/05_shap`): exact TreeSHAP of each outer-fold model
  on its own test part (refit predictions = step 4, asserted; additivity error ≤ 5e-5). Mean rank over the 5
  models — basic: t′ 1.0, w/cm 2.0, t − t′ 3.4, fcm28 3.6, cm 5.2, T 5.8; total: t − t′ 1.4, fcm28 1.6, t′ 3.0,
  w/cm 4.2, cm 5.8, t0 6.4, V/S 7.3, T 7.5, RH 7.8. Direction check (Spearman, input vs its SHAP, ET and GB):
  w/cm +0.58…+0.87, fcm28 −0.79…−0.89, t′ −0.86…−0.91, t − t′ +0.90…+0.98, t0 −0.59…−0.90 (all as physics
  expects); RH −0.57 (ET) / −0.03 (GB); V/S +0.38/+0.47 (opposite to physics, small importance: data range,
  not a causal effect); cm mixed (shares credit with w/cm, r = −0.89; Aas et al. 2021, Artif. Intell. 298:103502).
  Paper: ranking only, "not causal", as in the shrinkage paper.

### Rerun 2026-11-12 (current numbers; steps 2–5 above are superseded, old results in `04_results/_superseded_with_fc_above_110_2026-11-12`)
Owner decisions: drop the 8 Persson curves with fcm28 124/141 MPa (`common.FC_RANGE` 17–110, the advisor's filter;
all other advisor filters already met); report option A (J is the main metric; a creep-part table, J minus the
first reading, explains where ML loses). Run: `99_run_all.py`, log `04_results/run_all_2026-11-12.log`, all exit 0.
- Data: basic 47 curves / 418 readings / 27 mixes; total 143 / 1,371 / 115. Mixes with ≥ 2 curves: basic 8
  (28 curves), total 13 (41 curves).
- Tuning: random search within −2.1 to +2.5 points of the sequential result.
- Test, pooled out-of-fold, R² (NRMSE): basic B4TW-SCC 0.74 (45.5 %), ML 0.33–0.52 (GB best, 62.2 %); total
  B4TW-SCC 0.48 (68.2 %), ML 0.34–0.54 (GB 0.54 / 64.1 %, ET 0.52). Bootstrap of mixes, RMSE ML − B4TW: basic
  all intervals above 0 (GB 0.11 to 9.36); total all include 0.
- Creep part R²: basic B4TW 0.35, GB 0.55, XGB 0.40, ET 0.33; total B4TW 0.45, ET 0.55, GB 0.51. All bootstrap
  intervals include 0 except DT (worse). So: ML loses on the level at loading, not on the growth with time.
- Grouping check (not nested): by curve vs by mix, total +0.00 to +0.20 (ensembles 0.70–0.74 vs 0.52–0.59);
  basic −0.04 to +0.25.
- SHAP mean rank: basic t′ 1.0, t − t′ 2.8, fcm28 3.0, w/cm 3.2, cm 5.2, T 5.8; total t − t′ 1.0, fcm28 2.0,
  t′ 3.4, w/cm 4.0, t0 4.8, cm 6.4, V/S 7.4, T 7.8, RH 8.2. Direction (all 5 models): w/cm +, fcm28 −, t′ −,
  t − t′ +, t0 −, RH − (as physics); V/S mostly + (opposite, small importance); cm and T mixed.
- Representative curves (ET RMSE percentiles 25/50/75): basic c_104_05, c_079_03, c_082_08; total c_094_01,
  c_072_01, c_082_17.
- Quick check, not adopted (owner asked how to raise R²): adding 1/E(t′) as input raised basic ET/GB R² by
  0.11–0.16 but lowered total by 0.12–0.13 (defaults, not nested); choosing per creep type would select on the
  test, so not used. Larger options for the advisor: add SL/RS cement curves; hybrid with B4TW-SCC (= V1).

### Two test levels (owner 2026-11-12, option 2; V1 may never be submitted, so V2 keeps its useful finding)
- Main level 'curve' (mixes already tested): outer and inner folds grouped by curve, exactly the advisor's request
  ("chạy lại các thuật toán theo đường cong") and the shrinkage paper's split. All figures, the parameter table,
  max_depth and SHAP come from this level. Results: `04_results/level_curve`.
- Extra level 'mix' (new mixes): the rerun above, results moved to `04_results/level_mix`. In the paper: one
  table + 2–3 sentences, no figures; reason = the advisor's own shrinkage-paper sentence citing Wang et al. 2026
  (mixture-level validation).
- Code: `common.LEVEL` from env `CREEP_LEVEL` (default 'curve'); `C.GROUP` is the split group; bootstrap
  intervals stay by mix at both levels (mix = independent unit).
- Quick check before the choice (defaults, mix folds): ML correcting the B4TW-SCC residual (V1's hybrid idea)
  gave basic 0.63–0.69 vs B4TW 0.74, total 0.45–0.55 vs 0.48 → not added.
- Curve level run 2026-11-12 (`04_results/level_curve/run_all.log`, all exit 0). Test R² (NRMSE):
  total B4TW-SCC 0.48 (68.2 %), DT 0.53, RF 0.70, ET 0.72, GB 0.77 (45.3 %), XGB 0.76; bootstrap RMSE ML − B4TW
  for RF/ET/GB/XGB all intervals below 0 (ML better, sure). Basic B4TW-SCC 0.74 (45.5 %), ML 0.34–0.61 (GB best);
  RF and DT surely worse, ET/GB/XGB intervals just include 0. Creep part: total ET 0.69, GB 0.67 vs B4TW 0.45
  (ML better, sure for ET/GB/XGB); basic GB/XGB 0.51 vs 0.35 (not sure). Random search within −0.4 to +4.5
  points. SHAP rank: basic t′ 1.0, fcm28 2.0, t − t′ 3.2, w/cm 3.8; total t − t′ 1.0, fcm28 2.2, t′ and w/cm 3.6.
  Representative curves: basic c_014_28, c_083_01, c_006_04; total c_043_04, c_006_02, c_011_01.
- Message of the paper: for curves of mixes already tested, ML (RF, ET, GB, XGB) predicts total creep more
  accurately than B4TW-SCC (R² 0.70–0.77 vs 0.48); for new mixes it only matches it (0.44–0.54 vs 0.48). For
  basic creep (27 mixes) B4TW-SCC stays at least as accurate at both levels.

## Folders (data, code, results apart)
`paper_v2_simple\01_paper`, `02_data` (copy of the cleaned CSVs + advisor's B4TW code as received),
`03_code`, `04_results`. V1 folders are read-only.

### Step 7 draft built 2026-11-12 (stop point 2: owner reads the draft)
- Owner: claim and title approved ("Ok cả 2"); Fig. 1 compact redraw, database bar figure removed (counts in one
  sentence); representative-curve panels titled by w/cm, fcm28, t′ (no curve codes).
- `01_paper/build_v2.py` → `creep_scc_v2_draft.docx` / `.pdf`: 18 pages, 5,687 words (Word count), 11 figures,
  6 tables, 8 equations, 28 references. Test metrics recomputed from the predictions at full precision (the CSVs keep
  3 decimals: total B4TW-SCC R² 0.4754 had printed as 0.47); every other number read from `04_results`.
- Checks: `01_paper/check_numbers.py` exit 0 (hand-typed database and screening counts); presend_check: 0 banned
  hits, citations 22/85 sentences overall, Introduction 9/16 (above one third, owner rule "every prior work cited");
  mean sentence 27.9 words, several > 35 words (language pass after content approval). Overlap with cited PDFs not run.
- New helper scripts: `03_code/11_sensitivity.py` (validation-NRMSE range per hyperparameter, first pass),
  `12_residual_bias.py` (mean residual by load duration).
- Wording of the claim in the text: "held-out curves (other curves of the same mix can be in the training data)";
  only 41 of 143 total curves belong to mixes with more than one curve.
- Stop point 2 passed 2026-11-12: owner "Duyệt nội dung" (content and the "held-out curves" wording approved).

### Language pass applied 2026-11-12 (owner: "duyệt cả 3")
- Checks (`_control/language_check_2026-11-12/`): LanguageTool en-GB, corpus wording (13.2 M words), 8-word overlap
  (`01_paper/overlap_check.py`, 1,279 files), Opus sentence copy-edit (164 sentences, 52 rows); proposals in
  `PROPOSALS_2026-11-12.md`.
- Applied in `01_paper/build_v2.py` (backup of the previous builder in the session scratchpad): title "for predicting";
  8 sentences reworded to remove overlap with the advisor's shrinkage paper (longest run 23 → 10 words, the rest are
  lists of algorithm names, table headers, one caption of the advisor's form); 52 copy-edit rows; 6 corpus wording
  changes; ageing, Optimisation; Figure 2 key without RH (not a basic creep input).
- After: 18 pages, 5,745 words, abstract 237 words; check_numbers exit 0; LanguageTool: only the known false alarm
  ("the later of"); pages 1, 2, 7, 13, 16 viewed.

### Figures and cold reads 2026-11-13 (owner: "Duyệt theo đề xuất")
- Figs 2-11 redrawn at print size by `03_code/13_print_figures.py` (variant B, Times 7-8 pt; added to 99_run_all).
- Cold reads (2 figure readers, 1 blind reviewer): `_control/language_check_2026-11-12/BLIND_READ_2026-11-13.md`.
- Applied: parity caption "predicted = 0.6 and 1/0.6 times measured" (advisor's lines kept); tuning captions of
  Figs 3, 5 and Tables 2, 3 say the all-curve tuning only reports the tuning; 8 text fixes (Fig 8(b) wording, stale
  SHAP key, Table 6 order, elastic reading clause, flexibility sentences cut, "likely"/"consistent with"/"appear to",
  "mixes already in the database"). Rebuild: 18 pages, 5,765 words, check_numbers exit 0.
- Owner asked for wider but simple analyses (e.g. more seeds), backed by papers/books: research running; plan to be
  approved before any run. One nested run of 03_tune.py takes ~25 min (both creep types).

### Robustness analyses 2026-11-13 (owner: "Chạy cả 3", run on the owner's Mac; results "Duyệt theo đề xuất")
- `03_code/14_robustness.py` run on the Mac (commit 82b3da8, branch robustness-2026-11-13 of the private repo
  Duongnuc/DL): 10 seeds at both levels, random split of readings, monotonicity check. Outputs in
  `04_results/level_<L>/07_robustness/`. Recomputed from the prediction files: 240/240 rows match.
- Seed 2 on the Mac picked other tuned values in 30/50 jobs (curve level); on Windows the same script reproduces
  the paper values (3 DT jobs checked), so the difference is the machine. Seed-2 test R² differ by 0.00-0.05;
  19/20 paper R² lie inside the 10-seed min-max. Owner: use the Mac run as is.
- Conclusions hold in every seed: total creep, 4 ensembles lower RMSE than B4TW-SCC 10/10; basic creep, B4TW-SCC
  better 10/10; new mixes, ensembles mean R² 0.47-0.50 vs 0.475.
- Paper: 3.1 sentence on seeds 1-10 [Bouckaert & Frank 2004; Bouthillier et al. 2021]; 4.3 two sentences (10 seeds,
  mean R² 0.68-0.74; random split R² 0.95-0.98 as in [nguyen_jscbm]), numbers read from the CSV with asserts.
  To stay at 18 pages: reference spacing 2 → 1 pt; last sentence of 4.4 (repeat of its first sentence) cut (approved by the owner 2026-11-13),
  Monotonicity result kept out of the paper (owner), kept for the response to reviewers:
  predicted J drops > 0.5 % on 45-90 % of total-creep test curves for the ensembles, 0 % for B4TW-SCC.
- Rebuild: 18 pages, 5,876 words, check_numbers exit 0; pages 5, 13, 15, 16, 18 viewed.

### Blind read 2, 2026-11-13 (owner: "Duyệt theo đề xuất")
- Fresh Opus reviewer on the 18-page PDF: major revision; big objections as in blind read 1. Verification table:
  `_control/language_check_2026-11-12/BLIND_READ_2026-11-13b.md`.
- Applied: Fig 8(b) and Fig 11 growth/start-level clauses cut (data: ensembles rise 11.7-17.2 vs measured 28.9;
  model spread grows 44 → 94); conclusion (4) "lose accuracy mainly at the start level" cut (4.2 keeps "appear");
  seed sentence "B4TW-SCC had a lower RMSE than every algorithm for basic creep" (assert over all five, 10 seeds);
  "differ only in the age at loading and, for total creep, the age at the start of drying"; "RH between 0.45 and
  0.65"; Tables 2-3 note "-: not tuned"; "random forest and extra trees average many randomised trees, and
  gradient boosting and XGBoost add trees by boosting". Seed 2 vs 10-seed mean: kept as is (owner).
- LanguageTool: no new grammar issue; overlap: longest run 10 words (name lists, table headers).
- Rebuild: 18 pages, 5,832 words, check_numbers exit 0; changed pages 3, 6, 8, 9, 10, 12, 13, 14, 16 viewed.

### Abstract and introduction rewrite 2026-11-13 (owner: "Duyệt theo đề xuất")
- Checked against the abstract move sequence (sciwrite), the JOMC template ("limitation of previous research works",
  "main results obtained") and the advisor's shrinkage abstract.
- Abstract (239 → 277 words): opens with the gap (adapted models still inaccurate; earlier R² 0.95 with readings
  of one curve in training and test); adds ten seeds and random split (ensembles 0.95-0.98, read from
  07_robustness). Intro: "models adapted to SCC have therefore been proposed ... One of them"; "For the creep of
  SCC, such studies are few."; closing sentence "The results show where tree-based algorithms can replace a
  calibrated creep model for SCC and where they cannot."
- No 8-word overlap with the advisor's shrinkage papers except the list of algorithm names; LanguageTool clean.
- Rebuild: 18 pages, 5,894 words, check_numbers exit 0; pages 1-3 viewed.

### Package for the advisor 2026-11-13 (owner: "làm lại báo cáo, folder code tương tự để gửi kèm")
- `E:\05deep L\02_Creep_SCC_ML\creep_SCC_JOMC_V2_2026-11-13\` built by `01_paper/build_package.py` (readmes in
  `01_paper/package_readmes/`): 00_READ_FIRST (docx, pdf), 01_paper_and_report (paper, report_v2 3 pages),
  02_data (data only), 03_code (scripts, advisor's original B4TW-SCC code, requirements, README), 04_results.
- Report numbers read from the result files; five questions for the advisor (novelty vs [6], B4TW-SCC calibration
  curves, a/cm and admixture inputs, publication split, public data and code) and the monotonicity note.
- Checks: `14_robustness.py smoke` run inside the package ("smoke ok", folds match); 04_results identical to the
  source; report and READ_FIRST pages viewed.
- Short version (owner: "bản rút gọn", chose quick code): `03_code/15_quick_check.py` refits from the saved
  outer-fold hyperparameters and prints Tables 4-6 next to the saved values; both levels in ~17 s, 24/24 rows match
  (in the source and inside the package). Added to the package, README and READ_FIRST.
