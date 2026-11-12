# Run the robustness analyses on a Mac

Three extra analyses for the V2 creep paper (approved 2026-11-13). Script: `03_code/14_robustness.py`. Its header
lists what each analysis does and the sources it follows.

## 1. Set up (once)

```bash
brew install libomp                # OpenMP, needed by xgboost on macOS
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run, from the `03_code` folder

```bash
cd 03_code
CREEP_LEVEL=curve python 14_robustness.py smoke      # seconds; must print "smoke ok"
CREEP_LEVEL=curve python 14_robustness.py mono       # about 2 min
CREEP_LEVEL=curve python 14_robustness.py random     # about 25 min on an 8-core PC
CREEP_LEVEL=curve python 14_robustness.py seeds      # 10 seeds, main level, about 3-4 h on an 8-core PC
CREEP_LEVEL=mix   python 14_robustness.py seeds      # 10 seeds, new-mix level, about 3-4 h
```

- All cores but one are used; set `--jobs N` to change this.
- A stopped run can be started again: finished jobs (`*_best.json`) are skipped.
- Seed 2 is the seed of the paper. `07_robustness/reproduction_check.txt` records how far seed 2 is from the
  paper predictions. A tiny difference on another CPU is expected; report it if it is larger than about 0.1.

## 3. Send the results back

```bash
cd ..
git add 04_results/level_curve/07_robustness 04_results/level_mix/07_robustness
git commit -m "Robustness results from Mac"
git push
```

Main outputs in `04_results/level_<level>/07_robustness/`: `summary.txt`, `seeds_metrics.csv`,
`random_split_metrics.csv`, `monotonic.csv`, `reproduction_check.txt`.

Library versions of the Windows run that produced the paper results: Python 3.11.9, scikit-learn 1.8.0,
xgboost 3.2.0, numpy 2.2.1, pandas 3.0.3, joblib 1.4.2, scipy 1.14.1.
