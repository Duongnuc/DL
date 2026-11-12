"""99_run_all.py   (03_code)
PURPOSE    Runs steps 2-6 in order, each script in its own process; stops at the first error.
           Step 1 (01_check_b4tw_scc.py) checks the port on all cleaned curves and is not repeated here.
"""
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS = ['02_make_folds.py', '03_tune.py', '04_tuning_outputs.py', '05_random_search_check.py',
           '06_out_of_fold.py', '07_shap.py', '10_grouping_check.py', '08_figures.py', '09_fig01_process.py', '11_sensitivity.py', '12_residual_bias.py',
           '13_print_figures.py']
for s in SCRIPTS:
    t0 = time.time()
    print(f'=== {s}', flush=True)
    r = subprocess.run([sys.executable, '-X', 'utf8', s], cwd=HERE)
    print(f'=== {s}: exit {r.returncode}, {time.time() - t0:.0f} s', flush=True)
    if r.returncode:
        sys.exit(r.returncode)
print('ALL DONE')
