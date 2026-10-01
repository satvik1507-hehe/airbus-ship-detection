"""Run the whole experiment end to end (after the data has been downloaded).

  python run_all.py
"""
import subprocess
import sys

STEPS = [
    ["src.features"],                       # hand-engineered features
    ["src.train_traditional"],              # LDA, KNN, NB, RF, SVM + 10-fold CV
    ["src.train_cnn", "--model", "tl"],     # transfer-learning CNN
    ["src.train_cnn", "--model", "sim"],    # simple CNN from scratch
    ["src.evaluate_cnn"],                   # thresholds, confusion matrices, plots
    ["src.make_figures"],                   # dataset / feature / summary figures
]

for step in STEPS:
    print("\n" + "=" * 70 + f"\n>>> python -m {' '.join(step)}\n" + "=" * 70, flush=True)
    subprocess.run([sys.executable, "-m", *step], check=True)
