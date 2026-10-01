"""Step 3 - traditional machine-learning classifiers (Sections 4.1 and 5.1).

Experiment A (Table 1): every algorithm x every feature combination.
    Fit on train, report accuracy on test.
Experiment B (Figure 5): 10-fold cross-validation with all features,
    summarised as a box plot.

Run:  python -m src.train_traditional
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C
from .features import combine, extract_split

# column name -> descriptors used (same eight columns as Table 1 of the report)
FEATURE_SETS = {
    "His": ("His",), "Ha": ("Ha",), "Hu": ("Hu",),
    "His+Ha": ("His", "Ha"), "Ha+Hu": ("Ha", "Hu"), "Hu+His": ("Hu", "His"),
    "w/ all": ("His", "Ha", "Hu"), "w/o (raw pixels)": ("raw",),
}
ALL = "w/ all"


def make_models():
    """The five classifiers of the report. Features are standardised first
    (zero mean / unit variance, statistics learned on the training data only)."""
    return {
        "LDA": LinearDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(n_neighbors=5),                 # Euclidean distance
        "NB": GaussianNB(),
        "RF": RandomForestClassifier(n_estimators=70, random_state=C.SEED, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=C.SEED),
    }


def pipeline(model):
    return make_pipeline(StandardScaler(), model)


def main():
    ftrain, ytrain = extract_split("train")
    fdev, ydev = extract_split("dev")
    ftest, ytest = extract_split("test")
    baseline = max(ytest.mean(), 1 - ytest.mean())     # always predict the majority class
    print(f"train {len(ytrain)}  dev {len(ydev)}  test {len(ytest)}  "
          f"majority-class baseline accuracy = {baseline:.3f}")

    # ---------------- Experiment A: algorithm x feature set -----------------
    acc = pd.DataFrame(index=make_models().keys(), columns=FEATURE_SETS.keys(), dtype=float)
    detail = {}
    for col, names in FEATURE_SETS.items():
        Xtr, Xte = combine(ftrain, names), combine(ftest, names)
        for alg, model in make_models().items():
            t0 = time.time()
            clf = pipeline(model).fit(Xtr, ytrain)
            pred = clf.predict(Xte)
            acc.loc[alg, col] = accuracy_score(ytest, pred)
            detail[f"{alg}|{col}"] = {
                "accuracy": accuracy_score(ytest, pred),
                "precision": precision_score(ytest, pred, zero_division=0),
                "recall": recall_score(ytest, pred, zero_division=0),
                "f1": f1_score(ytest, pred, zero_division=0),
            }
            print(f"{alg:4s} {col:17s} acc={acc.loc[alg, col]:.3f}  ({time.time() - t0:.1f}s)")
            if col == ALL and alg == "RF":
                joblib.dump(clf, C.MODEL_DIR / "random_forest.joblib")   # used by the demo

    acc.round(4).to_csv(C.RESULTS_DIR / "table1_test_accuracy.csv")
    print("\nTest accuracy (rows = algorithm, columns = features)\n", acc.round(3).to_string())

    # ---------------- Experiment B: 10-fold cross-validation ----------------
    X = np.vstack([combine(ftrain, FEATURE_SETS[ALL]), combine(fdev, FEATURE_SETS[ALL])])
    y = np.concatenate([ytrain, ydev])
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=C.SEED)
    cv = {alg: cross_val_score(pipeline(m), X, y, cv=folds, scoring="accuracy", n_jobs=-1)
          for alg, m in make_models().items()}
    for alg, s in cv.items():
        print(f"10-fold CV {alg:4s} mean={s.mean():.3f} std={s.std():.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(list(cv.values()), tick_labels=list(cv.keys()))
    ax.axhline(max(y.mean(), 1 - y.mean()), ls="--", c="gray", lw=1, label="majority-class baseline")
    ax.set(ylabel="Accuracy", title="10-fold cross-validation (all features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "cv_boxplot.png", dpi=160)

    json.dump({
        "n_train": int(len(ytrain)), "n_dev": int(len(ydev)), "n_test": int(len(ytest)),
        "ship_rate_test": float(ytest.mean()), "majority_baseline": float(baseline),
        "detail": detail,
        "cv": {a: {"mean": float(s.mean()), "std": float(s.std()), "scores": s.tolist()}
               for a, s in cv.items()},
    }, open(C.RESULTS_DIR / "traditional_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
