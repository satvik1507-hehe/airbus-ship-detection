"""Step 5 - evaluate both CNNs (Section 5.2).

  * training curves (loss / accuracy per epoch)
  * threshold scanning on the dev set (0.1 ... 0.9) - because the classes are
    imbalanced, 0.5 is not necessarily the best cut-off
  * test accuracy, precision, recall, specificity at the chosen threshold
  * normalised confusion matrices

Run:  python -m src.evaluate_cnn
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from . import config as C
from .data import load_split
from .models import SimCNN, TLHead
from .train_cnn import DEVICE, densenet_features, to_tensor

NAMES = {"tl": "TL-CNN", "sim": "SIM-CNN"}


@torch.no_grad()
def predict_proba(kind, X, batch=128):
    """P(ship) for every row of X (pixels for 'sim', DenseNet features for 'tl')."""
    model = (SimCNN() if kind == "sim" else TLHead(X.shape[1])).to(DEVICE)
    model.load_state_dict(torch.load(C.MODEL_DIR / f"{kind}_cnn.pt", map_location=DEVICE))
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = to_tensor(X[i:i + batch]) if kind == "sim" else torch.from_numpy(X[i:i + batch])
        out.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else 0.0
    return {"threshold": thr, "accuracy": div(tp + tn, len(y)), "precision": div(tp, tp + fp),
            "recall": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def main():
    results, scans = {}, {}
    for kind in ("tl", "sim"):
        if kind == "sim":
            Xdv, ydv, _ = load_split("dev")
            Xte, yte, _ = load_split("test")
        else:
            Xdv, ydv = densenet_features("dev")
            Xte, yte = densenet_features("test")
        pdv, pte = predict_proba(kind, Xdv), predict_proba(kind, Xte)
        np.save(C.RESULTS_DIR / f"{kind}_test_proba.npy", pte)

        scans[kind] = [float(((pdv >= t).astype(int) == ydv).mean()) for t in C.THRESHOLDS]
        best_thr = C.THRESHOLDS[int(np.argmax(scans[kind]))]      # chosen on DEV, never on test
        results[kind] = {
            "dev_threshold_scan": dict(zip(map(str, C.THRESHOLDS), scans[kind])),
            "test_at_best_threshold": metrics(yte, pte, best_thr),
            "test_at_0.5": metrics(yte, pte, 0.5),
            "test_roc_auc": float(roc_auc_score(yte, pte)),
        }
        m = results[kind]["test_at_best_threshold"]
        print(f"{NAMES[kind]}: threshold {best_thr}  test acc {m['accuracy']:.4f}  "
              f"precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
              f"specificity {m['specificity']:.3f}  AUC {results[kind]['test_roc_auc']:.3f}")
    results["majority_baseline"] = float(max(yte.mean(), 1 - yte.mean()))
    json.dump(results, open(C.RESULTS_DIR / "cnn_results.json", "w"), indent=2)

    # --- Figure: training curves -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for kind, color in (("tl", "tab:orange"), ("sim", "tab:blue")):
        h = json.load(open(C.RESULTS_DIR / f"{kind}_history.json"))
        ep = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], c=color, label=f"{NAMES[kind]} train")
        axes[0].plot(ep, h["dev_loss"], c=color, ls="--", label=f"{NAMES[kind]} dev")
        axes[1].plot(ep, h["train_acc"], c=color, label=f"{NAMES[kind]} train")
        axes[1].plot(ep, h["dev_acc"], c=color, ls="--", label=f"{NAMES[kind]} dev")
    axes[0].set(xlabel="Epoch", ylabel="Binary cross-entropy loss", title="Loss")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "cnn_training_curves.png", dpi=160)

    # --- Figure: threshold scan ---------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for kind, color in (("tl", "tab:orange"), ("sim", "tab:blue")):
        ax.plot(C.THRESHOLDS, scans[kind], "o-", c=color, label=NAMES[kind])
    ax.set(xlabel="Decision threshold", ylabel="Dev accuracy", title="Threshold scanning")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "threshold_scan.png", dpi=160)

    # --- Figure: normalised confusion matrices --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for ax, kind in zip(axes, ("tl", "sim")):
        m = results[kind]["test_at_best_threshold"]
        cm = np.array(m["confusion"], dtype=float)
        cm /= cm.sum(axis=1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > .5 else "black", fontsize=13)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES,
               xlabel="Predicted", ylabel="True",
               title=f"{NAMES[kind]} (threshold {m['threshold']})")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "confusion_matrices.png", dpi=160)


if __name__ == "__main__":
    main()
