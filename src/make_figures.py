"""Step 6 - descriptive figures used in the write-up and slides.

  sample_images.png      example "Ship" / "No ship" images
  feature_examples.png   original vs greyscale vs HSV (what the descriptors see)
  class_balance.png      ship rate per split
  model_comparison.png   test accuracy of every model vs the majority baseline

Run:  python -m src.make_figures
"""
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C
from .data import load_labels, read_image


def main():
    df = load_labels()

    # --- sample images -------------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for row, label in enumerate((1, 0)):
        ids = df[df.has_ship == label].ImageId.sample(5, random_state=C.SEED)
        for ax, i in zip(axes[row], ids):
            ax.imshow(read_image(C.IMAGE_DIR / i)[..., ::-1])
            ax.set_title(C.CLASS_NAMES[label], fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "sample_images.png", dpi=140)

    # --- what the feature extractors look at ---------------------------------
    img = read_image(C.IMAGE_DIR / df[df.has_ship == 1].ImageId.sample(1, random_state=3).iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    axes[0].imshow(img[..., ::-1]); axes[0].set_title("Original (RGB)")
    axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap="gray")
    axes[1].set_title("Greyscale -> Haralick, Hu")
    axes[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)); axes[2].set_title("HSV -> colour histogram")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "feature_examples.png", dpi=140)

    # --- class balance ---------------------------------------------------------
    counts = df.groupby(["split", "has_ship"]).size().unstack().loc[["train", "dev", "test"]]
    ax = counts.rename(columns={0: "No ship", 1: "Ship"}).plot.bar(
        stacked=True, figsize=(5, 3.6), color=["#9ecae1", "#e6550d"], rot=0)
    ax.set(ylabel="Images", xlabel="", title="Class balance per split")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "class_balance.png", dpi=140)

    # --- final comparison ------------------------------------------------------
    trad = json.load(open(C.RESULTS_DIR / "traditional_results.json"))
    cnn = json.load(open(C.RESULTS_DIR / "cnn_results.json"))
    acc = {a: trad["detail"][f"{a}|w/ all"]["accuracy"] for a in ("LDA", "KNN", "NB", "RF", "SVM")}
    acc["TL-CNN"] = cnn["tl"]["test_at_best_threshold"]["accuracy"]
    acc["SIM-CNN"] = cnn["sim"]["test_at_best_threshold"]["accuracy"]
    s = pd.Series(acc)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(s.index, s.values, color=["#6baed6"] * 5 + ["#fd8d3c", "#e6550d"])
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.axhline(trad["majority_baseline"], ls="--", c="gray", label="majority-class baseline")
    ax.set(ylim=(0.4, 1.0), ylabel="Test accuracy", title="All models on the same test set")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "model_comparison.png", dpi=140)
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()
