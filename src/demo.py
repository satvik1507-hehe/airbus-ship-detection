"""Live demonstration - classify satellite images with all three trained models.

  python -m src.demo                       # 8 random test images
  python -m src.demo --n 12 --seed 7       # a different random selection
  python -m src.demo --image my_photo.jpg  # any image file(s) of your own

Prints P(ship) from Random Forest, TL-CNN and SIM-CNN for every image and
saves (and opens) a picture grid with the SIM-CNN verdict on each image.
"""
import argparse
import json
import os

import joblib
import matplotlib
import numpy as np
import torch

from . import config as C
from .data import load_labels, read_image
from .features import combine, extract_one
from .models import DenseNetBackbone, SimCNN, TLHead
from .train_cnn import DEVICE, to_tensor


def load_thresholds():
    try:
        r = json.load(open(C.RESULTS_DIR / "cnn_results.json"))
        return {k: r[k]["test_at_best_threshold"]["threshold"] for k in ("sim", "tl")}
    except FileNotFoundError:
        return {"sim": 0.5, "tl": 0.5}


@torch.no_grad()
def predict(images):
    """images: uint8 [N,256,256,3] BGR -> dict of P(ship) arrays per model."""
    feats = [extract_one(img) for img in images]
    X = np.stack([combine(f, ("His", "Ha", "Hu")) for f in feats])
    rf = joblib.load(C.MODEL_DIR / "random_forest.joblib")
    out = {"Random Forest": rf.predict_proba(X)[:, 1]}

    x = to_tensor(images).to(DEVICE)
    tl = TLHead().to(DEVICE)
    tl.load_state_dict(torch.load(C.MODEL_DIR / "tl_cnn.pt", map_location=DEVICE))
    out["TL-CNN"] = torch.sigmoid(tl.eval()(DenseNetBackbone().to(DEVICE)(x))).cpu().numpy()

    sim = SimCNN().to(DEVICE)
    sim.load_state_dict(torch.load(C.MODEL_DIR / "sim_cnn.pt", map_location=DEVICE))
    out["SIM-CNN"] = torch.sigmoid(sim.eval()(x)).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="+", help="path(s) to your own image(s)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-show", action="store_true", help="only save the figure")
    args = ap.parse_args()

    if args.image:
        paths, truth = args.image, [None] * len(args.image)
    else:
        test = load_labels().query("split == 'test'")
        # half ships / half no-ships so the demo shows both classes
        pick = test.groupby("has_ship").sample(n=args.n // 2, random_state=args.seed)
        pick = pick.sample(frac=1, random_state=args.seed)
        paths, truth = [C.IMAGE_DIR / i for i in pick.ImageId], pick.has_ship.tolist()

    images = np.stack([read_image(p) for p in paths])
    proba = predict(images)
    thr = load_thresholds()

    print(f"\n{'image':16s} {'truth':8s} " + " ".join(f"{m:>14s}" for m in proba))
    for i, p in enumerate(paths):
        t = "-" if truth[i] is None else C.CLASS_NAMES[truth[i]]
        print(f"{os.path.basename(str(p)):16s} {t:8s} " +
              " ".join(f"{proba[m][i]:14.3f}" for m in proba))
    print(f"\nValues are P(ship). Decision thresholds: Random Forest 0.5, "
          f"TL-CNN {thr['tl']}, SIM-CNN {thr['sim']}")

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = min(4, len(paths))
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.7 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, ax in enumerate(axes.ravel()[:len(paths)]):
        ax.imshow(images[i][..., ::-1])
        p = proba["SIM-CNN"][i]
        pred = int(p >= thr["sim"])
        ok = truth[i] is None or pred == truth[i]
        title = f"SIM-CNN: {C.CLASS_NAMES[pred]} (p={p:.2f})"
        if truth[i] is not None:
            title += f"\ntruth: {C.CLASS_NAMES[truth[i]]}"
        ax.set_title(title, color="green" if ok else "red", fontsize=10)
    fig.tight_layout()
    out = C.FIG_DIR / "demo.png"
    fig.savefig(out, dpi=130)
    print("Saved", out)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
