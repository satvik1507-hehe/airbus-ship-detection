"""Step 2 - hand-engineered global image descriptors (Section 3.2.1 of the report).

  His : colour histogram in HSV space, 8x8x8 bins -> 512 numbers (colour)
  Ha  : Haralick texture from the grey-level co-occurrence matrix -> 13 numbers (texture)
  Hu  : Hu moments of the grey image -> 7 numbers (shape)
  raw : the image itself, down-scaled to 32x32x3 and flattened -> 3072 numbers
        (this is the "w/o feature engineering" column)

Run:  python -m src.features
"""
import cv2
import mahotas
import numpy as np
from joblib import Parallel, delayed

from . import config as C
from .data import SPLITS, load_split

FEATURE_NAMES = ("His", "Ha", "Hu")
BINS = 8


def color_histogram(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [BINS] * 3, [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def haralick(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 13 Haralick statistics, averaged over the 4 GLCM directions
    return mahotas.features.haralick(gray).mean(axis=0)


def hu_moments(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(gray)).flatten()


def raw_pixels(img_bgr):
    small = cv2.resize(img_bgr, (C.RAW_SIZE, C.RAW_SIZE), interpolation=cv2.INTER_AREA)
    return small.flatten().astype(np.float32) / 255.0


EXTRACTORS = {"His": color_histogram, "Ha": haralick, "Hu": hu_moments, "raw": raw_pixels}


def extract_one(img_bgr):
    return {name: fn(img_bgr) for name, fn in EXTRACTORS.items()}


def extract_split(split):
    """Return dict name -> [N, d] array for one split (cached on disk)."""
    cache = C.CACHE_DIR / f"features_{split}.npz"
    X, y, _ = load_split(split)
    if cache.exists():
        feats = dict(np.load(cache))
        if len(feats["His"]) == len(X):
            return feats, y
    rows = Parallel(n_jobs=-1, verbose=1)(delayed(extract_one)(img) for img in X)
    feats = {name: np.stack([r[name] for r in rows]).astype(np.float32) for name in EXTRACTORS}
    np.savez_compressed(cache, **feats)
    return feats, y


def combine(feats, names):
    """Concatenate the chosen descriptors column-wise, e.g. ('His', 'Ha')."""
    return np.hstack([feats[n] for n in names])


if __name__ == "__main__":
    for s in SPLITS:
        f, y = extract_split(s)
        print(s, {k: v.shape for k, v in f.items()}, "ship rate %.3f" % y.mean())
