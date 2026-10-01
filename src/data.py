"""Loading helpers shared by the traditional-ML and CNN pipelines."""
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as C

SPLITS = ("train", "dev", "test")


def load_labels():
    if not C.LABELS_CSV.exists():
        raise SystemExit("data/labels.csv not found - run `python -m src.download_data` first.")
    return pd.read_csv(C.LABELS_CSV)


def read_image(path):
    """Read an image as a 256x256x3 uint8 BGR array (OpenCV channel order)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[:2] != (C.IMG_SIZE, C.IMG_SIZE):
        img = cv2.resize(img, (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


def load_split(split):
    """Return (X, y, ids): X is uint8 [N,256,256,3] BGR, y is 0/1. Cached as .npy."""
    df = load_labels()
    df = df[df.split == split]
    y = df.has_ship.to_numpy(dtype=np.int64)
    cache = C.CACHE_DIR / f"pixels_{split}.npy"
    if cache.exists():
        X = np.load(cache)
        if len(X) == len(df):
            return X, y, df.ImageId.tolist()
    X = np.stack([read_image(C.IMAGE_DIR / i) for i in tqdm(df.ImageId, desc=f"load {split}")])
    np.save(cache, X)
    return X, y, df.ImageId.tolist()
