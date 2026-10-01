"""Step 1 - build our working subset of the Kaggle Airbus Ship Detection data.

The full competition data is 192k images / 31 GB, far more than a laptop needs
(the Stanford report also used only a ~15k subset). We stay under 1 GB:

  1. download train_ship_segmentations_v2.csv - the OFFICIAL labels
     (one row per ship, run-length-encoded mask; an empty mask = no ship),
  2. download a ready-made 494 MB subset of the competition's train_v2 images
     (Kaggle dataset `aravjain007/aerial-ship-detection-dataset`) as one zip,
  3. optionally add N images sampled uniformly at random from the competition
     (`--n-random N`; Kaggle rate-limits per-file downloads to a few hundred),
  4. resize everything to 256x256 and store it in data/images/,
  5. label every image from the official CSV and write data/labels.csv with a
     stratified 7:3:5 train/dev/test split (same proportions as the report).

Needs a Kaggle API token (~/.kaggle/access_token or kaggle.json) and the
competition rules accepted on kaggle.com.

  python -m src.download_data                 # steps 1-5
  python -m src.download_data --n-random 500  # also add 500 random images
  python -m src.download_data --existing-only # only rebuild labels.csv from data/images

The exact image list and split we used is committed as data/labels.csv.
"""
import argparse
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import config as C


def kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def fetch_from_kaggle(api, name, out_dir, retries=4):
    """Download one competition file into out_dir and return its path.

    Kaggle throttles per-file downloads (HTTP 429), so a failed request is
    retried with an increasing pause.
    """
    out_dir = Path(out_dir)
    for attempt in range(retries):
        try:
            api.competition_download_file(C.KAGGLE_COMPETITION, name, path=str(out_dir), quiet=True)
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(20 * 2 ** attempt)
    path = out_dir / Path(name).name
    zipped = Path(str(path) + ".zip")
    if zipped.exists():                      # large files arrive zipped
        with zipfile.ZipFile(zipped) as z:
            z.extractall(out_dir)
        zipped.unlink()
    return path


def build_labels(seg_csv):
    """One row per image: n_ships (number of masks) and has_ship (0/1)."""
    seg = pd.read_csv(seg_csv)
    seg = seg[~seg.ImageId.isin(C.CORRUPT_IMAGES)]
    labels = (seg.groupby("ImageId").EncodedPixels.count()
                 .rename("n_ships").reset_index())
    labels["has_ship"] = (labels.n_ships > 0).astype(int)
    return labels


def save_resized(img, image_id):
    """Resize a decoded BGR image to 256x256 and store it as JPEG."""
    if img is None:
        return False
    img = cv2.resize(img, (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.imwrite(str(C.IMAGE_DIR / image_id), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def add_subset_zip(api, known_ids):
    """Images from the ready-made Kaggle subset (downloaded as a single zip)."""
    folder = C.DATA_DIR / "extra"
    folder.mkdir(exist_ok=True)
    zips = list(folder.glob("*.zip"))
    if not zips:
        print("Downloading", C.KAGGLE_SUBSET_DATASET)
        api.dataset_download_files(C.KAGGLE_SUBSET_DATASET, path=str(folder), quiet=True)
        zips = list(folder.glob("*.zip"))
    with zipfile.ZipFile(zips[0]) as z:
        names = [n for n in z.namelist() if n.endswith(".jpg") and Path(n).name in known_ids]
        for n in tqdm(names, desc="subset zip"):
            image_id = Path(n).name
            if not (C.IMAGE_DIR / image_id).exists():
                buf = np.frombuffer(z.read(n), np.uint8)
                save_resized(cv2.imdecode(buf, cv2.IMREAD_COLOR), image_id)


def add_random_images(api, labels, n, workers):
    """n images sampled uniformly at random from the whole competition."""
    ids = labels.ImageId.sample(n=n, random_state=C.SEED)

    def get(image_id):
        if (C.IMAGE_DIR / image_id).exists():
            return True
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = fetch_from_kaggle(api, f"train_v2/{image_id}", tmp)
                return save_resized(cv2.imread(str(path), cv2.IMREAD_COLOR), image_id)
        except Exception as e:
            print("failed:", image_id, str(e)[:60])
            return False

    with ThreadPoolExecutor(workers) as pool:
        ok = list(tqdm(pool.map(get, ids), total=n, desc="random images"))
    print(f"random images: {sum(ok)}/{n} downloaded")


def split(df):
    """Stratified train/dev/test split in the proportions of config.SPLIT."""
    tr, dv, te = C.SPLIT
    rest, test = train_test_split(df, test_size=te / (tr + dv + te),
                                  stratify=df.has_ship, random_state=C.SEED)
    train, dev = train_test_split(rest, test_size=dv / (tr + dv),
                                  stratify=rest.has_ship, random_state=C.SEED)
    return pd.concat([train.assign(split="train"), dev.assign(split="dev"),
                      test.assign(split="test")]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n-random", type=int, default=0,
                    help="additionally download N uniformly random competition images")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--existing-only", action="store_true",
                    help="download nothing; just rebuild labels.csv from data/images")
    args = ap.parse_args()

    seg_csv = C.DATA_DIR / C.KAGGLE_CSV
    if not args.existing_only:
        api = kaggle_api()
        if not seg_csv.exists():
            print("Downloading", C.KAGGLE_CSV)
            fetch_from_kaggle(api, C.KAGGLE_CSV, C.DATA_DIR)
    labels = build_labels(seg_csv)
    print(f"Full competition data: {len(labels)} images, {labels.has_ship.mean():.1%} contain a ship")

    if not args.existing_only:
        add_subset_zip(api, set(labels.ImageId))
        if args.n_random:
            add_random_images(api, labels, args.n_random, args.workers)

    have = sorted(p.name for p in C.IMAGE_DIR.glob("*.jpg"))
    subset = split(labels[labels.ImageId.isin(have)].sort_values("ImageId"))
    subset.to_csv(C.LABELS_CSV, index=False)
    print(f"Saved labels for {len(subset)} images -> {C.LABELS_CSV}")
    print(subset.groupby("split").has_ship.agg(["size", "mean"]).rename(
        columns={"size": "images", "mean": "ship_rate"}).loc[["train", "dev", "test"]])


if __name__ == "__main__":
    main()
