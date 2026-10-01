"""Central configuration: paths, split sizes and hyper-parameters.

Every other module imports from here, so changing a number in this file
changes it everywhere.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"        # 256x256 JPEGs
LABELS_CSV = DATA_DIR / "labels.csv"   # ImageId, has_ship, n_ships, split
CACHE_DIR = DATA_DIR / "cache"         # .npy arrays (pixels, features)
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"

SEED = 229
IMG_SIZE = 256                         # images are resized to 256x256x3 (as in the paper)
RAW_SIZE = 32                          # side of the down-scaled image used as "raw pixel" features

# Train / dev / test proportions - the Stanford report used 7k / 3k / 5k.
SPLIT = (7, 3, 5)

KAGGLE_COMPETITION = "airbus-ship-detection"
KAGGLE_CSV = "train_ship_segmentations_v2.csv"
KAGGLE_SUBSET_DATASET = "aravjain007/aerial-ship-detection-dataset"   # 494 MB of train_v2 images
CORRUPT_IMAGES = {"6384c3e78.jpg"}     # known truncated file in the Kaggle data

# CNN training (Section 4.3 of the report)
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]   # 0.1 ... 0.9

CLASS_NAMES = ["No ship", "Ship"]

for _d in (DATA_DIR, IMAGE_DIR, CACHE_DIR, RESULTS_DIR, FIG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
