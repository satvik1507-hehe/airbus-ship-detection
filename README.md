# Airbus Ship Detection — Traditional ML vs. Convolutional Neural Networks

UE24CS352A Machine Learning — Mini-Project (Problem Statement 35)

| | |
|---|---|
| **Team** | 9 |
| **Members** | Rohan Suresh (PES1UG24CS383), Satvik L (PES1UG24CS423) |
| **Reference** | Chen, Zheng, Zhou — *Airbus Ship Detection: Traditional v.s. Convolutional Neural Network Approach*, Stanford CS229, 2018 ([report](https://cs229.stanford.edu/proj2018/report/58.pdf), [poster](https://cs229.stanford.edu/proj2018/poster/58.pdf)) |

## Problem

Given a satellite image, decide whether it contains **at least one ship** (binary
classification). We compare two families of models on the same data:

* **Traditional ML** on hand-engineered features — LDA, k-NN, Naive Bayes, Random Forest, SVM
* **Deep learning** — a transfer-learning CNN (frozen DenseNet-169) and a simple CNN trained from scratch

## Dataset

[Kaggle Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection/data):
about 192k training images of 768×768 pixels (≈31 GB) plus a CSV of run-length-encoded ship masks.
An image with no mask contains no ship; roughly one image in five contains a ship.

Like the reference report we do **not** use the whole dataset. We stayed under a 1 GB
download budget and used **3,829 images**:

* 3,325 images from a ready-made 494 MB subset of the competition's `train_v2` images
  (Kaggle dataset `aravjain007/aerial-ship-detection-dataset`, one zip download), and
* 515 images sampled uniformly at random from the competition (Kaggle rate-limits
  per-file downloads to a few hundred); 11 images are in both.

Every label comes from the **official competition CSV**. The subset is ship-enriched:
**67.2 % ship / 32.8 % no ship**, so always predicting "ship" scores 67.2 %.
Images are resized to 256×256×3 and split with stratification in the report's 7:3:5
ratio: **1,786 train / 766 dev / 1,277 test**. The exact image list and split is
committed as `data/labels.csv`.

## Setup

Python 3.11–3.13 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

On Windows keep the project in a short path (e.g. `C:\projects\airbus-ship-detection`);
PyTorch fails to load its DLLs from very long paths.

### Getting the data

1. Create a Kaggle account and accept the rules of the
   [competition](https://www.kaggle.com/competitions/airbus-ship-detection) (Join / Late Submission).
2. Kaggle → Settings → API Tokens → create a token and save it as
   `C:\Users\<you>\.kaggle\access_token` (Linux/macOS: `~/.kaggle/access_token`).
3. Download and prepare the data (about 0.55 GB):

```bash
python -m src.download_data
```

This fetches the label CSV and the 494 MB subset, resizes the images and rebuilds the
split. Add `--n-random 500` to also fetch uniformly random competition images
(slow and rate-limited). The 515 random images of our run cannot be re-downloaded
deterministically, so a fresh download gives 3,325 images and slightly different numbers.

## Running

Everything, end to end (features → traditional models → both CNNs → evaluation → figures):

```bash
python run_all.py
```

Or one step at a time:

```bash
python -m src.features                 # colour histogram, Haralick texture, Hu moments
python -m src.train_traditional        # 5 algorithms x 8 feature sets + 10-fold CV
python -m src.train_cnn --model tl     # transfer-learning CNN (frozen DenseNet-169)
python -m src.train_cnn --model sim    # simple CNN trained from scratch
python -m src.evaluate_cnn             # threshold scan, confusion matrices, curves
python -m src.make_figures             # dataset / comparison figures
```

### Live demo

```bash
python -m src.demo                     # 8 random test images, all three models
python -m src.demo --n 12 --seed 7     # another random selection
python -m src.demo --image some.jpg    # your own image(s)
```

The demo prints P(ship) from the Random Forest, TL-CNN and SIM-CNN for each image and
shows a picture grid with the SIM-CNN verdict (green = correct, red = wrong).

### Live training demo (about 40 seconds)

```bash
python -m src.live_training                # ~40 s, 30 epochs
python -m src.live_training --seconds 60   # slower, easier to narrate
```

Trains the TL-CNN head from random weights on the cached DenseNet-169 features
(same training code as `train_cnn.py`) and animates each epoch: loss and accuracy
curves, the dev-set P(ship) histogram separating, and 12 dev images whose predictions
turn from guesses into correct answers. It ends by scoring the best epoch on the test
set. It does not overwrite the trained models in `models/`. It needs
`data/cache/densenet_*.npy`, created by `python -m src.train_cnn --model tl`.

## Repository layout

```
src/
  config.py             paths, split sizes, hyper-parameters (single source of truth)
  download_data.py      download the image subset, resize to 256x256, write labels.csv
  data.py               image / label loading with .npy caching
  features.py           hand-engineered descriptors: His, Ha, Hu (+ raw pixels)
  train_traditional.py  LDA, KNN, NB, RF, SVM; feature-combination table; 10-fold CV
  models.py             SimCNN (4,292,001 params) and TLHead (216,577 params) + DenseNet backbone
  train_cnn.py          training loop, data augmentation, best-checkpoint saving
  evaluate_cnn.py       threshold scanning, test metrics, confusion matrices, curves
  make_figures.py       descriptive figures for the write-up and slides
  demo.py               live demonstration
  live_training.py      ~40-second animated training run for the live demo
run_all.py              runs every step in order
results/                metrics (CSV/JSON) and figures produced by the code
docs/                   two-page write-up, slide deck, viva guide
```

`data/` (images and caches) is not committed — it is re-created by `download_data.py`.

## Method in brief

**Features** (global descriptors of the whole image):
*His* — 8×8×8 colour histogram in HSV space (512 values);
*Ha* — 13 Haralick texture statistics from the grey-level co-occurrence matrix;
*Hu* — 7 Hu shape moments;
*raw* — the image down-scaled to 32×32×3 and flattened (the "no feature engineering" baseline).

**Traditional models** — features are standardised, then LDA, k-NN (k = 5),
Gaussian Naive Bayes, Random Forest (70 trees) and RBF-kernel SVM are trained on
every combination of descriptors and scored on the test set; 10-fold
cross-validation measures stability.

**SIM-CNN** — 4 × (3×3 conv → ReLU → 2×2 max-pool) with 16/32/64/128 filters, then
FC(128) → FC(1) → sigmoid. Trained from scratch with augmentation (flip, rotation,
shift, zoom, shear, brightness).

**TL-CNN** — ImageNet-pretrained DenseNet-169 with all weights frozen, global max
pooling, then BatchNorm → Dropout → FC(128) → FC(1) → sigmoid. Only the head is trained.

**Training** — binary cross-entropy, Adam, batch size 64, 30 epochs, learning rate
halved when the dev loss stalls, best-dev-loss checkpoint kept.

**Imbalance** — the classes are not balanced, so we report the majority-class baseline,
precision, recall and specificity alongside accuracy, and choose the decision
threshold by scanning 0.1–0.9 on the dev set instead of assuming 0.5.

## Results (1,277 test images)

| Model | Accuracy | Precision | Recall | Notes |
|---|---|---|---|---|
| LDA (all features) | 85.4 % | 87.0 % | 92.0 % | 10-fold CV 83.4 ± 2.5 % |
| k-NN (all features) | 83.9 % | 85.6 % | 91.4 % | 10-fold CV 82.9 ± 2.4 % |
| Naive Bayes (all features) | 53.2 % | 91.4 % | 33.4 % | 10-fold CV 51.5 ± 2.8 % |
| **Random Forest (all features)** | **91.9 %** | 93.2 % | 94.9 % | 10-fold CV 92.4 ± 1.7 % |
| SVM (all features) | 85.0 % | 84.3 % | 95.5 % | 10-fold CV 84.6 ± 2.5 % |
| **TL-CNN** (threshold 0.5) | **95.3 %** | 96.3 % | 96.7 % | specificity 92.4 %, AUC 0.982 |
| SIM-CNN (threshold 0.6) | 88.0 % | 91.1 % | 91.0 % | specificity 81.9 %, AUC 0.928 |
| Always predict "ship" | 67.2 % | 67.2 % | 100 % | baseline |

* Random Forest is the best traditional model, as in the reference report (93 %).
* TL-CNN beat SIM-CNN here, the opposite of the report (90 % vs 94 %); we trained on
  1,786 images instead of 7,000, which is the likely reason. SIM-CNN's dev loss was
  still falling at epoch 30.
* Full tables and figures: `results/`. Write-up, slides and viva guide: `docs/`
  (rebuild with `python docs/build_docs.py`; needs `reportlab` and `python-pptx`).

Trained models are committed in `models/`, so `python -m src.demo --image <file>` works
straight after cloning; the random-test-image demo also needs `data/`.

## Differences from the reference report

* 3,829 images with 67 % ships instead of 15,000 images with about 16 % ships, so
  absolute numbers are not directly comparable.
* The raw-pixel baseline uses a 32×32 down-scaled image (3,072 values) rather than the
  full 196,608-value image, to keep k-NN and SVM tractable on a laptop.
* Random Forest uses 70 trees (the poster's setting; the report text says 10 trees of depth 5).
* TL-CNN: because the backbone is frozen, DenseNet features are computed once and
  cached, so the head trains in seconds; augmentation is therefore applied to SIM-CNN only.
* Implemented in PyTorch and trained on CPU (the report used Keras on Azure GPUs).
