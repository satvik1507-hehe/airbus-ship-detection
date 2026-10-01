"""Step 4 - train the CNNs (Section 4.3).

  python -m src.train_cnn --model sim     # SIM-CNN, trained from scratch
  python -m src.train_cnn --model tl      # TL-CNN, frozen DenseNet-169 + small head

Setup taken from the report: binary cross-entropy loss, Adam, mini-batches of 64,
30 epochs, decaying learning rate, and the weights are only kept when the
development-set loss improves ("best checkpoint").
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .data import load_split
from .models import DenseNetBackbone, SimCNN, TLHead, count_trainable

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_tensor(x_uint8_bgr):
    """uint8 [B,H,W,3] BGR numpy -> float [B,3,H,W] RGB tensor in [0,1]."""
    x = torch.from_numpy(np.ascontiguousarray(x_uint8_bgr[..., ::-1]))
    return x.permute(0, 3, 1, 2).float().div(255.0)


def augment(x):
    """Random flip / rotation / shift / zoom / shear / brightness on a batch.

    A ship is still a ship after any of these, so they create "new" training
    images for free and make the network less likely to memorise the data.
    """
    b = x.size(0)
    if torch.rand(1) < 0.5:
        x = x.flip(3)                                         # horizontal flip
    if torch.rand(1) < 0.5:
        x = x.flip(2)                                         # vertical flip
    ang = (torch.rand(b) * 2 - 1) * math.radians(20)          # rotation +-20 deg
    zoom = 1 + (torch.rand(b) * 2 - 1) * 0.10                 # zoom +-10 %
    shear = (torch.rand(b) * 2 - 1) * 0.10                    # shear
    tx, ty = (torch.rand(b) * 2 - 1) * 0.2, (torch.rand(b) * 2 - 1) * 0.2   # shift +-10 %
    theta = torch.stack([
        torch.stack([zoom * torch.cos(ang), -zoom * torch.sin(ang) + shear, tx], 1),
        torch.stack([zoom * torch.sin(ang), zoom * torch.cos(ang), ty], 1)], 1).to(x.device)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    bright = 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * 0.2    # brightness +-20 %
    return (x * bright).clamp(0, 1)


def densenet_features(split, batch=32):
    """Pooled DenseNet-169 features for a split. The backbone is frozen, so its
    output for an image never changes - we compute it once and cache it."""
    cache = C.CACHE_DIR / f"densenet_{split}.npy"
    X, y, _ = load_split(split)
    if cache.exists() and len(np.load(cache, mmap_mode="r")) == len(X):
        return np.load(cache), y
    backbone = DenseNetBackbone().to(DEVICE)
    out = []
    for i in range(0, len(X), batch):
        out.append(backbone(to_tensor(X[i:i + batch]).to(DEVICE)).cpu().numpy())
        if (i // batch) % 20 == 0:
            print(f"  densenet features {split}: {i}/{len(X)}", flush=True)
    feats = np.concatenate(out)
    np.save(cache, feats)
    return feats, y


def run_epoch(model, X, y, prepare, loss_fn, opt=None):
    """One pass over (X, y). Trains when an optimiser is given, else evaluates."""
    train = opt is not None
    model.train(train)
    order = np.random.permutation(len(X)) if train else np.arange(len(X))
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(train):
        for i in range(0, len(X), C.BATCH_SIZE):
            idx = np.sort(order[i:i + C.BATCH_SIZE])
            xb = prepare(X[idx], train).to(DEVICE)
            yb = torch.from_numpy(y[idx]).float().to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(idx)
            correct += ((logits > 0).float() == yb).sum().item()
    return total_loss / len(X), correct / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sim", "tl"], required=True)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    args = ap.parse_args()
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)

    if args.model == "sim":
        Xtr, ytr, _ = load_split("train")
        Xdv, ydv, _ = load_split("dev")
        model = SimCNN()
        prepare = lambda xb, train: augment(to_tensor(xb)) if train else to_tensor(xb)
    else:
        Xtr, ytr = densenet_features("train")
        Xdv, ydv = densenet_features("dev")
        model = TLHead(Xtr.shape[1])
        prepare = lambda xb, train: torch.from_numpy(xb)
    model.to(DEVICE)
    print(f"{args.model}: {count_trainable(model):,} trainable parameters, device={DEVICE}")

    loss_fn = nn.BCEWithLogitsLoss()                      # sigmoid + binary cross-entropy
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    history = {k: [] for k in ("train_loss", "train_acc", "dev_loss", "dev_acc", "lr")}
    best = float("inf")
    ckpt = C.MODEL_DIR / f"{args.model}_cnn.pt"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, Xtr, ytr, prepare, loss_fn, opt)
        dl, da = run_epoch(model, Xdv, ydv, prepare, loss_fn)
        sched.step(dl)
        for k, v in zip(history, (tl, ta, dl, da, opt.param_groups[0]["lr"])):
            history[k].append(v)
        kept = ""
        if dl < best:                                     # keep only the best weights
            best = dl
            torch.save(model.state_dict(), ckpt)
            kept = "  * saved"
        print(f"epoch {epoch:2d}/{args.epochs}  train loss {tl:.4f} acc {ta:.4f} | "
              f"dev loss {dl:.4f} acc {da:.4f}  ({time.time() - t0:.0f}s){kept}", flush=True)

    history["trainable_parameters"] = count_trainable(model)
    json.dump(history, open(C.RESULTS_DIR / f"{args.model}_history.json", "w"), indent=2)


if __name__ == "__main__":
    main()
