"""Live demo - watch a network learn, in about 40 seconds.

Trains the TL-CNN classifier head from scratch (random weights) on the cached,
frozen DenseNet-169 features, using the same code as the real training
(src/train_cnn.py), and animates every epoch:

  * training and dev loss falling
  * training and dev accuracy rising
  * the dev-set P(ship) histogram separating into "ship" and "no ship"
  * 12 dev images whose predictions flip from guesses to correct answers

At the end the best epoch's weights are scored on the test set.
Nothing is saved over the real trained model in models/.

  python -m src.live_training                 # ~40 s, 30 epochs
  python -m src.live_training --seconds 60    # slower, easier to narrate
  python -m src.live_training --seed 3        # different random start / images

Needs data/cache/densenet_*.npy (created by `python -m src.train_cnn --model tl`).
"""
import argparse
import copy
import time

T_START = time.time()

import matplotlib
import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .data import load_split
from .models import TLHead, count_trainable
from .train_cnn import run_epoch


@torch.no_grad()
def probabilities(model, X):
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--seconds", type=float, default=40, help="approximate total run time")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--no-show", action="store_true", help="no window; just save the final frame")
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading cached DenseNet-169 features and images ...")
    import pandas as pd
    labels = pd.read_csv(C.LABELS_CSV)
    y = {s: labels[labels.split == s].has_ship.to_numpy(np.int64) for s in ("train", "dev", "test")}
    X = {s: np.load(C.CACHE_DIR / f"densenet_{s}.npy") for s in ("train", "dev", "test")}
    for s in X:
        if len(X[s]) != len(y[s]):
            raise SystemExit("Feature cache does not match labels.csv - rerun `python -m src.train_cnn --model tl`.")
    dev_pixels, _, _ = load_split("dev")
    t_loaded = time.time()

    # 6 ship + 6 no-ship dev images to watch
    rng = np.random.default_rng(args.seed)
    show = np.concatenate([rng.choice(np.where(y["dev"] == c)[0], 6, replace=False) for c in (1, 0)])
    rng.shuffle(show)

    model = TLHead(X["train"].shape[1])
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    prepare = lambda xb, train: torch.from_numpy(xb)

    # ------------------------------------------------------------------ figure
    plt.ion()
    fig = plt.figure(figsize=(15, 8.6))
    fig.canvas.manager.set_window_title("Live training - TL-CNN head")
    gs = fig.add_gridspec(3, 6, height_ratios=[1.25, 1, 1], hspace=0.55, wspace=0.42,
                          left=0.05, right=0.98, top=0.86, bottom=0.04)
    ax_loss, ax_acc, ax_hist = (fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
                                fig.add_subplot(gs[0, 4:6]))
    img_axes = [fig.add_subplot(gs[1 + i // 6, i % 6]) for i in range(12)]
    title = fig.suptitle("", fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.912, f"Frozen ImageNet DenseNet-169 features -> BatchNorm -> Dropout -> FC(128) -> FC(1) -> sigmoid   |   "
             f"{count_trainable(model):,} trainable parameters   |   {len(y['train']):,} training images, "
             f"{len(y['dev']):,} dev images", ha="center", fontsize=10, color="#444")

    hist = {k: [] for k in ("train_loss", "train_acc", "dev_loss", "dev_acc")}
    best = {"loss": float("inf"), "epoch": 0, "state": None}

    # Artists are created once and only their data changes each epoch (fast redraws).
    lines = {}
    for ax, key, name in ((ax_loss, "loss", "Binary cross-entropy loss"), (ax_acc, "acc", "Accuracy")):
        lines[f"train_{key}"], = ax.plot([], [], "o-", c="tab:orange", ms=3, label="train")
        lines[f"dev_{key}"], = ax.plot([], [], "o--", c="tab:blue", ms=3, label="dev (unseen)")
        lines[f"best_{key}"] = ax.axvline(-5, c="green", lw=1.2, ls=":", label="best dev loss")
        ax.set(xlim=(0.5, args.epochs + 0.5), xlabel="Epoch", title=name)
        ax.grid(alpha=.3)
    ax_loss.set_ylim(0, 0.8)
    ax_acc.set_ylim(0.5, 1.0)
    base = max(y["dev"].mean(), 1 - y["dev"].mean())
    ax_acc.axhline(base, c="gray", ls="--", lw=1, label=f"always 'ship' ({base:.0%})")
    ax_loss.legend(fontsize=8, loc="upper right")
    ax_acc.legend(fontsize=8, loc="lower right")

    bins = np.linspace(0, 1, 26)
    width = bins[1] - bins[0]
    bars_neg = ax_hist.bar(bins[:-1], np.zeros(25), width=width, align="edge", color="tab:blue",
                           alpha=.75, label="really no ship")
    bars_pos = ax_hist.bar(bins[:-1], np.zeros(25), width=width, align="edge", color="tab:red",
                           alpha=.6, label="really ship")
    ax_hist.axvline(0.5, c="k", ls="--", lw=1)
    ax_hist.set(xlim=(0, 1), ylim=(0, 60), xlabel="Predicted P(ship) on dev images", ylabel="Images",
                title="Do the two classes separate?")
    ax_hist.legend(fontsize=8, loc="upper center")

    img_titles = []
    for ax, i in zip(img_axes, show):
        ax.imshow(dev_pixels[i][..., ::-1])
        ax.set_xticks([]), ax.set_yticks([])
        img_titles.append(ax.set_title("", fontsize=9))
        for s in ax.spines.values():
            s.set_linewidth(3)

    draw_times = []

    def wait(seconds):
        """Keep the window responsive without forcing another full redraw (plt.pause would)."""
        if seconds > 0:
            fig.canvas.start_event_loop(seconds)

    def draw(epoch, lr):
        p_dev = probabilities(model, X["dev"])
        title.set_text(f"Live training - epoch {epoch}/{args.epochs}   (learning rate {lr:.2g})"
                       if epoch else "Before training - random weights, the network is guessing")
        ep = np.arange(1, len(hist["train_loss"]) + 1)
        for key in ("train_loss", "dev_loss", "train_acc", "dev_acc"):
            lines[key].set_data(ep, hist[key])
        for key in ("best_loss", "best_acc"):
            lines[key].set_xdata([best["epoch"] or -5] * 2)
        if hist["train_loss"]:
            ax_loss.set_ylim(0, max(0.8, *hist["train_loss"], *hist["dev_loss"]) * 1.05)

        neg, _ = np.histogram(p_dev[y["dev"] == 0], bins=bins)
        pos, _ = np.histogram(p_dev[y["dev"] == 1], bins=bins)
        for b, h in zip(bars_neg, neg):
            b.set_height(h)
        for b, h in zip(bars_pos, pos):
            b.set_height(h)
        ax_hist.set_ylim(0, max(60, neg.max(), pos.max()) * 1.1)

        for ax, t, i in zip(img_axes, img_titles, show):
            p, truth = p_dev[i], y["dev"][i]
            ok = (p >= 0.5) == truth
            color = "green" if ok else "red"
            t.set_text(f"P(ship) = {p:.2f}  {'✓' if ok else '✗'}\ntruth: {C.CLASS_NAMES[truth]}")
            t.set_color(color)
            for s in ax.spines.values():
                s.set_edgecolor(color)
        t = time.time()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        draw_times.append(time.time() - t)

    # ------------------------------------------------------------------ train
    # Pace the run so it takes about --seconds in total, however fast the machine is.
    deadline = T_START + args.seconds - 1.0
    plt.show(block=False)
    draw(0, C.LEARNING_RATE)
    wait(min(3.0, args.seconds * 0.1))                 # let the audience see the "guessing" state
    print(f"{'epoch':>5} {'train loss':>10} {'train acc':>9} {'dev loss':>9} {'dev acc':>8}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, X["train"], y["train"], prepare, loss_fn, opt)
        dl, da = run_epoch(model, X["dev"], y["dev"], prepare, loss_fn)
        sched.step(dl)
        for k, v in zip(hist, (tl, ta, dl, da)):
            hist[k].append(v)
        if dl < best["loss"]:
            best.update(loss=dl, epoch=epoch, state=copy.deepcopy(model.state_dict()))
        print(f"{epoch:5d} {tl:10.4f} {ta:9.3f} {dl:9.4f} {da:8.3f}{'  * best so far' if best['epoch'] == epoch else ''}")
        draw(epoch, opt.param_groups[0]["lr"])
        per_epoch = (deadline - t0) / (args.epochs - epoch + 1)
        wait(per_epoch - (time.time() - t0))

    # ------------------------------------------------------------------ test
    model.load_state_dict(best["state"])
    p_test = probabilities(model, X["test"])
    pred = (p_test >= 0.5).astype(int)
    yt = y["test"]
    tp, tn = int(((pred == 1) & (yt == 1)).sum()), int(((pred == 0) & (yt == 0)).sum())
    fp, fn = int(((pred == 1) & (yt == 0)).sum()), int(((pred == 0) & (yt == 1)).sum())
    acc = (tp + tn) / len(yt)
    summary = (f"Finished. Kept the weights from epoch {best['epoch']} (lowest dev loss).   "
               f"TEST SET ({len(yt):,} never-seen images): accuracy {acc:.1%}, "
               f"precision {tp / max(tp + fp, 1):.1%}, recall {tp / max(tp + fn, 1):.1%}   "
               f"[TP {tp}  TN {tn}  FP {fp}  FN {fn}]")
    title.set_text(summary.split("   ")[0])
    fig.text(0.5, 0.947, summary.split("   ", 1)[1], ha="center", fontsize=11, color="darkgreen",
             fontweight="bold")
    print("\n" + summary)
    print(f"Run time {time.time() - T_START:.0f} s (startup {t_loaded - T_START:.1f} s, "
          f"average redraw {np.mean(draw_times):.2f} s)")
    out = C.FIG_DIR / "live_training.png"
    fig.savefig(out, dpi=110)
    print("Saved final frame ->", out)
    if not args.no_show:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
