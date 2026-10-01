"""Builds the documents in docs/ from the numbers in results/.

  python docs/build_docs.py

Produces
  Team9_Project35_Writeup.pdf     two-page summary
  Team9_Project35_Slides.pptx     presentation deck
  Team9_Project35_Viva_Guide.pdf  explanation of the project + likely viva questions
"""
import json
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES, FIG, DOCS = ROOT / "results", ROOT / "results" / "figures", ROOT / "docs"

TEAM = "Team 9  |  Project No. 35"
MEMBERS = "Rohan Suresh (PES1UG24CS383)  ·  Satvik L (PES1UG24CS423)"
COURSE = "UE24CS352A – Machine Learning  |  Mini-Project"
TITLE = "Airbus Ship Detection: Traditional ML vs. Convolutional Neural Networks"

# ----------------------------------------------------------------- numbers
trad = json.load(open(RES / "traditional_results.json"))
cnn = json.load(open(RES / "cnn_results.json"))
table1 = pd.read_csv(RES / "table1_test_accuracy.csv", index_col=0)
hist = {k: json.load(open(RES / f"{k}_history.json")) for k in ("sim", "tl")}
labels = pd.read_csv(ROOT / "data" / "labels.csv")

N = len(labels)
N_TR, N_DV, N_TE = trad["n_train"], trad["n_dev"], trad["n_test"]
SHIP = labels.has_ship.mean()
BASE = trad["majority_baseline"]
ALGS = ["LDA", "KNN", "NB", "RF", "SVM"]
ALL = "w/ all"
RAW = "w/o (raw pixels)"
acc_all = {a: trad["detail"][f"{a}|{ALL}"]["accuracy"] for a in ALGS}
rf = trad["detail"][f"RF|{ALL}"]
best_trad = max(acc_all, key=acc_all.get)
sim, tl = cnn["sim"]["test_at_best_threshold"], cnn["tl"]["test_at_best_threshold"]
sim5, tl5 = cnn["sim"]["test_at_0.5"], cnn["tl"]["test_at_0.5"]
sim_auc, tl_auc = cnn["sim"]["test_roc_auc"], cnn["tl"]["test_roc_auc"]
cv = trad["cv"]
best_overall = max({**acc_all, "TL-CNN": tl["accuracy"], "SIM-CNN": sim["accuracy"]}.items(),
                   key=lambda kv: kv[1])
best_cell = table1.stack().idxmax()
P_SIM, P_TL = hist["sim"]["trainable_parameters"], hist["tl"]["trainable_parameters"]
below_base = [a for a in ALGS if acc_all[a] < BASE]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


cnn_order = ("SIM-CNN", "TL-CNN") if sim["accuracy"] >= tl["accuracy"] else ("TL-CNN", "SIM-CNN")
cnn_hi, cnn_lo = max(sim["accuracy"], tl["accuracy"]), min(sim["accuracy"], tl["accuracy"])
CNN_SENTENCE = (f"{cnn_order[0]} ({pct(cnn_hi)}) scored higher than {cnn_order[1]} ({pct(cnn_lo)}) "
                f"on our test set")
FE_GAIN = {a: table1.loc[a, ALL] - table1.loc[a, RAW] for a in ALGS}
fe_helped = [a for a in ALGS if FE_GAIN[a] > 0.01]
fe_hurt = [a for a in ALGS if FE_GAIN[a] < -0.01]

# ----------------------------------------------------------------- pdf helpers
ss = getSampleStyleSheet()


def styles(base=9.6):
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=base + 5.5, leading=base + 8,
                                spaceAfter=2),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=base - 0.6, leading=base + 2,
                              alignment=TA_CENTER, textColor=colors.HexColor("#444444")),
        "h": ParagraphStyle("h", parent=ss["Heading2"], fontSize=base + 1.6, leading=base + 4,
                            spaceBefore=6, spaceAfter=2, textColor=colors.HexColor("#0b3d6b")),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=base + 5, leading=base + 8,
                             spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#0b3d6b")),
        "b": ParagraphStyle("b", parent=ss["Normal"], fontSize=base, leading=base + 3.2,
                            spaceAfter=3),
        "q": ParagraphStyle("q", parent=ss["Normal"], fontSize=base, leading=base + 3.2,
                            spaceBefore=5, spaceAfter=1, fontName="Helvetica-Bold"),
        "cap": ParagraphStyle("c", parent=ss["Normal"], fontSize=base - 1.4, leading=base + 1,
                              alignment=TA_CENTER, textColor=colors.HexColor("#444444")),
        "code": ParagraphStyle("code", parent=ss["Code"], fontSize=base - 1, leading=base + 2,
                               backColor=colors.HexColor("#f2f2f2"), borderPadding=4,
                               spaceBefore=3, spaceAfter=6),
    }


def bullets(items, st, left=12):
    return ListFlowable([ListItem(Paragraph(i, st), leftIndent=left) for i in items],
                        bulletType="bullet", start="•", leftIndent=left, bulletFontSize=7)


def grid(data, col_widths=None, font=8.2, header=True, bold_cells=()):
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    st = [("FONTSIZE", (0, 0), (-1, -1), font), ("LEADING", (0, 0), (-1, -1), font + 2.2),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
          ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe7f3")),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    for (r, c) in bold_cells:
        st.append(("FONTNAME", (c, r), (c, r), "Helvetica-Bold"))
    t.setStyle(TableStyle(st))
    return t


def table1_rows():
    cols = list(table1.columns)
    head = ["", "His", "Ha", "Hu", "His+Ha", "Ha+Hu", "Hu+His", "All three", "Raw pixels"]
    rows = [head] + [[a] + [f"{table1.loc[a, c]:.3f}" for c in cols] for a in ALGS]
    r, c = ALGS.index(best_cell[0]) + 1, cols.index(best_cell[1]) + 1
    return rows, [(r, c)]


def summary_rows():
    rows = [["Model", "Accuracy", "Precision", "Recall", "Specificity / notes"]]
    for a in ALGS:
        d = trad["detail"][f"{a}|{ALL}"]
        rows.append([f"{a} (all features)", pct(d["accuracy"]), pct(d["precision"]),
                     pct(d["recall"]), f"10-fold CV {pct(cv[a]['mean'])} ± {pct(cv[a]['std'])}"])
    rows.append([f"TL-CNN (thr {tl['threshold']})", pct(tl["accuracy"]), pct(tl["precision"]),
                 pct(tl["recall"]), f"specificity {pct(tl['specificity'])}, AUC {tl_auc:.3f}"])
    rows.append([f"SIM-CNN (thr {sim['threshold']})", pct(sim["accuracy"]), pct(sim["precision"]),
                 pct(sim["recall"]), f"specificity {pct(sim['specificity'])}, AUC {sim_auc:.3f}"])
    rows.append(["Always predict 'Ship'", pct(BASE), pct(SHIP), "100.0%", "specificity 0% (baseline)"])
    return rows


def page_number(canvas, doc):
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(A4[0] / 2, 0.8 * cm, f"{TEAM}  –  page {doc.page}")


# ----------------------------------------------------------------- 1. write-up
def build_writeup():
    S = styles(9.1)
    b = S["b"]
    doc = SimpleDocTemplate(str(DOCS / "Team9_Project35_Writeup.pdf"), pagesize=A4,
                            leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.3 * cm,
                            title=TITLE, author="Rohan Suresh, Satvik L")
    t1, bold = table1_rows()
    story = [
        Paragraph(TITLE, S["title"]),
        Paragraph(f"{COURSE}<br/>{TEAM}  |  {MEMBERS}", S["sub"]),

        Paragraph("1. Problem statement", S["h"]),
        Paragraph(
            "Maritime monitoring (illegal fishing, piracy, accident response) needs ships to be found "
            "in satellite imagery automatically. Following the Stanford CS229 project <i>Airbus Ship "
            "Detection – Traditional v.s. Convolutional Neural Network Approach</i> (Chen, Zheng, Zhou, "
            "2018), we treat it as <b>binary image classification</b>: given a satellite image, decide "
            "whether it contains at least one ship. The aim is to compare traditional machine-learning "
            "classifiers on hand-engineered features against convolutional neural networks that learn "
            "their features from pixels. It is hard because ships are tiny relative to the image and "
            "scenes vary widely (open sea, docks, clouds, haze, wakes).", b),

        Paragraph("2. Dataset", S["h"]),
        Paragraph(
            "Source: the Kaggle <b>Airbus Ship Detection Challenge</b> – 192,555 labelled 768×768 RGB "
            "satellite images (31 GB) and a CSV of run-length-encoded ship masks; an image with no mask "
            "has no ship (22.1% of all images contain one). The reference report used a 15k-image "
            f"subset; with a 1 GB download budget we used <b>{N:,} images</b>: a ready-made 494 MB "
            "subset of the competition's training images plus 515 images sampled uniformly at random "
            "(Kaggle rate-limits per-image downloads). Every label comes from the official competition "
            f"CSV. Our subset is ship-enriched – <b>{pct(SHIP)} ship / {pct(1 - SHIP)} no ship</b> – so "
            f"the trivial 'always predict ship' baseline is {pct(BASE)}, and we report precision, recall "
            "and specificity next to accuracy. Images are resized to 256×256×3 and split with "
            f"stratification in the report's 7:3:5 ratio: <b>{N_TR:,} train / {N_DV:,} dev / "
            f"{N_TE:,} test</b>. The test set is used only for the final numbers.", b),

        Paragraph("3. Approach", S["h"]),
        Paragraph(
            "<b>Feature engineering (traditional models).</b> Three global descriptors per image: "
            "<b>His</b> – an 8×8×8 colour histogram in HSV space (512 values, colour); <b>Ha</b> – 13 "
            "Haralick statistics of the grey-level co-occurrence matrix (texture); <b>Hu</b> – the 7 Hu "
            "moments (shape). As the 'no feature engineering' baseline the image is shrunk to 32×32×3 "
            "and flattened (3,072 raw pixel values). Features are standardised using training-set "
            "statistics.", b),
        Paragraph(
            "<b>Traditional classifiers.</b> Linear Discriminant Analysis, k-Nearest Neighbours (k = 5, "
            "Euclidean), Gaussian Naive Bayes, Random Forest (70 trees) and an RBF-kernel SVM, each "
            "trained on all 8 feature combinations (40 experiments), plus stratified 10-fold "
            "cross-validation on train + dev to measure stability.", b),
        Paragraph(
            f"<b>SIM-CNN (from scratch).</b> 4 × [3×3 convolution → ReLU → 2×2 max-pool] with "
            f"16/32/64/128 filters, then FC(128) → dropout → FC(1) → sigmoid; {P_SIM:,} trainable "
            "parameters, identical to the report. Trained with augmentation (flips, rotation, shift, "
            "zoom, shear, brightness). "
            f"<b>TL-CNN (transfer learning).</b> ImageNet-pretrained DenseNet-169 with all weights "
            f"frozen → global max-pool → BatchNorm → dropout → FC(128) → FC(1) → sigmoid; only the "
            f"{P_TL:,}-parameter head is trained (also identical to the report).", b),
        Paragraph(
            "<b>Training and evaluation.</b> Binary cross-entropy loss, Adam (lr 10<sup>-3</sup>, "
            "halved when the dev loss stalls), batch size 64, 30 epochs, keeping the checkpoint with "
            "the lowest dev loss. Because the classes are imbalanced, the decision threshold is chosen "
            "by scanning 0.1–0.9 on the dev set instead of assuming 0.5.", b),

        Paragraph("4. Implementation overview", S["h"]),
        Paragraph(
            "Python 3.13; scikit-learn for the traditional models, OpenCV and mahotas for features, "
            "PyTorch/torchvision for the CNNs; trained on a laptop CPU. The code is a small package "
            "(<font face='Courier'>src/</font>) with one module per stage: "
            "<font face='Courier'>download_data</font> (subset download, resize, labels, split) → "
            "<font face='Courier'>features</font> → <font face='Courier'>train_traditional</font> → "
            "<font face='Courier'>train_cnn</font> → <font face='Courier'>evaluate_cnn</font> → "
            "<font face='Courier'>make_figures</font>, plus <font face='Courier'>demo</font> for live "
            "prediction. <font face='Courier'>run_all.py</font> runs every stage; all settings live in "
            "<font face='Courier'>config.py</font>; random seeds are fixed and the exact image list and "
            "split are committed (<font face='Courier'>data/labels.csv</font>). Since the DenseNet "
            "backbone is frozen, its output for each image is computed once and cached, so the TL-CNN "
            "head trains in seconds.", b),

        Paragraph("5. Results", S["h"]),
        Paragraph(f"<b>Table 1.</b> Test accuracy of the traditional models for each feature set "
                  f"({N_TE:,} test images; best value in bold).", S["cap"]),
        grid(t1, col_widths=[1.3 * cm] + [1.85 * cm] * 8, font=7.8, bold_cells=bold),
        Spacer(1, 5),
        KeepTogether([
            Paragraph("<b>Table 2.</b> All models on the same test set (traditional models use all "
                      "three descriptors).", S["cap"]),
            grid(summary_rows(), col_widths=[4.1 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 6.2 * cm],
                 font=7.8)]),
        Spacer(1, 4),
        Table([[Image(str(FIG / "model_comparison.png"), width=8.7 * cm, height=4.72 * cm),
                Image(str(FIG / "confusion_matrices.png"), width=8.7 * cm, height=3.89 * cm)]],
              hAlign="CENTER"),
        Paragraph("<b>Figure 1.</b> Left: test accuracy of all models against the majority-class "
                  "baseline. Right: normalised confusion matrices of the two CNNs.", S["cap"]),

        Paragraph("6. Conclusions", S["h"]),
        bullets([
            f"<b>Random Forest is the best traditional model</b> ({pct(acc_all['RF'])} test accuracy, "
            f"10-fold CV {pct(cv['RF']['mean'])} ± {pct(cv['RF']['std'])}), in agreement with the "
            "reference report (93%).",
            f"<b>Feature engineering matters:</b> going from raw pixels to the three descriptors "
            f"changed accuracy by {', '.join(f'{a} {100 * FE_GAIN[a]:+.1f}' for a in ALGS)} points. "
            + (f"It helped {', '.join(fe_helped)}" if fe_helped else "")
            + (f" but hurt {', '.join(fe_hurt)}" if fe_hurt else "")
            + ". More features are not always better – the 512 correlated histogram bins break Naive "
              "Bayes's independence assumption, as the report also observed.",
            f"<b>CNNs:</b> {CNN_SENTENCE}; the best model overall was {best_overall[0]} "
            f"({pct(best_overall[1])}). "
            + ("Unlike the report, transfer learning beat the from-scratch network. A likely reason "
               f"is data size: we trained on {N_TR:,} images instead of 7,000, and a 4.3 M-parameter "
               "network trained from scratch needs far more data than a small head on features "
               "pre-learned from 1.2 M ImageNet images. SIM-CNN also scored below Random Forest."
               if tl["accuracy"] > sim["accuracy"] else
               "As in the report, the network trained from scratch on this specific imagery beat the "
               "frozen ImageNet features."),
            "<b>Limitations:</b> a small, ship-enriched subset (so absolute numbers are not directly "
            "comparable with the report's 84%-no-ship data), CPU-only training, a single run per "
            "model, SIM-CNN's dev loss was still falling at epoch 30 (it had not converged), and "
            "image-level classification only. <b>Future work:</b> train on the full "
            "dataset, fine-tune the backbone, and segment/localise the ships.",
        ], b),
    ]
    doc.build(story)


# ----------------------------------------------------------------- 2. slides
NAVY, GREY = RGBColor(0x0B, 0x3D, 0x6B), RGBColor(0x44, 0x44, 0x44)


def build_slides():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    def text(slide, x, y, w, h, lines, size=20, bold=False, color=GREY):
        tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
        tf.word_wrap = True
        for i, line in enumerate(lines if isinstance(lines, list) else [lines]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color
            p.space_after = Pt(size * 0.45)
        return tf

    def slide(title, n=[0]):
        s = prs.slides.add_slide(blank)
        text(s, 0.6, 0.35, 12.1, 0.9, title, size=32, bold=True, color=NAVY)
        bar = s.shapes.add_shape(1, Inches(0.6), Inches(1.2), Inches(12.1), Inches(0.05))
        bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
        n[0] += 1
        text(s, 0.6, 7.0, 12.1, 0.4, f"{TEAM}   ·   Airbus Ship Detection   ·   {n[0] + 1}", size=11)
        return s

    def bl(s, items, x=0.6, y=1.5, w=12.1, h=5.3, size=20):
        text(s, x, y, w, h, ["•  " + i for i in items], size=size)

    def pic(s, name, x, y, w=None, h=None):
        kw = {}
        if w: kw["width"] = Inches(w)
        if h: kw["height"] = Inches(h)
        s.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), **kw)

    def tbl(s, rows, x, y, w, h, size=13, bold_cells=()):
        t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                cell = t.cell(r, c)
                cell.text = str(val)
                f = cell.text_frame.paragraphs[0].font
                f.size = Pt(size)
                f.bold = r == 0 or (r, c) in bold_cells

    # 1 title
    s = prs.slides.add_slide(blank)
    text(s, 0.8, 2.0, 11.7, 1.6, "Airbus Ship Detection", size=50, bold=True, color=NAVY)
    text(s, 0.8, 3.2, 11.7, 0.8, "Traditional Machine Learning vs. Convolutional Neural Networks", size=26)
    text(s, 0.8, 4.6, 11.7, 1.6, [COURSE, TEAM, MEMBERS,
                                  "Reference: Chen, Zheng, Zhou – Stanford CS229 project, 2018"], size=18)

    s = slide("Problem statement")
    bl(s, ["Given a satellite image, decide: does it contain at least one ship?  (binary classification)",
           "Why: maritime monitoring – illegal fishing, piracy, accident response, traffic management",
           "Why it is hard: ships are tiny, scenes vary (open sea, docks, clouds, haze, wakes)",
           "Question we answer: do hand-engineered features + classical models compete with CNNs "
           "that learn features from pixels?",
           "We reproduce the Stanford CS229 study on a laptop-sized subset of the same data"])

    s = slide("Dataset")
    bl(s, ["Kaggle Airbus Ship Detection Challenge: 192,555 images, 768×768 RGB, 31 GB",
           "Labels: CSV of run-length-encoded ship masks; no mask = no ship (22% have a ship)",
           f"Our subset: {N:,} images (≈0.6 GB download), labels from the official CSV",
           f"Ship-enriched: {pct(SHIP)} ship / {pct(1 - SHIP)} no ship → baseline accuracy {pct(BASE)}",
           f"Resized to 256×256; stratified 7:3:5 split → {N_TR:,} train / {N_DV:,} dev / {N_TE:,} test"],
       w=7.2, size=18)
    pic(s, "sample_images.png", 7.9, 1.6, w=5.1)
    pic(s, "class_balance.png", 8.9, 4.0, h=2.9)

    s = slide("Pipeline")
    bl(s, ["download_data  →  subset, resize to 256×256, labels, train/dev/test split",
           "features  →  colour histogram, Haralick texture, Hu moments, raw pixels",
           "train_traditional  →  LDA, KNN, Naive Bayes, Random Forest, SVM × 8 feature sets + 10-fold CV",
           "train_cnn  →  TL-CNN (frozen DenseNet-169) and SIM-CNN (from scratch)",
           "evaluate_cnn  →  threshold scan on dev, test metrics, confusion matrices",
           "demo  →  live prediction on unseen images with all three best models",
           "One command runs everything: python run_all.py   (settings in config.py, seeds fixed)"], size=19)

    s = slide("Feature engineering (traditional models)")
    bl(s, ["His – colour histogram in HSV, 8×8×8 bins = 512 values  →  colour",
           "Ha – 13 Haralick statistics of the grey-level co-occurrence matrix  →  texture",
           "Hu – 7 Hu moments of the grey image  →  shape",
           "Raw – image shrunk to 32×32×3 and flattened = 3,072 values  →  'no feature engineering'",
           "All are global descriptors: one vector describes the whole image",
           "Standardised (zero mean, unit variance) with training-set statistics"], w=7.4, size=18)
    pic(s, "feature_examples.png", 8.1, 2.2, w=5.0)

    s = slide("Five traditional classifiers")
    bl(s, ["LDA – Gaussian classes with a shared covariance → linear decision boundary",
           "k-NN (k = 5) – majority vote of the 5 closest training images (Euclidean distance)",
           "Naive Bayes – Bayes' rule assuming features are independent given the class",
           "Random Forest – 70 decision trees on bootstrap samples and random feature subsets, majority vote",
           "SVM (RBF kernel) – maximum-margin boundary in a non-linear feature space",
           "Each trained on 8 feature combinations = 40 experiments, plus 10-fold cross-validation"], size=19)

    rows, bold = table1_rows()
    s = slide("Results – traditional models (test accuracy)")
    tbl(s, rows, 0.6, 1.5, 12.1, 2.6, size=14, bold_cells=bold)
    bl(s, [f"Random Forest is best: {pct(acc_all['RF'])} with all features (paper: 93%)",
           f"Feature engineering vs raw pixels: " + ", ".join(f"{a} {100 * FE_GAIN[a]:+.0f}" for a in ALGS) + " points",
           "'More is less': the 512 correlated histogram bins break Naive Bayes's independence assumption",
           f"Baseline (always 'ship') = {pct(BASE)}"
           + (f"; {', '.join(below_base)} with all features falls below it" if below_base else "")],
       y=4.4, h=2.5, size=17)

    s = slide("10-fold cross-validation (all features)")
    pic(s, "cv_boxplot.png", 0.6, 1.5, h=5.2)
    bl(s, [f"{a}: {pct(cv[a]['mean'])} ± {pct(cv[a]['std'])}" for a in ALGS]
       + ["Random Forest: highest mean and among the smallest spread → most reliable"], x=8.6, w=4.3, size=18)

    s = slide("Two convolutional neural networks")
    bl(s, ["SIM-CNN – trained from scratch",
           "    4 × [3×3 conv → ReLU → 2×2 max-pool], filters 16 / 32 / 64 / 128",
           "    256×256×3 → 16×16×128 → FC(128) → dropout → FC(1) → sigmoid",
           f"    {P_SIM:,} trainable parameters; augmentation: flip, rotate, shift, zoom, shear, brightness",
           "TL-CNN – transfer learning",
           "    ImageNet-pretrained DenseNet-169, all weights frozen (feature extractor)",
           "    global max-pool → BatchNorm → dropout → FC(128) → FC(1) → sigmoid",
           f"    only the head is trained: {P_TL:,} parameters",
           "Both: binary cross-entropy, Adam, batch 64, 30 epochs, LR decay, best-dev-loss checkpoint"], size=18)

    s = slide("CNN training curves")
    pic(s, "cnn_training_curves.png", 0.6, 1.5, w=9.0)
    bl(s, [f"TL-CNN: final train acc {pct(hist['tl']['train_acc'][-1])}, best dev acc {pct(max(hist['tl']['dev_acc']))}",
           f"SIM-CNN: final train acc {pct(hist['sim']['train_acc'][-1])}, best dev acc {pct(max(hist['sim']['dev_acc']))}",
           "Solid = train, dashed = dev",
           "We keep the weights of the epoch with the lowest dev loss"], x=9.8, w=3.2, size=15)

    s = slide("Threshold scanning and confusion matrices")
    pic(s, "threshold_scan.png", 0.6, 1.5, h=3.3)
    pic(s, "confusion_matrices.png", 5.6, 1.5, h=3.3)
    bl(s, ["Imbalanced classes → 0.5 is not automatically the best cut-off; we scan 0.1–0.9 on the dev set",
           f"TL-CNN @ {tl['threshold']}: accuracy {pct(tl['accuracy'])}, precision {pct(tl['precision'])}, "
           f"recall {pct(tl['recall'])}, specificity {pct(tl['specificity'])}",
           f"SIM-CNN @ {sim['threshold']}: accuracy {pct(sim['accuracy'])}, precision {pct(sim['precision'])}, "
           f"recall {pct(sim['recall'])}, specificity {pct(sim['specificity'])}"], y=5.0, h=1.9, size=16)

    s = slide("All models on the same test set")
    pic(s, "model_comparison.png", 0.6, 1.5, w=7.6)
    tbl(s, [r[:4] for r in summary_rows()], 8.4, 1.5, 4.5, 4.2, size=11)
    text(s, 0.6, 6.25, 12.1, 0.6, f"Best overall: {best_overall[0]} ({pct(best_overall[1])}).  "
         f"Best traditional: Random Forest ({pct(acc_all['RF'])}).", size=18, bold=True, color=NAVY)

    s = slide("Comparison with the Stanford report")
    tbl(s, [["", "Stanford report", "Our reproduction"],
            ["Images (train / dev / test)", "7,000 / 3,000 / 5,000", f"{N_TR:,} / {N_DV:,} / {N_TE:,}"],
            ["Class balance", "≈16% ship (natural)", f"{pct(SHIP, 0)} ship (ship-enriched subset)"],
            ["Best traditional model", "Random Forest, 93%", f"Random Forest, {pct(acc_all['RF'])}"],
            ["TL-CNN test accuracy", "90%", pct(tl["accuracy"])],
            ["SIM-CNN test accuracy", "94%", pct(sim["accuracy"])],
            ["Hardware / framework", "Azure GPU, Keras", "Laptop CPU, PyTorch"]],
        0.6, 1.5, 12.1, 3.4, size=16)
    bl(s, ["Same conclusions for traditional models: Random Forest wins, feature engineering helps, Naive Bayes is fragile",
           ("CNN ranking differs: TL-CNN beat SIM-CNN here, probably because we had about 4× less training data"
            if tl["accuracy"] > sim["accuracy"] else
            "Same CNN ranking: the from-scratch network beats the frozen pretrained one"),
           "Absolute numbers are not directly comparable – different subset and class balance"], y=5.1, h=1.8, size=16)

    s = slide("Conclusions, limitations, future work")
    bl(s, [f"Random Forest on hand-engineered features is a strong, cheap baseline ({pct(acc_all['RF'])})",
           f"CNNs: {CNN_SENTENCE}",
           "Feature choice matters as much as the algorithm; more features can hurt",
           "Limitations: small ship-enriched subset, CPU-only, one run per model, image-level labels only",
           "Future work: full dataset, fine-tune the DenseNet backbone, segment/localise ships (U-Net)"], size=20)

    s = slide("Live demo")
    bl(s, ["python -m src.demo                     → 8 random unseen test images",
           "python -m src.demo --n 12 --seed 7     → a different selection",
           "python -m src.demo --image my.jpg      → any image",
           "Prints P(ship) from Random Forest, TL-CNN and SIM-CNN; shows the images with the verdict"],
       w=6.4, size=17)
    if (FIG / "demo.png").exists():
        pic(s, "demo.png", 7.1, 1.6, w=5.9)

    prs.save(str(DOCS / "Team9_Project35_Slides.pptx"))


if __name__ == "__main__":
    from docs.viva_guide import build_viva_guide
    build_writeup()
    build_slides()
    build_viva_guide(globals())
    print("built:", *[p.name for p in sorted(DOCS.glob("Team9_*"))], sep="\n  ")
