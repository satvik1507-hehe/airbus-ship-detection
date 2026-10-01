"""Viva / study guide PDF. Called from build_docs.py, which passes its globals as `g`."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


def build_viva_guide(g):
    S = g["styles"](10)
    b, h1, h, q, code = S["b"], S["h1"], S["h"], S["q"], S["code"]
    pct, bullets, grid = g["pct"], g["bullets"], g["grid"]
    acc, cv, sim, tl = g["acc_all"], g["cv"], g["sim"], g["tl"]
    N, N_TR, N_DV, N_TE, SHIP, BASE = g["N"], g["N_TR"], g["N_DV"], g["N_TE"], g["SHIP"], g["BASE"]
    table1, ALGS, FE_GAIN = g["table1"], g["ALGS"], g["FE_GAIN"]
    tl_wins = tl["accuracy"] > sim["accuracy"]
    hist = g["hist"]

    def P(t): return Paragraph(t, b)
    def Q(question, answer): return [Paragraph("Q. " + question, q), Paragraph(answer, b)]
    def C(t): return Paragraph(t.replace("\n", "<br/>").replace(" ", "&nbsp;"), code)

    doc = SimpleDocTemplate(str(g["DOCS"] / "Team9_Project35_Viva_Guide.pdf"), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, title="Viva guide – Airbus Ship Detection")
    t1, bold = g["table1_rows"]()
    s = [
        Paragraph("Viva &amp; Study Guide", S["title"]),
        Paragraph(f"{g['TITLE']}<br/>{g['TEAM']}  |  {g['MEMBERS']}", S["sub"]),
        Spacer(1, 6),
        P("This guide explains what the project does, why each choice was made, what the numbers "
          "mean, how the code is organised, and how to answer the questions an evaluation panel is "
          "likely to ask. Read sections 1–3 first; they are enough to present the project. Sections "
          "4–7 are the theory behind it. Section 10 is a question bank."),

        # ------------------------------------------------------------------ 1
        Paragraph("1. The project in one minute", h1),
        P("<b>Task.</b> Look at a satellite photo and answer one yes/no question: is there at least "
          "one ship in it? That is <b>binary image classification</b>."),
        P("<b>What we compare.</b> Two ways of solving it. (a) <b>Traditional machine learning</b>: we "
          "first compute a few hand-designed numbers that describe each image (colour, texture, shape) "
          "and feed those numbers to five classical classifiers. (b) <b>Deep learning</b>: two "
          "convolutional neural networks (CNNs) that look at the raw pixels and learn their own "
          "features – one trained from scratch (SIM-CNN) and one that reuses a network pre-trained on "
          "ImageNet (TL-CNN, 'transfer learning')."),
        P("<b>Where it comes from.</b> It reproduces a Stanford CS229 student project (Chen, Zheng, "
          "Zhou, 2018) on a smaller subset of the same Kaggle dataset."),
        P(f"<b>Headline results on our {N_TE:,} test images.</b> Random Forest was the best traditional "
          f"model at {pct(acc['RF'])}. TL-CNN reached {pct(tl['accuracy'])} and SIM-CNN "
          f"{pct(sim['accuracy'])}. Always answering 'ship' would score {pct(BASE)}, so that is the "
          "number every model has to beat."),

        # ------------------------------------------------------------------ 2
        Paragraph("2. What we did, step by step", h1),
        bullets([
            "<b>Got the data</b> (<font face='Courier'>src/download_data.py</font>). Downloaded the "
            "official label CSV and a subset of the Kaggle images, resized every image to 256×256, "
            "labelled each image 'ship' or 'no ship' from the CSV, and split them into train / dev / "
            "test.",
            "<b>Extracted features</b> (<font face='Courier'>src/features.py</font>). For every image: "
            "a colour histogram (512 numbers), Haralick texture (13 numbers), Hu moments (7 numbers), "
            "and a tiny 32×32 copy of the image (3,072 numbers) as the 'no feature engineering' option.",
            "<b>Trained five classical models</b> (<font face='Courier'>src/train_traditional.py</font>) "
            "on eight different feature combinations = 40 experiments, then ran 10-fold "
            "cross-validation to see how stable each model is.",
            "<b>Trained two CNNs</b> (<font face='Courier'>src/train_cnn.py</font>, architectures in "
            "<font face='Courier'>src/models.py</font>) for 30 epochs each.",
            "<b>Evaluated the CNNs</b> (<font face='Courier'>src/evaluate_cnn.py</font>): picked the "
            "best decision threshold on the dev set, then measured accuracy, precision, recall, "
            "specificity and the confusion matrix on the test set.",
            "<b>Made figures and a demo</b> (<font face='Courier'>src/make_figures.py</font>, "
            "<font face='Courier'>src/demo.py</font>).",
        ], b),

        # ------------------------------------------------------------------ 3
        Paragraph("3. The dataset – know this cold", h1),
        bullets([
            "<b>Source:</b> Kaggle 'Airbus Ship Detection Challenge' (2018). 192,555 labelled training "
            "images, each 768×768 pixels, RGB, about 31 GB in total.",
            "<b>Labels:</b> a CSV file with one row per ship. Each row holds the ship's pixel mask in "
            "<i>run-length encoding</i> (pairs of 'start pixel, run length'). An image whose row has an "
            "empty mask contains no ship. The competition was about segmentation (outlining ships); we "
            "only use the masks to derive a yes/no label: <i>has at least one mask = ship</i>.",
            "<b>Natural imbalance:</b> in the full dataset only 22.1% of images contain a ship.",
            f"<b>Our subset:</b> {N:,} images. The paper used 15,000; we had a 1 GB download budget. "
            "3,325 images come from a ready-made 494 MB subset of the same competition images published "
            "on Kaggle (aravjain007/aerial-ship-detection-dataset), and 515 were sampled uniformly at "
            "random from the competition itself before Kaggle's per-file rate limit stopped us (11 "
            "images were in both). <b>All labels come from the official competition CSV</b>, not from "
            "the third-party subset.",
            f"<b>Our class balance:</b> {pct(SHIP)} ship, {pct(1 - SHIP)} no ship. The ready-made subset "
            "was built for ship detection, so it over-represents ship images. This is the opposite "
            "imbalance from the paper (their data was about 84% no-ship). Say this openly if asked.",
            f"<b>Split:</b> {N_TR:,} train / {N_DV:,} dev / {N_TE:,} test – the paper's 7:3:5 ratio. It "
            "is <i>stratified</i>, meaning each part has the same ship percentage. Train is for "
            "learning, dev is for decisions (threshold, which epoch to keep), test is touched only "
            "once for the final numbers.",
            "<b>Pre-processing:</b> resize to 256×256×3 (as in the paper). For CNNs pixel values are "
            "scaled to 0–1.",
        ], b),

        # ------------------------------------------------------------------ 4
        Paragraph("4. Feature engineering explained", h1),
        P("A classical model cannot take a picture directly in a useful way – 256×256×3 is 196,608 "
          "numbers, and shifting a ship by one pixel changes all of them. So we summarise each image "
          "with a few <b>global descriptors</b> (numbers describing the whole image)."),
        P("<b>Colour histogram (His).</b> Convert the image from RGB to <b>HSV</b> (Hue = which colour, "
          "Saturation = how vivid, Value = how bright). HSV separates colour from brightness, so it is "
          "less sensitive to lighting. Split each of H, S, V into 8 ranges, giving 8×8×8 = 512 boxes, "
          "and count how many pixels fall in each. Open sea gives a narrow blue/grey spike; a ship or "
          "dock adds other colours. It ignores <i>where</i> the colours are."),
        P("<b>Haralick texture (Ha).</b> Convert to greyscale and build the <b>grey-level co-occurrence "
          "matrix (GLCM)</b>: a table counting how often a pixel of brightness <i>i</i> sits next to a "
          "pixel of brightness <i>j</i>. From that table Haralick defined statistics such as contrast, "
          "correlation, energy, entropy and homogeneity. We take 13 of them, averaged over four "
          "directions (0°, 45°, 90°, 135°). Smooth water and a textured port give very different "
          "values."),
        P("<b>Hu moments (Hu).</b> Image moments are weighted averages of pixel intensities (area, "
          "centroid, spread, orientation). Hu combined them into 7 numbers that do not change when the "
          "image is shifted, scaled or rotated. They describe overall shape. On a whole satellite "
          f"scene they are weak (our accuracy with Hu alone: {', '.join(f'{a} {table1.loc[a, chr(72) + chr(117)]:.2f}' for a in ALGS)})."),
        P("<b>Standardisation.</b> The features have very different scales, so we subtract the mean "
          "and divide by the standard deviation of each feature. The mean and standard deviation are "
          "computed on the <i>training set only</i> and reused for test – otherwise information about "
          "the test set would leak into training. In the code this is the "
          "<font face='Courier'>StandardScaler</font> inside a scikit-learn "
          "<font face='Courier'>Pipeline</font>."),

        # ------------------------------------------------------------------ 5
        Paragraph("5. The five traditional classifiers", h1),
        P("<b>Linear Discriminant Analysis (LDA).</b> Assumes each class is a Gaussian (bell-shaped) "
          "cloud and that both classes share the same covariance matrix. Under that assumption the "
          "best boundary between the classes is a straight line/plane. Fast, no tuning; fails when "
          "the boundary is not linear."),
        P("<b>k-Nearest Neighbours (k = 5).</b> No real training: to classify an image, find the 5 "
          "training images with the closest feature vectors (Euclidean distance) and take a majority "
          "vote. Simple, but slow at prediction time and it suffers when there are many irrelevant "
          "dimensions ('curse of dimensionality')."),
        P("<b>Gaussian Naive Bayes.</b> Uses Bayes' rule, P(class | features) is proportional to P(class) × "
          "P(features | class), and 'naively' assumes every feature is independent of the others "
          "given the class, each following a Gaussian. Our 512 histogram bins are strongly correlated "
          "(they sum to a constant and most are near zero), which violates both assumptions – that "
          f"is why NB does badly whenever His is included ({table1.loc['NB', 'His']:.2f}) and better on "
          f"Haralick alone ({table1.loc['NB', 'Ha']:.2f}). The paper saw the same pattern (0.49 vs 0.84)."),
        P("<b>Random Forest (70 trees).</b> An <i>ensemble</i> of decision trees. Each tree is trained "
          "on a bootstrap sample (random sample with replacement) of the training images and, at each "
          "split, may only choose among a random subset of features. The forest predicts by majority "
          "vote. Averaging many different trees cancels their individual errors (lower variance), it "
          "handles non-linear patterns, does not care about feature scale, and ignores useless "
          "features. That is why it wins among the traditional models."),
        P("<b>Support Vector Machine (RBF kernel).</b> Finds the boundary with the largest margin "
          "between the classes. The RBF (Gaussian) kernel lets the boundary be curved by implicitly "
          "comparing points in a higher-dimensional space. Training cost grows faster than "
          "quadratically with the number of samples."),

        # ------------------------------------------------------------------ 6
        Paragraph("6. CNN concepts", h1),
        bullets([
            "<b>Convolution layer:</b> slides small 3×3 filters over the image; each filter responds "
            "to a local pattern (an edge, a bright blob). The filter weights are learned. The same "
            "filter is used at every position, so a ship is detected wherever it is.",
            "<b>ReLU:</b> max(0, x). Adds non-linearity so stacked layers can represent complex "
            "patterns.",
            "<b>Max-pooling 2×2:</b> keeps the largest value in each 2×2 block, halving width and "
            "height. Reduces computation and gives tolerance to small shifts.",
            "<b>Fully connected (FC) layer:</b> ordinary neural-network layer that combines all "
            "extracted features into the decision.",
            "<b>Sigmoid:</b> squashes the final number into 0–1, read as P(ship).",
            "<b>Binary cross-entropy loss:</b> −[y·log(p) + (1−y)·log(1−p)]. Small when the predicted "
            "probability p is close to the true label y; it punishes confident wrong answers heavily.",
            "<b>Adam:</b> the optimiser – gradient descent with a per-weight adaptive step size and "
            "momentum. Learning rate 0.001, halved when the dev loss stops improving for 2 epochs "
            "('decaying learning rate').",
            "<b>Epoch / batch:</b> one epoch = one pass over all training images; weights are updated "
            "after every mini-batch of 64 images.",
            "<b>Dropout (0.5):</b> randomly switches off half the neurons during training so the "
            "network cannot rely on any single one – reduces overfitting.",
            "<b>Batch normalisation:</b> rescales a layer's inputs to zero mean / unit variance per "
            "batch; stabilises training. Used in the TL-CNN head.",
            "<b>Data augmentation:</b> random flips, rotation (±20°), shift (±10%), zoom (±10%), shear "
            "and brightness (±20%) applied to training images each epoch. A rotated ship is still a "
            "ship, so this creates extra training variety for free and reduces overfitting.",
            "<b>Best checkpoint:</b> after each epoch we measure the dev loss and save the weights "
            "only if it improved. This is a form of early stopping.",
            "<b>Transfer learning:</b> reuse a network trained on another large dataset. DenseNet-169 "
            "was trained on 1.2 million ImageNet photos; its layers already detect edges, textures "
            "and shapes. We <i>freeze</i> it (its weights never change) and train only a small new "
            "classifier on top.",
            "<b>DenseNet:</b> a CNN in which every layer receives the outputs of all earlier layers "
            "in its block ('dense connections'), which helps gradients flow in deep networks. "
            "DenseNet-169 has 169 layers and outputs 1,664 feature maps.",
        ], b),
        Paragraph("Architectures and parameter counts", h),
        P("<b>SIM-CNN:</b> 256×256×3 → conv16+pool → 128×128×16 → conv32+pool → 64×64×32 → conv64+pool "
          "→ 32×32×64 → conv128+pool → 16×16×128 → flatten (32,768) → FC 128 → dropout → FC 1 → sigmoid."),
        P("Parameters of a conv layer = (3×3×input channels + 1 bias) × filters: 448 + 4,640 + 18,496 + "
          "73,856 = 97,440. FC1 = 32,768×128 + 128 = 4,194,432. FC2 = 128 + 1 = 129. "
          "<b>Total 4,292,001</b> – exactly the number in the paper. Note that 98% of the weights sit "
          "in the first FC layer."),
        P("<b>TL-CNN:</b> image → frozen DenseNet-169 → 8×8×1,664 → global max-pool → 1,664 → BatchNorm "
          "→ dropout → FC 128 → FC 1 → sigmoid. Trainable: BatchNorm 2×1,664 = 3,328; FC1 = 1,664×128 "
          "+ 128 = 213,120; FC2 = 129. <b>Total 216,577</b> – also the paper's number. The DenseNet "
          "itself has about 12.5 million weights, but they are frozen."),
        P("<b>Implementation trick:</b> because the backbone is frozen, its output for a given image "
          "never changes. We run every image through DenseNet once, save the 1,664 numbers, and train "
          "the head on those – which is why TL-CNN epochs take under a second. The cost is that we "
          "cannot apply augmentation to TL-CNN."),

        # ------------------------------------------------------------------ 7
        Paragraph("7. Evaluation metrics", h1),
        P("Take 'ship' as the positive class. TP = ship correctly found, TN = empty image correctly "
          "called empty, FP = false alarm, FN = missed ship."),
        bullets([
            "<b>Accuracy</b> = (TP + TN) / all. Misleading with imbalanced classes: always answering "
            f"'ship' already gives {pct(BASE)} here.",
            "<b>Precision</b> = TP / (TP + FP). Of the images we called 'ship', how many really were?",
            "<b>Recall (sensitivity)</b> = TP / (TP + FN). Of the real ship images, how many did we find?",
            "<b>Specificity</b> = TN / (TN + FP). Of the empty images, how many did we correctly call empty?",
            "<b>Confusion matrix:</b> the 2×2 table of TN, FP, FN, TP. 'Normalised' means each row is "
            "divided by its total, so the diagonal shows specificity and recall.",
            "<b>ROC-AUC:</b> the probability that a random ship image gets a higher score than a "
            "random empty image; 0.5 = guessing, 1.0 = perfect. It does not depend on the threshold.",
            "<b>Decision threshold:</b> the model outputs P(ship); we say 'ship' if it is above a "
            "threshold. 0.5 is only a default. We try 0.1 … 0.9 on the dev set and keep the one with "
            f"the best dev accuracy (TL-CNN: {tl['threshold']}, SIM-CNN: {sim['threshold']}). Lower "
            "threshold → more ships found (recall up) but more false alarms (precision down).",
            "<b>10-fold cross-validation:</b> split the data into 10 parts, train on 9 and test on the "
            "10th, rotate 10 times. Gives a mean and a spread, showing how much a result depends on "
            "the particular split. We run it on train + dev (never on test).",
            "<b>Majority-class baseline:</b> the accuracy of always predicting the most common class. "
            "A model below it has learned nothing useful.",
        ], b),

        # ------------------------------------------------------------------ 8
        Paragraph("8. Our results and how to read them", h1),
        Paragraph("Test accuracy, traditional models (rows) by feature set (columns)", S["cap"]),
        grid(t1, col_widths=[1.3 * cm] + [1.9 * cm] * 8, font=8, bold_cells=bold),
        Spacer(1, 6),
        grid(g["summary_rows"](), col_widths=[4.2 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 6.4 * cm], font=8),
        Spacer(1, 6),
        bullets([
            f"<b>Random Forest wins among traditional models</b> – {pct(acc['RF'])} on test and "
            f"{pct(cv['RF']['mean'])} ± {pct(cv['RF']['std'])} in cross-validation, the highest mean. "
            "Same finding as the paper (93%).",
            "<b>Feature engineering vs raw pixels</b> (change in accuracy points): "
            + ", ".join(f"{a} {100 * FE_GAIN[a]:+.1f}" for a in ALGS)
            + ". The paper reported a gain for every model; check the table for where ours agrees.",
            "<b>'More is less':</b> adding features does not always help. Naive Bayes is best on "
            "Haralick alone and collapses when the 512 histogram bins are added.",
            f"<b>Colour is the most useful single descriptor for Random Forest</b> "
            f"(His {table1.loc['RF', 'His']:.3f}, Ha {table1.loc['RF', 'Ha']:.3f}, Hu {table1.loc['RF', 'Hu']:.3f}); "
            "Hu moments are the weakest for every model.",
            f"<b>TL-CNN:</b> {pct(tl['accuracy'])} accuracy, precision {pct(tl['precision'])}, recall "
            f"{pct(tl['recall'])}, specificity {pct(tl['specificity'])}, AUC {g['tl_auc']:.3f}.",
            f"<b>SIM-CNN:</b> {pct(sim['accuracy'])} accuracy, precision {pct(sim['precision'])}, recall "
            f"{pct(sim['recall'])}, specificity {pct(sim['specificity'])}, AUC {g['sim_auc']:.3f}.",
            (f"<b>TL-CNN beat SIM-CNN</b> – the opposite of the paper (SIM 94% vs TL 90%). Our "
             f"explanation: we trained on {N_TR:,} images, about a quarter of the paper's 7,000. A "
             "4.3-million-parameter network trained from scratch needs a lot of data; pretrained "
             "features do not. This is a hypothesis consistent with our results, not something we "
             "tested by varying the training-set size."
             if tl_wins else
             "<b>SIM-CNN beat TL-CNN</b>, as in the paper: features learned on this imagery were "
             "better than frozen ImageNet features."),
            f"<b>Training curves:</b> TL-CNN reaches {pct(hist['tl']['train_acc'][-1])} training accuracy "
            f"while its dev loss rises after the first few epochs – overfitting, which is why we keep "
            f"the best-dev-loss checkpoint. SIM-CNN ends at {pct(hist['sim']['train_acc'][-1])} training "
            f"accuracy and {pct(hist['sim']['dev_acc'][-1])} dev accuracy.",
        ], b),

        # ------------------------------------------------------------------ 9
        Paragraph("9. Differences from the paper and limitations – be upfront", h1),
        grid([["", "Stanford paper", "Our project", "Why"],
              ["Data size", "7k / 3k / 5k", f"{N_TR:,} / {N_DV:,} / {N_TE:,}", "1 GB budget, Kaggle rate limit"],
              ["Class balance", "about 16% ship", f"{pct(SHIP, 0)} ship", "ready-made subset is ship-enriched"],
              ["Raw-pixel baseline", "256×256×3 flattened", "32×32×3 flattened", "keeps KNN/SVM fast"],
              ["Random Forest", "10 trees depth 5 (report)\n70 trees (poster)", "70 trees", "followed the poster"],
              ["TL-CNN augmentation", "yes", "no", "frozen features are cached"],
              ["Framework / hardware", "Keras, Azure GPU", "PyTorch, laptop CPU", "what we had"],
              ["Feature scaling", "not stated", "StandardScaler", "needed by KNN, SVM, LDA"]],
             col_widths=[3.3 * cm, 4.0 * cm, 3.9 * cm, 5.1 * cm], font=8),
        Spacer(1, 6),
        bullets([
            "Our absolute accuracies are <b>not directly comparable</b> to the paper's: different "
            "images, different class balance, less data.",
            "Each model was trained once with one random seed; small differences between models "
            f"(a point or so on {N_TE:,} test images) are within noise.",
            "Hyper-parameters were not tuned (no grid search) – we used the paper's settings and "
            "library defaults.",
            "The threshold was chosen by dev <i>accuracy</i>; in a real application one would choose "
            "it from the cost of a missed ship versus a false alarm.",
            "SIM-CNN's dev loss was still falling at epoch 30, so it had not fully converged; "
            "more epochs might narrow its gap to the other models. We kept 30 to match the paper.",
            "SIM-CNN scored below Random Forest here, whereas in the paper it scored above it.",
            "We classify whole images; we do not say where the ship is.",
        ], b),

        # ------------------------------------------------------------------ 10
        PageBreak(),
        Paragraph("10. Question bank", h1),
        Paragraph("About the problem and data", h),
        *Q("What is your problem statement?",
           "Binary classification of satellite images: does the image contain at least one ship? We "
           "compare traditional ML on hand-engineered features with CNNs, reproducing a Stanford CS229 "
           "project."),
        *Q("Is this detection, classification or segmentation?",
           "Classification. The Kaggle competition was segmentation (outline every ship). The paper – "
           "and we – simplify it to an image-level yes/no. The title says 'detection' in the everyday "
           "sense of detecting whether a ship is present."),
        *Q("How did you get the labels?",
           "From the official train_ship_segmentations_v2.csv. Each row is one ship mask in run-length "
           "encoding. We group by image and count masks: zero masks = no ship, one or more = ship."),
        *Q("Why didn't you use the full dataset?",
           "It is 31 GB and 192k images; the paper itself used only 15k because of compute limits. We "
           f"had a 1 GB budget and a CPU-only laptop, so we used {N:,} images."),
        *Q("Your data is 67% ships but the real data is 22%. Isn't that a problem?",
           "It is a limitation and we state it. The ready-made subset we could download in one file "
           "favours ship images. It means our accuracy is not directly comparable to the paper and "
           "the baseline is 67%, not 84%. To stay honest we report the baseline, precision, recall "
           "and specificity, and we stratify the split so train, dev and test have the same balance. "
           "With more download budget we would sample uniformly from the full data."),
        *Q("Why three sets – train, dev and test?",
           "Train to fit the model. Dev to make choices (decision threshold, which epoch's weights to "
           "keep, learning-rate decay). Test only for the final unbiased number. If we tuned on test, "
           "the reported accuracy would be optimistic."),
        *Q("What is data leakage and how did you avoid it?",
           "Leakage is when information from the test data influences training. We fit the "
           "StandardScaler on training data only (it is inside the pipeline, so cross-validation "
           "refits it per fold), choose thresholds on dev only, and no image appears in two splits."),

        Paragraph("About features and traditional models", h),
        *Q("Why HSV instead of RGB for the histogram?",
           "HSV separates colour (hue) from brightness (value), so the histogram is less affected by "
           "lighting changes than RGB, where all three channels change with brightness."),
        *Q("What is a GLCM?",
           "Grey-level co-occurrence matrix: entry (i, j) counts how often a pixel with grey level i "
           "is adjacent to one with grey level j. Haralick features are statistics of that matrix "
           "describing texture."),
        *Q("Why did Random Forest do best?",
           "It models non-linear boundaries, averages many de-correlated trees (low variance), is "
           "insensitive to feature scale, and effectively ignores uninformative features – useful "
           "when 512 histogram bins are mostly empty."),
        *Q("Why did Naive Bayes do so badly with the colour histogram?",
           "Its two assumptions fail: histogram bins are highly correlated, not independent, and most "
           "bins are near zero with a few large values, not Gaussian. The product of 512 badly "
           "estimated probabilities gives over-confident wrong answers."),
        *Q("Why standardise features?",
           "KNN and RBF-SVM use distances, so a feature with a large numeric range would dominate. "
           "Standardising puts all features on the same scale. Random Forest does not need it."),
        *Q("What does 'bagging' mean?",
           "Bootstrap aggregating: train each model on a random sample (with replacement) of the "
           "training data and average the predictions. Random Forest = bagging of trees + random "
           "feature subsets."),
        *Q("What is the kernel trick?",
           "Computing similarity between points as if they were mapped to a higher-dimensional space, "
           "without doing the mapping. The RBF kernel K(x, y) = exp(−γ·|x − y|<super>2</super>) gives the SVM a "
           "curved boundary."),
        *Q("Why 10-fold cross-validation if you already have a test set?",
           "A single split gives one number; cross-validation shows the spread across ten different "
           "splits, so we can say which model is stable. Random Forest had the highest mean."),

        Paragraph("About the CNNs", h),
        *Q("Why is a CNN suited to images?",
           "Local connectivity and weight sharing: a 3×3 filter has 9 weights per channel regardless "
           "of image size, and detects its pattern anywhere. Stacked layers build from edges to "
           "shapes. A fully connected network on 196,608 inputs would need far more weights and would "
           "ignore spatial structure."),
        *Q("Explain your SIM-CNN architecture.",
           "Four blocks of 3×3 convolution, ReLU and 2×2 max-pooling with 16, 32, 64 and 128 filters, "
           "shrinking 256×256 to 16×16; flatten to 32,768; a 128-unit FC layer with dropout; one "
           "output with sigmoid. 4,292,001 parameters."),
        *Q("How do you count the parameters of a convolution layer?",
           "(kernel height × kernel width × input channels + 1) × number of filters. First layer: "
           "(3×3×3 + 1) × 16 = 448."),
        *Q("What is transfer learning and why freeze the backbone?",
           "Reusing a network trained on another task. Freezing keeps the pretrained weights fixed, "
           "so we only train 216,577 head weights – fast and hard to overfit. The alternative, "
           "fine-tuning, also updates the backbone and usually performs better but needs more compute."),
        *Q("Why binary cross-entropy and not mean squared error?",
           "It is the natural loss for a probability output (maximum likelihood for a Bernoulli "
           "label), and with a sigmoid its gradient does not vanish when the prediction is confidently "
           "wrong, unlike MSE."),
        *Q("What is overfitting and how did you fight it?",
           "The model memorises training data and does worse on new data – training accuracy keeps "
           "rising while dev loss gets worse. We used dropout, data augmentation (SIM-CNN), a frozen "
           "backbone (TL-CNN), and kept the checkpoint with the lowest dev loss."),
        *Q("Why did you scan the threshold?",
           "With imbalanced classes the probability 0.5 is not necessarily the best cut-off. The paper "
           "scans 0.1–0.9 on the dev set; we did the same and applied the chosen threshold to test."),
        *Q(("Why did transfer learning beat your from-scratch CNN when the paper found the opposite?"
            if tl_wins else "Why did the from-scratch CNN beat transfer learning?"),
           (f"Most likely data size: we trained on {N_TR:,} images, the paper on 7,000. A network with "
            "4.3 million weights trained from scratch needs many examples, while DenseNet's features "
            "were already learned from 1.2 million images. We did not run an experiment to confirm it."
            if tl_wins else
            "Satellite images look very different from ImageNet photos, and the frozen backbone "
            "cannot adapt, while SIM-CNN learns filters specific to this imagery. It also has about 20 "
            "times more trainable parameters than the TL-CNN head.")),
        *Q("How would you improve the results?",
           "More data (the full 192k images), fine-tune the last DenseNet block instead of freezing "
           "everything, tune hyper-parameters, add batch normalisation to SIM-CNN, combine models, and "
           "move to segmentation (U-Net) to locate ships."),
        *Q("Which model would you deploy?",
           f"On these numbers {g['best_overall'][0]} has the highest accuracy ({pct(g['best_overall'][1])}). "
           "Random Forest is the cheapest to train and run and needs no GPU, so it is a reasonable "
           "choice if compute is limited. The final choice depends on whether missed ships or false "
           "alarms cost more."),

        Paragraph("About the code and process", h),
        *Q("Walk us through your code structure.",
           "A package <font face='Courier'>src/</font> with one module per stage – download_data, "
           "data, features, train_traditional, models, train_cnn, evaluate_cnn, make_figures, demo – "
           "and <font face='Courier'>config.py</font> holding every path and hyper-parameter. "
           "<font face='Courier'>run_all.py</font> runs the stages in order. Results go to "
           "<font face='Courier'>results/</font>, trained models to <font face='Courier'>models/</font>."),
        *Q("Which libraries did you use and for what?",
           "OpenCV (image reading, resizing, HSV, histogram, Hu moments), mahotas (Haralick), "
           "scikit-learn (the five classifiers, scaling, cross-validation, metrics), PyTorch and "
           "torchvision (CNNs, pretrained DenseNet-169), pandas/NumPy, matplotlib, and the Kaggle API."),
        *Q("Is your work reproducible?",
           "Random seeds are fixed (229), the exact image list and split are committed as "
           "data/labels.csv, the README gives the commands, and requirements.txt lists the packages. "
           "CNN results can still vary slightly between machines."),
        *Q("How long does it take to run?",
           "On a laptop CPU: features and the 40 classical experiments a few minutes, DenseNet feature "
           "extraction several minutes, SIM-CNN about 25 seconds per epoch (roughly 13 minutes for 30 "
           "epochs)."),

        # ------------------------------------------------------------------ 11
        Paragraph("11. Code map", h1),
        grid([["File", "What it does", "Key functions"],
              ["config.py", "paths, seed, image size, split ratio,\nepochs, batch size, thresholds", "–"],
              ["download_data.py", "label CSV + image subset from Kaggle,\nresize, label, stratified split",
               "build_labels, add_subset_zip,\nadd_random_images, split"],
              ["data.py", "loads a split as a NumPy array (cached)", "load_split, read_image"],
              ["features.py", "His, Ha, Hu and raw-pixel features", "color_histogram, haralick,\nhu_moments, combine"],
              ["train_traditional.py", "5 models × 8 feature sets,\n10-fold CV, box plot", "make_models, pipeline, main"],
              ["models.py", "network definitions", "SimCNN, TLHead,\nDenseNetBackbone"],
              ["train_cnn.py", "training loop, augmentation,\ncheckpointing", "augment, run_epoch,\ndensenet_features"],
              ["evaluate_cnn.py", "threshold scan, test metrics,\nconfusion matrices, curves", "predict_proba, metrics"],
              ["make_figures.py", "sample images, class balance,\nmodel comparison chart", "main"],
              ["demo.py", "live prediction with RF, TL-CNN, SIM-CNN", "predict, main"]],
             col_widths=[3.6 * cm, 6.6 * cm, 5.6 * cm], font=8),

        # ------------------------------------------------------------------ 12
        Paragraph("12. Running the live demo", h1),
        P("Open a terminal in the project folder with the Python environment active. Test this on the "
          "exact laptop you will present from, the day before."),
        C("python -m src.demo                    # 8 random unseen test images\n"
          "python -m src.demo --n 12 --seed 7    # a different selection\n"
          "python -m src.demo --image photo.jpg  # any image file"),
        P("It prints a table – image name, true label, and P(ship) from Random Forest, TL-CNN and "
          "SIM-CNN – and opens a picture grid where each title is SIM-CNN's verdict (green = correct, "
          "red = wrong). Things worth saying while it runs: the images come from the test set, which "
          "no model saw during training; point at one mistake and explain it (tiny ship, cloud, "
          "dock); note where the three models disagree."),
        P("If asked to show training live, run the classical part, which takes a few minutes:"),
        C("python -m src.train_traditional"),
        P("To show the networks and parameter counts:"),
        C("python -m src.models"),
        P("The demo needs the <font face='Courier'>data/</font> and <font face='Courier'>models/</font> "
          "folders, which are not stored in GitHub (too large). Keep them on the demo laptop, or "
          "re-create them with <font face='Courier'>python -m src.download_data</font> followed by "
          "<font face='Courier'>python run_all.py</font>."),

        # ------------------------------------------------------------------ 13
        Paragraph("13. Before the review – checklist", h1),
        bullets([
            "Both of you can explain every section of this guide, not just 'your half'. The panel "
            "marks individual contribution and Q&amp;A separately.",
            "Decide who presents which slides. A natural split: one person takes data, features and "
            "traditional models; the other takes the CNNs, evaluation and the demo. Describe your "
            "contributions truthfully if asked how the work was done.",
            "The GitHub repository is private and shared with the faculty (and TAs).",
            "The write-up PDF and slide deck are submitted before the deadline (10 October 2026, "
            "11:59 PM).",
            "The demo runs on the presentation laptop without internet.",
            "You have opened each source file at least once and can find the function named in the "
            "code map.",
        ], b),
    ]

    def footer(canvas, d):
        canvas.setFont("Helvetica", 7.5)
        canvas.drawCentredString(A4[0] / 2, 0.9 * cm, f"Viva guide – {g['TEAM']} – page {d.page}")

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
