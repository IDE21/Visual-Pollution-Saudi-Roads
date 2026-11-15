# Visual-Pollution-Saudi-Roads

> **Status: under revision.** This repository contains a first baseline (v0). A review found errors in how its results were reported and limitations in how it was trained and evaluated. They are documented below, and the numbers here are now the corrected ones. A rebuilt version based on a real YOLO detector is in progress (see [Roadmap](#roadmap)).

## Overview

This project detects three types of visual pollution in images of Saudi public roads: **barriers**, **sidewalks** (dilapidated sidewalks) and **potholes**.

The current notebook ([`YOLO_VSMODEL.ipynb`](YOLO_VSMODEL.ipynb)) is a **baseline**. Its model is a small custom CNN that is **YOLO-inspired, not YOLO**. For each image it predicts:

- one class label, and
- one bounding box.

It cannot detect more than one object per image. It has not been benchmarked for speed, so no real-time claim is made.

## Dataset

The data is the **Saudi Arabia Public Roads Visual Pollution Dataset** by AlElaiwi et al. (2022), published on Mendeley Data under the **CC BY 4.0** license. This project uses the dataset as published. The annotations are the dataset authors' work.

- Download: https://data.mendeley.com/datasets/bb7b8vtwry/1
- Classes used here: `barriers`, `sidewalks`, `pothole`
- **The official train/val split leaks.** 1,158 val images (16%) are byte-identical copies of train images, and 1,648 incident tickets have photos in both splits. See the [data audit](reports/data_audit/report.md).
- **New splits (v1):** after removing 1,271 exact duplicates, the 30,524 unique images are split 80/10/10 by incident ticket, so no incident appears in two splits. Lists are in [`splits/`](splits/summary.md) and are rebuilt with `python src/make_splits.py --data <root> --install`.
- Images loaded by the notebook: 31,795. This counts images, not objects: the baseline keeps only the first object listed in each label file.

```
AlElaiwi, M., Ahmad, H., Hussain, J., Al-antari, M., Almarri, B., & Azhar, A. (2022).
Saudi Arabia Public Roads Visual Pollution Dataset (Version 1) [Data set]. Mendeley Data.
https://doi.org/10.17632/bb7b8vtwry.1 — Licensed under CC BY 4.0
```

## Baseline model (v0)

- **Input:** images resized to 128×128 without keeping the aspect ratio
- **Backbone:** 3 × (Conv2D + MaxPooling)
- **Heads:**
  - a softmax head for the class
  - a sigmoid head for one box `[cx, cy, w, h]`
- **Losses:**
  - focal loss (γ=2, α=0.25) for the class
  - MSE for the box
- **Training:**
  - Adam (lr=1e-4), batch size 8
  - early stopping on the class loss (stopped at epoch 22, best weights from epoch 17)

## Results (v0, corrected)

Evaluated on a random 20% split of train+val (6,359 images).

> ⚠️ This split was also used for early stopping, so these numbers are **optimistic**. There is no independent test set yet.

### Classification

| Class | Precision | Recall | F1 |
|---|---|---|---|
| barriers | 0.467 | 0.379 | 0.419 |
| sidewalks | 0.589 | 0.384 | 0.465 |
| pothole | 0.786 | 0.903 | 0.841 |

- **Accuracy:** 0.720.
- **Majority-class baseline:** always predicting `pothole` would score about 0.648 on this split.

![Classification confusion matrix](output/classification_confusion_matrix.png)

Barriers and sidewalks are most often predicted as potholes, which is also the dominant class (about 64% of images).

### Localization

| Metric | Value |
|---|---|
| Mean IoU | 0.166 |
| IoU ≥ 0.5 | 8.1% |
| IoU ≥ 0.75 | 0.6% |
| Mean absolute error of box coordinates | 0.124 |

![Bounding box localization matrix](output/bbox_confusion_matrix.png)

**How to read this matrix** (rows: is the class correct? columns: is the box correct, IoU ≥ 0.5?):

| | Box correct | Box incorrect |
|---|---|---|
| **Class correct** | **456 (7.2%)**: fully correct detections | 4,121: right class, wrong box |
| **Class incorrect** | 0 | 1,782: wrong class and wrong box |

Only **456 of 6,359** test images (7.2%) have both the right class and a box with IoU ≥ 0.5.

> An earlier version of this README labelled 1,782 as "true positives", 456 as "false positives" and 4,121 as "false negatives". That reading was wrong.

> The code that produced this matrix is not in the notebook. Its total of 456 boxes with IoU ≥ 0.5 also differs from the notebook's printed 8.1% (about 514). It will be regenerated in the next version.

### Sample predictions

![Sample predictions](output/detection_results.png)

Green boxes are the ground truth, red boxes the predictions. Known display issues in this figure:

- Colours are swapped: the image is OpenCV BGR shown as RGB.
- Some labels are cut off at the image edge.

## Known issues in v0

1. **Not a YOLO detector.** The model predicts one box per image. Every label line after the first is discarded.
2. **Augmentation corrupts boxes.** Flips, shifts, rotation and zoom are applied to the images, but the box coordinates are not transformed with them. This is a major cause of the low IoU.
3. **Evaluation leakage.** Train and val were merged and re-split at random, and the "test" split was used for early stopping.
4. **Class imbalance is not addressed.** Focal loss uses one α for all classes, and no class weighting is applied.
5. **Low resolution.** Resizing to 128×128 without keeping the aspect ratio makes small barriers and sidewalks hard to see.
6. **Not reproducible.**
   - Paths are hard-coded to Google Drive.
   - There is no `requirements.txt`.
   - The trained weights are not published.
   - Training is forced onto the CPU.

## Roadmap

1. ~~**Audit the data.**~~ Done: [`reports/data_audit`](reports/data_audit/report.md).
2. ~~**Fix the evaluation.**~~ Done: leak-free v1 splits by incident ticket in [`splits/`](splits/summary.md).
3. **Retrain with a real detector.** Train Ultralytics YOLO on the original multi-object labels and report mAP@50 and mAP@50-95 per class, with the v0 model evaluated on the same test set for comparison.
4. **Analyse errors per class and object size.** Try a higher resolution or class rebalancing only if this analysis shows they are needed.
5. **Make it reproducible.** Add scripts, configs, `requirements.txt` and released weights.
6. **Publish a demo** and measure inference speed on named hardware.

## Contributions

Issues and pull requests are welcome.
