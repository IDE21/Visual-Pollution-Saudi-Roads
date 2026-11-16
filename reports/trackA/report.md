# Track A report: how much does a random split inflate the score?

The protocol was fixed before training ([`reports/leak_protocol.md`](../leak_protocol.md)). All mAP values come with 95% bootstrap CIs (2,000 resamples of whole incidents) from [`src/eval_ci.py`](../../src/eval_ci.py).

## Models

| Model | Training split | Train images | Best epoch | Weights |
|---|---|---|---|---|
| **v1** | Grouped by incident, duplicates removed (80/10/10) | 24,418 | 46 / 50 | `yolo11s_640_v1` |
| **randA** | Random at image level, duplicates kept (70/10/20), C held out | 20,029 | 40 / 50 | `yolo11s_640_randA` |

Both models are YOLO11s at 640 px, trained for 50 epochs with identical settings and seed 42.

## Test sets

| Set | Images | What it is |
|---|---|---|
| **C** | 3,053 | Clean test: its incidents and bytes appear in no training set |
| **R** | 5,723 | randA's random test set: 315 images (5.5%) are byte-identical to a train image, and 929 (16%) share an incident with train |
| **R-dedup** | 5,408 | R without the 315 duplicates |

## Results

### mAP (point estimate [95% CI])

| Model → test | mAP50 | mAP50-95 |
|---|---|---|
| randA → R | **0.698** [0.685, 0.712] | 0.478 [0.468, 0.490] |
| randA → R-dedup | 0.685 [0.673, 0.699] | 0.465 [0.455, 0.477] |
| randA → C | 0.674 [0.656, 0.693] | 0.452 [0.438, 0.468] |
| **v1 → C** | **0.697** [0.680, 0.716] | **0.475** [0.463, 0.491] |
| randA → the 315 duplicates only | 0.903 | 0.720 |

### Inflation (randA model)

| Difference | mAP50 | mAP50-95 |
|---|---|---|
| R − C (total inflation) | **+0.024** [+0.001, +0.048] | **+0.027** [+0.007, +0.045] |
| R − R-dedup (duplicates) | +0.013 [+0.009, +0.016] | +0.013 [+0.010, +0.017] |
| R-dedup − C (shared incidents and other) | +0.011 [−0.012, +0.036] | +0.013 [−0.007, +0.032] |

### Which split trains a better model? (paired, same C images)

| v1 − randA on C | mAP50 | mAP50-95 |
|---|---|---|
| barriers | +0.026 [+0.007, +0.047] | +0.027 [+0.010, +0.043] |
| sidewalks | +0.024 [−0.002, +0.052] | +0.025 [+0.006, +0.042] |
| pothole | +0.018 [+0.009, +0.027] | +0.019 [+0.012, +0.027] |
| **all** | **+0.023** [+0.012, +0.035] | **+0.024** [+0.015, +0.032] |

### Paper-style numbers (P, R, F1 at the max-F1 confidence, IoU 0.5, averaged over classes)

| Model → test | Precision | Recall | F1 | mAP50 |
|---|---|---|---|---|
| v1 → C | 0.666 | 0.653 | 0.659 | 0.697 |
| randA → C | 0.644 | 0.645 | 0.644 | 0.674 |
| randA → R | 0.656 | 0.666 | 0.661 | 0.698 |
| Paper (YOLOv5x, random per-class split, DAL labels) | 0.89 | 0.88 | 0.89 | 0.93 (type not stated) |

## Findings

1. **The random split inflates our model's score, but only a little.** mAP50 rises by 2.4 points (CI 0.1–4.8), and mAP50-95 by 2.7 points.
2. **Exact duplicates are memorized.** On the 315 duplicated test images, mAP50 is 0.90, compared with 0.69 on the rest. Because they are only 5.5% of R, they add about 1.3 points to the total.
3. **Shared incidents add no clear inflation.** Their effect is +1.1 points, but the CI includes 0. Photos of the same incident are apparently different enough from each other.
4. **Training on the clean split gives a better model.** v1 beats randA on C by 2.3 mAP50 points, and the CI excludes 0. Part of this may simply come from v1 having 22% more training images, so the two effects cannot be separated here.
5. **Under our replication, leakage does not explain the gap to the paper.** A random split takes our model from 0.674 to 0.698 mAP50, still far below 0.93. The remaining gap must come from factors we could not reproduce:
   - **Per-class splitting:** 2,665 images contain two classes and can be listed in both train and test.
   - **Test labels:** the paper's test labels were produced by the same deep-active-learning loop that trained the models.
   - **Model:** YOLOv5x, which is about 10× larger, trained for 100–250 epochs with tuned hyperparameters.
   - **Metric definition:** the paper does not state which mAP it reports, at what IoU threshold, or at what confidence its P/R/F1 are taken.
   - **Data:** the paper's test images are not the same images as ours.
6. **Our starting hypothesis is not supported.** We expected leakage to explain much of the gap to the paper; under this replication it does not. No public text should claim that the paper's number is inflated by leakage.

## Limits

- One seed per model.
- randA reproduces image-level random splitting, not the paper's per-class splitting.
- The difference between C and R also reflects different image sets, not only leakage. R-dedup, a subset of R, is the closest like-for-like comparison.
