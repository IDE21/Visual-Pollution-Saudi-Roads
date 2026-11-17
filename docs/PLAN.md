# Plan: next phase

> **Superseded.** Track A is done; its result is in [`reports/trackA/report.md`](../reports/trackA/report.md). The open parts of this plan continue in [`PLAN_v2.md`](PLAN_v2.md).

This plan was reviewed by an advisory council (methodologist, domain expert, critical reviewer, licensing/privacy guardian, strategist). Their main changes are folded in below.

## Where we are

| Item | Status |
|---|---|
| Data audit | Done ([`reports/data_audit`](../reports/data_audit/report.md)) |
| Leak-free v1 splits (by incident, exact duplicates removed) | Done ([`splits/`](../splits/summary.md)) |
| YOLO11s, 640 px, 50 epochs, v1 splits | Done ([`reports/yolo11s_640_v1`](../reports/yolo11s_640_v1)) |
| Clean test result | mAP50 0.699, mAP50-95 0.473 |
| Track A (random-split inflation) | Done: +2.4 mAP50 points, too small to explain the gap to the paper |

The dataset paper (AlElaiwi et al., 2022, *Mathematics* 11(1):186) reports 89% precision/F1 and 93% mAP. That number comes from:

- a random per-class 70/10/20 split, with no grouping by incident;
- labels mostly produced by deep active learning from 1,200 hand-labelled images;
- YOLOv5x.

On an external pothole dataset, the paper reports 70% mAP.

## Ground rules

- Every decision is made on **val**.
- The clean test set **C** (3,053 images) is used **once**, at the end of each track.
- Every headline number comes with a 95% bootstrap confidence interval. The bootstrap resamples by incident.
- Cleaning or relabelling is applied to **train and val only**. C stays exactly as it is.
- Comparison with the paper uses neutral wording. We measure what a random split does to **our** model. We do not claim to re-measure the paper's number.

## Step 0: quick checks (before any long run)

1. **Dataset version.** Mendeley has more than one version: v1 and at least v5. Our copy has 31,795 images, and the paper mentions 34,460. Compare our copy with the latest version: image count, file names, label changes.
2. **Analysis protocol.** Write `reports/leak_protocol.md` and `src/eval_ci.py`, which computes the bootstrap CI by incident, before training.

## Track P: pre-processing (train and val only)

1. **Quality filter.** Flag images that are very dark, blurred (low Laplacian variance) or off-topic, then review the flagged images by hand. Remove a flagged image only after review, and log every removal.
2. **Near-duplicates.** The audit found dHash near-duplicate candidates. Review them and drop the redundant copies inside train.
3. **Label review.** Export the top 200 background false positives and the top 100 false negatives from val as image grids. Classify each one:
   - missing label;
   - wrong class;
   - wrong box;
   - true model error.

   If more than about a quarter are label errors, fix the labels in train and val.
4. **Class balance.** Test oversampling of barrier and sidewalk images against the baseline on val.
5. **Augmentation.** Test mixup 0.1 and copy-paste, which are off today, and multi-scale.

Each change is kept only if it improves val mAP50-95 beyond the CI.

## Track A: measure the leak (done)

Result: a random split raised our model's mAP50 by 2.4 points (95% CI 0.1–4.8). Leakage does not explain the gap to the paper's 93% under our replication. Details: [`reports/trackA/report.md`](../reports/trackA/report.md).

Original protocol:

1. Hold out C and every incident it contains.
2. From the rest, build a random image-level 70/10/20 split, as in the paper, keeping the duplicates.
3. Train YOLO11s at 640 px for 50 epochs with the same settings (about 11 h on the M5 Pro).
4. Evaluate on three test sets:
   - **R**: the random test set;
   - **R-dedup**: R with exact duplicates of train removed;
   - **C**: the clean test set.

   Then compute:
   - **R − R-dedup** = effect of exact duplicates;
   - **R-dedup − C** = effect of shared incidents.

   Report the differences per class.
5. Report P, R, F1, mAP50 and mAP50-95, naming the mAP type explicitly.

## Track B: raise real performance on C

1. Screening runs (fraction 0.5, 40 epochs): imgsz 640 vs 960, and 1280 for barriers only. Also try tiling/SAHI at inference.
2. Full run of the winner for 100 epochs with patience 20:
   - yolo11s at 960 px locally (about 50 h);
   - or yolo11m at 960 px on a cloud GPU.
3. Without retraining:
   - per-class confidence thresholds;
   - TTA, tuned on val.
4. Run a single final evaluation on C. Rough target: mAP50 0.75–0.80. This is an estimate, not a promise.

## Track C: publish

- README: a table with the random-split result next to the clean-split result, plus the paper's figures and their protocol.
- Weights on Hugging Face or a GitHub Release, with a model card.
- A Gradio demo, with faces and number plates blurred in any example images.
- Licensing:
  - the Ultralytics code is AGPL-3.0, so the trained weights and the demo inherit it;
  - the dataset is CC BY 4.0, so it must be cited.

## Order and timeline (about 30 days)

| Days | Work |
|---|---|
| 1–2 | Step 0, then start Track A overnight |
| 3–6 | Track A results; Track P steps 1–3 in parallel |
| 7–10 | Track P steps 4–5; Track B screening |
| 11–18 | Track B full run, thresholds, TTA |
| 19–21 | Final evaluation on C; README |
| 22–30 | Weights, demo, review, release |
