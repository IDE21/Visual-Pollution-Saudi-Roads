# Plan v2 (after Track A)

**Status:** proposed, not started. Each step needs the owner's go-ahead.

This plan replaces the open parts of [`PLAN.md`](PLAN.md). It folds in the advisory council's review of the Track A results ([`reports/trackA/report.md`](../reports/trackA/report.md)).

## What changed

- **The random split inflates our model by only about 2.4 mAP50 points.** Leakage does not explain the gap to the paper's 93%. Any wording that suggests it does must go.
- **The paper's one directly comparable number is its external test.** It reports 81% P, 75% R and 70% mAP on the Roboflow "Pothole Detection Dataset" (1,482 images). Evaluating our model on the same dataset is the fairest head-to-head comparison available.
- **The clean split also trains a better model** (+2.3 mAP50 on C). But v1 had 22% more training images, so this difference is not yet attributed to the split alone.

## Steps

| # | Step | Why | Effort / compute | Needs |
|---|---|---|---|---|
| 1 ✅ | **Fix the wording** in README and PLAN.md: report Track A honestly, with no claim that the paper's number is inflated | The council's top priority, from both Reviewer 2 and the Guardian | S, no training | Approval to edit |
| 2 ✅ | **Done: v1 mAP50 0.551 vs the paper's 70% ([report](../reports/external_pothole/report.md)).** External pothole benchmark. Download the Roboflow dataset the paper used and check its licence. Evaluate v1, potholes only, with P, R, F1 and mAP50 (CI included). Compare with the paper's 81/75/70 | The only like-for-like comparison with the paper; cross-dataset generalization | S–M, about 1 h, inference only | Approval to download. Roboflow may need an account or API key, which the owner provides |
| 3 | **Size-matched control.** Train on the clean split subsampled to 20,029 images, then evaluate on C | Separates "clean split" from "more data" in the v1 vs randA gap | About 9 h | Approval to run |
| 4 ✅ | **Done: about 56% of flagged cases are label errors, roughly 11–18% of val labels ([report](../reports/label_audit/report.md)).** Label review (Track P). Export grids of the top 200 background false positives and top 100 false negatives from **val**. The owner, or a second annotator, labels each one as missing label, wrong class, wrong box or true error | Measures the label-noise ceiling before spending compute on bigger models | M; tooling 1 h, review a few hours | The owner's review time |
| 5 | **Resolution screening.** Fraction 0.5, 40 epochs: 640 vs 960 px, plus SAHI tiling at inference. Decide on val only | Barriers are tiny (median 0.81% of the image); resolution is the most likely real gain | About 2 × 8 h | Approval to run |
| 6 | **Full run of the winner.** yolo11s at 960 px for 100 epochs locally (about 50 h), or yolo11m on a cloud GPU | The main accuracy step | L | Explicit approval (over 50 h), or a cloud budget |
| 7 | **Final single evaluation on C**, with CIs, plus the external benchmark again | Closing numbers for the README | S | — |
| 8 | **Publish** (Track C): weights with a model card, a Gradio demo, licence notes (AGPL and CC BY) | Portfolio delivery | M | Approval to push or publish |

### Deferred

- **Per-class random split (randB)**, to mimic the paper's splitting exactly (about 9 h). Its value is low now that leakage turned out small. Revisit only if step 2 shows a large gap.

## Order

1 → 2 → 4 (in parallel with 3) → 5 → 6 → 7 → 8.

Steps 1, 2 and 4 cost almost no compute and should come first.
