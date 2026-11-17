# Plan: label cleaning

**Status:** proposed, not started. Each step needs the owner's go-ahead.

## Why

The [label audit](../reports/label_audit/report.md) found that about 56% of the cases the model flags on val are label errors. That is roughly 11–18% of val labels, and train comes from the same pipeline. Noisy labels limit how well the model can learn, and they also hide real gains when we measure.

## Steps

| # | Step | Compute | Owner time |
|---|---|---|---|
| 1 | Settle the open questions in [`LABEL_GUIDE.md`](LABEL_GUIDE.md) (4 decisions) | — | About 10 min |
| 2 | **Find errors in train (k-fold).** Split train into 5 folds by incident. Train 5 models, each on 4 folds, at the screening winner's resolution, fraction 1.0, about 30 epochs. Predict the held-out fold, then run `src/label_audit.py flag` on all of train | About 5 × 8–9 h ≈ 45 h, can run as separate nights | — |
| 3 | **Triage.** Fix automatically only the clearest cases: wrong class with model conf ≥ 0.8 and IoU ≥ 0.7, and spurious labels the guide rules out (for example, labels on shop signs). Everything else goes to review | Minutes | — |
| 4 | **Review the remaining candidates** on the review page, highest score first, with a time budget (for example, 2 hours ≈ 400 cases). Unreviewed candidates keep their original labels | — | About 2 h |
| 5 | **C-relabelled.** The owner relabels 300 test images, chosen at random, following the guide. These form a separate copy; the original C stays untouched | — | About 2 h |
| 6 | **Retrain on the cleaned labels.** Use the same settings as the comparison run, then evaluate on val, the original C, C-relabelled and the external pothole set | One run | — |

## Decision rules

- **Keep the cleaned labels only if** the model improves on val, and also on C-relabelled, beyond the CI.
- Report the original C as well, even if it does not move. The reason is explained above.
- Every changed label is logged in a CSV: image, old label, new label, reason, and who decided. The changes are reversible.

## Risks

- **Circularity.** Using the model to find errors favours labels the model already agrees with. Mitigation: human review, k-fold out-of-fold predictions, and a separate human-labelled C copy.
- **Time.** Step 2 is the long part. It can run overnight in pieces (one fold per night), and none of it starts without the owner's approval.
